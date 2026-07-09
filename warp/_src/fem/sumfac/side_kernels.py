# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Fused sum-factorized side (DG face) apply kernels for linear forms (Phase 6, stage 2).

This module provides the side-domain analog of the fused ``B^T D B`` linear
apply in :mod:`warp._src.fem.sumfac.kernels`, behind the same explicit
``integrate(..., assembly="sumfac")`` opt-in: linear forms over
:class:`warp.fem.Sides`/:class:`warp.fem.BoundarySides` (or side subdomains)
of ``Grid2D``/``Grid3D`` geometries are evaluated matrix-free with
sum-factorized face traces.

Gather formulation
------------------

The kernel launches one block per *cell* adjacent to the side set (no
atomics; races are impossible by construction). Each block loops over the
cell's ``2 * dim`` faces; for every face that belongs to the integration
domain it

1. **B stage** -- loads the side's inner and outer element DOF tensors (one
   cooperative ``tile_load`` each; the layout predicate guarantees
   element-major contiguous DOFs over the whole space partition) and contracts
   them to the four traces at the side quadrature points: the value trace is
   the contraction with the endpoint interpolation row (a one-hot for GLL
   bases), the normal-gradient trace with the endpoint derivative row
   (:func:`warp._src.fem.sumfac.face_trace.normal_derivative_row`), and the
   remaining ``dim - 1`` tangential axes with 1D interpolation/derivative
   matrices built at the side quadrature points mapped through the stage-1
   grid side conventions (cyclic side axes, longitude flips, identity
   inner-to-outer permutation -- see
   :mod:`warp._src.fem.sumfac.face_trace`).
2. **D stage** -- per face quadrature point, evaluates the *transformed user
   integrand* with the test field replaced by a
   :class:`warp._src.fem.field.SideSeedField` (inner/outer value and gradient
   trace seeds carried by ``Sample.test_dof``) and the input field by a
   :class:`warp._src.fem.field.SideTraceInjectedField` whose ``EvalArg`` is
   filled from the ``B``-stage trace tiles. Geometry factors (side measure,
   normals, inverse cell Jacobians for physical gradients) come from the
   native side machinery. The cell accumulates the coefficients of its own
   trace channels: the inner channels when it is the side's inner cell, the
   outer channels when it is the outer cell, and *both* on boundary sides
   (inner == outer; this matches the native trace test field, whose
   ``2 * nodes_per_element`` trace nodes map twice onto the same cell nodes).
   Interior numerical fluxes are therefore evaluated once per adjacent cell
   (twice per side) -- an accepted cost of the store-free formulation.
3. **B^T stage** -- the per-face coefficient tiles are contracted back to a
   cell-local residual tile with the transposed trace operators and stored
   once per cell; a scatter kernel accumulates the per-cell residuals into
   the partition node vector (one writer per node for discontinuous spaces).

Bilinear side forms stay on the default assembly path.
"""

from dataclasses import dataclass
from functools import cached_property

import numpy as np

import warp as wp
from warp._src.fem import cache
from warp._src.fem.domain import GeometryDomain
from warp._src.fem.field import SideSeedField, SideTraceInjectedField, TestField
from warp._src.fem.field.nodal_field import NodalField
from warp._src.fem.geometry import Grid2D, Grid3D
from warp._src.fem.polynomial import quadrature_1d
from warp._src.fem.quadrature import Quadrature, RegularQuadrature
from warp._src.fem.space.basis_space import ShapeBasisSpace, TraceBasisSpace
from warp._src.fem.space.partition import WholeSpacePartition
from warp._src.fem.space.shape.cube_shape_function import CubeTripolynomialShapeFunctions
from warp._src.fem.space.shape.square_shape_function import SquareBipolynomialShapeFunctions
from warp._src.fem.space.topology import RegularDiscontinuousSpaceTopologyMixin, TraceSpaceTopology
from warp._src.fem.sumfac.face_trace import normal_derivative_row
from warp._src.fem.sumfac.kernels import (
    _GRAD_OPERATORS,
    _SUPPORTED_INPUT_OPERATORS,
    SumfacNotApplicableError,
    _check_seedable_operators,
    _copy_qp_fields,
    _get_operator_arrays,
    _set_injected_eval_arg,
)
from warp._src.fem.sumfac.operators_1d import build_derivative_matrix, build_interpolation_matrix
from warp._src.fem.sumfac.qfunction import _SUPPORTED_TEST_OPERATORS
from warp._src.fem.types import (
    NULL_DOF_INDEX,
    NULL_NODE_INDEX,
    DofIndex,
    ElementKind,
    cached_coords_type,
    make_free_sample,
)
from warp._src.fem.utils import masked_indices

__all__ = [
    "SideConstantGeometryDomain",
    "SumfacSideLayout",
    "SumfacSidePlan",
    "find_sumfac_side_layout",
    "get_integrate_side_sumfac_kernel",
    "get_side_gather_arrays",
    "get_sumfac_side_scatter_kernel",
    "make_sumfac_side_plan",
    "sumfac_side_block_dim",
]


def sumfac_side_block_dim(device) -> int:
    """Pick the tile-kernel block size of the fused side kernel for ``device``.

    CPU tile kernels run serialized with a single thread per block. On CUDA
    the side kernel uses 32 threads (one warp, the ``tile_matmul`` cuBLASDx
    minimum) instead of the volume kernels' 64: its GEMMs are small and its
    seeded D stage is scalar code executed redundantly by every thread of the
    block, so halving the block halves the redundant lanes (measured faster
    on the SIP side-apply benchmark; see
    ``design/sumfac-phase6-perf-analysis.md``).
    """
    return 1 if wp.get_device(device).is_cpu else 32


class SideConstantGeometryDomain(GeometryDomain):
    """Side-domain stand-in whose side-constant geometry factors are injected per face.

    On ``Grid2D``/``Grid3D`` sides the normal, the side measure, and the
    measure ratio are constant per side. The fused side kernel evaluates them
    ONCE per face through the native machinery and injects them into the
    integrand's ``Domain`` argument (extra ``ElementArg`` members), instead of
    re-deriving them inside every seeded integrand evaluation -- each
    derivation re-decodes the grid side, which the perf analysis identified as
    the dominant scalar cost of the D stage
    (``design/sumfac-phase6-perf-analysis.md``). Every other domain operator
    (position, deformation gradient, inner/outer cell maps, ...) forwards to
    the native implementation through the embedded base element arg, so
    integrands keep their full generality; the side-apply oracle tests verify
    exact equality with the native evaluation (including the
    position-dependent coefficient forms).

    This class only serves generated-type and operator-resolution duties; it
    is never used host-side to fill launch arguments (the kernel fills the
    injected members in-kernel, block-uniformly per face, and the field
    arguments keep receiving the native element arg through the kernel's
    ``domain_arg`` parameter).

    Args:
        base: The native side domain being stood in for.
    """

    def __init__(self, base: GeometryDomain):
        super().__init__(base.geometry_partition)
        self._base = base
        # Sides do not support position lookups; mirror the base domain.
        self.element_lookup = getattr(base, "element_lookup", None)
        self.element_partition_lookup = getattr(base, "element_partition_lookup", None)

    @cached_property
    def name(self) -> str:
        """Unique name, discriminating the injected stand-in from its base domain."""
        return f"{self._base.name}_SideConstInj"

    def __eq__(self, other) -> bool:
        return isinstance(other, SideConstantGeometryDomain) and self._base == other._base

    @property
    def element_kind(self) -> ElementKind:
        """Kind of elements contained in the domain (always sides)."""
        return self._base.element_kind

    @property
    def dimension(self) -> int:
        """Dimension of the side elements."""
        return self._base.dimension

    def element_count(self) -> int:
        """Number of elements in the domain."""
        return self._base.element_count()

    def geometry_element_count(self) -> int:
        """Number of elements in the underlying geometry."""
        return self._base.geometry_element_count()

    def reference_element(self):
        """Reference element of the base domain."""
        return self._base.reference_element()

    def supports_lookup(self, device) -> bool:
        """Whether position lookups are supported (forwards to the base domain)."""
        return self._base.supports_lookup(device)

    def element_arg_value(self, device):
        """Unsupported: the injected element arg is filled in-kernel, per face."""
        raise RuntimeError(
            "SideConstantGeometryDomain has no host-side element arg value; the fused side kernel fills its "
            "ElementArg in-kernel"
        )

    def cell_domain(self):
        """Cell domain of the base side domain."""
        return self._base.cell_domain()

    @property
    def ElementIndexArg(self):
        """Element indexing argument struct (the base domain's)."""
        return self._base.ElementIndexArg

    @property
    def element_index(self):
        """Device function mapping domain element indices to side indices (the base domain's)."""
        return self._base.element_index

    @property
    def element_partition_index(self):
        """Device function mapping side indices to domain element indices (the base domain's)."""
        return self._base.element_partition_index

    @cached_property
    def ElementArg(self):
        """Element arg embedding the base arg plus the injected side-constant factors."""
        geometry = self.geometry
        base_arg = self._base.ElementArg
        normal_type = cache.cached_vec_type(length=geometry.dimension, dtype=geometry.scalar_type)
        scalar_type = geometry.scalar_type

        @cache.dynamic_struct(suffix=self.name)
        class SideConstantElementArg:
            base: base_arg
            normal: normal_type
            measure: scalar_type
            measure_ratio: scalar_type

        return SideConstantElementArg

    # -- Injected side-constant quantities -----------------------------------

    @cached_property
    def element_normal(self):
        """Device function returning the injected per-face normal."""

        @cache.dynamic_func(suffix=self.name)
        def side_const_normal(args: self.ElementArg, s: self.geometry.sample_type):
            return args.normal

        return side_const_normal

    @cached_property
    def element_measure(self):
        """Device function returning the injected per-face side measure."""

        @cache.dynamic_func(suffix=self.name)
        def side_const_measure(args: self.ElementArg, s: self.geometry.sample_type):
            return args.measure

        return side_const_measure

    @cached_property
    def element_measure_ratio(self):
        """Device function returning the injected per-face measure ratio."""

        @cache.dynamic_func(suffix=self.name)
        def side_const_measure_ratio(args: self.ElementArg, s: self.geometry.sample_type):
            return args.measure_ratio

        return side_const_measure_ratio

    # -- Operators forwarded to the native side machinery ---------------------

    def _forward_sample_func(self, base_fn, fn_suffix: str):
        @cache.dynamic_func(suffix=f"{self.name}_{fn_suffix}")
        def side_inj_forward(args: self.ElementArg, s: self.geometry.sample_type):
            return base_fn(args.base, s)

        return side_inj_forward

    @cached_property
    def element_position(self):
        """Device function forwarding position evaluation to the base domain."""
        return self._forward_sample_func(self._base.element_position, "pos")

    @cached_property
    def element_deformation_gradient(self):
        """Device function forwarding deformation-gradient evaluation to the base domain."""
        return self._forward_sample_func(self._base.element_deformation_gradient, "defgrad")

    @cached_property
    def element_environment_index(self):
        """Device function forwarding environment-index evaluation to the base domain."""
        return self._forward_sample_func(self._base.element_environment_index, "envidx")

    @cached_property
    def element_inner_cell_index(self):
        """Device function forwarding inner-cell index lookup to the base domain."""
        base_fn = self._base.element_inner_cell_index

        @cache.dynamic_func(suffix=self.name)
        def side_inj_inner_cell_index(args: self.ElementArg, side_index: int):
            return base_fn(args.base, side_index)

        return side_inj_inner_cell_index

    @cached_property
    def element_outer_cell_index(self):
        """Device function forwarding outer-cell index lookup to the base domain."""
        base_fn = self._base.element_outer_cell_index

        @cache.dynamic_func(suffix=self.name)
        def side_inj_outer_cell_index(args: self.ElementArg, side_index: int):
            return base_fn(args.base, side_index)

        return side_inj_outer_cell_index

    @cached_property
    def element_inner_cell_coords(self):
        """Device function forwarding inner-cell coordinate mapping to the base domain."""
        base_fn = self._base.element_inner_cell_coords
        coords_t = cached_coords_type(self.geometry.scalar_type)

        @cache.dynamic_func(suffix=self.name)
        def side_inj_inner_cell_coords(args: self.ElementArg, side_index: int, side_coords: coords_t):
            return base_fn(args.base, side_index, side_coords)

        return side_inj_inner_cell_coords

    @cached_property
    def element_outer_cell_coords(self):
        """Device function forwarding outer-cell coordinate mapping to the base domain."""
        base_fn = self._base.element_outer_cell_coords
        coords_t = cached_coords_type(self.geometry.scalar_type)

        @cache.dynamic_func(suffix=self.name)
        def side_inj_outer_cell_coords(args: self.ElementArg, side_index: int, side_coords: coords_t):
            return base_fn(args.base, side_index, side_coords)

        return side_inj_outer_cell_coords

    @cached_property
    def cell_to_element_coords(self):
        """Device function forwarding cell-to-side coordinate conversion to the base domain."""
        base_fn = self._base.cell_to_element_coords
        coords_t = cached_coords_type(self.geometry.scalar_type)

        @cache.dynamic_func(suffix=self.name)
        def side_inj_cell_to_element_coords(
            args: self.ElementArg, side_index: int, element_index: int, element_coords: coords_t
        ):
            return base_fn(args.base, side_index, element_index, element_coords)

        return side_inj_cell_to_element_coords

    @cached_property
    def element_coordinates(self):
        """Device function forwarding world-to-side coordinate queries to the base domain."""
        base_fn = self._base.element_coordinates
        pos_t = cache.cached_vec_type(length=self.geometry.dimension, dtype=self.geometry.scalar_type)

        @cache.dynamic_func(suffix=self.name)
        def side_inj_coordinates(args: self.ElementArg, element_index: int, pos: pos_t):
            return base_fn(args.base, element_index, pos)

        return side_inj_coordinates

    @cached_property
    def element_closest_point(self):
        """Device function forwarding closest-point queries to the base domain."""
        base_fn = self._base.element_closest_point
        pos_t = cache.cached_vec_type(length=self.geometry.dimension, dtype=self.geometry.scalar_type)

        @cache.dynamic_func(suffix=self.name)
        def side_inj_closest_point(args: self.ElementArg, element_index: int, pos: pos_t):
            return base_fn(args.base, element_index, pos)

        return side_inj_closest_point

    @cached_property
    def domain_cell_arg(self):
        """Device function mapping the injected domain arg to the base cell-domain arg."""
        base_fn = self._base.domain_cell_arg
        BaseDomainArg = self._base.DomainArg

        @cache.dynamic_func(suffix=self.name)
        def side_inj_domain_cell_arg(x: self.DomainArg):
            return base_fn(BaseDomainArg(x.geo.base, x.index))

        return side_inj_domain_cell_arg


def _tensor_product_side_quadrature_points_1d(quadrature: RegularQuadrature, face_dim: int) -> np.ndarray:
    """Extract the 1D point set of a lexicographic tensor-product side quadrature.

    The fused side kernel assumes the per-side quadrature points enumerate the
    outer product of a single 1D rule over the ``face_dim`` side reference
    coordinates, with the first side coordinate slowest; this is verified
    explicitly rather than assumed. Raises
    :class:`SumfacNotApplicableError` otherwise.
    """

    def tensor_product_error():
        return SumfacNotApplicableError(
            "assembly='sumfac' requires the side quadrature points to form the lexicographic tensor product "
            f"of a single 1D rule along each side axis; the {count} points per side of this RegularQuadrature do not"
        )

    count = quadrature.max_points_per_element()
    if count is None or count < 1:
        raise SumfacNotApplicableError(
            "assembly='sumfac' requires a quadrature rule with a fixed, nonzero number of points per side"
        )
    q = round(count ** (1.0 / face_dim))
    while q**face_dim < count:
        q += 1
    if q < 1 or q**face_dim != count:
        raise tensor_product_error()

    points = np.array([[p[i] for i in range(face_dim)] for p in quadrature.points], dtype=np.float64)
    # With the first side coordinate slowest, the last coordinate of the first q points is the 1D rule
    qpoints_1d = points[:q, face_dim - 1].copy()
    if len(np.unique(qpoints_1d)) != q:
        raise tensor_product_error()

    grids = np.meshgrid(*([qpoints_1d] * face_dim), indexing="ij")
    expected = np.stack([g.reshape(-1) for g in grids], axis=-1)
    atol = 1e-5 if quadrature.domain.geometry.scalar_type == wp.float32 else 1e-12
    if not np.allclose(points, expected, rtol=0.0, atol=atol):
        raise tensor_product_error()
    return qpoints_1d


def _check_side_tensor_product_space(space, dim: int):
    """Check that ``space`` is the trace of a scalar, discontinuous, tensor-product polynomial space.

    Raises :class:`SumfacNotApplicableError` naming the first unmet requirement.
    """
    if space.NODE_DOF_COUNT != 1 or space.VALUE_DOF_COUNT != 1:
        raise SumfacNotApplicableError(
            f"assembly='sumfac' requires a scalar function space; space '{space.name}' has "
            f"{space.NODE_DOF_COUNT} DOF(s) per node and {space.VALUE_DOF_COUNT} value DOF(s) (1 and 1 required)"
        )
    topology = space.topology
    if not isinstance(topology, TraceSpaceTopology) or not isinstance(
        topology.full_space_topology(), RegularDiscontinuousSpaceTopologyMixin
    ):
        raise SumfacNotApplicableError(
            "assembly='sumfac' side integration requires the trace of a discontinuous (DG) space over a regular "
            f"tensor-product grid; space '{space.name}' uses {type(topology).__name__}"
        )
    basis = space.basis
    shape_cls = SquareBipolynomialShapeFunctions if dim == 2 else CubeTripolynomialShapeFunctions
    cell_basis = basis._basis if isinstance(basis, TraceBasisSpace) else None
    if not isinstance(cell_basis, ShapeBasisSpace) or type(cell_basis.shape) is not shape_cls:
        raise SumfacNotApplicableError(
            "assembly='sumfac' requires square/cube tensor-product polynomial shape functions; "
            f"space '{space.name}' uses {type(basis).__name__}"
        )
    degree = space.degree
    if degree < 1 or topology.full_space_topology().MAX_NODES_PER_ELEMENT != (degree + 1) ** dim:
        raise SumfacNotApplicableError(
            f"assembly='sumfac' requires a complete tensor-product polynomial basis of degree >= 1; "
            f"space '{space.name}' has degree {degree}"
        )
    return cell_basis


@dataclass
class SumfacSideLayout:
    """Host-side description of a qualifying tensor-product linear side form."""

    degree: int
    n: int
    q: int
    dim: int
    qpoints_1d: np.ndarray
    test_uses_grad: bool
    input_name: str
    input_field: NodalField.Trace
    cell_basis: ShapeBasisSpace


@dataclass
class SumfacSidePlan:
    """Launch-side description of a sum-factorized linear side-form integration.

    Built by :func:`make_sumfac_side_plan` once a form has qualified; carries
    the original input trace field and its underlying cell field (for the
    in-kernel DOF gather), the substituted
    :class:`warp._src.fem.field.SideSeedField`/
    :class:`warp._src.fem.field.SideTraceInjectedField` instances, the baked
    tile sizes, and the host 1D operator matrices:

    * ``end_ops`` (4, n): rows ``[value(end 0); d/dx(end 0); value(end 1);
      d/dx(end 1)]`` -- the endpoint interpolation (one-hot for GLL bases) and
      derivative rows collapsing the face-normal axis.
    * ``tang_ops`` (4q, n): rows ``[interp; deriv; interp_flipped;
      deriv_flipped]`` -- the 1D operators at the side quadrature points and
      at their longitude-flipped images ``1 - t``.
    * ``end_ops_t`` / ``tang_ops_t``: contiguous transposes of the above, so
      the kernel never feeds a transposed operand to ``tile_matmul``.
    """

    test: TestField
    test_name: str
    input_name: str
    input_field: NodalField.Trace
    cell_field: NodalField
    seed_field: SideSeedField
    injected_field: SideTraceInjectedField
    injected_domain: SideConstantGeometryDomain
    degree: int
    n: int
    q: int
    dim: int
    test_uses_grad: bool
    end_ops: np.ndarray
    tang_ops: np.ndarray
    end_ops_t: np.ndarray
    tang_ops_t: np.ndarray

    @property
    def nodes_per_element(self) -> int:
        """Number of nodes per cell, ``n**dim``."""
        return self.n**self.dim

    def operator_arrays(self, dtype, device) -> tuple[wp.array, wp.array, wp.array, wp.array]:
        """Return the ``(end_ops, end_ops_t, tang_ops, tang_ops_t)`` device arrays in ``dtype`` on ``device``, cached."""
        ops = _get_operator_arrays(self.end_ops, self.tang_ops, dtype, device)
        ops_t = _get_operator_arrays(self.end_ops_t, self.tang_ops_t, dtype, device)
        return ops[0], ops_t[0], ops[1], ops_t[1]


def find_sumfac_side_layout(
    integrand,
    arguments,
    test: TestField,
    quadrature: Quadrature,
    domain: GeometryDomain,
) -> SumfacSideLayout:
    """Return the sum-factorization layout of a qualifying linear side form.

    Every structural assumption of the fused side kernel is verified; anything
    unproven raises :class:`SumfacNotApplicableError` naming the first unmet
    requirement: a side domain over a ``Grid2D``/``Grid3D`` geometry
    (unstructured quad/hex meshes have per-side orientations and are not
    supported), the trace of a scalar discontinuous tensor-product space on
    the whole space partition for both the test and the input field, a
    lexicographic tensor-product :class:`RegularQuadrature` over the side
    domain, and only seedable trace operators (``inner``, ``outer``, ``grad``,
    ``grad_outer``, ``degree``) on the test and input fields.

    Args:
        integrand: The form being integrated; ``integrand.operators`` must
            have been populated (via ``_find_integrand_operators``).
        arguments: Parsed integrand arguments (before field substitution).
        test: The (plain) test field of the linear side form.
        quadrature: Quadrature formula of the integration.
        domain: Side integration domain.
    """
    if domain.element_kind != ElementKind.SIDE:
        raise SumfacNotApplicableError("find_sumfac_side_layout requires a side domain")

    geometry = domain.geometry
    if type(geometry) not in (Grid2D, Grid3D):
        raise SumfacNotApplicableError(
            "assembly='sumfac' side integration requires a Grid2D or Grid3D geometry; got "
            f"{type(geometry).__name__} (unstructured meshes have per-side orientations that the "
            "sum-factorized side path does not support)"
        )
    dim = geometry.dimension

    cell_basis = _check_side_tensor_product_space(test.space, dim)

    if not isinstance(test.space_partition, WholeSpacePartition):
        raise SumfacNotApplicableError(
            "assembly='sumfac' side integration requires the test field to be defined over the whole space "
            f"partition (got {type(test.space_partition).__name__}); the gather kernel indexes element DOFs "
            "directly"
        )

    if not isinstance(quadrature, RegularQuadrature) or quadrature.domain != domain:
        raise SumfacNotApplicableError(
            "assembly='sumfac' requires a RegularQuadrature defined over the integration domain; "
            f"got {type(quadrature).__name__}"
        )
    qpoints_1d = _tensor_product_side_quadrature_points_1d(quadrature, dim - 1)

    test_operators = _check_seedable_operators(integrand, arguments.test_name, _SUPPORTED_TEST_OPERATORS, "test field")

    # Exactly one injectable input field: a nodal trace field over the same
    # (trace) space as the test field, on the whole space partition (so its
    # cell DOF storage is element-major contiguous and the trace gather is a
    # plain tile load), accessed only through value/gradient trace operators.
    input_name = None
    input_field = None
    rejected = []
    for name, field in arguments.field_args.items():
        if name == arguments.test_name or not isinstance(field, NodalField.Trace):
            continue
        if field.space.name != test.space.name:
            rejected.append(f"'{name}' is defined over space '{field.space.name}', not the test space")
            continue
        if not isinstance(field.space_partition, WholeSpacePartition):
            rejected.append(
                f"'{name}' is defined over a partial space partition "
                f"({type(field.space_partition).__name__}); the trace gather requires the whole space"
            )
            continue
        field_operators = integrand.operators.get(name, set())
        unsupported = field_operators - _SUPPORTED_INPUT_OPERATORS
        if unsupported:
            op_names = ", ".join(sorted(op.name for op in unsupported))
            rejected.append(f"'{name}' is accessed through unsupported operator(s) [{op_names}]")
            continue
        input_name = name
        input_field = field
        break
    if input_name is None:
        detail = "; ".join(rejected) if rejected else "no discrete nodal trace field argument found"
        raise SumfacNotApplicableError(
            "assembly='sumfac' side integration requires an input trace field (e.g. ``field.trace()``) defined "
            f"over the same space as the test field '{arguments.test_name}' and accessed only through "
            f"value/gradient trace operators: {detail}"
        )

    degree = test.space.degree
    return SumfacSideLayout(
        degree=degree,
        n=degree + 1,
        q=len(qpoints_1d),
        dim=dim,
        qpoints_1d=qpoints_1d,
        # Seed (and lift) the test-gradient trace channels only when the
        # integrand actually applies a gradient operator to the test field.
        test_uses_grad=bool(_GRAD_OPERATORS & test_operators),
        input_name=input_name,
        input_field=input_field,
        cell_basis=cell_basis,
    )


#: Plan cache: rebuilding the side plan dominates the per-apply host cost
#: (operator-matrix construction, quadrature tensor-product verification,
#: stand-in field creation), so plans are cached per argument-object identity.
#: The cached value pins strong references to every id()-keyed object, so a
#: key can never alias a collected object.
_side_plan_cache: dict = {}


def _side_plan_cache_key(integrand, arguments, test, quadrature, domain):
    """Identity-based cache key of a side plan: same objects, same plan."""
    field_ids = []
    for name, field in arguments.field_args.items():
        if isinstance(field, NodalField.Trace):
            # Re-creating ``field.trace()`` per apply is the common matrix-free
            # pattern; the plan only depends on the underlying cell field.
            field_ids.append((name, "trace", id(field.cell_field)))
        elif isinstance(field, GeometryDomain):
            field_ids.append((name, "domain", id(field)))
        else:
            field_ids.append((name, type(field).__qualname__, id(field)))
    return (id(integrand), id(test), id(quadrature), id(domain), tuple(field_ids))


def make_sumfac_side_plan(
    integrand,
    arguments,
    test: TestField,
    quadrature: Quadrature,
    domain: GeometryDomain,
) -> SumfacSidePlan:
    """Build (or fetch from cache) the side launch plan and substitute the stand-in fields in ``arguments``.

    On success, ``arguments.field_args`` is mutated in place: the test field
    is replaced by a :class:`warp._src.fem.field.SideSeedField`, the input
    trace field by a :class:`warp._src.fem.field.SideTraceInjectedField`, and
    the ``Domain`` argument by a :class:`SideConstantGeometryDomain`, so
    that the downstream ``FieldStruct`` / ``IntegrandTransformer`` machinery
    generates the in-kernel Q-function. Raises
    :class:`SumfacNotApplicableError` if the form does not qualify (the
    layout is validated before any mutation of ``arguments``). Plans are
    cached on the identity of the argument objects (with traced input fields
    keyed by their underlying cell field, so the usual
    ``fields={"u": u.trace(), ...}``-per-apply pattern hits the cache).
    """
    cache_key = _side_plan_cache_key(integrand, arguments, test, quadrature, domain)
    cached = _side_plan_cache.get(cache_key)
    if cached is not None:
        plan = cached[0]
        arguments.field_args[plan.test_name] = plan.seed_field
        arguments.field_args[plan.input_name] = plan.injected_field
        if arguments.domain_name is not None:
            arguments.field_args[arguments.domain_name] = plan.injected_domain
        return plan

    layout = find_sumfac_side_layout(integrand, arguments, test, quadrature, domain)

    n, q = layout.n, layout.q
    nodes_1d = np.asarray(quadrature_1d(point_count=n, family=layout.cell_basis.shape.family)[0], dtype=np.float64)

    interp = build_interpolation_matrix(nodes_1d, layout.qpoints_1d)
    deriv = build_derivative_matrix(nodes_1d, layout.qpoints_1d)
    interp_flip = build_interpolation_matrix(nodes_1d, 1.0 - layout.qpoints_1d)
    deriv_flip = build_derivative_matrix(nodes_1d, 1.0 - layout.qpoints_1d)
    tang_ops = np.ascontiguousarray(np.concatenate([interp, deriv, interp_flip, deriv_flip], axis=0))
    assert tang_ops.shape == (4 * q, n)

    value_row_0 = np.zeros(n)
    value_row_0[0] = 1.0
    value_row_1 = np.zeros(n)
    value_row_1[-1] = 1.0
    end_ops = np.ascontiguousarray(
        np.stack([value_row_0, normal_derivative_row(nodes_1d, 0), value_row_1, normal_derivative_row(nodes_1d, 1)])
    )

    seed_field = SideSeedField.from_field(test)
    injected_field = SideTraceInjectedField.from_field(layout.input_field, domain)
    arguments.field_args[arguments.test_name] = seed_field
    arguments.field_args[layout.input_name] = injected_field

    # Substitute the integrand's Domain argument with the side-constant
    # geometry stand-in: the kernel evaluates normal/measure/measure_ratio
    # once per face and injects them into the channel evaluations instead of
    # re-running the grid side decode inside every seeded integrand call.
    injected_domain = SideConstantGeometryDomain(domain)
    if arguments.domain_name is not None:
        arguments.field_args[arguments.domain_name] = injected_domain

    plan = SumfacSidePlan(
        test=test,
        test_name=arguments.test_name,
        input_name=layout.input_name,
        input_field=layout.input_field,
        cell_field=layout.input_field.cell_field,
        seed_field=seed_field,
        injected_field=injected_field,
        injected_domain=injected_domain,
        degree=layout.degree,
        n=n,
        q=q,
        dim=layout.dim,
        test_uses_grad=layout.test_uses_grad,
        end_ops=end_ops,
        tang_ops=tang_ops,
        end_ops_t=np.ascontiguousarray(end_ops.T),
        tang_ops_t=np.ascontiguousarray(tang_ops.T),
    )
    # Pin the id()-keyed objects alongside the plan (see _side_plan_cache).
    _side_plan_cache[cache_key] = (plan, (integrand, test, quadrature, domain))
    return plan


# -- Cell-to-side gather map ----------------------------------------------------


def _get_side_face_map_kernel(domain: GeometryDomain):
    """Build (and cache) the kernel filling the per-cell face-to-domain-side map."""
    geometry = domain.geometry
    dim = geometry.dimension
    scalar_type = geometry.scalar_type
    coords_type = cached_coords_type(scalar_type)

    @cache.dynamic_kernel(suffix=domain.name, kernel_options={"enable_backward": False})
    def fill_side_face_map(
        domain_arg: domain.ElementArg,
        domain_index_arg: domain.ElementIndexArg,
        face_map: wp.array2d(dtype=int),
        cell_mask: wp.array(dtype=int),
    ):
        domain_side_index = wp.tid()
        side_index = domain.element_index(domain_index_arg, domain_side_index)

        # The side center has all tangential coordinates at 0.5, so exactly
        # one coordinate of the mapped cell coords sits at 0 or 1: the face
        # normal axis and the cell's end along it (exact comparisons -- the
        # grid side machinery assigns the constants 0.0/1.0 directly).
        center = coords_type(scalar_type(0.5), scalar_type(0.5), scalar_type(0.0))
        inner_cell = geometry.side_inner_cell_index(domain_arg, side_index)
        outer_cell = geometry.side_outer_cell_index(domain_arg, side_index)
        inner_coords = geometry.side_inner_cell_coords(domain_arg, side_index, center)
        outer_coords = geometry.side_outer_cell_coords(domain_arg, side_index, center)

        axis = int(0)
        inner_end = int(0)
        for d in range(dim):
            if inner_coords[d] == scalar_type(0.0):
                axis = d
                inner_end = 0
            elif inner_coords[d] == scalar_type(1.0):
                axis = d
                inner_end = 1
        outer_end = wp.where(outer_coords[axis] == scalar_type(1.0), 1, 0)

        face_map[inner_cell, 2 * axis + inner_end] = domain_side_index
        cell_mask[inner_cell] = 1
        if outer_cell != inner_cell or outer_end != inner_end:
            face_map[outer_cell, 2 * axis + outer_end] = domain_side_index
            cell_mask[outer_cell] = 1

    return fill_side_face_map


def get_side_gather_arrays(domain: GeometryDomain, device, temporary_store=None) -> tuple[wp.array, wp.array]:
    """Return the gather-formulation launch arrays of a side domain, cached per device.

    Builds (once per ``(domain, device)``) the ``(cell_count, 2 * dim)``
    ``face_map`` array holding, for each cell and local face
    ``2 * axis + end``, the *domain* element index of the side covering that
    face (or ``-1`` when the side is not part of the integration domain), and
    the compacted list of active cells (cells adjacent to at least one domain
    side) that forms the kernel launch space.

    Args:
        domain: Side integration domain.
        device: Warp device on which to build (and cache) the arrays.
        temporary_store: Shared pool from which to allocate build temporaries.

    Returns:
        The ``(face_map, active_cells)`` pair of device arrays.
    """
    device = wp.get_device(device)
    gather_cache = domain.__dict__.setdefault("_sumfac_side_gather_cache", {})
    cached = gather_cache.get(device.alias)
    if cached is not None:
        return cached

    geometry = domain.geometry
    dim = geometry.dimension
    cell_count = geometry.cell_count()

    face_map = wp.full((cell_count, 2 * dim), -1, dtype=int, device=device)
    cell_mask = wp.zeros(cell_count, dtype=int, device=device)

    side_count = domain.element_count()
    if side_count > 0:
        wp.launch(
            _get_side_face_map_kernel(domain),
            dim=side_count,
            inputs=[
                domain.element_arg_value(device),
                domain.element_index_arg_value(device),
                face_map,
                cell_mask,
            ],
            device=device,
        )

    active_cells, _ = masked_indices(cell_mask, temporary_store=temporary_store)
    active_cells = active_cells.detach()

    gather_cache[device.alias] = (face_map, active_cells)
    return face_map, active_cells


# -- Fused side kernel factory ---------------------------------------------------


def get_integrate_side_sumfac_kernel(
    integrand_func: wp.Function,
    domain: GeometryDomain,
    quadrature: Quadrature,
    FieldStruct,
    ValueStruct,
    test: TestField,
    cell_field: NodalField,
    injected_field: SideTraceInjectedField,
    *,
    n: int,
    q: int,
    dim: int,
    test_uses_grad: bool,
    accumulate_dtype,
    injected_domain: SideConstantGeometryDomain,
):
    """Build the fused gather-formulation side kernel body for a qualifying linear form.

    Returns a ``kernel_fn`` closure to be compiled through
    ``cache.get_integrand_kernel`` with a
    :class:`warp._src.fem.sumfac.kernels.SumfacQPFieldsTransformer` and a
    ``PassFieldArgsToIntegrand`` (with ``fields_var_name="qp_fields"``) in its
    code transformers, exactly like the volume linear factory.

    The kernel is launched with ``wp.launch_tiled(dim=[active_cell_count])``,
    one cell per block. Per cell it runs one *dynamic* loop over the ``dim``
    element axes (the runtime ``two`` argument keeps every loop bound
    runtime-valued so tile temporaries are emitted -- and their shared memory
    allocated -- once instead of per unrolled iteration); each iteration
    processes the cell's two opposing faces along that axis with *stacked*
    operators (the perf-round redesign; see
    ``design/sumfac-phase6-perf-analysis.md``):

    1. **B stage** -- the endpoint value and derivative rows of *both*
       opposing faces form the single ``(4, n)`` ``end_ops`` operator, applied
       to the concatenation ``[own | neighbor(end 0) | neighbor(end 1)]`` of
       the axis-major element DOF tensors in one GEMM (the own-cell tensor is
       loaded once per cell). One further GEMM against the stacked
       ``(n, 4q)`` transposed tangential operator (interpolation, derivative,
       and their longitude-flipped variants) yields every trace channel of
       every face/neighbor at the side quadrature points; the per-face flip
       and inner/outer role pick rows/columns at read time.
    2. **D stage** -- per face quadrature point, the transformed integrand is
       evaluated with the test field replaced by a
       :class:`warp._src.fem.field.SideSeedField`, the input field by a
       :class:`warp._src.fem.field.SideTraceInjectedField` filled from the
       B-stage trace tile, and the ``Domain`` argument by a
       :class:`SideConstantGeometryDomain` whose normal/measure/measure-ratio
       members are evaluated ONCE per face (they are side constants on grids)
       instead of re-decoded inside every seeded evaluation; the inverse cell
       Jacobians and the side measure used for the quadrature scale are
       hoisted per face the same way. Faces outside the integration domain
       are skipped with block-uniform branches (axis level and face level)
       rather than computed-and-zeroed. The extracted test-channel
       coefficients are written into a flip-*padded* coefficient tile (each
       face's block sits at its flip's column/row offset, the other variant's
       half stays zero).
    3. **B^T stage** -- the padded layout lets both faces of the axis lift
       through shared operands: one GEMM against the full tangential operator
       and one against the ``(n, 4)`` transposed endpoint operator accumulate
       the axis residual (the zero pad halves absorb the per-face flip
       selection without per-face GEMMs).

    This brings the per-cell ``tile_matmul`` count from 24 to 8 in 2D (and
    from ~100 to ~21 in 3D) with identical math per coefficient -- only
    floating-point summation order changes (well inside the 1e-9 oracle
    tolerance of the side-apply tests).

    Tile-view discipline: NO ``tile_view`` result is ever passed to
    ``tile_matmul`` (the known cuBLASDx strided-operand hazard -- see
    ``design/sumfac-status.md``). Views appear only as ``tile_assign``
    sources (element-wise copies, stride-safe) and as full-width row blocks
    reshaped for assignment; every GEMM operand is an owned, contiguous tile.

    Args:
        integrand_func: Transformed integrand (with the seed/injected fields
            substituted).
        domain: Side domain of the integration.
        quadrature: Tensor-product side quadrature formula.
        FieldStruct: Generated field-argument struct (post-substitution).
        ValueStruct: Generated value-argument struct.
        test: Original (plain) side test field.
        cell_field: Cell-level nodal field underlying the input trace field,
            used for the in-kernel element DOF gather.
        injected_field: The :class:`warp._src.fem.field.SideTraceInjectedField`
            standing in for the input field; provides the per-quadrature-point
            ``EvalArg``.
        n: Nodes per axis of the (test and input) space, ``degree + 1``.
        q: Quadrature points per side axis.
        dim: Spatial dimension (2 or 3).
        test_uses_grad: Whether the integrand applies a gradient operator to
            the test field; when ``False`` the gradient seeds are omitted.
        accumulate_dtype: Scalar type used for the tile contractions and
            coefficient accumulation.
        injected_domain: The :class:`SideConstantGeometryDomain` standing in
            for the integrand's ``Domain`` argument; provides the per-face
            injected ``ElementArg``.
    """
    geometry = domain.geometry
    space = test.space
    SampleType = geometry.sample_type
    scalar_type = geometry.scalar_type
    coords_type = cached_coords_type(scalar_type)
    value_type = injected_field.dtype
    grad_type = injected_field.gradient_dtype
    InjectedEvalArg = injected_field.EvalArg
    InjectedGeoArg = injected_domain.ElementArg

    inner_grad_transform = space.element_inner_reference_gradient_transform
    outer_grad_transform = space.element_outer_reference_gradient_transform

    OUTER_VALUE_SEED = wp.constant(1 + dim)
    OUTER_GRAD_BEGIN = wp.constant(2 + dim)

    meta_int2 = cache.cached_vec_type(length=2, dtype=int)
    meta_scalar2 = cache.cached_vec_type(length=2, dtype=accumulate_dtype)
    acc_grad_vec = cache.cached_vec_type(length=dim, dtype=accumulate_dtype)

    n_c = wp.constant(n)
    q_c = wp.constant(q)
    q2_c = wp.constant(2 * q)
    q3_c = wp.constant(3 * q)
    q4_c = wp.constant(4 * q)
    nn_c = wp.constant(n * n)
    n3_c = wp.constant(3 * n)

    if dim == 2:

        def integrate_kernel_fn(
            qp_arg: quadrature.Arg,
            domain_arg: domain.ElementArg,
            domain_index_arg: domain.ElementIndexArg,
            fields: FieldStruct,
            values: ValueStruct,
            input_eval_arg: cell_field.EvalArg,
            end_ops: wp.array2d(dtype=accumulate_dtype),
            end_ops_t: wp.array2d(dtype=accumulate_dtype),
            tang_ops: wp.array2d(dtype=accumulate_dtype),
            tang_ops_t: wp.array2d(dtype=accumulate_dtype),
            face_map: wp.array2d(dtype=int),
            active_cells: wp.array(dtype=int),
            two: int,
            result_elem: wp.array2d(dtype=accumulate_dtype),
        ):
            block_index = wp.tid()
            cell_index = active_cells[block_index]

            E_all = wp.tile_load(end_ops, shape=(4, n_c))  # rows [v(e0); d(e0); v(e1); d(e1)]
            Et_all = wp.tile_load(end_ops_t, shape=(n_c, 4))
            W_all = wp.tile_load(tang_ops, shape=(q4_c, n_c))  # rows [interp; deriv; interp_f; deriv_f]
            Wt_all = wp.tile_load(tang_ops_t, shape=(n_c, q4_c))

            # Own-cell DOF tensor, loaded ONCE per cell
            u_own_flat = wp.tile_load(input_eval_arg.dof_values, shape=(nn_c,), offset=(cell_index * nn_c,))
            U_own = wp.tile_astype(wp.tile_reshape(u_own_flat, shape=(n_c, n_c)), dtype=accumulate_dtype)

            r_a = wp.tile_zeros(shape=(n_c, n_c), dtype=accumulate_dtype)  # axis-0 faces: [i, j]
            r_b = wp.tile_zeros(shape=(n_c, n_c), dtype=accumulate_dtype)  # axis-1 faces: [j, i]

            qp_fields = FieldStruct()
            _copy_qp_fields()

            # Injected domain arg: side-constant geometry (normal, measure,
            # measure ratio) is evaluated once per face below and read by the
            # integrand through the SideConstantGeometryDomain stand-in.
            qp_domain_geo = InjectedGeoArg()
            qp_domain_geo.base = domain_arg
            face_center = coords_type(scalar_type(0.5), scalar_type(0.5), scalar_type(0.0))

            # Dynamic axis loop (runtime bound): tile temporaries below are
            # emitted/allocated once, not per unrolled iteration.
            for axis in range(two):
                # ---- per-face metadata (block-uniform scalars) ------------
                dse2 = meta_int2()
                side2 = meta_int2()
                nbend2 = meta_int2()
                flip2 = meta_int2()
                bnd2 = meta_int2()
                w_in2 = meta_scalar2()
                w_out2 = meta_scalar2()

                # face end 0 (own cell is the side's outer cell on interior sides)
                dse_raw0 = face_map[cell_index, 2 * axis + 0]
                active0 = dse_raw0 >= 0
                dse0 = wp.max(dse_raw0, 0)
                side_index0 = domain.element_index(domain_index_arg, dse0)
                in_c0 = geometry.side_inner_cell_index(domain_arg, side_index0)
                out_c0 = geometry.side_outer_cell_index(domain_arg, side_index0)
                boundary0 = in_c0 == out_c0
                nb_cell0 = in_c0  # == cell_index on boundary sides
                dse2[0] = dse0
                side2[0] = side_index0
                nbend2[0] = wp.where(boundary0, 0, 1)
                bnd2[0] = wp.where(boundary0, 1, 0)
                # 2D longitude flip: (axis == 0) == (altitude == 0)
                flip2[0] = wp.where(axis == 0, wp.where(boundary0, 1, 0), wp.where(boundary0, 0, 1))
                w_in2[0] = wp.where(active0 and boundary0, accumulate_dtype(1.0), accumulate_dtype(0.0))
                w_out2[0] = wp.where(active0, accumulate_dtype(1.0), accumulate_dtype(0.0))

                # face end 1 (own cell is the side's inner cell on interior sides)
                dse_raw1 = face_map[cell_index, 2 * axis + 1]
                active1 = dse_raw1 >= 0
                dse1 = wp.max(dse_raw1, 0)
                side_index1 = domain.element_index(domain_index_arg, dse1)
                in_c1 = geometry.side_inner_cell_index(domain_arg, side_index1)
                out_c1 = geometry.side_outer_cell_index(domain_arg, side_index1)
                boundary1 = in_c1 == out_c1
                nb_cell1 = out_c1
                dse2[1] = dse1
                side2[1] = side_index1
                nbend2[1] = wp.where(boundary1, 1, 0)
                bnd2[1] = wp.where(boundary1, 1, 0)
                flip2[1] = wp.where(axis == 0, 0, 1)
                w_in2[1] = wp.where(active1, accumulate_dtype(1.0), accumulate_dtype(0.0))
                w_out2[1] = wp.where(active1 and boundary1, accumulate_dtype(1.0), accumulate_dtype(0.0))

                act0 = wp.where(active0, 1, 0)
                act1 = wp.where(active1, 1, 0)

                # Skip the whole axis when neither face belongs to the
                # integration domain (block-uniform branch; e.g. on
                # BoundarySides most of a boundary cell's faces are inactive).
                if active0 or active1:
                    # Hoisted side-constant geometry (constant per side on the
                    # qualifying grid geometries): normal, side measure,
                    # measure ratio, and the inverse cell Jacobians.
                    fs0 = make_free_sample(side_index0, face_center)
                    nor0 = domain.element_normal(domain_arg, fs0)
                    meas0 = domain.element_measure(domain_arg, fs0)
                    ratio0 = domain.element_measure_ratio(domain_arg, fs0)
                    xf_in0 = inner_grad_transform(domain_arg, fs0)
                    xf_out0 = outer_grad_transform(domain_arg, fs0)
                    fs1 = make_free_sample(side_index1, face_center)
                    nor1 = domain.element_normal(domain_arg, fs1)
                    meas1 = domain.element_measure(domain_arg, fs1)
                    ratio1 = domain.element_measure_ratio(domain_arg, fs1)
                    xf_in1 = inner_grad_transform(domain_arg, fs1)
                    xf_out1 = outer_grad_transform(domain_arg, fs1)

                    # ---- B stage: stacked endpoint + tangential contractions ---
                    # U_cat = [own | nb(end 0) | nb(end 1)] in axis-major layout
                    # (the tangential node axis is the column axis of each block).
                    U_cat = wp.tile_zeros(shape=(n_c, n3_c), dtype=accumulate_dtype)
                    nb0_flat = wp.tile_load(input_eval_arg.dof_values, shape=(nn_c,), offset=(nb_cell0 * nn_c,))
                    NB0 = wp.tile_astype(wp.tile_reshape(nb0_flat, shape=(n_c, n_c)), dtype=accumulate_dtype)
                    nb1_flat = wp.tile_load(input_eval_arg.dof_values, shape=(nn_c,), offset=(nb_cell1 * nn_c,))
                    NB1 = wp.tile_astype(wp.tile_reshape(nb1_flat, shape=(n_c, n_c)), dtype=accumulate_dtype)
                    if axis == 0:
                        wp.tile_assign(U_cat, U_own, offset=(0, 0))
                        wp.tile_assign(U_cat, NB0, offset=(0, n_c))
                        wp.tile_assign(U_cat, NB1, offset=(0, 2 * n_c))
                    else:
                        wp.tile_assign(U_cat, wp.tile_transpose(U_own), offset=(0, 0))
                        wp.tile_assign(U_cat, wp.tile_transpose(NB0), offset=(0, n_c))
                        wp.tile_assign(U_cat, wp.tile_transpose(NB1), offset=(0, 2 * n_c))

                    # Endpoint traces of all three tensors at BOTH ends in one GEMM
                    T = wp.tile_matmul(E_all, U_cat)  # (4, 3n)

                    # Restack per-tensor column blocks into rows for the shared
                    # tangential GEMM (tile_assign accepts the strided views).
                    TS = wp.tile_zeros(shape=(12, n_c), dtype=accumulate_dtype)
                    for b in range(3):
                        wp.tile_assign(TS, wp.tile_view(T, offset=(0, b * n_c), shape=(4, n_c)), offset=(4 * b, 0))

                    # Every trace channel at the side QPs, both flip variants:
                    # rows = [own; nb0; nb1] x [v(e0); d(e0); v(e1); d(e1)],
                    # cols = [interp; deriv; interp_f; deriv_f] blocks of q.
                    V = wp.tile_matmul(TS, Wt_all)  # (12, 4q)

                    # ---- D stage: seeded integrand evaluations per face QP -----
                    # Flip-padded coefficient tile: rows [c0|ct (end 0); cn (end 0);
                    # c0|ct (end 1); cn (end 1)], each face's block at its flip's
                    # column offset (the other half stays zero).
                    CF = wp.tile_zeros(shape=(4, q4_c), dtype=accumulate_dtype)

                    for it in range(two * q_c):
                        e = it // q_c
                        r = it - e * q_c
                        # Skip faces outside the integration domain entirely
                        # (block-uniform branch).
                        active_f = wp.where(e == 0, act0, act1)
                        if active_f == 1:
                            tang_axis = 1 - axis
                            dse = dse2[e]
                            side_index = side2[e]
                            w_in = w_in2[e]
                            w_out = w_out2[e]
                            col = flip2[e] * q2_c

                            qp_index = quadrature.point_index(domain_arg, qp_arg, dse, side_index, r)
                            qp_coords = quadrature.point_coords(domain_arg, qp_arg, dse, side_index, r)
                            qp_weight = quadrature.point_weight(domain_arg, qp_arg, dse, side_index, r)

                            # Per-face hoisted geometry (side constants on grids)
                            meas = wp.where(e == 0, meas0, meas1)
                            scale = accumulate_dtype(qp_weight * meas)
                            xf_in = wp.where(e == 0, xf_in0, xf_in1)  # J^{-1} of the inner cell
                            xf_out = wp.where(e == 0, xf_out0, xf_out1)
                            qp_domain_geo.normal = wp.where(e == 0, nor0, nor1)
                            qp_domain_geo.measure = meas
                            qp_domain_geo.measure_ratio = wp.where(e == 0, ratio0, ratio1)

                            # Trace-tile rows of this face's inner/outer roles
                            own_v = 2 * e
                            nb_v = 4 + 4 * e + 2 * nbend2[e]
                            own_role_inner = bnd2[e] == 1 or e == 1
                            in_v = wp.where(own_role_inner, own_v, nb_v)
                            out_v = wp.where(bnd2[e] == 1 or e == 0, own_v, nb_v)

                            # Inject the B-stage traces (value + physical gradient, inner and outer)
                            inj_arg = InjectedEvalArg()
                            inj_arg.inner_value = value_type(V[in_v, col + r])
                            g_in = grad_type()
                            g_in[axis] = value_type(V[in_v + 1, col + r])
                            g_in[tang_axis] = value_type(V[in_v, col + q_c + r])
                            inj_arg.inner_gradient = wp.transpose(xf_in) * g_in
                            inj_arg.outer_value = value_type(V[out_v, col + r])
                            g_out = grad_type()
                            g_out[axis] = value_type(V[out_v + 1, col + r])
                            g_out[tang_axis] = value_type(V[out_v, col + q_c + r])
                            inj_arg.outer_gradient = wp.transpose(xf_out) * g_out
                            _set_injected_eval_arg()

                            # Seeded test-channel extraction; this cell lifts its own
                            # role's channels (both roles on boundary sides).
                            c0 = accumulate_dtype(0.0)
                            cg = acc_grad_vec()
                            if w_in != accumulate_dtype(0.0):
                                sample = SampleType(
                                    side_index, qp_coords, qp_index, qp_weight, DofIndex(0, 0), NULL_DOF_INDEX
                                )
                                c0 += w_in * accumulate_dtype(integrand_func(sample, qp_fields, values))
                                if wp.static(test_uses_grad):
                                    fg = grad_type()
                                    for seed in range(2):
                                        sample = SampleType(
                                            side_index,
                                            qp_coords,
                                            qp_index,
                                            qp_weight,
                                            DofIndex(seed + 1, 0),
                                            NULL_DOF_INDEX,
                                        )
                                        fg[seed] = value_type(integrand_func(sample, qp_fields, values))
                                    fg_ref = xf_in * fg
                                    for k in range(2):
                                        cg[k] += w_in * accumulate_dtype(fg_ref[k])
                            if w_out != accumulate_dtype(0.0):
                                sample = SampleType(
                                    side_index,
                                    qp_coords,
                                    qp_index,
                                    qp_weight,
                                    DofIndex(OUTER_VALUE_SEED, 0),
                                    NULL_DOF_INDEX,
                                )
                                c0 += w_out * accumulate_dtype(integrand_func(sample, qp_fields, values))
                                if wp.static(test_uses_grad):
                                    fg = grad_type()
                                    for seed in range(2):
                                        sample = SampleType(
                                            side_index,
                                            qp_coords,
                                            qp_index,
                                            qp_weight,
                                            DofIndex(OUTER_GRAD_BEGIN + seed, 0),
                                            NULL_DOF_INDEX,
                                        )
                                        fg[seed] = value_type(integrand_func(sample, qp_fields, values))
                                    fg_ref = xf_out * fg
                                    for k in range(2):
                                        cg[k] += w_out * accumulate_dtype(fg_ref[k])

                            CF[2 * e, col + r] = scale * c0
                            if wp.static(test_uses_grad):
                                CF[2 * e, col + q_c + r] = scale * cg[tang_axis]
                                CF[2 * e + 1, col + r] = scale * cg[axis]

                    # ---- B^T stage: padded lift, both faces in two GEMMs -------
                    Y = wp.tile_matmul(CF, W_all)  # (4, n): [c0-lift(e0); cn-lift(e0); c0-lift(e1); cn-lift(e1)]
                    if axis == 0:
                        wp.tile_matmul(Et_all, Y, r_a)
                    else:
                        wp.tile_matmul(Et_all, Y, r_b)

            # Combine the two face orientations: r[i, j] = r_a[i, j] + r_b[j, i].
            # Written into a FRESH tile: an in-place r_a update would be a
            # read-modify-write on shared memory racing across the redundant
            # per-thread element writes (one warp's store can land before
            # another's load of the same element -- caught by racecheck).
            r_c = wp.tile_zeros(shape=(n_c, n_c), dtype=accumulate_dtype)
            for t in range(nn_c):
                i = t // n_c
                j = t - i * n_c
                r_c[i, j] = r_a[i, j] + r_b[j, i]

            r_flat = wp.tile_reshape(r_c, shape=(1, nn_c))
            wp.tile_store(result_elem, r_flat, offset=(block_index, 0))

        return integrate_kernel_fn

    # ----------------------------- 3D -------------------------------------
    # Same staged structure as 2D with rank-2 face tensors. Per axis the
    # element DOF tensors are permuted to axis-major layout [a, rest]; the
    # face frame is the SORTED pair of remaining axes (t1 < t2), so the side
    # frame's cyclic order means: axis 0 -> (t1, t2) = (longitude, latitude),
    # axis 1 -> (latitude, longitude) [swapped], axis 2 -> (longitude,
    # latitude); the longitude flip applies to rows (t1) for axes 0/2 and to
    # columns (t2) for axis 1, selected at runtime.
    nnn_c = wp.constant(n * n * n)
    qq_c = wp.constant(q * q)
    n12_c = wp.constant(12 * n)
    q16_c = wp.constant(16 * q)
    nn3_c = wp.constant(3 * n * n)
    n4_c = wp.constant(4 * n)
    q6_c = wp.constant(6 * q)

    def integrate_kernel_fn(
        qp_arg: quadrature.Arg,
        domain_arg: domain.ElementArg,
        domain_index_arg: domain.ElementIndexArg,
        fields: FieldStruct,
        values: ValueStruct,
        input_eval_arg: cell_field.EvalArg,
        end_ops: wp.array2d(dtype=accumulate_dtype),
        end_ops_t: wp.array2d(dtype=accumulate_dtype),
        tang_ops: wp.array2d(dtype=accumulate_dtype),
        tang_ops_t: wp.array2d(dtype=accumulate_dtype),
        face_map: wp.array2d(dtype=int),
        active_cells: wp.array(dtype=int),
        two: int,
        result_elem: wp.array2d(dtype=accumulate_dtype),
    ):
        block_index = wp.tid()
        cell_index = active_cells[block_index]

        E_all = wp.tile_load(end_ops, shape=(4, n_c))  # rows [v(e0); d(e0); v(e1); d(e1)]
        Et_all = wp.tile_load(end_ops_t, shape=(n_c, 4))
        W_all = wp.tile_load(tang_ops, shape=(q4_c, n_c))  # rows [interp; deriv; interp_f; deriv_f]
        Wt_all = wp.tile_load(tang_ops_t, shape=(n_c, q4_c))

        # Own-cell DOF tensor, loaded ONCE per cell, in all three axis-major
        # layouts: rows [a*n : (a+1)*n] hold the [x_a, rest] layout.
        u_own_flat = wp.tile_load(input_eval_arg.dof_values, shape=(nnn_c,), offset=(cell_index * nnn_c,))
        U_own = wp.tile_astype(wp.tile_reshape(u_own_flat, shape=(n_c, nn_c)), dtype=accumulate_dtype)  # [i, jk]
        OWN3 = wp.tile_zeros(shape=(n3_c, nn_c), dtype=accumulate_dtype)
        wp.tile_assign(OWN3, U_own, offset=(0, 0))
        for i in range(n_c):
            wp.tile_assign(
                OWN3,
                wp.tile_reshape(wp.tile_view(U_own, offset=(i, 0), shape=(1, nn_c)), shape=(n_c, n_c)),
                offset=(n_c, i * n_c),
            )  # [j, ik]
        wp.tile_assign(
            OWN3, wp.tile_transpose(wp.tile_reshape(U_own, shape=(nn_c, n_c))), offset=(2 * n_c, 0)
        )  # [k, ij]

        r_ax0 = wp.tile_zeros(shape=(n_c, nn_c), dtype=accumulate_dtype)  # [i, jk]
        r_ax1 = wp.tile_zeros(shape=(n_c, nn_c), dtype=accumulate_dtype)  # [j, ik]
        r_ax2 = wp.tile_zeros(shape=(n_c, nn_c), dtype=accumulate_dtype)  # [k, ij]

        qp_fields = FieldStruct()
        _copy_qp_fields()

        # Injected domain arg: side-constant geometry (normal, measure,
        # measure ratio) is evaluated once per face below and read by the
        # integrand through the SideConstantGeometryDomain stand-in.
        qp_domain_geo = InjectedGeoArg()
        qp_domain_geo.base = domain_arg
        face_center = coords_type(scalar_type(0.5), scalar_type(0.5), scalar_type(0.0))

        # Dynamic axis loop (runtime bound = dim): tile temporaries below are
        # emitted/allocated once, not per unrolled iteration.
        for axis in range(two + 1):
            # ---- per-face metadata (block-uniform scalars) -----------------
            dse2 = meta_int2()
            side2 = meta_int2()
            nbend2 = meta_int2()
            flip2 = meta_int2()
            bnd2 = meta_int2()
            w_in2 = meta_scalar2()
            w_out2 = meta_scalar2()

            dse_raw0 = face_map[cell_index, 2 * axis + 0]
            active0 = dse_raw0 >= 0
            dse0 = wp.max(dse_raw0, 0)
            side_index0 = domain.element_index(domain_index_arg, dse0)
            in_c0 = geometry.side_inner_cell_index(domain_arg, side_index0)
            out_c0 = geometry.side_outer_cell_index(domain_arg, side_index0)
            boundary0 = in_c0 == out_c0
            nb_cell0 = in_c0
            dse2[0] = dse0
            side2[0] = side_index0
            nbend2[0] = wp.where(boundary0, 0, 1)
            bnd2[0] = wp.where(boundary0, 1, 0)
            flip2[0] = wp.where(boundary0, 1, 0)  # 3D longitude flip: altitude == 0
            w_in2[0] = wp.where(active0 and boundary0, accumulate_dtype(1.0), accumulate_dtype(0.0))
            w_out2[0] = wp.where(active0, accumulate_dtype(1.0), accumulate_dtype(0.0))

            dse_raw1 = face_map[cell_index, 2 * axis + 1]
            active1 = dse_raw1 >= 0
            dse1 = wp.max(dse_raw1, 0)
            side_index1 = domain.element_index(domain_index_arg, dse1)
            in_c1 = geometry.side_inner_cell_index(domain_arg, side_index1)
            out_c1 = geometry.side_outer_cell_index(domain_arg, side_index1)
            boundary1 = in_c1 == out_c1
            nb_cell1 = out_c1
            dse2[1] = dse1
            side2[1] = side_index1
            nbend2[1] = wp.where(boundary1, 1, 0)
            bnd2[1] = wp.where(boundary1, 1, 0)
            flip2[1] = 0
            w_in2[1] = wp.where(active1, accumulate_dtype(1.0), accumulate_dtype(0.0))
            w_out2[1] = wp.where(active1 and boundary1, accumulate_dtype(1.0), accumulate_dtype(0.0))

            act2 = meta_int2()
            act2[0] = wp.where(active0, 1, 0)
            act2[1] = wp.where(active1, 1, 0)

            # Skip the whole axis when neither face belongs to the integration
            # domain (block-uniform branch; e.g. on BoundarySides most of a
            # boundary cell's faces are inactive).
            if active0 or active1:
                # Hoisted side-constant geometry (constant per side on the
                # qualifying grid geometries): normal, side measure, measure
                # ratio, and the inverse cell Jacobians.
                fs0 = make_free_sample(side_index0, face_center)
                nor0 = domain.element_normal(domain_arg, fs0)
                meas0 = domain.element_measure(domain_arg, fs0)
                ratio0 = domain.element_measure_ratio(domain_arg, fs0)
                xf_in0 = inner_grad_transform(domain_arg, fs0)
                xf_out0 = outer_grad_transform(domain_arg, fs0)
                fs1 = make_free_sample(side_index1, face_center)
                nor1 = domain.element_normal(domain_arg, fs1)
                meas1 = domain.element_measure(domain_arg, fs1)
                ratio1 = domain.element_measure_ratio(domain_arg, fs1)
                xf_in1 = inner_grad_transform(domain_arg, fs1)
                xf_out1 = outer_grad_transform(domain_arg, fs1)

                # ---- B stage: stacked endpoint + tangential contractions -------
                # U_cat = [own | nb0 | nb1] in axis-major layout
                U_cat = wp.tile_zeros(shape=(n_c, nn3_c), dtype=accumulate_dtype)
                wp.tile_assign(U_cat, wp.tile_view(OWN3, offset=(axis * n_c, 0), shape=(n_c, nn_c)), offset=(0, 0))
                nb0_flat = wp.tile_load(input_eval_arg.dof_values, shape=(nnn_c,), offset=(nb_cell0 * nnn_c,))
                NB0 = wp.tile_astype(wp.tile_reshape(nb0_flat, shape=(n_c, nn_c)), dtype=accumulate_dtype)  # [i, jk]
                nb1_flat = wp.tile_load(input_eval_arg.dof_values, shape=(nnn_c,), offset=(nb_cell1 * nnn_c,))
                NB1 = wp.tile_astype(wp.tile_reshape(nb1_flat, shape=(n_c, nn_c)), dtype=accumulate_dtype)
                if axis == 0:
                    wp.tile_assign(U_cat, NB0, offset=(0, nn_c))
                    wp.tile_assign(U_cat, NB1, offset=(0, 2 * nn_c))
                elif axis == 1:
                    for i in range(n_c):
                        wp.tile_assign(
                            U_cat,
                            wp.tile_reshape(wp.tile_view(NB0, offset=(i, 0), shape=(1, nn_c)), shape=(n_c, n_c)),
                            offset=(0, nn_c + i * n_c),
                        )
                        wp.tile_assign(
                            U_cat,
                            wp.tile_reshape(wp.tile_view(NB1, offset=(i, 0), shape=(1, nn_c)), shape=(n_c, n_c)),
                            offset=(0, 2 * nn_c + i * n_c),
                        )
                else:
                    wp.tile_assign(U_cat, wp.tile_transpose(wp.tile_reshape(NB0, shape=(nn_c, n_c))), offset=(0, nn_c))
                    wp.tile_assign(
                        U_cat, wp.tile_transpose(wp.tile_reshape(NB1, shape=(nn_c, n_c))), offset=(0, 2 * nn_c)
                    )

                # Endpoint traces of all three tensors at BOTH ends in one GEMM:
                # rows [v(e0); d(e0); v(e1); d(e1)], each row 3 face tensors [t1, t2]
                T = wp.tile_matmul(E_all, U_cat)  # (4, 3nn)

                # Contract the SECOND face axis (t2) of every face tensor with the
                # full stacked tangential operator (both flips) in one GEMM: the
                # (12n, n) reshape's row r = (channel * 3 + block) * n + t1.
                C1 = wp.tile_matmul(wp.tile_reshape(T, shape=(n12_c, n_c)), Wt_all)  # (12n, 4q)

                # ---- per-face row (t1) contraction + D stage -------------------
                CBA = wp.tile_zeros(shape=(q4_c, q6_c), dtype=accumulate_dtype)  # flip-padded lift coefficients

                for e in range(two):
                    # Skip faces outside the integration domain entirely
                    # (block-uniform branch).
                    if act2[e] == 1:
                        nbe = nbend2[e]
                        # Row-flip / column-flip selection: the side longitude runs
                        # along t1 for axes 0/2 and along t2 for axis 1.
                        rflip = wp.where(axis == 1, 0, flip2[e])
                        cflip = wp.where(axis == 1, flip2[e], 0)

                        # The four needed face tensors as full-width row blocks of C1
                        C1sel = wp.tile_zeros(
                            shape=(n_c, q16_c), dtype=accumulate_dtype
                        )  # [v_own | d_own | v_nb | d_nb]
                        wp.tile_assign(
                            C1sel, wp.tile_view(C1, offset=(6 * e * n_c, 0), shape=(n_c, q4_c)), offset=(0, 0)
                        )
                        wp.tile_assign(
                            C1sel, wp.tile_view(C1, offset=((6 * e + 3) * n_c, 0), shape=(n_c, q4_c)), offset=(0, q4_c)
                        )
                        wp.tile_assign(
                            C1sel,
                            wp.tile_view(C1, offset=((6 * nbe + 1 + e) * n_c, 0), shape=(n_c, q4_c)),
                            offset=(0, 2 * q4_c),
                        )
                        wp.tile_assign(
                            C1sel,
                            wp.tile_view(C1, offset=((6 * nbe + 4 + e) * n_c, 0), shape=(n_c, q4_c)),
                            offset=(0, 3 * q4_c),
                        )

                        # Row (t1) contraction for all four tensors in one GEMM; rows
                        # of QF are [interp(q); deriv(q)] at this face's row flip.
                        rowop = wp.tile_load(tang_ops, shape=(q2_c, n_c), offset=(rflip * q2_c, 0))
                        QF = wp.tile_matmul(rowop, C1sel)  # (2q, 16q)

                        # ---- D stage: seeded integrand evaluations per face QP -----
                        dse = dse2[e]
                        side_index = side2[e]
                        w_in = w_in2[e]
                        w_out = w_out2[e]
                        t1_axis = wp.where(axis == 0, 1, 0)  # face-frame first element axis
                        t2_axis = wp.where(axis == 2, 1, 2)  # face-frame second element axis
                        qoff = cflip * q2_c
                        rbase = rflip * q2_c
                        cb = q3_c * e

                        # Per-face hoisted geometry (side constants on grids)
                        meas = wp.where(e == 0, meas0, meas1)
                        xf_in = wp.where(e == 0, xf_in0, xf_in1)  # J^{-1} of the inner cell
                        xf_out = wp.where(e == 0, xf_out0, xf_out1)
                        qp_domain_geo.normal = wp.where(e == 0, nor0, nor1)
                        qp_domain_geo.measure = meas
                        qp_domain_geo.measure_ratio = wp.where(e == 0, ratio0, ratio1)

                        for r in range(qq_c):
                            r0 = r // q_c
                            r1 = r - r0 * q_c
                            # t1/t2 quadrature indices: s0 (longitude, slow) runs along
                            # t1 for axes 0/2 and along t2 for axis 1.
                            ri = wp.where(axis == 1, r1, r0)
                            ci = wp.where(axis == 1, r0, r1)

                            qp_index = quadrature.point_index(domain_arg, qp_arg, dse, side_index, r)
                            qp_coords = quadrature.point_coords(domain_arg, qp_arg, dse, side_index, r)
                            qp_weight = quadrature.point_weight(domain_arg, qp_arg, dse, side_index, r)
                            scale = accumulate_dtype(qp_weight * meas)

                            # Column-block bases of this face's inner/outer roles in QF
                            own_role_inner = bnd2[e] == 1 or e == 1
                            in_vb = wp.where(own_role_inner, 0, 2 * q4_c)
                            out_vb = wp.where(bnd2[e] == 1 or e == 0, 0, 2 * q4_c)

                            # Inject the B-stage traces (value + physical gradient, inner and outer)
                            inj_arg = InjectedEvalArg()
                            inj_arg.inner_value = value_type(QF[ri, in_vb + qoff + ci])
                            g_in = grad_type()
                            g_in[axis] = value_type(QF[ri, in_vb + q4_c + qoff + ci])
                            g_in[t1_axis] = value_type(QF[q_c + ri, in_vb + qoff + ci])
                            g_in[t2_axis] = value_type(QF[ri, in_vb + qoff + q_c + ci])
                            inj_arg.inner_gradient = wp.transpose(xf_in) * g_in
                            inj_arg.outer_value = value_type(QF[ri, out_vb + qoff + ci])
                            g_out = grad_type()
                            g_out[axis] = value_type(QF[ri, out_vb + q4_c + qoff + ci])
                            g_out[t1_axis] = value_type(QF[q_c + ri, out_vb + qoff + ci])
                            g_out[t2_axis] = value_type(QF[ri, out_vb + qoff + q_c + ci])
                            inj_arg.outer_gradient = wp.transpose(xf_out) * g_out
                            _set_injected_eval_arg()

                            # Seeded test-channel extraction; this cell lifts its own
                            # role's channels (both roles on boundary sides).
                            c0 = accumulate_dtype(0.0)
                            cg = acc_grad_vec()
                            if w_in != accumulate_dtype(0.0):
                                sample = SampleType(
                                    side_index, qp_coords, qp_index, qp_weight, DofIndex(0, 0), NULL_DOF_INDEX
                                )
                                c0 += w_in * accumulate_dtype(integrand_func(sample, qp_fields, values))
                                if wp.static(test_uses_grad):
                                    fg = grad_type()
                                    for seed in range(3):
                                        sample = SampleType(
                                            side_index,
                                            qp_coords,
                                            qp_index,
                                            qp_weight,
                                            DofIndex(seed + 1, 0),
                                            NULL_DOF_INDEX,
                                        )
                                        fg[seed] = value_type(integrand_func(sample, qp_fields, values))
                                    fg_ref = xf_in * fg
                                    for k in range(3):
                                        cg[k] += w_in * accumulate_dtype(fg_ref[k])
                            if w_out != accumulate_dtype(0.0):
                                sample = SampleType(
                                    side_index,
                                    qp_coords,
                                    qp_index,
                                    qp_weight,
                                    DofIndex(OUTER_VALUE_SEED, 0),
                                    NULL_DOF_INDEX,
                                )
                                c0 += w_out * accumulate_dtype(integrand_func(sample, qp_fields, values))
                                if wp.static(test_uses_grad):
                                    fg = grad_type()
                                    for seed in range(3):
                                        sample = SampleType(
                                            side_index,
                                            qp_coords,
                                            qp_index,
                                            qp_weight,
                                            DofIndex(OUTER_GRAD_BEGIN + seed, 0),
                                            NULL_DOF_INDEX,
                                        )
                                        fg[seed] = value_type(integrand_func(sample, qp_fields, values))
                                    fg_ref = xf_out * fg
                                    for k in range(3):
                                        cg[k] += w_out * accumulate_dtype(fg_ref[k])

                            # Lift coefficients in the face frame, flip-padded: this
                            # face's CB block sits at rows rbase..rbase+2q of columns
                            # cb..cb+2q ([C0 | Ct2; Ct1 | 0]) and its CA block (normal
                            # derivative, interp rows both sides) at columns cb+2q.
                            CBA[rbase + ri, cb + ci] = scale * c0
                            if wp.static(test_uses_grad):
                                CBA[rbase + ri, cb + q_c + ci] = scale * cg[t2_axis]
                                CBA[rbase + q_c + ri, cb + ci] = scale * cg[t1_axis]
                                CBA[rbase + ri, cb + q2_c + ci] = scale * cg[axis]

                # ---- B^T stage: padded lift, both faces in three GEMMs ---------
                # X2 = tang_ops^T @ CBA: per face block, X = S_row(rflip)^T @ CB
                # (n, 2q) and Y = I_row(rflip)^T @ CA (n, q).
                X2 = wp.tile_matmul(Wt_all, CBA)  # (n, 6q)

                # Pad the column-side flip: X rows at this face's cflip offset
                # contract with S_col(cflip), Y rows with I_col(cflip).
                XY = wp.tile_zeros(shape=(n4_c, q4_c), dtype=accumulate_dtype)
                cflip0 = wp.where(axis == 1, flip2[0], 0)
                cflip1 = wp.where(axis == 1, flip2[1], 0)
                wp.tile_assign(XY, wp.tile_view(X2, offset=(0, 0), shape=(n_c, q2_c)), offset=(0, cflip0 * q2_c))
                wp.tile_assign(XY, wp.tile_view(X2, offset=(0, q2_c), shape=(n_c, q_c)), offset=(n_c, cflip0 * q2_c))
                wp.tile_assign(
                    XY, wp.tile_view(X2, offset=(0, q3_c), shape=(n_c, q2_c)), offset=(2 * n_c, cflip1 * q2_c)
                )
                wp.tile_assign(
                    XY, wp.tile_view(X2, offset=(0, q3_c + q2_c), shape=(n_c, q_c)), offset=(3 * n_c, cflip1 * q2_c)
                )
                AvAn = wp.tile_matmul(XY, W_all)  # (4n, n): [Av(e0); An(e0); Av(e1); An(e1)]

                # Endpoint lift of both faces in one GEMM: end_ops^T rows pair
                # [v(e0); d(e0); v(e1); d(e1)] with [Av(e0); An(e0); Av(e1); An(e1)].
                Z = wp.tile_zeros(shape=(4, nn_c), dtype=accumulate_dtype)
                for c in range(4):
                    wp.tile_assign(
                        Z,
                        wp.tile_reshape(wp.tile_view(AvAn, offset=(c * n_c, 0), shape=(n_c, n_c)), shape=(1, nn_c)),
                        offset=(c, 0),
                    )
                if axis == 0:
                    wp.tile_matmul(Et_all, Z, r_ax0)
                elif axis == 1:
                    wp.tile_matmul(Et_all, Z, r_ax1)
                else:
                    wp.tile_matmul(Et_all, Z, r_ax2)

        # Combine the three face orientations:
        # r[i, j*n + k] = r_ax0[i, jk] + r_ax1[j, ik] + r_ax2[k, ij].
        # Written into a FRESH tile: an in-place r_ax0 update would be a
        # read-modify-write on shared memory racing across the redundant
        # per-thread element writes (one warp's store can land before
        # another's load of the same element -- caught by racecheck).
        r_c = wp.tile_zeros(shape=(n_c, nn_c), dtype=accumulate_dtype)
        for t in range(nnn_c):
            i = t // nn_c
            jk = t - i * nn_c
            j = jk // n_c
            k = jk - j * n_c
            r_c[i, jk] = r_ax0[i, jk] + r_ax1[j, i * n_c + k] + r_ax2[k, i * n_c + j]

        r_flat = wp.tile_reshape(r_c, shape=(1, nnn_c))
        wp.tile_store(result_elem, r_flat, offset=(block_index, 0))

    return integrate_kernel_fn


def get_sumfac_side_scatter_kernel(
    geometry,
    space,
    space_partition,
    staging_dtype,
    output_dtype,
):
    """Build (and cache) the kernel scattering per-active-cell residuals to partition nodes.

    For discontinuous spaces every node belongs to exactly one cell and every
    active cell appears exactly once in the launch space, so the accumulation
    is race-free without atomics; nodes outside the partition are skipped.

    Args:
        geometry: Geometry owning the cells.
        space: Test (trace) function space; its full-space topology provides
            the cell-to-node map.
        space_partition: Test space partition indexing the output vector rows.
        staging_dtype: Scalar type of the per-cell staging array.
        output_dtype: Scalar type of the output vector.
    """
    cell_topology = space.topology.full_space_topology()

    @cache.dynamic_kernel(
        suffix=(
            "sumfac_side_scatter",
            geometry.name,
            space.name,
            space_partition.name,
            cache.pod_type_key(staging_dtype),
            cache.pod_type_key(output_dtype),
        ),
        kernel_options={"enable_backward": False},
    )
    def sumfac_side_scatter_kernel(
        cell_arg: geometry.CellArg,
        topo_arg: cell_topology.TopologyArg,
        partition_arg: space_partition.PartitionArg,
        active_cells: wp.array(dtype=int),
        staging: wp.array2d(dtype=staging_dtype),
        result: wp.array2d(dtype=output_dtype),
    ):
        row, node_in_cell = wp.tid()
        cell_index = active_cells[row]
        node_index = cell_topology.element_node_index(cell_arg, topo_arg, cell_index, node_in_cell)
        partition_node_index = space_partition.partition_node_index(partition_arg, node_index)
        if partition_node_index == NULL_NODE_INDEX:
            return
        result[partition_node_index, 0] += output_dtype(staging[row, node_in_cell])

    return sumfac_side_scatter_kernel
