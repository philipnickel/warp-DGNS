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
    "SumfacSideLayout",
    "SumfacSidePlan",
    "find_sumfac_side_layout",
    "get_integrate_side_sumfac_kernel",
    "get_side_gather_arrays",
    "get_sumfac_side_scatter_kernel",
    "make_sumfac_side_plan",
]


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
    """

    test: TestField
    test_name: str
    input_name: str
    input_field: NodalField.Trace
    cell_field: NodalField
    seed_field: SideSeedField
    injected_field: SideTraceInjectedField
    degree: int
    n: int
    q: int
    dim: int
    test_uses_grad: bool
    end_ops: np.ndarray
    tang_ops: np.ndarray

    @property
    def nodes_per_element(self) -> int:
        """Number of nodes per cell, ``n**dim``."""
        return self.n**self.dim

    def operator_arrays(self, dtype, device) -> tuple[wp.array, wp.array]:
        """Return the ``(end_ops, tang_ops)`` device arrays in ``dtype`` on ``device``, cached."""
        return _get_operator_arrays(self.end_ops, self.tang_ops, dtype, device)


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


def make_sumfac_side_plan(
    integrand,
    arguments,
    test: TestField,
    quadrature: Quadrature,
    domain: GeometryDomain,
) -> SumfacSidePlan:
    """Build the side launch plan and substitute the seed/injected fields in ``arguments``.

    On success, ``arguments.field_args`` is mutated in place: the test field
    is replaced by a :class:`warp._src.fem.field.SideSeedField` and the input
    trace field by a :class:`warp._src.fem.field.SideTraceInjectedField`, so
    that the downstream ``FieldStruct`` / ``IntegrandTransformer`` machinery
    generates the in-kernel Q-function. Raises
    :class:`SumfacNotApplicableError` if the form does not qualify (the
    layout is validated before any mutation of ``arguments``).
    """
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

    return SumfacSidePlan(
        test=test,
        test_name=arguments.test_name,
        input_name=layout.input_name,
        input_field=layout.input_field,
        cell_field=layout.input_field.cell_field,
        seed_field=seed_field,
        injected_field=injected_field,
        degree=layout.degree,
        n=n,
        q=q,
        dim=layout.dim,
        test_uses_grad=layout.test_uses_grad,
        end_ops=end_ops,
        tang_ops=tang_ops,
    )


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
):
    """Build the fused gather-formulation side kernel body for a qualifying linear form.

    Returns a ``kernel_fn`` closure to be compiled through
    ``cache.get_integrand_kernel`` with a
    :class:`warp._src.fem.sumfac.kernels.SumfacQPFieldsTransformer` and a
    ``PassFieldArgsToIntegrand`` (with ``fields_var_name="qp_fields"``) in its
    code transformers, exactly like the volume linear factory.

    The kernel is launched with ``wp.launch_tiled(dim=[active_cell_count])``,
    one cell per block; per-face control flow is uniform over the block (faces
    outside the integration domain contribute zero through their role
    weights, never through divergent tile operations).

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
    """
    geometry = domain.geometry
    space = test.space
    SampleType = geometry.sample_type
    value_type = injected_field.dtype
    grad_type = injected_field.gradient_dtype
    InjectedEvalArg = injected_field.EvalArg

    inner_grad_transform = space.element_inner_reference_gradient_transform
    outer_grad_transform = space.element_outer_reference_gradient_transform

    OUTER_VALUE_SEED = wp.constant(1 + dim)
    OUTER_GRAD_BEGIN = wp.constant(2 + dim)

    face_count = 2 * dim
    meta_int_vec = cache.cached_vec_type(length=face_count, dtype=int)
    meta_scalar_vec = cache.cached_vec_type(length=face_count, dtype=accumulate_dtype)
    acc_grad_vec = cache.cached_vec_type(length=dim, dtype=accumulate_dtype)

    n_c = wp.constant(n)
    q_c = wp.constant(q)
    q2_c = wp.constant(2 * q)
    nn_c = wp.constant(n * n)

    if dim == 2:
        # Tile row layouts (2D): TR holds 6 trace channels per face
        # [u_in, du_in/dn(ref), du_in/dt(ref), u_out, du_out/dn, du_out/dt];
        # CF holds the 2 lift rows per face [[c0 | c_tang], [c_norm | 0]].
        tr_rows_c = wp.constant(4 * 6)
        cf_rows_c = wp.constant(4 * 2)
        dloop_c = wp.constant(4 * q)

        def integrate_kernel_fn(
            qp_arg: quadrature.Arg,
            domain_arg: domain.ElementArg,
            domain_index_arg: domain.ElementIndexArg,
            fields: FieldStruct,
            values: ValueStruct,
            input_eval_arg: cell_field.EvalArg,
            end_ops: wp.array2d(dtype=accumulate_dtype),
            tang_ops: wp.array2d(dtype=accumulate_dtype),
            face_map: wp.array2d(dtype=int),
            active_cells: wp.array(dtype=int),
            result_elem: wp.array2d(dtype=accumulate_dtype),
        ):
            block_index = wp.tid()
            cell_index = active_cells[block_index]

            # Runtime-valued loop bound (always 2): keeps the face/neighbor
            # loops dynamic so their tile temporaries are emitted (and their
            # shared memory allocated) once instead of per unrolled iteration.
            dyn2 = wp.min(2, cell_index + 2)

            # Per-face metadata recorded by the B stage for the D and B^T stages
            dse_v = meta_int_vec()
            side_v = meta_int_vec()
            flip_v = meta_int_vec()
            w_in_v = meta_scalar_vec()
            w_out_v = meta_scalar_vec()

            TR = wp.tile_zeros(shape=(tr_rows_c, q_c), dtype=accumulate_dtype)
            CF = wp.tile_zeros(shape=(cf_rows_c, q2_c), dtype=accumulate_dtype)

            # --- B stage, axis 0 faces (normal along element axis 0) -------
            for end in range(dyn2):
                face = end
                dse_raw = face_map[cell_index, face]
                active = dse_raw >= 0
                dse = wp.max(dse_raw, 0)
                side_index = domain.element_index(domain_index_arg, dse)
                inner_cell = geometry.side_inner_cell_index(domain_arg, side_index)
                outer_cell = geometry.side_outer_cell_index(domain_arg, side_index)
                boundary = inner_cell == outer_cell
                alt0 = boundary and end == 0
                inner_end = wp.where(alt0, 0, 1)
                outer_end = wp.where(boundary and end == 1, 1, 0)
                # 2D longitude flip: (axis == 0) == (altitude == 0)
                flip = wp.where(alt0, 1, 0)

                dse_v[face] = dse
                side_v[face] = side_index
                flip_v[face] = flip
                w_in_v[face] = wp.where(
                    active and inner_cell == cell_index, accumulate_dtype(1.0), accumulate_dtype(0.0)
                )
                w_out_v[face] = wp.where(
                    active and outer_cell == cell_index, accumulate_dtype(1.0), accumulate_dtype(0.0)
                )

                W_x0 = wp.tile_load(tang_ops, shape=(q2_c, n_c), offset=(flip * q2_c, 0))
                for nb in range(dyn2):
                    nb_cell = wp.where(nb == 0, inner_cell, outer_cell)
                    nb_end = wp.where(nb == 0, inner_end, outer_end)
                    u_flat_x0 = wp.tile_load(input_eval_arg.dof_values, shape=(nn_c,), offset=(nb_cell * nn_c,))
                    U_x0 = wp.tile_astype(wp.tile_reshape(u_flat_x0, shape=(n_c, n_c)), dtype=accumulate_dtype)
                    E_nb_x0 = wp.tile_load(end_ops, shape=(2, n_c), offset=(2 * nb_end, 0))
                    T_x0 = wp.tile_matmul(E_nb_x0, U_x0)  # (2, n): [value; d/dn] face nodal
                    V_x0 = wp.tile_matmul(T_x0, wp.tile_transpose(W_x0))  # (2, 2q)
                    for rr in range(q_c):
                        TR[face * 6 + nb * 3 + 0, rr] = V_x0[0, rr]
                        TR[face * 6 + nb * 3 + 1, rr] = V_x0[1, rr]
                        TR[face * 6 + nb * 3 + 2, rr] = V_x0[0, q_c + rr]

            # --- B stage, axis 1 faces --------------------------------------
            for end in range(dyn2):
                face = 2 + end
                dse_raw = face_map[cell_index, face]
                active = dse_raw >= 0
                dse = wp.max(dse_raw, 0)
                side_index = domain.element_index(domain_index_arg, dse)
                inner_cell = geometry.side_inner_cell_index(domain_arg, side_index)
                outer_cell = geometry.side_outer_cell_index(domain_arg, side_index)
                boundary = inner_cell == outer_cell
                alt0 = boundary and end == 0
                inner_end = wp.where(alt0, 0, 1)
                outer_end = wp.where(boundary and end == 1, 1, 0)
                # 2D longitude flip: (axis == 0) == (altitude == 0)
                flip = wp.where(alt0, 0, 1)

                dse_v[face] = dse
                side_v[face] = side_index
                flip_v[face] = flip
                w_in_v[face] = wp.where(
                    active and inner_cell == cell_index, accumulate_dtype(1.0), accumulate_dtype(0.0)
                )
                w_out_v[face] = wp.where(
                    active and outer_cell == cell_index, accumulate_dtype(1.0), accumulate_dtype(0.0)
                )

                W_x1 = wp.tile_load(tang_ops, shape=(q2_c, n_c), offset=(flip * q2_c, 0))
                for nb in range(dyn2):
                    nb_cell = wp.where(nb == 0, inner_cell, outer_cell)
                    nb_end = wp.where(nb == 0, inner_end, outer_end)
                    u_flat_x1 = wp.tile_load(input_eval_arg.dof_values, shape=(nn_c,), offset=(nb_cell * nn_c,))
                    U_x1 = wp.tile_astype(wp.tile_reshape(u_flat_x1, shape=(n_c, n_c)), dtype=accumulate_dtype)
                    E_nb_x1 = wp.tile_load(end_ops, shape=(2, n_c), offset=(2 * nb_end, 0))
                    T_x1 = wp.tile_matmul(E_nb_x1, wp.tile_transpose(U_x1))  # (2, n): face frame = axis 0
                    V_x1 = wp.tile_matmul(T_x1, wp.tile_transpose(W_x1))  # (2, 2q)
                    for rr in range(q_c):
                        TR[face * 6 + nb * 3 + 0, rr] = V_x1[0, rr]
                        TR[face * 6 + nb * 3 + 1, rr] = V_x1[1, rr]
                        TR[face * 6 + nb * 3 + 2, rr] = V_x1[0, q_c + rr]

            # --- D stage: seeded integrand evaluations per face quadrature point
            qp_fields = FieldStruct()
            _copy_qp_fields()

            for it in range(dloop_c):
                face = it // q_c
                r = it - face * q_c
                axis = face // 2
                tang_axis = 1 - axis
                dse = dse_v[face]
                side_index = side_v[face]
                w_in = w_in_v[face]
                w_out = w_out_v[face]

                qp_index = quadrature.point_index(domain_arg, qp_arg, dse, side_index, r)
                qp_coords = quadrature.point_coords(domain_arg, qp_arg, dse, side_index, r)
                qp_weight = quadrature.point_weight(domain_arg, qp_arg, dse, side_index, r)

                free_sample = make_free_sample(side_index, qp_coords)
                vol = domain.element_measure(domain_arg, free_sample)
                scale = accumulate_dtype(qp_weight * vol)
                xf_in = inner_grad_transform(domain_arg, free_sample)  # J^{-1} of the inner cell
                xf_out = outer_grad_transform(domain_arg, free_sample)

                # Inject the B-stage traces (value + physical gradient, inner and outer)
                base = face * 6
                inj_arg = InjectedEvalArg()
                inj_arg.inner_value = value_type(TR[base + 0, r])
                g_in = grad_type()
                g_in[axis] = value_type(TR[base + 1, r])
                g_in[tang_axis] = value_type(TR[base + 2, r])
                inj_arg.inner_gradient = wp.transpose(xf_in) * g_in
                inj_arg.outer_value = value_type(TR[base + 3, r])
                g_out = grad_type()
                g_out[axis] = value_type(TR[base + 4, r])
                g_out[tang_axis] = value_type(TR[base + 5, r])
                inj_arg.outer_gradient = wp.transpose(xf_out) * g_out
                _set_injected_eval_arg()

                # Seeded test-channel extraction; this cell lifts its own
                # role's channels (both roles on boundary sides).
                c0 = accumulate_dtype(0.0)
                cg = acc_grad_vec()
                if w_in != accumulate_dtype(0.0):
                    sample = SampleType(side_index, qp_coords, qp_index, qp_weight, DofIndex(0, 0), NULL_DOF_INDEX)
                    c0 += w_in * accumulate_dtype(integrand_func(sample, qp_fields, values))
                    if wp.static(test_uses_grad):
                        fg = grad_type()
                        for seed in range(2):
                            sample = SampleType(
                                side_index, qp_coords, qp_index, qp_weight, DofIndex(seed + 1, 0), NULL_DOF_INDEX
                            )
                            fg[seed] = value_type(integrand_func(sample, qp_fields, values))
                        fg_ref = xf_in * fg
                        for k in range(2):
                            cg[k] += w_in * accumulate_dtype(fg_ref[k])
                if w_out != accumulate_dtype(0.0):
                    sample = SampleType(
                        side_index, qp_coords, qp_index, qp_weight, DofIndex(OUTER_VALUE_SEED, 0), NULL_DOF_INDEX
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

                CF[face * 2 + 0, r] = scale * c0
                if wp.static(test_uses_grad):
                    CF[face * 2 + 0, q_c + r] = scale * cg[tang_axis]
                    CF[face * 2 + 1, r] = scale * cg[axis]

            # --- B^T stage: lift the coefficients back to cell nodal residuals
            r_a = wp.tile_zeros(shape=(n_c, n_c), dtype=accumulate_dtype)  # axis-0 faces: [i, j]
            r_b = wp.tile_zeros(shape=(n_c, n_c), dtype=accumulate_dtype)  # axis-1 faces: [j, i]

            for end in range(dyn2):
                E_own = wp.tile_load(end_ops, shape=(2, n_c), offset=(2 * end, 0))
                # axis 0 face
                face = end
                W0 = wp.tile_load(tang_ops, shape=(q2_c, n_c), offset=(flip_v[face] * q2_c, 0))
                Ff0 = wp.tile_view(CF, offset=(face * 2, 0), shape=(2, q2_c))
                X0 = wp.tile_matmul(wp.tile_transpose(E_own), Ff0)  # (n, 2q)
                wp.tile_matmul(X0, W0, r_a)
                # axis 1 face
                face = 2 + end
                W1 = wp.tile_load(tang_ops, shape=(q2_c, n_c), offset=(flip_v[face] * q2_c, 0))
                Ff1 = wp.tile_view(CF, offset=(face * 2, 0), shape=(2, q2_c))
                X1 = wp.tile_matmul(wp.tile_transpose(E_own), Ff1)  # (n, 2q)
                wp.tile_matmul(X1, W1, r_b)

            # Combine the two face orientations: r[i, j] = r_a[i, j] + r_b[j, i]
            for t in range(nn_c):
                i = t // n_c
                j = t - i * n_c
                r_a[i, j] = r_a[i, j] + r_b[j, i]

            r_flat = wp.tile_reshape(r_a, shape=(1, nn_c))
            wp.tile_store(result_elem, r_flat, offset=(block_index, 0))

        return integrate_kernel_fn

    # ----------------------------- 3D -------------------------------------
    # Tile row layouts: TR holds 8 trace channels per face
    # [u_in, du_in/dn, du_in/d(ga0), du_in/d(ga1), u_out, ...] at the qq side
    # QPs in canonical order r = r0 * q + r1; CB holds the (2q, 2q) lift
    # blocks [[C0, Ct_col], [Ct_row, 0]] per face in the face frame (rows =
    # face-frame-first axis); CA the (q, q) normal-derivative coefficients.
    nnn_c = wp.constant(n * n * n)
    qq_c = wp.constant(q * q)
    tr_rows_c = wp.constant(6 * 8)
    cb_rows_c = wp.constant(6 * 2 * q)
    ca_rows_c = wp.constant(6 * q)
    dloop_c = wp.constant(6 * q * q)

    def integrate_kernel_fn(
        qp_arg: quadrature.Arg,
        domain_arg: domain.ElementArg,
        domain_index_arg: domain.ElementIndexArg,
        fields: FieldStruct,
        values: ValueStruct,
        input_eval_arg: cell_field.EvalArg,
        end_ops: wp.array2d(dtype=accumulate_dtype),
        tang_ops: wp.array2d(dtype=accumulate_dtype),
        face_map: wp.array2d(dtype=int),
        active_cells: wp.array(dtype=int),
        result_elem: wp.array2d(dtype=accumulate_dtype),
    ):
        block_index = wp.tid()
        cell_index = active_cells[block_index]

        # Runtime-valued loop bound (always 2): keeps the face/neighbor loops
        # dynamic so their tile temporaries are emitted once.
        dyn2 = wp.min(2, cell_index + 2)

        dse_v = meta_int_vec()
        side_v = meta_int_vec()
        flip_v = meta_int_vec()
        w_in_v = meta_scalar_vec()
        w_out_v = meta_scalar_vec()

        TR = wp.tile_zeros(shape=(tr_rows_c, qq_c), dtype=accumulate_dtype)
        CB = wp.tile_zeros(shape=(cb_rows_c, q2_c), dtype=accumulate_dtype)
        CA = wp.tile_zeros(shape=(ca_rows_c, q_c), dtype=accumulate_dtype)

        # --- B stage, axis 0 faces (face frame (1, 2) = (longitude, latitude))
        for end in range(dyn2):
            face = end
            dse_raw = face_map[cell_index, face]
            active = dse_raw >= 0
            dse = wp.max(dse_raw, 0)
            side_index = domain.element_index(domain_index_arg, dse)
            inner_cell = geometry.side_inner_cell_index(domain_arg, side_index)
            outer_cell = geometry.side_outer_cell_index(domain_arg, side_index)
            boundary = inner_cell == outer_cell
            alt0 = boundary and end == 0
            inner_end = wp.where(alt0, 0, 1)
            outer_end = wp.where(boundary and end == 1, 1, 0)
            flip = wp.where(alt0, 1, 0)  # 3D longitude flip: altitude == 0

            dse_v[face] = dse
            side_v[face] = side_index
            flip_v[face] = flip
            w_in_v[face] = wp.where(active and inner_cell == cell_index, accumulate_dtype(1.0), accumulate_dtype(0.0))
            w_out_v[face] = wp.where(active and outer_cell == cell_index, accumulate_dtype(1.0), accumulate_dtype(0.0))

            # Face frame first axis = longitude (ga0 = 1), second = latitude (ga1 = 2)
            S_row_x0 = wp.tile_load(tang_ops, shape=(q2_c, n_c), offset=(flip * q2_c, 0))
            S_col_x0 = wp.tile_load(tang_ops, shape=(q2_c, n_c), offset=(0, 0))
            I_row_x0 = wp.tile_load(tang_ops, shape=(q_c, n_c), offset=(flip * q2_c, 0))
            I_col_x0 = wp.tile_load(tang_ops, shape=(q_c, n_c), offset=(0, 0))
            for nb in range(dyn2):
                nb_cell = wp.where(nb == 0, inner_cell, outer_cell)
                nb_end = wp.where(nb == 0, inner_end, outer_end)
                u_flat_x0 = wp.tile_load(input_eval_arg.dof_values, shape=(nnn_c,), offset=(nb_cell * nnn_c,))
                U_x0 = wp.tile_astype(wp.tile_reshape(u_flat_x0, shape=(n_c, nn_c)), dtype=accumulate_dtype)  # [i, jk]
                E_nb_x0 = wp.tile_load(end_ops, shape=(2, n_c), offset=(2 * nb_end, 0))
                T_x0 = wp.tile_matmul(E_nb_x0, U_x0)  # (2, nn): [value; d/dn] over [j, k]
                Fv_x0 = wp.tile_reshape(wp.tile_view(T_x0, offset=(0, 0), shape=(1, nn_c)), shape=(n_c, n_c))
                Fg_x0 = wp.tile_reshape(wp.tile_view(T_x0, offset=(1, 0), shape=(1, nn_c)), shape=(n_c, n_c))
                P_x0 = wp.tile_matmul(S_row_x0, Fv_x0)  # (2q, n)
                Q_x0 = wp.tile_matmul(P_x0, wp.tile_transpose(S_col_x0))  # (2q, 2q)
                G1_x0 = wp.tile_matmul(I_row_x0, Fg_x0)  # (q, n)
                Gn_x0 = wp.tile_matmul(G1_x0, wp.tile_transpose(I_col_x0))  # (q, q)
                for t in range(qq_c):
                    r0 = t // q_c
                    r1 = t - r0 * q_c
                    TR[face * 8 + nb * 4 + 0, t] = Q_x0[r0, r1]
                    TR[face * 8 + nb * 4 + 1, t] = Gn_x0[r0, r1]
                    TR[face * 8 + nb * 4 + 2, t] = Q_x0[q_c + r0, r1]  # d/d ga0 (longitude)
                    TR[face * 8 + nb * 4 + 3, t] = Q_x0[r0, q_c + r1]  # d/d ga1 (latitude)

        # --- B stage, axis 1 faces (face frame (0, 2) = (latitude, longitude): swapped)
        for end in range(dyn2):
            face = 2 + end
            dse_raw = face_map[cell_index, face]
            active = dse_raw >= 0
            dse = wp.max(dse_raw, 0)
            side_index = domain.element_index(domain_index_arg, dse)
            inner_cell = geometry.side_inner_cell_index(domain_arg, side_index)
            outer_cell = geometry.side_outer_cell_index(domain_arg, side_index)
            boundary = inner_cell == outer_cell
            alt0 = boundary and end == 0
            inner_end = wp.where(alt0, 0, 1)
            outer_end = wp.where(boundary and end == 1, 1, 0)
            flip = wp.where(alt0, 1, 0)

            dse_v[face] = dse
            side_v[face] = side_index
            flip_v[face] = flip
            w_in_v[face] = wp.where(active and inner_cell == cell_index, accumulate_dtype(1.0), accumulate_dtype(0.0))
            w_out_v[face] = wp.where(active and outer_cell == cell_index, accumulate_dtype(1.0), accumulate_dtype(0.0))

            # Face frame first axis = latitude (ga1 = 0, never flipped),
            # second = longitude (ga0 = 2, flip-selected)
            S_row_x1 = wp.tile_load(tang_ops, shape=(q2_c, n_c), offset=(0, 0))
            S_col_x1 = wp.tile_load(tang_ops, shape=(q2_c, n_c), offset=(flip * q2_c, 0))
            I_row_x1 = wp.tile_load(tang_ops, shape=(q_c, n_c), offset=(0, 0))
            I_col_x1 = wp.tile_load(tang_ops, shape=(q_c, n_c), offset=(flip * q2_c, 0))
            for nb in range(dyn2):
                nb_cell = wp.where(nb == 0, inner_cell, outer_cell)
                nb_end = wp.where(nb == 0, inner_end, outer_end)
                u_flat_x1 = wp.tile_load(input_eval_arg.dof_values, shape=(nnn_c,), offset=(nb_cell * nnn_c,))
                U_x1 = wp.tile_astype(wp.tile_reshape(u_flat_x1, shape=(n_c, nn_c)), dtype=accumulate_dtype)  # [i, jk]
                E_nb_x1 = wp.tile_load(end_ops, shape=(2, n_c), offset=(2 * nb_end, 0))
                # Slab contraction over the middle axis j: Fv_x1/Fg_x1 in face frame [i, k]
                Fv_x1 = wp.tile_zeros(shape=(n_c, n_c), dtype=accumulate_dtype)
                Fg_x1 = wp.tile_zeros(shape=(n_c, n_c), dtype=accumulate_dtype)
                for i in range(n_c):
                    blk_x1 = wp.tile_reshape(
                        wp.tile_view(U_x1, offset=(i, 0), shape=(1, nn_c)), shape=(n_c, n_c)
                    )  # [j, k]
                    Ti_x1 = wp.tile_matmul(E_nb_x1, blk_x1)  # (2, n): [value; d/dn] over [k]
                    wp.tile_assign(Fv_x1, wp.tile_view(Ti_x1, offset=(0, 0), shape=(1, n_c)), offset=(i, 0))
                    wp.tile_assign(Fg_x1, wp.tile_view(Ti_x1, offset=(1, 0), shape=(1, n_c)), offset=(i, 0))
                P_x1 = wp.tile_matmul(S_row_x1, Fv_x1)  # (2q, n): latitude rows
                Q_x1 = wp.tile_matmul(P_x1, wp.tile_transpose(S_col_x1))  # (2q, 2q): [r1-blocks, r0-blocks]
                G1_x1 = wp.tile_matmul(I_row_x1, Fg_x1)
                Gn_x1 = wp.tile_matmul(G1_x1, wp.tile_transpose(I_col_x1))  # [r1, r0]
                for t in range(qq_c):
                    r0 = t // q_c
                    r1 = t - r0 * q_c
                    TR[face * 8 + nb * 4 + 0, t] = Q_x1[r1, r0]
                    TR[face * 8 + nb * 4 + 1, t] = Gn_x1[r1, r0]
                    TR[face * 8 + nb * 4 + 2, t] = Q_x1[r1, q_c + r0]  # d/d ga0 (longitude)
                    TR[face * 8 + nb * 4 + 3, t] = Q_x1[q_c + r1, r0]  # d/d ga1 (latitude)

        # --- B stage, axis 2 faces (face frame (0, 1) = (longitude, latitude))
        for end in range(dyn2):
            face = 4 + end
            dse_raw = face_map[cell_index, face]
            active = dse_raw >= 0
            dse = wp.max(dse_raw, 0)
            side_index = domain.element_index(domain_index_arg, dse)
            inner_cell = geometry.side_inner_cell_index(domain_arg, side_index)
            outer_cell = geometry.side_outer_cell_index(domain_arg, side_index)
            boundary = inner_cell == outer_cell
            alt0 = boundary and end == 0
            inner_end = wp.where(alt0, 0, 1)
            outer_end = wp.where(boundary and end == 1, 1, 0)
            flip = wp.where(alt0, 1, 0)

            dse_v[face] = dse
            side_v[face] = side_index
            flip_v[face] = flip
            w_in_v[face] = wp.where(active and inner_cell == cell_index, accumulate_dtype(1.0), accumulate_dtype(0.0))
            w_out_v[face] = wp.where(active and outer_cell == cell_index, accumulate_dtype(1.0), accumulate_dtype(0.0))

            # Face frame first axis = longitude (ga0 = 0), second = latitude (ga1 = 1)
            S_row_x2 = wp.tile_load(tang_ops, shape=(q2_c, n_c), offset=(flip * q2_c, 0))
            S_col_x2 = wp.tile_load(tang_ops, shape=(q2_c, n_c), offset=(0, 0))
            I_row_x2 = wp.tile_load(tang_ops, shape=(q_c, n_c), offset=(flip * q2_c, 0))
            I_col_x2 = wp.tile_load(tang_ops, shape=(q_c, n_c), offset=(0, 0))
            for nb in range(dyn2):
                nb_cell = wp.where(nb == 0, inner_cell, outer_cell)
                nb_end = wp.where(nb == 0, inner_end, outer_end)
                u_flat_x2 = wp.tile_load(input_eval_arg.dof_values, shape=(nnn_c,), offset=(nb_cell * nnn_c,))
                U_x2 = wp.tile_astype(wp.tile_reshape(u_flat_x2, shape=(nn_c, n_c)), dtype=accumulate_dtype)  # [ij, k]
                E_nb_x2 = wp.tile_load(end_ops, shape=(2, n_c), offset=(2 * nb_end, 0))
                T_x2 = wp.tile_matmul(E_nb_x2, wp.tile_transpose(U_x2))  # (2, nn): [value; d/dn] over [i, j]
                Fv_x2 = wp.tile_reshape(wp.tile_view(T_x2, offset=(0, 0), shape=(1, nn_c)), shape=(n_c, n_c))
                Fg_x2 = wp.tile_reshape(wp.tile_view(T_x2, offset=(1, 0), shape=(1, nn_c)), shape=(n_c, n_c))
                P_x2 = wp.tile_matmul(S_row_x2, Fv_x2)  # (2q, n)
                Q_x2 = wp.tile_matmul(P_x2, wp.tile_transpose(S_col_x2))  # (2q, 2q)
                G1_x2 = wp.tile_matmul(I_row_x2, Fg_x2)
                Gn_x2 = wp.tile_matmul(G1_x2, wp.tile_transpose(I_col_x2))
                for t in range(qq_c):
                    r0 = t // q_c
                    r1 = t - r0 * q_c
                    TR[face * 8 + nb * 4 + 0, t] = Q_x2[r0, r1]
                    TR[face * 8 + nb * 4 + 1, t] = Gn_x2[r0, r1]
                    TR[face * 8 + nb * 4 + 2, t] = Q_x2[q_c + r0, r1]  # d/d ga0 (longitude)
                    TR[face * 8 + nb * 4 + 3, t] = Q_x2[r0, q_c + r1]  # d/d ga1 (latitude)

        # --- D stage: seeded integrand evaluations per face quadrature point
        qp_fields = FieldStruct()
        _copy_qp_fields()

        for it in range(dloop_c):
            face = it // qq_c
            r = it - face * qq_c
            r0 = r // q_c
            r1 = r - r0 * q_c
            axis = face // 2
            ga0 = (axis + 1) % 3
            ga1 = (axis + 2) % 3
            dse = dse_v[face]
            side_index = side_v[face]
            w_in = w_in_v[face]
            w_out = w_out_v[face]

            qp_index = quadrature.point_index(domain_arg, qp_arg, dse, side_index, r)
            qp_coords = quadrature.point_coords(domain_arg, qp_arg, dse, side_index, r)
            qp_weight = quadrature.point_weight(domain_arg, qp_arg, dse, side_index, r)

            free_sample = make_free_sample(side_index, qp_coords)
            vol = domain.element_measure(domain_arg, free_sample)
            scale = accumulate_dtype(qp_weight * vol)
            xf_in = inner_grad_transform(domain_arg, free_sample)  # J^{-1} of the inner cell
            xf_out = outer_grad_transform(domain_arg, free_sample)

            # Inject the B-stage traces (value + physical gradient, inner and outer)
            base = face * 8
            inj_arg = InjectedEvalArg()
            inj_arg.inner_value = value_type(TR[base + 0, r])
            g_in = grad_type()
            g_in[axis] = value_type(TR[base + 1, r])
            g_in[ga0] = value_type(TR[base + 2, r])
            g_in[ga1] = value_type(TR[base + 3, r])
            inj_arg.inner_gradient = wp.transpose(xf_in) * g_in
            inj_arg.outer_value = value_type(TR[base + 4, r])
            g_out = grad_type()
            g_out[axis] = value_type(TR[base + 5, r])
            g_out[ga0] = value_type(TR[base + 6, r])
            g_out[ga1] = value_type(TR[base + 7, r])
            inj_arg.outer_gradient = wp.transpose(xf_out) * g_out
            _set_injected_eval_arg()

            # Seeded test-channel extraction; this cell lifts its own role's
            # channels (both roles on boundary sides).
            c0 = accumulate_dtype(0.0)
            cg = acc_grad_vec()
            if w_in != accumulate_dtype(0.0):
                sample = SampleType(side_index, qp_coords, qp_index, qp_weight, DofIndex(0, 0), NULL_DOF_INDEX)
                c0 += w_in * accumulate_dtype(integrand_func(sample, qp_fields, values))
                if wp.static(test_uses_grad):
                    fg = grad_type()
                    for seed in range(3):
                        sample = SampleType(
                            side_index, qp_coords, qp_index, qp_weight, DofIndex(seed + 1, 0), NULL_DOF_INDEX
                        )
                        fg[seed] = value_type(integrand_func(sample, qp_fields, values))
                    fg_ref = xf_in * fg
                    for k in range(3):
                        cg[k] += w_in * accumulate_dtype(fg_ref[k])
            if w_out != accumulate_dtype(0.0):
                sample = SampleType(
                    side_index, qp_coords, qp_index, qp_weight, DofIndex(OUTER_VALUE_SEED, 0), NULL_DOF_INDEX
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

            # Write the coefficients in the per-face lift frame: rows of CB
            # are contracted by the face-frame-first axis operator. For
            # axis == 1 the face frame is (latitude, longitude): swap.
            swapped = wp.where(axis == 1, 1, 0)
            rr = wp.where(swapped == 1, r1, r0)
            cc = wp.where(swapped == 1, r0, r1)
            CB[face * q2_c + rr, cc] = scale * c0
            if wp.static(test_uses_grad):
                f0_axis = wp.where(axis == 0, 1, 0)  # face-frame first element axis
                f1_axis = wp.where(axis == 2, 1, 2)  # face-frame second element axis
                CB[face * q2_c + q_c + rr, cc] = scale * cg[f0_axis]
                CB[face * q2_c + rr, q_c + cc] = scale * cg[f1_axis]
                CA[face * q_c + rr, cc] = scale * cg[axis]

        # --- B^T stage: lift the coefficients back to cell nodal residuals
        r_ax0 = wp.tile_zeros(shape=(n_c, nn_c), dtype=accumulate_dtype)  # [i, jk]
        r_ax1 = wp.tile_zeros(shape=(n_c, nn_c), dtype=accumulate_dtype)  # [j, ik]
        r_ax2 = wp.tile_zeros(shape=(n_c, nn_c), dtype=accumulate_dtype)  # [k, ij]

        # axis 0 faces: row op = longitude (flip-selected), col op = latitude
        for end in range(dyn2):
            face = end
            flip = flip_v[face]
            E_own_l0 = wp.tile_load(end_ops, shape=(2, n_c), offset=(2 * end, 0))
            S_row_l0 = wp.tile_load(tang_ops, shape=(q2_c, n_c), offset=(flip * q2_c, 0))
            S_col_l0 = wp.tile_load(tang_ops, shape=(q2_c, n_c), offset=(0, 0))
            I_row_l0 = wp.tile_load(tang_ops, shape=(q_c, n_c), offset=(flip * q2_c, 0))
            I_col_l0 = wp.tile_load(tang_ops, shape=(q_c, n_c), offset=(0, 0))
            CBf_l0 = wp.tile_view(CB, offset=(face * q2_c, 0), shape=(q2_c, q2_c))
            X_l0 = wp.tile_matmul(wp.tile_transpose(S_row_l0), CBf_l0)  # (n, 2q)
            Av_l0 = wp.tile_matmul(X_l0, S_col_l0)  # (n, n): [j, k]
            CAf_l0 = wp.tile_view(CA, offset=(face * q_c, 0), shape=(q_c, q_c))
            Y_l0 = wp.tile_matmul(wp.tile_transpose(I_row_l0), CAf_l0)  # (n, q)
            An_l0 = wp.tile_matmul(Y_l0, I_col_l0)  # (n, n)
            Z_l0 = wp.tile_zeros(shape=(2, nn_c), dtype=accumulate_dtype)
            wp.tile_assign(Z_l0, wp.tile_reshape(Av_l0, shape=(1, nn_c)), offset=(0, 0))
            wp.tile_assign(Z_l0, wp.tile_reshape(An_l0, shape=(1, nn_c)), offset=(1, 0))
            wp.tile_matmul(wp.tile_transpose(E_own_l0), Z_l0, r_ax0)

        # axis 1 faces: row op = latitude (straight), col op = longitude
        for end in range(dyn2):
            face = 2 + end
            flip = flip_v[face]
            E_own_l1 = wp.tile_load(end_ops, shape=(2, n_c), offset=(2 * end, 0))
            S_row_l1 = wp.tile_load(tang_ops, shape=(q2_c, n_c), offset=(0, 0))
            S_col_l1 = wp.tile_load(tang_ops, shape=(q2_c, n_c), offset=(flip * q2_c, 0))
            I_row_l1 = wp.tile_load(tang_ops, shape=(q_c, n_c), offset=(0, 0))
            I_col_l1 = wp.tile_load(tang_ops, shape=(q_c, n_c), offset=(flip * q2_c, 0))
            CBf_l1 = wp.tile_view(CB, offset=(face * q2_c, 0), shape=(q2_c, q2_c))
            X_l1 = wp.tile_matmul(wp.tile_transpose(S_row_l1), CBf_l1)  # (n, 2q)
            Av_l1 = wp.tile_matmul(X_l1, S_col_l1)  # (n, n): [i, k]
            CAf_l1 = wp.tile_view(CA, offset=(face * q_c, 0), shape=(q_c, q_c))
            Y_l1 = wp.tile_matmul(wp.tile_transpose(I_row_l1), CAf_l1)
            An_l1 = wp.tile_matmul(Y_l1, I_col_l1)
            Z_l1 = wp.tile_zeros(shape=(2, nn_c), dtype=accumulate_dtype)
            wp.tile_assign(Z_l1, wp.tile_reshape(Av_l1, shape=(1, nn_c)), offset=(0, 0))
            wp.tile_assign(Z_l1, wp.tile_reshape(An_l1, shape=(1, nn_c)), offset=(1, 0))
            wp.tile_matmul(wp.tile_transpose(E_own_l1), Z_l1, r_ax1)

        # axis 2 faces: row op = longitude (flip-selected), col op = latitude
        for end in range(dyn2):
            face = 4 + end
            flip = flip_v[face]
            E_own_l2 = wp.tile_load(end_ops, shape=(2, n_c), offset=(2 * end, 0))
            S_row_l2 = wp.tile_load(tang_ops, shape=(q2_c, n_c), offset=(flip * q2_c, 0))
            S_col_l2 = wp.tile_load(tang_ops, shape=(q2_c, n_c), offset=(0, 0))
            I_row_l2 = wp.tile_load(tang_ops, shape=(q_c, n_c), offset=(flip * q2_c, 0))
            I_col_l2 = wp.tile_load(tang_ops, shape=(q_c, n_c), offset=(0, 0))
            CBf_l2 = wp.tile_view(CB, offset=(face * q2_c, 0), shape=(q2_c, q2_c))
            X_l2 = wp.tile_matmul(wp.tile_transpose(S_row_l2), CBf_l2)
            Av_l2 = wp.tile_matmul(X_l2, S_col_l2)  # (n, n): [i, j]
            CAf_l2 = wp.tile_view(CA, offset=(face * q_c, 0), shape=(q_c, q_c))
            Y_l2 = wp.tile_matmul(wp.tile_transpose(I_row_l2), CAf_l2)
            An_l2 = wp.tile_matmul(Y_l2, I_col_l2)
            Z_l2 = wp.tile_zeros(shape=(2, nn_c), dtype=accumulate_dtype)
            wp.tile_assign(Z_l2, wp.tile_reshape(Av_l2, shape=(1, nn_c)), offset=(0, 0))
            wp.tile_assign(Z_l2, wp.tile_reshape(An_l2, shape=(1, nn_c)), offset=(1, 0))
            wp.tile_matmul(wp.tile_transpose(E_own_l2), Z_l2, r_ax2)

        # Combine the three face orientations:
        # r[i, j*n + k] = r_ax0[i, jk] + r_ax1[j, ik] + r_ax2[k, ij]
        for t in range(nnn_c):
            i = t // nn_c
            jk = t - i * nn_c
            j = jk // n_c
            k = jk - j * n_c
            r_ax0[i, jk] = r_ax0[i, jk] + r_ax1[j, i * n_c + k] + r_ax2[k, i * n_c + j]

        r_flat = wp.tile_reshape(r_ax0, shape=(1, nnn_c))
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
