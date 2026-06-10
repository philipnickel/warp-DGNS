# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Fused sum-factorized ``B^T D B`` apply kernels for linear forms (Phase 3).

This module provides the staged kernel factory behind the transparent
sum-factorization dispatch in :mod:`warp._src.fem.integrate`:

1. **B stage** -- the input field's element DOFs are gathered in-kernel and
   interpolated to the quadrature points (value and reference gradient) with
   per-axis :func:`warp.tile_matmul` contractions against the 1D interpolation
   and derivative operators, following the canonical layouts of
   :mod:`warp._src.fem.sumfac.tensor_contract`.
2. **D stage** -- per quadrature point, the *transformed user integrand* is
   evaluated ``d + 1`` times with the test field replaced by a
   :class:`warp._src.fem.field.SeedField` (value/gradient seeds carried by
   ``Sample.test_dof``) and the input field by a
   :class:`warp._src.fem.field.ValueInjectedField` whose ``EvalArg`` is filled
   from the ``B``-stage tiles (value and ``J^{-T}``-mapped physical gradient).
   The physical test-gradient coefficients are mapped back to reference space
   with ``J^{-1}`` and folded with ``weight * |det J|``.
3. **B^T stage** -- the ``(f0, f1)`` coefficient tiles are contracted back to
   element nodal residuals with the transposed operators and stored per
   element; a small scatter kernel accumulates them into the partition node
   vector (one writer per node for discontinuous spaces).

The kernel is generated through the standard ``cache.get_integrand_kernel`` +
``PassFieldArgsToIntegrand`` machinery (see
``design/sumfac-phase3-spike-findings.md``), which registers each
``(n, q, E_b, dim, dtype)`` specialization in its own dynamic module with
``enable_backward=False`` (autodiff through the staged kernel is deferred;
every backward-enabled ``tile_matmul`` would triple the LTO compile cost).

Structural applicability is decided host-side by :func:`find_sumfac_layout`,
which conservatively verifies every assumption baked into the kernel: cell
domain, tensor-product discontinuous scalar space, lexicographically ordered
tensor-product ``RegularQuadrature`` points, supported operators on the test
field, and a single injectable input field. As with the rest of the
``B^T D B`` machinery, the integrand must be *linear* in the test function
(weak forms are, by construction; affine offsets cannot be represented by the
seeded extraction).
"""

import ast
from dataclasses import dataclass
from typing import Any

import numpy as np

import warp as wp
from warp._src.context import capture_pause, capture_resume
from warp._src.fem import cache
from warp._src.fem import operator as fem_operator
from warp._src.fem.domain import GeometryDomain
from warp._src.fem.field import TestField, TrialField
from warp._src.fem.field.nodal_field import NodalField
from warp._src.fem.field.virtual import SeedField, TrialSeedField, ValueInjectedField
from warp._src.fem.polynomial import quadrature_1d
from warp._src.fem.quadrature import Quadrature, RegularQuadrature
from warp._src.fem.space import FunctionSpace, SpacePartition
from warp._src.fem.space.basis_space import ShapeBasisSpace
from warp._src.fem.space.shape.cube_shape_function import CubeTripolynomialShapeFunctions
from warp._src.fem.space.shape.square_shape_function import SquareBipolynomialShapeFunctions
from warp._src.fem.space.topology import RegularDiscontinuousSpaceTopologyMixin
from warp._src.fem.sumfac.operators_1d import build_derivative_matrix, build_interpolation_matrix
from warp._src.fem.sumfac.qfunction import _SUPPORTED_TEST_OPERATORS
from warp._src.fem.types import (
    NULL_DOF_INDEX,
    NULL_NODE_INDEX,
    DofIndex,
    ElementKind,
    make_free_sample,
)
from warp._src.optim.linear import LinearOperator
from warp._src.types import type_scalar_type

__all__ = [
    "SumfacBilinearPlan",
    "SumfacPlan",
    "find_sumfac_bilinear_layout",
    "find_sumfac_layout",
    "get_integrate_bilinear_sumfac_kernel",
    "get_integrate_linear_sumfac_kernel",
    "get_sumfac_scatter_kernel",
    "get_sumfac_triplet_fill_kernel",
    "make_sumfac_bilinear_plan",
    "make_sumfac_linear_operator",
    "make_sumfac_plan",
    "sumfac_block_dim",
]

#: Panel width (elements per block). Only ``1`` is implemented for now; the
#: ``E_b > 1`` wider-GEMM variant is a Phase 5 performance knob.
SUMFAC_ELEMENT_BATCH = 1

# Operators through which the injected input field may be accessed: the
# B stage only interpolates the value and gradient of the input field.
_SUPPORTED_INPUT_OPERATORS = frozenset(
    (
        fem_operator.inner,
        fem_operator.outer,
        fem_operator.grad,
        fem_operator.grad_outer,
        fem_operator.degree,
    )
)

# Operators that require seeding (and contracting) the gradient channels of a
# test or trial field.
_GRAD_OPERATORS = frozenset((fem_operator.grad, fem_operator.grad_outer))


def sumfac_block_dim(device) -> int:
    """Pick the tile-kernel block size for ``device``.

    CPU tile kernels run serialized with a single thread per block; on CUDA
    the ``tile_matmul`` cuBLASDx path requires at least a full warp.
    """
    return 1 if wp.get_device(device).is_cpu else 64


@dataclass
class _SumfacLayoutBase:
    """Host-side description shared by the linear and bilinear layouts."""

    degree: int
    n: int
    q: int
    dim: int
    qpoints_1d: np.ndarray
    test_uses_grad: bool


@dataclass
class SumfacLayout(_SumfacLayoutBase):
    """Host-side description of a qualifying tensor-product linear form."""

    input_name: str
    input_field: NodalField


@dataclass
class _SumfacPlanBase:
    """Launch-side fields shared by the linear and bilinear plans."""

    test: TestField
    test_name: str
    degree: int
    n: int
    q: int
    dim: int
    element_batch: int
    interp: np.ndarray
    deriv: np.ndarray
    test_uses_grad: bool

    @property
    def nodes_per_element(self) -> int:
        """Number of nodes per element, ``n**dim``."""
        return self.n**self.dim

    def operator_arrays(self, dtype, device) -> tuple[wp.array, wp.array]:
        """Return the ``(interp, deriv)`` device arrays in ``dtype`` on ``device``, cached."""
        return _get_operator_arrays(self.interp, self.deriv, dtype, device)


@dataclass
class SumfacPlan(_SumfacPlanBase):
    """Launch-side description of a sum-factorized linear-form integration.

    Built by :func:`make_sumfac_plan` once a form has qualified; carries the
    original input field (for the in-kernel DOF gather), the substituted
    :class:`SeedField`/:class:`ValueInjectedField` instances, the baked tile
    sizes, and the host 1D operator matrices.
    """

    input_name: str
    input_field: NodalField
    seed_field: SeedField
    injected_field: ValueInjectedField


_operator_array_cache: dict[Any, tuple[wp.array, wp.array]] = {}


def _get_operator_arrays(interp: np.ndarray, deriv: np.ndarray, dtype, device):
    device = wp.get_device(device)
    key = (interp.tobytes(), deriv.tobytes(), type_scalar_type(dtype), device.alias)
    arrays = _operator_array_cache.get(key)
    if arrays is None:
        # Pause graph capture while we copy from host, so that the first use of
        # a given operator pair inside a wp.ScopedCapture still succeeds and the
        # cached arrays remain valid outside of the graph (same pattern as
        # RegularQuadrature.CachedFormula.arg_value).
        graph = capture_pause() if device.is_capturing else None

        np_dtype = np.float32 if type_scalar_type(dtype) == wp.float32 else np.float64
        arrays = (
            wp.array(interp.astype(np_dtype), dtype=dtype, device=device),
            wp.array(deriv.astype(np_dtype), dtype=dtype, device=device),
        )
        _operator_array_cache[key] = arrays

        if graph is not None:
            capture_resume(graph)
    return arrays


def _tensor_product_quadrature_points_1d(quadrature: RegularQuadrature, dim: int) -> np.ndarray | None:
    """Extract the 1D point set if the quadrature is a lexicographic tensor product, else ``None``.

    The fused kernel assumes the per-element quadrature points enumerate the
    outer product of a single 1D rule with the first axis slowest (matching
    the lexicographic node ordering of the square/cube shape functions); this
    is verified explicitly rather than assumed (silent-correctness risk).
    """
    count = quadrature.max_points_per_element()
    if count is None or count < 1:
        return None
    q = round(count ** (1.0 / dim))
    while q**dim < count:
        q += 1
    if q < 1 or q**dim != count:
        return None

    points = np.array([[p[i] for i in range(dim)] for p in quadrature.points], dtype=np.float64)
    # With the first axis slowest, the last coordinate of the first q points is the 1D rule
    qpoints_1d = points[:q, dim - 1].copy()
    if len(np.unique(qpoints_1d)) != q:
        return None

    grids = np.meshgrid(*([qpoints_1d] * dim), indexing="ij")
    expected = np.stack([g.reshape(-1) for g in grids], axis=-1)
    atol = 1e-5 if quadrature.domain.geometry.scalar_type == wp.float32 else 1e-12
    if not np.allclose(points, expected, rtol=0.0, atol=atol):
        return None
    return qpoints_1d


def _is_tensor_product_scalar_dg_space(space: FunctionSpace, dim: int) -> bool:
    """Check that ``space`` is a scalar, discontinuous, tensor-product polynomial space."""
    if space.NODE_DOF_COUNT != 1 or space.VALUE_DOF_COUNT != 1:
        return False
    if not isinstance(space.topology, RegularDiscontinuousSpaceTopologyMixin):
        return False
    basis = space.basis
    if not isinstance(basis, ShapeBasisSpace):
        return False
    shape_cls = SquareBipolynomialShapeFunctions if dim == 2 else CubeTripolynomialShapeFunctions
    if type(basis.shape) is not shape_cls:
        return False
    degree = space.degree
    if degree < 1:
        return False
    return space.topology.MAX_NODES_PER_ELEMENT == (degree + 1) ** dim


def _find_tensor_product_core(
    integrand,
    arguments,
    test: TestField,
    quadrature: Quadrature,
    domain: GeometryDomain,
) -> tuple[int, np.ndarray, set] | None:
    """Verify the geometric/space/quadrature/operator assumptions shared by the apply and assembly paths.

    Returns ``(dim, qpoints_1d, test_operators)`` for a cell domain over a
    tensor-product geometry with a scalar discontinuous tensor-product test
    space, a matching lexicographic tensor-product :class:`RegularQuadrature`,
    and only seedable operators on the test field, else ``None``.
    """
    if domain.element_kind != ElementKind.CELL:
        return None
    geometry = domain.geometry
    dim = geometry.dimension
    if dim not in (2, 3) or geometry.cell_dimension != dim:
        return None

    if not _is_tensor_product_scalar_dg_space(test.space, dim):
        return None

    if not isinstance(quadrature, RegularQuadrature) or quadrature.domain != domain:
        return None
    qpoints_1d = _tensor_product_quadrature_points_1d(quadrature, dim)
    if qpoints_1d is None:
        return None

    if integrand.operators is None:
        return None
    test_operators = integrand.operators.get(arguments.test_name, set())
    if not test_operators <= _SUPPORTED_TEST_OPERATORS:
        return None

    return dim, qpoints_1d, test_operators


def find_sumfac_layout(
    integrand,
    arguments,
    test: TestField,
    quadrature: Quadrature,
    domain: GeometryDomain,
) -> SumfacLayout | None:
    """Return the sum-factorization layout if the form qualifies, else ``None``.

    Conservative predicate core: every structural assumption of the fused
    kernel is verified, and anything unproven returns ``None`` (the caller
    falls through to the legacy kernels, which is always correct).

    Args:
        integrand: The form being integrated; ``integrand.operators`` must
            have been populated (via ``_find_integrand_operators``).
        arguments: Parsed integrand arguments (before field substitution).
        test: The (plain) test field of the linear form.
        quadrature: Quadrature formula of the integration.
        domain: Integration domain.
    """
    core = _find_tensor_product_core(integrand, arguments, test, quadrature, domain)
    if core is None:
        return None
    dim, qpoints_1d, test_operators = core
    space = test.space

    # Exactly one injectable input field: a nodal field over the same space as
    # the test field, accessed only through value/gradient operators. Other
    # field arguments are evaluated through their standard machinery.
    input_name = None
    input_field = None
    for name, field in arguments.field_args.items():
        if name == arguments.test_name or not isinstance(field, NodalField):
            continue
        if field.space.name != space.name:
            continue
        field_operators = integrand.operators.get(name, set())
        if not field_operators <= _SUPPORTED_INPUT_OPERATORS:
            continue
        input_name = name
        input_field = field
        break
    if input_name is None:
        return None

    degree = space.degree
    return SumfacLayout(
        input_name=input_name,
        input_field=input_field,
        degree=degree,
        n=degree + 1,
        q=len(qpoints_1d),
        dim=dim,
        qpoints_1d=qpoints_1d,
        # Seed (and contract) the test-gradient channels only when the
        # integrand actually applies a gradient operator to the test field;
        # this matches the legacy dispatch path, which restricts the Taylor
        # DOFs to the operators in use (LocalAdjointField.notify_operator_usage).
        test_uses_grad=bool(_GRAD_OPERATORS & test_operators),
    )


def _build_1d_operator_matrices(test: TestField, layout: _SumfacLayoutBase) -> tuple[np.ndarray, np.ndarray]:
    """Build the ``(q, n)`` 1D interpolation/derivative matrices for the test space's basis nodes."""
    shape = test.space.basis.shape
    nodes_1d = np.asarray(quadrature_1d(point_count=layout.n, family=shape.family)[0], dtype=np.float64)
    interp = build_interpolation_matrix(nodes_1d, layout.qpoints_1d)
    deriv = build_derivative_matrix(nodes_1d, layout.qpoints_1d)
    return interp, deriv


def make_sumfac_plan(
    integrand,
    arguments,
    test: TestField,
    quadrature: Quadrature,
    domain: GeometryDomain,
    element_batch: int = SUMFAC_ELEMENT_BATCH,
) -> SumfacPlan | None:
    """Build the launch plan and substitute the seed/injected fields in ``arguments``.

    On success, ``arguments.field_args`` is mutated in place: the test field
    is replaced by a :class:`SeedField` and the input field by a
    :class:`ValueInjectedField`, so that the downstream ``FieldStruct`` /
    ``IntegrandTransformer`` machinery generates the in-kernel Q-function.
    Returns ``None`` if the form does not qualify (no mutation happens).
    """
    layout = find_sumfac_layout(integrand, arguments, test, quadrature, domain)
    if layout is None:
        return None

    interp, deriv = _build_1d_operator_matrices(test, layout)

    seed_field = SeedField.from_field(test)
    injected_field = ValueInjectedField.from_field(layout.input_field, domain)
    arguments.field_args[arguments.test_name] = seed_field
    arguments.field_args[layout.input_name] = injected_field

    return SumfacPlan(
        test=test,
        test_name=arguments.test_name,
        input_name=layout.input_name,
        input_field=layout.input_field,
        seed_field=seed_field,
        injected_field=injected_field,
        degree=layout.degree,
        n=layout.n,
        q=layout.q,
        dim=layout.dim,
        element_batch=element_batch,
        interp=interp,
        deriv=deriv,
        test_uses_grad=layout.test_uses_grad,
    )


# -- Bilinear (assembly) path ---------------------------------------------------
#
# For a bilinear form a(u, v), linear in both arguments, the per-quadrature-
# point D stage is the (1 + d) x (1 + d) channel matrix
#   c[a, b](qp) = w |J| * integrand(test seed a, trial seed b)
# (seed 0 = value, seed 1 + i = physical gradient e_i; gradient channels are
# mapped to reference space with J^{-1} on both sides). The element-local
# block is then
#   K_e = sum_{a, b} B_a^T diag(c[a, b]) B_b
# with B_0 the value interpolation operator and B_{1+i} the reference-
# gradient operator along axis i. The fused kernel evaluates the channels
# ONCE per element ((1+d)^2 q^d integrand evaluations instead of the naive
# n^{2d} q^d), then forms K_e column by column as the B^T D B action on the
# n^d one-hot trial vectors, reusing the Phase 1/3 contraction pipelines:
# O(d n^d q) tile FLOPs per column, O(d n^{2d} q) per element overall versus
# the naive O(n^{2d} q^d).

#: Conservative shared-memory budget (bytes) for the fused bilinear kernel's
#: tile working set. Shapes whose estimated footprint exceeds it do not
#: qualify (the caller falls back to the legacy kernels) instead of failing
#: at module load; the A100-class opt-in limit is ~163 KiB and the estimate
#: deliberately keeps a wide margin for codegen temporaries.
SUMFAC_BILINEAR_SMEM_BUDGET = 96 * 1024


def _sumfac_bilinear_smem_estimate(n: int, q: int, dim: int, scalar_bytes: int) -> int:
    """Estimate the fused bilinear kernel's live tile footprint in bytes (full-gradient case)."""
    nch = (1 + dim) ** 2
    ops = 4 * q * n
    if dim == 2:
        channels = nch * q * q
        column = n * n + 2 * q * n + 3 * q * q  # one-hot DOFs, B stages, (value, grad) at QPs
        combine = 2 * 3 * q * q  # f tiles and elementwise temporaries
        backward = 3 * n * q + n * n
    else:
        channels = nch * q * q * q
        column = n**3 + 2 * q * n * n + 4 * q**3  # one-hot DOFs, B stages, (value, grad) at QPs
        column += 3 * q * n + 4 * q * q  # per-slab intermediates
        combine = 2 * 4 * q**3
        backward = 4 * n * q * q + 4 * n * q + n * n + n**3
    # Safety factor for expression temporaries and allocator padding
    return int(1.5 * scalar_bytes * (ops + channels + column + combine + backward))


@dataclass
class SumfacBilinearLayout(_SumfacLayoutBase):
    """Host-side description of a qualifying tensor-product bilinear form."""

    trial_uses_grad: bool


@dataclass
class SumfacBilinearPlan(_SumfacPlanBase):
    """Launch-side description of a sum-factorized bilinear-form assembly.

    Built by :func:`make_sumfac_bilinear_plan` once a form has qualified;
    carries the substituted :class:`SeedField`/:class:`TrialSeedField`
    instances, the baked tile sizes, and the host 1D operator matrices.
    """

    trial: TrialField
    trial_name: str
    test_seed: SeedField
    trial_seed: TrialSeedField
    trial_uses_grad: bool


def find_sumfac_bilinear_layout(
    integrand,
    arguments,
    test: TestField,
    trial: TrialField,
    quadrature: Quadrature,
    domain: GeometryDomain,
) -> SumfacBilinearLayout | None:
    """Return the sum-factorization layout for a bilinear form if it qualifies, else ``None``.

    In addition to the structural requirements of :func:`find_sumfac_layout`'s
    core (cell domain, tensor-product discontinuous scalar test space,
    lexicographic tensor-product quadrature), the trial field must be defined
    over the *same* function space as the test field (mixed test/trial spaces
    are rejected) and accessed only through value/gradient operators, and the
    fused kernel's estimated tile working set must fit the shared-memory
    budget.

    Args:
        integrand: The form being integrated; ``integrand.operators`` must
            have been populated (via ``_find_integrand_operators``).
        arguments: Parsed integrand arguments (before field substitution).
        test: The (plain) test field of the bilinear form.
        trial: The (plain) trial field of the bilinear form.
        quadrature: Quadrature formula of the integration.
        domain: Integration domain.
    """
    core = _find_tensor_product_core(integrand, arguments, test, quadrature, domain)
    if core is None:
        return None
    dim, qpoints_1d, test_operators = core
    space = test.space

    # Same scalar space on both sides; the column node indexing and the
    # shared 1D operators both assume it.
    if trial.space.name != space.name:
        return None

    trial_operators = integrand.operators.get(arguments.trial_name, set())
    if not trial_operators <= _SUPPORTED_TEST_OPERATORS:
        return None

    degree = space.degree
    n = degree + 1
    q = len(qpoints_1d)
    scalar_bytes = 4 if type_scalar_type(space.dtype) == wp.float32 else 8
    if _sumfac_bilinear_smem_estimate(n, q, dim, scalar_bytes) > SUMFAC_BILINEAR_SMEM_BUDGET:
        return None

    return SumfacBilinearLayout(
        degree=degree,
        n=n,
        q=q,
        dim=dim,
        qpoints_1d=qpoints_1d,
        test_uses_grad=bool(_GRAD_OPERATORS & test_operators),
        trial_uses_grad=bool(_GRAD_OPERATORS & trial_operators),
    )


def make_sumfac_bilinear_plan(
    integrand,
    arguments,
    test: TestField,
    trial: TrialField,
    quadrature: Quadrature,
    domain: GeometryDomain,
    element_batch: int = SUMFAC_ELEMENT_BATCH,
) -> SumfacBilinearPlan | None:
    """Build the assembly launch plan and substitute the seed fields in ``arguments``.

    On success, ``arguments.field_args`` is mutated in place: the test field
    is replaced by a :class:`SeedField` and the trial field by a
    :class:`TrialSeedField`, so that the downstream ``FieldStruct`` /
    ``IntegrandTransformer`` machinery generates the in-kernel channel
    extraction. Returns ``None`` if the form does not qualify (no mutation
    happens).
    """
    layout = find_sumfac_bilinear_layout(integrand, arguments, test, trial, quadrature, domain)
    if layout is None:
        return None

    interp, deriv = _build_1d_operator_matrices(test, layout)

    test_seed = SeedField.from_field(test)
    trial_seed = TrialSeedField.from_field(trial)
    arguments.field_args[arguments.test_name] = test_seed
    arguments.field_args[arguments.trial_name] = trial_seed

    return SumfacBilinearPlan(
        test=test,
        trial=trial,
        test_name=arguments.test_name,
        trial_name=arguments.trial_name,
        test_seed=test_seed,
        trial_seed=trial_seed,
        degree=layout.degree,
        n=layout.n,
        q=layout.q,
        dim=layout.dim,
        element_batch=element_batch,
        interp=interp,
        deriv=deriv,
        test_uses_grad=layout.test_uses_grad,
        trial_uses_grad=layout.trial_uses_grad,
    )


# -- Kernel body markers -------------------------------------------------------
#
# The fused kernel must build a fresh local FieldStruct per block and a fresh
# injected EvalArg per quadrature point (kernel-parameter structs are
# effectively immutable), but the struct member names are integrand-specific.
# The kernel body therefore contains marker calls that the
# SumfacQPFieldsTransformer expands into the per-member assignments before
# Warp code generation (the same AST-rewriting mechanism that
# PassFieldArgsToIntegrand uses for its `_get_test_arg()` markers).


def _copy_qp_fields():
    """Marker: expanded into ``qp_fields.<name> = fields.<name>`` per non-injected member."""


def _set_injected_eval_arg():
    """Marker: expanded into ``qp_fields.<input_name> = inj_arg``."""


class SumfacQPFieldsTransformer(ast.NodeTransformer):
    """Expand the sum-factorized kernel's field-struct marker calls.

    Args:
        copied_field_names: Names of the ``FieldStruct`` members copied from
            the kernel's ``fields`` argument (everything but the injected
            input field).
        injected_field_name: Name of the injected input-field member, assigned
            from the per-quadrature-point local ``inj_arg`` variable.
    """

    _FIELDS_VAR = "fields"
    _QP_FIELDS_VAR = "qp_fields"
    _INJECTED_ARG_VAR = "inj_arg"

    def __init__(self, copied_field_names: list[str], injected_field_name: str):
        self._copied_field_names = list(copied_field_names)
        self._injected_field_name = injected_field_name

    @staticmethod
    def _member_assign(dst_var: str, member: str, src_var: str, src_member: str | None, template: ast.AST):
        value: ast.expr = ast.Name(id=src_var, ctx=ast.Load())
        if src_member is not None:
            value = ast.Attribute(value=value, attr=src_member, ctx=ast.Load())
        assign = ast.Assign(
            targets=[
                ast.Attribute(
                    value=ast.Name(id=dst_var, ctx=ast.Load()),
                    attr=member,
                    ctx=ast.Store(),
                )
            ],
            value=value,
        )
        ast.copy_location(assign, template)
        ast.fix_missing_locations(assign)
        return assign

    def visit_Expr(self, node: ast.Expr):
        call = node.value
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)):
            return node

        if call.func.id == _copy_qp_fields.__name__:
            return [
                self._member_assign(self._QP_FIELDS_VAR, name, self._FIELDS_VAR, name, node)
                for name in self._copied_field_names
            ] or ast.copy_location(ast.Pass(), node)

        if call.func.id == _set_injected_eval_arg.__name__:
            return self._member_assign(
                self._QP_FIELDS_VAR, self._injected_field_name, self._INJECTED_ARG_VAR, None, node
            )

        return node


# -- Fused kernel factory -------------------------------------------------------


def get_integrate_linear_sumfac_kernel(
    integrand_func: wp.Function,
    domain: GeometryDomain,
    quadrature: Quadrature,
    FieldStruct,
    ValueStruct,
    test: TestField,
    input_field: NodalField,
    injected_field: ValueInjectedField,
    *,
    n: int,
    q: int,
    dim: int,
    element_batch: int,
    test_uses_grad: bool,
    accumulate_dtype,
):
    """Build the fused ``B^T D B`` kernel body for a qualifying linear form.

    Returns a ``kernel_fn`` closure to be compiled through
    ``cache.get_integrand_kernel`` with a :class:`SumfacQPFieldsTransformer`
    and a ``PassFieldArgsToIntegrand`` (with ``fields_var_name="qp_fields"``)
    in its code transformers, exactly like the other
    ``get_integrate_*_kernel`` factories in :mod:`warp._src.fem.integrate`.

    The kernel is launched with ``wp.launch_tiled(dim=[element_count])``, one
    element per block; all loops have compile-time bounds and there are no
    early returns (uniform control flow over the tile operations).

    Args:
        integrand_func: Transformed integrand (with the seed/injected fields
            substituted).
        domain: Cell domain of the integration.
        quadrature: Tensor-product quadrature formula.
        FieldStruct: Generated field-argument struct (post-substitution).
        ValueStruct: Generated value-argument struct.
        test: Original (plain) test field.
        input_field: Original input nodal field, used for the in-kernel
            element DOF gather.
        injected_field: The :class:`ValueInjectedField` standing in for the
            input field; provides the per-quadrature-point ``EvalArg``.
        n: Nodes per axis of the (test and input) space, ``degree + 1``.
        q: Quadrature points per axis.
        dim: Spatial dimension (2 or 3).
        element_batch: Elements per block; only ``1`` is implemented.
        test_uses_grad: Whether the integrand applies a gradient operator to
            the test field. When ``False`` the gradient seeds and the ``f1``
            contractions are omitted entirely, matching the legacy dispatch
            path (which restricts the Taylor DOFs to the operators in use)
            and saving ``dim`` integrand evaluations per quadrature point.
        accumulate_dtype: Scalar type used for the tile contractions and
            coefficient accumulation.
    """
    if element_batch != 1:
        raise NotImplementedError("Sum-factorized integration currently requires element_batch == 1")

    SampleType = domain.geometry.sample_type
    value_type = injected_field.dtype
    grad_type = injected_field.gradient_dtype
    InjectedEvalArg = injected_field.EvalArg
    InputElementEvalArg = input_field.ElementEvalArg
    read_node_value = input_field._read_node_value

    # The per-quadrature-point and gather loops are written as single flat
    # loops with index arithmetic so that Warp does not unroll them (their
    # trip counts exceed the unroll limit): a fully unrolled D stage inlines
    # hundreds of seeded integrand evaluations and makes the generated source
    # pathologically large for NVRTC at high degrees.
    n_c = wp.constant(n)
    q_c = wp.constant(q)
    nn_c = wp.constant(n * n)
    qq_c = wp.constant(q * q)

    if dim == 2:

        def integrate_kernel_fn(
            qp_arg: quadrature.Arg,
            domain_arg: domain.ElementArg,
            domain_index_arg: domain.ElementIndexArg,
            fields: FieldStruct,
            values: ValueStruct,
            input_eval_arg: input_field.EvalArg,
            interp: wp.array2d(dtype=accumulate_dtype),
            deriv: wp.array2d(dtype=accumulate_dtype),
            result_elem: wp.array2d(dtype=accumulate_dtype),
        ):
            domain_element_index = wp.tid()
            element_index = domain.element_index(domain_index_arg, domain_element_index)

            a_tile = wp.tile_load(interp, shape=(q_c, n_c))
            d_tile = wp.tile_load(deriv, shape=(q_c, n_c))
            a_t = wp.tile_transpose(a_tile)
            d_t = wp.tile_transpose(d_tile)

            # --- B stage: gather element DOFs and interpolate to quadrature points
            input_args = InputElementEvalArg(domain_arg, input_eval_arg)
            u_mat = wp.tile_zeros(shape=(n_c, n_c), dtype=accumulate_dtype)
            for node in range(nn_c):
                i = node // n_c
                j = node - i * n_c
                u_mat[i, j] = accumulate_dtype(read_node_value(input_args, element_index, node))

            stage_i = wp.tile_matmul(a_tile, u_mat)  # (q, n) [qx, j]
            stage_d = wp.tile_matmul(d_tile, u_mat)
            uq = wp.tile_zeros(shape=(q_c, q_c), dtype=accumulate_dtype)
            wp.tile_matmul(stage_i, a_t, uq)  # value [qx, qy]
            g_xi = wp.tile_zeros(shape=(q_c, q_c), dtype=accumulate_dtype)
            wp.tile_matmul(stage_d, a_t, g_xi)  # d/dxi
            g_eta = wp.tile_zeros(shape=(q_c, q_c), dtype=accumulate_dtype)
            wp.tile_matmul(stage_i, d_t, g_eta)  # d/deta

            # --- D stage: seeded integrand evaluations per quadrature point
            f0 = wp.tile_zeros(shape=(q_c, q_c), dtype=accumulate_dtype)
            if wp.static(test_uses_grad):
                f1_xi = wp.tile_zeros(shape=(q_c, q_c), dtype=accumulate_dtype)
                f1_eta = wp.tile_zeros(shape=(q_c, q_c), dtype=accumulate_dtype)

            qp_fields = FieldStruct()
            _copy_qp_fields()

            for qp in range(qq_c):
                qx = qp // q_c
                qy = qp - qx * q_c
                qp_index = quadrature.point_index(domain_arg, qp_arg, domain_element_index, element_index, qp)
                qp_coords = quadrature.point_coords(domain_arg, qp_arg, domain_element_index, element_index, qp)
                qp_weight = quadrature.point_weight(domain_arg, qp_arg, domain_element_index, element_index, qp)

                free_sample = make_free_sample(element_index, qp_coords)
                vol = domain.element_measure(domain_arg, free_sample)
                scale = accumulate_dtype(qp_weight * vol)
                jac = domain.element_deformation_gradient(domain_arg, free_sample)
                jac_inv = wp.inverse(jac)

                # Inject the B-stage interpolated value and physical gradient
                inj_arg = InjectedEvalArg()
                inj_arg.value = value_type(uq[qx, qy])
                g_ref = grad_type(value_type(g_xi[qx, qy]), value_type(g_eta[qx, qy]))
                inj_arg.gradient = wp.transpose(jac_inv) * g_ref
                _set_injected_eval_arg()

                # Value seed (v = 1, grad v = 0) -> f0
                sample = SampleType(element_index, qp_coords, qp_index, qp_weight, DofIndex(0, 0), NULL_DOF_INDEX)
                f0[qx, qy] = scale * accumulate_dtype(integrand_func(sample, qp_fields, values))

                # Gradient seeds (v = 0, grad v = e_i) -> f1; the seeds are
                # physical, so map back to reference space with J^{-1}.
                if wp.static(test_uses_grad):
                    f1_phys = grad_type()
                    for seed in range(2):
                        sample = SampleType(
                            element_index, qp_coords, qp_index, qp_weight, DofIndex(seed + 1, 0), NULL_DOF_INDEX
                        )
                        f1_phys[seed] = value_type(integrand_func(sample, qp_fields, values))
                    f1_ref = jac_inv * f1_phys
                    f1_xi[qx, qy] = scale * accumulate_dtype(f1_ref[0])
                    f1_eta[qx, qy] = scale * accumulate_dtype(f1_ref[1])

            # --- B^T stage: contract (f0, f1) back to nodal residuals
            # r = Kron(I, I)^T f0 + Kron(D, I)^T f1_xi + Kron(I, D)^T f1_eta
            r = wp.tile_zeros(shape=(n_c, n_c), dtype=accumulate_dtype)
            tmp0 = wp.tile_matmul(a_t, f0)  # (n, q)
            wp.tile_matmul(tmp0, a_tile, r)
            if wp.static(test_uses_grad):
                tmp1 = wp.tile_matmul(d_t, f1_xi)
                wp.tile_matmul(tmp1, a_tile, r)
                tmp2 = wp.tile_matmul(a_t, f1_eta)
                wp.tile_matmul(tmp2, d_tile, r)

            r_flat = wp.tile_reshape(r, shape=(1, nn_c))
            wp.tile_store(result_elem, r_flat, offset=(domain_element_index, 0))

        return integrate_kernel_fn

    nnn_c = wp.constant(n * n * n)
    qqq_c = wp.constant(q * q * q)

    def integrate_kernel_fn(
        qp_arg: quadrature.Arg,
        domain_arg: domain.ElementArg,
        domain_index_arg: domain.ElementIndexArg,
        fields: FieldStruct,
        values: ValueStruct,
        input_eval_arg: input_field.EvalArg,
        interp: wp.array2d(dtype=accumulate_dtype),
        deriv: wp.array2d(dtype=accumulate_dtype),
        result_elem: wp.array2d(dtype=accumulate_dtype),
    ):
        domain_element_index = wp.tid()
        element_index = domain.element_index(domain_index_arg, domain_element_index)

        a_tile = wp.tile_load(interp, shape=(q_c, n_c))
        d_tile = wp.tile_load(deriv, shape=(q_c, n_c))
        a_t = wp.tile_transpose(a_tile)
        d_t = wp.tile_transpose(d_tile)

        # --- B stage: gather element DOFs and interpolate to quadrature points
        input_args = InputElementEvalArg(domain_arg, input_eval_arg)
        u_mat = wp.tile_zeros(shape=(n_c, nn_c), dtype=accumulate_dtype)  # [i, j*n + k]
        for node in range(nnn_c):
            i = node // nn_c
            jk = node - i * nn_c
            u_mat[i, jk] = accumulate_dtype(read_node_value(input_args, element_index, node))

        stage_i = wp.tile_matmul(a_tile, u_mat)  # (q, n^2) [qx, j*n + k]
        stage_d = wp.tile_matmul(d_tile, u_mat)

        uq = wp.tile_zeros(shape=(q_c, qq_c), dtype=accumulate_dtype)  # [qx, qy*q + qz]
        g_xi = wp.tile_zeros(shape=(q_c, qq_c), dtype=accumulate_dtype)
        g_eta = wp.tile_zeros(shape=(q_c, qq_c), dtype=accumulate_dtype)
        g_zeta = wp.tile_zeros(shape=(q_c, qq_c), dtype=accumulate_dtype)

        for qx in range(q_c):
            row_i = wp.tile_view(stage_i, offset=(qx, 0), shape=(1, nn_c))
            block_i = wp.tile_reshape(row_i, shape=(n_c, n_c))  # [j, k]
            row_d = wp.tile_view(stage_d, offset=(qx, 0), shape=(1, nn_c))
            block_d = wp.tile_reshape(row_d, shape=(n_c, n_c))

            slab_ii = wp.tile_matmul(a_tile, block_i)  # (q, n) [qy, k]
            slab_di = wp.tile_matmul(d_tile, block_i)
            slab_id = wp.tile_matmul(a_tile, block_d)

            val_qx = wp.tile_matmul(slab_ii, a_t)  # (q, q) [qy, qz]
            gz_qx = wp.tile_matmul(slab_ii, d_t)
            gy_qx = wp.tile_matmul(slab_di, a_t)
            gx_qx = wp.tile_matmul(slab_id, a_t)

            wp.tile_assign(uq, wp.tile_reshape(val_qx, shape=(1, qq_c)), offset=(qx, 0))
            wp.tile_assign(g_xi, wp.tile_reshape(gx_qx, shape=(1, qq_c)), offset=(qx, 0))
            wp.tile_assign(g_eta, wp.tile_reshape(gy_qx, shape=(1, qq_c)), offset=(qx, 0))
            wp.tile_assign(g_zeta, wp.tile_reshape(gz_qx, shape=(1, qq_c)), offset=(qx, 0))

        # --- D stage: seeded integrand evaluations per quadrature point
        f0 = wp.tile_zeros(shape=(q_c, qq_c), dtype=accumulate_dtype)
        if wp.static(test_uses_grad):
            f1_xi = wp.tile_zeros(shape=(q_c, qq_c), dtype=accumulate_dtype)
            f1_eta = wp.tile_zeros(shape=(q_c, qq_c), dtype=accumulate_dtype)
            f1_zeta = wp.tile_zeros(shape=(q_c, qq_c), dtype=accumulate_dtype)

        qp_fields = FieldStruct()
        _copy_qp_fields()

        for qp in range(qqq_c):
            qx = qp // qq_c
            idx = qp - qx * qq_c  # qy * q + qz
            qp_index = quadrature.point_index(domain_arg, qp_arg, domain_element_index, element_index, qp)
            qp_coords = quadrature.point_coords(domain_arg, qp_arg, domain_element_index, element_index, qp)
            qp_weight = quadrature.point_weight(domain_arg, qp_arg, domain_element_index, element_index, qp)

            free_sample = make_free_sample(element_index, qp_coords)
            vol = domain.element_measure(domain_arg, free_sample)
            scale = accumulate_dtype(qp_weight * vol)
            jac = domain.element_deformation_gradient(domain_arg, free_sample)
            jac_inv = wp.inverse(jac)

            # Inject the B-stage interpolated value and physical gradient
            inj_arg = InjectedEvalArg()
            inj_arg.value = value_type(uq[qx, idx])
            g_ref = grad_type(value_type(g_xi[qx, idx]), value_type(g_eta[qx, idx]), value_type(g_zeta[qx, idx]))
            inj_arg.gradient = wp.transpose(jac_inv) * g_ref
            _set_injected_eval_arg()

            # Value seed (v = 1, grad v = 0) -> f0
            sample = SampleType(element_index, qp_coords, qp_index, qp_weight, DofIndex(0, 0), NULL_DOF_INDEX)
            f0[qx, idx] = scale * accumulate_dtype(integrand_func(sample, qp_fields, values))

            # Gradient seeds (v = 0, grad v = e_i) -> f1; the seeds are
            # physical, so map back to reference space with J^{-1}.
            if wp.static(test_uses_grad):
                f1_phys = grad_type()
                for seed in range(3):
                    sample = SampleType(
                        element_index, qp_coords, qp_index, qp_weight, DofIndex(seed + 1, 0), NULL_DOF_INDEX
                    )
                    f1_phys[seed] = value_type(integrand_func(sample, qp_fields, values))
                f1_ref = jac_inv * f1_phys
                f1_xi[qx, idx] = scale * accumulate_dtype(f1_ref[0])
                f1_eta[qx, idx] = scale * accumulate_dtype(f1_ref[1])
                f1_zeta[qx, idx] = scale * accumulate_dtype(f1_ref[2])

        # --- B^T stage: contract the coefficient channels back to nodal residuals
        # channels: f0 -> Kron(I,I,I)^T, f1_xi -> Kron(D,I,I)^T,
        #           f1_eta -> Kron(I,D,I)^T, f1_zeta -> Kron(I,I,D)^T
        g0 = wp.tile_matmul(a_t, f0)  # (n, q^2) [i, qy*q + qz]
        if wp.static(test_uses_grad):
            g1 = wp.tile_matmul(d_t, f1_xi)
            g2 = wp.tile_matmul(a_t, f1_eta)
            g3 = wp.tile_matmul(a_t, f1_zeta)

        r = wp.tile_zeros(shape=(n_c, nn_c), dtype=accumulate_dtype)  # [i, j*n + k]
        for i in range(n_c):
            b0 = wp.tile_reshape(wp.tile_view(g0, offset=(i, 0), shape=(1, qq_c)), shape=(q_c, q_c))  # [qy, qz]
            h0 = wp.tile_matmul(a_t, b0)  # (n, q) [j, qz]

            r_blk = wp.tile_zeros(shape=(n_c, n_c), dtype=accumulate_dtype)  # [j, k]
            wp.tile_matmul(h0, a_tile, r_blk)

            if wp.static(test_uses_grad):
                b1 = wp.tile_reshape(wp.tile_view(g1, offset=(i, 0), shape=(1, qq_c)), shape=(q_c, q_c))
                b2 = wp.tile_reshape(wp.tile_view(g2, offset=(i, 0), shape=(1, qq_c)), shape=(q_c, q_c))
                b3 = wp.tile_reshape(wp.tile_view(g3, offset=(i, 0), shape=(1, qq_c)), shape=(q_c, q_c))

                h1 = wp.tile_matmul(a_t, b1)
                h2 = wp.tile_matmul(d_t, b2)
                h3 = wp.tile_matmul(a_t, b3)

                wp.tile_matmul(h1, a_tile, r_blk)
                wp.tile_matmul(h2, a_tile, r_blk)
                wp.tile_matmul(h3, d_tile, r_blk)

            wp.tile_assign(r, wp.tile_reshape(r_blk, shape=(1, nn_c)), offset=(i, 0))

        r_flat = wp.tile_reshape(r, shape=(1, nnn_c))
        wp.tile_store(result_elem, r_flat, offset=(domain_element_index, 0))

    return integrate_kernel_fn


def get_integrate_bilinear_sumfac_kernel(
    integrand_func: wp.Function,
    domain: GeometryDomain,
    quadrature: Quadrature,
    FieldStruct,
    ValueStruct,
    test: TestField,
    trial: TrialField,
    *,
    n: int,
    q: int,
    dim: int,
    element_batch: int,
    test_uses_grad: bool,
    trial_uses_grad: bool,
    accumulate_dtype,
):
    """Build the fused sum-factorized assembly kernel body for a qualifying bilinear form.

    Returns a ``kernel_fn`` closure to be compiled through
    ``cache.get_integrand_kernel`` with a ``PassFieldArgsToIntegrand`` code
    transformer, exactly like the other ``get_integrate_*_kernel`` factories
    in :mod:`warp._src.fem.integrate`. Unlike the linear apply kernel, no
    field injection is needed: both the test and the trial field have been
    substituted with seed fields whose selectors travel through the
    ``Sample``, so the kernel's ``fields`` argument is passed to the
    transformed integrand unchanged.

    The kernel is launched with ``wp.launch_tiled(dim=[element_count])``, one
    element per block. Per element it first evaluates the per-quadrature-
    point channel coefficients ``c[a, b]`` (the ``D`` stage, ``(1 + d)^2``
    seeded integrand evaluations per point, gradient channels mapped to
    reference space with ``J^{-1}`` on both sides), then forms the local
    block column by column as the ``B^T D B`` action on the one-hot trial
    vectors, storing each column to the per-element staging array. All loops
    have compile-time bounds and there are no early returns (uniform control
    flow over the tile operations).

    Args:
        integrand_func: Transformed integrand (with the seed fields
            substituted).
        domain: Cell domain of the integration.
        quadrature: Tensor-product quadrature formula.
        FieldStruct: Generated field-argument struct (post-substitution).
        ValueStruct: Generated value-argument struct.
        test: Original (plain) test field.
        trial: Original (plain) trial field (same space as ``test``).
        n: Nodes per axis of the (test and trial) space, ``degree + 1``.
        q: Quadrature points per axis.
        dim: Spatial dimension (2 or 3).
        element_batch: Elements per block; only ``1`` is implemented.
        test_uses_grad: Whether the integrand applies a gradient operator to
            the test field; when ``False`` the test gradient channels are
            omitted entirely (matching the legacy dispatch path).
        trial_uses_grad: Same for the trial field.
        accumulate_dtype: Scalar type used for the tile contractions and
            coefficient accumulation.
    """
    if element_batch != 1:
        raise NotImplementedError("Sum-factorized integration currently requires element_batch == 1")

    SampleType = domain.geometry.sample_type
    value_type = type_scalar_type(test.dtype)
    grad_type = test.gradient_dtype
    chmat_type = cache.cached_mat_type(shape=(dim + 1, dim + 1), dtype=value_type)

    TEST_CHANNELS = 1 + (dim if test_uses_grad else 0)
    TRIAL_CHANNELS = 1 + (dim if trial_uses_grad else 0)

    # Flat loops with index arithmetic (see get_integrate_linear_sumfac_kernel)
    n_c = wp.constant(n)
    q_c = wp.constant(q)
    nn_c = wp.constant(n * n)
    qq_c = wp.constant(q * q)

    if dim == 2:

        def integrate_kernel_fn(
            qp_arg: quadrature.Arg,
            domain_arg: domain.ElementArg,
            domain_index_arg: domain.ElementIndexArg,
            fields: FieldStruct,
            values: ValueStruct,
            interp: wp.array2d(dtype=accumulate_dtype),
            deriv: wp.array2d(dtype=accumulate_dtype),
            staging: wp.array2d(dtype=accumulate_dtype),
        ):
            domain_element_index = wp.tid()
            element_index = domain.element_index(domain_index_arg, domain_element_index)

            a_tile = wp.tile_load(interp, shape=(q_c, n_c))
            d_tile = wp.tile_load(deriv, shape=(q_c, n_c))
            a_t = wp.tile_transpose(a_tile)
            d_t = wp.tile_transpose(d_tile)

            # --- D stage: seeded channel coefficients at every quadrature point
            c_vv = wp.tile_zeros(shape=(q_c, q_c), dtype=accumulate_dtype)
            if wp.static(trial_uses_grad):
                c_vg_x = wp.tile_zeros(shape=(q_c, q_c), dtype=accumulate_dtype)
                c_vg_y = wp.tile_zeros(shape=(q_c, q_c), dtype=accumulate_dtype)
            if wp.static(test_uses_grad):
                c_gv_x = wp.tile_zeros(shape=(q_c, q_c), dtype=accumulate_dtype)
                c_gv_y = wp.tile_zeros(shape=(q_c, q_c), dtype=accumulate_dtype)
            if wp.static(test_uses_grad and trial_uses_grad):
                c_gg_xx = wp.tile_zeros(shape=(q_c, q_c), dtype=accumulate_dtype)
                c_gg_xy = wp.tile_zeros(shape=(q_c, q_c), dtype=accumulate_dtype)
                c_gg_yx = wp.tile_zeros(shape=(q_c, q_c), dtype=accumulate_dtype)
                c_gg_yy = wp.tile_zeros(shape=(q_c, q_c), dtype=accumulate_dtype)

            for qp in range(qq_c):
                qx = qp // q_c
                qy = qp - qx * q_c
                qp_index = quadrature.point_index(domain_arg, qp_arg, domain_element_index, element_index, qp)
                qp_coords = quadrature.point_coords(domain_arg, qp_arg, domain_element_index, element_index, qp)
                qp_weight = quadrature.point_weight(domain_arg, qp_arg, domain_element_index, element_index, qp)

                free_sample = make_free_sample(element_index, qp_coords)
                vol = domain.element_measure(domain_arg, free_sample)
                scale = accumulate_dtype(qp_weight * vol)
                jac = domain.element_deformation_gradient(domain_arg, free_sample)
                jac_inv = wp.inverse(jac)

                # Physical channel matrix: integrand at (test seed a, trial seed b)
                ch = chmat_type()
                for a in range(TEST_CHANNELS):
                    for b in range(TRIAL_CHANNELS):
                        sample = SampleType(
                            element_index, qp_coords, qp_index, qp_weight, DofIndex(a, 0), DofIndex(b, 0)
                        )
                        ch[a, b] = value_type(integrand_func(sample, fields, values))

                # Map the physical gradient channels to reference space with
                # J^{-1} on both sides (grad_phys = J^{-T} grad_ref).
                if wp.static(test_uses_grad):
                    for b in range(TRIAL_CHANNELS):
                        gv = jac_inv * grad_type(ch[1, b], ch[2, b])
                        ch[1, b] = gv[0]
                        ch[2, b] = gv[1]
                if wp.static(trial_uses_grad):
                    for a in range(TEST_CHANNELS):
                        gu = jac_inv * grad_type(ch[a, 1], ch[a, 2])
                        ch[a, 1] = gu[0]
                        ch[a, 2] = gu[1]

                c_vv[qx, qy] = scale * accumulate_dtype(ch[0, 0])
                if wp.static(trial_uses_grad):
                    c_vg_x[qx, qy] = scale * accumulate_dtype(ch[0, 1])
                    c_vg_y[qx, qy] = scale * accumulate_dtype(ch[0, 2])
                if wp.static(test_uses_grad):
                    c_gv_x[qx, qy] = scale * accumulate_dtype(ch[1, 0])
                    c_gv_y[qx, qy] = scale * accumulate_dtype(ch[2, 0])
                if wp.static(test_uses_grad and trial_uses_grad):
                    c_gg_xx[qx, qy] = scale * accumulate_dtype(ch[1, 1])
                    c_gg_xy[qx, qy] = scale * accumulate_dtype(ch[1, 2])
                    c_gg_yx[qx, qy] = scale * accumulate_dtype(ch[2, 1])
                    c_gg_yy[qx, qy] = scale * accumulate_dtype(ch[2, 2])

            # --- Local block, column by column: B^T D B on one-hot trial vectors
            for col in range(nn_c):
                j1 = col // n_c
                j2 = col - j1 * n_c
                u_mat = wp.tile_zeros(shape=(n_c, n_c), dtype=accumulate_dtype)
                u_mat[j1, j2] = accumulate_dtype(1.0)

                # B stage: one-hot trial values (and reference gradients) at QPs
                stage_i = wp.tile_matmul(a_tile, u_mat)  # (q, n) [qx, j]
                uq = wp.tile_matmul(stage_i, a_t)  # value [qx, qy]
                if wp.static(trial_uses_grad):
                    stage_d = wp.tile_matmul(d_tile, u_mat)
                    g_xi = wp.tile_matmul(stage_d, a_t)
                    g_eta = wp.tile_matmul(stage_i, d_t)

                # Channel combine: f_a = sum_b c[a, b] * phi_b
                if wp.static(trial_uses_grad):
                    f0 = c_vv * uq + c_vg_x * g_xi + c_vg_y * g_eta
                else:
                    f0 = c_vv * uq
                if wp.static(test_uses_grad and trial_uses_grad):
                    f1_xi = c_gv_x * uq + c_gg_xx * g_xi + c_gg_xy * g_eta
                    f1_eta = c_gv_y * uq + c_gg_yx * g_xi + c_gg_yy * g_eta
                elif wp.static(test_uses_grad):
                    f1_xi = c_gv_x * uq
                    f1_eta = c_gv_y * uq

                # B^T stage: contract the coefficients back to test nodal values
                r = wp.tile_zeros(shape=(n_c, n_c), dtype=accumulate_dtype)
                tmp0 = wp.tile_matmul(a_t, f0)  # (n, q)
                wp.tile_matmul(tmp0, a_tile, r)
                if wp.static(test_uses_grad):
                    tmp1 = wp.tile_matmul(d_t, f1_xi)
                    wp.tile_matmul(tmp1, a_tile, r)
                    tmp2 = wp.tile_matmul(a_t, f1_eta)
                    wp.tile_matmul(tmp2, d_tile, r)

                r_flat = wp.tile_reshape(r, shape=(1, nn_c))
                wp.tile_store(staging, r_flat, offset=(domain_element_index * nn_c + col, 0))

        return integrate_kernel_fn

    nnn_c = wp.constant(n * n * n)
    qqq_c = wp.constant(q * q * q)

    def integrate_kernel_fn(
        qp_arg: quadrature.Arg,
        domain_arg: domain.ElementArg,
        domain_index_arg: domain.ElementIndexArg,
        fields: FieldStruct,
        values: ValueStruct,
        interp: wp.array2d(dtype=accumulate_dtype),
        deriv: wp.array2d(dtype=accumulate_dtype),
        staging: wp.array2d(dtype=accumulate_dtype),
    ):
        domain_element_index = wp.tid()
        element_index = domain.element_index(domain_index_arg, domain_element_index)

        a_tile = wp.tile_load(interp, shape=(q_c, n_c))
        d_tile = wp.tile_load(deriv, shape=(q_c, n_c))
        a_t = wp.tile_transpose(a_tile)
        d_t = wp.tile_transpose(d_tile)

        # --- D stage: seeded channel coefficients at every quadrature point
        c_vv = wp.tile_zeros(shape=(q_c, qq_c), dtype=accumulate_dtype)  # [qx, qy*q + qz]
        if wp.static(trial_uses_grad):
            c_vg_x = wp.tile_zeros(shape=(q_c, qq_c), dtype=accumulate_dtype)
            c_vg_y = wp.tile_zeros(shape=(q_c, qq_c), dtype=accumulate_dtype)
            c_vg_z = wp.tile_zeros(shape=(q_c, qq_c), dtype=accumulate_dtype)
        if wp.static(test_uses_grad):
            c_gv_x = wp.tile_zeros(shape=(q_c, qq_c), dtype=accumulate_dtype)
            c_gv_y = wp.tile_zeros(shape=(q_c, qq_c), dtype=accumulate_dtype)
            c_gv_z = wp.tile_zeros(shape=(q_c, qq_c), dtype=accumulate_dtype)
        if wp.static(test_uses_grad and trial_uses_grad):
            c_gg_xx = wp.tile_zeros(shape=(q_c, qq_c), dtype=accumulate_dtype)
            c_gg_xy = wp.tile_zeros(shape=(q_c, qq_c), dtype=accumulate_dtype)
            c_gg_xz = wp.tile_zeros(shape=(q_c, qq_c), dtype=accumulate_dtype)
            c_gg_yx = wp.tile_zeros(shape=(q_c, qq_c), dtype=accumulate_dtype)
            c_gg_yy = wp.tile_zeros(shape=(q_c, qq_c), dtype=accumulate_dtype)
            c_gg_yz = wp.tile_zeros(shape=(q_c, qq_c), dtype=accumulate_dtype)
            c_gg_zx = wp.tile_zeros(shape=(q_c, qq_c), dtype=accumulate_dtype)
            c_gg_zy = wp.tile_zeros(shape=(q_c, qq_c), dtype=accumulate_dtype)
            c_gg_zz = wp.tile_zeros(shape=(q_c, qq_c), dtype=accumulate_dtype)

        for qp in range(qqq_c):
            qx = qp // qq_c
            idx = qp - qx * qq_c  # qy * q + qz
            qp_index = quadrature.point_index(domain_arg, qp_arg, domain_element_index, element_index, qp)
            qp_coords = quadrature.point_coords(domain_arg, qp_arg, domain_element_index, element_index, qp)
            qp_weight = quadrature.point_weight(domain_arg, qp_arg, domain_element_index, element_index, qp)

            free_sample = make_free_sample(element_index, qp_coords)
            vol = domain.element_measure(domain_arg, free_sample)
            scale = accumulate_dtype(qp_weight * vol)
            jac = domain.element_deformation_gradient(domain_arg, free_sample)
            jac_inv = wp.inverse(jac)

            # Physical channel matrix: integrand at (test seed a, trial seed b)
            ch = chmat_type()
            for a in range(TEST_CHANNELS):
                for b in range(TRIAL_CHANNELS):
                    sample = SampleType(element_index, qp_coords, qp_index, qp_weight, DofIndex(a, 0), DofIndex(b, 0))
                    ch[a, b] = value_type(integrand_func(sample, fields, values))

            # Map the physical gradient channels to reference space with
            # J^{-1} on both sides (grad_phys = J^{-T} grad_ref).
            if wp.static(test_uses_grad):
                for b in range(TRIAL_CHANNELS):
                    gv = jac_inv * grad_type(ch[1, b], ch[2, b], ch[3, b])
                    ch[1, b] = gv[0]
                    ch[2, b] = gv[1]
                    ch[3, b] = gv[2]
            if wp.static(trial_uses_grad):
                for a in range(TEST_CHANNELS):
                    gu = jac_inv * grad_type(ch[a, 1], ch[a, 2], ch[a, 3])
                    ch[a, 1] = gu[0]
                    ch[a, 2] = gu[1]
                    ch[a, 3] = gu[2]

            c_vv[qx, idx] = scale * accumulate_dtype(ch[0, 0])
            if wp.static(trial_uses_grad):
                c_vg_x[qx, idx] = scale * accumulate_dtype(ch[0, 1])
                c_vg_y[qx, idx] = scale * accumulate_dtype(ch[0, 2])
                c_vg_z[qx, idx] = scale * accumulate_dtype(ch[0, 3])
            if wp.static(test_uses_grad):
                c_gv_x[qx, idx] = scale * accumulate_dtype(ch[1, 0])
                c_gv_y[qx, idx] = scale * accumulate_dtype(ch[2, 0])
                c_gv_z[qx, idx] = scale * accumulate_dtype(ch[3, 0])
            if wp.static(test_uses_grad and trial_uses_grad):
                c_gg_xx[qx, idx] = scale * accumulate_dtype(ch[1, 1])
                c_gg_xy[qx, idx] = scale * accumulate_dtype(ch[1, 2])
                c_gg_xz[qx, idx] = scale * accumulate_dtype(ch[1, 3])
                c_gg_yx[qx, idx] = scale * accumulate_dtype(ch[2, 1])
                c_gg_yy[qx, idx] = scale * accumulate_dtype(ch[2, 2])
                c_gg_yz[qx, idx] = scale * accumulate_dtype(ch[2, 3])
                c_gg_zx[qx, idx] = scale * accumulate_dtype(ch[3, 1])
                c_gg_zy[qx, idx] = scale * accumulate_dtype(ch[3, 2])
                c_gg_zz[qx, idx] = scale * accumulate_dtype(ch[3, 3])

        # --- Local block, column by column: B^T D B on one-hot trial vectors
        for col in range(nnn_c):
            j1 = col // nn_c
            jk = col - j1 * nn_c  # j2 * n + j3
            u_mat = wp.tile_zeros(shape=(n_c, nn_c), dtype=accumulate_dtype)  # [i, j*n + k]
            u_mat[j1, jk] = accumulate_dtype(1.0)

            # B stage: one-hot trial values (and reference gradients) at QPs
            stage_i = wp.tile_matmul(a_tile, u_mat)  # (q, n^2) [qx, j*n + k]
            if wp.static(trial_uses_grad):
                stage_d = wp.tile_matmul(d_tile, u_mat)

            uq = wp.tile_zeros(shape=(q_c, qq_c), dtype=accumulate_dtype)  # [qx, qy*q + qz]
            if wp.static(trial_uses_grad):
                g_xi = wp.tile_zeros(shape=(q_c, qq_c), dtype=accumulate_dtype)
                g_eta = wp.tile_zeros(shape=(q_c, qq_c), dtype=accumulate_dtype)
                g_zeta = wp.tile_zeros(shape=(q_c, qq_c), dtype=accumulate_dtype)

            for qx in range(q_c):
                row_i = wp.tile_view(stage_i, offset=(qx, 0), shape=(1, nn_c))
                block_i = wp.tile_reshape(row_i, shape=(n_c, n_c))  # [j, k]
                slab_ii = wp.tile_matmul(a_tile, block_i)  # (q, n) [qy, k]
                val_qx = wp.tile_matmul(slab_ii, a_t)  # (q, q) [qy, qz]
                wp.tile_assign(uq, wp.tile_reshape(val_qx, shape=(1, qq_c)), offset=(qx, 0))

                if wp.static(trial_uses_grad):
                    row_d = wp.tile_view(stage_d, offset=(qx, 0), shape=(1, nn_c))
                    block_d = wp.tile_reshape(row_d, shape=(n_c, n_c))
                    slab_di = wp.tile_matmul(d_tile, block_i)
                    slab_id = wp.tile_matmul(a_tile, block_d)
                    gz_qx = wp.tile_matmul(slab_ii, d_t)
                    gy_qx = wp.tile_matmul(slab_di, a_t)
                    gx_qx = wp.tile_matmul(slab_id, a_t)
                    wp.tile_assign(g_xi, wp.tile_reshape(gx_qx, shape=(1, qq_c)), offset=(qx, 0))
                    wp.tile_assign(g_eta, wp.tile_reshape(gy_qx, shape=(1, qq_c)), offset=(qx, 0))
                    wp.tile_assign(g_zeta, wp.tile_reshape(gz_qx, shape=(1, qq_c)), offset=(qx, 0))

            # Channel combine: f_a = sum_b c[a, b] * phi_b
            if wp.static(trial_uses_grad):
                f0 = c_vv * uq + c_vg_x * g_xi + c_vg_y * g_eta + c_vg_z * g_zeta
            else:
                f0 = c_vv * uq
            if wp.static(test_uses_grad and trial_uses_grad):
                f1_xi = c_gv_x * uq + c_gg_xx * g_xi + c_gg_xy * g_eta + c_gg_xz * g_zeta
                f1_eta = c_gv_y * uq + c_gg_yx * g_xi + c_gg_yy * g_eta + c_gg_yz * g_zeta
                f1_zeta = c_gv_z * uq + c_gg_zx * g_xi + c_gg_zy * g_eta + c_gg_zz * g_zeta
            elif wp.static(test_uses_grad):
                f1_xi = c_gv_x * uq
                f1_eta = c_gv_y * uq
                f1_zeta = c_gv_z * uq

            # B^T stage: contract the coefficients back to test nodal values
            g0 = wp.tile_matmul(a_t, f0)  # (n, q^2) [i, qy*q + qz]
            if wp.static(test_uses_grad):
                g1 = wp.tile_matmul(d_t, f1_xi)
                g2 = wp.tile_matmul(a_t, f1_eta)
                g3 = wp.tile_matmul(a_t, f1_zeta)

            r = wp.tile_zeros(shape=(n_c, nn_c), dtype=accumulate_dtype)  # [i, j*n + k]
            for i in range(n_c):
                b0 = wp.tile_reshape(wp.tile_view(g0, offset=(i, 0), shape=(1, qq_c)), shape=(q_c, q_c))  # [qy, qz]
                h0 = wp.tile_matmul(a_t, b0)  # (n, q) [j, qz]

                r_blk = wp.tile_zeros(shape=(n_c, n_c), dtype=accumulate_dtype)  # [j, k]
                wp.tile_matmul(h0, a_tile, r_blk)

                if wp.static(test_uses_grad):
                    b1 = wp.tile_reshape(wp.tile_view(g1, offset=(i, 0), shape=(1, qq_c)), shape=(q_c, q_c))
                    b2 = wp.tile_reshape(wp.tile_view(g2, offset=(i, 0), shape=(1, qq_c)), shape=(q_c, q_c))
                    b3 = wp.tile_reshape(wp.tile_view(g3, offset=(i, 0), shape=(1, qq_c)), shape=(q_c, q_c))

                    h1 = wp.tile_matmul(a_t, b1)
                    h2 = wp.tile_matmul(d_t, b2)
                    h3 = wp.tile_matmul(a_t, b3)

                    wp.tile_matmul(h1, a_tile, r_blk)
                    wp.tile_matmul(h2, a_tile, r_blk)
                    wp.tile_matmul(h3, d_tile, r_blk)

                wp.tile_assign(r, wp.tile_reshape(r_blk, shape=(1, nn_c)), offset=(i, 0))

            r_flat = wp.tile_reshape(r, shape=(1, nnn_c))
            wp.tile_store(staging, r_flat, offset=(domain_element_index * nnn_c + col, 0))

    return integrate_kernel_fn


def get_sumfac_triplet_fill_kernel(
    domain: GeometryDomain,
    test: TestField,
    trial: TrialField,
    staging_dtype,
    output_dtype,
):
    """Build (and cache) the kernel scattering per-element local blocks to BSR triplets.

    The fused assembly kernel stores column ``j`` of element ``e``'s local
    block at staging row ``e * nodes_per_element + j``; this kernel writes
    the matching ``(row, column, value)`` triplet for every entry, with the
    triplet index equal to the flattened staging index so that
    ``bsr_set_from_triplets`` receives one triplet per local-block entry.
    Nodes outside the test (row) or trial (column) partitions are marked with
    ``NULL_NODE_INDEX`` and ignored by the BSR construction, mirroring the
    legacy bilinear kernels.

    Args:
        domain: Cell domain of the integration.
        test: Test field (provides the row topology and partition).
        trial: Trial field (provides the column topology and partition).
        staging_dtype: Scalar type of the per-element staging array.
        output_dtype: Scalar type of the triplet values array.
    """
    test_topology = test.space.topology
    trial_topology = trial.space.topology
    test_partition = test.space_partition
    trial_partition = trial.space_partition

    @cache.dynamic_kernel(
        suffix=(
            "sumfac_triplet_fill",
            domain.name,
            test.space.name,
            test_partition.name,
            trial.space.name,
            trial_partition.name,
            cache.pod_type_key(staging_dtype),
            cache.pod_type_key(output_dtype),
        ),
        kernel_options={"enable_backward": False},
    )
    def sumfac_triplet_fill_kernel(
        domain_arg: domain.ElementArg,
        domain_index_arg: domain.ElementIndexArg,
        test_topo_arg: test_topology.TopologyArg,
        test_partition_arg: test_partition.PartitionArg,
        trial_topo_arg: trial_topology.TopologyArg,
        trial_partition_arg: trial_partition.PartitionArg,
        nodes_per_element: int,
        staging: wp.array2d(dtype=staging_dtype),
        triplet_rows: wp.array(dtype=int),
        triplet_cols: wp.array(dtype=int),
        triplet_values: wp.array3d(dtype=output_dtype),
    ):
        domain_element_index, trial_node, test_node = wp.tid()
        element_index = domain.element_index(domain_index_arg, domain_element_index)

        staging_row = domain_element_index * nodes_per_element + trial_node
        triplet_index = staging_row * nodes_per_element + test_node

        test_node_index = test_topology.element_node_index(domain_arg, test_topo_arg, element_index, test_node)
        row = test_partition.partition_node_index(test_partition_arg, test_node_index)
        trial_node_index = trial_topology.element_node_index(domain_arg, trial_topo_arg, element_index, trial_node)
        col = trial_partition.partition_node_index(trial_partition_arg, trial_node_index)
        if row == NULL_NODE_INDEX or col == NULL_NODE_INDEX:
            # Will get ignored when converting to BSR
            row = NULL_NODE_INDEX
            col = NULL_NODE_INDEX

        triplet_rows[triplet_index] = row
        triplet_cols[triplet_index] = col
        triplet_values[triplet_index, 0, 0] = output_dtype(staging[staging_row, test_node])

    return sumfac_triplet_fill_kernel


def get_sumfac_scatter_kernel(
    domain: GeometryDomain,
    space: FunctionSpace,
    space_partition: SpacePartition,
    staging_dtype,
    output_dtype,
):
    """Build (and cache) the kernel scattering per-element residuals to partition nodes.

    For discontinuous spaces every node belongs to exactly one element, so the
    accumulation is race-free without atomics; nodes outside the partition are
    skipped.

    Args:
        domain: Cell domain of the integration.
        space: Test function space (provides the element-to-node topology).
        space_partition: Test space partition indexing the output vector rows.
        staging_dtype: Scalar type of the per-element staging array.
        output_dtype: Scalar type of the output vector.
    """
    topology = space.topology

    @cache.dynamic_kernel(
        suffix=(
            "sumfac_scatter",
            domain.name,
            space.name,
            space_partition.name,
            cache.pod_type_key(staging_dtype),
            cache.pod_type_key(output_dtype),
        ),
        kernel_options={"enable_backward": False},
    )
    def sumfac_scatter_kernel(
        domain_arg: domain.ElementArg,
        domain_index_arg: domain.ElementIndexArg,
        topo_arg: topology.TopologyArg,
        partition_arg: space_partition.PartitionArg,
        staging: wp.array2d(dtype=staging_dtype),
        result: wp.array2d(dtype=output_dtype),
    ):
        domain_element_index, node_in_element = wp.tid()
        element_index = domain.element_index(domain_index_arg, domain_element_index)
        node_index = topology.element_node_index(domain_arg, topo_arg, element_index, node_in_element)
        partition_node_index = space_partition.partition_node_index(partition_arg, node_index)
        if partition_node_index == NULL_NODE_INDEX:
            return
        result[partition_node_index, 0] += output_dtype(staging[domain_element_index, node_in_element])

    return sumfac_scatter_kernel


# -- Matrix-free operator -------------------------------------------------------


def make_sumfac_linear_operator(
    integrand,
    fields: dict[str, Any],
    quadrature: Quadrature | None = None,
    values: dict[str, Any] | None = None,
    input_name: str | None = None,
    temporary_store=None,
    device=None,
) -> LinearOperator:
    """Wrap the sum-factorized apply of a linear form as a matrix-free operator.

    The returned :class:`warp.optim.linear.LinearOperator` computes
    ``z = alpha * A @ x + beta * y`` where ``A @ x`` is the integration of
    ``integrand`` with the input field's degrees of freedom set to ``x``. It
    can be passed directly to the :mod:`warp.optim.linear` iterative solvers
    (pass ``use_cuda_graph=False``, as each apply allocates temporaries).

    Each apply runs ``integrate()`` with the sum-factorization mode forced on
    for that call only (the global mode set through
    ``warp.fem.set_sumfac_mode`` is not touched); if the form does not
    structurally qualify, the apply transparently (and correctly) falls back
    to the legacy kernels. During an apply the input field's ``dof_values``
    are temporarily rebound to ``x`` and restored afterwards.

    Args:
        integrand: Linear form to apply, decorated with :func:`warp.fem.integrand`.
        fields: Field arguments to the integrand; must contain exactly one
            test field and the input :class:`warp.fem.DiscreteField` whose
            DOFs are replaced by ``x`` on every apply.
        quadrature: Quadrature formula; deduced from the field degrees if ``None``.
        values: Additional value arguments to the integrand.
        input_name: Name of the input field argument; deduced when ``fields``
            contains a single discrete field.
        temporary_store: Shared pool from which to allocate temporary arrays.
        device: Device on which to run the apply; defaults to the input
            field's DOF array device.
    """
    from warp._src.fem import integrate as fem_integrate  # noqa: PLC0415 (circular import)
    from warp._src.fem.linalg import array_axpy  # noqa: PLC0415 (circular import)

    test = next((f for f in fields.values() if isinstance(f, TestField)), None)
    if test is None:
        raise ValueError("The fields dictionary must contain a test field")

    if input_name is None:
        candidates = [name for name, f in fields.items() if isinstance(f, NodalField)]
        if len(candidates) != 1:
            raise ValueError(
                "Could not deduce the input field; pass input_name to select among "
                f"{candidates or 'no discrete fields'}"
            )
        input_name = candidates[0]
    input_field = fields[input_name]
    if not isinstance(input_field, NodalField):
        raise ValueError(f"Field '{input_name}' is not a discrete nodal field")

    device = wp.get_device(device if device is not None else input_field.dof_values.device)
    scalar_type = type_scalar_type(input_field.dof_dtype)
    node_count = test.space_partition.node_count()

    result = wp.empty(node_count, dtype=scalar_type, device=device)

    def matvec(x: wp.array, y: wp.array, z: wp.array, alpha, beta):
        # Force sum-factorization for this call only (via the private
        # integrate() argument): mutating the process-global mode here would
        # leak into fem.integrate() calls running concurrently on other
        # threads. The input field's DOFs are rebound to the solver iterate
        # for the duration of the apply and restored afterwards.
        previous_dof_values = input_field.dof_values
        input_field.dof_values = x
        try:
            fem_integrate.integrate(
                integrand,
                fields=fields,
                values=values,
                quadrature=quadrature,
                output=result,
                device=device,
                temporary_store=temporary_store,
                _sumfac_mode="force",
            )
        finally:
            input_field.dof_values = previous_dof_values

        if z.ptr != y.ptr:
            wp.copy(z, y)
        array_axpy(x=result, y=z, alpha=alpha, beta=beta)

    return LinearOperator(shape=(node_count, node_count), dtype=scalar_type, device=device, matvec=matvec)
