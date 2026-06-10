# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Pointwise Q-function extraction for the sum-factorized ``B^T D B`` pipeline.

Any integrand that is linear in the test function ``v`` and accesses it only
through its value and gradient (the ``inner``/``outer``/``grad``/``grad_outer``
operators) can be written, at a single quadrature point, as
``f0 * v + dot(f1, grad v)``. The ``D`` stage of
the partial-assembly factorization only needs the coefficients ``(f0, f1)``:
the ``B^T`` stage then contracts them against the test basis. This module
recovers ``(f0, f1)`` from the *unmodified* user integrand by substituting the
test field with a :class:`warp._src.fem.field.SeedField` and evaluating the
integrand ``d + 1`` times per quadrature point with the seeds ``(v=1,
grad v=0)`` (yielding ``f0``) and ``(v=0, grad v=e_i)`` (yielding ``f1[i]``).
Because the form is linear in ``v``, the ``d + 1`` evaluations are exact, and
because :mod:`warp._src.fem.operator` resolves ``inner``/``grad`` directly to
the field's ``eval_inner``/``eval_grad_inner`` device functions, the
integrand body requires no source changes.

Seed injection channel
----------------------

The seed selector is encoded in the ``test_dof`` slot of the :class:`Sample`
(see :class:`warp._src.fem.field.SeedField` for the rationale): the consumer
builds a fresh ``Sample`` per seeded evaluation, so the seeds live entirely in
registers — no per-launch state, no extra memory traffic. This is the same
mechanism the Phase 3 fused kernel will use: after its ``B`` stage produces
interpolated ``(u, ref-grad u)`` tiles, its ``D`` stage loops the ``d + 1``
seeds per quadrature point by rebuilding ``Sample.test_dof``.

Trial-value injection (Phase 3)
-------------------------------

The fused kernel must also inject the *trial* values interpolated by the ``B``
stage — the interpolated ``(u, grad u)`` at the quadrature point — instead of
re-evaluating the trial basis. One-hot seeds cannot represent arbitrary
values, so this will be served by a sibling of ``SeedField`` whose ``EvalArg``
holds plain ``value``/``gradient`` struct members that the staged kernel fills
in-kernel (per quadrature point, from shared-memory tiles) before calling the
transformed integrand; the operator-resolution mechanism is identical, only
the storage channel differs. In this phase, the host-driven path simply
evaluates the trial argument as a regular :class:`DiscreteField`, which is
numerically equivalent to injecting the ``B``-stage interpolation (Phase 1
guarantees ``B == dense basis evaluation``), including the ``grad u = J^{-T}
ref-grad u`` mapping that the standard machinery applies.

Geometry folding
----------------

With ``fold_geometry=True`` (the default), the extracted coefficients are
multiplied by the quadrature weight times ``|det J|``, i.e. exactly the factor
the standard integration kernels apply per sample, so that the ``B^T`` stage
reduces to a plain transposed contraction. :func:`reference_geometry_factors`
exposes the per-quadrature-point ``(J^{-T}, |det J|)`` pair; it is evaluated
per point even though affine geometries (``Grid2D``/``Grid3D``) yield
element-constant values, so curved elements can slot in without interface
changes.

The extraction strategy is intentionally swappable: a one-pass Warp AD
extraction can later replace the ``d + 1`` seeded evaluations behind the same
entry points if heavy Q-functions make the ``d + 1`` factor noticeable.
"""

from typing import Any

import warp as wp
from warp._src.fem import cache
from warp._src.fem import operator as fem_operator
from warp._src.fem.domain import GeometryDomain
from warp._src.fem.field import FieldLike, SeedField, SideSeedField
from warp._src.fem.integrate import (
    IntegrandTransformer,
    PassFieldArgsToIntegrand,
    _check_field_compat,
    _find_integrand_operators,
    _gen_field_struct,
    _notify_operator_usage,
    _parse_integrand_arguments,
)
from warp._src.fem.operator import Integrand
from warp._src.fem.quadrature import Quadrature
from warp._src.fem.types import (
    NULL_DOF_INDEX,
    DofIndex,
    ElementKind,
    make_free_sample,
)

__all__ = [
    "extract_qfunction",
    "extract_side_qfunction",
    "reference_geometry_factors",
]

# Operators through which the seeded test field may be accessed. The seeded
# representation ``f0 * v + dot(f1, grad v)`` only captures the value and
# gradient of the test function; node-based operators (``at_node``,
# ``node_index``, ...) would silently misinterpret the seed selector as a node
# index, and divergence seeding is not defined for scalar spaces.
_SUPPORTED_TEST_OPERATORS = frozenset(
    (
        fem_operator.inner,
        fem_operator.outer,
        fem_operator.grad,
        fem_operator.grad_outer,
        fem_operator.degree,
    )
)


def _make_qfunction_eval_kernel_fn(
    integrand_func: wp.Function,
    domain: GeometryDomain,
    quadrature: Quadrature,
    FieldStruct: type,
    ValueStruct: type,
):
    """Build the host-driven kernel body recording ``(f0, f1)`` per quadrature point.

    The kernel evaluates the seeded integrand ``d + 1`` times per quadrature
    point, rebuilding the ``Sample`` with the seed selector encoded in
    ``test_dof`` (see module docstring). The variable names ``sample``,
    ``fields``, ``values``, ``domain_arg`` and ``domain_index_arg`` are
    significant: the ``PassFieldArgsToIntegrand`` code transformer rewrites the
    ``integrand_func`` calls in terms of them.
    """

    SampleType = domain.geometry.sample_type
    scalar_type = domain.geometry.scalar_type
    GRAD_DIM = domain.geometry.dimension
    grad_vec_type = cache.cached_vec_type(length=GRAD_DIM, dtype=scalar_type)

    def qfunction_eval_kernel_fn(
        qp_arg: quadrature.Arg,
        domain_arg: domain.ElementArg,
        domain_index_arg: domain.ElementIndexArg,
        fields: FieldStruct,
        values: ValueStruct,
        fold_geometry: int,
        f0: wp.array2d(dtype=scalar_type),
        f1: wp.array2d(dtype=grad_vec_type),
    ):
        domain_element_index, qp = wp.tid()

        element_index = domain.element_index(domain_index_arg, domain_element_index)
        qp_point_count = quadrature.point_count(domain_arg, qp_arg, domain_element_index, element_index)
        if qp >= qp_point_count:
            return

        qp_index = quadrature.point_index(domain_arg, qp_arg, domain_element_index, element_index, qp)
        qp_coords = quadrature.point_coords(domain_arg, qp_arg, domain_element_index, element_index, qp)
        qp_weight = quadrature.point_weight(domain_arg, qp_arg, domain_element_index, element_index, qp)

        vol = domain.element_measure(domain_arg, make_free_sample(element_index, qp_coords))
        scale = wp.where(fold_geometry != 0, qp_weight * vol, scalar_type(1.0))

        # Value seed (v = 1, grad v = 0) -> f0
        sample = SampleType(element_index, qp_coords, qp_index, qp_weight, DofIndex(0, 0), NULL_DOF_INDEX)
        f0[domain_element_index, qp] = scale * scalar_type(integrand_func(sample, fields, values))

        # Gradient seeds (v = 0, grad v = e_i) -> f1[i]
        f1_val = grad_vec_type()
        for seed in range(GRAD_DIM):
            sample = SampleType(element_index, qp_coords, qp_index, qp_weight, DofIndex(seed + 1, 0), NULL_DOF_INDEX)
            f1_val[seed] = scale * scalar_type(integrand_func(sample, fields, values))
        f1[domain_element_index, qp] = f1_val

    return qfunction_eval_kernel_fn


def _get_qfunction_kernel(
    integrand: Integrand,
    domain: GeometryDomain,
    quadrature: Quadrature,
    arguments,
):
    """Generate (or fetch from cache) the seeded Q-function evaluation kernel."""

    field_names = tuple((k, f.name) for k, f in arguments.field_args.items())
    kernel_suffix = ("qfunc", quadrature.name, field_names)

    kernel, field_struct, value_struct = cache.get_integrand_kernel(integrand=integrand, suffix=kernel_suffix)
    if kernel is not None:
        return kernel, field_struct, value_struct

    FieldStruct = _gen_field_struct(arguments.field_args)
    ValueStruct = cache.get_argument_struct(arguments.value_args)

    integrand_func = IntegrandTransformer.apply(
        integrand, arguments.field_args, sample_type=domain.geometry.sample_type
    )

    kernel_fn = _make_qfunction_eval_kernel_fn(integrand_func, domain, quadrature, FieldStruct, ValueStruct)

    kernel, _field_struct, _value_struct = cache.get_integrand_kernel(
        integrand=integrand,
        kernel_fn=kernel_fn,
        suffix=kernel_suffix,
        code_transformers=[
            PassFieldArgsToIntegrand(
                arg_names=integrand.argspec.args, parsed_args=arguments, integrand_func=integrand_func
            )
        ],
        FieldStruct=FieldStruct,
        ValueStruct=ValueStruct,
    )

    return kernel, FieldStruct(), ValueStruct()


def extract_qfunction(
    integrand: Integrand,
    fields: dict[str, FieldLike],
    quadrature: Quadrature,
    values: dict[str, Any] | None = None,
    fold_geometry: bool = True,
    device=None,
) -> tuple[wp.array, wp.array]:
    """Evaluate the pointwise Q-function coefficients ``(f0, f1)`` of an integrand.

    The test field found in ``fields`` is replaced by a
    :class:`warp._src.fem.field.SeedField` and the integrand is evaluated
    ``d + 1`` times per quadrature point with unit seeds, recovering the
    coefficients such that ``integrand == f0 * v + dot(f1, grad v)`` at every
    quadrature point (exact, since the form is linear in the test function).
    The gradient seeds are *physical* unit vectors, so ``f1`` multiplies the
    physical test gradient; the trial-side ``grad u = J^{-T} ref-grad u``
    mapping is applied by the standard field-evaluation machinery.

    Args:
        integrand: Form to extract from, decorated with :func:`warp.fem.integrand`.
            Must be linear in the (single) test field argument and must not take
            a trial field argument.
        fields: Field arguments to the integrand, keyed by parameter name. Must
            contain exactly one test field (e.g. from :func:`warp.fem.make_test`).
        quadrature: Quadrature formula defining the evaluation points; its
            domain is used as the integration domain.
        values: Additional value arguments to the integrand, keyed by parameter name.
        fold_geometry: If ``True``, multiply the coefficients by the quadrature
            weight times ``|det J|`` so that the ``B^T`` stage reduces to a plain
            transposed contraction.
        device: Warp device on which to evaluate.

    Returns:
        A pair of arrays of shape ``(element_count, points_per_element)``:
        ``f0`` with the geometry scalar type, and ``f1`` with the matching
        ``d``-vector type.

    Raises:
        NotImplementedError: If the quadrature domain is not a cell domain
            (the cell seeds cannot distinguish inner from outer traces, so
            e.g. ``jump(v)`` would silently extract as zero; use
            :func:`extract_side_qfunction` for DG face/side forms), or if the
            integrand accesses the test field through an operator other than
            ``inner``/``outer``/``grad``/``grad_outer``/``degree``
            (node-based operators are not representable as
            per-quadrature-point ``(f0, f1)`` coefficients).
    """

    domain = quadrature.domain
    if domain.element_kind != ElementKind.CELL:
        raise NotImplementedError(
            "Q-function extraction is only implemented for cell domains; "
            "use extract_side_qfunction for side (DG face) forms"
        )

    arguments = _parse_integrand_arguments(integrand, fields)
    _check_field_compat(integrand, arguments, domain)
    if arguments.test_name is None:
        raise ValueError("Q-function extraction requires a test field argument")
    if arguments.trial_name is not None:
        raise NotImplementedError(
            "Q-function extraction with a trial field is not supported yet; "
            "pass the trial argument as a DiscreteField instead"
        )

    test = arguments.field_args[arguments.test_name]
    if test.domain != domain:
        raise ValueError("Test field and quadrature must be defined over the same domain")

    field_args = dict(arguments.field_args)
    field_args[arguments.test_name] = SeedField.from_field(test)
    if arguments.domain_name is not None:
        field_args[arguments.domain_name] = domain
    arguments = arguments._replace(field_args=field_args)

    _find_integrand_operators(integrand, field_args)

    test_operators = integrand.operators.get(arguments.test_name, set())
    unsupported = test_operators - _SUPPORTED_TEST_OPERATORS
    if unsupported:
        op_names = ", ".join(sorted(op.name for op in unsupported))
        raise NotImplementedError(
            f"Q-function extraction does not support operator(s) [{op_names}] on the test field "
            f"'{arguments.test_name}'; only value and gradient evaluation "
            "(inner, outer, grad, grad_outer, degree) can be seeded"
        )

    _notify_operator_usage(integrand, field_args)

    kernel, field_struct, value_struct = _get_qfunction_kernel(integrand, domain, quadrature, arguments)

    device = wp.get_device(device)

    for name, field in field_args.items():
        if isinstance(field, FieldLike):
            field.fill_eval_arg(getattr(field_struct, name), device=device)
    cache.populate_argument_struct(value_struct, values, func_name=integrand.name)

    element_count = domain.element_count()
    points_per_element = quadrature.max_points_per_element()
    scalar_type = domain.geometry.scalar_type
    grad_vec_type = cache.cached_vec_type(length=domain.geometry.dimension, dtype=scalar_type)

    f0 = wp.zeros((element_count, points_per_element), dtype=scalar_type, device=device)
    f1 = wp.zeros((element_count, points_per_element), dtype=grad_vec_type, device=device)

    wp.launch(
        kernel,
        dim=(element_count, points_per_element),
        inputs=[
            quadrature.arg_value(device),
            domain.element_arg_value(device),
            domain.element_index_arg_value(device),
            field_struct,
            value_struct,
            1 if fold_geometry else 0,
            f0,
            f1,
        ],
        device=device,
    )

    return f0, f1


def _make_side_qfunction_eval_kernel_fn(
    integrand_func: wp.Function,
    domain: GeometryDomain,
    quadrature: Quadrature,
    FieldStruct: type,
    ValueStruct: type,
):
    """Build the host-driven kernel body recording the side trace-channel coefficients.

    The kernel evaluates the seeded integrand ``2 * (d + 1)`` times per side
    quadrature point, rebuilding the ``Sample`` with the
    :class:`warp._src.fem.field.SideSeedField` seed selector encoded in
    ``test_dof`` (inner value, inner gradient, outer value, outer gradient).
    The variable names ``sample``, ``fields``, ``values``, ``domain_arg`` and
    ``domain_index_arg`` are significant for ``PassFieldArgsToIntegrand``.
    """

    SampleType = domain.geometry.sample_type
    scalar_type = domain.geometry.scalar_type
    GRAD_DIM = domain.geometry.dimension
    OUTER_VALUE_SEED = wp.constant(1 + GRAD_DIM)
    OUTER_GRAD_BEGIN = wp.constant(2 + GRAD_DIM)
    grad_vec_type = cache.cached_vec_type(length=GRAD_DIM, dtype=scalar_type)

    def side_qfunction_eval_kernel_fn(
        qp_arg: quadrature.Arg,
        domain_arg: domain.ElementArg,
        domain_index_arg: domain.ElementIndexArg,
        fields: FieldStruct,
        values: ValueStruct,
        fold_geometry: int,
        f0_inner: wp.array2d(dtype=scalar_type),
        f1_inner: wp.array2d(dtype=grad_vec_type),
        f0_outer: wp.array2d(dtype=scalar_type),
        f1_outer: wp.array2d(dtype=grad_vec_type),
    ):
        domain_element_index, qp = wp.tid()

        element_index = domain.element_index(domain_index_arg, domain_element_index)
        qp_point_count = quadrature.point_count(domain_arg, qp_arg, domain_element_index, element_index)
        if qp >= qp_point_count:
            return

        qp_index = quadrature.point_index(domain_arg, qp_arg, domain_element_index, element_index, qp)
        qp_coords = quadrature.point_coords(domain_arg, qp_arg, domain_element_index, element_index, qp)
        qp_weight = quadrature.point_weight(domain_arg, qp_arg, domain_element_index, element_index, qp)

        vol = domain.element_measure(domain_arg, make_free_sample(element_index, qp_coords))
        scale = wp.where(fold_geometry != 0, qp_weight * vol, scalar_type(1.0))

        # Inner value seed (inner(v) = 1, all other channels 0) -> f0_inner
        sample = SampleType(element_index, qp_coords, qp_index, qp_weight, DofIndex(0, 0), NULL_DOF_INDEX)
        f0_inner[domain_element_index, qp] = scale * scalar_type(integrand_func(sample, fields, values))

        # Inner gradient seeds (grad(v) = e_i) -> f1_inner[i]
        f1_in = grad_vec_type()
        for seed in range(GRAD_DIM):
            sample = SampleType(element_index, qp_coords, qp_index, qp_weight, DofIndex(seed + 1, 0), NULL_DOF_INDEX)
            f1_in[seed] = scale * scalar_type(integrand_func(sample, fields, values))
        f1_inner[domain_element_index, qp] = f1_in

        # Outer value seed (outer(v) = 1) -> f0_outer
        sample = SampleType(
            element_index, qp_coords, qp_index, qp_weight, DofIndex(OUTER_VALUE_SEED, 0), NULL_DOF_INDEX
        )
        f0_outer[domain_element_index, qp] = scale * scalar_type(integrand_func(sample, fields, values))

        # Outer gradient seeds (grad_outer(v) = e_i) -> f1_outer[i]
        f1_out = grad_vec_type()
        for seed in range(GRAD_DIM):
            sample = SampleType(
                element_index, qp_coords, qp_index, qp_weight, DofIndex(OUTER_GRAD_BEGIN + seed, 0), NULL_DOF_INDEX
            )
            f1_out[seed] = scale * scalar_type(integrand_func(sample, fields, values))
        f1_outer[domain_element_index, qp] = f1_out

    return side_qfunction_eval_kernel_fn


def _get_side_qfunction_kernel(
    integrand: Integrand,
    domain: GeometryDomain,
    quadrature: Quadrature,
    arguments,
):
    """Generate (or fetch from cache) the seeded side Q-function evaluation kernel."""

    field_names = tuple((k, f.name) for k, f in arguments.field_args.items())
    kernel_suffix = ("side_qfunc", quadrature.name, field_names)

    kernel, field_struct, value_struct = cache.get_integrand_kernel(integrand=integrand, suffix=kernel_suffix)
    if kernel is not None:
        return kernel, field_struct, value_struct

    FieldStruct = _gen_field_struct(arguments.field_args)
    ValueStruct = cache.get_argument_struct(arguments.value_args)

    integrand_func = IntegrandTransformer.apply(
        integrand, arguments.field_args, sample_type=domain.geometry.sample_type
    )

    kernel_fn = _make_side_qfunction_eval_kernel_fn(integrand_func, domain, quadrature, FieldStruct, ValueStruct)

    kernel, _field_struct, _value_struct = cache.get_integrand_kernel(
        integrand=integrand,
        kernel_fn=kernel_fn,
        suffix=kernel_suffix,
        code_transformers=[
            PassFieldArgsToIntegrand(
                arg_names=integrand.argspec.args, parsed_args=arguments, integrand_func=integrand_func
            )
        ],
        FieldStruct=FieldStruct,
        ValueStruct=ValueStruct,
    )

    return kernel, FieldStruct(), ValueStruct()


def extract_side_qfunction(
    integrand: Integrand,
    fields: dict[str, FieldLike],
    quadrature: Quadrature,
    values: dict[str, Any] | None = None,
    fold_geometry: bool = True,
    device=None,
) -> tuple[wp.array, wp.array, wp.array, wp.array]:
    """Evaluate the side trace-channel Q-function coefficients of an integrand.

    The test field found in ``fields`` is replaced by a
    :class:`warp._src.fem.field.SideSeedField` and the integrand is evaluated
    ``2 * (d + 1)`` times per side quadrature point with unit seeds,
    recovering the coefficients such that ``integrand == f0_inner * inner(v)
    + dot(f1_inner, grad(v)) + f0_outer * outer(v) + dot(f1_outer,
    grad_outer(v))`` at every quadrature point (exact, since the form is
    linear in the test function; ``jump``/``average``/``grad_jump``/
    ``grad_average`` are linear combinations of those channels). The gradient
    seeds are *physical* unit vectors, so ``f1_inner``/``f1_outer`` multiply
    the physical test gradients.

    Args:
        integrand: Form to extract from, decorated with :func:`warp.fem.integrand`.
            Must be linear in the (single) test field argument and must not take
            a trial field argument.
        fields: Field arguments to the integrand, keyed by parameter name. Must
            contain exactly one test field over a side domain.
        quadrature: Quadrature formula defining the evaluation points; its
            (side) domain is used as the integration domain.
        values: Additional value arguments to the integrand, keyed by parameter name.
        fold_geometry: If ``True``, multiply the coefficients by the quadrature
            weight times the side measure, i.e. the factor the standard side
            integration kernels apply per sample.
        device: Warp device on which to evaluate.

    Returns:
        A 4-tuple of arrays of shape ``(side_count, points_per_side)``:
        ``(f0_inner, f1_inner, f0_outer, f1_outer)``, the value coefficients
        with the geometry scalar type and the gradient coefficients with the
        matching ``d``-vector type.

    Raises:
        NotImplementedError: If the quadrature domain is not a side domain, or
            if the integrand accesses the test field through an operator other
            than ``inner``/``outer``/``grad``/``grad_outer``/``degree``.
    """

    domain = quadrature.domain
    if domain.element_kind != ElementKind.SIDE:
        raise NotImplementedError(
            "Side Q-function extraction is only implemented for side domains; use extract_qfunction for cell domains"
        )

    arguments = _parse_integrand_arguments(integrand, fields)
    _check_field_compat(integrand, arguments, domain)
    if arguments.test_name is None:
        raise ValueError("Q-function extraction requires a test field argument")
    if arguments.trial_name is not None:
        raise NotImplementedError(
            "Q-function extraction with a trial field is not supported yet; "
            "pass the trial argument as a DiscreteField instead"
        )

    test = arguments.field_args[arguments.test_name]
    if test.domain != domain:
        raise ValueError("Test field and quadrature must be defined over the same domain")

    field_args = dict(arguments.field_args)
    field_args[arguments.test_name] = SideSeedField.from_field(test)
    if arguments.domain_name is not None:
        field_args[arguments.domain_name] = domain
    arguments = arguments._replace(field_args=field_args)

    _find_integrand_operators(integrand, field_args)

    test_operators = integrand.operators.get(arguments.test_name, set())
    unsupported = test_operators - _SUPPORTED_TEST_OPERATORS
    if unsupported:
        op_names = ", ".join(sorted(op.name for op in unsupported))
        raise NotImplementedError(
            f"Side Q-function extraction does not support operator(s) [{op_names}] on the test field "
            f"'{arguments.test_name}'; only value and gradient evaluation "
            "(inner, outer, grad, grad_outer, degree) can be seeded"
        )

    _notify_operator_usage(integrand, field_args)

    kernel, field_struct, value_struct = _get_side_qfunction_kernel(integrand, domain, quadrature, arguments)

    device = wp.get_device(device)

    for name, field in field_args.items():
        if isinstance(field, FieldLike):
            field.fill_eval_arg(getattr(field_struct, name), device=device)
    cache.populate_argument_struct(value_struct, values, func_name=integrand.name)

    element_count = domain.element_count()
    points_per_element = quadrature.max_points_per_element()
    scalar_type = domain.geometry.scalar_type
    grad_vec_type = cache.cached_vec_type(length=domain.geometry.dimension, dtype=scalar_type)

    f0_inner = wp.zeros((element_count, points_per_element), dtype=scalar_type, device=device)
    f1_inner = wp.zeros((element_count, points_per_element), dtype=grad_vec_type, device=device)
    f0_outer = wp.zeros((element_count, points_per_element), dtype=scalar_type, device=device)
    f1_outer = wp.zeros((element_count, points_per_element), dtype=grad_vec_type, device=device)

    wp.launch(
        kernel,
        dim=(element_count, points_per_element),
        inputs=[
            quadrature.arg_value(device),
            domain.element_arg_value(device),
            domain.element_index_arg_value(device),
            field_struct,
            value_struct,
            1 if fold_geometry else 0,
            f0_inner,
            f1_inner,
            f0_outer,
            f1_outer,
        ],
        device=device,
    )

    return f0_inner, f1_inner, f0_outer, f1_outer


def reference_geometry_factors(
    domain: GeometryDomain,
    quadrature: Quadrature,
    device=None,
) -> tuple[wp.array, wp.array]:
    """Evaluate the geometry factors ``(J^{-T}, |det J|)`` at every quadrature point.

    ``J`` is the deformation gradient of the element mapping at the sample
    point. Affine geometries (``Grid2D``/``Grid3D``) produce element-constant
    factors, but the evaluation is per quadrature point so that curved
    elements can reuse the same interface later.

    Args:
        domain: Cell domain over which to evaluate; the geometry's cell
            dimension must match its embedding dimension.
        quadrature: Quadrature formula defining the evaluation points.
        device: Warp device on which to evaluate.

    Returns:
        A pair of arrays of shape ``(element_count, points_per_element)``:
        ``J^{-T}`` with a ``(d, d)`` matrix type and ``|det J|`` with the
        geometry scalar type.
    """

    if domain.element_kind != ElementKind.CELL:
        raise NotImplementedError("Geometry factors are only implemented for cell domains")

    dim = domain.geometry.dimension
    if domain.geometry.cell_dimension != dim:
        raise NotImplementedError("Geometry factors require square element Jacobians (cells of embedding dimension)")

    scalar_type = domain.geometry.scalar_type
    mat_type = cache.cached_mat_type(shape=(dim, dim), dtype=scalar_type)

    @cache.dynamic_kernel(suffix=(domain.name, quadrature.name), kernel_options={"enable_backward": False})
    def geometry_factors_kernel(
        qp_arg: quadrature.Arg,
        domain_arg: domain.ElementArg,
        domain_index_arg: domain.ElementIndexArg,
        jac_inv_t: wp.array2d(dtype=mat_type),
        det_abs: wp.array2d(dtype=scalar_type),
    ):
        domain_element_index, qp = wp.tid()

        element_index = domain.element_index(domain_index_arg, domain_element_index)
        qp_point_count = quadrature.point_count(domain_arg, qp_arg, domain_element_index, element_index)
        if qp >= qp_point_count:
            return

        qp_coords = quadrature.point_coords(domain_arg, qp_arg, domain_element_index, element_index, qp)
        sample = make_free_sample(element_index, qp_coords)

        jac = domain.element_deformation_gradient(domain_arg, sample)
        jac_inv_t[domain_element_index, qp] = wp.transpose(wp.inverse(jac))
        det_abs[domain_element_index, qp] = wp.abs(wp.determinant(jac))

    device = wp.get_device(device)

    element_count = domain.element_count()
    points_per_element = quadrature.max_points_per_element()

    jac_inv_t = wp.zeros((element_count, points_per_element), dtype=mat_type, device=device)
    det_abs = wp.zeros((element_count, points_per_element), dtype=scalar_type, device=device)

    wp.launch(
        geometry_factors_kernel,
        dim=(element_count, points_per_element),
        inputs=[
            quadrature.arg_value(device),
            domain.element_arg_value(device),
            domain.element_index_arg_value(device),
            jac_inv_t,
            det_abs,
        ],
        device=device,
    )

    return jac_inv_t, det_abs
