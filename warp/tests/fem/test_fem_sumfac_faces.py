# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Tests for DG face handling with the explicit sum-factorization opt-in (Phases 5 and 6).

Phase 6 stage 2 brings linear side (DG face) forms onto the sum-factorized
path: ``assembly="sumfac"`` over ``fem.Sides``/``fem.BoundarySides`` (or side
subdomains) of ``Grid2D``/``Grid3D`` runs the fused gather-formulation side
kernel, so a fully matrix-free SIPG operator (sumfac volume apply + sumfac
side apply) works end to end. Bilinear side forms stay on the default path
and raise a descriptive error.

These tests verify, against the naive default-path ``integrate()`` oracle
(the same discrete operator, ``rtol=1e-9``): the sum-factorized side apply
for SIP, upwind-advection, and asymmetric trace-channel forms on all side
domain kinds in 2D and 3D; the side Q-function seed machinery (linearity
identity); the sum-factorized gradient-trace conventions against the native
``fem.grad``/``fem.grad_outer`` side evaluations (closing the stage-1 review
gap); the fully matrix-free SIPG composition; descriptive errors for
unqualified side forms; and the Phase 5 mixed volume-sumfac/side-default
assembly.

All GPU-compiling test shapes stay on discontinuous P=4 spaces with
``RegularQuadrature`` order ``2 * degree`` (n = q = 5 per axis) and
``wp.float64`` values, matching the shared P=4 sum-factorization coverage of
the apply and assembly tests.
"""

import contextlib
import unittest

import numpy as np

import warp as wp
import warp._src.fem.integrate as fem_integrate
import warp.fem as fem
import warp.sparse as sparse
from warp._src.fem.linalg import array_axpy
from warp._src.fem.polynomial import quadrature_1d
from warp._src.fem.sumfac.face_trace import (
    face_axes,
    face_trace_normal_gradient,
    face_trace_value,
    grid_side_face_ends,
    grid_side_to_face_coords,
)
from warp._src.fem.sumfac.operators_1d import build_derivative_matrix, build_interpolation_matrix
from warp._src.fem.sumfac.qfunction import extract_side_qfunction
from warp.optim.linear import LinearOperator, cg
from warp.tests.unittest_utils import *


@fem.integrand
def volume_stiffness_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    return wp.dot(fem.grad(u, s), fem.grad(v, s))


@fem.integrand
def implicit_volume_form(s: fem.Sample, u: fem.Field, v: fem.Field, viscosity: wp.float64):
    return u(s) * v(s) + viscosity * wp.dot(fem.grad(u, s), fem.grad(v, s))


@fem.integrand
def side_flux_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    return fem.jump(u, s) * fem.jump(v, s) + wp.float64(0.25) * fem.average(u, s) * fem.jump(v, s)


@fem.integrand
def side_penalty_form(s: fem.Sample, domain: fem.Domain, u: fem.Field, v: fem.Field, penalty_scale: wp.float64):
    penalty = penalty_scale * fem.measure_ratio(domain, s) * wp.float64(fem.degree(u) * fem.degree(v))
    return penalty * fem.jump(u, s) * fem.jump(v, s)


@fem.integrand
def manufactured_rhs_form(s: fem.Sample, domain: fem.Domain, v: fem.Field):
    pos = domain(s)
    pi = wp.float64(3.141592653589793)
    source = wp.float64(1.0) + wp.float64(0.25) * wp.sin(wp.float64(2.0) * pi * pos[0]) * wp.cos(pi * pos[1])
    return source * v(s)


@fem.integrand
def sip_side_form(s: fem.Sample, domain: fem.Domain, u: fem.Field, v: fem.Field, penalty_scale: wp.float64):
    # Standard symmetric interior penalty: penalty + consistency + symmetry
    nor = fem.normal(domain, s)
    penalty = penalty_scale * fem.measure_ratio(domain, s) * wp.float64(fem.degree(u) * fem.degree(u))
    return (
        penalty * fem.jump(u, s) * fem.jump(v, s)
        - wp.dot(fem.grad_average(u, s), nor) * fem.jump(v, s)
        - wp.dot(fem.grad_average(v, s), nor) * fem.jump(u, s)
    )


@fem.integrand
def upwind_form_2d(s: fem.Sample, domain: fem.Domain, u: fem.Field, v: fem.Field, vel: wp.vec2d):
    nor = fem.normal(domain, s)
    vn = wp.dot(vel, nor)
    flux = fem.average(u, s) * vn + wp.float64(0.5) * wp.abs(vn) * fem.jump(u, s)
    return flux * fem.jump(v, s)


@fem.integrand
def upwind_form_3d(s: fem.Sample, domain: fem.Domain, u: fem.Field, v: fem.Field, vel: wp.vec3d):
    nor = fem.normal(domain, s)
    vn = wp.dot(vel, nor)
    flux = fem.average(u, s) * vn + wp.float64(0.5) * wp.abs(vn) * fem.jump(u, s)
    return flux * fem.jump(v, s)


@fem.integrand
def asym_trace_form(s: fem.Sample, domain: fem.Domain, u: fem.Field, v: fem.Field):
    # Exercises every inner/outer trace channel pairing asymmetrically and is
    # non-degenerate on boundary sides (where jump-only forms vanish).
    nor = fem.normal(domain, s)
    return (
        wp.float64(2.5) * fem.inner(u, s) * fem.inner(v, s)
        + fem.average(u, s) * fem.jump(v, s)
        - wp.dot(fem.grad(u, s), nor) * fem.outer(v, s)
        + wp.float64(0.5) * fem.jump(u, s) * wp.dot(fem.grad_average(v, s), nor)
        + wp.float64(0.25) * fem.outer(u, s) * wp.dot(fem.grad_jump(v, s), nor)
    )


@fem.integrand
def tangential_grad_form_2d(s: fem.Sample, u: fem.Field, v: fem.Field, tvec: wp.vec2d):
    # Tangential-gradient oracle: every other committed form dots gradients
    # with the (axis-aligned) grid side normal, so without this form the
    # tangential trace/lift code paths carry exactly zero.
    return (
        wp.dot(fem.grad_average(u, s), tvec) * fem.jump(v, s)
        + fem.jump(u, s) * wp.dot(fem.grad_jump(v, s), tvec)
        + wp.float64(0.5) * wp.dot(fem.grad_jump(u, s), tvec) * wp.dot(fem.grad_average(v, s), tvec)
    )


@fem.integrand
def tangential_grad_form_3d(s: fem.Sample, u: fem.Field, v: fem.Field, tvec: wp.vec3d):
    return (
        wp.dot(fem.grad_average(u, s), tvec) * fem.jump(v, s)
        + fem.jump(u, s) * wp.dot(fem.grad_jump(v, s), tvec)
        + wp.float64(0.5) * wp.dot(fem.grad_jump(u, s), tvec) * wp.dot(fem.grad_average(v, s), tvec)
    )


@fem.integrand
def sip_position_coef_form_2d(
    s: fem.Sample, domain: fem.Domain, u: fem.Field, v: fem.Field, penalty_scale: wp.float64, tvec: wp.vec2d
):
    # Position-dependent coefficient oracle: the symmetric Gauss rule plus
    # position-independent integrands make any CONSISTENT longitude-flip
    # transcription error an exact quadrature-point relabeling, invisible to
    # the constant-coefficient forms; the asymmetric coefficient breaks that
    # symmetry on the value, normal-gradient, and tangential-gradient channels.
    pos = fem.position(domain, s)
    nor = fem.normal(domain, s)
    coef = wp.float64(1.0) + wp.float64(0.5) * wp.sin(wp.float64(2.3) * pos[0] + wp.float64(1.1) * pos[1])
    penalty = penalty_scale * fem.measure_ratio(domain, s)
    return coef * (
        penalty * fem.jump(u, s) * fem.jump(v, s)
        - wp.dot(fem.grad_average(u, s), nor) * fem.jump(v, s)
        - wp.dot(fem.grad_average(v, s), nor) * fem.jump(u, s)
        + wp.dot(fem.grad_jump(u, s), tvec) * fem.jump(v, s)
        + fem.average(u, s) * wp.dot(fem.grad_average(v, s), tvec)
    )


@fem.integrand
def sip_position_coef_form_3d(
    s: fem.Sample, domain: fem.Domain, u: fem.Field, v: fem.Field, penalty_scale: wp.float64, tvec: wp.vec3d
):
    pos = fem.position(domain, s)
    nor = fem.normal(domain, s)
    coef = wp.float64(1.0) + wp.float64(0.5) * wp.sin(
        wp.float64(2.3) * pos[0] + wp.float64(1.1) * pos[1] + wp.float64(1.7) * pos[2]
    )
    penalty = penalty_scale * fem.measure_ratio(domain, s)
    return coef * (
        penalty * fem.jump(u, s) * fem.jump(v, s)
        - wp.dot(fem.grad_average(u, s), nor) * fem.jump(v, s)
        - wp.dot(fem.grad_average(v, s), nor) * fem.jump(u, s)
        + wp.dot(fem.grad_jump(u, s), tvec) * fem.jump(v, s)
        + fem.average(u, s) * wp.dot(fem.grad_average(v, s), tvec)
    )


@fem.integrand
def side_at_node_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    # Linear in v, but reads the test field through a node-based operator
    return u(fem.at_node(v, s)) * fem.inner(v, s)


@fem.integrand
def _interior_side_probe(s: fem.Sample, domain: fem.Domain, mask: wp.array(dtype=int)):
    s_in = fem.to_inner_cell(domain, s)
    s_out = fem.to_outer_cell(domain, s)
    if s_in.element_index != s_out.element_index:
        mask[s.qp_index] = 1


def _make_grid_2d():
    return fem.Grid2D(
        res=wp.vec2i(3, 2),
        bounds_lo=wp.vec2d(0.0, 0.0),
        bounds_hi=wp.vec2d(1.0, 1.5),
        scalar_type=wp.float64,
    )


def _make_grid_3d():
    return fem.Grid3D(
        res=wp.vec3i(3, 2, 2),
        bounds_lo=wp.vec3d(0.0, 0.0, 0.0),
        bounds_hi=wp.vec3d(1.0, 1.5, 0.5),
        scalar_type=wp.float64,
    )


def _make_dg_case():
    """Build the P=4 discontinuous ``Grid2D`` case shared by the face tests."""

    degree = 4
    geo = _make_grid_2d()
    space = fem.make_polynomial_space(geo, degree=degree, discontinuous=True, dtype=wp.float64)

    cell_domain = fem.Cells(geometry=geo)
    cell_test = fem.make_test(space=space, domain=cell_domain)
    cell_trial = fem.make_trial(space=space, domain=cell_domain)
    cell_quadrature = fem.RegularQuadrature(cell_domain, order=2 * degree)

    side_domain = fem.Sides(geometry=geo)
    side_test = fem.make_test(space=space, domain=side_domain)
    side_trial = fem.make_trial(space=space, domain=side_domain)
    side_quadrature = fem.RegularQuadrature(side_domain, order=2 * degree)

    return space, cell_test, cell_trial, cell_quadrature, side_test, side_trial, side_quadrature


def _interior_sides(geo):
    """Build a ``Subdomain`` of the interior sides of ``geo`` (the InteriorSides analog)."""
    all_sides = fem.Sides(geometry=geo)
    probe_quad = fem.RegularQuadrature(all_sides, order=0)
    mask = wp.zeros(all_sides.element_count(), dtype=int)
    fem.interpolate(_interior_side_probe, at=probe_quad, values={"mask": mask})
    return fem.Subdomain(all_sides, element_mask=mask)


@contextlib.contextmanager
def _capture_integrate_kernels():
    """Capture the kernels selected by ``fem.integrate`` for dispatch assertions."""

    captured = []
    original = fem_integrate._launch_integrate_kernel

    def spy(*args, **kwargs):
        captured.append(kwargs["kernel"])
        return original(*args, **kwargs)

    fem_integrate._launch_integrate_kernel = spy
    try:
        yield captured
    finally:
        fem_integrate._launch_integrate_kernel = original


def _is_sumfac_kernel(kernel):
    return getattr(kernel, "_wp_fem_sumfac_", False)


def _bsr_to_dense(matrix):
    """Convert a BSR matrix to a dense NumPy array."""

    nnz = matrix.nnz_sync()
    offsets = matrix.offsets.numpy()
    columns = matrix.columns.numpy()[:nnz]
    brows, bcols = matrix.block_shape
    values = matrix.values.numpy()[:nnz].reshape(nnz, brows, bcols)
    dense = np.zeros(matrix.shape)
    for row in range(matrix.nrow):
        for b in range(offsets[row], offsets[row + 1]):
            col = columns[b]
            dense[row * brows : (row + 1) * brows, col * bcols : (col + 1) * bcols] += values[b]
    return dense


def _integrate_mixed_matrix(volume_assembly):
    """Assemble one BSR matrix from a volume term (sumfac or default) plus a default-path side term."""

    _space, cell_test, cell_trial, cell_quadrature, side_test, side_trial, side_quadrature = _make_dg_case()
    with _capture_integrate_kernels() as kernels:
        matrix = fem.integrate(
            volume_stiffness_form,
            fields={"u": cell_trial, "v": cell_test},
            quadrature=cell_quadrature,
            output_dtype=wp.float64,
            assembly=volume_assembly,
        )
        fem.integrate(
            side_penalty_form,
            fields={"u": side_trial, "v": side_test},
            values={"penalty_scale": wp.float64(1.0)},
            quadrature=side_quadrature,
            output=matrix,
            add=True,
            output_dtype=wp.float64,
        )
    return matrix, kernels


def _assemble_dg_system(volume_assembly):
    """Assemble a compact P=4 SIP-style diffusion system and manufactured RHS.

    Only the volume term may use ``assembly="sumfac"``; the side penalty is a
    bilinear side form (outside the sum-factorized path) and the RHS form has
    no injectable input field, so both always use the default assembly.
    """

    _space, cell_test, cell_trial, cell_quadrature, side_test, side_trial, side_quadrature = _make_dg_case()
    matrix = fem.integrate(
        implicit_volume_form,
        fields={"u": cell_trial, "v": cell_test},
        values={"viscosity": wp.float64(0.05)},
        quadrature=cell_quadrature,
        output_dtype=wp.float64,
        assembly=volume_assembly,
    )
    fem.integrate(
        side_penalty_form,
        fields={"u": side_trial, "v": side_test},
        values={"penalty_scale": wp.float64(0.05)},
        quadrature=side_quadrature,
        output=matrix,
        add=True,
        output_dtype=wp.float64,
    )
    rhs = fem.integrate(
        manufactured_rhs_form,
        fields={"v": cell_test},
        quadrature=cell_quadrature,
        output_dtype=wp.float64,
    )
    return matrix, rhs


def test_mixed_volume_sumfac_plus_side_default(test, device):
    """Volume sumfac + default side terms accumulate into one BSR identical to the all-default matrix."""

    with wp.ScopedDevice(device):
        mixed_sumfac, sumfac_kernels = _integrate_mixed_matrix("sumfac")
        mixed_default, default_kernels = _integrate_mixed_matrix(None)

        test.assertEqual(len(sumfac_kernels), 2)
        test.assertTrue(_is_sumfac_kernel(sumfac_kernels[0]), "volume stiffness should use the sum-factorized kernel")
        test.assertFalse(_is_sumfac_kernel(sumfac_kernels[1]), "side penalty must use the default path")
        test.assertFalse(_is_sumfac_kernel(default_kernels[0]))
        test.assertFalse(_is_sumfac_kernel(default_kernels[1]))
        np.testing.assert_allclose(_bsr_to_dense(mixed_sumfac), _bsr_to_dense(mixed_default), rtol=1e-9, atol=1e-10)


def test_end_to_end_dg_p4(test, device):
    with wp.ScopedDevice(device):
        matrix_sumfac, rhs_sumfac = _assemble_dg_system("sumfac")
        matrix_default, rhs_default = _assemble_dg_system(None)

        np.testing.assert_allclose(rhs_sumfac.numpy(), rhs_default.numpy(), rtol=1e-9, atol=1e-10)

        x_sumfac = wp.zeros_like(rhs_sumfac)
        x_default = wp.zeros_like(rhs_default)
        cg(matrix_sumfac, rhs_sumfac, x_sumfac, tol=1e-12, maxiter=500, use_cuda_graph=False)
        cg(matrix_default, rhs_default, x_default, tol=1e-12, maxiter=500, use_cuda_graph=False)

        np.testing.assert_allclose(x_sumfac.numpy(), x_default.numpy(), rtol=1e-6, atol=1e-9)

        probe = wp.array(np.linspace(-0.5, 0.5, rhs_sumfac.shape[0]), dtype=wp.float64)
        y_sumfac = sparse.bsr_mv(matrix_sumfac, probe)
        y_default = sparse.bsr_mv(matrix_default, probe)
        np.testing.assert_allclose(y_sumfac.numpy(), y_default.numpy(), rtol=1e-9, atol=1e-10)


# -- Phase 6 stage 2: sum-factorized side apply ----------------------------------


def _check_side_apply_matches_naive(test, geo, domain, form, values, rng):
    """Sumfac side apply == naive default-path integrate over the same domain and quadrature."""
    degree = 4
    space = fem.make_polynomial_space(geo, degree=degree, discontinuous=True, dtype=wp.float64)
    side_test = fem.make_test(space=space, domain=domain)
    quadrature = fem.RegularQuadrature(domain, order=2 * degree)

    u = space.make_field()
    u.dof_values = wp.array(rng.uniform(-1.0, 1.0, size=space.node_count()), dtype=wp.float64)

    kwargs = {
        "fields": {"u": u.trace(), "v": side_test},
        "quadrature": quadrature,
        "values": values,
        "output_dtype": wp.float64,
    }
    y_naive = fem.integrate(form, **kwargs)
    with _capture_integrate_kernels() as kernels:
        y_sumfac = fem.integrate(form, assembly="sumfac", **kwargs)
    test.assertTrue(_is_sumfac_kernel(kernels[0]), "side apply should use the sum-factorized side kernel")

    a = y_naive.numpy()
    scale = max(np.abs(a).max(), 1.0)
    np.testing.assert_allclose(y_sumfac.numpy(), a, rtol=1e-9, atol=1e-12 * scale)


def test_side_apply_equals_naive_2d(test, device):
    rng = np.random.default_rng(61)
    with wp.ScopedDevice(device):
        geo = _make_grid_2d()
        sides = fem.Sides(geometry=geo)
        boundary = fem.BoundarySides(geometry=geo)
        interior = _interior_sides(geo)
        sip_values = {"penalty_scale": wp.float64(8.0)}
        upwind_values = {"vel": wp.vec2d(0.7, -0.3)}
        tang_values = {"tvec": wp.vec2d(0.6, -1.3)}
        pos_values = {"penalty_scale": wp.float64(4.0), "tvec": wp.vec2d(0.7, 0.4)}
        for domain in (sides, boundary, interior):
            _check_side_apply_matches_naive(test, geo, domain, sip_side_form, sip_values, rng)
            _check_side_apply_matches_naive(test, geo, domain, upwind_form_2d, upwind_values, rng)
            _check_side_apply_matches_naive(test, geo, domain, tangential_grad_form_2d, tang_values, rng)
            _check_side_apply_matches_naive(test, geo, domain, sip_position_coef_form_2d, pos_values, rng)
        # Asymmetric form: exercises every channel pairing; non-degenerate on boundaries
        _check_side_apply_matches_naive(test, geo, sides, asym_trace_form, {}, rng)
        _check_side_apply_matches_naive(test, geo, boundary, asym_trace_form, {}, rng)


def test_side_apply_equals_naive_3d(test, device):
    rng = np.random.default_rng(62)
    with wp.ScopedDevice(device):
        geo = _make_grid_3d()
        sides = fem.Sides(geometry=geo)
        boundary = fem.BoundarySides(geometry=geo)
        interior = _interior_sides(geo)
        sip_values = {"penalty_scale": wp.float64(8.0)}
        upwind_values = {"vel": wp.vec3d(0.7, -0.3, 0.4)}
        tang_values = {"tvec": wp.vec3d(0.6, -1.3, 0.8)}
        pos_values = {"penalty_scale": wp.float64(4.0), "tvec": wp.vec3d(0.7, 0.4, -0.9)}
        for domain in (sides, boundary, interior):
            _check_side_apply_matches_naive(test, geo, domain, sip_side_form, sip_values, rng)
            _check_side_apply_matches_naive(test, geo, domain, upwind_form_3d, upwind_values, rng)
            _check_side_apply_matches_naive(test, geo, domain, tangential_grad_form_3d, tang_values, rng)
            _check_side_apply_matches_naive(test, geo, domain, sip_position_coef_form_3d, pos_values, rng)
        _check_side_apply_matches_naive(test, geo, sides, asym_trace_form, {}, rng)
        _check_side_apply_matches_naive(test, geo, boundary, asym_trace_form, {}, rng)


@fem.integrand
def _trace_value_probe_2d(
    s: fem.Sample,
    w: fem.Field,
    w_in: wp.array(dtype=wp.float64),
    w_out: wp.array(dtype=wp.float64),
    gw_in: wp.array(dtype=wp.vec2d),
    gw_out: wp.array(dtype=wp.vec2d),
):
    qp = s.qp_index
    w_in[qp] = fem.inner(w, s)
    w_out[qp] = fem.outer(w, s)
    gw_in[qp] = fem.grad(w, s)
    gw_out[qp] = fem.grad_outer(w, s)


def test_side_qfunction_linearity(test, device):
    """Side Q-function extraction satisfies the linearity identity against the integrate() oracle.

    For any discrete field ``w`` of the test space, contracting the extracted
    trace-channel coefficients with the native traces of ``w`` at the side
    quadrature points must reproduce ``dot(integrate(form, v), w_dofs)``.
    """
    rng = np.random.default_rng(63)
    with wp.ScopedDevice(device):
        degree = 4
        geo = _make_grid_2d()
        space = fem.make_polynomial_space(geo, degree=degree, discontinuous=True, dtype=wp.float64)
        sides = fem.Sides(geometry=geo)
        side_test = fem.make_test(space=space, domain=sides)
        quadrature = fem.RegularQuadrature(sides, order=2 * degree)

        u = space.make_field()
        u.dof_values = wp.array(rng.uniform(-1.0, 1.0, size=space.node_count()), dtype=wp.float64)
        w = space.make_field()
        w_dofs = rng.uniform(-1.0, 1.0, size=space.node_count())
        w.dof_values = wp.array(w_dofs, dtype=wp.float64)

        values = {"penalty_scale": wp.float64(2.5)}
        f0_in, f1_in, f0_out, f1_out = extract_side_qfunction(
            sip_side_form,
            fields={"u": u.trace(), "v": side_test},
            quadrature=quadrature,
            values=values,
        )

        total = quadrature.total_point_count()
        arrays = {
            "w_in": wp.zeros(total, dtype=wp.float64),
            "w_out": wp.zeros(total, dtype=wp.float64),
            "gw_in": wp.zeros(total, dtype=wp.vec2d),
            "gw_out": wp.zeros(total, dtype=wp.vec2d),
        }
        fem.interpolate(_trace_value_probe_2d, at=quadrature, fields={"w": w.trace()}, values=arrays)

        nqp = quadrature.max_points_per_element()
        ns = sides.element_count()
        lhs = (
            (f0_in.numpy() * arrays["w_in"].numpy().reshape(ns, nqp)).sum()
            + (f1_in.numpy() * arrays["gw_in"].numpy().reshape(ns, nqp, 2)).sum()
            + (f0_out.numpy() * arrays["w_out"].numpy().reshape(ns, nqp)).sum()
            + (f1_out.numpy() * arrays["gw_out"].numpy().reshape(ns, nqp, 2)).sum()
        )

        result = fem.integrate(
            sip_side_form,
            fields={"u": u.trace(), "v": side_test},
            quadrature=quadrature,
            values=values,
            output_dtype=wp.float64,
        )
        rhs = float(np.dot(result.numpy(), w_dofs))
        np.testing.assert_allclose(lhs, rhs, rtol=1e-9)


def test_fully_matrix_free_sipg(test, device):
    """Sumfac volume apply + sumfac side apply == assembled SIPG matvec; CG solves match.

    The volume term includes a mass contribution so that the composed system
    is symmetric positive definite (the SIP interior-penalty terms vanish on
    boundary sides under the native trace semantics, so the pure SIPG
    Laplacian would retain the constant null space).
    """
    rng = np.random.default_rng(64)
    with wp.ScopedDevice(device):
        degree = 4
        geo = _make_grid_2d()
        space = fem.make_polynomial_space(geo, degree=degree, discontinuous=True, dtype=wp.float64)
        cells = fem.Cells(geometry=geo)
        sides = fem.Sides(geometry=geo)

        cell_test = fem.make_test(space=space, domain=cells)
        cell_trial = fem.make_trial(space=space, domain=cells)
        cell_quadrature = fem.RegularQuadrature(cells, order=2 * degree)
        side_test = fem.make_test(space=space, domain=sides)
        side_trial = fem.make_trial(space=space, domain=sides)
        side_quadrature = fem.RegularQuadrature(sides, order=2 * degree)

        sip_values = {"penalty_scale": wp.float64(8.0)}
        volume_values = {"viscosity": wp.float64(1.0)}

        # Assembled reference operator: volume mass + stiffness plus SIP side matrix
        matrix = fem.integrate(
            implicit_volume_form,
            fields={"u": cell_trial, "v": cell_test},
            values=volume_values,
            quadrature=cell_quadrature,
            output_dtype=wp.float64,
        )
        fem.integrate(
            sip_side_form,
            fields={"u": side_trial, "v": side_test},
            values=sip_values,
            quadrature=side_quadrature,
            output=matrix,
            add=True,
            output_dtype=wp.float64,
        )

        n_dofs = space.node_count()
        u_field = space.make_field()
        u_trace = u_field.trace()
        result = wp.empty(n_dofs, dtype=wp.float64)

        def apply_sipg(x: wp.array, output: wp.array):
            u_field.dof_values = x
            fem.integrate(
                implicit_volume_form,
                fields={"u": u_field, "v": cell_test},
                values=volume_values,
                quadrature=cell_quadrature,
                output=output,
                assembly="sumfac",
            )
            fem.integrate(
                sip_side_form,
                fields={"u": u_trace, "v": side_test},
                values=sip_values,
                quadrature=side_quadrature,
                output=output,
                add=True,
                assembly="sumfac",
            )

        # Matrix-free apply == assembled matvec
        x = wp.array(rng.uniform(-1.0, 1.0, size=n_dofs), dtype=wp.float64)
        apply_sipg(x, result)
        y_ref = sparse.bsr_mv(matrix, x).numpy()
        scale = max(np.abs(y_ref).max(), 1.0)
        np.testing.assert_allclose(result.numpy(), y_ref, rtol=1e-9, atol=1e-12 * scale)

        # CG solve of the manufactured problem: fully matrix-free == assembled
        rhs = fem.integrate(
            manufactured_rhs_form,
            fields={"v": cell_test},
            quadrature=cell_quadrature,
            output_dtype=wp.float64,
        )

        def matvec(x, y, z, alpha, beta):
            apply_sipg(x, result)
            if z.ptr != y.ptr:
                wp.copy(z, y)
            array_axpy(x=result, y=z, alpha=alpha, beta=beta)

        operator = LinearOperator(shape=(n_dofs, n_dofs), dtype=wp.float64, device=wp.get_device(), matvec=matvec)

        x_mf = wp.zeros_like(rhs)
        x_assembled = wp.zeros_like(rhs)
        cg(operator, rhs, x_mf, tol=1e-12, maxiter=2000, use_cuda_graph=False)
        cg(matrix, rhs, x_assembled, tol=1e-12, maxiter=2000, use_cuda_graph=False)

        x_scale = max(np.abs(x_assembled.numpy()).max(), 1.0)
        np.testing.assert_allclose(x_mf.numpy(), x_assembled.numpy(), rtol=1e-6, atol=1e-8 * x_scale)


devices = get_test_devices()


@fem.integrand
def _grad_trace_probe_2d(
    s: fem.Sample,
    domain: fem.Domain,
    u: fem.Field,
    pos: wp.array(dtype=wp.vec2d),
    nrm: wp.array(dtype=wp.vec2d),
    gin: wp.array(dtype=wp.vec2d),
    gout: wp.array(dtype=wp.vec2d),
    icell: wp.array(dtype=int),
    ocell: wp.array(dtype=int),
    scoord: wp.array(dtype=wp.vec3d),
):
    qp = s.qp_index
    pos[qp] = domain(s)
    nrm[qp] = fem.normal(domain, s)
    gin[qp] = fem.grad(u, s)
    gout[qp] = fem.grad_outer(u, s)
    icell[qp] = fem.to_inner_cell(domain, s).element_index
    ocell[qp] = fem.to_outer_cell(domain, s).element_index
    scoord[qp] = s.element_coords


@fem.integrand
def _grad_trace_probe_3d(
    s: fem.Sample,
    domain: fem.Domain,
    u: fem.Field,
    pos: wp.array(dtype=wp.vec3d),
    nrm: wp.array(dtype=wp.vec3d),
    gin: wp.array(dtype=wp.vec3d),
    gout: wp.array(dtype=wp.vec3d),
    icell: wp.array(dtype=int),
    ocell: wp.array(dtype=int),
    scoord: wp.array(dtype=wp.vec3d),
):
    qp = s.qp_index
    pos[qp] = domain(s)
    nrm[qp] = fem.normal(domain, s)
    gin[qp] = fem.grad(u, s)
    gout[qp] = fem.grad_outer(u, s)
    icell[qp] = fem.to_inner_cell(domain, s).element_index
    ocell[qp] = fem.to_outer_cell(domain, s).element_index
    scoord[qp] = s.element_coords


class TestFemSumfacFaces(unittest.TestCase):
    def test_side_sumfac_raises(self):
        """Side forms under assembly="sumfac": bilinear forms raise; qualifying linear forms do not."""

        _space, _cell_test, _cell_trial, _cell_quadrature, side_test, side_trial, side_quadrature = _make_dg_case()
        # Bilinear side forms stay on the default path
        with self.assertRaisesRegex(NotImplementedError, "bilinear forms over side"):
            fem.integrate(
                side_flux_form,
                fields={"u": side_trial, "v": side_test},
                quadrature=side_quadrature,
                output_dtype=wp.float64,
                assembly="sumfac",
            )

    def test_side_sumfac_raises_unqualified(self):
        """Unqualified side linear forms raise a descriptive error (host-side, no kernel compiles)."""

        rng = np.random.default_rng(65)
        degree = 4
        geo = _make_grid_2d()
        space = fem.make_polynomial_space(geo, degree=degree, discontinuous=True, dtype=wp.float64)
        sides = fem.Sides(geometry=geo)
        side_test = fem.make_test(space=space, domain=sides)
        side_quadrature = fem.RegularQuadrature(sides, order=2 * degree)
        u = space.make_field()
        u.dof_values = wp.array(rng.uniform(-1.0, 1.0, size=space.node_count()), dtype=wp.float64)

        def integrate_sumfac(form, fields, **kwargs):
            kwargs.setdefault("output_dtype", wp.float64)
            kwargs.setdefault("assembly", "sumfac")
            return fem.integrate(form, fields=fields, **kwargs)

        # Unstructured quadmesh side domains have per-side orientations
        pos_grid = np.stack(np.meshgrid(np.linspace(0.0, 1.0, 3), np.linspace(0.0, 1.0, 3), indexing="ij"), axis=-1)
        positions = wp.array(pos_grid.reshape(-1, 2), dtype=wp.vec2d)
        quads = []
        for i in range(2):
            for j in range(2):
                v00 = i * 3 + j
                quads.append([v00, v00 + 3, v00 + 4, v00 + 1])
        quad_vidx = wp.array(np.array(quads, dtype=np.int32), dtype=int)
        quad_geo = fem.Quadmesh2D(quad_vertex_indices=quad_vidx, positions=positions)
        quad_space = fem.make_polynomial_space(quad_geo, degree=2, discontinuous=True, dtype=wp.float64)
        quad_sides = fem.Sides(geometry=quad_geo)
        quad_test = fem.make_test(space=quad_space, domain=quad_sides)
        quad_quadrature = fem.RegularQuadrature(quad_sides, order=4)
        quad_u = quad_space.make_field()
        quad_u.dof_values = wp.array(rng.uniform(-1.0, 1.0, size=quad_space.node_count()), dtype=wp.float64)
        with self.assertRaisesRegex(NotImplementedError, "Grid2D or Grid3D"):
            integrate_sumfac(side_flux_form, {"u": quad_u.trace(), "v": quad_test}, quadrature=quad_quadrature)

        # Partial space partition on the input field
        geo_partition = fem.LinearGeometryPartition(geo, 0, 2)
        partial_partition = fem.make_space_partition(space.topology, geometry_partition=geo_partition)
        u_partial = space.make_field(space_partition=partial_partition)
        with self.assertRaisesRegex(NotImplementedError, "partial space partition"):
            integrate_sumfac(side_flux_form, {"u": u_partial.trace(), "v": side_test}, quadrature=side_quadrature)

        # Node-based operators on the test field cannot be seeded
        with self.assertRaisesRegex(NotImplementedError, "at_node"):
            integrate_sumfac(side_at_node_form, {"u": u.trace(), "v": side_test}, quadrature=side_quadrature)

        # No injectable input trace field (pure right-hand-side form)
        with self.assertRaisesRegex(NotImplementedError, "input trace field"):
            integrate_sumfac(manufactured_rhs_form, {"v": side_test}, quadrature=side_quadrature)

    def test_side_grad_trace_matches_native(self):
        # Closes the stage-1 review gap: the sum-factorized gradient traces
        # (endpoint derivative row along the normal axis, 1D derivative
        # matrices at the mapped side QPs along the tangential axes, J^{-T}
        # mapping) equal fem.grad / fem.grad_outer through the native path,
        # for every side of a Grid2D and a Grid3D. CPU-only by design.
        for dim in (2, 3):
            with self.subTest(dim=dim):
                self._check_side_grad_trace(dim)

    def _check_side_grad_trace(self, dim):
        device = "cpu"
        degree = 3
        n = degree + 1

        if dim == 2:
            res = (3, 2)
            lo, hi = (0.0, 0.0), (1.0, 1.5)
            geo = fem.Grid2D(
                res=wp.vec2i(*res), bounds_lo=wp.vec2d(*lo), bounds_hi=wp.vec2d(*hi), scalar_type=wp.float64
            )
            probe, vec_type = _grad_trace_probe_2d, wp.vec2d
        else:
            res = (3, 2, 4)
            lo, hi = (0.0, 0.0, 0.0), (1.0, 1.5, 0.5)
            geo = fem.Grid3D(
                res=wp.vec3i(*res), bounds_lo=wp.vec3d(*lo), bounds_hi=wp.vec3d(*hi), scalar_type=wp.float64
            )
            probe, vec_type = _grad_trace_probe_3d, wp.vec3d

        space = fem.make_polynomial_space(geo, degree=degree, discontinuous=True, dtype=wp.float64)
        rng = np.random.default_rng(66 + dim)
        dofs_flat = rng.uniform(-1.0, 1.0, size=space.node_count())
        field = space.make_field()
        field.dof_values = wp.array(dofs_flat, dtype=wp.float64, device=device)
        dofs = dofs_flat.reshape(geo.cell_count(), n**dim)

        sides = fem.Sides(geometry=geo)
        quadrature = fem.RegularQuadrature(sides, order=2 * degree)
        total = quadrature.total_point_count()
        nqp = quadrature.max_points_per_element()

        arrays = {
            "pos": wp.zeros(total, dtype=vec_type, device=device),
            "nrm": wp.zeros(total, dtype=vec_type, device=device),
            "gin": wp.zeros(total, dtype=vec_type, device=device),
            "gout": wp.zeros(total, dtype=vec_type, device=device),
            "icell": wp.zeros(total, dtype=int, device=device),
            "ocell": wp.zeros(total, dtype=int, device=device),
            "scoord": wp.zeros(total, dtype=wp.vec3d, device=device),
        }
        fem.interpolate(probe, at=quadrature, fields={"u": field.trace()}, values=arrays, device=device)
        observed = {key: arr.numpy() for key, arr in arrays.items()}

        nodes = np.asarray(quadrature_1d(point_count=n, family=space.basis.shape.family)[0], dtype=np.float64)
        cell_size = np.array([(hi[d] - lo[d]) / res[d] for d in range(dim)])

        def interpolate_face(face_vals, face_coords, deriv_axis=None):
            """Contract face-frame nodal values to side QPs; optionally differentiate along one face axis."""
            mats = []
            for f in range(dim - 1):
                build = build_derivative_matrix if deriv_axis == f else build_interpolation_matrix
                mats.append(build(nodes, face_coords[:, f]))
            if dim == 2:
                return mats[0] @ face_vals
            return np.einsum("ru,rv,uv->r", mats[0], mats[1], face_vals.reshape(n, n))

        for side in range(sides.element_count()):
            qps = slice(side * nqp, (side + 1) * nqp)
            normal = observed["nrm"][qps][0]
            axis = int(np.argmax(np.abs(normal)))
            altitude = int(round((observed["pos"][qps][0][axis] - lo[axis]) / cell_size[axis]))
            inner_cell = int(observed["icell"][qps][0])
            outer_cell = int(observed["ocell"][qps][0])

            inner_end, outer_end = grid_side_face_ends(altitude, res[axis])
            side_qp = observed["scoord"][qps][:, : dim - 1]
            face_coords = grid_side_to_face_coords(dim, axis, altitude, side_qp)
            f_axes = list(face_axes(dim, axis))

            with self.subTest(dim=dim, side=side, axis=axis, altitude=altitude):
                for cell, end, key in ((inner_cell, inner_end, "gin"), (outer_cell, outer_end, "gout")):
                    face_value = face_trace_value(dofs[cell][None, :], n, dim, axis, end)[0]
                    face_normal_grad = face_trace_normal_gradient(dofs[cell][None, :], n, dim, axis, end)[0]
                    # Reference gradient at the side QPs, element axes
                    g_ref = np.zeros((nqp, dim))
                    g_ref[:, axis] = interpolate_face(face_normal_grad, face_coords)
                    for f, element_axis in enumerate(f_axes):
                        g_ref[:, element_axis] = interpolate_face(face_value, face_coords, deriv_axis=f)
                    # Physical gradient (axis-aligned grid: J^{-T} = diag(1 / cell_size))
                    g_phys = g_ref / cell_size[np.newaxis, :]
                    np.testing.assert_allclose(observed[key][qps], g_phys, rtol=0.0, atol=1e-11)


add_function_test(
    TestFemSumfacFaces,
    "test_mixed_volume_sumfac_plus_side_default",
    test_mixed_volume_sumfac_plus_side_default,
    devices=devices,
)
add_function_test(TestFemSumfacFaces, "test_end_to_end_dg_p4", test_end_to_end_dg_p4, devices=devices)
add_function_test(
    TestFemSumfacFaces, "test_side_apply_equals_naive_2d", test_side_apply_equals_naive_2d, devices=devices
)
add_function_test(
    TestFemSumfacFaces, "test_side_apply_equals_naive_3d", test_side_apply_equals_naive_3d, devices=devices
)
add_function_test(TestFemSumfacFaces, "test_side_qfunction_linearity", test_side_qfunction_linearity, devices=devices)
add_function_test(TestFemSumfacFaces, "test_fully_matrix_free_sipg", test_fully_matrix_free_sipg, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
