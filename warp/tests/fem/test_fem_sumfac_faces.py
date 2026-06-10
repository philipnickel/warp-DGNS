# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Tests for DG face fallback with the sum-factorization dispatcher (Phase 5).

DG face integrals over :class:`warp.fem.Sides` are intentionally outside the
cell-only sum-factorized path until surface sum-factorization lands. These
tests force the dispatcher on and verify that side terms still take the legacy
kernel while complete DG operators, assembled from qualifying volume terms plus
falling-back side terms, remain numerically identical to the legacy path.

All test shapes stay on discontinuous P=5 ``Grid2D`` spaces with
``RegularQuadrature`` order ``2 * degree`` and ``wp.float64`` values, matching
the cached high-order sum-factorization coverage used by the apply and assembly
tests.
"""

import contextlib
import unittest

import numpy as np

import warp as wp
import warp._src.fem.integrate as fem_integrate
import warp.fem as fem
import warp.sparse as sparse
from warp.optim.linear import cg
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


def _make_grid_2d():
    return fem.Grid2D(
        res=wp.vec2i(3, 2),
        bounds_lo=wp.vec2d(0.0, 0.0),
        bounds_hi=wp.vec2d(1.0, 1.5),
        scalar_type=wp.float64,
    )


def _make_dg_case():
    """Build the P=5 discontinuous ``Grid2D`` case shared by the face tests."""

    degree = 5
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


@contextlib.contextmanager
def _sumfac_mode(mode):
    """Set the sum-factorization dispatch mode, restoring the previous one on exit."""

    previous = fem_integrate.get_sumfac_mode()
    fem_integrate.set_sumfac_mode(mode)
    try:
        yield
    finally:
        fem_integrate.set_sumfac_mode(previous)


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


def _integrate_side_matrix(mode):
    """Assemble the pure side DG flux matrix under ``mode``."""

    _space, _cell_test, _cell_trial, _cell_quadrature, side_test, side_trial, side_quadrature = _make_dg_case()
    with _sumfac_mode(mode), _capture_integrate_kernels() as kernels:
        matrix = fem.integrate(
            side_flux_form,
            fields={"u": side_trial, "v": side_test},
            quadrature=side_quadrature,
            output_dtype=wp.float64,
        )
    return matrix, kernels


def _integrate_mixed_matrix(mode):
    """Assemble one BSR matrix from a sum-factorizable volume term plus a side fallback term."""

    _space, cell_test, cell_trial, cell_quadrature, side_test, side_trial, side_quadrature = _make_dg_case()
    with _sumfac_mode(mode), _capture_integrate_kernels() as kernels:
        matrix = fem.integrate(
            volume_stiffness_form,
            fields={"u": cell_trial, "v": cell_test},
            quadrature=cell_quadrature,
            output_dtype=wp.float64,
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


def _assemble_dg_system(mode):
    """Assemble a compact P=5 SIP-style diffusion system and manufactured RHS."""

    _space, cell_test, cell_trial, cell_quadrature, side_test, side_trial, side_quadrature = _make_dg_case()
    with _sumfac_mode(mode):
        matrix = fem.integrate(
            implicit_volume_form,
            fields={"u": cell_trial, "v": cell_test},
            values={"viscosity": wp.float64(0.05)},
            quadrature=cell_quadrature,
            output_dtype=wp.float64,
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


def test_face_terms_fallback_correct(test, device):
    with wp.ScopedDevice(device):
        side_force, side_force_kernels = _integrate_side_matrix("force")
        side_off, side_off_kernels = _integrate_side_matrix("off")

        test.assertEqual(len(side_force_kernels), 1)
        test.assertFalse(_is_sumfac_kernel(side_force_kernels[0]), "side forms must fall back under mode force")
        test.assertFalse(_is_sumfac_kernel(side_off_kernels[0]))
        np.testing.assert_allclose(_bsr_to_dense(side_force), _bsr_to_dense(side_off), rtol=1e-9, atol=1e-10)

        mixed_force, mixed_force_kernels = _integrate_mixed_matrix("force")
        mixed_off, mixed_off_kernels = _integrate_mixed_matrix("off")

        test.assertEqual(len(mixed_force_kernels), 2)
        test.assertTrue(_is_sumfac_kernel(mixed_force_kernels[0]), "P=5 volume stiffness should use sumfac")
        test.assertFalse(_is_sumfac_kernel(mixed_force_kernels[1]), "side penalty must fall back under mode force")
        test.assertFalse(_is_sumfac_kernel(mixed_off_kernels[0]))
        test.assertFalse(_is_sumfac_kernel(mixed_off_kernels[1]))
        np.testing.assert_allclose(_bsr_to_dense(mixed_force), _bsr_to_dense(mixed_off), rtol=1e-9, atol=1e-10)


def test_end_to_end_dg_p5(test, device):
    with wp.ScopedDevice(device):
        matrix_force, rhs_force = _assemble_dg_system("force")
        matrix_off, rhs_off = _assemble_dg_system("off")

        np.testing.assert_allclose(rhs_force.numpy(), rhs_off.numpy(), rtol=1e-9, atol=1e-10)

        x_force = wp.zeros_like(rhs_force)
        x_off = wp.zeros_like(rhs_off)
        cg(matrix_force, rhs_force, x_force, tol=1e-12, maxiter=500, use_cuda_graph=False)
        cg(matrix_off, rhs_off, x_off, tol=1e-12, maxiter=500, use_cuda_graph=False)

        np.testing.assert_allclose(x_force.numpy(), x_off.numpy(), rtol=1e-6, atol=1e-9)

        probe = wp.array(np.linspace(-0.5, 0.5, rhs_force.shape[0]), dtype=wp.float64)
        y_force = sparse.bsr_mv(matrix_force, probe)
        y_off = sparse.bsr_mv(matrix_off, probe)
        np.testing.assert_allclose(y_force.numpy(), y_off.numpy(), rtol=1e-9, atol=1e-10)


devices = get_test_devices()


class TestFemSumfacFaces(unittest.TestCase):
    def setUp(self):
        self._sumfac_mode = fem_integrate.get_sumfac_mode()

    def tearDown(self):
        fem_integrate.set_sumfac_mode(self._sumfac_mode)


add_function_test(
    TestFemSumfacFaces, "test_face_terms_fallback_correct", test_face_terms_fallback_correct, devices=devices
)
add_function_test(TestFemSumfacFaces, "test_end_to_end_dg_p5", test_end_to_end_dg_p5, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
