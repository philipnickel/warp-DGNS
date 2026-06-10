# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Tests for DG face handling with the explicit sum-factorization opt-in (Phase 5).

DG face integrals over :class:`warp.fem.Sides` are intentionally outside the
cell-only sum-factorized path until surface sum-factorization lands: requesting
``assembly="sumfac"`` for a side form raises a descriptive error, and complete
DG operators are assembled from sum-factorized volume terms plus default-path
side terms. These tests verify the error, the mixed assembly, and that the
resulting end-to-end DG system remains numerically identical to the all-default
path.

All test shapes stay on discontinuous P=4 ``Grid2D`` spaces with
``RegularQuadrature`` order ``2 * degree`` and ``wp.float64`` values, matching
the shared P=4 sum-factorization coverage of the apply and assembly tests.
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

    Only the volume term may use ``assembly="sumfac"``; the side penalty falls
    outside the cell-only path and the RHS form has no injectable input field,
    so both always use the default assembly.
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


devices = get_test_devices()


class TestFemSumfacFaces(unittest.TestCase):
    def test_side_sumfac_raises(self):
        """Requesting assembly="sumfac" for a side (DG face) form raises a descriptive error."""

        _space, _cell_test, _cell_trial, _cell_quadrature, side_test, side_trial, side_quadrature = _make_dg_case()
        with self.assertRaisesRegex(NotImplementedError, "cell domains"):
            fem.integrate(
                side_flux_form,
                fields={"u": side_trial, "v": side_test},
                quadrature=side_quadrature,
                output_dtype=wp.float64,
                assembly="sumfac",
            )


add_function_test(
    TestFemSumfacFaces,
    "test_mixed_volume_sumfac_plus_side_default",
    test_mixed_volume_sumfac_plus_side_default,
    devices=devices,
)
add_function_test(TestFemSumfacFaces, "test_end_to_end_dg_p4", test_end_to_end_dg_p4, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
