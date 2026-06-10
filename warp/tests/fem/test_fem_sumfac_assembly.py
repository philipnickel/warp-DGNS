# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the sum-factorized assembly of bilinear forms to BSR (Phase 4).

When ``integrate()`` is called with both a test and a trial field on a
qualifying form (cell domain, same scalar discontinuous tensor-product space
on both sides, tensor-product ``RegularQuadrature``, value and gradient
operators only), the sum-factorized path assembles the element-block-diagonal
sparse matrix through the fused ``B^T D B`` action instead of the naive
per-entry quadrature loop, and feeds the same ``bsr_set_from_triplets`` tail
as the legacy kernels. The naive ``integrate()`` path is the golden oracle
throughout: the assembled matrices must agree entry-wise in dense form.

Degrees are kept to a modest set ({1, 3, 5} in 2D, {1, 2, 4} in 3D) and tile
shapes are shared with the Phase 3 apply tests to bound first-run CUDA
compile times.
"""

import contextlib
import unittest

import numpy as np

import warp as wp
import warp._src.fem.integrate as fem_integrate
import warp.fem as fem
import warp.sparse as sparse
from warp.fem.utils import grid_to_tris
from warp.tests.unittest_utils import *


@fem.integrand
def mass_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    return u(s) * v(s)


@fem.integrand
def stiffness_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    return wp.dot(fem.grad(u, s), fem.grad(v, s))


@fem.integrand
def advection_form_2d(s: fem.Sample, u: fem.Field, v: fem.Field, vel: wp.vec2d):
    return wp.dot(vel, fem.grad(u, s)) * v(s)


@fem.integrand
def advection_form_3d(s: fem.Sample, u: fem.Field, v: fem.Field, vel: wp.vec3d):
    return wp.dot(vel, fem.grad(u, s)) * v(s)


# Transposed advection: the gradient channel sits on the TEST side, catching
# test/trial channel transposition errors that the symmetric forms cannot.
@fem.integrand
def advection_transpose_form_2d(s: fem.Sample, u: fem.Field, v: fem.Field, vel: wp.vec2d):
    return u(s) * wp.dot(vel, fem.grad(v, s))


@fem.integrand
def vec_mass_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    return wp.dot(u(s), v(s))


def _make_grid_2d():
    return fem.Grid2D(
        res=wp.vec2i(3, 2),
        bounds_lo=wp.vec2d(0.0, 0.0),
        bounds_hi=wp.vec2d(1.0, 1.5),
        scalar_type=wp.float64,
    )


def _make_grid_3d():
    return fem.Grid3D(
        res=wp.vec3i(2, 2, 2),
        bounds_lo=wp.vec3d(0.0, 0.0, 0.0),
        bounds_hi=wp.vec3d(1.0, 2.0, 0.5),
        scalar_type=wp.float64,
    )


def _make_dg_case(geo, degree):
    """Build a discontinuous tensor-product space with test/trial fields and quadrature."""

    space = fem.make_polynomial_space(geo, degree=degree, discontinuous=True, dtype=wp.float64)
    domain = fem.Cells(geometry=geo)
    test_field = fem.make_test(space=space, domain=domain)
    trial_field = fem.make_trial(space=space, domain=domain)
    quadrature = fem.RegularQuadrature(domain, order=2 * degree)
    return space, domain, test_field, trial_field, quadrature


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
def _capture_integrate_kernel():
    """Capture the kernel selected by ``fem.integrate`` for dispatch assertions."""

    captured = {}
    original = fem_integrate._launch_integrate_kernel

    def spy(*args, **kwargs):
        captured["kernel"] = kwargs["kernel"]
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


def _integrate_matrix(form, test_field, trial_field, quadrature, mode, values=None):
    with _sumfac_mode(mode), _capture_integrate_kernel() as captured:
        matrix = fem.integrate(
            form,
            fields={"u": trial_field, "v": test_field},
            quadrature=quadrature,
            values=values,
            output_dtype=wp.float64,
        )
    return matrix, captured["kernel"]


def _check_assembly_matches_naive(test, form, geo, degree, values=None):
    """Sum-factorized assembled BSR vs the legacy-assembled matrix, in dense form."""

    _space, _domain, test_field, trial_field, quadrature = _make_dg_case(geo, degree)

    matrix_naive, kernel_naive = _integrate_matrix(form, test_field, trial_field, quadrature, "off", values=values)
    matrix_sumfac, kernel_sumfac = _integrate_matrix(form, test_field, trial_field, quadrature, "force", values=values)

    test.assertFalse(_is_sumfac_kernel(kernel_naive))
    test.assertTrue(_is_sumfac_kernel(kernel_sumfac), "expected the sum-factorized assembly kernel to be selected")

    np.testing.assert_allclose(_bsr_to_dense(matrix_sumfac), _bsr_to_dense(matrix_naive), rtol=1e-9, atol=1e-10)


def test_assembled_equals_naive_mass_2d(test, device):
    with wp.ScopedDevice(device):
        geo = _make_grid_2d()
        for degree in (1, 3, 5):
            _check_assembly_matches_naive(test, mass_form, geo, degree)


def test_assembled_equals_naive_mass_3d(test, device):
    with wp.ScopedDevice(device):
        geo = _make_grid_3d()
        for degree in (1, 2, 4):
            _check_assembly_matches_naive(test, mass_form, geo, degree)


def test_assembled_equals_naive_stiffness(test, device):
    with wp.ScopedDevice(device):
        _check_assembly_matches_naive(test, stiffness_form, _make_grid_2d(), 3)
        _check_assembly_matches_naive(test, stiffness_form, _make_grid_3d(), 2)


def test_assembled_equals_naive_advection(test, device):
    with wp.ScopedDevice(device):
        _check_assembly_matches_naive(test, advection_form_2d, _make_grid_2d(), 3, values={"vel": wp.vec2d(0.7, -0.3)})
        _check_assembly_matches_naive(
            test, advection_form_3d, _make_grid_3d(), 2, values={"vel": wp.vec3d(0.7, -0.3, 0.45)}
        )
        # Gradient channel on the test side (transposed advection)
        _check_assembly_matches_naive(
            test, advection_transpose_form_2d, _make_grid_2d(), 3, values={"vel": wp.vec2d(0.7, -0.3)}
        )


def test_blockdiag_structure(test, device):
    """The DG sum-factorized matrix has the same element-block-diagonal sparsity as the naive DG matrix."""

    with wp.ScopedDevice(device):
        geo = _make_grid_2d()
        degree = 3
        _space, domain, test_field, trial_field, quadrature = _make_dg_case(geo, degree)

        matrix_naive, _kernel = _integrate_matrix(mass_form, test_field, trial_field, quadrature, "off")
        matrix_sumfac, kernel_sumfac = _integrate_matrix(mass_form, test_field, trial_field, quadrature, "force")
        test.assertTrue(_is_sumfac_kernel(kernel_sumfac))

        nodes_per_element = (degree + 1) ** 2
        expected_nnz = domain.element_count() * nodes_per_element**2

        nnz_naive = matrix_naive.nnz_sync()
        nnz_sumfac = matrix_sumfac.nnz_sync()
        test.assertEqual(nnz_sumfac, nnz_naive)
        test.assertEqual(nnz_sumfac, expected_nnz)

        # Identical block layout (offsets and sorted column indices)
        assert_np_equal(matrix_sumfac.offsets.numpy(), matrix_naive.offsets.numpy())
        assert_np_equal(matrix_sumfac.columns.numpy()[:nnz_sumfac], matrix_naive.columns.numpy()[:nnz_naive])

        # Every block couples nodes of the same element (element-block-diagonal)
        rows = np.repeat(np.arange(matrix_sumfac.nrow), np.diff(matrix_sumfac.offsets.numpy()))
        cols = matrix_sumfac.columns.numpy()[:nnz_sumfac]
        assert_np_equal(rows // nodes_per_element, cols // nodes_per_element)


def test_assembly_apply_consistency(test, device):
    """``K_sumfac @ x`` (bsr_mv) matches the Phase 3 sum-factorized apply on ``x``."""

    rng = np.random.default_rng(60)
    with wp.ScopedDevice(device):
        cases = (
            (mass_form, _make_grid_2d(), 3, None),
            (stiffness_form, _make_grid_3d(), 2, None),
        )
        for form, geo, degree, values in cases:
            space, _domain, test_field, trial_field, quadrature = _make_dg_case(geo, degree)

            matrix, kernel_mat = _integrate_matrix(form, test_field, trial_field, quadrature, "force", values=values)
            test.assertTrue(_is_sumfac_kernel(kernel_mat))

            x = wp.array(rng.uniform(-1.0, 1.0, size=space.node_count()), dtype=wp.float64)
            y_mat = sparse.bsr_mv(matrix, x)

            u = space.make_field()
            u.dof_values = x
            with _sumfac_mode("force"), _capture_integrate_kernel() as captured:
                y_apply = fem.integrate(
                    form,
                    fields={"u": u, "v": test_field},
                    quadrature=quadrature,
                    values=values,
                    output_dtype=wp.float64,
                )
            test.assertTrue(_is_sumfac_kernel(captured["kernel"]))

            np.testing.assert_allclose(y_mat.numpy(), y_apply.numpy(), rtol=1e-9, atol=1e-10)


def _gen_trimesh(nx, ny):
    x = np.linspace(0.0, 1.0, nx + 1)
    y = np.linspace(0.0, 1.0, ny + 1)
    positions = np.transpose(np.meshgrid(x, y, indexing="ij"), axes=(1, 2, 0)).reshape(-1, 2)
    vidx = grid_to_tris(nx, ny)
    return wp.array(positions, dtype=wp.vec2d), wp.array(vidx, dtype=int)


def _check_falls_back_to_legacy(test, form, fields, values=None, quadrature=None):
    """Under mode "force" the form must still take the legacy path, with results identical to mode "off"."""

    def run(mode):
        with _sumfac_mode(mode), _capture_integrate_kernel() as captured:
            result = fem.integrate(
                form,
                fields=fields,
                values=values,
                quadrature=quadrature,
                output_dtype=wp.float64,
            )
        return captured["kernel"], result

    kernel_force, mat_force = run("force")
    kernel_off, mat_off = run("off")

    test.assertFalse(_is_sumfac_kernel(kernel_force), "unqualified form must fall through to the legacy kernel")
    test.assertIs(kernel_force, kernel_off, "fallback must reuse the exact legacy kernel")
    assert_np_equal(_bsr_to_dense(mat_force), _bsr_to_dense(mat_off))


def test_dispatch_and_fallback_bilinear(test, device):
    """Mode force/off/auto behave as for the linear path; unqualified bilinear forms fall back to legacy."""

    with wp.ScopedDevice(device):
        geo = _make_grid_2d()

        def integrate_and_capture(degree, mode):
            _space, _domain, test_field, trial_field, quadrature = _make_dg_case(geo, degree)
            matrix, kernel = _integrate_matrix(mass_form, test_field, trial_field, quadrature, mode)
            return kernel, _bsr_to_dense(matrix)

        kernel_force, mat_force = integrate_and_capture(5, "force")
        kernel_off, mat_off = integrate_and_capture(5, "off")
        kernel_auto_high, mat_auto_high = integrate_and_capture(5, "auto")
        kernel_auto_low, _mat = integrate_and_capture(1, "auto")
        kernel_force_low, _mat = integrate_and_capture(1, "force")

        test.assertTrue(_is_sumfac_kernel(kernel_force))
        test.assertFalse(_is_sumfac_kernel(kernel_off))
        test.assertTrue(_is_sumfac_kernel(kernel_auto_high), "P=5 should qualify under auto mode")
        test.assertFalse(_is_sumfac_kernel(kernel_auto_low), "P=1 is below the low-degree threshold")
        test.assertTrue(_is_sumfac_kernel(kernel_force_low), "force bypasses the low-degree threshold")

        np.testing.assert_allclose(mat_force, mat_off, rtol=1e-9, atol=1e-10)
        np.testing.assert_allclose(mat_auto_high, mat_off, rtol=1e-9, atol=1e-10)

        # Mixed test/trial spaces: not supported, must fall back to legacy
        domain = fem.Cells(geometry=geo)
        space_p2 = fem.make_polynomial_space(geo, degree=2, discontinuous=True, dtype=wp.float64)
        space_p3 = fem.make_polynomial_space(geo, degree=3, discontinuous=True, dtype=wp.float64)
        mixed_test = fem.make_test(space=space_p2, domain=domain)
        mixed_trial = fem.make_trial(space=space_p3, domain=domain)
        mixed_quadrature = fem.RegularQuadrature(domain, order=5)
        _check_falls_back_to_legacy(
            test, mass_form, fields={"u": mixed_trial, "v": mixed_test}, quadrature=mixed_quadrature
        )

        # Side (boundary) domain: faces are not supported by the cell-only sum-factorized path
        space = fem.make_polynomial_space(geo, degree=2, discontinuous=True, dtype=wp.float64)
        sides = fem.BoundarySides(geo)
        side_test = fem.make_test(space=space, domain=sides)
        side_trial = fem.make_trial(space=space, domain=sides)
        side_quadrature = fem.RegularQuadrature(sides, order=4)
        _check_falls_back_to_legacy(
            test, mass_form, fields={"u": side_trial, "v": side_test}, quadrature=side_quadrature
        )

        # Simplex geometry: triangle shape functions are not tensor products
        positions, tri_vidx = _gen_trimesh(3, 2)
        tri_geo = fem.Trimesh2D(tri_vertex_indices=tri_vidx, positions=positions)
        tri_space = fem.make_polynomial_space(tri_geo, degree=2, discontinuous=True, dtype=wp.float64)
        tri_domain = fem.Cells(geometry=tri_geo)
        tri_test = fem.make_test(space=tri_space, domain=tri_domain)
        tri_trial = fem.make_trial(space=tri_space, domain=tri_domain)
        tri_quadrature = fem.RegularQuadrature(tri_domain, order=4)
        _check_falls_back_to_legacy(test, mass_form, fields={"u": tri_trial, "v": tri_test}, quadrature=tri_quadrature)

        # Vector-valued space: seeding is scalar-only
        vec_space = fem.make_polynomial_space(geo, degree=2, discontinuous=True, dtype=wp.vec2d)
        vec_domain = fem.Cells(geometry=geo)
        vec_test = fem.make_test(space=vec_space, domain=vec_domain)
        vec_trial = fem.make_trial(space=vec_space, domain=vec_domain)
        vec_quadrature = fem.RegularQuadrature(vec_domain, order=4)
        _check_falls_back_to_legacy(
            test, vec_mass_form, fields={"u": vec_trial, "v": vec_test}, quadrature=vec_quadrature
        )


devices = get_test_devices()


class TestFemSumfacAssembly(unittest.TestCase):
    pass


add_function_test(
    TestFemSumfacAssembly,
    "test_assembled_equals_naive_mass_2d",
    test_assembled_equals_naive_mass_2d,
    devices=devices,
)
add_function_test(
    TestFemSumfacAssembly,
    "test_assembled_equals_naive_mass_3d",
    test_assembled_equals_naive_mass_3d,
    devices=devices,
)
add_function_test(
    TestFemSumfacAssembly,
    "test_assembled_equals_naive_stiffness",
    test_assembled_equals_naive_stiffness,
    devices=devices,
)
add_function_test(
    TestFemSumfacAssembly,
    "test_assembled_equals_naive_advection",
    test_assembled_equals_naive_advection,
    devices=devices,
)
add_function_test(TestFemSumfacAssembly, "test_blockdiag_structure", test_blockdiag_structure, devices=devices)
add_function_test(
    TestFemSumfacAssembly,
    "test_assembly_apply_consistency",
    test_assembly_apply_consistency,
    devices=devices,
)
add_function_test(
    TestFemSumfacAssembly,
    "test_dispatch_and_fallback_bilinear",
    test_dispatch_and_fallback_bilinear,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
