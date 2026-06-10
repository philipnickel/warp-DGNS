# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the fused sum-factorized ``B^T D B`` apply path (Phase 3).

The transparent sum-factorization path replaces the standard linear-form
integration kernel with a fused tile kernel (``B`` interpolation stage,
seeded in-kernel Q-function ``D`` stage, ``B^T`` contraction stage) when the
form qualifies: cell domain, tensor-product discontinuous scalar space,
tensor-product ``RegularQuadrature``, plain test field, no trial field, and a
single injectable input field. The naive ``integrate()`` path is the golden
oracle throughout: the sum-factorized apply of a form must match the
matrix-vector product with the legacy-assembled BSR matrix of the same form.

The dispatch is driven by ``warp.fem.set_sumfac_mode``: ``"force"`` bypasses
the low-degree threshold (structural requirements still hold), ``"off"``
always selects the legacy kernels, and ``"auto"`` selects sum-factorization
only when applicable. Unqualified forms (side domains, simplex geometries,
vector-valued spaces, nodal assembly) must silently fall through to the
legacy path with identical results.

Degrees are kept to a modest set ({1, 3, 5} in 2D, {1, 2, 4} in 3D) and tile
shapes are reused across tests to bound first-run CUDA compile times.
"""

import contextlib
import unittest

import numpy as np

import warp as wp
import warp._src.fem.integrate as fem_integrate
import warp.fem as fem
import warp.sparse as sparse
from warp._src.fem.sumfac import make_sumfac_linear_operator
from warp.fem.utils import grid_to_tris
from warp.optim.linear import cg
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


@fem.integrand
def vec_mass_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    return wp.dot(u(s), v(s))


# Affine in the test function (nonzero at v = 0): only the value channel of
# the seeded extraction may be contracted, matching the legacy dispatch path
# which restricts the Taylor DOFs to the operators actually used.
@fem.integrand
def affine_offset_form(s: fem.Sample, u: fem.Field, v: fem.Field, w: fem.Field):
    return u(s) * v(s) + w(s)


# Dedicated integrands for the cache-isolation test, so the in-process kernel
# generation order (legacy first vs sum-factorized first) is controlled here.
@fem.integrand
def iso_form_legacy_first(s: fem.Sample, u: fem.Field, v: fem.Field):
    return u(s) * v(s)


@fem.integrand
def iso_form_sumfac_first(s: fem.Sample, u: fem.Field, v: fem.Field):
    return u(s) * v(s)


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


def _make_dg_case(geo, degree, rng):
    """Build a discontinuous tensor-product space with test/trial fields and quadrature."""

    space = fem.make_polynomial_space(geo, degree=degree, discontinuous=True, dtype=wp.float64)
    domain = fem.Cells(geometry=geo)
    test_field = fem.make_test(space=space, domain=domain)
    trial_field = fem.make_trial(space=space, domain=domain)
    quadrature = fem.RegularQuadrature(domain, order=2 * degree)

    x = rng.uniform(-1.0, 1.0, size=space.node_count())
    return space, domain, test_field, trial_field, quadrature, x


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


def _check_apply_matches_naive(test, form, geo, degree, rng, values=None):
    """Sum-factorized apply on a random vector vs legacy-assembled BSR matrix-vector product."""

    space, _domain, test_field, trial_field, quadrature, x = _make_dg_case(geo, degree, rng)

    matrix = fem.integrate(
        form,
        fields={"u": trial_field, "v": test_field},
        quadrature=quadrature,
        values=values,
        output_dtype=wp.float64,
    )
    x_wp = wp.array(x, dtype=wp.float64)
    y_naive = sparse.bsr_mv(matrix, x_wp)

    u = space.make_field()
    u.dof_values = x_wp

    with _sumfac_mode("force"), _capture_integrate_kernel() as captured:
        y_sumfac = fem.integrate(
            form,
            fields={"u": u, "v": test_field},
            quadrature=quadrature,
            values=values,
            output_dtype=wp.float64,
        )
    test.assertTrue(_is_sumfac_kernel(captured["kernel"]), "expected the sum-factorized kernel to be selected")

    np.testing.assert_allclose(y_sumfac.numpy(), y_naive.numpy(), rtol=1e-9, atol=1e-10)


def test_apply_equals_naive_mass_2d(test, device):
    rng = np.random.default_rng(50)
    with wp.ScopedDevice(device):
        geo = _make_grid_2d()
        for degree in (1, 3, 5):
            _check_apply_matches_naive(test, mass_form, geo, degree, rng)


def test_apply_equals_naive_mass_3d(test, device):
    rng = np.random.default_rng(51)
    with wp.ScopedDevice(device):
        geo = _make_grid_3d()
        for degree in (1, 2, 4):
            _check_apply_matches_naive(test, mass_form, geo, degree, rng)


def test_apply_equals_naive_stiffness(test, device):
    rng = np.random.default_rng(52)
    with wp.ScopedDevice(device):
        _check_apply_matches_naive(test, stiffness_form, _make_grid_2d(), 3, rng)
        _check_apply_matches_naive(test, stiffness_form, _make_grid_3d(), 2, rng)


def test_apply_equals_naive_advection(test, device):
    rng = np.random.default_rng(53)
    with wp.ScopedDevice(device):
        _check_apply_matches_naive(
            test, advection_form_2d, _make_grid_2d(), 3, rng, values={"vel": wp.vec2d(0.7, -0.3)}
        )
        _check_apply_matches_naive(
            test, advection_form_3d, _make_grid_3d(), 2, rng, values={"vel": wp.vec3d(0.7, -0.3, 0.45)}
        )


def test_dispatch_selects_sumfac(test, device):
    """Mode "force" selects the sum-factorized kernel, "off" the legacy one, "auto" only when applicable."""

    rng = np.random.default_rng(54)
    with wp.ScopedDevice(device):
        geo = _make_grid_2d()

        def make_case(degree):
            space, _domain, test_field, _trial, quadrature, x = _make_dg_case(geo, degree, rng)
            u = space.make_field()
            u.dof_values = wp.array(x, dtype=wp.float64)
            return u, test_field, quadrature

        def integrate_and_capture(case, mode):
            u, test_field, quadrature = case
            with _sumfac_mode(mode), _capture_integrate_kernel() as captured:
                result = fem.integrate(
                    mass_form,
                    fields={"u": u, "v": test_field},
                    quadrature=quadrature,
                    output_dtype=wp.float64,
                )
            return captured["kernel"], result.numpy()

        case_high = make_case(5)
        case_low = make_case(1)

        kernel_force, res_force = integrate_and_capture(case_high, "force")
        kernel_off, res_off = integrate_and_capture(case_high, "off")
        kernel_auto_high, res_auto_high = integrate_and_capture(case_high, "auto")
        kernel_auto_low, _res = integrate_and_capture(case_low, "auto")
        kernel_force_low, _res = integrate_and_capture(case_low, "force")

        test.assertTrue(_is_sumfac_kernel(kernel_force))
        test.assertFalse(_is_sumfac_kernel(kernel_off))
        test.assertTrue(_is_sumfac_kernel(kernel_auto_high), "P=5 should qualify under auto mode")
        test.assertFalse(_is_sumfac_kernel(kernel_auto_low), "P=1 is below the low-degree threshold")
        test.assertTrue(_is_sumfac_kernel(kernel_force_low), "force bypasses the low-degree threshold")

        np.testing.assert_allclose(res_force, res_off, rtol=1e-9, atol=1e-10)
        np.testing.assert_allclose(res_auto_high, res_off, rtol=1e-9, atol=1e-10)


def _gen_trimesh(nx, ny):
    x = np.linspace(0.0, 1.0, nx + 1)
    y = np.linspace(0.0, 1.0, ny + 1)
    positions = np.transpose(np.meshgrid(x, y, indexing="ij"), axes=(1, 2, 0)).reshape(-1, 2)
    vidx = grid_to_tris(nx, ny)
    return wp.array(positions, dtype=wp.vec2d), wp.array(vidx, dtype=int)


def _check_falls_back_to_legacy(test, form, fields, values=None, quadrature=None, assembly=None):
    """Under mode "force" the form must still take the legacy path, with results identical to mode "off"."""

    def run(mode):
        with _sumfac_mode(mode), _capture_integrate_kernel() as captured:
            result = fem.integrate(
                form,
                fields=fields,
                values=values,
                quadrature=quadrature,
                assembly=assembly,
                output_dtype=wp.float64,
            )
        return captured["kernel"], result.numpy()

    kernel_force, res_force = run("force")
    kernel_off, res_off = run("off")

    test.assertFalse(_is_sumfac_kernel(kernel_force), "unqualified form must fall through to the legacy kernel")
    test.assertIs(kernel_force, kernel_off, "fallback must reuse the exact legacy kernel")
    assert_np_equal(res_force, res_off)


def test_fallback_unqualified(test, device):
    rng = np.random.default_rng(55)
    with wp.ScopedDevice(device):
        # Side (boundary) domain: faces are not supported by the cell-only sum-factorized path
        geo = _make_grid_2d()
        space = fem.make_polynomial_space(geo, degree=2, discontinuous=True, dtype=wp.float64)
        sides = fem.BoundarySides(geo)
        side_test = fem.make_test(space=space, domain=sides)
        side_quadrature = fem.RegularQuadrature(sides, order=4)
        u = space.make_field()
        u.dof_values = wp.array(rng.uniform(-1.0, 1.0, size=space.node_count()), dtype=wp.float64)
        _check_falls_back_to_legacy(
            test, mass_form, fields={"u": u.trace(), "v": side_test}, quadrature=side_quadrature
        )

        # Simplex geometry: triangle shape functions are not tensor products
        positions, tri_vidx = _gen_trimesh(3, 2)
        tri_geo = fem.Trimesh2D(tri_vertex_indices=tri_vidx, positions=positions)
        tri_space = fem.make_polynomial_space(tri_geo, degree=2, discontinuous=True, dtype=wp.float64)
        tri_domain = fem.Cells(geometry=tri_geo)
        tri_test = fem.make_test(space=tri_space, domain=tri_domain)
        tri_quadrature = fem.RegularQuadrature(tri_domain, order=4)
        tri_u = tri_space.make_field()
        tri_u.dof_values = wp.array(rng.uniform(-1.0, 1.0, size=tri_space.node_count()), dtype=wp.float64)
        _check_falls_back_to_legacy(test, mass_form, fields={"u": tri_u, "v": tri_test}, quadrature=tri_quadrature)

        # Vector-valued space: seeding is scalar-only
        vec_space = fem.make_polynomial_space(geo, degree=2, discontinuous=True, dtype=wp.vec2d)
        vec_domain = fem.Cells(geometry=geo)
        vec_test = fem.make_test(space=vec_space, domain=vec_domain)
        vec_quadrature = fem.RegularQuadrature(vec_domain, order=4)
        vec_u = vec_space.make_field()
        vec_u.dof_values = wp.array(rng.uniform(-1.0, 1.0, size=(vec_space.node_count(), 2)), dtype=wp.vec2d)
        _check_falls_back_to_legacy(test, vec_mass_form, fields={"u": vec_u, "v": vec_test}, quadrature=vec_quadrature)

        # Nodal assembly: no quadrature, integrates at test nodes
        nodal_space = fem.make_polynomial_space(geo, degree=2, discontinuous=True, dtype=wp.float64)
        nodal_domain = fem.Cells(geometry=geo)
        nodal_test = fem.make_test(space=nodal_space, domain=nodal_domain)
        nodal_u = nodal_space.make_field()
        nodal_u.dof_values = wp.array(rng.uniform(-1.0, 1.0, size=nodal_space.node_count()), dtype=wp.float64)
        _check_falls_back_to_legacy(test, mass_form, fields={"u": nodal_u, "v": nodal_test}, assembly="nodal")

        # Explicit assembly="generic": the documented escape hatch making no
        # assumption about the integrand's content must be honored verbatim,
        # even when the form structurally qualifies for sum-factorization.
        generic_quadrature = fem.RegularQuadrature(nodal_domain, order=4)
        _check_falls_back_to_legacy(
            test,
            mass_form,
            fields={"u": nodal_u, "v": nodal_test},
            quadrature=generic_quadrature,
            assembly="generic",
        )


def test_affine_form_matches_dispatch(test, device):
    """A grad-free affine-in-test form must reproduce the legacy dispatch semantics exactly.

    The gradient seeds must not be evaluated when the integrand never applies
    a gradient operator to the test field; otherwise the affine offset would
    contaminate the ``f1`` channels and silently change the result relative to
    the legacy dispatch path.
    """

    rng = np.random.default_rng(58)
    with wp.ScopedDevice(device):
        geo = _make_grid_2d()
        space, _domain, test_field, _trial, quadrature, x = _make_dg_case(geo, 3, rng)
        u = space.make_field()
        u.dof_values = wp.array(x, dtype=wp.float64)
        w = space.make_field()
        w.dof_values = wp.array(rng.uniform(-1.0, 1.0, size=space.node_count()), dtype=wp.float64)

        def run(mode):
            with _sumfac_mode(mode), _capture_integrate_kernel() as captured:
                result = fem.integrate(
                    affine_offset_form,
                    fields={"u": u, "v": test_field, "w": w},
                    quadrature=quadrature,
                    output_dtype=wp.float64,
                )
            return captured["kernel"], result.numpy()

        kernel_force, res_force = run("force")
        kernel_off, res_off = run("off")

        test.assertTrue(_is_sumfac_kernel(kernel_force))
        test.assertFalse(_is_sumfac_kernel(kernel_off))
        np.testing.assert_allclose(res_force, res_off, rtol=1e-9, atol=1e-10)


def test_matrix_free_solve(test, device):
    """CG with the matrix-free sum-factorized operator matches the assembled-matrix solve."""

    rng = np.random.default_rng(56)
    mode_before = fem_integrate.get_sumfac_mode()
    with wp.ScopedDevice(device):
        geo = _make_grid_2d()
        space, _domain, test_field, trial_field, quadrature, _x = _make_dg_case(geo, 3, rng)

        matrix = fem.integrate(
            mass_form,
            fields={"u": trial_field, "v": test_field},
            quadrature=quadrature,
            output_dtype=wp.float64,
        )

        u_op = space.make_field()
        u_op_dofs = u_op.dof_values
        operator = make_sumfac_linear_operator(mass_form, fields={"u": u_op, "v": test_field}, quadrature=quadrature)
        test.assertEqual(operator.shape, (space.node_count(), space.node_count()))

        b = wp.array(rng.uniform(-1.0, 1.0, size=space.node_count()), dtype=wp.float64)

        # Direct apply check against the assembled matrix
        x_probe = wp.array(rng.uniform(-1.0, 1.0, size=space.node_count()), dtype=wp.float64)
        z = wp.zeros_like(b)
        y = wp.zeros_like(b)
        operator.matvec(x_probe, y, z, alpha=1.0, beta=0.0)
        np.testing.assert_allclose(z.numpy(), sparse.bsr_mv(matrix, x_probe).numpy(), rtol=1e-9, atol=1e-10)

        x_assembled = wp.zeros_like(b)
        cg(matrix, b, x_assembled, tol=1e-12, maxiter=1000, use_cuda_graph=False)

        x_matfree = wp.zeros_like(b)
        cg(operator, b, x_matfree, tol=1e-12, maxiter=1000, use_cuda_graph=False)

        np.testing.assert_allclose(x_matfree.numpy(), x_assembled.numpy(), rtol=1e-6, atol=1e-9)

        # The apply must not leave the caller's field bound to the solver iterate,
        # nor flip the process-global sum-factorization mode.
        test.assertIs(u_op.dof_values, u_op_dofs, "matvec must restore the input field's dof_values")
        test.assertEqual(fem_integrate.get_sumfac_mode(), mode_before)


def _check_cache_isolation(test, form, geo, rng, first_mode, second_mode):
    """Generating the kernels in either order never returns the wrong cached kernel."""

    space, _domain, test_field, _trial, quadrature, x = _make_dg_case(geo, 3, rng)
    u = space.make_field()
    u.dof_values = wp.array(x, dtype=wp.float64)

    def run(mode):
        with _sumfac_mode(mode), _capture_integrate_kernel() as captured:
            result = fem.integrate(
                form, fields={"u": u, "v": test_field}, quadrature=quadrature, output_dtype=wp.float64
            )
        return captured["kernel"], result.numpy()

    kernel_first, res_first = run(first_mode)
    kernel_second, res_second = run(second_mode)

    test.assertIsNot(kernel_first, kernel_second, "legacy and sum-factorized kernels must not share a cache entry")
    sumfac_kernel = kernel_first if first_mode == "force" else kernel_second
    legacy_kernel = kernel_second if first_mode == "force" else kernel_first
    test.assertTrue(_is_sumfac_kernel(sumfac_kernel))
    test.assertFalse(_is_sumfac_kernel(legacy_kernel))

    # Repeat in the same order: each mode must hit its own cached kernel
    kernel_first_again, _res = run(first_mode)
    kernel_second_again, _res = run(second_mode)
    test.assertIs(kernel_first_again, kernel_first)
    test.assertIs(kernel_second_again, kernel_second)

    np.testing.assert_allclose(res_first, res_second, rtol=1e-9, atol=1e-10)


def test_seed_substitution_cache_isolation(test, device):
    rng = np.random.default_rng(57)
    with wp.ScopedDevice(device):
        geo = _make_grid_2d()
        _check_cache_isolation(test, iso_form_legacy_first, geo, rng, first_mode="off", second_mode="force")
        _check_cache_isolation(test, iso_form_sumfac_first, geo, rng, first_mode="force", second_mode="off")


devices = get_test_devices()


class TestFemSumfacApply(unittest.TestCase):
    pass


add_function_test(
    TestFemSumfacApply, "test_apply_equals_naive_mass_2d", test_apply_equals_naive_mass_2d, devices=devices
)
add_function_test(
    TestFemSumfacApply, "test_apply_equals_naive_mass_3d", test_apply_equals_naive_mass_3d, devices=devices
)
add_function_test(
    TestFemSumfacApply, "test_apply_equals_naive_stiffness", test_apply_equals_naive_stiffness, devices=devices
)
add_function_test(
    TestFemSumfacApply, "test_apply_equals_naive_advection", test_apply_equals_naive_advection, devices=devices
)
add_function_test(TestFemSumfacApply, "test_dispatch_selects_sumfac", test_dispatch_selects_sumfac, devices=devices)
add_function_test(TestFemSumfacApply, "test_fallback_unqualified", test_fallback_unqualified, devices=devices)
add_function_test(
    TestFemSumfacApply, "test_affine_form_matches_dispatch", test_affine_form_matches_dispatch, devices=devices
)
add_function_test(TestFemSumfacApply, "test_matrix_free_solve", test_matrix_free_solve, devices=devices)
add_function_test(
    TestFemSumfacApply,
    "test_seed_substitution_cache_isolation",
    test_seed_substitution_cache_isolation,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
