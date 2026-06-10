# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the fused sum-factorized ``B^T D B`` apply path (Phase 3).

The sum-factorized path is an explicit opt-in: ``fem.integrate(...,
assembly="sumfac")`` replaces the standard linear-form integration kernel with
a fused tile kernel (``B`` interpolation stage, seeded in-kernel Q-function
``D`` stage, ``B^T`` contraction stage) when the form qualifies: cell domain,
tensor-product discontinuous scalar space, tensor-product
``RegularQuadrature``, plain test field, no trial field, and a single
injectable input field. The default ``integrate()`` path is the golden oracle
throughout: the sum-factorized apply of a form must match the matrix-vector
product with the default-assembled BSR matrix of the same form.

The sum-factorized path is never selected automatically, and forms that do not
qualify raise a descriptive error instead of silently falling back.

GPU-compiling correctness tests use degree P=4 exclusively (n = q = 5 per
axis, 2D and 3D) so the apply, assembly, and faces tests share one tile shape
per dimension, bounding first-run CUDA compile times.
"""

import contextlib
import unittest

import numpy as np

import warp as wp
import warp._src.fem.integrate as fem_integrate
import warp.fem as fem
import warp.sparse as sparse
from warp._src.fem.sumfac import make_sumfac_linear_operator
from warp.fem.utils import grid_to_quads, grid_to_tris
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


@fem.integrand
def rhs_only_form(s: fem.Sample, v: fem.Field):
    return v(s)


# Affine in the test function (nonzero at v = 0): only the value channel of
# the seeded extraction may be contracted, matching the legacy dispatch path
# which restricts the Taylor DOFs to the operators in use.
@fem.integrand
def affine_offset_form(s: fem.Sample, u: fem.Field, v: fem.Field, w: fem.Field):
    return u(s) * v(s) + w(s)


# Dedicated integrands for the cache-isolation test, so the in-process kernel
# generation order (default first vs sum-factorized first) is controlled here.
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
    """Sum-factorized apply on a random vector vs default-assembled BSR matrix-vector product."""

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

    with _capture_integrate_kernel() as captured:
        y_sumfac = fem.integrate(
            form,
            fields={"u": u, "v": test_field},
            quadrature=quadrature,
            values=values,
            output_dtype=wp.float64,
            assembly="sumfac",
        )
    test.assertTrue(_is_sumfac_kernel(captured["kernel"]), "expected the sum-factorized kernel to be selected")

    np.testing.assert_allclose(y_sumfac.numpy(), y_naive.numpy(), rtol=1e-9, atol=1e-10)


def test_apply_equals_naive_mass_2d(test, device):
    rng = np.random.default_rng(50)
    with wp.ScopedDevice(device):
        _check_apply_matches_naive(test, mass_form, _make_grid_2d(), 4, rng)


def test_apply_equals_naive_mass_3d(test, device):
    rng = np.random.default_rng(51)
    with wp.ScopedDevice(device):
        _check_apply_matches_naive(test, mass_form, _make_grid_3d(), 4, rng)


def test_apply_equals_naive_stiffness(test, device):
    rng = np.random.default_rng(52)
    with wp.ScopedDevice(device):
        _check_apply_matches_naive(test, stiffness_form, _make_grid_2d(), 4, rng)
        _check_apply_matches_naive(test, stiffness_form, _make_grid_3d(), 4, rng)


def test_apply_equals_naive_advection(test, device):
    rng = np.random.default_rng(53)
    with wp.ScopedDevice(device):
        _check_apply_matches_naive(
            test, advection_form_2d, _make_grid_2d(), 4, rng, values={"vel": wp.vec2d(0.7, -0.3)}
        )
        _check_apply_matches_naive(
            test, advection_form_3d, _make_grid_3d(), 4, rng, values={"vel": wp.vec3d(0.7, -0.3, 0.45)}
        )


def test_assembly_arg_selects_sumfac(test, device):
    """assembly="sumfac" always selects the sum-factorized kernel; the default never does."""

    rng = np.random.default_rng(54)
    with wp.ScopedDevice(device):
        geo = _make_grid_2d()
        space, _domain, test_field, _trial, quadrature, x = _make_dg_case(geo, 4, rng)
        u = space.make_field()
        u.dof_values = wp.array(x, dtype=wp.float64)

        def integrate_and_capture(assembly):
            with _capture_integrate_kernel() as captured:
                result = fem.integrate(
                    mass_form,
                    fields={"u": u, "v": test_field},
                    quadrature=quadrature,
                    output_dtype=wp.float64,
                    assembly=assembly,
                )
            return captured["kernel"], result.numpy()

        kernel_sumfac, res_sumfac = integrate_and_capture("sumfac")
        kernel_default, res_default = integrate_and_capture(None)
        kernel_dispatch, res_dispatch = integrate_and_capture("dispatch")

        test.assertTrue(_is_sumfac_kernel(kernel_sumfac))
        test.assertFalse(_is_sumfac_kernel(kernel_default), "sumfac must never be selected automatically")
        test.assertFalse(_is_sumfac_kernel(kernel_dispatch), "explicit dispatch must be honored verbatim")

        np.testing.assert_allclose(res_sumfac, res_default, rtol=1e-9, atol=1e-10)
        np.testing.assert_allclose(res_dispatch, res_default, rtol=1e-9, atol=1e-10)


def _gen_trimesh(nx, ny):
    x = np.linspace(0.0, 1.0, nx + 1)
    y = np.linspace(0.0, 1.0, ny + 1)
    positions = np.transpose(np.meshgrid(x, y, indexing="ij"), axes=(1, 2, 0)).reshape(-1, 2)
    vidx = grid_to_tris(nx, ny)
    return wp.array(positions, dtype=wp.vec2d), wp.array(vidx, dtype=int)


def _gen_nonaffine_quadmesh(nx, ny, rng):
    """Sheared quad mesh with perturbed interior vertices (per-element, non-constant Jacobians)."""
    x = np.linspace(0.0, 1.0, nx + 1)
    y = np.linspace(0.0, 1.0, ny + 1)
    grid = np.transpose(np.meshgrid(x, y, indexing="ij"), axes=(1, 2, 0)).reshape(-1, 2)
    shear = np.array([[1.0, 0.35], [0.2, 1.1]])
    positions = grid @ shear.T
    interior = (grid[:, 0] > 0.0) & (grid[:, 0] < 1.0) & (grid[:, 1] > 0.0) & (grid[:, 1] < 1.0)
    h = 1.0 / max(nx, ny)
    positions[interior] += rng.uniform(-0.15 * h, 0.15 * h, size=(int(interior.sum()), 2))
    vidx = grid_to_quads(nx, ny)
    return wp.array(positions, dtype=wp.vec2d), wp.array(vidx, dtype=int)


def test_apply_equals_naive_nonaffine_quadmesh(test, device):
    rng = np.random.default_rng(59)
    with wp.ScopedDevice(device):
        positions, quad_vidx = _gen_nonaffine_quadmesh(3, 2, rng)
        geo = fem.Quadmesh2D(quad_vertex_indices=quad_vidx, positions=positions)
        _check_apply_matches_naive(test, mass_form, geo, 4, rng)
        _check_apply_matches_naive(test, stiffness_form, geo, 4, rng)


def test_affine_form_matches_default(test, device):
    """A grad-free affine-in-test form must reproduce the default dispatch semantics exactly.

    The gradient seeds must not be evaluated when the integrand never applies
    a gradient operator to the test field; otherwise the affine offset would
    contaminate the ``f1`` channels and silently change the result relative to
    the default dispatch path.
    """

    rng = np.random.default_rng(58)
    with wp.ScopedDevice(device):
        geo = _make_grid_2d()
        space, _domain, test_field, _trial, quadrature, x = _make_dg_case(geo, 4, rng)
        u = space.make_field()
        u.dof_values = wp.array(x, dtype=wp.float64)
        w = space.make_field()
        w.dof_values = wp.array(rng.uniform(-1.0, 1.0, size=space.node_count()), dtype=wp.float64)

        def run(assembly):
            with _capture_integrate_kernel() as captured:
                result = fem.integrate(
                    affine_offset_form,
                    fields={"u": u, "v": test_field, "w": w},
                    quadrature=quadrature,
                    output_dtype=wp.float64,
                    assembly=assembly,
                )
            return captured["kernel"], result.numpy()

        kernel_sumfac, res_sumfac = run("sumfac")
        kernel_default, res_default = run(None)

        test.assertTrue(_is_sumfac_kernel(kernel_sumfac))
        test.assertFalse(_is_sumfac_kernel(kernel_default))
        np.testing.assert_allclose(res_sumfac, res_default, rtol=1e-9, atol=1e-10)


def test_matrix_free_solve(test, device):
    """CG with the matrix-free sum-factorized operator matches the assembled-matrix solve."""

    rng = np.random.default_rng(56)
    with wp.ScopedDevice(device):
        geo = _make_grid_2d()
        space, _domain, test_field, trial_field, quadrature, _x = _make_dg_case(geo, 4, rng)

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

        # The apply must not leave the caller's field bound to the solver iterate.
        test.assertIs(u_op.dof_values, u_op_dofs, "matvec must restore the input field's dof_values")


def _check_cache_isolation(test, form, geo, rng, first_assembly, second_assembly):
    """Generating the kernels in either order never returns the wrong cached kernel."""

    space, _domain, test_field, _trial, quadrature, x = _make_dg_case(geo, 4, rng)
    u = space.make_field()
    u.dof_values = wp.array(x, dtype=wp.float64)

    def run(assembly):
        with _capture_integrate_kernel() as captured:
            result = fem.integrate(
                form,
                fields={"u": u, "v": test_field},
                quadrature=quadrature,
                output_dtype=wp.float64,
                assembly=assembly,
            )
        return captured["kernel"], result.numpy()

    kernel_first, res_first = run(first_assembly)
    kernel_second, res_second = run(second_assembly)

    test.assertIsNot(kernel_first, kernel_second, "default and sum-factorized kernels must not share a cache entry")
    sumfac_kernel = kernel_first if first_assembly == "sumfac" else kernel_second
    legacy_kernel = kernel_second if first_assembly == "sumfac" else kernel_first
    test.assertTrue(_is_sumfac_kernel(sumfac_kernel))
    test.assertFalse(_is_sumfac_kernel(legacy_kernel))

    # Repeat in the same order: each assembly must hit its own cached kernel
    kernel_first_again, _res = run(first_assembly)
    kernel_second_again, _res = run(second_assembly)
    test.assertIs(kernel_first_again, kernel_first)
    test.assertIs(kernel_second_again, kernel_second)

    np.testing.assert_allclose(res_first, res_second, rtol=1e-9, atol=1e-10)


def test_seed_substitution_cache_isolation(test, device):
    rng = np.random.default_rng(57)
    with wp.ScopedDevice(device):
        geo = _make_grid_2d()
        _check_cache_isolation(test, iso_form_legacy_first, geo, rng, first_assembly=None, second_assembly="sumfac")
        _check_cache_isolation(test, iso_form_sumfac_first, geo, rng, first_assembly="sumfac", second_assembly=None)


devices = get_test_devices()


class TestFemSumfacApply(unittest.TestCase):
    def test_sumfac_raises_unqualified(self):
        """assembly="sumfac" raises a descriptive error for every unqualified form (host-side, no compiles)."""

        rng = np.random.default_rng(55)
        geo = _make_grid_2d()
        space, _domain, test_field, _trial, quadrature, x = _make_dg_case(geo, 4, rng)
        u = space.make_field()
        u.dof_values = wp.array(x, dtype=wp.float64)

        def integrate_sumfac(form, fields, **kwargs):
            kwargs.setdefault("output_dtype", wp.float64)
            return fem.integrate(form, fields=fields, assembly="sumfac", **kwargs)

        # Bilinear side forms are not supported by the sum-factorized path
        # (linear side forms are -- see test_fem_sumfac_faces.py)
        sides = fem.BoundarySides(geo)
        side_test = fem.make_test(space=space, domain=sides)
        side_trial = fem.make_trial(space=space, domain=sides)
        side_quadrature = fem.RegularQuadrature(sides, order=4)
        with self.assertRaisesRegex(NotImplementedError, "bilinear forms over side"):
            integrate_sumfac(mass_form, {"u": side_trial, "v": side_test}, quadrature=side_quadrature)

        # Simplex geometry: triangle shape functions are not tensor products
        positions, tri_vidx = _gen_trimesh(3, 2)
        tri_geo = fem.Trimesh2D(tri_vertex_indices=tri_vidx, positions=positions)
        tri_space = fem.make_polynomial_space(tri_geo, degree=2, discontinuous=True, dtype=wp.float64)
        tri_domain = fem.Cells(geometry=tri_geo)
        tri_test = fem.make_test(space=tri_space, domain=tri_domain)
        tri_quadrature = fem.RegularQuadrature(tri_domain, order=4)
        tri_u = tri_space.make_field()
        tri_u.dof_values = wp.array(rng.uniform(-1.0, 1.0, size=tri_space.node_count()), dtype=wp.float64)
        with self.assertRaisesRegex(NotImplementedError, "tensor-product"):
            integrate_sumfac(mass_form, {"u": tri_u, "v": tri_test}, quadrature=tri_quadrature)

        # Vector-valued space: seeding is scalar-only
        vec_space = fem.make_polynomial_space(geo, degree=2, discontinuous=True, dtype=wp.vec2d)
        vec_domain = fem.Cells(geometry=geo)
        vec_test = fem.make_test(space=vec_space, domain=vec_domain)
        vec_quadrature = fem.RegularQuadrature(vec_domain, order=4)
        vec_u = vec_space.make_field()
        vec_u.dof_values = wp.array(rng.uniform(-1.0, 1.0, size=(vec_space.node_count(), 2)), dtype=wp.vec2d)
        with self.assertRaisesRegex(NotImplementedError, "scalar"):
            integrate_sumfac(vec_mass_form, {"u": vec_u, "v": vec_test}, quadrature=vec_quadrature)

        # Backward kernel generation is unsupported
        with self.assertRaisesRegex(NotImplementedError, "backward"):
            integrate_sumfac(
                mass_form,
                {"u": u, "v": test_field},
                quadrature=quadrature,
                kernel_options={"enable_backward": True},
            )

        # Differentiable field inputs are unsupported
        u_grad = space.make_field()
        u_grad.dof_values = wp.array(x, dtype=wp.float64, requires_grad=True)
        with self.assertRaisesRegex(NotImplementedError, "differentiable field inputs"):
            integrate_sumfac(mass_form, {"u": u_grad, "v": test_field}, quadrature=quadrature)

        # Differentiable outputs are unsupported
        out_grad = wp.zeros(space.node_count(), dtype=wp.float64, requires_grad=True)
        with self.assertRaisesRegex(NotImplementedError, "differentiable outputs"):
            integrate_sumfac(mass_form, {"u": u, "v": test_field}, quadrature=quadrature, output=out_grad)

        # Accumulation dtype must be float32 or float64
        with self.assertRaisesRegex(NotImplementedError, "accumulation"):
            integrate_sumfac(
                mass_form,
                {"u": u, "v": test_field},
                quadrature=quadrature,
                accumulate_dtype=wp.float16,
            )

        # No injectable input field (pure right-hand-side form)
        with self.assertRaisesRegex(NotImplementedError, "input field"):
            integrate_sumfac(rhs_only_form, {"v": test_field}, quadrature=quadrature)


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
add_function_test(
    TestFemSumfacApply,
    "test_apply_equals_naive_nonaffine_quadmesh",
    test_apply_equals_naive_nonaffine_quadmesh,
    devices=devices,
)
add_function_test(
    TestFemSumfacApply, "test_assembly_arg_selects_sumfac", test_assembly_arg_selects_sumfac, devices=devices
)
add_function_test(
    TestFemSumfacApply, "test_affine_form_matches_default", test_affine_form_matches_default, devices=devices
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
