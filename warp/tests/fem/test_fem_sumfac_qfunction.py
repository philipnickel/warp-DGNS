# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Tests for SeedField-based Q-function extraction (sum-factorized DG, Phase 2).

The ``D`` stage of the ``B^T D B`` factorization evaluates the user's integrand
pointwise at every quadrature point with the test function replaced by seeds
``(v=1, grad v=0)`` and ``(v=0, grad v=e_i)``, recovering the coefficients
``(f0, f1)`` such that the integrand equals ``f0 * v + dot(f1, grad v)``
(exactly, because weak forms are linear in the test function). These tests pin
the extracted coefficients against closed forms for the mass, stiffness, and
advection bilinear forms, validate the affine geometry factors ``(J^{-T},
|det J|)`` -- including a sheared quad mesh whose non-diagonal Jacobian
distinguishes ``J^{-T}`` from ``J^{-1}`` -- and link seeded evaluations back to
the standard ``integrate()`` oracle through the linearity identity, in 2D and
in 3D at degree 5. Unsupported inputs (side/face domains, node-based operators
on the test field, fields incompatible with the quadrature domain) must raise
instead of extracting silently wrong coefficients.

All comparisons use ``wp.float64`` geometries so the 1e-9 tolerances from the
implementation plan are meaningful. Function spaces are discontinuous
(DG) tensor-product spaces, including one degree >= 5 case.
"""

import unittest

import numpy as np

import warp as wp
import warp.fem as fem
from warp._src.fem.sumfac.qfunction import extract_qfunction, reference_geometry_factors
from warp.tests.unittest_utils import *


@fem.integrand
def mass_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    return u(s) * v(s)


@fem.integrand
def stiffness_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    return wp.dot(fem.grad(u, s), fem.grad(v, s))


@fem.integrand
def advection_form_3d(s: fem.Sample, u: fem.Field, v: fem.Field, b: wp.vec3d):
    return wp.dot(b, fem.grad(u, s)) * v(s)


@fem.integrand
def mass_stiffness_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    return u(s) * v(s) + wp.dot(fem.grad(u, s), fem.grad(v, s))


@fem.integrand
def grad_expr(s: fem.Sample, u: fem.Field):
    return fem.grad(u, s)


@fem.integrand
def jump_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    return fem.jump(u, s) * fem.jump(v, s)


@fem.integrand
def at_node_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    # Linear in v, but reads the test field through a node-based operator
    return u(fem.at_node(v, s)) * v(s)


def _make_grid_2d():
    return fem.Grid2D(
        res=wp.vec2i(3, 2),
        bounds_lo=wp.vec2d(0.0, 0.0),
        bounds_hi=wp.vec2d(1.0, 1.5),
        scalar_type=wp.float64,
    )


def _make_grid_3d(res=(2, 2, 2)):
    return fem.Grid3D(
        res=wp.vec3i(*res),
        bounds_lo=wp.vec3d(0.0, 0.0, 0.0),
        bounds_hi=wp.vec3d(1.0, 2.0, 0.5),
        scalar_type=wp.float64,
    )


def _make_case(geo, degree, rng):
    """Build a random DG field, test field, and quadrature on ``geo``."""

    space = fem.make_polynomial_space(geo, degree=degree, discontinuous=True, dtype=wp.float64)
    domain = fem.Cells(geometry=geo)
    test_field = fem.make_test(space=space, domain=domain)
    quadrature = fem.RegularQuadrature(domain, order=2 * degree)

    u = space.make_field()
    u.dof_values = wp.array(rng.uniform(-1.0, 1.0, size=space.node_count()), dtype=wp.float64)

    return space, domain, test_field, quadrature, u


def _interpolate_at_qp(field_or_integrand, quadrature, dtype, fields=None):
    """Evaluate a field (or integrand) at all quadrature points, reshaped per element."""

    dest = wp.zeros(quadrature.total_point_count(), dtype=dtype)
    if fields is None:
        fem.interpolate(field_or_integrand, dest=dest, at=quadrature)
    else:
        fem.interpolate(field_or_integrand, dest=dest, at=quadrature, fields=fields)

    n_elt = quadrature.domain.element_count()
    return dest.numpy().reshape(n_elt, quadrature.max_points_per_element(), -1)


def _fold_scale(quadrature, det_abs):
    """Return the per-QP geometry folding factor ``weight * |det J|`` as a NumPy array."""

    weights = np.asarray(quadrature.weights, dtype=np.float64)
    return weights[np.newaxis, :] * det_abs


def test_mass_qfunction(test, device):
    """Mass integrand ``u * v``: f0 == u at QPs, f1 == 0; folding multiplies by ``w |det J|``."""

    rng = np.random.default_rng(42)

    with wp.ScopedDevice(device):
        geo = _make_grid_2d()
        for degree in (2, 5):
            _space, domain, test_field, quadrature, u = _make_case(geo, degree, rng)

            f0, f1 = extract_qfunction(
                mass_form, fields={"u": u, "v": test_field}, quadrature=quadrature, fold_geometry=False
            )
            u_qp = _interpolate_at_qp(u, quadrature, wp.float64)[..., 0]

            np.testing.assert_allclose(f0.numpy(), u_qp, atol=1e-12, rtol=0.0)
            np.testing.assert_allclose(f1.numpy(), 0.0, atol=1e-12, rtol=0.0)

            # Geometry folding: folded coefficients equal raw ones times weight * |det J|
            f0_folded, _f1_folded = extract_qfunction(
                mass_form, fields={"u": u, "v": test_field}, quadrature=quadrature, fold_geometry=True
            )
            _jac_inv_t, det_abs = reference_geometry_factors(domain, quadrature)
            scale = _fold_scale(quadrature, det_abs.numpy())
            np.testing.assert_allclose(f0_folded.numpy(), scale * u_qp, atol=1e-12, rtol=0.0)


def test_stiffness_qfunction(test, device):
    """Stiffness integrand ``dot(grad u, grad v)``: f0 == 0, f1 == physical grad u (J^{-T} applied)."""

    rng = np.random.default_rng(43)

    with wp.ScopedDevice(device):
        for geo, degree in ((_make_grid_3d(), 2), (_make_grid_3d(res=(1, 2, 1)), 5)):
            _space, _domain, test_field, quadrature, u = _make_case(geo, degree, rng)

            f0, f1 = extract_qfunction(
                stiffness_form, fields={"u": u, "v": test_field}, quadrature=quadrature, fold_geometry=False
            )
            grad_u_qp = _interpolate_at_qp(grad_expr, quadrature, wp.vec3d, fields={"u": u})

            np.testing.assert_allclose(f0.numpy(), 0.0, atol=1e-9, rtol=0.0)
            np.testing.assert_allclose(f1.numpy(), grad_u_qp, atol=1e-9, rtol=0.0)


def test_advection_qfunction(test, device):
    """Advection integrand ``dot(b, grad u) * v``: f0 == b . grad u, f1 == 0."""

    rng = np.random.default_rng(44)
    b = (0.7, -1.3, 2.1)

    with wp.ScopedDevice(device):
        geo = _make_grid_3d()
        _space, _domain, test_field, quadrature, u = _make_case(geo, 3, rng)

        f0, f1 = extract_qfunction(
            advection_form_3d,
            fields={"u": u, "v": test_field},
            values={"b": wp.vec3d(*b)},
            quadrature=quadrature,
            fold_geometry=False,
        )
        grad_u_qp = _interpolate_at_qp(grad_expr, quadrature, wp.vec3d, fields={"u": u})

        np.testing.assert_allclose(f0.numpy(), grad_u_qp @ np.asarray(b), atol=1e-9, rtol=0.0)
        np.testing.assert_allclose(f1.numpy(), 0.0, atol=1e-12, rtol=0.0)


def test_geometry_factors_affine(test, device):
    """On a uniform Grid3D, ``|det J|`` and ``J^{-T}`` match the analytic grid spacings."""

    res = (2, 3, 4)
    extents = (1.0, 2.0, 0.5)
    spacings = np.asarray(extents) / np.asarray(res)

    with wp.ScopedDevice(device):
        geo = fem.Grid3D(
            res=wp.vec3i(*res),
            bounds_lo=wp.vec3d(0.0, 0.0, 0.0),
            bounds_hi=wp.vec3d(*extents),
            scalar_type=wp.float64,
        )
        domain = fem.Cells(geometry=geo)
        quadrature = fem.RegularQuadrature(domain, order=4)

        jac_inv_t, det_abs = reference_geometry_factors(domain, quadrature)

        n_elt = domain.element_count()
        n_qp = quadrature.max_points_per_element()
        test.assertEqual(det_abs.shape, (n_elt, n_qp))
        test.assertEqual(jac_inv_t.shape, (n_elt, n_qp))

        np.testing.assert_allclose(det_abs.numpy(), np.prod(spacings), atol=1e-14, rtol=0.0)

        expected_jac_inv_t = np.broadcast_to(np.diag(1.0 / spacings), (n_elt, n_qp, 3, 3))
        np.testing.assert_allclose(jac_inv_t.numpy(), expected_jac_inv_t, atol=1e-12, rtol=0.0)


def test_geometry_factors_sheared(test, device):
    """On a uniformly sheared quad mesh, ``J`` is non-diagonal so ``J^{-T} != J^{-1}``.

    Grid geometries have diagonal deformation gradients, for which a transpose
    error in ``reference_geometry_factors`` would be invisible; this affine but
    non-orthogonal mesh pins the ``J^{-T}`` orientation down.
    """

    res = (2, 3)
    extents = (1.0, 1.5)
    spacings = np.asarray(extents) / np.asarray(res)
    # Non-diagonal, non-symmetric linear map with positive determinant
    shear = np.array([[1.0, 0.3], [0.2, 1.1]])

    with wp.ScopedDevice(device):
        x = np.linspace(0.0, extents[0], res[0] + 1)
        y = np.linspace(0.0, extents[1], res[1] + 1)
        grid_pos = np.stack(np.meshgrid(x, y, indexing="ij"), axis=-1).reshape(-1, 2)
        positions = grid_pos @ shear.T

        geo = fem.Quadmesh2D(
            quad_vertex_indices=wp.array(fem.utils.grid_to_quads(*res), dtype=int),
            positions=wp.array(positions, dtype=wp.vec2d),
        )
        domain = fem.Cells(geometry=geo)
        quadrature = fem.RegularQuadrature(domain, order=3)

        jac_inv_t, det_abs = reference_geometry_factors(domain, quadrature)

        # Every cell is the same parallelogram: J = shear @ diag(spacings)
        jac = shear @ np.diag(spacings)
        n_elt = domain.element_count()
        n_qp = quadrature.max_points_per_element()

        np.testing.assert_allclose(det_abs.numpy(), abs(np.linalg.det(jac)), atol=1e-14, rtol=0.0)

        expected_jac_inv_t = np.broadcast_to(np.linalg.inv(jac).T, (n_elt, n_qp, 2, 2))
        np.testing.assert_allclose(jac_inv_t.numpy(), expected_jac_inv_t, atol=1e-12, rtol=0.0)


def test_seedfield_eval_matches_basis_sum(test, device):
    """The d+1 seeded evaluations reproduce the action on any test-basis combination.

    For a form linear in ``v``, ``sum_k c_k * integrand(v=phi_k)`` integrated by the
    standard ``integrate()`` machinery must equal ``sum_qp w |det J| * (f0 * w_c +
    dot(f1, grad w_c))`` where ``w_c = sum_k c_k phi_k`` is the discrete field with
    DOF values ``c``. This links the SeedField evaluation to the AdjointField basis
    machinery it stands in for.

    The 3D anisotropic case at degree 5 exercises the gradient/``J^{-T}`` path
    against the ``integrate()`` oracle (rather than against ``fem.interpolate``,
    which shares the field-evaluation code path with the extraction kernel).
    """

    rng = np.random.default_rng(45)

    with wp.ScopedDevice(device):
        cases = (
            (_make_grid_2d(), 2, wp.vec2d),
            (_make_grid_3d(res=(1, 2, 1)), 5, wp.vec3d),
        )
        for geo, degree, grad_dtype in cases:
            space, _domain, test_field, quadrature, u = _make_case(geo, degree, rng)

            # Oracle: standard linear-form integration against every test basis function
            r = fem.integrate(
                mass_stiffness_form,
                fields={"u": u, "v": test_field},
                quadrature=quadrature,
                output_dtype=wp.float64,
            )

            # Random test-basis combination w_c = sum_k c_k phi_k, as a discrete field
            c = rng.uniform(-1.0, 1.0, size=space.node_count())
            c_field = space.make_field()
            c_field.dof_values = wp.array(c, dtype=wp.float64)

            lhs = float(c @ r.numpy())

            # Reconstruction from the geometry-folded seeded coefficients
            f0, f1 = extract_qfunction(
                mass_stiffness_form, fields={"u": u, "v": test_field}, quadrature=quadrature, fold_geometry=True
            )
            c_qp = _interpolate_at_qp(c_field, quadrature, wp.float64)[..., 0]
            grad_c_qp = _interpolate_at_qp(grad_expr, quadrature, grad_dtype, fields={"u": c_field})

            rhs = float(np.sum(f0.numpy() * c_qp) + np.sum(f1.numpy() * grad_c_qp))

            np.testing.assert_allclose(rhs, lhs, atol=1e-9, rtol=1e-12)


devices = get_test_devices()


class TestFemSumfacQFunction(unittest.TestCase):
    # Input-validation tests: these raise on the host before any kernel is
    # generated, so they run once on the default device.

    def test_side_domain_raises(self):
        """Side (DG face) forms must raise: jump(v) would silently extract as zero."""

        rng = np.random.default_rng(46)
        geo = _make_grid_2d()
        space = fem.make_polynomial_space(geo, degree=2, discontinuous=True, dtype=wp.float64)
        u = space.make_field()
        u.dof_values = wp.array(rng.uniform(-1.0, 1.0, size=space.node_count()), dtype=wp.float64)

        sides = fem.Sides(geometry=geo)
        v = fem.make_test(space=space, domain=sides)
        quadrature = fem.RegularQuadrature(sides, order=4)

        with self.assertRaisesRegex(NotImplementedError, "cell domains"):
            extract_qfunction(jump_form, fields={"u": u.trace(), "v": v}, quadrature=quadrature)

    def test_unsupported_test_operator_raises(self):
        """Node-based operators on the test field are not seedable and must raise."""

        rng = np.random.default_rng(47)
        geo = _make_grid_2d()
        _space, _domain, test_field, quadrature, u = _make_case(geo, 2, rng)

        with self.assertRaisesRegex(NotImplementedError, "at_node"):
            extract_qfunction(at_node_form, fields={"u": u, "v": test_field}, quadrature=quadrature)

    def test_incompatible_field_raises(self):
        """A field defined on the wrong element kind must fail the compatibility check."""

        rng = np.random.default_rng(48)
        geo = _make_grid_2d()
        _space, _domain, test_field, quadrature, u = _make_case(geo, 2, rng)

        with self.assertRaisesRegex(ValueError, "same kind of elements"):
            extract_qfunction(mass_form, fields={"u": u.trace(), "v": test_field}, quadrature=quadrature)


add_function_test(TestFemSumfacQFunction, "test_mass_qfunction", test_mass_qfunction, devices=devices)
add_function_test(TestFemSumfacQFunction, "test_stiffness_qfunction", test_stiffness_qfunction, devices=devices)
add_function_test(TestFemSumfacQFunction, "test_advection_qfunction", test_advection_qfunction, devices=devices)
add_function_test(TestFemSumfacQFunction, "test_geometry_factors_affine", test_geometry_factors_affine, devices=devices)
add_function_test(
    TestFemSumfacQFunction, "test_geometry_factors_sheared", test_geometry_factors_sheared, devices=devices
)
add_function_test(
    TestFemSumfacQFunction,
    "test_seedfield_eval_matches_basis_sum",
    test_seedfield_eval_matches_basis_sum,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
