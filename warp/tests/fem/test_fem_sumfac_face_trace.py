# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the host-side face-trace primitives (surface sum-factorization, stage 1).

Three properties of :mod:`warp._src.fem.sumfac.face_trace` are pinned down:

1. **GLL value trace is a DOF slice**: with endpoint-including (GLL) 1D nodes,
   the face value trace equals the dense evaluation of the element's nodal
   interpolant restricted to the face (Kronecker of 1D interpolation matrices
   with the face coordinate pinned to 0/1), and the analytic restriction of a
   random separable polynomial interpolant.
2. **Normal-gradient trace**: collapsing the face-normal axis with the endpoint
   row of the 1D derivative matrix equals the dense Kronecker reference and the
   analytic normal derivative of a random separable polynomial interpolant.
3. **Side-machinery ordering (the critical one, CPU-only)**: the side-to-element
   coordinate mapping encoded by ``grid_side_face_ends`` /
   ``grid_side_to_face_coords`` / ``grid_outer_face_permutation`` reproduces,
   to ``atol=1e-12``, the inner and outer field values that ``warp.fem``'s own
   side machinery (``fem.inner`` / ``fem.outer`` over a ``Sides`` domain with a
   ``RegularQuadrature``) produces on ``Grid2D`` and ``Grid3D`` -- including
   the per-quadrature-point element coordinates reported by
   ``fem.to_inner_cell`` / ``fem.to_outer_cell``. This locks the face
   orientation risk (design spec section 9.8) before any surface tile kernel is
   written.

Tests 1 and 2 are host-side NumPy only. Test 3 compiles only standard
``warp.fem`` interpolation kernels on CPU (no sum-factorized tile kernels).
"""

import unittest

import numpy as np

import warp as wp
import warp.fem as fem
from warp._src.fem.polynomial import is_closed
from warp._src.fem.sumfac.face_trace import (
    face_axes,
    face_slice_indices,
    face_trace_normal_gradient,
    face_trace_value,
    grid_outer_face_permutation,
    grid_side_face_ends,
    grid_side_to_face_coords,
)
from warp._src.fem.sumfac.operators_1d import (
    build_derivative_matrix,
    build_interpolation_matrix,
    default_basis_nodes,
)
from warp.tests.unittest_utils import *

_DEGREES = (1, 3, 5)
_DIMS = (2, 3)


@fem.integrand
def _side_probe_2d(
    s: fem.Sample,
    domain: fem.Domain,
    u: fem.Field,
    pos: wp.array(dtype=wp.vec2d),
    nrm: wp.array(dtype=wp.vec2d),
    inner_val: wp.array(dtype=wp.float64),
    outer_val: wp.array(dtype=wp.float64),
    inner_cell: wp.array(dtype=int),
    outer_cell: wp.array(dtype=int),
    inner_coords: wp.array(dtype=wp.vec3d),
    outer_coords: wp.array(dtype=wp.vec3d),
    side_coords: wp.array(dtype=wp.vec3d),
):
    qp = s.qp_index
    pos[qp] = domain(s)
    nrm[qp] = fem.normal(domain, s)
    inner_val[qp] = fem.inner(u, s)
    outer_val[qp] = fem.outer(u, s)
    s_in = fem.to_inner_cell(domain, s)
    s_out = fem.to_outer_cell(domain, s)
    inner_cell[qp] = s_in.element_index
    outer_cell[qp] = s_out.element_index
    inner_coords[qp] = s_in.element_coords
    outer_coords[qp] = s_out.element_coords
    side_coords[qp] = s.element_coords


@fem.integrand
def _side_probe_3d(
    s: fem.Sample,
    domain: fem.Domain,
    u: fem.Field,
    pos: wp.array(dtype=wp.vec3d),
    nrm: wp.array(dtype=wp.vec3d),
    inner_val: wp.array(dtype=wp.float64),
    outer_val: wp.array(dtype=wp.float64),
    inner_cell: wp.array(dtype=int),
    outer_cell: wp.array(dtype=int),
    inner_coords: wp.array(dtype=wp.vec3d),
    outer_coords: wp.array(dtype=wp.vec3d),
    side_coords: wp.array(dtype=wp.vec3d),
):
    qp = s.qp_index
    pos[qp] = domain(s)
    nrm[qp] = fem.normal(domain, s)
    inner_val[qp] = fem.inner(u, s)
    outer_val[qp] = fem.outer(u, s)
    s_in = fem.to_inner_cell(domain, s)
    s_out = fem.to_outer_cell(domain, s)
    inner_cell[qp] = s_in.element_index
    outer_cell[qp] = s_out.element_index
    inner_coords[qp] = s_in.element_coords
    outer_coords[qp] = s_out.element_coords
    side_coords[qp] = s.element_coords


def _kron_face_reference(dofs, n, dim, axis, end, derivative):
    """Dense face-trace reference: Kronecker of 1D matrices, face coordinate pinned to ``end``.

    The face-normal axis carries the single-row interpolation (or derivative)
    matrix at the endpoint; the remaining axes carry the interpolation matrix
    evaluated at the GLL nodes. Row ordering of the Kronecker product is the
    face frame of ``face_axes`` (remaining axes increasing, slowest first).
    """
    nodes = default_basis_nodes(n - 1)
    endpoint = np.array([float(end)])
    mats = []
    for d in range(dim):
        if d == axis:
            mats.append(
                build_derivative_matrix(nodes, endpoint) if derivative else build_interpolation_matrix(nodes, endpoint)
            )
        else:
            mats.append(build_interpolation_matrix(nodes, nodes))
    op = mats[0]
    for mat in mats[1:]:
        op = np.kron(op, mat)
    return np.asarray(dofs) @ op.T


def _separable_polynomial_case(rng, degree, dim):
    """Random separable polynomial ``u(x) = prod_d p_d(x_d)`` of per-axis degree ``degree``.

    Returns the 1D polynomials and the lexicographic nodal DOF vector of its
    interpolant on the GLL node tensor product (exact, as ``u`` lies in the
    tensor-product space).
    """
    polys = [np.polynomial.Polynomial(rng.uniform(-1.0, 1.0, degree + 1)) for _ in range(dim)]
    nodes = default_basis_nodes(degree)
    vals_1d = [p(nodes) for p in polys]
    grids = np.meshgrid(*vals_1d, indexing="ij")
    dofs = grids[0].copy()
    for g in grids[1:]:
        dofs = dofs * g
    return polys, vals_1d, dofs.reshape(1, -1)


def _face_tangential_tensor(vals_1d, dim, axis):
    """Tensor product of the 1D nodal values over the face axes, in face-frame order."""
    tang = [vals_1d[d] for d in range(dim) if d != axis]
    if dim == 2:
        return tang[0]
    return np.multiply.outer(tang[0], tang[1]).reshape(-1)


class TestFemSumfacFaceTrace(unittest.TestCase):
    def test_face_value_trace_is_dof_slice(self):
        rng = np.random.default_rng(20260610)
        for degree in _DEGREES:
            n = degree + 1
            for dim in _DIMS:
                dofs = rng.uniform(-1.0, 1.0, size=(3, n**dim))
                polys, vals_1d, poly_dofs = _separable_polynomial_case(rng, degree, dim)
                for axis in range(dim):
                    for end in (0, 1):
                        with self.subTest(degree=degree, dim=dim, axis=axis, end=end):
                            traced = face_trace_value(dofs, n, dim, axis, end)
                            self.assertEqual(traced.shape, (3, n ** (dim - 1)))

                            # Dense interpolant restricted to the face (GLL endpoint property).
                            dense = _kron_face_reference(dofs, n, dim, axis, end, derivative=False)
                            np.testing.assert_allclose(traced, dense, rtol=0.0, atol=1e-14)

                            # Analytic restriction of a separable polynomial interpolant.
                            face_tensor = _face_tangential_tensor(vals_1d, dim, axis)
                            expected = polys[axis](float(end)) * face_tensor
                            poly_traced = face_trace_value(poly_dofs, n, dim, axis, end)[0]
                            np.testing.assert_allclose(poly_traced, expected, rtol=0.0, atol=1e-13)

    def test_face_normal_gradient_trace(self):
        rng = np.random.default_rng(20260611)
        for degree in _DEGREES:
            n = degree + 1
            for dim in _DIMS:
                dofs = rng.uniform(-1.0, 1.0, size=(3, n**dim))
                polys, vals_1d, poly_dofs = _separable_polynomial_case(rng, degree, dim)
                for axis in range(dim):
                    for end in (0, 1):
                        with self.subTest(degree=degree, dim=dim, axis=axis, end=end):
                            traced = face_trace_normal_gradient(dofs, n, dim, axis, end)
                            self.assertEqual(traced.shape, (3, n ** (dim - 1)))

                            # Dense Kronecker reference with the endpoint derivative row.
                            dense = _kron_face_reference(dofs, n, dim, axis, end, derivative=True)
                            np.testing.assert_allclose(traced, dense, rtol=0.0, atol=1e-12)

                            # Analytic normal derivative of a separable polynomial interpolant.
                            face_tensor = _face_tangential_tensor(vals_1d, dim, axis)
                            expected = polys[axis].deriv()(float(end)) * face_tensor
                            poly_traced = face_trace_normal_gradient(poly_dofs, n, dim, axis, end)[0]
                            np.testing.assert_allclose(poly_traced, expected, rtol=0.0, atol=1e-12)

    def test_face_trace_ordering_matches_side_machinery(self):
        # THE critical stage-1 test (CPU-only by design): the encoded
        # side-to-element mapping must reproduce warp.fem's own inner/outer
        # side evaluations on every side of a Grid2D and a Grid3D.
        for dim in _DIMS:
            with self.subTest(dim=dim):
                self._check_side_trace_ordering(dim)

    def _check_side_trace_ordering(self, dim):
        device = "cpu"
        degree = 3
        n = degree + 1

        if dim == 2:
            res = (3, 2)
            bounds_lo, bounds_hi = (0.0, 0.0), (1.0, 1.5)
            geo = fem.Grid2D(
                res=wp.vec2i(*res),
                bounds_lo=wp.vec2d(*bounds_lo),
                bounds_hi=wp.vec2d(*bounds_hi),
                scalar_type=wp.float64,
            )
            probe, vec_type = _side_probe_2d, wp.vec2d
        else:
            # Anisotropic resolution: distinct res per axis (so a res-axis
            # mix-up cannot pass) and interior altitudes >= 2 on two axes.
            res = (3, 2, 4)
            bounds_lo, bounds_hi = (0.0, 0.0, 0.0), (1.0, 1.5, 0.5)
            geo = fem.Grid3D(
                res=wp.vec3i(*res),
                bounds_lo=wp.vec3d(*bounds_lo),
                bounds_hi=wp.vec3d(*bounds_hi),
                scalar_type=wp.float64,
            )
            probe, vec_type = _side_probe_3d, wp.vec3d

        space = fem.make_polynomial_space(geo, degree=degree, discontinuous=True, dtype=wp.float64)
        # The value-trace-is-slice property holds only for endpoint-including
        # (closed) node families; fail loudly if the default ever changes.
        self.assertTrue(is_closed(space.basis.shape.family))

        rng = np.random.default_rng(20260612 + dim)
        dofs_flat = rng.uniform(-1.0, 1.0, size=space.node_count())
        field = space.make_field()
        field.dof_values = wp.array(dofs_flat, dtype=wp.float64, device=device)
        # Discontinuous topology: global node index = cell * nodes_per_element
        # + lexicographic node index in element.
        dofs = dofs_flat.reshape(geo.cell_count(), n**dim)

        sides = fem.Sides(geometry=geo)
        quad = fem.RegularQuadrature(sides, order=2 * degree)
        total = quad.total_point_count()
        nqp = quad.max_points_per_element()
        q_1d = round(nqp ** (1.0 / (dim - 1)))
        self.assertEqual(q_1d ** (dim - 1), nqp)

        arrays = {
            "pos": wp.zeros(total, dtype=vec_type, device=device),
            "nrm": wp.zeros(total, dtype=vec_type, device=device),
            "inner_val": wp.zeros(total, dtype=wp.float64, device=device),
            "outer_val": wp.zeros(total, dtype=wp.float64, device=device),
            "inner_cell": wp.zeros(total, dtype=int, device=device),
            "outer_cell": wp.zeros(total, dtype=int, device=device),
            "inner_coords": wp.zeros(total, dtype=wp.vec3d, device=device),
            "outer_coords": wp.zeros(total, dtype=wp.vec3d, device=device),
            "side_coords": wp.zeros(total, dtype=wp.vec3d, device=device),
        }
        fem.interpolate(probe, at=quad, fields={"u": field.trace()}, values=arrays, device=device)
        observed = {key: arr.numpy() for key, arr in arrays.items()}

        nodes = default_basis_nodes(degree)
        cell_size = np.array([(bounds_hi[d] - bounds_lo[d]) / res[d] for d in range(dim)])
        outer_perm = grid_outer_face_permutation(q_1d, dim)

        def contract_to_side_qps(face_vals, face_coords):
            """Interpolate face-frame nodal values at the mapped side QP coordinates."""
            mat_0 = build_interpolation_matrix(nodes, face_coords[:, 0])
            if dim == 2:
                return mat_0 @ face_vals
            mat_1 = build_interpolation_matrix(nodes, face_coords[:, 1])
            return np.einsum("ru,rv,uv->r", mat_0, mat_1, face_vals.reshape(n, n))

        interior_count = 0
        for side in range(sides.element_count()):
            qps = slice(side * nqp, (side + 1) * nqp)
            normal = observed["nrm"][qps][0]
            axis = int(np.argmax(np.abs(normal)))
            altitude = int(round((observed["pos"][qps][0][axis] - bounds_lo[axis]) / cell_size[axis]))
            inner_cell = int(observed["inner_cell"][qps][0])
            outer_cell = int(observed["outer_cell"][qps][0])
            interior = inner_cell != outer_cell
            interior_count += interior

            with self.subTest(dim=dim, side=side, axis=axis, altitude=altitude, interior=interior):
                if interior:
                    # The side normal points from the inner to the outer cell (+axis).
                    self.assertGreater(normal[axis], 0.0)
                else:
                    # Boundary normals point outward: -axis at altitude 0, +axis at res.
                    self.assertGreater(normal[axis] * (1.0 if altitude == res[axis] else -1.0), 0.0)

                inner_end, outer_end = grid_side_face_ends(altitude, res[axis])
                side_qp = observed["side_coords"][qps][:, : dim - 1]
                face_coords = grid_side_to_face_coords(dim, axis, altitude, side_qp)
                f_axes = list(face_axes(dim, axis))

                # Coordinate-level lock: the mapped face coordinates reproduce the
                # element coordinates warp.fem assigns to each side QP.
                for name, end in (("inner_coords", inner_end), ("outer_coords", outer_end)):
                    coords = observed[name][qps]
                    np.testing.assert_allclose(coords[:, axis], float(end), rtol=0.0, atol=1e-12)
                    np.testing.assert_allclose(coords[:, f_axes], face_coords, rtol=0.0, atol=1e-12)

                # Value-level lock: GLL DOF slice + (d-1)-dim 1D interpolation at
                # the mapped coordinates == fem.inner / fem.outer.
                inner_face = face_trace_value(dofs[inner_cell][None, :], n, dim, axis, inner_end)[0]
                predicted_inner = contract_to_side_qps(inner_face, face_coords)
                np.testing.assert_allclose(observed["inner_val"][qps], predicted_inner, rtol=0.0, atol=1e-12)

                outer_face = face_trace_value(dofs[outer_cell][None, :], n, dim, axis, outer_end)[0]
                # Empirical finding: the outer trace needs no flip or permutation
                # relative to the inner trace (identity permutation).
                predicted_outer = contract_to_side_qps(outer_face, face_coords)[outer_perm]
                np.testing.assert_allclose(observed["outer_val"][qps], predicted_outer, rtol=0.0, atol=1e-12)

        # Make sure the shared-face (jump/average) case was actually exercised.
        expected_interior = sum((res[a] - 1) * np.prod([res[d] for d in range(dim) if d != a]) for a in range(dim))
        self.assertEqual(interior_count, expected_interior)

    def test_face_slice_indices_basic(self):
        # 2D, n=2: lexicographic nodes (i, j) -> index 2 * i + j.
        np.testing.assert_array_equal(face_slice_indices(2, 2, 0, 0), [0, 1])  # i = 0
        np.testing.assert_array_equal(face_slice_indices(2, 2, 0, 1), [2, 3])  # i = 1
        np.testing.assert_array_equal(face_slice_indices(2, 2, 1, 0), [0, 2])  # j = 0
        np.testing.assert_array_equal(face_slice_indices(2, 2, 1, 1), [1, 3])  # j = 1
        # 3D, n=2, axis=1: face frame is (axis 0 slow, axis 2 fast).
        np.testing.assert_array_equal(face_slice_indices(2, 3, 1, 0), [0, 1, 4, 5])
        np.testing.assert_array_equal(face_slice_indices(2, 3, 1, 1), [2, 3, 6, 7])


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
