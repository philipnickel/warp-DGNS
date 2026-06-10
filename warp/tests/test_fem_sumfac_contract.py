# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Tests for sum-factorized tensor-product contractions B and B^T (sum-factorized DG, Phase 1b).

These tests pin down the E_b-wide ``tile_matmul`` contraction kernels in
:mod:`warp._src.fem.sumfac.tensor_contract`, which map element nodal DOFs to the
value and reference gradient at quadrature points for quad (2D) and hex (3D)
tensor-product elements. The golden reference is the dense Kronecker-product
operator ``Kron(...) @ vec(dofs)`` built with NumPy:

* 2D value reproduces ``Kron(I, I) @ vec(U)`` (sub-step 1, P=1..5).
* 2D reference gradient reproduces ``Kron(D_ref, I)`` and ``Kron(I, D_ref)``
  (sub-step 2, P=1..5).
* 3D value and reference gradient reproduce the three-factor Kronecker products
  (sub-step 3, P=1..4).
* Batching ``E_b = 4`` elements into one wider RHS tile matches looping the
  ``E_b = 1`` path element by element (sub-step 4, P=1..5, 2D and 3D).
* The transpose contraction ``B^T`` reproduces ``g @ Kron(...)`` for the value
  operator and every gradient operator, in 2D and 3D, including ``E_b = 4``
  batching, and the residual accumulation driver matches the dense
  ``(1 + dim)``-term sum.
* The full ``B^T D B`` mass pipeline (``D`` = diagonal tensor-product GL
  weights) matches the dense Kronecker-built mass-matrix action.
* Over-integration (``q != n``) with rectangular ``(q, n)`` operators matches
  the dense references in 2D and 3D, forward and transpose, so the kernel
  factories stay general in ``(rows_in, rows_out)``.

The contraction runs in Warp tile kernels (``block_dim = 1`` on CPU) so the
tests run on every device returned by ``get_test_devices()``.
"""

import unittest

import numpy as np

from warp._src.fem.polynomial import Polynomial, quadrature_1d
from warp._src.fem.sumfac.operators_1d import (
    build_derivative_matrix,
    build_interpolation_matrix,
    default_basis_nodes,
    default_quadrature_points,
)
from warp._src.fem.sumfac.tensor_contract import (
    build_operator_arrays,
    contract_transpose_2d,
    contract_transpose_3d,
    interpolate_2d,
    interpolate_3d,
    pack_dofs_2d,
    pack_dofs_3d,
)
from warp.tests.unittest_utils import *

# Force Warp runtime initialization at import time so its one-time startup banner
# is flushed before any test runs (otherwise ``CheckOutput`` flags it as
# unexpected stdout).
get_test_devices()

# Tolerances. The contraction kernels run in float64, so the error is at the
# round-off level; keep a comfortable margin for the higher orders.
ATOL_F64 = 1.0e-10


def _dense_kron_2d(left, right):
    """Dense 2D operator ``Kron(left, right)`` acting on row-major ``vec(U)``."""
    return np.kron(left, right)


def _dense_kron_3d(a, b, c):
    """Dense 3D operator ``Kron(a, b, c)`` acting on row-major ``vec(U)``."""
    return np.kron(np.kron(a, b), c)


def contract_transpose_residual_2d(f0, f1_xi, f1_eta, interp, deriv, n, q, element_batch=1, device=None):
    """Accumulate the 2D ``B^T`` residual ``r = ValueOp^T f0 + sum_axis GradOp_axis^T f1_axis``.

    Test-local composition of :func:`contract_transpose_2d` (one launch per
    operator, summed at the NumPy level); the production apply path fuses this
    into a single kernel in :mod:`warp._src.fem.sumfac.kernels`.
    """
    residual = contract_transpose_2d(f0, interp, interp, n, q, element_batch, device)
    residual += contract_transpose_2d(f1_xi, deriv, interp, n, q, element_batch, device)
    residual += contract_transpose_2d(f1_eta, interp, deriv, n, q, element_batch, device)
    return residual


def contract_transpose_residual_3d(f0, f1_xi, f1_eta, f1_zeta, interp, deriv, n, q, element_batch=1, device=None):
    """3D counterpart of :func:`contract_transpose_residual_2d` (``1 + dim = 4`` contributions)."""
    residual = contract_transpose_3d(f0, interp, interp, interp, n, q, element_batch, device)
    residual += contract_transpose_3d(f1_xi, deriv, interp, interp, n, q, element_batch, device)
    residual += contract_transpose_3d(f1_eta, interp, deriv, interp, n, q, element_batch, device)
    residual += contract_transpose_3d(f1_zeta, interp, interp, deriv, n, q, element_batch, device)
    return residual


def test_interpolate_2d_value(test, device):
    # Sub-step 1: 2D value interpolation vs dense Kron(I, I), E_b = 1.
    rng = np.random.default_rng(0)
    for degree in range(1, 6):
        interp, _ = build_operator_arrays(degree)
        n = degree + 1
        q = len(default_quadrature_points(degree))

        num_elements = 3
        dofs = rng.standard_normal((num_elements, n * n))

        result = interpolate_2d(dofs, interp, interp, n, q, element_batch=1, device=device)

        op = _dense_kron_2d(interp, interp)  # (q*q, n*n)
        expected = dofs @ op.T  # (num_elements, q*q)

        np.testing.assert_allclose(
            result, expected, atol=ATOL_F64, err_msg=f"2D value interpolation failed for degree={degree}"
        )


def test_interpolate_2d_gradient(test, device):
    # Sub-step 2: 2D reference gradient (d/dxi, d/deta) vs dense Kron(D, I) / Kron(I, D).
    rng = np.random.default_rng(1)
    for degree in range(1, 6):
        interp, deriv = build_operator_arrays(degree)
        n = degree + 1
        q = len(default_quadrature_points(degree))

        num_elements = 3
        dofs = rng.standard_normal((num_elements, n * n))

        grad_xi = interpolate_2d(dofs, deriv, interp, n, q, element_batch=1, device=device)
        grad_eta = interpolate_2d(dofs, interp, deriv, n, q, element_batch=1, device=device)

        expected_xi = dofs @ _dense_kron_2d(deriv, interp).T
        expected_eta = dofs @ _dense_kron_2d(interp, deriv).T

        np.testing.assert_allclose(
            grad_xi, expected_xi, atol=ATOL_F64, err_msg=f"2D d/dxi gradient failed for degree={degree}"
        )
        np.testing.assert_allclose(
            grad_eta, expected_eta, atol=ATOL_F64, err_msg=f"2D d/deta gradient failed for degree={degree}"
        )


def test_interpolate_3d_value_and_gradient(test, device):
    # Sub-step 3: 3D value and reference gradient vs dense three-factor Kron products.
    rng = np.random.default_rng(2)
    for degree in range(1, 5):
        interp, deriv = build_operator_arrays(degree)
        n = degree + 1
        q = len(default_quadrature_points(degree))

        num_elements = 2
        dofs = rng.standard_normal((num_elements, n * n * n))

        value = interpolate_3d(dofs, interp, interp, interp, n, q, element_batch=1, device=device)
        grad_xi = interpolate_3d(dofs, deriv, interp, interp, n, q, element_batch=1, device=device)
        grad_eta = interpolate_3d(dofs, interp, deriv, interp, n, q, element_batch=1, device=device)
        grad_zeta = interpolate_3d(dofs, interp, interp, deriv, n, q, element_batch=1, device=device)

        expected_value = dofs @ _dense_kron_3d(interp, interp, interp).T
        expected_xi = dofs @ _dense_kron_3d(deriv, interp, interp).T
        expected_eta = dofs @ _dense_kron_3d(interp, deriv, interp).T
        expected_zeta = dofs @ _dense_kron_3d(interp, interp, deriv).T

        np.testing.assert_allclose(
            value, expected_value, atol=ATOL_F64, err_msg=f"3D value interpolation failed for degree={degree}"
        )
        np.testing.assert_allclose(
            grad_xi, expected_xi, atol=ATOL_F64, err_msg=f"3D d/dxi gradient failed for degree={degree}"
        )
        np.testing.assert_allclose(
            grad_eta, expected_eta, atol=ATOL_F64, err_msg=f"3D d/deta gradient failed for degree={degree}"
        )
        np.testing.assert_allclose(
            grad_zeta, expected_zeta, atol=ATOL_F64, err_msg=f"3D d/dzeta gradient failed for degree={degree}"
        )


def test_interpolate_2d_batched(test, device):
    # Sub-step 4: E_b = 4 batching matches looping the E_b = 1 path element by element.
    rng = np.random.default_rng(3)
    element_batch = 4
    for degree in range(1, 6):
        interp, _ = build_operator_arrays(degree)
        n = degree + 1
        q = len(default_quadrature_points(degree))

        # Use a multiple of element_batch so the panel split is exact.
        num_elements = 2 * element_batch
        dofs = rng.standard_normal((num_elements, n * n))

        batched = interpolate_2d(dofs, interp, interp, n, q, element_batch=element_batch, device=device)
        looped = interpolate_2d(dofs, interp, interp, n, q, element_batch=1, device=device)

        np.testing.assert_allclose(
            batched, looped, atol=ATOL_F64, err_msg=f"2D batched (E_b=4) vs looped mismatch for degree={degree}"
        )


def test_interpolate_3d_batched(test, device):
    # 3D analog of sub-step 4: E_b = 4 batching matches the E_b = 1 path.
    rng = np.random.default_rng(4)
    element_batch = 4
    for degree in (1, 2):
        interp, _ = build_operator_arrays(degree)
        n = degree + 1
        q = len(default_quadrature_points(degree))

        num_elements = 2 * element_batch
        dofs = rng.standard_normal((num_elements, n * n * n))

        batched = interpolate_3d(dofs, interp, interp, interp, n, q, element_batch=element_batch, device=device)
        looped = interpolate_3d(dofs, interp, interp, interp, n, q, element_batch=1, device=device)

        np.testing.assert_allclose(
            batched, looped, atol=ATOL_F64, err_msg=f"3D batched (E_b=4) vs looped mismatch for degree={degree}"
        )


# -- Backward (B^T) contraction tests -----------------------------------------


def test_backward_contract_is_transpose_2d(test, device):
    # B^T stage: per-QP coefficients g (E, q*q) -> residual (E, n*n); the result
    # must equal the dense transpose action g @ Kron(A, B) for the value operator
    # and both gradient operators.
    rng = np.random.default_rng(5)
    for degree in (1, 3, 5):
        interp, deriv = build_operator_arrays(degree)
        n = degree + 1
        q = len(default_quadrature_points(degree))

        num_elements = 3
        g = rng.standard_normal((num_elements, q * q))

        for a_mat, b_mat, label in ((interp, interp, "value"), (deriv, interp, "d/dxi"), (interp, deriv, "d/deta")):
            result = contract_transpose_2d(g, a_mat, b_mat, n, q, element_batch=1, device=device)
            expected = g @ _dense_kron_2d(a_mat, b_mat)  # (E, n*n)
            np.testing.assert_allclose(
                result,
                expected,
                atol=ATOL_F64,
                err_msg=f"2D transpose contraction ({label}) failed for degree={degree}",
            )

        # Residual accumulation driver: ValueOp^T f0 + sum_axis GradOp_axis^T f1_axis.
        f0 = rng.standard_normal((num_elements, q * q))
        f1_xi = rng.standard_normal((num_elements, q * q))
        f1_eta = rng.standard_normal((num_elements, q * q))
        residual = contract_transpose_residual_2d(
            f0, f1_xi, f1_eta, interp, deriv, n, q, element_batch=1, device=device
        )
        expected_residual = (
            f0 @ _dense_kron_2d(interp, interp)
            + f1_xi @ _dense_kron_2d(deriv, interp)
            + f1_eta @ _dense_kron_2d(interp, deriv)
        )
        np.testing.assert_allclose(
            residual, expected_residual, atol=ATOL_F64, err_msg=f"2D residual accumulation failed for degree={degree}"
        )


def test_backward_contract_is_transpose_3d(test, device):
    # 3D B^T stage vs the dense three-factor Kronecker transpose action.
    rng = np.random.default_rng(6)
    for degree in (1, 2, 4):
        interp, deriv = build_operator_arrays(degree)
        n = degree + 1
        q = len(default_quadrature_points(degree))

        num_elements = 2
        g = rng.standard_normal((num_elements, q * q * q))

        cases = (
            (interp, interp, interp, "value"),
            (deriv, interp, interp, "d/dxi"),
            (interp, deriv, interp, "d/deta"),
            (interp, interp, deriv, "d/dzeta"),
        )
        for a_mat, b_mat, c_mat, label in cases:
            result = contract_transpose_3d(g, a_mat, b_mat, c_mat, n, q, element_batch=1, device=device)
            expected = g @ _dense_kron_3d(a_mat, b_mat, c_mat)  # (E, n^3)
            np.testing.assert_allclose(
                result,
                expected,
                atol=ATOL_F64,
                err_msg=f"3D transpose contraction ({label}) failed for degree={degree}",
            )

        # Residual accumulation driver, 3D: 1 + dim = 4 contractions.
        f0 = rng.standard_normal((num_elements, q * q * q))
        f1_xi = rng.standard_normal((num_elements, q * q * q))
        f1_eta = rng.standard_normal((num_elements, q * q * q))
        f1_zeta = rng.standard_normal((num_elements, q * q * q))
        residual = contract_transpose_residual_3d(
            f0, f1_xi, f1_eta, f1_zeta, interp, deriv, n, q, element_batch=1, device=device
        )
        expected_residual = (
            f0 @ _dense_kron_3d(interp, interp, interp)
            + f1_xi @ _dense_kron_3d(deriv, interp, interp)
            + f1_eta @ _dense_kron_3d(interp, deriv, interp)
            + f1_zeta @ _dense_kron_3d(interp, interp, deriv)
        )
        np.testing.assert_allclose(
            residual, expected_residual, atol=ATOL_F64, err_msg=f"3D residual accumulation failed for degree={degree}"
        )


def test_backward_batched(test, device):
    # E_b = 4 transpose contraction matches the E_b = 1 path, 2D and 3D.
    rng = np.random.default_rng(7)
    element_batch = 4

    for degree in (1, 3):
        interp, deriv = build_operator_arrays(degree)
        n = degree + 1
        q = len(default_quadrature_points(degree))

        num_elements = 2 * element_batch
        g = rng.standard_normal((num_elements, q * q))

        batched = contract_transpose_2d(g, deriv, interp, n, q, element_batch=element_batch, device=device)
        looped = contract_transpose_2d(g, deriv, interp, n, q, element_batch=1, device=device)
        np.testing.assert_allclose(
            batched, looped, atol=ATOL_F64, err_msg=f"2D batched (E_b=4) transpose mismatch for degree={degree}"
        )

    for degree in (1, 2):
        interp, deriv = build_operator_arrays(degree)
        n = degree + 1
        q = len(default_quadrature_points(degree))

        num_elements = 2 * element_batch
        g = rng.standard_normal((num_elements, q * q * q))

        batched = contract_transpose_3d(g, interp, deriv, interp, n, q, element_batch=element_batch, device=device)
        looped = contract_transpose_3d(g, interp, deriv, interp, n, q, element_batch=1, device=device)
        np.testing.assert_allclose(
            batched, looped, atol=ATOL_F64, err_msg=f"3D batched (E_b=4) transpose mismatch for degree={degree}"
        )


def test_forward_backward_roundtrip_mass(test, device):
    # Full B^T D B pipeline shape for the mass operator: B^T (W * (B u)) must
    # equal the dense mass-matrix action with D = diag of tensor-product GL weights.
    rng = np.random.default_rng(8)
    degree = 3
    interp, _ = build_operator_arrays(degree)
    n = degree + 1
    q = len(default_quadrature_points(degree))

    _, w1d = quadrature_1d(point_count=degree + 1, family=Polynomial.GAUSS_LEGENDRE)
    w1d = np.asarray(w1d, dtype=float)
    w2d = np.kron(w1d, w1d)  # (q*q,) tensor-product weights, lexicographic (qx, qy)

    num_elements = 3
    u = rng.standard_normal((num_elements, n * n))

    bu = interpolate_2d(u, interp, interp, n, q, element_batch=1, device=device)  # (E, q*q)
    r = contract_transpose_2d(w2d[None, :] * bu, interp, interp, n, q, element_batch=1, device=device)

    b_dense = _dense_kron_2d(interp, interp)  # (q*q, n*n)
    mass = b_dense.T @ np.diag(w2d) @ b_dense  # (n*n, n*n)
    expected = u @ mass.T

    np.testing.assert_allclose(r, expected, atol=ATOL_F64, err_msg="B^T D B mass roundtrip failed")


def test_over_integration_rectangular(test, device):
    # q != n (over-integration): the kernel factories must be general in
    # (rows_in, rows_out). With the default quadrature q == n every compiled
    # tile shape is square, so a mix-up between the n- and q-sized constants
    # (or between the rows_in/rows_out arguments of the drivers) would be
    # invisible to the rest of the suite; this rectangular case pins it down.
    rng = np.random.default_rng(9)
    degree = 2
    n = degree + 1  # 3 basis nodes
    q = 5  # 5 Gauss--Legendre points: q != n

    nodes = default_basis_nodes(degree)
    points, _ = quadrature_1d(point_count=q, family=Polynomial.GAUSS_LEGENDRE)
    points = np.asarray(points, dtype=float)
    interp = build_interpolation_matrix(nodes, points)  # (q, n)
    deriv = build_derivative_matrix(nodes, points)  # (q, n)

    # 2D forward and backward, E_b = 1 and E_b = 2.
    for element_batch in (1, 2):
        num_elements = 2 * element_batch
        dofs = rng.standard_normal((num_elements, n * n))
        g = rng.standard_normal((num_elements, q * q))

        for a_mat, b_mat, label in ((interp, interp, "value"), (deriv, interp, "d/dxi"), (interp, deriv, "d/deta")):
            forward = interpolate_2d(dofs, a_mat, b_mat, n, q, element_batch=element_batch, device=device)
            np.testing.assert_allclose(
                forward,
                dofs @ _dense_kron_2d(a_mat, b_mat).T,
                atol=ATOL_F64,
                err_msg=f"2D forward q != n ({label}, E_b={element_batch}) failed",
            )

            backward = contract_transpose_2d(g, a_mat, b_mat, n, q, element_batch=element_batch, device=device)
            np.testing.assert_allclose(
                backward,
                g @ _dense_kron_2d(a_mat, b_mat),
                atol=ATOL_F64,
                err_msg=f"2D transpose q != n ({label}, E_b={element_batch}) failed",
            )

    # 3D forward and backward, E_b = 1 (exercises the nn/qn/qq tile-shape
    # constants of the 3D kernel, which all coincide when q == n).
    num_elements = 2
    dofs = rng.standard_normal((num_elements, n * n * n))
    g = rng.standard_normal((num_elements, q * q * q))

    cases = (
        (interp, interp, interp, "value"),
        (deriv, interp, interp, "d/dxi"),
        (interp, deriv, interp, "d/deta"),
        (interp, interp, deriv, "d/dzeta"),
    )
    for a_mat, b_mat, c_mat, label in cases:
        forward = interpolate_3d(dofs, a_mat, b_mat, c_mat, n, q, element_batch=1, device=device)
        np.testing.assert_allclose(
            forward,
            dofs @ _dense_kron_3d(a_mat, b_mat, c_mat).T,
            atol=ATOL_F64,
            err_msg=f"3D forward q != n ({label}) failed",
        )

        backward = contract_transpose_3d(g, a_mat, b_mat, c_mat, n, q, element_batch=1, device=device)
        np.testing.assert_allclose(
            backward,
            g @ _dense_kron_3d(a_mat, b_mat, c_mat),
            atol=ATOL_F64,
            err_msg=f"3D transpose q != n ({label}) failed",
        )


# -- Smoke check that the host packing helpers round-trip the layout --


def test_pack_dofs_layout(test, device):
    n = 3
    num_elements = 2
    dofs_2d = np.arange(num_elements * n * n, dtype=float).reshape(num_elements, n * n)
    packed_2d = pack_dofs_2d(dofs_2d, n)
    test.assertEqual(packed_2d.shape, (n, num_elements * n))
    for e in range(num_elements):
        block = packed_2d[:, e * n : (e + 1) * n]
        np.testing.assert_array_equal(block, dofs_2d[e].reshape(n, n))

    dofs_3d = np.arange(num_elements * n * n * n, dtype=float).reshape(num_elements, n * n * n)
    packed_3d = pack_dofs_3d(dofs_3d, n)
    test.assertEqual(packed_3d.shape, (n, num_elements * n * n))
    for e in range(num_elements):
        block = packed_3d[:, e * n * n : (e + 1) * n * n]
        np.testing.assert_array_equal(block, dofs_3d[e].reshape(n, n * n))


# -- Device setup and test registration --


class TestFemSumfacContract(unittest.TestCase):
    pass


devices = get_test_devices()

add_function_test(TestFemSumfacContract, "test_interpolate_2d_value", test_interpolate_2d_value, devices=devices)
add_function_test(TestFemSumfacContract, "test_interpolate_2d_gradient", test_interpolate_2d_gradient, devices=devices)
add_function_test(
    TestFemSumfacContract,
    "test_interpolate_3d_value_and_gradient",
    test_interpolate_3d_value_and_gradient,
    devices=devices,
)
add_function_test(TestFemSumfacContract, "test_interpolate_2d_batched", test_interpolate_2d_batched, devices=devices)
add_function_test(TestFemSumfacContract, "test_interpolate_3d_batched", test_interpolate_3d_batched, devices=devices)
add_function_test(
    TestFemSumfacContract,
    "test_backward_contract_is_transpose_2d",
    test_backward_contract_is_transpose_2d,
    devices=devices,
)
add_function_test(
    TestFemSumfacContract,
    "test_backward_contract_is_transpose_3d",
    test_backward_contract_is_transpose_3d,
    devices=devices,
)
add_function_test(TestFemSumfacContract, "test_backward_batched", test_backward_batched, devices=devices)
add_function_test(
    TestFemSumfacContract,
    "test_forward_backward_roundtrip_mass",
    test_forward_backward_roundtrip_mass,
    devices=devices,
)
add_function_test(
    TestFemSumfacContract,
    "test_over_integration_rectangular",
    test_over_integration_rectangular,
    devices=devices,
)
add_function_test(TestFemSumfacContract, "test_pack_dofs_layout", test_pack_dofs_layout, devices=None)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
