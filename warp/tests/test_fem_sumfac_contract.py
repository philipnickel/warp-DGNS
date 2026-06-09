# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Tests for sum-factorized tensor-product interpolation B (sum-factorized DG, Phase 1b).

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
  ``E_b = 1`` path element by element (sub-step 4, P=1..5).

The contraction runs in Warp tile kernels (``block_dim = 1`` on CPU) so the
tests run on every device returned by ``get_test_devices()``.
"""

import unittest

import numpy as np

from warp._src.fem.sumfac.operators_1d import (
    default_quadrature_points,
)
from warp._src.fem.sumfac.tensor_contract import (
    build_operator_arrays,
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
add_function_test(TestFemSumfacContract, "test_pack_dofs_layout", test_pack_dofs_layout, devices=None)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
