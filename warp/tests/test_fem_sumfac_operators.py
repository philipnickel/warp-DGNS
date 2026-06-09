# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Tests for 1D sum-factorization operator matrices (sum-factorized DG, Phase 1a).

These tests pin down the barycentric-Lagrange interpolation matrix ``I`` and
derivative matrix ``D_ref`` built by :mod:`warp._src.fem.sumfac.operators_1d`:

* Partition of unity: every row of ``I`` sums to ``1`` (the Lagrange basis is a
  partition of unity, so interpolating the constant function reproduces it).
* Polynomial reproduction: for a degree-``P`` basis (``P + 1`` nodes), ``I``
  reproduces the values at the evaluation points of any polynomial of degree
  ``<= P`` exactly, and ``D_ref`` reproduces that polynomial's derivative.
* Collocation: when the evaluation points coincide with the basis nodes, ``I``
  is the identity matrix.

All computation is NumPy on the CPU at setup time (no Warp kernels), so the
tests run without a device.
"""

import unittest

import numpy as np

from warp._src.fem.sumfac.operators_1d import (
    build_derivative_matrix,
    build_interpolation_matrix,
    default_basis_nodes,
    default_quadrature_points,
)
from warp.tests.unittest_utils import *

# Force Warp runtime initialization at import time so its one-time startup banner
# is flushed before any test runs (otherwise ``CheckOutput`` flags it as
# unexpected stdout). The matrices themselves are pure NumPy and need no device.
get_test_devices()


def test_interpolation_partition_of_unity(test, device):
    # Rows of I must sum to 1 for any node set and any evaluation points.
    for degree in range(1, 9):
        nodes = default_basis_nodes(degree)
        points = default_quadrature_points(degree)
        interp = build_interpolation_matrix(nodes, points)

        test.assertEqual(interp.shape, (len(points), len(nodes)))
        row_sums = np.sum(interp, axis=1)
        np.testing.assert_allclose(
            row_sums, np.ones(len(points)), atol=1e-12, err_msg=f"partition of unity failed for degree={degree}"
        )


def test_interpolation_reproduces_polynomial(test, device):
    # An n-node Lagrange basis reproduces every polynomial of degree < n exactly.
    rng = np.random.default_rng(0)
    for degree in range(1, 9):
        nodes = default_basis_nodes(degree)
        points = default_quadrature_points(degree)
        interp = build_interpolation_matrix(nodes, points)

        coeffs = rng.standard_normal(degree + 1)  # polynomial of degree `degree`
        poly = np.polynomial.Polynomial(coeffs)

        nodal_values = poly(nodes)
        interpolated = interp @ nodal_values
        expected = poly(points)

        np.testing.assert_allclose(
            interpolated, expected, atol=1e-10, err_msg=f"interpolation reproduction failed for degree={degree}"
        )


def test_derivative_reproduces_polynomial_derivative(test, device):
    # D_ref applied to nodal values reproduces the polynomial's derivative at points.
    rng = np.random.default_rng(1)
    for degree in range(1, 9):
        nodes = default_basis_nodes(degree)
        points = default_quadrature_points(degree)
        deriv = build_derivative_matrix(nodes, points)

        test.assertEqual(deriv.shape, (len(points), len(nodes)))

        coeffs = rng.standard_normal(degree + 1)  # polynomial of degree `degree`
        poly = np.polynomial.Polynomial(coeffs)
        dpoly = poly.deriv()

        nodal_values = poly(nodes)
        differentiated = deriv @ nodal_values
        expected = dpoly(points)

        np.testing.assert_allclose(
            differentiated, expected, atol=1e-10, err_msg=f"derivative reproduction failed for degree={degree}"
        )


def test_collocation_is_identity(test, device):
    # When evaluation points coincide with the basis nodes, I is the identity.
    for degree in range(1, 9):
        nodes = default_basis_nodes(degree)
        interp = build_interpolation_matrix(nodes, nodes)

        np.testing.assert_allclose(
            interp, np.eye(len(nodes)), atol=1e-12, err_msg=f"collocation identity failed for degree={degree}"
        )


def test_derivative_constant_is_zero(test, device):
    # The derivative of the constant function (all-ones nodal values) is zero,
    # equivalently every row of D_ref sums to zero.
    for degree in range(1, 9):
        nodes = default_basis_nodes(degree)
        points = default_quadrature_points(degree)
        deriv = build_derivative_matrix(nodes, points)

        row_sums = np.sum(deriv, axis=1)
        np.testing.assert_allclose(
            row_sums, np.zeros(len(points)), atol=1e-10, err_msg=f"derivative-of-constant failed for degree={degree}"
        )


# -- Device setup and test registration --


class TestFemSumfacOperators(unittest.TestCase):
    pass


add_function_test(
    TestFemSumfacOperators,
    "test_interpolation_partition_of_unity",
    test_interpolation_partition_of_unity,
    devices=None,
)
add_function_test(
    TestFemSumfacOperators,
    "test_interpolation_reproduces_polynomial",
    test_interpolation_reproduces_polynomial,
    devices=None,
)
add_function_test(
    TestFemSumfacOperators,
    "test_derivative_reproduces_polynomial_derivative",
    test_derivative_reproduces_polynomial_derivative,
    devices=None,
)
add_function_test(
    TestFemSumfacOperators,
    "test_collocation_is_identity",
    test_collocation_is_identity,
    devices=None,
)
add_function_test(
    TestFemSumfacOperators,
    "test_derivative_constant_is_zero",
    test_derivative_constant_is_zero,
    devices=None,
)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
