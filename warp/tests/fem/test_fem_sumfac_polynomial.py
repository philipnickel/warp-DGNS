# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Tests for arbitrary-P 1D quadrature generation (sum-factorized DG, Phase 0).

These tests pin down three properties of the Gauss--Legendre (GL) and
Lobatto--Gauss--Legendre (GLL) quadrature generators in
:mod:`warp._src.fem.polynomial`:

1. Backward compatibility: the generalized generators reproduce the legacy
   hardcoded tables (GL ``n=1..5``, GLL ``n=2..5``) to ``atol=1e-13``. The rules
   are compared as coordinate-sorted ``(coord, weight)`` pairs because a
   quadrature rule is defined by its set of point/weight pairs, not by the order
   in which they are listed (the legacy tables used an inconsistent
   center-first/outward-pairs ordering).
2. Polynomial exactness on the ``[0, 1]`` interval: GL with ``n`` points
   integrates ``x^k`` exactly for ``k <= 2n - 1``; GLL for ``k <= 2n - 3``.
3. The caps are removed: ``fem.make_polynomial_space(..., discontinuous=True)``
   now constructs for degrees ``5, 6, 7, 8`` (previously raised at degree 5).
"""

import unittest

import numpy as np

import warp as wp
import warp.fem as fem
from warp._src.fem.polynomial import (
    Polynomial,
    _gauss_legendre_quadrature_1d,
    _lobatto_gauss_legendre_quadrature_1d,
    quadrature_1d,
)
from warp.tests.unittest_utils import *

# Legacy hardcoded reference tables on [0, 1], captured from the pre-arbitrary-P
# implementation. Stored as coordinate-sorted (coord, weight) pairs so the
# reproduction test is independent of listing order.
_LEGACY_GL = {
    1: (
        [0.5],
        [1.0],
    ),
    2: (
        [0.21132486540518713, 0.7886751345948129],
        [0.5, 0.5],
    ),
    3: (
        [0.1127016653792583, 0.5, 0.8872983346207417],
        [0.2777777777777778, 0.4444444444444444, 0.2777777777777778],
    ),
    4: (
        [0.06943184420297371, 0.33000947820757187, 0.6699905217924281, 0.9305681557970262],
        [0.17392742256872692, 0.3260725774312731, 0.3260725774312731, 0.17392742256872692],
    ),
    5: (
        [0.04691007703066802, 0.2307653449471585, 0.5, 0.7692346550528415, 0.9530899229693319],
        [
            0.11846344252809454,
            0.23931433524968324,
            0.28444444444444444,
            0.23931433524968324,
            0.11846344252809454,
        ],
    ),
}

_LEGACY_GLL = {
    2: (
        [0.0, 1.0],
        [0.5, 0.5],
    ),
    3: (
        [0.0, 0.5, 1.0],
        [0.16666666666666666, 0.6666666666666666, 0.16666666666666666],
    ),
    4: (
        [0.0, 0.27639320225002106, 0.7236067977499789, 1.0],
        [0.08333333333333333, 0.4166666666666667, 0.4166666666666667, 0.08333333333333333],
    ),
    5: (
        [0.0, 0.17267316464601146, 0.5, 0.8273268353539885, 1.0],
        [0.05, 0.2722222222222222, 0.35555555555555557, 0.2722222222222222, 0.05],
    ),
}


def _sorted_rule(coords, weights):
    """Return ``(coords, weights)`` sorted by ascending coordinate."""
    coords = np.asarray(coords, dtype=float)
    weights = np.asarray(weights, dtype=float)
    order = np.argsort(coords)
    return coords[order], weights[order]


def test_gauss_legendre_reproduces_legacy_table(test, device):
    for n, (ref_coords, ref_weights) in _LEGACY_GL.items():
        coords, weights = _gauss_legendre_quadrature_1d(n)
        coords, weights = _sorted_rule(coords, weights)
        np.testing.assert_allclose(coords, ref_coords, atol=1e-13, err_msg=f"GL coords n={n}")
        np.testing.assert_allclose(weights, ref_weights, atol=1e-13, err_msg=f"GL weights n={n}")


def test_lobatto_gauss_legendre_reproduces_legacy_table(test, device):
    for n, (ref_coords, ref_weights) in _LEGACY_GLL.items():
        coords, weights = _lobatto_gauss_legendre_quadrature_1d(n)
        coords, weights = _sorted_rule(coords, weights)
        np.testing.assert_allclose(coords, ref_coords, atol=1e-13, err_msg=f"GLL coords n={n}")
        np.testing.assert_allclose(weights, ref_weights, atol=1e-13, err_msg=f"GLL weights n={n}")


def test_gauss_legendre_polynomial_exactness(test, device):
    # GL with n points integrates polynomials up to degree 2n - 1 exactly on [0, 1].
    for n in range(1, 9):
        coords, weights = _gauss_legendre_quadrature_1d(n)
        coords = np.asarray(coords, dtype=float)
        weights = np.asarray(weights, dtype=float)

        test.assertEqual(len(coords), n)
        np.testing.assert_allclose(np.sum(weights), 1.0, atol=1e-13, err_msg=f"GL weight sum n={n}")

        for k in range(2 * n):  # k = 0 .. 2n - 1
            integral = np.sum(weights * coords**k)
            reference = 1.0 / (k + 1)
            np.testing.assert_allclose(integral, reference, atol=1e-12, err_msg=f"GL n={n} failed to integrate x^{k}")


def test_lobatto_gauss_legendre_polynomial_exactness(test, device):
    # GLL with n points integrates polynomials up to degree 2n - 3 exactly on [0, 1].
    for n in range(2, 9):
        coords, weights = _lobatto_gauss_legendre_quadrature_1d(n)
        coords = np.asarray(coords, dtype=float)
        weights = np.asarray(weights, dtype=float)

        test.assertEqual(len(coords), n)
        np.testing.assert_allclose(np.sum(weights), 1.0, atol=1e-13, err_msg=f"GLL weight sum n={n}")

        # GLL includes the interval endpoints.
        np.testing.assert_allclose(np.min(coords), 0.0, atol=1e-13, err_msg=f"GLL min n={n}")
        np.testing.assert_allclose(np.max(coords), 1.0, atol=1e-13, err_msg=f"GLL max n={n}")

        for k in range(2 * n - 2):  # k = 0 .. 2n - 3
            integral = np.sum(weights * coords**k)
            reference = 1.0 / (k + 1)
            np.testing.assert_allclose(integral, reference, atol=1e-12, err_msg=f"GLL n={n} failed to integrate x^{k}")


def test_quadrature_1d_dispatch_high_order(test, device):
    # The public dispatcher must also support arbitrary point counts.
    for n in range(6, 9):
        gl_coords, gl_weights = quadrature_1d(point_count=n, family=Polynomial.GAUSS_LEGENDRE)
        test.assertEqual(len(gl_coords), n)
        np.testing.assert_allclose(np.sum(gl_weights), 1.0, atol=1e-13)

        gll_coords, gll_weights = quadrature_1d(point_count=n, family=Polynomial.LOBATTO_GAUSS_LEGENDRE)
        test.assertEqual(len(gll_coords), n)
        np.testing.assert_allclose(np.sum(gll_weights), 1.0, atol=1e-13)


def test_high_order_polynomial_space_constructs(test, device):
    # Previously make_polynomial_space raised NotImplementedError at degree 5.
    geo = fem.Grid3D(res=wp.vec3i(1, 1, 1))
    for degree in (5, 6, 7, 8):
        space = fem.make_polynomial_space(geo, degree=degree, discontinuous=True)
        expected_nodes = (degree + 1) ** 3
        test.assertEqual(space.node_count(), expected_nodes)


# -- Device setup and test registration --


class TestFemSumfacPolynomial(unittest.TestCase):
    pass


add_function_test(
    TestFemSumfacPolynomial,
    "test_gauss_legendre_reproduces_legacy_table",
    test_gauss_legendre_reproduces_legacy_table,
    devices=None,
)
add_function_test(
    TestFemSumfacPolynomial,
    "test_lobatto_gauss_legendre_reproduces_legacy_table",
    test_lobatto_gauss_legendre_reproduces_legacy_table,
    devices=None,
)
add_function_test(
    TestFemSumfacPolynomial,
    "test_gauss_legendre_polynomial_exactness",
    test_gauss_legendre_polynomial_exactness,
    devices=None,
)
add_function_test(
    TestFemSumfacPolynomial,
    "test_lobatto_gauss_legendre_polynomial_exactness",
    test_lobatto_gauss_legendre_polynomial_exactness,
    devices=None,
)
add_function_test(
    TestFemSumfacPolynomial,
    "test_quadrature_1d_dispatch_high_order",
    test_quadrature_1d_dispatch_high_order,
    devices=None,
)
add_function_test(
    TestFemSumfacPolynomial,
    "test_high_order_polynomial_space_constructs",
    test_high_order_polynomial_space_constructs,
    devices=get_test_devices(),
)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
