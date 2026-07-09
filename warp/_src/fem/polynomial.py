# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import Enum

import numpy as np

_wp_module_name_ = "warp.fem.polynomial"


class Polynomial(Enum):
    """Polynomial family defining interpolation nodes over an interval."""

    GAUSS_LEGENDRE = "GL"
    """Gauss--Legendre 1D polynomial family (does not include endpoints)."""

    LOBATTO_GAUSS_LEGENDRE = "LGL"
    """Lobatto--Gauss--Legendre 1D polynomial family (includes endpoints)."""

    EQUISPACED_CLOSED = "closed"
    """Closed 1D polynomial family with uniformly distributed nodes (includes endpoints)."""

    EQUISPACED_OPEN = "open"
    """Open 1D polynomial family with uniformly distributed nodes (does not include endpoints)."""

    def __str__(self):
        return self.value


def is_closed(family: Polynomial):
    """Whether the polynomial roots include interval endpoints."""
    return family == Polynomial.LOBATTO_GAUSS_LEGENDRE or family == Polynomial.EQUISPACED_CLOSED


def _gauss_legendre_quadrature_1d(n: int):
    """Gauss--Legendre quadrature with ``n`` points, shifted to the ``[0, 1]`` interval.

    The rule is exact for polynomials up to degree ``2n - 1``. Points are returned in
    ascending order and the weights sum to ``1`` (the measure of ``[0, 1]``).
    """
    if n < 1:
        raise ValueError(f"Gauss--Legendre quadrature requires at least one point (got n={n}).")

    # Roots of the degree-n Legendre polynomial on [-1, 1] with their weights.
    coords, weights = np.polynomial.legendre.leggauss(n)

    # Shift from [-1, 1] to [0, 1]; the weights pick up a factor of 1/2 from the change
    # of measure. ``leggauss`` already returns the roots in ascending order.
    weights = 0.5 * weights
    coords = 0.5 * coords + 0.5

    return coords, weights


def _lobatto_gauss_legendre_quadrature_1d(n: int):
    """Lobatto--Gauss--Legendre quadrature with ``n`` points, shifted to ``[0, 1]``.

    The rule includes the interval endpoints and is exact for polynomials up to degree
    ``2n - 3``. Points are returned in ascending order and the weights sum to ``1`` (the
    measure of ``[0, 1]``).
    """
    if n < 2:
        raise ValueError(f"Lobatto--Gauss--Legendre quadrature requires at least two points (got n={n}).")

    # Interior nodes are the roots of P'_{n-1}; together with the endpoints they are the
    # roots of (1 - x^2) P'_{n-1}(x). Solve for all nodes on [-1, 1] by Newton iteration,
    # initialized at the Chebyshev--Gauss--Lobatto points, then apply the Christoffel
    # weights w_i = 2 / (n (n-1) [P_{n-1}(x_i)]^2).
    coords = np.cos(np.pi * np.arange(n) / (n - 1))  # descending Chebyshev guess
    coords = np.sort(coords)  # ascending

    legendre = np.zeros((n, n))
    previous = 2.0 * np.ones(n)
    # Newton's method on the Legendre--Lobatto residual x P_{n-1} - P_{n-2}.
    while np.max(np.abs(coords - previous)) > 1e-15:
        previous = coords.copy()

        legendre[:, 0] = 1.0
        legendre[:, 1] = coords
        for k in range(2, n):
            legendre[:, k] = ((2 * k - 1) * coords * legendre[:, k - 1] - (k - 1) * legendre[:, k - 2]) / k

        coords = previous - (coords * legendre[:, n - 1] - legendre[:, n - 2]) / (n * legendre[:, n - 1])

    # Pin the endpoints exactly.
    coords[0] = -1.0
    coords[-1] = 1.0

    p_nm1 = legendre[:, n - 1]
    weights = 2.0 / ((n - 1) * n * p_nm1 * p_nm1)

    # Shift from [-1, 1] to [0, 1]; the weights pick up a factor of 1/2.
    weights = 0.5 * weights
    coords = 0.5 * coords + 0.5

    return coords, weights


def _uniform_open_quadrature_1d(n: int):
    step = 1.0 / (n + 1)
    coords = np.linspace(step, 1.0 - step, n)
    weights = np.full(n, 1.0 / (n + 1))

    # Boundaries have 3/2 the weight
    weights[0] = 1.5 / (n + 1)
    weights[-1] = 1.5 / (n + 1)

    return coords, weights


def _uniform_closed_quadrature_1d(n: int):
    coords = np.linspace(0.0, 1.0, n)
    weights = np.full(n, 1.0 / (n - 1))

    # Boundaries have half the weight
    weights[0] = 0.5 / (n - 1)
    weights[-1] = 0.5 / (n - 1)

    return coords, weights


def _open_newton_cotes_quadrature_1d(n: int):
    step = 1.0 / (n + 1)
    coords = np.linspace(step, 1.0 - step, n)

    # Weisstein, Eric W. "Newton-Cotes Formulas." From MathWorld--A Wolfram Web Resource.
    # https://mathworld.wolfram.com/Newton-CotesFormulas.html

    if n == 1:
        weights = np.array([1.0])
    elif n == 2:
        weights = np.array([0.5, 0.5])
    elif n == 3:
        weights = np.array([2.0, -1.0, 2.0]) / 3.0
    elif n == 4:
        weights = np.array([11.0, 1.0, 1.0, 11.0]) / 24.0
    elif n == 5:
        weights = np.array([11.0, -14.0, 26.0, -14.0, 11.0]) / 20.0
    elif n == 6:
        weights = np.array([611.0, -453.0, 562.0, 562.0, -453.0, 611.0]) / 1440.0
    elif n == 7:
        weights = np.array([460.0, -954.0, 2196.0, -2459.0, 2196.0, -954.0, 460.0]) / 945.0
    else:
        raise NotImplementedError

    return coords, weights


def _closed_newton_cotes_quadrature_1d(n: int):
    coords = np.linspace(0.0, 1.0, n)

    # OEIS: A093735, A093736

    if n == 2:
        weights = np.array([1.0, 1.0]) / 2.0
    elif n == 3:
        weights = np.array([1.0, 4.0, 1.0]) / 3.0
    elif n == 4:
        weights = np.array([3.0, 9.0, 9.0, 3.0]) / 8.0
    elif n == 5:
        weights = np.array([14.0, 64.0, 24.0, 64.0, 14.0]) / 45.0
    elif n == 6:
        weights = np.array([95.0 / 288.0, 125.0 / 96.0, 125.0 / 144.0, 125.0 / 144.0, 125.0 / 96.0, 95.0 / 288.0])
    elif n == 7:
        weights = np.array([41, 54, 27, 68, 27, 54, 41], dtype=float) / np.array(
            [140, 35, 140, 35, 140, 35, 140], dtype=float
        )
    elif n == 8:
        weights = np.array(
            [
                5257,
                25039,
                343,
                20923,
                20923,
                343,
                25039,
                5257,
            ]
        ) / np.array(
            [
                17280,
                17280,
                640,
                17280,
                17280,
                640,
                17280,
                17280,
            ],
            dtype=float,
        )
    else:
        raise NotImplementedError

    # Normalize with interval length
    weights = weights / (n - 1)

    return coords, weights


def quadrature_1d(point_count: int, family: Polynomial):
    """Return quadrature points and weights for the given family and point count.

    Args:
        point_count: Number of quadrature points.
        family: Polynomial family defining the quadrature rule.

    Returns:
        A tuple ``(coords, weights)`` in which ``coords`` is a NumPy array of quadrature point
        coordinates in the interval ``[0, 1]``, and ``weights`` is a NumPy array of corresponding
        quadrature weights.
    """

    if family == Polynomial.GAUSS_LEGENDRE:
        return _gauss_legendre_quadrature_1d(point_count)
    if family == Polynomial.LOBATTO_GAUSS_LEGENDRE:
        return _lobatto_gauss_legendre_quadrature_1d(point_count)
    if family == Polynomial.EQUISPACED_CLOSED:
        return _closed_newton_cotes_quadrature_1d(point_count)
    if family == Polynomial.EQUISPACED_OPEN:
        return _open_newton_cotes_quadrature_1d(point_count)

    raise NotImplementedError


def lagrange_scales(coords: np.ndarray) -> np.ndarray:
    """Return the scaling factors for Lagrange polynomials with roots at ``coords``.

    Args:
        coords: NumPy array of coordinates defining the polynomial roots.

    Returns:
        A NumPy array of scaling factors for the Lagrange basis polynomials, in which
        the i-th element is the reciprocal of the product of differences between
        ``coords[i]`` and all other coordinates.
    """
    lagrange_scale = np.empty_like(coords)
    for i in range(len(coords)):
        deltas = coords[i] - coords
        deltas[i] = 1.0
        lagrange_scale[i] = 1.0 / np.prod(deltas)

    return lagrange_scale
