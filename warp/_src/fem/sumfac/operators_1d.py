# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""1D sum-factorization operator matrices.

The ``B^T D B`` factorization for tensor-product elements (quad/hex) contracts
element DOFs against two small ``(P+1)``-sized 1D matrices, applied one axis at a
time:

* the **interpolation matrix** ``I[q, a] = L_a(points[q])``, which evaluates the
  nodal basis at the quadrature points, and
* the **reference derivative matrix** ``D_ref[q, a] = L'_a(points[q])``.

Here ``L_a`` is the Lagrange basis polynomial associated with node ``a`` of the
1D nodal basis (the roots ``nodes``). Both matrices are built once, on the CPU,
at setup time via the **barycentric** form of Lagrange interpolation, which is
numerically stable at the high polynomial orders (``P >= 5``) targeted by this
module.

The barycentric weights are ``w_a = 1 / prod_{b != a} (x_a - x_b)``. For an
evaluation point ``p`` that does not coincide with any node,

    L_a(p) = (w_a / (p - x_a)) / sum_b (w_b / (p - x_b)),

and the derivative follows from differentiating that quotient. When ``p`` equals
a node ``x_c`` the formulas above are singular and the well-known collocation
limits are used instead (see Berrut & Trefethen, *Barycentric Lagrange
Interpolation*, SIAM Review 2004).
"""

from __future__ import annotations

import numpy as np

from warp._src.fem.polynomial import Polynomial, lagrange_scales, quadrature_1d

__all__ = [
    "build_derivative_matrix",
    "build_interpolation_matrix",
    "default_basis_nodes",
    "default_quadrature_points",
]


def _barycentric_weights(nodes: np.ndarray) -> np.ndarray:
    """Return the barycentric weights ``w_a = 1 / prod_{b != a} (x_a - x_b)``.

    This is exactly the Lagrange scaling already used by the shape functions, so
    the two stay consistent.
    """
    return lagrange_scales(nodes)


def build_interpolation_matrix(nodes: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Build the 1D interpolation matrix ``I[q, a] = L_a(points[q])``.

    The matrix maps nodal values at ``nodes`` to interpolated values at
    ``points``: ``I @ u_nodal`` evaluates the nodal interpolant at every
    evaluation point. Each row is a partition of unity (sums to ``1``).

    Args:
        nodes: 1D array of ``n`` basis-node coordinates (the Lagrange roots).
        points: 1D array of ``q`` evaluation-point coordinates.

    Returns:
        A ``(q, n)`` NumPy array of basis-function values.
    """
    nodes = np.asarray(nodes, dtype=float)
    points = np.asarray(points, dtype=float)

    weights = _barycentric_weights(nodes)

    n = len(nodes)
    q = len(points)
    interp = np.zeros((q, n))

    # diff[r, a] = points[r] - nodes[a]
    diff = points[:, None] - nodes[None, :]

    for r in range(q):
        # Detect collocation (evaluation point coincides with a node).
        match = np.flatnonzero(diff[r] == 0.0)
        if match.size > 0:
            interp[r, match[0]] = 1.0
            continue

        terms = weights / diff[r]
        interp[r] = terms / np.sum(terms)

    return interp


def build_derivative_matrix(nodes: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Build the 1D reference derivative matrix ``D_ref[q, a] = L'_a(points[q])``.

    The matrix maps nodal values at ``nodes`` to the derivative of the nodal
    interpolant at ``points``: ``D_ref @ u_nodal`` evaluates the interpolant's
    derivative at every evaluation point. Each row sums to ``0`` (the derivative
    of the constant function vanishes).

    Args:
        nodes: 1D array of ``n`` basis-node coordinates (the Lagrange roots).
        points: 1D array of ``q`` evaluation-point coordinates.

    Returns:
        A ``(q, n)`` NumPy array of basis-function derivatives.
    """
    nodes = np.asarray(nodes, dtype=float)
    points = np.asarray(points, dtype=float)

    weights = _barycentric_weights(nodes)

    n = len(nodes)
    q = len(points)
    deriv = np.zeros((q, n))

    # diff[r, a] = points[r] - nodes[a]
    diff = points[:, None] - nodes[None, :]

    for r in range(q):
        match = np.flatnonzero(diff[r] == 0.0)
        if match.size > 0:
            # Evaluation point coincides with node c: use the collocation
            # (node-to-node) barycentric differentiation formula.
            c = match[0]
            for a in range(n):
                if a == c:
                    continue
                deriv[r, a] = (weights[a] / weights[c]) / (nodes[c] - nodes[a])
            # Diagonal entry is minus the sum of the off-diagonal entries
            # (rows of the differentiation matrix sum to zero).
            deriv[r, c] = -np.sum(deriv[r])
            continue

        # General evaluation point: differentiate the barycentric quotient
        #   L_a(p) = t_a / S,   t_a = w_a / (p - x_a),   S = sum_b t_b.
        terms = weights / diff[r]
        s = np.sum(terms)
        interp_row = terms / s

        # d/dp [ w_a / (p - x_a) ] = -w_a / (p - x_a)^2 = -t_a / (p - x_a)
        dterms = -terms / diff[r]
        ds = np.sum(dterms)

        deriv[r] = (dterms - interp_row * ds) / s

    return deriv


def default_basis_nodes(degree: int) -> np.ndarray:
    """Return the default 1D basis-node coordinates for a degree-``degree`` element.

    The default nodal basis for tensor-product (quad/hex) shape functions uses
    Lobatto--Gauss--Legendre (GLL) points, which include the interval endpoints
    and give ``degree + 1`` nodes on ``[0, 1]``.

    Args:
        degree: Polynomial degree ``P`` of the element.

    Returns:
        A ``(degree + 1,)`` NumPy array of node coordinates in ascending order.
    """
    if degree < 1:
        raise ValueError(f"Element degree must be at least 1 (got degree={degree}).")

    coords, _ = quadrature_1d(point_count=degree + 1, family=Polynomial.LOBATTO_GAUSS_LEGENDRE)
    return np.asarray(coords, dtype=float)


def default_quadrature_points(degree: int) -> np.ndarray:
    """Return the default 1D quadrature-point coordinates for a degree-``degree`` element.

    The default quadrature for tensor-product cells is Gauss--Legendre (GL). With
    ``degree + 1`` points the rule is exact to degree ``2 * degree + 1``, which
    integrates the mass matrix (degree ``2 * degree``) exactly.

    Args:
        degree: Polynomial degree ``P`` of the element.

    Returns:
        A ``(degree + 1,)`` NumPy array of quadrature coordinates in ascending order.
    """
    if degree < 1:
        raise ValueError(f"Element degree must be at least 1 (got degree={degree}).")

    coords, _ = quadrature_1d(point_count=degree + 1, family=Polynomial.GAUSS_LEGENDRE)
    return np.asarray(coords, dtype=float)
