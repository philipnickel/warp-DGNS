# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Host-side face-trace operators for surface sum-factorization (Phase 6, stage 1).

On a quad/hex face normal to element axis ``a`` of a tensor-product element
whose 1D basis nodes include the interval endpoints (GLL / closed families,
:func:`warp._src.fem.polynomial.is_closed`), the trace of the nodal basis
factorizes:

* **Value trace:** ``L_i(0) = delta_{i,0}`` and ``L_i(1) = delta_{i,n-1}``, so
  the face value tensor is the boundary *slice* of the element DOF tensor along
  axis ``a`` -- no contraction (:func:`face_slice_indices`,
  :func:`face_trace_value`).
* **Normal-gradient trace:** one 1D contraction collapsing axis ``a`` with the
  endpoint row of the 1D derivative matrix (:func:`normal_derivative_row`,
  :func:`face_trace_normal_gradient`).

Element-axis to face-axis convention
------------------------------------

Element DOFs are lexicographic with axis 0 slowest (``warp.fem`` square/cube
shape functions, ``_node_ij``/``_node_ijk``). The **face frame** used by every
function in this module keeps the remaining element axes in *increasing* order,
slowest first (:func:`face_axes`):

* 2D, axis ``a``: face axis 0 = element axis ``1 - a``.
* 3D, axis 0: face axes = element axes ``(1, 2)``; axis 1: ``(0, 2)``;
  axis 2: ``(0, 1)``.

so a face tensor ``F`` of a 3D element with ``axis=1`` is indexed
``F[i_0, i_2]`` with ``i_0`` (element axis 0) slowest.

Observed ``warp.fem`` side orientation conventions (Grid2D / Grid3D)
--------------------------------------------------------------------

The mapping between side reference coordinates and the inner/outer element
frames below was extracted *empirically* by evaluating ``fem.inner`` /
``fem.outer`` / ``fem.to_inner_cell`` / ``fem.to_outer_cell`` of discrete
fields at actual side quadrature points (see
``warp/tests/fem/test_fem_sumfac_face_trace.py``), and matches the source:

* A grid side with ``Side.axis = a`` is normal to world/element axis ``a``;
  for interior sides the geometric normal points from the inner cell to the
  outer cell, in the ``+a`` direction.
* **Normal axis ends:** the inner cell touches an interior side at its local
  coordinate ``1.0`` along ``a`` and the outer cell at ``0.0``
  (``side_inner_cell_coords`` / ``side_outer_cell_coords``,
  ``warp/_src/fem/geometry/grid_2d.py:507-528`` and
  ``warp/_src/fem/geometry/grid_3d.py:529-543``). At domain boundaries
  (side altitude ``0`` or ``res``) inner == outer with end ``0`` / ``1``
  (:func:`grid_side_face_ends`).
* **Tangential axes:** in 2D the single side coordinate ``s0`` runs along
  element axis ``1 - a`` (``Grid2D.orient``,
  ``warp/_src/fem/geometry/grid_2d.py:269-275``). In 3D the side frame is the
  *cyclic* permutation ``(a, a+1, a+2) mod 3``: side coordinate ``s0``
  (longitude) runs along element axis ``(a + 1) % 3`` and ``s1`` (latitude)
  along ``(a + 2) % 3`` (``Grid3D._local_to_world``,
  ``warp/_src/fem/geometry/grid_3d.py:286-287``). For ``a = 1`` this is
  ``(s0 -> z, s1 -> x)``, i.e. *reversed* with respect to the sorted face
  frame above -- :func:`grid_side_to_face_coords` reorders accordingly.
* **Longitude flip:** the longitude coordinate is reversed
  (``t = 1 - s0``) exactly when
  ``Grid2D.is_flipped = (a == 0) == (altitude == 0)``
  (``warp/_src/fem/geometry/grid_2d.py:277-279``; interior 2D sides normal to
  ``y`` are flipped, those normal to ``x`` are not), and in 3D when
  ``altitude == 0`` (``warp/_src/fem/geometry/grid_3d.py:532,542``; *no*
  interior 3D side is flipped, only the boundary sides on the lower domain
  faces). The 3D latitude coordinate ``s1`` is never flipped.
  (:func:`grid_side_longitude_flipped`).
* **Inner vs. outer:** the flip/permutation formulas are *identical* for the
  inner and the outer cell (same ``side_coord`` expression in
  ``side_inner_cell_coords`` and ``side_outer_cell_coords``,
  ``warp/_src/fem/geometry/grid_2d.py:513,526`` and
  ``warp/_src/fem/geometry/grid_3d.py:532,542``), so the two traces of a
  shared interior side are traversed in the *same* face order: the
  inner-to-outer permutation is the identity
  (:func:`grid_outer_face_permutation`).

What stage 2 consumes
---------------------

The surface tile kernels (stage 2) gather the face DOF slice with
:func:`face_slice_indices` (value trace) or collapse the normal axis with
:func:`normal_derivative_row` (normal-gradient trace), then contract the
remaining ``d - 1`` axes against 1D interpolation/derivative matrices built at
the side quadrature points mapped through :func:`grid_side_to_face_coords`
(with :func:`grid_side_face_ends` selecting the slice ends). Because the
inner/outer permutation is the identity, jump/average terms may combine the two
traces pointwise without reindexing.
"""

from __future__ import annotations

import numpy as np

from warp._src.fem.sumfac.operators_1d import build_derivative_matrix, default_basis_nodes

__all__ = [
    "face_axes",
    "face_slice_indices",
    "face_trace_normal_gradient",
    "face_trace_value",
    "grid_outer_face_permutation",
    "grid_side_axes",
    "grid_side_face_ends",
    "grid_side_longitude_flipped",
    "grid_side_to_face_coords",
    "normal_derivative_row",
]


def _check_axis(dim: int, axis: int) -> None:
    if dim not in (2, 3):
        raise ValueError(f"Unsupported dimension {dim} (expected 2 or 3).")
    if not 0 <= axis < dim:
        raise ValueError(f"Face normal axis must be in [0, {dim}) (got axis={axis}).")


def _check_end(end: int) -> None:
    if end not in (0, 1):
        raise ValueError(f"Face end must be 0 or 1 (got end={end}).")


def face_axes(dim: int, axis: int) -> tuple[int, ...]:
    """Return the element axes spanning the face normal to ``axis``, in face-frame order.

    The face frame keeps the remaining element axes in increasing order,
    slowest first; this is the ordering of the face tensors returned by
    :func:`face_trace_value` and :func:`face_trace_normal_gradient`.

    Args:
        dim: Spatial dimension of the element (2 or 3).
        axis: Element axis the face is normal to.

    Returns:
        A tuple of ``dim - 1`` element axes.
    """
    _check_axis(dim, axis)
    return tuple(d for d in range(dim) if d != axis)


def face_slice_indices(n: int, dim: int, axis: int, end: int) -> np.ndarray:
    """Return the indices of the face DOF slice in the lexicographic element DOF vector.

    With endpoint-including (GLL) 1D nodes, the value trace of the element
    interpolant on the face normal to ``axis`` at coordinate ``end`` is exactly
    the element DOF tensor sliced at node ``0`` (``end=0``) or ``n - 1``
    (``end=1``) along ``axis``. The returned indices enumerate that slice in
    the face frame of :func:`face_axes` (remaining element axes in increasing
    order, slowest first).

    Args:
        n: Number of 1D nodes per axis (``P + 1``).
        dim: Spatial dimension of the element (2 or 3).
        axis: Element axis the face is normal to.
        end: Face position along ``axis``: 0 (coordinate 0.0) or 1 (coordinate 1.0).

    Returns:
        An ``(n**(dim - 1),)`` NumPy index array into the ``n**dim`` element
        DOF vector.
    """
    _check_axis(dim, axis)
    _check_end(end)
    indices = np.arange(n**dim).reshape((n,) * dim)
    return np.ascontiguousarray(np.take(indices, -1 if end == 1 else 0, axis=axis).reshape(-1))


def normal_derivative_row(nodes: np.ndarray, end: int) -> np.ndarray:
    """Return the endpoint row of the 1D derivative matrix.

    ``row[i] = L'_i(end)`` contracts the element DOF tensor along the face
    normal axis to produce the normal-derivative trace (in reference
    coordinates): ``du/dx_a|_face = sum_i row[i] * u(i, ...)``.

    The row is always the ``+axis`` reference-coordinate derivative,
    regardless of ``end``: the caller applies the geometric side-normal sign
    (for grid sides, ``-axis`` on ``altitude == 0`` boundary sides, ``+axis``
    otherwise) and the metric factors.

    Args:
        nodes: 1D array of ``n`` basis-node coordinates (the Lagrange roots).
        end: Face position: 0 (evaluate ``L'`` at 0.0) or 1 (at 1.0).

    Returns:
        An ``(n,)`` NumPy array.
    """
    _check_end(end)
    return build_derivative_matrix(nodes, np.array([float(end)]))[0]


def face_trace_value(dofs: np.ndarray, n: int, dim: int, axis: int, end: int) -> np.ndarray:
    """Return the value trace of element DOF tensors on a face (GLL slice).

    Args:
        dofs: ``(num_elements, n**dim)`` NumPy array of lexicographic element
            DOFs (axis 0 slowest).
        n: Number of 1D nodes per axis (``P + 1``).
        dim: Spatial dimension of the element (2 or 3).
        axis: Element axis the face is normal to.
        end: Face position along ``axis`` (0 or 1).

    Returns:
        A ``(num_elements, n**(dim - 1))`` NumPy array of face nodal values in
        the face frame of :func:`face_axes`.
    """
    dofs = np.asarray(dofs)
    if dofs.ndim != 2 or dofs.shape[1] != n**dim:
        raise ValueError(f"Expected a (num_elements, {n**dim}) DOF array, got shape {dofs.shape}.")
    return np.ascontiguousarray(dofs[:, face_slice_indices(n, dim, axis, end)])


def face_trace_normal_gradient(
    dofs: np.ndarray, n: int, dim: int, axis: int, end: int, nodes: np.ndarray | None = None
) -> np.ndarray:
    """Return the reference normal-derivative trace of element DOF tensors on a face.

    Contracts the element DOF tensor along ``axis`` with
    :func:`normal_derivative_row`: the result holds the derivative of the
    nodal interpolant with respect to the reference coordinate ``x_axis``,
    evaluated at the face nodes (the GLL nodes of the remaining axes with
    ``x_axis`` pinned to ``end``). This is always the ``+axis`` reference
    derivative, regardless of ``end``; the caller applies the geometric
    side-normal sign and the metric factors (on interior grid sides the side
    normal is ``+axis`` for both traces, so the value is directly
    normal-aligned).

    Args:
        dofs: ``(num_elements, n**dim)`` NumPy array of lexicographic element
            DOFs (axis 0 slowest).
        n: Number of 1D nodes per axis (``P + 1``).
        dim: Spatial dimension of the element (2 or 3).
        axis: Element axis the face is normal to.
        end: Face position along ``axis`` (0 or 1).
        nodes: Optional 1D basis-node coordinates; defaults to the GLL nodes of
            degree ``n - 1`` (:func:`default_basis_nodes`).

    Returns:
        A ``(num_elements, n**(dim - 1))`` NumPy array of face nodal normal
        derivatives in the face frame of :func:`face_axes`.
    """
    _check_axis(dim, axis)
    dofs = np.asarray(dofs)
    if nodes is None:
        nodes = default_basis_nodes(n - 1)
    row = normal_derivative_row(nodes, end)
    num_elements = dofs.shape[0]
    tensors = dofs.reshape((num_elements,) + (n,) * dim)
    # Contract the face-normal axis (axis + 1 accounts for the element axis).
    collapsed = np.tensordot(tensors, row, axes=([axis + 1], [0]))
    return np.ascontiguousarray(collapsed.reshape(num_elements, n ** (dim - 1)))


# -- Grid2D / Grid3D side-to-element coordinate mapping ------------------------
#
# The functions below encode the side orientation conventions of the structured
# grid geometries (warp/_src/fem/geometry/grid_2d.py, grid_3d.py), extracted
# empirically from fem.inner / fem.outer evaluations at side quadrature points
# (test_fem_sumfac_face_trace.py, test_face_trace_ordering_matches_side_machinery).
# They do NOT apply to unstructured Quadmesh2D / Hexmesh sides, whose face
# orientations depend on the mesh connectivity.


def grid_side_axes(dim: int, axis: int) -> tuple[int, ...]:
    """Return the element axes the side reference coordinates run along.

    ``warp.fem`` grid sides traverse the face in the *cyclic* frame
    ``(axis, axis + 1, axis + 2) mod dim``: side coordinate ``s0`` (longitude)
    runs along element axis ``(axis + 1) % dim`` and, in 3D, ``s1`` (latitude)
    along ``(axis + 2) % 3`` (``Grid2D.orient``, grid_2d.py:269-275;
    ``Grid3D._local_to_world``, grid_3d.py:286-287). Note for 3D ``axis=1``
    this order (``(2, 0)``) is the reverse of the sorted face frame
    (:func:`face_axes`).

    Args:
        dim: Spatial dimension (2 or 3).
        axis: Element axis the side is normal to.

    Returns:
        A tuple of ``dim - 1`` element axes, ordered like the side coordinates.
    """
    _check_axis(dim, axis)
    return tuple((axis + k) % dim for k in range(1, dim))


def grid_side_longitude_flipped(dim: int, axis: int, altitude: int) -> bool:
    """Return whether the side longitude coordinate is reversed in the element frame.

    When flipped, the element-frame coordinate along
    ``grid_side_axes(dim, axis)[0]`` is ``1 - s0`` instead of ``s0``. Empirical
    convention (identical for the inner and outer cell):

    * 2D: flipped iff ``(axis == 0) == (altitude == 0)``
      (``Grid2D.is_flipped``, grid_2d.py:277-279). In particular *interior*
      sides normal to ``y`` (``axis == 1``) are flipped, interior sides normal
      to ``x`` are not.
    * 3D: flipped iff ``altitude == 0`` (grid_3d.py:532,542); *no interior 3D
      side is flipped*. The latitude coordinate ``s1`` is never flipped.

    Args:
        dim: Spatial dimension (2 or 3).
        axis: Element axis the side is normal to.
        altitude: Grid altitude of the side along ``axis`` (0 to ``res[axis]``);
            interior sides have ``0 < altitude < res[axis]``.

    Returns:
        True when the longitude coordinate must be reversed.
    """
    _check_axis(dim, axis)
    if dim == 2:
        return (axis == 0) == (altitude == 0)
    return altitude == 0


def grid_side_face_ends(altitude: int, res_axis: int) -> tuple[int, int]:
    """Return the ``(inner, outer)`` face ends of a grid side along its normal axis.

    For interior sides the inner cell touches the side at its local coordinate
    ``1.0`` along the normal axis and the outer cell at ``0.0`` (the normal
    points from inner to outer, in the ``+axis`` direction). At domain
    boundaries inner == outer: both ends are 0 at ``altitude == 0`` and 1 at
    ``altitude == res_axis`` (grid_2d.py:507-528, grid_3d.py:529-543).

    Args:
        altitude: Grid altitude of the side along its normal axis.
        res_axis: Grid resolution along the normal axis.

    Returns:
        A pair of slice ends (each 0 or 1) for :func:`face_slice_indices`:
        ``(inner_end, outer_end)``.
    """
    if not 0 <= altitude <= res_axis:
        raise ValueError(f"Side altitude must be in [0, {res_axis}] (got altitude={altitude}).")
    inner_end = 0 if altitude == 0 else 1
    outer_end = 1 if altitude == res_axis else 0
    return inner_end, outer_end


def grid_side_to_face_coords(dim: int, axis: int, altitude: int, side_coords: np.ndarray) -> np.ndarray:
    """Map grid side reference coordinates to face-frame coordinates.

    Applies the longitude flip (:func:`grid_side_longitude_flipped`) and
    reorders the side coordinates from the cyclic side frame
    (:func:`grid_side_axes`) into the sorted face frame (:func:`face_axes`),
    yielding the element-local coordinates of each side point along the face
    axes. The same mapping applies to the inner and the outer cell
    (:func:`grid_outer_face_permutation`).

    Args:
        dim: Spatial dimension (2 or 3).
        axis: Element axis the side is normal to.
        altitude: Grid altitude of the side along ``axis``.
        side_coords: ``(num_points, dim - 1)`` array of side reference
            coordinates (``s0`` longitude and, in 3D, ``s1`` latitude).

    Returns:
        A ``(num_points, dim - 1)`` NumPy array; column ``f`` is the
        element-local coordinate along ``face_axes(dim, axis)[f]``.
    """
    side_coords = np.atleast_2d(np.asarray(side_coords, dtype=float))
    if side_coords.shape[1] != dim - 1:
        raise ValueError(f"Expected (num_points, {dim - 1}) side coordinates, got shape {side_coords.shape}.")

    coords = side_coords.copy()
    if grid_side_longitude_flipped(dim, axis, altitude):
        coords[:, 0] = 1.0 - coords[:, 0]

    # Reorder from the cyclic side frame into the sorted face frame.
    side_to_element = grid_side_axes(dim, axis)
    order = [side_to_element.index(element_axis) for element_axis in face_axes(dim, axis)]
    return np.ascontiguousarray(coords[:, order])


def grid_outer_face_permutation(n: int, dim: int) -> np.ndarray:
    """Return the face-node permutation between the inner and outer trace of a grid side.

    Empirical finding (the stage-2 de-risking result): for Grid2D and Grid3D
    interior sides, the outer cell traverses the shared face with *exactly* the
    same side-to-element tangential map as the inner cell -- the longitude
    flip and the cyclic axis order depend only on the side, not on which cell
    is being traced (grid_2d.py:513,526; grid_3d.py:532,542). The
    inner-to-outer face permutation is therefore the **identity**: face node
    (or quadrature point) ``r`` of the inner trace coincides geometrically with
    face node ``r`` of the outer trace.

    Args:
        n: Number of 1D nodes (or quadrature points) per face axis.
        dim: Spatial dimension of the *element* (2 or 3).

    Returns:
        An ``(n**(dim - 1),)`` NumPy index array (the identity permutation).
    """
    if dim not in (2, 3):
        raise ValueError(f"Unsupported dimension {dim} (expected 2 or 3).")
    return np.arange(n ** (dim - 1))
