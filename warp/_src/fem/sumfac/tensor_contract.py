# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Sum-factorized contractions ``B`` and ``B^T`` for tensor-product elements.

The ``B`` stage of the ``B^T D B`` factorization maps element nodal DOFs to the
value and *reference* gradient at every quadrature point, one axis at a time,
contracting against the small 1D interpolation matrix ``I`` and derivative
matrix ``D_ref`` from :mod:`warp._src.fem.sumfac.operators_1d`. The ``B^T``
stage is the exact transpose: it maps per-quadrature-point coefficients back to
element nodal residuals via the same separable contraction with the transposed
``(n, q)`` 1D matrices.

For a quad element with lexicographic nodal DOFs ``u(i, j)`` (``i`` the slowest
axis, ``j`` the fastest -- the ordering used by ``warp.fem`` square/cube shape
functions), the value at quadrature point ``(qx, qy)`` is the separable
contraction::

    V(qx, qy) = sum_{i, j} A[qx, i] B[qy, j] u(i, j)  ==  A @ U @ B^T,

where ``U`` is the ``(n, n)`` matrix ``U(i, j) = u(i, j)`` and ``n = P + 1``.
Choosing ``A`` and ``B`` selects the output:

* value:        ``A = I``,     ``B = I``      (``Kron(I, I)``)
* d/dxi:        ``A = D_ref``, ``B = I``      (``Kron(D_ref, I)``)
* d/deta:       ``A = I``,     ``B = D_ref``  (``Kron(I, D_ref)``)

The 3D hex case adds a third axis ``k`` (the fastest) and a third operator
matrix ``C``, with the analogous three-factor contraction
``sum_{i,j,k} A[qx,i] B[qy,j] C[qz,k] u(i,j,k)``.

Each directional contraction is a small dense matrix product issued with
:func:`warp.tile_matmul`, so it maps to tensor cores on GPU and to the scalar
fallback GEMM on CPU. Because the same 1D matrices apply to every element, a
panel of ``E_b`` elements is contracted together: the first (slowest-axis)
contraction is a single wide :func:`warp.tile_matmul` over an ``(n, n * E_b)``
(2D) or ``(n, n^2 * E_b)`` (3D) right-hand side; the remaining axes are folded
in with per-element, per-slice tile matmuls. ``E_b = 1`` recovers one element
per block.

These kernels are written to be correct on CPU (``block_dim = 1``, serialized)
and fast on GPU. ``P``, ``n = P + 1``, ``q``, ``E_b`` and their products are
baked in as :func:`warp.constant` values so the tile shapes are static and the
loops unroll.

The kernel factories and packing helpers are written for **arbitrary**
``(rows_out, rows_in)`` operator shapes; nothing in them assumes that the
operator maps nodes to quadrature points. The forward (``B``) drivers pass the
``(q, n)`` operators on an ``n``-wide packed input; the backward (``B^T``)
drivers pass the transposed ``(n, q)`` operators on a ``q``-wide packed input.
One kernel code path therefore serves both directions, and with the default
``q = n`` quadrature both directions share the same compiled tile shapes.
"""

from __future__ import annotations

import numpy as np

import warp as wp
from warp._src.fem import cache as fem_cache
from warp._src.fem.sumfac.operators_1d import (
    build_derivative_matrix,
    build_interpolation_matrix,
    default_basis_nodes,
    default_quadrature_points,
)

__all__ = [
    "build_operator_arrays",
    "contract_transpose_2d",
    "contract_transpose_3d",
    "interpolate_2d",
    "interpolate_3d",
    "make_interpolation_kernel_2d",
    "make_interpolation_kernel_3d",
    "pack_dofs_2d",
    "pack_dofs_3d",
]


def build_operator_arrays(degree: int, dtype=np.float64):
    """Build the 1D interpolation/derivative matrices for a degree-``degree`` element.

    Args:
        degree: Polynomial degree ``P`` of the element. The basis has ``n = P + 1``
            nodes and the default quadrature has ``q = n`` points.
        dtype: NumPy floating-point dtype for the returned matrices.

    Returns:
        A pair ``(interp, deriv)`` of ``(q, n)`` NumPy arrays, where
        ``interp[q, a] = L_a(points[q])`` and ``deriv[q, a] = L'_a(points[q])``.
    """
    nodes = default_basis_nodes(degree)
    points = default_quadrature_points(degree)
    interp = build_interpolation_matrix(nodes, points).astype(dtype)
    deriv = build_derivative_matrix(nodes, points).astype(dtype)
    return interp, deriv


# -- Host-side packing helpers ------------------------------------------------
#
# The contraction kernels read a panel of E_b elements as a single wide tile so
# the first (slowest-axis) contraction is one wide matmul. The per-element
# directional matrix is stored with the contracted axis ``i`` as the leading
# (row) dimension and the remaining axes (plus the batch index) flattened into
# the columns.


def pack_dofs_2d(dofs: np.ndarray, n: int) -> np.ndarray:
    """Pack ``(num_elements, n*n)`` lexicographic DOFs into the wide ``(n, num_elements*n)`` layout.

    Element ``e`` occupies columns ``[e*n : (e+1)*n]``; within that block the
    array is ``U_e(i, j) = dofs[e, i*n + j]`` (``i`` the leading row, ``j`` the
    column).
    """
    num_elements = dofs.shape[0]
    u = np.asarray(dofs).reshape(num_elements, n, n)  # [e, i, j]
    # move e between i and j -> [i, e, j] -> (n, num_elements*n)
    return np.ascontiguousarray(u.transpose(1, 0, 2).reshape(n, num_elements * n))


def pack_dofs_3d(dofs: np.ndarray, n: int) -> np.ndarray:
    """Pack ``(num_elements, n^3)`` lexicographic DOFs into the wide ``(n, num_elements*n*n)`` layout.

    Element ``e`` occupies columns ``[e*n*n : (e+1)*n*n]``; within that block the
    array is ``U_e(i, j*n + k) = dofs[e, i*n*n + j*n + k]`` (``i`` the leading
    row).
    """
    num_elements = dofs.shape[0]
    u = np.asarray(dofs).reshape(num_elements, n, n * n)  # [e, i, (j,k)]
    return np.ascontiguousarray(u.transpose(1, 0, 2).reshape(n, num_elements * n * n))


# -- Kernel factories ---------------------------------------------------------
#
# ``P``, ``n``, ``q``, ``E_b`` and the products needed by tile shapes are baked
# in as ``wp.constant`` values captured in the closure so the tile dimensions
# are compile-time constants and the loops unroll. ``fem_cache.dynamic_kernel``
# dedupes by suffix, so a given (n, q, E_b, dtype) shape compiles only once.
#
# Each specialization is registered in its own dynamic module (via
# ``fem_cache.dynamic_kernel``) rather than accumulating in this module:
# appending kernels to a shared module changes its hash and recompiles every
# previously built shape, making a sweep over (n, q, E_b) quadratic in CUDA
# compile time. Isolated modules keep it linear and cache-stable.
#
# The contraction kernels do not need autodiff yet (design risk 9.7 defers AD),
# hence ``enable_backward=False`` in every factory: with backward enabled every
# tile_matmul builds three GEMM LTOs (forward + two adjoints), tripling the
# first-run CUDA compile time of every kernel shape. Re-enable when the AD path
# lands.


def make_interpolation_kernel_2d(n: int, q: int, element_batch: int, dtype):
    """Build (and cache) the 2D contraction kernel ``out_e = A @ U_e @ B^T``.

    The kernel processes one ``element_batch``-wide panel per launch index. The
    first contraction (over axis ``i``) is a single wide ``tile_matmul`` over
    the packed ``(n, element_batch * n)`` panel; the second contraction (over
    axis ``j``) is one ``tile_matmul`` per element against ``B^T``.

    The kernel is agnostic to what the per-axis dimensions represent: ``n`` is
    the per-axis size of the *input* (operator columns) and ``q`` the per-axis
    size of the *output* (operator rows). The forward ``B`` stage passes
    ``(q, n)`` node-to-quadrature operators; the transpose ``B^T`` stage reuses
    this factory with the roles swapped and the transposed ``(n, q)`` operators.

    Args:
        n: Per-axis input size (operator columns); ``P + 1`` nodes in the
            forward direction.
        q: Per-axis output size (operator rows); the 1D quadrature point count
            in the forward direction.
        element_batch: Number of elements per panel ``E_b``.
        dtype: Warp scalar dtype (e.g. ``wp.float64``).

    Returns:
        A compiled ``wp.Kernel`` taking ``(A, B, packed, out)`` arrays.
    """
    n_c = wp.constant(n)
    q_c = wp.constant(q)
    eb_c = wp.constant(element_batch)
    qq_c = wp.constant(q * q)
    ebn_c = wp.constant(element_batch * n)

    @fem_cache.dynamic_kernel(
        suffix=f"contract2d_{n}_{q}_{element_batch}_{dtype.__name__}",
        kernel_options={"enable_backward": False},
    )
    def kernel(
        a_mat: wp.array2d(dtype=dtype),
        b_mat: wp.array2d(dtype=dtype),
        packed: wp.array2d(dtype=dtype),
        out: wp.array2d(dtype=dtype),
    ):
        panel = wp.tid()

        a_tile = wp.tile_load(a_mat, shape=(q_c, n_c))
        b_tile = wp.tile_load(b_mat, shape=(q_c, n_c))
        b_transpose = wp.tile_transpose(b_tile)

        # Wide first contraction over axis i: (q, n) @ (n, E_b*n) -> (q, E_b*n).
        rhs = wp.tile_load(packed, shape=(n_c, ebn_c), offset=(0, panel * ebn_c))
        stage1 = wp.tile_matmul(a_tile, rhs)  # [qx, e*n + j]

        for e in range(eb_c):
            # The per-element panel slice is strided (row stride E_b*n); copy it into a
            # contiguous tile first, because the cuBLASDx tile_matmul path assumes packed
            # operands (no leading dimension is passed to the GEMM).
            stage1_view = wp.tile_view(stage1, offset=(0, e * n_c), shape=(q_c, n_c))  # [qx, j]
            stage1_e = wp.tile_zeros(shape=(q_c, n_c), dtype=dtype)
            wp.tile_assign(stage1_e, stage1_view, offset=(0, 0))
            value_e = wp.tile_zeros(shape=(q_c, q_c), dtype=dtype)
            wp.tile_matmul(stage1_e, b_transpose, value_e)  # (q, q) [qx, qy]
            value_flat = wp.tile_reshape(value_e, shape=(1, qq_c))
            wp.tile_store(out, value_flat, offset=(panel * eb_c + e, 0))

    return kernel


def make_interpolation_kernel_3d(n: int, q: int, element_batch: int, dtype):
    """Build (and cache) the 3D contraction kernel ``out_e = (A,B,C) . U_e``.

    The kernel processes one ``element_batch``-wide panel per launch index:

    1. Wide first contraction over axis ``i``: one ``tile_matmul`` over the
       packed ``(n, element_batch * n^2)`` panel.
    2. Contraction over axis ``j``: per element, per ``qx``-slice, ``B @ slice``.
    3. Contraction over axis ``k``: per element, per ``qx``-slice, ``slice @ C^T``.

    All intermediate slices are contiguous (offset-only) tile views.

    As in the 2D factory, ``n`` is the per-axis input size and ``q`` the
    per-axis output size; the transpose ``B^T`` stage reuses this factory with
    the roles swapped and the transposed ``(n, q)`` operators.

    Args:
        n: Per-axis input size (operator columns); ``P + 1`` nodes in the
            forward direction.
        q: Per-axis output size (operator rows); the 1D quadrature point count
            in the forward direction.
        element_batch: Number of elements per panel ``E_b``.
        dtype: Warp scalar dtype (e.g. ``wp.float64``).

    Returns:
        A compiled ``wp.Kernel`` taking ``(A, B, C, packed, out)`` arrays.
    """
    n_c = wp.constant(n)
    q_c = wp.constant(q)
    eb_c = wp.constant(element_batch)
    nn_c = wp.constant(n * n)
    qn_c = wp.constant(q * n)
    qq_c = wp.constant(q * q)
    ebnn_c = wp.constant(element_batch * n * n)

    @fem_cache.dynamic_kernel(
        suffix=f"contract3d_{n}_{q}_{element_batch}_{dtype.__name__}",
        kernel_options={"enable_backward": False},
    )
    def kernel(
        a_mat: wp.array2d(dtype=dtype),
        b_mat: wp.array2d(dtype=dtype),
        c_mat: wp.array2d(dtype=dtype),
        packed: wp.array2d(dtype=dtype),
        out: wp.array2d(dtype=dtype),
    ):
        panel = wp.tid()

        a_tile = wp.tile_load(a_mat, shape=(q_c, n_c))
        b_tile = wp.tile_load(b_mat, shape=(q_c, n_c))
        c_tile = wp.tile_load(c_mat, shape=(q_c, n_c))
        c_transpose = wp.tile_transpose(c_tile)

        # Wide first contraction over axis i: (q, n) @ (n, E_b*n^2) -> (q, E_b*n^2).
        rhs = wp.tile_load(packed, shape=(n_c, ebnn_c), offset=(0, panel * ebnn_c))
        stage1 = wp.tile_matmul(a_tile, rhs)  # [qx, e*n^2 + j*n + k]

        for e in range(eb_c):
            stage1_e = wp.tile_view(stage1, offset=(0, e * nn_c), shape=(q_c, nn_c))  # [qx, j*n + k]

            # Contraction over axis j, per qx-slice. stage2: (q, q*n) [qx, qy*n + k].
            stage2 = wp.tile_zeros(shape=(q_c, qn_c), dtype=dtype)
            for qx in range(q_c):
                row = wp.tile_view(stage1_e, offset=(qx, 0), shape=(1, nn_c))  # [j*n + k]
                block = wp.tile_reshape(row, shape=(n_c, n_c))  # [j, k]
                slab = wp.tile_matmul(b_tile, block)  # (q, n) [qy, k]
                slab_flat = wp.tile_reshape(slab, shape=(1, qn_c))
                wp.tile_assign(stage2, slab_flat, offset=(qx, 0))

            # Contraction over axis k, per qx-slice. stage3: (q, q*q) [qx, qy*q + qz].
            for qx in range(q_c):
                row2 = wp.tile_view(stage2, offset=(qx, 0), shape=(1, qn_c))  # [qy*n + k]
                block2 = wp.tile_reshape(row2, shape=(q_c, n_c))  # [qy, k]
                slab2 = wp.tile_matmul(block2, c_transpose)  # (q, q) [qy, qz]
                slab2_flat = wp.tile_reshape(slab2, shape=(1, qq_c))
                wp.tile_store(out, slab2_flat, offset=(panel * eb_c + e, qx * qq_c))

    return kernel


# -- High-level NumPy drivers -------------------------------------------------


def _np_to_wp_dtype(np_dtype):
    return wp.float32 if np.dtype(np_dtype) == np.float32 else wp.float64


def _default_block_dim(device) -> int:
    """Pick the tile-kernel block size for ``device``.

    CPU tile kernels run serialized with a single thread per block. On CUDA the
    ``tile_matmul`` cuBLASDx path requires at least a full warp per block; a
    couple of warps is a safe default for the small matrices used here.
    """
    return 1 if wp.get_device(device).is_cpu else 64


def _contract(values, operators, rows_in, rows_out, dim, element_batch, device):
    """Apply the ``dim``-axis separable contraction to every element.

    Generic core shared by the forward (``B``) and transpose (``B^T``) drivers.
    Each operator in ``operators`` has shape ``(rows_out, rows_in)`` and is
    applied along one axis (slowest first), so the per-element result is
    ``Kron(operators...) @ vec(values_e)``.

    Args:
        values: ``(num_elements, rows_in**dim)`` NumPy array of lexicographic
            per-element coefficients. ``num_elements`` must be a multiple of
            ``element_batch``.
        operators: ``dim`` operator matrices of shape ``(rows_out, rows_in)``,
            ordered slowest axis first.
        rows_in: Per-axis input size (operator columns).
        rows_out: Per-axis output size (operator rows).
        dim: Spatial dimension (2 or 3).
        element_batch: Panel width ``E_b``.
        device: Warp device to run on.

    Returns:
        A ``(num_elements, rows_out**dim)`` NumPy array, lexicographically
        ordered (slowest axis first).
    """
    values = np.ascontiguousarray(values)
    num_elements = values.shape[0]
    if num_elements % element_batch != 0:
        raise ValueError(f"num_elements ({num_elements}) must be a multiple of element_batch ({element_batch}).")

    np_dtype = values.dtype if values.dtype in (np.float32, np.float64) else np.float64
    wp_dtype = _np_to_wp_dtype(np_dtype)

    if dim == 2:
        packed = pack_dofs_2d(values.astype(np_dtype), rows_in)
        kernel = make_interpolation_kernel_2d(rows_in, rows_out, element_batch, wp_dtype)
    elif dim == 3:
        packed = pack_dofs_3d(values.astype(np_dtype), rows_in)
        kernel = make_interpolation_kernel_3d(rows_in, rows_out, element_batch, wp_dtype)
    else:
        raise ValueError(f"Unsupported dimension {dim} (expected 2 or 3).")

    op_wp = [wp.array(np.ascontiguousarray(op, dtype=np_dtype), dtype=wp_dtype, device=device) for op in operators]
    packed_wp = wp.array(packed, dtype=wp_dtype, device=device)
    out_wp = wp.zeros((num_elements, rows_out**dim), dtype=wp_dtype, device=device)

    num_panels = num_elements // element_batch
    wp.launch_tiled(
        kernel,
        dim=[num_panels],
        inputs=[*op_wp, packed_wp, out_wp],
        block_dim=_default_block_dim(device),
        device=device,
    )
    wp.synchronize_device(device)
    return out_wp.numpy()


def interpolate_2d(dofs, a_mat, b_mat, n, q, element_batch=1, device=None):
    """Apply the 2D separable contraction ``out_e = A @ U_e @ B^T`` to every element.

    Args:
        dofs: ``(num_elements, n*n)`` NumPy array of lexicographic nodal DOFs.
            ``num_elements`` must be a multiple of ``element_batch``.
        a_mat: ``(q, n)`` operator matrix applied along the slow axis ``i``.
        b_mat: ``(q, n)`` operator matrix applied along the fast axis ``j``.
        n: Number of 1D nodes ``P + 1``.
        q: Number of 1D quadrature points.
        element_batch: Panel width ``E_b`` (default 1).
        device: Warp device to run on.

    Returns:
        A ``(num_elements, q*q)`` NumPy array of interpolated values at the
        quadrature points, ordered lexicographically ``(qx, qy)``.
    """
    return _contract(dofs, (a_mat, b_mat), n, q, 2, element_batch, device)


def interpolate_3d(dofs, a_mat, b_mat, c_mat, n, q, element_batch=1, device=None):
    """Apply the 3D separable contraction ``out_e = (A, B, C) . U_e`` to every element.

    Args:
        dofs: ``(num_elements, n^3)`` NumPy array of lexicographic nodal DOFs.
            ``num_elements`` must be a multiple of ``element_batch``.
        a_mat: ``(q, n)`` operator matrix applied along the slow axis ``i``.
        b_mat: ``(q, n)`` operator matrix applied along the middle axis ``j``.
        c_mat: ``(q, n)`` operator matrix applied along the fast axis ``k``.
        n: Number of 1D nodes ``P + 1``.
        q: Number of 1D quadrature points.
        element_batch: Panel width ``E_b`` (default 1).
        device: Warp device to run on.

    Returns:
        A ``(num_elements, q^3)`` NumPy array of interpolated values at the
        quadrature points, ordered lexicographically ``(qx, qy, qz)``.
    """
    return _contract(dofs, (a_mat, b_mat, c_mat), n, q, 3, element_batch, device)


def contract_transpose_2d(qvals, a_mat, b_mat, n, q, element_batch=1, device=None):
    """Apply the 2D transpose contraction ``r_e = Kron(A, B)^T @ vec(g_e)`` to every element.

    This is the ``B^T`` stage of the ``B^T D B`` factorization: per-quadrature-
    point coefficients are contracted back to element nodal residuals with the
    transposed 1D operators, reusing the forward kernel with the input/output
    roles swapped.

    Args:
        qvals: ``(num_elements, q*q)`` NumPy array of per-quadrature-point
            coefficients, ordered lexicographically ``(qx, qy)``.
            ``num_elements`` must be a multiple of ``element_batch``.
        a_mat: ``(q, n)`` operator matrix of the slow axis (transposed internally).
        b_mat: ``(q, n)`` operator matrix of the fast axis (transposed internally).
        n: Number of 1D nodes ``P + 1``.
        q: Number of 1D quadrature points.
        element_batch: Panel width ``E_b`` (default 1).
        device: Warp device to run on.

    Returns:
        A ``(num_elements, n*n)`` NumPy array of nodal residuals, ordered
        lexicographically ``(i, j)``.
    """
    a_t = np.ascontiguousarray(np.asarray(a_mat).T)
    b_t = np.ascontiguousarray(np.asarray(b_mat).T)
    return _contract(qvals, (a_t, b_t), q, n, 2, element_batch, device)


def contract_transpose_3d(qvals, a_mat, b_mat, c_mat, n, q, element_batch=1, device=None):
    """Apply the 3D transpose contraction ``r_e = Kron(A, B, C)^T @ vec(g_e)`` to every element.

    Args:
        qvals: ``(num_elements, q^3)`` NumPy array of per-quadrature-point
            coefficients, ordered lexicographically ``(qx, qy, qz)``.
            ``num_elements`` must be a multiple of ``element_batch``.
        a_mat: ``(q, n)`` operator matrix of the slow axis (transposed internally).
        b_mat: ``(q, n)`` operator matrix of the middle axis (transposed internally).
        c_mat: ``(q, n)`` operator matrix of the fast axis (transposed internally).
        n: Number of 1D nodes ``P + 1``.
        q: Number of 1D quadrature points.
        element_batch: Panel width ``E_b`` (default 1).
        device: Warp device to run on.

    Returns:
        A ``(num_elements, n^3)`` NumPy array of nodal residuals, ordered
        lexicographically ``(i, j, k)``.
    """
    a_t = np.ascontiguousarray(np.asarray(a_mat).T)
    b_t = np.ascontiguousarray(np.asarray(b_mat).T)
    c_t = np.ascontiguousarray(np.asarray(c_mat).T)
    return _contract(qvals, (a_t, b_t, c_t), q, n, 3, element_batch, device)
