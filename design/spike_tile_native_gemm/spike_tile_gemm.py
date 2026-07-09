# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Spike: tile-native, GEMM-pure sum-factorized DG element kernels.

A from-scratch formulation of the ``B^T D B`` element apply using ONLY Warp
tile collectives -- no scalar gather loops, no per-quadrature-point loops, no
branchy callees. Every stage is a ``tile_matmul`` (cuBLASDx, tensor cores when
available) or an elementwise tile product:

1. **Stacked operator.** ``A_hat = [A; D_hat]`` of shape ``(2q, n)`` stacks the
   1D interpolation matrix on the 1D derivative matrix. One pair of GEMMs
   ``S = A_hat @ U @ A_hat^T`` produces a ``(2q, 2q)`` tile holding ALL
   channels as blocks: values ``A U A^T``, x-gradient ``D U A^T``, y-gradient
   ``A U D^T``, and a cross block ``D U D^T`` (free; used by anisotropic
   coefficients, multiplied by zero otherwise).
2. **Pointwise D stage as one Hadamard.** ``F = G (*) S`` with a precomputed
   per-element ``(2q, 2q)`` geometry tile ``G``: mass weights in block (0,0),
   gradient weights in blocks (1,0)/(0,1). Affine geometry folds J and detJ
   into ``G`` on the host; spatially varying coefficients just change ``G``.
3. **Transpose pair.** ``R = A_hat^T @ F @ A_hat`` collapses by block algebra
   to exactly ``A^T f0 A + D^T f1x A + A^T f1y D`` -- the full B^T stage in
   two GEMMs.

The same pattern covers 3D by axis rotation (transpose + reshape between three
sweeps) and multi-element batching by widening the first contraction's columns
to ``n * E_b`` (Roget-style panels). GEMM shapes at P=4 (n=q=5, f64):
``(10,5)x(5,5)`` and ``(10,5)x(5,10)`` instead of five separate
``(5,5)x(5,5)`` -- twice the DMMA row fill per multiply and four GEMMs total.

Validation: dense Kronecker references in NumPy, degrees covering n=4..6 --
deliberately including n=5, the shape on which the current fused scalar-gather
kernel is miscompiled by nvJitLink's LTO pass (see design/sumfac-status.md).
The GEMM-pure kernels contain none of the structure involved in that trigger.

Run: uv run design/spike_tile_native_gemm/spike_tile_gemm.py
"""

import numpy as np

import warp as wp
from warp._src.fem.polynomial import Polynomial, quadrature_1d
from warp._src.fem.sumfac.operators_1d import (
    build_derivative_matrix,
    build_interpolation_matrix,
    default_basis_nodes,
    default_quadrature_points,
)

wp.init()

# ---------------------------------------------------------------------------
# Kernel factories: pure tile pipelines, one element per block
# ---------------------------------------------------------------------------


def make_bilinear_2d_kernel(n: int, q: int, dtype):
    """2D B^T D B apply with value + gradient channels: 4 GEMMs + 1 Hadamard."""
    n_c = wp.constant(n)
    q2_c = wp.constant(2 * q)
    nn_c = wp.constant(n * n)

    @wp.kernel(enable_backward=False)
    def kernel(
        a_hat: wp.array2d(dtype=dtype),  # (2q, n) stacked [A; D_hat]
        geom: wp.array3d(dtype=dtype),  # (num_elements, 2q, 2q) per-element G
        dofs: wp.array(dtype=dtype),  # (num_elements * n*n) element-major
        out: wp.array(dtype=dtype),  # (num_elements * n*n)
    ):
        e = wp.tid()

        op = wp.tile_load(a_hat, shape=(q2_c, n_c))
        op_t = wp.tile_transpose(op)

        u_flat = wp.tile_load(dofs, shape=(nn_c,), offset=(e * nn_c,))
        u = wp.tile_reshape(u_flat, shape=(n_c, n_c))

        # B stage: all channels in one (2q, 2q) tile
        t = wp.tile_matmul(op, u)  # (2q, n)
        s = wp.tile_matmul(t, op_t)  # (2q, 2q): [[V, Uy], [Ux, Uxy]]

        # D stage: pointwise geometry/coefficients, one Hadamard
        g = wp.tile_load(geom[e], shape=(q2_c, q2_c))
        f = s * g

        # B^T stage
        r1 = wp.tile_matmul(op_t, f)  # (n, 2q)
        r = wp.tile_matmul(r1, op)  # (n, n)

        r_flat = wp.tile_reshape(r, shape=(nn_c,))
        wp.tile_store(out, r_flat, offset=(e * nn_c,))

    return kernel


def make_mass_2d_panel_kernel(n: int, q: int, element_batch: int, dtype):
    """2D mass with an E_b-wide first contraction: the Roget panel shape."""
    n_c = wp.constant(n)
    q_c = wp.constant(q)
    qq_c = wp.constant(q * q)
    ebn_c = wp.constant(element_batch * n)
    eb_c = wp.constant(element_batch)

    @wp.kernel(enable_backward=False)
    def kernel(
        a_mat: wp.array2d(dtype=dtype),  # (q, n)
        weights: wp.array2d(dtype=dtype),  # (q, q) tensor-product w * detJ
        packed: wp.array2d(dtype=dtype),  # (n, num_elements * n) packed panels
        out: wp.array2d(dtype=dtype),  # (num_elements, q*q) quadrature values
    ):
        panel = wp.tid()

        a_tile = wp.tile_load(a_mat, shape=(q_c, n_c))
        a_t = wp.tile_transpose(a_tile)
        w_tile = wp.tile_load(weights, shape=(q_c, q_c))

        # One wide GEMM across the whole panel: (q, n) @ (n, E_b * n)
        rhs = wp.tile_load(packed, shape=(n_c, ebn_c), offset=(0, panel * ebn_c))
        stage = wp.tile_matmul(a_tile, rhs)  # (q, E_b * n)

        for e in range(eb_c):
            view = wp.tile_view(stage, offset=(0, e * n_c), shape=(q_c, n_c))
            cur = wp.tile_zeros(shape=(q_c, n_c), dtype=dtype)
            wp.tile_assign(cur, view, offset=(0, 0))
            vals = wp.tile_matmul(cur, a_t)  # (q, q)
            wvals = vals * w_tile
            flat = wp.tile_reshape(wvals, shape=(1, qq_c))
            wp.tile_store(out, flat, offset=(panel * eb_c + e, 0))

    return kernel


def make_mass_3d_kernel(n: int, q: int, dtype):
    """3D mass via axis rotation: every contraction is ONE wide GEMM.

    Layout discipline: rows = contracted axis, columns = remaining axes
    flattened. After each GEMM, transpose + reshape rotates the next axis into
    the rows. Six GEMMs total (three out, three back); the symmetric reverse
    sweep returns the lexicographic nodal layout.
    """
    n_c = wp.constant(n)
    q_c = wp.constant(q)
    nn_c = wp.constant(n * n)
    nnn_c = wp.constant(n * n * n)
    qn_c = wp.constant(q * n)
    qq_c = wp.constant(q * q)
    qqq_c = wp.constant(q * q * q)

    @wp.kernel(enable_backward=False)
    def kernel(
        a_mat: wp.array2d(dtype=dtype),  # (q, n)
        weights: wp.array(dtype=dtype),  # (q*q*q,) separable weights * detJ
        dofs: wp.array(dtype=dtype),  # (num_elements * n^3) element-major
        out: wp.array(dtype=dtype),  # (num_elements * n^3)
    ):
        e = wp.tid()

        a_tile = wp.tile_load(a_mat, shape=(q_c, n_c))
        a_t = wp.tile_transpose(a_tile)

        u_flat = wp.tile_load(dofs, shape=(nnn_c,), offset=(e * nnn_c,))
        u = wp.tile_reshape(u_flat, shape=(n_c, nn_c))  # rows x, cols (y,z)

        # --- forward: contract x, y, z; rotate between sweeps
        t1 = wp.tile_matmul(a_tile, u)  # (q, n^2) rows qx
        t1t = wp.tile_zeros(shape=(nn_c, q_c), dtype=dtype)
        wp.tile_assign(t1t, wp.tile_transpose(t1), offset=(0, 0))
        t1r = wp.tile_reshape(t1t, shape=(n_c, qn_c))  # rows y, cols (z,qx)

        t2 = wp.tile_matmul(a_tile, t1r)  # (q, n*q) rows qy
        t2t = wp.tile_zeros(shape=(qn_c, q_c), dtype=dtype)
        wp.tile_assign(t2t, wp.tile_transpose(t2), offset=(0, 0))
        t2r = wp.tile_reshape(t2t, shape=(n_c, qq_c))  # rows z, cols (qx,qy)

        t3 = wp.tile_matmul(a_tile, t2r)  # (q, q^2) rows qz, cols (qx,qy)

        # --- D stage: separable weights, one Hadamard
        w_flat = wp.tile_load(weights, shape=(qqq_c,))
        w = wp.tile_reshape(w_flat, shape=(q_c, qq_c))
        f = t3 * w

        # --- backward: same rotation pattern with A^T restores the layout
        s1 = wp.tile_matmul(a_t, f)  # (n, q^2) rows z
        s1t = wp.tile_zeros(shape=(qq_c, n_c), dtype=dtype)
        wp.tile_assign(s1t, wp.tile_transpose(s1), offset=(0, 0))
        s1r = wp.tile_reshape(s1t, shape=(q_c, qn_c))  # rows qx, cols (qy,z)

        s2 = wp.tile_matmul(a_t, s1r)  # (n, q*n) rows x
        s2t = wp.tile_zeros(shape=(qn_c, n_c), dtype=dtype)
        wp.tile_assign(s2t, wp.tile_transpose(s2), offset=(0, 0))
        s2r = wp.tile_reshape(s2t, shape=(q_c, nn_c))  # rows qy, cols (z,x)

        s3 = wp.tile_matmul(a_t, s2r)  # (n, n^2) rows y, cols (z,x)

        r_flat = wp.tile_reshape(s3, shape=(nnn_c,))
        wp.tile_store(out, r_flat, offset=(e * nnn_c,))

    return kernel


# ---------------------------------------------------------------------------
# Dense NumPy references
# ---------------------------------------------------------------------------


def operators(degree):
    nodes = default_basis_nodes(degree)
    points = default_quadrature_points(degree)
    interp = build_interpolation_matrix(nodes, points)
    deriv = build_derivative_matrix(nodes, points)
    _, w1d = quadrature_1d(point_count=degree + 1, family=Polynomial.GAUSS_LEGENDRE)
    return interp, deriv, np.asarray(w1d, dtype=float)


def check_bilinear_2d(degree, device, rng):
    interp, deriv, w1d = operators(degree)
    n = degree + 1
    q = len(w1d)
    w2d = np.kron(w1d, w1d)  # (q*q,), lexicographic (qx, qy)

    num_elements = 4
    mass_coef = 0.7
    diff_coef = np.array([1.3, 0.4])  # diagonal anisotropic diffusion

    # Stacked operator and per-element geometry tile (affine unit elements)
    a_hat = np.vstack([interp, deriv])  # (2q, n)
    g = np.zeros((2 * q, 2 * q))
    w_tile = w2d.reshape(q, q)
    g[:q, :q] = mass_coef * w_tile
    g[q:, :q] = diff_coef[0] * w_tile  # f1x <- Ux block (rows = D-rows)
    g[:q, q:] = diff_coef[1] * w_tile  # f1y <- Uy block
    geom = np.broadcast_to(g, (num_elements, 2 * q, 2 * q)).copy()

    dofs = rng.standard_normal(num_elements * n * n)

    kernel = make_bilinear_2d_kernel(n, q, wp.float64)
    a_hat_wp = wp.array(a_hat, dtype=wp.float64, device=device)
    geom_wp = wp.array(geom, dtype=wp.float64, device=device)
    dofs_wp = wp.array(dofs, dtype=wp.float64, device=device)
    out_wp = wp.zeros(num_elements * n * n, dtype=wp.float64, device=device)
    block_dim = 1 if wp.get_device(device).is_cpu else 64
    wp.launch_tiled(
        kernel, dim=[num_elements], inputs=[a_hat_wp, geom_wp, dofs_wp, out_wp], block_dim=block_dim, device=device
    )
    wp.synchronize_device(device)

    # Dense reference: M = sum_c Bc^T diag(w * coef) Bc
    bv = np.kron(interp, interp)
    bx = np.kron(deriv, interp)
    by = np.kron(interp, deriv)
    mass = bv.T @ np.diag(mass_coef * w2d) @ bv
    mass += bx.T @ np.diag(diff_coef[0] * w2d) @ bx
    mass += by.T @ np.diag(diff_coef[1] * w2d) @ by
    expected = (dofs.reshape(num_elements, -1) @ mass.T).reshape(-1)

    np.testing.assert_allclose(out_wp.numpy(), expected, atol=1e-10)
    print(
        f"  2D bilinear (mass + anisotropic stiffness), n={n} q={q}: OK  [4 GEMMs: ({2 * q},{n})x({n},{n}), ({2 * q},{n})x({n},{2 * q}), ({n},{2 * q})x({2 * q},{2 * q}), ({n},{2 * q})x({2 * q},{n})]"
    )


def check_mass_2d_panel(degree, element_batch, device, rng):
    interp, _deriv, w1d = operators(degree)
    n = degree + 1
    q = len(w1d)
    w2d = np.kron(w1d, w1d).reshape(q, q)

    num_elements = 2 * element_batch
    dofs = rng.standard_normal((num_elements, n * n))
    # pack: element e occupies columns [e*n, (e+1)*n), rows are the slow axis
    packed = np.ascontiguousarray(dofs.reshape(num_elements, n, n).transpose(1, 0, 2).reshape(n, num_elements * n))

    kernel = make_mass_2d_panel_kernel(n, q, element_batch, wp.float64)
    a_wp = wp.array(interp, dtype=wp.float64, device=device)
    w_wp = wp.array(w2d, dtype=wp.float64, device=device)
    packed_wp = wp.array(packed, dtype=wp.float64, device=device)
    out_wp = wp.zeros((num_elements, q * q), dtype=wp.float64, device=device)
    block_dim = 1 if wp.get_device(device).is_cpu else 64
    wp.launch_tiled(
        kernel,
        dim=[num_elements // element_batch],
        inputs=[a_wp, w_wp, packed_wp, out_wp],
        block_dim=block_dim,
        device=device,
    )
    wp.synchronize_device(device)

    bv = np.kron(interp, interp)
    expected = (dofs @ bv.T) * np.kron(w1d, w1d)[None, :]
    np.testing.assert_allclose(out_wp.numpy(), expected, atol=1e-10)
    print(f"  2D mass, E_b={element_batch} panel, n={n} q={q}: OK  [first GEMM: ({q},{n})x({n},{element_batch * n})]")


def check_mass_3d(degree, device, rng):
    interp, _deriv, w1d = operators(degree)
    n = degree + 1
    q = len(w1d)

    num_elements = 3
    dofs = rng.standard_normal(num_elements * n**3)

    # Weights in the rotated layout the kernel reaches: rows qz, cols (qx, qy)
    w_rot = np.einsum("z,x,y->zxy", w1d, w1d, w1d).reshape(-1)

    kernel = make_mass_3d_kernel(n, q, wp.float64)
    a_wp = wp.array(interp, dtype=wp.float64, device=device)
    w_wp = wp.array(w_rot, dtype=wp.float64, device=device)
    dofs_wp = wp.array(dofs, dtype=wp.float64, device=device)
    out_wp = wp.zeros(num_elements * n**3, dtype=wp.float64, device=device)
    block_dim = 1 if wp.get_device(device).is_cpu else 64
    wp.launch_tiled(
        kernel, dim=[num_elements], inputs=[a_wp, w_wp, dofs_wp, out_wp], block_dim=block_dim, device=device
    )
    wp.synchronize_device(device)

    bv = np.kron(np.kron(interp, interp), interp)
    w3d = np.einsum("x,y,z->xyz", w1d, w1d, w1d).reshape(-1)
    mass = bv.T @ np.diag(w3d) @ bv
    u = dofs.reshape(num_elements, -1)
    expected_lex = (u @ mass.T).reshape(num_elements, n, n, n)
    # the symmetric rotation pipeline returns layout rows y, cols (z, x)
    expected = expected_lex.transpose(0, 2, 3, 1).reshape(-1)

    np.testing.assert_allclose(out_wp.numpy(), expected, atol=1e-10)
    print(f"  3D mass via axis rotation, n={n} q={q}: OK  [6 GEMMs, widest: ({q},{n})x({n},{n * n})]")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    devices = ["cpu"] + (["cuda:0"] if wp.get_cuda_device_count() > 0 else [])
    for device in devices:
        print(f"device: {device}")
        for degree in (3, 4, 5):
            check_bilinear_2d(degree, device, rng)
            check_mass_2d_panel(degree, element_batch=4, device=device, rng=rng)
            check_mass_3d(degree, device, rng)
    print("all tile-native GEMM spikes passed")
