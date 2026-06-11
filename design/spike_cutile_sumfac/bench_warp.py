"""Warp-tiles baseline for the sum-factorized 2D DG cell apply (FP64).

Mirrors the production sumfac volume kernel: one element per block, the
4-GEMM stacked-operator chain via wp.tile_matmul (cuBLASDx), native unpadded
shapes. Two D-stage variants:
  tile    elementwise tile multiply (best case: constant-geometry mask)
  scalar  per-QP scalar loop, block-redundant (mirrors the production seeded
          D stage's execution pattern, minus the integrand call overhead)

Run with: uv run bench_warp.py --help
"""

import argparse

import numpy as np
import oracle

import warp as wp

wp.set_module_options({"enable_backward": False})


def make_kernels(n: int, q: int):
    m = 2 * q
    nn = n * n

    @wp.kernel
    def cell_apply_tile(
        u: wp.array2d(dtype=wp.float64),
        r: wp.array2d(dtype=wp.float64),
        op_arr: wp.array2d(dtype=wp.float64),
        wq_arr: wp.array2d(dtype=wp.float64),
    ):
        e = wp.tid()
        U = wp.tile_reshape(wp.tile_load(u, shape=(1, nn), offset=(e, 0)), shape=(n, n))
        OP = wp.tile_load(op_arr, shape=(m, n))
        WQ = wp.tile_load(wq_arr, shape=(m, m))
        T = wp.tile_matmul(OP, U)
        S = wp.tile_matmul(T, wp.tile_transpose(OP))
        F = wp.tile_map(wp.mul, S, WQ)
        X = wp.tile_matmul(wp.tile_transpose(OP), F)
        R = wp.tile_matmul(X, OP)
        wp.tile_store(r, wp.tile_reshape(R, shape=(1, nn)), offset=(e, 0))

    @wp.kernel
    def cell_apply_scalar_d(
        u: wp.array2d(dtype=wp.float64),
        r: wp.array2d(dtype=wp.float64),
        op_arr: wp.array2d(dtype=wp.float64),
        wq_arr: wp.array2d(dtype=wp.float64),
    ):
        e = wp.tid()
        U = wp.tile_reshape(wp.tile_load(u, shape=(1, nn), offset=(e, 0)), shape=(n, n))
        OP = wp.tile_load(op_arr, shape=(m, n))
        T = wp.tile_matmul(OP, U)
        S = wp.tile_matmul(T, wp.tile_transpose(OP))
        # Block-redundant scalar D stage: every thread walks all QPs, like the
        # production seeded integrand loop.
        F = wp.tile_zeros(shape=(m, m), dtype=wp.float64)
        for i in range(q):
            for j in range(q):
                F[i, j] = wq_arr[i, j] * S[i, j]
                F[q + i, j] = wq_arr[q + i, j] * S[q + i, j]
                F[i, q + j] = wq_arr[i, q + j] * S[i, q + j]
        X = wp.tile_matmul(wp.tile_transpose(OP), F)
        R = wp.tile_matmul(X, OP)
        wp.tile_store(r, wp.tile_reshape(R, shape=(1, nn)), offset=(e, 0))

    return {"tile": cell_apply_tile, "scalar": cell_apply_scalar_d}


def make_wide_kernel(n: int, q: int, eb: int):
    """E_b wide-panel variant: all four stages are wide GEMMs over E_b
    elements; per-element transpose restaging between stages (contracted
    index kept leading, cf. the core reference's cyclic layouts)."""
    m = 2 * q
    neb = n * eb
    meb = m * eb

    @wp.kernel
    def cell_apply_wide(
        up: wp.array2d(dtype=wp.float64),  # (n, E*n) element panels
        rp: wp.array2d(dtype=wp.float64),  # (n, E*n) result panels (transposed blocks)
        op_arr: wp.array2d(dtype=wp.float64),
        wqt_arr: wp.array2d(dtype=wp.float64),  # (m, m*eb) WQ^T replicated
    ):
        b = wp.tid()
        OP = wp.tile_load(op_arr, shape=(m, n))
        OPT = wp.tile_transpose(OP)
        WQTC = wp.tile_load(wqt_arr, shape=(m, meb))

        Ucat = wp.tile_load(up, shape=(n, neb), offset=(0, b * neb))
        T = wp.tile_matmul(OP, Ucat)  # (m, n*eb): T_e column blocks

        # Restage: TT holds T_e^T blocks so stage 2 is a wide LEFT multiply.
        TT = wp.tile_zeros(shape=(n, meb), dtype=wp.float64)
        tmp1 = wp.tile_zeros(shape=(m, n), dtype=wp.float64)
        for e in range(eb):
            wp.tile_assign(tmp1, wp.tile_view(T, offset=(0, e * n), shape=(m, n)), offset=(0, 0))
            wp.tile_assign(TT, wp.tile_transpose(tmp1), offset=(0, e * m))
        ST = wp.tile_matmul(OP, TT)  # (m, m*eb): S_e^T blocks

        FT = wp.tile_map(wp.mul, ST, WQTC)

        # Restage: F_e blocks (un-transpose) so B^T stage is a left multiply.
        FC = wp.tile_zeros(shape=(m, meb), dtype=wp.float64)
        tmp2 = wp.tile_zeros(shape=(m, m), dtype=wp.float64)
        for e in range(eb):
            wp.tile_assign(tmp2, wp.tile_view(FT, offset=(0, e * m), shape=(m, m)), offset=(0, 0))
            wp.tile_assign(FC, wp.tile_transpose(tmp2), offset=(0, e * m))
        X = wp.tile_matmul(OPT, FC)  # (n, m*eb): X_e blocks

        XT = wp.tile_zeros(shape=(m, neb), dtype=wp.float64)
        tmp3 = wp.tile_zeros(shape=(n, m), dtype=wp.float64)
        for e in range(eb):
            wp.tile_assign(tmp3, wp.tile_view(X, offset=(0, e * m), shape=(n, m)), offset=(0, 0))
            wp.tile_assign(XT, wp.tile_transpose(tmp3), offset=(0, e * n))
        RT = wp.tile_matmul(OPT, XT)  # (n, n*eb): R_e^T blocks

        wp.tile_store(rp, RT, offset=(0, b * neb))

    return cell_apply_wide


def make_wide2_kernel(n: int, q: int, eb: int):
    """Lean wide-panel variant: 4 wide GEMMs, only TWO restages (B^T runs on
    transposed blocks directly since R^T = OP^T F^T OP has the same form),
    transposes taken directly from strided views (no tmp copies)."""
    m = 2 * q
    neb = n * eb
    meb = m * eb

    @wp.kernel
    def cell_apply_wide2(
        up: wp.array2d(dtype=wp.float64),
        rp: wp.array2d(dtype=wp.float64),
        op_arr: wp.array2d(dtype=wp.float64),
        wqt_arr: wp.array2d(dtype=wp.float64),
    ):
        b = wp.tid()
        OP = wp.tile_load(op_arr, shape=(m, n))
        OPT = wp.tile_transpose(OP)
        WQTC = wp.tile_load(wqt_arr, shape=(m, meb))

        Ucat = wp.tile_load(up, shape=(n, neb), offset=(0, b * neb))
        T = wp.tile_matmul(OP, Ucat)  # (m, neb): T_e blocks

        TT = wp.tile_zeros(shape=(n, meb), dtype=wp.float64)
        for e in range(eb):
            wp.tile_assign(
                TT,
                wp.tile_transpose(wp.tile_view(T, offset=(0, e * n), shape=(m, n))),
                offset=(0, e * m),
            )
        ST = wp.tile_matmul(OP, TT)  # (m, meb): S_e^T blocks
        FT = wp.tile_map(wp.mul, ST, WQTC)  # F_e^T blocks (mask is symmetric-free)

        G = wp.tile_matmul(OPT, FT)  # (n, meb): G_e = OP^T F_e^T
        GT = wp.tile_zeros(shape=(m, neb), dtype=wp.float64)
        for e in range(eb):
            wp.tile_assign(
                GT,
                wp.tile_transpose(wp.tile_view(G, offset=(0, e * m), shape=(n, m))),
                offset=(0, e * n),
            )
        R = wp.tile_matmul(OPT, GT)  # (n, neb): R_e = OP^T G_e^T (natural!)
        wp.tile_store(rp, R, offset=(0, b * neb))

    return cell_apply_wide2


def make_kron_kernel(n: int, eb: int):
    n2 = n * n

    @wp.kernel
    def cell_apply_kron(
        ut: wp.array2d(dtype=wp.float64),
        rt: wp.array2d(dtype=wp.float64),
        k_ops: wp.array3d(dtype=wp.float64),
        kt_ops: wp.array3d(dtype=wp.float64),
    ):
        b = wp.tid()
        U = wp.tile_load(ut, shape=(n2, eb), offset=(0, b * eb))
        R = wp.tile_zeros(shape=(n2, eb), dtype=wp.float64)
        for c in range(3):
            Kc = wp.tile_load(k_ops[c], shape=(n2, n2))
            KTc = wp.tile_load(kt_ops[c], shape=(n2, n2))
            Bc = wp.tile_matmul(Kc, U)
            wp.tile_matmul(KTc, Bc, R)  # accumulate
        wp.tile_store(rt, R, offset=(0, b * eb))

    return cell_apply_kron


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--grid", type=int, default=128)
    ap.add_argument("--variant", default="tile")
    ap.add_argument("--eb", type=int, default=64, help="kron variant: elements per panel")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--block-dim", type=int, default=64)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    wp.init()
    device = "cuda:0"
    n, q, grid = args.n, args.n, args.grid
    E = grid * grid
    _, _, op, WQ, _ = oracle.make_setup(n, q, 1.0 / grid)

    rng = np.random.default_rng(42)
    U = rng.standard_normal((E, n, n))
    ref = oracle.reference_apply(U, op, WQ)

    if args.variant in ("wide", "wide2"):
        eb = args.eb
        up = np.ascontiguousarray(U.transpose(1, 0, 2).reshape(n, E * n))
        up_d = wp.array(up, dtype=wp.float64, device=device)
        rp_d = wp.zeros((n, E * n), dtype=wp.float64, device=device)
        op_d = wp.array(op, dtype=wp.float64, device=device)
        wqt_d = wp.array(np.ascontiguousarray(np.tile(WQ.T, (1, eb))), dtype=wp.float64, device=device)
        maker = make_wide2_kernel if args.variant == "wide2" else make_wide_kernel
        kernel = maker(n, q, eb)

        def apply_once():
            wp.launch_tiled(
                kernel,
                dim=[E // eb],
                inputs=[up_d, rp_d, op_d, wqt_d],
                block_dim=args.block_dim,
                device=device,
            )
    elif args.variant == "kron":
        from numpy.polynomial import legendre as leg  # noqa: PLC0415

        _, qw = leg.leggauss(q)
        w2 = np.outer(qw, qw).flatten()
        mc = (1.0 / grid) ** 2 / 4.0
        A, D = op[:q], op[q:]
        K = np.stack([np.kron(A, A), np.kron(D, A), np.kron(A, D)])
        KT = np.stack([K[0].T * (mc * w2)[None, :], K[1].T * w2[None, :], K[2].T * w2[None, :]])
        ut_d = wp.array(np.ascontiguousarray(U.reshape(E, n * n).T), dtype=wp.float64, device=device)
        rt_d = wp.zeros((n * n, E), dtype=wp.float64, device=device)
        k_d = wp.array(K, dtype=wp.float64, device=device)
        kt_d = wp.array(KT, dtype=wp.float64, device=device)
        kernel = make_kron_kernel(n, args.eb)

        def apply_once():
            wp.launch_tiled(
                kernel,
                dim=[E // args.eb],
                inputs=[ut_d, rt_d, k_d, kt_d],
                block_dim=args.block_dim,
                device=device,
            )
    else:
        u_d = wp.array(U.reshape(E, n * n), dtype=wp.float64, device=device)
        r_d = wp.zeros((E, n * n), dtype=wp.float64, device=device)
        op_d = wp.array(op, dtype=wp.float64, device=device)
        wq_d = wp.array(WQ, dtype=wp.float64, device=device)
        kernel = make_kernels(n, q)[args.variant]

        def apply_once():
            wp.launch_tiled(
                kernel,
                dim=[E],
                inputs=[u_d, r_d, op_d, wq_d],
                block_dim=args.block_dim,
                device=device,
            )

    for _ in range(args.warmup):
        apply_once()
    wp.synchronize_device(device)

    if args.check:
        if args.variant == "wide2":
            # rp holds R_e blocks in natural layout
            got = rp_d.numpy().reshape(n, E, n).transpose(1, 0, 2)
        elif args.variant == "wide":
            # rp holds R_e^T column blocks: (n, E*n) -> (E, n, n) untransposed
            got = rp_d.numpy().reshape(n, E, n).transpose(1, 2, 0)
        elif args.variant == "kron":
            got = rt_d.numpy().T.reshape(E, n, n)
        else:
            got = r_d.numpy().reshape(E, n, n)
        err = np.abs(got - ref).max() / np.abs(ref).max()
        print(f"check warp-{args.variant}: rel err {err:.3e}")
        assert err < 1e-13, "FAIL"

    start = wp.Event(enable_timing=True)
    stop = wp.Event(enable_timing=True)
    wp.record_event(start)
    for _ in range(args.iters):
        apply_once()
    wp.record_event(stop)
    wp.synchronize_device(device)
    ms = wp.get_event_elapsed_time(start, stop) / args.iters
    gdofs = E * n * n / (ms * 1e-3) / 1e9
    print(f"warp-{args.variant} n={n} grid={grid} block_dim={args.block_dim}: {ms:.4f} ms/apply  ({gdofs:.3f} GDOF/s)")


if __name__ == "__main__":
    main()
