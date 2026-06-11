"""cuTile variants of the sum-factorized 2D DG cell apply (FP64).

Variants:
  b1      one element per block, 2D tiles (padded to powers of 2)
  eb{N}   tilegrid formulation: the element-blocked DOF array (E, np, np) IS
          the tile space; one block loads an (N, np, np) tile = N elements and
          the 1D operators broadcast across the batch via 3D matmul
  tg2d{N} literal 2D tilegrid: DOF array viewed as (Ey, np, Ex, np); one block
          loads a (By, np, Bx, np) tile (By*Bx = N elements with 2D locality)

Run with the cuTile venv: /tmp/cutile-venv/bin/python bench_cutile.py --help
"""

import argparse
import math

import cuda.tile as ct
import cupy as cp
import numpy as np
import oracle

# ---------------------------------------------------------------- kernels


@ct.kernel
def cell_apply_b1(u, r, opp, opt, wq, NP: ct.Constant[int], MP: ct.Constant[int]):
    e = ct.bid(0)
    U = ct.reshape(
        ct.load(u, (e, 0, 0), shape=(1, NP, NP), padding_mode=ct.PaddingMode.ZERO),
        (NP, NP),
    )
    OP = ct.load(opp, (0, 0), shape=(MP, NP), padding_mode=ct.PaddingMode.ZERO)
    OPT = ct.load(opt, (0, 0), shape=(NP, MP), padding_mode=ct.PaddingMode.ZERO)
    WQ = ct.load(wq, (0, 0), shape=(MP, MP), padding_mode=ct.PaddingMode.ZERO)
    T = OP @ U  # (MP, NP)  = [A;D] @ U
    S = T @ OPT  # (MP, MP)  quadrants [V, Vy; Vx, junk]
    F = S * WQ  # D stage: constant-geometry elementwise mask
    X = OPT @ F  # (NP, MP)
    R = X @ OP  # (NP, NP)
    ct.store(r, (e, 0, 0), ct.reshape(R, (1, NP, NP)))


@ct.kernel
def cell_apply_eb(u, r, opp, opt, wq, EB: ct.Constant[int], NP: ct.Constant[int], MP: ct.Constant[int]):
    b = ct.bid(0)
    U = ct.load(u, (b, 0, 0), shape=(EB, NP, NP), padding_mode=ct.PaddingMode.ZERO)
    OP = ct.load(opp, (0, 0), shape=(MP, NP), padding_mode=ct.PaddingMode.ZERO)
    OPT = ct.load(opt, (0, 0), shape=(NP, MP), padding_mode=ct.PaddingMode.ZERO)
    WQ = ct.load(wq, (0, 0), shape=(MP, MP), padding_mode=ct.PaddingMode.ZERO)
    T = OP @ U  # (EB, MP, NP): operator broadcast over the batch
    S = T @ OPT  # (EB, MP, MP)
    F = S * WQ  # WQ (MP, MP) broadcasts over the batch dim
    X = OPT @ F  # (EB, NP, MP)
    R = X @ OP  # (EB, NP, NP)
    ct.store(r, (b, 0, 0), R)


@ct.kernel
def cell_apply_tg2d(
    u4,
    r4,
    opp,
    opt,
    wq,
    BY: ct.Constant[int],
    BX: ct.Constant[int],
    NP: ct.Constant[int],
    MP: ct.Constant[int],
):
    by = ct.bid(0)
    bx = ct.bid(1)
    U4 = ct.load(u4, (by, 0, bx, 0), shape=(BY, NP, BX, NP), padding_mode=ct.PaddingMode.ZERO)
    U = ct.reshape(ct.permute(U4, (0, 2, 1, 3)), (BY * BX, NP, NP))
    OP = ct.load(opp, (0, 0), shape=(MP, NP), padding_mode=ct.PaddingMode.ZERO)
    OPT = ct.load(opt, (0, 0), shape=(NP, MP), padding_mode=ct.PaddingMode.ZERO)
    WQ = ct.load(wq, (0, 0), shape=(MP, MP), padding_mode=ct.PaddingMode.ZERO)
    T = OP @ U
    S = T @ OPT
    F = S * WQ
    X = OPT @ F
    R = X @ OP
    R4 = ct.permute(ct.reshape(R, (BY, BX, NP, NP)), (0, 2, 1, 3))
    ct.store(r4, (by, 0, bx, 0), R4)


@ct.kernel
def cell_apply_kron(ut, rt, k0, k1, k2, kt0, kt1, kt2, EB: ct.Constant[int], KP: ct.Constant[int]):
    # Axis-paired (Kronecker) formulation: 6 chained square GEMMs, no
    # elementwise stage (weights folded into the transposed operators).
    b = ct.bid(0)
    U = ct.load(ut, (0, b), shape=(KP, EB))
    K0 = ct.load(k0, (0, 0), shape=(KP, KP))
    K1 = ct.load(k1, (0, 0), shape=(KP, KP))
    K2 = ct.load(k2, (0, 0), shape=(KP, KP))
    KT0 = ct.load(kt0, (0, 0), shape=(KP, KP))
    KT1 = ct.load(kt1, (0, 0), shape=(KP, KP))
    KT2 = ct.load(kt2, (0, 0), shape=(KP, KP))
    B0 = K0 @ U
    B1 = K1 @ U
    B2 = K2 @ U
    acc = KT2 @ B2
    acc = ct.mma(KT1, B1, acc)
    acc = ct.mma(KT0, B0, acc)
    ct.store(rt, (0, b), acc)


def setup_kron(n: int, grid: int):
    from numpy.polynomial import legendre as leg  # noqa: PLC0415

    q = n
    h = 1.0 / grid
    A, D, op, WQ, mc = oracle.make_setup(n, q, h)
    _, qw = leg.leggauss(q)
    w2 = np.outer(qw, qw).flatten()
    K = [np.kron(A, A), np.kron(D, A), np.kron(A, D)]
    KT = [K[0].T * (mc * w2)[None, :], K[1].T * w2[None, :], K[2].T * w2[None, :]]

    n2 = n * n
    KP = oracle.pad_pow2(n2)
    E = grid * grid
    rng = np.random.default_rng(42)
    U = rng.standard_normal((E, n, n))
    ref = oracle.reference_apply(U, op, WQ)

    def padk(m):
        out = np.zeros((KP, KP))
        out[:n2, :n2] = m
        return cp.asarray(out)

    ut = np.zeros((KP, E))
    ut[:n2] = U.reshape(E, n2).T
    dev = {
        "ut": cp.asarray(ut),
        "rt": cp.zeros((KP, E)),
        "k0": padk(K[0]),
        "k1": padk(K[1]),
        "k2": padk(K[2]),
        "kt0": padk(KT[0]),
        "kt1": padk(KT[1]),
        "kt2": padk(KT[2]),
    }
    return dev, ref, KP, E


# ---------------------------------------------------------------- harness


def setup(n: int, grid: int, q: int = 0):
    q = q or n
    h = 1.0 / grid
    _, _, op, WQ, _ = oracle.make_setup(n, q, h)
    NP = oracle.pad_pow2(n)
    MP = oracle.pad_pow2(2 * q)
    E = grid * grid

    rng = np.random.default_rng(42)
    U = rng.standard_normal((E, n, n))

    u_pad = np.zeros((E, NP, NP))
    u_pad[:, :n, :n] = U
    opp = np.zeros((MP, NP))
    opp[: 2 * q, :n] = op
    wqp = np.zeros((MP, MP))
    wqp[: 2 * q, : 2 * q] = WQ

    dev = {
        "u": cp.asarray(u_pad),
        "r": cp.zeros((E, NP, NP)),
        "opp": cp.asarray(opp),
        "opt": cp.asarray(np.ascontiguousarray(opp.T)),
        "wq": cp.asarray(wqp),
    }
    ref = oracle.reference_apply(U, op, WQ)
    return dev, ref, NP, MP, E


def launch(variant: str, dev, NP, MP, E, grid, stream):
    if variant == "b1":
        ct.launch(stream, (E,), cell_apply_b1, (dev["u"], dev["r"], dev["opp"], dev["opt"], dev["wq"], NP, MP))
    elif variant.startswith("tg2d"):
        nb = int(variant[4:])
        side = int(math.isqrt(nb))
        assert side * side == nb, "tg2d batch must be a perfect square"
        # One-time contiguous (Ey, NP, Ex, NP) copies, built host-side (the
        # cupy device transpose would need an NVRTC kernel this venv lacks).
        if "u4" not in dev:
            u4 = cp.asnumpy(dev["u"]).reshape(grid, grid, NP, NP).transpose(0, 2, 1, 3)
            dev["u4"] = cp.asarray(np.ascontiguousarray(u4))
            dev["r4"] = cp.zeros(dev["u4"].shape)
        ct.launch(
            stream,
            (grid // side, grid // side),
            cell_apply_tg2d,
            (dev["u4"], dev["r4"], dev["opp"], dev["opt"], dev["wq"], side, side, NP, MP),
        )
    else:
        eb = int(variant[2:])
        ct.launch(
            stream,
            (E // eb,),
            cell_apply_eb,
            (dev["u"], dev["r"], dev["opp"], dev["opt"], dev["wq"], eb, NP, MP),
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5, help="1D dofs per element (P+1)")
    ap.add_argument("--q", type=int, default=0, help="quadrature points per axis (default n)")
    ap.add_argument("--grid", type=int, default=128)
    ap.add_argument("--variant", default="b1")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--warps", type=int, default=None, help="num_worker_warps hint (4 or 8)")
    ap.add_argument("--occupancy", type=int, default=None, help="occupancy hint [1,32]")
    args = ap.parse_args()

    if args.warps or args.occupancy:
        hints = {}
        if args.warps:
            hints["num_worker_warps"] = args.warps
        if args.occupancy:
            hints["occupancy"] = args.occupancy
        for kname in ("cell_apply_b1", "cell_apply_eb", "cell_apply_tg2d"):
            globals()[kname] = globals()[kname].replace_hints(**hints)

    stream = cp.cuda.get_current_stream()
    if args.variant.startswith("kron"):
        dev, ref, KP, E = setup_kron(args.n, args.grid)
        eb = int(args.variant[4:])
        kernel = cell_apply_kron
        if args.warps or args.occupancy:
            kernel = kernel.replace_hints(**hints)

        def do_launch():
            ct.launch(
                stream,
                (E // eb,),
                kernel,
                (dev["ut"], dev["rt"], dev["k0"], dev["k1"], dev["k2"], dev["kt0"], dev["kt1"], dev["kt2"], eb, KP),
            )
    else:
        dev, ref, NP, MP, E = setup(args.n, args.grid, args.q)

        def do_launch():
            launch(args.variant, dev, NP, MP, E, args.grid, stream)

    for _ in range(args.warmup):
        do_launch()
    stream.synchronize()

    if args.check:
        if args.variant.startswith("kron"):
            n2 = args.n * args.n
            got = cp.asnumpy(dev["rt"])[:n2].T.reshape(E, args.n, args.n)
        elif args.variant.startswith("tg2d"):
            r4 = cp.asnumpy(dev["r4"])  # (Ey, NP, Ex, NP)
            got = r4.transpose(0, 2, 1, 3).reshape(E, NP, NP)[:, : args.n, : args.n]
        else:
            got = cp.asnumpy(dev["r"])[:, : args.n, : args.n]
        err = np.abs(got - ref).max() / np.abs(ref).max()
        print(f"check {args.variant}: rel err {err:.3e}")
        assert err < 1e-13, "FAIL"

    start, stop = cp.cuda.Event(), cp.cuda.Event()
    start.record(stream)
    for _ in range(args.iters):
        do_launch()
    stop.record(stream)
    stop.synchronize()
    ms = cp.cuda.get_elapsed_time(start, stop) / args.iters
    gdofs = E * args.n * args.n / (ms * 1e-3) / 1e9
    print(f"{args.variant} n={args.n} grid={args.grid}: {ms:.4f} ms/apply  ({gdofs:.3f} GDOF/s)")


if __name__ == "__main__":
    main()
