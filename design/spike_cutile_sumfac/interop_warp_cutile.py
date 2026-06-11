"""End-to-end Warp <-> cuTile interop: sum-factorized DG cell apply.

All device storage is allocated as wp.array (zero-copy into cuTile via
__cuda_array_interface__); the kernel launches on Warp's own CUDA stream
(raw CUstream handle); results are checked against the NumPy oracle.

Run with: uv run --with "cuda-tile[tileiras]" interop_warp_cutile.py
"""

import cuda.tile as ct
import numpy as np
import oracle

import warp as wp


@ct.kernel
def cell_apply_eb(u, r, opp, opt, wq, EB: ct.Constant[int], NP: ct.Constant[int], MP: ct.Constant[int]):
    b = ct.bid(0)
    U = ct.load(u, (b, 0, 0), shape=(EB, NP, NP), padding_mode=ct.PaddingMode.ZERO)
    OP = ct.load(opp, (0, 0), shape=(MP, NP), padding_mode=ct.PaddingMode.ZERO)
    OPT = ct.load(opt, (0, 0), shape=(NP, MP), padding_mode=ct.PaddingMode.ZERO)
    WQ = ct.load(wq, (0, 0), shape=(MP, MP), padding_mode=ct.PaddingMode.ZERO)
    T = OP @ U
    S = T @ OPT
    F = S * WQ
    X = OPT @ F
    R = X @ OP
    ct.store(r, (b, 0, 0), R)


def main():
    wp.init()
    device = "cuda:0"
    n = q = 5
    grid = 128
    eb = 4
    E = grid * grid
    NP = oracle.pad_pow2(n)
    MP = oracle.pad_pow2(2 * q)
    _, _, op, WQ, _ = oracle.make_setup(n, q, 1.0 / grid)

    rng = np.random.default_rng(42)
    U = rng.standard_normal((E, n, n))
    ref = oracle.reference_apply(U, op, WQ)

    u_pad = np.zeros((E, NP, NP))
    u_pad[:, :n, :n] = U
    opp = np.zeros((MP, NP))
    opp[: 2 * q, :n] = op
    wqp = np.zeros((MP, MP))
    wqp[: 2 * q, : 2 * q] = WQ

    # Device storage: Warp arrays only.
    u_d = wp.array(u_pad, dtype=wp.float64, device=device)
    r_d = wp.zeros((E, NP, NP), dtype=wp.float64, device=device)
    opp_d = wp.array(opp, dtype=wp.float64, device=device)
    opt_d = wp.array(np.ascontiguousarray(opp.T), dtype=wp.float64, device=device)
    wq_d = wp.array(wqp, dtype=wp.float64, device=device)

    # Warp's stream, as the raw CUstream handle cuTile accepts.
    stream = wp.get_stream(device).cuda_stream

    ct.launch(stream, (E // eb,), cell_apply_eb, (u_d, r_d, opp_d, opt_d, wq_d, eb, NP, MP))
    wp.synchronize_device(device)

    got = r_d.numpy()[:, :n, :n]
    err = np.abs(got - ref).max() / np.abs(ref).max()
    print(f"warp->cutile->warp cell apply (eb={eb}, P={n - 1}): rel err {err:.3e}")
    assert err < 1e-13, "FAIL"

    # Round-trip back into a Warp kernel to prove shared ownership.
    @wp.kernel
    def scale_inplace(a: wp.array3d(dtype=wp.float64), s: wp.float64):
        i, j, k = wp.tid()
        a[i, j, k] = a[i, j, k] * s

    wp.launch(scale_inplace, dim=r_d.shape, inputs=[r_d, wp.float64(2.0)], device=device)
    wp.synchronize_device(device)
    err2 = np.abs(r_d.numpy()[:, :n, :n] - 2.0 * ref).max() / np.abs(ref).max()
    print(f"warp kernel on cuTile-written array: rel err {err2:.3e}")
    assert err2 < 1e-13, "FAIL"
    print("PASS - zero-copy wp.array interop, shared stream, both directions")


if __name__ == "__main__":
    main()
