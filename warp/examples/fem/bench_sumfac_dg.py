# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark the sum-factorized tensor-product contraction against a naive dense baseline.

Times the ``E_b``-wide ``tile_matmul`` interpolation kernels from
:mod:`warp._src.fem.sumfac.tensor_contract` (the ``B`` stage of the ``B^T D B``
factorization) against a naive per-thread dense Kronecker matrix-vector kernel,
sweeping polynomial order ``P`` and the panel width ``E_b``.

The naive baseline applies the dense ``(q^d, n^d)`` operator row by row (one
thread per element and output point), which is the same arithmetic the existing
non-factorized integration kernels perform per quadrature point.

Run on a CUDA machine (the tile path engages cuBLASDx / tensor cores there)::

    uv run warp/examples/fem/bench_sumfac_dg.py --dim 3 --orders 1 2 3 4 5 6 7 8
    uv run warp/examples/fem/bench_sumfac_dg.py --compile-only   # prewarm LTO cache

Timing assertions deliberately do not exist here; unit tests must not time
(see the sum-factorized DG design doc, section 8).
"""

import argparse
import time

import numpy as np

import warp as wp
from warp._src.fem.sumfac.operators_1d import default_quadrature_points
from warp._src.fem.sumfac.tensor_contract import (
    _default_block_dim,
    build_operator_arrays,
    make_interpolation_kernel_2d,
    make_interpolation_kernel_3d,
    pack_dofs_2d,
    pack_dofs_3d,
)

_naive_kernel_cache = {}


def _make_naive_kernel(dtype):
    """Build (and cache) the naive dense matvec kernel ``out[e, r] = sum_c op[r, c] dofs[e, c]``."""
    key = dtype
    if key in _naive_kernel_cache:
        return _naive_kernel_cache[key]

    @wp.kernel
    def naive_kernel(
        op: wp.array2d(dtype=dtype),
        dofs: wp.array2d(dtype=dtype),
        out: wp.array2d(dtype=dtype),
    ):
        e, r = wp.tid()
        acc = op.dtype(0.0)
        for c in range(op.shape[1]):
            acc += op[r, c] * dofs[e, c]
        out[e, r] = acc

    _naive_kernel_cache[key] = naive_kernel
    return naive_kernel


def _time_launches(launch, device, warmup: int, repeats: int) -> float:
    """Return the average seconds per launch using host timing around device sync."""
    for _ in range(warmup):
        launch()
    wp.synchronize_device(device)

    start = time.perf_counter()
    for _ in range(repeats):
        launch()
    wp.synchronize_device(device)
    return (time.perf_counter() - start) / repeats


def bench_config(dim, degree, element_batch, num_elements, wp_dtype, np_dtype, device, warmup, repeats, compile_only):
    """Benchmark one (dim, degree, E_b) configuration; return a result dict or None."""
    n = degree + 1
    q = len(default_quadrature_points(degree))
    interp, _deriv = build_operator_arrays(degree, dtype=np_dtype)

    # Round element count down to a whole number of panels.
    num_panels = max(1, num_elements // element_batch)
    num_elements = num_panels * element_batch

    rng = np.random.default_rng(42)
    dofs = rng.standard_normal((num_elements, n**dim)).astype(np_dtype)

    op_wp = wp.array(np.ascontiguousarray(interp), dtype=wp_dtype, device=device)

    if dim == 2:
        packed = pack_dofs_2d(dofs, n)
        kernel = make_interpolation_kernel_2d(n, q, element_batch, wp_dtype)
        inputs = [op_wp, op_wp]
        # Per element: first contraction 2*q*n*n, second 2*q*q*n.
        sumfac_flops = 2 * q * n * (n + q)
    else:
        packed = pack_dofs_3d(dofs, n)
        kernel = make_interpolation_kernel_3d(n, q, element_batch, wp_dtype)
        inputs = [op_wp, op_wp, op_wp]
        sumfac_flops = 2 * q * n * n * n + 2 * q * q * n * n + 2 * q * q * q * n

    packed_wp = wp.array(packed, dtype=wp_dtype, device=device)
    out_wp = wp.zeros((num_elements, q**dim), dtype=wp_dtype, device=device)
    block_dim = _default_block_dim(device)

    def launch_sumfac():
        wp.launch_tiled(
            kernel,
            dim=[num_panels],
            inputs=[*inputs, packed_wp, out_wp],
            block_dim=block_dim,
            device=device,
        )

    # Naive baseline: dense Kronecker operator applied row by row.
    op_dense = interp
    for _ in range(dim - 1):
        op_dense = np.kron(op_dense, interp)
    op_dense_wp = wp.array(np.ascontiguousarray(op_dense.astype(np_dtype)), dtype=wp_dtype, device=device)
    dofs_wp = wp.array(dofs, dtype=wp_dtype, device=device)
    out_naive_wp = wp.zeros((num_elements, q**dim), dtype=wp_dtype, device=device)
    naive_kernel = _make_naive_kernel(wp_dtype)
    naive_flops = 2 * (q**dim) * (n**dim)

    def launch_naive():
        wp.launch(
            naive_kernel,
            dim=(num_elements, q**dim),
            inputs=[op_dense_wp, dofs_wp, out_naive_wp],
            device=device,
        )

    # Correctness cross-check (also forces compilation).
    launch_sumfac()
    launch_naive()
    wp.synchronize_device(device)
    if compile_only:
        return None

    atol = 1e-10 if np_dtype == np.float64 else 1e-3
    np.testing.assert_allclose(out_wp.numpy(), out_naive_wp.numpy(), atol=atol)

    t_sumfac = _time_launches(launch_sumfac, device, warmup, repeats)
    t_naive = _time_launches(launch_naive, device, warmup, repeats)

    return {
        "dim": dim,
        "P": degree,
        "E_b": element_batch,
        "elements": num_elements,
        "t_sumfac_ms": t_sumfac * 1e3,
        "t_naive_ms": t_naive * 1e3,
        "speedup": t_naive / t_sumfac,
        "sumfac_gflops": num_elements * sumfac_flops / t_sumfac / 1e9,
        "naive_gflops": num_elements * naive_flops / t_naive / 1e9,
        "elem_per_s": num_elements / t_sumfac,
    }


def main():
    parser = argparse.ArgumentParser(description="Sum-factorization E_b sweep benchmark.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dim", type=int, default=3, choices=(2, 3))
    parser.add_argument("--orders", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8])
    parser.add_argument("--eb", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--dtype", default="float64", choices=("float32", "float64"))
    parser.add_argument("--elements", type=int, default=1 << 14)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--compile-only", action="store_true", help="Compile all kernels, skip timing.")
    args = parser.parse_args()

    wp.init()
    np_dtype = np.float32 if args.dtype == "float32" else np.float64
    wp_dtype = wp.float32 if args.dtype == "float32" else wp.float64

    header = f"{'dim':>3} {'P':>2} {'E_b':>4} {'elems':>7} {'sumfac ms':>10} {'naive ms':>9} {'speedup':>8} {'sf GF/s':>8} {'nv GF/s':>8}"
    print(header)
    print("-" * len(header))
    for degree in args.orders:
        for eb in args.eb:
            try:
                r = bench_config(
                    args.dim,
                    degree,
                    eb,
                    args.elements,
                    wp_dtype,
                    np_dtype,
                    args.device,
                    args.warmup,
                    args.repeats,
                    args.compile_only,
                )
            except Exception as exc:
                print(f"{args.dim:>3} {degree:>2} {eb:>4}  FAILED: {type(exc).__name__}: {exc}")
                continue
            if r is None:
                print(f"{args.dim:>3} {degree:>2} {eb:>4}  compiled")
                continue
            print(
                f"{r['dim']:>3} {r['P']:>2} {r['E_b']:>4} {r['elements']:>7}"
                f" {r['t_sumfac_ms']:>10.3f} {r['t_naive_ms']:>9.3f} {r['speedup']:>8.2f}"
                f" {r['sumfac_gflops']:>8.1f} {r['naive_gflops']:>8.1f}"
            )


if __name__ == "__main__":
    main()
