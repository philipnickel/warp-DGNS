# Sum-Factorization Benchmark Results — A100, 2026-06-10

Forward (`B`-stage) contraction vs naive dense Kronecker matvec.
`uv run warp/examples/fem/bench_sumfac_dg.py --dim 3`, float64, 16384 hex elements,
NVIDIA A100-PCIE-40GB (sm_80), CUDA 13.1, 50 timed repeats after warmup.

## 3D sweep (dim 3, float64)

| P | E_b | sumfac ms | naive ms | speedup | sumfac GF/s | naive GF/s |
|---|-----|-----------|----------|---------|-------------|------------|
| 1 | 4   | 0.078     | 0.022    | 0.28    | 20.2        | 95.7       |
| 2 | 8   | 0.087     | 0.059    | 0.68    | 91.4        | 403.4      |
| 3 | 4   | 0.150     | 0.823    | **5.5** | 167.6       | 163.2      |
| 4 | 1   | 0.144     | 0.787    | **5.5** | 427.3       | 650.7      |
| 5 | 2   | 0.293     | 4.142    | **14.1**| 435.1       | 369.1      |
| 6 | 1   | 0.360     | 11.281   | **31.3**| 655.2       | 341.7      |
| 7 | 1   | 0.484     | 30.616   | **63.3**| 832.0       | 280.6      |
| 8 | 1   | 0.701     | 43.171   | **61.6**| 920.0       | 403.4      |

(Best `E_b` per order shown; full sweep over `E_b ∈ {1, 2, 4, 8, 16}` in the bench logs.)

## Conclusions

- **Crossover ≈ P=3**, exactly as the design predicted (spec §3, Roget anchor): P ≤ 2 the naive
  kernel wins (0.3–0.7×) — the low-P fallback in `sumfac_applicable` is confirmed correct.
- **Target regime far exceeds expectations:** 14× at P=5 (Roget's reference: ~8.5×), growing to
  ~62× at P=7–8. The naive path's O((P+1)^{2d}) wall makes high order effectively free for the
  sum-factorized path: P=8 sum-fac (0.70 ms) costs less than naive P=3 (0.82 ms).
- **Best `E_b` is small and shrinks with order:** E_b=4–8 helps only near the crossover where the
  per-element GEMM RHS is narrow; at P ≥ 4 a single element already provides an n² ≥ 25-column
  RHS and wider panels only cost shared memory/occupancy (E_b=16 degrades everywhere; P=8/E_b=8
  exceeds the launchable shared-memory limit). Production default: `E_b = 1` for P ≥ 4,
  `E_b = 4` for P = 3.
- **FP64 tensor cores confirmed:** the P=8 contraction kernel's PTX contains 144
  `mma.sync.aligned.m8n8k4.row.col.f64` (DMMA) instructions and zero scalar `fma.rn.f64` — all
  GEMM flops are issued to tensor cores (hardware counters are admin-locked on this box, so the
  check is via generated code rather than `ncu`).
- Peak achieved: **920 GF/s f64** at P=8 (~9.4% of the A100's 9.7 TF/s FP64-FMA peak, ~4.7% of
  the 19.5 TF/s DMMA peak) on the B-stage alone; the fused BᵀDB kernel (Phase 3) raises
  arithmetic intensity further by keeping the D-stage in shared memory.

## Compile-time notes (development cost, not runtime)

- Each (n, q, E_b, dtype) kernel shape costs ~10–60 s of first-run CUDA compile (mathdx LTO +
  ptxas); compile time grows with E_b because the per-element loop fully unrolls.
- Two fixes keep the shape sweep linear and 20×+ faster than the initial state:
  `enable_backward=False` (3 GEMM LTOs → 1) and per-shape dynamic-module isolation
  (commit 312cee42; shared-module registration recompiled all earlier shapes on every addition).
