# Phase 6 stage 2 — side-apply performance analysis (RTX 5090, 2026-06-10)

Measured (Grid2D 128x128, P=4, f64, SIP side apply, 50 iters): sumfac side apply **5.07 ms**
vs 6.57 ms naive matrix-free and 0.365 ms assembled `bsr_mv` (280 MB matrix). NCU: SM 82%,
DRAM 0.03%, smem 69%, occupancy 50% (register-limited). Correct to 6.8e-16; zero storage.

## Why it is slow (it is too *cheap*, not too expensive)

1. **Work-to-overhead ratio inverted vs the volume kernel.** Per 2D P=4 cell the side kernel
   does ~2600 useful MACs (volume: ~3000) but split over **24 tile_matmul calls** with
   height-2 operands (the (2,n) endpoint operator), vs the volume's 4 calls at (10,5)x(5,10).
   Each cooperative GEMM has fixed cost (smem staging, barriers, cuBLASDx prologue) that a
   5-number face contraction cannot amortize.
2. **The SIP D stage is heavy and warp-redundant.** Per face QP: normal, measure_ratio,
   element_measure, two J^-1 transforms, injected trace struct, then 3-6 seeded integrand
   evals, each re-running the jump/average channel combinations and re-decoding grid side
   geometry internally (`_get_side_from_grid` ~10+ times per QP). All block threads execute
   this identical scalar stream — zero intra-block parallelism. That is the NCU signature:
   SM busy on integer/address math and barriers, not FLOPs.
3. **Gather doubles interior flux evaluations** (accepted, race-free) — a 2x on the dominant
   D stage.
4. **The competitor is pure bandwidth**: `bsr_mv` streams the matrix at ~40% DRAM with no
   math. Beating 0.365 ms needs ~300 ns/cell total; the math content allows it, the current
   call structure does not.

## Fix plan, by leverage

1. **Stack faces into wide GEMMs**: all 2d endpoint rows in one (2*2d, n) operator; one
   batched tangential contraction per stage. 24 calls -> ~4 well-shaped ones.
2. **E_b > 1 multi-cell panels**: faces are exactly where cross-element batching pays —
   single-face shapes are degenerate. (Reuses the project's core E_b design goal.)
3. **Hoist side-constant geometry out of the seeded evals**: on grids, normal/measure_ratio
   are constant per side; evaluate once per face, pass into channel evals. Requires a
   side-aware Q-function contract instead of re-running the unmodified integrand per seed.
4. **block_dim 32** for the side kernel (halves redundant lanes; new LTO set) and **plan/arg
   caching** to cut the ~0.46 ms per-apply host floor (measured at res 8).

## Where the balance shifts without any of this

Naive side apply pays O(n^d) per QP (explodes with P); the assembled matrix grows
~n^{2(d-1)} per side plus multi-second assembly; the sumfac face cost is nearly flat in P,
and 3D faces are 2D objects with decently shaped GEMMs. The 2D P=4 case measured here is the
worst case for the current structure. The footprint win stands as measured: ~3 MB of face
traffic vs a 280 MB matrix.

## Stage-2 review follow-ups (adversarial review 2026-06-10: zero correctness bugs; these are queued)

Test coverage (cheap, no new tile shapes): a side form with tangential gradient content
(every committed form dots gradients with the axis-aligned normal, so tangential trace/lift
paths carry exactly zero); a position-dependent side coefficient (symmetric Gauss + constant
integrands make a *consistent* flip error an invisible QP relabeling); the test-field
partial-partition raise branch and the trace-introspection branch of _field_requiring_grad.
Kernel (perf round, in-design): the ADDENDUM's stacked opposing-face operators (still thin
per-face GEMMs); skip inactive faces with block-uniform branches (BoundarySides wastes
~(2d-1)/(2d) of B/D work); don't compute identical inner/outer traces twice on boundary
faces; move block-uniform per-face metadata out of per-thread registers; comment/guard the
(sound) full-width-row-block tile_view convention at its 8 call sites; replace the
wp.min(2, cell_index+2) unroll-defeat with a plain int kernel arg; add domain.name directly
to the cache suffix (currently discriminated only indirectly via quadrature.name).

## Core reference (added 2026-06-10)

Tu, Karlin, Camier, Dobrev, Kolev, Henneking, Ghattas, **"Accelerating High-Order Finite
Element Simulations at Extreme Scale with FP64 Tensor Cores"**, arXiv:2603.09038 (MFEM /
2025 Gordon Bell tsunami digital twin; GH200/GB200, 9,216 GPUs). THE core reference for this
project. What it establishes, and how it re-ranks our plan:

- **Validates the architecture end to end**: fused sum-factorized B^T D B on FP64 DMMA gives
  2x (35-59% from DMMA alone, the rest from fusion) at p=3/4 with our exact GEMM shapes
  (m25n5k4). Mechanism: shared-memory DATA MOTION is the bottleneck, not FLOPs (baseline 97%
  LSU wavefronts vs 14% FP64 pipe; DMMA cuts smem reads 4.6x, 9000->1960 B per GEMM). They
  reject CUTLASS/cuBLAS padding for O(10) GEMMs and name cuBLASDx as the future alternative
  to their inline PTX -- i.e. our Warp-tiles/cuBLASDx stack is the layer they predicted.
- **What it does NOT cover (our open territory)**: no interior DG faces at all (mixed
  Galerkin, boundary integrals only); no multi-element batching into MMA tiles (they accept
  ~49% instruction-lane waste at m25n5 -- the waste E_b panels exist to eliminate); no A100
  data; no order sweep; no DMMA-vs-FMA rounding analysis.
- **Re-ranked side-kernel fix plan (evidence-weighted)**: (1) E_b multi-cell panels +
  stacked opposing-face operators (their admitted lane waste + "discretize to fit the
  tensor-core architecture" make tile-filling the dominant lever; our (2,n) GEMMs waste >75%
  of the m-extent); (2) fused/stacked side stages (fusion was half their 2x); (3) hoist
  side-constant geometry PA-style -- their Fused-PA beats Fused-MF in runtime DESPITE worse
  roofline position ("gains in FLOP/s efficiency are outweighed by the additional FLOPs"),
  which also cautions the deep-GEMM bilinear: judge by ms/apply, never by utilization;
  (4) block_dim/redundancy (their GEMMs use 4 cooperating warps).
- **New levers**: cyclic tensor-index reordering (contracted index always fastest-changing;
  removes transposes and makes stages bank-conflict-free) for tensor_contract staging;
  Nsight diagnostic pair "L1: Data Pipe Lsu Wavefronts" vs "SM: Pipe Fp64/Dmma Cycles
  Active" for the A100 pass; report GDOF/s and MDOF/Watt (capture power via NVML).
- **A100 expectation setting**: A100 FP64-TC peak is 19.5 TFLOP/s (~3.4x below H100's 67);
  related work (CEED-MS40) saw DMMA gains on A100 "mainly for orders above 10" -- at P=4
  the win must come from filling tiles (E_b/stacking), not from DMMA per se.
