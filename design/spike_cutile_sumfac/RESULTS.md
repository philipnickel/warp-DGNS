# cuTile vs Warp-tiles spike — sum-factorized 2D DG cell apply (2026-06-11)

Hardware: RTX 5090 (sm_120, FP64 1/64 rate, no FP64 tensor cores), driver 580.95.05.
Software: cuda-tile 1.4.0 + tileiras 13.3.36 (`/tmp/cutile-venv`); Warp 1.15.0.dev0
(branch `philipnickel/sumfac-cutile` @ upstream/main ec3c429d); NCU 2025.3.1.
Problem: mass + Laplace stiffness cell apply, Grid2D 128x128 (E=16384), FP64,
affine square elements (D stage = constant elementwise mask = best case for both
sides). Same 4-GEMM stacked-operator chain `op@U, T@op^T, F=S*WQ, op^T@F, X@op`
in every variant; all variants bit-consistent with the NumPy oracle (rel err
~2-3e-16). Timing via CUDA events over 100-200 iters after warmup.

## The tilegrid formulation (hypothesis under test)

The element-blocked DOF array `(E, np, np)` IS a cuTile tile space: one
`ct.load` with tile shape `(E_b, np, np)` fetches an element batch, and the 1D
operators broadcast across the batch via 3D `matmul` — E_b multi-element
batching becomes a tile-shape choice instead of kernel surgery (contrast: the
Warp sumfac fused kernels still raise NotImplementedError for E_b > 1). The
literal 2D tilegrid `(Ey, np, Ex, np)` with 4D tile loads + in-kernel
`ct.permute` also works and is bit-exact; it matches 1D batching for cell apply
(no neighbor access) but is the natural shape for future face terms.
**Hypothesis confirmed functionally.**

## Timings (ms/apply; GDOF/s in parens)

P=4 (n=q=5; cuTile tiles padded 5->8, 10->16):

| variant | ms/apply | note |
|---|---|---|
| warp-tile bd64 | **0.0715** (5.73) | best overall; elementwise D stage |
| warp-tile bd32 | 0.1058 | |
| warp-scalar-D bd64 | 0.1679 (2.44) | production-style block-redundant D stage |
| cutile b1 | 0.4037 | one element/block |
| cutile eb4 | 0.2933 (1.40) | best cuTile; 0.2690 with occupancy=4 hint |
| cutile eb8/eb16/eb32 | 0.294 / 0.318 / 0.367 | |
| cutile tg2d4/tg2d16/tg2d64 | 0.296 / 0.314 / 0.354 | 2D tilegrid = 1D batching perf |

P=3 (n=q=4; exact powers of two, zero padding):

| variant | ms/apply | note |
|---|---|---|
| warp-tile bd64 | **0.0310** (8.45) | |
| cutile b1 | 0.2166 | |
| cutile eb8/eb16 | 0.0516 / **0.0494** (5.31) | batching pays 4.4x over b1 |

Hint sweep (`replace_hints`): `occupancy=4` gives 8% on eb4 n=5; `num_worker_warps`
4 vs 8 indistinguishable.

P=7 (n=q=8; native powers of two) and the padding-axis decomposition (added
2026-06-11, same session):

| variant | ms/apply | note |
|---|---|---|
| warp-tile n=8 bd64 | **0.1113** (9.42) | |
| cutile b1 n=8 | 0.4040 | |
| cutile eb4 n=8 | **0.2906** (3.61) | = padded-n=5 wall clock exactly |
| cutile eb4 n=5 q=8 | 0.2915 | over-integration: same cost as zero-pad |

The P=4 padding waste decomposes as 1.6 (M: 2q 10->16) x 1.6 (K: n 5->8) x
1.6 (N: n 5->8) = 4.1x. The three axis classes differ in fixability:
- **Batch axes** (elements/faces): align for free — pick E_b a power of 2
  (the eb variants already do).
- **Quadrature axis (M)**: pad with REAL quadrature points — q=8
  over-integration measured at identical runtime to zero-padded q=5
  (0.2915 vs 0.2933 ms). Tile shapes fix the cost; over-integration converts
  dead rows into accuracy. Never zero-pad this axis.
- **Basis axes (K, N)**: fixed by the element (n = P+1); padding is pure
  waste and is exactly equivalent to running the next pow2 order: padded
  P=4 and native P=7 have identical wall clock (0.2933 vs 0.2906 ms) while
  P=7 carries 2.56x the DOFs. Under cuTile's constraint the rational orders
  are P=3 and P=7 ("discretize to fit the architecture", per the core
  reference). Cross-element line packing could in principle densify N, but
  element width 5 never aligns with pow2 tile spaces — stage-2 regrouping
  would need gathers or a multi-kernel chain for at most ~1.6x recovery.

## NCU (one apply, curated metrics)

Useful MACs/apply: n=5: 24.6M; n=4: 12.6M.

| kernel | dur (under NCU) | DFMA | DMUL | DADD | regs/thr | smem LSU wavefronts |
|---|---|---|---|---|---|---|
| warp-tile n=5 | 105 us | 62.9M | 2.1M | 0 | 62 | 5.04M |
| warp-scalar-D n=5 | 248 us | 62.9M | 78.6M | 0 | 80 | 13.5M |
| cutile eb4 n=5 | 430 us | 0 | 104.9M | 110.1M | 255 | 11.3M |
| warp-tile n=4 | 46 us | 25.2M | 1.05M | 0 | 48 | 3.90M |
| cutile eb8 n=4 | 74 us | 0 | 13.6M | 14.9M | 255 | 1.82M |

## E_b wide-panel Warp variant — NEW FASTEST (added same session)

All four sumfac stages as wide GEMMs over E_b elements — (10,5)@(5,5Eb),
(10,5)@(5,10Eb), (5,10)@(10,10Eb), (5,10)@(10,5Eb) — with per-element
transpose restaging between stages (tile_view -> tile_assign of transposed
blocks; the strided views feed only tile_assign, never tile_matmul, per the
production convention). Element panels stored (n, E*n) host-side; D stage is
one wide tile_map against a host-replicated WQ^T panel. Bit-exact (3.4e-16).

| config | ms/apply | NCU |
|---|---|---|
| wide eb4 bd64 | **0.0611** (6.70 GDOF/s) | 41.9M DFMA (pad 1.70x), SM 71%, 110 regs |
| wide eb2 bd64 | 0.0638 | |
| wide eb8 bd64 / eb4 bd128 / eb4 bd32 | 0.086 / 0.092 / 0.103 | restaging + smem pressure |
| tile per-element (prev best) | 0.0715 | 65.0M DFMA (pad 2.64x), SM 91% |

E_b panels did exactly what the theory said: cuBLASDx padding fell 2.64x ->
1.70x (65.0M -> 41.9M DFMA, -36% instructions), buying -15% wall despite the
in-kernel restaging costing 20 points of SM efficiency (91% -> 71%).

### NCU-driven iteration 2: wide2 (lean restaging) — 0.0536 ms

Stall profile of wide eb4 named the bottlenecks: short_scoreboard 5.18 (smem
load latency from restaging) + wait 5.41 (FP64 dependency chains) at only
1.97 active warps/scheduler. Fixes in `wide2`:
- Algebraic: R^T = OP^T F^T OP has the same form as the forward stage, so
  B^T runs directly on the transposed F blocks — 3 restages -> 2, two live
  tiles eliminated, result lands in natural layout.
- Transposes taken directly from strided tile_views into tile_assign (no tmp
  copies) — verified bit-exact (4.6e-16), so transpose-of-view + assign is
  sound (matmul remains the only consumer that corrupts on strided views).

| config | ms/apply | NCU |
|---|---|---|
| wide2 eb4 bd64, grid 128 | **0.0536** (7.65 GDOF/s) | SM 79.3%, LSU 3.98M (was 5.23M), 103 regs |
| wide2 eb4 bd64, grid 256 | 0.1976 (**8.29 GDOF/s**) | 128^2 was partially tail-bound |
| wide2 eb8 / eb2 / bd32 / bd128 | 0.062 / 0.062 / 0.080 / 0.090 | eb4+bd64 is the optimum |

Cumulative NCU-driven round: 0.0715 -> 0.0611 -> 0.0536 ms (-25%), instruction
count -36%, smem traffic -24%. Remaining stalls are still smem latency + FP64
waits at ~2 warps/scheduler; the structural exits are (a) occupancy via
smaller live-tile footprint, and (b) 3D slabs, where restage cost per element
is O(m*n) while GEMM work grows ~n^3 — the restaging amortizes much better at
m25-class shapes. ~27 us remains the perfect-padding ceiling.

## Axis-pairing (Kronecker) experiment — larger regular GEMMs (added same session)

Pair both 2D axes into one operator per channel: K_c in {A(x)A, D(x)A, A(x)D}
(25,25), weights folded into K_c^T host-side. The whole apply = 6 chained
square GEMMs (25,25)@(25,E_b), no elementwise stage, no inter-stage
restaging, element batch lands directly on N. Validated to 4.6e-16 on both
stacks. Cost: 3750 useful MACs/elem vs sumfac's 1500 (2.5x).

| variant | ms/apply | NCU |
|---|---|---|
| warp-kron eb64 bd256 | 0.1171 | 78.6M DFMA (pad 1.28x), SM 65.8%, 76 regs |
| warp-kron eb64 bd64 | 0.2664 | block_dim sensitive |
| warp-tile (sumfac, ref) | 0.0715 | 65.0M DFMA (pad 2.64x), SM 90.9% |
| cutile kron32 | 0.3305 | same padded MAC volume as sumfac -> same time |
| cutile eb4 (sumfac, ref) | 0.2933 | |

Verdict: regular (25,25) shapes pad 2x better (1.28x vs 2.64x) but the 2.5x
algorithmic FLOP increase dominates on FMA-pipe hardware — the scalar FP64
pipe pays per slot and does not reward regularity. Axis pairing is an
A100/DMMA-contingent lever (tensor cores pay 2x rate for shapes they can
tile), not a 5090 lever. Note Tu et al.'s m25n5k4 GEMMs are NOT full
Kronecker — m25=q^2 is the natural slab shape of a *3D* sumfac stage: 3D
sum factorization gets the regularity for free, 2D is the degenerate case.

## Speed-of-light analysis (NCU SOL + roofline, n=5 kernels)

| | warp-tile | cutile eb4 |
|---|---|---|
| Compute (SM) throughput | **90.9%** | 73.0% |
| Memory throughput / DRAM | 20.2% / 1.8% | 7.8% / 1.1% |
| Achieved occupancy | 63.1% | 10.5% (theoretical 16.7%) |
| Block limit, registers | 16 blocks/SM | **1 block/SM** (255 regs/thr) |
| Issued warp per scheduler | 0.19 | 0.07 |

Both kernels are pure FP64-pipe problems (DRAM ~1%). The chip's demonstrated
FP64 issue rate (from the Warp kernel: 65M FMA-slots in 71.5 us at 91% SM) is
~1.0e12 slots/s, putting the useful-work physics floor (24.6M MACs as fused
FMA) at ~25 us. Measured decomposition vs that floor:

- warp-tile: 71.5 us = 25 us x **2.6 (cuBLASDx internal tile padding)** x
  1.1 (91% SM). It runs AT speed of light for the instruction stream it
  issues; its only recoverable waste is the 2.6x shape padding (the E_b
  lever).
- cutile eb4: 293 us = 25 us x **4.3 (pow2 padding)** x **2.0 (unfused
  DMUL+DADD)** x 1.37 (73% SM; register-capped at 1 block/SM). 8.4% of the
  physics floor.

cuTile executes its (inflated) stream fairly well given 10.5% occupancy; its
losses are dominated by WHAT it issues, not how. All three cuTile factors are
outside user control except via the padding playbook above.

## Findings

1. **cuTile FP64 GEMMs are unfused DMUL+DADD** (zero DFMA in every cuTile
   kernel) — kernel-level confirmation of the AOT-disassembly finding. Two
   instruction slots per MAC on the 1/64-rate pipe.
2. **Power-of-2 padding costs 4.3x at n=5**: 104.9M DMULs vs 24.6M useful MACs.
   Combined with no fusion: 8.7x instruction slots vs a perfect fused kernel.
   cuBLASDx also over-issues (62.9M DFMA = 2.6x useful) but fused and scheduled
   better. Net wall gap at P=4: **cuTile 4.1x slower**.
3. **At n=4 (no padding) cuTile is nearly instruction-minimal**: 13.6M MAC
   pairs = 1.13x useful vs Warp's 2.08x. Instruction slots almost tie (28.5M vs
   26.2M) — the remaining 1.66x wall gap is scheduling: cuTile maxes registers
   (255/thread) vs Warp's 48. If tileiras ever emits DFMA, P=3-class shapes
   should reach near-parity.
4. **cuTile's compiler-managed layouts are genuinely good**: lowest
   shared-memory LSU traffic of all variants (1.82M vs Warp's 3.90M at n=4) —
   the P7/P8 (lane maps / cyclic layouts) story is handled, for free.
5. **The block-redundant scalar D stage costs 2.3x even for a trivial
   qfunction** (0.168 vs 0.072 ms). Its NCU signature is exact: 78,643,200
   DMULs = 75 QP values x 16384 elements x 64 threads — every thread computes
   everything. cuTile eliminates this waste class by construction (tile
   elementwise ops are compiler-distributed). For Warp, the same fix is
   available whenever the D stage can be phrased as tile elementwise ops
   (constant/injected geometry): the warp-tile variant here IS that fix.
6. **E_b batching pays on both stacks**: 27-37% at n=5 (optimum E_b=4-8), 4.4x
   at n=4 (optimum E_b=16). Supports the E_b campaign; cuTile shows the
   expression cost can be near-zero with a tilegrid layout.

## Implications for the project

- Keep Warp/cuBLASDx as the production path: 4.1x faster at P=4 today, has a
  CPU path, and hosts the integrand codegen.
- Port finding 5 back to the sumfac kernels: a tile-elementwise D stage for
  side-constant/affine geometry (mask precomputed per plan) instead of the
  seeded scalar loop, where the form permits.
- Re-run this spike when tileiras updates FP64 codegen (DFMA fusion or DMMA);
  the harness is one command per table row.

## Warp <-> cuTile interop (verified end-to-end)

`interop_warp_cutile.py`: the eb4 sum-factorized cell apply with ALL device
storage as `wp.array` (zero-copy via `__cuda_array_interface__`), launched via
`ct.launch(wp.get_stream(device).cuda_stream, ...)` (raw CUstream handle —
cuTile does not accept the wp.Stream object itself), then a Warp kernel
mutating the cuTile-written array in place. Both directions bit-exact
(2.3e-16 / 4.6e-16). No copies, one process, one stream.

Environment: `uv run --with "cuda-tile[tileiras]" ...` resolves cuda-tile
1.4.0 + the tileiras compiler wheels alongside the repo's editable Warp —
no pyproject change needed for spike work (add as an optional extra only if
committed code grows a cuTile dependency). The standalone /tmp/cutile-venv
(cupy-based) remains for the AOT disassembly harness.

## Repro

```
/tmp/cutile-venv/bin/python oracle.py
/tmp/cutile-venv/bin/python bench_cutile.py --variant eb4 --check
/tmp/cutile-venv/bin/python bench_cutile.py --variant tg2d4 --check
/tmp/cutile-venv/bin/python bench_cutile.py --variant eb16 --n 4 --check
uv run bench_warp.py --variant tile --check
uv run bench_warp.py --variant scalar --check
# NCU rows:
ncu --metrics gpu__time_duration.sum,smsp__sass_thread_inst_executed_op_dfma_pred_on.sum,\
smsp__sass_thread_inst_executed_op_dmul_pred_on.sum,smsp__sass_thread_inst_executed_op_dadd_pred_on.sum,\
l1tex__data_pipe_lsu_wavefronts_mem_shared.sum,launch__registers_per_thread \
  -k regex:cell_apply --launch-skip 8 --launch-count 1 --csv \
  <python> <bench>.py --variant <v> --warmup 5 --iters 5
```
