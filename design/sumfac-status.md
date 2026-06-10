# Sum-Factorized DG — Project Status

**Updated:** 2026-06-10 · **Branch:** `philipnickel/sumfac-dg` (all work committed and pushed to `origin` = philipnickel/warp-DGNS)

## Phase ledger

| Phase | Status | Key commits / artifacts |
|---|---|---|
| 0 · Arbitrary-P quadrature | **DONE** | `a50c2412`; `warp/_src/fem/polynomial.py`, tests `test_fem_sumfac_polynomial.py` |
| 1 · 1D operators + B/Bᵀ tile contraction | **DONE** (GPU-validated) | `a50c2412`, `5853f484` (CUDA fixes), `2f6441cc` (Bᵀ), `312cee42` (per-shape module isolation); `warp/_src/fem/sumfac/{operators_1d,tensor_contract}.py` |
| 2 · SeedField Q-function | **DONE** | `1eaf7571`; `warp/_src/fem/sumfac/qfunction.py`, `SeedField`/`TrialSeedField` in `field/virtual.py` |
| 3 · Fused apply + transparent dispatch | **DONE** (integration gate green; dispatch superseded — see consolidation row) | `ec1491a9`; `warp/_src/fem/sumfac/kernels.py`, dispatch in `integrate.py`, `ValueInjectedField`; spike findings in `design/sumfac-phase3-spike-findings.md` |
| 4 · Bilinear assembly to BSR | **DONE** | `33e8c080`; channel-seeded action-on-unit-vectors, shared-memory budget gate (now errors — see consolidation row) |
| 5 · Face fallback + end-to-end DG | **DONE** (reworked to assembly-arg form at P=4 — see consolidation row) | `43a048e2`; `test_fem_sumfac_faces.py` |
| Bench (A100) | **DONE** | `7e59cbf8`; `design/sumfac-bench-results.md` — crossover P=3, 14× at P=5 → ~62× at P=7–8, 920 GF/s f64, FP64 DMMA confirmed via PTX; `warp/examples/fem/bench_sumfac_dg.py` |
| C · API consolidation (`assembly="sumfac"`) | **DONE** (CUDA-validated, one known issue below) | Replace the transparent auto/force/off dispatch with an explicit `integrate(..., assembly="sumfac")` opt-in (never auto-selected); unqualified forms raise `SumfacNotApplicableError` naming the unmet requirement, including the bilinear smem-budget gate. `set_sumfac_mode`/`get_sumfac_mode`/`WARP_FEM_SUMFAC`/`SUMFAC_DEGREE_THRESHOLD` removed; `make_sumfac_linear_operator` now calls `assembly="sumfac"`. GPU correctness tests collapsed to P=4 (contract tests keep one E_b>1 and one rectangular q≠n case). Also: shared layout/plan dataclass bases; tensor_contract test-only residual drivers moved into the contract tests; verified (against generated source) that in-loop `wp.static` guards do NOT force-unroll the flat D-stage loops — closure booleans are replaced at declaration time, so the loops compile dynamic as intended. CUDA-revalidated on RTX 5090 (sm_120, CUDA 13.0 via pixi toolkit): operators/polynomial/contract/qfunction/faces/assembly suites green incl. the end-to-end DG P=4 solve and all E_b>1 tile kernels; the 2D *linear apply* shape n=5 hits the known issue below. |
| T · Tile-native linear kernels | **DONE** | Rewrite the fused linear apply kernels to the tile-native formulation (`design/sumfac-tile-native-operator.md`): B-stage gather = one `tile_load` (elements as tile space; requires `WholeSpacePartition`, enforced in `find_sumfac_layout`); 2D uses the stacked `[A; D̂]` operator (all channels from 2 GEMMs, Bᵀ in 2 GEMMs); 3D keeps its GEMM slab structure with the tile-load gather. Removes the nvJitLink trigger by construction; validated by the isolated spike (`design/spike_tile_native_gemm/`) and the full P=4 suites on CPU+CUDA. Follow-up (spec R6): dimension-generic mode-k emitter unifying 2D/3D/faces. |
| T2 · Direct-store bilinear assembly | **DONE** | The fused bilinear kernel `tile_store`s each local-block column straight into `BsrMatrix.values` (viewed as `(rows, N)`); the compact block-diagonal topology is written in closed form by `set_block_diagonal_topology()` + `notify_nnz_changed()` — the staging array, scalar triplet-fill kernel, triplet temporaries, and `bsr_set_from_triplets` radix sort are gone from the sumfac path. Requires the whole space partition on test and trial fields (enforced in `find_sumfac_bilinear_layout`, raising otherwise); `bsr_options['topology']='padded'` raises; `capacity='reuse'`, `output=`, and `add=True` covered by a new oracle test. Spec section "Output: direct block-diagonal store". Suites green CPU+CUDA incl. non-affine quadmesh; example composed-system check unchanged (f32 roundoff). |
| 6 · Surface sum-factorization | **Stage 2 DONE** | Stage 1 (face traces + ordering locks, `design/sumfac-phase6-stage1-spec.md`): `warp/_src/fem/sumfac/face_trace.py` + `test_fem_sumfac_face_trace.py`. Key empirical findings (4-lens adversarial review: all confirmed against grid_2d/3d source, zero blocker/major): grid side normal-axis ends are inner=1.0/outer=0.0; 2D longitude flip iff `(a==0)==(alt==0)` (interior y-normal sides flipped); 3D side frame is the CYCLIC `(a+1, a+2)%3` order (a=1 needs a transpose vs the sorted face frame) with flips only on `alt==0` boundary sides; and **the inner→outer face permutation on interior grid sides is the IDENTITY** — jump/average combine pointwise, no reindexing. Conventions do NOT apply to unstructured Quadmesh/Hexmesh (per-side orientations; stage 2+ scope decision). Stage 2 (`design/sumfac-phase6-stage2-spec.md`): `assembly="sumfac"` now covers LINEAR side forms over `Sides`/`BoundarySides`/side `Subdomain`s of Grid2D/Grid3D — gather formulation (one block per adjacent cell, no atomics; interior fluxes evaluated once per adjacent cell), `warp/_src/fem/sumfac/side_kernels.py` + `SideSeedField`/`SideTraceInjectedField` (2·(1+d) trace seed channels) + `extract_side_qfunction`; boundary sides lift BOTH inner and outer channels (matches the native trace test field exactly). Oracle tests (SIP/upwind/asymmetric trace forms, 2D+3D, all side domain kinds, vs naive integrate at rtol 1e-9), grad-trace-vs-native lock, side Q-function linearity lock, and the fully matrix-free SIPG composition test are in `test_fem_sumfac_faces.py`. Bilinear side forms still raise. Stage-2 empirical note: dynamic-loop tile SSA merges require unique tile names per code section (mismatched tile types across loops fail to compile). Stage-2 perf (RTX 5090, Grid2D 128x128 P=4 f64 SIP apply, 50 iters): sumfac side apply 5.07 ms/apply vs 6.57 ms naive and 0.365 ms assembled `bsr_mv` (same run; ~0.46 ms of the 5.07 is per-apply host overhead, measured at res 8). NCU on the side kernel: SM 82%, shared-mem 69%, DRAM 0.03%, occupancy 50% (register-limited) — compute-bound on the warp-redundant scalar D stage (up to 2·(1+d) seeded SIP evaluations per face QP, all `block_dim` threads redundant) and the tiny per-face GEMMs (m=2 trace shapes). The 0.44 ms target is NOT met; follow-ups: smaller `block_dim` for the side kernel (new LTO set), batching faces into wider GEMMs, distributing D-stage QPs across warps. Stage 3 = fully-sumfac DG example adoption. |
| 7 · High-order example adoption | **STARTED** | `example_convection_diffusion_dg.py` gained `--sumfac` (volume terms via `assembly="sumfac"`, side terms default; `tri` mesh rejected). Verified on the example's f32 grid/quad setups: per-form matrices, the fully composed system (incl. side `add=True`), the rhs apply, and exact f64 solves of both systems all agree at f32 roundoff (≤5e-7); end-to-end φ trajectories differ only by bicgstab `tol=1e-4` slop (κ≈2e2 → bounded by κ·tol), which an iterative f32 solve cannot distinguish. Remaining: other examples, AOT prewarm (`wp.compile_aot_module`). |

## Known open issues (from adversarial reviews, none blocking)

- **[RESOLVED on the kernel side by the tile-native rewrite]** The fused linear kernels no longer contain the
  scalar-gather structure that nvJitLink miscompiled: the B-stage gather is now a single `tile_load` (mesh
  elements as the tile space; layout guaranteed by a `WholeSpacePartition` check in the layout predicate),
  and the 2D kernel uses the stacked `[A; D̂]` operator pipeline (4 GEMMs total) per
  `design/sumfac-tile-native-operator.md`. All sumfac suites pass on the RTX 5090 (CPU + CUDA, including the
  formerly failing n=5 2D linear apply cases). The nvJitLink miscompilation itself remains unfixed upstream —
  the artifact pair and the record below stay valid for the NVIDIA/Warp reports.
- **nvJitLink LTO miscompiles the (former) scalar-gather 2D linear apply kernel at n=5 into an infinite loop** (found
  2026-06-10 by the new P=4 2D coverage — this shape had never run on any GPU; the old sweeps used 2D
  n∈{2,4,6}). Root cause established by reading the final linked PTX: the B-stage gather loop
  (`for node in range(nn_c)`, kernels.py) is emitted with an **unconditional back-branch and no exit
  condition**, and everything after it (matmuls, D-stage, BᵀT, store) is deleted as unreachable — the kernel
  body truncates right after the loop. At runtime the loop walks `read_node_value` past the DOF array
  (instrumentation showed `node` reaching 200+; the first OOB index is always `node_count`, which is what
  `compute-sanitizer` reported) → CUDA error 700.
  Localization: compiling the **identical generated source with NVRTC alone (no `-dlto`) produces a correct
  loop** — the damage happens only in the LTO link step, where nvJitLink inlines the cuBLASDx
  `dot_5_5_5_*` LTOIR and re-optimizes. Reproduced with nvJitLink/NVRTC **12.9.86, 13.0.88 (and the
  `.alt` NVRTC build), and 13.3.33** (13.3 output inspected from the kernel cache; its PTX ISA 9.3 cannot
  load on the 13.0 driver). Insensitive to `--Ofast-compile` levels, `--extra-device-vectorization`,
  `--restrict`, block_dim 32/64, and `ptx` vs `cubin` output. Given the LTO dependence this is almost
  certainly **not Blackwell-specific** and would reproduce on the A100 as well.
  Shape envelope (runtime-verified): 2D linear n=5 broken for any q∈{4,5,6}; 2D n=4/n=6, 3D n=5, the
  bilinear kernel at n=5, and the standalone contraction kernels using the same 5×5×5 GEMM are all fine. So
  `test_apply_*_2d`/`test_affine_form_matches_default`/`test_assembly_apply_consistency` fail on CUDA while
  everything else (incl. end-to-end DG P=4) passes. A nested-loop gather rewrite avoids it at runtime but was
  rejected as a fix (perturbs the design to dodge a linker bug). Both officially-pinned toolchain pairings
  (CUDA 12.9 + libmathdx 0.3.1 and CUDA 13.0 + libmathdx 0.3.2) reproduce it, and libmathdx 0.3.2 is the
  newest cu13 build on NVIDIA's redist (0.3.3+/0.4.0 probed: 404), so there is no newer LTOIR producer to
  try. Next steps: report to NVIDIA (artifact pair: cached truncated `*.sm120.ptx` vs the correct NVRTC-only
  compile of the same cached `.cu`) and to NVIDIA/warp; re-test when a newer nvJitLink/libmathdx ships.

- ~~Bilinear sumfac tests use axis-aligned grids only~~ **Resolved:** non-affine (sheared + vertex-perturbed
  Quadmesh2D) oracle tests added for both the bilinear assembly (mass/stiffness/advection) and the linear
  apply at P=4, passing on CPU+CUDA.
- `E_b > 1` is implemented only in the standalone contraction primitives, not the fused kernels (bench says E_b=1 is optimal at P≥4 anyway; E_b≈4 would help near the crossover).
- Sum-fac kernels have no autodiff (guarded: differentiation requests under `assembly="sumfac"` raise a descriptive error).
- Assembly is the shared-channel action-on-unit-vectors variant, O(d·n^{2d}·q) FLOPs; the fully factorized O(n^{2d+1}) variant needs global-memory accumulation (deferred).
- First-compile of a new GEMM shape ≈10–60 s (3D bilinear ≈155 s) is dominated by cuBLASDx LTO generation — measured insensitive to `optimization_level` (−8%) and `max_unroll` (±0). Remaining levers: parallel prewarm across processes, AOT precompilation.

## Environment / process notes

### A100 box

- A100-PCIE-40GB, CUDA 13.1. Static NVRTC libs were restored manually (see memory or: extract `cuda-nvrtc-dev-13-1` deb → `/usr/local/cuda-13.1/targets/x86_64-linux/lib/`). `uv run build_lib.py --quick` for the native lib.

### RTX 5090 box (no system CUDA toolkit)

- RTX 5090 (sm_120), driver 580.95.05 (CUDA 13.0 branch). The toolkit comes from a pixi env at `/root/cuda-env`
  (`pixi add cuda-toolkit cuda-version=13.0 cuda-nvrtc-static "libnvjitlink-static=13.0"` — the nvjitlink pin
  matters: an unpinned 13.3 static lib emits PTX ISA 9.3, which the 13.0 driver cannot load). Three symlink
  bridges are needed in the prefix: `targets/x86_64-linux/include/*` → `include/`, `targets/x86_64-linux/lib/*`
  → `lib/`, and `lib64` → `lib`. Build with
  `NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++" WARP_CUDA_PATH=/root/cuda-env/.pixi/envs/default uv run build_lib.py --quick`
  (the env's conda gcc 14.3 breaks nvcc 13.0's frontend; the system g++ works).
- Never run `-m warp.tests` during iteration (clears kernel cache). Run test files directly. All sumfac tests warm-run in seconds.
- NCU (nsight-compute 2025.3.1, `/root/cuda-env/.pixi/envs/default/bin/ncu`) works on this box incl. GPU perf
  counters. First SIPG matvec profile (Grid2D 128x128 P=4 f64, 409,600 DOFs, hybrid = sumfac volume apply +
  assembled SIP side matrix, script pattern in git history / `/tmp/sipg_matvec_profile.py`): volume sumfac apply
  0.62 ms (NCU: SM compute 83%, DRAM 0.2% -- compute-bound, intermediates fully on-chip), side `bsr_mv` 0.44 ms
  (DRAM 40%, SM 12% -- bandwidth-bound), full matvec 1.04 ms. Quantifies the Phase 6 payoff: ~40% of the hybrid
  operator is the memory-bound non-sumfac side path. Note sm_120 has no FP64 tensor cores; DMMA utilization is
  an A100 metric. No nsys in the pixi env (`pixi add nsight-systems` if a timeline is needed).
- Fully matrix-free SIPG side apply works TODAY through the native path (linear side form over
  `u_field.trace()` with jump/grad_average; matches `K_side @ x` to 7e-16) at 6.59 ms/apply -- 15x the
  assembled-matrix matvec, since each side QP pays the naive O(n^d) trace evaluation. This is the Phase 6
  stage-2/3 baseline: sumfac faces must beat 0.44 ms (assembled) while keeping the 0 MB footprint of the
  6.59 ms naive path (face DOF traffic is ~3 MB vs the 280 MB side matrix, so the roofline is far below both).
- `test_gimp_quadrature` / `test_cube_shape_functions` CheckOutput failures on CUDA + `test_volume` NVDB import error are pre-existing environmental noise (confirmed against pristine main).
- Codex CLI works for implementation tasks with: `codex exec --skip-git-repo-check -m gpt-5.5 --config model_reasoning_effort="high" --sandbox danger-full-access -c approval_policy="never" -C /root/warp-DGNS "<prompt>" </dev/null` (bubblewrap sandboxing is non-functional in this container; `--full-auto` silently overrides the sandbox mode — do not pass it).
- Orchestration pattern that worked: spec → implement (Codex or workflow agent) → direct-run test gates on cpu+cuda → adversarial review with refutation lenses → fix round → signed commit → push.
