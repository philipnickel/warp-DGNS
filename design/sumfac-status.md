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
| C · API consolidation (`assembly="sumfac"`) | **DONE** | Replace the transparent auto/force/off dispatch with an explicit `integrate(..., assembly="sumfac")` opt-in (never auto-selected); unqualified forms raise `SumfacNotApplicableError` naming the unmet requirement, including the bilinear smem-budget gate. `set_sumfac_mode`/`get_sumfac_mode`/`WARP_FEM_SUMFAC`/`SUMFAC_DEGREE_THRESHOLD` removed; `make_sumfac_linear_operator` now calls `assembly="sumfac"`. GPU correctness tests collapsed to P=4 (contract tests keep one E_b>1 and one rectangular q≠n case). Also: shared layout/plan dataclass bases; tensor_contract test-only residual drivers moved into the contract tests; verified (against generated source) that in-loop `wp.static` guards do NOT force-unroll the flat D-stage loops — closure booleans are replaced at declaration time, so the loops compile dynamic as intended. CUDA revalidation of the consolidated branch is pending (this round was CPU-only: no CUDA toolkit on the dev box). |
| 6 · Surface sum-factorization | **NEXT — not started** | Spec for stage 1 (face traces + ordering locks): `design/sumfac-phase6-stage1-spec.md`. Stage 2 = tile kernels + dispatch relaxation; stage 3 = fully-sumfac DG e2e. Plan section "Phase 6" in `sumfac-dg-implementation-plan.md`; risk §9.8 (inner/outer face orientation) MUST be locked by the stage-1 ordering test before any flux kernel. |
| 7 · High-order example adoption | **TODO** | Plan section "Phase 7"; `example_convection_diffusion_dg.py` already has `--degree`. AOT prewarm (`wp.compile_aot_module`) is a candidate deliverable. |

## Known open issues (from adversarial reviews, none blocking)

- Bilinear sumfac tests use axis-aligned grids only; a non-affine (Quadmesh2D) bilinear oracle test would catch off-diagonal-Jacobian errors in the both-sides J⁻¹ channel mapping (the *apply* path does have a non-affine check).
- `E_b > 1` is implemented only in the standalone contraction primitives, not the fused kernels (bench says E_b=1 is optimal at P≥4 anyway; E_b≈4 would help near the crossover).
- Sum-fac kernels have no autodiff (guarded: differentiation requests under `assembly="sumfac"` raise a descriptive error).
- Assembly is the shared-channel action-on-unit-vectors variant, O(d·n^{2d}·q) FLOPs; the fully factorized O(n^{2d+1}) variant needs global-memory accumulation (deferred).
- First-compile of a new GEMM shape ≈10–60 s (3D bilinear ≈155 s) is dominated by cuBLASDx LTO generation — measured insensitive to `optimization_level` (−8%) and `max_unroll` (±0). Remaining levers: parallel prewarm across processes, AOT precompilation.

## Environment / process notes (this machine)

- A100-PCIE-40GB, CUDA 13.1. Static NVRTC libs were restored manually (see memory or: extract `cuda-nvrtc-dev-13-1` deb → `/usr/local/cuda-13.1/targets/x86_64-linux/lib/`). `uv run build_lib.py --quick` for the native lib.
- Never run `-m warp.tests` during iteration (clears kernel cache). Run test files directly. All sumfac tests warm-run in seconds.
- `test_gimp_quadrature` / `test_cube_shape_functions` CheckOutput failures on CUDA + `test_volume` NVDB import error are pre-existing environmental noise (confirmed against pristine main).
- Codex CLI works for implementation tasks with: `codex exec --skip-git-repo-check -m gpt-5.5 --config model_reasoning_effort="high" --sandbox danger-full-access -c approval_policy="never" -C /root/warp-DGNS "<prompt>" </dev/null` (bubblewrap sandboxing is non-functional in this container; `--full-auto` silently overrides the sandbox mode — do not pass it).
- Orchestration pattern that worked: spec → implement (Codex or workflow agent) → direct-run test gates on cpu+cuda → adversarial review with refutation lenses → fix round → signed commit → push.
