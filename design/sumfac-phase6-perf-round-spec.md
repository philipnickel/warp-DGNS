You are implementing the Phase 6 SIDE-KERNEL PERF ROUND of the sum-factorized DG project on
the Warp fork at /root/warp-DGNS (branch philipnickel/sumfac-dg). Do NOT commit; leave all
changes in the working tree.

GOAL: reduce the sum-factorized side apply from 5.07 ms/apply toward (ideally below) the
0.365 ms assembled `bsr_mv` baseline on the benchmark (Grid2D 128x128, P=4, f64, SIP form,
RTX 5090; script `/tmp/sipg_matvec_profile.py --side-mf`, extend it to also time the sumfac
side apply if it does not already). Numerics must be bit-for-bit oracle-equivalent in the
test sense: all existing suites stay green at their tolerances (they currently pass at
~1e-15 vs the naive-integrate oracle).

READ FIRST (authoritative):
1. design/sumfac-phase6-perf-analysis.md — the measured bottleneck breakdown, the
   evidence-weighted fix plan (core reference arXiv:2603.09038), and the queued review
   follow-ups. This document IS the rationale for everything below.
2. warp/_src/fem/sumfac/side_kernels.py (the kernel under optimization), face_trace.py
   (locked conventions), kernels.py (volume kernel patterns, stacked-operator precedent),
   warp/tests/fem/test_fem_sumfac_faces.py (the oracles).
3. design/sumfac-phase6-stage2-spec.md including its ADDENDUM (GEMM-purity rules).

ORDER OF WORK (strict — strengthen the oracle BEFORE touching the kernel):

STEP 0 — Test strengthening (from the stage-2 adversarial review; cheap, no new tile shapes):
  a. Add a side form with TANGENTIAL gradient content (e.g. gradients dotted with a fixed
     non-axis-aligned vector, or grad_jump(u) contracted with a constant non-normal vector):
     every committed form dots gradients with the axis-aligned normal, so all tangential
     trace/lift code paths currently carry exactly zero.
  b. Add a POSITION-DEPENDENT side coefficient to one oracle form (integrand reads
     fem.position(domain, s)): the symmetric Gauss rule plus position-independent integrands
     make any CONSISTENT longitude-flip transcription error an exact QP relabeling, invisible
     to the current oracle.
  Both go into test_fem_sumfac_faces.py's side-apply-equals-naive sweep (2D+3D, interior and
  boundary domains), P=4. Run and confirm green BEFORE any kernel change; these are the
  guards for everything that follows.

STEP 1 — Stacked face operators (top lever; 24 GEMM calls -> ~4-6 per cell):
  Per axis, stack the value and derivative endpoint rows of BOTH opposing faces into one
  (4, n) trace operator (2 faces x [value; deriv]); where the inner/outer pairing allows,
  batch the neighbor traces the same way. Batch the tangential interpolation/derivative
  contractions across faces/channels into the fewest well-shaped GEMMs the algebra permits.
  Load the own-cell DOF tensor ONCE per cell, not once per face. Follow the (2,n)->(4,n)
  stacking pattern of the volume kernel's [A; D-hat] precedent. Keep the cyclic-index idea
  from the core reference in mind: choose intermediate tile layouts so each GEMM contracts
  over the leading (contiguous) index — no transposes, no strided operands into tile_matmul.

STEP 2 — D-stage cost reduction (second lever):
  a. Hoist side-constant geometry: on grids, normal and measure_ratio are constant per face;
     evaluate once per face (block-uniform), not inside every seeded integrand evaluation.
     If this needs a side-aware Q-function contract (pass precomputed geometry into the
     channel evals instead of re-running the unmodified integrand per seed), implement it —
     the oracle equality at 1e-9 is the contract.
  b. Boundary faces: stop computing identical inner and outer traces twice.
  c. Inactive faces (BoundarySides/Subdomain): skip with block-uniform branches instead of
     computing-and-zeroing (~(2d-1)/(2d) of B/D work wasted there today).

STEP 3 — Launch/host costs (third lever):
  a. Measure a block_dim=32 variant of the side kernel (halves redundant D-stage lanes; new
     LTO set — keep it to the benchmark shapes; only adopt if it wins on ms/apply).
  b. Cut the ~0.46 ms per-apply host floor (measured at res 8): cache the side plan, face
     map, operator arrays, and launch arguments across applies (e.g. on the plan object or
     via cache.borrow_temporary patterns); profile what remains with a quick
     time.perf_counter breakdown.

STEP 4 — Hygiene (from the review; do these while in the file):
  a. Comment (and where cheap, guard) the full-width-row-block tile_view convention at its
     call sites (sound today; enforced only by convention).
  b. Replace the wp.min(2, cell_index + 2) unroll-defeat with a plain int kernel argument.
  c. Add domain.name directly to the "sumfac-side" cache suffix.

EXPLICITLY OUT OF SCOPE: E_b multi-cell panels (the biggest lever but a separate campaign —
do not start it); 3D-specific tuning beyond what the stacked operators give for free;
autodiff; unstructured meshes.

MEASUREMENT PROTOCOL (after each step, record in your summary):
- ms/apply on the benchmark (50 iters, warm) for: sumfac side apply, plus the unchanged
  volume apply and assembled bsr_mv as controls.
- After the final step: NCU on the side kernel with the core reference's diagnostic pair —
  "L1: Data Pipe Lsu Wavefronts" and the FP64/DMMA pipe activity — plus achieved occupancy
  (ncu at /root/cuda-env/.pixi/envs/default/bin/ncu; see design/sumfac-status.md env notes).
- GDOF/s for the full fully-matrix-free SIPG matvec before/after.

RULES (hard): uv run only; run test files directly (never -m warp.tests); CUDA env prefix
NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++" WARP_CUDA_PATH=/root/cuda-env/.pixi/envs/default;
iterate CPU-first, finish every step with a CUDA run of test_fem_sumfac_faces.py; the volume
suites (apply/assembly/qfunction/face_trace) must stay green at the end; never
wp.clear_kernel_cache; P=4 only for GPU tests; minimize new LTO shapes (the stacked (4,n)
operators introduce a small new set — acceptable; list every new shape in the summary);
np.testing.assert_allclose; pre-commit on changed files at the end.

SUMMARY REQUIRED: per-step ms/apply progression (table); final NCU metrics vs the stage-2
baseline (SM 82%, smem 69%, DRAM 0.03%, occupancy 50%); exact unittest summary lines for
every test file run (CPU and CUDA); new LTO shapes introduced; files changed; deviations
with reasons; remaining bottleneck analysis if the 0.365 ms target is still not met.
