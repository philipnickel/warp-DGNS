You are implementing Phase 6 STAGE 2 (sum-factorized side apply) of the sum-factorized DG project
on the Warp fork at /root/warp-DGNS (branch philipnickel/sumfac-dg). Do NOT commit; leave changes
in the working tree.

GOAL: `fem.integrate(side_form, ..., assembly="sumfac")` works for qualifying LINEAR forms over
`fem.Sides`/`fem.InteriorSides`/`fem.BoundarySides` of Grid2D/Grid3D, so that a fully matrix-free
SIPG operator (sumfac volume apply + sumfac side apply) runs end to end. Bilinear side assembly
stays on the default path (out of scope).

READ FIRST (authoritative):
1. design/sumfac-dg-implementation-plan.md section "Phase 6" (math) and design/sumfac-status.md
   (Phase 6 row: stage-1 findings; environment notes; the 0.44 ms / 6.59 ms baselines).
2. warp/_src/fem/sumfac/face_trace.py — the stage-1 conventions, LOCKED by tests: value trace =
   GLL boundary slice (face_slice_indices); normal-gradient trace = endpoint derivative row;
   grid side frames (grid_side_axes cyclic order, longitude flips, grid_side_face_ends);
   inner->outer permutation = IDENTITY on grid interior sides. Build on these; do not re-derive.
3. warp/_src/fem/sumfac/kernels.py — the volume linear kernel factory
   (get_integrate_linear_sumfac_kernel), the layout/plan dataclasses, the raising predicates, the
   ValueInjectedField/SeedField substitution flow in make_sumfac_plan, and integrate.py's sumfac
   dispatch + launch sections. The side path must follow the same architecture: raising layout
   finder -> plan -> kernel factory -> launch branch, cache-keyed with a "sumfac-side" suffix.
4. warp/_src/fem/sumfac/qfunction.py (cell-side seeded extraction; you will add the side analog)
   and warp/_src/fem/field/virtual.py (SeedField/ValueInjectedField; side traces need
   inner+outer channels).
5. warp/tests/fem/test_fem_sumfac_face_trace.py and test_fem_sumfac_faces.py for conventions and
   oracles. /tmp/sipg_matvec_profile.py for the SIPG operator and baselines.

DESIGN DECISIONS (locked in design discussion; do not revisit):
- GATHER formulation: launch one block per CELL of the side domain's geometry (or per cell
  adjacent to the side set). Each block loops over the cell's 2*dim faces; for each face in the
  integration domain it loads the neighbor's face trace data and accumulates the test-channel
  lift into a cell-local residual tile; ONE store at the end (or scatter through the existing
  sumfac scatter kernel). No atomics, races impossible by construction. Interior numerical
  fluxes are evaluated twice (once per adjacent cell) — accepted cost.
- Traces: inner value trace = strided tile_load of the cell's boundary DOF slice (never a
  strided in-register tile_view into tile_matmul — known cuBLASDx corruption, copy to a
  contiguous tile first); neighbor value trace = tile_load of the neighbor's slice (identity
  permutation on grids). Normal-gradient traces contract the full element tensor with the
  endpoint derivative row. Tangential axes contract with the (d-1)-dim 1D interpolation/
  derivative matrices at the side QPs mapped through the stage-1 conventions.
- D-stage generality boundary: arbitrary integrand bodies via seeded test channels, like the
  volume path. At each face QP the test field has 2*(1+dim) trace channels (inner/outer value,
  inner/outer gradient); jump/average/grad_jump/grad_average/inner/outer on the test field are
  linear in those channels. The input field u (one injectable NodalField over the same space,
  whole partition — same requirement as the volume path) has its inner/outer value+gradient
  traces precomputed by the B stage and injected (ValueInjectedField analog for traces).
  Supported operators on u and v: inner, outer, jump, average, grad_inner(=grad), grad_outer,
  grad_jump, grad_average, degree, normal/measure_ratio/domain position on the Domain arg.
  Anything else raises SumfacNotApplicableError naming the operator.
- Qualifying layouts (raise descriptively otherwise): side domain whose geometry is Grid2D or
  Grid3D (unstructured Quadmesh/Hexmesh raise — per-side orientations are stage 2+ follow-up);
  scalar discontinuous GLL tensor-product space on the WHOLE partition for both test and input
  fields; RegularQuadrature over the side domain with lexicographic tensor-product points in
  the (d-1) face axes (reuse/extend the stage-1 quadrature checks); f32/f64; no autodiff.
- Boundary sides: inner == outer cell; jump(u) = u_inner, average(u) = u_inner (match the
  native operators' boundary semantics EXACTLY — read warp/_src/fem/operator.py and the field
  trace implementations, and lock with a BoundarySides oracle test). Note the geometric normal
  on altitude-0 boundaries points -axis (stage-1 finding): get normals/measure factors from the
  native side machinery (side_arg) per QP, not from hand-derived formulas.
- Geometry factors per face QP (normal, measure ratio, Jacobians for physical gradients) come
  from the NATIVE side machinery inside the kernel (domain/geometry side args), exactly like
  the volume kernel gets cell Jacobians natively. Only the trace evaluation is sumfac-owned.

DELIVERABLES:
1. Side Q-function/seed support: extend qfunction.py + virtual.py with side-trace seed fields
   (test) and trace value injection (input field). Keep the cell path untouched and green.
2. kernels.py (or a new sumfac/side_kernels.py if cleaner): find_sumfac_side_layout (raising),
   make_sumfac_side_plan, get_integrate_side_sumfac_kernel factory (gather formulation, 2D and
   3D), launch wiring in integrate.py behind assembly="sumfac" for linear side forms. Forced
   enable_backward=False; kernel tagged _wp_fem_sumfac_; cache suffix discriminates side vs
   cell ("sumfac-side", n, q_face, dim, axis-handling constants, field names, dtypes).
3. Tests (warp/tests/fem/test_fem_sumfac_faces.py — extend; register nothing new unless you add
   a file): at P=4 exclusively for GPU-compiling tests (n=q=5 per axis; reuse the volume tile
   shapes where possible to bound LTO compiles):
   a. test_side_apply_equals_naive: SIP form (penalty + consistency + symmetry, the form in
      /tmp/sipg_matvec_profile.py) and an upwind-advection flux form, on Grid2D AND Grid3D
      Sides + InteriorSides + BoundarySides, random DG DOFs: sumfac side apply == default
      integrate to rtol 1e-9 (the naive path is the oracle; it is the same discrete operator).
   b. test_side_grad_trace_matches_native (closes the stage-1 review gap): the injected
      normal-gradient traces equal fem.grad of inner/outer at side QPs through the native path.
   c. test_side_sumfac_raises_unqualified: unstructured quadmesh side domain, partial
      partition, unsupported operator (e.g. at_node), bilinear side form -> descriptive raises.
      UPDATE the existing test_side_sumfac_raises (which currently asserts side domains always
      raise "cell domains") to the new contract: qualifying side LINEAR forms now work; side
      BILINEAR forms still raise; the volume-path error message must no longer claim side
      integrals are unsupported if they now are.
   d. test_fully_matrix_free_sipg: compose sumfac volume apply + sumfac side apply for the SIP
      operator on Grid2D P=4; compare y against (assembled volume + assembled side) @ x to
      rtol 1e-9; then a CG solve of the manufactured problem matches the hybrid solve.
4. Perf evidence (report, do NOT assert timings in tests): extend /tmp/sipg_matvec_profile.py
   with the sumfac side apply and report ms/apply at res 128 P=4 f64 vs the 0.44 ms (assembled)
   and 6.59 ms (naive) baselines on this box.
5. CHANGELOG.md: extend the existing Unreleased sumfac entry (side linear forms now supported;
   fully matrix-free DG operators). design/sumfac-status.md: update the Phase 6 row.

RULES (hard): uv run only; run test files directly (never -m warp.tests); CUDA env prefix
NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++" WARP_CUDA_PATH=/root/cuda-env/.pixi/envs/default;
iterate CPU-first but finish with CUDA runs of every new/changed test file; never
wp.clear_kernel_cache; no nested scalar gather loops in tile kernels (tile_load everything);
never feed a strided tile_view to tile_matmul; degrees {4} for GPU tests; np.testing.assert_allclose;
SPDX 2026 headers; Google docstrings; uvx pre-commit run --files <changed> at the end. The
existing volume suites (test_fem_sumfac_apply/assembly/qfunction/face_trace) must stay green —
run them after your changes.

SUMMARY REQUIRED: files changed; exact unittest summary lines per file (CPU and CUDA); the
measured side-apply ms/apply vs both baselines; any deviation from this spec and why; any
convention you had to extract empirically beyond stage 1 (state it precisely).

ADDENDUM (design discussion, GEMM purity):
- Stack the face-trace operators per axis, mirroring the volume kernel's stacked [A; D-hat]
  trick: the two endpoint derivative rows of one axis form a (2, n) operator (one GEMM for both
  opposing faces' normal-gradient traces instead of two row contractions); the two opposing
  value-trace slices pair the same way for the tangential interpolation to face QPs. Prefer
  fewer, fatter GEMMs over per-face thin GEMVs wherever the operands are dense.
- Keep the primitive-selection rule: dense contraction -> tile_matmul; diagonal scaling
  (weights/geometry factors) -> elementwise tile arithmetic, never a GEMM; value-trace
  selection -> slice load, never a multiplication by a 0/1 matrix; arbitrary integrand
  evaluation -> the seeded scalar loop (generality boundary, R2).
