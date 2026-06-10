# Sum-Factorized High-Order DG Operators — TDD Implementation Plan

**Companion to:** `docs/superpowers/specs/2026-06-09-sumfac-dg-operators-design.md`
**Branch:** `philipnickel/sumfac-dg` (create from `main`, never commit to `main`)
**Target:** fork of NVIDIA/Warp
**Date:** 2026-06-09

## How to use this plan

Each phase below is self-contained and ordered so that **tests are written and seen to fail
FIRST**, then the implementation is written until the test passes (red → green → refactor).
Every phase lists: files to create/edit, public signatures, the test (with the concrete
assertion), the run command, and an acceptance gate. The existing naive `integrate()` is the
**golden oracle** for every numerical check (spec §8).

**Repo rules baked into every step (AGENTS.md):**
- Run Python only via `uv run` (CPU-only Mac, native lib already built in `warp/bin/`).
- Tests use stdlib `unittest`, NOT pytest. New `@wp.kernel`/`@fem.integrand` code goes in `.py`
  files only (codegen calls `inspect.getsourcelines()`), never `python -c`.
- Use `np.testing.assert_allclose` (not `np.allclose`); add `wp.synchronize_device()` before
  reading device results in a test that does not otherwise sync.
- NEVER call `wp.clear_kernel_cache()` / `wp.clear_lto_cache()` anywhere.
- New files get the SPDX header:
  ```
  # SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
  # SPDX-License-Identifier: Apache-2.0
  ```
- Internal modules import from `warp._src.fem.*`; public-facing/test code imports from `warp` /
  `warp.fem`. Capitalize NumPy and Warp in docstrings/messages. Google-style docstrings.
- Register every new test module in `default_suite()` in
  `warp/tests/unittest_suites.py` (import near line 108, append to `test_classes` near line 309).
- Add a `CHANGELOG.md` "Unreleased" entry for each user-facing change (arbitrary-P, the
  transparent sum-fac path).
- Run `uvx pre-commit run --files <changed files>` before each commit; `git commit -s` (DCO).

## Hardware verifiability summary

| Phase | What it delivers | Where verifiable |
|---|---|---|
| 0 · Arbitrary-P quadrature | `polynomial.py` GL+GLL, caps removed | **CPU (here)** |
| 1 · 1D operators + tile contraction | `operators_1d.py`, `tensor_contract.py` | **CPU (here)** — correctness only (`block_dim=1`) |
| 2 · Q-function extraction | `SeedField` + `qfunction.py` + geometry | **CPU (here)** |
| 3 · Apply + transparent dispatch | `kernels.py` apply, `sumfac_applicable`, `integrate.py` hook | **CPU (here)** — correctness |
| 4 · Assembly to BSR | action-on-unit-vectors → BSR | **CPU (here)** — correctness |
| 5 · Faces + benchmark | surface terms, end-to-end DG, roofline | **GPU required (NOT here)** |

Phases 0–4 are fully **CPU-correctness-verifiable on this Mac**. Phase 5's *correctness* parts
(face fallback, end-to-end DG solve at P≥5) are CPU-verifiable, but the **performance/tensor-core
roofline gate needs a CUDA GPU** and must be deferred to a CUDA machine.

---

## Phase 0 · Arbitrary-P quadrature (CPU)

**Goal:** Replace the hardcoded `n≤5` GL/GLL tables in `polynomial.py` with NumPy-based
arbitrary-P generation, preserving the `[0,1]` shift and the `0.5×` weight-scaling convention,
so that `fem.make_polynomial_space(..., degree=P, discontinuous=True)` constructs for P up to 8.

### Files
- **EDIT** `warp/_src/fem/polynomial.py`
- **CREATE** `warp/tests/fem/test_fem_sumfac_polynomial.py`
- **EDIT** `warp/tests/unittest_suites.py` (register `TestFemSumfacPolynomial`)
- **EDIT** `CHANGELOG.md` (Unreleased: "Extend tensor-product polynomial order beyond P=4 …")

### Public functions / signatures (in `polynomial.py`)
Keep names and 2-tuple return contract identical (callers in `element.py`,
`quadrature.py`, `space/shape/__init__.py` rely on them):
```python
def _gauss_legendre_quadrature_1d(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss--Legendre nodes/weights on [0, 1] for arbitrary n (>= 1)."""
def _lobatto_gauss_legendre_quadrature_1d(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Lobatto--Gauss--Legendre nodes/weights on [0, 1] for arbitrary n (>= 2)."""
```
Algorithm:
- GL: `x, w = np.polynomial.legendre.leggauss(n)`; then shift `x = 0.5*x + 0.5`,
  scale `w = 0.5*w`. (Roots already ascending → preserves node-index ordering, spec risk §9.2.)
- GLL: Newton iteration on `P'_{n-1}` (Jacobi) for interior nodes, endpoints `±1` fixed;
  weights `w_i = 2/(n(n-1) [P_{n-1}(x_i)]^2)`; then the same `[0,1]` shift + `0.5×` scale.
  Implement via `numpy.polynomial.legendre.Legendre` derivative roots, or an explicit
  Newton–Raphson loop. Keep it pure NumPy (no SciPy dependency).
- Remove the `raise NotImplementedError` for `n>5` in both. Leave Newton–Cotes families
  untouched (out of scope; note in docstring).

### Tests to write FIRST (`test_fem_sumfac_polynomial.py`)
CPU-only, pure NumPy. Use `unittest.TestCase` methods directly (fixed device).
1. `test_gl_reproduces_legacy_tables`: for n=1..5, assert new GL output ==
   `np.polynomial.legendre.leggauss(n)` shifted/scaled — **`assert_allclose(..., atol=1e-14)`**.
2. `test_gll_reproduces_legacy_tables`: for n=2..5, assert new GLL nodes/weights match the
   removed hardcoded tables (paste them into the test as the reference) — **`atol=1e-13`**.
3. `test_gl_exactness_degree_2n_minus_1`: for n=1..8, integrate monomials `x^k` on `[0,1]`,
   `k=0..2n-1`; **`assert_allclose(sum(w*x**k), 1/(k+1), atol=1e-12)`**.
4. `test_gll_exactness_degree_2n_minus_3`: for n=2..8, monomials `k=0..2n-3`;
   **`assert_allclose(sum(w*x**k), 1/(k+1), atol=1e-12)`**.
5. `test_weights_sum_to_one`: GL and GLL, n=1..8, **`assert_allclose(sum(w), 1.0, atol=1e-13)`**
   (because of the `0.5×` `[0,1]` convention).
6. `test_high_order_space_constructs`: integration-level — for P in 5..8,
   `fem.make_polynomial_space(Grid3D(...), degree=P, discontinuous=True)` does **not** raise.
   (Anchors the established fact that P=5 previously hit the cap.)

### Run command
```
uv run python warp/tests/fem/test_fem_sumfac_polynomial.py
```
(Or `uv run python -m unittest warp.tests.fem.test_fem_sumfac_polynomial -v`.)

### Acceptance gate
All 6 tests pass on CPU. `make_polynomial_space(... degree=8, discontinuous=True)` constructs.
Existing `warp/tests/fem/test_fem_quadrature.py` still passes (no regression on n≤5 path):
`uv run python warp/tests/fem/test_fem_quadrature.py`.

---

## Phase 1 · 1D operators + E_b-wide tile contraction (CPU correctness)

**Goal:** Build the per-axis 1D interpolation `I` and reference-derivative `D̂` matrices from
basis nodes + quadrature points (barycentric/Lagrange), and the `B`/`Bᵀ` directional contraction
kernels using `tile_matmul` / `tile_reshape` / `tile_view`, correct on CPU (`block_dim=1`).

**Ordering precondition (verified in recon, re-asserted by test):** tensor-product nodes and
quadrature points share lexicographic outer-product order — SQUARE `(i↔x slowest, j↔y fastest)`,
CUBE `(i↔x slowest, j↔y middle, k↔z fastest)`. Node maps:
`square_shape_function.py:86-89`, `cube_shape_function.py:108-115`; QP order `element.py:130,151`.

### Files
- **CREATE** `warp/_src/fem/sumfac/__init__.py`
- **CREATE** `warp/_src/fem/sumfac/operators_1d.py`
- **CREATE** `warp/_src/fem/sumfac/tensor_contract.py`
- **CREATE** `warp/tests/fem/test_fem_sumfac_operators.py`
- **EDIT** `warp/tests/unittest_suites.py` (register `TestFemSumfacOperators`)

### Public functions / signatures
`operators_1d.py` (pure NumPy — host-side matrix construction):
```python
def interpolation_matrix_1d(nodes: np.ndarray, qpoints: np.ndarray) -> np.ndarray:
    """I[q, a] = L_a(qpoints[q]); shape (Q, n). Lagrange basis with roots ``nodes``."""
def derivative_matrix_1d(nodes: np.ndarray, qpoints: np.ndarray) -> np.ndarray:
    """Dhat[q, a] = L_a'(qpoints[q]); shape (Q, n). Reference-derivative of Lagrange basis."""
def basis_nodes_1d(order: int, family: Polynomial) -> np.ndarray:
    """1D node coordinates on [0, 1] for the basis (reuses polynomial.quadrature_1d nodes)."""
def quadrature_points_1d(order: int, family: Polynomial) -> tuple[np.ndarray, np.ndarray]:
    """1D quadrature nodes+weights matching element.py tensoring (GL by default)."""
```
Use barycentric weights from `polynomial.lagrange_scales` for numerically stable `L_a`/`L_a'`.

`tensor_contract.py` (Warp tile kernels + a NumPy dense reference for tests):
```python
def dense_interpolation_matrix(I1d, D1d, dim: int) -> tuple[np.ndarray, ...]:
    """Dense reference B blocks: value (Q^d x n^d) and per-axis grad (Q^d x n^d) via Kronecker."""
def make_forward_contract_kernel(dim: int, n: int, q: int, E_b: int) -> wp.Kernel:
    """B-stage: DOFs (E_b, n^d) -> (value, ref-grad) at QPs (E_b, Q^d, 1+dim) via dim tile_matmuls."""
def make_backward_contract_kernel(dim: int, n: int, q: int, E_b: int) -> wp.Kernel:
    """Bᵀ-stage: (f0, f1) at QPs (E_b, Q^d, 1+dim) -> element residual (E_b, n^d)."""
```
Kernels: one block per `E_b`-element panel; `tile_reshape` nodal field → directional matrices;
`tile_matmul(D1d, U_axis)` per axis; transpose path uses `tile_matmul(transpose(D1d), F_axis)`.
`E_b` is a kwarg defaulting to 1 (recovers one-element-per-block; CPU runs serialized).

### Tests to write FIRST (`test_fem_sumfac_operators.py`)
Use `add_function_test(..., devices=...)` so they run on CPU now and CUDA later. For pure-NumPy
matrix tests, `unittest.TestCase` methods are fine.
1. `test_partition_of_unity`: rows of `I` sum to 1 →
   **`assert_allclose(I.sum(axis=1), np.ones(Q), atol=1e-13)`**, for order 1..8, GL & GLL.
2. `test_derivative_reproduces_polynomials`: for a degree-`order` monomial sampled at nodes,
   `Dhat @ nodal_values` equals the analytic derivative at QPs →
   **`assert_allclose(Dhat @ f_nodal, fprime_at_qp, atol=1e-10)`**, order 1..8.
3. `test_ordering_node_qp_match`: build the dense tensor `B` two ways — (a) Kronecker of 1D
   `I`/`D̂` in lexicographic order, (b) by querying the actual shape function
   `element_inner_weight` at the actual element QPs — and assert they are equal →
   **`assert_allclose(B_kron, B_shapefn, atol=1e-12)`**. (Locks spec risk §9.2: silent ordering.)
4. `test_forward_contract_matches_dense` (2D & 3D, P=1..8): random DOFs;
   **`assert_allclose(sumfac_forward(dofs), B_dense @ dofs, atol=1e-10)`** for value and each
   gradient component. `wp.synchronize_device()` before `.numpy()`.
5. `test_backward_contract_is_transpose` (2D & 3D): `assert_allclose(Bᵀ_sumfac(g),
   B_dense.T @ g, atol=1e-10)`.
6. `test_Eb_panel_equivalence`: same inputs with `E_b=1` vs `E_b=4` give identical per-element
   outputs → **`assert_allclose(out_Eb4, out_Eb1, atol=1e-12)`**.

### Run command
```
uv run python warp/tests/fem/test_fem_sumfac_operators.py
```

### Acceptance gate
All 6 tests pass on CPU. Test 3 (ordering) green is the silent-correctness prerequisite.
Forward/backward contraction matches the dense Kronecker reference at P up to 8 in 2D and 3D.

---

## Phase 2 · Q-function extraction (SeedField) + geometry (CPU)

**Goal:** Add a `SeedField` (an `AdjointField` subclass) whose generated `eval_inner` returns an
**injected value** and `eval_grad_inner` an **injected gradient** instead of reading
`s.test_dof_index`. Because `operator.py` resolves `inner`/`grad`/`div` directly to those
`wp.func`s (`operator.py:323-342`), the user's integrand body is reused verbatim as the pointwise
Q-function. `qfunction.py` evaluates the integrand at `(v=1,∇v=0)`→`f₀` and `(v=0,∇v=eᵢ)`→`f₁ᵢ`
(d+1 evals — exact because the form is linear in the test function), folding geometry
(`∇u=J⁻ᵀ∇̂u`, `weight·|det J|`).

### Files
- **EDIT** `warp/_src/fem/field/virtual.py` (add `SeedField`; mirror `AdjointField`
  `_dynamic_attribute_constructors` at line 33, override `_make_eval_inner` /
  `_make_eval_grad_inner` to read injected seeds from the sample/arg instead of `_get_dof`)
- **CREATE** `warp/_src/fem/sumfac/qfunction.py`
- **CREATE** `warp/tests/fem/test_fem_sumfac_qfunction.py`
- **EDIT** `warp/tests/unittest_suites.py` (register `TestFemSumfacQFunction`)

### Public functions / signatures
`virtual.py`:
```python
class SeedField(AdjointField):
    """AdjointField whose value/gradient are injected seeds, for Q-function extraction.

    ``eval_inner`` returns the seed value and ``eval_grad_inner`` the seed gradient,
    so an integrand body becomes a reusable pointwise Q-function.
    """
```
`qfunction.py`:
```python
def extract_qfunction(integrand, test_field, fields, values, dim: int):
    """Build a wp.func evaluating (f0, f1[dim]) from interpolated (u, ref-grad) at a QP.

    Runs the integrand body d+1 times with seeds (v=1,grad=0) and (v=0,grad=e_i),
    applying ``grad u = J^{-T} ref_grad`` and folding ``weight * |det J|``.
    """
def reference_geometry_factors(domain, sample) -> tuple:
    """Return (J^{-T}, |det J|) at a sample (affine grid first; per-QP for curved)."""
```

### Tests to write FIRST (`test_fem_sumfac_qfunction.py`)
Use `@fem.integrand` bodies in the test file (NOT inline). Compare extracted `(f₀,f₁)` against
known closed forms (spec §8):
1. `test_mass_qfunction`: integrand `u*v` → extraction gives **f₀=u, f₁=0** →
   `assert_allclose(f0, u_at_qp)`, `assert_allclose(f1, 0)`.
2. `test_stiffness_qfunction`: integrand `dot(grad(u),grad(v))` → **f₀=0, f₁=∇u** (after J⁻ᵀ) →
   `assert_allclose(f1, grad_u_physical, atol=1e-9)`.
3. `test_advection_qfunction`: integrand `dot(b,grad(u))*v` → **f₀ = b·∇u, f₁=0**.
4. `test_geometry_factors_affine`: on a uniform `Grid3D`, `|det J|` and `J⁻ᵀ` match the analytic
   constant grid spacing → `assert_allclose`.
5. `test_seedfield_eval_matches_basis_sum`: for a fixed seed, `SeedField.eval_grad_inner`
   equals the corresponding combination of basis gradient weights (sanity vs `AdjointField`).

Each compares against the **naive `integrate()` oracle**: assemble the same form with the existing
path and check the per-QP coefficient (or, where direct QP read is awkward, check the assembled
local block equals `Bᵀ diag(f) B`).

### Run command
```
uv run python warp/tests/fem/test_fem_sumfac_qfunction.py
```

### Acceptance gate
Extracted `(f₀,f₁)` match mass/stiffness/advection closed forms and affine geometry factors on
CPU. SeedField generates without codegen errors (`inspect.getsourcelines()` clean — kernels live
in files).

---

## Phase 3 · Apply + transparent dispatch (CPU correctness) — INTEGRATION GATE

> **Superseded (2026-06):** the transparent dispatch was replaced by an explicit
> `integrate(..., assembly="sumfac")` opt-in; unqualified forms raise instead of falling back.
> See `design/sumfac-status.md`.

**Goal:** Assemble the staged BᵀDB **apply** kernel from Phases 1–2; add `sumfac_applicable`
and the dispatch hook in `_generate_integrate_kernel` (`integrate.py:1110`, before the existing
linear/bilinear branches); expose a matrix-free `LinearOperator`. **Spec risk §9.1 spike first:**
confirm the staged kernel can be generated inside `IntegrandTransformer` /
`cache.get_integrand_kernel`, else use a parallel generator in `sumfac/kernels.py`. Run the spike
at the very start of this phase (timebox ~½ day) before writing `kernels.py`.

### Files
- **CREATE** `warp/_src/fem/sumfac/kernels.py`
- **EDIT** `warp/_src/fem/integrate.py` (add `sumfac_applicable(...)` + dispatch hook at ~1110;
  add a `warp.fem` config flag `FEM_SUMFAC` = `auto|force|off` to override for A/B)
- **CREATE** `warp/tests/fem/test_fem_sumfac_apply.py`
- **EDIT** `warp/tests/unittest_suites.py` (register `TestFemSumfacApply`)
- **EDIT** `CHANGELOG.md` (Unreleased: transparent sum-factorization path)

### Public functions / signatures
`kernels.py`:
```python
def make_sumfac_apply_kernel(integrand, test, trial, quadrature, domain, *, E_b=1) -> wp.Kernel:
    """Fused BᵀDB apply: B (Phase 1) -> Q-function (Phase 2) -> Bᵀ (Phase 1), DG scatter."""
def sumfac_linear_operator(integrand, test, trial, quadrature, domain) -> wp.fem? / LinearOperator:
    """Matrix-free operator whose apply == naive bsr_mv for the same form."""
```
`integrate.py`:
```python
def sumfac_applicable(test, trial, quadrature, domain) -> bool:
    """Conservative predicate: tensor-product space AND tensor-product quadrature AND cell
    domain (no side/face) AND value+grad operators only (no 2nd derivative) AND order above the
    low-P threshold. Anything unproven returns False (falls through to existing branches)."""
```
Hook (pseudocode, inserted before existing branches):
```python
if _sumfac_enabled() and sumfac_applicable(test, trial, quadrature, domain):
    return make_sumfac_apply_kernel(...)
# else: existing linear/bilinear branches unchanged
```

### Tests to write FIRST (`test_fem_sumfac_apply.py`)
Use `add_function_test` across `get_test_devices()`.
1. `test_apply_equals_naive_bsr_mv_mass` (2D & 3D, P=1..8): build the same mass form; assemble
   naive BSR `K`, compute `K @ x` (`bsr_mv`); compute sum-fac apply on `x`;
   **`assert_allclose(sumfac_apply(x), bsr_mv(K, x), rtol=1e-9, atol=1e-10)`**.
2. `test_apply_equals_naive_stiffness` (2D & 3D, P=3..8): same for stiffness.
3. `test_apply_equals_naive_advection`.
4. `test_dispatch_selects_sumfac`: with `FEM_SUMFAC="force"`, the returned kernel is the sum-fac
   kernel (assert kernel name/marker); with `"off"`, it is the legacy kernel; `"auto"` selects
   sum-fac only when `sumfac_applicable` is True.
5. `test_fallback_unqualified`: a side/face domain or a second-derivative form makes
   `sumfac_applicable` return False and `integrate()` still produces the correct legacy result →
   `assert_allclose(result, naive_result)`.
6. `test_matrix_free_dg_solve`: small DG diffusion problem; CG/GMRES with the matrix-free
   operator converges to the same solution as the assembled-matrix solve →
   `assert_allclose(u_matfree, u_assembled, rtol=1e-6)`.

### Run command
```
uv run python warp/tests/fem/test_fem_sumfac_apply.py
```

### Acceptance gate
Sum-fac apply == naive `bsr_mv` for mass/stiffness/advection in 2D & 3D, P up to 8 (CPU
correctness). Dispatch flag honored; unqualified forms fall through with unchanged results.
This is the **integration gate** — no later phase proceeds until this is green.

---

## Phase 4 · Assembly to BSR (CPU correctness)

**Goal:** Produce assembled BSR by running the apply kernel on the `(P+1)^d` unit vectors per
element (action-on-unit-vectors → local block) and feeding the existing
`bsr_set_from_triplets` tail. Result must equal the naive assembled matrix.

### Files
- **EDIT** `warp/_src/fem/sumfac/kernels.py` (add assembly factory)
- **EDIT** `warp/_src/fem/integrate.py` (route the bilinear/assembly branch when
  `sumfac_applicable`)
- **CREATE** `warp/tests/fem/test_fem_sumfac_assembly.py`
- **EDIT** `warp/tests/unittest_suites.py` (register `TestFemSumfacAssembly`)

### Public functions / signatures
```python
def make_sumfac_assembly_kernel(integrand, test, trial, quadrature, domain, *, E_b=1) -> wp.Kernel:
    """Run BᵀDB on the n^d unit vectors per element to emit the dense local block, then scatter
    into BSR triplets (DG: block-diagonal, no inter-element coupling)."""
```

### Tests to write FIRST (`test_fem_sumfac_assembly.py`)
1. `test_assembled_equals_naive_mass` (2D & 3D, P=1..8): build BSR via sum-fac and via naive
   `integrate(...)`; compare dense forms →
   **`assert_allclose(K_sumfac.todense(), K_naive.todense(), rtol=1e-9, atol=1e-10)`**.
2. `test_assembled_equals_naive_stiffness` (2D & 3D, P=3..8).
3. `test_assembled_blockdiag_structure_dg`: assert DG sum-fac matrix has the expected
   block-diagonal sparsity (`nnz` and block layout match naive DG).
4. `test_assembly_apply_consistency`: `K_sumfac @ x == sumfac_apply(x)` →
   `assert_allclose`, closing the loop between Phases 3 and 4.

### Run command
```
uv run python warp/tests/fem/test_fem_sumfac_assembly.py
```

### Acceptance gate
Assembled sum-fac BSR == naive assembled BSR (dense compare) for mass/stiffness, 2D & 3D, P up
to 8, on CPU. Assembly/apply consistency holds.

---

## Phase 5 · Faces + benchmark (GPU REQUIRED for performance)

**Goal:** Add DG **face/surface** terms (side domains) — initially they fall back to the legacy
path (already true via `sumfac_applicable` returning False for side domains), then add a
surface sum-fac path. Reproduce `example_convection_diffusion_dg.py` end-to-end at P≥5 and
produce the speedup / tensor-core roofline curve.

**Hardware split:**
- **CPU-verifiable here:** face integrals correctly fall back; end-to-end DG example runs and
  matches the legacy result at P≥5 (correctness only, slow).
- **GPU-REQUIRED (NOT verifiable on this Mac):** tensor-core throughput, `E_b` sweep, crossover
  vs naive (~P=3 per Roget), ~8.5× at P=5 cooperative GEMM, roofline. cuBLASDx tile path only
  engages on CUDA. Defer all timing work to a CUDA machine.

### Files
- **EDIT** `warp/_src/fem/sumfac/kernels.py` (surface term factory; later)
- **EDIT** `warp/_src/fem/integrate.py` (relax `sumfac_applicable` for qualifying side domains)
- **CREATE** `warp/tests/fem/test_fem_sumfac_faces.py`
- **CREATE** `warp/examples/fem/bench_sumfac_dg.py` (bench script — **no timing asserts in unit
  tests**, spec §8)
- **EDIT** `warp/tests/unittest_suites.py` (register `TestFemSumfacFaces`)
- **EDIT** `CHANGELOG.md`

### Public functions / signatures
```python
# kernels.py
def make_sumfac_surface_kernel(integrand, test, trial, side_domain, *, E_b=1) -> wp.Kernel: ...
# bench_sumfac_dg.py
def benchmark(orders=(1,2,3,4,5,6,7,8), E_b_values=(1,2,4,8), device="cuda"): ...
```

### Tests to write FIRST (`test_fem_sumfac_faces.py`)
1. `test_face_terms_fallback_correct` (CPU): a DG form with interior-face flux integrals over a
   side domain still produces the correct legacy result →
   `assert_allclose(result, naive_result)`.
2. `test_end_to_end_dg_p5` (CPU correctness): run the convection-diffusion DG example at P=5
   with sum-fac forced on; final field matches the legacy-path solution →
   `assert_allclose(u_sumfac, u_legacy, rtol=1e-6)`.
3. **(GPU-only, skipped on CPU via `@unittest.skipUnless(cuda_available)`):**
   `test_surface_sumfac_matches_naive` — once surface sum-fac lands, apply == naive on CUDA.

### Run commands
```
uv run python warp/tests/fem/test_fem_sumfac_faces.py        # CPU correctness parts
uv run python warp/examples/fem/bench_sumfac_dg.py           # GPU only — run on CUDA machine
```

### Acceptance gate
- CPU here: face fallback correct; end-to-end DG at P=5 matches legacy.
- **GPU machine (deferred):** speedup curve produced; crossover ≈ P=3; ≥~8× at P=5; roofline
  shows tensor-core utilization. These performance gates are explicitly **out of scope for this
  CPU-only Mac**.

---

## Phase 6 · Surface sum-factorization (GPU)

**Goal:** Sum-factorize the DG face/flux terms so the complete DG operator (volume + surface)
runs on the fast path. Quad/hex only.

**Math:** On a hex face normal to axis ``a``, the trace of the tensor-product basis factorizes:
- **Value trace:** with GLL (endpoint-including) nodes, ``L_i(0) = δ_{i,0}`` and
  ``L_i(1) = δ_{i,n-1}``, so the face value is simply the boundary **slice** of the DOF tensor —
  no contraction along the normal axis at all.
- **Normal-gradient trace:** one extra 1D contraction collapsing the normal axis with the
  endpoint row of ``D̂`` (``∂u/∂x_a|_face = Σ_i D̂[face_end, i] · u(i, ·)``).
- The remaining ``d−1`` axes contract exactly like the volume case (1D ``I``/``D̂`` against the
  face quadrature points), so the face ``B`` stage is an ``E_b``-wide (d−1)-dimensional version
  of the existing kernels. Flux terms (jump/average) need BOTH the inner and the outer element's
  traces; the outer element's face is traversed in its own local frame, so the node/QP
  **permutation between the two frames** must match what the existing side-domain machinery
  (`Sides`, `element_inner_weight`/`element_outer_weight`) produces — this is the Phase 6 analog
  of the Phase 1 ordering risk (spec §9.8) and gets its own test before any flux kernel.

### Files
- **EDIT** `warp/_src/fem/sumfac/tensor_contract.py` (face-trace contraction kernels)
- **EDIT** `warp/_src/fem/sumfac/qfunction.py` (seeded extraction for side integrands:
  jump/average operators on SeedFields, inner+outer seeds)
- **EDIT** `warp/_src/fem/sumfac/kernels.py` (surface apply/assembly factories)
- **EDIT** `warp/_src/fem/integrate.py` (relax `sumfac_applicable` for qualifying side domains)
- **EDIT** `warp/tests/fem/test_fem_sumfac_faces.py` (extend with surface sum-fac tests)
- **EDIT** `CHANGELOG.md`

### Tests to write FIRST
1. `test_face_trace_ordering`: inner and outer sum-fac traces of a shared face equal the values
   of `element_inner_weight`/`element_outer_weight`-based evaluation at the actual side QPs
   (locks the orientation/permutation risk, spec §9.8).
2. `test_face_value_trace_is_dof_slice`: GLL endpoint property — sum-fac face value == boundary
   slice of the DOF tensor, P=1..8.
3. `test_face_normal_gradient_trace`: endpoint-``D̂``-row contraction == dense reference.
4. `test_surface_apply_equals_naive`: DG flux forms (upwind advection, SIP diffusion penalty)
   on `Sides` domains — sum-fac apply == naive `integrate()`, 2D & 3D, P up to 8.
5. `test_end_to_end_dg_fully_sumfac`: convection-diffusion DG example at P≥5 with BOTH volume
   and surface terms forced onto the sum-fac path matches the legacy solution.

### Acceptance gate
Surface apply == naive on side domains (CPU + CUDA); the end-to-end DG example at P≥5 runs with
no naive fallback anywhere (assert via dispatch instrumentation) and matches legacy. Benchmark
updated: volume+surface speedup curve.

---

## Phase 7 · High-order example adoption (GPU)

**Goal:** Demonstrate the transparent pass on real workloads by adapting existing
`warp/examples/fem/` examples to high polynomial order on the sum-factorization path. The
physics and example structure stay as upstream wrote them — the adaptation is raising
``degree``, switching to tensor-product discontinuous spaces where the original used low-order
ones, and (optionally) printing the sum-fac vs. legacy timing.

### Candidate examples (in priority order)
1. `example_convection_diffusion_dg.py` → `example_convection_diffusion_dg_p5.py` (or a
   ``--degree`` arg on the original): THE target workload — DG volume + flux terms at P≥5.
2. `example_diffusion_3d.py`: hex stiffness operator, the canonical BᵀDB case; matrix-free CG
   at P≥5 vs. assembled legacy solve.
3. `example_burgers.py` or `example_kelvin_helmholtz.py`: nonlinear DG transport — exercises
   repeated `integrate()` calls per step where apply-path speedup compounds.

### Approach
- Prefer adding a ``--degree`` / ``--sumfac {auto,force,off}`` CLI arg to the existing example
  over forking a copy; fork only when high-order needs structural changes (e.g. coarser mesh to
  keep DOF count comparable).
- Each adapted example must run twice in its test: ``sumfac=force`` vs. ``sumfac=off`` on the
  same inputs, asserting field agreement (rtol≈1e-6, matching the Phase 5 end-to-end gate).
- Register in `warp/tests/fem/test_fem_examples.py` following the existing example-test pattern
  (CUDA-gated where runtime demands).
- Report per-example wall-clock speedup at P=5..8 in the PR description / bench notes (no
  timing asserts in tests).

### Acceptance gate
Adapted examples produce fields matching their legacy-path runs at P≥5 on CUDA; example tests
green; per-example speedups recorded alongside the Phase 5 roofline numbers.

---

## Cross-cutting: full-suite regression & finalize

After each phase, run the touched FEM tests plus the existing ones to catch regressions:
```
uv run --extra dev -m warp.tests -s autodetect -k TestFemSumfac -k TestFemQuadrature -k TestFemIntegrate
```
Before any commit:
```
uvx pre-commit run --files <changed files>
```
If `warp/_src/fem/builtins.py`-adjacent or `.pyi`-affecting changes occur (not expected here),
run `uv run --extra docs build_docs.py`. Commit per phase with `git commit -s`, imperative
subject, `(GH-XXX)` if an issue exists. Open the PR with `.github/PULL_REQUEST_TEMPLATE.md`
(or the GitLab template if targeting GitLab), appending CHANGELOG entries for the two user-facing
changes (arbitrary-P order; transparent sum-factorization).

## Test-module registration checklist (do this for every new test file)
In `warp/tests/unittest_suites.py`:
1. Import near line 108: `from warp.tests.fem.test_fem_sumfac_X import TestFemSumfacX`.
2. Append `TestFemSumfacX,` to the `test_classes` list near line 309.
Files to register: `TestFemSumfacPolynomial`, `TestFemSumfacOperators`, `TestFemSumfacQFunction`,
`TestFemSumfacApply`, `TestFemSumfacAssembly`, `TestFemSumfacFaces`.

## Risk → mitigation map (from spec §9, pinned to phases)
- §9.1 codegen invasiveness → **Phase 3 spike** before writing `kernels.py`.
- §9.2 ordering match → **Phase 1 test 3** (`test_ordering_node_qp_match`) locks it.
- §9.3 qualification detection → **Phase 3** `sumfac_applicable` conservative default + flag.
- §9.4 heavy Q-function d+1× → swappable extraction interface in `qfunction.py` (Phase 2).
- §9.5 `tile_matmul` small-n / `E_b` → perf-only, **Phase 5 GPU** sweep.
- §9.6 non-affine geometry → Phase 2 affine first; per-QP Jacobian hook for curved hexes.
- §9.7 AD later → keep tile ops AD-friendly (`adj_tile_matmul` exists).
