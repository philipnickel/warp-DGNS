# Sum-Factorized High-Order DG Operators for `warp.fem`

**Status:** Approved design (2026-06-09) · **Branch:** `philipnickel/sumfac-dg` · **Target:** fork of NVIDIA/Warp (not upstream)

## 1. Goal

Add a **transparent, automatic sum-factorization path** to `warp.fem` so that operator
evaluation (`integrate()` / matrix-free apply) on tensor-product elements (quad/hex) runs at
the sum-factorized complexity **O(d·(P+1)^{d+1})** instead of the naive **O((P+1)^{2d})**, and
**extend polynomial order to arbitrary P** (today hard-capped at P≈4). Primary regime:
**P ≥ 5**, **NVIDIA GPU with tensor cores**, **discontinuous Galerkin** (element-local DOFs).

This is the user-chosen "general DSL compiler pass" deliverable: same `@integrand` / `integrate()`
API, with sum-factorization selected internally and a fallback to the existing kernels.

## 2. Mathematical model — BᵀDB / partial assembly

Any qualifying multilinear form factors per element as **Bᵀ · D · B**:

```
 element DOFs ──B──▶ (u, ∇̂u) at QPs ──D──▶ (f₀, f₁) at QPs ──Bᵀ──▶ element residual
  (P+1)^d           sum-factorized       pointwise Q-function     sum-factorized      (P+1)^d
```

- **B (interpolate):** contract DOFs against 1D interpolation `I` and derivative `D̂` matrices,
  one axis at a time → value and *reference* gradient at every quadrature point.
- **D (Q-function):** pointwise. Apply geometry (∇u = J⁻ᵀ∇̂u, weight·|det J|) and the user's
  integrand body, producing the coefficients `(f₀, f₁)` that multiply `v` and `∇v`.
- **Bᵀ (contract):** transpose contraction of `(f₀, f₁)` back to element DOFs. DG scatters
  directly (no inter-element assembly).

This matches Roget (NumPEx/Exa-DI 2026), deal.II `FEEvaluation`, MFEM/libCEED partial assembly.
For the stiffness operator (Roget slide 7):
`(Ku)_ijk = Σ_α D_αᵀ ( W⊙|J|⊙G^{αβ} ⊙ (D_β u) )_ijk`, with `G̃^{αβ}=W⊙|J|⊙G^{αβ}` precomputed at QPs.

**Apply** = run once on the real DOF vector. **Assembly** = run on the (P+1)^d unit vectors per
element → local block → existing `bsr_set_from_triplets` tail.

## 3. GPU kernel strategy — wider-GEMM tile contraction (tensor-core native)

The directional contraction `D · U` uses the **same** small `D (n×n)` for every element (n=P+1).
We therefore **batch a panel of `E_b` elements into one wider GEMM**: stack their reshaped
directional matrices into `U ∈ ℝ^{n × (n²·E_b)}` and issue a single `wp.tile_matmul(D, U)`.
Widening the RHS with more elements raises arithmetic intensity and feeds **tensor cores**
natively, instead of many tiny `n×n·n×n²` GEMMs.

Kernel structure (one block per `E_b`-element panel, fused, shared-memory scratch), per
Roget's 5-step workflow expressed with Warp tile ops:

1. **Reshape** nodal field → directional matrices via `tile_reshape` (`U_ξ(i,j+kn)=u(i,j,k)` etc.).
2. **Forward** reference derivatives: 3× `tile_matmul(D, U_axis)`.
3. **Pointwise D-stage** on per-element `tile_view`s: geometry + Q-function (§4).
4. **Backward** contraction: 3× `tile_matmul(Dᵀ, F_axis)`.
5. **Inverse reshape + sum** directional contributions → element residual.

`E_b` is a **tuning knob** (default benchmarked; `E_b=1` recovers one-element-per-block). The
contraction is one `tile_matmul`-based kernel that is **correct on CPU** (`block_dim=1`,
serialized) and **fast on GPU** (cuBLASDx / tensor cores; CPU/no-MathDx fallback GEMM exists).
For **mixed-order / p-adaptive** meshes (future), `tile_stack` push/pop buckets elements by
(type, order) into homogeneous panels before the GEMM.

**Empirical anchor (Roget):** crossover vs. naive ≈ P=3; at P=5 the cooperative per-block GEMM
reaches ~8.5×. Below P≈4 the tile path has overhead → **fall back to the naive kernel at low P**.

## 4. Q-function extraction — SeedField

`SeedField` is an `AdjointField` (`warp/_src/fem/field/virtual.py`) whose generated `eval_inner`
returns an injected **value** and `eval_grad_inner` an injected **gradient** instead of reading
`s.test_dof_index`. Because `operator.py` resolves `inner`/`grad`/`div` directly to those
`wp.func`s, the **user's integrand body is reused verbatim** as the pointwise Q-function.

Per quadrature point the D-stage: reads interpolated `(u, ∇̂u)` from B; applies `∇u=J⁻ᵀ∇̂u` and
folds `weight·|det J|`; evaluates the integrand with seeds `(v=1,∇v=0)`→`f₀` and
`(v=0,∇v=eᵢ)`→`f₁ᵢ` (d+1 evals — exact because the form is linear in the test function).
Extraction is **swappable** (seed-field now; one-pass Warp AD later) so a heavy Q-function
isn't penalized by the d+1× factor.

## 5. Transparent dispatch & fallback

Hook in `_generate_integrate_kernel` (`warp/_src/fem/integrate.py:1110`), before the existing
linear/bilinear branches:

```
if sumfac_applicable(test, trial, quadrature, domain):  # tensor-product space + tensor-product
    return get_sumfac_kernel(...)                        # quadrature; linear in test (&trial);
else:                                                    # value+grad operators only; cell domain;
    ... existing branches unchanged ...                  # P above low-order threshold
```

`sumfac_applicable` is conservative — anything unproven falls through. `integrate()` semantics
never change. A `warp.fem` config flag forces it on/off for A/B testing. DG **face** integrals
(side domains) fall back initially (Phase 5); full surface sum-factorization lands in Phase 6 so
that the complete DG operator (volume + flux terms) runs sum-factorized.

## 6. Module layout

```
warp/_src/fem/
├── polynomial.py            [EDIT]  arbitrary-P GL + GLL generation (remove n≤5 caps)
├── sumfac/                  [NEW]
│   ├── __init__.py
│   ├── operators_1d.py        1D I / D̂ matrices from basis nodes + quadrature points (barycentric)
│   ├── tensor_contract.py     B / Bᵀ kernels: E_b-wide tile_matmul + tile_reshape/tile_view
│   ├── qfunction.py           SeedField-based extraction of (f₀, f₁) + geometric factors
│   └── kernels.py             staged BᵀDB kernel factories (apply + assembly)
├── field/virtual.py         [EDIT]  add SeedField
└── integrate.py             [EDIT]  sumfac_applicable + dispatch hook
```

Self-contained subpackage + small reviewable hooks → contained diff against upstream, easy rebase.

## 7. Phased delivery

| Phase | Deliverable | Validation gate | HW |
|---|---|---|---|
| **0 · Arbitrary-P** | `polynomial.py` GL+GLL generation, caps removed | reproduce n≤5 tables ~1e-14; polynomial exactness; P=8 space constructs | CPU |
| **1 · Primitives** | `operators_1d.py` (I, D̂); `tensor_contract.py` (E_b-wide tile_matmul B/Bᵀ) | vs. dense reference, 2D/3D, P=1…8 | CPU |
| **2 · Q-function** | `SeedField` + `qfunction.py` + geometry | extraction matches mass/stiffness/advection coefficients | CPU |
| **3 · Apply + dispatch** | `kernels.py` apply; `sumfac_applicable` + `integrate.py` hook; matrix-free `LinearOperator` | apply == naive `bsr_mv`; matrix-free DG solve | CPU corr. |
| **4 · Assembly** | action-on-unit-vectors → BSR | assembled == naive assembly | CPU corr. |
| **5 · Face fallback + bench** | face integrals fall back correctly; end-to-end DG; speedup + tensor-core roofline | example reproduced at P≥5; speedup curve | **GPU** |
| **6 · Surface sum-fac** | face/flux terms sum-factorized: GLL endpoint traces (value = DOF slice, normal grad = endpoint ``D̂`` row), (d−1)-dim tensor-product contraction, inner+outer traces for jumps/averages | face apply == naive on side domains; DG example fully sum-factorized at P≥5 | **GPU** |
| **7 · High-order examples** | existing `warp/examples/fem/` examples adapted to high order on the sum-fac path (DG convection-diffusion, diffusion 3D, Burgers/Kelvin–Helmholtz) | each example matches its legacy-path result at P≥5; registered in example tests; per-example speedup reported | **GPU** |

Phases 0 and 1 are independent. Phase 3 is the integration gate. **GPU/tensor-core performance
validation (Phase 5) requires a CUDA machine — not the local CPU-only Mac.**

## 8. Testing strategy

The existing naive `integrate()` **is the golden oracle** — every sum-fac result is checked
against it in-repo.

- Arbitrary-P quadrature: reproduce existing tables ~1e-14; exactness to degree 2n-1 (GL)/2n-3 (GLL).
- 1D operators: partition-of-unity (rows of `I` sum to 1); `D̂` reproduces derivatives at QPs.
- Contraction: `assert_allclose(B_dense @ dofs, sumfac_interp(dofs))`, 2D+3D, P=1…8, incl. `E_b>1`.
- SeedField: `(f₀,f₁)` matches known forms (mass→f₀=u,f₁=0; stiffness→f₁=∇u; advection→f₀=b·∇u).
- Apply == naive `bsr_mv`; Assembly == naive BSR.
- End-to-end: reproduce `example_convection_diffusion_dg.py` at P≥5.
- `add_function_test` across `get_test_devices()` (CPU + CUDA); new modules in `default_suite()`.
- Speedup/roofline via `asv` or a bench script — **no timing assertions in unit tests** (flaky).
- `np.testing.assert_allclose` (not `np.allclose`); `wp.synchronize_device()` where needed; never
  clear kernel/LTO caches in tests.

## 9. Risks & open questions (ranked)

1. **Codegen invasiveness (highest):** can the staged kernel be generated inside the existing
   `IntegrandTransformer` / `cache.get_integrand_kernel` machinery, or does it need a parallel
   generator? → resolve with a spike at the start of Phase 3.
2. **Ordering match:** verify `_node_ijk` lexicographic numbering == outer-product quadrature
   order. Silent-correctness risk → pin down & test in Phase 1.
3. **Qualification detection:** test/trial are virtual (linear by construction), so the check
   mainly rejects second-derivative and side/face operators. Conservative default + force/disable flag.
4. **Heavy Q-function (d+1)×:** mitigated by swappable AD extraction; hoist input interpolation into B.
5. **`tile_matmul` at small `n` / `E_b` choice / block mapping:** perf-only; swept by `E_b` knob;
   GPU-validated.
6. **Non-affine geometry:** validate affine grids first, then trilinear/curved hexes (per-QP Jacobian in D).
7. **AD later:** keep ops AD-friendly; `adj_tile_matmul` exists for the backward path.
8. **Face orientation match (Phase 6):** the inner and outer elements of a shared face traverse it
   in their own local coordinate frames; the trace contraction must apply the same node/QP
   permutation that the existing side-domain machinery uses. Silent-correctness risk of the same
   kind as §9.2 — pin down with a dedicated inner/outer trace-equality test before any flux term.

## 10. Alternatives considered

- **Symbolic AST form compiler (TSFC/GEM-style):** most optimal, but a research-grade form
  compiler — over-scope. Documented future direction.
- **Explicit opt-in primitives (libCEED user model):** safe, but not transparent — kept as the
  fallback API beneath the transparent pass.
- **Cross-element large GEMM / unfused global Q-vector (Abdelfattah device-level batch GEMM):**
  higher BLAS throughput at extreme P/scale but more global-memory traffic. Subsumed here as the
  `E_b > 1` wider-GEMM knob within the fused tile kernel.

## References

- A. Roget, *High-Performance Spectral Element Operators on GPUs: A GEMM-Based Reformulation for
  Modern Accelerators*, NumPEx/Exa-DI 2026 Annual Meeting.
- Abdelfattah et al., *High-Order FEM using Standard and Device-Level Batch GEMM on GPUs* (ICL/ORNL).
- *Architecture-aware h-to-p optimisation: spectral/hp element operators for mixed-element meshes*,
  arXiv:2604.04644.
- Kronbichler & Kormann, deal.II matrix-free (`FEEvaluation`); MFEM/libCEED partial assembly.
