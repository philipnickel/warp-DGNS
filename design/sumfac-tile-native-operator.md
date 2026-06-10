# Tile-Native Sum-Factorized DG Operators

**Status**: In Progress

**Issue**: none (fork design doc; see `design/sumfac-status.md`)

## Motivation

The fused sum-factorized kernels behind `fem.integrate(..., assembly="sumfac")`
mix tile collectives (`tile_matmul`) with scalar per-thread code: a flat
gather loop calling `read_node_value` per node, and a per-quadrature-point
loop evaluating the seeded integrand. The scalar gather is an anti-pattern in
the tile programming model, and is the structure on which nvJitLink's LTO
optimizer miscompiles the 2D linear kernel at n=5 into an infinite loop
(reproduced across nvJitLink 12.9/13.0/13.3; see the known-issues entry in
`design/sumfac-status.md`).

Independently, NVIDIA's tile-programming direction (Warp tiles today, cuTile /
Tile IR next — see the cuTile deep dive, https://www.youtube.com/watch?v=YFrP03KuMZ8)
makes the performance contract explicit: the system automates block
parallelism, memory movement (sync/`cp.async`/TMA), memory-tier placement, and
tensor-core mapping **for code expressed as array operations on
constant-shaped tiles**. Scalar loops, per-element writes, and views are
either unsupported (cuTile forbids tile element assignment and views entirely)
or opaque to the optimizer. Designing the DG operator *around* that contract
yields kernels that are simpler, faster-by-construction, portable to future
tile stacks, and free of the structure that triggers the linker bug.

## Requirements

| ID  | Requirement                                                                  | Priority | Notes |
| --- | ---------------------------------------------------------------------------- | -------- | ----- |
| R1  | All B/Bᵀ-stage data movement and FLOPs expressed as tile collectives          | Must     | `tile_load`/`tile_matmul`/`tile_reshape`/elementwise; no scalar gather |
| R2  | Keep the seeded-integrand D stage (arbitrary `@integrand` bodies)             | Must     | generality of `integrate()` is non-negotiable |
| R3  | Identical results to the legacy path at the existing P=4 oracle tests        | Must     | rtol 1e-9 vs default assembly |
| R4  | No regression of the explicit `assembly="sumfac"` API and error contract     | Must     | unsupported layouts raise `SumfacNotApplicableError` |
| R5  | Multi-element batching expressible as a tile-shape change, not a new layout  | Should   | Roget-style panels stay open (E_b machinery retained) |
| R6  | Dimension-generic emitter (one body for 2D/3D/faces)                          | Could    | follow-up; 2D stacked + 3D rotation proven in spike |

**Non-goals**: changing warp.fem public APIs; unstructured-mesh support beyond
the current predicate; even-odd decomposition (incompatible with a GEMM-pure
formulation and skipped even by deal.II's GPU port).

## Design

### Representation: elements are the tile space

For the qualifying sumfac case (regular discontinuous tensor-product space,
whole space partition), the DOF storage is already element-major contiguous:
`dof_values[e * n^d + node]`. Element `e` is therefore literally the `e`-th
tile of the DOF array — the mesh element space *is* the tile space, and the
intra-tile indices are the DOFs. The B-stage gather is one `tile_load` with
`offset=(element_index * n^d,)` plus a `tile_reshape`; the topology and
partition indirections vanish from the hot path. This requires (and the
layout predicate now enforces) a `WholeSpacePartition` on the input field.

On grid geometries this extends naturally to a d-dimensional tile space (the
field as one `(Ex·n, Ey·n)` array; faces = adjacent tiles; boundaries = ghost
padding), which is the intended basis for the Phase 6 face kernels — a face
trace is a mode-k contraction with a 1-row operator, and the neighbor is the
adjacent tile. `fem.cells()` / `to_inner_cell` remain the warp.fem-level
counterparts for relating side and cell views.

### Operator algebra: stacked-operator GEMM pipeline

At its core a sum-factorized DG operator is d applications of one primitive —
contract one axis of a small d-way tensor with a small 1D matrix (a mode-k
product = one GEMM after flattening) — around one pointwise stage. Values,
gradients, and face traces differ only in which 1D matrix each axis receives
(A, D̂, or an endpoint row).

The 2D fused apply uses the **stacked operator** `Â = [A; D̂]` of shape
`(2q, n)`, assembled in-kernel from the existing interp/deriv arrays:

1. **B stage** — `S = Â U Âᵀ` (two GEMMs): the `(2q, 2q)` tile holds all
   channels as blocks — values `S[0:q,0:q]`, ∂ξ `S[q:,0:q]`, ∂η `S[0:q,q:]`,
   and a cross block (zero-weighted unless an anisotropic coefficient uses it).
2. **D stage** — unchanged seeded-integrand loop, now reading channel values
   from `S` blocks and writing `f₀/f₁` into the matching blocks of a
   zero-initialized `(2q, 2q)` tile `F`. This is the generality boundary where
   warp.fem's integrand machinery plugs in.
3. **Bᵀ stage** — `R = Âᵀ F Â` (two GEMMs); block algebra collapses exactly to
   `Aᵀf₀A + D̂ᵀf₁ξA + Aᵀf₁ηD̂`.

GEMM shapes at P=4 f64 go from five `(5,5)×(5,5)` to four GEMMs at
`(10,5)/(5,10)` — twice the MMA row fill — and cuBLASDx engages tensor cores
automatically where the hardware has them. The 3D kernel keeps its existing
(already GEMM-based) slab structure and only replaces the scalar gather with
`tile_load`; the rotation-based dimension-generic emitter validated in the
spike is the follow-up that unifies 2D/3D/faces (R6).

### What the system owns

Batching, memory tiers, load strategy, and tensor-core mapping are the tile
runtime's job. E_b appears nowhere in the kernel structure — a wider panel is
a different constant tile shape over the same storage (validated in the
spike). The only residual knobs are `block_dim` and tile-shape constants, both
single tunables. Discipline carried over from cuTile's value semantics:
intermediates are copies, never strided views into `tile_matmul` (Warp's known
sharp edge).

### Alternatives considered

- **Nested/unrolled gather loops**: dodges the linker bug but keeps the scalar
  gather anti-pattern; rejected.
- **Per-channel separate tiles (status quo)**: more, smaller GEMMs; no path to
  padding/stacking benefits.
- **Toolchain substitution**: nvJitLink 12.9/13.0/13.3 all reproduce the
  miscompilation; not actionable.

### Validation spike

`design/spike_tile_native_gemm/spike_tile_gemm.py` validates the formulation
in isolation against dense Kronecker references on CPU and CUDA at n ∈
{4, 5, 6}: 2D bilinear (mass + anisotropic stiffness) in 4 GEMMs + 1 Hadamard,
E_b=4 panels with the wide first contraction, and 3D mass via axis rotation —
including n=5 on the RTX 5090, the shape on which the scalar-gather kernel is
miscompiled.

## Testing Strategy

The existing P=4 suites are the oracle (`test_fem_sumfac_apply/assembly/faces`
vs the default assembly path, CPU + CUDA). The previously failing CUDA cases
(2D linear apply at n=5: `test_apply_equals_naive_mass_2d`,
`test_affine_form_matches_default`, `test_assembly_apply_consistency`) become
the regression canaries for the rewrite: they must pass once the gather is
tile-native. The contract tests keep one E_b>1 and one rectangular q≠n case;
host-side NumPy suites are unaffected.
