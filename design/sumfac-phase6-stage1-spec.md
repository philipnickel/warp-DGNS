You are implementing Phase 6 STAGE 1 (face-trace primitives + ordering locks) of the sum-factorized DG project on the Warp fork at /root/warp-DGNS (branch philipnickel/sumfac-dg). Do NOT commit; leave changes in the working tree.

READ FIRST (authoritative):
1. design/sumfac-dg-implementation-plan.md — section "Phase 6 · Surface sum-factorization" (the math is spelled out there).
2. design/sumfac-dg-design.md — section 7 row "6 · Surface sum-fac" and risk §9.8 (face orientation match).
3. warp/_src/fem/sumfac/operators_1d.py and tensor_contract.py — the 1D operator construction and the (d-1)-dimensional contraction machinery you will reuse (note: each kernel specialization in its own dynamic module via fem cache.dynamic_kernel, enable_backward=False; never feed a strided tile_view to tile_matmul — copy to a contiguous tile first; block_dim >= 32 on CUDA via _default_block_dim).
4. How warp.fem side machinery works: warp/_src/fem/geometry/grid_2d.py + grid_3d.py (side_arg/side coordinates), warp/_src/fem/space/shape/square_shape_function.py + cube_shape_function.py (element_inner_weight at side coords), and how Sides domains map side sample coords to inner/outer element coords (look for side_inner_cell_coords / side_outer_cell_coords or equivalent in warp/_src/fem/geometry/*.py).
5. warp/tests/fem/test_fem_sumfac_apply.py for test conventions; AGENTS.md for all rules (especially "Tile-kernel development speed").

THE MATH (from the plan):
On a quad/hex face normal to axis ``a`` of a tensor-product element with GLL (endpoint-including) basis nodes:
- Value trace: L_i(0) = delta_{i,0}, L_i(1) = delta_{i,n-1}, so the face value tensor is literally the boundary SLICE of the element DOF tensor along axis a (no contraction).
- Normal-gradient trace: one 1D contraction collapsing axis a with the endpoint row of the derivative matrix D_ref (build_derivative_matrix evaluated at coordinate 0.0 or 1.0).
- The remaining d-1 axes contract against the face quadrature points exactly like the volume case (1D interpolation/derivative matrices), so the face B stage is the existing (d-1)-dimensional contraction applied to the sliced/collapsed DOF tensor.
- THE RISK (§9.8): the inner and outer elements of a shared interior face traverse the face in their own local coordinate frames. The mapping between the face quadrature points and each element's local axes (including possible axis reversal/permutation) MUST match what the existing side machinery produces. Do not guess this mapping — extract it empirically from warp.fem itself in the tests (evaluate element_inner_weight / element_outer_weight at actual side sample coords) and encode what you find as host-side index maps.

DELIVERABLES:
1. NEW FILE warp/_src/fem/sumfac/face_trace.py: host-side NumPy construction of the face-trace operators:
   - face_slice_indices(n, dim, axis, end) -> index array selecting the face DOF slice from the lexicographic element DOF vector (value trace).
   - normal_derivative_row(nodes, end) -> the (n,) endpoint row of the 1D derivative matrix (use operators_1d.build_derivative_matrix with points=[0.0] or [1.0]).
   - face_trace_value(dofs, n, dim, axis, end) and face_trace_normal_gradient(dofs, n, dim, axis, end) -> NumPy reference implementations operating on (num_elements, n^dim) lexicographic DOFs, returning (num_elements, n^(dim-1)) face-node values / normal derivatives in the face's own lexicographic order (document which element axes map to face axes and in what order).
   This stage is NumPy-only (reference semantics + index maps); the tile kernels come in stage 2. SPDX 2026 header, Google docstrings.
2. NEW FILE warp/tests/fem/test_fem_sumfac_face_trace.py (register TestFemSumfacFaceTrace in default_suite in warp/tests/unittest_suites.py):
   a. test_face_value_trace_is_dof_slice: for P in {1, 3, 5}, dim in {2, 3}, every axis and both ends: face_trace_value == dense evaluation of the element's nodal interpolant restricted to the face at the face nodes (build via element shape functions or via Kronecker of 1D interpolation matrices evaluated at GLL nodes with the face coordinate pinned to 0/1). This pins the GLL endpoint property end-to-end.
   b. test_face_normal_gradient_trace: same sweep; face_trace_normal_gradient == analytic derivative of a random polynomial interpolant evaluated on the face (build the dense reference via Kronecker with the endpoint derivative row on the collapsed axis).
   c. test_face_trace_ordering_matches_side_machinery (THE CRITICAL ONE, CPU-only is fine): build a small Grid2D (and Grid3D) discontinuous space at P=3; create a Sides domain and a RegularQuadrature on it; for a handful of interior sides, evaluate the discrete field's inner and outer values at the actual side quadrature points using the existing warp.fem machinery (e.g. fem.interpolate or a tiny @fem.integrand storing fem.inner(u, s) and fem.outer(u, s) per qp — kernels in the test .py file), and compare against: face_trace_value of the inner/outer elements' DOFs contracted to the side QPs with the (d-1)-dim 1D interpolation matrix, using YOUR index maps plus whatever side-to-element coordinate mapping you derived. assert_allclose atol=1e-12. If inner passes but outer needs an axis flip/permutation, encode that mapping as a documented function in face_trace.py (e.g. outer_face_permutation(...)) — that function IS the deliverable that de-risks stage 2.
3. Document in the face_trace.py module docstring: the exact element-axis -> face-axis convention, the side orientation convention you observed from warp.fem (with file:line references), and what stage 2 will consume.

RULES (hard): uv run only; run test files directly (never -m warp.tests); unittest + np.testing.assert_allclose; no wp.clear_kernel_cache/clear_lto_cache; degrees {1,3,5} max (cached shapes); pre-commit --files at the end. If CUDA is unavailable in your environment, validate on CPU and say so.

SUMMARY: files changed, tests added, exact unittest summary lines, the orientation/permutation findings (this is the most important part — be precise about what mapping warp.fem uses between side QPs and inner/outer element frames), and any deviations.
