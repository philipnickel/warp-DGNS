"""NumPy oracle for the sum-factorized 2D DG cell apply (mass + Laplace stiffness).

Mirrors the production sumfac volume kernel structure on an affine square grid:
per element R = A^T (mc*W o V) A + D^T (W o Vx) A + A^T (W o Vy) D with
V = A U A^T, Vx = D U A^T, Vy = A U D^T, W = outer(w, w).

For square elements of size h with reference domain [-1, 1]^2 the Laplace
geometric factor is exactly 1 (detJ * (2/h)^2 = 1) and the mass factor is
mc = h^2 / 4, so the D stage is a constant elementwise mask -- the cell-apply
analog of the side kernels' side-constant geometry injection.
"""

import numpy as np


def gll_nodes(n: int) -> np.ndarray:
    """Gauss-Lobatto-Legendre nodes on [-1, 1] (n points, degree n-1 basis)."""
    from numpy.polynomial import legendre as leg  # noqa: PLC0415

    c = np.zeros(n)
    c[-1] = 1.0
    interior = leg.legroots(leg.legder(c))
    return np.concatenate(([-1.0], interior, [1.0]))


def lagrange_operators(nodes: np.ndarray, qpts: np.ndarray):
    """Barycentric Lagrange interpolation and derivative matrices (q, n)."""
    n = len(nodes)
    w_bary = np.ones(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                w_bary[i] /= nodes[i] - nodes[j]

    A = np.zeros((len(qpts), n))
    D = np.zeros((len(qpts), n))
    for k, x in enumerate(qpts):
        diff = x - nodes
        on_node = np.isclose(diff, 0.0, atol=1e-14)
        if on_node.any():
            i = int(np.argmax(on_node))
            A[k, i] = 1.0
            for j in range(n):
                if j != i:
                    D[k, j] = w_bary[j] / (w_bary[i] * (x - nodes[j]))
            D[k, i] = -np.sum(D[k, np.arange(n) != i])
        else:
            t = w_bary / diff
            s = t.sum()
            A[k] = t / s
            # l_j'(x) = t_j'/s - l_j * s'/s with t_j' = -t_j/(x-x_j),
            # s' = -sum_k t_k/(x-x_k)  (barycentric second form).
            D[k] = -(t / diff) / s + A[k] * (np.sum(t / diff) / s)
    return A, D


def make_setup(n: int, q: int, h: float):
    """Operators and the D-stage mask. Returns (A, D, op, WQ, mc)."""
    from numpy.polynomial import legendre as leg  # noqa: PLC0415

    nodes = gll_nodes(n)
    qpts, qw = leg.leggauss(q)
    A, D = lagrange_operators(nodes, qpts)
    op = np.vstack([A, D])  # (2q, n) stacked [A; D]
    W = np.outer(qw, qw)
    mc = h * h / 4.0
    # Mask over the (2q, 2q) S = [V, Vy; Vx, junk] quadrant layout.
    WQ = np.zeros((2 * q, 2 * q))
    WQ[:q, :q] = mc * W
    WQ[q:, :q] = W  # Vx quadrant, stiffness factor 1 on affine square cells
    WQ[:q, q:] = W  # Vy quadrant
    return A, D, op, WQ, mc


def reference_apply(U: np.ndarray, op: np.ndarray, WQ: np.ndarray) -> np.ndarray:
    """Apply mass+stiffness to U (E, n, n) via the same 4-GEMM structure."""
    T = np.einsum("mi,eij->emj", op, U)
    S = np.einsum("emj,kj->emk", T, op)
    F = S * WQ[None]
    X = np.einsum("im,emk->eik", op.T, F)
    return np.einsum("eik,kj->eij", X, op)


def reference_apply_direct(U, A, D, WQ, q):
    """Independent check: quadrant algebra written out explicitly."""
    V = np.einsum("qi,eij,rj->eqr", A, U, A)
    Vx = np.einsum("qi,eij,rj->eqr", D, U, A)
    Vy = np.einsum("qi,eij,rj->eqr", A, U, D)
    F00 = WQ[:q, :q][None] * V
    F10 = WQ[q:, :q][None] * Vx
    F01 = WQ[:q, q:][None] * Vy
    R = np.einsum("qi,eqr,rj->eij", A, F00, A)
    R += np.einsum("qi,eqr,rj->eij", D, F10, A)
    R += np.einsum("qi,eqr,rj->eij", A, F01, D)
    return R


def pad_pow2(x: int) -> int:
    p = 1
    while p < x:
        p *= 2
    return p


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for n in (4, 5):
        q = n
        _, _, op, WQ, _ = make_setup(n, q, h=1.0 / 128.0)
        U = rng.standard_normal((7, n, n))
        A, D = op[:q], op[q:]
        r1 = reference_apply(U, op, WQ)
        r2 = reference_apply_direct(U, A, D, WQ, q)
        np.testing.assert_allclose(r1, r2, rtol=1e-12, atol=1e-14)
        # Patch-test sanity: stiffness of a constant is zero; mass is positive.
        Uc = np.ones((1, n, n))
        Rc = reference_apply(Uc, op, WQ)
        mass_only = (1.0 / 128.0) ** 2 / 4.0
        print(
            f"n={n}: oracle forms agree; const-field residual sum "
            f"{Rc.sum():.6e} (pure mass, expect ~h^2={mass_only * 4:.3e} scale)"
        )
    print("oracle self-check PASS")
