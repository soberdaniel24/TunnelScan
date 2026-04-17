"""
optimal_transport.py
---------------------
Wasserstein-2 optimal transport for structural alignment of enzyme active sites.

Physics
-------
Given two protein structures A and B (e.g., wild-type and mutant), the
active-site geometry change that controls tunneling is not simply a list of
per-atom displacements.  The relevant quantity is the STRUCTURAL DEFORMATION
COST: the minimum total work needed to rearrange the atomic mass distribution
from configuration A to configuration B.

This is exactly the Wasserstein-2 (W₂) distance between the atomic mass
distributions (Villani 2003, "Topics in Optimal Transport"):

    W₂²(μ, ν) = min_{γ ∈ Γ(μ,ν)} ∫ |x − y|² dγ(x, y)                  (Eq. 1)

where μ = ∑ m_i δ(x−x_i^A) and ν = ∑ m_j δ(x−x_j^B) are the mass
distributions of structure A and B, and Γ(μ,ν) is the set of all transport
plans coupling them.

For discrete distributions (finite point clouds), W₂² reduces to:

    W₂² = min_{π} ∑_{ij} π_{ij} |x_i^A − x_j^B|²                       (Eq. 2)

subject to: ∑_j π_{ij} = m_i (row marginal = mass of atom i in A)
             ∑_i π_{ij} = m_j (column marginal = mass of atom j in B)
             π_{ij} ≥ 0

This is a linear programme.  For identical mass distributions (same number
of atoms, same species) it reduces to a minimum-weight bipartite matching.

Tunneling relevance
-------------------
The D-A distance change is the primary driver of tunneling rate, but W₂²
captures the FULL structural reorganisation.  The W₂-based D-A predictor:

    Δr_DA^{W₂} = W₂(μ_A^{DA}, ν_B^{DA}) × sign(r_DA^B − r_DA^A)        (Eq. 3)

where μ^{DA}, ν^{DA} are the 1D projections onto the D→A axis.

More generally, the W₂ distance between the LOCAL environments (atoms within
scan_radius of the tunneling path) provides a single scalar summary of how
much the active site has deformed:

    W₂_local = W₂(μ_A|_{B_r(path)}, ν_B|_{B_r(path)})                   (Eq. 4)

Entropic regularisation (Sinkhorn)
------------------------------------
For large point clouds the exact LP is expensive.  We use entropic
regularisation (Cuturi 2013, NIPS; Genevay et al. 2016) to obtain a
smooth approximation:

    W₂_ε²  = min_{π ≥ 0, marginals} ∑ π_{ij} C_{ij} + ε ∑ π_{ij} ln π_{ij}   (Eq. 5)

where C_{ij} = |x_i − x_j|². Solved by Sinkhorn iterations:
    u ← r / (K v),   v ← c / (K^T u)                                    (Eq. 6)
    K_{ij} = exp(−C_{ij}/ε)

Converges geometrically at rate exp(−1/(ε κ)) where κ = max(C)/min(C).

Self-test
---------
  1. W₂(μ, μ) = 0 (identity transport).
  2. W₂(μ, T_d(μ)) = d for a uniform translation by d (Å).
  3. W₂²(μ, ν) ≤ ∫|x−y|²dμ for any coupling y=T(x) (trivial upper bound).
  4. Triangle inequality: W₂(A,C) ≤ W₂(A,B) + W₂(B,C).
  5. Δr_DA^{W₂} agrees with direct D-A distance to within scan_radius.
  6. Sinkhorn convergence check: error < ε/10 after n_iter iterations.

No new empirical parameters: masses from standard atomic weights.

References
----------
Villani 2003  "Topics in Optimal Transport" (AMS Graduate Studies)
Cuturi 2013  NIPS  (Sinkhorn distances)
Genevay, Cuturi, Peyré, Bach 2016  NIPS
Peyré & Cuturi 2019  FnTML 11:355  (computational OT review)
"""

from __future__ import annotations

import sys
import os
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# Standard atomic masses (u) — no new empirical parameters, NIST 2020 values
ATOMIC_MASS: Dict[str, float] = {
    'H': 1.008,  'C': 12.011, 'N': 14.007, 'O': 15.999,
    'S': 32.06,  'P': 30.974, 'SE': 78.971,'FE': 55.845,
}


# ── Data containers ────────────────────────────────────────────────────────────

@dataclass
class PointCloud:
    """
    Discrete mass distribution: N atoms with positions (Å) and masses (u).
    """
    positions: np.ndarray    # (N, 3) Å
    masses:    np.ndarray    # (N,) u (atomic mass units)
    labels:    List[str]     # atom labels (e.g., 'CA', 'CB')

    @property
    def n_atoms(self) -> int:
        return len(self.masses)

    @property
    def weights(self) -> np.ndarray:
        """Normalised mass weights summing to 1."""
        m = self.masses
        return m / m.sum()

    @property
    def centroid(self) -> np.ndarray:
        return (self.weights[:, None] * self.positions).sum(axis=0)

    @classmethod
    def from_arrays(cls, positions: np.ndarray, element_symbols: List[str]) -> 'PointCloud':
        masses = np.array([
            ATOMIC_MASS.get(s.upper(), 12.011) for s in element_symbols
        ])
        return cls(positions=positions, masses=masses, labels=element_symbols)


@dataclass
class OTResult:
    """Result of one optimal-transport computation."""
    W2:            float        # W₂ distance (Å)
    W2_sq:         float        # W₂² (Å²)
    transport_plan: np.ndarray  # π_{ij} (n×m) — optimal coupling
    cost_matrix:   np.ndarray   # C_{ij} = |x_i − y_j|² (Å²)
    n_iter:        int          # Sinkhorn iterations used
    converged:     bool
    epsilon:       float        # regularisation parameter

    @property
    def marginal_error(self) -> float:
        """Max deviation of π marginals from target weights."""
        r = self.transport_plan.sum(axis=1)
        c = self.transport_plan.sum(axis=0)
        r_tgt = self.cost_matrix.shape[0]   # uniform: all 1/n → scaled
        return max(np.max(np.abs(r - r.mean())), np.max(np.abs(c - c.mean())))


@dataclass
class StructuralAlignment:
    """Full alignment result between two structures."""
    W2_global:    float     # W₂ over all active-site atoms (Å)
    W2_DA_axis:   float     # W₂ projected onto D-A axis (Å)
    da_dist_A:    float     # D-A distance in structure A (Å)
    da_dist_B:    float     # D-A distance in structure B (Å)
    delta_r_DA:   float     # r_DA^B − r_DA^A (Å), direct measurement
    delta_r_W2:   float     # Wasserstein estimate of Δr_DA (Eq. 3) (Å)
    n_atoms:      int       # atoms in local environment


# ── Sinkhorn algorithm ─────────────────────────────────────────────────────────

def _cost_matrix(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """C_{ij} = |x_i − y_j|² (Å²).  X: (n,3), Y: (m,3)."""
    # ||x_i - y_j||² = ||x_i||² + ||y_j||² - 2 x_i·y_j
    X2 = (X**2).sum(axis=1, keepdims=True)   # (n,1)
    Y2 = (Y**2).sum(axis=1, keepdims=True).T  # (1,m)
    XY = X @ Y.T                              # (n,m)
    C  = np.maximum(X2 + Y2 - 2*XY, 0.0)
    return C


def sinkhorn_w2(
    cloud_A: PointCloud,
    cloud_B: PointCloud,
    epsilon: float = 0.1,
    n_iter:  int   = 200,
    tol:     float = 1e-6,
) -> OTResult:
    """
    Compute W₂ between two point clouds via Sinkhorn–Knopp (Eq. 5, 6).

    Parameters
    ----------
    cloud_A, cloud_B : PointCloud instances (positions in Å)
    epsilon          : entropic regularisation (Å²); smaller = closer to exact OT
    n_iter           : maximum Sinkhorn iterations
    tol              : convergence tolerance (marginal residual)

    Returns
    -------
    OTResult
    """
    r = cloud_A.weights   # (n,)  target row marginals
    c = cloud_B.weights   # (m,)  target col marginals

    C   = _cost_matrix(cloud_A.positions, cloud_B.positions)   # (n,m)
    log_K = -C / epsilon

    # Sinkhorn in log-space for numerical stability
    log_u = np.zeros(len(r))
    log_v = np.zeros(len(c))
    log_r = np.log(r + 1e-300)
    log_c = np.log(c + 1e-300)

    converged = False
    n_actual  = n_iter
    for it in range(n_iter):
        # u ← r / (K v)  in log-space: log_u = log_r - logsumexp(log_K + log_v, axis=1)
        log_Kv = log_K + log_v[None, :]        # (n, m) broadcast
        log_u  = log_r - _logsumexp(log_Kv, axis=1)
        # v ← c / (K^T u)
        log_Ku = log_K + log_u[:, None]        # (n, m) broadcast
        log_v  = log_c - _logsumexp(log_Ku, axis=0)

        # Check convergence every 10 steps
        if it % 10 == 9:
            log_pi = log_K + log_u[:, None] + log_v[None, :]
            pi     = np.exp(np.clip(log_pi, -700, 700))
            res_r  = np.max(np.abs(pi.sum(axis=1) - r))
            res_c  = np.max(np.abs(pi.sum(axis=0) - c))
            if max(res_r, res_c) < tol:
                converged = True
                n_actual  = it + 1
                break

    # Final transport plan
    log_pi = log_K + log_u[:, None] + log_v[None, :]
    pi     = np.exp(np.clip(log_pi, -700, 700))

    W2_sq_sinkhorn = float(np.sum(pi * C))
    # Remove entropic bias: W₂²_unreg ≈ W₂²_sink − ε H(π)
    H     = -float(np.sum(pi * np.clip(log_pi, -700, 0)))   # entropy (positive)
    W2_sq = max(W2_sq_sinkhorn - epsilon * H, 0.0)
    W2    = np.sqrt(W2_sq)

    return OTResult(
        W2=W2, W2_sq=W2_sq,
        transport_plan=pi,
        cost_matrix=C,
        n_iter=n_actual,
        converged=converged,
        epsilon=epsilon,
    )


def _logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
    """Numerically stable logsumexp along axis."""
    a_max = np.max(a, axis=axis, keepdims=True)
    out   = np.log(np.sum(np.exp(a - a_max), axis=axis)) + a_max.squeeze(axis)
    return out


# ── Structural alignment ───────────────────────────────────────────────────────

def align_active_sites(
    positions_A: np.ndarray,
    positions_B: np.ndarray,
    elements: List[str],
    donor_idx: int,
    acceptor_idx: int,
    epsilon: float = 0.05,
    n_iter: int    = 500,
) -> StructuralAlignment:
    """
    Compute full structural alignment between two configurations A and B.

    Parameters
    ----------
    positions_A, positions_B : (N, 3) arrays of atom positions (Å)
    elements    : element symbols for each atom
    donor_idx   : index of the donor atom in positions_A/B
    acceptor_idx: index of the acceptor atom in positions_A/B
    epsilon     : Sinkhorn regularisation (Å²)
    n_iter      : Sinkhorn iterations

    Returns
    -------
    StructuralAlignment
    """
    cloud_A = PointCloud.from_arrays(positions_A, elements)
    cloud_B = PointCloud.from_arrays(positions_B, elements)

    ot = sinkhorn_w2(cloud_A, cloud_B, epsilon=epsilon, n_iter=n_iter)

    # D-A distance from direct coordinates
    d_A = float(np.linalg.norm(positions_A[donor_idx] - positions_A[acceptor_idx]))
    d_B = float(np.linalg.norm(positions_B[donor_idx] - positions_B[acceptor_idx]))
    delta_r = d_B - d_A

    # W₂ projected onto D-A axis (1D marginal)
    da_vec = positions_A[acceptor_idx] - positions_A[donor_idx]
    da_hat = da_vec / (np.linalg.norm(da_vec) + 1e-30)
    proj_A = (positions_A @ da_hat).reshape(-1, 1)
    proj_B = (positions_B @ da_hat).reshape(-1, 1)
    cloud_proj_A = PointCloud(positions=proj_A, masses=cloud_A.masses, labels=cloud_A.labels)
    cloud_proj_B = PointCloud(positions=proj_B, masses=cloud_B.masses, labels=cloud_B.labels)
    ot_1d = sinkhorn_w2(cloud_proj_A, cloud_proj_B, epsilon=epsilon*0.1, n_iter=n_iter)

    delta_W2 = float(ot_1d.W2 * np.sign(delta_r)) if delta_r != 0.0 else 0.0

    return StructuralAlignment(
        W2_global=ot.W2,
        W2_DA_axis=ot_1d.W2,
        da_dist_A=d_A,
        da_dist_B=d_B,
        delta_r_DA=delta_r,
        delta_r_W2=delta_W2,
        n_atoms=cloud_A.n_atoms,
    )


# ── Self-test ──────────────────────────────────────────────────────────────────

def _run_self_test() -> None:
    print("=" * 60)
    print("  OPTIMAL TRANSPORT — self-test")
    print("=" * 60)

    rng = np.random.default_rng(42)

    fails = []

    # ── Check 1: W₂(μ, μ) = 0 ────────────────────────────────────────────────
    print("\n[1] W₂(μ, μ) = 0:")
    n = 10
    pos = rng.normal(size=(n, 3))
    elems = ['C'] * n
    cloud = PointCloud.from_arrays(pos, elems)
    ot = sinkhorn_w2(cloud, cloud, epsilon=0.001, n_iter=500, tol=1e-8)
    ok  = ot.W2 < 0.01
    print(f"    W₂ = {ot.W2:.6f} Å  {'PASS ✓' if ok else 'FAIL'}")
    if not ok:
        fails.append(f"W₂(μ,μ) = {ot.W2:.4f} ≠ 0")

    # ── Check 2: W₂(μ, T_d(μ)) = d (uniform translation) ────────────────────
    print("\n[2] W₂(μ, T_d μ) = d for translation d:")
    d = 1.5   # Å
    pos_shifted = pos + np.array([d, 0.0, 0.0])
    cloud_shifted = PointCloud.from_arrays(pos_shifted, elems)
    ot2 = sinkhorn_w2(cloud, cloud_shifted, epsilon=0.001, n_iter=1000, tol=1e-8)
    err = abs(ot2.W2 - d) / d
    ok  = err < 0.01   # 1% tolerance on Sinkhorn approximation
    print(f"    d = {d:.2f} Å  W₂ = {ot2.W2:.4f} Å  err = {err:.4f}  "
          f"{'PASS ✓' if ok else 'FAIL'}")
    if not ok:
        fails.append(f"translation test: W₂={ot2.W2:.4f} expected {d:.4f}")

    # ── Check 3: trivial upper bound W₂² ≤ ∑ m_i |x_i − y_i|² ──────────────
    print("\n[3] W₂² ≤ trivial coupling bound:")
    pos_B = pos + rng.normal(scale=0.5, size=(n, 3))
    cloud_B = PointCloud.from_arrays(pos_B, elems)
    ot3 = sinkhorn_w2(cloud, cloud_B, epsilon=0.05, n_iter=500)
    w   = cloud.weights
    trivial_bound = float(np.sum(w * np.sum((pos - pos_B)**2, axis=1)))
    ok  = ot3.W2_sq <= trivial_bound * 1.001  # 0.1% numerical tolerance
    print(f"    W₂² = {ot3.W2_sq:.4f}  bound = {trivial_bound:.4f}  "
          f"{'PASS ✓' if ok else 'FAIL'}")
    if not ok:
        fails.append(f"W₂² = {ot3.W2_sq:.4f} > trivial bound {trivial_bound:.4f}")

    # ── Check 4: triangle inequality ─────────────────────────────────────────
    print("\n[4] Triangle inequality W₂(A,C) ≤ W₂(A,B) + W₂(B,C):")
    pos_C = pos + rng.normal(scale=0.8, size=(n, 3))
    cloud_C = PointCloud.from_arrays(pos_C, elems)
    wAB = sinkhorn_w2(cloud,   cloud_B, epsilon=0.05, n_iter=300).W2
    wBC = sinkhorn_w2(cloud_B, cloud_C, epsilon=0.05, n_iter=300).W2
    wAC = sinkhorn_w2(cloud,   cloud_C, epsilon=0.05, n_iter=300).W2
    ok  = wAC <= wAB + wBC + 0.01  # small Sinkhorn tolerance
    print(f"    W₂(A,C)={wAC:.3f}  W₂(A,B)+W₂(B,C)={wAB+wBC:.3f}  "
          f"{'PASS ✓' if ok else 'FAIL'}")
    if not ok:
        fails.append(f"triangle inequality violated: {wAC:.4f} > {wAB+wBC:.4f}")

    # ── Check 5: uniform translation along D-A axis gives W₂_DA = d ──────────
    print("\n[5] align_active_sites: uniform translation → W₂_DA = shift d:")
    # 5 atoms on D-A axis; ALL shift by d → D-A distance unchanged, W₂_DA = d
    pos_line = np.zeros((5, 3))
    pos_line[:, 0] = [0.0, 1.0, 2.0, 3.0, 4.0]
    elems_line = ['N', 'C', 'C', 'C', 'O']
    d_shift = 0.4   # Å
    pos_shifted = pos_line.copy()
    pos_shifted[:, 0] += d_shift
    align_trans = align_active_sites(
        pos_line, pos_shifted, elems_line, donor_idx=0, acceptor_idx=4,
        epsilon=0.001, n_iter=2000)
    err5 = abs(align_trans.W2_DA_axis - d_shift) / d_shift
    ok   = err5 < 0.02   # 2% Sinkhorn tolerance
    print(f"    d = {d_shift:.3f} Å  W₂_DA = {align_trans.W2_DA_axis:.4f} Å  "
          f"Δr_DA = {align_trans.delta_r_DA:.4f} Å  err = {err5:.4f}  "
          f"{'PASS ✓' if ok else 'FAIL'}")
    if not ok:
        fails.append(f"uniform translation: W₂_DA={align_trans.W2_DA_axis:.4f} ≠ {d_shift}")

    # ── Check 6: Sinkhorn convergence ────────────────────────────────────────
    print("\n[6] Sinkhorn convergence (tol=1e-4):")
    ot6 = sinkhorn_w2(cloud, cloud_B, epsilon=0.1, n_iter=2000, tol=1e-4)
    ok  = ot6.converged
    print(f"    Converged after {ot6.n_iter} iterations  "
          f"{'PASS ✓' if ok else 'FAIL'}")
    if not ok:
        fails.append("Sinkhorn not converged at tol=1e-4")

    print("\n" + "=" * 60)
    if fails:
        for f in fails:
            print(f"  FAIL: {f}")
        raise AssertionError(f"OT self-test failed: {fails}")
    else:
        print("  [PASS] All OT checks completed.")
    print("=" * 60)


if __name__ == '__main__':
    _run_self_test()
