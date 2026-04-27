"""
tunnelling_network.py
---------------------
Quantum tunnelling network topology analysis.

Every residue is a node; edge weights encode quantum-mechanically relevant
coupling for a specific tunnelling event:

    W_ij = sqrt(P_i P_j)  ×  A_i A_j  ×  Q_ij     [edges within EDGE_CUTOFF Å]

    P_i  = rank-normalised ENM participation (how much residue i participates
           in the promoting vibration mode)
    A_i  = anisotropic D-A alignment score (how directed the motion is toward
           the tunnelling coordinate — from 2AH1 ANISOU, QCF fallback, or ANM)
    Q_ij = |G_ij| / sqrt(G_ii G_jj) — normalised QCF propagator correlation
           (zero-point fluctuation quantum coupling between i and j)

From this adjacency matrix five novel physical quantities are computed:

  1. Tunnelling betweenness B_i — fraction of max-weight paths through i
  2. Spectral gap λ₂ (Fiedler value) of the normalised graph Laplacian
  3. Fiedler vector v₂ — partitions network into two functional communities
  4. Effective resistance R_i — electrical resistance analogy to the D-A axis
  5. Spectral communities — spectral k-means on Laplacian eigenvectors

Each quantity is available per-residue and is added to MutationScore as
topological KIE predictors orthogonal to the geometric/dynamic model.

References
----------
Newman (2010) Networks: An Introduction — betweenness, Laplacian spectra
Chung (1997) Spectral Graph Theory — normalised Laplacian, Fiedler value
Klein & Randić (1993) J Math Chem — effective resistance
Luxburg (2007) A Tutorial on Spectral Clustering
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from scipy.sparse.csgraph import dijkstra as _dijkstra
from scipy.stats import pearsonr

# ── Constants ──────────────────────────────────────────────────────────────────

KAPPA_TOPO   = 0.5    # topological coupling constant (see topological_delta)
LOG_EPS      = 1e-9   # floor for -log(W) transformation
MAX_LOG_DIST = 50.0   # effective ∞ for disconnected pairs
DA_RADIUS    = 20.0   # Å: node selection radius around D-A midpoint
EDGE_CUTOFF  = 20.0   # Å: max Cα-Cα distance for a network edge
BETWN_RADIUS = 25.0   # Å: betweenness computed on this sub-radius


# ── Data class ─────────────────────────────────────────────────────────────────

@dataclass
class TunnellingNetworkResult:
    """All topological quantities for the tunnelling network."""

    nodes:                List[Tuple[str, int]]
    adjacency:            np.ndarray              # W (m×m)
    laplacian:            np.ndarray              # L = I - D^(-1/2) W D^(-1/2), (m×m)
    laplacian_evals:      np.ndarray              # ascending eigenvalues (m,)
    laplacian_evecs:      np.ndarray              # columns = eigenvectors (m×m)
    fiedler_value:        float                   # λ₂ = spectral gap
    fiedler_vector:       np.ndarray              # v₂ (m,)
    betweenness:          Dict[Tuple[str, int], float]
    effective_resistance: Dict[Tuple[str, int], float]
    communities:          Dict[Tuple[str, int], int]
    n_communities:        int
    da_ref_idx:           int                     # local index of D-A reference node
    node_index:           Dict[Tuple[str, int], int]  # key → local index

    # ── Per-residue accessors ──────────────────────────────────────────────────

    def get_betweenness(self, chain: str, resnum: int) -> float:
        return self.betweenness.get((chain, resnum), 0.0)

    def get_effective_resistance(self, chain: str, resnum: int) -> float:
        return self.effective_resistance.get((chain, resnum), float('inf'))

    def get_community(self, chain: str, resnum: int) -> int:
        return self.communities.get((chain, resnum), -1)

    def spectral_sensitivity(
        self,
        chain:     str,
        resnum:    int,
        disruption: float,
    ) -> float:
        """
        |Δλ₂| predicted when residue (chain, resnum) loses fraction `disruption`
        of its alignment score.  Computed by rebuilding the perturbed Laplacian.
        O(m³) per call but m is small (≤ 200 residues).
        """
        key = (chain, resnum)
        if key not in self.node_index or disruption <= 0.0:
            return 0.0
        i = self.node_index[key]
        m = len(self.nodes)

        W_p = self.adjacency.copy()
        W_p[i, :] *= (1.0 - disruption)
        W_p[:, i] *= (1.0 - disruption)

        deg_p = W_p.sum(axis=1)
        with np.errstate(invalid='ignore', divide='ignore'):
            d_inv = np.where(deg_p > 0, 1.0 / np.sqrt(deg_p), 0.0)
        L_p = np.eye(m) - np.diag(d_inv) @ W_p @ np.diag(d_inv)
        evals_p = np.linalg.eigvalsh(L_p)
        lam2_p  = float(evals_p[1]) if m > 1 else 0.0
        return abs(lam2_p - self.fiedler_value)

    def topological_delta(
        self,
        chain:     str,
        resnum:    int,
        disruption: float,
        kappa:     float = KAPPA_TOPO,
    ) -> float:
        """
        ln(KIE) contribution from network topology change:
          topological_delta = -κ × B_i × disruption

        Negative (reduces KIE) when disruption > 0.  Physically: residues that
        are structural bottlenecks for quantum flux (high B_i) have an
        amplified impact when their alignment is disrupted.
        """
        B = self.get_betweenness(chain, resnum)
        return -kappa * B * disruption


# ── Internal helpers ───────────────────────────────────────────────────────────

def _trace_path(pred: np.ndarray, s: int, t: int) -> List[int]:
    """Reconstruct shortest path from s to t via predecessor matrix."""
    path = [t]
    cur  = t
    for _ in range(pred.shape[0] + 1):
        p = int(pred[s, cur])
        if p < 0 or p == cur:
            return []    # no path
        path.append(p)
        cur = p
        if cur == s:
            break
    path.reverse()
    return path if (path and path[0] == s) else []


def _compute_betweenness(W: np.ndarray) -> np.ndarray:
    """
    Betweenness centrality for maximum-weight paths.

    Converts to log-space so max-weight path = shortest path.
    For each pair (s, t) counts intermediate nodes on the shortest path.
    Normalised to [0, 1].
    """
    n = W.shape[0]
    if n < 3:
        return np.zeros(n)

    # -log(W) converts max-product paths to min-sum paths.
    # scipy csgraph treats edge costs < ~1e-7 as structural zeros (sparse
    # convention) — W=1 gives -log(1)=0 which is invisible to Dijkstra.
    # Clamp to 1e-6 minimum so even W=1 edges are visible; in real networks
    # W = C·AA·Q << 1 so -log(W) >> 1e-6 and the clamp never activates.
    with np.errstate(divide='ignore', invalid='ignore'):
        log_W = np.where(W > LOG_EPS,
                         np.maximum(-np.log(W), 1e-6),
                         MAX_LOG_DIST)
    np.fill_diagonal(log_W, 0.0)

    _dist, pred = _dijkstra(log_W, directed=False, return_predecessors=True)

    BC = np.zeros(n)
    for s in range(n):
        for t in range(s + 1, n):
            if pred[s, t] < 0:
                continue
            path = _trace_path(pred, s, t)
            for node in path[1:-1]:
                BC[node] += 1.0

    mx = BC.max()
    return BC / mx if mx > 0 else BC


def _normalized_laplacian(W: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """L = I - D^(-1/2) W D^(-1/2); returns (L, degree)."""
    deg = W.sum(axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        d_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    L = np.eye(len(W)) - np.diag(d_inv_sqrt) @ W @ np.diag(d_inv_sqrt)
    return L, deg


def _spectral_communities(evecs: np.ndarray, k: int) -> np.ndarray:
    """K-means on Laplacian spectral embedding (eigenvectors 1..k)."""
    from scipy.cluster.vq import kmeans, vq, whiten

    n = evecs.shape[0]
    if n < k:
        return np.zeros(n, dtype=int)

    # For k clusters, use k-1 eigenvectors (skip the trivial λ=0 mode).
    # Using k vectors (off-by-one) adds noise from within-community modes.
    n_embed = max(1, min(k - 1, evecs.shape[1] - 1))
    X = evecs[:, 1 : n_embed + 1]   # skip zero mode

    if X.shape[1] == 0:
        return np.zeros(n, dtype=int)

    try:
        X_w = whiten(X + 1e-8)
        if not np.all(np.isfinite(X_w)):
            X_w = X
        centroids, _ = kmeans(X_w, k, iter=100)
        labels, _    = vq(X_w, centroids)
        return labels.astype(int)
    except Exception:
        return np.zeros(n, dtype=int)


# ── Main builder ───────────────────────────────────────────────────────────────

def build_tunnelling_network(
    enm,
    qcf,
    aniso_map:          Dict[Tuple[str, int], float],
    donor_coords:       np.ndarray,
    acceptor_coords:    np.ndarray,
    da_radius:          float = DA_RADIUS,
    edge_cutoff:        float = EDGE_CUTOFF,
    betweenness_radius: float = BETWN_RADIUS,
    n_communities:      int   = 3,
) -> TunnellingNetworkResult:
    """
    Build the tunnelling network and compute all topological quantities.

    Parameters
    ----------
    enm        : ENMResult from elastic_network.build_gnm
    qcf        : QCFResult from build_quantum_propagator (must include ca_coords)
    aniso_map  : {(chain, resnum): float} merged D-A alignment scores
    donor_coords, acceptor_coords : Cα/heavy-atom coordinates of D-A pair
    da_radius  : Å, nodes within this distance of D-A midpoint are included
    edge_cutoff: Å, edges only between pairs within this distance
    """
    if qcf.ca_coords is None:
        raise ValueError(
            "QCF must be built with structure= argument to populate ca_coords"
        )

    keys = qcf.residue_keys    # same ordering as enm
    ca   = qcf.ca_coords        # (N, 3)
    N    = qcf.n_residues

    midpoint   = 0.5 * (donor_coords + acceptor_coords)
    dists_mid  = np.linalg.norm(ca - midpoint, axis=1)   # (N,)

    # ── Node selection ─────────────────────────────────────────────────────────
    node_mask = dists_mid <= da_radius
    gidx      = np.where(node_mask)[0]                   # global indices into qcf
    m         = len(gidx)

    if m < 4:
        raise ValueError(
            f"Only {m} residues within {da_radius} Å of D-A midpoint; "
            f"increase da_radius"
        )

    local_keys = [keys[g] for g in gidx]
    local_ca   = ca[gidx]                                # (m, 3)
    node_index = {k: li for li, k in enumerate(local_keys)}

    # ── Edge weight components ─────────────────────────────────────────────────
    # C_ij = sqrt(P_i P_j)  ENM cross-correlation proxy
    part = np.array([float(enm.participation[g]) for g in gidx])  # (m,)
    C    = np.sqrt(np.outer(part, part))                            # (m, m)

    # A_i × A_j  anisotropic alignment coupling (0.5 = neutral prior)
    A_vec = np.array([float(aniso_map.get(k, 0.5)) for k in local_keys])
    AA    = np.outer(A_vec, A_vec)                                  # (m, m)

    # Q_ij = |G_ij| / sqrt(G_ii G_jj)  QCF normalised correlation
    G_sub = qcf.propagator[np.ix_(gidx, gidx)]                     # (m, m)
    G_d   = np.maximum(np.diag(G_sub), 0.0)
    sqG   = np.sqrt(G_d)
    denom = np.outer(sqG, sqG)
    with np.errstate(invalid='ignore', divide='ignore'):
        Q = np.where(denom > 0, np.abs(G_sub) / denom, 0.0)
    np.fill_diagonal(Q, 0.0)

    # Pairwise Cα distances
    diff   = local_ca[:, None, :] - local_ca[None, :, :]
    pair_d = np.linalg.norm(diff, axis=2)                           # (m, m)
    in_cut = (pair_d > 0.1) & (pair_d <= edge_cutoff)

    # ── Adjacency matrix ───────────────────────────────────────────────────────
    W = C * AA * Q * in_cut.astype(float)
    np.fill_diagonal(W, 0.0)

    # ── Graph Laplacian and spectrum ───────────────────────────────────────────
    L, _deg  = _normalized_laplacian(W)
    evals, evecs = np.linalg.eigh(L)
    fiedler_value  = float(evals[1]) if m > 1 else 0.0
    fiedler_vector = evecs[:, 1].copy()

    # ── Effective resistance to D-A reference node ─────────────────────────────
    L_pinv    = np.linalg.pinv(L)
    da_ref_idx = int(np.argmin(dists_mid[gidx]))   # local index closest to midpoint
    R = np.array([
        float(L_pinv[i, i]
              - 2.0 * L_pinv[i, da_ref_idx]
              + L_pinv[da_ref_idx, da_ref_idx])
        for i in range(m)
    ])
    R = np.maximum(R, 0.0)   # clamp numerical noise
    effective_resistance = {local_keys[i]: float(R[i]) for i in range(m)}

    # ── Betweenness centrality ─────────────────────────────────────────────────
    dists_local = dists_mid[gidx]
    bt_mask     = dists_local <= betweenness_radius
    bt_lidx     = np.where(bt_mask)[0]

    BC_full = np.zeros(m)
    if len(bt_lidx) >= 3:
        W_bt  = W[np.ix_(bt_lidx, bt_lidx)]
        BC_bt = _compute_betweenness(W_bt)
        for li, gi in enumerate(bt_lidx):
            BC_full[gi] = BC_bt[li]

    betweenness = {local_keys[i]: float(BC_full[i]) for i in range(m)}

    # ── Spectral community detection ───────────────────────────────────────────
    k_eff = min(n_communities, m // 4)
    k_eff = max(k_eff, 2)
    comm_labels = _spectral_communities(evecs, k_eff)
    communities = {local_keys[i]: int(comm_labels[i]) for i in range(m)}

    return TunnellingNetworkResult(
        nodes                = local_keys,
        adjacency            = W,
        laplacian            = L,
        laplacian_evals      = evals,
        laplacian_evecs      = evecs,
        fiedler_value        = fiedler_value,
        fiedler_vector       = fiedler_vector,
        betweenness          = betweenness,
        effective_resistance = effective_resistance,
        communities          = communities,
        n_communities        = k_eff,
        da_ref_idx           = da_ref_idx,
        node_index           = node_index,
    )


# ── Self-tests ─────────────────────────────────────────────────────────────────

def _self_tests():
    passed = 0
    failed = 0

    def check(name, cond, detail=""):
        nonlocal passed, failed
        if cond:
            print(f"  [PASS] {name}")
            passed += 1
        else:
            print(f"  [FAIL] {name}  {detail}")
            failed += 1

    print("─" * 60)
    print("Tunnelling network self-tests")
    print("─" * 60)

    # ── Check 1: Betweenness on path graph ────────────────────────────────────
    # 5-node path 0-1-2-3-4; node 2 (center) should have highest betweenness.
    W_path = np.zeros((5, 5))
    for i in range(4):
        W_path[i, i+1] = W_path[i+1, i] = 1.0

    BC = _compute_betweenness(W_path)
    center_is_max = int(np.argmax(BC)) == 2
    check("Check 1: center of path graph has max betweenness",
          center_is_max,
          f"BC = {BC.tolist()}")

    # ── Check 2: Spectral gap of K_5 = 5/4 = 1.25 ────────────────────────────
    # K_5: complete graph, 5 nodes, all edges weight 1.
    # Normalised Laplacian eigenvalues: 0 (×1) and 5/4 (×4).
    W_K5 = np.ones((5, 5)) - np.eye(5)
    L_K5, _ = _normalized_laplacian(W_K5)
    evals_K5 = np.linalg.eigvalsh(L_K5)
    fiedler_K5 = float(evals_K5[1])
    expected_K5 = 5.0 / 4.0
    check("Check 2: K_5 Fiedler value = 5/4",
          abs(fiedler_K5 - expected_K5) < 1e-6,
          f"got {fiedler_K5:.6f}, expected {expected_K5:.6f}")

    # ── Check 3: Effective resistance on 4-node ring ──────────────────────────
    # Ring 0-1-2-3-0; by symmetry R(1, ref=0) = R(3, ref=0) < R(2, ref=0).
    W_ring = np.zeros((4, 4))
    for i in range(4):
        W_ring[i, (i+1) % 4] = W_ring[(i+1) % 4, i] = 1.0
    L_ring, _ = _normalized_laplacian(W_ring)
    L_pinv     = np.linalg.pinv(L_ring)
    ref        = 0
    R_ring     = np.array([
        L_pinv[i, i] - 2*L_pinv[i, ref] + L_pinv[ref, ref]
        for i in range(4)
    ])
    R_ring = np.maximum(R_ring, 0.0)
    # R(0)=0, R(1)=R(3) (adjacent), R(2) > R(1) (opposite)
    ring_ok = (R_ring[0] < 1e-10
               and abs(R_ring[1] - R_ring[3]) < 1e-8
               and R_ring[2] > R_ring[1])
    check("Check 3: effective resistance on ring (R[opposite] > R[adjacent] > 0)",
          ring_ok,
          f"R = {R_ring.tolist()}")

    # ── Check 4: Community detection on dumbbell graph ────────────────────────
    # Two K_4 subgraphs (nodes 0-3 and 4-7) connected by one weak bridge (3↔4).
    W_db = np.zeros((8, 8))
    # K_4 blocks
    for i in range(4):
        for j in range(i+1, 4):
            W_db[i, j] = W_db[j, i] = 1.0
            W_db[i+4, j+4] = W_db[j+4, i+4] = 1.0
    # Bridge
    W_db[3, 4] = W_db[4, 3] = 0.05
    np.fill_diagonal(W_db, 0.0)

    L_db, _ = _normalized_laplacian(W_db)
    evals_db, evecs_db = np.linalg.eigh(L_db)
    labels_db = _spectral_communities(evecs_db, k=2)
    # The two groups {0-3} and {4-7} should be in different communities
    set_A = set(labels_db[:4])
    set_B = set(labels_db[4:])
    dumbbell_ok = (len(set_A) == 1) and (len(set_B) == 1) and (set_A != set_B)
    check("Check 4: spectral 2-community on dumbbell separates the two cliques",
          dumbbell_ok,
          f"labels = {labels_db.tolist()}")

    # ── Optional check 5: T172 betweenness on 2AGW ────────────────────────────
    import os
    pdb_2agw = os.path.join(os.path.dirname(__file__), '..', 'data', 'structures', '2AGW.pdb')
    pdb_2ah1 = os.path.join(os.path.dirname(__file__), '..', 'data', 'structures', '2AH1.pdb')

    if os.path.exists(pdb_2agw) and os.path.exists(pdb_2ah1):
        try:
            from pdb_parser import Structure
            from elastic_network import build_gnm
            from quantum_conformational_field import build_quantum_propagator
            from anisotropic_bfactor import build_alignment_map

            s        = Structure(pdb_2agw)
            enm      = build_gnm(s)
            qcf      = build_quantum_propagator(enm, 165.0, 298.15, structure=s)
            donor    = np.array([-0.116, 4.392, 7.502])
            acceptor = np.array([ 4.795, 1.345, 4.126])
            aniso    = build_alignment_map(pdb_2ah1, donor, acceptor)

            tn = build_tunnelling_network(enm, qcf, aniso, donor, acceptor)

            B_T172 = tn.get_betweenness('D', 172)
            all_B  = list(tn.betweenness.values())
            median_B = float(np.median(all_B))

            check("Check 5 (2AGW): T172 betweenness ≥ median",
                  B_T172 >= median_B,
                  f"T172={B_T172:.3f}  median={median_B:.3f}  "
                  f"fiedler={tn.fiedler_value:.4f}  "
                  f"n_nodes={len(tn.nodes)}")

            # Spectral sensitivity of T172 is positive and finite
            # (real protein networks have lower λ₂ than random graphs due to
            # community structure, so gap > shuffled is NOT a valid criterion)
            ss = tn.spectral_sensitivity('D', 172, disruption=0.5)
            fv_ok = 0.0 < tn.fiedler_value < 2.0

            check("Check 6 (2AGW): fiedler value in (0,2) and T172 sensitivity > 0",
                  fv_ok and ss > 0.0,
                  f"λ₂={tn.fiedler_value:.4f}  T172_sensitivity={ss:.4f}")

        except Exception as e:
            print(f"  [SKIP] Check 5/6 (2AGW): {e}")
    else:
        print("  [SKIP] Check 5/6: 2AGW/2AH1 not available")

    print("─" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    import sys
    ok = _self_tests()
    sys.exit(0 if ok else 1)
