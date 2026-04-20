"""
topological_analysis.py
-----------------------
Persistent homology of the enzyme active-site residue contact network.

Physics / Mathematical Background
----------------------------------
Persistent homology (Edelsbrunner & Harer 2010, *Computational Topology*,
AMS) tracks topological features (connected components, loops, voids) of a
point cloud as a filtration parameter ε grows.  For the Vietoris-Rips
complex:

    VR(X, ε) = { σ ⊂ X : d(x,y) ≤ ε  ∀ x,y ∈ σ }                 (Eq. 1)

A k-dimensional simplex σ = {x₀,…,xₖ} is added at the scale ε(σ) =
max_{i<j} d(xᵢ, xⱼ).  The birth–death pair (b, d) of a homological class
records when a k-cycle is born at ε = b and filled at ε = d.  Persistence
is p = d − b.

Active-site relevance
---------------------
The donor–acceptor tunnel imposes a 1-cycle (loop) in the contact graph
whose persistence p₁ encodes the topological rigidity of the hydrophobic
cage.  Mutations that alter this loop (e.g. T172V, I14V in DHFR) are
detected as changes in the H₁ persistence diagram.

The 0th-dimensional persistence captures connected-component structure:
long-lived H₀ pairs indicate disconnected sub-networks (e.g. flexible
loops not in contact with the core).

Algorithm (matrix reduction)
-----------------------------
We implement the standard boundary-matrix reduction algorithm
(Zomorodian & Carlsson 2005, *Discrete Comput Geom* 33:249) in Fₚ (p=2):

    Reduce D to R by left-to-right column operations:
    while ∃ j₁ < j₂ with pivot(R[j₁]) == pivot(R[j₂]):
        R[j₂] += R[j₁]  (mod 2)

Persistence pairs: σᵢ (born at εᵢ) is paired with σⱼ (born at εⱼ > εᵢ)
if pivot(R[j]) == i.  Unpaired simplices are essential classes.

Complexity: O(N³) worst case for N simplices.  We cap the filtration at
ε_max = 10 Å and cap the complex at 3-simplices (tetrahedra) so that N
stays tractable for active-site point clouds (≤ 50 residues → ≤ 50 000
simplices at typical density).

Self-test
---------
  1. Triangle (equilateral, side a): one H₀ class born at ε=0, dies at ε=a;
     one H₁ class born at ε=a, dies at ε=a (boundary fills immediately).
  2. Square (unit side): H₁ class born at ε=1 (loop appears), dies at
     ε=√2 (diagonal fills it).  Persistence = √2 − 1.
  3. Uniform translation invariance: moving all points by a rigid shift
     leaves the persistence diagram unchanged.
  4. Merge test: two disjoint clusters → H₀ class with persistence equal
     to the distance between nearest neighbours across clusters.
  5. Betti numbers: for a tetrahedron (4 points, all mutual edges ≤ ε_fill)
     β₀ = 1, β₁ = 0 after full filling.

No new empirical parameters.

References
----------
Edelsbrunner & Harer 2010  Computational Topology, AMS
Zomorodian & Carlsson 2005  Discrete Comput Geom 33:249
Carlsson 2009  Bull Amer Math Soc 46:255 (review)
Kovacev-Nikolic et al. 2016  Mol BioSyst 12:2357 (topology of biomolecules)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class Simplex:
    """A k-simplex with its filtration value."""
    vertices:  Tuple[int, ...]  # sorted vertex indices, length = dim+1
    filt_val:  float            # ε at which this simplex is added

    @property
    def dim(self) -> int:
        return len(self.vertices) - 1


@dataclass
class PersistencePair:
    """A birth–death pair in dimension k."""
    dim:        int
    birth:      float           # ε at birth
    death:      float           # ε at death (np.inf for essential classes)
    birth_simplex: Tuple[int, ...]
    death_simplex: Tuple[int, ...]

    @property
    def persistence(self) -> float:
        return self.death - self.birth

    @property
    def is_essential(self) -> bool:
        return np.isinf(self.death)


@dataclass
class PersistenceDiagram:
    """Complete persistence diagram for a point cloud."""
    pairs:      List[PersistencePair]
    n_points:   int
    labels:     Optional[List[str]]   # residue labels if supplied

    def pairs_in_dim(self, k: int) -> List[PersistencePair]:
        return [p for p in self.pairs if p.dim == k]

    def betti(self, k: int, epsilon: float) -> int:
        """β_k(ε): number of alive k-classes at scale ε."""
        return sum(
            1 for p in self.pairs_in_dim(k)
            if p.birth <= epsilon and (p.is_essential or p.death > epsilon)
        )

    def total_persistence(self, k: int) -> float:
        """Sum of persistence values for finite pairs in dimension k."""
        return sum(p.persistence for p in self.pairs_in_dim(k) if not p.is_essential)

    def most_persistent(self, k: int, n: int = 5) -> List[PersistencePair]:
        """Top n most persistent finite pairs in dimension k."""
        finite = [p for p in self.pairs_in_dim(k) if not p.is_essential]
        return sorted(finite, key=lambda p: p.persistence, reverse=True)[:n]


# ── Vietoris-Rips filtration ───────────────────────────────────────────────────

def _pairwise_distances(points: np.ndarray) -> np.ndarray:
    """Euclidean distance matrix, shape (n, n)."""
    diff = points[:, None, :] - points[None, :, :]   # (n,n,d)
    return np.sqrt(np.sum(diff**2, axis=-1))


def build_vietoris_rips(
    points:    np.ndarray,
    epsilon_max: float = 10.0,
    max_dim:   int    = 2,
) -> List[Simplex]:
    """
    Build the Vietoris-Rips filtration up to dimension max_dim (Eq. 1).

    Parameters
    ----------
    points      : (n, d) array of point coordinates (e.g. Cα positions in Å)
    epsilon_max : cutoff distance; edges with d > epsilon_max are excluded
    max_dim     : highest simplex dimension to include (0=vertices, 1=edges,
                  2=triangles, 3=tetrahedra)

    Returns
    -------
    List of Simplex objects sorted by filtration value then dimension.
    """
    n = len(points)
    dist = _pairwise_distances(points)

    simplices: List[Simplex] = []

    # Vertices (0-simplices): all born at ε=0
    for i in range(n):
        simplices.append(Simplex(vertices=(i,), filt_val=0.0))

    if max_dim < 1:
        return sorted(simplices, key=lambda s: (s.filt_val, s.dim))

    # Edges (1-simplices): add if d_ij ≤ epsilon_max
    edge_list = []
    for i in range(n):
        for j in range(i + 1, n):
            d = dist[i, j]
            if d <= epsilon_max:
                edge_list.append((d, i, j))

    for d, i, j in edge_list:
        simplices.append(Simplex(vertices=(i, j), filt_val=d))

    if max_dim < 2:
        return sorted(simplices, key=lambda s: (s.filt_val, s.dim))

    # Build adjacency set for triangle enumeration
    nbrs: Dict[int, set] = {i: set() for i in range(n)}
    for d, i, j in edge_list:
        nbrs[i].add(j)
        nbrs[j].add(i)

    # Triangles (2-simplices): {i,j,k} iff all three edges present
    for i in range(n):
        for j in sorted(nbrs[i]):
            if j <= i:
                continue
            for k in sorted(nbrs[i] & nbrs[j]):
                if k <= j:
                    continue
                fv = max(dist[i, j], dist[j, k], dist[i, k])
                if fv <= epsilon_max:
                    simplices.append(Simplex(vertices=(i, j, k), filt_val=fv))

    if max_dim < 3:
        return sorted(simplices, key=lambda s: (s.filt_val, s.dim))

    # Tetrahedra (3-simplices): {i,j,k,l} iff all C(4,2)=6 edges present
    # Build triangle lookup for efficiency
    tri_set: set = set()
    for s in simplices:
        if s.dim == 2:
            tri_set.add(s.vertices)

    for i in range(n):
        for j in sorted(nbrs[i]):
            if j <= i:
                continue
            for k in sorted(nbrs[i] & nbrs[j]):
                if k <= j:
                    continue
                for l in sorted(nbrs[i] & nbrs[j] & nbrs[k]):
                    if l <= k:
                        continue
                    fv = max(dist[i, j], dist[i, k], dist[i, l],
                             dist[j, k], dist[j, l], dist[k, l])
                    if fv <= epsilon_max:
                        simplices.append(Simplex(vertices=(i, j, k, l), filt_val=fv))

    return sorted(simplices, key=lambda s: (s.filt_val, s.dim))


# ── Boundary matrix reduction (Zomorodian-Carlsson algorithm) ─────────────────

def _boundary(simplex: Simplex) -> List[Tuple[int, ...]]:
    """
    Faces of a simplex (boundary operator ∂_k).

    For σ = {v₀, …, vₖ}, ∂σ = Σᵢ {v₀,…,v̂ᵢ,…,vₖ}  (omit vertex i).
    Returns sorted tuples for lookup.
    """
    verts = simplex.vertices
    return [tuple(verts[:i] + verts[i+1:]) for i in range(len(verts))]


def compute_persistence(
    simplices: List[Simplex],
) -> PersistenceDiagram:
    """
    Standard persistence algorithm (Zomorodian & Carlsson 2005).

    Works over F₂ (arithmetic mod 2): addition = XOR on column bit-vectors.

    Returns
    -------
    PersistenceDiagram with all birth–death pairs.
    """
    n = len(simplices)
    # Map vertex-tuple → index in filtration order
    simplex_index: Dict[Tuple[int, ...], int] = {
        s.vertices: idx for idx, s in enumerate(simplices)
    }

    # Boundary matrix D stored as list of sets (non-zero row indices, mod 2)
    D: List[set] = []
    for s in simplices:
        if s.dim == 0:
            D.append(set())
        else:
            col = set()
            for face in _boundary(s):
                idx = simplex_index.get(face)
                if idx is not None:
                    col.add(idx)
            D.append(col)

    # Reduced matrix R (same structure); low[j] = max row index in R[j]
    R: List[set] = [set(col) for col in D]

    def _low(col: set) -> int:
        return max(col) if col else -1

    pivot_col: Dict[int, int] = {}   # row → leftmost column with that pivot

    for j in range(n):
        while R[j]:
            low_j = _low(R[j])
            if low_j in pivot_col:
                R[j] ^= R[pivot_col[low_j]]   # XOR = mod-2 addition
            else:
                pivot_col[low_j] = j
                break

    # Extract pairs
    paired: set = set()
    pairs: List[PersistencePair] = []

    for j in range(n):
        if R[j]:
            i = _low(R[j])
            paired.add(i)
            paired.add(j)
            b = simplices[i].filt_val
            d = simplices[j].filt_val
            if d > b:   # non-trivial
                pairs.append(PersistencePair(
                    dim=simplices[i].dim,
                    birth=b,
                    death=d,
                    birth_simplex=simplices[i].vertices,
                    death_simplex=simplices[j].vertices,
                ))

    # Unpaired simplices → essential classes (death = ∞)
    for i, s in enumerate(simplices):
        if i not in paired:
            pairs.append(PersistencePair(
                dim=s.dim,
                birth=s.filt_val,
                death=np.inf,
                birth_simplex=s.vertices,
                death_simplex=(),
            ))

    return pairs


# ── Public API ─────────────────────────────────────────────────────────────────

def analyse_active_site(
    positions:   np.ndarray,
    labels:      Optional[List[str]] = None,
    epsilon_max: float = 10.0,
    max_dim:     int   = 2,
) -> PersistenceDiagram:
    """
    Compute the Vietoris-Rips persistent homology of an active-site point cloud.

    Parameters
    ----------
    positions   : (n, 3) array of Cα coordinates (Å)
    labels      : optional list of n residue labels
    epsilon_max : filtration cutoff in Å (default 10 Å, typical contact range)
    max_dim     : highest simplex dimension (default 2)

    Returns
    -------
    PersistenceDiagram
    """
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must be (n, 3)")

    simplices = build_vietoris_rips(positions, epsilon_max=epsilon_max, max_dim=max_dim)
    pairs = compute_persistence(simplices)
    return PersistenceDiagram(pairs=pairs, n_points=len(positions), labels=labels)


def tunnel_loop_persistence(diagram: PersistenceDiagram) -> Optional[float]:
    """
    Return the persistence of the most prominent H₁ loop.

    In the tunnelling context this encodes the topological rigidity of the
    hydrophobic cage enclosing the D-A pair.  Returns None if no finite H₁
    pair exists.
    """
    top = diagram.most_persistent(k=1, n=1)
    return top[0].persistence if top else None


def mutation_topology_delta(
    wt_diagram:  PersistenceDiagram,
    mut_diagram: PersistenceDiagram,
    k:           int = 1,
) -> float:
    """
    Bottleneck-like distance between WT and mutant persistence diagrams in
    dimension k (heuristic: compare sorted persistence vectors).

    d_B ≈ max_i |p_i^WT − p_i^mut|  where p_i are sorted finite persistences.
    Exact bottleneck distance requires the Hungarian algorithm; this
    approximation is sufficient for ranking mutations.
    """
    def _pers_vec(dgm: PersistenceDiagram) -> np.ndarray:
        vals = sorted(
            [p.persistence for p in dgm.pairs_in_dim(k) if not p.is_essential],
            reverse=True,
        )
        return np.array(vals, dtype=float)

    wt_v  = _pers_vec(wt_diagram)
    mut_v = _pers_vec(mut_diagram)

    # Pad to equal length with zeros
    n = max(len(wt_v), len(mut_v))
    if n == 0:
        return 0.0
    wt_v  = np.pad(wt_v,  (0, n - len(wt_v)))
    mut_v = np.pad(mut_v, (0, n - len(mut_v)))
    return float(np.max(np.abs(wt_v - mut_v)))


# ── Self-test ──────────────────────────────────────────────────────────────────

def _run_self_test() -> None:
    print("=" * 60)
    print("  TOPOLOGICAL ANALYSIS — self-test")
    print("=" * 60)

    fails = []

    # ── Check 1: triangle ──────────────────────────────────────────────────
    # Equilateral triangle with side a=2.0 Å
    print("\n[1] Equilateral triangle (side a=2): H₁ born at a, dies at a → persistence=0")
    a = 2.0
    pts_tri = np.array([
        [0.0,      0.0,       0.0],
        [a,        0.0,       0.0],
        [a / 2.0,  a * np.sqrt(3) / 2.0, 0.0],
    ])
    dgm_tri = analyse_active_site(pts_tri, epsilon_max=a + 0.1)
    h0_tri = dgm_tri.pairs_in_dim(0)
    h1_tri = dgm_tri.pairs_in_dim(1)
    # One H₀ essential (the whole component) plus two finite H₀ pairs at ε=a
    h0_finite = [p for p in h0_tri if not p.is_essential]
    h0_essential = [p for p in h0_tri if p.is_essential]
    ok1a = len(h0_essential) == 1
    # For equilateral triangle all edges have length a; H₁ is born at ε=a when
    # the loop closes, and immediately filled by the 2-face → p = 0 (zero-persistence)
    h1_finite = [p for p in h1_tri if not p.is_essential]
    ok1b = all(abs(p.persistence) < 1e-12 for p in h1_finite)
    ok = ok1a and ok1b
    print(f"    H₀ essential={len(h0_essential)} (expect 1)  "
          f"H₁ finite pairs={len(h1_finite)} all p=0: {ok1b}  "
          f"{'PASS ✓' if ok else 'FAIL'}")
    if not ok:
        fails.append(f"triangle: H₀_ess={len(h0_essential)}, H₁_p0={ok1b}")

    # ── Check 2: square ────────────────────────────────────────────────────
    # Unit square (side 1); diagonal = √2
    # H₁ class born at ε=1 (loop), dies at ε=√2 (filled by long diagonal) → p = √2-1
    print("\n[2] Unit square: H₁ born at ε=1, dies at ε=√2, persistence=√2-1")
    pts_sq = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    dgm_sq = analyse_active_site(pts_sq, epsilon_max=2.0)
    h1_sq = [p for p in dgm_sq.pairs_in_dim(1) if not p.is_essential]
    expected_p = np.sqrt(2.0) - 1.0
    # The main H₁ pair should have persistence close to √2−1
    top_h1 = sorted(h1_sq, key=lambda p: p.persistence, reverse=True)
    if top_h1:
        top_p = top_h1[0].persistence
        err2 = abs(top_p - expected_p) / expected_p
        ok2  = err2 < 0.01
    else:
        top_p = None
        ok2   = False
    print(f"    Top H₁ persistence={top_p}  expected={expected_p:.6f}  "
          f"err={err2 if top_h1 else 'N/A'}  {'PASS ✓' if ok2 else 'FAIL'}")
    if not ok2:
        fails.append(f"square H₁: {top_p} vs {expected_p:.6f}")

    # ── Check 3: translation invariance ───────────────────────────────────
    print("\n[3] Rigid translation → same persistence diagram:")
    shift = np.array([[5.0, 3.0, -2.0]])
    pts_shifted = pts_sq + shift
    dgm_shifted = analyse_active_site(pts_shifted, epsilon_max=2.0)
    h1_sh = [p for p in dgm_shifted.pairs_in_dim(1) if not p.is_essential]
    top_sh = sorted(h1_sh, key=lambda p: p.persistence, reverse=True)
    if top_sh and top_h1:
        ok3 = abs(top_sh[0].persistence - top_h1[0].persistence) < 1e-10
    else:
        ok3 = False
    print(f"    Original={top_p:.8f}  Shifted={top_sh[0].persistence:.8f}  "
          f"{'PASS ✓' if ok3 else 'FAIL'}")
    if not ok3:
        fails.append("translation broke persistence")

    # ── Check 4: two disjoint clusters ────────────────────────────────────
    # Two triangles separated by gap G: nearest-neighbour distance = G
    # → H₀ class with persistence G (one cluster merged at ε=G)
    print("\n[4] Two disjoint clusters: H₀ persistence = gap distance:")
    G = 5.0
    # Cluster A centred left of 0; cluster B starts at G → nearest gap = G exactly
    pts_cl = np.array([
        [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-0.5, 0.8, 0.0],   # cluster A (rightmost at x=0)
        [G,    0.0, 0.0], [G+1, 0.0, 0.0], [G+0.5, 0.8, 0.0],  # cluster B (leftmost at x=G)
    ])
    dgm_cl = analyse_active_site(pts_cl, epsilon_max=G + 2.0)
    h0_cl  = [p for p in dgm_cl.pairs_in_dim(0) if not p.is_essential]
    # The H₀ pair that dies latest should die at ε = G (nearest-neighbour gap)
    h0_cl_sorted = sorted(h0_cl, key=lambda p: p.death, reverse=True)
    if h0_cl_sorted:
        merge_eps = h0_cl_sorted[0].death
        err4 = abs(merge_eps - G) / G
        ok4  = err4 < 0.01
    else:
        merge_eps = None
        ok4 = False
    print(f"    Merge ε={merge_eps:.4f}  expected={G}  "
          f"{'PASS ✓' if ok4 else 'FAIL'}")
    if not ok4:
        fails.append(f"cluster merge at {merge_eps}, expected {G}")

    # ── Check 5: tetrahedron Betti numbers ────────────────────────────────
    # 4 points at unit simplex corners; all mutual distances = 1 (regular tetrahedron)
    # After filling all tetrahedra β₀=1, β₁=0, β₂=0
    print("\n[5] Tetrahedron: β₀=1, β₁=0, β₂=0 after full filling:")
    s = 1.0
    pts_tet = np.array([
        [0.0,         0.0,              0.0],
        [s,           0.0,              0.0],
        [s / 2.0,     s * np.sqrt(3) / 2.0, 0.0],
        [s / 2.0,     s / (2.0 * np.sqrt(3)), s * np.sqrt(2.0 / 3.0)],
    ])
    dgm_tet = analyse_active_site(pts_tet, epsilon_max=2.0, max_dim=3)
    eps_fill = s * 1.001   # just above the edge length
    b0 = dgm_tet.betti(0, eps_fill)
    b1 = dgm_tet.betti(1, eps_fill)
    b2 = dgm_tet.betti(2, eps_fill)
    ok5 = (b0 == 1) and (b1 == 0) and (b2 == 0)
    print(f"    β₀={b0} β₁={b1} β₂={b2}  (expect 1,0,0)  "
          f"{'PASS ✓' if ok5 else 'FAIL'}")
    if not ok5:
        fails.append(f"tetrahedron betti=({b0},{b1},{b2})")

    print("\n" + "=" * 60)
    if fails:
        for f in fails:
            print(f"  FAIL: {f}")
        raise AssertionError(f"Topological analysis self-test failed: {fails}")
    else:
        print("  [PASS] All topology checks completed.")
    print("=" * 60)


if __name__ == '__main__':
    _run_self_test()
