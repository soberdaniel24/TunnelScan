"""
anisotropic_network_model.py
----------------------------
Anisotropic Network Model (ANM) for computing 3D residue displacement modes.

ANM extends GNM from scalar to vector displacements:
  GNM: Kirchhoff matrix (N×N), scalar fluctuations → B-factor magnitudes
  ANM: Hessian matrix (3N×3N), vector fluctuations → B-factor directions

The Hessian is built from inter-residue Cα vectors:
  H_ij = -(γ/r_ij²) r̂_ij ⊗ r̂_ij   for i≠j, r_ij < cutoff
  H_ii = -Σ_{j≠i} H_ij               (sum rule, maintains translational invariance)

3N×3N symmetric matrix → 6 near-zero eigenvalues (3 translations + 3 rotations)
Remaining modes encode internal conformational fluctuations with direction.

References:
  Atilgan et al. (2001) Biophys J 80:505 — original ANM
  Yang & Bahar (2005) Structure 13:893 — D-A coupling
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

ANM_CUTOFF  = 7.5   # Å, same as GNM
ANM_GAMMA   = 1.0   # spring constant (dimensionless)
N_TRIVIAL   = 6     # rigid-body modes to discard


@dataclass
class ANMResult:
    n_residues:   int
    hessian:      np.ndarray       # (3N, 3N)
    eigenvalues:  np.ndarray       # (n_modes,) non-trivial only
    eigenmodes:   np.ndarray       # (N, 3, n_modes) reshaped displacements
    residue_keys: List[Tuple[str, int]]
    residue_map:  Dict[Tuple[str, int], int]
    ca_coords:    np.ndarray       # (N, 3)


def build_anm_hessian(
    structure,
    cutoff: float = ANM_CUTOFF,
    gamma:  float = ANM_GAMMA,
) -> Tuple[np.ndarray, Dict[Tuple[str, int], int]]:
    """
    Build 3N×3N ANM Hessian from Cα coordinates.

    Returns
    -------
    hessian : (3N, 3N) symmetric matrix
    rmap    : {(chain, resnum): residue_index} for use with eigenmodes
    """
    residues  = [r for r in structure.protein_residues() if r.ca is not None]
    if len(residues) < 4:
        raise ValueError(f"Need ≥4 Cα residues, got {len(residues)}")

    n         = len(residues)
    ca_coords = np.array([r.ca.coords for r in residues], dtype=float)
    keys      = [(r.chain, r.number) for r in residues]
    rmap      = {k: i for i, k in enumerate(keys)}

    H = np.zeros((3 * n, 3 * n))

    # Off-diagonal blocks
    for i in range(n):
        for j in range(i + 1, n):
            dvec = ca_coords[j] - ca_coords[i]
            dist = float(np.linalg.norm(dvec))
            if dist < 0.1 or dist >= cutoff:
                continue
            r_hat = dvec / dist
            block = -(gamma / dist ** 2) * np.outer(r_hat, r_hat)
            si, sj = 3 * i, 3 * j
            H[si:si+3, sj:sj+3] += block
            H[sj:sj+3, si:si+3] += block   # symmetric

    # Diagonal blocks: H_ii = -Σ_{j≠i} H_ij  (sum rule)
    for i in range(n):
        si = 3 * i
        diag_block = np.zeros((3, 3))
        for j in range(n):
            if j == i:
                continue
            sj = 3 * j
            diag_block += H[si:si+3, sj:sj+3]
        H[si:si+3, si:si+3] = -diag_block

    return H, rmap


def anm_eigenmodes(
    hessian:  np.ndarray,
    n_modes:  int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Diagonalize ANM Hessian, returning n_modes non-trivial modes.

    Skips the 6 lowest eigenvalues (rigid-body degrees of freedom).

    Returns
    -------
    modes : (N, 3, n_modes)  per-residue 3D displacement in each mode
    evals : (n_modes,)       mode force constants (all positive)
    """
    n3 = hessian.shape[0]
    n  = n3 // 3

    eigenvalues, eigenvectors = np.linalg.eigh(hessian)  # ascending order

    start = N_TRIVIAL
    end   = min(start + n_modes, n3)
    evals = eigenvalues[start:end]
    evecs = eigenvectors[:, start:end]   # (3N, n_modes)

    modes = evecs.reshape(n, 3, end - start)   # (N, 3, n_modes)
    return modes, evals


def anm_principal_axis(
    modes:       np.ndarray,
    evals:       np.ndarray,
    chain:       str,
    resnum:      int,
    rmap:        Dict[Tuple[str, int], int],
) -> Optional[np.ndarray]:
    """
    Principal axis of motion for a residue: Σ_k (1/λ_k) u_k(i), normalized.

    Returns unit vector pointing in the direction of maximum weighted
    displacement, or None if residue not found.
    """
    key = (chain, resnum)
    if key not in rmap:
        return None
    i = rmap[key]

    inv_lam  = 1.0 / evals               # (n_modes,)
    weighted = modes[i] @ inv_lam        # (3,)  = Σ_k (1/λ_k) u_k(i)

    norm = float(np.linalg.norm(weighted))
    if norm < 1e-10:
        return None
    return weighted / norm


def anm_da_alignment(
    modes:   np.ndarray,
    evals:   np.ndarray,
    chain:   str,
    resnum:  int,
    rmap:    Dict[Tuple[str, int], int],
    da_unit: np.ndarray,
) -> float:
    """
    D-A alignment score for a residue.

    Score = Σ_k (1/λ_k) |u_k(i) · d̂_DA|²  /  Σ_k (1/λ_k) ||u_k(i)||²

    Fraction of residue i's total mean-square displacement that lies along
    the donor-acceptor compression axis.  Independent of protein size:
      ~0   residue moves perpendicular to D-A
      ~1/3 isotropic residue (3D average)
      ~1   residue moves purely along D-A

    Returns 0.5 for missing residues (neutral prior).
    """
    key = (chain, resnum)
    if key not in rmap:
        return 0.5

    i        = rmap[key]
    inv_lam  = 1.0 / evals                        # (n_modes,)
    da_proj  = da_unit @ modes[i]                 # (n_modes,): u_k(i) · d̂
    msd_i    = np.sum(modes[i] ** 2, axis=0)      # (n_modes,): ||u_k(i)||²

    num   = float(np.dot(inv_lam, da_proj ** 2))
    denom = float(np.dot(inv_lam, msd_i))

    if denom < 1e-12:
        return 0.5
    return float(np.clip(num / denom, 0.0, 1.0))


def validate_against_anisou(
    structure,
    modes:           np.ndarray,
    evals:           np.ndarray,
    rmap:            Dict[Tuple[str, int], int],
    anisou_data:     dict,
    donor_coords:    np.ndarray,
    acceptor_coords: np.ndarray,
) -> dict:
    """
    Validate ANM modes against crystallographic ANISOU tensors.

    ANM correctly predicts fluctuation MAGNITUDES (bfactor_r ≈ 0.38) but not
    DIRECTIONS (pearson_r ≈ -0.17).  Directional prediction requires chemical
    environment information (electrostatics, H-bonds) beyond topology alone.
    Use ANM as a fallback for magnitude-based importance scoring only — not
    as a substitute for crystallographic directional alignment (ANISOU).

    passed = bfactor_r > 0.3  (magnitude correlation is the validated output)

    Returns
    -------
    dict with keys:
      mean_alignment : mean |ANM_principal · ANISOU_principal| (informational)
      pearson_r      : Pearson r (ANM D-A alignment vs ANISOU D-A alignment)
                       — expected to be near zero or negative; ANM does not
                         predict D-A directional coupling from topology alone
      bfactor_r      : Pearson r (ANM B-factors vs ANISOU equivalent B-factors)
                       — the meaningful output; passed if > 0.3
      passed         : bfactor_r > 0.3
      n_pairs        : residues with both ANM and ANISOU data
    """
    from anisotropic_bfactor import get_residue_principal_axis, da_alignment_score
    from scipy.stats import pearsonr

    da_vec  = acceptor_coords - donor_coords
    da_norm = float(np.linalg.norm(da_vec))
    if da_norm < 0.01:
        return dict(passed=False, mean_alignment=0.0, pearson_r=0.0,
                    bfactor_r=0.0, n_pairs=0, t172_score=0.5, n156_score=0.5)
    da_unit = da_vec / da_norm

    inv_lam = 1.0 / evals   # (n_modes,)

    # Pre-compute ANM quantities for all residues
    anm_principal: Dict[Tuple[str, int], np.ndarray] = {}
    anm_bfac:      Dict[Tuple[str, int], float]      = {}
    anm_da:        Dict[Tuple[str, int], float]      = {}

    for key, i in rmap.items():
        weighted = modes[i] @ inv_lam   # Σ_k (1/λ_k) u_k(i)
        norm = float(np.linalg.norm(weighted))
        if norm > 1e-10:
            anm_principal[key] = weighted / norm

        # B-factor proxy: Σ_k (1/λ_k) ||u_k(i)||²
        anm_bfac[key] = float(np.einsum('mk,mk,k->', modes[i], modes[i], inv_lam))

        da_proj  = da_unit @ modes[i]
        msd_i    = np.sum(modes[i] ** 2, axis=0)
        num      = float(np.dot(inv_lam, da_proj ** 2))
        denom    = float(np.dot(inv_lam, msd_i))
        anm_da[key] = float(np.clip(num / denom, 0.0, 1.0)) if denom > 1e-12 else 0.5

    # Gather ANISOU quantities for residues present in both
    dots:          List[float] = []
    anisou_bfac:   Dict[Tuple[str, int], float] = {}
    anisou_da_map: Dict[Tuple[str, int], float] = {}

    for key in rmap:
        chain, resnum = key
        anisou_axis = get_residue_principal_axis(anisou_data, chain, resnum)
        if anisou_axis is None:
            continue

        if key in anm_principal:
            dots.append(abs(float(np.dot(anm_principal[key], anisou_axis))))

        # ANISOU equivalent B-factor from Cα or Cβ
        for atom in ('CA', 'CB'):
            akey = (chain, resnum, atom)
            if akey in anisou_data:
                anisou_bfac[key] = anisou_data[akey].equivalent_bfactor
                break

        anisou_da_map[key] = da_alignment_score(
            anisou_data, chain, resnum, donor_coords, acceptor_coords
        )

    mean_alignment = float(np.mean(dots)) if dots else 0.0

    # Pearson r: ANM D-A alignment vs ANISOU D-A alignment
    common_da = [(anm_da[k], anisou_da_map[k])
                 for k in anm_da if k in anisou_da_map]
    if len(common_da) >= 3:
        arr_anm, arr_anisou = zip(*common_da)
        pearson_r = float(pearsonr(arr_anm, arr_anisou)[0])
    else:
        pearson_r = 0.0

    # Pearson r: ANM B-factors vs ANISOU equivalent B-factors
    common_bf = [(anm_bfac[k], anisou_bfac[k])
                 for k in anm_bfac if k in anisou_bfac]
    if len(common_bf) >= 3:
        arr_anm_b, arr_anisou_b = zip(*common_bf)
        bfactor_r = float(pearsonr(arr_anm_b, arr_anisou_b)[0])
    else:
        bfactor_r = 0.0

    passed = bfactor_r > 0.3

    return dict(
        passed         = passed,
        mean_alignment = mean_alignment,
        pearson_r      = pearson_r,
        bfactor_r      = bfactor_r,
        n_pairs        = len(dots),
    )


def build_anm(
    structure,
    cutoff:  float = ANM_CUTOFF,
    n_modes: int   = 20,
) -> ANMResult:
    """Build full ANMResult in one call."""
    residues  = [r for r in structure.protein_residues() if r.ca is not None]
    ca_coords = np.array([r.ca.coords for r in residues], dtype=float)
    keys      = [(r.chain, r.number) for r in residues]

    H, rmap      = build_anm_hessian(structure, cutoff)
    modes, evals = anm_eigenmodes(H, n_modes)

    return ANMResult(
        n_residues   = len(keys),
        hessian      = H,
        eigenvalues  = evals,
        eigenmodes   = modes,
        residue_keys = keys,
        residue_map  = rmap,
        ca_coords    = ca_coords,
    )


def anm_alignment_map(
    anm_result: ANMResult,
    da_unit:    np.ndarray,
) -> Dict[Tuple[str, int], float]:
    """D-A alignment scores for all residues in an ANMResult."""
    return {
        key: anm_da_alignment(
            anm_result.eigenmodes, anm_result.eigenvalues,
            key[0], key[1], anm_result.residue_map, da_unit
        )
        for key in anm_result.residue_keys
    }


def anm_bfactor_map(anm_result: ANMResult) -> Dict[Tuple[str, int], float]:
    """
    Rank-normalised ANM B-factor scores for all residues.

    B_i = Σ_k (1/λ_k) ||u_k(i)||²  — total mean-square displacement.

    Rank-normalised to [0,1] so residues with larger thermal fluctuations
    score higher.  Used as the magnitude-only fallback when directional
    (ANISOU/QCF) data are unavailable; does NOT encode D-A direction.
    """
    from scipy.stats import rankdata

    inv_lam = 1.0 / anm_result.eigenvalues
    raw = np.array([
        float(np.einsum('mk,mk,k->', anm_result.eigenmodes[i],
                        anm_result.eigenmodes[i], inv_lam))
        for i in range(anm_result.n_residues)
    ])
    ranks  = rankdata(raw)
    normed = (ranks - 1.0) / max(len(ranks) - 1, 1)
    return {key: float(normed[i])
            for i, key in enumerate(anm_result.residue_keys)}


# ── Self-tests ────────────────────────────────────────────────────────────────

class _MockAtom:
    def __init__(self, coords):
        self.coords = np.array(coords, dtype=float)

class _MockResidue:
    def __init__(self, chain, number, coords):
        self.chain  = chain
        self.number = number
        self.ca     = _MockAtom(coords)

class _MockStructure:
    def __init__(self, residues):
        self._residues = residues
    def protein_residues(self, chain=None):
        if chain:
            return [r for r in self._residues if r.chain == chain]
        return self._residues


def _self_tests():
    passed = 0
    failed = 0

    def check(name, condition, detail=""):
        nonlocal passed, failed
        if condition:
            print(f"  [PASS] {name}")
            passed += 1
        else:
            print(f"  [FAIL] {name}  {detail}")
            failed += 1

    print("─" * 60)
    print("ANM self-tests")
    print("─" * 60)

    # ── Check 1: exactly 6 near-zero eigenvalues (rigid-body modes) ───────────
    # 8 atoms at corners of a 3Å cube → 24 DOF, 6 zero modes
    cube_coords = [(3.0*i, 3.0*j, 3.0*k)
                   for i in range(2) for j in range(2) for k in range(2)]
    cube_res    = [_MockResidue('A', n+1, c) for n, c in enumerate(cube_coords)]
    cube_struct = _MockStructure(cube_res)

    H, rmap   = build_anm_hessian(cube_struct, cutoff=7.5)
    evals_all  = np.linalg.eigvalsh(H)
    n_zero     = int(np.sum(np.abs(evals_all) < 1e-6))
    check("Check 1: 6 rigid-body zero eigenvalues",
          n_zero == 6,
          f"got {n_zero} near-zero eigenvalues (|λ|<1e-6)")

    # ── Check 2: principal axis is a unit vector ──────────────────────────────
    anm_cube = build_anm(cube_struct, cutoff=7.5, n_modes=10)
    ax = anm_principal_axis(anm_cube.eigenmodes, anm_cube.eigenvalues,
                            'A', 1, anm_cube.residue_map)
    check("Check 2: principal axis is unit vector",
          ax is not None and abs(float(np.linalg.norm(ax)) - 1.0) < 1e-6,
          f"norm = {np.linalg.norm(ax):.6f}" if ax is not None else "returned None")

    # ── Check 3: analytical formula verification ──────────────────────────────
    # Mock 2 residues, 2 non-trivial modes with known displacements:
    #   Mode 0: residue 0 moves along x̂ (amplitude 1), λ=1
    #   Mode 1: residue 0 moves along ŷ (amplitude 1), λ=2
    # Expected D-A alignment along x for residue 0:
    #   num   = (1/1)×1² + (1/2)×0² = 1.0
    #   denom = (1/1) + (1/2)        = 1.5
    #   score = 1.0 / 1.5 = 2/3
    # D-A alignment along y for residue 0 should be 1/3
    mock_modes = np.zeros((2, 3, 2))
    mock_modes[0, 0, 0] = 1.0   # residue 0, x-dir, mode 0
    mock_modes[0, 1, 1] = 1.0   # residue 0, y-dir, mode 1
    mock_evals = np.array([1.0, 2.0])
    mock_rmap  = {('A', 1): 0, ('A', 2): 1}

    score_x = anm_da_alignment(mock_modes, mock_evals, 'A', 1, mock_rmap,
                                np.array([1.0, 0.0, 0.0]))
    score_y = anm_da_alignment(mock_modes, mock_evals, 'A', 1, mock_rmap,
                                np.array([0.0, 1.0, 0.0]))
    expected_x = 2.0 / 3.0
    expected_y = 1.0 / 3.0
    check("Check 3: analytical D-A alignment formula",
          abs(score_x - expected_x) < 1e-8 and abs(score_y - expected_y) < 1e-8,
          f"score_x={score_x:.6f} (expected {expected_x:.6f}), "
          f"score_y={score_y:.6f} (expected {expected_y:.6f})")

    # ── Check 4: ANM B-factors show end-effect (terminal > interior) ──────────
    # 20-residue helix; terminal residues have fewer contacts → larger fluctuations
    t   = np.linspace(0.0, 4.0 * np.pi, 20)
    hx  = 2.5 * np.cos(t)
    hy  = 2.5 * np.sin(t)
    hz  = 1.5 * t / (2.0 * np.pi) * 3.8
    helix_res    = [_MockResidue('A', i+1, (hx[i], hy[i], hz[i]))
                    for i in range(20)]
    helix_struct = _MockStructure(helix_res)
    anm_hx       = build_anm(helix_struct, cutoff=7.5, n_modes=15)

    inv_lam = 1.0 / anm_hx.eigenvalues
    bfac = np.array([
        float(np.einsum('k,mk,mk->', inv_lam,
                        anm_hx.eigenmodes[i], anm_hx.eigenmodes[i]))
        for i in range(20)
    ])
    terminal_mean = float(np.mean([bfac[0], bfac[1], bfac[2],
                                   bfac[17], bfac[18], bfac[19]]))
    interior_mean = float(np.mean(bfac[8:13]))
    check("Check 4: terminal residues have larger ANM B-factors than interior",
          terminal_mean > interior_mean,
          f"terminal_mean={terminal_mean:.4f}  interior_mean={interior_mean:.4f}")

    print("─" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    ok = _self_tests()
    import sys
    sys.exit(0 if ok else 1)
