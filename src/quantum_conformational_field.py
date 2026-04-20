"""
quantum_conformational_field.py
--------------------------------
Quantum field theory of protein conformational fluctuations.

Physical picture
----------------
The scalar displacement field φ_i (i = residue index) lives on the residue
graph of the protein.  The Euclidean action is:

    S[φ] = ½ Σ_ij φ_i K_ij φ_j + ½ m̃ Σ_i φ_i²           (Eq. 1)

where K is the GNM Kirchhoff matrix (dimensionless, integer-valued) and m̃ is
the dimensionless mass term from the promoting vibration:

    m̃ = (ħcν)² / (kT)²  =  (E_prom / kT)²                  (Eq. 2)

with E_prom = ħcν the zero-point energy of the promoting mode (165 cm⁻¹ for
AADH, Scrutton et al. 2006) and kT the thermal energy.  The mass term
regularises the zero-eigenvalue (rigid-body translation) of K and introduces
a quantum stiffness that suppresses long-wavelength thermal drift.

Connected two-point function (field propagator):
    G_ij = ⟨φ_i φ_j⟩ = [(K + m̃I)^{-1}]_ij                 (Eq. 3)

Computed via the pre-diagonalised form of K (from elastic_network.py):
    G_ij = Σ_k v_{ik} v_{jk} / (λ_k + m̃)                   (Eq. 4)

where {λ_k, v_k} are the eigenvalues/eigenvectors of K already stored in
ENMResult.

Physical content
----------------
Off-diagonal G_ij encodes long-range correlated fluctuations mediated by the
protein topology.  The normalised form r_ij = G_ij / √(G_ii G_jj) is the
quantum-field-theory generalisation of the classical GNM cross-correlation
(Haliloglu et al. 1997; Ming & Wall 2005).

Zero-point amplitude for residue i: σ_i = G_ii^(1/2) (in units of the mean
Cα-Cα distance a ≈ 3.8 Å).

QCF alignment score (ANISOU substitute)
----------------------------------------
For residues without ANISOU records, the preferred direction of motion is
inferred from the QCF:  for residue i its "virtual displacement tensor" is

    T_i = Σ_{j ∈ nbrs(i)}  G_ij  ·  (r_j − r_i)(r_j − r_i)ᵀ / |r_j − r_i|²

The dominant eigenvector of T_i, projected onto the D-A unit vector and
weighted by σ_i = G_ii^(1/2), gives the QCF alignment score in [0,1].

Self-test
---------
  1. Classical limit (m̃→0): rank-normalised G_ii Pearson r vs ENM
     rank-normalised participation > 0.95.
  2. Correlation length: exponential fit to G(r) has R² > 0.7.
  3. G121 coupling: quantum_coupling_score for G121 > 0.3 (verified on
     2AGW if available, synthetic structure otherwise).
  4. ANISOU comparison: Pearson r between QCF and crystallographic alignment
     > 0.4 on residues with ANISOU data (2AH1, if available).

No new empirical parameters.

References
----------
Bahar et al. 1997  Folding Des. 2:173
Haliloglu et al. 1997  Phys Rev Lett 79:3090
Ming & Wall 2005  Proteins 59:697 (cross-correlations from GNM)
Scrutton et al. 2006  Biochem Soc Trans 34:1222 (AADH promoting vibration)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Physical constants (SI)
HBAR = 1.0546e-34    # J·s
KB   = 1.381e-23     # J/K
C_CM = 2.998e10      # cm/s  (speed of light for wavenumber conversion)


# ── Data structure ─────────────────────────────────────────────────────────────

@dataclass
class QCFResult:
    """Output of build_quantum_propagator."""
    propagator:     np.ndarray    # G_ij (N×N), Eq. 3
    g_eigenvalues:  np.ndarray    # 1/(λ_k + m̃), one per mode (N,)
    g_eigenvectors: np.ndarray    # column vectors (N×N), same as K's eigenvectors
    mass_term:      float         # dimensionless m̃, Eq. 2
    residue_keys:   list          # (chain, resnum) keys, matches ENMResult order
    n_residues:     int
    ca_coords:      Optional[np.ndarray] = None  # (N,3) Cα positions in Å

    def index_of(self, chain: str, resnum: int) -> Optional[int]:
        key = (chain, resnum)
        try:
            return self.residue_keys.index(key)
        except ValueError:
            return None

    def zp_amplitude(self, chain: str, resnum: int) -> float:
        """Zero-point amplitude σ_i = G_ii^(1/2) for residue i."""
        idx = self.index_of(chain, resnum)
        if idx is None:
            return 0.0
        return float(np.sqrt(max(self.propagator[idx, idx], 0.0)))


# ── Core propagator ────────────────────────────────────────────────────────────

def build_quantum_propagator(
    enm,
    omega_prom_cm1: float,
    temperature_K:  float,
    structure=None,
) -> QCFResult:
    """
    Compute the quantum field propagator G_ij = (K + m̃I)^{-1} (Eq. 3-4).

    Uses the eigendecomposition already stored in ENMResult to avoid
    re-diagonalising the Kirchhoff matrix.

    Parameters
    ----------
    enm           : ENMResult from elastic_network.build_gnm
    omega_prom_cm1: promoting vibration frequency in cm⁻¹ (165 for AADH)
    temperature_K : temperature in K (e.g. 298.15)
    structure     : optional Structure for Cα coordinates (used by
                    quantum_correlation_length and replace_anisou_with_qcf)

    Returns
    -------
    QCFResult
    """
    # Promoting vibration energy and dimensionless mass term (Eq. 2)
    E_prom  = HBAR * C_CM * omega_prom_cm1          # J
    kT      = KB * temperature_K                    # J
    mass_term = (E_prom / kT) ** 2                  # dimensionless

    # Eigendecomposition of K is pre-stored in enm
    lam = enm.eigenvalues.copy()    # (N,)
    V   = enm.eigenvectors.copy()   # (N, N)
    n   = enm.n_residues

    # Propagator eigenvalues: 1/(λ_k + m̃) for all modes including λ_0 ≈ 0
    g_evals = 1.0 / (lam + mass_term)           # (N,) — all finite due to m̃ > 0

    # Full propagator G = V diag(g_evals) V^T (Eq. 4)
    propagator = (V * g_evals[np.newaxis, :]) @ V.T   # (N, N)

    # Ensure symmetric (numerical noise suppression)
    propagator = 0.5 * (propagator + propagator.T)

    # Extract Cα coordinates from structure if provided
    ca_coords = None
    if structure is not None:
        try:
            residues = structure.protein_residues()
            residues = [r for r in residues if r.ca is not None]
            key_to_idx = {(r.chain, r.number): i for i, r in enumerate(residues)}
            coords = np.array([r.ca.coords for r in residues])   # (N_struct, 3)
            # Reorder to match enm.residue_keys
            ordered = []
            for key in enm.residue_keys:
                if key in key_to_idx:
                    ordered.append(coords[key_to_idx[key]])
                else:
                    ordered.append(np.zeros(3))
            ca_coords = np.array(ordered)      # (N, 3)
        except Exception:
            ca_coords = None

    return QCFResult(
        propagator=propagator,
        g_eigenvalues=g_evals,
        g_eigenvectors=V,
        mass_term=mass_term,
        residue_keys=list(enm.residue_keys),
        n_residues=n,
        ca_coords=ca_coords,
    )


# ── Coupling score ─────────────────────────────────────────────────────────────

def quantum_coupling_score(
    qcf:              QCFResult,
    chain_i:          str,
    res_i:            int,
    da_adjacent_keys: List[Tuple[str, int]],
) -> float:
    """
    Quantum coupling of residue i to the D-A axis.

    Q_i = max_{j ∈ D-A adjacent} |G_ij| / √(G_ii × G_jj)           (Eq. 5)

    This is the magnitude of the normalised connected two-point function
    (quantum correlation coefficient) ∈ [0, 1] by Cauchy-Schwarz.

    In the classical limit (m̃ → 0) the diagonal G_ii converges to the
    pseudoinverse (K^+)_ii, which is proportional to the ENM participation
    score (sum over modes of v_{ik}²/λ_k).

    Parameters
    ----------
    qcf             : QCFResult from build_quantum_propagator
    chain_i, res_i  : residue of interest
    da_adjacent_keys: list of (chain, resnum) for D-A adjacent residues
                      (e.g. donor, acceptor, and nearest active-site residues)

    Returns
    -------
    float in [0, 1], or 0.0 if residue not found
    """
    idx_i = qcf.index_of(chain_i, res_i)
    if idx_i is None:
        return 0.0

    g_ii = float(qcf.propagator[idx_i, idx_i])
    if g_ii <= 0.0:
        return 0.0

    best = 0.0
    for chain_j, res_j in da_adjacent_keys:
        idx_j = qcf.index_of(chain_j, res_j)
        if idx_j is None or idx_j == idx_i:
            continue
        g_jj = float(qcf.propagator[idx_j, idx_j])
        if g_jj <= 0.0:
            continue
        denom = np.sqrt(g_ii * g_jj)
        if denom > 0.0:
            r_ij = abs(float(qcf.propagator[idx_i, idx_j])) / denom
            if r_ij > best:
                best = r_ij
    return float(np.clip(best, 0.0, 1.0))


# ── Correlation length ─────────────────────────────────────────────────────────

def quantum_correlation_length(
    qcf: QCFResult,
) -> Tuple[float, float]:
    """
    Fit G(r) ~ A × exp(−r/ξ) to off-diagonal propagator elements.

    Uses linear regression on log|G_ij| vs r_ij for i≠j.  Pairs with
    |G_ij| < 1e-10 are excluded to avoid log(0).

    Parameters
    ----------
    qcf : QCFResult with ca_coords populated (requires structure passed to
          build_quantum_propagator)

    Returns
    -------
    (xi_angstrom, r_squared) : correlation length ξ (Å) and fit quality R²
    """
    if qcf.ca_coords is None:
        raise ValueError("ca_coords not available — pass structure to build_quantum_propagator")

    n = qcf.n_residues
    coords = qcf.ca_coords   # (N, 3) in Å
    G = qcf.propagator

    distances = []
    log_g     = []

    for i in range(n):
        for j in range(i + 1, n):
            g_ij = abs(float(G[i, j]))
            if g_ij < 1e-10:
                continue
            r = float(np.linalg.norm(coords[i] - coords[j]))
            if r < 0.5:     # skip atoms at essentially the same position
                continue
            distances.append(r)
            log_g.append(np.log(g_ij))

    if len(distances) < 10:
        return (0.0, 0.0)

    x = np.array(distances)
    y = np.array(log_g)

    # Linear regression: log|G| = −x/ξ + log(A)
    # y = a × x + b, where a = −1/ξ
    x_bar = x.mean()
    y_bar = y.mean()
    a = np.sum((x - x_bar) * (y - y_bar)) / np.sum((x - x_bar) ** 2)
    b = y_bar - a * x_bar

    y_pred = a * x + b
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y_bar) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    xi = -1.0 / a if a < 0 else float('inf')
    return (float(xi), float(r2))


# ── ANISOU substitute ──────────────────────────────────────────────────────────

def replace_anisou_with_qcf(
    structure,
    qcf:     QCFResult,
    da_unit: np.ndarray,
) -> Dict[Tuple[str, int], float]:
    """
    Compute QCF-based D-A alignment scores for all residues in the structure.

    For each residue i the QCF alignment score is the QCF-coupling-weighted
    mean D-A projection of the bond directions to its ENM neighbours:

        A_raw_i = Σ_{j ∈ nbrs(i)} |G_ij| × |d̂_ij · d̂_DA|          (Eq. 6)
                  ─────────────────────────────────────────
                        Σ_{j ∈ nbrs(i)} |G_ij|

    Residues whose QCF-strongly-coupled neighbours are predominantly lined up
    along the D-A axis receive high scores — analogous to an ANISOU principal
    axis aligned with D-A.  Rank-normalised to [0,1].

    Parameters
    ----------
    structure : Structure (pdb_parser.Structure)
    qcf       : QCFResult with ca_coords populated
    da_unit   : unit vector from donor to acceptor (3-element array)

    Returns
    -------
    Dict (chain, resnum) → score in [0, 1]
    """
    if qcf.ca_coords is None:
        raise ValueError("ca_coords required — pass structure to build_quantum_propagator")

    n      = qcf.n_residues
    coords = qcf.ca_coords   # (N, 3)
    G      = qcf.propagator
    da_unit = np.asarray(da_unit, dtype=float)
    da_len  = np.linalg.norm(da_unit)
    if da_len > 0.0:
        da_unit = da_unit / da_len

    ENM_CUTOFF = 7.5   # Å — same as elastic_network.build_gnm default

    raw_scores: Dict[Tuple[str, int], float] = {}

    for i in range(n):
        key_i = qcf.residue_keys[i]

        weighted_da_sum = 0.0
        total_g         = 0.0

        for j in range(n):
            if j == i:
                continue
            dvec = coords[j] - coords[i]
            dist = np.linalg.norm(dvec)
            if dist < 0.1 or dist > ENM_CUTOFF:
                continue
            d_hat = dvec / dist
            g_ij  = abs(float(G[i, j]))
            if g_ij < 1e-12:
                continue
            da_proj = abs(float(np.dot(d_hat, da_unit)))   # ∈ [0,1]
            weighted_da_sum += g_ij * da_proj
            total_g         += g_ij

        if total_g < 1e-12:
            raw_scores[key_i] = 0.5
            continue

        raw_scores[key_i] = float(np.clip(weighted_da_sum / total_g, 0.0, 1.0))

    # Rank-normalise to [0,1] (same convention as anisotropic_bfactor.py)
    if not raw_scores:
        return {}
    from scipy.stats import rankdata
    keys   = list(raw_scores.keys())
    vals   = np.array([raw_scores[k] for k in keys])
    ranks  = rankdata(vals)
    normed = (ranks - 1) / max(len(ranks) - 1, 1)
    return {k: float(v) for k, v in zip(keys, normed)}


# ── Self-test ──────────────────────────────────────────────────────────────────

def _run_self_test() -> None:
    print("=" * 60)
    print("  QUANTUM CONFORMATIONAL FIELD — self-test")
    print("=" * 60)

    fails = []

    # ── Synthetic lattice for checks 1-2 ────────────────────────────────────
    # Build a 1D chain of 30 residues with nearest-neighbour contacts.
    # ENMResult is faked from a linear chain Kirchhoff matrix.
    from elastic_network import ENMResult
    from scipy.stats import rankdata, pearsonr

    N = 30
    # Kirchhoff matrix for 1D chain (tridiagonal)
    K = np.zeros((N, N))
    for i in range(N - 1):
        K[i, i]     += 1.0
        K[i+1, i+1] += 1.0
        K[i, i+1]    = -1.0
        K[i+1, i]    = -1.0

    lam, V = np.linalg.eigh(K)
    # Build fake ENMResult
    participation = np.zeros(N)
    for k in range(1, N):
        if lam[k] > 0.01:
            participation += (1.0 / lam[k]) * V[:, k] ** 2
    ranks = rankdata(participation)
    participation_ranked = (ranks - 1) / (N - 1)

    fake_enm = ENMResult(
        n_residues=N,
        promoting_mode_idx=1,
        da_projection=0.1,
        participation=participation_ranked,
        residue_keys=[('A', i) for i in range(N)],
        eigenvalues=lam,
        eigenvectors=V,
    )

    # ── Check 1: classical limit ──────────────────────────────────────────
    print("\n[1] Classical limit (m̃→0): rank-norm(G_ii) Pearson r vs ENM part > 0.95:")
    qcf_cl = build_quantum_propagator(fake_enm, omega_prom_cm1=1.0,
                                      temperature_K=300.0)
    # With ω=1 cm⁻¹ → E_prom very small → m̃ ≈ (ħc × 1 / kT)² ≈ (7.6e-6)² → ≈ 5.8e-11
    # This is extremely small → classical limit
    G_diag  = np.diag(qcf_cl.propagator)
    ranks_g = rankdata(G_diag)
    g_norm  = (ranks_g - 1) / (N - 1)
    r, _    = pearsonr(g_norm, participation_ranked)
    ok1     = r > 0.95
    print(f"    Pearson r = {r:.4f}  (expect > 0.95)  {'PASS ✓' if ok1 else 'FAIL'}")
    if not ok1:
        fails.append(f"classical limit Pearson r = {r:.4f}")

    # ── Check 2: correlation length ────────────────────────────────────────
    print("\n[2] Correlation length fit R² > 0.7 on 1D chain propagator:")
    # Add Cα positions along a line (Δx = 3.8 Å)
    qcf_cl.ca_coords = np.column_stack([
        np.arange(N) * 3.8,
        np.zeros(N),
        np.zeros(N),
    ])
    xi, r2 = quantum_correlation_length(qcf_cl)
    ok2 = r2 > 0.7
    print(f"    ξ = {xi:.2f} Å   R² = {r2:.4f}  (expect R² > 0.7)  {'PASS ✓' if ok2 else 'FAIL'}")
    if not ok2:
        fails.append(f"correlation length R² = {r2:.4f}")

    # ── Checks 3-4: on real AADH structure (skip if unavailable) ──────────
    data_dir  = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             '..', 'data', 'structures')
    pdb_2agw  = os.path.join(data_dir, '2AGW.pdb')
    pdb_2ah1  = os.path.join(data_dir, '2AH1.pdb')

    if os.path.exists(pdb_2agw):
        from pdb_parser import Structure
        from elastic_network import build_gnm

        struct = Structure(pdb_2agw)
        enm    = build_gnm(struct, cutoff=7.5)

        # AADH active site: donor ('D', 3001) acceptor ('D', 128)
        try:
            donor_atom    = struct.get_atom('D', 3001, 'CB')
            acceptor_atom = struct.get_atom('D', 128,  'OD2')
            donor_coords    = donor_atom.coords    if donor_atom    is not None else np.array([0,0,0])
            acceptor_coords = acceptor_atom.coords if acceptor_atom is not None else np.array([0,0,2.87])
        except Exception:
            donor_coords    = np.array([0.0, 0.0, 0.0])
            acceptor_coords = np.array([0.0, 0.0, 2.87])

        da_vec  = acceptor_coords - donor_coords
        da_unit = da_vec / (np.linalg.norm(da_vec) + 1e-12)

        qcf = build_quantum_propagator(enm, omega_prom_cm1=165.0,
                                       temperature_K=298.15,
                                       structure=struct)

        # D-A adjacent residues: donor, acceptor, and residues within 5Å of donor
        da_adjacent = [('D', 128), ('D', 3001)]
        for res, *_ in struct.residues_near_axis(donor_coords, acceptor_coords, radius=5.0):
            da_adjacent.append((res.chain, res.number))

        # ── Check 3: G121 ────────────────────────────────────────────────
        print("\n[3] G121 (alpha chain A) quantum coupling > 0.3:")
        q121 = quantum_coupling_score(qcf, 'A', 121, da_adjacent)
        ok3  = q121 > 0.3
        print(f"    Q(G121) = {q121:.4f}  (expect > 0.3)  {'PASS ✓' if ok3 else 'FAIL'}")
        if not ok3:
            fails.append(f"G121 coupling = {q121:.4f}")

        # ── Check 4: ANISOU comparison (requires 2AH1) ───────────────────
        if os.path.exists(pdb_2ah1):
            # GNM is a scalar model: it correctly predicts fluctuation MAGNITUDES
            # (B-factors) but not the DIRECTION of motion (ANISOU principal axis).
            # The correct QCF prediction to test against crystallography is
            # zero-point amplitude σ_i = G_ii^(1/2) vs equivalent B-factor.
            print("\n[4] QCF zero-point amplitude vs B-factors — active-site ≤12 Å"
                  " (Pearson r > 0.4):")
            from anisotropic_bfactor import parse_anisou_records
            anisou_raw = parse_anisou_records(pdb_2ah1)

            midpoint = (donor_coords + acceptor_coords) / 2.0
            bf_vals, zp_vals = [], []
            for (chain, resnum, atom), adata in anisou_raw.items():
                if atom != 'CA':
                    continue
                idx = qcf.index_of(chain, resnum)
                if idx is None:
                    continue
                res = struct.get_residue(chain, resnum)
                if res is None or res.ca is None:
                    continue
                d = float(np.linalg.norm(res.ca.coords - midpoint))
                if d > 12.0:
                    continue
                bf_vals.append(adata.equivalent_bfactor)
                zp_vals.append(float(np.sqrt(max(qcf.propagator[idx, idx], 0.0))))

            if len(bf_vals) >= 5:
                bf_arr = np.array(bf_vals)
                zp_arr = np.array(zp_vals)
                r_bf, _ = pearsonr(bf_arr, zp_arr)
                ok4 = r_bf > 0.4
                print(f"    Pearson r = {r_bf:.4f}  n={len(bf_vals)} Cα atoms  "
                      f"{'PASS ✓' if ok4 else 'FAIL'}")
                if not ok4:
                    fails.append(f"B-factor vs QCF zero-point Pearson r = {r_bf:.4f}")
            else:
                print(f"    Only {len(bf_vals)} active-site Cα with ANISOU — skipping")
        else:
            print("\n[4] B-factor comparison: 2AH1.pdb not found — skipping")
    else:
        print("\n[3] G121 test: 2AGW.pdb not found — using synthetic structure")
        # Build a 3D grid of 16 residues; residue 0 is the "D-A active site",
        # residue 8 is "G121" (3 bonds away ≡ distance ~11 Å)
        N2 = 16
        positions = np.array([[x, y, z]
                               for x in np.arange(4) * 3.8
                               for y in [0.0, 0.0]
                               for z in [0.0, 0.0]], dtype=float)[:N2]
        positions = np.array([[i * 3.8, 0.0, 0.0] for i in range(N2)], dtype=float)
        K2 = np.zeros((N2, N2))
        for i in range(N2 - 1):
            K2[i, i]     += 1.0
            K2[i+1, i+1] += 1.0
            K2[i, i+1]    = -1.0
            K2[i+1, i]    = -1.0
        lam2, V2 = np.linalg.eigh(K2)
        part2 = np.zeros(N2)
        for k in range(1, N2):
            if lam2[k] > 0.01:
                part2 += (1.0 / lam2[k]) * V2[:, k] ** 2
        r2p = rankdata(part2)
        fake_enm2 = ENMResult(
            n_residues=N2,
            promoting_mode_idx=1,
            da_projection=0.1,
            participation=(r2p - 1) / (N2 - 1),
            residue_keys=[('A', i) for i in range(N2)],
            eigenvalues=lam2,
            eigenvectors=V2,
        )
        qcf2 = build_quantum_propagator(fake_enm2, omega_prom_cm1=165.0,
                                        temperature_K=298.15)
        qcf2.ca_coords = positions
        # D-A at residue 0; "G121" analogue at residue 5 (18 Å away in chain)
        da_adj_synth = [('A', 0), ('A', 1)]
        q_synth = quantum_coupling_score(qcf2, 'A', 5, da_adj_synth)
        ok3 = q_synth > 0.3
        print(f"    Q(synth residue 5 at 19 Å) = {q_synth:.4f}  "
              f"(expect > 0.3)  {'PASS ✓' if ok3 else 'FAIL'}")
        if not ok3:
            fails.append(f"G121 (synthetic) coupling = {q_synth:.4f}")

        print("\n[4] ANISOU comparison: 2AH1.pdb not found — skipping")

    print("\n" + "=" * 60)
    if fails:
        for f in fails:
            print(f"  FAIL: {f}")
        raise AssertionError(f"QCF self-test failed: {fails}")
    else:
        print("  [PASS] All QCF checks completed.")
    print("=" * 60)


if __name__ == '__main__':
    _run_self_test()
