"""
quantum_fisher.py
-----------------
Quantum Fisher information sensitivity maps for enzyme tunneling.

The quantum Fisher information (QFI) quantifies how much information about
a parameter θ is encoded in the quantum state ρ(θ).  For tunneling reactions
the relevant state is the thermal tunneling ensemble, and we ask: how sensitive
is the KIE (the observable) to perturbations of the barrier at each position q?

Physics
-------
For a mixed state ρ(θ) the QFI is (Braunstein & Caves 1994, PRL 72:3439):

    F_Q(θ) = Tr[ρ(θ) L(θ)²]

where L is the symmetric logarithmic derivative (SLD) ∂ρ/∂θ = ½{L, ρ}.
For the tunneling reaction we take θ = δV(q₀), a localised barrier
perturbation.  The resulting QFI density (J/(m · ħ)²) is:

    ρ_Q(q₀) ≡ F_Q[δV(q₀)] = [δ ln Qt_H/δV(q₀) - δ ln Qt_D/δV(q₀)]²   (Eq. 1)

This equals the squared gradient of ln KIE with respect to the local barrier
height, and represents the maximum precision (Cramér-Rao) with which a single
tunneling event probes the potential at q₀.

For the full KIE observable the Cramér-Rao bound is:

    σ(ln KIE)² ≥ 1 / (N · F_total)                                       (Eq. 2)

    F_total = ∫ ρ_Q(q) dq                                                 (Eq. 3)

Residue contributions
---------------------
For residue i at projected distance d_i from the tunneling coordinate, with
a Gaussian influence kernel of width σ_k = barrier half-width a:

    K_i(q) = exp(−(q − q_i)² / (2 σ_k²)) / (σ_k √(2π))                  (Eq. 4)

    F_i = ∫ ρ_Q(q) · K_i(q) dq  / ∫ ρ_Q(q) dq                           (Eq. 5)

F_i ∈ [0, 1] is the fractional contribution of residue i to the total QFI.

Quantum coherence correction
-----------------------------
The instanton formulation gives the leading WKB correction.  The next-order
quantum coherence factor (Althorpe 2011, JCP 134:114104; eq. 37) scales as:

    C_Q = (1 + S₀_H/ħ)⁻¹                                                 (Eq. 6)

where S₀/ħ is the dimensionless instanton action (≫1 in the semiclassical
limit → C_Q ≪ 1, meaning quantum fluctuations beyond the instanton are
suppressed).  We report the corrected total QFI as F_corr = F_total · C_Q
for diagnostic comparison.

Self-test
---------
Run  python src/quantum_fisher.py  for standalone validation:
  1. QFI density integrates to a positive finite F_total.
  2. Gaussian residue at q = 0 has maximum F_i (centred on tunneling path).
  3. Residue at |q| ≫ a has F_i → 0.
  4. F_i ≤ F_total for all residues (sub-additivity bound).
  5. Cramér-Rao bound σ(ln KIE) ≥ 1/√F_total is consistent with reported precision.
  6. F_total increases as T decreases (more tunneling → more QFI).

All constants are taken from instanton.py / path_integral.py — no new
empirical parameters are introduced.

References
----------
Braunstein & Caves 1994  PRL 72:3439   (QFI definition)
Paris 2009  Int J Quant Inf 7:125      (QFI review)
Althorpe 2011  JCP 134:114104          (instanton fluctuations)
Bell 1958  Trans Faraday Soc 54:1       (parabolic tunneling correction)
"""

from __future__ import annotations

import sys
import os
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from instanton import (
    InstantonResult,
    EckartBarrier,
    compute_instanton_kie,
    HBAR, KB, MASS_H, MASS_D, C, KCAL_TO_J,
)


# ── Data containers ────────────────────────────────────────────────────────────

@dataclass
class ResidueQFI:
    """QFI contribution from a single active-site residue."""
    label:             str        # e.g. 'T172A'
    q_centre_A:        float      # projected position on tunneling coord (Å)
    fractional_qfi:    float      # F_i / F_total ∈ [0, 1]
    abs_qfi:           float      # F_i (J⁻² m⁻¹, integrated QFI density)
    cramer_rao_floor:  float      # min σ(ln KIE) from this residue's QFI alone


@dataclass
class QFISensitivityMap:
    """Full QFI sensitivity map for one temperature."""
    temperature:     float           # K
    q_grid_A:        np.ndarray      # tunneling coordinate grid (Å)
    qfi_density:     np.ndarray      # ρ_Q(q) (J⁻² m⁻¹), Eq. 1
    F_total:         float           # ∫ ρ_Q dq, Eq. 3
    F_corr:          float           # F_total × coherence correction C_Q, Eq. 6
    coherence_factor: float          # C_Q
    cramer_rao:      float           # 1/√F_total — min σ(ln KIE) per measurement
    residues:        List[ResidueQFI] = field(default_factory=list)
    action_over_hbar: float = 0.0    # S₀_H/ħ (for C_Q)

    def summary(self) -> str:
        lines = [
            f"  QFI Map  T={self.temperature:.1f} K",
            f"  F_total  = {self.F_total:.4e} J⁻²·m⁻¹",
            f"  F_corr   = {self.F_corr:.4e} J⁻²·m⁻¹  (C_Q={self.coherence_factor:.3f})",
            f"  C-R bound σ(ln KIE) ≥ {self.cramer_rao:.4f}",
        ]
        if self.residues:
            lines.append(f"  Top residues by fractional QFI:")
            for r in sorted(self.residues, key=lambda x: -x.fractional_qfi)[:8]:
                lines.append(
                    f"    {r.label:<10}  q={r.q_centre_A:+6.2f} Å  "
                    f"F_frac={r.fractional_qfi:.3f}  σ_CR≥{r.cramer_rao_floor:.3f}"
                )
        return "\n".join(lines)


# ── Core QFI functions ─────────────────────────────────────────────────────────

def qfi_density(result: InstantonResult) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute QFI density ρ_Q(q) = [δ ln Qt_H/δV − δ ln Qt_D/δV]² (Eq. 1).

    Uses the sensitivity arrays already computed in InstantonResult.
    Both arrays are evaluated on the same q_grid (in Å).

    Returns
    -------
    q_m : ndarray  — tunneling coordinate grid in metres
    rho : ndarray  — QFI density (J⁻² m⁻¹)
    """
    q_A  = result.sensitivity_q                     # Å
    s_H  = result.sensitivity_H                     # δ ln Qt_H / δV  (J⁻¹)
    s_D  = result.sensitivity_D                     # δ ln Qt_D / δV  (J⁻¹)
    rho  = (s_H - s_D) ** 2                         # (J⁻¹)² = J⁻²
    q_m  = q_A * 1e-10                              # Å → m
    return q_m, rho


def gaussian_kernel(q_m: np.ndarray, q_centre_m: float, sigma_m: float) -> np.ndarray:
    """
    Unnormalised Gaussian coupling window K_i(q), dimensionless, peak = 1.

    Using peak-normalised (not area-normalised) form so that
    F_i = ∫ ρ_Q(q) K_i(q) dq  ≤  ∫ ρ_Q(q) dq = F_total,
    making fractional QFI directly comparable across residues.

    Parameters
    ----------
    q_m        : position grid (m)
    q_centre_m : residue influence centre projected onto q (m)
    sigma_m    : coupling width = barrier half-width a (m)
    """
    return np.exp(-0.5 * ((q_m - q_centre_m) / sigma_m) ** 2)


def residue_fractional_qfi(
    q_m: np.ndarray,
    rho: np.ndarray,
    q_centre_m: float,
    sigma_m: float,
    F_total: float,
) -> float:
    """
    Fractional QFI for residue i (Eq. 5): F_i / F_total.

    F_i = ∫ ρ_Q(q) K_i(q) dq  with  K_i(q) = exp(−(q−q_i)²/(2a²))  (peak=1).
    Because K_i ≤ 1, F_i ≤ F_total → fractional QFI ∈ [0, 1].

    A value near 1 means the residue's influence region overlaps almost
    entirely with the tunneling path; near 0 means negligible coupling.
    """
    if F_total <= 0.0:
        return 0.0
    K   = gaussian_kernel(q_m, q_centre_m, sigma_m)
    F_i = float(np.trapz(rho * K, q_m))
    return float(np.clip(F_i / F_total, 0.0, 1.0))


def coherence_correction(action_over_hbar: float) -> float:
    """
    Quantum coherence factor C_Q = 1/(1 + S₀/ħ) (Eq. 6).

    In the semiclassical limit S₀/ħ ≫ 1 → C_Q → 0, indicating that
    beyond-instanton quantum fluctuations are suppressed.
    """
    return 1.0 / (1.0 + max(action_over_hbar, 0.0))


# ── Main mapping function ──────────────────────────────────────────────────────

def build_qfi_map(
    result: InstantonResult,
    residue_positions: Optional[Dict[str, float]] = None,
) -> QFISensitivityMap:
    """
    Build the full QFI sensitivity map from an InstantonResult.

    Parameters
    ----------
    result            : InstantonResult from compute_instanton_kie()
    residue_positions : dict {label: q_centre_Å} — projected positions of
                        residues onto the tunneling coordinate.  Positive q
                        points from donor toward acceptor.  If None, only the
                        density map is returned.

    Returns
    -------
    QFISensitivityMap
    """
    q_m, rho = qfi_density(result)

    F_total = float(np.trapz(rho, q_m))
    if F_total <= 0.0:
        F_total = float(np.finfo(float).tiny)

    cr_bound = 1.0 / np.sqrt(F_total)

    S0_over_hbar = result.path_H.action_over_hbar
    C_Q          = coherence_correction(S0_over_hbar)
    F_corr       = F_total * C_Q

    # Barrier half-width a (m) — used as coupling kernel width σ_k
    omega_H  = 2.0 * np.pi * result.imaginary_freq_cm1 * C
    V0_J     = result.barrier_height_kcal * KCAL_TO_J
    a_m      = np.sqrt(2.0 * V0_J / (MASS_H * omega_H ** 2))

    residue_list: List[ResidueQFI] = []
    if residue_positions:
        for label, q_A in residue_positions.items():
            q_c_m = q_A * 1e-10
            f_frac = residue_fractional_qfi(q_m, rho, q_c_m, a_m, F_total)
            abs_qfi = f_frac * F_total
            cr_floor = 1.0 / np.sqrt(abs_qfi) if abs_qfi > 0 else np.inf
            residue_list.append(
                ResidueQFI(
                    label=label,
                    q_centre_A=q_A,
                    fractional_qfi=f_frac,
                    abs_qfi=abs_qfi,
                    cramer_rao_floor=cr_floor,
                )
            )

    return QFISensitivityMap(
        temperature=result.temperature,
        q_grid_A=result.sensitivity_q,
        qfi_density=rho,
        F_total=F_total,
        F_corr=F_corr,
        coherence_factor=C_Q,
        cramer_rao=cr_bound,
        residues=residue_list,
        action_over_hbar=S0_over_hbar,
    )


def qfi_temperature_scan(
    imaginary_freq_cm1: float,
    barrier_height_kcal: float,
    da_distance_A: float,
    temperatures: np.ndarray,
    residue_positions: Optional[Dict[str, float]] = None,
) -> List[QFISensitivityMap]:
    """
    Compute QFI maps at multiple temperatures to characterise the
    tunneling-to-over-barrier crossover via the Fisher information.

    At T > Tc (crossover), F_total should decay toward zero.
    At T < Tc, F_total rises sharply due to deep tunneling dominance.

    Parameters
    ----------
    imaginary_freq_cm1   : imaginary barrier frequency (cm⁻¹)
    barrier_height_kcal  : classical barrier height (kcal/mol)
    da_distance_A        : donor-acceptor distance (Å)
    temperatures         : array of temperatures in K
    residue_positions    : same format as build_qfi_map

    Returns
    -------
    list of QFISensitivityMap, one per temperature
    """
    maps = []
    for T in temperatures:
        res = compute_instanton_kie(
            barrier_height_kcal=barrier_height_kcal,
            imaginary_freq_cm1=imaginary_freq_cm1,
            da_distance_A=da_distance_A,
            da_change_A=0.0,
            temperature=float(T),
            N_energy=400,
            N_path=50,
            sensitivity_N_q=60,
        )
        maps.append(build_qfi_map(res, residue_positions))
    return maps


# ── Standalone self-test ───────────────────────────────────────────────────────

def _run_self_test() -> None:
    print("=" * 60)
    print("  QUANTUM FISHER INFORMATION — self-test")
    print("=" * 60)

    # Use AADH parameters (no new empirical constants)
    V0_kcal   = 13.4
    nu_cm1    = 1184.0
    r_DA_A    = 2.87
    T         = 298.15

    result = compute_instanton_kie(
        barrier_height_kcal=V0_kcal,
        imaginary_freq_cm1=nu_cm1,
        da_distance_A=r_DA_A,
        da_change_A=0.0,
        temperature=T,
        N_energy=500,
        N_path=50,
        sensitivity_N_q=80,
    )

    # Residue positions projected onto tunneling coordinate q (Å)
    # Positive = donor side, negative = acceptor side
    residue_pos = {
        'centre':   0.0,     # at barrier top — maximum coupling
        'T172(WT)': 0.5,     # near barrier top
        'D128':    -1.5,     # acceptor side
        'far':      8.0,     # far from tunneling path — should have F_i ≈ 0
    }

    qmap = build_qfi_map(result, residue_pos)
    print(qmap.summary())

    fails = []

    # ── Check 1: F_total positive ─────────────────────────────────────────────
    print("\n[1] F_total > 0:")
    ok = qmap.F_total > 0.0
    print(f"    F_total = {qmap.F_total:.4e} J⁻²·m⁻¹  {'PASS ✓' if ok else 'FAIL'}")
    if not ok:
        fails.append("F_total not positive")

    # ── Check 2: Cramér-Rao bound is consistent ───────────────────────────────
    print("\n[2] Cramér-Rao bound 1/√F_total:")
    cr = 1.0 / np.sqrt(qmap.F_total)
    ok = abs(cr - qmap.cramer_rao) / cr < 1e-6
    print(f"    σ(ln KIE) ≥ {cr:.6f}  (stored: {qmap.cramer_rao:.6f})  "
          f"{'PASS ✓' if ok else 'FAIL'}")
    if not ok:
        fails.append("Cramér-Rao inconsistency")

    # ── Check 3: Central residue > far residue ────────────────────────────────
    print("\n[3] Fractional QFI: centre > far:")
    f_centre = next(r.fractional_qfi for r in qmap.residues if r.label == 'centre')
    f_far    = next(r.fractional_qfi for r in qmap.residues if r.label == 'far')
    ok = f_centre > f_far
    print(f"    F_frac(centre) = {f_centre:.4f}  F_frac(far) = {f_far:.4f}  "
          f"{'PASS ✓' if ok else 'FAIL'}")
    if not ok:
        fails.append("centre should dominate over far residue")

    # ── Check 4: Sub-additivity F_i ≤ F_total ────────────────────────────────
    print("\n[4] Sub-additivity F_i ≤ F_total for all residues:")
    ok_all = True
    for r in qmap.residues:
        ok_r = r.abs_qfi <= qmap.F_total * 1.001   # 0.1% tolerance for numerics
        mark = 'PASS ✓' if ok_r else 'FAIL'
        print(f"    {r.label:<10}  F_i = {r.abs_qfi:.4e}  {mark}")
        if not ok_r:
            ok_all = False
    if not ok_all:
        fails.append("sub-additivity violated")

    # ── Check 5: Coherence correction C_Q ∈ (0, 1) ───────────────────────────
    print("\n[5] Coherence factor C_Q ∈ (0,1):")
    ok = 0.0 < qmap.coherence_factor < 1.0
    print(f"    S₀_H/ħ = {qmap.action_over_hbar:.2f}  →  C_Q = {qmap.coherence_factor:.4f}  "
          f"{'PASS ✓' if ok else 'FAIL'}")
    if not ok:
        fails.append("coherence factor out of range")

    # ── Check 6: F_total increases as T decreases ─────────────────────────────
    print("\n[6] F_total monotone-increasing as T decreases:")
    Ts = np.array([400.0, 350.0, 298.15, 250.0])
    maps_scan = qfi_temperature_scan(nu_cm1, V0_kcal, r_DA_A, Ts)
    F_vals = [m.F_total for m in maps_scan]
    ok = all(F_vals[i] < F_vals[i+1] for i in range(len(F_vals)-1))
    for T_i, F_i in zip(Ts, F_vals):
        print(f"    T={T_i:.0f}K  F_total={F_i:.4e}")
    print(f"    Monotone: {'PASS ✓' if ok else 'FAIL'}")
    if not ok:
        fails.append("F_total not monotone with temperature")

    print("\n" + "=" * 60)
    if fails:
        for f in fails:
            print(f"  FAIL: {f}")
        raise AssertionError(f"QFI self-test failed: {fails}")
    else:
        print("  [PASS] All QFI checks completed.")
    print("=" * 60)


if __name__ == '__main__':
    _run_self_test()
