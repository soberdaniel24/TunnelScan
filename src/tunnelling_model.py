"""
tunnelling_model.py
-------------------
Computes quantum tunnelling contribution to enzymatic H-transfer reactions.

The core physics:
  Classical transition state theory predicts a rate using the Arrhenius equation.
  Bell correction adds a first-order tunnelling term based on the imaginary
  frequency at the transition state (the curvature of the barrier at its peak).
  
  KIE = kH / kD (kinetic isotope effect)
  Classical KIE ≈ 6-7 at 25°C (Swain-Schaad limit)
  Tunnelling inflates this: AADH wild-type KIE ≈ 55 (deeply tunnelling)
  
  If your predicted KIE >> 7, tunnelling is significant.
  If predicted KIE matches literature, your pipeline is working.

Usage:
  from tunnelling_model import bell_correction, TunnellingResult
  result = bell_correction(barrier_height_kcal, imaginary_freq_cm1, da_distance_angstrom)
  print(result)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


# ── Physical constants ────────────────────────────────────────────────────────

KB   = 1.380649e-23    # Boltzmann constant, J/K
H    = 6.62607015e-34  # Planck constant, J·s
HBAR = H / (2 * np.pi)
C    = 2.99792458e10   # Speed of light, cm/s (note: cm not m, for wavenumber conversion)
NA   = 6.02214076e23   # Avogadro's number
KCAL_TO_J = 4184.0 / NA  # kcal/mol → J per molecule

# Mass of hydrogen and deuterium in kg
MASS_H = 1.6735575e-27  # kg
MASS_D = 3.3444e-27     # kg

# Typical C-H stretch frequency in the REACTANT state (cm⁻¹)
# Used for classical KIE calculation.
# C-H: ~2900-3000 cm⁻¹  |  C-D: ≈ C-H / sqrt(2) ≈ 2100 cm⁻¹
# This gives the Swain-Schaad classical limit of ~6.9 at 298K.
# Note: this is NOT the imaginary TS frequency — that governs tunnelling width,
# not the classical ZPE-based KIE.
REACTANT_CH_FREQ = 2950.0   # cm⁻¹ (representative C-H stretch)

# Proper single-letter amino acid codes (needed for mutation naming)
THREE_TO_ONE = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C',
    'GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I',
    'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P',
    'SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V',
}


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class TunnellingResult:
    """All computed tunnelling quantities for one enzyme variant."""
    temperature:          float   # Kelvin
    barrier_height:       float   # kcal/mol (classical ΔE‡)
    imaginary_freq:       float   # cm⁻¹ (ν‡ from QM/MM)
    da_distance:          float   # Angstroms (donor-acceptor at TS)

    # Bell correction outputs
    Qt_H:                 float   # tunnelling transmission coeff for H
    Qt_D:                 float   # tunnelling transmission coeff for D
    predicted_KIE:        float   # Qt_H / Qt_D — compare to experimental kH/kD
    tunnelling_fraction:  float   # fraction of total rate from tunnelling
    classical_KIE:        float   # Swain-Schaad limit (~6.9 at 298K)

    # Qualitative assessment
    tunnelling_regime:    str     # 'deep', 'moderate', 'minimal'
    experimental_KIE:     Optional[float] = None  # set if known, for validation
    prediction_error:     Optional[float] = None  # |predicted - experimental| / experimental

    def __str__(self):
        lines = [
            f"\n{'─'*50}",
            f"  TUNNELLING ANALYSIS",
            f"{'─'*50}",
            f"  Temperature:          {self.temperature:.1f} K ({self.temperature - 273.15:.1f} °C)",
            f"  Classical barrier:    {self.barrier_height:.2f} kcal/mol",
            f"  Imaginary frequency:  {self.imaginary_freq:.1f} cm⁻¹",
            f"  D-A distance at TS:   {self.da_distance:.3f} Å",
            f"{'─'*50}",
            f"  Qt (H):               {self.Qt_H:.4f}",
            f"  Qt (D):               {self.Qt_D:.4f}",
            f"  Classical KIE:        {self.classical_KIE:.2f}",
            f"  Predicted KIE (H/D):  {self.predicted_KIE:.1f}",
            f"  Tunnelling fraction:  {self.tunnelling_fraction:.1%}",
            f"  Tunnelling regime:    {self.tunnelling_regime.upper()}",
        ]
        if self.experimental_KIE is not None:
            lines += [
                f"{'─'*50}",
                f"  Experimental KIE:     {self.experimental_KIE:.1f}",
                f"  Prediction error:     {self.prediction_error:.1%}",
            ]
        lines.append(f"{'─'*50}\n")
        return "\n".join(lines)


@dataclass
class MutantPrediction:
    """Predicted tunnelling properties for one point mutant."""
    residue_number:    int
    original_aa:       str
    new_aa:            str
    name:              str          # e.g. "T172A"

    da_change:         float        # Δ D-A distance vs wild-type (Å), negative = shorter
    predicted_KIE:     float        # predicted KIE for this mutant
    fold_enhancement:  float        # predicted_KIE / wt_KIE
    priority:          str          # 'HIGH', 'MEDIUM', 'LOW'
    rationale:         str          # plain-English explanation

    def __str__(self):
        return (
            f"  {self.name:<12} "
            f"ΔD-A: {self.da_change:+.3f} Å   "
            f"KIE: {self.predicted_KIE:>5.1f}   "
            f"Enhancement: {self.fold_enhancement:.2f}x   "
            f"[{self.priority}]"
        )


# ── Core calculation: Bell correction ─────────────────────────────────────────

def bell_correction(
    barrier_height_kcal: float,
    imaginary_freq_cm1:  float,
    da_distance_angstrom: float,
    temperature:         float = 298.15,
    experimental_KIE:    Optional[float] = None
) -> TunnellingResult:
    """
    Bell correction for quantum tunnelling in enzymatic H-transfer.

    This is the standard first-order treatment. It works well for reactions
    where the imaginary frequency is not too large (< 2000 cm⁻¹). For deeply
    tunnelling enzymes like AADH it underpredicts — use as a lower bound.

    Parameters
    ----------
    barrier_height_kcal : float
        Classical activation energy in kcal/mol (from QM/MM PES scan).
        Typical range for enzyme H-transfer: 8–20 kcal/mol.

    imaginary_freq_cm1 : float
        Magnitude of the imaginary frequency at the transition state in cm⁻¹.
        This is the curvature of the barrier peak.
        Obtained from a frequency calculation in ORCA at the TS geometry.
        Typical range: 500–1800 cm⁻¹ for H-transfer.
        IMPORTANT: Enter the magnitude (positive number), not the imaginary value.

    da_distance_angstrom : float
        Donor-acceptor distance at the transition state in Angstroms.
        The distance between the heavy atom donating H and the heavy atom
        accepting it (e.g., C-to-N distance in AADH).
        Shorter distance → more tunnelling.
        Typical WT range: 2.7–3.4 Å. Optimised mutants can reach 2.5 Å.

    temperature : float
        Temperature in Kelvin. Default 298.15 K (25°C).

    experimental_KIE : float, optional
        If you have a measured KIE from the literature or your own experiment,
        provide it here to get the prediction error.

    Returns
    -------
    TunnellingResult
        Full set of computed quantities. See class definition above.

    Examples
    --------
    # Wild-type AADH (Scrutton group benchmark)
    # Literature KIE ≈ 55 at 25°C
    # Literature imaginary freq ≈ 1184 cm⁻¹
    >>> result = bell_correction(
    ...     barrier_height_kcal=13.4,
    ...     imaginary_freq_cm1=1184.0,
    ...     da_distance_angstrom=2.87,
    ...     experimental_KIE=55.0
    ... )
    >>> print(result)
    """

    kT = KB * temperature  # thermal energy in Joules

    # Convert imaginary frequency from cm⁻¹ to rad/s (angular frequency)
    # ν (cm⁻¹) → ω (rad/s): multiply by 2π * c (cm/s)
    omega_ts = 2 * np.pi * imaginary_freq_cm1 * C  # rad/s

    # ── Bell correction formula ──────────────────────────────────────────────
    # Qt = 1 + (1/24)(u²)   where u = ħω / kT
    # This is the first-order expansion of the full Bell correction.
    # u is dimensionless — it's the ratio of the zero-point energy of the
    # imaginary mode to the thermal energy.

    u_H = (HBAR * omega_ts) / kT
    Qt_H = 1 + (u_H ** 2) / 24

    # For deuterium: same geometry, different mass → frequency scales as 1/√(mass)
    # ω_D = ω_H / √(m_D / m_H) = ω_H / √2  (since m_D ≈ 2 * m_H)
    omega_ts_D = omega_ts / np.sqrt(MASS_D / MASS_H)
    u_D = (HBAR * omega_ts_D) / kT
    Qt_D = 1 + (u_D ** 2) / 24

    # ── Classical KIE (Swain-Schaad) ────────────────────────────────────────
    # The classical KIE comes from ZPE difference in the REACTANT ground state,
    # not from the imaginary frequency at the TS.
    #
    # Physical picture:
    #   C-H bond has higher ZPE than C-D (lighter H vibrates faster → more ZPE).
    #   At the TS, the ZPE difference is mostly lost (reaction coordinate is
    #   the imaginary mode — it has no ZPE).
    #   So the KIE ≈ exp(ΔZPE_reactant / kT).
    #
    # Using representative C-H stretch frequency of 2950 cm⁻¹:
    #   ν_CH ≈ 2950 cm⁻¹  →  ω_CH = 2π * 2950 * c  rad/s
    #   ν_CD ≈ ν_CH / √2  (isotope effect on harmonic oscillator)
    #   ΔZPE = 0.5 * ħ * (ω_CH - ω_CD)
    #   Classical KIE = exp(ΔZPE / kT) ≈ 6-7 at 298K ✓

    omega_CH = 2 * np.pi * REACTANT_CH_FREQ * C       # rad/s
    omega_CD = omega_CH / np.sqrt(MASS_D / MASS_H)    # scales as 1/√m

    zpe_reactant_H = 0.5 * HBAR * omega_CH
    zpe_reactant_D = 0.5 * HBAR * omega_CD
    delta_zpe = zpe_reactant_H - zpe_reactant_D        # J, always positive

    classical_KIE = np.exp(delta_zpe / kT)             # should be ~6-7 at 298K

    # ── Tunnelling-corrected KIE ─────────────────────────────────────────────
    predicted_KIE = classical_KIE * (Qt_H / Qt_D)

    # ── Tunnelling fraction ──────────────────────────────────────────────────
    # What fraction of the H-transfer rate comes from below-barrier (tunnelling) paths?
    # Rough estimate: (Qt - 1) / Qt
    tunnelling_fraction = (Qt_H - 1.0) / Qt_H

    # ── D-A distance correction ──────────────────────────────────────────────
    # Bell correction doesn't explicitly use D-A distance, but we can apply
    # a Marcus-inspired scaling to account for it.
    # Tunnelling probability ∝ exp(-α * r) where α ≈ 25 Å⁻¹ for H-transfer
    # We use this in the mutation ranker, not here — here we just store the value.

    # ── Tunnelling regime classification ────────────────────────────────────
    if predicted_KIE > 25:
        regime = "deep"
    elif predicted_KIE > 10:
        regime = "moderate"
    else:
        regime = "minimal"

    # ── Prediction error vs experiment ──────────────────────────────────────
    pred_error = None
    if experimental_KIE is not None:
        pred_error = abs(predicted_KIE - experimental_KIE) / experimental_KIE

    return TunnellingResult(
        temperature=temperature,
        barrier_height=barrier_height_kcal,
        imaginary_freq=imaginary_freq_cm1,
        da_distance=da_distance_angstrom,
        Qt_H=Qt_H,
        Qt_D=Qt_D,
        predicted_KIE=predicted_KIE,
        tunnelling_fraction=tunnelling_fraction,
        classical_KIE=classical_KIE,
        tunnelling_regime=regime,
        experimental_KIE=experimental_KIE,
        prediction_error=pred_error,
    )


# ── Mutation ranking ──────────────────────────────────────────────────────────

# Marcus decay constant for H-transfer tunnelling (Å⁻¹)
# Tunnelling probability ∝ exp(-α * r_DA)
# α ≈ 25-28 Å⁻¹ for proton/hydride transfer (from Marcus theory)
ALPHA_H = 26.0

# Amino acid sidechain volumes (Å³) — proxy for steric bulk
# Used to predict how mutations change active site geometry
AA_VOLUME = {
    'GLY': 60.1,  'ALA': 88.6,  'VAL': 140.0, 'LEU': 166.7,
    'ILE': 166.7, 'PRO': 112.7, 'PHE': 189.9, 'TRP': 227.8,
    'MET': 162.9, 'SER': 89.0,  'THR': 116.1, 'CYS': 108.5,
    'TYR': 193.6, 'HIS': 153.2, 'ASP': 111.1, 'GLU': 138.4,
    'ASN': 114.1, 'GLN': 143.8, 'LYS': 168.6, 'ARG': 173.4,
}

# Sensible substitution candidates for tunnelling enhancement
# Logic: propose smaller residues (less steric bulk → compresses active site)
SUBSTITUTION_CANDIDATES = {
    'PHE': ['ALA', 'VAL', 'LEU', 'ILE'],
    'TYR': ['PHE', 'ALA', 'VAL', 'LEU'],
    'TRP': ['PHE', 'LEU', 'ALA'],
    'ILE': ['ALA', 'VAL', 'GLY'],
    'LEU': ['ALA', 'VAL', 'GLY', 'ILE'],
    'MET': ['ALA', 'VAL', 'LEU'],
    'HIS': ['ALA', 'ASN', 'GLN'],
    'ASN': ['ALA', 'SER', 'THR', 'GLY'],
    'THR': ['ALA', 'VAL', 'SER', 'GLY'],
    'SER': ['ALA', 'GLY', 'THR'],
    'GLN': ['ALA', 'ASN', 'SER'],
    'GLU': ['ALA', 'ASP', 'GLN'],
    'LYS': ['ALA', 'ARG', 'GLN'],
    'ARG': ['ALA', 'LYS', 'GLN'],
    'VAL': ['ALA', 'GLY'],
    'ASP': ['ALA', 'ASN', 'GLU'],
    'CYS': ['ALA', 'SER'],
}


def predict_mutation_effect(
    wt_result: TunnellingResult,
    residue_number: int,
    original_aa: str,
    new_aa: str,
    position_type: str = 'flanking',
) -> MutantPrediction:
    """
    Predicts the tunnelling effect of a point mutation.

    Uses a semi-empirical model:
      1. Estimate change in D-A distance from change in sidechain volume
         (smaller residue → active site compresses → D-A shortens)
      2. Compute tunnelling enhancement from Marcus exponential decay
      3. Scale predicted KIE accordingly

    Parameters
    ----------
    wt_result : TunnellingResult
        Wild-type tunnelling calculation.

    residue_number : int
        Residue number in the structure (e.g. 172 for Thr172).

    original_aa : str
        Three-letter or one-letter code of original residue (e.g. 'THR' or 'T').

    new_aa : str
        Proposed substitution (e.g. 'ALA' or 'A').

    position_type : str
        'donor_side'  – residue packs against the hydride donor
        'acceptor_side' – residue packs against the hydride acceptor
        'flanking'    – residue influences geometry less directly
        Affects the magnitude of the predicted D-A change.

    Returns
    -------
    MutantPrediction
    """

    # Normalise to three-letter codes and get single-letter for naming
    one_to_three = {
        'G':'GLY','A':'ALA','V':'VAL','L':'LEU','I':'ILE','P':'PRO',
        'F':'PHE','W':'TRP','M':'MET','S':'SER','T':'THR','C':'CYS',
        'Y':'TYR','H':'HIS','D':'ASP','E':'GLU','N':'ASN','Q':'GLN',
        'K':'LYS','R':'ARG'
    }
    orig = one_to_three.get(original_aa.upper(), original_aa.upper())
    new  = one_to_three.get(new_aa.upper(), new_aa.upper())

    # Use proper single-letter codes for mutation name (e.g. T172A not T172A)
    orig_1 = THREE_TO_ONE.get(orig, orig[0])
    new_1  = THREE_TO_ONE.get(new,  new[0])
    name = f"{orig_1}{residue_number}{new_1}"

    vol_orig = AA_VOLUME.get(orig, 120.0)
    vol_new  = AA_VOLUME.get(new,  120.0)
    delta_vol = vol_new - vol_orig  # negative = smaller substitution

    # Convert volume change to D-A distance change
    # Empirical scaling: 100 Å³ volume reduction ≈ 0.1–0.15 Å D-A compression
    # Depends on how directly the sidechain packs against the reaction axis
    position_factor = {
        'donor_side':    0.0015,  # most direct
        'acceptor_side': 0.0015,
        'flanking':      0.0008,  # less direct
    }.get(position_type, 0.0008)

    da_change = delta_vol * position_factor  # Å (negative = D-A shortens)

    new_da = wt_result.da_distance + da_change

    # Tunnelling enhancement from Marcus exponential
    # ΔkKIE / kKIE_WT = exp(-α * Δr_DA) / 1
    tunnelling_ratio = np.exp(-ALPHA_H * da_change)  # >1 if D-A shortens

    predicted_KIE = wt_result.predicted_KIE * tunnelling_ratio
    fold_enhancement = predicted_KIE / wt_result.predicted_KIE

    # Priority scoring
    if fold_enhancement > 2.0 and da_change < -0.05:
        priority = 'HIGH'
    elif fold_enhancement > 1.3 and da_change < -0.02:
        priority = 'MEDIUM'
    else:
        priority = 'LOW'

    # Human-readable rationale
    if da_change < 0:
        rationale = (
            f"{orig} → {new}: sidechain volume -{abs(delta_vol):.0f} Å³, "
            f"predicted D-A compression of {abs(da_change):.3f} Å, "
            f"tunnelling enhancement {fold_enhancement:.2f}x."
        )
    else:
        rationale = (
            f"{orig} → {new}: larger sidechain (+{delta_vol:.0f} Å³), "
            f"predicted D-A elongation — likely reduces tunnelling. "
            f"Included for completeness."
        )

    return MutantPrediction(
        residue_number=residue_number,
        original_aa=orig,
        new_aa=new,
        name=name,
        da_change=da_change,
        predicted_KIE=predicted_KIE,
        fold_enhancement=fold_enhancement,
        priority=priority,
        rationale=rationale,
    )


def rank_mutations(
    wt_result: TunnellingResult,
    candidates: list,  # list of (residue_number, original_aa, new_aa, position_type)
) -> list:
    """
    Ranks a list of candidate mutations by predicted tunnelling enhancement.

    Parameters
    ----------
    wt_result : TunnellingResult
        Wild-type baseline.

    candidates : list of tuples
        Each tuple: (residue_number, original_aa, new_aa, position_type)
        e.g. [(172, 'THR', 'ALA', 'donor_side'), (198, 'ASN', 'ALA', 'flanking')]

    Returns
    -------
    list of MutantPrediction, sorted by predicted_KIE descending
    """
    predictions = []
    for residue_number, original_aa, new_aa, position_type in candidates:
        pred = predict_mutation_effect(
            wt_result, residue_number, original_aa, new_aa, position_type
        )
        predictions.append(pred)

    return sorted(predictions, key=lambda x: x.predicted_KIE, reverse=True)
