"""
electrostatics.py
-----------------
Computes the electrostatic contribution to quantum tunnelling.

Physical basis:
  The tunnelling barrier height is not purely geometric — it has an
  electrostatic component. Warshel's group (Liu & Warshel 2007) showed
  that electrostatic preorganisation accounts for a significant fraction
  of enzyme catalytic power. For tunnelling specifically:

  1. Charged residues near the D-A axis create an electrostatic field
     that stabilises the transition state geometry, effectively lowering
     the barrier for the tunnelling-ready configuration.

  2. When a charged residue is mutated to alanine (common in alanine
     scanning), the electrostatic stabilisation is lost. This raises the
     effective barrier height and reduces tunnelling.

  3. The effect is distance-dependent (Coulomb: 1/r) and orientation-
     dependent (the projection along the D-A axis matters most).

Model:
  For each charged residue near the D-A axis:

    E_elec = sum_i [ q_i / (eps × r_i) × cos(theta_i) ]

  where:
    q_i     = partial charge of residue i
    r_i     = distance from residue i to D-A midpoint
    eps     = effective dielectric constant (~20 in protein interior)
    theta_i = angle between residue-to-DA vector and DA axis
    cos(θ)  = projection: only component along D-A axis contributes

  When a charged residue is mutated to neutral:
    delta_E = -E_elec_i (loss of stabilisation = barrier increase)
    delta_KIE = exp(-delta_E / kT) × tunnelling_sensitivity

Reference:
  Warshel et al. (2006) Chem. Rev. 106:3210 — electrostatic catalysis
  Liu & Warshel (2007) J. Phys. Chem. B 111:7863 — tunnelling + electrostatics
  Hay & Scrutton (2012) Nat. Chem. 4:161 — AADH electrostatic environment
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pdb_parser import Structure, Residue


# ── Constants ─────────────────────────────────────────────────────────────────

EPSILON_PROTEIN  = 20.0     # effective dielectric in protein interior
BOLTZMANN_KCAL   = 0.593    # kT in kcal/mol at 298K
TUNNELLING_SENS  = 0.15     # electrostatic sensitivity of tunnelling
                            # (fraction of barrier change that affects KIE)
                            # calibrated to give ~0.1-0.3 ln(KIE) change
                            # for Arg/Lys mutations at 5-8A from D-A axis

# Formal charges for amino acid residues (in standard protonation state at pH 7)
# Positive: Arg (+1), Lys (+1), His (0 at pH 7, +1 protonated)
# Negative: Asp (-1), Glu (-1)
# Neutral: all others
RESIDUE_CHARGE = {
    'ARG': +1.0,
    'LYS': +1.0,
    'HIS': +0.1,   # partial — ~10% protonated at pH 7
    'ASP': -1.0,
    'GLU': -1.0,
}

# Partial charge change when mutated to ALA or GLY
# Positive = mutation adds positive charge (rare)
# Negative = mutation removes positive charge
def charge_change_on_mutation(orig_aa: str, new_aa: str) -> float:
    """
    Net charge change when orig_aa is mutated to new_aa.
    Returns delta_charge = charge(new) - charge(orig).
    """
    q_orig = RESIDUE_CHARGE.get(orig_aa, 0.0)
    q_new  = RESIDUE_CHARGE.get(new_aa, 0.0)
    return q_new - q_orig


# ── Electrostatic scoring ──────────────────────────────────────────────────────

def electrostatic_field_at_da(
    structure: Structure,
    donor_coords: np.ndarray,
    acceptor_coords: np.ndarray,
    radius: float = 12.0,
    epsilon: float = EPSILON_PROTEIN
) -> Dict[Tuple[str, int], float]:
    """
    Compute the electrostatic field contribution of each charged residue
    at the D-A midpoint, projected along the D-A compression axis.

    Returns dict: (chain, resnum) → electrostatic field contribution (kcal/mol/e)

    Positive = stabilises (reduces barrier)
    Negative = destabilises (increases barrier)
    """
    da_midpoint = (donor_coords + acceptor_coords) / 2.0
    da_vec      = acceptor_coords - donor_coords
    da_unit     = da_vec / np.linalg.norm(da_vec)

    field_contributions = {}

    for (chain, resnum), res in structure.residues.items():
        if res.is_hetatm:
            continue
        charge = RESIDUE_CHARGE.get(res.name, 0.0)
        if abs(charge) < 0.05:
            continue   # skip uncharged residues

        centroid = res.sidechain_centroid
        if centroid is None:
            continue

        r_vec = da_midpoint - centroid
        r     = float(np.linalg.norm(r_vec))

        if r > radius or r < 0.5:
            continue

        # Coulomb field: E = q / (eps * r)
        field_magnitude = charge / (epsilon * r)

        # Project along D-A axis: only component that compresses D-A matters
        r_unit    = r_vec / r
        cos_theta = float(np.dot(r_unit, da_unit))

        # Projected field contribution
        field_proj = field_magnitude * abs(cos_theta)

        field_contributions[(chain, resnum)] = field_proj

    return field_contributions


def electrostatic_delta(
    orig_aa: str,
    new_aa: str,
    field_contribution: float,
    sensitivity: float = TUNNELLING_SENS
) -> float:
    """
    Change in ln(KIE) from the electrostatic effect of a mutation.

    When a charged residue is mutated:
      - Field contribution is lost (or changed)
      - This changes the effective barrier height
      - Which changes the tunnelling rate

    Parameters
    ----------
    orig_aa : str
        Original amino acid (three-letter).
    new_aa : str
        New amino acid after mutation.
    field_contribution : float
        Electrostatic field contribution of this residue at D-A midpoint.
    sensitivity : float
        Fraction of electrostatic change that affects tunnelling KIE.

    Returns
    -------
    float : delta ln(KIE) from electrostatic effect.
             Positive = mutation enhances tunnelling (removes destabilising charge)
             Negative = mutation hurts tunnelling (removes stabilising charge)
    """
    dq = charge_change_on_mutation(orig_aa, new_aa)

    if abs(dq) < 0.05:
        return 0.0  # no charge change, no electrostatic effect

    # Change in field = delta_charge × (field per unit charge)
    # field_contribution already normalised per unit charge of orig residue
    q_orig = RESIDUE_CHARGE.get(orig_aa, 1.0)
    if abs(q_orig) < 0.05:
        return 0.0

    field_per_unit = field_contribution / q_orig
    delta_field    = dq * field_per_unit

    # Convert field change to barrier change: dE = delta_field × e × d_DA
    # where d_DA ≈ 0.6 A (tunnelling distance in AADH)
    # Units: (kcal/mol/e) × e × A → need to convert to ln(KIE) units
    # Sensitivity parameter absorbs the unit conversion and calibration
    delta_ln_kie = delta_field * sensitivity

    return float(delta_ln_kie)


@dataclass
class ElectrostaticsMap:
    """Pre-computed electrostatic field contributions for all charged residues."""
    field_map: Dict[Tuple[str, int], float]
    n_charged: int
    mean_field: float

    def get_contribution(self, chain: str, resnum: int) -> float:
        return self.field_map.get((chain, resnum), 0.0)

    def get_delta(self, chain: str, resnum: int,
                  orig_aa: str, new_aa: str) -> float:
        field = self.get_contribution(chain, resnum)
        return electrostatic_delta(orig_aa, new_aa, field)


def build_electrostatics_map(
    structure: Structure,
    donor_coords: np.ndarray,
    acceptor_coords: np.ndarray,
    radius: float = 12.0
) -> ElectrostaticsMap:
    """
    Build the complete electrostatics map for a structure.
    Call once per scan, then query per mutation.
    """
    field_map = electrostatic_field_at_da(
        structure, donor_coords, acceptor_coords, radius=radius
    )
    if field_map:
        mean_f = float(np.mean(list(field_map.values())))
    else:
        mean_f = 0.0

    return ElectrostaticsMap(
        field_map=field_map,
        n_charged=len(field_map),
        mean_field=mean_f
    )
