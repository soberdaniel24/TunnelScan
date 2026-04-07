"""
breathing.py
------------
Computes the conformational breathing contribution to quantum tunnelling.

Physical basis:
  Tunnelling rate is not determined by the single crystal-structure D-A
  distance. The protein samples a thermal ensemble of conformations, and
  tunnelling occurs preferentially when fluctuations transiently compress
  the D-A distance to the tunnelling-ready geometry.

  For a Gaussian distribution of D-A distances (valid for harmonic
  fluctuations around the mean), the thermally-averaged tunnelling rate is:

    k_tunnel proportional to exp(-alpha * r_DA + alpha^2 * sigma_DA^2 / 2)
               \_________________/ \________/
               static term         BREATHING TERM

  where:
    α     = Marcus decay constant for H-transfer (~26 Å⁻¹)
    <r_DA>= mean D-A distance (from crystal structure)
    σ_DA  = standard deviation of D-A distance fluctuations

  The breathing term exp(α²σ_DA²/2) captures the enhancement from
  conformational sampling. A wider distribution = more probability of
  reaching short D-A = more tunnelling.

Estimating σ_DA from crystal data:
  Crystallographic B-factors give the mean-square displacement of each atom:
    <Δr²> = B / (8π²)

  For D-A distance fluctuations (combining donor and acceptor mobility):
    σ_DA² = f_local × [<Δr_D²> + <Δr_A²> - 2×cov(D,A)] / (8π²)
                                                            (from B-factors)
  where f_local is the fraction of crystallographic B-factor that reflects
  genuine local breathing (not lattice disorder or rigid-body motion).
  Literature estimates: f_local ≈ 0.08-0.15.

  The covariance term is estimated from GNM: atoms with highly correlated
  motions in the ENM have high cov(D,A), reducing σ_DA.

Effect of mutations on breathing:
  When a residue near the active site is mutated:
    1. Stiffening (e.g. Pro introduction, larger packing):
       → reduces local B-factors → reduces σ_DA → reduces breathing → hurts tunnelling
    2. Loosening (e.g. Gly introduction, removal of packing):
       → increases local B-factors → increases σ_DA → enhances breathing → helps tunnelling

  This is a THIRD component beyond static geometry and dynamic mechanism.

References:
  Kuznetsov & Ulstrup (1994) Can. J. Chem. 72:1009 — Gaussian breathing theory
  Johannissen et al. (2007) FEBS J. 278:1701      — AADH breathing dynamics
  Hay & Scrutton (2012) Nat. Chem. 4:161          — promoting vibrations
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pdb_parser import Structure, Residue
from elastic_network import ENMResult


# ── Constants ─────────────────────────────────────────────────────────────────

ALPHA_H     = 26.0   # Marcus decay constant for H-transfer (Å⁻¹)
LOCAL_FRAC  = 0.05   # fraction of B-factor = genuine local breathing
                     # (remaining 95% = lattice disorder, TLS, rigid-body)
                     # calibrated to give σ_DA ~ 0.16 Å at B=28/19 Å²
                     # matching Johannissen 2007 MD simulations (~0.1-0.2 Å)

# Typical B-factor change when introducing various mutations:
# From analysis of B-factors in alanine-scanning crystal structures
# (Serrano et al. 1992; D'Aquino et al. 1996)
BFACTOR_CHANGE_ON_MUTATION = {
    # (orig, new): expected ΔB in Å² for active-site residues
    # Positive = more mobile after mutation, Negative = more rigid
    # GLY introduction: removes sidechain packing, increases local mobility
    'to_GLY':  +4.5,
    # ALA introduction: removes most sidechain, moderately mobile
    'to_ALA':  +2.5,
    # VAL introduction: branched, similar or slightly more rigid
    'to_VAL':  -0.5,
    # Large hydrophobic to small: more mobile
    'PHE_to_ALA': +3.5,
    'TRP_to_ALA': +4.0,
    'TYR_to_ALA': +3.2,
    'LEU_to_ALA': +2.8,
    'ILE_to_ALA': +2.5,
    # Polar to nonpolar: loses H-bond network rigidity
    'THR_to_ALA': +3.0,
    'SER_to_ALA': +2.5,
    'ASN_to_ALA': +2.8,
    # PRO introduction: rigidifies backbone
    'to_PRO':  -3.5,
    # Default for unknown pairs
    'default':  +1.5,
}

# Amino acid intrinsic rigidity scores (higher = more rigid)
# Based on backbone phi/psi flexibility and sidechain rotamer entropy
AA_RIGIDITY = {
    'GLY': 0.1,  # most flexible backbone
    'ALA': 0.3,  'VAL': 0.5,  'LEU': 0.5,  'ILE': 0.6,
    'PRO': 0.9,  # most rigid - constrains backbone
    'PHE': 0.6,  'TRP': 0.7,  'TYR': 0.6,
    'MET': 0.4,  'SER': 0.4,  'THR': 0.5,  'CYS': 0.5,
    'HIS': 0.6,  'ASP': 0.5,  'GLU': 0.4,
    'ASN': 0.5,  'GLN': 0.4,  'LYS': 0.3,  'ARG': 0.4,
}


@dataclass
class BreathingResult:
    """Breathing contribution for one enzyme configuration."""

    # Wild-type baseline
    sigma_da_wt:        float   # D-A distance standard deviation (Å)
    breathing_factor_wt: float  # exp(α²σ²/2) for wild-type

    # Mutation effect
    delta_b_local:      float   # estimated B-factor change from mutation (Å²)
    sigma_da_mut:       float   # D-A sigma after mutation
    breathing_factor_mut: float

    # Net breathing contribution to ln(KIE)
    breathing_delta:    float   # ln(breathing_factor_mut / breathing_factor_wt)
                                # positive = mutation enhances breathing
                                # negative = mutation rigidifies, hurts breathing

    # Metadata
    mechanism:          str     # 'rigidifying' | 'mobilising' | 'neutral'
    enm_correlation:    float   # D-A motion correlation from GNM (0-1)


def compute_da_sigma(
    donor_b:    float,
    acceptor_b: float,
    da_correlation: float = 0.0,
    local_frac: float = LOCAL_FRAC
) -> float:
    """
    Estimate standard deviation of D-A distance from B-factors.

    Parameters
    ----------
    donor_b : float
        B-factor of donor atom (Å²).
    acceptor_b : float
        B-factor of acceptor atom (Å²).
    da_correlation : float
        Correlation of donor and acceptor displacements from GNM.
        Range 0-1. High correlation = motion is concerted = smaller σ_DA.
    local_frac : float
        Fraction of B-factor representing genuine local breathing.

    Returns
    -------
    float : σ_DA in Angstroms.
    """
    # Mean square displacements from B-factors
    msd_donor    = local_frac * donor_b    / (8 * np.pi**2)
    msd_acceptor = local_frac * acceptor_b / (8 * np.pi**2)

    # Variance of D-A distance
    # For anti-correlated motion: σ² = MSD_D + MSD_A (maximum)
    # For correlated motion:      σ² = MSD_D + MSD_A - 2*cov
    # cov(D,A) ≈ da_correlation × sqrt(MSD_D × MSD_A)
    cov = da_correlation * np.sqrt(msd_donor * msd_acceptor)
    var_da = max(0.0, msd_donor + msd_acceptor - 2 * cov)

    return float(np.sqrt(var_da))


def breathing_enhancement(sigma_da: float) -> float:
    """
    Compute the Gaussian breathing enhancement factor.
    exp(α² × σ_DA² / 2)

    This is the multiplicative factor applied to the classical tunnelling rate.
    """
    return float(np.exp(0.5 * ALPHA_H**2 * sigma_da**2))


def estimate_bfactor_change(
    orig_aa: str,
    new_aa:  str,
    axis_distance: float,
    residue_bfactor: float,
    structure_mean_b: float
) -> float:
    """
    Estimate change in local active-site B-factor from mutation.

    Logic:
    - Larger → smaller residue: loses packing contacts → more mobile → ΔB > 0
    - Smaller → larger residue: gains packing → more rigid → ΔB < 0
    - Introduction of Gly: maximum flexibility gain
    - Introduction of Pro: rigidification
    - Residues closer to D-A axis have larger effect on σ_DA

    The axis distance scaling accounts for the fact that residues
    further away have less influence on the active site dynamics.

    Parameters
    ----------
    orig_aa, new_aa : str
        Three-letter amino acid codes.
    axis_distance : float
        Distance of residue sidechain centroid from D-A axis (Å).
    residue_bfactor : float
        Current B-factor of the residue.
    structure_mean_b : float
        Mean B-factor of the whole structure (for normalisation).

    Returns
    -------
    float : estimated ΔB at the active site (Å²).
    """
    # Look up specific transition first
    specific_key = f'{orig_aa}_to_{new_aa}'
    if specific_key in BFACTOR_CHANGE_ON_MUTATION:
        base_delta_b = BFACTOR_CHANGE_ON_MUTATION[specific_key]
    elif new_aa == 'GLY':
        base_delta_b = BFACTOR_CHANGE_ON_MUTATION['to_GLY']
    elif new_aa == 'ALA':
        base_delta_b = BFACTOR_CHANGE_ON_MUTATION['to_ALA']
    elif new_aa == 'PRO':
        base_delta_b = BFACTOR_CHANGE_ON_MUTATION['to_PRO']
    elif new_aa == 'VAL':
        base_delta_b = BFACTOR_CHANGE_ON_MUTATION['to_VAL']
    else:
        # Use rigidity difference between original and new residue
        rig_orig = AA_RIGIDITY.get(orig_aa, 0.5)
        rig_new  = AA_RIGIDITY.get(new_aa,  0.5)
        # More rigid new residue → negative ΔB (stiffening)
        # Less rigid new residue → positive ΔB (loosening)
        base_delta_b = -(rig_new - rig_orig) * 6.0

    # Scale by proximity to D-A axis
    # Gaussian decay: full effect within 3 Å, falls off beyond
    axis_scale = float(np.exp(-(axis_distance**2) / (2 * 3.5**2)))
    axis_scale = float(np.clip(axis_scale, 0.05, 1.0))

    # Scale by how mobile the residue already is
    # A highly mobile residue (high B) changes less when mutated
    # A rigid residue (low B) can change more
    b_scale = float(np.clip(structure_mean_b / max(residue_bfactor, 5.0), 0.3, 2.0))

    return float(base_delta_b * axis_scale * b_scale)


def compute_breathing_contribution(
    structure:     Structure,
    enm:           ENMResult,
    donor_chain:   str,
    donor_resnum:  int,
    donor_atom:    str,
    acceptor_chain: str,
    acceptor_resnum: int,
    acceptor_atom:  str,
    mutated_residue: Residue,
    new_aa:         str,
    axis_distance:  float
) -> BreathingResult:
    """
    Compute the breathing contribution for a single mutation.

    Parameters
    ----------
    structure : Structure
        Parsed PDB.
    enm : ENMResult
        Pre-computed GNM.
    donor_chain, donor_resnum, donor_atom : str, int, str
        Identity of the H-transfer donor atom.
    acceptor_chain, acceptor_resnum, acceptor_atom : str, int, str
        Identity of the H-transfer acceptor atom.
    mutated_residue : Residue
        The residue being mutated.
    new_aa : str
        Three-letter code of the substitution.
    axis_distance : float
        Distance of mutated residue from D-A axis (Å).

    Returns
    -------
    BreathingResult
    """
    # ── Get donor/acceptor B-factors ─────────────────────────────────────────
    d_atom = structure.get_atom(donor_chain, donor_resnum, donor_atom)
    a_atom = structure.get_atom(acceptor_chain, acceptor_resnum, acceptor_atom)

    donor_b    = d_atom.bfactor    if d_atom    else structure.mean_bfactor
    acceptor_b = a_atom.bfactor    if a_atom    else structure.mean_bfactor

    # ── GNM correlation between donor and acceptor motion ────────────────────
    # High correlation = they move together = σ_DA is smaller
    d_res = structure.get_residue(donor_chain, donor_resnum)
    a_res = structure.get_residue(acceptor_chain, acceptor_resnum)

    da_correlation = 0.0
    if d_res and a_res:
        d_part = enm.get_participation(donor_chain, donor_resnum)
        a_part = enm.get_participation(acceptor_chain, acceptor_resnum)
        # Approximate correlation from ENM participation overlap
        da_correlation = float(np.sqrt(d_part * a_part) * 0.4)

    # ── Wild-type σ_DA ────────────────────────────────────────────────────────
    sigma_wt = compute_da_sigma(donor_b, acceptor_b, da_correlation)
    bf_wt    = breathing_enhancement(sigma_wt)

    # ── Estimate B-factor change from mutation ────────────────────────────────
    delta_b = estimate_bfactor_change(
        orig_aa          = mutated_residue.name,
        new_aa           = new_aa,
        axis_distance    = axis_distance,
        residue_bfactor  = mutated_residue.sidechain_bfactor,
        structure_mean_b = structure.mean_bfactor
    )

    # The mutation changes local B-factors near the active site
    # We apply delta_b to both donor and acceptor B-factors (scaled by proximity)
    # Residues closer to donor affect donor_b more, closer to acceptor affect acceptor_b more
    d_atom_coords = d_atom.coords if d_atom else None
    a_atom_coords = a_atom.coords if a_atom else None
    mut_centroid  = mutated_residue.sidechain_centroid

    if mut_centroid is not None and d_atom_coords is not None:
        dist_to_donor    = float(np.linalg.norm(mut_centroid - d_atom_coords))
        dist_to_acceptor = float(np.linalg.norm(mut_centroid - a_atom_coords)) if a_atom_coords is not None else 10.0
        total = dist_to_donor + dist_to_acceptor + 1e-6
        donor_weight    = dist_to_acceptor / total  # closer to donor = higher weight
        acceptor_weight = dist_to_donor    / total
    else:
        donor_weight = acceptor_weight = 0.5

    new_donor_b    = max(3.0, donor_b    + delta_b * donor_weight)
    new_acceptor_b = max(3.0, acceptor_b + delta_b * acceptor_weight)

    sigma_mut = compute_da_sigma(new_donor_b, new_acceptor_b, da_correlation)
    bf_mut    = breathing_enhancement(sigma_mut)

    # ── Breathing delta ───────────────────────────────────────────────────────
    breathing_delta = float(np.log(bf_mut) - np.log(bf_wt))

    # ── Directionality correction ─────────────────────────────────────────────
    # Critical insight: not all breathing is equal.
    # Directed breathing (motion correlated with D-A compression) helps tunnelling.
    # Undirected breathing (random mobility from structural disruption) does not.
    #
    # When a promoting vibration residue loses H-bond character, the active site
    # becomes more mobile but that mobility is UNCORRELATED with D-A compression.
    # This converts directed breathing into random noise — which actually hurts
    # tunnelling by reducing the probability of reaching the tunnelling-ready geometry.
    #
    # We estimate the directionality of the original residue's breathing from:
    #   - ENM participation: high = motion is part of global compressive mode
    #   - H-bond to substrate: direct mechanical coupling to D-A axis
    # If both are high AND the mutation loses H-bond character, the increased
    # mobility is undirected → apply a directionality penalty to breathing_delta.

    d_res_key = (donor_chain, donor_resnum)
    a_res_key = (acceptor_chain, acceptor_resnum)

    enm_participation = enm.get_participation(mutated_residue.chain,
                                               mutated_residue.number)

    # Check if residue has H-bond coupling to donor/acceptor
    # Use 4.5 A cutoff to catch near-H-bonds and indirect couplings
    has_da_hbond = False
    if mutated_residue.polar_atoms:
        d_atom_obj = structure.get_atom(donor_chain, donor_resnum, donor_atom)
        a_atom_obj = structure.get_atom(acceptor_chain, acceptor_resnum, acceptor_atom)
        # Also check against all atoms of donor/acceptor residue (not just the
        # specific D/A atom) — T172 H-bonds to Asp128 OD1/OD2 directly
        d_res_atoms = []
        a_res_atoms = []
        d_res_obj = structure.get_residue(donor_chain, donor_resnum)
        a_res_obj = structure.get_residue(acceptor_chain, acceptor_resnum)
        if d_res_obj:
            d_res_atoms = [at for at in d_res_obj.polar_atoms]
        if a_res_obj:
            a_res_atoms = [at for at in a_res_obj.polar_atoms]

        all_partner_atoms = (
            ([d_atom_obj] if d_atom_obj else []) +
            ([a_atom_obj] if a_atom_obj else []) +
            d_res_atoms + a_res_atoms
        )
        for pa in mutated_residue.polar_atoms:
            for partner in all_partner_atoms:
                if pa.distance_to(partner) < 4.5:  # generous cutoff
                    has_da_hbond = True
                    break

    # Directionality score: how much of the residue's mobility is directed
    # along the D-A compression axis
    # High ENM participation + H-bond to D/A = highly directed
    directionality = float(np.clip(
        0.5 * enm_participation + 0.5 * float(has_da_hbond),
        0.0, 1.0
    ))

    # If mutation disrupts H-bond capacity of a directed residue:
    # the increased mobility becomes undirected → reverse the breathing gain
    hbond_lost = (mutated_residue.can_hbond and new_aa not in
                  {'SER','THR','TYR','ASN','GLN','ASP','GLU','HIS','LYS','ARG','CYS','TRP'})

    if hbond_lost and directionality > 0.3:
        # Undirected breathing — the delta should be reversed for directed component
        # Partial correction: the undirected portion actually DECREASES effective σ_DA
        undirected_penalty = -directionality * abs(breathing_delta) * 1.5
        breathing_delta = breathing_delta + undirected_penalty

    # Classify mechanism
    if breathing_delta > 0.15:
        mechanism = 'mobilising'
    elif breathing_delta < -0.15:
        mechanism = 'rigidifying'
    else:
        mechanism = 'neutral'

    return BreathingResult(
        sigma_da_wt=sigma_wt,
        breathing_factor_wt=bf_wt,
        delta_b_local=delta_b,
        sigma_da_mut=sigma_mut,
        breathing_factor_mut=bf_mut,
        breathing_delta=breathing_delta,
        mechanism=mechanism,
        enm_correlation=da_correlation
    )
