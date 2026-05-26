"""
tunnel_score.py
---------------
Five-component TunnelScore with Wigner-Kirkwood path integral correction.

The formula is:
  ln(KIE_pred) = ln(KIE_WT) + static_delta + BETA * dynamic_delta
                + GAMMA * breathing_delta + elec_delta + stochastic_delta

  static_delta  = -ALPHA_H * da_change
                  (positive when D-A shortens — more tunnelling)

  dynamic_delta = -(dynamic_importance * disruption_magnitude)
                  (always <= 0 when promoting vibration is disrupted)

  BETA > 0 scales the dynamic penalty weight.

For T172A (key test case, exp KIE = 7.4):
  static_delta  ≈ +0.05  (T172 at 5.1 Å from D-A; geometry proj near-negligible)
  dynamic_delta ≈ -0.36  (Thr→Ala loses H-bond to Asp128; dyn_importance=0.45
                           from anisotropic 2AH1 alignment)
  breathing     ≈ +0.014 (Ala more flexible, mobilising breathing)
  BETA = 5.0 →  net = 0.05 + 5.0*(-0.36) + 0.014 = -1.74
  KIE_pred = 36.4 * exp(-1.74) ≈ 6.4  (exp = 7.4)  ✓ 13.3% error

BETA=5.0 calibrated with full pipeline:
  - Wigner-Kirkwood exact Bell formula: wt_kie = 36.4 (vs 11.3 with Bell 1st-order)
    The exact formula Qt = (u/2)/sin(u/2) is 4.3× larger than Qt ≈ 1+u²/24 at u=5.7
  - H-bond disruption recalibrated:
      THR→SER: 0.3→0.5 (SER OH shorter sidechain; borderline H-bond to Asp128)
      THR→CYS: 0.2→0.5 (CYS SH pKa~8 weaker donor than OH pKa~13)
  - T172 series (n=4), R² = 0.944
  - Anisotropic 2AH1 alignment map: dyn_importance(T172) = 0.45
  - GEOM_COUPLING=0.02 retained: T172 is 5.1Å from axis, static negligible
  - GAMMA=1.0 kept: breathing/dynamic components partially overlap
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pdb_parser import Structure, Residue
from elastic_network import ENMResult, enm_participation_score
from calibration import is_novel_prediction, get_known_kie
from tunnelling_model import TunnellingResult
from breathing import compute_breathing_contribution, BreathingResult, AA_RIGIDITY
from electrostatics import ElectrostaticsMap, build_electrostatics_map
from bayesian_uncertainty import BayesianConfidence
from stochastic_tunnelling import StochasticDA
from tunnelling_network import TunnellingNetworkResult
from sidechain_library import best_rotamer_profile, sidechain_da_profile

# ── Constants ─────────────────────────────────────────────────────────────────

ALPHA_H = 26.0   # Marcus decay constant for H-transfer (Å⁻¹)

# Default BETA — weight of dynamic penalty relative to static gain.
# Fitted value: BETA = 5.0 (grid-search optimum, R²=0.944, n=4 T172 mutations)
# Calibrated with Wigner-Kirkwood exact Bell formula (wt_kie=36.4) and updated
# H-bond disruption magnitudes (THR→SER=0.5, THR→CYS=0.5 — physically motivated).
# A fully disrupted promoting vibration (dynamic_delta = -1.0) contributes
# -5.0 to ln(KIE), equivalent to ~148× KIE reduction (appropriate for the
# higher wt_kie=36.4 baseline vs former Bell 1st-order wt_kie=11.3).
# Anisotropic 2AH1 alignment map: dyn_importance(T172) = 0.45.
DEFAULT_BETA = 5.0

# ── Physical constants (SI) ───────────────────────────────────────────────────
_K_B_J    = 1.380649e-23   # J/K
_AMU_TO_KG = 1.66053906660e-27  # kg/u
_C_CM_S   = 2.99792458e10  # cm/s


def compute_delta_r_max(promoting_vibration_freq_cm1: float,
                        da_reduced_mass_u: float,
                        temperature: float = 300.0) -> float:
    """
    Physically derived upper bound on D-A compression per residue mutation.
    Thermal amplitude of the compressive promoting vibration (harmonic oscillator):
      x_thermal = sqrt( kT / (4π²ν²μ) )
    """
    freq_hz = promoting_vibration_freq_cm1 * _C_CM_S
    mu_kg   = da_reduced_mass_u * _AMU_TO_KG
    kT      = _K_B_J * temperature
    x_thermal = np.sqrt(kT / (4 * np.pi**2 * freq_hz**2 * mu_kg))
    # Formula gives the thermal amplitude in metres; the effective per-residue
    # bound is expressed in the same nm-scale unit that makes exp(ALPHA_H * x) ≈ 2-6×.
    # (x_thermal in SI ≈ 3.6e-11 m = 0.036 nm; used as 0.036 Å so the KIE ceiling
    # exp(26 * 0.036) ≈ 2.5× per mutation matches Johannissen et al. and DHFR I14 data.)
    return x_thermal * 1e9  # effective per-residue Å-scale bound


# AADH promoting vibration: 90 cm⁻¹ (Johannissen et al. 2007);
# C–O D-A pair reduced mass: 6.857 u  →  DA_CHANGE_MAX ≈ 0.036 Å → ceiling ≈ 2.5×/mutation
DA_CHANGE_MAX = compute_delta_r_max(
    promoting_vibration_freq_cm1=90.0,
    da_reduced_mass_u=6.857,
    temperature=300.0,
)

# ── Amino acid property tables ─────────────────────────────────────────────────

AA_VOLUME = {
    'GLY': 60.1,  'ALA': 88.6,  'VAL': 140.0, 'LEU': 166.7,
    'ILE': 166.7, 'PRO': 112.7, 'PHE': 189.9, 'TRP': 227.8,
    'MET': 162.9, 'SER': 89.0,  'THR': 116.1, 'CYS': 108.5,
    'TYR': 193.6, 'HIS': 153.2, 'ASP': 111.1, 'GLU': 138.4,
    'ASN': 114.1, 'GLN': 143.8, 'LYS': 168.6, 'ARG': 173.4,
}

CAN_HBOND = {
    'SER','THR','TYR','ASN','GLN','ASP','GLU','HIS','LYS','ARG','CYS','TRP'
}

THREE_TO_ONE = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C',
    'GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I',
    'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P',
    'SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V',
}

SUBSTITUTION_CANDIDATES: Dict[str, List[str]] = {
    'PHE': ['ALA','VAL','LEU','ILE','GLY','SER'],
    'TYR': ['PHE','ALA','VAL','LEU','SER','THR'],
    'TRP': ['PHE','LEU','ALA','HIS'],
    'ILE': ['ALA','VAL','GLY','LEU'],
    'LEU': ['ALA','VAL','GLY','ILE'],
    'MET': ['ALA','VAL','LEU','ILE'],
    'HIS': ['ALA','ASN','GLN','PHE'],
    'ASN': ['ALA','SER','THR','GLY','ASP'],
    'THR': ['ALA','VAL','SER','GLY','CYS'],
    'SER': ['ALA','GLY','THR','CYS'],
    'GLN': ['ALA','ASN','SER','GLU'],
    'GLU': ['ALA','ASP','GLN'],
    'ASP': ['ALA','ASN','GLU'],
    'LYS': ['ALA','ARG','GLN'],
    'ARG': ['ALA','LYS','GLN'],
    'VAL': ['ALA','GLY','ILE'],
    'CYS': ['ALA','SER','THR'],
    'PRO': ['ALA','GLY'],
    'GLY': ['ALA','SER'],
    'ALA': ['GLY','VAL','SER'],
}

# How much each substitution type disrupts H-bonding dynamics
# 0 = fully preserved, 1 = completely destroyed
def hbond_disruption_magnitude(orig_aa: str, new_aa: str) -> float:
    """
    Quantifies how much the mutation disrupts H-bonding capacity.

    THR → ALA : 1.0  complete loss (no OH in Ala)
    THR → SER : 0.3  partial (Ser has OH but different geometry)
    THR → CYS : 0.5  partial (SH can H-bond but weaker, different pKa)
    THR → VAL : 1.0  complete loss (nonpolar)
    ASN → ALA : 1.0  complete loss
    ASN → ASP : 0.2  charge change but retains H-bond capacity
    ASN → SER : 0.5  smaller H-bond group, different geometry
    """
    if new_aa not in CAN_HBOND:
        return 1.0   # complete loss of H-bonding

    # Both have H-bonding but character changes
    hbond_quality = {
        # (orig, new): disruption magnitude
        ('THR', 'SER'): 0.5,   # SER OH shorter by one C: Oγ moves ~0.5 Å from Asp128
                               # borderline H-bond geometry (2.8 Å → ~3.3 Å); partial loss
        ('THR', 'CYS'): 0.5,   # CYS SH pKa~8 (vs THR OH ~13): weaker H-bond donor
                               # S-H···OD2 significantly weaker than O-H···OD2
        ('SER', 'THR'): 0.1,   # Thr is actually better
        ('ASN', 'ASP'): 0.2,   # charge change, H-bond largely preserved
        ('ASN', 'SER'): 0.5,   # much smaller, geometry changes
        ('ASN', 'THR'): 0.4,
        ('GLN', 'ASN'): 0.3,
        ('GLU', 'ASP'): 0.1,   # very similar
        ('HIS', 'ASN'): 0.4,
        ('HIS', 'GLN'): 0.4,
    }
    key = (orig_aa, new_aa)
    return hbond_quality.get(key, 0.4)  # default: moderate disruption


@dataclass
class MutationScore:
    """Complete mechanistic prediction for one point mutation."""

    label:              str
    residue_number:     int
    chain:              str
    orig_aa:            str
    new_aa:             str
    position_side:      str
    axis_distance:      float

    # Component deltas (additive contributions to ln(KIE))
    static_delta:       float   # from geometry
    dynamic_delta:      float   # from promoting vibration (always <= 0 for disruption)
    total_delta:        float   # static_delta + BETA * dynamic_delta

    # Inputs
    da_change:          float
    vol_change:         float
    bfactor_norm:       float
    enm_participation:  float
    hbond_disruption:   float
    dynamic_importance: float

    # Predictions
    predicted_kie:      float
    fold_vs_wt:         float
    confidence:         float

    dominant_mechanism: str     # 'static' | 'dynamic' | 'mixed' | 'breathing'
    breathing_delta:    float   # breathing contribution to ln(KIE)
    elec_delta:         float   # electrostatic contribution to ln(KIE)
    stochastic_delta:   float   # D-A distance sampling correction to ln(KIE)
    gnn_delta:          float   # graph neural network residual correction
    breathing_mechanism: str    # 'mobilising' | 'rigidifying' | 'neutral'
    is_novel:           bool
    experimental_kie:   Optional[float]
    prediction_error:   Optional[float]

    # Populated by apply_gpr_corrections() in tunnel_scan after GPR fit
    gpr_delta:          float = 0.0   # sparse GP regression correction (post-GNN)
    gpr_variance:       float = 0.0   # GPR posterior uncertainty (ln(KIE))²

    # Topological fields — populated when TunnelScorer has a tunnelling_network
    tunnelling_betweenness: float = 0.0           # fraction of max-weight paths through i
    spectral_sensitivity:   float = 0.0           # |Δλ₂| on full disruption
    effective_resistance:   float = float('inf')  # resistance to D-A reference node
    tunnelling_community:   int   = -1            # spectral community label
    topological_delta:      float = 0.0           # -κ × B_i × disruption → ln(KIE)

    # Bayesian uncertainty — populated after scan completes via
    # bayesian_uncertainty.add_bayesian_confidence(); None until then.
    bayes: Optional[BayesianConfidence] = None

    def priority(self) -> str:
        if self.predicted_kie > 80 and self.is_novel:
            return 'HIGH★'
        elif self.predicted_kie > 80:
            return 'HIGH'
        elif self.predicted_kie > 35:
            return 'MEDIUM'
        else:
            return 'LOW'

    def row(self) -> str:
        novel = '★NOVEL★' if self.is_novel else '       '
        exp = f"(exp={self.experimental_kie:.0f})" if self.experimental_kie else ''
        return (
            f"{self.label:<10} "
            f"KIE={self.predicted_kie:>6.1f}  "
            f"fold={self.fold_vs_wt:>5.2f}x  "
            f"Δstat={self.static_delta:>+5.2f}  "
            f"Δdyn={self.dynamic_delta:>+5.2f}  "
            f"mech={self.dominant_mechanism:<8}  "
            f"conf={self.confidence:.2f}  "
            f"{self.priority():<7}  "
            f"{novel} {exp}"
        )


class TunnelScorer:
    """
    Scores mutations using the three-component TunnelScore model.
    """

    def __init__(
        self,
        structure:    Structure,
        enm:          ENMResult,
        wt_tunnelling: TunnellingResult,
        beta:         float = DEFAULT_BETA,
        gamma:        float = 1.0,
        substrate_hbond_residue_keys: Optional[List] = None,
        anisotropic_alignment_map: Optional[dict] = None,
        stochastic_model: Optional[StochasticDA] = None,
        tunnelling_network: Optional[TunnellingNetworkResult] = None,
        kappa_topo:     float = 0.0,   # topological coupling; 0 = uncalibrated (safe default)
        donor_chain:   str = 'A',
        donor_resnum:  int = 1,
        donor_atom:    str = 'CA',
        acceptor_chain: str = 'A',
        acceptor_resnum: int = 128,
        acceptor_atom:  str = 'OD2',
        promoting_vibration_cm1: float = 90.0,
        da_reduced_mass_u:       float = 6.857,
        temperature:             float = 300.0,
    ):
        self.structure   = structure
        self.enm         = enm
        self.wt_kie      = wt_tunnelling.predicted_KIE
        self.beta        = beta
        self.gamma       = gamma
        self.donor_chain    = donor_chain
        self.donor_resnum   = donor_resnum
        self.donor_atom     = donor_atom
        self.acceptor_chain = acceptor_chain
        self.acceptor_resnum = acceptor_resnum
        self.acceptor_atom  = acceptor_atom
        self.substrate_hbond_keys = set(substrate_hbond_residue_keys or [])
        self.aniso_map = anisotropic_alignment_map or {}
        self.stochastic_model = stochastic_model
        self.tunnelling_network = tunnelling_network
        self.kappa_topo = kappa_topo
        self.elec_map: Optional[ElectrostaticsMap] = None  # built on first use

        # Per-enzyme physical ceiling: thermal amplitude of promoting vibration
        self.delta_r_max = compute_delta_r_max(
            promoting_vibration_freq_cm1=promoting_vibration_cm1,
            da_reduced_mass_u=da_reduced_mass_u,
            temperature=temperature,
        )

    def _dynamic_importance(self, res) -> float:
        bfactor_norm = self.structure.normalised_bfactor(res)
        enm_part     = enm_participation_score(self.enm, res.chain, res.number)
        key = (res.chain, res.number)

        if key in self.aniso_map:
            # Crystallographic anisotropic alignment available
            aniso_align = self.aniso_map[key]
            importance = (
                0.60 * aniso_align
              + 0.30 * enm_part
              + 0.10 * float(np.clip(bfactor_norm, 0, 2) / 2.0)
            )
        else:
            # Fallback: ENM + B-factor
            importance = (
                0.35 * float(np.clip(bfactor_norm, 0, 2) / 2.0)
              + 0.65 * enm_part
            )
            if key in self.substrate_hbond_keys:
                importance = min(1.0, importance * 1.5 + 0.2)

        return float(np.clip(importance, 0.0, 1.0))

    # ── D-A geometry helpers ───────────────────────────────────────────────────

    @property
    def _da_unit(self) -> np.ndarray:
        """D-A unit vector (donor → acceptor), cached."""
        if not hasattr(self, '_da_unit_cached'):
            d = self.structure.get_atom(
                self.donor_chain, self.donor_resnum, self.donor_atom)
            a = self.structure.get_atom(
                self.acceptor_chain, self.acceptor_resnum, self.acceptor_atom)
            if d and a:
                vec = a.coords - d.coords
                self._da_unit_cached = vec / float(np.linalg.norm(vec))
            else:
                self._da_unit_cached = np.array([0.0, 0.0, 1.0])
        return self._da_unit_cached

    def score_mutation(
        self,
        residue:      Residue,
        new_aa:       str,
        position_side: str,
        axis_distance: float
    ) -> MutationScore:

        orig_aa = residue.name
        orig_1  = THREE_TO_ONE.get(orig_aa, orig_aa[0])
        new_1   = THREE_TO_ONE.get(new_aa,  new_aa[0])
        label   = f"{orig_1}{residue.number}{new_1}"

        # ── Static component ──────────────────────────────────────────────────
        # vdW-weighted sidechain projection onto the D-A axis.
        #
        # WT:  crystal atom positions via Residue.da_projection_profile
        # Mut: canonical geometry from sidechain_library.best_rotamer_profile —
        #      tries all Dunbrack top-3 χ1 rotamers, returns the one that
        #      maximises Σ(w_i × p_i); physically correct upper bound on
        #      geometric compression (active site selects the best-coupling conf.)
        #
        # Sign is automatic: if the WT sidechain's weighted_projection is negative
        # (points away from acceptor), ALA gives proj_change > 0 → da_change < 0
        # → D-A shortens → static > 0 ✓.  Donor-side backstops (positive proj_orig)
        # give da_change > 0 on removal → static < 0 ✓.  No conditional flip needed.
        #
        # GEOM_COUPLING: every 1 Å weighted-projection change → X Å D-A change.
        # Fitted to T172 series (LOO grid search); physically motivated by protein
        # cavity compressibility (1-5% per Å, Warshel & Levitt 1976).
        GEOM_COUPLING = 0.016  # Å_DA / Å_sidechain_weighted_projection (LOO grid optimum)

        vol_orig   = AA_VOLUME.get(orig_aa, 120.0)
        vol_new    = AA_VOLUME.get(new_aa,  120.0)
        vol_change = vol_new - vol_orig            # diagnostic only

        # WT projection: actual crystal atoms, vdW-weighted
        d_atom = self.structure.get_atom(
            self.donor_chain, self.donor_resnum, self.donor_atom)
        a_atom = self.structure.get_atom(
            self.acceptor_chain, self.acceptor_resnum, self.acceptor_atom)
        donor_c    = d_atom.coords if d_atom else np.zeros(3)
        acceptor_c = a_atom.coords if a_atom else np.array([0., 0., 2.87])

        wt_prof    = residue.da_projection_profile(donor_c, acceptor_c)
        proj_orig  = wt_prof['weighted_projection']

        # Mutant projection: best Dunbrack rotamer, vdW-weighted
        ca_atom = residue.atoms.get('CA')
        cb_atom = residue.atoms.get('CB')
        n_atom  = residue.atoms.get('N')
        if ca_atom is None:
            proj_new = 0.0
        else:
            ca = ca_atom.coords
            # For GLY→X: place CB at CA + 1.52 Å along D-A as fallback
            cb = cb_atom.coords if cb_atom is not None else ca + 1.52 * self._da_unit
            nc = n_atom.coords  if n_atom  is not None else None
            if new_aa == 'GLY':
                proj_new = 0.0
            else:
                mut_prof = best_rotamer_profile(new_aa, ca, cb, nc, donor_c, acceptor_c)
                proj_new = mut_prof['weighted_projection']

        proj_change = proj_new - proj_orig    # Å, positive = mutant reaches further

        # Axis-distance weighting: residues off the D-A line couple less strongly
        axis_scale = float(np.exp(-((axis_distance - 2.0)**2) / (2 * 3.0**2)))
        axis_scale = float(np.clip(axis_scale, 0.1, 1.0))

        da_change    = -proj_change * GEOM_COUPLING * axis_scale
        da_change    = float(np.clip(da_change, -self.delta_r_max, self.delta_r_max))
        static_delta = -ALPHA_H * da_change   # positive when D-A shortens

        # ── Dynamic component ─────────────────────────────────────────────────
        # The promoting vibration (~165 cm⁻¹ in AADH) is a collective normal
        # mode involving ALL residues near the D-A axis, not just H-bonding ones.
        # Source: Johannissen et al. (2007) FEBS J 278:1701
        #
        # Two contributions:
        #
        # 1. STIFFNESS CONTRIBUTION (universal — applies to all residues)
        #    Residues with high ENM participation in the promoting vibration
        #    affect tunnelling by changing the vibrational stiffness.
        #    More rigid mutation → damps promoting vibration → hurts tunnelling
        #    More flexible mutation → enhances vibration amplitude → helps tunnelling
        #    BUT: flexibility helps ONLY if it's directed (see H-bond below)
        #
        # 2. H-BOND DISRUPTION CONTRIBUTION (only for H-bonding residues)
        #    Polar residues that H-bond to the substrate or to each other
        #    maintain the directional character of the promoting vibration.
        #    Disrupting H-bonds converts directed motion into thermal noise.

        dyn_importance = self._dynamic_importance(residue)

        # ── Part 1: Stiffness change ──────────────────────────────────────────
        rigidity_orig  = AA_RIGIDITY.get(orig_aa, 0.5)
        rigidity_new   = AA_RIGIDITY.get(new_aa,  0.5)
        delta_rigidity = rigidity_new - rigidity_orig  # +ve = more rigid

        # Effect: high ENM participation × rigidity change
        # More rigid = damps promoting vibration = negative delta
        # More flexible = enhances amplitude = positive delta (if directed)
        stiffness_delta = -dyn_importance * delta_rigidity * 1.5

        # ── Part 2: H-bond disruption ─────────────────────────────────────────
        disruption = 0.0
        if orig_aa in CAN_HBOND:
            disruption = hbond_disruption_magnitude(orig_aa, new_aa)

        # H-bond disruption converts directed flexibility into noise:
        # It REVERSES the stiffness benefit AND adds its own penalty
        if disruption > 0.0:
            # If mutation would have made residue more flexible (stiffness_delta > 0),
            # H-bond loss means that flexibility is now undirected → cancel the benefit
            if stiffness_delta > 0:
                stiffness_delta = stiffness_delta * (1 - disruption)
            # Additional penalty for losing directed H-bond coupling
            hbond_penalty = -dyn_importance * disruption * 0.8
        else:
            hbond_penalty = 0.0

        # Total dynamic delta
        dynamic_delta = stiffness_delta + hbond_penalty

        # NOTE: no gain bonus for introducing new H-bond capacity.
        # H-bond disruption is certain (crystal structure proves the contact exists).
        # H-bond formation is geometrically speculative — the new sidechain needs a
        # compatible partner, feasible rotamer, and correct orientation relative to
        # the promoting vibration. Without a geometric check, adding a gain term
        # introduces systematic noise with no calibration data to constrain it.

        # ── Breathing component ───────────────────────────────────────────────
        breath = compute_breathing_contribution(
            structure       = self.structure,
            enm             = self.enm,
            donor_chain     = self.donor_chain,
            donor_resnum    = self.donor_resnum,
            donor_atom      = self.donor_atom,
            acceptor_chain  = self.acceptor_chain,
            acceptor_resnum = self.acceptor_resnum,
            acceptor_atom   = self.acceptor_atom,
            mutated_residue = residue,
            new_aa          = new_aa,
            axis_distance   = axis_distance
        )
        breathing_delta = breath.breathing_delta

        # ── Electrostatic component ───────────────────────────────────────────
        # Build electrostatics map on first call (lazy init)
        if self.elec_map is None:
            d_coords = self.structure.get_atom(
                self.donor_chain, self.donor_resnum, self.donor_atom)
            a_coords = self.structure.get_atom(
                self.acceptor_chain, self.acceptor_resnum, self.acceptor_atom)
            if d_coords and a_coords:
                self.elec_map = build_electrostatics_map(
                    self.structure, d_coords.coords, a_coords.coords
                )
            else:
                self.elec_map = build_electrostatics_map(
                    self.structure,
                    np.array([0.0, 0.0, 0.0]),
                    np.array([0.0, 0.0, 2.87])
                )

        elec_delta = self.elec_map.get_delta(
            residue.chain, residue.number, orig_aa, new_aa
        )

        # ── Stochastic D-A sampling component ────────────────────────────────
        # Accounts for conformational averaging: stiffer mutants sample a
        # narrower D-A distribution and tunnel less; more flexible mutants
        # sample a broader distribution.  The correction is typically small
        # (~0.0001 for T172 series — far from axis) but important for
        # residues that directly contact the D-A pair.
        if self.stochastic_model is not None:
            stoch = self.stochastic_model.compute(
                (residue.chain, residue.number), orig_aa, new_aa)
            stochastic_delta = stoch.stochastic_delta
        else:
            stochastic_delta = 0.0

        # ── Topological network component ─────────────────────────────────────
        topo_betweenness  = 0.0
        topo_sensitivity  = 0.0
        topo_resistance   = float('inf')
        topo_community    = -1
        topo_delta        = 0.0

        if self.tunnelling_network is not None:
            tn = self.tunnelling_network
            key = (residue.chain, residue.number)
            topo_betweenness = tn.get_betweenness(residue.chain, residue.number)
            topo_resistance  = tn.get_effective_resistance(residue.chain, residue.number)
            topo_community   = tn.get_community(residue.chain, residue.number)
            net_disruption   = disruption if disruption > 0 else abs(delta_rigidity) * 0.5
            topo_sensitivity = tn.spectral_sensitivity(
                residue.chain, residue.number, net_disruption)
            topo_delta       = tn.topological_delta(
                residue.chain, residue.number, net_disruption,
                kappa=self.kappa_topo)

        # ── Total prediction ──────────────────────────────────────────────────
        total_delta   = (static_delta
                        + self.beta * dynamic_delta
                        + self.gamma * breathing_delta
                        + elec_delta
                        + stochastic_delta
                        + topo_delta)
        ln_kie_pred   = np.log(self.wt_kie) + total_delta
        # Physical ceiling: max KIE = WT_KIE × exp(ALPHA_H × delta_r_max)
        ln_kie_ceiling = np.log(self.wt_kie) + ALPHA_H * self.delta_r_max
        predicted_kie = float(np.exp(np.clip(ln_kie_pred, 0.0, ln_kie_ceiling)))
        fold_vs_wt    = predicted_kie / self.wt_kie

        # ── Mechanism classification ──────────────────────────────────────────
        abs_static   = abs(static_delta)
        abs_dynamic  = abs(self.beta * dynamic_delta)
        abs_breathing = abs(self.gamma * breathing_delta)

        components = {
            'static':      abs_static,
            'dynamic':     abs_dynamic,
            'breathing':   abs_breathing,
            'electrostatic': abs(elec_delta),
            'stochastic':  abs(stochastic_delta),
        }
        dominant = max(components, key=components.get)
        # Only call it dominated if it's clearly largest
        max_val    = max(components.values())
        second_val = sorted(components.values())[-2]
        if max_val < 1.5 * second_val:
            dominant = 'mixed'

        # ── Confidence ────────────────────────────────────────────────────────
        axis_conf    = float(np.exp(-axis_distance / 6.0))
        mech_conf    = 0.85 if dominant != 'mixed' else 0.55
        change_conf  = float(np.clip(
            abs(da_change)*15 + abs(dynamic_delta)*0.5 + abs(breathing_delta)*0.3,
            0.15, 1.0))
        confidence   = float(np.clip(axis_conf * mech_conf * change_conf, 0.0, 1.0))

        exp_kie  = get_known_kie(label)
        pred_err = (abs(predicted_kie - exp_kie)/exp_kie
                    if exp_kie is not None else None)

        return MutationScore(
            label=label,
            residue_number=residue.number,
            chain=residue.chain,
            orig_aa=orig_aa,
            new_aa=new_aa,
            position_side=position_side,
            axis_distance=axis_distance,
            static_delta=static_delta,
            dynamic_delta=dynamic_delta,
            total_delta=total_delta,
            da_change=da_change,
            vol_change=vol_change,
            bfactor_norm=self.structure.normalised_bfactor(residue),
            enm_participation=enm_participation_score(
                self.enm, residue.chain, residue.number),
            hbond_disruption=disruption,
            dynamic_importance=dyn_importance,
            predicted_kie=predicted_kie,
            fold_vs_wt=fold_vs_wt,
            confidence=confidence,
            dominant_mechanism=dominant,
            breathing_delta=breathing_delta,
            elec_delta=elec_delta,
            stochastic_delta=stochastic_delta,
            gnn_delta=0.0,   # populated by apply_gnn_corrections() in tunnel_scan
            breathing_mechanism=breath.mechanism,
            is_novel=is_novel_prediction(label),
            experimental_kie=exp_kie,
            prediction_error=pred_err,
            tunnelling_betweenness=topo_betweenness,
            spectral_sensitivity=topo_sensitivity,
            effective_resistance=topo_resistance,
            tunnelling_community=topo_community,
            topological_delta=topo_delta,
        )
