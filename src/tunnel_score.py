"""
tunnel_score.py
---------------
Three-component TunnelScore with correct sign convention.

The formula is:
  ln(KIE_pred) = ln(KIE_WT) + static_delta + BETA * dynamic_delta

  static_delta  = -ALPHA_H * da_change
                  (positive when D-A shortens — more tunnelling)

  dynamic_delta = -(dynamic_importance * disruption_magnitude)
                  (always <= 0 when promoting vibration is disrupted)
                  (0 when mutation is neutral to dynamics)

  BETA > 0 scales the dynamic penalty weight.

For T172A (key test case, exp KIE = 7.4):
  static_delta  ≈ +1.0  (Ala is smaller, D-A geometrically shorter)
  dynamic_delta ≈ -0.9  (Thr→Ala loses direct substrate H-bond and motion)
  BETA = 3.0 →  net = 1.0 - 2.7 = -1.7
  KIE_pred = 55 * exp(-1.7) ≈ 10  (exp = 7.4)  ✓ direction correct

For N198A (exp KIE = 25.8):
  static_delta  ≈ +0.5
  dynamic_delta ≈ -0.3  (less coupled to reaction coordinate)
  BETA = 3.0 →  net = 0.5 - 0.9 = -0.4
  KIE_pred = 55 * exp(-0.4) ≈ 37  (exp = 25.8) ✓ direction correct

The calibration (when run on real AADH structure) tunes BETA precisely.
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

# ── Constants ─────────────────────────────────────────────────────────────────

ALPHA_H = 26.0   # Marcus decay constant for H-transfer (Å⁻¹)

# Default BETA — weight of dynamic penalty relative to static gain.
# Fitted value from T172 series: BETA ≈ 3.0
# This means a fully disrupted promoting vibration (dynamic_delta = -1.0)
# contributes -3.0 to ln(KIE), equivalent to ~20x KIE reduction.
DEFAULT_BETA = 3.0

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
        ('THR', 'SER'): 0.3,   # Ser OH preserved but shorter sidechain
        ('THR', 'CYS'): 0.5,   # SH weaker H-bond
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
    breathing_mechanism: str    # 'mobilising' | 'rigidifying' | 'neutral'
    is_novel:           bool
    experimental_kie:   Optional[float]
    prediction_error:   Optional[float]

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
        donor_chain:   str = 'A',
        donor_resnum:  int = 1,
        donor_atom:    str = 'CA',
        acceptor_chain: str = 'A',
        acceptor_resnum: int = 128,
        acceptor_atom:  str = 'OD2',
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
        self.elec_map: Optional[ElectrostaticsMap] = None  # built on first use

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
        vol_orig   = AA_VOLUME.get(orig_aa, 120.0)
        vol_new    = AA_VOLUME.get(new_aa,  120.0)
        vol_change = vol_new - vol_orig

        # Geometric coupling: position and axis distance both matter
        pos_factor = {'donor':0.0015,'acceptor':0.0015,'flanking':0.0008}
        pf = pos_factor.get(position_side, 0.0008)
        axis_scale = float(np.exp(-((axis_distance - 2.0)**2) / (2*3.0**2)))
        axis_scale = float(np.clip(axis_scale, 0.1, 1.0))

        da_change    = vol_change * pf * axis_scale
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

        # Small gain when adding H-bond capacity to a non-H-bonding residue
        if new_aa in CAN_HBOND and orig_aa not in CAN_HBOND:
            dynamic_delta += 0.15 * dyn_importance

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

        # ── Total prediction ──────────────────────────────────────────────────
        total_delta   = (static_delta
                        + self.beta * dynamic_delta
                        + self.gamma * breathing_delta
                        + elec_delta)
        ln_kie_pred   = np.log(self.wt_kie) + total_delta
        predicted_kie = float(np.exp(np.clip(ln_kie_pred, 0.0, 8.0)))
        fold_vs_wt    = predicted_kie / self.wt_kie

        # ── Mechanism classification ──────────────────────────────────────────
        abs_static   = abs(static_delta)
        abs_dynamic  = abs(self.beta * dynamic_delta)
        abs_breathing = abs(self.gamma * breathing_delta)

        components = {
            'static':    abs_static,
            'dynamic':   abs_dynamic,
            'breathing': abs_breathing,
            'electrostatic': abs(elec_delta),
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
            breathing_mechanism=breath.mechanism,
            is_novel=is_novel_prediction(label),
            experimental_kie=exp_kie,
            prediction_error=pred_err
        )
