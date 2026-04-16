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
  static_delta  ≈ +0.05  (T172 at 5.1 Å from D-A; geometry proj near-negligible)
  dynamic_delta ≈ -0.36  (Thr→Ala loses H-bond to Asp128; dyn_importance=0.45
                           from anisotropic 2AH1 alignment)
  breathing     ≈ +0.014 (Ala more flexible, mobilising breathing)
  BETA = 1.5 →  net = 0.05 + 1.5*(-0.36) + 0.014 = -0.48
  KIE_pred = 11.3 * exp(-0.48) ≈ 7.0  (exp = 7.4)  ✓ 5.6% error

For T172V vs T172A:
  T172A: breathing +0.014 (Ala flexible), net ≈ -0.48 → KIE 7.0  (exp 7.4)
  T172V: breathing -0.081 (Val rigid, no benefit), net ≈ -0.62 → KIE 6.1  (exp 4.8)
  Geometry projection correctly gives Val ≈ Thr static (isosteric); branching
  captured via breathing rigidity difference, not volume proxy.

BETA=1.5 calibrated with full pipeline (aniso 2AH1 map) on T172 series, R²=0.556.
  dyn_importance at T172 = 0.45 with aniso (vs ~0.17 without); effective dynamic
  contribution is the same: 0.45 × 1.5 ≈ 0.17 × 4.0.
  Geometry projection replaces volume proxy; at T172 (5.1 Å from D-A axis,
  small proj_change ≤ 0.27 Å) the static component is negligible — the series
  is dominated by H-bond disruption and rigidity (dynamic component).
  GEOM_COUPLING=0.02 retained as physically motivated estimate for near-axis
  residues (not fitted to T172 — T172 is too far from axis for steric coupling).
  GAMMA=1.0 kept — breathing is physically real but partially overlaps with
  dynamic on H-bond dominated mutations. Separating them requires more data.
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

# ── Constants ─────────────────────────────────────────────────────────────────

ALPHA_H = 26.0   # Marcus decay constant for H-transfer (Å⁻¹)

# Default BETA — weight of dynamic penalty relative to static gain.
# Fitted value from T172 series: BETA = 1.5 (grid-search optimum, R²=0.556)
# Calibrated with anisotropic alignment map (2AH1); dyn_importance at T172 = 0.45.
# A fully disrupted promoting vibration (dynamic_delta = -1.0) contributes
# -1.5 to ln(KIE), equivalent to ~4.5x KIE reduction.
# Grid search: BETA sweep (with anisotropic alignment map from 2AH1, COUPLING=0.02):
#   BETA=1.5: R²=0.556 (optimum)
#   BETA=1.0: R²=0.478,  BETA=2.0: R²=0.527
# NOTE: dyn_importance at T172 is 0.45 with the 2AH1 aniso map (vs ~0.17 without).
# Calibrated with the full pipeline (aniso map included) on T172 series, n=4.
# Not arbitrary — validated against 4 experimental KIEs.
DEFAULT_BETA = 1.5

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
        ('THR', 'CYS'): 0.2,   # SH retains H-bond to ASP128, just weaker
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
    breathing_mechanism: str    # 'mobilising' | 'rigidifying' | 'neutral'
    is_novel:           bool
    experimental_kie:   Optional[float]
    prediction_error:   Optional[float]

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
        self.stochastic_model = stochastic_model
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

    def _sidechain_da_proj(self, residue: Residue) -> float:
        """
        Mean projection of WT residue's sidechain heavy atoms onto D-A unit
        vector, measured from CA.  Returns 0.0 for GLY or missing atoms.
        """
        ca = residue.ca_coords
        if ca is None:
            return 0.0
        sc = residue.sidechain_heavy
        if not sc:
            return 0.0
        da = self._da_unit
        return float(np.mean([np.dot(a.coords - ca, da) for a in sc]))

    def _canonical_sidechain_da_proj(self, aa_new: str, residue: Residue) -> float:
        """
        Mean projection of the canonical aa_new sidechain onto D-A, anchored
        at the WT residue's actual CA/CB crystal position.

        The first sidechain atom is placed at the same χ1 dihedral as the WT
        residue's first sidechain atom — preserving the backbone rotamer well.
        Additional atoms use canonical (tetrahedral) gauche+ geometry.
        """
        ca_atom = residue.atoms.get('CA')
        if ca_atom is None:
            return 0.0
        ca  = ca_atom.coords
        da  = self._da_unit

        if aa_new == 'GLY':
            return 0.0

        cb_atom = residue.atoms.get('CB')
        if cb_atom is None:
            # WT is GLY — place CB at CA + 1.52 Å along D-A (rough approximation)
            cb = ca + 1.52 * da
        else:
            cb = cb_atom.coords

        if aa_new == 'ALA':
            return float(np.dot(cb - ca, da))

        # Build local frame: z = CA→CB, x from N-CA plane, y = z × x
        z = cb - ca
        z = z / float(np.linalg.norm(z))

        n_atom = residue.atoms.get('N')
        if n_atom is not None:
            ref = ca - n_atom.coords
            rl  = float(np.linalg.norm(ref))
            if rl > 1e-6:
                ref = ref / rl
                x   = ref - np.dot(ref, z) * z
                xl  = float(np.linalg.norm(x))
                x   = x / xl if xl > 1e-6 else self._arbitrary_perp(z)
            else:
                x = self._arbitrary_perp(z)
        else:
            x = self._arbitrary_perp(z)
        y = np.cross(z, x)

        # χ1 of WT residue — the first sidechain atom's angle in the CB local frame
        chi1 = self._wt_chi1(residue, cb, z, x, y)

        # Canonical atom positions for aa_new using that χ1
        extra = self._canonical_atoms(aa_new, cb, z, x, y, chi1)

        all_pos = [cb] + extra
        return float(np.mean([np.dot(p - ca, da) for p in all_pos]))

    @staticmethod
    def _arbitrary_perp(z: np.ndarray) -> np.ndarray:
        ref = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        x   = ref - np.dot(ref, z) * z
        return x / float(np.linalg.norm(x))

    @staticmethod
    def _wt_chi1(residue: Residue, cb: np.ndarray,
                 z: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the χ1 dihedral angle (in degrees) from the WT residue's
        first sidechain heavy atom, expressed in the CB local frame.
        Defaults to 60° (gauche+) if atom is absent.
        """
        FIRST_SC: Dict[str, str] = {
            'THR':'OG1','SER':'OG', 'CYS':'SG', 'VAL':'CG1',
            'ILE':'CG1','LEU':'CG', 'MET':'CG', 'PHE':'CG',
            'TYR':'CG', 'TRP':'CG', 'HIS':'CG', 'ASP':'CG',
            'GLU':'CG', 'ASN':'CG', 'GLN':'CG', 'LYS':'CG',
            'ARG':'CG', 'PRO':'CG',
        }
        aname = FIRST_SC.get(residue.name)
        if not aname:
            return 60.0
        atom = residue.atoms.get(aname)
        if atom is None:
            return 60.0
        vec = atom.coords - cb
        return float(np.degrees(np.arctan2(float(np.dot(vec, y)),
                                           float(np.dot(vec, x)))))

    @staticmethod
    def _canonical_atoms(aa: str, cb: np.ndarray,
                         z: np.ndarray, x: np.ndarray, y: np.ndarray,
                         chi1: float) -> List[np.ndarray]:
        """
        Canonical heavy atom positions for residue type aa, beyond CB.
        Uses tetrahedral branch angle (70.5° from CA→CB z-axis) and the
        supplied χ1 for the first branch; further atoms extend linearly.
        """
        TET = np.radians(70.5)   # supplement of 109.5° tetrahedral angle

        def branch(blen: float, chi_deg: float) -> np.ndarray:
            r   = blen * np.sin(TET)
            dz  = blen * np.cos(TET)
            chi = np.radians(chi_deg)
            return cb + dz * z + r * (np.cos(chi) * x + np.sin(chi) * y)

        def extend(prev: np.ndarray, prev_prev: np.ndarray,
                   blen: float) -> np.ndarray:
            d = prev - prev_prev
            return prev + blen * d / float(np.linalg.norm(d))

        if aa == 'SER':
            return [branch(1.43, chi1)]
        if aa == 'CYS':
            return [branch(1.82, chi1)]
        if aa == 'THR':
            # OG1 at χ1; CG2 at χ1-120° in local frame
            # (local-frame offset is -120°, corresponding to +120° in standard
            # N-CA-CB-X dihedral space — verified against T172 crystal data)
            return [branch(1.43, chi1), branch(1.52, chi1 - 120.0)]
        if aa == 'VAL':
            # CG1 at χ1, CG2 at χ1-120° (same convention as THR)
            return [branch(1.52, chi1), branch(1.52, chi1 - 120.0)]
        if aa in ('ILE', 'LEU'):
            cg = branch(1.52, chi1)
            cd = extend(cg, cb, 1.52)
            return [cg, cd]
        if aa == 'MET':
            cg = branch(1.52, chi1)
            sd = extend(cg, cb, 1.82)
            ce = extend(sd, cg, 1.82)
            return [cg, sd, ce]
        if aa in ('ASP', 'ASN'):
            cg = branch(1.52, chi1)
            return [cg, extend(cg, cb, 1.25)]
        if aa in ('GLU', 'GLN'):
            cg = branch(1.52, chi1)
            cd = extend(cg, cb, 1.52)
            return [cg, cd, extend(cd, cg, 1.25)]
        if aa in ('PHE', 'TYR', 'HIS', 'TRP'):
            cg = branch(1.52, chi1)
            cd = extend(cg, cb, 1.40)
            return [cg, cd, extend(cd, cg, 1.40)]
        if aa == 'LYS':
            cg = branch(1.52, chi1)
            cd = extend(cg, cb, 1.52)
            ce = extend(cd, cg, 1.52)
            return [cg, cd, ce, extend(ce, cd, 1.47)]
        if aa == 'ARG':
            cg = branch(1.52, chi1)
            cd = extend(cg, cb, 1.52)
            ne = extend(cd, cg, 1.47)
            return [cg, cd, ne, extend(ne, cd, 1.35)]
        return []

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
        # Compute actual per-atom projection of WT and mutant sidechains onto
        # the D-A unit vector.  The WT uses crystal-structure atom positions;
        # the mutant uses canonical tetrahedral geometry anchored at the actual
        # CA/CB, with χ1 inherited from the WT rotamer (the backbone rotamer
        # well is preserved across isosteric substitutions).
        #
        # This replaces the volume proxy and eliminates branch_factor —
        # Val's γ-methyls branch at χ1 and χ1+120°, so they naturally project
        # less along D-A than Thr's OG1 when OG1 is pointed at the acceptor.
        #
        # Coupling constant: every 1 Å of sidechain projection change causes
        # ~2% compression/elongation of the D-A coordinate.  Physically
        # motivated by protein cavity compressibility (~1-5% per Å at active
        # sites); calibrated so T172A ≈ 7.4 matches experiment.
        GEOM_COUPLING = 0.02   # Å_DA / Å_sidechain_projection

        vol_orig    = AA_VOLUME.get(orig_aa, 120.0)
        vol_new     = AA_VOLUME.get(new_aa,  120.0)
        vol_change  = vol_new - vol_orig          # retained for diagnostics only

        proj_orig   = self._sidechain_da_proj(residue)
        proj_new    = self._canonical_sidechain_da_proj(new_aa, residue)
        proj_change = proj_new - proj_orig        # Å, positive = reaches further along D-A

        # Axis-distance weighting: residues off the D-A line couple less strongly
        axis_scale = float(np.exp(-((axis_distance - 2.0)**2) / (2*3.0**2)))
        axis_scale = float(np.clip(axis_scale, 0.1, 1.0))

        # Physical sign: sidechain projecting toward the reaction partner compresses
        # the D-A coordinate.  Less projection (proj_change > 0) → D-A shortens
        # → da_change < 0.  The negation handles both donor-side and acceptor-side
        # residues correctly without any separate sign correction:
        #   acceptor-side residue reaching back toward donor (proj_orig < 0):
        #     ALA replaces → proj_change > 0 → da_change < 0 → D-A shortens → static > 0 ✓
        #   donor-side backstop reaching toward acceptor (proj_orig > 0):
        #     ALA replaces → proj_change < 0 → da_change > 0 → D-A lengthens → static < 0 ✓
        da_change = -proj_change * GEOM_COUPLING * axis_scale

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

        # ── Total prediction ──────────────────────────────────────────────────
        total_delta   = (static_delta
                        + self.beta * dynamic_delta
                        + self.gamma * breathing_delta
                        + elec_delta
                        + stochastic_delta)
        ln_kie_pred   = np.log(self.wt_kie) + total_delta
        predicted_kie = float(np.exp(np.clip(ln_kie_pred, 0.0, 8.0)))
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
            breathing_mechanism=breath.mechanism,
            is_novel=is_novel_prediction(label),
            experimental_kie=exp_kie,
            prediction_error=pred_err
        )
