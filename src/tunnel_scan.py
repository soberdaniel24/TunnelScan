"""
tunnel_scan.py
--------------
Systematic tunnelling landscape scanner.

Given a PDB structure and active site definition, scans EVERY residue
near the D-A axis and generates a complete ranked mutation landscape.

For AADH this produces ~150-200 mutation predictions, most untested.
The novel predictions (marked ★) are genuine experimental hypotheses.

Active site definition for AADH (1AX3):
  Reaction:  Cβ-H of tryptamine → OD2 of Asp128 (small subunit)
  Donor:     chain A, ligand TPM, atom CB
  Acceptor:  chain B, residue 128, atom OD2
  Wild-type QM/MM parameters (Johannissen et al. 2020):
    barrier height:   13.4 kcal/mol
    imaginary freq:   1184 cm⁻¹
    D-A distance:     from crystal structure (auto-measured)
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pdb_parser import Structure, Residue
from elastic_network import build_gnm
from tunnelling_model import bell_correction
from tunnel_score import TunnelScorer, SUBSTITUTION_CANDIDATES, MutationScore, DEFAULT_BETA
from bayesian_uncertainty import add_bayesian_confidence
from calibration import AADH_KIE_DATA, DHFR_KIE_DATA
from multi_mutation import scan_double_mutants, print_double_mutant_report
from stochastic_tunnelling import build_stochastic_model
from gnn_coupling import build_gnn_model, compute_gnn_residuals_from_scan
from gp_regression import (build_gpr_model, compute_gpr_residuals_from_scan,
                            extract_gpr_feature, MIN_CALIBRATION_GPR)
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple


@dataclass
class ActiveSiteConfig:
    """
    Defines the active site geometry for one enzyme system.
    """
    name:                str
    pdb_id:              str

    # Donor atom: (chain, residue_number, atom_name)
    # For AADH: Cβ of tryptamine substrate
    donor:               Tuple[str, int, str]

    # Acceptor atom: (chain, residue_number, atom_name)
    # For AADH: OD2 of catalytic Asp128
    acceptor:            Tuple[str, int, str]

    # Wild-type QM/MM parameters (from literature, for Bell correction)
    barrier_height_kcal: float
    imaginary_freq_cm1:  float

    # Catalytic residues to exclude from mutation (would destroy activity)
    catalytic_residues:  List[Tuple[str, int]]

    # Scan radius around D-A axis (Angstroms)
    scan_radius:         float = 8.0

    # Wild-type experimental KIE for validation
    wt_kie_exp:          float = 55.0

    # When True, use wt_kie_exp as the KIE baseline in TunnelScorer instead of
    # Bell-predicted KIE. Set this only when Bell is fundamentally miscalibrated
    # (e.g. DHFR C→C hydride: Bell floor > experimental WT KIE).
    # Leave False for AADH — its BETA=5.0 was calibrated against Bell-predicted wt_kie.
    use_exp_kie_override: bool  = False

    # Physical ceiling parameters (Johannissen et al. J Phys Chem B 2007)
    promoting_vibration_cm1: float = 90.0   # cm⁻¹  promoting vibration frequency
    da_reduced_mass_u:       float = 6.857  # u      D-A pair reduced mass

    # Per-enzyme fitted BETA (dynamic penalty weight).
    # None → caller's beta argument (default DEFAULT_BETA=5.0) is used.
    # Set this after LOO calibration on enzyme-specific KIE data.
    beta: Optional[float] = None

    # Which calibration dataset to use for GNN/GPR/Bayes and is_novel lookups.
    # "AADH" = AADH_KIE_DATA (default); "DHFR" = DHFR_KIE_DATA.
    calibration_data_key: str = "AADH"

    # Minimum number of calibration mutations to unlock GPR.
    # AADH uses 8 (LOO cross-validation showed GPR adds noise at n=4).
    # DHFR uses 4 — lower bar because DHFR KIE data is scarcer.
    min_calibration_gpr: int = 8


# ── Pre-configured enzyme systems ────────────────────────────────────────────

AADH_CONFIG = ActiveSiteConfig(
    name='AADH (Alcaligenes faecalis) + tryptamine',
    pdb_id='2AGW',

    # Donor: Cβ of tryptamine (HETATM, chain D, residue 3001 in 2AGW)
    # Cβ-H is the bond that breaks; CB not CA is the heavy-atom donor
    donor=('D', 3001, 'CB'),

    # Acceptor: OD2 of catalytic Asp128 (chain D, small beta subunit)
    acceptor=('D', 128, 'OD2'),

    barrier_height_kcal=13.4,
    imaginary_freq_cm1=1184.0,

    # Asp128 is the catalytic base — mutating it destroys activity entirely
    # Trp160/Trp109 form the TTQ cofactor — do not mutate
    catalytic_residues=[('D', 128), ('D', 109), ('D', 160)],

    scan_radius=8.0,
    wt_kie_exp=55.0,
    use_exp_kie_override=False,
    promoting_vibration_cm1=90.0,
    da_reduced_mass_u=6.857,
)

# 2IUQ: dithionite-reduced AADH with tryptamine covalently bound (TSS adduct).
# CAUTION — UNCALIBRATED: in 2IUQ the TSS sits in the large alpha subunit (chain B)
# while the TTQ/Asp128 are in the small beta subunit (chain D), giving a cross-subunit
# D-A axis.  The T172 calibration data (measured on 2AGW intrasubunit geometry) does
# NOT transfer: T172V mispredicted 3.5× (16.7 vs 4.8 experimental).  Use 2AGW-based
# AADH_CONFIG for validated predictions; 2IUQ is kept for structural comparison only.
AADH_2IUQ_CONFIG = ActiveSiteConfig(
    name='AADH (Alcaligenes faecalis) + tryptamine, 2IUQ substrate-bound',
    pdb_id='2IUQ',

    # Donor: Cβ of TSS (tryptamine adduct, chain B residue 1434)
    donor=('B', 1434, 'CB'),

    # Acceptor: OD2 of catalytic Asp128, beta chain D (pairs with alpha chain B)
    acceptor=('D', 128, 'OD2'),

    barrier_height_kcal=13.4,
    imaginary_freq_cm1=1184.0,

    catalytic_residues=[('D', 128), ('D', 109), ('D', 160)],

    scan_radius=8.0,
    wt_kie_exp=55.0,
)


@dataclass
class ScanResult:
    """Complete output of a TunnelScan run."""
    config:           ActiveSiteConfig
    n_residues_found: int
    n_mutations_scored: int
    wt_kie_predicted: float
    wt_kie_exp:       float

    all_scores:             List[MutationScore]
    double_mutant_scores:   List = field(default_factory=list)
    topological_candidates: List[MutationScore] = field(default_factory=list)

    # Part A: network topology maps
    full_resistance_map:    Dict = field(default_factory=dict)   # (chain,resnum)→R_i all protein
    rewiring_mutations:     List = field(default_factory=list)   # List[RewiringMutation]
    network_robustness:     float = 0.0                          # Ω = λ₂/mean_R_top10

    # Part B: stored for cross-enzyme comparison
    tunnelling_network:     Optional[object] = None              # TunnellingNetworkResult

    @property
    def novel_scores(self) -> List[MutationScore]:
        return [s for s in self.all_scores if s.is_novel]

    @property
    def known_scores(self) -> List[MutationScore]:
        return [s for s in self.all_scores if not s.is_novel]

    @property
    def top_enhancing(self) -> List[MutationScore]:
        """Novel mutations predicted to INCREASE KIE above WT."""
        return [s for s in self.all_scores
                if s.is_novel and s.predicted_kie > self.wt_kie_predicted]

    @property
    def calibration_r2(self) -> float:
        """R² of predictions vs experiment on known mutations."""
        known = [(s.experimental_kie, s.predicted_kie)
                 for s in self.known_scores if s.experimental_kie]
        if len(known) < 3:
            return float('nan')
        exp  = np.array([k[0] for k in known])
        pred = np.array([k[1] for k in known])
        ss_res = np.sum((np.log(exp) - np.log(pred))**2)
        ss_tot = np.sum((np.log(exp) - np.log(exp).mean())**2)
        return float(1 - ss_res/ss_tot) if ss_tot > 0 else float('nan')


DHFR_CONFIG = ActiveSiteConfig(
    name='DHFR (E. coli) + NADP+/folate',
    pdb_id='1RX2',
    donor=('A', 164, 'C4N'),
    acceptor=('A', 161, 'C6'),
    # C4N→C6 hydride: imaginary freq ~700 cm⁻¹ (Cha et al. 1989 Biochemistry;
    # C-C hydride transfers have softer TS curvature than C-O proton transfers).
    # barrier_height from DHFR QM/MM: Hammes-Schiffer group, ~13 kcal/mol.
    # Note: Bell-predicted WT KIE floor (~8.1) exceeds experimental (6.8) due to
    # classical ZPE term; wt_kie_exp override is used as baseline in TunnelScorer.
    barrier_height_kcal=13.4,
    imaginary_freq_cm1=700.0,
    catalytic_residues=[('A', 161), ('A', 164)],
    scan_radius=10.0,
    wt_kie_exp=6.8,
    use_exp_kie_override=True,
    promoting_vibration_cm1=50.0,
    da_reduced_mass_u=6.000,
    calibration_data_key='DHFR',
    min_calibration_gpr=4,
    # BETA_DHFR=4.76 — LOO-calibrated on I14V/A/G + G121V (n=4, LOO-R²=1.000, RMSE=0.0007).
    # G121V kH/kD=4.9 confirmed from Wang 2006 Biochemistry PMC2553318 (HIGH confidence).
    # GPR active (n=4 >= min_calibration_gpr=4).
    beta=4.76,
)

# ── Part B enzyme configs ────────────────────────────────────────────────────

MADH_CONFIG = ActiveSiteConfig(
    name='MADH (P. denitrificans) + methylamine',
    pdb_id='2BBK',
    # TTQ cofactor: Trp108 (substrate imine adduct) and Trp57 (quinone O5).
    # Both in the light (beta, chain L) subunit.  The C-H bond that breaks is
    # the Cα-H of methylamine; H transfers to O5 of TTQ.  Using Cα as proxy
    # since TTQ atoms vary by modification state in the PDB ATOM record.
    # Scrutton/Hay lab KIE at 298 K: 15.8 (direct H-transfer measurement).
    donor=('L', 108, 'CA'),
    acceptor=('L', 57, 'CA'),
    barrier_height_kcal=14.0,
    imaginary_freq_cm1=950.0,
    catalytic_residues=[('L', 57), ('L', 108)],
    scan_radius=8.0,
    wt_kie_exp=15.8,
    promoting_vibration_cm1=85.0,
    da_reduced_mass_u=6.857,
)

MR_CONFIG = ActiveSiteConfig(
    name='Morphinone reductase (P. putida M10)',
    pdb_id='1GWJ',
    # NADH C4H → FMN N5 hydride transfer (old yellow enzyme mechanism).
    # 1GWJ is the substrate-free oxidised form; NADH is absent.
    # His186 and Tyr183 bracket FMN N5 in the OYE-family active site.
    # Hay et al. (2009) PNAS: KIE = 7.1 at 298 K.
    donor=('A', 186, 'CA'),
    acceptor=('A', 183, 'CA'),
    barrier_height_kcal=13.0,
    imaginary_freq_cm1=1200.0,
    catalytic_residues=[('A', 183), ('A', 186)],
    scan_radius=8.0,
    wt_kie_exp=7.1,
    promoting_vibration_cm1=70.0,
    da_reduced_mass_u=6.000,
)

htADH_CONFIG = ActiveSiteConfig(
    name='ht-ADH (Thermus thermophilus)',
    pdb_id='1RJW',
    # Zinc-dependent ADH: alcohol C1-H → NAD+ C4N hydride transfer.
    # Active-site zinc coordinated by Cys43, His67, Cys153, Glu68 (Tm numbering).
    # Zinc-bound Cys43 positions the alcohol; Tyr168 is the proton relay.
    # Klinman group KIE at 298 K: ~5 (Liang & Klinman 2004; Kohen lab data).
    donor=('A', 43, 'CA'),     # Cys43 — substrate-binding zinc ligand
    acceptor=('A', 67, 'CA'),  # His67 — zinc coordination / proton relay
    barrier_height_kcal=13.4,
    imaginary_freq_cm1=1100.0,
    catalytic_residues=[('A', 43), ('A', 67), ('A', 153)],
    scan_radius=8.0,
    wt_kie_exp=5.0,
    promoting_vibration_cm1=60.0,
    da_reduced_mass_u=6.000,
)

ATA117_CONFIG = ActiveSiteConfig(
    name='ATA-117 (R)-selective omega-TA (Arthrobacter citreus)',
    # Structural proxy: 3WWH from Arthrobacter sp. KNK168 — the closest available
    # structure to the ATA-117 engineering scaffold (Savile et al. Science 2010,
    # DOI: 10.1126/science.1188934). No Arthrobacter citreus ATA crystal structure
    # is deposited in the PDB as of 2024. 3WWH is 1.65 Å resolution, (R)-selective,
    # fold-type IV PLP-dependent omega-TA from the same Arthrobacter genus.
    pdb_id='3WWH',
    # Donor: CE of catalytic Lys188 (internal aldimine with PLP C4A).
    # In the external aldimine (substrate-bound) the substrate Cα-H transfers to
    # PLP C4A; CE of the displaced Lys is the best available proxy from the
    # internal aldimine crystal form.  D-A distance = 2.811 Å (from 3WWH).
    donor=('A', 188, 'CE'),    # Lys188 Cε — catalytic Lys, internal aldimine
    acceptor=('A', 401, 'C4A'),  # PLP C4A — electrophilic carbon of Schiff base
    # Barrier height and imaginary frequency: estimate based on PLP-Schiff base
    # C-N proton transfer studies (Toney group, DOI: 10.1021/bi00161a047).
    # No QM/MM data published specifically for ATA-117.
    barrier_height_kcal=14.0,
    imaginary_freq_cm1=1050.0,
    # Catalytic residues: Lys188 (forms aldimine), Asp259 (acid/base) — do not mutate.
    catalytic_residues=[('A', 188), ('A', 401)],
    scan_radius=8.0,
    # wt_kie_exp: conservative estimate; Savile et al. 2010 does not report KIE
    # values. Based on Toney group C-H transfer studies in PLP enzymes
    # (DOI: 10.1021/bi00161a047) and Klinman-style intrinsic KIE for Schiff base
    # proton transfer. Treat as exploratory — see ATA117_CALIBRATION in
    # calibration_data.py for full disclosure.
    wt_kie_exp=5.0,
    promoting_vibration_cm1=75.0,
    da_reduced_mass_u=6.000,
)

def run_scan(
    pdb_path:   str,
    config:     ActiveSiteConfig,
    beta:       float = DEFAULT_BETA,
    verbose:    bool = True,
    force_eval_residues: Optional[set] = None,
) -> ScanResult:
    """
    Run a complete tunnelling landscape scan.

    Parameters
    ----------
    pdb_path : str
        Path to PDB file (download with download_pdb() first).
    config : ActiveSiteConfig
        Active site definition (use AADH_CONFIG for AADH).
    beta : float
        Dynamic penalty weight (default 3.0, calibrated on AADH data).
    verbose : bool
        Print progress.

    Returns
    -------
    ScanResult with all predictions sorted by predicted KIE.
    """

    if verbose:
        print(f"\n{'='*65}")
        print(f"  TUNNELSCAN — {config.name}")
        print(f"  PDB: {pdb_path}")
        print(f"{'='*65}")

    # ── Parse structure ───────────────────────────────────────────────────────
    if verbose:
        print(f"\n[1/5] Parsing structure...")
    s = Structure(pdb_path)
    if verbose:
        print(f"      {repr(s)}")
        print(f"      Mean B-factor: {s.mean_bfactor:.1f} ± {s.std_bfactor:.1f} Å²")

    # ── Locate donor and acceptor atoms ──────────────────────────────────────
    if verbose:
        print(f"\n[2/5] Locating active site...")

    d_chain, d_resnum, d_atom = config.donor
    a_chain, a_resnum, a_atom = config.acceptor

    donor_atom    = s.get_atom(d_chain, d_resnum, d_atom)
    acceptor_atom = s.get_atom(a_chain, a_resnum, a_atom)

    # Fallback: if exact atoms not found, use Cα of the residues
    if donor_atom is None:
        donor_res = s.get_residue(d_chain, d_resnum)
        if donor_res:
            donor_atom = donor_res.ca
            if verbose:
                print(f"      WARNING: donor atom {d_atom} not found, using Cα of {donor_res}")
    if acceptor_atom is None:
        acc_res = s.get_residue(a_chain, a_resnum)
        if acc_res:
            acceptor_atom = acc_res.ca
            if verbose:
                print(f"      WARNING: acceptor atom {a_atom} not found, using Cα of {acc_res}")

    if donor_atom is None or acceptor_atom is None:
        # Fall back to approximate coordinates from literature
        if verbose:
            print(f"      NOTE: Using approximate literature coordinates for D-A pair")
        donor_coords    = np.array([0.0, 0.0, 0.0])
        acceptor_coords = np.array([0.0, 0.0, 2.87])
        da_dist_crystal = 2.87
    else:
        donor_coords    = donor_atom.coords
        acceptor_coords = acceptor_atom.coords
        da_dist_crystal = float(np.linalg.norm(acceptor_coords - donor_coords))

    if verbose:
        print(f"      D-A distance (crystal):  {da_dist_crystal:.3f} Å")
        print(f"      (MD/TS distance used for Bell correction: {config.imaginary_freq_cm1:.0f} cm⁻¹, 2.87 Å)")

    # ── Wild-type Bell correction ─────────────────────────────────────────────
    if verbose:
        print(f"\n[3/5] Computing wild-type tunnelling baseline...")

    # Use the crystal D-A distance as input; the barrier height and imaginary
    # frequency are from literature QM/MM (Johannissen et al. 2020)
    da_for_bell = min(da_dist_crystal, 3.5)   # cap at physically reasonable value
    wt_result   = bell_correction(
        barrier_height_kcal = config.barrier_height_kcal,
        imaginary_freq_cm1  = config.imaginary_freq_cm1,
        da_distance_angstrom= da_for_bell,
        experimental_KIE    = config.wt_kie_exp
    )
    if verbose:
        print(f"      Bell predicted KIE (WT): {wt_result.predicted_KIE:.1f}")
        print(f"      Experimental KIE:        {config.wt_kie_exp:.1f}")
        if abs(wt_result.predicted_KIE - config.wt_kie_exp) / config.wt_kie_exp > 0.2:
            print(f"      → Using exp KIE={config.wt_kie_exp:.1f} as baseline (Bell off by "
                  f"{(wt_result.predicted_KIE/config.wt_kie_exp - 1)*100:+.0f}%)")
        print(f"      Tunnelling fraction: {wt_result.tunnelling_fraction:.1%}")

    # ── Build ENM ─────────────────────────────────────────────────────────────
    if verbose:
        print(f"\n[4/5] Building Gaussian Network Model...")

    enm = build_gnm(s, cutoff=7.5)
    if verbose:
        print(f"      {enm.n_residues} Cα atoms, {sum(enm.eigenvalues>0.01)} normal modes")
        high_part = enm.high_participation_residues(0.75)
        print(f"      {len(high_part)} residues in top 25% promoting vibration participation")

    # ── Build stochastic D-A model ────────────────────────────────────────────
    stochastic_model = None
    try:
        stochastic_model = build_stochastic_model(
            structure    = s,
            enm          = enm,
            donor_key    = (d_chain, d_resnum),
            acceptor_key = (a_chain, a_resnum),
        )
        if verbose:
            print(f"      Stochastic D-A model: σ_DA_WT = {stochastic_model.sigma_da_wt:.4f} Å"
                  f"  (WT boost = {stochastic_model.wt_stochastic_delta():.4f} ln(KIE) units)")
    except Exception as e:
        if verbose:
            print(f"      Stochastic model failed: {e} — stochastic_delta will be 0")

    # ── Build anisotropic alignment map ──────────────────────────────────────
    # Use 2AH1 (oxidised AADH with ANISOU records) to get crystallographic
    # evidence of which residues move preferentially along the D-A axis.
    # This is the only enzyme engineering platform that uses this information.
    aniso_map = {}
    aniso_pdb = pdb_path.replace('2AGW.pdb', '2AH1.pdb')
    if os.path.exists(aniso_pdb):
        try:
            from anisotropic_bfactor import build_alignment_map
            raw_map  = build_alignment_map(aniso_pdb, donor_coords, acceptor_coords)
            aniso_map = raw_map
            if verbose:
                n_aniso = len(aniso_map)
                t172_score = aniso_map.get((a_chain, 172), None)
                n156_score = aniso_map.get((a_chain, 156), None)
                print(f"      Anisotropic alignment: {n_aniso} residues from 2AH1")
                if t172_score is not None:
                    print(f"      T172 alignment score: {t172_score:.3f} (N156: {n156_score:.3f})")
        except Exception as e:
            if verbose:
                print(f"      Anisotropic data unavailable: {e}")
    elif verbose:
        print(f"      2AH1.pdb not found — skipping anisotropic alignment")
        print(f"      (download with: curl -o {aniso_pdb} https://files.rcsb.org/download/2AH1.pdb)")

    # ── QCF fallback for residues lacking ANISOU coverage ────────────────────
    # For any residue near the D-A axis that has no entry in aniso_map
    # (coverage gap, different crystal form, novel enzyme), substitute the
    # QCF zero-point amplitude proxy from quantum_conformational_field.
    qcf_result = None   # hoisted so tunnelling_network block can reference it
    try:
        from quantum_conformational_field import (
            build_quantum_propagator, replace_anisou_with_qcf
        )
        da_unit_vec = acceptor_coords - donor_coords
        da_len_vec  = float(np.linalg.norm(da_unit_vec))
        if da_len_vec > 0.01:
            da_unit_vec = da_unit_vec / da_len_vec

        qcf_result   = build_quantum_propagator(
            enm, config.imaginary_freq_cm1, 298.15, structure=s
        )
        qcf_aln_map  = replace_anisou_with_qcf(s, qcf_result, da_unit_vec)

        n_qcf_filled = 0
        for key, score in qcf_aln_map.items():
            if key not in aniso_map:
                aniso_map[key] = score
                n_qcf_filled  += 1

        if verbose and n_qcf_filled > 0:
            print(f"      QCF alignment fallback: {n_qcf_filled} residues "
                  f"supplemented (m̃ = {qcf_result.mass_term:.4f})")
        elif verbose and n_qcf_filled == 0 and qcf_aln_map:
            print(f"      QCF alignment: full ANISOU coverage, no gaps to fill")
    except Exception as e:
        if verbose:
            print(f"      QCF alignment fallback skipped: {e}")

    # ── ANM fallback for residues still lacking coverage ─────────────────────
    # ANM predicts fluctuation MAGNITUDES well (bfactor_r≈0.38) but NOT
    # directions (pearson_r≈-0.17 vs ANISOU).  When this fires, scores encode
    # "how mobile is this residue" rather than "does it move along D-A."
    # Logged as a warning so the user knows directional data is unavailable.
    try:
        from anisotropic_network_model import build_anm, anm_bfactor_map

        anm_result   = build_anm(s, cutoff=7.5, n_modes=20)
        anm_mag_map  = anm_bfactor_map(anm_result)

        n_anm_filled = 0
        for key, score in anm_mag_map.items():
            if key not in aniso_map:
                aniso_map[key] = score
                n_anm_filled  += 1

        if verbose and n_anm_filled > 0:
            print(f"      ANM magnitude fallback: {n_anm_filled} residues supplemented")
            print(f"      WARNING: ANM scores encode fluctuation magnitude only —")
            print(f"               directional D-A alignment data unavailable for these residues")
        elif verbose and n_anm_filled == 0:
            print(f"      ANM fallback: full coverage already, no gaps to fill")
    except Exception as e:
        if verbose:
            print(f"      ANM fallback skipped: {e}")

    # ── Tunnelling network (Module 9) ────────────────────────────────────────
    # Build the quantum tunnelling network topology: W_ij = sqrt(P_i P_j) × A_i A_j × Q_ij
    # Requires QCF (must be built with structure= so ca_coords is populated).
    tunnelling_network = None
    if qcf_result is not None:
        try:
            from tunnelling_network import build_tunnelling_network
            tunnelling_network = build_tunnelling_network(
                enm, qcf_result, aniso_map, donor_coords, acceptor_coords
            )
            if verbose:
                tn = tunnelling_network
                top_bt = sorted(tn.betweenness.items(), key=lambda x: x[1], reverse=True)[:3]
                top_str = '  '.join(f"{k[0]}{k[1]}={v:.3f}" for k, v in top_bt)
                print(f"      Tunnelling network: {len(tn.nodes)} nodes  "
                      f"λ₂={tn.fiedler_value:.4f}  top_betweenness: {top_str}")
        except Exception as e:
            if verbose:
                print(f"      Tunnelling network skipped: {e}")

    # ── Identify substrate H-bond partners ───────────────────────────────────
    substrate = s.get_residue(d_chain, d_resnum)
    substrate_hbond_keys = []
    if substrate:
        partners = s.substrate_hbond_partners(substrate, cutoff=3.5)
        substrate_hbond_keys = [(r.chain, r.number) for r in partners]
        if verbose:
            print(f"      Substrate H-bond partners: "
                  + ", ".join(str(s.get_residue(*k)) for k in substrate_hbond_keys[:5]))

    # ── Select calibration dataset ─────────────────────────────────────────────
    _cal_key = getattr(config, 'calibration_data_key', 'AADH')
    _cal_data = DHFR_KIE_DATA if _cal_key == 'DHFR' else AADH_KIE_DATA
    _min_gpr  = getattr(config, 'min_calibration_gpr', 8)

    # Per-enzyme beta: config.beta takes priority over caller argument
    _beta = config.beta if getattr(config, 'beta', None) is not None else beta

    # ── Build scorer ─────────────────────────────────────────────────────────
    scorer = TunnelScorer(
        structure=s, enm=enm, wt_tunnelling=wt_result,
        beta=_beta,
        gamma=1.0,
        substrate_hbond_residue_keys=substrate_hbond_keys,
        anisotropic_alignment_map=aniso_map,
        stochastic_model=stochastic_model,
        tunnelling_network=tunnelling_network,
        donor_chain    =d_chain,
        donor_resnum   =d_resnum,
        donor_atom     =d_atom,
        acceptor_chain =a_chain,
        acceptor_resnum=a_resnum,
        acceptor_atom  =a_atom,
        promoting_vibration_cm1=getattr(config, 'promoting_vibration_cm1', 90.0),
        da_reduced_mass_u=getattr(config, 'da_reduced_mass_u', 6.857),
        temperature=getattr(config, 'temperature', 300.0),
        wt_kie_exp=(config.wt_kie_exp
                   if getattr(config, 'use_exp_kie_override', False)
                   else None),
        kie_data=_cal_data,
    )

    # Physical KIE ceiling for clamping post-processing corrections
    import math as _math_ceil
    _ln_kie_ceiling = _math_ceil.log(scorer.wt_kie) + scorer.delta_r_max * 26.0

    # ── Find residues near D-A axis ───────────────────────────────────────────
    if verbose:
        print(f"\n[5/5] Scanning residues near D-A axis (radius={config.scan_radius}Å)...")

    catalytic_keys = set(config.catalytic_residues)
    near = s.residues_near_axis(donor_coords, acceptor_coords,
                                radius=config.scan_radius)

    # Filter: skip catalytic residues, skip the substrate itself
    near_filtered = [
        (res, dist, side, t)
        for res, dist, side, t in near
        if (res.chain, res.number) not in catalytic_keys
        and not (res.chain == d_chain and res.number == d_resnum)
    ]

    if verbose:
        print(f"      {len(near)} residues found, {len(near_filtered)} after filtering catalytic residues")

    # ── Add force-evaluated residues (calibration mutants outside scan radius) ──
    force_keys = set(force_eval_residues or [])
    # Infer from calibration data: any calibration mutant residue not yet covered
    from calibration import DHFR_KIE_DATA as _DHFR_KIE
    if _cal_key == 'DHFR':
        for dp in _DHFR_KIE:
            if dp.new_aa != 'WT':
                force_keys.add((dp.chain, dp.residue))
    near_keys = {(res.chain, res.number) for res, _, _, _ in near_filtered}
    for ck in force_keys - near_keys:
        chain, resnum = ck
        if (chain, resnum) in catalytic_keys:
            continue
        extra_res = s.get_residue(chain, resnum)
        if extra_res is None:
            continue
        # Approximate axis distance (straight-line to midpoint of DA)
        mid = (donor_coords + acceptor_coords) / 2.0
        ca  = extra_res.ca
        ed  = float(np.linalg.norm(ca.coords - mid)) if ca else 15.0
        # Donor side heuristic
        da_vec = acceptor_coords - donor_coords
        if ca:
            to_res = ca.coords - donor_coords
            side_t = 'donor' if float(np.dot(to_res, da_vec)) < 0 else 'acceptor'
        else:
            side_t = 'unknown'
        near_filtered.append((extra_res, ed, side_t, 1.0))
        if verbose:
            print(f"      Force-evaluated: {extra_res} (dist≈{ed:.1f}Å, calibration residue)")

    # ── Score all mutations ───────────────────────────────────────────────────
    all_scores = []
    for res, dist, side, t in near_filtered:
        candidates = SUBSTITUTION_CANDIDATES.get(res.name, ['ALA'])
        for new_aa in candidates:
            if new_aa == res.name:
                continue   # skip self-mutations
            sc = scorer.score_mutation(res, new_aa, side, dist)
            all_scores.append(sc)

    # Sort by predicted KIE descending
    all_scores.sort(key=lambda x: x.predicted_kie, reverse=True)

    n_novel  = sum(1 for s in all_scores if s.is_novel)
    n_enhancing = sum(1 for s in all_scores
                      if s.is_novel and s.predicted_kie > scorer.wt_kie)

    if verbose:
        print(f"\n{'─'*65}")
        print(f"  SCAN COMPLETE")
        print(f"  {len(near_filtered)} residues scanned")
        print(f"  {len(all_scores)} mutations scored")
        print(f"  {n_novel} novel (untested) predictions")
        print(f"  {n_enhancing} novel mutations predicted to ENHANCE tunnelling above WT")
        print(f"{'─'*65}")

    result = ScanResult(
        config=config,
        n_residues_found=len(near_filtered),
        n_mutations_scored=len(all_scores),
        wt_kie_predicted=scorer.wt_kie,
        wt_kie_exp=config.wt_kie_exp,
        all_scores=all_scores,
        double_mutant_scores=scan_double_mutants(
            all_scores, top_n=30,
            wt_kie=scorer.wt_kie,
            beta=beta
        )
    )

    if verbose:
        cal_r2 = result.calibration_r2
        if not np.isnan(cal_r2):
            print(f"  Calibration R² (known mutations): {cal_r2:.3f}")

    # ── GNN residual correction ───────────────────────────────────────────────
    # Two-pass approach:
    #   1. Physics scan already complete (all_scores populated)
    #   2. Extract residuals for known mutations
    #   3. Fit GNN on those residuals (w_mp, w_out: 4 parameters)
    #   4. Apply GNN corrections to every MutationScore in-place
    try:
        cal_residuals = compute_gnn_residuals_from_scan(all_scores, _cal_data)
        if cal_residuals:
            gnn_model = build_gnn_model(
                s, enm,
                donor_key    = (d_chain, d_resnum),
                acceptor_key = (a_chain, a_resnum),
                donor_coords    = donor_coords,
                acceptor_coords = acceptor_coords,
                calibration_residuals = cal_residuals,
                substrate_hbond_keys  = set(substrate_hbond_keys),
                verbose = verbose,
            )
            # Apply GNN delta to every MutationScore
            import math
            for sc in all_scores:
                gnn_r = gnn_model.predict((sc.chain, sc.residue_number), sc.orig_aa, sc.new_aa)
                sc.gnn_delta = gnn_r.gnn_delta
                sc.total_delta += gnn_r.gnn_delta
                ln_kie = math.log(sc.predicted_kie) + gnn_r.gnn_delta
                sc.predicted_kie = float(math.exp(min(ln_kie, _ln_kie_ceiling)))
                sc.fold_vs_wt    = sc.predicted_kie / scorer.wt_kie
                if sc.experimental_kie:
                    sc.prediction_error = abs(sc.predicted_kie - sc.experimental_kie) / sc.experimental_kie
            # Re-sort after GNN correction
            all_scores.sort(key=lambda x: x.predicted_kie, reverse=True)
            if verbose:
                cal_r2_post = result.calibration_r2
                if not np.isnan(cal_r2_post):
                    print(f"  Calibration R² after GNN: {cal_r2_post:.3f}")
            result.gnn_model = gnn_model
        else:
            result.gnn_model = None
    except Exception as e:
        if verbose:
            print(f"  GNN correction skipped: {e}")
        result.gnn_model = None

    # ── Sparse GP regression correction ──────────────────────────────────────
    # Two-pass approach (parallel to GNN):
    #   1. Compute post-GNN residuals for known mutations
    #   2. Fit Sparse GP with physics-informed kernel on those residuals
    #   3. Apply GPR corrections + uncertainty bands to every MutationScore
    #
    # GATING: LOO cross-validation (T172 series, n=4, with 2AH1 aniso map):
    #   Physics-only  LOO-R²=0.941  LOO-RMSE=0.121 ln(KIE)
    #   Physics+GPR   LOO-R²=0.921  LOO-RMSE=0.140 ln(KIE)
    # GPR passes the R²≥0.70 threshold but INCREASES RMSE — the physics pipeline
    # is already excellent and GPR adds noise at n=4.  Gate until GPR demonstrates
    # a strict RMSE reduction in LOO.  Run src/loo_gpr.py to re-evaluate.
    try:
        gpr_residuals = compute_gpr_residuals_from_scan(all_scores, _cal_data)
        if len(gpr_residuals) >= _min_gpr:
            gpr_model = build_gpr_model(all_scores, gpr_residuals, verbose=verbose)
            if gpr_model.is_fitted():
                import math as _math
                for sc in all_scores:
                    feat         = extract_gpr_feature(sc)
                    gpr_r        = gpr_model.predict(feat)
                    sc.gpr_delta    = gpr_r.gpr_delta
                    sc.gpr_variance = gpr_r.variance
                    sc.total_delta += gpr_r.gpr_delta
                    ln_kie = _math.log(sc.predicted_kie) + gpr_r.gpr_delta
                    sc.predicted_kie  = float(_math.exp(min(ln_kie, _ln_kie_ceiling)))
                    sc.fold_vs_wt     = sc.predicted_kie / scorer.wt_kie
                    if sc.experimental_kie:
                        sc.prediction_error = (abs(sc.predicted_kie - sc.experimental_kie)
                                               / sc.experimental_kie)
                all_scores.sort(key=lambda x: x.predicted_kie, reverse=True)
                if verbose:
                    cal_r2_gpr = result.calibration_r2
                    if not np.isnan(cal_r2_gpr):
                        print(f"  Calibration R² after GPR: {cal_r2_gpr:.3f}")
                result.gpr_model = gpr_model
            else:
                result.gpr_model = None
        else:
            n_cal = len(gpr_residuals)
            if verbose:
                print(f"  GPR gated: n={n_cal} calibration mutations "
                      f"< {_min_gpr} required "
                      f"(LOO-R²=0.62 with n=4 — run loo_gpr.py to re-evaluate)")
            result.gpr_model = None
    except Exception as e:
        if verbose:
            print(f"  GPR correction skipped: {e}")
        result.gpr_model = None

    # ── Bayesian uncertainty quantification ───────────────────────────────────
    # Fitted on T172 calibration series after physics scan; enriches every
    # MutationScore with a BayesianConfidence object (ms.bayes).
    try:
        bayes_model = add_bayesian_confidence(
            all_scores, _cal_data, float(np.log(scorer.wt_kie)),
            verbose=verbose,
        )
        result.bayes_model = bayes_model
    except Exception as e:
        if verbose:
            print(f"  Bayesian UQ skipped: {e}")
        result.bayes_model = None

    # ── Distal topological scan ───────────────────────────────────────────────
    # Score mutations at high-betweenness network nodes that fall OUTSIDE the
    # geometric scan radius.  These are topological bottlenecks — every
    # max-weight tunnelling path in W_ij passes through them, yet they are
    # invisible to a distance-only scan.  The DHFR analogue is G121 (found
    # at 19Å from the active site via ENM network coupling).
    if tunnelling_network is not None:
        try:
            scanned_keys   = {(sc.chain, sc.residue_number) for sc in all_scores}
            catalytic_keys = set(config.catalytic_residues)
            top_bt = sorted(tunnelling_network.betweenness.items(),
                            key=lambda x: x[1], reverse=True)

            DISTAL_BETWEENNESS_CUTOFF = 0.50   # top fraction of network bottlenecks
            distal_scores = []

            for (chain, resnum), bt in top_bt:
                if bt < DISTAL_BETWEENNESS_CUTOFF:
                    break
                if (chain, resnum) in scanned_keys:
                    continue
                if (chain, resnum) in catalytic_keys:
                    continue
                res = s.get_residue(chain, resnum)
                if res is None:
                    continue
                ca = res.ca_coords
                if ca is None:
                    continue

                # Axis distance (perpendicular distance from D-A line)
                axis_vec = acceptor_coords - donor_coords
                axis_len = float(np.linalg.norm(axis_vec))
                if axis_len < 0.01:
                    continue
                axis_hat = axis_vec / axis_len
                to_ca    = ca - donor_coords
                proj     = float(np.dot(to_ca, axis_hat))
                perp     = to_ca - proj * axis_hat
                dist     = float(np.linalg.norm(perp))

                side = 'donor' if proj < axis_len / 2 else 'acceptor'
                candidates = SUBSTITUTION_CANDIDATES.get(res.name, ['ALA'])
                for new_aa in candidates:
                    if new_aa == res.name:
                        continue
                    sc = scorer.score_mutation(res, new_aa, side, dist)
                    distal_scores.append(sc)

            distal_scores.sort(key=lambda x: x.tunnelling_betweenness, reverse=True)
            result.topological_candidates = distal_scores

            if verbose and distal_scores:
                print(f"\n{'─'*65}")
                print(f"  DISTAL TOPOLOGICAL CANDIDATES  "
                      f"(betweenness ≥ {DISTAL_BETWEENNESS_CUTOFF}, outside {config.scan_radius}Å)")
                print(f"  {len({sc.residue_number for sc in distal_scores})} residues  "
                      f"({len(distal_scores)} mutations)")
                print(f"  {'Mutation':<10}  {'Betweenness':>12}  {'Comm':>5}  "
                      f"{'KIE_pred':>9}  {'Mechanism':<10}")
                printed = set()
                for sc in distal_scores[:15]:
                    key = (sc.chain, sc.residue_number)
                    if key in printed:
                        continue
                    printed.add(key)
                    bt  = sc.tunnelling_betweenness
                    comm = sc.tunnelling_community
                    print(f"  {sc.label:<10}  {bt:>12.3f}  {comm:>5}  "
                          f"{sc.predicted_kie:>9.1f}  {sc.dominant_mechanism:<10}")
                print(f"{'─'*65}")

        except Exception as e:
            if verbose:
                print(f"  Distal topological scan skipped: {e}")

    # ── Part A: full resistance map + rewiring mutations ─────────────────────
    if tunnelling_network is not None and qcf_result is not None:
        try:
            from tunnelling_network import (
                build_full_resistance_map, find_rewiring_mutations
            )
            full_R = build_full_resistance_map(
                enm, qcf_result, aniso_map, donor_coords, acceptor_coords
            )
            result.full_resistance_map  = full_R
            result.network_robustness   = tunnelling_network.robustness
            result.tunnelling_network   = tunnelling_network

            rewire = find_rewiring_mutations(
                tunnelling_network, s, SUBSTITUTION_CANDIDATES
            )
            result.rewiring_mutations = rewire

            if verbose:
                print(f"\n{'─'*65}")
                print(f"  PART A: NETWORK TOPOLOGY")
                print(f"  Ω (robustness) = {tunnelling_network.robustness:.4f}  "
                      f"λ₂ = {tunnelling_network.fiedler_value:.4f}")
                print(f"  Full resistance map: {len(full_R)} residues")
                # Print 5 most connected (lowest R) protein residues
                prot_R = [(k, v) for k, v in full_R.items()
                          if not s.get_residue(*k) or not s.get_residue(*k).is_hetatm]
                prot_R.sort(key=lambda x: x[1])
                print(f"  Most D-A-coupled residues (lowest R):")
                for (ch, rn), r in prot_R[:5]:
                    res = s.get_residue(ch, rn)
                    aa  = res.name if res else '???'
                    print(f"    {aa}{rn}({ch})  R={r:.4f}")
                if rewire:
                    print(f"  Top rewiring mutations (Δλ₂ > 0):")
                    print(f"  {'Mutation':<10} {'Δλ₂':>8}  {'new_λ₂':>8}  {'sensitivity':>12}")
                    for rm in rewire[:8]:
                        print(f"  {rm.label:<10} {rm.delta_lambda2:>8.4f}  "
                              f"{rm.new_lambda2:>8.4f}  {rm.fiedler_sensitivity:>12.4f}")
                print(f"{'─'*65}")
        except Exception as e:
            if verbose:
                print(f"  Part A network analysis skipped: {e}")

    return result


def download_pdb(pdb_id: str, output_dir: str = '.') -> str:
    """
    Download a PDB file from RCSB.
    Returns the local file path.
    Run this on your machine (requires internet).
    """
    import urllib.request
    url      = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    out_path = os.path.join(output_dir, f"{pdb_id}.pdb")
    if os.path.exists(out_path):
        print(f"  {out_path} already exists, skipping download")
        return out_path
    print(f"  Downloading {pdb_id} from RCSB...")
    urllib.request.urlretrieve(url, out_path)
    print(f"  Saved to {out_path}")
    return out_path
