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
from calibration import AADH_KIE_DATA
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


# ── Pre-configured enzyme systems ────────────────────────────────────────────

AADH_CONFIG = ActiveSiteConfig(
    name='AADH (Alcaligenes faecalis) + tryptamine',
    pdb_id='2AGW',

    # Donor: Cβ of tryptamine (HETATM, chain A, residue 1 in 1AX3)
    # This is the carbon whose C-H bond breaks during the reaction
    donor=('D', 3001, 'CA'),

    # Acceptor: OD2 of catalytic Asp128 (chain B small subunit)
    # This is the oxygen that abstracts the proton via tunnelling
    acceptor=('D', 128, 'OD2'),

    barrier_height_kcal=13.4,
    imaginary_freq_cm1=1184.0,

    # Asp128 is the catalytic base — mutating it destroys activity entirely
    # Trp160/Trp109 form the TTQ cofactor — do not mutate
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

    all_scores:       List[MutationScore]
    double_mutant_scores: List = field(default_factory=list)

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
    barrier_height_kcal=13.4,
    imaginary_freq_cm1=1184.0,
    catalytic_residues=[('A', 161), ('A', 164)],
    scan_radius=10.0,
    wt_kie_exp=6.8,
)

def run_scan(
    pdb_path:   str,
    config:     ActiveSiteConfig,
    beta:       float = DEFAULT_BETA,
    verbose:    bool = True
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
        print(f"      Predicted KIE (WT): {wt_result.predicted_KIE:.1f}")
        print(f"      Experimental KIE:   {config.wt_kie_exp:.1f}")
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

    # ── Identify substrate H-bond partners ───────────────────────────────────
    substrate = s.get_residue(d_chain, d_resnum)
    substrate_hbond_keys = []
    if substrate:
        partners = s.substrate_hbond_partners(substrate, cutoff=3.5)
        substrate_hbond_keys = [(r.chain, r.number) for r in partners]
        if verbose:
            print(f"      Substrate H-bond partners: "
                  + ", ".join(str(s.get_residue(*k)) for k in substrate_hbond_keys[:5]))

    # ── Build scorer ─────────────────────────────────────────────────────────
    scorer = TunnelScorer(
        structure=s, enm=enm, wt_tunnelling=wt_result,
        beta=beta,
        gamma=1.0,
        substrate_hbond_residue_keys=substrate_hbond_keys,
        anisotropic_alignment_map=aniso_map,
        stochastic_model=stochastic_model,
        donor_chain    =d_chain,
        donor_resnum   =d_resnum,
        donor_atom     =d_atom,
        acceptor_chain =a_chain,
        acceptor_resnum=a_resnum,
        acceptor_atom  =a_atom,
    )

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
                      if s.is_novel and s.predicted_kie > wt_result.predicted_KIE)

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
        wt_kie_predicted=wt_result.predicted_KIE,
        wt_kie_exp=config.wt_kie_exp,
        all_scores=all_scores,
        double_mutant_scores=scan_double_mutants(
            all_scores, top_n=30,
            wt_kie=wt_result.predicted_KIE,
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
        cal_residuals = compute_gnn_residuals_from_scan(all_scores, AADH_KIE_DATA)
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
                sc.predicted_kie = float(math.exp(min(ln_kie, 8.0)))
                sc.fold_vs_wt    = sc.predicted_kie / wt_result.predicted_KIE
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
    # GATING: LOO cross-validation on the current T172 series (n=4) yields
    # LOO-R²=0.622 < 0.70 threshold, indicating overfitting with n<8.
    # GPR is held in reserve until ≥8 calibration mutations are available.
    # Run src/loo_gpr.py to re-evaluate when new experimental data arrives.
    try:
        gpr_residuals = compute_gpr_residuals_from_scan(all_scores, AADH_KIE_DATA)
        if len(gpr_residuals) >= MIN_CALIBRATION_GPR:
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
                    sc.predicted_kie  = float(_math.exp(min(ln_kie, 8.0)))
                    sc.fold_vs_wt     = sc.predicted_kie / wt_result.predicted_KIE
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
                      f"< {MIN_CALIBRATION_GPR} required "
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
            all_scores, AADH_KIE_DATA, float(np.log(wt_result.predicted_KIE)),
            verbose=verbose,
        )
        result.bayes_model = bayes_model
    except Exception as e:
        if verbose:
            print(f"  Bayesian UQ skipped: {e}")
        result.bayes_model = None

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
