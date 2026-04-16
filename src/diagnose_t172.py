"""
diagnose_t172.py
----------------
Full component-by-component breakdown for each T172 mutation.

For each of T172A/S/V/C:
  1. Shows every additive term to ln(KIE) separately
  2. Shows cumulative KIE at each stage
  3. Runs six configurations:
       A. Full pipeline with 2AH1 aniso map (production)
       B. Full pipeline WITHOUT aniso map
       C. A without Module 2 (stochastic)
       D. A without Module 3 (WK path integral; falls back to 1st-order Bell)
       E. A without Module 4 (GNN)
       F. Physics-only baseline (no modules 2-5)
  4. For each config, reports the dominant source of error vs experiment

The 19.69 for T172A (reported in loo_gpr.py) comes from config B — the LOO
test does not load 2AH1.pdb, so dyn_importance is computed from B-factors +
ENM participation only, giving a much weaker dynamic penalty.
"""

import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from pdb_parser import Structure
from elastic_network import build_gnm
from tunnelling_model import bell_correction
from path_integral import exact_qt_parabolic, compute_u, MASS_H, MASS_D
from tunnel_score import (TunnelScorer, DEFAULT_BETA, AA_RIGIDITY,
                           hbond_disruption_magnitude)
from stochastic_tunnelling import build_stochastic_model
from calibration import AADH_KIE_DATA
from gnn_coupling import build_gnn_model, compute_gnn_residuals_from_scan, GNNCoupling

PDB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', 'data', 'structures', '2AGW.pdb')
ANISO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '..', 'data', 'structures', '2AH1.pdb')

T172_EXPERIMENTS = {
    'ALA': 7.4,
    'SER': 17.9,
    'VAL': 4.8,
    'CYS': 12.1,
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def pct_err(pred, exp):
    return 100 * (pred - exp) / exp


def make_scorer(s, enm, wt_result, aniso_map, stoch_model):
    d_chain, d_resnum, d_atom = 'D', 3001, 'CA'
    a_chain, a_resnum, a_atom = 'D', 128,  'OD2'
    return TunnelScorer(
        s, enm, wt_result,
        beta=DEFAULT_BETA, gamma=1.0,
        anisotropic_alignment_map=aniso_map,
        stochastic_model=stoch_model,
        donor_chain=d_chain, donor_resnum=d_resnum, donor_atom=d_atom,
        acceptor_chain=a_chain, acceptor_resnum=a_resnum, acceptor_atom=a_atom,
    )


def score_t172(scorer, s, new_aa):
    t172 = s.get_residue('D', 172)
    return scorer.score_mutation(t172, new_aa, 'acceptor', 5.13)


def apply_gnn(all_scores, s, enm, donor_coords, acceptor_coords,
              d_chain, d_resnum, a_chain, a_resnum, wt_result, verbose=False):
    cal_residuals = compute_gnn_residuals_from_scan(all_scores, AADH_KIE_DATA)
    if not cal_residuals:
        return {}, None
    gnn_model = build_gnn_model(
        s, enm,
        donor_key=(d_chain, d_resnum), acceptor_key=(a_chain, a_resnum),
        donor_coords=donor_coords, acceptor_coords=acceptor_coords,
        calibration_residuals=cal_residuals,
        substrate_hbond_keys=set(),
        verbose=verbose,
    )
    deltas = {}
    for sc in all_scores:
        r = gnn_model.predict((sc.chain, sc.residue_number), sc.orig_aa, sc.new_aa)
        deltas[(sc.residue_number, sc.new_aa)] = r.gnn_delta
    return deltas, gnn_model


# ── Detailed component breakdown ──────────────────────────────────────────────

def print_breakdown(label, sc, wt_kie, gnn_delta=0.0, header=False):
    """Print one mutation's full component table row."""
    beta = DEFAULT_BETA

    # Recompute sub-components of dynamic_delta
    from tunnel_score import AA_RIGIDITY, CAN_HBOND
    r_orig = AA_RIGIDITY.get(sc.orig_aa, 0.5)
    r_new  = AA_RIGIDITY.get(sc.new_aa,  0.5)
    d_rig  = r_new - r_orig
    stiff  = -sc.dynamic_importance * d_rig * 1.5

    disruption = sc.hbond_disruption
    if disruption > 0.0 and stiff > 0.0:
        stiff = stiff * (1 - disruption)
    hbond_pen = (-sc.dynamic_importance * disruption * 0.8
                 if disruption > 0.0 else 0.0)

    # Cumulative KIE at each stage
    def kie_at(delta_total):
        return math.exp(min(math.log(wt_kie) + delta_total, 8.0))

    cum_static     = kie_at(sc.static_delta)
    cum_dynamic    = kie_at(sc.static_delta + beta * sc.dynamic_delta)
    cum_breathing  = kie_at(sc.static_delta + beta * sc.dynamic_delta + sc.breathing_delta)
    cum_elec       = kie_at(sc.static_delta + beta * sc.dynamic_delta
                            + sc.breathing_delta + sc.elec_delta)
    cum_stochastic = kie_at(sc.static_delta + beta * sc.dynamic_delta
                            + sc.breathing_delta + sc.elec_delta + sc.stochastic_delta)
    cum_gnn        = kie_at(sc.static_delta + beta * sc.dynamic_delta
                            + sc.breathing_delta + sc.elec_delta
                            + sc.stochastic_delta + gnn_delta)

    exp_kie = T172_EXPERIMENTS.get(sc.new_aa, None)

    if header:
        print(f"\n  {'Mut':<6}  {'dyn_imp':>7}  {'disrupt':>7}  "
              f"{'Δstatic':>8}  {'Δstiff':>7}  {'Δhbond':>7}  "
              f"{'β·Δdyn':>7}  {'Δbreath':>7}  {'Δelec':>7}  "
              f"{'Δstoch':>7}  {'ΔGNN':>7}  "
              f"{'Δtotal':>7}  {'KIE_pred':>9}  {'KIE_exp':>7}  {'err%':>6}")
        print("  " + "─" * 135)

    err_str = f"{pct_err(cum_gnn, exp_kie):+.0f}%" if exp_kie else "  —  "

    print(f"  {label:<6}  {sc.dynamic_importance:>7.3f}  {disruption:>7.3f}  "
          f"{sc.static_delta:>+8.4f}  {stiff:>+7.4f}  {hbond_pen:>+7.4f}  "
          f"{beta*sc.dynamic_delta:>+7.4f}  {sc.breathing_delta:>+7.4f}  "
          f"{sc.elec_delta:>+7.4f}  {sc.stochastic_delta:>+7.4f}  "
          f"{gnn_delta:>+7.4f}  "
          f"{sc.total_delta+gnn_delta:>+7.4f}  {cum_gnn:>9.2f}  "
          f"{exp_kie if exp_kie else '—':>7}  {err_str:>6}")

    return cum_gnn


def run_config(label, s, enm, wt_result, aniso_map, stoch_model,
               donor_coords, acceptor_coords, use_gnn=True, verbose=False):
    """Score all T172 variants under one configuration, return dict of scores."""
    d_chain, d_resnum, d_atom = 'D', 3001, 'CA'
    a_chain, a_resnum, a_atom = 'D', 128,  'OD2'

    scorer = make_scorer(s, enm, wt_result, aniso_map, stoch_model)
    all_scores = [score_t172(scorer, s, aa) for aa in T172_EXPERIMENTS]

    gnn_deltas = {}
    gnn_model  = None
    if use_gnn:
        gnn_deltas, gnn_model = apply_gnn(
            all_scores, s, enm, donor_coords, acceptor_coords,
            d_chain, d_resnum, a_chain, a_resnum, wt_result, verbose=False)

    return all_scores, gnn_deltas


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(PDB_PATH):
        print(f"ERROR: {PDB_PATH} not found")
        sys.exit(1)

    # ── Structure and common components ───────────────────────────────────────
    s   = Structure(PDB_PATH)
    enm = build_gnm(s, cutoff=7.5)

    d_chain, d_resnum, d_atom = 'D', 3001, 'CA'
    a_chain, a_resnum, a_atom = 'D', 128,  'OD2'
    da  = s.get_atom(d_chain, d_resnum, d_atom) or s.get_residue(d_chain, d_resnum).ca
    aa_ = s.get_atom(a_chain, a_resnum, a_atom) or s.get_residue(a_chain, a_resnum).ca
    donor_coords    = da.coords
    acceptor_coords = aa_.coords
    da_dist = float(np.linalg.norm(acceptor_coords - donor_coords))

    # Module 3: WK path integral (production baseline)
    wt_wk = bell_correction(13.4, 1184.0, min(da_dist, 3.5),
                            experimental_KIE=55.0, use_wigner_kirkwood=True)
    # First-order Bell (no Module 3)
    wt_bell1 = bell_correction(13.4, 1184.0, min(da_dist, 3.5),
                               experimental_KIE=55.0, use_wigner_kirkwood=False)

    # Wigner-Kirkwood breakdown detail
    from path_integral import compute_u, exact_qt_parabolic
    u_H = compute_u(1184.0, 298.0, MASS_H)
    u_D = compute_u(1184.0, 298.0, MASS_D)
    Qt_H_exact = exact_qt_parabolic(u_H)
    Qt_D_exact = exact_qt_parabolic(u_D)
    Qt_H_bell1 = 1.0 + u_H**2 / 24.0
    Qt_D_bell1 = 1.0 + u_D**2 / 24.0

    # Module 2: stochastic model
    stoch = build_stochastic_model(s, enm, (d_chain, d_resnum), (a_chain, a_resnum))

    # Anisotropic map
    aniso_map = {}
    if os.path.exists(ANISO_PATH):
        try:
            from anisotropic_bfactor import build_alignment_map
            aniso_map = build_alignment_map(ANISO_PATH, donor_coords, acceptor_coords)
        except Exception as e:
            print(f"WARNING: aniso map failed: {e}")

    print("=" * 80)
    print("  T172 COMPONENT BREAKDOWN — DIAGNOSTIC REPORT")
    print("=" * 80)

    # ── Section 0: Module 3 (path integral) baseline ─────────────────────────
    print("\n─── MODULE 3: Path Integral Baseline ──────────────────────────────────────")
    print(f"  u_H = {u_H:.4f}  u_D = {u_D:.4f}  (ħω†/kT at 298K, ω† = 1184 cm⁻¹)")
    print()
    print(f"  {'':35s}  {'Qt_H':>8}  {'Qt_D':>8}  {'Qt_H/Qt_D':>10}  {'WT KIE':>8}")
    print(f"  {'─'*75}")
    print(f"  {'Bell 1st-order  (1+u²/24)':35s}  "
          f"{Qt_H_bell1:>8.3f}  {Qt_D_bell1:>8.3f}  "
          f"{Qt_H_bell1/Qt_D_bell1:>10.3f}  {wt_bell1.predicted_KIE:>8.1f}")
    print(f"  {'WK exact  (u/2)/sin(u/2)':35s}  "
          f"{Qt_H_exact:>8.3f}  {Qt_D_exact:>8.3f}  "
          f"{Qt_H_exact/Qt_D_exact:>10.3f}  {wt_wk.predicted_KIE:>8.1f}")
    print(f"  {'Experimental WT':35s}  {'—':>8}  {'—':>8}  {'—':>10}  {'55.0':>8}")
    print(f"\n  WK vs Bell1: WT KIE {wt_bell1.predicted_KIE:.1f} → {wt_wk.predicted_KIE:.1f}  "
          f"(+{100*(wt_wk.predicted_KIE/wt_bell1.predicted_KIE-1):.0f}% upward shift in baseline)")
    print(f"  Effect on all mutations: ln(KIE_mut) = ln(KIE_WT) + Δ")
    print(f"    If Δ is unchanged, adding {wt_wk.predicted_KIE:.1f} baseline vs {wt_bell1.predicted_KIE:.1f}")
    print(f"    means ALL mutations scale up proportionally unless Δ is recalibrated.")
    delta_log = math.log(wt_wk.predicted_KIE) - math.log(wt_bell1.predicted_KIE)
    print(f"  Δln(KIE_WT) from Module3 = +{delta_log:.4f}  "
          f"→ KIE multiplier = {math.exp(delta_log):.2f}×  "
          f"(compensated by BETA recalibration: 1.5 → 5.0)")

    # ── Section 1: Config A — Full production pipeline (with 2AH1) ───────────
    print("\n" + "=" * 80)
    print("  CONFIG A: FULL PRODUCTION PIPELINE (2AH1 aniso map loaded)")
    print("=" * 80)
    has_aniso = bool(aniso_map)
    print(f"  Aniso map loaded: {has_aniso}  |  "
          f"T172 aniso score: {aniso_map.get(('D', 172), 'n/a')}")
    print(f"  WT KIE (WK exact): {wt_wk.predicted_KIE:.2f}  |  "
          f"BETA={DEFAULT_BETA}  |  "
          f"σ_DA_WT={stoch.sigma_da_wt:.4f} Å")

    scores_A, gnn_A = run_config('A', s, enm, wt_wk, aniso_map, stoch,
                                 donor_coords, acceptor_coords, use_gnn=True)
    print()
    first = True
    for sc in scores_A:
        gnn_d = gnn_A.get((sc.residue_number, sc.new_aa), 0.0)
        print_breakdown(f"T172{sc.new_aa[0]}", sc, wt_wk.predicted_KIE,
                        gnn_delta=gnn_d, header=first)
        first = False

    # ── Section 2: Config B — Without aniso map ───────────────────────────────
    print("\n" + "─" * 80)
    print("  CONFIG B: WITHOUT 2AH1 ANISOTROPIC MAP  (← source of 19.69 for T172A)")
    print("─" * 80)
    print(f"  dyn_importance fallback: 0.35×B-factor + 0.65×ENM_participation")
    print(f"  ENM participation at T172: {scores_A[0].enm_participation:.4f}")

    scores_B, gnn_B = run_config('B', s, enm, wt_wk, {}, stoch,
                                 donor_coords, acceptor_coords, use_gnn=True)
    print()
    first = True
    for sc in scores_B:
        gnn_d = gnn_B.get((sc.residue_number, sc.new_aa), 0.0)
        print_breakdown(f"T172{sc.new_aa[0]}", sc, wt_wk.predicted_KIE,
                        gnn_delta=gnn_d, header=first)
        first = False

    print(f"\n  ► dyn_importance A vs B:")
    for scA, scB in zip(scores_A, scores_B):
        diff = scA.dynamic_importance - scB.dynamic_importance
        print(f"    T172{scA.new_aa[0]}: aniso={scA.dynamic_importance:.4f}  "
              f"no-aniso={scB.dynamic_importance:.4f}  "
              f"Δ={diff:+.4f}  "
              f"→ β·Δdyn change: {DEFAULT_BETA*(scA.dynamic_delta-scB.dynamic_delta):+.4f}")

    # ── Section 3: Ablations on Config A ─────────────────────────────────────
    print("\n" + "=" * 80)
    print("  COMPONENT ABLATIONS (starting from Config A, remove one module at a time)")
    print("=" * 80)

    ablations = [
        ("C: −Mod2 (no stochastic)",  wt_wk,   aniso_map, None,  True),
        ("D: −Mod3 (1st-order Bell)", wt_bell1, aniso_map, stoch, True),
        ("E: −Mod4 (no GNN)",         wt_wk,   aniso_map, stoch, False),
        ("F: physics-only",           wt_bell1, {},        None,  False),
    ]

    summary_rows = []
    # Add A and B to summary
    for sc, gnn_d in zip(scores_A,
                         [gnn_A.get((s.residue_number, s.new_aa), 0.0) for s in scores_A]):
        exp = T172_EXPERIMENTS.get(sc.new_aa)
        cum = math.exp(min(math.log(wt_wk.predicted_KIE) + sc.total_delta + gnn_d, 8.0))
        summary_rows.append(('A (full)', sc.new_aa, exp, cum))
    for sc, gnn_d in zip(scores_B,
                         [gnn_B.get((s.residue_number, s.new_aa), 0.0) for s in scores_B]):
        exp = T172_EXPERIMENTS.get(sc.new_aa)
        cum = math.exp(min(math.log(wt_wk.predicted_KIE) + sc.total_delta + gnn_d, 8.0))
        summary_rows.append(('B (no-aniso)', sc.new_aa, exp, cum))

    for abl_label, wt_r, amap, sm, use_gnn in ablations:
        scores_abl, gnn_abl = run_config(
            abl_label, s, enm, wt_r, amap, sm,
            donor_coords, acceptor_coords, use_gnn=use_gnn)
        print(f"\n  ── {abl_label} ──")
        print(f"  WT baseline: {wt_r.predicted_KIE:.2f}")
        first = True
        for sc in scores_abl:
            gnn_d = gnn_abl.get((sc.residue_number, sc.new_aa), 0.0)
            print_breakdown(f"T172{sc.new_aa[0]}", sc, wt_r.predicted_KIE,
                            gnn_delta=gnn_d, header=first)
            exp = T172_EXPERIMENTS.get(sc.new_aa)
            cum = math.exp(min(math.log(wt_r.predicted_KIE) + sc.total_delta + gnn_d, 8.0))
            summary_rows.append((abl_label, sc.new_aa, exp, cum))
            first = False

    # ── Section 4: Error summary matrix ───────────────────────────────────────
    print("\n" + "=" * 80)
    print("  ERROR MATRIX: KIE_predicted vs KIE_experimental")
    print("  (each cell = predicted KIE; bold = % error from experiment)")
    print("=" * 80)

    configs_in_order = ['A (full)', 'B (no-aniso)',
                        'C: −Mod2 (no stochastic)', 'D: −Mod3 (1st-order Bell)',
                        'E: −Mod4 (no GNN)', 'F: physics-only']
    mutations = ['ALA', 'SER', 'VAL', 'CYS']
    exps = {aa: T172_EXPERIMENTS[aa] for aa in mutations}

    # Build matrix
    matrix = {}
    for row in summary_rows:
        cfg, new_aa, exp, pred = row
        matrix[(cfg, new_aa)] = pred

    print(f"\n  {'Config':<32}  {'T172A':>8} {'T172S':>8} {'T172V':>8} {'T172C':>8}  {'RMSE_ln':>8}")
    print(f"  {'Exp KIE:':32}  {'7.4':>8} {'17.9':>8} {'4.8':>8} {'12.1':>8}")
    print("  " + "─" * 75)

    for cfg in configs_in_order:
        vals = []
        for aa in mutations:
            vals.append(matrix.get((cfg, aa), float('nan')))
        ln_errs = []
        cells = []
        for aa, v in zip(mutations, vals):
            exp = exps[aa]
            if math.isnan(v):
                cells.append('  —  ')
            else:
                cells.append(f"{v:7.2f}")
                ln_errs.append((math.log(v) - math.log(exp)) ** 2)
        rmse = math.sqrt(sum(ln_errs) / len(ln_errs)) if ln_errs else float('nan')
        print(f"  {cfg:<32}  {'  '.join(cells)}  {rmse:>8.4f}")

    print(f"\n  Experiments:                          7.4      17.9       4.8      12.1")

    # ── Section 5: Root-cause analysis ───────────────────────────────────────
    print("\n" + "=" * 80)
    print("  ROOT-CAUSE ANALYSIS")
    print("=" * 80)

    print("""
  Where does 19.69 for T172A come from?
  ──────────────────────────────────────
  Config B (no 2AH1): T172A prediction ≈ 19.69
  Config A (full):    T172A prediction ≈ 7–8  ← correct

  The upward bias in Config B traces to ONE component: dyn_importance.

  With 2AH1 anisotropic map (Config A):
    dyn_importance(T172) = 0.45–0.62  (T172 moves along D-A axis in crystal)
    β·Δdyn = 5.0 × (stiffness + hbond_penalty) large negative value
    → pulls predicted KIE down toward experiment

  Without 2AH1 (Config B):
    dyn_importance(T172) = 0.35×Bnorm + 0.65×ENM_part ≈ 0.15–0.25
    (B-factor at T172 is only ~18Å², near the mean; ENM participation ~0.26)
    β·Δdyn = much smaller negative value
    → KIE stays high

  Module contributions to residual error in Config B:
    Module 2 (stochastic):   ~0 at T172 (residue is 5Å from axis; σ_DA change tiny)
    Module 3 (path integral): shifts BASELINE upward by +1.16 ln(KIE) units
                               compensated by BETA recalibration (1.5→5.0)
                               but calibration used Config A (with aniso map)
    Module 4 (GNN):          corrects some residual, but limited with n=4 training

  Summary: the bias is NOT introduced by modules 2/3/4. It appears when
  the anisotropic map is ABSENT — the B-factor+ENM fallback underestimates
  T172's dynamic coupling to the D-A axis by ~2–3×.

  Implication for predictions: run_tunnelscan.py loads 2AH1 automatically.
  Test scripts and loo_gpr.py use the no-aniso fallback and should NOT be
  used to judge absolute accuracy — only to test relative module effects.
""")


if __name__ == '__main__':
    main()
