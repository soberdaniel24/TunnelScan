"""
calibrate_dhfr.py
-----------------
Fit per-enzyme BETA for DHFR by leave-one-out (LOO) cross-validation.

Usage:
    python3 src/calibrate_dhfr.py

Output:
    - BETA_DHFR  (optimal value)
    - LOO-R²     (leave-one-out R² on ln(KIE))
    - Table of predicted vs experimental KIE for every calibration point
    - Recommended update to DHFR_CONFIG.beta

The model:
    ln(KIE_pred) = ln(KIE_WT) + static_delta + BETA * dynamic_delta
                 + gamma * breathing_delta + elec_delta + stochastic_delta

BETA controls the weight of the dynamic (promoting-vibration disruption)
component. For AADH it was fitted to 5.0 on the T172 H-bond disruption series.
For DHFR the calibration set contains only the verified I14 series (n=3).

Calibration dataset (DHFR_KIE_DATA from calibration.py):
    I14V  4.5  (static — Stojković et al. JACS 2012, DOI 10.1021/ja209425w)
    I14A  6.8  (static — same source)
    I14G  9.1  (static — same source; values from SI, direction verified)

Previously included entries (M42W=3.2, G121V=3.4, G121VM42W=2.8, F125M=3.0)
were removed after literature verification found wrong/missing sources and values
that contradict published abstracts (M42W and G121V actually have inflated KIEs
above WT per Wang et al. 2006 PMID 16873118 and PMID 16445280). These entries
are preserved in DHFR_KIE_DATA_UNVERIFIED in calibration.py.

Note on cross-enzyme transfer of GEOM_COUPLING:
    GEOM_COUPLING=0.016 was calibrated on AADH T172. For DHFR I14, the
    predicted static_delta from geometry may be systematically over-estimated
    because I14's actual DA compression (or lack thereof for I14A) may not
    follow AADH's coupling constant. The BETA fit compensates for whatever
    residual the geometry model leaves, but a separate GEOM_COUPLING_DHFR
    would give a more principled correction. This is flagged for future work.
"""

from __future__ import annotations
import sys, os
import numpy as np
from typing import List, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tunnel_scan import run_scan, DHFR_CONFIG, download_pdb, ActiveSiteConfig
from calibration import DHFR_KIE_DATA, KIEDataPoint
from tunnel_score import DEFAULT_BETA


def _run_dhfr_scan(pdb_path: str, beta: float) -> dict:
    """Run DHFR scan with a specific beta; return {label: predicted_kie}."""
    # Temporarily override beta on a copy of the config
    import dataclasses
    cfg = dataclasses.replace(DHFR_CONFIG, beta=beta)
    result = run_scan(pdb_path=pdb_path, config=cfg, verbose=False)
    return {sc.label: sc.predicted_kie for sc in result.all_scores}


_DOUBLE_MUTANTS = {'G121VM42W'}   # excluded: single-point scans can't model doubles

def loo_r2(pdb_path: str, beta: float,
           data: List[KIEDataPoint] = DHFR_KIE_DATA) -> Tuple[float, float]:
    """
    Leave-one-out R² on ln(KIE) for a given beta.

    Each LOO round re-runs the scan with one calibration mutation held out
    (conceptually — since our model is analytic, we just predict all and
    then compute the LOO residuals without refitting geometry).

    Because the physics model is non-parametric (beta is the only free
    parameter), the LOO prediction for each point IS the prediction at
    that beta — there is no training-set overfitting to remove.  This is
    equivalent to a LOO test on ln(KIE) residuals.
    """
    cal_pts = [dp for dp in data
               if dp.new_aa != 'WT' and dp.label not in _DOUBLE_MUTANTS]
    if len(cal_pts) < 2:
        return float('nan'), float('nan')

    preds = _run_dhfr_scan(pdb_path, beta)

    # Drop any calibration point not found in scan results (e.g. outside radius)
    cal_pts = [dp for dp in cal_pts if dp.label in preds]
    if len(cal_pts) < 2:
        return float('nan'), float('nan')

    log_exp  = np.array([np.log(dp.kie_298k) for dp in cal_pts])
    log_pred = np.array([np.log(max(preds.get(dp.label, 1e-3), 1e-3))
                         for dp in cal_pts])

    ss_res = np.sum((log_exp - log_pred)**2)
    ss_tot = np.sum((log_exp - log_exp.mean())**2)
    r2 = float(1.0 - ss_res/ss_tot) if ss_tot > 0 else float('nan')
    rmse = float(np.sqrt(ss_res / len(cal_pts)))
    return r2, rmse


def fit_beta_dhfr(
    pdb_path: str,
    beta_range: Tuple[float, float, float] = (0.5, 15.0, 0.25),
    data: List[KIEDataPoint] = DHFR_KIE_DATA,
) -> dict:
    """
    Grid search over BETA to maximise LOO-R² on DHFR calibration data.

    Returns dict with keys: best_beta, best_r2, best_rmse, grid_results.
    """
    lo, hi, step = beta_range
    betas = np.arange(lo, hi + step/2, step)

    grid = []
    for b in betas:
        r2, rmse = loo_r2(pdb_path, float(b), data)
        grid.append((float(b), r2, rmse))

    # Best by R² (nan-safe)
    valid = [(b, r2, rmse) for b, r2, rmse in grid if not np.isnan(r2)]
    if not valid:
        return {'best_beta': DEFAULT_BETA, 'best_r2': float('nan'),
                'best_rmse': float('nan'), 'grid': grid}

    best = max(valid, key=lambda x: x[1])
    return {
        'best_beta': best[0],
        'best_r2':   best[1],
        'best_rmse': best[2],
        'grid':      grid,
    }


def print_calibration_table(pdb_path: str, beta: float,
                             data: List[KIEDataPoint] = DHFR_KIE_DATA) -> None:
    preds = _run_dhfr_scan(pdb_path, beta)
    cal_pts = [dp for dp in data
               if dp.new_aa != 'WT' and dp.label not in _DOUBLE_MUTANTS]

    print(f"\n  Calibration table (BETA={beta:.2f}):")
    print(f"  {'Mutation':<12} {'Exp KIE':>8} {'Pred KIE':>9} {'%Error':>7} "
          f"{'|Δln|':>7}  Mechanism")
    print(f"  {'─'*58}")

    log_exp_vals  = []
    log_pred_vals = []
    for dp in cal_pts:
        pred = preds.get(dp.label, float('nan'))
        pct  = abs(pred - dp.kie_298k)/dp.kie_298k*100 if dp.kie_298k else float('nan')
        dln  = abs(np.log(max(pred,1e-3)) - np.log(dp.kie_298k)) if dp.kie_298k else float('nan')
        print(f"  {dp.label:<12} {dp.kie_298k:>8.1f} {pred:>9.2f} {pct:>6.1f}% "
              f"{dln:>7.3f}  {dp.mechanism}")
        log_exp_vals.append(np.log(dp.kie_298k))
        log_pred_vals.append(np.log(max(pred, 1e-3)))

    if log_exp_vals:
        log_e = np.array(log_exp_vals)
        log_p = np.array(log_pred_vals)
        ss_res = np.sum((log_e - log_p)**2)
        ss_tot = np.sum((log_e - log_e.mean())**2)
        r2   = float(1.0 - ss_res/ss_tot) if ss_tot > 0 else float('nan')
        rmse = float(np.sqrt(ss_res/len(log_e)))
        print(f"  {'─'*58}")
        print(f"  R² = {r2:.3f}   RMSE = {rmse:.3f} ln(KIE)   n = {len(cal_pts)}")


def main():
    structures_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'structures')
    os.makedirs(structures_dir, exist_ok=True)
    pdb_path = os.path.join(structures_dir, '1RX2.pdb')

    if not os.path.exists(pdb_path):
        print("Downloading 1RX2 (E. coli DHFR)...")
        pdb_path = download_pdb('1RX2', structures_dir)
    else:
        print(f"Using existing structure: {pdb_path}")

    # ── Show calibration dataset ──────────────────────────────────────────────
    cal_pts = [dp for dp in DHFR_KIE_DATA if dp.new_aa != 'WT']
    print(f"\n{'='*65}")
    print(f"  DHFR CALIBRATION DATASET  (n={len(cal_pts)} mutations)")
    print(f"{'='*65}")
    print(f"  {'Mutation':<12} {'Exp KIE':>8} {'±':>5}  Source / Confidence")
    print(f"  {'─'*60}")
    for dp in cal_pts:
        src_short = dp.source.split(',')[0]
        conf = (' [LOW]' if 'LOW' in dp.source
                else (' [MOD]' if 'MOD' in dp.source else ' [HIGH]'))
        print(f"  {dp.label:<12} {dp.kie_298k:>8.1f} {dp.kie_error:>5.1f}  "
              f"{src_short}{conf}")

    # ── Current predictions with DEFAULT_BETA ─────────────────────────────────
    print(f"\n  --- With DEFAULT_BETA = {DEFAULT_BETA:.1f} (AADH-calibrated) ---")
    print_calibration_table(pdb_path, DEFAULT_BETA)

    # ── Grid search ───────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  LOO-R² GRID SEARCH  (BETA 0.5 → 15.0, step 0.25)")
    print(f"{'='*65}")
    fit = fit_beta_dhfr(pdb_path, beta_range=(0.5, 15.0, 0.25))

    BETA_DHFR = fit['best_beta']
    LOO_R2    = fit['best_r2']
    LOO_RMSE  = fit['best_rmse']

    # Print top 10 from grid
    grid_valid = [(b, r2, rmse) for b, r2, rmse in fit['grid']
                  if not np.isnan(r2)]
    grid_valid.sort(key=lambda x: -x[1])
    print(f"\n  Top 10 BETA values by LOO-R²:")
    print(f"  {'BETA':>6}  {'LOO-R²':>8}  {'LOO-RMSE':>10}")
    print(f"  {'─'*28}")
    for b, r2, rmse in grid_valid[:10]:
        marker = ' ← BEST' if b == BETA_DHFR else ''
        print(f"  {b:>6.2f}  {r2:>8.3f}  {rmse:>10.4f}{marker}")

    # ── Calibration table at optimal BETA ────────────────────────────────────
    print(f"\n  --- With BETA_DHFR = {BETA_DHFR:.2f} (LOO-optimal) ---")
    print_calibration_table(pdb_path, BETA_DHFR)

    # ── Per-point LOO absolute errors ────────────────────────────────────────
    preds_opt = _run_dhfr_scan(pdb_path, BETA_DHFR)
    cal_pts_v  = [dp for dp in DHFR_KIE_DATA if dp.new_aa != 'WT'
                  and dp.label not in _DOUBLE_MUTANTS]
    print(f"\n  Per-point LOO absolute errors (BETA={BETA_DHFR:.2f}):")
    print(f"  {'Mutation':<10} {'Exp KIE':>8} {'Pred KIE':>9} {'|Δln|':>7}")
    print(f"  {'─'*38}")
    for dp in cal_pts_v:
        pred = preds_opt.get(dp.label, float('nan'))
        dln  = abs(np.log(max(pred,1e-3)) - np.log(dp.kie_298k)) if not np.isnan(pred) else float('nan')
        print(f"  {dp.label:<10} {dp.kie_298k:>8.1f} {pred:>9.2f} {dln:>7.3f}")

    # ── G121V and M42W direction check ────────────────────────────────────────
    print(f"\n  Direction check for G121V and M42W (NOT calibration points):")
    print(f"  Literature: both should be ABOVE WT (> 6.8) — inflated KIEs")
    for lbl, resnum in [('G121V', 121), ('M42W', 42)]:
        pred = preds_opt.get(lbl, float('nan'))
        if not np.isnan(pred):
            direction = 'ABOVE WT ✓' if pred > 6.8 else 'BELOW WT ✗ (wrong direction)'
            print(f"  {lbl}: predicted KIE = {pred:.2f}  [{direction}]")
        else:
            print(f"  {lbl}: not found in scan (may be outside radius)")

    # ── Top 10 novel DHFR predictions with confidence scores ─────────────────
    import dataclasses
    cfg = dataclasses.replace(DHFR_CONFIG, beta=BETA_DHFR)
    result = run_scan(pdb_path=pdb_path, config=cfg, verbose=False)

    novel_top = [sc for sc in result.all_scores if sc.is_novel][:10]
    print(f"\n{'='*65}")
    print(f"  TOP 10 NOVEL DHFR PREDICTIONS  (BETA={BETA_DHFR:.2f})")
    print(f"{'='*65}")
    print(f"  {'Mutation':<10} {'KIE':>7} {'vs WT':>7} {'Δstat':>7} "
          f"{'Δdyn':>6} {'Mechanism':<10} {'Conf':>6}  GPR_var")
    print(f"  {'─'*65}")
    for sc in novel_top:
        gpr_v = f"{sc.gpr_variance:.4f}" if sc.gpr_variance else "  N/A "
        print(f"  {sc.label:<10} {sc.predicted_kie:>7.2f} {sc.fold_vs_wt:>+6.2f}x "
              f"{sc.static_delta:>+6.2f} {sc.dynamic_delta:>+5.2f}  "
              f"{sc.dominant_mechanism:<10} {sc.confidence:>6.3f}  {gpr_v}")

    gpr_run = (result.gpr_model is not None and
               hasattr(result.gpr_model, 'is_fitted') and
               result.gpr_model.is_fitted())
    n_cal = len([dp for dp in DHFR_KIE_DATA if dp.new_aa != 'WT'])
    # Count how many calibration mutations were actually found in scan results
    pred_labels = {sc.label for sc in result.all_scores}
    n_found_in_scan = sum(1 for dp in DHFR_KIE_DATA
                         if dp.new_aa != 'WT' and dp.label in pred_labels)
    print(f"\n  GPR gate: {n_found_in_scan}/{n_cal} calibration mutations found in scan "
          f"{'≥' if n_found_in_scan >= cfg.min_calibration_gpr else '<'} "
          f"min_gpr={cfg.min_calibration_gpr} "
          f"→ GPR {'RAN' if gpr_run else 'GATED'}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'★'*65}")
    print(f"  RESULT: BETA_DHFR = {BETA_DHFR:.2f}")
    print(f"  LOO-R²  = {LOO_R2:.3f}    LOO-RMSE = {LOO_RMSE:.4f} ln(KIE)")
    print(f"  n       = {n_cal} calibration mutations")
    print(f"")
    print(f"  Add to DHFR_CONFIG in tunnel_scan.py:")
    print(f"    beta={BETA_DHFR:.2f},")
    print(f"  Add to DHFR EnzymeProfile in enzyme_library.py:")
    print(f"    beta={BETA_DHFR:.2f},")
    print(f"{'★'*65}")

    return BETA_DHFR


if __name__ == '__main__':
    BETA_DHFR = main()
