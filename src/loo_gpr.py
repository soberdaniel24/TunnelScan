"""
loo_gpr.py
----------
Leave-one-out cross-validation for the Sparse GP regression module.

Trains on n-1 T172 calibration mutations, predicts the held-out mutation,
rotates through all n combinations. Reports LOO-R² and LOO-RMSE in
ln(KIE) space. Verdict: if LOO-R² < 0.70 the GPR is overfitting and
should be gated until more experimental data is available.
"""

import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from pdb_parser import Structure
from elastic_network import build_gnm
from tunnelling_model import bell_correction
from tunnel_score import TunnelScorer, DEFAULT_BETA
from stochastic_tunnelling import build_stochastic_model
from calibration import AADH_KIE_DATA
from gp_regression import SparseGP, extract_gpr_feature

PDB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', 'data', 'structures', '2AGW.pdb')
LOO_R2_THRESHOLD = 0.70


def r2(y_true, y_pred):
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return (1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def main():
    if not os.path.exists(PDB_PATH):
        print(f"ERROR: {PDB_PATH} not found")
        sys.exit(1)

    # ── Build physics pipeline ────────────────────────────────────────────────
    s   = Structure(PDB_PATH)
    enm = build_gnm(s, cutoff=7.5)

    d_chain, d_resnum, d_atom = 'D', 3001, 'CA'
    a_chain, a_resnum, a_atom = 'D', 128,  'OD2'
    da = s.get_atom(d_chain, d_resnum, d_atom) or s.get_residue(d_chain, d_resnum).ca
    aa = s.get_atom(a_chain, a_resnum, a_atom) or s.get_residue(a_chain, a_resnum).ca
    da_dist = float(np.linalg.norm(aa.coords - da.coords))

    wt = bell_correction(13.4, 1184.0, min(da_dist, 3.5),
                         experimental_KIE=55.0, use_wigner_kirkwood=True)
    stoch = build_stochastic_model(s, enm, (d_chain, d_resnum), (a_chain, a_resnum))

    scorer = TunnelScorer(
        s, enm, wt, beta=DEFAULT_BETA, gamma=1.0,
        stochastic_model=stoch,
        donor_chain=d_chain, donor_resnum=d_resnum, donor_atom=d_atom,
        acceptor_chain=a_chain, acceptor_resnum=a_resnum, acceptor_atom=a_atom,
    )

    # ── Score T172 mutations ──────────────────────────────────────────────────
    t172_res = s.get_residue('D', 172)
    cal_data = []
    for dp in AADH_KIE_DATA:
        if dp.new_aa == 'WT' or dp.residue != 172:
            continue
        sc = scorer.score_mutation(t172_res, dp.new_aa, 'acceptor', 5.13)
        residual = math.log(dp.kie_298k) - math.log(sc.predicted_kie)
        cal_data.append({
            'label':    dp.label,
            'new_aa':   dp.new_aa,
            'exp_kie':  dp.kie_298k,
            'pred_kie': sc.predicted_kie,
            'residual': residual,
            'feature':  extract_gpr_feature(sc),
        })

    n = len(cal_data)
    labels = [d['label'] for d in cal_data]
    print("=" * 60)
    print("  GPR — LEAVE-ONE-OUT CROSS-VALIDATION")
    print("=" * 60)
    print(f"  n = {n} calibration mutations: {labels}")
    print()

    # Print physics-only baseline
    print("  Physics-only residuals (before GPR):")
    for d in cal_data:
        r_str = f"{d['residual']:+.4f}"
        print(f"    {d['label']}: physics={d['pred_kie']:.2f}  "
              f"exp={d['exp_kie']:.1f}  residual={r_str}")
    print()

    # ── LOO loop ──────────────────────────────────────────────────────────────
    loo_results = []
    for i in range(n):
        held   = cal_data[i]
        train  = [cal_data[j] for j in range(n) if j != i]

        train_feats = [d['feature'] for d in train]
        train_cal   = [
            (('D', 172), d['new_aa'], d['new_aa'], d['residual'])
            for d in train
        ]

        gpr = SparseGP(optimize_hp=True)
        gpr.fit(train_cal, train_feats, verbose=False)

        result    = gpr.predict(held['feature'])
        gpr_delta = result.gpr_delta
        std       = result.std

        ln_corrected = math.log(held['pred_kie']) + gpr_delta
        kie_corrected = math.exp(min(ln_corrected, 8.0))

        residual_after = held['residual'] - gpr_delta

        loo_results.append({
            'label':         held['label'],
            'exp_kie':       held['exp_kie'],
            'physics_kie':   held['pred_kie'],
            'gpr_corrected': kie_corrected,
            'gpr_delta':     gpr_delta,
            'gpr_std':       std,
            'residual_before': held['residual'],
            'residual_after':  residual_after,
        })

        train_str = ', '.join(d['label'] for d in train)
        print(f"  Fold {i+1}: hold out {held['label']}, train on [{train_str}]")
        print(f"    GPR prediction: delta={gpr_delta:+.4f}  std={std:.4f}")
        print(f"    KIE:  physics={held['pred_kie']:.2f}  "
              f"corrected={kie_corrected:.2f}  exp={held['exp_kie']:.1f}")
        print(f"    Residual: {held['residual']:+.4f} → {residual_after:+.4f}")
        print()

    # ── Metrics ───────────────────────────────────────────────────────────────
    y_exp  = np.array([math.log(d['exp_kie'])                     for d in loo_results])
    y_phys = np.array([math.log(d['physics_kie'])                 for d in loo_results])
    y_gpr  = np.array([math.log(max(d['gpr_corrected'], 0.01))   for d in loo_results])

    loo_r2_phys  = r2(y_exp, y_phys)
    loo_r2_gpr   = r2(y_exp, y_gpr)
    loo_rmse_phys = rmse(y_exp, y_phys)
    loo_rmse_gpr  = rmse(y_exp, y_gpr)

    print("=" * 60)
    print("  LOO SUMMARY")
    print("=" * 60)
    print(f"  Physics-only:   LOO-R² = {loo_r2_phys:.4f}  "
          f"LOO-RMSE = {loo_rmse_phys:.4f} ln(KIE)")
    print(f"  Physics + GPR:  LOO-R² = {loo_r2_gpr:.4f}  "
          f"LOO-RMSE = {loo_rmse_gpr:.4f} ln(KIE)")
    print()

    gpr_improves = loo_r2_gpr > loo_r2_phys
    exceeds_threshold = loo_r2_gpr >= LOO_R2_THRESHOLD

    print(f"  LOO-R² threshold: {LOO_R2_THRESHOLD}")
    print(f"  GPR improves over physics: {gpr_improves}")

    if not exceeds_threshold:
        print()
        print(f"  *** LOO-R² = {loo_r2_gpr:.4f} < {LOO_R2_THRESHOLD} THRESHOLD ***")
        print(f"  *** GPR IS OVERFITTING — DISABLING FROM MAIN PIPELINE ***")
        print(f"  *** Reason: with n={n} calibration points the GP interpolates")
        print(f"  ***   the training set but cannot generalise reliably. ***")
        print(f"  *** Action: GPR block will be gated behind a minimum-data")
        print(f"  ***   check (n ≥ 8 calibration mutations required). ***")
        verdict = 'DISABLE'
    else:
        print()
        print(f"  LOO-R² = {loo_r2_gpr:.4f} ≥ {LOO_R2_THRESHOLD} — GPR is predictive")
        verdict = 'KEEP'

    print()
    print(f"  Verdict: {verdict}")
    print("=" * 60)
    return verdict, loo_r2_gpr, loo_r2_phys


if __name__ == '__main__':
    verdict, loo_r2_gpr, loo_r2_phys = main()
    sys.exit(0 if verdict == 'KEEP' else 1)
