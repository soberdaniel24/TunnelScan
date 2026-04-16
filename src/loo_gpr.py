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

PDB_PATH   = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '..', 'data', 'structures', '2AGW.pdb')
ANISO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '..', 'data', 'structures', '2AH1.pdb')
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
    da  = s.get_atom(d_chain, d_resnum, d_atom) or s.get_residue(d_chain, d_resnum).ca
    acc = s.get_atom(a_chain, a_resnum, a_atom) or s.get_residue(a_chain, a_resnum).ca
    da_dist = float(np.linalg.norm(acc.coords - da.coords))

    donor_coords    = da.coords
    acceptor_coords = acc.coords

    wt = bell_correction(13.4, 1184.0, min(da_dist, 3.5),
                         experimental_KIE=55.0, use_wigner_kirkwood=True)
    stoch = build_stochastic_model(s, enm, (d_chain, d_resnum), (a_chain, a_resnum))

    # Load 2AH1 anisotropic alignment map — identical path logic to run_tunnelscan.py
    aniso_map = {}
    if os.path.exists(ANISO_PATH):
        try:
            from anisotropic_bfactor import build_alignment_map
            aniso_map = build_alignment_map(ANISO_PATH, donor_coords, acceptor_coords)
            t172_score = aniso_map.get((a_chain, 172), None)
            print(f"  Anisotropic map loaded: {len(aniso_map)} residues from 2AH1"
                  + (f"  (T172 score: {t172_score:.4f})" if t172_score else ""))
        except Exception as e:
            print(f"  WARNING: aniso map failed — {e}")
    else:
        print(f"  WARNING: {ANISO_PATH} not found — dyn_importance will use B-factor+ENM fallback")

    scorer = TunnelScorer(
        s, enm, wt, beta=DEFAULT_BETA, gamma=1.0,
        anisotropic_alignment_map=aniso_map,
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

    gpr_improves_r2   = loo_r2_gpr   > loo_r2_phys
    gpr_improves_rmse = loo_rmse_gpr < loo_rmse_phys
    exceeds_threshold = loo_r2_gpr  >= LOO_R2_THRESHOLD

    print(f"  LOO-R² threshold: {LOO_R2_THRESHOLD}")
    print(f"  GPR improves LOO-R²:   {gpr_improves_r2}")
    print(f"  GPR improves LOO-RMSE: {gpr_improves_rmse}")

    # Two conditions required to activate GPR:
    #   1. LOO-R² ≥ 0.70  (GP generalises, not wildly overfitting)
    #   2. LOO-RMSE(GPR) < LOO-RMSE(physics)  (GPR actually reduces error)
    # Condition 2 guards against the case where physics is already excellent and
    # GPR only adds noise (passes R² gate but doesn't help).
    if not exceeds_threshold:
        print()
        print(f"  *** LOO-R² = {loo_r2_gpr:.4f} < {LOO_R2_THRESHOLD} THRESHOLD ***")
        print(f"  *** GPR IS OVERFITTING — keep gated ***")
        verdict = 'DISABLE'
    elif not gpr_improves_rmse:
        print()
        print(f"  *** LOO-R² = {loo_r2_gpr:.4f} ≥ {LOO_R2_THRESHOLD} (passes threshold) ***")
        print(f"  *** BUT GPR LOO-RMSE ({loo_rmse_gpr:.4f}) ≥ physics ({loo_rmse_phys:.4f}) ***")
        print(f"  *** Physics pipeline already excellent — GPR adds no improvement ***")
        print(f"  *** Keep gated until GPR demonstrates RMSE reduction ***")
        verdict = 'DISABLE'
    else:
        print()
        print(f"  LOO-R² = {loo_r2_gpr:.4f} ≥ {LOO_R2_THRESHOLD} AND RMSE reduced "
              f"({loo_rmse_phys:.4f} → {loo_rmse_gpr:.4f}) — GPR is beneficial")
        verdict = 'KEEP'

    print()
    print(f"  Verdict: {verdict}")
    print("=" * 60)
    return verdict, loo_r2_gpr, loo_r2_phys


if __name__ == '__main__':
    verdict, loo_r2_gpr, loo_r2_phys = main()
    sys.exit(0 if verdict == 'KEEP' else 1)
