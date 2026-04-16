"""
test_gp_regression.py
---------------------
Unit tests for gp_regression.py (Module 5).

Tests:
  1. GPRFeature extraction produces valid values from 2AGW scan results
  2. Kernel is symmetric and strictly positive on the diagonal
  3. Covariance matrix is positive semi-definite (all eigenvalues ≥ 0)
  4. Inducing point selection: K-means finds M ≤ N points covering input space
  5. DTC log marginal likelihood is finite and higher with sensible hyps
  6. fit() converges without error on T172 calibration residuals
  7. After fitting: GPR posterior mean reduces residual SS vs zero-correction
  8. Posterior variance is non-negative for all T172 variants
  9. Missing feature (unseen mutation type) returns finite, reasonable prediction
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np, math

from pdb_parser import Structure
from elastic_network import build_gnm
from tunnelling_model import bell_correction
from tunnel_score import TunnelScorer, DEFAULT_BETA, SUBSTITUTION_CANDIDATES
from stochastic_tunnelling import build_stochastic_model
from calibration import AADH_KIE_DATA
from gp_regression import (
    GPRFeature, GPRResult, SparseGP,
    compute_kernel, build_covariance_matrix,
    extract_gpr_feature, compute_gpr_residuals_from_scan, build_gpr_model,
    N_FEATURES, MAX_INDUCING, MECHANISM_IDS,
    DEFAULT_SIGMA_SIGNAL, DEFAULT_SIGMA_NOISE,
    DEFAULT_LENGTH_RESIDUE, DEFAULT_LENGTH_MAGNITUDE,
)

PDB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', 'data', 'structures', '2AGW.pdb')

PASS = "\033[92m  PASS\033[0m"
FAIL = "\033[91m  FAIL\033[0m"

def check(cond, msg):
    print(f"  {PASS if cond else FAIL}  {msg}")
    return cond


def build_scan_scores(s, enm, wt_result, stoch_model):
    """Run a minimal scan to get MutationScore objects for testing."""
    scorer = TunnelScorer(
        s, enm, wt_result, beta=DEFAULT_BETA, gamma=1.0,
        stochastic_model=stoch_model,
        donor_chain='D',    donor_resnum=3001, donor_atom='CA',
        acceptor_chain='D', acceptor_resnum=128, acceptor_atom='OD2',
    )
    donor_atom    = s.get_atom('D', 3001, 'CA') or s.get_residue('D', 3001).ca
    acceptor_atom = s.get_atom('D', 128, 'OD2') or s.get_residue('D', 128).ca
    donor_coords    = donor_atom.coords
    acceptor_coords = acceptor_atom.coords
    near = s.residues_near_axis(donor_coords, acceptor_coords, radius=8.0)
    catalytic = {('D', 128), ('D', 109), ('D', 160)}
    scores = []
    for res, dist, side, _ in near:
        if (res.chain, res.number) in catalytic:
            continue
        if res.chain == 'D' and res.number == 3001:
            continue
        for new_aa in SUBSTITUTION_CANDIDATES.get(res.name, ['ALA'])[:2]:
            if new_aa == res.name:
                continue
            sc = scorer.score_mutation(res, new_aa, side, dist)
            scores.append(sc)
    return scores


def main():
    if not os.path.exists(PDB_PATH):
        print(f"ERROR: {PDB_PATH} not found")
        sys.exit(1)

    print("=" * 60)
    print("  SPARSE GP REGRESSION MODULE — UNIT TESTS")
    print("=" * 60)

    # ── Setup ─────────────────────────────────────────────────────────────────
    print("\n[Setup] Parsing structure and building physics pipeline...")
    s   = Structure(PDB_PATH)
    enm = build_gnm(s, cutoff=7.5)
    print(f"  ENM: {enm.n_residues} residues")

    d_chain, d_resnum, d_atom = 'D', 3001, 'CA'
    a_chain, a_resnum, a_atom = 'D', 128,  'OD2'
    donor_atom    = s.get_atom(d_chain, d_resnum, d_atom) or s.get_residue(d_chain, d_resnum).ca
    acceptor_atom = s.get_atom(a_chain, a_resnum, a_atom) or s.get_residue(a_chain, a_resnum).ca
    donor_coords    = donor_atom.coords
    acceptor_coords = acceptor_atom.coords
    da_dist = float(np.linalg.norm(acceptor_coords - donor_coords))

    wt_result = bell_correction(13.4, 1184.0, min(da_dist, 3.5),
                                experimental_KIE=55.0, use_wigner_kirkwood=True)
    stoch = build_stochastic_model(s, enm, (d_chain, d_resnum), (a_chain, a_resnum))

    print("  Running minimal scan (T172 neighbourhood)...")
    all_scores = build_scan_scores(s, enm, wt_result, stoch)
    print(f"  {len(all_scores)} mutations scored")

    t172_scores = [sc for sc in all_scores if sc.residue_number == 172]
    if not t172_scores:
        print("ERROR: T172 not found in scan results")
        sys.exit(1)

    # ── Test 1: Feature extraction ────────────────────────────────────────────
    print("\n[Test 1] GPRFeature extraction from MutationScore")
    feats = [extract_gpr_feature(sc) for sc in t172_scores]
    for f, sc in zip(feats, t172_scores):
        check(f.axis_distance >= 0.0 and f.axis_distance <= 20.0,
              f"  {sc.label}: axis_distance={f.axis_distance:.2f} Å in [0,20]")
        check(0.0 <= f.enm_participation <= 1.0 + 1e-6,
              f"  {sc.label}: enm_participation={f.enm_participation:.4f} in [0,1]")
        check(f.mechanism_id in (0, 1, 2, 3),
              f"  {sc.label}: mechanism_id={f.mechanism_id} valid")
        check(0.0 <= f.hbond_disruption <= 1.0 + 1e-6,
              f"  {sc.label}: hbond_disruption={f.hbond_disruption:.4f} in [0,1]")
    check(len(feats) == len(t172_scores), f"Feature count matches score count ({len(feats)})")

    # ── Test 2: Kernel symmetry and positivity ────────────────────────────────
    print("\n[Test 2] Kernel is symmetric and positive on the diagonal")
    f0, f1 = feats[0], feats[-1]
    k00 = compute_kernel(f0, f0)
    k11 = compute_kernel(f1, f1)
    k01 = compute_kernel(f0, f1)
    k10 = compute_kernel(f1, f0)
    check(k00 > 0.0 and k11 > 0.0, f"Diagonal entries positive: k(f0,f0)={k00:.4f}, k(f1,f1)={k11:.4f}")
    check(abs(k01 - k10) < 1e-10,  f"Kernel is symmetric: k(f0,f1)={k01:.6f} == k(f1,f0)={k10:.6f}")
    check(k00 <= compute_kernel(f0, f0, sigma_signal=2.0),
          "Larger σ_s → larger kernel value")
    # Cauchy-Schwarz: k(x,y)² ≤ k(x,x) × k(y,y)
    cs = k01 ** 2 <= k00 * k11 + 1e-10
    check(cs, f"Cauchy-Schwarz: k(f0,f1)²={k01**2:.6f} ≤ k(f0,f0)×k(f1,f1)={k00*k11:.6f}")

    # ── Test 3: Covariance matrix is PSD ─────────────────────────────────────
    print("\n[Test 3] Covariance matrix is positive semi-definite")
    all_feats = [extract_gpr_feature(sc) for sc in all_scores[:20]]
    K = build_covariance_matrix(all_feats)
    check(K.shape == (len(all_feats), len(all_feats)), f"K shape: {K.shape}")
    check(np.allclose(K, K.T, atol=1e-10), "K is symmetric")
    evals = np.linalg.eigvalsh(K)
    min_ev = float(evals.min())
    check(min_ev >= -1e-8, f"All eigenvalues ≥ 0 (min={min_ev:.2e})")
    check(float(evals.max()) > 0, f"At least one positive eigenvalue (max={float(evals.max()):.4f})")

    # ── Test 4: Inducing point selection ─────────────────────────────────────
    print("\n[Test 4] Inducing point selection via K-means")
    all_feats_full = [extract_gpr_feature(sc) for sc in all_scores]
    gpr_test = SparseGP(max_inducing=10)
    inducing = gpr_test._select_inducing(all_feats_full)
    M = len(inducing)
    check(M <= min(10, len(all_feats_full)),
          f"M={M} ≤ min(max_inducing=10, N={len(all_feats_full)})")
    check(M >= 1, f"At least one inducing point selected")
    # All inducing points should be valid GPRFeature objects
    check(all(isinstance(z, GPRFeature) for z in inducing),
          "All inducing points are GPRFeature instances")
    # With N < max_inducing, should return exact GP
    small_feats = all_feats_full[:5]
    gpr_exact = SparseGP(max_inducing=20)
    ind_exact  = gpr_exact._select_inducing(small_feats)
    check(len(ind_exact) == 5,
          f"Exact GP when N({len(small_feats)}) ≤ max_inducing(20): M={len(ind_exact)}")

    # ── Test 5: Log marginal likelihood is finite ─────────────────────────────
    print("\n[Test 5] DTC log marginal likelihood is finite")
    cal_feats = [extract_gpr_feature(sc)
                 for sc in t172_scores
                 if sc.experimental_kie is not None]
    cal_y     = np.array([math.log(sc.experimental_kie) - math.log(sc.predicted_kie)
                          for sc in t172_scores
                          if sc.experimental_kie is not None])
    if len(cal_y) >= 2:
        gpr_lml = SparseGP()
        ind_lml = gpr_lml._select_inducing(cal_feats)
        lml_default = gpr_lml._dtc_lml(
            cal_feats, ind_lml, cal_y,
            DEFAULT_SIGMA_SIGNAL, 0.3, 3.0, 0.5)
        lml_bad = gpr_lml._dtc_lml(
            cal_feats, ind_lml, cal_y,
            0.001, 10.0, 100.0, 100.0)  # absurd hyperparameters
        check(np.isfinite(lml_default), f"LML (default hyps) = {lml_default:.4f} is finite")
        check(lml_default > lml_bad,
              f"Default hyps ({lml_default:.2f}) beat absurd hyps ({lml_bad:.2f})")
    else:
        print("  Skipped (insufficient calibration data)")

    # ── Test 6: fit() on T172 calibration residuals ───────────────────────────
    print("\n[Test 6] fit() converges on T172 calibration residuals")
    # Build calibration residuals from raw (pre-GPR) physics scores
    cal_residuals_raw = []
    for dp in AADH_KIE_DATA:
        if dp.new_aa == 'WT':
            continue
        for sc in all_scores:
            if sc.residue_number == dp.residue and sc.new_aa == dp.new_aa:
                residual = math.log(dp.kie_298k) - math.log(sc.predicted_kie)
                cal_residuals_raw.append(
                    ((sc.chain, sc.residue_number), sc.orig_aa, sc.new_aa, residual))
                print(f"  {dp.label}: predicted={sc.predicted_kie:.2f}  "
                      f"exp={dp.kie_298k:.1f}  residual={residual:.4f}")
                break

    if len(cal_residuals_raw) >= 2:
        try:
            gpr = build_gpr_model(all_scores, cal_residuals_raw, verbose=True)
            check(True, "build_gpr_model() completed without error")
            check(gpr.is_fitted(), "SparseGP is fitted")
        except Exception as e:
            check(False, f"build_gpr_model() raised: {e}")
            gpr = None
    else:
        print("  Skipped (insufficient data)")
        gpr = None

    # ── Test 7: GPR reduces residual magnitude ────────────────────────────────
    print("\n[Test 7] GPR predictions reduce residual SS vs zero-correction")
    if gpr is not None and gpr.is_fitted():
        before_ss = sum(r ** 2 for _, _, _, r in cal_residuals_raw)
        after_ss  = 0.0
        for (key, orig_aa, new_aa, residual) in cal_residuals_raw:
            chain, resnum = key
            sc = next((s for s in all_scores
                       if s.residue_number == resnum and s.new_aa == new_aa), None)
            if sc is None:
                continue
            feat   = extract_gpr_feature(sc)
            result = gpr.predict(feat)
            after_ss += (residual - result.gpr_delta) ** 2
        check(after_ss <= before_ss + 1e-8,
              f"SS_residuals: {before_ss:.5f} → {after_ss:.5f} "
              f"(GPR reduces or leaves unchanged)")
        print(f"  Model: l_r={gpr._model.length_residue:.3f}  "
              f"l_m={gpr._model.length_magnitude:.3f}  "
              f"σ_s={gpr._model.sigma_signal:.3f}  "
              f"σ_n={gpr._model.sigma_noise:.3f}")
    else:
        print("  Skipped (GPR not fitted)")

    # ── Test 8: Posterior variance non-negative ───────────────────────────────
    print("\n[Test 8] Posterior variance ≥ 0 for all T172 variants")
    if gpr is not None and gpr.is_fitted():
        for sc in t172_scores:
            feat   = extract_gpr_feature(sc)
            result = gpr.predict(feat)
            check(result.variance >= -1e-8,
                  f"  {sc.label}: variance={result.variance:.6f} ≥ 0")
            check(np.isfinite(result.mean),
                  f"  {sc.label}: mean={result.mean:.6f} is finite")
            print(f"    {sc.label}: gpr_delta={result.gpr_delta:+.5f}  "
                  f"std={result.std:.5f}")
    else:
        print("  Skipped (GPR not fitted)")

    # ── Test 9: Novel / unseen mutation prediction ─────────────────────────────
    print("\n[Test 9] Novel mutation prediction is finite and bounded")
    if gpr is not None and gpr.is_fitted():
        novel_feat = GPRFeature(
            axis_distance=5.0, enm_participation=0.3,
            mechanism_id=1, rigidity_change=-0.2,
            vol_change_norm=-0.3, hbond_disruption=0.8,
            dynamic_importance=0.4,
        )
        result = gpr.predict(novel_feat)
        check(np.isfinite(result.mean),     f"Novel mean={result.mean:.5f} is finite")
        check(result.variance >= 0.0,       f"Novel variance={result.variance:.5f} ≥ 0")
        check(abs(result.mean) < 10.0,
              f"Novel mean={result.mean:.5f} is bounded (|mean| < 10)")
        # Uncertainty should be larger for novel than for training points
        train_var = np.mean([
            gpr.predict(extract_gpr_feature(sc)).variance
            for sc in t172_scores if sc.experimental_kie is not None
        ])
        print(f"  Novel variance={result.variance:.5f}  "
              f"Train avg variance={train_var:.5f}")
    else:
        print("  Skipped (GPR not fitted)")

    print("\n" + "=" * 60)
    print("  DONE")
    print("=" * 60)


if __name__ == '__main__':
    main()
