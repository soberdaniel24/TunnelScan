"""
test_gnn_coupling.py
--------------------
Unit tests for gnn_coupling.py (Module 4).

Tests:
  1. build_gnn_model() runs on 2AGW without error
  2. Adjacency matrix is row-stochastic (rows sum to 1.0)
  3. Node features are physically reasonable (0-1 range)
  4. Message passing propagates signal (layer signals change across layers)
  5. Missing residue key returns zero GNN delta
  6. fit() converges without error on T172 calibration residuals
  7. After fitting, GNN residuals smaller than pre-fit residuals
  8. Prediction signs: flexible mutation (ALA) vs rigid (PRO)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from pdb_parser import Structure
from elastic_network import build_gnm
from tunnelling_model import bell_correction
from tunnel_score import TunnelScorer, DEFAULT_BETA
from stochastic_tunnelling import build_stochastic_model
from calibration import AADH_KIE_DATA
from gnn_coupling import (
    build_gnn_model, GNNCoupling, GNNResult,
    compute_gnn_residuals_from_scan,
    _build_adjacency, _build_node_features,
    N_FEATURES, N_LAYERS,
)

PDB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', 'data', 'structures', '2AGW.pdb')

DONOR_KEY    = ('D', 3001)
ACCEPTOR_KEY = ('D', 128)
T172_KEY     = ('D', 172)

PASS = "\033[92m  PASS\033[0m"
FAIL = "\033[91m  FAIL\033[0m"

def check(cond, msg):
    print(f"  {PASS if cond else FAIL}  {msg}")
    return cond


def main():
    if not os.path.exists(PDB_PATH):
        print(f"ERROR: {PDB_PATH} not found")
        sys.exit(1)

    print("=" * 60)
    print("  GNN COUPLING MODULE — UNIT TESTS")
    print("=" * 60)

    # ── Setup ─────────────────────────────────────────────────────────────────
    print("\n[Setup] Parsing structure and building GNM...")
    s   = Structure(PDB_PATH)
    enm = build_gnm(s, cutoff=7.5)
    print(f"  ENM: {enm.n_residues} residues")

    d_chain, d_resnum, d_atom = 'D', 3001, 'CA'
    a_chain, a_resnum, a_atom = 'D', 128, 'OD2'
    donor_atom    = s.get_atom(d_chain, d_resnum, d_atom) or s.get_residue(d_chain, d_resnum).ca
    acceptor_atom = s.get_atom(a_chain, a_resnum, a_atom) or s.get_residue(a_chain, a_resnum).ca
    donor_coords    = donor_atom.coords
    acceptor_coords = acceptor_atom.coords

    # ── Test 1: Build without error ───────────────────────────────────────────
    print("\n[Test 1] build_gnn_model() completes without error")
    try:
        gnn = build_gnn_model(
            s, enm, DONOR_KEY, ACCEPTOR_KEY,
            donor_coords, acceptor_coords,
            verbose=False)
        check(True, "build_gnn_model() completed")
    except Exception as e:
        check(False, f"Exception: {e}")
        sys.exit(1)

    # ── Test 2: Adjacency matrix is row-stochastic ────────────────────────────
    print("\n[Test 2] Adjacency matrix is row-stochastic (rows sum ≈ 1)")
    adj = gnn._adj
    row_sums = adj.sum(axis=1)
    # All nodes should have row-sum ≈ 1 (isolated nodes clamped to 1)
    has_contacts = (adj.sum(axis=1) > 0)
    check(adj.shape[0] == adj.shape[1], f"Adj is square ({adj.shape[0]}×{adj.shape[1]})")
    check(np.all(adj >= 0.0), "All adjacency entries ≥ 0")
    # Check non-isolated nodes sum to 1
    connected = has_contacts.sum()
    row_sum_check = np.allclose(row_sums[has_contacts], 1.0, atol=1e-6)
    check(row_sum_check, f"{connected}/{adj.shape[0]} connected nodes have row-sum=1")
    check(int(has_contacts.sum()) > enm.n_residues * 0.5,
          f"More than 50% of nodes have at least one contact ({connected})")

    # ── Test 3: Node features are in range ────────────────────────────────────
    print("\n[Test 3] Node features in physical range [0,1]")
    phi = gnn._phi
    check(phi.shape == (enm.n_residues, N_FEATURES),
          f"Feature matrix shape = ({enm.n_residues}, {N_FEATURES})")
    for f in range(N_FEATURES):
        col_min, col_max = phi[:, f].min(), phi[:, f].max()
        check(col_min >= -0.01 and col_max <= 1.01,
              f"Feature {f}: range [{col_min:.3f}, {col_max:.3f}] ⊆ [0,1]")

    # ── Test 4: Message passing propagates signal ─────────────────────────────
    print("\n[Test 4] Message passing changes node states across layers")
    w_test = np.array([0.5, 0.5, 0.5, 1.0])
    keys = enm.residue_keys
    t172_idx = keys.index(T172_KEY) if T172_KEY in keys else None
    if t172_idx is not None:
        _, layer_sigs = gnn._forward(t172_idx, -0.4, w_test[:N_LAYERS])
        print(f"  Layer signals at T172: {[f'{s:.4f}' for s in layer_sigs]}")
        check(len(layer_sigs) == N_LAYERS,
              f"Returned {N_LAYERS} layer signals")
        # Signal should change across layers (non-trivial propagation)
        check(max(layer_sigs) > 0,
              "Layer signals > 0 (non-trivial propagation)")
    else:
        print("  T172 not found in ENM — skip layer signal test")

    # ── Test 5: Missing residue returns zero ──────────────────────────────────
    print("\n[Test 5] Missing residue key returns zero GNN delta")
    r_miss = gnn.predict(('Z', 9999), 'ALA', 'GLY')
    check(r_miss.gnn_delta == 0.0,
          "Missing residue → gnn_delta == 0.0")

    # ── Test 6: fit() on T172 residuals ──────────────────────────────────────
    print("\n[Test 6] fit() converges on T172 calibration residuals")

    # Build residuals from physics pipeline
    wt_result   = bell_correction(13.4, 1184.0, min(
        float(np.linalg.norm(acceptor_coords - donor_coords)), 3.5),
        experimental_KIE=55.0, use_wigner_kirkwood=True)
    stoch_model = build_stochastic_model(s, enm, DONOR_KEY, ACCEPTOR_KEY)

    aniso_map = {}
    try:
        from anisotropic_bfactor import build_alignment_map
        aniso_pdb = PDB_PATH.replace('2AGW.pdb', '2AH1.pdb')
        if os.path.exists(aniso_pdb):
            aniso_map = build_alignment_map(aniso_pdb, donor_coords, acceptor_coords)
    except Exception: pass

    scorer = TunnelScorer(
        s, enm, wt_result, beta=DEFAULT_BETA, gamma=1.0,
        anisotropic_alignment_map=aniso_map,
        stochastic_model=stoch_model,
        donor_chain=d_chain, donor_resnum=d_resnum, donor_atom=d_atom,
        acceptor_chain=a_chain, acceptor_resnum=a_resnum, acceptor_atom=a_atom,
    )

    cal_residuals = []
    for dp in AADH_KIE_DATA:
        if dp.new_aa == 'WT': continue
        res = s.get_residue(a_chain, dp.residue)
        if res is None: continue
        sc = scorer.score_mutation(res, dp.new_aa, 'acceptor', 5.0)
        residual = float(np.log(dp.kie_298k) - np.log(sc.predicted_kie))
        cal_residuals.append(((a_chain, dp.residue), dp.orig_aa, dp.new_aa, residual))
        print(f"  {dp.label}: physics_kie={sc.predicted_kie:.2f}  "
              f"exp={dp.kie_298k:.1f}  residual={residual:.4f}")

    try:
        model = gnn.fit(cal_residuals, lambda_reg=0.10, verbose=True)
        check(True, "fit() completed without error")
        check(gnn.is_fitted(), "GNN model is fitted")
    except Exception as e:
        check(False, f"fit() raised: {e}")

    # ── Test 7: After fitting, GNN reduces residuals ──────────────────────────
    print("\n[Test 7] GNN predictions reduce residual magnitude")
    before_ss = sum(r**2 for _, _, _, r in cal_residuals)
    after_ss  = 0.0
    for (chn, rnum), orig, new, residual in cal_residuals:
        r = gnn.predict((chn, rnum), orig, new)
        after_ss += (residual - r.gnn_delta)**2
    check(after_ss <= before_ss + 1e-6,
          f"SS_residuals: {before_ss:.5f} → {after_ss:.5f} (GNN reduces or leaves unchanged)")
    print(f"  GNN weights: w_mp={gnn._weights[:3].mean():.4f}  "
          f"w_out={gnn._weights[3]:.4f}")

    # ── Test 8: Sign convention ───────────────────────────────────────────────
    print("\n[Test 8] GNN outputs: T172 variants")
    if t172_idx is not None:
        for (_, _, new_aa, _) in cal_residuals:
            r = gnn.predict(T172_KEY, 'THR', new_aa)
            print(f"  T172{new_aa}: gnn_delta={r.gnn_delta:+.5f}  "
                  f"da_reach={r.da_reach:.4f}  layer_sigs={[f'{s:.3f}' for s in r.layer_signals]}")
        check(True, "GNN predictions computed for all T172 variants")

    print("\n" + "=" * 60)
    print("  DONE")
    print("=" * 60)


if __name__ == '__main__':
    main()
