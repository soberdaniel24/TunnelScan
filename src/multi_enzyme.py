"""
multi_enzyme.py
---------------
Cross-enzyme quantum tunnelling network comparison (Part B).

Runs TunnelScan on four H-tunnelling enzymes and computes:
  1. λ₂ vs WT KIE correlation — tests whether better-connected tunnelling
     networks correspond to larger primary KIEs across unrelated sequences
  2. Network robustness Ω comparison table (AADH → MADH → MR → ht-ADH)
  3. Structural equivalents of AADH distal candidates (L87, V175) in each
     enzyme — based purely on betweenness rank and D-A positional analogy,
     no sequence alignment required
  4. Betweenness rank conservation — do topologically equivalent positions
     couple tunnelling across structurally unrelated sequences?

The central hypothesis: enzymes that evolved for quantum tunnelling (high
primary KIE) have higher algebraic connectivity (λ₂) in their tunnelling
network — a consequence of evolutionary pressure on network topology.

References
----------
Hay et al. (2009) PNAS — morphinone reductase KIE
Scrutton lab (2003–2012) — MADH KIE temperature dependence
Klinman & Kohen (2013) Ann. Rev. Biochem. — ht-ADH
"""

from __future__ import annotations

import os
import sys
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ── Data class ─────────────────────────────────────────────────────────────────

@dataclass
class EnzymeNetworkStats:
    """Per-enzyme summary for cross-enzyme comparison."""
    enzyme_name:     str
    pdb_id:          str
    wt_kie:          float
    lambda2:         float         # Fiedler value λ₂
    robustness:      float         # Ω = λ₂ / mean_R_top10
    n_nodes:         int           # nodes in tunnelling network
    # [(chain, resnum, aa_name, B_i, position_side)]: top network nodes by betweenness
    top_betweenness: List[Tuple]
    distal_keys:     List[Tuple]   # (chain, resnum) of distal topological candidates
    full_resistance: Dict          # (chain, resnum) → R_i


# ── Extraction from completed ScanResult ─────────────────────────────────────

def extract_enzyme_stats(result, structure) -> Optional[EnzymeNetworkStats]:
    """
    Extract EnzymeNetworkStats from a completed ScanResult.
    Requires result.tunnelling_network and result.full_resistance_map to be populated.
    """
    tn = result.tunnelling_network
    if tn is None:
        return None

    # Top betweenness residues with amino acid names and position_side
    bt_sorted = sorted(tn.betweenness.items(), key=lambda x: x[1], reverse=True)

    # Build position_side lookup from scored mutations
    position_map: Dict[Tuple, str] = {}
    for sc in result.all_scores + result.topological_candidates:
        position_map[(sc.chain, sc.residue_number)] = sc.position_side

    top_bt = []
    for (ch, rn), b in bt_sorted[:20]:
        res  = structure.get_residue(ch, rn)
        aa   = res.name if res else '???'
        side = position_map.get((ch, rn), 'flanking')
        top_bt.append((ch, rn, aa, b, side))

    # Distal candidates (from topological scan)
    distal_keys = []
    seen: set = set()
    for sc in result.topological_candidates:
        key = (sc.chain, sc.residue_number)
        if key not in seen:
            seen.add(key)
            distal_keys.append(key)

    return EnzymeNetworkStats(
        enzyme_name     = result.config.name,
        pdb_id          = result.config.pdb_id,
        wt_kie          = result.config.wt_kie_exp,
        lambda2         = tn.fiedler_value,
        robustness      = tn.robustness,
        n_nodes         = len(tn.nodes),
        top_betweenness = top_bt,
        distal_keys     = distal_keys,
        full_resistance = result.full_resistance_map,
    )


# ── Cross-enzyme correlation ───────────────────────────────────────────────────

def correlate_lambda2_kie(stats_list: List[EnzymeNetworkStats]) -> dict:
    """
    Pearson and Spearman correlations of λ₂ vs WT KIE and Ω vs WT KIE.
    With n=4 data points these are exploratory — report r alongside p-value.
    The direction (positive r) is the physically motivated prediction.
    """
    kies    = np.array([s.wt_kie  for s in stats_list])
    lam2s   = np.array([s.lambda2 for s in stats_list])
    omegas  = np.array([s.robustness for s in stats_list])

    result = {}
    if len(stats_list) >= 3:
        r_lam2, p_lam2 = pearsonr(lam2s, kies)
        r_omega, p_omega = pearsonr(omegas, kies)
        rs_lam2, ps_lam2 = spearmanr(lam2s, kies)

        result['r_lambda2_kie']    = float(r_lam2)
        result['p_lambda2_kie']    = float(p_lam2)
        result['rho_lambda2_kie']  = float(rs_lam2)  # Spearman
        result['r_omega_kie']      = float(r_omega)
        result['p_omega_kie']      = float(p_omega)
    return result


# ── Structural equivalents ────────────────────────────────────────────────────

def find_structural_equivalents(
    target:          EnzymeNetworkStats,
    ref_side:        str,    # 'donor' | 'acceptor' | 'flanking'
    ref_bt_rank:     float,  # normalised betweenness of reference [0,1]
    n:               int = 5,
) -> List[dict]:
    """
    Find residues in the target enzyme topologically analogous to a reference
    residue characterised by (ref_side, ref_bt_rank).

    Analogy score = |B_i_norm − ref_bt_rank| + 0.3 × (0 if side matches else 1)
    Lower score = better structural equivalent.

    This is purely topological: "same relative betweenness rank AND same side
    of the D-A axis."  No sequence alignment needed.
    """
    if not target.top_betweenness:
        return []

    max_bt = max(b for _, _, _, b, _ in target.top_betweenness)
    if max_bt <= 0:
        return []

    candidates = []
    for ch, rn, aa, bt, side in target.top_betweenness:
        bt_norm     = bt / max_bt
        side_penalty = 0.0 if side == ref_side else 0.3
        score        = abs(bt_norm - ref_bt_rank) + side_penalty
        candidates.append({
            'chain': ch, 'resnum': rn, 'aa': aa,
            'betweenness': bt, 'bt_norm': bt_norm,
            'side': side,
            'R': target.full_resistance.get((ch, rn), float('inf')),
            'analogy_score': score,
        })

    candidates.sort(key=lambda x: x['analogy_score'])
    return candidates[:n]


# ── Main runner ───────────────────────────────────────────────────────────────

def run_multi_enzyme(
    configs_and_paths: List[Tuple],   # [(config, pdb_path), ...]
    beta:              float = 5.0,
    verbose:           bool = True,
) -> List[Tuple]:
    """
    Run TunnelScan on each (config, pdb_path) pair.
    Returns [(config, ScanResult, Structure), ...] — Structure needed for stat extraction.
    """
    from tunnel_scan import run_scan
    from pdb_parser import Structure

    results = []
    for config, pdb_path in configs_and_paths:
        if not os.path.exists(pdb_path):
            if verbose:
                print(f"\n  SKIP {config.pdb_id}: PDB not found at {pdb_path}")
            continue
        if verbose:
            print(f"\n{'='*65}")
            print(f"  Running {config.name}...")
        try:
            result = run_scan(pdb_path=pdb_path, config=config,
                              beta=beta, verbose=verbose)
            structure = Structure(pdb_path)
            results.append((config, result, structure))
        except Exception as e:
            if verbose:
                print(f"  ERROR running {config.pdb_id}: {e}")
    return results


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_multi_enzyme_report(
    run_results: List[Tuple],
    aadh_idx:    int = 0,   # index of AADH in run_results (reference enzyme)
) -> None:
    """
    Print formatted cross-enzyme comparison table + structural equivalents.
    """
    # Extract stats
    stats_list = []
    for config, result, structure in run_results:
        st = extract_enzyme_stats(result, structure)
        if st is not None:
            stats_list.append(st)

    if not stats_list:
        print("  No completed enzyme scans to compare.")
        return

    # ── Table 1: Network topology summary ─────────────────────────────────────
    print("\n" + "="*70)
    print("  PART B: CROSS-ENZYME TUNNELLING NETWORK COMPARISON")
    print("="*70)
    print(f"\n  {'Enzyme':<35} {'PDB':>5}  {'WT KIE':>8}  {'λ₂':>8}  "
          f"{'Ω':>8}  {'nodes':>6}")
    print(f"  {'─'*65}")
    for st in stats_list:
        print(f"  {st.enzyme_name[:35]:<35} {st.pdb_id:>5}  "
              f"{st.wt_kie:>8.1f}  {st.lambda2:>8.4f}  "
              f"{st.robustness:>8.4f}  {st.n_nodes:>6}")

    # ── Correlation: λ₂ vs KIE ────────────────────────────────────────────────
    corr = correlate_lambda2_kie(stats_list)
    if corr:
        print(f"\n  λ₂ vs WT KIE correlation  (n={len(stats_list)}):")
        print(f"    Pearson  r = {corr['r_lambda2_kie']:+.3f}  "
              f"p = {corr['p_lambda2_kie']:.3f}")
        print(f"    Spearman ρ = {corr['rho_lambda2_kie']:+.3f}")
        print(f"  Ω vs WT KIE:")
        print(f"    Pearson  r = {corr['r_omega_kie']:+.3f}  "
              f"p = {corr['p_omega_kie']:.3f}")
        if corr['r_lambda2_kie'] > 0.7:
            print(f"\n  ★ Positive λ₂–KIE correlation: higher tunnelling correlates")
            print(f"    with better-connected quantum flux networks.")
        elif corr['r_lambda2_kie'] < -0.3:
            print(f"\n  NOTE: Negative λ₂–KIE correlation — tunnelling may be")
            print(f"    dominated by single-bottleneck (low λ₂) networks.")

    # ── Table 2: Top betweenness per enzyme ───────────────────────────────────
    print(f"\n  Top betweenness residues per enzyme (network bottlenecks):")
    print(f"  {'─'*50}")
    for st in stats_list:
        top3 = st.top_betweenness[:3]
        labels = '  '.join(f"{aa}{rn}({ch})={b:.3f}"
                           for ch, rn, aa, b, *_ in top3)
        print(f"  {st.pdb_id}:  {labels}")

    # ── Table 3: Structural equivalents of AADH L87 and V175 ─────────────────
    if len(stats_list) >= 2:
        # AADH L87: top betweenness (rank=1.0), donor-side
        # AADH V175: betweenness rank ≈ 0.525, acceptor-side
        ref_residues = [
            ('L87 (AADH top bottleneck, donor-side)',      'donor',    1.0),
            ('V175 (AADH distal acceptor-side)',           'acceptor', 0.525),
        ]
        print(f"\n  Structural equivalents of AADH distal candidates")
        print(f"  (ranked by betweenness rank + D-A side; purely topological):")
        print(f"  {'─'*60}")
        for ref_name, ref_side, ref_bt_rank in ref_residues:
            print(f"\n  Reference: {ref_name}  (rank={ref_bt_rank:.3f}, side={ref_side})")
            for st in stats_list:
                if 'AADH' in st.enzyme_name:
                    continue
                equiv = find_structural_equivalents(st, ref_side, ref_bt_rank, n=3)
                if equiv:
                    top = equiv[0]
                    print(f"    {st.pdb_id}: {top['aa']}{top['resnum']}({top['chain']})  "
                          f"B={top['betweenness']:.3f}  side={top['side']}  "
                          f"Δ={top['analogy_score']:.3f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    if len(stats_list) >= 2:
        best_kie   = max(stats_list, key=lambda s: s.wt_kie)
        best_lam2  = max(stats_list, key=lambda s: s.lambda2)
        best_omega = max(stats_list, key=lambda s: s.robustness)
        print(f"\n  Highest WT KIE:        {best_kie.pdb_id} ({best_kie.wt_kie:.1f})")
        print(f"  Highest λ₂:            {best_lam2.pdb_id} ({best_lam2.lambda2:.4f})")
        print(f"  Highest Ω (robustness):{best_omega.pdb_id} ({best_omega.robustness:.4f})")

    print("="*70)
