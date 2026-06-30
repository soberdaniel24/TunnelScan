"""
run_petnr.py
------------
PETNR zero-parameter cross-validation on AADH BETA=5.0.

3KFT: PETNR (Enterobacter cloacae) complex with 1,4,5,6-tetrahydro-NADH.
Actual D-A pair: C4N(NAD 366) → N5(FMN 365), d=3.793 Å.
WT KIE experimental: 7.0 (NADPH, Pudney 2013 JACS / Hay 2026 Table 1).

Known mutants / literature context:
  Pudney 2013 JACS (10.1021/ja312593d): heavy-enzyme isotope labelling —
    NO amino acid mutations; establishes causality between fast protein
    motions and KIE temperature dependence.
  Longbotham 2016 JACS (PMID 27676389): both protein AND FMN cofactor modes
    contribute to enzyme isotope effect (contributions not additive).
  Hay 2018 Biochem (PMID 31119061): second-sphere mutants L25I, L25A,
    I107L, I107A studied for NADH and NADPH half-reactions.

Usage:
    python3 src/run_petnr.py
"""

import sys
import os
import io
import contextlib
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tunnel_scan import PETNR_CONFIG, run_scan, download_pdb
from tunnel_score import DEFAULT_BETA


def main():
    structures_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'structures')
    results_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'results')
    os.makedirs(structures_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    pdb_path = os.path.join(structures_dir, '3KFT.pdb')
    if not os.path.exists(pdb_path):
        print("Downloading 3KFT (PETNR)...")
        pdb_path = download_pdb('3KFT', structures_dir)

    print(f"Running PETNR scan on {pdb_path}...")
    result = run_scan(pdb_path=pdb_path, config=PETNR_CONFIG, verbose=True)

    # ── Build formatted report ────────────────────────────────────────────────
    lines = []

    def p(s=''):
        lines.append(s)

    w = 65
    bar = '█' * w

    p(bar)
    p('  TUNNELSCAN REPORT — PETNR (Enterobacter cloacae)')
    p(bar)
    p(f'  Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    p(f'  Enzyme:    PETNR (Pentaerythritol Tetranitrate Reductase)')
    p(f'  PDB:       3KFT (TH-NADH complex, 2.10 Å)')
    p(f'  BETA:      {DEFAULT_BETA:.1f} (AADH-calibrated; zero-parameter transfer test)')
    p(f'  Residues scanned:    {result.n_residues_found}')
    p(f'  Mutations predicted: {result.n_mutations_scored}')
    p(f'  Novel predictions:   {len(result.novel_scores)}')

    p()
    p('─' * w)
    p('  ACTIVE SITE')
    p('─' * w)
    p('  D-A pair: C4N(TH-NADH res 366) → N5(FMN res 365)')
    p('  Transfer: NADPH C4H → FMN N5 (hydride, OYE mechanism)')
    p(f'  D-A distance (crystal): 3.793 Å  (capped to 3.5 Å for Bell)')
    p('  Donor in structure: 1,4,5,6-tetrahydro-NADH (actual substrate analog)')

    p()
    p('─' * w)
    p('  WT KIE')
    p('─' * w)
    p(f'  Predicted (Bell, before exp override): see verbose output')
    p(f'  Experimental (override):  {result.wt_kie_exp:.1f}  '
      f'(NADPH; Pudney 2013 JACS 10.1021/ja312593d / Hay 2026 Table 1)')
    p(f'  Model baseline used:      {result.wt_kie_predicted:.1f}')
    pred_wt = result.wt_kie_predicted
    in_range = abs(pred_wt - result.wt_kie_exp) / result.wt_kie_exp <= 0.30
    p(f'  Within 30% of exp:        {"YES ✓" if in_range else "NO ✗"}  '
      f'(pred={pred_wt:.1f}, exp={result.wt_kie_exp:.1f})')

    # ── Network topology ──────────────────────────────────────────────────────
    tn = getattr(result, 'tunnelling_network', None)
    p()
    p('─' * w)
    p('  NETWORK TOPOLOGY')
    p('─' * w)
    if tn is not None:
        p(f'  Nodes (D-A network):  {len(tn.nodes)}')
        p(f'  λ₂ (Fiedler gap):     {tn.fiedler_value:.4f}')
        p(f'  Ω (robustness):       {tn.robustness:.4f}')
        p('  Top betweenness residues:')
        top_bt = sorted(tn.betweenness.items(), key=lambda x: x[1], reverse=True)[:8]
        for (ch, rn), bt in top_bt:
            from pdb_parser import Structure
            s_struct = Structure(pdb_path)
            res = s_struct.get_residue(ch, rn)
            aa = res.name if res else '???'
            p(f'    {aa}{rn} ({ch})  B={bt:.3f}')
        p()
        p('  Longbotham 2016 JACS: protein AND FMN cofactor modes both contribute.')
        p('  Check: FMN-adjacent residues in top betweenness?')
        fmn_adj_resnums = {25, 26, 57, 58}  # residues within 8Å of FMN N5 from distance search
        top_bt_resnums = {rn for (ch, rn), bt in top_bt}
        overlap = fmn_adj_resnums & top_bt_resnums
        if overlap:
            p(f'  FMN-adjacent top-betweenness residues: {sorted(overlap)} ✓')
        else:
            p('  No FMN-adjacent residues in top 8 betweenness (cofactor modes may not be')
            p('  captured by protein-only ENM — consistent with Longbotham 2016 finding).')
    else:
        p('  Network topology unavailable (HETATM donor/acceptor outside protein ENM).')
        p('  Note: ENM uses protein Cα only; TH-NADH/FMN are HETATM cofactors.')

    # ── Known mutant direction check ──────────────────────────────────────────
    p()
    p('─' * w)
    p('  KNOWN MUTANT DIRECTION CHECK')
    p('─' * w)
    p('  Pudney 2013 JACS (10.1021/ja312593d): HEAVY ENZYME study only.')
    p('  All non-exchangeable atoms replaced with 13C/15N/2H isotopes.')
    p('  → No amino acid mutations in this paper. Direction check N/A.')
    p()
    p('  Hay 2018 Biochem (PMID 31119061): second-sphere mutants')
    p('  L25I, L25A, I107L, I107A (NADPH half-reaction studied).')
    p('  L25 is 3.5 Å from FMN N5 (nearest protein residue to acceptor).')

    # Find L25 and I107 in scan results
    l25_scores = [sc for sc in result.all_scores if sc.residue_number == 25]
    i107_scores = [sc for sc in result.all_scores if sc.residue_number == 107]

    if l25_scores:
        p(f'  L25 predictions from scan:')
        for sc in sorted(l25_scores, key=lambda x: x.label):
            p(f'    {sc.label:<10} KIE={sc.predicted_kie:>6.1f}  {sc.dominant_mechanism}')
    else:
        p('  L25 not found in scan (outside 8Å radius or catalytic exclusion).')

    if i107_scores:
        p(f'  I107 predictions from scan:')
        for sc in sorted(i107_scores, key=lambda x: x.label):
            p(f'    {sc.label:<10} KIE={sc.predicted_kie:>6.1f}  {sc.dominant_mechanism}')
    else:
        p('  I107 not found in scan (outside 8Å radius).')

    p()
    p('  Note: experimental KIE values for L25x/I107x not available in')
    p('  open-access main text — direction comparison requires full text access.')

    # ── Top 10 novel predictions ──────────────────────────────────────────────
    p()
    p('─' * w)
    p(f'  TOP 10 NOVEL PREDICTIONS (all novel — no PETNR mutation KIE data in scan)')
    p('─' * w)
    p(f'  {"Mutation":<10} {"KIE":>7} {"vsWT":>7} {"Δstat":>7} '
      f'{"Δdyn":>6}  {"Mechanism":<10}  {"Conf":>6}')
    p('  ' + '─' * (w - 2))
    novel = [sc for sc in result.all_scores if sc.is_novel][:10]
    for sc in novel:
        vs = sc.predicted_kie / result.wt_kie_predicted
        p(f'  {sc.label:<10} {sc.predicted_kie:>7.1f} {vs:>+6.2f}x '
          f'{sc.static_delta:>+6.2f} {sc.dynamic_delta:>+5.2f}  '
          f'{sc.dominant_mechanism:<10}  {sc.confidence:>6.3f}')

    # ── Top 5 reducing ────────────────────────────────────────────────────────
    p()
    p('─' * w)
    p('  TOP 5 REDUCING MUTATIONS (negative controls)')
    p('─' * w)
    reducing = [sc for sc in reversed(result.all_scores)
                if sc.is_novel and sc.predicted_kie < result.wt_kie_predicted][:5]
    for sc in reducing:
        p(f'  {sc.label:<12} KIE={sc.predicted_kie:>5.1f}  {sc.dominant_mechanism}')

    # ── Rewiring mutations ────────────────────────────────────────────────────
    if result.rewiring_mutations:
        p()
        p('─' * w)
        p('  TOP REWIRING MUTATIONS (increase spectral gap λ₂)')
        p('─' * w)
        for rm in result.rewiring_mutations[:5]:
            p(f'  {rm.label}  Δλ₂={rm.delta_lambda2:+.4f}')

    # ── Verdict ───────────────────────────────────────────────────────────────
    p()
    p(bar)
    p('  VERDICT: SECOND CROSS-VALIDATION')
    p(bar)

    # WT within 30%?
    wt_pass = abs(result.wt_kie_predicted - result.wt_kie_exp) / result.wt_kie_exp <= 0.30
    # Mutant direction: unknown (no open-access experimental values)
    # Cross-validation claim: WT KIE recovery with zero parameter changes
    p(f'  WT KIE: pred={result.wt_kie_predicted:.1f}  exp={result.wt_kie_exp:.1f}  '
      f'→ {"PASS (≤30%)" if wt_pass else "FAIL (>30%)"}')
    p(f'  Mutant direction check: UNAVAILABLE')
    p(f'    Pudney 2013 is a heavy-enzyme isotope study (no amino acid mutations).')
    p(f'    Hay 2018 L25/I107 mutants scanned but experimental KIE not in open access.')
    p()
    if wt_pass:
        p('  VERDICT: PARTIAL CROSS-VALIDATION')
        p('  WT KIE recovered within 30% with zero PETNR-specific parameters.')
        p('  Cannot assert directional accuracy for amino acid mutations — Pudney 2013')
        p('  contains no such data. Cite as: "WT KIE recovered; direction test pending')
        p('  access to Hay 2018 full text for L25/I107 comparison."')
    else:
        p('  VERDICT: WT KIE NOT RECOVERED.')
        p(f'  pred={result.wt_kie_predicted:.1f} vs exp={result.wt_kie_exp:.1f}  '
          f'({abs(result.wt_kie_predicted - result.wt_kie_exp)/result.wt_kie_exp*100:.0f}% error)')
        p('  Do not cite as cross-validation without investigation.')
    p(bar)
    p('  TunnelScan v0.1 — zero-parameter transfer (AADH BETA=5.0)')
    p('  No PETNR-specific calibration. WT recovery test only.')

    # Print to console
    report_text = '\n'.join(lines)
    print('\n' + report_text)

    # Save to file
    out_path = os.path.join(results_dir, 'tunnelscan_petnr.txt')
    with open(out_path, 'w') as f:
        f.write(report_text + '\n')
    print(f'\nReport saved: {out_path}')

    return result


if __name__ == '__main__':
    main()
