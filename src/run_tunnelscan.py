"""
run_tunnelscan.py
-----------------
Main entry point. Downloads 1AX3 and runs the complete AADH scan.

Usage:
  python3 run_tunnelscan.py

Output:
  - tunnelscan_results.txt  (full report)
  - Console summary with novel predictions

This is the command that generates novel scientific results.
"""

import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tunnel_scan import run_scan, download_pdb, AADH_CONFIG
from tunnel_score import compute_delta_r_max, ALPHA_H
from report import generate_report, print_quick_summary
from multi_mutation import print_double_mutant_report
from temperature_dependence import predict_temperature_dependence, print_temperature_report


def main():
    # ── Step 1: Get the PDB file ──────────────────────────────────────────────
    structures_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'structures')
    os.makedirs(structures_dir, exist_ok=True)

    pdb_path = os.path.join(structures_dir, '2AGW.pdb')

    if not os.path.exists(pdb_path):
        print("Downloading 2AGW (AADH crystal structure)...")
        try:
            pdb_path = download_pdb('2AGW', structures_dir)
        except Exception as e:
            print(f"Download failed: {e}")
            print("Please manually download 2AGW.pdb from https://rcsb.org")
            print(f"and place it at: {pdb_path}")
            sys.exit(1)
    else:
        print(f"Using existing structure: {pdb_path}")

    # ── Physical ceiling header ───────────────────────────────────────────────
    _delta_r_max = compute_delta_r_max(
        AADH_CONFIG.promoting_vibration_cm1,
        AADH_CONFIG.da_reduced_mass_u,
        getattr(AADH_CONFIG, 'temperature', 300.0)
    )
    _kie_ceiling_fold = np.exp(ALPHA_H * _delta_r_max)
    _kie_ceiling_abs  = AADH_CONFIG.wt_kie_exp * _kie_ceiling_fold
    print(f"\nPhysical ceiling (derived from Johannissen et al. J Phys Chem B 2007):")
    print(f"  Promoting vibration: {AADH_CONFIG.promoting_vibration_cm1:.1f} cm-1")
    print(f"  D-A reduced mass: {AADH_CONFIG.da_reduced_mass_u:.3f} u  [C->O proton transfer]")
    print(f"  Max D-A compression/mutation: {_delta_r_max:.3f} A  [= thermal amplitude at 300K]")
    print(f"  Max KIE fold/mutation: {_kie_ceiling_fold:.1f}x  [= exp(ALPHA_H x {_delta_r_max:.3f})]")
    print(f"  Physical KIE ceiling: {_kie_ceiling_abs:.0f}  [= WT x {_kie_ceiling_fold:.1f}x]")

    # ── Step 2: Run the scan ─────────────────────────────────────────────────
    result = run_scan(
        pdb_path=pdb_path,
        config=AADH_CONFIG,
        verbose=True
    )

    # ── Physical ceiling assertion ────────────────────────────────────────────
    delta_r_max   = compute_delta_r_max(AADH_CONFIG.promoting_vibration_cm1,
                                        AADH_CONFIG.da_reduced_mass_u)
    kie_ceiling   = result.wt_kie_predicted * np.exp(ALPHA_H * delta_r_max)
    max_kie       = max(s.predicted_kie for s in result.all_scores)
    assert max_kie <= kie_ceiling + 1e-6, (
        f"Physical ceiling violated: max KIE {max_kie:.1f} > ceiling {kie_ceiling:.1f}"
    )
    print(f"\n  [ASSERT PASS] max KIE {max_kie:.1f} <= ceiling {kie_ceiling:.1f}")

    # ── Step 3: Print summary ─────────────────────────────────────────────────
    print_quick_summary(result)

    # ── Step 4: Save full report ──────────────────────────────────────────────
    results_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'results')
    os.makedirs(results_dir, exist_ok=True)
    report_path = os.path.join(results_dir, 'tunnelscan_aadh.txt')

    report = generate_report(result, output_path=report_path)

    # Also print the novel predictions section to terminal
    print("\n" + "★"*65)
    print("  NOVEL PREDICTIONS — copy these into your lab notebook")
    print("★"*65)
    enhancing = result.top_enhancing
    if enhancing:
        print(f"\n  {len(enhancing)} mutations predicted to EXCEED wild-type KIE:\n")
        print(f"  {'Mutation':<10} {'KIE':>6} {'vs WT':>7} {'Δstat':>7} "
              f"{'Δdyn':>6} {'Mechanism':<10} {'Confidence':>10}")
        print(f"  {'─'*60}")
        for sc in enhancing:
            print(f"  {sc.label:<10} {sc.predicted_kie:>6.1f} "
                  f"{sc.fold_vs_wt:>+6.2f}x "
                  f"{sc.static_delta:>+6.2f} "
                  f"{sc.dynamic_delta:>+6.2f}  "
                  f"{sc.dominant_mechanism:<10} "
                  f"{sc.confidence:>10.2f}")
    else:
        print("\n  No mutations predicted to exceed WT KIE in this scan.")
        print("  (This means WT is already near-optimal geometrically)")
        print("  Top novel predictions by absolute KIE:")
        for sc in result.novel_scores[:10]:
            print(f"  {sc.label:<10} KIE={sc.predicted_kie:.1f}  ({sc.dominant_mechanism})")

    print(f"\n  Full report saved: {report_path}")
    print("★"*65)

    # ── Step 5: Distal topological candidates ────────────────────────────────
    if result.topological_candidates:
        print("\n" + "="*65)
        print("  DISTAL TOPOLOGICAL CANDIDATES")
        print("  High-betweenness network bottlenecks outside geometric scan")
        print("="*65)
        seen = set()
        rows = []
        for sc in result.topological_candidates:
            key = (sc.chain, sc.residue_number)
            if key not in seen:
                seen.add(key)
                rows.append(sc)
        rows.sort(key=lambda x: x.tunnelling_betweenness, reverse=True)
        print(f"\n  {'Residue':<8} {'Betweenness':>12}  {'Top mutation':>13}  "
              f"{'KIE':>7}  {'Mechanism':<10}")
        print(f"  {'─'*58}")
        for sc in rows[:10]:
            top_mut = max(
                [m for m in result.topological_candidates
                 if m.chain == sc.chain and m.residue_number == sc.residue_number],
                key=lambda x: x.predicted_kie
            )
            print(f"  {sc.orig_aa}{sc.residue_number} ({sc.chain})   "
                  f"{sc.tunnelling_betweenness:>12.3f}  "
                  f"{top_mut.label:>13}  "
                  f"{top_mut.predicted_kie:>7.1f}  "
                  f"{top_mut.dominant_mechanism:<10}")
        print(f"\n  These residues are network bottlenecks identified by quantum")
        print(f"  tunnelling network topology — not reachable by geometry alone.")
        print("="*65)

    # ── Step 6: Part A — Network robustness + rewiring mutations ─────────────
    if result.network_robustness > 0 or result.full_resistance_map:
        print("\n" + "="*65)
        print("  PART A: TUNNELLING NETWORK TOPOLOGY")
        print("="*65)

        print(f"\n  Network robustness  Ω = {result.network_robustness:.4f}")
        print(f"  (Ω = λ₂ / mean_R_top10; higher = better-connected quantum flux path)")

        if result.full_resistance_map:
            from pdb_parser import Structure
            pdb_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                '..', 'data', 'structures', '2AGW.pdb')
            try:
                s_temp = Structure(pdb_path)
                prot_R = [
                    (k, v) for k, v in result.full_resistance_map.items()
                    if not (s_temp.get_residue(*k) and s_temp.get_residue(*k).is_hetatm)
                ]
                prot_R.sort(key=lambda x: x[1])
                print(f"\n  Full-protein effective resistance to D-A axis ({len(prot_R)} residues)")
                print(f"  Most tightly coupled (lowest R):")
                print(f"  {'Residue':<12} {'R_i':>8}")
                print(f"  {'─'*22}")
                for (ch, rn), r in prot_R[:10]:
                    res = s_temp.get_residue(ch, rn)
                    aa  = res.name if res else '???'
                    print(f"  {aa}{rn}({ch})    {r:>8.4f}")
                print(f"\n  Most distal from D-A (highest R):")
                for (ch, rn), r in prot_R[-5:][::-1]:
                    res = s_temp.get_residue(ch, rn)
                    aa  = res.name if res else '???'
                    print(f"  {aa}{rn}({ch})    {r:>8.4f}")
            except Exception:
                pass

        if result.rewiring_mutations:
            print(f"\n  Rewiring mutations (substitutions that increase λ₂):")
            print(f"  {'Mutation':<10}  {'Δλ₂':>8}  {'new_λ₂':>8}  {'F-sens':>10}")
            print(f"  {'─'*42}")
            for rm in result.rewiring_mutations[:12]:
                print(f"  {rm.label:<10}  {rm.delta_lambda2:>8.4f}  "
                      f"{rm.new_lambda2:>8.4f}  {rm.fiedler_sensitivity:>10.4f}")
            print(f"\n  These mutations are predicted to improve quantum flux")
            print(f"  connectivity without necessarily changing active-site geometry.")
        print("="*65)

    # ── Step 7: Double mutant predictions ────────────────────────────────────
    if result.double_mutant_scores:
        print_double_mutant_report(
            result.double_mutant_scores,
            wt_kie=result.wt_kie_predicted
        )

    # ── Step 7: Temperature dependence predictions ────────────────────────
    top_novel = result.top_enhancing[:15]
    if top_novel:
        temp_preds = [
            predict_temperature_dependence(sc.label, sc.predicted_kie)
            for sc in top_novel
        ]
        # Also add known mutations for validation
        from calibration import AADH_KIE_DATA
        for dp in AADH_KIE_DATA:
            if dp.new_aa != 'WT':
                temp_preds.append(
                    predict_temperature_dependence(dp.label, dp.kie_298k)
                )
        print_temperature_report(temp_preds)


if __name__ == '__main__':
    main()
