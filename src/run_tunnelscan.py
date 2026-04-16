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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tunnel_scan import run_scan, download_pdb, AADH_CONFIG
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

    # ── Step 2: Run the scan ─────────────────────────────────────────────────
    result = run_scan(
        pdb_path=pdb_path,
        config=AADH_CONFIG,
        verbose=True
    )

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

    # ── Step 5: Double mutant predictions ────────────────────────────────────
    if result.double_mutant_scores:
        print_double_mutant_report(
            result.double_mutant_scores,
            wt_kie=result.wt_kie_predicted
        )

    # ── Step 6: Temperature dependence predictions ────────────────────────
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
