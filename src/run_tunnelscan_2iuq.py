"""
run_tunnelscan_2iuq.py
----------------------
Runs the AADH TunnelScan on 2IUQ (dithionite-reduced, tryptamine covalently
bound as carbinolamine intermediate TSS).  Compares key predictions against
the 2AGW-based scan to assess structural sensitivity.

Usage:
  python3 run_tunnelscan_2iuq.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tunnel_scan import run_scan, AADH_2IUQ_CONFIG
from report import generate_report, print_quick_summary
from multi_mutation import print_double_mutant_report
from temperature_dependence import predict_temperature_dependence, print_temperature_report


def main():
    structures_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'structures')

    pdb_path = os.path.join(structures_dir, '2IUQ.pdb')
    if not os.path.exists(pdb_path):
        from tunnel_scan import download_pdb
        print("Downloading 2IUQ...")
        pdb_path = download_pdb('2IUQ', structures_dir)
    else:
        print(f"Using existing structure: {pdb_path}")

    result = run_scan(pdb_path=pdb_path, config=AADH_2IUQ_CONFIG, verbose=True)

    print_quick_summary(result)

    results_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'results')
    os.makedirs(results_dir, exist_ok=True)
    report_path = os.path.join(results_dir, 'tunnelscan_aadh_2iuq.txt')
    generate_report(result, output_path=report_path)

    print("\n" + "★"*65)
    print("  NOVEL PREDICTIONS (2IUQ substrate-bound structure)")
    print("★"*65)
    enhancing = result.top_enhancing
    if enhancing:
        print(f"\n  {len(enhancing)} mutations predicted to EXCEED wild-type KIE:\n")
        print(f"  {'Mutation':<10} {'KIE':>6} {'vs WT':>7} {'Δstat':>7} "
              f"{'Δdyn':>6} {'Mechanism':<10} {'Confidence':>10}")
        print(f"  {'─'*60}")
        for sc in enhancing[:20]:
            print(f"  {sc.label:<10} {sc.predicted_kie:>6.1f} "
                  f"{sc.fold_vs_wt:>+6.2f}x "
                  f"{sc.static_delta:>+6.2f} "
                  f"{sc.dynamic_delta:>+6.2f}  "
                  f"{sc.dominant_mechanism:<10} "
                  f"{sc.confidence:>10.2f}")
    else:
        print("\n  No mutations predicted to exceed WT KIE.")
        for sc in result.novel_scores[:10]:
            print(f"  {sc.label:<10} KIE={sc.predicted_kie:.1f}  ({sc.dominant_mechanism})")

    print(f"\n  Full report saved: {report_path}")
    print("★"*65)

    if result.double_mutant_scores:
        print_double_mutant_report(result.double_mutant_scores, wt_kie=result.wt_kie_predicted)

    top_novel = result.top_enhancing[:10]
    if top_novel:
        temp_preds = [predict_temperature_dependence(sc.label, sc.predicted_kie)
                      for sc in top_novel]
        print_temperature_report(temp_preds)


if __name__ == '__main__':
    main()
