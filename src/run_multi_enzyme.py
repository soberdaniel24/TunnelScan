"""
run_multi_enzyme.py
-------------------
Cross-enzyme tunnelling network comparison (Part B entry point).

Downloads PDB files for AADH, MADH, MR, and ht-ADH and runs TunnelScan
on each, then computes the cross-enzyme network topology comparison:

  - λ₂ vs WT KIE correlation across 4 enzymes
  - Network robustness Ω comparison
  - Structural equivalents of AADH L87/V175 in other enzymes

Usage:
  python3 run_multi_enzyme.py

Output:
  - Console report (cross-enzyme comparison table)
  - data/results/multi_enzyme_report.txt
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tunnel_scan import (
    AADH_CONFIG, MADH_CONFIG, MR_CONFIG, htADH_CONFIG, download_pdb
)
from multi_enzyme import run_multi_enzyme, print_multi_enzyme_report


def main():
    structures_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'structures')
    os.makedirs(structures_dir, exist_ok=True)

    results_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'results')
    os.makedirs(results_dir, exist_ok=True)

    # ── Download PDB files ────────────────────────────────────────────────────
    configs = [AADH_CONFIG, MADH_CONFIG, MR_CONFIG, htADH_CONFIG]
    configs_and_paths = []

    for cfg in configs:
        pdb_path = os.path.join(structures_dir, f"{cfg.pdb_id}.pdb")
        if not os.path.exists(pdb_path):
            print(f"Downloading {cfg.pdb_id}...")
            try:
                pdb_path = download_pdb(cfg.pdb_id, structures_dir)
            except Exception as e:
                print(f"  WARNING: Could not download {cfg.pdb_id}: {e}")
                print(f"  Place {cfg.pdb_id}.pdb in {structures_dir} to include it.")
                continue
        else:
            print(f"Using existing structure: {pdb_path}")
        configs_and_paths.append((cfg, pdb_path))

    if not configs_and_paths:
        print("No PDB files available. Exiting.")
        sys.exit(1)

    # ── Run scans ────────────────────────────────────────────────────────────
    print(f"\nRunning TunnelScan on {len(configs_and_paths)} enzyme(s)...")
    run_results = run_multi_enzyme(configs_and_paths, verbose=True)

    if not run_results:
        print("No scans completed successfully.")
        sys.exit(1)

    # ── Print cross-enzyme report ─────────────────────────────────────────────
    print_multi_enzyme_report(run_results)

    # ── Save report ───────────────────────────────────────────────────────────
    report_path = os.path.join(results_dir, 'multi_enzyme_report.txt')
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_multi_enzyme_report(run_results)
    with open(report_path, 'w') as f:
        f.write(buf.getvalue())
    print(f"\nReport saved: {report_path}")


if __name__ == '__main__':
    main()
