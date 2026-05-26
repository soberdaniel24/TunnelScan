"""
run_multi_enzyme.py
-------------------
Cross-enzyme tunnelling network comparison (Part B entry point).

Downloads PDB files for AADH, MADH, MR, ht-ADH, DHFR, and SLO and runs TunnelScan
on each, then computes the cross-enzyme network topology comparison:

  - Physical ceiling table: ν(cm-1), mu(u), delta_r_max, ceiling fold, WT KIE
  - validate_ceiling_against_literature() check
  - λ₂ vs WT KIE correlation across enzymes
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
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tunnel_scan import (
    AADH_CONFIG, MADH_CONFIG, MR_CONFIG, htADH_CONFIG, DHFR_CONFIG, download_pdb
)
from multi_enzyme import run_multi_enzyme, print_multi_enzyme_report
from enzyme_library import ALL_ENZYMES, validate_ceiling_against_literature, PhysicalCeilingViolation


def print_ceiling_table(ceiling_results: dict) -> None:
    """Print per-enzyme physical ceiling table."""
    print("\n" + "="*75)
    print("  PHYSICAL CEILING TABLE (Johannissen et al. J Phys Chem B 2007)")
    print("="*75)
    print(f"  {'Enzyme':<12} {'nu(cm-1)':>9} {'mu(u)':>7} {'dr_max(A)':>10} {'Ceil/mut':>10} {'KIE_WT':>8} {'KIE_ceil':>10}")
    print(f"  {'-'*73}")
    for enzyme in ALL_ENZYMES:
        v = ceiling_results[enzyme.name]
        short_name = enzyme.name.split('(')[0].split(' ')[0]
        if 'Methylamine' in enzyme.name:
            short_name = 'MADH'
        elif 'Morphinone' in enzyme.name:
            short_name = 'MR'
        elif 'Thermophilic' in enzyme.name:
            short_name = 'htADH'
        elif 'Dihydrofolate' in enzyme.name:
            short_name = 'DHFR'
        elif 'Soybean' in enzyme.name:
            short_name = 'SLO'
        elif 'Aromatic' in enzyme.name:
            short_name = 'AADH'
        print(
            f"  {short_name:<12} "
            f"{enzyme.promoting_vibration_cm1:>9.0f} "
            f"{enzyme.da_reduced_mass_u:>7.3f} "
            f"{v['delta_r_max']:>10.4f} "
            f"{v['ceiling_fold']:>9.2f}x "
            f"{enzyme.kie_wt_experimental:>8.1f} "
            f"{v['kie_ceiling']:>10.1f}"
        )
    print("="*75)


def main():
    structures_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'structures')
    os.makedirs(structures_dir, exist_ok=True)

    results_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'results')
    os.makedirs(results_dir, exist_ok=True)

    # ── Run validate_ceiling_against_literature() first ───────────────────────
    print("\n" + "="*65)
    print("  STEP 0: Physical ceiling validation")
    print("="*65)
    try:
        ceiling_results = validate_ceiling_against_literature()
        print("  validate_ceiling_against_literature() PASSED")
        print_ceiling_table(ceiling_results)
    except PhysicalCeilingViolation as e:
        print(f"  CEILING VIOLATION: {e}")
        sys.exit(1)

    # ── Download PDB files ────────────────────────────────────────────────────
    # Core 4 configs that have calibration data; add DHFR if PDB available
    configs = [AADH_CONFIG, MADH_CONFIG, MR_CONFIG, htADH_CONFIG, DHFR_CONFIG]
    # Also try SLO (3PZW) if available
    try:
        from tunnel_scan import ActiveSiteConfig
        slo_pdb = os.path.join(structures_dir, '3PZW.pdb')
        if os.path.exists(slo_pdb):
            SLO_CONFIG = ActiveSiteConfig(
                name='SLO-1 (Glycine max)',
                pdb_id='3PZW',
                donor=('A', 553, 'CA'),
                acceptor=('A', 546, 'CA'),
                barrier_height_kcal=14.0,
                imaginary_freq_cm1=1000.0,
                catalytic_residues=[('A', 553), ('A', 546)],
                scan_radius=8.0,
                wt_kie_exp=36.0,
                promoting_vibration_cm1=40.0,
                da_reduced_mass_u=6.857,
            )
            configs.append(SLO_CONFIG)
    except Exception:
        pass

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

    # ── Spearman correlation across enzymes ───────────────────────────────────
    try:
        from scipy.stats import spearmanr
        # run_results is List[(config, ScanResult, Structure)]
        wt_kies_exp  = [r[1].wt_kie_exp      for r in run_results]
        wt_kies_pred = [r[1].wt_kie_predicted for r in run_results]
        if len(wt_kies_exp) >= 3:
            rho, pval = spearmanr(wt_kies_exp, wt_kies_pred)
            print(f"\n  Cross-enzyme Spearman rho (exp vs pred WT KIE): {rho:.3f}  (p={pval:.3f}, n={len(wt_kies_exp)})")
        else:
            print(f"\n  Only {len(wt_kies_exp)} enzymes scanned; Spearman rho requires >= 3")
    except ImportError:
        print("\n  scipy not available — skipping Spearman rho")

    # ── Save report ───────────────────────────────────────────────────────────
    report_path = os.path.join(results_dir, 'multi_enzyme_report.txt')
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_ceiling_table(ceiling_results)
        print_multi_enzyme_report(run_results)
    with open(report_path, 'w') as f:
        f.write(buf.getvalue())
    print(f"\nReport saved: {report_path}")


if __name__ == '__main__':
    main()
