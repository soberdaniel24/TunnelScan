"""
run_tunnelscan_ata117.py
------------------------
Run TunnelScan on ATA-117 (R)-selective omega-transaminase.

Structural proxy: 3WWH (Arthrobacter sp. KNK168, 1.65 Å resolution).
No Arthrobacter citreus ATA crystal structure is deposited in the PDB as of 2024.
3WWH is the closest available homologue (same genus, same fold type IV, same
(R)-selectivity, same catalytic Lys-PLP internal aldimine architecture).

Reference: Savile et al. Science 2010, DOI: 10.1126/science.1188934

Usage:
  python3 src/run_tunnelscan_ata117.py

IMPORTANT: scan predictions are UNCALIBRATED for ATA-117. No mutant KIE data
is available in Savile et al. 2010 or the public literature. See
calibration_data.ATA117_CALIBRATION for full disclosure.
"""

import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tunnel_scan import run_scan, download_pdb, ATA117_CONFIG
from tunnel_score import compute_delta_r_max, ALPHA_H


def main():
    structures_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'structures')
    os.makedirs(structures_dir, exist_ok=True)

    pdb_path = os.path.join(structures_dir, f'{ATA117_CONFIG.pdb_id}.pdb')

    if not os.path.exists(pdb_path):
        print(f"Downloading {ATA117_CONFIG.pdb_id}...")
        try:
            pdb_path = download_pdb(ATA117_CONFIG.pdb_id, structures_dir)
        except Exception as e:
            print(f"Download failed: {e}")
            print(f"Please manually place {ATA117_CONFIG.pdb_id}.pdb in {structures_dir}")
            sys.exit(1)
    else:
        print(f"Using existing structure: {pdb_path}")

    # ── Physical ceiling ──────────────────────────────────────────────────────
    delta_r_max = compute_delta_r_max(
        ATA117_CONFIG.promoting_vibration_cm1,
        ATA117_CONFIG.da_reduced_mass_u,
    )
    kie_ceiling_fold = np.exp(ALPHA_H * delta_r_max)
    kie_ceiling = ATA117_CONFIG.wt_kie_exp * kie_ceiling_fold

    print(f"\nATA-117 TunnelScan")
    print(f"  Structure: {ATA117_CONFIG.pdb_id} (Arthrobacter sp. KNK168 proxy for ATA-117)")
    print(f"  Promoting vibration: {ATA117_CONFIG.promoting_vibration_cm1:.1f} cm-1")
    print(f"  D-A reduced mass:    {ATA117_CONFIG.da_reduced_mass_u:.3f} u")
    print(f"  delta_r_max:         {delta_r_max:.4f} A")
    print(f"  Physical ceiling:    {kie_ceiling:.1f}  "
          f"(WT={ATA117_CONFIG.wt_kie_exp} x {kie_ceiling_fold:.2f}x)")
    print(f"  NOTE: WT KIE = {ATA117_CONFIG.wt_kie_exp} is an estimate — "
          f"no KIE published for ATA-117 (Savile et al. 2010)")

    # ── Run scan ──────────────────────────────────────────────────────────────
    result = run_scan(pdb_path, ATA117_CONFIG, verbose=True)

    # ── Top 10 predictions ────────────────────────────────────────────────────
    print("\nTop 10 predictions:")
    print(f"  {'Mutation':<10} {'KIE_pred':>9}  {'Mechanism':<10}  {'Conf':>6}  {'Novel':>6}")
    for sc in result.all_scores[:10]:
        novel_str = 'NOVEL' if sc.is_novel else ''
        print(f"  {sc.label:<10} {sc.predicted_kie:>9.1f}  "
              f"{sc.dominant_mechanism:<10}  {sc.confidence:>6.2f}  {novel_str:>6}")

    # ── Tunnelling network ────────────────────────────────────────────────────
    if result.tunnelling_network:
        tn = result.tunnelling_network
        print(f"\nNetwork topology:")
        print(f"  lambda_2 (Fiedler value) = {tn.fiedler_value:.4f}")
        print(f"  Omega (robustness)       = {tn.robustness:.4f}")

    # ── Check against Codexis directed-evolution mutations ────────────────────
    try:
        from calibration_data import ATA117_CALIBRATION
        codexis_labels = {m['label'] for m in ATA117_CALIBRATION.get(
            'codexis_evolution_mutations', [])}
        hits = [sc for sc in result.all_scores if sc.label in codexis_labels]
        if hits:
            print(f"\nCodexis directed-evolution mutations in scan results "
                  f"({len(hits)} hits — numbering uncertain, see ATA117_CALIBRATION):")
            for sc in hits:
                print(f"  {sc.label:<10} KIE={sc.predicted_kie:>6.1f}  "
                      f"mech={sc.dominant_mechanism}")
        else:
            print("\nNo Codexis directed-evolution mutations found in scan radius.")
            print("  (Expected: numbering offset between ATA-117 and 3WWH may apply)")
    except ImportError:
        pass

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  SUMMARY")
    print(f"  Residues scanned:    {result.n_residues_found}")
    print(f"  Mutations scored:    {result.n_mutations_scored}")
    print(f"  Novel predictions:   {len(result.novel_scores)}")
    print(f"  KIE-enhancing novel: {len(result.top_enhancing)}")
    print(f"  WARNING: predictions are UNCALIBRATED (no ATA-117 KIE mutant data)")
    print(f"{'='*65}")


if __name__ == '__main__':
    main()
