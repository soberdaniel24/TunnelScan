import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tunnel_scan import run_scan, DHFR_CONFIG, download_pdb
from report import print_quick_summary

structures_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'structures')
pdb_path = os.path.join(structures_dir, '1RX2.pdb')

if not os.path.exists(pdb_path):
    download_pdb('1RX2', structures_dir)

result = run_scan(pdb_path, DHFR_CONFIG, beta=3.0, verbose=True)
print_quick_summary(result)

# Show key residues from literature
print("\nLiterature check - these residues should appear near D-A axis:")
print("  G121 - distal dynamic network (19A from active site)")
print("  M42  - distal dynamic network (15A from active site)")
print("  I14  - active site, backs H-donor")
print()
known = {'G121', 'M42', 'I14', 'F125'}
for sc in result.all_scores:
    label_root = ''.join(c for c in sc.label if not c.isdigit())[:-1]
    num = ''.join(c for c in sc.label if c.isdigit())
    if num in {'121', '42', '14', '125'}:
        print(f"  {sc.label}: KIE={sc.predicted_kie:.1f} mech={sc.dominant_mechanism}")
