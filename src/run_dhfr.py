import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from pdb_parser import Structure
from elastic_network import build_gnm
from tunnelling_model import bell_correction
from tunnel_scan import run_scan, DHFR_CONFIG, download_pdb
from tunnel_score import TunnelScorer, SUBSTITUTION_CANDIDATES
from network_coupling import find_network_residues
from report import print_quick_summary

structures_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'structures')
pdb_path = os.path.join(structures_dir, '1RX2.pdb')

if not os.path.exists(pdb_path):
    download_pdb('1RX2', structures_dir)

# ── Run standard local scan ───────────────────────────────────────────────────
result = run_scan(pdb_path, DHFR_CONFIG, beta=3.0, verbose=True)
print_quick_summary(result)

# ── Network scan: distal residues coupled via ENM ─────────────────────────────
print("\n[+] Running network coupling scan (distal residues)...")

s   = Structure(pdb_path)
enm = build_gnm(s, cutoff=7.5)
wt  = bell_correction(13.4, 1184.0, 3.338, experimental_KIE=6.8)

donor    = np.array([31.149, 45.947, 14.124])
acceptor = np.array([31.895, 44.651, 11.140])

local_keys = [(sc.chain, sc.residue_number)
              for sc in result.all_scores]
local_keys  = list(set(local_keys))
catalytic   = {('A', 161), ('A', 164)}

network = find_network_residues(
    s, enm, donor, acceptor,
    local_scan_keys=local_keys,
    catalytic_keys=catalytic,
    network_threshold=0.25,
    verbose=True
)

scorer = TunnelScorer(
    s, enm, wt, beta=3.0, gamma=1.0,
    donor_chain='A', donor_resnum=164, donor_atom='C4N',
    acceptor_chain='A', acceptor_resnum=161, acceptor_atom='C6',
)

network_scores = []
for nr in network:  # all network residues
    for new_aa in SUBSTITUTION_CANDIDATES.get(nr.residue.name, ['ALA']):
        if new_aa == nr.residue.name:
            continue
        sc = scorer.score_mutation(nr.residue, new_aa, 'flanking',
                                   nr.dist_to_axis)
        # Override: network residues have no static geometric component
        # Their effect is purely dynamic (mode coupling)
        network_scores.append(sc)

network_scores.sort(key=lambda x: x.predicted_kie, reverse=True)

print(f"\n  Top 10 network (distal) mutations:")
print(f"  {'Label':<12} {'KIE':>7} {'Δdyn':>7} {'dist':>7} {'novel'}")
print(f"  {'-'*50}")
for sc in network_scores[:10]:
    nr_match = next((n for n in network if n.number == sc.residue_number), None)
    dist = nr_match.dist_to_midpoint if nr_match else 0
    print(f"  {sc.label:<12} {sc.predicted_kie:>7.1f} "
          f"{sc.dynamic_delta:>+7.2f} {dist:>7.1f}A  "
          f"{'★NOVEL★' if sc.is_novel else ''}")

print(f"\n  Literature check (should appear in network scan):")
for resnum, label in [(121,'G121'),(42,'M42')]:
    # M42 is in local scan
    local_match = [sc for sc in result.all_scores if sc.residue_number == resnum]
    network_match = [sc for sc in network_scores if sc.residue_number == resnum]
    all_match = local_match + network_match
    if all_match:
        for sc in all_match[:2]:
            src = "local" if sc in local_match else "network"
            print(f"  {sc.label} [{src}]: KIE={sc.predicted_kie:.1f}  "
                  f"dyn={sc.dynamic_delta:+.2f}")
    else:
        print(f"  {label}: not found in either scan")
