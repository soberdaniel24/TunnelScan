"""
Fixed QM/MM KIE pipeline — AADH WT and DHFR WT (G121V comparison).

Active site model approach:
  Instead of radius-based extraction (which cuts through aromatic rings and
  creates link atom clashes), we build EXPLICIT MINIMAL MODELS using only
  the crystal coordinates of the key sp3/sp2 atoms directly involved in
  the transfer reaction.

  AADH: CB (donor, sp3 CH2) + Asp128 side chain (CG/OD1/OD2)
        13 atoms, no rings, fully GFN2-xTB compatible.
  DHFR: C4 (donor, NADH) + C6 (acceptor, folate) vicinity
        Similar minimal sp2 model.

Fixes applied:
  1. Minimal model: no aromatic ring cuts, no link atom clashes
  2. ConstrainedEngine: DA soft wall (cutoff = crystal_DA + 0.5 Å)
     + position restraints on outer shell (k=100 kcal/mol/Å²)
  3. Staged equilibration: ASE LBFGS + 100K→200K→300K heating
  4. dt = 0.5 fs
  5. Energy drift rejection (>5 kcal/mol rollback)

Usage:
    python scripts/run_qmmm_kie.py
"""
from __future__ import annotations
import os, sys, time, math, logging, warnings, tempfile

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="  %(message)s")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

try:
    from tblite.ase import TBLite
    QM_BACKEND = "GFN2-xTB (tblite)"
except ImportError:
    QM_BACKEND = "ASE EMT (tblite not found)"
print(f"QM backend: {QM_BACKEND}\n")

EV_TO_KCAL = 23.0605


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Atom coordinate parser
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _parse_pdb_atoms(pdb_path):
    """Return dict (chain, resnum, aname) -> np.ndarray(3) of crystal coordinates."""
    coords = {}
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith(('ATOM', 'HETATM')):
                continue
            ch = line[21]
            try:
                rn = int(line[22:26])
                an = line[12:16].strip()
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                coords[(ch, rn, an)] = np.array([x, y, z])
            except (ValueError, IndexError):
                continue
    return coords


def _tetra_h_positions(center, b1, b2, bond_len=1.09):
    """Two tetrahedral H positions on sp3 center given two bonded partners."""
    v1 = b1 - center; v1 /= np.linalg.norm(v1)
    v2 = b2 - center; v2 /= np.linalg.norm(v2)
    perp = np.cross(v1, v2)
    if np.linalg.norm(perp) < 1e-6:
        perp = np.array([1.0, 0.0, 0.0])
    perp /= np.linalg.norm(perp)
    bisect = -(v1 + v2)
    bn = np.linalg.norm(bisect)
    if bn < 1e-6:
        bisect = perp
    else:
        bisect /= bn
    H1 = center + bond_len * (bisect + 0.816 * perp) / np.linalg.norm(bisect + 0.816 * perp)
    H2 = center + bond_len * (bisect - 0.816 * perp) / np.linalg.norm(bisect - 0.816 * perp)
    return H1, H2


def _write_minimal_pdb(symbols, positions, path):
    lines = ["REMARK  TunnelScan active site model\n"]
    for i, (sym, pos) in enumerate(zip(symbols, positions)):
        line = (f"ATOM  {i+1:5d}  {sym:<3s} LIG A   1    "
                f"{pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}"
                f"  1.00  0.00          {sym:>2s}\n")
        lines.append(line)
    lines.append("END\n")
    with open(path, "w") as f:
        f.writelines(lines)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Build AADH minimal model
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def build_aadh_model(pdb_2agw):
    """
    13-atom AADH active site: tryptamine CB+CA stub + Asp128 side chain.

    Donor:    CB of tryptamine (chain D, res 3001) — sp3 CH2 carbon
    Acceptor: OD2 of Asp128 (chain D, res 128)
    H:        Estimated on CB, pointing toward OD2

    Atoms:
      CA (tryptamine), CB (donor), 2×H on CA, H_transfer, H_other,
      link_H_ring (CB→CG cap), link_H_amine (CA→N1 cap),
      Asp_CG, Asp_OD1, Asp_OD2, link_H_Asp_backbone
    """
    c = _parse_pdb_atoms(pdb_2agw)

    p_CA     = c[('D', 3001, 'CA')]
    p_CB     = c[('D', 3001, 'CB')]
    p_CG_t   = c[('D', 3001, 'CG')]   # tryptamine ring C (not included)
    p_N1     = c.get(('D', 3001, 'N1'), c.get(('D', 3001, 'N'), p_CA + np.array([0,0,1.47])))
    p_Asp_CB = c[('D', 128, 'CB')]
    p_Asp_CG = c[('D', 128, 'CG')]
    p_Asp_OD1= c[('D', 128, 'OD1')]
    p_Asp_OD2= c[('D', 128, 'OD2')]

    da_crystal = float(np.linalg.norm(p_CB - p_Asp_OD2))

    # Two H positions on CB (sp3, bonded to CA, CG, H_a, H_b)
    H_CB_a, H_CB_b = _tetra_h_positions(p_CB, p_CA, p_CG_t)
    # H_transfer is the one closer to OD2
    if np.linalg.norm(H_CB_a - p_Asp_OD2) < np.linalg.norm(H_CB_b - p_Asp_OD2):
        H_transfer, H_other = H_CB_a, H_CB_b
    else:
        H_transfer, H_other = H_CB_b, H_CB_a

    # Two H on CA (sp3, bonded to N1, CB)
    H_CA_a, H_CA_b = _tetra_h_positions(p_CA, p_N1, p_CB)

    # ONIOM link atoms (g = 0.71, applied to FULL bond vector):
    link_ring   = p_CB     + 0.71 * (p_CG_t   - p_CB)      # CB→CG ring cap
    link_amine  = p_CA     + 0.71 * (p_N1     - p_CA)       # CA→N1 amine cap
    link_Asp_bb = p_Asp_CG + 0.71 * (p_Asp_CB - p_Asp_CG)  # Asp_CG→Asp_CB cap

    # Check all link atoms are > 1.0 Å from all heavy atoms
    heavy_pos = [p_CA, p_CB, p_Asp_CG, p_Asp_OD1, p_Asp_OD2]
    for lp, name in [(link_ring, 'link_ring'), (link_amine, 'link_amine'),
                     (link_Asp_bb, 'link_Asp_bb')]:
        dists = [np.linalg.norm(lp - hp) for hp in heavy_pos]
        if min(dists) < 0.9:
            print(f"  WARNING: {name} link atom is {min(dists):.2f} Å from a heavy atom")

    #                0   1    2       3       4           5          6          7
    symbols  = ['C', 'C', 'H',    'H',    'H',        'H',       'H',       'H',
    #                8          9          10         11
                 'C',       'O',       'O',       'H']
    positions = np.array([
        p_CA,                  # 0: CA (tryptamine)
        p_CB,                  # 1: CB = donor
        H_CA_a, H_CA_b,        # 2,3: H on CA
        H_transfer, H_other,   # 4: H_transfer = transferring H; 5: other H on CB
        link_ring, link_amine, # 6: link H caps ring side of CB; 7: caps amine side of CA
        p_Asp_CG,              # 8: CG of Asp128
        p_Asp_OD1,             # 9: OD1 of Asp128
        p_Asp_OD2,             # 10: OD2 of Asp128 = acceptor
        link_Asp_bb,           # 11: link H caps backbone side of CG
    ])

    donor_idx    = 1   # CB
    h_idx        = 4   # H_transfer
    acceptor_idx = 10  # OD2

    # Outer shell: atoms farther than 3.5 Å from H_transfer (except D, H, A)
    outer = [i for i, p in enumerate(positions)
             if i not in (donor_idx, h_idx, acceptor_idx)
             and float(np.linalg.norm(p - H_transfer)) > 3.5]

    return dict(symbols=symbols, positions=positions,
                donor_idx=donor_idx, h_idx=h_idx, acceptor_idx=acceptor_idx,
                outer_shell_indices=outer, da_distance_crystal=da_crystal,
                label="AADH WT — tryptamine CB + Asp128 (13 atoms, no rings)")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Build DHFR minimal model
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def build_dhfr_model(pdb_1rx2):
    """
    Minimal DHFR active site: NADH C4 (donor) + folate C6 (acceptor) stub.
    Both are sp2 carbons; use N and C atoms bonded to them as context.
    """
    c = _parse_pdb_atoms(pdb_1rx2)

    # NADH/NADPH (residue 164, chain A): C4N is the donor
    # Folate (residue 161, chain A): C6 is the acceptor
    donor_key = ('A', 164, 'C4N')
    if donor_key not in c:
        donor_key = ('A', 164, 'C4')
    accept_key = ('A', 161, 'C6')

    if donor_key not in c or accept_key not in c:
        return None

    p_D = c[donor_key]
    p_A = c[accept_key]
    da_crystal = float(np.linalg.norm(p_D - p_A))

    # Estimate H position on donor sp2 carbon
    h_est = p_D + 1.09 * (p_A - p_D) / (np.linalg.norm(p_A - p_D) + 1e-12)

    # Find a few context atoms bonded to donor and acceptor
    # N4N bonded to C4N in nicotinamide ring
    context_keys_D = [('A', 164, 'N1'), ('A', 164, 'C3'), ('A', 164, 'C5')]
    context_keys_A = [('A', 161, 'C5'), ('A', 161, 'N5'), ('A', 161, 'N8')]

    d_ctx = [c[k] for k in context_keys_D if k in c][:1]
    a_ctx = [c[k] for k in context_keys_A if k in c][:1]

    # Build minimal model
    syms  = ['C', 'H', 'C']  # donor, H, acceptor
    pos   = [p_D.copy(), h_est.copy(), p_A.copy()]
    for p in d_ctx:
        syms.append('C'); pos.append(p.copy())
    for p in a_ctx:
        syms.append('C'); pos.append(p.copy())

    # Add link H for any C context atoms (simplified: one H per context C)
    for p in d_ctx:
        vec = p - p_D
        syms.append('H'); pos.append(p_D + 0.71 * vec)
    for p in a_ctx:
        vec = p - p_A
        syms.append('H'); pos.append(p_A + 0.71 * vec)

    positions = np.array(pos)
    donor_idx = 0; h_idx = 1; acceptor_idx = 2
    outer = [i for i, p in enumerate(positions)
             if i not in (donor_idx, h_idx, acceptor_idx)
             and float(np.linalg.norm(p - h_est)) > 2.5]

    return dict(symbols=syms, positions=positions,
                donor_idx=donor_idx, h_idx=h_idx, acceptor_idx=acceptor_idx,
                outer_shell_indices=outer, da_distance_crystal=da_crystal,
                label="DHFR WT (G121V active site = WT; G121 not in QM region)")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GFN2-xTB validation of a model
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def validate_model_sp(model):
    """Single-point GFN2-xTB, return (E_kcal, Fmax_kcal, ok)."""
    import ase
    from ase.constraints import FixAtoms
    atoms = ase.Atoms(symbols=model['symbols'], positions=model['positions'])
    atoms.calc = TBLite(method='GFN2-xTB', verbosity=0)
    if model['outer_shell_indices']:
        atoms.set_constraint(FixAtoms(indices=model['outer_shell_indices']))
    try:
        e = atoms.get_potential_energy() * EV_TO_KCAL
        fmax = float(np.max(np.abs(atoms.get_forces()))) * EV_TO_KCAL
        return e, fmax, e < 0 and math.isfinite(e)
    except Exception as ex:
        return float('nan'), float('nan'), False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Run fixed pipeline on a model dict
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def run_model_pipeline(model, experimental_kie=None, qm_charge=0):
    from tunnelscan.pipeline.runner import run_tunnelscan

    da_crystal = model['da_distance_crystal']
    da_cutoff  = da_crystal + 0.5   # drift prevention only

    restraints = [(i, model['positions'][i].copy(), 100.0)
                  for i in model['outer_shell_indices']]

    with tempfile.TemporaryDirectory() as tmpdir:
        pdb_tmp = os.path.join(tmpdir, 'model.pdb')
        out_dir = os.path.join(tmpdir, 'out')
        _write_minimal_pdb(model['symbols'], model['positions'], pdb_tmp)

        t0 = time.time()
        try:
            result = run_tunnelscan(
                pdb_path=pdb_tmp,
                donor_atom_index=model['donor_idx'],
                acceptor_atom_index=model['acceptor_idx'],
                hydrogen_atom_index=model['h_idx'],
                temperature=300.0,
                n_beads=8,
                n_tps_paths=10,
                run_convergence_check=False,
                experimental_kie=experimental_kie,
                output_dir=out_dir,
                fast_test=True,
                da_constraint=True,
                da_k=50.0, da_d0=2.7, da_cutoff=da_cutoff,
                position_restraints=restraints,
                dt=0.5,
                staged_equilibration=True,
                n_centroid_steps_override=300,
                qm_charge=qm_charge,
            )
            elapsed = time.time() - t0
            return result, elapsed
        except RuntimeError as ex:
            print(f"  Pipeline RuntimeError: {ex}")
            return None, time.time() - t0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("=" * 65)
print("  TunnelScan QM/MM KIE — fixed pipeline")
print("=" * 65)

# ── AADH ─────────────────────────────────────────────────────────────────────
print("\n" + "━"*65)
print("  AADH WT — minimal 13-atom model (CB + Asp128 side chain)")
print("━"*65)

aadh_model = build_aadh_model("data/structures/2AGW.pdb")
from collections import Counter
print(f"  Atoms: {len(aadh_model['symbols'])}  "
      f"composition: {dict(Counter(aadh_model['symbols']))}")
print(f"  D-A crystal: {aadh_model['da_distance_crystal']:.3f} Å")
print(f"  Outer shell (restrained): {len(aadh_model['outer_shell_indices'])}")
print(f"  {aadh_model['label']}")

# GFN2-xTB validation
e_sp, fmax_sp, sp_ok = validate_model_sp(aadh_model)
print(f"  GFN2-xTB SP: E={e_sp:.1f} kcal/mol  Fmax={fmax_sp:.1f} kcal/mol/Å  "
      f"valid={sp_ok}")

if sp_ok:
    # AADH active site: deprotonated Asp128 carboxylate → charge = -1
    print("  Running pipeline (qm_charge=-1, deprotonated Asp128)...")
    result_aadh, elapsed_aadh = run_model_pipeline(aadh_model, experimental_kie=55.0,
                                                    qm_charge=-1)
    if result_aadh:
        zpe = result_aadh.zpe
        kie = result_aadh.kie_theoretical
        ddg = result_aadh.delta_delta_G
        zpe_ok = math.isfinite(zpe) and zpe != 0.0
        kie_ok = math.isfinite(kie) and kie > 1.0
        print(f"\n  Wall time:      {elapsed_aadh:.0f} s")
        print(f"  Post-equil E:   reported in logs above")
        print(f"  Classical ΔG‡:  see logs")
        print(f"  Centroid-H ΔG‡: see logs")
        print(f"  ZPE:            {zpe:.4f} kcal/mol  {'PHYSICAL ✓' if zpe_ok else 'ZERO ✗'}")
        print(f"  ΔΔG:            {ddg:.4f} kcal/mol")
        print(f"  KIE:            {kie:.4f}  {'KIE > 1 ✓' if kie_ok else 'KIE ≤ 1 ✗'}")
        print(f"  Classification: {result_aadh.classification}")
        print(f"  Deuterium check: {result_aadh.deuterium_check_passed}")
        print(f"  Flags:          {result_aadh.flags}")
    else:
        result_aadh = None
        print("  Pipeline failed (RuntimeError — see above)")
else:
    print("  SP invalid — skipping pipeline run")
    result_aadh = None
    elapsed_aadh = 0

# ── DHFR ─────────────────────────────────────────────────────────────────────
print("\n" + "━"*65)
print("  DHFR WT (G121V comparison)")
print("━"*65)

dhfr_model = build_dhfr_model("data/structures/1RX2.pdb")
result_dhfr = None
elapsed_dhfr = 0

if dhfr_model:
    print(f"  Atoms: {len(dhfr_model['symbols'])}  "
          f"composition: {dict(Counter(dhfr_model['symbols']))}")
    print(f"  D-A crystal: {dhfr_model['da_distance_crystal']:.3f} Å")
    print(f"  {dhfr_model['label']}")
    print()
    print("  G121V note: G121 is ~19 Å from active site → NOT in 4.5 Å QM model.")
    print("  DHFR WT and G121V active sites are structurally identical.")
    print("  Any WT vs G121V KIE difference requires full protein dynamics")
    print("  (captured by the ENM BETA term in the scoring model, not QM/MM).")

    e_sp_d, fmax_sp_d, sp_ok_d = validate_model_sp(dhfr_model)
    print(f"  GFN2-xTB SP: E={e_sp_d:.1f} kcal/mol  Fmax={fmax_sp_d:.1f} kcal/mol/Å  "
          f"valid={sp_ok_d}")

    if sp_ok_d:
        print("  Running pipeline...")
        result_dhfr, elapsed_dhfr = run_model_pipeline(dhfr_model, experimental_kie=6.8)
else:
    print("  DHFR: could not locate C4N or C6 in 1RX2 — check HETATM atom names")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FINAL SUMMARY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 65)
print("  FINAL SUMMARY")
print("=" * 65)

def _row(label, r):
    if r is None:
        print(f"  {label:<30s}  FAILED")
        return False
    zpe_ok = math.isfinite(r.zpe) and r.zpe != 0.0
    kie_ok = math.isfinite(r.kie_theoretical) and r.kie_theoretical > 1.0
    phys = zpe_ok and kie_ok
    print(f"  {label:<30s}  ZPE={r.zpe:.4f}  ΔΔG={r.delta_delta_G:.4f}  "
          f"KIE={r.kie_theoretical:.4f}  physical={'YES' if phys else 'NO'}")
    return phys

aadh_phys = _row("AADH WT", result_aadh)
dhfr_phys = _row("DHFR WT", result_dhfr)

print()
print("  Scoring model reference:")
print("   DHFR WT = 6.8  |  G121V = 4.897 (BELOW WT — dynamic penalty)")
print("   QM/MM: WT ≈ G121V (G121 not in QM region; distal effect via ENM)")

print()
print("  Fixes applied:")
print("   1. Minimal explicit model: CB + Asp128 side chain (13 atoms, no rings)")
print("   2. Link atoms: ONIOM full-vector formula (lp = r_i + 0.71*(r_j - r_i))")
print("   3. ASE LBFGS minimisation via _EngineAsCalc adapter")
print("   4. DA soft wall: cutoff = crystal_DA + 0.5 Å (drift only)")
print("   5. Position restraints on outer shell (k=100 kcal/mol/Å²)")
print("   6. dt = 0.5 fs; energy drift rejection (>5 kcal/mol rollback)")
