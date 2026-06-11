"""
Extract a minimal QM active site from an enzyme PDB.

Selects all heavy atoms within inner_radius of the estimated H position,
caps dangling bonds with H link atoms, identifies the outer shell for
position restraints, and writes the model to XYZ.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class ActiveSiteModel:
    symbols: List[str]
    positions: np.ndarray          # (n, 3) Å
    donor_idx: int                 # index in THIS model
    h_idx: int
    acceptor_idx: int
    outer_shell_indices: List[int] # atoms > outer_radius from H; restrained
    ref_positions: np.ndarray      # crystal reference for restraints
    da_distance_crystal: float     # Å
    n_link_atoms: int
    comment: str


def _h_estimate(donor_pos: np.ndarray, acceptor_pos: np.ndarray,
                bond_length: float = 1.09) -> np.ndarray:
    """Place H along D→A axis at bond_length from D."""
    v = acceptor_pos - donor_pos
    d = float(np.linalg.norm(v))
    if d < 1e-6:
        return donor_pos + np.array([bond_length, 0.0, 0.0])
    return donor_pos + bond_length * v / d


def _bonded(sym1: str, p1: np.ndarray, sym2: str, p2: np.ndarray) -> bool:
    d = float(np.linalg.norm(p1 - p2))
    if 'H' in (sym1, sym2):
        return d < 1.3
    if 'S' in (sym1, sym2):
        return d < 2.0
    return d < 1.85


def extract_active_site(
    pdb_path: str,
    donor_chain: str, donor_resnum: int, donor_atom: str,
    acceptor_chain: str, acceptor_resnum: int, acceptor_atom: str,
    inner_radius: float = 4.5,
    outer_radius: float = 3.5,
) -> ActiveSiteModel:
    import ase.io
    full_atoms = ase.io.read(pdb_path, format='proteindatabank')
    all_pos = full_atoms.get_positions()
    all_sym = full_atoms.get_chemical_symbols()
    n_full = len(full_atoms)

    # Parse atom metadata
    res_info: list[tuple[str, int, str, str]] = []
    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith(('ATOM', 'HETATM')):
                continue
            ch = line[21]
            rn = int(line[22:26].strip())
            rname = line[17:20].strip()
            an = line[12:16].strip()
            res_info.append((ch, rn, rname, an))

    if len(res_info) != n_full:
        raise ValueError(f"Atom count mismatch: PDB records={len(res_info)}, ASE atoms={n_full}")

    donor_idx_full = acceptor_idx_full = None
    for i, (ch, rn, rname, an) in enumerate(res_info):
        if ch == donor_chain and rn == donor_resnum and an == donor_atom:
            donor_idx_full = i
        if ch == acceptor_chain and rn == acceptor_resnum and an == acceptor_atom:
            acceptor_idx_full = i

    if donor_idx_full is None:
        raise ValueError(f"Donor not found: chain={donor_chain} res={donor_resnum} atom={donor_atom}")
    if acceptor_idx_full is None:
        raise ValueError(f"Acceptor not found: chain={acceptor_chain} res={acceptor_resnum} atom={acceptor_atom}")

    donor_pos = all_pos[donor_idx_full]
    acceptor_pos = all_pos[acceptor_idx_full]
    da_crystal = float(np.linalg.norm(donor_pos - acceptor_pos))
    h_est = _h_estimate(donor_pos, acceptor_pos)

    # Check if explicit H already exists near h_est (within 0.6 Å)
    h_explicit_idx = None
    for i in range(n_full):
        if all_sym[i] == 'H' and float(np.linalg.norm(all_pos[i] - h_est)) < 0.6:
            h_explicit_idx = i
            break

    # Select atoms within inner_radius of h_est
    sel_full: list[int] = []
    for i in range(n_full):
        if float(np.linalg.norm(all_pos[i] - h_est)) <= inner_radius:
            sel_full.append(i)

    # Always include donor and acceptor
    for must in (donor_idx_full, acceptor_idx_full):
        if must not in sel_full:
            sel_full.append(must)
    if h_explicit_idx is not None and h_explicit_idx not in sel_full:
        sel_full.append(h_explicit_idx)

    # ── Complete any partial aromatic rings ───────────────────────────
    # Cutting through an aromatic ring creates adjacent link atoms that
    # clash (< 1 Å apart) and cause GFN2-xTB SCF failure.  Completing
    # the ring avoids those bonds entirely; the ring → backbone bond is
    # then the only cut, producing a single clean link atom.
    sel_set_tmp = set(sel_full)
    changed = True
    while changed:
        changed = False
        # Build adjacency for all non-H atoms
        heavy = [i for i in range(n_full) if all_sym[i] != 'H']
        for hi in heavy:
            for hj in heavy:
                if hi >= hj:
                    continue
                if not _bonded(all_sym[hi], all_pos[hi], all_sym[hj], all_pos[hj]):
                    continue
                # If exactly one of them is in the selection, check ring
                # membership: detect 5- or 6-membered rings containing this bond
                hi_in = hi in sel_set_tmp
                hj_in = hj in sel_set_tmp
                if hi_in == hj_in:
                    continue  # both in or both out — no partial cut
                # BFS to find short cycle through this bond
                inner_node = hi if hi_in else hj
                outer_node = hj if hi_in else hi
                # Find whether a 5- or 6-membered cycle exists containing both
                # nodes by searching all short paths from outer_node back to inner_node
                in_ring = False
                from collections import deque
                q: deque = deque([(outer_node, [outer_node])])
                while q:
                    cur, path = q.popleft()
                    if len(path) > 6:
                        continue
                    for nb in heavy:
                        if nb == cur:
                            continue
                        if not _bonded(all_sym[cur], all_pos[cur], all_sym[nb], all_pos[nb]):
                            continue
                        if nb == inner_node and len(path) >= 4:
                            in_ring = True
                            break
                        if nb not in path:
                            q.append((nb, path + [nb]))
                    if in_ring:
                        break
                if in_ring and outer_node not in sel_set_tmp:
                    sel_set_tmp.add(outer_node)
                    # Also add the explicit H atoms on outer_node
                    for hi2 in range(n_full):
                        if all_sym[hi2] == 'H' and _bonded('H', all_pos[hi2], all_sym[outer_node], all_pos[outer_node]):
                            sel_set_tmp.add(hi2)
                    changed = True
    sel_full = sorted(sel_set_tmp)

    sel_pos = all_pos[sel_full]
    sel_sym = [all_sym[i] for i in sel_full]
    full_to_sel = {f: s for s, f in enumerate(sel_full)}
    sel_set = set(sel_full)

    # Add H link atoms for each cut heavy-heavy bond
    # Only one link per bond; only when the cut is at a single-bond
    # (ring cuts eliminated above — only backbone/chain cuts remain).
    link_syms: list[str] = []
    link_pos: list[np.ndarray] = []
    for sel_i, full_i in enumerate(sel_full):
        if all_sym[full_i] == 'H':
            continue
        for full_j in range(n_full):
            if full_j in sel_set or all_sym[full_j] == 'H':
                continue
            if _bonded(all_sym[full_i], all_pos[full_i],
                       all_sym[full_j], all_pos[full_j]):
                vec = all_pos[full_j] - all_pos[full_i]
                d = float(np.linalg.norm(vec))
                if d < 0.5:
                    continue
                # Standard ONIOM link atom: r_link = r_i + g * (r_j - r_i)
                # where g = d(C-H)/d(C-C) ≈ 0.71.  Multiply against the FULL
                # bond vector (not unit vector): |r_link - r_i| = 0.71 * d_bond.
                lp = all_pos[full_i] + 0.71 * vec   # do NOT divide by d
                # Sanity check: reject if this link H would be < 1.0 Å from
                # any existing atom (indicates a ring-cut that slipped through)
                clash = any(
                    float(np.linalg.norm(lp - all_pos[existing])) < 1.0
                    for existing in sel_set
                    if existing != full_i
                )
                if not clash:
                    link_syms.append('H')
                    link_pos.append(lp)

    # Build final model
    final_sym = sel_sym + link_syms
    final_pos_list = [sel_pos] + ([np.array(lp)[None, :] for lp in link_pos] if link_pos else [])
    final_pos = np.vstack(final_pos_list)

    n_sel = len(sel_sym)
    donor_new = full_to_sel[donor_idx_full]
    acceptor_new = full_to_sel[acceptor_idx_full]

    # H index: prefer explicit H from PDB, else add one
    if h_explicit_idx is not None and h_explicit_idx in full_to_sel:
        h_new = full_to_sel[h_explicit_idx]
    else:
        # Check link atoms
        h_new = None
        for li, lp in enumerate(link_pos):
            if float(np.linalg.norm(lp - h_est)) < 0.8:
                h_new = n_sel + li
                break
        if h_new is None:
            # Add explicit H
            h_pos_new = _h_estimate(final_pos[donor_new], final_pos[acceptor_new])
            final_sym.append('H')
            final_pos = np.vstack([final_pos, h_pos_new[None, :]])
            h_new = len(final_sym) - 1

    h_pos_new = final_pos[h_new]

    # Outer shell: model atoms > outer_radius from H (excluding D, H, A)
    outer = [i for i, p in enumerate(final_pos)
             if i not in (donor_new, h_new, acceptor_new)
             and float(np.linalg.norm(p - h_pos_new)) > outer_radius]

    comment = (f"D-A crystal distance: {da_crystal:.3f} A | "
               f"{n_sel} heavy + {len(link_pos)} link + H transfer | "
               f"{len(outer)} outer-shell restrained")

    return ActiveSiteModel(
        symbols=final_sym,
        positions=final_pos,
        donor_idx=donor_new,
        h_idx=h_new,
        acceptor_idx=acceptor_new,
        outer_shell_indices=outer,
        ref_positions=final_pos.copy(),
        da_distance_crystal=da_crystal,
        n_link_atoms=len(link_pos),
        comment=comment,
    )


def write_xyz(model: ActiveSiteModel, path: str) -> None:
    lines = [str(len(model.symbols)), model.comment]
    for sym, pos in zip(model.symbols, model.positions):
        lines.append(f"{sym:2s}  {pos[0]:12.6f}  {pos[1]:12.6f}  {pos[2]:12.6f}")
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def model_to_ase(model: ActiveSiteModel):
    import ase
    return ase.Atoms(symbols=model.symbols, positions=model.positions)
