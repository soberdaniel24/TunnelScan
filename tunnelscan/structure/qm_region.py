from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class QMRegion:
    qm_indices: list[int]
    mm_indices: list[int]
    boundary_pairs: list[tuple[int, int]]
    link_positions: np.ndarray


def _find_aromatic_rings(atoms, indices_set):
    """Detect simple 5/6-membered rings using connectivity; return list of rings (sets)."""
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    n = len(atoms)
    # Build adjacency for all atoms
    adj = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(positions[i] - positions[j])
            sym_i, sym_j = symbols[i], symbols[j]
            max_bond = 1.9 if ("S" in (sym_i, sym_j)) else 1.7
            if d < max_bond:
                adj[i].append(j)
                adj[j].append(i)

    rings = []
    visited_rings = set()

    def dfs(start, current, path, depth):
        if depth > 6:
            return
        for nb in adj[current]:
            if nb == start and depth >= 4:
                key = frozenset(path)
                if key not in visited_rings:
                    visited_rings.add(key)
                    rings.append(set(path))
            elif nb not in path:
                dfs(start, nb, path + [nb], depth + 1)

    aromatic_elements = {"C", "N", "O", "S"}
    for i in range(n):
        if symbols[i] in aromatic_elements:
            dfs(i, i, [i], 1)

    return [r for r in rings if len(r) in (5, 6)]


def select_qm_region(atoms, topology, donor_idx: int, acceptor_idx: int,
                     h_idx: int, cutoff: float = 4.5) -> QMRegion:
    positions = atoms.get_positions()
    n = len(atoms)

    seed = {donor_idx, acceptor_idx, h_idx}

    # Expand by distance
    qm_set = set(seed)
    for i in range(n):
        for ref in seed:
            d = np.linalg.norm(positions[i] - positions[ref])
            if d <= cutoff:
                qm_set.add(i)
                break

    # Always include non-water HETATM atoms (cofactors)
    qm_set |= topology.hetatm_indices

    # Detect cut aromatic rings and complete them
    rings = _find_aromatic_rings(atoms, qm_set)
    changed = True
    while changed:
        changed = False
        for ring in rings:
            partial = ring & qm_set
            if partial and partial != ring:
                qm_set |= ring
                changed = True

    qm_indices = sorted(qm_set)
    mm_indices = [i for i in range(n) if i not in qm_set]

    # Find boundary pairs: QM atom bonded to MM atom
    symbols = atoms.get_chemical_symbols()
    boundary_pairs = []
    for qi in qm_indices:
        for mi in mm_indices:
            d = np.linalg.norm(positions[qi] - positions[mi])
            sym_q, sym_m = symbols[qi], symbols[mi]
            max_bond = 1.9 if ("S" in (sym_q, sym_m)) else 1.7
            if d < max_bond:
                boundary_pairs.append((qi, mi))

    # Place link atoms
    link_positions = np.zeros((len(boundary_pairs), 3))
    for k, (qi, mi) in enumerate(boundary_pairs):
        r_q = positions[qi]
        r_m = positions[mi]
        diff = r_m - r_q
        dist = np.linalg.norm(diff)
        if dist > 1e-10:
            link_positions[k] = r_q + 1.0 * diff / dist
        else:
            link_positions[k] = r_q

    return QMRegion(
        qm_indices=qm_indices,
        mm_indices=mm_indices,
        boundary_pairs=boundary_pairs,
        link_positions=link_positions,
    )
