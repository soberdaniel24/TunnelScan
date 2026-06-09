from __future__ import annotations
import numpy as np


def place_link_atoms(atoms, boundary_pairs: list[tuple[int, int]],
                     link_distance: float = 1.0) -> np.ndarray:
    positions = atoms.get_positions()
    link_positions = np.zeros((len(boundary_pairs), 3))
    for k, (qi, mi) in enumerate(boundary_pairs):
        r_q = positions[qi]
        r_m = positions[mi]
        diff = r_m - r_q
        dist = np.linalg.norm(diff)
        if dist > 1e-10:
            link_positions[k] = r_q + link_distance * diff / dist
        else:
            link_positions[k] = r_q
    return link_positions


def get_excluded_mm_charges(atoms, qm_indices: list[int], topology) -> set:
    """Return MM indices directly bonded to QM region (excluded from electrostatic embedding)."""
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    qm_set = set(qm_indices)
    n = len(atoms)
    excluded = set()
    for mm_idx in range(n):
        if mm_idx in qm_set:
            continue
        for qi in qm_indices:
            d = np.linalg.norm(positions[mm_idx] - positions[qi])
            sym_q = symbols[qi]
            sym_m = symbols[mm_idx]
            max_bond = 1.9 if ("S" in (sym_q, sym_m)) else 1.7
            if d < max_bond:
                excluded.add(mm_idx)
                break
    return excluded
