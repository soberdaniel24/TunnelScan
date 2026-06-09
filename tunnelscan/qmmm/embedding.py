from __future__ import annotations
import numpy as np

_ELEMENT_CHARGES = {
    "C": -0.1, "N": -0.3, "O": -0.4, "H": 0.1, "S": -0.2,
}


def _get_charge(symbol: str) -> float:
    return _ELEMENT_CHARGES.get(symbol, 0.0)


def _bonded_within_n(atoms, start: int, n_bonds: int) -> set:
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    n = len(atoms)

    def bonded(i, j):
        d = np.linalg.norm(positions[i] - positions[j])
        sym_i, sym_j = symbols[i], symbols[j]
        max_b = 1.9 if "S" in (sym_i, sym_j) else 1.7
        return d < max_b

    visited = {start}
    frontier = {start}
    for _ in range(n_bonds):
        next_f = set()
        for fi in frontier:
            for j in range(n):
                if j not in visited and bonded(fi, j):
                    visited.add(j)
                    next_f.add(j)
        frontier = next_f
    visited.discard(start)
    return visited


def build_point_charges(atoms, mm_indices: list[int],
                        excluded_indices: set) -> np.ndarray:
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()

    # Find atoms within 2 bonds of any boundary (QM-adjacent MM atoms)
    qm_set = set(range(len(atoms))) - set(mm_indices)
    boundary_mm = set()
    for qi in qm_set:
        for mi in mm_indices:
            d = np.linalg.norm(positions[qi] - positions[mi])
            sym_q = symbols[qi]
            sym_m = symbols[mi]
            max_b = 1.9 if "S" in (sym_q, sym_m) else 1.7
            if d < max_b:
                # mm atom bonded to QM atom — find 2-bond neighborhood
                near = _bonded_within_n(atoms, mi, 2)
                boundary_mm.update(near & set(mm_indices))
                boundary_mm.add(mi)

    result = []
    for mi in mm_indices:
        if mi in excluded_indices:
            continue
        pos = positions[mi]
        q = _get_charge(symbols[mi])
        if mi in boundary_mm:
            q *= 0.8
        result.append([pos[0], pos[1], pos[2], q])

    if result:
        return np.array(result)
    return np.zeros((0, 4))
