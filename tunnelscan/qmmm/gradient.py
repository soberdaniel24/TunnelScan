from __future__ import annotations
import numpy as np


def project_link_forces(forces: np.ndarray, boundary_pairs: list[tuple[int, int]],
                        link_positions: np.ndarray, atoms) -> np.ndarray:
    """Distribute link atom forces back onto QM boundary and adjacent MM atoms."""
    forces = forces.copy()
    positions = atoms.get_positions()
    d_link = 1.0  # Å

    for k, (qi, mi) in enumerate(boundary_pairs):
        r_q = positions[qi]
        r_m = positions[mi]
        d_bond = np.linalg.norm(r_m - r_q)
        if d_bond < 1e-10:
            continue
        ratio = d_link / d_bond
        # Link atom force
        f_link = forces[qi].copy()  # stored at qm index from QM calc
        # Redistribute: F_qm_link_contribution = (1 - ratio)*F_link
        #               F_mm = ratio * F_link
        # The link force was already counted in forces[qi] from QM calculation
        # We now also add the MM projected contribution
        forces[qi] += (1.0 - ratio) * f_link
        forces[mi] += ratio * f_link

    return forces
