"""
ConstrainedEngine — transparent wrapper adding two physical constraints
on top of any base QM or QM/MM engine:

  1. D-A soft wall: harmonic restraint (k, d0) that activates when
     d(donor, acceptor) > da_cutoff, pulling toward d0. Keeps the
     transferring geometry near the tunnelling window.

  2. Position restraints: harmonic springs (k_pos) anchoring chosen
     atoms to their reference positions. Used for the outer shell of
     an active-site model so they cannot drift into unphysical geometry.
"""
from __future__ import annotations
import numpy as np
from typing import List, Optional, Tuple


class ConstrainedEngine:
    def __init__(
        self,
        base_engine,
        donor_idx: int,
        acceptor_idx: int,
        da_k: float = 50.0,          # kcal/mol/Å²
        da_d0: float = 2.7,           # Å target D-A distance
        da_cutoff: float = 3.2,       # Å threshold above which wall activates
        position_restraints: Optional[List[Tuple[int, np.ndarray, float]]] = None,
    ):
        self.base = base_engine
        self.atoms = base_engine.atoms
        self.donor_idx = donor_idx
        self.acceptor_idx = acceptor_idx
        self.da_k = da_k
        self.da_d0 = da_d0
        self.da_cutoff = da_cutoff
        # Each entry: (atom_index, ref_position_array, k_pos)
        self.position_restraints: List[Tuple[int, np.ndarray, float]] = (
            list(position_restraints) if position_restraints else []
        )

    def energy_and_forces(self) -> tuple[float, np.ndarray]:
        e, f = self.base.energy_and_forces()
        pos = self.atoms.get_positions()
        f = np.array(f, dtype=float)

        # ── D-A soft wall ──────────────────────────────────────────────
        r_D = pos[self.donor_idx]
        r_A = pos[self.acceptor_idx]
        d_DA = float(np.linalg.norm(r_D - r_A)) + 1e-14
        if d_DA > self.da_cutoff:
            delta = d_DA - self.da_d0
            e += 0.5 * self.da_k * delta * delta
            grad_mag = self.da_k * delta / d_DA
            unit = (r_D - r_A)
            f[self.donor_idx]   -= grad_mag * unit
            f[self.acceptor_idx] += grad_mag * unit

        # ── Position restraints ────────────────────────────────────────
        for idx, ref, k_pos in self.position_restraints:
            disp = pos[idx] - np.asarray(ref)
            e += 0.5 * k_pos * float(np.dot(disp, disp))
            f[idx] -= k_pos * disp

        return float(e), f

    def update_positions(self, positions: np.ndarray) -> None:
        self.base.update_positions(positions)
