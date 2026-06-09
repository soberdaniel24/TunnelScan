from __future__ import annotations
import numpy as np


class MockEngine:
    """Configurable mock engine for testing."""

    def __init__(self, potential="harmonic", n_atoms=3, k=1.0, x0=0.0,
                 donor_idx=0, h_idx=1, acceptor_idx=2,
                 atom_positions=None, mass=1.0):
        self.potential = potential
        self.n_atoms = n_atoms
        self.k = k
        self.x0 = x0
        self.donor_idx = donor_idx
        self.h_idx = h_idx
        self.acceptor_idx = acceptor_idx
        self.mass = mass

        if atom_positions is not None:
            self._positions = np.array(atom_positions, dtype=float)
        else:
            self._positions = np.zeros((n_atoms, 3))
            # Default D-H-A linear molecule
            if n_atoms >= 3:
                self._positions[0] = [0.9, 0.0, 0.0]  # Donor (DH = 0.9 Å)
                self._positions[1] = [0.0, 0.0, 0.0]  # H
                self._positions[2] = [-2.5, 0.0, 0.0] # Acceptor (HA = 2.5 Å)

        import ase
        symbols = ["N", "H", "O"] if n_atoms == 3 else ["H"] * n_atoms
        self.atoms = ase.Atoms(symbols[:n_atoms], positions=self._positions)

    def energy_and_forces(self):
        pos = self._positions
        e = 0.0
        f = np.zeros_like(pos)

        if self.potential == "harmonic":
            x = pos[self.h_idx, 0]
            e = 0.5 * self.k * (x - self.x0) ** 2
            f[self.h_idx, 0] = -self.k * (x - self.x0)

        elif self.potential == "double_well":
            # V(x) = (x^2 - 1)^2; clip x to avoid float overflow at large displacements
            x = np.clip(pos[self.h_idx, 0], -10.0, 10.0)
            e = (x**2 - 1.0)**2
            dV = 4.0 * x * (x**2 - 1.0)
            f[self.h_idx, 0] = -dV

        elif self.potential == "dha_transfer":
            # Morse-like: V = A*(d_DH - r0)^2 + A*(d_HA - r0)^2
            r_D = pos[self.donor_idx]
            r_H = pos[self.h_idx]
            r_A = pos[self.acceptor_idx]
            dDH = np.linalg.norm(r_D - r_H)
            dHA = np.linalg.norm(r_H - r_A)
            e = self.k * (dDH - 1.0)**2 + self.k * (dHA - 1.0)**2
            e_dDH = 2.0 * self.k * (dDH - 1.0) / (dDH + 1e-12)
            e_dHA = 2.0 * self.k * (dHA - 1.0) / (dHA + 1e-12)
            f[self.h_idx] = e_dDH * (r_D - r_H) + e_dHA * (r_A - r_H)
            f[self.donor_idx] = -e_dDH * (r_D - r_H)
            f[self.acceptor_idx] = -e_dHA * (r_A - r_H)

        return float(e), f

    def update_positions(self, positions: np.ndarray):
        self._positions = np.array(positions)
        self.atoms.set_positions(positions)

    def get_masses(self):
        return np.array([14.0, self.mass, 16.0][:self.n_atoms])
