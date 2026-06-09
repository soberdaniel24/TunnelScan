from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class ProductionTrajectory:
    positions: np.ndarray   # (n_frames, n_atoms, 3) Å
    velocities: np.ndarray  # (n_frames, n_atoms, 3) Å/fs
    xi: np.ndarray          # (n_frames,) Å
    energies: np.ndarray    # (n_frames,) kcal/mol
    dt: float               # fs


def _get_masses(atoms) -> np.ndarray:
    try:
        masses = atoms.get_masses()
    except Exception:
        from ase.data import atomic_masses, atomic_numbers
        syms = atoms.get_chemical_symbols()
        masses = np.array([atomic_masses[atomic_numbers[s]] for s in syms])
    return masses


def run_production(engine, eq_result, donor_idx: int, acceptor_idx: int,
                   h_idx: int, n_steps: int = 500, dt: float = 1.0,
                   record_every: int = 1) -> ProductionTrajectory:
    from tunnelscan.config import K_B_KCAL

    atoms = engine.atoms
    masses = _get_masses(atoms)
    n_atoms = len(atoms)

    positions = eq_result.positions.copy()
    velocities = eq_result.velocities.copy()
    engine.update_positions(positions)

    gamma = 1e-3  # ps^-1 in 1/fs = 1e-3 /fs (1 ps^-1 = 0.001 /fs)
    temperature = 300.0

    rng = np.random.default_rng(0)
    noise_sigma = np.sqrt(2.0 * gamma * K_B_KCAL * temperature / masses)  # Å/fs per sqrt(fs)

    frame_positions = []
    frame_velocities = []
    frame_xi = []
    frame_energies = []

    energy, forces = engine.energy_and_forces()

    for step in range(n_steps):
        # Half-kick
        acc = forces / masses[:, None]
        velocities += 0.5 * dt * acc

        # Full position update
        positions += velocities * dt
        engine.update_positions(positions)

        # New forces
        energy, forces = engine.energy_and_forces()
        acc_new = forces / masses[:, None]

        # Half-kick
        velocities += 0.5 * dt * acc_new

        # Langevin thermostat
        c1 = np.exp(-gamma * dt)
        c2 = noise_sigma * np.sqrt(dt)
        noise = rng.standard_normal((n_atoms, 3))
        velocities = c1 * velocities + c2[:, None] * noise

        if (step + 1) % record_every == 0:
            frame_positions.append(positions.copy())
            frame_velocities.append(velocities.copy())
            r_D = positions[donor_idx]
            r_H = positions[h_idx]
            r_A = positions[acceptor_idx]
            xi_val = np.linalg.norm(r_D - r_H) - np.linalg.norm(r_H - r_A)
            frame_xi.append(xi_val)
            frame_energies.append(energy)

    return ProductionTrajectory(
        positions=np.array(frame_positions),
        velocities=np.array(frame_velocities),
        xi=np.array(frame_xi),
        energies=np.array(frame_energies),
        dt=dt,
    )
