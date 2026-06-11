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
                   record_every: int = 1,
                   max_energy_drift: float = 5.0) -> ProductionTrajectory:
    """
    Velocity-Verlet Langevin MD with adaptive-timestep energy-drift rejection.

    If the total energy changes by more than max_energy_drift kcal/mol in a
    single step, the step is rejected: positions are restored and the timestep
    is halved for the next 10 steps before reverting to dt.  This prevents
    GFN2-xTB SCF failures from propagating garbage forces into the trajectory.
    """
    from tunnelscan.config import K_B_KCAL

    atoms = engine.atoms
    masses = _get_masses(atoms)
    n_atoms = len(atoms)

    positions = eq_result.positions.copy()
    velocities = eq_result.velocities.copy()
    engine.update_positions(positions)

    gamma = 1e-3          # Langevin damping, /fs (= 1 ps⁻¹)
    temperature = 300.0

    rng = np.random.default_rng(0)
    noise_sigma = np.sqrt(2.0 * gamma * K_B_KCAL * temperature / masses)  # Å/fs per √fs

    frame_positions: list[np.ndarray] = []
    frame_velocities: list[np.ndarray] = []
    frame_xi: list[float] = []
    frame_energies: list[float] = []

    energy, forces = engine.energy_and_forces()

    dt_current = dt
    dt_reduce_countdown = 0  # steps remaining at reduced timestep

    for step in range(n_steps):
        # Restore normal timestep after reduction period
        if dt_reduce_countdown > 0:
            dt_reduce_countdown -= 1
            if dt_reduce_countdown == 0:
                dt_current = dt

        # Save state for potential rollback
        pos_saved = positions.copy()
        vel_saved = velocities.copy()
        e_prev = energy

        # ── Velocity-Verlet half-kick ─────────────────────────────────
        acc = forces / masses[:, None]
        velocities += 0.5 * dt_current * acc

        # ── Full position update ───────────────────────────────────────
        positions += velocities * dt_current
        engine.update_positions(positions)

        # ── New forces ────────────────────────────────────────────────
        energy_new, forces_new = engine.energy_and_forces()

        # ── Energy drift check (also catches NaN from failed SCF) ────────
        import math as _math
        drift = abs(energy_new - e_prev)
        if _math.isnan(drift) or _math.isinf(drift) or drift > max_energy_drift:
            # Reject step: restore state, reduce timestep
            positions[:] = pos_saved
            velocities[:] = vel_saved
            engine.update_positions(positions)
            energy_new = e_prev
            forces_new = forces
            dt_current = max(dt_current * 0.5, 0.05)
            dt_reduce_countdown = 10
            # Still record this frame so trajectory length is maintained
        else:
            forces = forces_new
            energy = energy_new
            acc_new = forces / masses[:, None]
            velocities += 0.5 * dt_current * acc_new

        # ── Langevin thermostat ───────────────────────────────────────
        c1 = np.exp(-gamma * dt_current)
        c2 = noise_sigma * np.sqrt(dt_current)
        velocities = c1 * velocities + c2[:, None] * rng.standard_normal((n_atoms, 3))

        if (step + 1) % record_every == 0:
            frame_positions.append(positions.copy())
            frame_velocities.append(velocities.copy())
            r_D = positions[donor_idx]
            r_H = positions[h_idx]
            r_A = positions[acceptor_idx]
            dDH = float(np.linalg.norm(r_D - r_H)) + 1e-14
            dHA = float(np.linalg.norm(r_H - r_A)) + 1e-14
            frame_xi.append(dDH - dHA)
            frame_energies.append(float(energy))

    return ProductionTrajectory(
        positions=np.array(frame_positions),
        velocities=np.array(frame_velocities),
        xi=np.array(frame_xi),
        energies=np.array(frame_energies),
        dt=dt,
    )
