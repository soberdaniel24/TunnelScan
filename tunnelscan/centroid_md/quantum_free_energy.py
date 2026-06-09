from __future__ import annotations
import numpy as np
from tunnelscan.config import (MASS_H, K_B_KCAL, N_BEADS_DEFAULT,
                                DT_SLOW_FS, DT_FAST_FS)
from tunnelscan.classical_md.production import ProductionTrajectory
from tunnelscan.classical_md.free_energy import FreeEnergyResult


def run_centroid_md(engine, eq_result, donor_idx: int, acceptor_idx: int,
                    h_idx: int, n_beads: int = N_BEADS_DEFAULT,
                    n_steps: int = 500, dt_slow: float = DT_SLOW_FS,
                    temperature: float = 300.0, mass: float = MASS_H) -> ProductionTrajectory:
    from tunnelscan.centroid_md.ring_polymer import make_ring_polymer, centroid_position
    from tunnelscan.centroid_md.normal_modes import build_transform, normal_mode_freqs
    from tunnelscan.centroid_md.propagator import propagate_step

    init_pos = eq_result.positions[h_idx]
    rp = make_ring_polymer(n_beads, mass, temperature, init_pos)
    C = build_transform(n_beads)
    freqs = normal_mode_freqs(n_beads, mass, temperature)

    n_atoms = len(engine.atoms)
    frame_positions = []
    frame_velocities = []
    frame_xi = []
    frame_energies = []

    # Start from equilibration state
    engine.update_positions(eq_result.positions)
    all_positions = eq_result.positions.copy()
    all_velocities = eq_result.velocities.copy()

    for step in range(n_steps):
        # Compute forces at centroid position
        centroid_pos = centroid_position(rp)
        # Update H position to centroid for force evaluation
        eval_pos = all_positions.copy()
        eval_pos[h_idx] = centroid_pos
        engine.update_positions(eval_pos)
        energy, forces = engine.energy_and_forces()

        # Extract force on H for the ring polymer
        h_forces_bead = np.tile(forces[h_idx], (n_beads, 1))

        # Propagate ring polymer
        rp = propagate_step(rp, h_forces_bead, C, freqs,
                            dt_slow=dt_slow, dt_fast=DT_FAST_FS,
                            temperature=temperature)

        # Update centroid position in main trajectory
        centroid = centroid_position(rp)
        all_positions[h_idx] = centroid
        # Simple Langevin for all other atoms
        _propagate_other_atoms(all_positions, all_velocities, forces, engine,
                               h_idx, dt_slow, temperature)

        r_D = all_positions[donor_idx]
        r_H = centroid
        r_A = all_positions[acceptor_idx]
        xi_val = np.linalg.norm(r_D - r_H) - np.linalg.norm(r_H - r_A)

        frame_positions.append(all_positions.copy())
        frame_velocities.append(all_velocities.copy())
        frame_xi.append(xi_val)
        frame_energies.append(energy)

    return ProductionTrajectory(
        positions=np.array(frame_positions),
        velocities=np.array(frame_velocities),
        xi=np.array(frame_xi),
        energies=np.array(frame_energies),
        dt=dt_slow,
    )


def _propagate_other_atoms(positions, velocities, forces, engine, h_idx,
                            dt, temperature):
    from tunnelscan.config import K_B_KCAL
    try:
        masses = engine.atoms.get_masses()
    except Exception:
        masses = np.ones(len(positions))

    gamma = 1e-3
    rng = np.random.default_rng()
    n = len(positions)
    c1 = np.exp(-gamma * dt)
    noise_sigma = np.sqrt(2.0 * gamma * K_B_KCAL * temperature / masses * dt)

    acc = forces / (masses[:, None] * 2390.06)
    for i in range(n):
        if i == h_idx:
            continue
        velocities[i] += 0.5 * dt * acc[i]
        positions[i] += velocities[i] * dt
        velocities[i] += 0.5 * dt * acc[i]
        noise = rng.standard_normal(3)
        velocities[i] = c1 * velocities[i] + noise_sigma[i] * noise
    engine.update_positions(positions)


def extract_quantum_barrier(classical_fe: FreeEnergyResult,
                             centroid_traj: ProductionTrajectory,
                             temperature: float = 300.0,
                             n_beads: int = N_BEADS_DEFAULT) -> FreeEnergyResult:
    from tunnelscan.classical_md.free_energy import compute_free_energy_from_trajectory

    centroid_fe = compute_free_energy_from_trajectory(centroid_traj, method="centroid_temp")

    # ZPE estimate: difference between centroid and classical reactant minima
    xi_c = centroid_fe.xi_values
    mask = xi_c < 0.0
    if mask.any():
        centroid_reactant = float(centroid_fe.free_energy[mask].min())
    else:
        centroid_reactant = float(centroid_fe.free_energy.min())

    xi_cl = classical_fe.xi_values
    mask_cl = xi_cl < 0.0
    if mask_cl.any():
        classical_reactant = float(classical_fe.free_energy[mask_cl].min())
    else:
        classical_reactant = float(classical_fe.free_energy.min())

    zpe = centroid_reactant - classical_reactant

    return FreeEnergyResult(
        xi_values=centroid_fe.xi_values,
        free_energy=centroid_fe.free_energy,
        barrier_height=centroid_fe.barrier_height,
        reactant_energy=centroid_fe.reactant_energy,
        ts_energy=centroid_fe.ts_energy,
        zpe=float(zpe),
        method=f"centroid_N{n_beads}",
        n_trajectories=1,
        n_frames=centroid_fe.n_frames,
    )
