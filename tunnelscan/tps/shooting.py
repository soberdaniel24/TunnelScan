from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional
from tunnelscan.config import PERTURBATION_DELTA, K_B_KCAL, MASS_H


@dataclass
class ShootingResult:
    accepted: bool
    path: Optional[object]
    t_star: int


def shooting_move(engine, path, donor_idx: int, acceptor_idx: int,
                  h_idx: int, delta: float = PERTURBATION_DELTA,
                  temperature: float = 300.0,
                  max_steps: int = 500) -> ShootingResult:
    from tunnelscan.tps.basins import is_reactant, is_product
    from tunnelscan.classical_md.production import run_production, ProductionTrajectory
    from tunnelscan.classical_md.equilibration import EquilibrationResult

    n_frames = len(path.xi)
    lo = max(1, int(0.1 * n_frames))
    hi = min(n_frames - 1, int(0.9 * n_frames))
    if lo >= hi:
        lo = 0
        hi = n_frames - 1

    rng = np.random.default_rng()
    t_star = int(rng.integers(lo, hi + 1))

    coords_star = path.positions[t_star].copy()
    vel_star = path.velocities[t_star].copy()

    # Perturb H velocity
    try:
        masses = engine.atoms.get_masses()
    except Exception:
        masses = np.ones(len(coords_star))
    m_H = masses[h_idx]

    sigma_H = np.sqrt(K_B_KCAL * temperature / m_H / 2390.06)
    dv = rng.normal(0.0, sigma_H * delta, 3)

    # Rescale to conserve KE
    v_H_orig = vel_star[h_idx].copy()
    ke_orig = 0.5 * m_H * np.dot(v_H_orig, v_H_orig) * 2390.06
    v_H_new = v_H_orig + dv
    ke_new = 0.5 * m_H * np.dot(v_H_new, v_H_new) * 2390.06
    if ke_new > 1e-20 and ke_orig > 1e-20:
        v_H_new *= np.sqrt(ke_orig / ke_new)
    vel_star[h_idx] = v_H_new

    eq_star = EquilibrationResult(
        positions=coords_star,
        velocities=vel_star,
        box_vectors=np.zeros((3, 3)),
        potential_energy=0.0,
    )

    n_forward = max_steps - t_star
    n_backward = t_star

    # Forward leg
    fwd = run_production(engine, eq_star, donor_idx, acceptor_idx, h_idx,
                         n_steps=max(n_forward, 5), dt=path.dt)

    # Backward leg: reverse velocities
    vel_back = -vel_star
    eq_back = EquilibrationResult(
        positions=coords_star,
        velocities=vel_back,
        box_vectors=np.zeros((3, 3)),
        potential_energy=0.0,
    )
    bwd = run_production(engine, eq_back, donor_idx, acceptor_idx, h_idx,
                         n_steps=max(n_backward, 5), dt=path.dt)

    # Concatenate: reversed backward + forward
    bwd_pos = bwd.positions[::-1]
    bwd_vel = -bwd.velocities[::-1]
    bwd_xi = bwd.xi[::-1]
    bwd_e = bwd.energies[::-1]

    trial_pos = np.concatenate([bwd_pos, fwd.positions], axis=0)
    trial_vel = np.concatenate([bwd_vel, fwd.velocities], axis=0)
    trial_xi = np.concatenate([bwd_xi, fwd.xi], axis=0)
    trial_e = np.concatenate([bwd_e, fwd.energies], axis=0)

    trial = ProductionTrajectory(
        positions=trial_pos,
        velocities=trial_vel,
        xi=trial_xi,
        energies=trial_e,
        dt=path.dt,
    )

    n_trial = len(trial_xi)
    check = min(5, n_trial // 2)

    starts_reactant = any(
        is_reactant(trial.positions[i], donor_idx, h_idx, acceptor_idx)
        for i in range(check)
    )
    ends_product = any(
        is_product(trial.positions[-(i + 1)], donor_idx, h_idx, acceptor_idx)
        for i in range(check)
    )

    accepted = starts_reactant and ends_product
    return ShootingResult(accepted=accepted, path=trial if accepted else None, t_star=t_star)
