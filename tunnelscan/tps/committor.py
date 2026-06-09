from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class CommittorResult:
    p_B_values: list[float]
    mean_pB: float
    n_false_ts: int


def committor_analysis(engine, paths, donor_idx: int, acceptor_idx: int,
                       h_idx: int, n_ts_frames: int = 20,
                       n_shoots_per_frame: int = 50) -> CommittorResult:
    from tunnelscan.tps.basins import is_product, xi as compute_xi
    from tunnelscan.classical_md.production import run_production
    from tunnelscan.classical_md.equilibration import EquilibrationResult

    rng = np.random.default_rng(99)

    # Collect all frames near TS (|xi| < 0.2 Å)
    ts_frames = []
    for path in paths:
        for t in range(len(path.xi)):
            if abs(path.xi[t]) < 0.2:
                ts_frames.append((path.positions[t].copy(), path.velocities[t].copy()))

    if not ts_frames:
        return CommittorResult(p_B_values=[], mean_pB=float("nan"), n_false_ts=0)

    # Sample n_ts_frames
    indices = rng.choice(len(ts_frames), size=min(n_ts_frames, len(ts_frames)), replace=False)
    selected = [ts_frames[i] for i in indices]

    p_B_values = []
    try:
        masses = engine.atoms.get_masses()
    except Exception:
        masses = np.ones(len(selected[0][0]))

    for pos, vel in selected:
        n_product = 0
        for _ in range(n_shoots_per_frame):
            # Randomize velocities from Maxwell-Boltzmann at 300K
            from tunnelscan.config import K_B_KCAL
            sigma = np.sqrt(K_B_KCAL * 300.0 / masses / 2390.06)
            rand_vel = rng.normal(0.0, 1.0, vel.shape) * sigma[:, None]

            eq = EquilibrationResult(
                positions=pos, velocities=rand_vel,
                box_vectors=np.zeros((3, 3)), potential_energy=0.0
            )
            short_traj = run_production(engine, eq, donor_idx, acceptor_idx, h_idx,
                                        n_steps=100, dt=1.0)
            # Check if ends in product
            check = min(5, len(short_traj.xi))
            if any(is_product(short_traj.positions[-(i + 1)], donor_idx, h_idx, acceptor_idx)
                   for i in range(check)):
                n_product += 1

        p_B = n_product / n_shoots_per_frame
        p_B_values.append(p_B)

    n_false_ts = sum(1 for p in p_B_values if not (0.3 <= p <= 0.7))
    mean_pB = float(np.mean(p_B_values)) if p_B_values else float("nan")
    return CommittorResult(p_B_values=p_B_values, mean_pB=mean_pB, n_false_ts=n_false_ts)
