from __future__ import annotations
import logging
import numpy as np
from dataclasses import dataclass, field
from tunnelscan.config import PERTURBATION_DELTA

log = logging.getLogger(__name__)


@dataclass
class PathEnsemble:
    paths: list
    n_accepted: int
    n_rejected: int
    n_independent: int
    delta_history: list[float]


def find_initial_path(engine, eq_result, donor_idx: int, acceptor_idx: int,
                      h_idx: int, max_attempts: int = 1000,
                      max_steps: int = 500):
    from tunnelscan.classical_md.production import run_production
    from tunnelscan.tps.basins import is_reactant, is_product

    for attempt in range(max_attempts):
        traj = run_production(
            engine, eq_result, donor_idx, acceptor_idx, h_idx,
            n_steps=max_steps, dt=1.0
        )
        n = len(traj.xi)
        check = min(5, n // 2)
        starts_r = any(
            is_reactant(traj.positions[i], donor_idx, h_idx, acceptor_idx)
            for i in range(check)
        )
        ends_p = any(
            is_product(traj.positions[-(i + 1)], donor_idx, h_idx, acceptor_idx)
            for i in range(check)
        )
        if starts_r and ends_p:
            log.info(f"Found initial reactive path at attempt {attempt + 1}")
            return traj

        if attempt % 100 == 0:
            log.debug(f"find_initial_path: attempt {attempt}/{max_attempts}")

    raise RuntimeError(f"No reactive path found in {max_attempts} attempts")


def collect_paths(engine, initial_path, donor_idx: int, acceptor_idx: int,
                  h_idx: int, n_paths: int = 300,
                  delta: float = PERTURBATION_DELTA) -> PathEnsemble:
    from tunnelscan.tps.shooting import shooting_move

    paths = []
    n_accepted = 0
    n_rejected = 0
    delta_history = [delta]
    current_path = initial_path
    recent = []

    for i in range(n_paths):
        result = shooting_move(
            engine, current_path, donor_idx, acceptor_idx, h_idx,
            delta=delta, max_steps=len(current_path.xi)
        )
        recent.append(result.accepted)
        if result.accepted:
            current_path = result.path
            paths.append(result.path)
            n_accepted += 1
        else:
            n_rejected += 1

        # Adaptive delta based on last 50 moves
        if len(recent) >= 50:
            window = recent[-50:]
            acc_rate = sum(window) / len(window)
            if acc_rate < 0.20:
                delta *= 0.90
            elif acc_rate > 0.60:
                delta *= 1.10
            delta = max(1e-4, min(1.0, delta))
            delta_history.append(delta)

    n_independent = max(1, n_accepted // 150)
    return PathEnsemble(
        paths=paths,
        n_accepted=n_accepted,
        n_rejected=n_rejected,
        n_independent=n_independent,
        delta_history=delta_history,
    )
