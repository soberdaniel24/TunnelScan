from __future__ import annotations
import numpy as np


def extract_populations(paths: list, xi_min: float = -2.0, xi_max: float = 2.0,
                        n_bins: int = 15) -> np.ndarray:
    edges = np.linspace(xi_min, xi_max, n_bins + 1)
    counts = np.zeros(n_bins, dtype=float)
    for path in paths:
        h, _ = np.histogram(path.xi, bins=edges)
        counts += h
    return counts


def supplement_basin_sampling(engine, eq_result, donor_idx: int, acceptor_idx: int,
                               h_idx: int, populations: np.ndarray,
                               xi_edges: np.ndarray) -> np.ndarray:
    from tunnelscan.classical_md.production import run_production

    populations = populations.copy()
    short_traj = run_production(engine, eq_result, donor_idx, acceptor_idx, h_idx,
                                n_steps=200, dt=1.0)
    new_counts, _ = np.histogram(short_traj.xi, bins=xi_edges)

    for i in range(len(populations)):
        if populations[i] < 10:
            populations[i] += new_counts[i]
    return populations
