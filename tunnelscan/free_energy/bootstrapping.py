from __future__ import annotations
import numpy as np
from tunnelscan.config import K_B_KCAL


def _barrier_from_xi(xi_vals: np.ndarray, temperature: float, n_bins: int) -> float:
    if len(xi_vals) < 2:
        return float("nan")
    xi_min, xi_max = xi_vals.min() - 0.01, xi_vals.max() + 0.01
    counts, edges = np.histogram(xi_vals, bins=n_bins, range=(xi_min, xi_max))
    centres = (edges[:-1] + edges[1:]) / 2.0
    counts = counts.astype(float)
    counts[counts < 1] = 1.0
    F = -K_B_KCAL * temperature * np.log(counts)
    F -= F.min()
    mask = centres < 0.0
    reactant = float(F[mask].min()) if mask.any() else float(F.min())
    return float(F.max() - reactant)


def per_trajectory_barriers(paths: list, donor_idx: int, acceptor_idx: int,
                             h_idx: int, n_bins: int = 15,
                             temperature: float = 300.0) -> np.ndarray:
    barriers = []
    for path in paths:
        b = _barrier_from_xi(path.xi, temperature, n_bins)
        barriers.append(b)
    return np.array(barriers)


def bootstrap_barrier(paths: list, donor_idx: int, acceptor_idx: int,
                      h_idx: int, n_bootstrap: int = 1000,
                      n_bins: int = 15, temperature: float = 300.0) -> dict:
    rng = np.random.default_rng(42)
    n = len(paths)
    per_path = per_trajectory_barriers(paths, donor_idx, acceptor_idx, h_idx,
                                        n_bins, temperature)

    bootstrap_barriers = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        xi_combined = np.concatenate([paths[i].xi for i in idx])
        b = _barrier_from_xi(xi_combined, temperature, n_bins)
        bootstrap_barriers.append(b)

    bootstrap_barriers = np.array(bootstrap_barriers)
    bootstrap_barriers = bootstrap_barriers[np.isfinite(bootstrap_barriers)]
    mean = float(np.mean(bootstrap_barriers)) if len(bootstrap_barriers) else float("nan")
    std = float(np.std(bootstrap_barriers)) if len(bootstrap_barriers) else float("nan")
    lo = float(np.percentile(bootstrap_barriers, 2.5)) if len(bootstrap_barriers) else float("nan")
    hi = float(np.percentile(bootstrap_barriers, 97.5)) if len(bootstrap_barriers) else float("nan")

    return {
        "mean": mean,
        "std": std,
        "ci_95": (lo, hi),
        "per_trajectory_barriers": list(per_path),
    }
