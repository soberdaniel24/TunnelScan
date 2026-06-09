from __future__ import annotations
import numpy as np
from tunnelscan.config import K_B_KCAL
from tunnelscan.classical_md.free_energy import FreeEnergyResult


def boltzmann_inversion(populations: np.ndarray, xi_edges: np.ndarray,
                         temperature: float = 300.0) -> tuple[np.ndarray, np.ndarray]:
    xi_centres = (xi_edges[:-1] + xi_edges[1:]) / 2.0
    pop = populations.astype(float).copy()
    pop[pop < 1] = 1.0
    F = -K_B_KCAL * temperature * np.log(pop)
    F -= F.min()
    return xi_centres, F


def stitch_windows(window_populations: list[np.ndarray], xi_edges: np.ndarray,
                   temperature: float = 300.0) -> FreeEnergyResult:
    total = np.zeros(len(xi_edges) - 1, dtype=float)
    for pop in window_populations:
        if len(pop) == len(total):
            total += pop

    xi_centres, F = boltzmann_inversion(total, xi_edges, temperature)

    mask = xi_centres < 0.0
    if mask.any():
        reactant = float(F[mask].min())
    else:
        reactant = float(F.min())

    barrier = float(F.max() - reactant)
    ts = float(F.max())

    n_frames = int(total.sum())
    n_traj = len(window_populations)

    return FreeEnergyResult(
        xi_values=xi_centres,
        free_energy=F,
        barrier_height=barrier,
        reactant_energy=reactant,
        ts_energy=ts,
        zpe=float("nan"),
        method="window_stitching",
        n_trajectories=n_traj,
        n_frames=n_frames,
    )
