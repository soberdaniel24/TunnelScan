from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class VarianceReport:
    classical_barriers: np.ndarray
    centroid_barriers: np.ndarray
    classical_std: float
    centroid_std: float
    enzyme_type: str
    multi_pathway: bool


def compute_variance(paths_classical: list, paths_centroid: list,
                     donor_idx: int, acceptor_idx: int,
                     h_idx: int) -> VarianceReport:
    from tunnelscan.free_energy.bootstrapping import per_trajectory_barriers

    classical_barriers = per_trajectory_barriers(
        paths_classical, donor_idx, acceptor_idx, h_idx
    )
    centroid_barriers = per_trajectory_barriers(
        paths_centroid, donor_idx, acceptor_idx, h_idx
    )

    classical_barriers = classical_barriers[np.isfinite(classical_barriers)]
    centroid_barriers = centroid_barriers[np.isfinite(centroid_barriers)]

    classical_std = float(np.std(classical_barriers)) if len(classical_barriers) > 1 else 0.0
    centroid_std = float(np.std(centroid_barriers)) if len(centroid_barriers) > 1 else 0.0

    enzyme_type = "natural" if classical_std < 1.0 else "designed"
    multi_pathway = centroid_std > 2.0 * classical_std

    return VarianceReport(
        classical_barriers=classical_barriers,
        centroid_barriers=centroid_barriers,
        classical_std=classical_std,
        centroid_std=centroid_std,
        enzyme_type=enzyme_type,
        multi_pathway=multi_pathway,
    )
