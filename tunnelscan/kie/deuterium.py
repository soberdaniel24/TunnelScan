from __future__ import annotations
from tunnelscan.config import MASS_D, DEUTERIUM_CHECK_TOLERANCE
from tunnelscan.classical_md.free_energy import FreeEnergyResult


def run_deuterium_protocol(engine, eq_result, donor_idx: int, acceptor_idx: int,
                            h_idx: int, classical_fe: FreeEnergyResult,
                            **kwargs) -> FreeEnergyResult:
    from tunnelscan.centroid_md.quantum_free_energy import run_centroid_md, extract_quantum_barrier

    n_beads = kwargs.get("n_beads", 8)
    n_steps = kwargs.get("n_steps", 500)
    temperature = kwargs.get("temperature", 300.0)

    centroid_traj = run_centroid_md(
        engine, eq_result, donor_idx, acceptor_idx, h_idx,
        n_beads=n_beads, n_steps=n_steps, temperature=temperature,
        mass=MASS_D,
    )
    return extract_quantum_barrier(classical_fe, centroid_traj,
                                   temperature=temperature, n_beads=n_beads)


def check_deuterium_consistency(fe_D: FreeEnergyResult,
                                 fe_classical: FreeEnergyResult) -> bool:
    return abs(fe_D.barrier_height - fe_classical.barrier_height) < DEUTERIUM_CHECK_TOLERANCE
