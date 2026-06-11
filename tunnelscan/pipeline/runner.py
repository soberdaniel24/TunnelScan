from __future__ import annotations
import logging
import os
from typing import Optional, List, Tuple

import numpy as np

log = logging.getLogger(__name__)


def run_tunnelscan(
    pdb_path: str,
    donor_atom_index: int,
    acceptor_atom_index: int,
    hydrogen_atom_index: int,
    temperature: float = 300.0,
    n_beads: int = 8,
    n_tps_paths: int = 300,
    force_field: str = "amber14-all",
    water_model: str = "tip3p",
    run_convergence_check: bool = True,
    experimental_kie: Optional[float] = None,
    output_dir: str = "./tunnelscan_output",
    fast_test: bool = False,
    use_gpu: bool = False,
    # ── Constrained MD parameters ──────────────────────────────────────
    da_constraint: bool = False,
    da_k: float = 50.0,
    da_d0: float = 2.7,
    da_cutoff: float = 3.2,
    position_restraints: Optional[List[Tuple[int, np.ndarray, float]]] = None,
    # ── Integrator parameters ──────────────────────────────────────────
    dt: Optional[float] = None,          # fs; None → 0.5 in fast_test, 1.0 otherwise
    staged_equilibration: bool = False,  # staged L-BFGS + heating; required for GFN2-xTB
    n_centroid_steps_override: Optional[int] = None,
    qm_charge: int = 0,                  # formal charge of QM region (e.g. -1 for Asp⁻)
):
    """
    Full TunnelScan KIE pipeline.

    New parameters vs. original:
      da_constraint           — wrap engine with D-A soft wall + optional position
                                restraints (ConstrainedEngine).
      position_restraints     — list of (idx, ref_pos, k) tuples; passed only when
                                da_constraint=True.
      dt                      — MD timestep (fs). Default: 0.5 in fast_test, 1.0 otherwise.
      staged_equilibration    — run L-BFGS minimisation + 100K→200K→300K heating.
                                Mandatory for GFN2-xTB to avoid divergent geometry.
      n_centroid_steps_override — override centroid MD step count independently of
                                  fast_test flag.
    """
    from tunnelscan.structure.loader import load_pdb
    from tunnelscan.structure.qm_region import select_qm_region
    from tunnelscan.qmmm.engine import QMMMEngine
    from tunnelscan.classical_md.equilibration import equilibrate
    from tunnelscan.classical_md.production import run_production
    from tunnelscan.classical_md.free_energy import compute_free_energy_from_trajectory
    from tunnelscan.tps.path_ensemble import find_initial_path, collect_paths
    from tunnelscan.free_energy.bootstrapping import bootstrap_barrier
    from tunnelscan.centroid_md.quantum_free_energy import run_centroid_md, extract_quantum_barrier
    from tunnelscan.kie.tunnelling_contribution import assess_tunnelling
    from tunnelscan.uncertainty.trajectory_variance import compute_variance
    from tunnelscan.pipeline.output import save_results
    from tunnelscan.config import MASS_H, MASS_D, N_BEADS_CONVERGENCE

    n_classical_steps = 50 if fast_test else 500
    n_centroid_steps = (n_centroid_steps_override if n_centroid_steps_override is not None
                        else (50 if fast_test else 500))
    n_tps = 20 if fast_test else n_tps_paths
    n_tps_max_attempts = 20 if fast_test else 1000

    # Timestep: 0.5 fs for fast_test (GFN2-xTB needs smaller steps), 1.0 otherwise
    dt_md = dt if dt is not None else (0.5 if fast_test else 1.0)

    log.info("Step 1: Loading PDB")
    atoms, topology = load_pdb(pdb_path)

    log.info("Step 2: Selecting QM region")
    qm_region = select_qm_region(
        atoms, topology, donor_atom_index, acceptor_atom_index, hydrogen_atom_index
    )

    log.info("Step 3: Building QM/MM engine")
    base_engine = QMMMEngine(atoms, qm_region, temperature=temperature,
                             use_gpu=use_gpu, qm_charge=qm_charge)

    # ── Optionally wrap with constraints ──────────────────────────────
    if da_constraint:
        from tunnelscan.qmmm.constrained_engine import ConstrainedEngine
        engine = ConstrainedEngine(
            base_engine,
            donor_idx=donor_atom_index,
            acceptor_idx=acceptor_atom_index,
            da_k=da_k,
            da_d0=da_d0,
            da_cutoff=da_cutoff,
            position_restraints=position_restraints or [],
        )
        log.info(f"  D-A soft wall active: d0={da_d0} Å, cutoff={da_cutoff} Å, k={da_k}")
        if position_restraints:
            log.info(f"  Position restraints: {len(position_restraints)} atoms")
    else:
        engine = base_engine

    log.info("Step 4: Equilibration")
    eq_result = equilibrate(engine, temperature=temperature,
                            fast_test=(fast_test and not staged_equilibration),
                            staged=staged_equilibration)
    log.info(f"  Post-equilibration energy: {eq_result.potential_energy:.3f} kcal/mol")

    log.info(f"Step 5: Classical production MD  dt={dt_md} fs")
    classical_traj = run_production(
        engine, eq_result, donor_atom_index, acceptor_atom_index, hydrogen_atom_index,
        n_steps=n_classical_steps, dt=dt_md
    )

    log.info("Step 6: Classical free energy")
    fe_classical = compute_free_energy_from_trajectory(classical_traj, method="classical")
    log.info(f"  Classical barrier: {fe_classical.barrier_height:.3f} kcal/mol")

    log.info("Step 7: TPS — hydrogen")
    try:
        initial_path_H = find_initial_path(
            engine, eq_result, donor_atom_index, acceptor_atom_index,
            hydrogen_atom_index, max_steps=n_classical_steps,
            max_attempts=n_tps_max_attempts,
        )
        ensemble_H = collect_paths(
            engine, initial_path_H, donor_atom_index, acceptor_atom_index,
            hydrogen_atom_index, n_paths=n_tps
        )
        paths_H = ensemble_H.paths if ensemble_H.paths else [classical_traj]
    except RuntimeError as e:
        log.warning(f"TPS H failed: {e}, using classical trajectory")
        paths_H = [classical_traj]

    log.info("Step 8: TPS — deuterium")
    try:
        initial_path_D = find_initial_path(
            engine, eq_result, donor_atom_index, acceptor_atom_index,
            hydrogen_atom_index, max_steps=n_classical_steps,
            max_attempts=n_tps_max_attempts,
        )
        ensemble_D = collect_paths(
            engine, initial_path_D, donor_atom_index, acceptor_atom_index,
            hydrogen_atom_index, n_paths=n_tps
        )
        paths_D = ensemble_D.paths if ensemble_D.paths else [classical_traj]
    except RuntimeError as e:
        log.warning(f"TPS D failed: {e}, using classical trajectory")
        paths_D = [classical_traj]

    log.info("Step 9: Bootstrap barriers")
    bootstrap_H = bootstrap_barrier(
        paths_H, donor_atom_index, acceptor_atom_index, hydrogen_atom_index,
        n_bootstrap=200
    )
    bootstrap_D = bootstrap_barrier(
        paths_D, donor_atom_index, acceptor_atom_index, hydrogen_atom_index,
        n_bootstrap=200
    )

    log.info("Step 10: Centroid MD — hydrogen")
    centroid_traj_H = run_centroid_md(
        engine, eq_result, donor_atom_index, acceptor_atom_index, hydrogen_atom_index,
        n_beads=n_beads, n_steps=n_centroid_steps, temperature=temperature,
        mass=MASS_H,
    )
    fe_H = extract_quantum_barrier(fe_classical, centroid_traj_H,
                                    temperature=temperature, n_beads=n_beads)
    log.info(f"  Centroid-H barrier: {fe_H.barrier_height:.3f} kcal/mol  ZPE={fe_H.zpe:.4f}")

    log.info("Step 11: Centroid MD — deuterium")
    centroid_traj_D = run_centroid_md(
        engine, eq_result, donor_atom_index, acceptor_atom_index, hydrogen_atom_index,
        n_beads=n_beads, n_steps=n_centroid_steps, temperature=temperature,
        mass=MASS_D,
    )
    fe_D = extract_quantum_barrier(fe_classical, centroid_traj_D,
                                    temperature=temperature, n_beads=n_beads)
    log.info(f"  Centroid-D barrier: {fe_D.barrier_height:.3f} kcal/mol")

    if run_convergence_check and not fast_test:
        log.info("Step 12: Convergence check (N_beads=16)")
        conv_traj = run_centroid_md(
            engine, eq_result, donor_atom_index, acceptor_atom_index, hydrogen_atom_index,
            n_beads=N_BEADS_CONVERGENCE, n_steps=n_centroid_steps, temperature=temperature,
            mass=MASS_H,
        )
        fe_H_conv = extract_quantum_barrier(fe_classical, conv_traj,
                                             temperature=temperature, n_beads=N_BEADS_CONVERGENCE)
        barrier_diff = abs(fe_H_conv.barrier_height - fe_H.barrier_height)
        if barrier_diff > 0.5:
            log.warning(f"Convergence check: N_beads=16 barrier differs by {barrier_diff:.3f}")

    log.info("Step 13: Assess tunnelling")
    result = assess_tunnelling(
        fe_classical, fe_H, fe_D, bootstrap_H, bootstrap_D,
        kie_experimental=experimental_kie, temperature=temperature
    )

    log.info("Step 14: Compute variance")
    variance_report = compute_variance(
        paths_H, paths_D, donor_atom_index, acceptor_atom_index, hydrogen_atom_index
    )

    log.info("Step 15: Save results")
    save_results(
        output_dir, result, fe_classical, fe_H, fe_D,
        variance_report, bootstrap_H, bootstrap_D
    )

    return result
