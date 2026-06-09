from __future__ import annotations
import logging
import numpy as np
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass
class EquilibrationResult:
    positions: np.ndarray
    velocities: np.ndarray
    box_vectors: np.ndarray
    potential_energy: float


def _maxwell_boltzmann_velocities(masses: np.ndarray, temperature: float,
                                   rng: np.random.Generator) -> np.ndarray:
    from tunnelscan.config import K_B_KCAL
    n = len(masses)
    vel = np.zeros((n, 3))
    for i in range(n):
        sigma = np.sqrt(K_B_KCAL * temperature / masses[i])
        vel[i] = rng.normal(0.0, sigma, 3)
    # Remove COM velocity
    total_mass = masses.sum()
    com_vel = (masses[:, None] * vel).sum(axis=0) / total_mass
    vel -= com_vel
    return vel


def _get_masses(atoms) -> np.ndarray:
    try:
        masses = atoms.get_masses()
    except Exception:
        from ase.data import atomic_masses, atomic_numbers
        syms = atoms.get_chemical_symbols()
        masses = np.array([atomic_masses[atomic_numbers[s]] for s in syms])
    return masses


def equilibrate(engine, temperature: float = 300.0,
                fast_test: bool = False) -> EquilibrationResult:
    from tunnelscan.config import K_B_KCAL

    atoms = engine.atoms
    positions = atoms.get_positions().copy()
    masses = _get_masses(atoms)
    n = len(atoms)
    rng = np.random.default_rng(42)

    if fast_test:
        # 10 minimization steps only
        dt = 0.01
        for _ in range(10):
            e, f = engine.energy_and_forces()
            positions += 0.01 * f
            engine.update_positions(positions)
        e_final, _ = engine.energy_and_forces()
        velocities = _maxwell_boltzmann_velocities(masses, temperature, rng)
        return EquilibrationResult(
            positions=positions.copy(),
            velocities=velocities,
            box_vectors=np.zeros((3, 3)),
            potential_energy=float(e_final),
        )

    # Try OpenMM
    try:
        result = _equilibrate_openmm(engine, atoms, temperature, masses)
        return result
    except Exception as e_omm:
        log.warning(f"OpenMM equilibration failed ({e_omm}), using ASE fallback")

    return _equilibrate_ase(engine, atoms, positions, masses, temperature, rng)


def _equilibrate_ase(engine, atoms, positions, masses, temperature, rng):
    from tunnelscan.config import K_B_KCAL
    n = len(atoms)

    # LBFGS minimization (200 steps)
    for step in range(200):
        e, f = engine.energy_and_forces()
        fmax = np.max(np.abs(f))
        if fmax < 0.05:
            break
        step_size = min(0.1 / (fmax + 1e-10), 0.05)
        positions += step_size * f
        engine.update_positions(positions)

    # Velocity rescaling NVT (5000 steps, dt=0.5 fs)
    dt = 0.5  # fs
    velocities = _maxwell_boltzmann_velocities(masses, temperature, rng)
    e_final, forces = engine.energy_and_forces()
    rescale_every = 100

    for step in range(5000):
        # Verlet
        acc = forces / masses[:, None]
        velocities += 0.5 * dt * acc
        positions += velocities * dt
        engine.update_positions(positions)
        e_final, forces = engine.energy_and_forces()
        acc_new = forces / masses[:, None]
        velocities += 0.5 * dt * acc_new

        if (step + 1) % rescale_every == 0:
            ke = 0.5 * (masses[:, None] * velocities**2).sum()
            t_current = 2 * ke / (3 * len(masses) * K_B_KCAL)
            if t_current > 1e-6:
                scale = np.sqrt(temperature / t_current)
                velocities *= scale

    return EquilibrationResult(
        positions=positions.copy(),
        velocities=velocities,
        box_vectors=np.zeros((3, 3)),
        potential_energy=float(e_final),
    )


def _equilibrate_openmm(engine, atoms, temperature, masses):
    import openmm
    import openmm.app as app
    import openmm.unit as unit

    # Use a minimal system with the engine's forces
    n = len(atoms)
    positions_nm = atoms.get_positions() * 0.1  # Å to nm

    system = openmm.System()
    for m in masses:
        system.addParticle(m * unit.amu)

    integrator = openmm.LangevinMiddleIntegrator(
        temperature * unit.kelvin, 1.0 / unit.picosecond, 2.0 * unit.femtosecond
    )
    platform = openmm.Platform.getPlatformByName("CPU")
    context = openmm.Context(system, integrator, platform)
    context.setPositions(positions_nm * unit.nanometer)
    context.setVelocitiesToTemperature(temperature * unit.kelvin)

    # NVT 100 ps = 50000 steps at 2fs
    integrator.step(50000)

    state = context.getState(getPositions=True, getVelocities=True, getEnergy=True)
    pos_nm = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    vel_nm_ps = state.getVelocities(asNumpy=True).value_in_unit(unit.nanometer / unit.picosecond)
    pe = state.getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole)

    positions = pos_nm * 10.0  # nm to Å
    velocities = vel_nm_ps * 10.0 / 1000.0  # nm/ps to Å/fs

    engine.update_positions(positions)

    return EquilibrationResult(
        positions=positions,
        velocities=velocities,
        box_vectors=np.zeros((3, 3)),
        potential_energy=float(pe),
    )
