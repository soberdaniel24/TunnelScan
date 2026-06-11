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
                fast_test: bool = False,
                staged: bool = False) -> EquilibrationResult:
    """
    Equilibrate the engine's atomic positions and velocities.

    staged=True: run L-BFGS minimisation + staged Langevin heating
    (100 K→200 K→300 K, 50 steps each at 0.5 fs). Required for
    GFN2-xTB runs where a bare fast_test gives unphysical geometry.
    """
    from tunnelscan.config import K_B_KCAL

    atoms = engine.atoms
    positions = atoms.get_positions().copy()
    masses = _get_masses(atoms)
    n = len(atoms)
    rng = np.random.default_rng(42)

    if staged:
        return _staged_equilibrate(engine, positions, masses, temperature, rng)

    if fast_test:
        return _fast_minimise_equilibrate(engine, positions, masses, temperature, rng)

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


def _fast_minimise_equilibrate(engine, positions, masses, temperature,
                                rng) -> EquilibrationResult:
    """Original fast_test path: 10 gradient-descent steps."""
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


def _staged_equilibrate(engine, positions, masses, temperature,
                         rng) -> EquilibrationResult:
    """
    Staged equilibration for small QM active-site models:

    1. Geometry minimisation using ASE LBFGS (if available) or simple
       steepest descent with correct per-step displacement clamping.
       Converges to max|F| < 1 kcal/mol/Å, max 300 steps.
       Raises RuntimeError if converged energy is positive or NaN (bad geometry).

    2. Langevin NVT heating at 100 K → 200 K → 300 K (50 steps each, dt=0.5 fs).

    3. Post-heat single-point energy validation.
    """
    from tunnelscan.config import K_B_KCAL
    import math

    n = len(masses)
    dt_min = 0.5           # fs for heating phase

    # ── Stage 1: Geometry optimisation ───────────────────────────────
    EV_TO_KCAL = 23.0605

    # Try ASE LBFGS first — it handles stiff/complex PES far better
    # than steepest descent and doesn't need manual step size tuning.
    lbfgs_ok = False
    try:
        from ase.optimize import LBFGS
        from ase.calculators.calculator import Calculator as AseCalc

        class _EngineAsCalc(AseCalc):
            """Thin ASE calculator adapter so LBFGS can drive engine.energy_and_forces()."""
            implemented_properties = ["energy", "forces"]

            def __init__(self, eng):
                super().__init__()
                self._eng = eng

            def calculate(self, atoms=None, properties=("energy",),
                          system_changes=("positions",)):
                if atoms is not None:
                    self._eng.update_positions(atoms.get_positions())
                e_kcal, f_kcal = self._eng.energy_and_forces()
                self.results = {
                    "energy": e_kcal / EV_TO_KCAL,
                    "forces": f_kcal / EV_TO_KCAL,
                }

        import ase
        ase_opt_atoms = engine.atoms.copy()
        ase_opt_atoms.calc = _EngineAsCalc(engine)

        opt = LBFGS(ase_opt_atoms, logfile=None)
        opt.run(fmax=1.0 / EV_TO_KCAL, steps=300)  # fmax in eV/Å
        positions[:] = ase_opt_atoms.get_positions()
        engine.update_positions(positions)
        lbfgs_ok = True
    except Exception:
        pass  # fall through to simple steepest descent

    if not lbfgs_ok:
        # Simple steepest descent with correct per-step clamping:
        # max displacement per step = step_size Å (for the highest-force atom)
        e_prev, f = engine.energy_and_forces()
        step_size = 0.005

        for _step in range(500):
            fmax = float(np.max(np.abs(f)))
            if fmax < 1.0:
                break
            # Scale so max atom displacement = step_size Å
            scale = step_size / (fmax + 1e-10)
            pos_trial = positions + scale * f
            engine.update_positions(pos_trial)
            e_trial, f_trial = engine.energy_and_forces()

            if e_trial < e_prev:
                positions[:] = pos_trial
                e_prev = e_trial
                f = f_trial
                step_size = min(step_size * 1.2, 0.02)
            else:
                engine.update_positions(positions)
                step_size *= 0.5
                e_prev, f = engine.energy_and_forces()
                if step_size < 1e-8:
                    break  # stuck

    e_min, _ = engine.energy_and_forces()
    if math.isnan(e_min) or math.isinf(e_min) or e_min > 0.0:
        raise RuntimeError(
            f"Minimisation produced invalid energy: {e_min:.4f} kcal/mol. "
            "Active-site geometry invalid — likely a link atom clash or "
            "partial aromatic ring with radical character."
        )

    # ── Stage 2: Staged Langevin heating ─────────────────────────────
    velocities = _maxwell_boltzmann_velocities(masses, 100.0, rng)
    gamma = 1e-3           # /fs  (= 1 ps⁻¹)
    c1 = np.exp(-gamma * dt_min)

    for stage_T in (100.0, 200.0, temperature):
        noise_sigma = np.sqrt(2.0 * gamma * K_B_KCAL * stage_T / masses)
        _, forces = engine.energy_and_forces()
        for _ in range(50):
            acc = forces / masses[:, None]
            velocities += 0.5 * dt_min * acc
            positions  += velocities * dt_min
            engine.update_positions(positions)
            e_step, forces = engine.energy_and_forces()
            acc_new = forces / masses[:, None]
            velocities += 0.5 * dt_min * acc_new
            noise = rng.standard_normal((n, 3))
            velocities = c1 * velocities + noise_sigma[:, None] * noise * math.sqrt(dt_min)

    # ── Stage 3: Post-heat validation ─────────────────────────────────
    e_final, f_final = engine.energy_and_forces()
    if math.isnan(e_final) or math.isinf(e_final):
        import warnings
        warnings.warn(
            f"Post-heat energy is {e_final:.4f} — GFN2-xTB may not have "
            "converged at the heated geometry. Proceeding but KIE may be "
            "unreliable.",
            RuntimeWarning
        )

    return EquilibrationResult(
        positions=positions.copy(),
        velocities=velocities.copy(),
        box_vectors=np.zeros((3, 3)),
        potential_energy=float(e_final),
    )
