from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from tunnelscan.config import (K_B_KCAL, HBAR_KCAL_FS, AMU_TO_KG,
                                ANGSTROM_TO_M)


@dataclass
class RingPolymer:
    n_beads: int
    mass: float
    temperature: float
    positions: np.ndarray   # (n_beads, 3) Å
    velocities: np.ndarray  # (n_beads, 3) Å/fs
    k_spring: float         # kcal/mol/Å²


def _compute_k_spring(n_beads: int, mass: float, temperature: float) -> float:
    # omega_N = N * k_B * T / hbar  (in rad/fs, using kcal-fs units)
    omega_N = n_beads * K_B_KCAL * temperature / HBAR_KCAL_FS
    # k_spring = m * omega_N^2  (in kcal/mol/Å² with mass in amu and omega in rad/fs)
    # Units: amu * (rad/fs)^2 → need to convert to kcal/mol/Å²
    # 1 amu*(Å/fs)^2 = (1.66054e-27 kg) * (1e-10 m / 1e-15 s)^2
    #                = 1.66054e-27 * 1e10 m^2/s^2 ... (not directly kcal/mol/Å²)
    # Use: 1 kcal/mol = 6.9477e-21 J, 1 amu*(Å/fs)^2 = 1.66054e-27 * 1e10 = 1.66054e-17 J per atom
    # So 1 amu*(Å/fs)^2 / atom = 1.66054e-17 / 6.9477e-21 * 6.022e23 kcal/mol ... let's use direct:
    # Actually in Å/fs units: KE = 0.5*m(amu)*(v Å/fs)^2  and we want energy in kcal/mol
    # Conversion: 1 amu*(Å/fs)^2 = 1.66054e-27 kg * (1e-10/1e-15)^2 m^2/s^2
    #           = 1.66054e-27 * 1e10 J = 1.66054e-17 J
    # 1 kcal/mol = 4184/6.022e23 J = 6.9477e-21 J
    # So 1 amu*(Å/fs)^2 = 1.66054e-17 / 6.9477e-21 kcal/mol = 2390.06 kcal/mol
    CONV = 2390.06  # amu*(Å/fs)^2 to kcal/mol
    k_spring = mass * omega_N**2 * CONV
    return k_spring


def make_ring_polymer(n_beads: int, mass: float, temperature: float,
                      init_pos: np.ndarray) -> RingPolymer:
    init_pos = np.asarray(init_pos, dtype=float)
    if init_pos.ndim == 1:
        init_pos = init_pos[None, :]
    # Initialize beads with small random displacement from mean
    rng = np.random.default_rng(7)
    sigma = np.sqrt(K_B_KCAL * temperature / (mass * 2390.06)) * 0.1
    positions = np.tile(init_pos, (n_beads, 1)) + rng.normal(0, sigma, (n_beads, init_pos.shape[-1]))

    # Maxwell-Boltzmann velocities for each bead
    CONV = 2390.06
    vel_sigma = np.sqrt(K_B_KCAL * temperature / (mass * CONV))
    velocities = rng.normal(0.0, vel_sigma, positions.shape)

    k_spring = _compute_k_spring(n_beads, mass, temperature)
    return RingPolymer(
        n_beads=n_beads,
        mass=mass,
        temperature=temperature,
        positions=positions,
        velocities=velocities,
        k_spring=k_spring,
    )


def spring_energy(rp: RingPolymer) -> float:
    e = 0.0
    for i in range(rp.n_beads):
        diff = rp.positions[(i + 1) % rp.n_beads] - rp.positions[i]
        e += 0.5 * rp.k_spring * float(np.dot(diff, diff))
    return e


def spring_forces(rp: RingPolymer) -> np.ndarray:
    n = rp.n_beads
    f = np.zeros_like(rp.positions)
    for i in range(n):
        r_prev = rp.positions[(i - 1) % n]
        r_next = rp.positions[(i + 1) % n]
        f[i] = -rp.k_spring * (2 * rp.positions[i] - r_prev - r_next)
    return f


def centroid_position(rp: RingPolymer) -> np.ndarray:
    return rp.positions.mean(axis=0)


def centroid_velocity(rp: RingPolymer) -> np.ndarray:
    return rp.velocities.mean(axis=0)
