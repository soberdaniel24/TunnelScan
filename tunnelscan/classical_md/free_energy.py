from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class FreeEnergyResult:
    xi_values: np.ndarray
    free_energy: np.ndarray
    barrier_height: float
    reactant_energy: float
    ts_energy: float
    zpe: float
    method: str
    n_trajectories: int
    n_frames: int


def compute_free_energy_from_trajectory(traj, n_bins: int = 15,
                                         method: str = "classical") -> FreeEnergyResult:
    from tunnelscan.config import K_B_KCAL, T_DEFAULT
    T = T_DEFAULT

    xi = traj.xi
    xi_min, xi_max = xi.min() - 0.01, xi.max() + 0.01
    counts, edges = np.histogram(xi, bins=n_bins, range=(xi_min, xi_max))
    centres = (edges[:-1] + edges[1:]) / 2.0

    counts = counts.astype(float)
    counts[counts < 1] = 1.0

    F = -K_B_KCAL * T * np.log(counts)
    F -= F.min()

    barrier = float(F.max() - F[centres < 0.0].min()) if np.any(centres < 0.0) else float(F.max())
    reactant = float(F[centres < 0.0].min()) if np.any(centres < 0.0) else float(F.min())
    ts = float(F.max())

    return FreeEnergyResult(
        xi_values=centres,
        free_energy=F,
        barrier_height=barrier,
        reactant_energy=reactant,
        ts_energy=ts,
        zpe=float("nan"),
        method=method,
        n_trajectories=1,
        n_frames=len(xi),
    )


def wham_pmf(biased_histograms: list[np.ndarray],
             xi_centers: np.ndarray,
             xi0_values: np.ndarray,
             k_umbrella: float,
             temperature: float = 300.0,
             n_iter: int = 500,
             tol: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """
    Weighted Histogram Analysis Method (WHAM).

    Solves the self-consistent WHAM equations to recover the unbiased PMF
    from a set of umbrella-sampling windows.

    Parameters
    ----------
    biased_histograms : list of 1-D count arrays, one per window
    xi_centers        : bin centre positions (Å)
    xi0_values        : umbrella window centres (Å)
    k_umbrella        : spring constant (kcal/mol/Å²)
    temperature       : K
    n_iter            : maximum WHAM iterations
    tol               : convergence criterion on free energies (kcal/mol)

    Returns
    -------
    F  : unbiased PMF (kcal/mol), zeroed at minimum
    P  : normalised unbiased probability
    """
    from tunnelscan.config import K_B_KCAL

    beta = 1.0 / (K_B_KCAL * temperature)
    n_windows = len(biased_histograms)
    n_bins = len(xi_centers)

    H = np.array([h.astype(float) for h in biased_histograms])  # (W, B)
    N_i = H.sum(axis=1)                                          # (W,) counts per window

    # Bias potential for every (window, bin): w[i,j] = 0.5*k*(xi_j - xi0_i)^2
    w = 0.5 * k_umbrella * (xi_centers[None, :] - xi0_values[:, None]) ** 2  # (W, B)
    bw = beta * w  # (W, B) dimensionless

    # Initialise free energies (log partition functions in kT units)
    lnZ = np.zeros(n_windows)

    for iteration in range(n_iter):
        # WHAM equation 1: unbiased population at each bin
        #   P(xi_j) = [Σ_i H_i(xi_j)] / [Σ_k N_k * exp(-β w_k(xi_j) + lnZ_k)]
        log_denom = np.logaddexp.reduce(
            np.log(N_i[:, None] + 1e-300) - bw + lnZ[:, None], axis=0
        )  # (B,)
        log_num = np.log(H.sum(axis=0) + 1e-300)  # (B,)
        log_P = log_num - log_denom

        # Normalise
        log_P -= np.logaddexp.reduce(log_P)

        # WHAM eq 2: lnZ_i = β*f_i = -ln(Z_i), Z_i = Σ_j P(ξ_j)*exp(-β*w_i(ξ_j))
        lnZ_new = -np.logaddexp.reduce(log_P[None, :] - bw, axis=1)  # (W,)

        # Remove arbitrary additive constant (set first window to 0)
        lnZ_new -= lnZ_new[0]
        lnZ -= lnZ[0]

        if np.max(np.abs(lnZ_new - lnZ)) < tol:
            lnZ = lnZ_new
            break
        lnZ = lnZ_new

    P = np.exp(log_P)
    P[P < 1e-300] = 1e-300
    F = -np.log(P) / beta
    F -= F.min()
    return F, P


def _pmf_stats(F: np.ndarray, centres: np.ndarray, n_windows: int,
               n_frames: int, method: str) -> FreeEnergyResult:
    mask = centres < 0.0
    reactant = float(F[mask].min()) if mask.any() else float(F.min())
    ts = float(F.max())
    barrier = float(ts - reactant)
    return FreeEnergyResult(
        xi_values=centres,
        free_energy=F,
        barrier_height=barrier,
        reactant_energy=reactant,
        ts_energy=ts,
        zpe=float("nan"),
        method=method,
        n_trajectories=n_windows,
        n_frames=n_frames,
    )


def compute_pmf_umbrella(engine, eq_result, donor_idx: int, acceptor_idx: int,
                          h_idx: int, n_windows: int = 15,
                          k_umbrella: float = 50.0,
                          n_steps_per_window: int = 200) -> FreeEnergyResult:
    """
    Umbrella sampling + WHAM to compute the classical PMF.

    Places n_windows harmonic restraints evenly from xi=-2 to +2 Å,
    runs n_steps_per_window MD steps in each, then unbiases via WHAM.
    """
    from tunnelscan.config import T_DEFAULT
    from tunnelscan.classical_md.production import run_production

    xi0_values = np.linspace(-2.0, 2.0, n_windows)
    xi_edges = np.linspace(-2.5, 2.5, n_windows + 1)
    xi_centers = (xi_edges[:-1] + xi_edges[1:]) / 2.0

    per_window_histograms: list[np.ndarray] = []
    per_window_xi: list[np.ndarray] = []

    for xi_0 in xi0_values:
        class _Biased:
            def __init__(self, base, k, xi0, d, a, h):
                self.base = base
                self.k = k
                self.xi0 = xi0
                self.d = d
                self.a = a
                self.h = h
                self.atoms = base.atoms

            def energy_and_forces(self):
                e, f = self.base.energy_and_forces()
                pos = self.atoms.get_positions()
                r_D, r_H, r_A = pos[self.d], pos[self.h], pos[self.a]
                dDH = np.linalg.norm(r_D - r_H) + 1e-12
                dHA = np.linalg.norm(r_H - r_A) + 1e-12
                xi_val = dDH - dHA
                dphi = self.k * (xi_val - self.xi0)
                e += 0.5 * self.k * (xi_val - self.xi0) ** 2
                f[self.h] -= dphi * (-(r_D - r_H) / dDH + (r_H - r_A) / dHA)
                f[self.d] -= dphi * (r_D - r_H) / dDH
                f[self.a] += dphi * (r_H - r_A) / dHA
                return e, f

            def update_positions(self, positions):
                self.base.update_positions(positions)

        biased = _Biased(engine, k_umbrella, xi_0,
                         donor_idx, acceptor_idx, h_idx)
        traj = run_production(biased, eq_result, donor_idx, acceptor_idx, h_idx,
                              n_steps=n_steps_per_window, dt=1.0)
        counts, _ = np.histogram(traj.xi, bins=xi_edges)
        per_window_histograms.append(counts.astype(float))
        per_window_xi.append(traj.xi)

    # WHAM unbiasing
    F, _ = wham_pmf(per_window_histograms, xi_centers, xi0_values,
                    k_umbrella, temperature=T_DEFAULT)

    n_frames = sum(len(x) for x in per_window_xi)
    return _pmf_stats(F, xi_centers, n_windows, n_frames, "wham")
