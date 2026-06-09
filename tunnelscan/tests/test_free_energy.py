import numpy as np
import pytest
from tunnelscan.free_energy.boltzmann_inversion import boltzmann_inversion, stitch_windows
from tunnelscan.free_energy.bootstrapping import bootstrap_barrier, per_trajectory_barriers
from tunnelscan.classical_md.free_energy import wham_pmf
from tunnelscan.config import K_B_KCAL


def _fake_trajectory(xi_values):
    from tunnelscan.classical_md.production import ProductionTrajectory
    n = len(xi_values)
    positions = np.zeros((n, 3, 3))
    # Set donor at (0.9,0,0), H at (0,0,0), acceptor at (-2.5,0,0) for all frames
    positions[:, 0] = [0.9, 0.0, 0.0]
    positions[:, 2] = [-2.5, 0.0, 0.0]
    # Adjust H position to match xi
    for i, xv in enumerate(xi_values):
        # xi = dDH - dHA; dDH + dHA ≈ 3.4 (fixed total distance)
        # dDH = (3.4 + xv) / 2
        dDH = (3.4 + xv) / 2.0
        positions[i, 1] = [0.9 - dDH, 0.0, 0.0]
    return ProductionTrajectory(
        positions=positions,
        velocities=np.zeros((n, 3, 3)),
        xi=np.array(xi_values),
        energies=np.zeros(n),
        dt=1.0,
    )


def test_boltzmann_inversion():
    T = 300.0
    rng = np.random.default_rng(1)
    # Sample from Gaussian centered at -0.5 (reactant)
    xi_vals = rng.normal(-0.5, 0.3, size=5000)
    xi_min, xi_max = -2.0, 1.0
    counts, edges = np.histogram(xi_vals, bins=30, range=(xi_min, xi_max))
    centres, F = boltzmann_inversion(counts.astype(float), edges, temperature=T)

    # Find minimum of F
    i_min = np.argmin(F)
    x_min = centres[i_min]
    # Should be near -0.5
    assert abs(x_min - (-0.5)) < 0.3, f"FE minimum at {x_min}, expected ~-0.5"
    assert abs(F.min()) < 1e-10, f"FE minimum should be 0, got {F.min()}"

    # The parabola: F(x) ~ 0.5*k*(x-x0)^2 where k = kT/sigma^2
    k_expected = K_B_KCAL * T / 0.3**2
    # Check curvature roughly
    i_right = np.searchsorted(centres, x_min + 0.2)
    if 0 < i_right < len(F):
        expected_F = 0.5 * k_expected * 0.04  # (0.2)^2
        assert abs(F[i_right] - expected_F) < 0.1, (
            f"FE curvature wrong: got {F[i_right]:.4f}, expected ~{expected_F:.4f}"
        )


def test_bootstrap_ci():
    """Bootstrap 95% CI should capture true mean ~95% of the time."""
    rng = np.random.default_rng(42)
    true_mean = 5.0
    true_std = 1.0
    n_tests = 40
    captured = 0

    for _ in range(n_tests):
        # Generate 20 trajectories
        trajs = []
        for _ in range(20):
            # Gaussian xi centered at -1.0 (reactant) and 1.0 (product)
            xi_vals = np.concatenate([
                rng.normal(-1.0, 0.5, 100),
                rng.normal(1.0, 0.5, 10)
            ])
            trajs.append(_fake_trajectory(xi_vals))

        result = bootstrap_barrier(trajs, 0, 2, 1, n_bootstrap=200)
        lo, hi = result["ci_95"]
        if np.isfinite(lo) and np.isfinite(hi) and lo <= result["mean"] <= hi:
            captured += 1

    # CI should capture the point estimate nearly always
    assert captured >= n_tests * 0.80, (
        f"CI captured estimate only {captured}/{n_tests} times"
    )


def test_window_stitching():
    """Stitching two windows with known populations should give a continuous profile."""
    xi_edges = np.linspace(-2.0, 2.0, 16)

    # Window 1: peaked at -1.0
    xi1 = np.random.default_rng(10).normal(-1.0, 0.3, 500)
    counts1, _ = np.histogram(xi1, bins=xi_edges)

    # Window 2: peaked at 0.0
    xi2 = np.random.default_rng(11).normal(0.0, 0.3, 500)
    counts2, _ = np.histogram(xi2, bins=xi_edges)

    result = stitch_windows([counts1.astype(float), counts2.astype(float)],
                             xi_edges, temperature=300.0)

    assert result.free_energy.min() == pytest.approx(0.0, abs=1e-10)
    assert result.barrier_height >= 0.0
    assert len(result.xi_values) == 15
    # Profile should be continuous (no NaN)
    assert np.all(np.isfinite(result.free_energy))


def test_wham_recovers_known_pmf():
    """
    WHAM recovers an unbiased PMF from exact Gaussian umbrella samples.

    True PMF: F(xi) = 0.5 * k_true * xi^2 (harmonic, min at xi=0).
    Biased distribution for window i: Gaussian with
      mean = k_umbrella * xi0_i / (k_true + k_umbrella)
      variance = 1 / (beta * (k_true + k_umbrella))
    Exact histograms computed from the Gaussian CDF — no Monte Carlo noise.
    """
    from scipy.stats import norm as snorm

    T = 300.0
    beta = 1.0 / (K_B_KCAL * T)
    k_true = 1.0       # kcal/mol/Å²
    k_umb = 5.0        # kcal/mol/Å²
    k_eff = k_true + k_umb

    xi_edges = np.linspace(-3.0, 3.0, 25)
    xi_c = (xi_edges[:-1] + xi_edges[1:]) / 2.0
    xi0s = np.linspace(-2.0, 2.0, 9)

    hists = []
    N_per_window = 5000
    for xi0 in xi0s:
        mu = k_umb * xi0 / k_eff
        sigma = 1.0 / np.sqrt(beta * k_eff)
        # Exact histogram from CDF
        cdf = snorm.cdf(xi_edges, loc=mu, scale=sigma)
        counts = np.diff(cdf) * N_per_window
        hists.append(counts)

    F_wham, _ = wham_pmf(hists, xi_c, xi0s, k_umbrella=k_umb, temperature=T)

    # WHAM should be finite and zeroed at the minimum
    assert np.all(np.isfinite(F_wham[2:-2])), "WHAM produced non-finite values"
    assert abs(F_wham.min()) < 1e-8, f"Min not zero: {F_wham.min()}"

    # Minimum should be near xi=0 (true PMF minimum)
    i_min = np.argmin(F_wham[2:-2]) + 2
    assert abs(xi_c[i_min]) < 0.3, f"WHAM min at xi={xi_c[i_min]:.2f}, expected ~0"

    # At xi=1.0, F_true = 0.5*1.0*1^2 = 0.5 kcal/mol
    i_1 = np.argmin(np.abs(xi_c - 1.0))
    assert 0.2 < F_wham[i_1] < 0.9, (
        f"WHAM at xi≈1: {F_wham[i_1]:.3f}, expected ~0.5 kcal/mol"
    )
    # At xi=2.0, F_true = 2.0 kcal/mol
    i_2 = np.argmin(np.abs(xi_c - 2.0))
    assert 1.3 < F_wham[i_2] < 2.7, (
        f"WHAM at xi≈2: {F_wham[i_2]:.3f}, expected ~2.0 kcal/mol"
    )
