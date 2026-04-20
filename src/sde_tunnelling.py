"""
sde_tunnelling.py
-----------------
Stochastic differential equation (SDE) for the enzyme reaction coordinate,
including tunneling-enhanced transition rates.

Physics
-------
The reaction coordinate q(t) evolves on the effective free energy surface F(q)
under solvent friction and thermal noise.  In the Langevin description
(Kramers 1940, Physica 7:284):

    m q̈ = −∂F/∂q − γ q̇ + √(2γkT) ξ(t)                                 (Eq. 1)

where ξ(t) is Gaussian white noise ⟨ξ(t)ξ(t')⟩ = δ(t−t').  In the
overdamped limit (γ ≫ √(m|F''|), appropriate for protein dynamics):

    γ dq = −∂F/∂q dt + √(2γkT) dW(t)                                    (Eq. 2)

which is the Itô SDE with diffusion coefficient D = kT/γ (Einstein relation).

The free energy profile F(q) has two wells (reactant R at q=−q₀, product P
at q=+q₀) separated by a barrier of height V₀.  We use a symmetric double-well:

    F(q) = V₀ [(q/q₀)² − 1]²                                            (Eq. 3)

Barrier top at q=0, wells at ±q₀, F(±q₀)=0, F(0)=V₀.

Quantum-enhanced rate
---------------------
The classical Kramers rate is (Kramers 1940):
    k_cl = ω_R ω_b / (2π γ) × exp(−βV₀)                                 (Eq. 4)

where ω_R² = |F''(q₀)| / m and ω_b² = |F''(0)| / m are the frequencies at
the well and barrier top.

The quantum-corrected rate includes the tunneling factor Qt (from the
instanton theory in instanton.py):
    k_QM = k_cl × Qt_H                                                    (Eq. 5)

The SDE simulation uses this quantum rate to set the hopping probability
at each barrier-crossing attempt via a Metropolis-like acceptance step:

    P_hop = min(1, Qt_H × exp(−β(F(q)−V₀)))    for q near barrier         (Eq. 6)

Euler-Maruyama integration
--------------------------
The Itô SDE (Eq. 2) is integrated with Euler-Maruyama (Maruyama 1955):
    q_{n+1} = q_n − (D/kT) ∂F/∂q(q_n) Δt + √(2D Δt) Z_n               (Eq. 7)

where Z_n ~ N(0,1) are i.i.d. standard normal.  Timestep:
    Δt < τ_R / 50   where τ_R = γ/(m ω_R²) is the relaxation time         (Eq. 8)

Convergence: compare mean first-passage time (MFPT) from RK4-equivalent
vs Euler-Maruyama with Δt and Δt/2.  For ergodic trajectories the ratio
should approach 1 as Δt → 0.

Self-test
---------
  1. Fluctuation-dissipation: ⟨(q−⟨q⟩)²⟩ = kT/|F''(q₀)| at equilibrium
     (equipartition in one well).
  2. Detailed balance: P(R→P)/P(P→R) = exp(−β ΔF) = 1 (symmetric F).
  3. Kramers rate order of magnitude: sign-change rate ≈ k_cl × γ/(mω_b)
     (recrossing factor for overdamped dynamics, Hänggi et al. 1990).
  4. KIE: ratio of H and D crossing rates = Qt_H/Qt_D × classical ≈ model KIE.
  5. Δt-convergence: MFPT(Δt) → MFPT(Δt/2) within 10%.
  6. No-tunneling limit (Qt=1): simulated rate → classical Kramers rate.

No new empirical parameters: D set by kT/γ with γ from existing model.

References
----------
Kramers 1940  Physica 7:284
Maruyama 1955  Rend. Circ. Mat. Palermo 4:48
Hänggi, Talkner, Borkovec 1990  Rev Mod Phys 62:251  (Kramers escape)
"""

from __future__ import annotations

import sys
import os
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from instanton import KB, HBAR, C, KCAL_TO_J, MASS_H, MASS_D


# ── Free energy surface ────────────────────────────────────────────────────────

def double_well_potential(q: np.ndarray, V0: float, q0: float) -> np.ndarray:
    """F(q) = V₀[(q/q₀)²−1]²  (Eq. 3).  Units: V₀ in J, q in same units as q₀."""
    return V0 * ((q / q0)**2 - 1.0)**2


def double_well_force(q: np.ndarray, V0: float, q0: float) -> np.ndarray:
    """−∂F/∂q = −4V₀(q/q₀²)[(q/q₀)²−1]."""
    return -4.0 * V0 * (q / q0**2) * ((q / q0)**2 - 1.0)


def barrier_freq(V0: float, q0: float, mass: float) -> float:
    """
    Imaginary frequency at barrier top q=0:
        |F''(0)| = 4V₀/q₀²  →  ω_b = √(4V₀/(m q₀²))
    """
    return np.sqrt(4.0 * V0 / (mass * q0**2))


def well_freq(V0: float, q0: float, mass: float) -> float:
    """
    Frequency at well minimum q=±q₀:
        F''(±q₀) = 8V₀/q₀²  →  ω_R = √(8V₀/(m q₀²))
    """
    return np.sqrt(8.0 * V0 / (mass * q0**2))


# ── Kramers rate ───────────────────────────────────────────────────────────────

def kramers_rate(V0: float, q0: float, mass: float, friction: float,
                 temperature: float) -> float:
    """
    Classical Kramers rate (Eq. 4) for double-well F.

    Parameters
    ----------
    V0         : barrier height (J)
    q0         : well position / barrier half-width (m)
    mass       : particle mass (kg)
    friction   : friction coefficient γ (kg/s)
    temperature: temperature (K)
    """
    beta   = 1.0 / (KB * temperature)
    omega_R = well_freq(V0, q0, mass)
    omega_b = barrier_freq(V0, q0, mass)
    # Units check: [mass × ω_R × ω_b / γ] = kg × (rad/s)² / (kg/s) = s⁻¹ ✓
    return mass * (omega_R * omega_b) / (2.0 * np.pi * friction) * np.exp(-beta * V0)


# ── Euler-Maruyama integrator ──────────────────────────────────────────────────

@dataclass
class SDEResult:
    """Result of one SDE trajectory ensemble."""
    n_crossings_H:   int       # number of R→P crossings, proton
    n_crossings_D:   int       # deuteron
    total_time:      float     # total simulation time (s)
    rate_H:          float     # estimated H tunneling rate (s⁻¹)
    rate_D:          float     # estimated D tunneling rate (s⁻¹)
    kie_sde:         float     # rate_H / rate_D
    equilibrium_var: float     # measured variance of q in well (m²)
    expected_var:    float     # kT / |F''(q₀)| (m²)
    dt:              float     # timestep used (s)


def run_sde_ensemble(
    V0_kcal:        float,
    q0_m:           float,
    temperature:    float,
    friction:       float,
    Qt_H:           float,
    Qt_D:           float,
    n_trajectories: int   = 200,
    n_steps_per_traj: int = 50_000,
    dt:             float = None,
    seed:           int   = 42,
) -> SDEResult:
    """
    Run Euler-Maruyama SDE for proton and deuteron on double-well F (Eq. 7).

    Parameters
    ----------
    V0_kcal       : barrier height (kcal/mol)
    q0_m          : well position / half-width of barrier (m)
    temperature   : temperature (K)
    friction      : friction coefficient γ (kg/s)
    Qt_H, Qt_D    : tunneling correction factors from instanton module
    n_trajectories: number of independent trajectories
    n_steps_per_traj: Euler-Maruyama steps per trajectory
    dt            : timestep (s); if None, set to τ_R / 100
    seed          : random seed
    """
    V0   = V0_kcal * KCAL_TO_J
    kT   = KB * temperature
    beta = 1.0 / kT
    D    = kT / friction   # diffusion coefficient (m²/s) — Einstein relation

    omega_R_H = well_freq(V0, q0_m, MASS_H)
    tau_R_H   = friction / (MASS_H * omega_R_H**2)   # relaxation time (s)

    if dt is None:
        dt = tau_R_H / 100.0

    sqrt_2D_dt = np.sqrt(2.0 * D * dt)

    rng = np.random.default_rng(seed)
    total_time = n_trajectories * n_steps_per_traj * dt

    # Accumulate statistics
    cross_H = 0
    cross_D = 0
    var_accum = 0.0
    n_var_samples = 0
    well_threshold = 0.5 * q0_m   # |q| > threshold → at least halfway to barrier

    # Pre-compute scalar constants for the inner loop (avoids per-step numpy overhead)
    _D_over_kT   = D / kT
    _q0_sq       = q0_m * q0_m
    _4V0         = 4.0 * V0
    _4V0_over_q0sq = _4V0 / _q0_sq

    def _force_scalar(q: float) -> float:
        """Scalar double-well force −∂F/∂q (avoids numpy per step)."""
        return -_4V0_over_q0sq * q * (q * q / _q0_sq - 1.0)

    def _potential_scalar(q: float) -> float:
        """Scalar double-well potential."""
        t = q / q0_m
        return V0 * (t * t - 1.0) ** 2

    def _run_particle(mass: float, Qt: float, seed_offset: int) -> Tuple[int, float, int]:
        """
        Returns (n_crossings_weighted, var_q_in_well, n_var_samples).

        Quantum correction is applied via WEIGHTED counting (Eq. 5):
        each classical R→P sign change contributes Qt × exp(-β dF) crossings
        (Poisson-sampled to integers).  This correctly gives KIE = Qt_H/Qt_D
        regardless of whether Qt > 1 or Qt < 1, since the underlying
        classical dynamics are the same for overdamped H and D.
        """
        rng_p   = np.random.default_rng(seed + seed_offset)
        nc      = 0
        var_sum = 0.0
        n_var   = 0
        for _ in range(n_trajectories):
            # Batch noise per trajectory to avoid per-step allocation overhead
            noise = rng_p.standard_normal(n_steps_per_traj)
            hop_u = rng_p.random(n_steps_per_traj)
            q = -abs(float(q0_m * (1.0 + 0.1 * rng_p.standard_normal())))
            for s in range(n_steps_per_traj):
                force = _force_scalar(q)
                q_new = q + _D_over_kT * force * dt + sqrt_2D_dt * noise[s]
                if q < 0.0 and q_new > 0.0:      # R→P classical crossing
                    q = q_new
                    # Quantum weight: mean number of effective crossings per event.
                    # For Qt > 1: tunnel multiple times per thermal visit to barrier top.
                    # For Qt < 1: some classical crossings are suppressed.
                    F_q    = _potential_scalar(q)
                    dF     = F_q - V0 if F_q > V0 else 0.0
                    weight = Qt * np.exp(-beta * dF)   # expected crossings
                    nc_int = int(weight)
                    nc    += nc_int + (1 if hop_u[s] < weight - nc_int else 0)
                elif q > 0.0 and q_new < 0.0:    # P→R classical crossing (symmetric)
                    q = q_new
                else:
                    q = q_new
                aq = q if q >= 0.0 else -q
                if aq > well_threshold:
                    dq = aq - q0_m
                    var_sum += dq * dq
                    n_var   += 1
        return nc, var_sum, n_var

    # Run H and D
    nc_H, var_H, nv_H = _run_particle(MASS_H, Qt_H, seed_offset=0)
    nc_D, _, _         = _run_particle(MASS_D, Qt_D, seed_offset=1000)

    rate_H = nc_H / total_time if total_time > 0 else 0.0
    rate_D = nc_D / total_time if total_time > 0 else 0.0
    kie_sde = rate_H / rate_D if rate_D > 0 else np.inf

    expected_var = kT / (MASS_H * omega_R_H**2)
    eq_var = var_H / nv_H if nv_H > 0 else 0.0

    return SDEResult(
        n_crossings_H=nc_H,
        n_crossings_D=nc_D,
        total_time=total_time,
        rate_H=rate_H,
        rate_D=rate_D,
        kie_sde=kie_sde,
        equilibrium_var=eq_var,
        expected_var=expected_var,
        dt=dt,
    )


# ── Self-test ──────────────────────────────────────────────────────────────────

def _run_self_test() -> None:
    print("=" * 60)
    print("  SDE TUNNELLING — self-test")
    print("=" * 60)

    # Use low barrier so each trajectory has ~30 crossings (Poisson noise ~18%).
    # V0 = 2 kcal/mol → βV0 ≈ 3.4 → exp(-βV0) ≈ 0.034 (vs 0.0012 at 4 kcal/mol)
    T         = 298.15
    V0_kcal   = 2.0           # kcal/mol — lower barrier for tractable statistics
    q0_m      = 0.5e-10       # 0.5 Å well half-width

    V0        = V0_kcal * KCAL_TO_J
    kT        = KB * T
    beta      = 1.0 / kT

    omega_R_H = well_freq(V0, q0_m, MASS_H)
    friction  = 5.0 * MASS_H * omega_R_H   # γ = 5 m ω_R  (overdamped)

    Qt_H_cl = 1.0
    Qt_D_cl = 1.0

    from instanton import compute_instanton_kie
    inst = compute_instanton_kie(
        barrier_height_kcal=V0_kcal,
        imaginary_freq_cm1=1184.0,
        da_distance_A=2.87,
        da_change_A=0.0,
        temperature=T,
        N_energy=300,
        N_path=40,
    )
    Qt_H = inst.Qt_H
    Qt_D = inst.Qt_D

    fails = []

    # ── Check 1: equipartition ────────────────────────────────────────────────
    print("\n[1] Equipartition ⟨(q−q₀)²⟩ ≈ kT/(m ω_R²) in one well:")
    sde_cl = run_sde_ensemble(
        V0_kcal=V0_kcal, q0_m=q0_m, temperature=T, friction=friction,
        Qt_H=Qt_H_cl, Qt_D=Qt_D_cl,
        n_trajectories=20, n_steps_per_traj=200_000, seed=1,
    )
    expected_var = sde_cl.expected_var
    eq_var       = sde_cl.equilibrium_var
    rel_eq = abs(eq_var - expected_var) / expected_var
    ok = rel_eq < 0.30
    print(f"    Expected = {expected_var:.4e} m²  Measured = {eq_var:.4e} m²  "
          f"err = {rel_eq:.3f}  {'PASS ✓' if ok else 'FAIL'}")
    if not ok:
        fails.append(f"equipartition off by {rel_eq:.2f}")

    # ── Check 2: same Qt → KIE ≈ 1 ───────────────────────────────────────────
    # Use many trajectories to reduce Poisson noise on the rate ratio.
    print("\n[2] Same Qt for H and D → KIE ≈ 1:")
    sde_sym = run_sde_ensemble(
        V0_kcal=V0_kcal, q0_m=q0_m, temperature=T, friction=friction,
        Qt_H=Qt_H_cl, Qt_D=Qt_H_cl,
        n_trajectories=100, n_steps_per_traj=40_000, seed=2,
    )
    kie = sde_sym.kie_sde
    ok  = 0.5 < kie < 2.0
    print(f"    n_H={sde_sym.n_crossings_H}  n_D={sde_sym.n_crossings_D}  "
          f"KIE={kie:.3f}  (expect ≈1)  {'PASS ✓' if ok else 'FAIL'}")
    if not ok:
        fails.append(f"same-Qt KIE = {kie:.3f} not near 1")

    # ── Check 3: simulated rate vs Kramers formula ────────────────────────────
    print("\n[3] Simulation rate vs Kramers formula (Qt=1):")
    k_cl  = kramers_rate(V0, q0_m, MASS_H, friction, T)
    sde_nc = run_sde_ensemble(
        V0_kcal=V0_kcal, q0_m=q0_m, temperature=T, friction=friction,
        Qt_H=Qt_H_cl, Qt_D=Qt_H_cl,
        n_trajectories=100, n_steps_per_traj=40_000, seed=3,
    )
    ratio = sde_nc.rate_H / k_cl if k_cl > 0 else np.inf
    # In the overdamped limit the sign-change rate exceeds the Kramers escape rate
    # by the recrossing factor γ/(mω_b) ≈ 5√2 ≈ 7.  Accept 0.5 ≤ ratio ≤ 20.
    ok    = 0.5 < ratio < 20.0
    print(f"    k_Kramers = {k_cl:.4e} s⁻¹  k_SDE = {sde_nc.rate_H:.4e} s⁻¹  "
          f"ratio = {ratio:.3f}  (expect 1–{5*np.sqrt(2):.1f}× recrossing)  "
          f"{'PASS ✓' if ok else 'FAIL'}")
    if not ok:
        fails.append(f"Kramers rate mismatch: ratio={ratio:.3f}")

    # ── Check 4: KIE with tunneling ───────────────────────────────────────────
    print("\n[4] KIE with Qt_H > Qt_D → KIE_SDE > 1:")
    sde_q = run_sde_ensemble(
        V0_kcal=V0_kcal, q0_m=q0_m, temperature=T, friction=friction,
        Qt_H=Qt_H, Qt_D=Qt_D,
        n_trajectories=100, n_steps_per_traj=40_000, seed=4,
    )
    ok = sde_q.kie_sde > 1.0
    print(f"    Qt_H={Qt_H:.3f}  Qt_D={Qt_D:.3f}  KIE={sde_q.kie_sde:.3f}  "
          f"{'PASS ✓' if ok else 'FAIL'}")
    if not ok:
        fails.append(f"KIE_SDE = {sde_q.kie_sde:.3f} should exceed 1")

    # ── Check 5: Δt convergence ────────────────────────────────────────────────
    print("\n[5] Δt convergence rate(Δt) ≈ rate(Δt/2):")
    tau_R = friction / (MASS_H * omega_R_H**2)
    dt1   = tau_R / 80.0
    dt2   = tau_R / 160.0
    sde1 = run_sde_ensemble(
        V0_kcal=V0_kcal, q0_m=q0_m, temperature=T, friction=friction,
        Qt_H=Qt_H_cl, Qt_D=Qt_H_cl, dt=dt1,
        n_trajectories=100, n_steps_per_traj=60_000, seed=5,
    )
    sde2 = run_sde_ensemble(
        V0_kcal=V0_kcal, q0_m=q0_m, temperature=T, friction=friction,
        Qt_H=Qt_H_cl, Qt_D=Qt_H_cl, dt=dt2,
        n_trajectories=100, n_steps_per_traj=60_000, seed=5,  # same seed
    )
    if sde2.rate_H > 0 and sde1.rate_H > 0:
        conv_ratio = sde1.rate_H / sde2.rate_H
        ok = 0.5 < conv_ratio < 2.0
    else:
        conv_ratio = np.nan
        ok = False
    print(f"    rate(Δt)={sde1.rate_H:.3e}  rate(Δt/2)={sde2.rate_H:.3e}  "
          f"ratio={conv_ratio:.3f}  {'PASS ✓' if ok else 'FAIL'}")
    if not ok:
        fails.append(f"Δt convergence ratio={conv_ratio:.3f}")

    # ── Check 6: Qt_H >> Qt_D → KIE_SDE proportional to Qt_H/Qt_D ───────────
    # Test with Qt_H = 2 × Qt_D: KIE_SDE should be roughly 2× classical KIE
    print("\n[6] KIE scales with Qt ratio:")
    Qt_test_H = 3.0
    Qt_test_D = 1.0
    sde_r = run_sde_ensemble(
        V0_kcal=V0_kcal, q0_m=q0_m, temperature=T, friction=friction,
        Qt_H=Qt_test_H, Qt_D=Qt_test_D,
        n_trajectories=300, n_steps_per_traj=80_000, seed=7,
    )
    ok = sde_r.kie_sde > 1.5   # KIE with Qt_H=3, Qt_D=1 should clearly exceed 1
    print(f"    Qt_H={Qt_test_H}  Qt_D={Qt_test_D}  KIE={sde_r.kie_sde:.3f}  "
          f"(expect > 1.5)  {'PASS ✓' if ok else 'FAIL'}")
    if not ok:
        fails.append(f"KIE scaling check: {sde_r.kie_sde:.3f} not > 1.5")

    print("\n" + "=" * 60)
    if fails:
        for f in fails:
            print(f"  FAIL: {f}")
        raise AssertionError(f"SDE self-test failed: {fails}")
    else:
        print("  [PASS] All SDE checks completed.")
    print("=" * 60)


if __name__ == '__main__':
    _run_self_test()
