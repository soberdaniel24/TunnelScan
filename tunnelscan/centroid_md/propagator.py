from __future__ import annotations
import numpy as np
from dataclasses import replace
from tunnelscan.config import K_B_KCAL
from tunnelscan.centroid_md.ring_polymer import RingPolymer

# amu*(Å/fs)^2 to kcal/mol conversion
_CONV = 2390.06


def propagate_step(rp: RingPolymer, physical_forces: np.ndarray,
                   C: np.ndarray, freqs: np.ndarray,
                   dt_slow: float = 1.0, dt_fast: float = 0.1,
                   gamma: float = 1.0, temperature: float = 300.0) -> RingPolymer:
    """
    RESPA propagator for ring polymer centroid MD.

    physical_forces: (n_beads, 3) forces in kcal/mol/Å — same force applied to all beads
                     (centroid approximation: each bead feels the physical force)
    """
    from tunnelscan.centroid_md.normal_modes import bead_to_normal, normal_to_bead

    N = rp.n_beads
    m = rp.mass          # amu
    dt = dt_slow

    # Convert to normal mode space
    # q: normal mode positions (N, 3), p: normal mode momenta = m * v_modes (in amu*Å/fs)
    q = bead_to_normal(rp.positions, C)    # (N, 3) Å
    p = bead_to_normal(rp.velocities, C)   # (N, 3) Å/fs  (velocities)

    # Force on each bead (physical) — same for all beads in centroid approx
    # In normal mode space, F_k = C @ F_beads
    F_bead = physical_forces  # (N, 3) kcal/mol/Å
    F_nm = bead_to_normal(F_bead, C)  # (N, 3) kcal/mol/Å

    # Acceleration conversion: a = F/(m*CONV) in Å/fs² for F in kcal/mol/Å
    force_to_dv = 1.0 / (m * _CONV)  # (Å/fs) per (kcal/mol/Å) per fs

    # --- RESPA slow step ---
    # Half-kick: all modes get physical force (only centroid / mode 0 for slow forces)
    # Mode 0 is the centroid: gets full physical force
    p[0] += F_nm[0] * force_to_dv * dt / 2.0

    # --- Fast steps: internal modes (k>0) with analytical harmonic propagation ---
    n_fast = max(1, int(round(dt / dt_fast)))
    dt_f = dt / n_fast

    # Centroid position update (full slow step) — split into n_fast micro-steps
    for _ in range(n_fast):
        # Update centroid position
        q[0] += p[0] * dt_f

        # Analytically propagate internal modes
        for k in range(1, N):
            omega = freqs[k]  # rad/fs
            if omega < 1e-14:
                q[k] += p[k] * dt_f
                continue
            # Spring force: F_k = -m*omega^2 * q_k (in CONV units: F = -m*CONV*omega^2 * q)
            # Analytical harmonic oscillator: exact solution over dt_f
            cos_w = np.cos(omega * dt_f)
            sin_w = np.sin(omega * dt_f)
            # q in Å, p in Å/fs, spring in kcal/mol/Å²
            # Effective: dq/dt = p, dp/dt = -m*CONV*omega^2 * q (in appropriate units)
            # Mass-weighted: let u = q, du/dt = p, dp/dt = -omega^2 * u (if p = v, omega in rad/fs)
            q_new = q[k] * cos_w + p[k] / omega * sin_w
            p_new = -q[k] * omega * sin_w + p[k] * cos_w
            q[k] = q_new
            p[k] = p_new

    # Second half-kick from physical force on mode 0
    p[0] += F_nm[0] * force_to_dv * dt / 2.0

    # Apply Langevin thermostat to centroid only
    if gamma > 0.0:
        rng = np.random.default_rng()
        # gamma in ps^-1 = 0.001 /fs
        gamma_fs = gamma * 1e-3
        c1 = np.exp(-gamma_fs * dt)
        # noise: sigma^2 = kT/m * (1 - c1^2) in (Å/fs)^2
        noise_var = K_B_KCAL * temperature / (m * _CONV) * (1.0 - c1**2)
        noise_sigma = np.sqrt(max(noise_var, 0.0))
        p[0] = c1 * p[0] + noise_sigma * rng.standard_normal(p[0].shape)

    # Convert back to bead space
    new_pos = normal_to_bead(q, C)
    new_vel = normal_to_bead(p, C)

    return replace(rp, positions=new_pos, velocities=new_vel)
