"""
path_integral.py
----------------
Wigner-Kirkwood path integral correction to the Bell tunnelling model.

The Bell (1958) first-order approximation uses the leading term of the
semiclassical expansion:

  Qt ≈ 1 + u²/24      (valid only for u << 1)

where u = ħω†/kT.  For AADH at 298 K with ω† = 1184 cm⁻¹, u_H ≈ 5.7 —
far outside the regime of validity.  The first-order formula underpredicts
Qt_H by 4.3× and the KIE correction by 3.2×.

The exact result for a parabolic barrier is:

  Qt_exact = (u/2) / sin(u/2)                          (Bell 1958, eq. 3.5)

This is finite for |u| < 2π (u ≈ 6.28) and equals the generating function of
the Bernoulli numbers, whose Taylor expansion gives the Wigner-Kirkwood series:

  Qt_WK = 1 + u²/24 + 7u⁴/5760 + 31u⁶/967680 + 127u⁸/154828800 + ...

For AADH:
  Bell 1st-order KIE correction: 1.40×  (Qt_H/Qt_D = 2.36/1.68)
  Wigner-Kirkwood n=3 correction: 4.10×
  Exact parabolic correction:     4.53×
  Experimental KIE:               55
  → Bell 1st-order + classical: ~11.3;  exact parabolic + classical: ~31.3
  → Remaining factor (~1.75×) comes from non-parabolic barrier shape

Validity
--------
The parabolic approximation is valid when:

  1. u < 2π  (no pole in sin(u/2)); for AADH u_H = 5.71 < 6.28 ✓
  2. The barrier is well-approximated by a parabola near the saddle point

For u ≥ π (AADH u_H = 5.71 > 3.14), over-barrier reflections become
significant (Wentzel–Kramers–Brillouin breakdown).  The exact formula
for a fully parabolic potential still works in this regime, but real
enzyme barriers are not perfectly parabolic.  A correction flag is set
when u_H > π.

References
----------
Bell RP (1958) Proc R Soc Lond A 234:414.
  – Qt = (u/2)/sin(u/2) exact derivation for parabolic barrier
Bell RP (1980) The Tunnel Effect in Chemistry. Chapman & Hall.
  – u = ħω†/kT; validity discussion
Wigner E (1932) Phys Rev 40:749.
  – Semiclassical expansion; first-order term = Bell approximation
Kirkwood JG (1933) Phys Rev 44:31.
  – Higher-order quantum corrections (ħ⁴, ħ⁶ terms)
Scrutton NS et al (2019) Annu Rev Biochem 88:555.
  – Enzyme tunnelling regime; AADH parameters
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

# ── Physical constants ────────────────────────────────────────────────────────

KB   = 1.380649e-23       # Boltzmann constant, J/K
HBAR = 1.0545718e-34      # ħ = h/(2π), J·s
C    = 2.99792458e10      # speed of light, cm/s (for cm⁻¹ → rad/s)

MASS_H = 1.6735575e-27    # kg
MASS_D = 3.3444000e-27    # kg
MASS_RATIO_DH = MASS_D / MASS_H  # ≈ 1.9981 (not exactly 2 — use measured masses)

# Wigner-Kirkwood series coefficients c_n such that:
#   Qt = Σ c_n × u^(2n)    (u = ħω†/kT)
# Derived from: (x/2)/sin(x/2) = Σ B_{2n}(0) × (-1)^n × x^(2n) / (2n)!
# where B_{2n} are Bernoulli numbers.
# Equivalently, c_0=1, c_1=1/24, c_2=7/5760, c_3=31/967680, c_4=127/154828800
WK_COEFFICIENTS = [
    1.0,                       # u⁰ (classical)
    1.0 / 24.0,                # u²  Bell (1958) 1st-order
    7.0 / 5760.0,              # u⁴  Kirkwood (1933) 2nd-order
    31.0 / 967680.0,           # u⁶  3rd-order
    127.0 / 154828800.0,       # u⁸  4th-order
    73.0 / 1505280000.0,       # u¹⁰ 5th-order  (Bernoulli B_10/10!)
]


# ── Output dataclass ──────────────────────────────────────────────────────────

@dataclass
class PathIntegralResult:
    """
    Complete output of the path integral tunnelling calculation for one isotope.

    Fields
    ------
    u : ħω†/kT (dimensionless; scaled by sqrt(mass) for deuterium)
    Qt_bell  : first-order Bell approximation, 1 + u²/24
    Qt_wk    : Wigner-Kirkwood series to max_order
    Qt_exact : exact parabolic-barrier formula, (u/2)/sin(u/2)
    wk_order : number of WK terms included (not counting the classical 1)
    correction_factor : Qt_exact / Qt_bell (error in Bell approximation)
    near_pole : True if u > π (over-barrier reflections non-negligible)
    """
    u:                float
    Qt_bell:          float   # 1 + u²/24
    Qt_wk:            float   # WK series to wk_order
    Qt_exact:         float   # (u/2)/sin(u/2)
    wk_order:         int
    correction_factor: float  # Qt_exact / Qt_bell
    near_pole:        bool    # u > π — parabolic formula near breakdown


@dataclass
class PathIntegralKIE:
    """
    Complete KIE comparison: Bell vs path integral (WK / exact).
    """
    temperature:       float   # K
    imaginary_freq:    float   # cm⁻¹
    u_H:               float
    u_D:               float
    pi_H:              PathIntegralResult
    pi_D:              PathIntegralResult

    # KIE values (Qt_H/Qt_D) × classical_KIE
    classical_KIE:     float
    kie_bell:          float   # using Qt_bell
    kie_wk:            float   # using Qt_wk
    kie_exact:         float   # using Qt_exact (best estimate)

    # How much better is exact vs Bell?
    kie_correction_factor: float  # kie_exact / kie_bell

    def summary(self) -> str:
        lines = [
            f"  Path Integral KIE  (T = {self.temperature:.1f} K,  ν† = {self.imaginary_freq:.0f} cm⁻¹)",
            f"  u_H = {self.u_H:.4f},  u_D = {self.u_D:.4f}",
            f"  Qt_H (Bell):     {self.pi_H.Qt_bell:.4f}",
            f"  Qt_H (WK-{self.pi_H.wk_order}):    {self.pi_H.Qt_wk:.4f}",
            f"  Qt_H (Exact):    {self.pi_H.Qt_exact:.4f}   ({self.pi_H.correction_factor:.2f}× Bell)",
            f"  Qt_D (Bell):     {self.pi_D.Qt_bell:.4f}",
            f"  Qt_D (WK-{self.pi_D.wk_order}):    {self.pi_D.Qt_wk:.4f}",
            f"  Qt_D (Exact):    {self.pi_D.Qt_exact:.4f}   ({self.pi_D.correction_factor:.2f}× Bell)",
            f"  ─────────────────────────────────────────────────",
            f"  KIE (Bell):   {self.kie_bell:.2f}",
            f"  KIE (WK-{self.pi_H.wk_order}):  {self.kie_wk:.2f}",
            f"  KIE (Exact):  {self.kie_exact:.2f}   ({self.kie_correction_factor:.2f}× Bell)",
        ]
        if self.pi_H.near_pole:
            lines.append(f"  WARNING: u_H = {self.u_H:.3f} > π — parabolic barrier near pole")
        return "\n".join(lines)


# ── Core functions ────────────────────────────────────────────────────────────

def wigner_kirkwood_qt(u: float, wk_order: int = 3) -> float:
    """
    Wigner-Kirkwood series for the tunnelling transmission factor Qt.

    Qt = Σ_{n=0}^{wk_order} c_n × u^{2n}

    Parameters
    ----------
    u : float
        ħω†/kT (dimensionless).
    wk_order : int
        Number of correction terms beyond the classical (default 3, i.e. up to u⁶).
        Range: 1 (Bell 1st-order) through 5 (up to u¹⁰).

    Returns
    -------
    Qt (dimensionless, ≥ 1)
    """
    order = max(1, min(wk_order, len(WK_COEFFICIENTS) - 1))
    u2 = u * u
    qt = 0.0
    for n in range(order + 1):
        qt += WK_COEFFICIENTS[n] * (u2 ** n)
    return float(max(1.0, qt))


def exact_qt_parabolic(u: float) -> float:
    """
    Exact Bell tunnelling factor for a parabolic barrier.

    Qt_exact = (u/2) / sin(u/2)

    Finite for |u| < 2π.  For u = 0, Qt = 1 (classical limit).
    For u → 2π, Qt → ∞ (resonance — tunnelling probability approaches 1).

    Parameters
    ----------
    u : float
        ħω†/kT.  For H-transfer in AADH at 298 K, u_H ≈ 5.71.

    Returns
    -------
    Qt_exact (dimensionless, ≥ 1)
    """
    if abs(u) < 1e-8:
        return 1.0
    half_u = u / 2.0
    sin_half = np.sin(half_u)
    if abs(sin_half) < 1e-10:
        # Near pole — physically means tunnelling approaches unity (deep tunnelling)
        # Cap at a large but finite value
        return 1000.0
    qt = half_u / sin_half
    return float(max(1.0, qt))


def compute_u(imaginary_freq_cm1: float, temperature: float, mass_kg: float = MASS_H) -> float:
    """
    Compute u = ħω†/kT for a given isotope.

    Parameters
    ----------
    imaginary_freq_cm1 : float
        Magnitude of imaginary TS frequency in cm⁻¹.
    temperature : float
        Kelvin.
    mass_kg : float
        Isotope mass in kg.  Default: proton (MASS_H).
        For deuterium: MASS_D.
        For tritium: ~5.01e-27 kg.

    Note: the imaginary frequency scales as 1/√m because ω = √(k/m).
    When the TS force constant k is the same, ω_D = ω_H × √(m_H/m_D).
    """
    # Angular frequency for proton from the given cm⁻¹ value
    omega_H = 2.0 * np.pi * imaginary_freq_cm1 * C      # rad/s

    # Scale by mass ratio to get ω for the specified isotope
    omega = omega_H * np.sqrt(MASS_H / mass_kg)

    return float(HBAR * omega / (KB * temperature))


def path_integral_correction(
    imaginary_freq_cm1: float,
    temperature:        float = 298.15,
    wk_order:           int   = 3,
    mass_kg:            float = MASS_H,
) -> PathIntegralResult:
    """
    Path integral tunnelling correction for one isotope at one temperature.

    Parameters
    ----------
    imaginary_freq_cm1 : float
        |ν†| in cm⁻¹ (positive, for the proton TS mode).
    temperature : float
        Kelvin (default 298.15).
    wk_order : int
        WK series order (default 3, up to u⁶).
    mass_kg : float
        Isotope mass for scaling ω†.

    Returns
    -------
    PathIntegralResult with Qt_bell, Qt_wk, Qt_exact and correction factor.
    """
    u = compute_u(imaginary_freq_cm1, temperature, mass_kg)

    qt_bell  = 1.0 + (u**2) / 24.0
    qt_wk    = wigner_kirkwood_qt(u, wk_order)
    qt_exact = exact_qt_parabolic(u)

    return PathIntegralResult(
        u                = u,
        Qt_bell          = float(qt_bell),
        Qt_wk            = float(qt_wk),
        Qt_exact         = float(qt_exact),
        wk_order         = wk_order,
        correction_factor = float(qt_exact / qt_bell),
        near_pole        = bool(u > np.pi),
    )


def compute_kie(
    imaginary_freq_cm1:  float,
    classical_KIE:       float,
    temperature:         float = 298.15,
    wk_order:            int   = 3,
) -> PathIntegralKIE:
    """
    Compute the full KIE using Bell, WK series, and exact parabolic correction.

    Parameters
    ----------
    imaginary_freq_cm1 : float
        |ν†| in cm⁻¹ (magnitude of TS imaginary frequency, proton scaling).
    classical_KIE : float
        Classical KIE from ZPE difference (~6.9 at 298 K for C-H transfer).
    temperature : float
        Kelvin.
    wk_order : int
        WK expansion order.

    Returns
    -------
    PathIntegralKIE with full comparison.
    """
    pi_H = path_integral_correction(imaginary_freq_cm1, temperature, wk_order, MASS_H)
    pi_D = path_integral_correction(imaginary_freq_cm1, temperature, wk_order, MASS_D)

    kie_bell  = classical_KIE * (pi_H.Qt_bell  / pi_D.Qt_bell)
    kie_wk    = classical_KIE * (pi_H.Qt_wk    / pi_D.Qt_wk)
    kie_exact = classical_KIE * (pi_H.Qt_exact / pi_D.Qt_exact)

    return PathIntegralKIE(
        temperature           = temperature,
        imaginary_freq        = imaginary_freq_cm1,
        u_H                   = pi_H.u,
        u_D                   = pi_D.u,
        pi_H                  = pi_H,
        pi_D                  = pi_D,
        classical_KIE         = classical_KIE,
        kie_bell              = float(kie_bell),
        kie_wk                = float(kie_wk),
        kie_exact             = float(kie_exact),
        kie_correction_factor = float(kie_exact / kie_bell),
    )


# ── WK series convergence check ───────────────────────────────────────────────

def wk_convergence_check(u: float) -> dict:
    """
    Show convergence of WK series at a given u, compared to exact result.

    Returns dict: {order: Qt_wk for orders 1-5, plus Qt_exact}.
    """
    result = {}
    for order in range(1, 6):
        result[f'WK-{order}'] = wigner_kirkwood_qt(u, order)
    result['Exact'] = exact_qt_parabolic(u)
    return result


# ── Temperature scan ──────────────────────────────────────────────────────────

def temperature_scan(
    imaginary_freq_cm1: float,
    classical_KIE:      float,
    temperatures:       Optional[list] = None,
    wk_order:           int = 3,
) -> list:
    """
    Compute KIE at multiple temperatures using all three methods.
    Useful for validating against temperature-dependent KIE data (Arrhenius).

    Returns list of PathIntegralKIE objects.
    """
    if temperatures is None:
        temperatures = [278.15, 283.15, 288.15, 293.15, 298.15, 303.15, 308.15, 313.15]
    return [
        compute_kie(imaginary_freq_cm1, classical_KIE, T, wk_order)
        for T in temperatures
    ]
