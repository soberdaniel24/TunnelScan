"""
temperature_dependence.py
-------------------------
Predicts temperature-dependent KIE behaviour from 298K predictions.

Physical basis:
  For a tunnelling-dominated reaction, KIE varies with temperature as:

    KIE(T) = (AH/AD) × exp(ΔEa / RT)

  where:
    AH/AD  = isotopic Arrhenius pre-exponential factor ratio
    ΔEa    = EaD - EaH = isotopic activation energy difference

  In deep tunnelling (WT AADH, KIE~55):
    AH/AD >> 1 (experimentally 7-10 for AADH WT)
    Weak temperature dependence — KIE barely changes with T

  In reduced tunnelling (T172A, KIE~7.4):
    AH/AD < 1 (below semiclassical limit of 0.7)
    Strong temperature dependence — KIE rises with decreasing T

Theoretical connection:
  The isotopic activation energy difference ΔEa scales with the tunnelling
  probability, which scales with KIE_298. From Klinman & Kohen (2013):

    ΔEa ≈ ΔEa_wt × (KIE_wt / KIE_mut)^κ

  where κ ≈ 0.3 (empirical scaling from the DHFR, AADH, ADH datasets).
  ΔEa_wt for AADH is ~1.5 kcal/mol (Johannissen et al. 2007).

  This gives AH/AD directly:
    ln(AH/AD) = ln(KIE_298) - ΔEa / RT

Predictions:
  For each mutant, we output:
    - ΔEa: isotopic activation energy difference (kcal/mol)
    - AH/AD: pre-exponential ratio
    - KIE at any temperature T
    - T-dependence classification

  Classification (from Klinman & Kohen 2013):
    AH/AD > 3.3: tunnelling-dominated, weak T-dependence (like WT AADH)
    0.7 < AH/AD < 3.3: semiclassical regime
    AH/AD < 0.7: tunnelling with environmental coupling, strong T-dependence

References:
  Klinman & Kohen (2013) Annual Review of Biochemistry 82:471
  Johannissen et al. (2007) FEBS J 278:1701
  Scrutton et al. (2012) Nature Chemistry 4:161
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional


# ── Constants ─────────────────────────────────────────────────────────────────

R_KCAL       = 1.987e-3   # kcal/mol/K
T_STANDARD   = 298.0      # K (25°C)

# AADH wild-type calibration parameters
# From Johannissen et al. (2007) FEBS J 278:1701
DELTA_EA_WT  = 1.5        # kcal/mol, isotopic activation energy difference
KIE_WT_EXP   = 55.0       # experimental WT KIE at 298K
AHAD_WT_EXP  = 8.0        # experimental AH/AD for WT AADH (range 7-10)

# Klinman scaling exponent
# Empirical from DHFR, AADH, ADH datasets: κ ≈ 0.3
# Source: Klinman & Kohen (2013) Annual Review Biochemistry
KLINMAN_KAPPA = 0.3


# ── Core calculations ─────────────────────────────────────────────────────────

def predict_delta_ea(
    kie_mut: float,
    kie_wt: float = KIE_WT_EXP,
    delta_ea_wt: float = DELTA_EA_WT,
    kappa: float = KLINMAN_KAPPA
) -> float:
    """
    Predict the isotopic activation energy difference (EaD - EaH) for a mutant.

    Klinman scaling: ΔEa_mut = ΔEa_wt × (KIE_wt / KIE_mut)^κ
    Larger ΔEa = stronger temperature dependence = less tunnelling.

    Parameters
    ----------
    kie_mut : float
        Predicted or experimental KIE of the mutant at 298K.
    kie_wt : float
        Wild-type KIE at 298K (default: 55.0 for AADH).
    delta_ea_wt : float
        ΔEa for wild-type (kcal/mol, default: 1.5 for AADH).
    kappa : float
        Klinman scaling exponent (default: 0.3).

    Returns
    -------
    float : ΔEa in kcal/mol.
    """
    if kie_mut <= 0:
        return delta_ea_wt * 2.0
    return float(delta_ea_wt * (kie_wt / kie_mut) ** kappa)


def predict_ahad(
    kie_298: float,
    delta_ea: float,
    T: float = T_STANDARD
) -> float:
    """
    Predict the isotopic Arrhenius pre-exponential factor ratio AH/AD.

    From: ln(KIE) = ln(AH/AD) + ΔEa/RT
    → ln(AH/AD) = ln(KIE) - ΔEa/RT

    Parameters
    ----------
    kie_298 : float
        KIE at 298K.
    delta_ea : float
        ΔEa in kcal/mol.
    T : float
        Temperature in Kelvin.

    Returns
    -------
    float : AH/AD ratio.
    """
    if kie_298 <= 0:
        return 0.0
    ln_ahad = np.log(kie_298) - delta_ea / (R_KCAL * T)
    return float(np.exp(ln_ahad))


def kie_at_temperature(
    kie_298: float,
    delta_ea: float,
    T_new: float,
    T_ref: float = T_STANDARD
) -> float:
    """
    Predict KIE at a new temperature given KIE at reference temperature.

    KIE(T_new) = KIE(T_ref) × exp(ΔEa/R × (1/T_ref - 1/T_new))

    Parameters
    ----------
    kie_298 : float
        KIE at reference temperature (default 298K).
    delta_ea : float
        ΔEa in kcal/mol.
    T_new : float
        New temperature in Kelvin.
    T_ref : float
        Reference temperature in Kelvin.

    Returns
    -------
    float : Predicted KIE at T_new.
    """
    if kie_298 <= 0:
        return 0.0
    ln_scaling = (delta_ea / R_KCAL) * (1.0/T_ref - 1.0/T_new)
    return float(kie_298 * np.exp(ln_scaling))


def classify_tunnelling_regime(ahad: float) -> str:
    """
    Classify the tunnelling regime based on AH/AD.
    From Klinman & Kohen (2013) Annual Review Biochemistry.
    """
    if ahad > 3.3:
        return 'deep tunnelling'        # like WT AADH
    elif ahad > 0.7:
        return 'semiclassical'          # conventional KIE
    elif ahad > 0.1:
        return 'tunnelling+sampling'    # promoted tunnelling
    else:
        return 'strongly coupled'       # extreme tunnelling with dynamics


# ── Prediction dataclass ──────────────────────────────────────────────────────

@dataclass
class TemperaturePrediction:
    """Temperature-dependent KIE prediction for one mutation."""
    label:         str
    kie_298:       float   # KIE at 298K (input)
    delta_ea:      float   # ΔEa in kcal/mol
    ahad:          float   # AH/AD pre-exponential ratio
    tunnelling_regime: str

    # KIE at key temperatures
    kie_278:       float   # 5°C (cold)
    kie_298:       float   # 25°C (standard)
    kie_318:       float   # 45°C (warm)

    # T-dependence metric: fold change over 40°C range
    t_dependence:  float   # KIE_278 / KIE_318 (>1 = T-dependent)

    def summary(self) -> str:
        return (
            f"{self.label:<12} "
            f"KIE@298={self.kie_298:>6.1f}  "
            f"AH/AD={self.ahad:>5.2f}  "
            f"ΔEa={self.delta_ea:>5.2f}  "
            f"T-dep={self.t_dependence:>4.2f}x  "
            f"{self.tunnelling_regime}"
        )


def predict_temperature_dependence(
    label: str,
    kie_298: float,
    kie_wt: float = KIE_WT_EXP,
    temperatures: Optional[List[float]] = None
) -> TemperaturePrediction:
    """
    Full temperature dependence prediction for one mutation.

    Parameters
    ----------
    label : str
        Mutation label (e.g. 'T172A').
    kie_298 : float
        KIE at 298K (from TunnelScan single-mutation prediction).
    kie_wt : float
        Wild-type KIE (default 55.0).
    temperatures : list, optional
        Additional temperatures to compute (Kelvin).

    Returns
    -------
    TemperaturePrediction
    """
    delta_ea  = predict_delta_ea(kie_298, kie_wt)
    ahad      = predict_ahad(kie_298, delta_ea)
    regime    = classify_tunnelling_regime(ahad)

    kie_278 = kie_at_temperature(kie_298, delta_ea, 278.0)
    kie_318 = kie_at_temperature(kie_298, delta_ea, 318.0)
    t_dep   = kie_278 / max(kie_318, 0.01)

    return TemperaturePrediction(
        label=label,
        kie_298=kie_298,
        delta_ea=delta_ea,
        ahad=ahad,
        tunnelling_regime=regime,
        kie_278=kie_278,
        kie_318=kie_318,
        t_dependence=t_dep
    )


def print_temperature_report(
    predictions: List[TemperaturePrediction],
    top_n: int = 20
):
    """Print a formatted temperature dependence report."""
    print(f"\n{'='*70}")
    print(f"  TEMPERATURE DEPENDENCE PREDICTIONS")
    print(f"{'='*70}")
    print(f"  Based on Klinman-Arrhenius framework")
    print(f"  WT AADH calibration: ΔEa={DELTA_EA_WT} kcal/mol, AH/AD~{AHAD_WT_EXP}")
    print()
    print(f"  {'Mutation':<12} {'KIE@298':>8} {'KIE@278':>8} {'KIE@318':>8} "
          f"{'AH/AD':>7} {'ΔEa':>6} {'T-dep':>6} {'Regime'}")
    print(f"  {'-'*80}")

    for p in predictions[:top_n]:
        print(f"  {p.label:<12} {p.kie_298:>8.1f} {p.kie_278:>8.1f} "
              f"{p.kie_318:>8.1f} {p.ahad:>7.2f} "
              f"{p.delta_ea:>6.2f} {p.t_dependence:>6.2f}x  "
              f"{p.tunnelling_regime}")

    # Highlight deep tunnelling predictions
    deep = [p for p in predictions if p.tunnelling_regime == 'deep tunnelling']
    if deep:
        print(f"\n  ★ {len(deep)} mutations predicted in deep tunnelling regime (AH/AD > 3.3):")
        print(f"    These show WEAK temperature dependence — strong tunnelling signal")
        for p in deep[:5]:
            print(f"    {p.label:<12} AH/AD={p.ahad:.1f}  KIE@278={p.kie_278:.0f}  KIE@318={p.kie_318:.0f}")

    print(f"{'='*70}\n")
