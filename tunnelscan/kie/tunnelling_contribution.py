from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional
from tunnelscan.config import (K_B_KCAL, TUNNEL_THRESHOLD_WEAK,
                                TUNNEL_THRESHOLD_MODERATE, TUNNEL_THRESHOLD_DEEP,
                                DEUTERIUM_CHECK_TOLERANCE)
from tunnelscan.kie.calculator import compute_kie, propagate_kie_uncertainty
from tunnelscan.kie.deuterium import check_deuterium_consistency


@dataclass
class TunnellingResult:
    delta_delta_G: float
    tunnelling_factor: float
    kie_theoretical: float
    kie_experimental: Optional[float]
    kie_agreement: Optional[bool]
    zpe: float
    classification: str
    confidence_interval_95: tuple
    trajectory_variance: float
    deuterium_check_passed: bool
    flags: list[str]


def assess_tunnelling(fe_classical, fe_H, fe_D, bootstrap_H: dict,
                      bootstrap_D: dict, kie_experimental: Optional[float] = None,
                      temperature: float = 300.0) -> TunnellingResult:
    delta_delta_G = fe_classical.barrier_height - fe_H.barrier_height
    tunnelling_factor = math.exp(delta_delta_G / (K_B_KCAL * temperature))

    dg_H = fe_H.barrier_height
    dg_D = fe_D.barrier_height
    kie_theor = compute_kie(dg_H, dg_D, temperature)

    sigma_H = bootstrap_H.get("std", 0.0) or 0.0
    sigma_D = bootstrap_D.get("std", 0.0) or 0.0
    kie_unc = propagate_kie_uncertainty(dg_H, dg_D, sigma_H, sigma_D, temperature)

    ci_lo = bootstrap_H.get("ci_95", (float("nan"), float("nan")))[0]
    ci_hi = bootstrap_H.get("ci_95", (float("nan"), float("nan")))[1]
    ci_95 = (ci_lo, ci_hi)

    trajectory_variance = sigma_H

    deuterium_ok = check_deuterium_consistency(fe_D, fe_classical)

    # Classification
    if abs(delta_delta_G) < TUNNEL_THRESHOLD_WEAK:
        classification = "negligible"
    elif abs(delta_delta_G) < TUNNEL_THRESHOLD_MODERATE:
        classification = "weak"
    elif abs(delta_delta_G) < TUNNEL_THRESHOLD_DEEP:
        classification = "moderate"
    else:
        classification = "deep"

    # KIE agreement
    kie_agreement = None
    if kie_experimental is not None:
        kie_agreement = abs(kie_theor / kie_experimental - 1.0) < 1.0

    # Flags
    flags = []
    if not deuterium_ok:
        flags.append("deuterium_check_failed")
    if classification == "deep":
        flags.append("deep_tunnelling_regime")
    total_paths = (bootstrap_H.get("ci_95", (0, 0)) and
                   len(bootstrap_H.get("per_trajectory_barriers", [])))
    if total_paths is not None and isinstance(total_paths, int) and total_paths < 20:
        flags.append("insufficient_paths_for_statistics")
    if math.isnan(dg_H) or math.isnan(dg_D):
        flags.append("nan_barrier_detected")
    if kie_agreement is False:
        flags.append("kie_disagreement_with_experiment")

    return TunnellingResult(
        delta_delta_G=delta_delta_G,
        tunnelling_factor=tunnelling_factor,
        kie_theoretical=kie_theor,
        kie_experimental=kie_experimental,
        kie_agreement=kie_agreement,
        zpe=fe_H.zpe,
        classification=classification,
        confidence_interval_95=ci_95,
        trajectory_variance=trajectory_variance,
        deuterium_check_passed=deuterium_ok,
        flags=flags,
    )
