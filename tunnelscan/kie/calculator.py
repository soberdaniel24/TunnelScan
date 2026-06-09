from __future__ import annotations
import math
from tunnelscan.config import K_B_KCAL


def compute_kie(dg_H: float, dg_D: float, temperature: float = 300.0) -> float:
    return math.exp((dg_D - dg_H) / (K_B_KCAL * temperature))


def propagate_kie_uncertainty(dg_H: float, dg_D: float, sigma_H: float,
                              sigma_D: float, temperature: float = 300.0) -> float:
    kie = compute_kie(dg_H, dg_D, temperature)
    return kie * math.sqrt(sigma_H**2 + sigma_D**2) / (K_B_KCAL * temperature)
