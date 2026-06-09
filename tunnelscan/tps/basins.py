from __future__ import annotations
import numpy as np


def _norm(v):
    return float(np.linalg.norm(v))


def is_reactant(coords: np.ndarray, donor_idx: int, h_idx: int,
                acceptor_idx: int) -> bool:
    dDH = _norm(coords[donor_idx] - coords[h_idx])
    dHA = _norm(coords[h_idx] - coords[acceptor_idx])
    return dDH < 1.3 and dHA > 2.0 and (dDH - dHA) < -0.7


def is_product(coords: np.ndarray, donor_idx: int, h_idx: int,
               acceptor_idx: int) -> bool:
    dDH = _norm(coords[donor_idx] - coords[h_idx])
    dHA = _norm(coords[h_idx] - coords[acceptor_idx])
    return dHA < 1.3 and dDH > 2.0 and (dDH - dHA) > 0.7


def xi(coords: np.ndarray, donor_idx: int, h_idx: int,
       acceptor_idx: int) -> float:
    return _norm(coords[donor_idx] - coords[h_idx]) - _norm(coords[h_idx] - coords[acceptor_idx])
