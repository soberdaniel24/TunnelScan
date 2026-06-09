from __future__ import annotations
import numpy as np
from tunnelscan.config import K_B_KCAL, HBAR_KCAL_FS


def build_transform(n_beads: int) -> np.ndarray:
    N = n_beads
    C = np.zeros((N, N))
    for j in range(N):
        C[0, j] = 1.0 / np.sqrt(N)
    for k in range(1, N // 2 + (N % 2)):
        for j in range(N):
            C[k, j] = np.sqrt(2.0 / N) * np.cos(2.0 * np.pi * k * j / N)
    if N % 2 == 0:
        k = N // 2
        for j in range(N):
            C[k, j] = ((-1) ** j) / np.sqrt(N)
    for k in range(N // 2 + 1, N):
        for j in range(N):
            C[k, j] = np.sqrt(2.0 / N) * np.sin(2.0 * np.pi * k * j / N)
    return C


def bead_to_normal(positions: np.ndarray, C: np.ndarray) -> np.ndarray:
    return C @ positions


def normal_to_bead(modes: np.ndarray, C: np.ndarray) -> np.ndarray:
    return C.T @ modes


def normal_mode_freqs(n_beads: int, mass: float, temperature: float) -> np.ndarray:
    N = n_beads
    # omega_k = 2 * omega_N * sin(pi*k/N), omega_N = N*k_B*T/hbar
    omega_N = N * K_B_KCAL * temperature / HBAR_KCAL_FS
    freqs = np.zeros(N)
    for k in range(N):
        freqs[k] = 2.0 * omega_N * np.sin(np.pi * k / N)
    return freqs
