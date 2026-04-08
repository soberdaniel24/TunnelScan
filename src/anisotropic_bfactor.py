"""
anisotropic_bfactor.py
----------------------
Reads crystallographic anisotropic displacement parameters (ADPs) from
ANISOU records in PDB files and computes how well each residue's preferred
direction of motion aligns with the donor-acceptor compression axis.

Why this matters:
  Standard B-factors are isotropic — they tell you HOW MUCH an atom moves
  but not in which direction. ANISOU records give the full 3x3 displacement
  tensor (U matrix), which encodes the preferred direction of motion.

  A residue that moves preferentially ALONG the D-A axis is a genuine
  promoting vibration residue — its thermal motion directly compresses the
  tunnelling distance. A residue that moves perpendicular to the D-A axis
  contributes random noise, not directed tunnelling enhancement.

  This is what distinguishes T172 from N156 at the crystallographic level:
  T172's thermal motion should be preferentially aligned with D-A compression,
  N156's should not.

The ANISOU record format (PDB):
  ANISOU serial name resName chain resSeq  U11 U22 U33 U12 U13 U23
  Values are in units of 10^-4 Å²

  The U tensor maps to a 3x3 matrix:
    [[U11, U12, U13],
     [U12, U22, U23],
     [U13, U23, U33]]

  The principal axes of motion are the eigenvectors of U.
  The largest eigenvalue gives the direction of maximum displacement.

Reference:
  Trueblood et al. (1996) Acta Cryst. A52:770 — ADP conventions
  Painter & Merritt (2006) Acta Cryst. D62:439 — ANISOU interpretation
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


@dataclass
class AnisotropicData:
    """Anisotropic displacement parameters for one atom."""
    u11: float
    u22: float
    u33: float
    u12: float
    u13: float
    u23: float

    @property
    def tensor(self) -> np.ndarray:
        """Full 3x3 U tensor."""
        return np.array([
            [self.u11, self.u12, self.u13],
            [self.u12, self.u22, self.u23],
            [self.u13, self.u23, self.u33]
        ]) * 1e-4  # convert from 10^-4 Å² to Å²

    @property
    def principal_axis(self) -> np.ndarray:
        """Direction of maximum displacement (eigenvector of largest eigenvalue)."""
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(self.tensor)
            # eigh returns in ascending order — last is largest
            return eigenvectors[:, -1]
        except np.linalg.LinAlgError:
            return np.array([1.0, 0.0, 0.0])

    @property
    def anisotropy_ratio(self) -> float:
        """
        Ratio of largest to smallest eigenvalue.
        1.0 = perfectly isotropic
        >> 1.0 = strongly anisotropic (has a preferred direction)
        """
        try:
            eigenvalues = np.linalg.eigvalsh(self.tensor)
            eigenvalues = np.abs(eigenvalues)
            if eigenvalues.min() < 1e-10:
                return 1.0
            return float(eigenvalues.max() / eigenvalues.min())
        except np.linalg.LinAlgError:
            return 1.0

    @property
    def equivalent_bfactor(self) -> float:
        """Isotropic equivalent B-factor: B = 8π²/3 × trace(U)."""
        return float(8 * np.pi**2 / 3 * (self.u11 + self.u22 + self.u33) * 1e-4)


def parse_anisou_records(pdb_path: str) -> Dict[Tuple[str, int, str], AnisotropicData]:
    """
    Parse all ANISOU records from a PDB file.

    Returns dict keyed by (chain, resnum, atom_name) → AnisotropicData.
    """
    anisou = {}

    with open(pdb_path) as f:
        for line in f:
            if not line.startswith('ANISOU'):
                continue
            try:
                atom_name = line[12:16].strip()
                chain     = line[21].strip()
                resnum    = int(line[22:26])
                u11       = int(line[28:35])
                u22       = int(line[35:42])
                u33       = int(line[42:49])
                u12       = int(line[49:56])
                u13       = int(line[56:63])
                u23       = int(line[63:70])

                key = (chain, resnum, atom_name)
                anisou[key] = AnisotropicData(u11, u22, u33, u12, u13, u23)
            except (ValueError, IndexError):
                continue

    return anisou


def get_residue_principal_axis(
    anisou_data: Dict[Tuple[str, int, str], AnisotropicData],
    chain: str,
    resnum: int,
    preferred_atoms: list = ('CA', 'CB', 'CG')
) -> Optional[np.ndarray]:
    """
    Get the principal axis of motion for a residue.
    Uses Cα by default, falls back to CB, CG.
    Returns unit vector in direction of maximum displacement, or None.
    """
    for atom in preferred_atoms:
        key = (chain, resnum, atom)
        if key in anisou_data:
            axis = anisou_data[key].principal_axis
            norm = np.linalg.norm(axis)
            if norm > 0:
                return axis / norm
    return None


def da_alignment_score(
    anisou_data: Dict[Tuple[str, int, str], AnisotropicData],
    chain: str,
    resnum: int,
    donor_coords: np.ndarray,
    acceptor_coords: np.ndarray
) -> float:
    """
    Compute how well a residue's preferred motion direction aligns
    with the donor-acceptor compression axis.

    Returns a score in [0, 1]:
      1.0 = residue moves perfectly along D-A axis (promoting vibration)
      0.0 = residue moves perpendicular to D-A axis (no tunnelling contribution)
      0.5 = isotropic or at 45° to D-A axis

    This is the crystallographic evidence for promoting vibration participation —
    no other enzyme engineering tool uses this information.

    Parameters
    ----------
    anisou_data : dict
        Parsed ANISOU records from parse_anisou_records().
    chain, resnum : str, int
        Residue identity.
    donor_coords, acceptor_coords : np.ndarray
        3D coordinates of donor and acceptor atoms.

    Returns
    -------
    float : D-A alignment score [0, 1]
    """
    # Get principal axis of motion for this residue
    principal = get_residue_principal_axis(anisou_data, chain, resnum)

    if principal is None:
        return 0.5  # no anisotropic data — assume neutral

    # D-A compression axis (unit vector from donor toward acceptor)
    da_vec = acceptor_coords - donor_coords
    da_len = np.linalg.norm(da_vec)
    if da_len < 0.01:
        return 0.5
    da_unit = da_vec / da_len

    # Alignment = |cos θ| between principal axis and D-A axis
    # |cos θ| because motion in either direction along the axis is useful
    alignment = float(abs(np.dot(principal, da_unit)))

    # Weight by anisotropy: a strongly anisotropic residue with
    # good alignment is much more significant than a near-isotropic one
    # Get anisotropy ratio from Cα
    anisotropy = 1.0
    for atom in ('CA', 'CB'):
        key = (chain, resnum, atom)
        if key in anisou_data:
            anisotropy = anisou_data[key].anisotropy_ratio
            break

    # Scale alignment by anisotropy
    # Perfectly isotropic (ratio=1): alignment weighted 50% (no directional info)
    # Strongly anisotropic (ratio=5+): alignment weighted fully
    aniso_weight = float(np.clip((anisotropy - 1.0) / 4.0, 0.0, 1.0))
    weighted_alignment = 0.5 + (alignment - 0.5) * aniso_weight

    return float(np.clip(weighted_alignment, 0.0, 1.0))


def build_alignment_map(
    pdb_path: str,
    donor_coords: np.ndarray,
    acceptor_coords: np.ndarray,
    chains: list = None
) -> Dict[Tuple[str, int], float]:
    """
    Build a complete map of D-A alignment scores for all residues in the structure.

    Parameters
    ----------
    pdb_path : str
        Path to PDB file with ANISOU records.
    donor_coords, acceptor_coords : np.ndarray
        Active site geometry.
    chains : list, optional
        If given, only compute for these chains.

    Returns
    -------
    dict : (chain, resnum) → alignment_score [0, 1]
    """
    anisou_data = parse_anisou_records(pdb_path)

    if not anisou_data:
        return {}

    # Get all unique (chain, resnum) combinations
    residue_keys = set((c, r) for (c, r, a) in anisou_data.keys())
    if chains:
        residue_keys = {(c, r) for c, r in residue_keys if c in chains}

    alignment_map = {}
    for chain, resnum in residue_keys:
        score = da_alignment_score(
            anisou_data, chain, resnum,
            donor_coords, acceptor_coords
        )
        alignment_map[(chain, resnum)] = score

    return alignment_map


def normalised_alignment_map(
    alignment_map: Dict[Tuple[str, int], float]
) -> Dict[Tuple[str, int], float]:
    """
    Rank-normalise alignment scores to [0, 1] so that the most aligned
    residue scores 1.0 regardless of the absolute distribution.
    Same approach as ENM rank normalisation.
    """
    if not alignment_map:
        return {}

    from scipy.stats import rankdata
    keys   = list(alignment_map.keys())
    values = np.array([alignment_map[k] for k in keys])
    ranks  = rankdata(values)
    normed = (ranks - 1) / max(len(ranks) - 1, 1)

    return {k: float(v) for k, v in zip(keys, normed)}
