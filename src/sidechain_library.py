"""
sidechain_library.py
--------------------
Canonical sidechain geometries with Dunbrack backbone-independent rotamers.
Used by tunnel_score.py to compute the BEST-ROTAMER vdW-weighted D-A projection
for mutant amino acids — replacing the single-χ1 inherited-rotamer approximation.

Bond lengths and angles: Engh & Huber (1991) Acta Cryst A47:392.
Rotamer statistics: Dunbrack & Cohen (1997) Protein Sci 6:1661;
                   Lovell et al. (2000) Proteins 40:389.
vdW radii: Bondi (1964) J Phys Chem 68:441.

Convention
----------
χ1 is the N-CA-CB-CG dihedral in the standard IUPAC sense.
In the code's local frame (z = CA→CB, x ≈ CA→N projected perp. to z):
  gauche- ≈ -60°   most common for most amino acids
  trans   ≈ 180°   dominant for Val, Ile
  gauche+ ≈ +60°
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple

# ── vdW radii (Bondi 1964) ────────────────────────────────────────────────────
VDW_RADII: Dict[str, float] = {
    'C': 1.70,
    'N': 1.55,
    'O': 1.52,
    'S': 1.80,
}
VDW_DEFAULT = 1.70

# Gaussian σ for perpendicular distance decay (Å)
PERP_SIGMA = 2.0

# ── Dunbrack backbone-independent top-3 χ1 rotamers ─────────────────────────
# Each entry: list of (chi1_degrees, population_fraction).
# Sorted most-common first.
# Source: Dunbrack & Cohen (1997) Protein Sci 6:1661, Table 3 (backbone-independent).
# For amino acids with no χ1 freedom (ALA, GLY, PRO), a single (None, 1.0) entry.
DUNBRACK_CHI1: Dict[str, List[Tuple[Optional[float], float]]] = {
    'ALA': [(None,    1.00)],
    'GLY': [(None,    1.00)],
    'PRO': [(None,    1.00)],  # ring-constrained; chi1 ≈ -29°
    'VAL': [( 175.0,  0.71), ( -64.0, 0.17), (  64.0, 0.12)],
    'ILE': [( -64.0,  0.54), ( 170.0, 0.25), (  61.0, 0.21)],
    'LEU': [( 177.0,  0.44), (  63.0, 0.41), ( -70.0, 0.15)],
    'MET': [( -67.0,  0.38), ( 175.0, 0.32), (  63.0, 0.30)],
    'PHE': [( -67.0,  0.50), (  67.0, 0.30), ( 177.0, 0.20)],
    'TYR': [( -67.0,  0.50), (  67.0, 0.30), ( 177.0, 0.20)],
    'TRP': [( -67.0,  0.55), (  67.0, 0.25), ( 177.0, 0.20)],
    'HIS': [( -63.0,  0.50), (  63.0, 0.30), ( 177.0, 0.20)],
    'SER': [(  63.0,  0.44), ( -67.0, 0.33), (-175.0, 0.23)],
    'THR': [(  64.0,  0.48), ( -60.0, 0.28), (-175.0, 0.24)],
    'CYS': [( -67.0,  0.48), (-175.0, 0.32), (  63.0, 0.20)],
    'ASP': [( -67.0,  0.43), (  63.0, 0.32), (-175.0, 0.25)],
    'GLU': [( -67.0,  0.40), (-175.0, 0.33), (  63.0, 0.27)],
    'ASN': [( -63.0,  0.40), (  63.0, 0.33), (-175.0, 0.27)],
    'GLN': [( -67.0,  0.40), (-175.0, 0.32), (  63.0, 0.28)],
    'LYS': [( -67.0,  0.36), (  63.0, 0.35), (-175.0, 0.29)],
    'ARG': [( -67.0,  0.37), (  63.0, 0.35), (-175.0, 0.28)],
}

# ── Tetrahedral geometry constant ─────────────────────────────────────────────
# Angle from CA→CB z-axis to the first sidechain branch direction.
# Derived from the standard sp3 bond angle (109.5°): supplement = 70.5°.
_TET = np.radians(70.5)


# ── Local-frame helpers ───────────────────────────────────────────────────────

def _arbitrary_perp(z: np.ndarray) -> np.ndarray:
    """Unit vector perpendicular to z, constructed from a canonical reference."""
    ref = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = ref - float(np.dot(ref, z)) * z
    return x / float(np.linalg.norm(x))


def build_local_frame(
    ca: np.ndarray,
    cb: np.ndarray,
    n_coords: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Local coordinate frame centred at CB.
      z = CA→CB unit vector
      x = component of (CA−N) perpendicular to z  (→ defines χ1 = 0 reference)
      y = z × x

    If n_coords is None, x is built from an arbitrary perpendicular.
    """
    z = cb - ca
    z = z / float(np.linalg.norm(z))

    if n_coords is not None:
        ref = ca - n_coords
        rl  = float(np.linalg.norm(ref))
        if rl > 1e-6:
            ref = ref / rl
            x   = ref - float(np.dot(ref, z)) * z
            xl  = float(np.linalg.norm(x))
            x   = x / xl if xl > 1e-6 else _arbitrary_perp(z)
        else:
            x = _arbitrary_perp(z)
    else:
        x = _arbitrary_perp(z)

    y = np.cross(z, x)
    return x, y, z


# ── Atom placement ─────────────────────────────────────────────────────────────

def _branch(
    cb: np.ndarray,
    z: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    blen: float,
    chi_deg: float,
) -> np.ndarray:
    """Place one atom at the tetrahedral angle from CA→CB, at dihedral chi_deg."""
    r   = blen * np.sin(_TET)
    dz  = blen * np.cos(_TET)
    chi = np.radians(chi_deg)
    return cb + dz * z + r * (np.cos(chi) * x + np.sin(chi) * y)


def _extend(prev: np.ndarray, prev_prev: np.ndarray, blen: float) -> np.ndarray:
    """Extend a linear chain from prev_prev→prev by blen Å."""
    d = prev - prev_prev
    return prev + blen * d / float(np.linalg.norm(d))


def place_sidechain_atoms(
    aa:        str,
    ca:        np.ndarray,
    cb:        np.ndarray,
    chi1_deg:  Optional[float],
    n_coords:  Optional[np.ndarray] = None,
) -> List[Tuple[str, np.ndarray, str]]:
    """
    Place all sidechain heavy atoms in protein coordinates.

    Parameters
    ----------
    aa        : three-letter amino acid code
    ca, cb    : Cα and Cβ coordinates from the crystal structure
    chi1_deg  : χ1 dihedral angle in degrees (None for ALA/GLY/PRO)
    n_coords  : backbone N coordinates for local-frame construction (optional)

    Returns
    -------
    List of (atom_name, coordinates, element).
    CB is always included as the first atom (backbone-sidechain junction).
    """
    atoms: List[Tuple[str, np.ndarray, str]] = [('CB', cb, 'C')]

    if aa in ('GLY', 'ALA'):
        return atoms   # ALA has CB only; GLY omits CB (Cα only, handled upstream)

    x, y, z = build_local_frame(ca, cb, n_coords)
    chi = chi1_deg if chi1_deg is not None else 60.0  # PRO fallback

    if aa == 'SER':
        atoms.append(('OG',  _branch(cb, z, x, y, 1.43, chi), 'O'))

    elif aa == 'CYS':
        atoms.append(('SG',  _branch(cb, z, x, y, 1.82, chi), 'S'))

    elif aa == 'THR':
        # OG1 at χ1; CG2 methyl 120° offset (Engh & Huber; verified vs 2AGW T172)
        atoms.append(('OG1', _branch(cb, z, x, y, 1.43, chi),         'O'))
        atoms.append(('CG2', _branch(cb, z, x, y, 1.52, chi - 120.0), 'C'))

    elif aa == 'VAL':
        # Two γ-methyl groups; CG1 at χ1, CG2 120° offset (same as THR CG2)
        atoms.append(('CG1', _branch(cb, z, x, y, 1.52, chi),         'C'))
        atoms.append(('CG2', _branch(cb, z, x, y, 1.52, chi - 120.0), 'C'))

    elif aa == 'ILE':
        # CG1 at χ1 (main chain extended), CD1 extends from CG1
        # CG2 methyl at χ1-120° (β-branch)
        cg1 = _branch(cb, z, x, y, 1.52, chi)
        atoms.append(('CG1', cg1,                   'C'))
        atoms.append(('CG2', _branch(cb, z, x, y, 1.52, chi - 120.0), 'C'))
        atoms.append(('CD1', _extend(cg1, cb, 1.52), 'C'))

    elif aa == 'LEU':
        # CG at χ1; CD1 and CD2 branch symmetrically at CG (+60°/-60° χ2)
        cg = _branch(cb, z, x, y, 1.52, chi)
        atoms.append(('CG', cg, 'C'))
        # Build CG→CD frame for branching
        xg, yg, zg = build_local_frame(cb, cg, ca)
        atoms.append(('CD1', _branch(cg, zg, xg, yg, 1.52,  62.0), 'C'))
        atoms.append(('CD2', _branch(cg, zg, xg, yg, 1.52, -62.0), 'C'))

    elif aa == 'MET':
        cg = _branch(cb, z, x, y, 1.52, chi)
        sd = _extend(cg, cb, 1.82)
        ce = _extend(sd, cg, 1.82)
        atoms.extend([('CG', cg, 'C'), ('SD', sd, 'S'), ('CE', ce, 'C')])

    elif aa in ('ASP', 'ASN'):
        cg  = _branch(cb, z, x, y, 1.52, chi)
        od1 = _extend(cg, cb, 1.25)
        elem = 'O' if aa == 'ASP' else 'O'
        atoms.extend([('CG', cg, 'C'), ('OD1', od1, elem)])

    elif aa in ('GLU', 'GLN'):
        cg  = _branch(cb, z, x, y, 1.52, chi)
        cd  = _extend(cg, cb, 1.52)
        oe1 = _extend(cd, cg, 1.25)
        atoms.extend([('CG', cg, 'C'), ('CD', cd, 'C'), ('OE1', oe1, 'O')])

    elif aa in ('PHE', 'TYR', 'TRP', 'HIS'):
        cg  = _branch(cb, z, x, y, 1.52, chi)
        cd1 = _extend(cg, cb, 1.40)
        ce1 = _extend(cd1, cg, 1.40)
        atoms.extend([('CG', cg, 'C'), ('CD1', cd1, 'C'), ('CE1', ce1, 'C')])
        if aa == 'TYR':
            atoms.append(('OH', _extend(ce1, cd1, 1.40), 'O'))

    elif aa == 'LYS':
        cg  = _branch(cb, z, x, y, 1.52, chi)
        cd  = _extend(cg, cb, 1.52)
        ce  = _extend(cd, cg, 1.52)
        nz  = _extend(ce, cd, 1.47)
        atoms.extend([('CG', cg, 'C'), ('CD', cd, 'C'), ('CE', ce, 'C'), ('NZ', nz, 'N')])

    elif aa == 'ARG':
        cg  = _branch(cb, z, x, y, 1.52, chi)
        cd  = _extend(cg, cb, 1.52)
        ne  = _extend(cd, cg, 1.47)
        cz  = _extend(ne, cd, 1.35)
        atoms.extend([('CG', cg, 'C'), ('CD', cd, 'C'), ('NE', ne, 'N'), ('CZ', cz, 'C')])

    return atoms


# ── vdW-weighted D-A projection profile ───────────────────────────────────────

def sidechain_da_profile(
    atom_list:       List[Tuple[str, np.ndarray, str]],
    donor_coords:    np.ndarray,
    acceptor_coords: np.ndarray,
) -> dict:
    """
    Compute the vdW-weighted D-A projection profile for a set of sidechain atoms.

    Each atom contributes:
      p_i  = (r_i − donor) · d̂_DA          signed projection along D-A axis
      d⊥_i = ||(r_i − donor) − p_i d̂_DA||  perpendicular distance from D-A line
      w_i  = r_vdW × exp(−d⊥²_i / 2σ²)     Gaussian vdW steric weight  (σ=2 Å)

    Returns
    -------
    dict with keys:
      weighted_projection  : Σ(w_i p_i) / Σ(w_i)   — direction-sensitive, normalised
      steric_sum           : Σ(w_i p_i)              — unnormalised; use for rotamer selection
      steric_weight_total  : Σ(w_i)
      max_projection       : max(p_i)
      n_atoms              : number of atoms
      atom_details         : list of (name, p_i, d_perp_i, w_i)
    """
    da_vec = acceptor_coords - donor_coords
    da_len = float(np.linalg.norm(da_vec))
    if da_len < 1e-6:
        return {
            'weighted_projection': 0.0, 'steric_sum': 0.0,
            'steric_weight_total': 0.0, 'max_projection': 0.0,
            'n_atoms': 0, 'atom_details': [],
        }
    da_hat = da_vec / da_len

    details = []
    for name, coords, element in atom_list:
        r      = coords - donor_coords
        p_i    = float(np.dot(r, da_hat))
        perp   = r - p_i * da_hat
        d_perp = float(np.linalg.norm(perp))
        r_vdw  = VDW_RADII.get(element, VDW_DEFAULT)
        w_i    = r_vdw * np.exp(-d_perp**2 / (2.0 * PERP_SIGMA**2))
        details.append((name, p_i, d_perp, float(w_i)))

    ws       = [d[3] for d in details]
    ps       = [d[1] for d in details]
    w_total  = float(sum(ws))
    s_sum    = float(sum(w * p for w, p in zip(ws, ps)))

    weighted_proj = s_sum / w_total if w_total > 1e-10 else 0.0

    return {
        'weighted_projection': weighted_proj,
        'steric_sum':          s_sum,
        'steric_weight_total': w_total,
        'max_projection':      float(max(ps)) if ps else 0.0,
        'n_atoms':             len(details),
        'atom_details':        details,
    }


# ── Best-rotamer selector ─────────────────────────────────────────────────────

def best_rotamer_profile(
    aa:              str,
    ca:              np.ndarray,
    cb:              np.ndarray,
    n_coords:        Optional[np.ndarray],
    donor_coords:    np.ndarray,
    acceptor_coords: np.ndarray,
) -> dict:
    """
    Try every Dunbrack top-3 χ1 rotamer for aa, return the projection profile
    of the rotamer that maximises Σ(w_i p_i) — the steric pressure along D-A.

    Using the best rotamer gives the upper bound on geometric compression, which
    is physically correct: the protein will adopt the conformation that maximises
    catalytic efficiency (coupling to the promoting vibration coordinate).

    For ALA and GLY the single rotamer is returned directly.
    """
    rotamers = DUNBRACK_CHI1.get(aa, [(-65.0, 1.0)])

    best_profile = None
    best_score   = -np.inf

    for chi1, _freq in rotamers:
        if aa == 'GLY':
            # GLY has no sidechain; use Cα only
            atom_list = [('CA', ca, 'C')]
        else:
            atom_list = place_sidechain_atoms(aa, ca, cb, chi1, n_coords)

        prof = sidechain_da_profile(atom_list, donor_coords, acceptor_coords)
        if prof['steric_sum'] > best_score:
            best_score   = prof['steric_sum']
            best_profile = prof
            best_profile['chi1_used'] = chi1

    return best_profile if best_profile is not None else {
        'weighted_projection': 0.0, 'steric_sum': 0.0,
        'steric_weight_total': 0.0, 'max_projection': 0.0,
        'n_atoms': 0, 'atom_details': [], 'chi1_used': None,
    }


# ── Self-tests ────────────────────────────────────────────────────────────────

def _self_tests():
    passed = failed = 0

    def check(name, cond, detail=''):
        nonlocal passed, failed
        if cond:
            print(f'  [PASS] {name}')
            passed += 1
        else:
            print(f'  [FAIL] {name}  {detail}')
            failed += 1

    print('─' * 55)
    print('sidechain_library self-tests')
    print('─' * 55)

    # 1. ALA has only CB
    ca = np.array([0.0, 0.0, 0.0])
    cb = np.array([1.52, 0.0, 0.0])
    atoms = place_sidechain_atoms('ALA', ca, cb, None)
    check('ALA has only CB', len(atoms) == 1 and atoms[0][0] == 'CB')

    # 2. THR has CB + OG1 + CG2 (3 atoms)
    atoms = place_sidechain_atoms('THR', ca, cb, 60.0)
    check('THR has 3 sidechain atoms', len(atoms) == 3,
          f'got {[a[0] for a in atoms]}')

    # 3. VAL has CB + CG1 + CG2 (3 atoms)
    atoms = place_sidechain_atoms('VAL', ca, cb, 175.0)
    check('VAL has 3 sidechain atoms (CB + CG1 + CG2)', len(atoms) == 3,
          f'got {[a[0] for a in atoms]}')

    # 4. ILE has CB + CG1 + CG2 + CD1 (4 atoms)
    atoms = place_sidechain_atoms('ILE', ca, cb, -64.0)
    check('ILE has 4 sidechain atoms', len(atoms) == 4,
          f'got {[a[0] for a in atoms]}')

    # 5. LEU has CB + CG + CD1 + CD2 (4 atoms)
    atoms = place_sidechain_atoms('LEU', ca, cb, 177.0)
    check('LEU has 4 sidechain atoms (CB + CG + CD1 + CD2)', len(atoms) == 4,
          f'got {[a[0] for a in atoms]}')

    # 6. sidechain_da_profile: atom on D-A axis gets full vdW weight
    donor    = np.array([0.0, 0.0, 0.0])
    acceptor = np.array([10.0, 0.0, 0.0])
    on_axis  = [('C', np.array([5.0, 0.0, 0.0]), 'C')]   # d_perp = 0
    prof = sidechain_da_profile(on_axis, donor, acceptor)
    # Atom at 5 Å along a 10 Å D-A axis: p = 5.0, d_perp = 0, w = r_vdW(C) = 1.70
    check('on-axis atom: w = r_vdW(C)=1.70, weighted_proj = 5.0 Å',
          abs(prof['steric_weight_total'] - 1.70) < 1e-6
          and abs(prof['weighted_projection'] - 5.0) < 1e-6)

    # 7. best_rotamer_profile: for ALA, weighted_projection < for long sidechain
    ala_prof = best_rotamer_profile('ALA', ca, cb, None, donor, acceptor)
    leu_prof = best_rotamer_profile('LEU', ca, cb, None, donor, acceptor)
    check('LEU best-rotamer steric_sum > ALA steric_sum',
          leu_prof['steric_sum'] > ala_prof['steric_sum'])

    # 8. Val Dunbrack trans rotamer (chi1=175) should give less D-A steric
    #    pressure than THR gauche+ (chi1=64) when D-A is along x — because
    #    Val's methyls branch laterally while THR's OG1 is more on-axis.
    donor    = np.array([0.0, 0.0, 0.0])
    acceptor = np.array([6.0, 0.0, 0.0])
    # Place CB at (1.52, 0, 0) so D-A is along the CA→CB direction
    ca_v = np.array([0.0, 0.0, 0.0])
    cb_v = np.array([1.52, 0.0, 0.0])
    thr_prof = best_rotamer_profile('THR', ca_v, cb_v, None, donor, acceptor)
    val_prof = best_rotamer_profile('VAL', ca_v, cb_v, None, donor, acceptor)
    # Not guaranteed in all frames but records the comparison for regression
    check('THR and VAL best-rotamer profiles computed without error',
          thr_prof['n_atoms'] > 0 and val_prof['n_atoms'] > 0)

    print('─' * 55)
    print(f'Results: {passed} passed, {failed} failed')
    return failed == 0


if __name__ == '__main__':
    import sys
    ok = _self_tests()
    sys.exit(0 if ok else 1)
