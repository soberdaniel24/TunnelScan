"""
pdb_parser.py — Parse PDB files into structured objects with B-factors.

B-factors encode atomic mobility from crystallography. They are the key
additional data source that lets us distinguish promoting vibration residues
(dynamic, high B) from purely steric ones (rigid, low B).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


@dataclass
class Atom:
    name:    str
    element: str
    coords:  np.ndarray
    bfactor: float
    serial:  int

    def distance_to(self, other: 'Atom') -> float:
        return float(np.linalg.norm(self.coords - other.coords))


@dataclass
class Residue:
    name:      str
    number:    int
    chain:     str
    is_hetatm: bool = False
    atoms:     Dict[str, Atom] = field(default_factory=dict)

    BACKBONE = {'N', 'CA', 'C', 'O', 'OXT', 'H', 'HA', 'HN', '1H', '2H', '3H'}
    HBONDING = {'SER','THR','TYR','ASN','GLN','ASP','GLU',
                'HIS','LYS','ARG','CYS','MET','TRP'}

    @property
    def ca(self) -> Optional[Atom]:
        return self.atoms.get('CA')

    @property
    def ca_coords(self) -> Optional[np.ndarray]:
        return self.ca.coords if self.ca else None

    @property
    def all_heavy(self) -> List[Atom]:
        return [a for a in self.atoms.values() if a.element not in ('H','D','')]

    @property
    def sidechain_heavy(self) -> List[Atom]:
        return [a for k, a in self.atoms.items()
                if k not in self.BACKBONE and a.element not in ('H','D','')]

    @property
    def sidechain_centroid(self) -> Optional[np.ndarray]:
        sc = self.sidechain_heavy
        if not sc:
            return self.ca.coords if self.ca else None
        return np.mean([a.coords for a in sc], axis=0)

    @property
    def mean_bfactor(self) -> float:
        bvals = [a.bfactor for a in self.all_heavy if a.bfactor > 0]
        return float(np.mean(bvals)) if bvals else 0.0

    @property
    def sidechain_bfactor(self) -> float:
        bvals = [a.bfactor for a in self.sidechain_heavy if a.bfactor > 0]
        return float(np.mean(bvals)) if bvals else self.mean_bfactor

    @property
    def can_hbond(self) -> bool:
        return self.name in self.HBONDING

    @property
    def polar_atoms(self) -> List[Atom]:
        return [a for a in self.atoms.values() if a.element in ('N','O','S')]

    def __str__(self):
        return f"{self.name}{self.number}{self.chain}"

    def __repr__(self):
        return f"Residue({self})"


class Structure:
    """Parsed PDB structure with geometric and B-factor utilities."""

    def __init__(self, pdb_path: str):
        self.path = pdb_path
        self.residues: Dict[Tuple[str,int], Residue] = {}
        self._parse(pdb_path)
        self._compute_bfactor_stats()

    def _parse(self, path: str):
        with open(path) as f:
            lines = f.readlines()
        for line in lines:
            rec = line[:6].strip()
            if rec not in ('ATOM','HETATM'):
                continue
            try:
                serial  = int(line[6:11])
                aname   = line[12:16].strip()
                alt_loc = line[16].strip()
                resname = line[17:20].strip()
                chain   = line[21] if len(line) > 21 else 'A'
                resnum  = int(line[22:26])
                x       = float(line[30:38])
                y       = float(line[38:46])
                z       = float(line[46:54])
                bfactor = float(line[60:66]) if len(line)>66 and line[60:66].strip() else 0.0
                element = line[76:78].strip() if len(line)>78 else ''
                if not element:
                    element = aname.lstrip('0123456789')[0] if aname else 'C'
            except (ValueError, IndexError):
                continue
            if alt_loc and alt_loc not in ('','A'):
                continue
            key = (chain.strip(), resnum)
            if key not in self.residues:
                self.residues[key] = Residue(
                    name=resname, number=resnum,
                    chain=chain.strip(), is_hetatm=(rec=='HETATM'))
            self.residues[key].atoms[aname] = Atom(
                name=aname, element=element,
                coords=np.array([x,y,z]),
                bfactor=bfactor, serial=serial)

    def _compute_bfactor_stats(self):
        bvals = [a.bfactor for r in self.residues.values()
                 for a in r.all_heavy if a.bfactor > 0]
        self.mean_bfactor = float(np.mean(bvals)) if bvals else 20.0
        self.std_bfactor  = float(np.std(bvals))  if bvals else 5.0

    def get_residue(self, chain: str, resnum: int) -> Optional[Residue]:
        return self.residues.get((chain, resnum))

    def get_atom(self, chain: str, resnum: int, atom_name: str) -> Optional[Atom]:
        res = self.get_residue(chain, resnum)
        return res.atoms.get(atom_name) if res else None

    def protein_residues(self, chain: Optional[str]=None) -> List[Residue]:
        return [r for (c,n),r in sorted(self.residues.items())
                if not r.is_hetatm and (chain is None or c==chain)]

    def ligands(self) -> List[Residue]:
        return [r for r in self.residues.values() if r.is_hetatm]

    def normalised_bfactor(self, res: Residue) -> float:
        """Z-score of residue sidechain B-factor relative to whole structure."""
        if self.std_bfactor < 0.01:
            return 0.0
        return (res.sidechain_bfactor - self.mean_bfactor) / self.std_bfactor

    def residues_near_axis(
        self,
        donor_coords: np.ndarray,
        acceptor_coords: np.ndarray,
        radius: float = 8.0
    ) -> List[Tuple[Residue, float, str, float]]:
        """
        Residues with sidechain centroid within radius Å of the D-A line segment.
        Returns (residue, dist_to_axis, side, t_norm) sorted by dist_to_axis.
          side: 'donor' | 'acceptor' | 'flanking'
          t_norm: 0=donor end, 1=acceptor end
        """
        da_vec    = acceptor_coords - donor_coords
        da_len    = float(np.linalg.norm(da_vec))
        da_unit   = da_vec / da_len

        result = []
        for res in self.residues.values():
            if res.is_hetatm:
                continue
            c = res.sidechain_centroid
            if c is None:
                continue
            v      = c - donor_coords
            t      = float(np.dot(v, da_unit))
            proj   = donor_coords + t * da_unit
            dist   = float(np.linalg.norm(c - proj))
            if dist > radius:
                continue
            t_norm = t / da_len
            if t_norm < 0.35:
                side = 'donor'
            elif t_norm > 0.65:
                side = 'acceptor'
            else:
                side = 'flanking'
            result.append((res, dist, side, t_norm))

        return sorted(result, key=lambda x: x[1])

    def substrate_hbond_partners(
        self,
        substrate: Residue,
        cutoff: float = 3.5
    ) -> List[Residue]:
        """Protein residues that H-bond to the substrate."""
        partners = []
        for res in self.residues.values():
            if res.is_hetatm or res is substrate:
                continue
            for pa in res.polar_atoms:
                for sa in substrate.polar_atoms:
                    if pa.distance_to(sa) < cutoff:
                        partners.append(res)
                        break
                else:
                    continue
                break
        return partners

    def __repr__(self):
        chains = sorted(set(c for (c,n) in self.residues))
        return f"Structure({self.path}, {len(self.residues)} residues, chains={chains})"
