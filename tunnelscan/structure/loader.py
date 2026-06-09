from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Topology:
    residues: list[dict]
    hetatm_indices: set
    water_indices: set


def load_pdb(path: str, active_site_center: Optional[np.ndarray] = None):
    import ase.io
    atoms = ase.io.read(path, format="proteindatabank")

    residues = []
    hetatm_indices = set()
    water_res_indices = {}  # reskey -> list of atom indices

    res_map = {}  # (chain, resnum, resname) -> {'name':..., 'atom_indices':list}

    with open(path) as fh:
        atom_serial_to_idx = {}
        ase_idx = 0
        for line in fh:
            rec = line[:6].strip()
            if rec not in ("ATOM", "HETATM"):
                continue
            serial = int(line[6:11])
            name = line[12:16].strip()
            resname = line[17:20].strip()
            chain = line[21].strip()
            resnum = int(line[22:26])
            key = (chain, resnum, resname)

            if key not in res_map:
                res_map[key] = {"name": resname, "chain": chain, "resnum": resnum, "atom_indices": []}
            res_map[key]["atom_indices"].append(ase_idx)

            if rec == "HETATM":
                hetatm_indices.add(ase_idx)

            if resname in ("HOH", "WAT", "TIP", "TIP3", "SOL"):
                if key not in water_res_indices:
                    water_res_indices[key] = []
                water_res_indices[key].append(ase_idx)

            atom_serial_to_idx[serial] = ase_idx
            ase_idx += 1

    residues = list(res_map.values())
    positions = atoms.get_positions()

    # Find water O indices (element O) within 5Å of active site centroid
    water_indices = set()
    if active_site_center is not None:
        center = np.asarray(active_site_center)
        for key, idxs in water_res_indices.items():
            for idx in idxs:
                sym = atoms[idx].symbol
                if sym == "O":
                    dist = np.linalg.norm(positions[idx] - center)
                    if dist <= 5.0:
                        water_indices.add(idx)
    else:
        # Include all water O atoms
        for key, idxs in water_res_indices.items():
            for idx in idxs:
                if atoms[idx].symbol == "O":
                    water_indices.add(idx)

    # Remove water hetatm indices from hetatm_indices
    water_all = set()
    for idxs in water_res_indices.values():
        water_all.update(idxs)
    hetatm_indices -= water_all

    topology = Topology(residues=residues, hetatm_indices=hetatm_indices, water_indices=water_indices)
    return atoms, topology
