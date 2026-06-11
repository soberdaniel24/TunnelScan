from __future__ import annotations
import logging
import numpy as np

log = logging.getLogger(__name__)

try:
    from tblite.ase import TBLite
    _TBLITE_AVAILABLE = True
except ImportError:
    _TBLITE_AVAILABLE = False

try:
    import openmm
    import openmm.app as app
    import openmm.unit as unit
    _OPENMM_AVAILABLE = True
except ImportError:
    _OPENMM_AVAILABLE = False


class QMMMEngine:
    def __init__(self, atoms, qm_region, temperature: float = 300.0, use_gpu: bool = False,
                 qm_charge: int = 0):
        self.atoms = atoms.copy()
        self.qm_region = qm_region
        self.temperature = temperature
        self.use_gpu = use_gpu
        self.qm_charge = qm_charge  # formal charge of QM region (e.g. -1 for Asp carboxylate)
        self._setup_qm()
        self._setup_mm()

    def _setup_qm(self):
        if _TBLITE_AVAILABLE:
            self._qm_method = "gfn2"
        else:
            log.warning("tblite not available — using ASE EMT as QM fallback (testing only)")
            self._qm_method = "emt"

    def _setup_mm(self):
        if _OPENMM_AVAILABLE:
            try:
                self._setup_openmm()
                self._mm_method = "openmm"
                return
            except Exception as e:
                log.warning(f"OpenMM setup failed ({e}), falling back to LennardJones")
        self._mm_method = "lj"

    def _setup_openmm(self):
        import openmm.app as app
        import openmm.unit as unit
        ff = app.ForceField("amber14-all.xml", "amber14/tip3p.xml")
        modeller = app.Modeller(
            app.Topology(),
            []
        )
        self._openmm_ff = ff

    def _qm_energy_forces(self, positions: np.ndarray, indices: list[int],
                           link_pos: np.ndarray | None = None):
        import ase
        sub = self.atoms[indices].copy()
        sub.set_positions(positions[indices])
        if link_pos is not None and len(link_pos) > 0:
            from ase import Atoms
            link_atoms = Atoms(["H"] * len(link_pos), positions=link_pos)
            from ase import Atoms as AseAtoms
            combined_pos = np.vstack([sub.get_positions(), link_pos])
            combined_syms = list(sub.get_chemical_symbols()) + ["H"] * len(link_pos)
            import ase
            sub2 = ase.Atoms(symbols=combined_syms, positions=combined_pos)
        else:
            sub2 = sub

        if self._qm_method == "gfn2":
            calc = TBLite(method="GFN2-xTB", charge=self.qm_charge)
        else:
            from ase.calculators.emt import EMT
            calc = EMT()

        sub2.calc = calc
        try:
            e = sub2.get_potential_energy()
            f = sub2.get_forces()
        except Exception:
            e = 0.0
            f = np.zeros((len(sub2), 3))

        # Convert from eV to kcal/mol
        EV_TO_KCAL = 23.0605
        e_kcal = e * EV_TO_KCAL
        f_kcal = f[:len(indices)] * EV_TO_KCAL  # only QM atom forces, not link
        return e_kcal, f_kcal

    def _mm_energy_forces_lj(self, positions: np.ndarray, indices: list[int] | None = None):
        """Simple Lennard-Jones fallback for MM."""
        if indices is None:
            indices = list(range(len(positions)))
        pos = positions[indices]
        n = len(pos)
        e = 0.0
        f = np.zeros((n, 3))
        eps = 0.1  # kcal/mol
        sigma = 3.4  # Angstrom
        for i in range(n):
            for j in range(i + 1, n):
                r_vec = pos[i] - pos[j]
                r = np.linalg.norm(r_vec)
                if r < 0.1:
                    continue
                sr6 = (sigma / r) ** 6
                sr12 = sr6 * sr6
                e += 4 * eps * (sr12 - sr6)
                f_mag = 4 * eps * (12 * sr12 - 6 * sr6) / r
                f_dir = f_mag * r_vec / r
                f[i] += f_dir
                f[j] -= f_dir
        full_f = np.zeros((len(positions), 3))
        for k, idx in enumerate(indices):
            full_f[idx] = f[k]
        return e, full_f

    def energy_and_forces(self) -> tuple[float, np.ndarray]:
        positions = self.atoms.get_positions()
        n = len(positions)

        # QM region with link atoms
        from tunnelscan.structure.link_atoms import place_link_atoms
        link_pos = place_link_atoms(self.atoms, self.qm_region.boundary_pairs)

        e_qm, f_qm_partial = self._qm_energy_forces(
            positions, self.qm_region.qm_indices, link_pos
        )

        # MM for all atoms
        e_mm_all, f_mm_all = self._mm_energy_forces_lj(positions)

        # MM for QM atoms only (subtraction scheme)
        if self.qm_region.qm_indices:
            e_mm_qm, f_mm_qm_partial = self._mm_energy_forces_lj(
                positions, self.qm_region.qm_indices
            )
        else:
            e_mm_qm = 0.0
            f_mm_qm_partial = np.zeros((n, 3))

        e_total = e_qm + e_mm_all - e_mm_qm

        # Assemble forces
        forces = f_mm_all - f_mm_qm_partial
        for k, idx in enumerate(self.qm_region.qm_indices):
            if k < len(f_qm_partial):
                forces[idx] += f_qm_partial[k]

        # Distribute link atom forces
        if len(self.qm_region.boundary_pairs) > 0:
            from tunnelscan.qmmm.gradient import project_link_forces
            forces = project_link_forces(
                forces, self.qm_region.boundary_pairs, link_pos, self.atoms
            )

        return float(e_total), forces

    def update_positions(self, positions: np.ndarray):
        self.atoms.set_positions(positions)
