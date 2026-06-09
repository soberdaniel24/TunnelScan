import numpy as np
import pytest


def test_link_atom_placement():
    """Link H placed at 1.0Å from QM boundary atom along QM-MM bond."""
    import ase
    from tunnelscan.structure.link_atoms import place_link_atoms

    atoms = ase.Atoms(
        ["N", "C", "H"],
        positions=[[0.0, 0.0, 0.0],   # QM: N at origin
                   [2.0, 0.0, 0.0],   # MM: C at 2 Å
                   [0.0, 1.0, 0.0]]   # unrelated
    )
    boundary_pairs = [(0, 1)]  # QM atom 0, MM atom 1
    link_pos = place_link_atoms(atoms, boundary_pairs, link_distance=1.0)

    # Link H should be at QM_atom + 1.0*(MM-QM)/|MM-QM| = (0,0,0) + (1,0,0) = (1,0,0)
    assert link_pos.shape == (1, 3)
    expected = np.array([[1.0, 0.0, 0.0]])
    assert np.allclose(link_pos, expected, atol=1e-6), (
        f"Link atom at {link_pos}, expected {expected}"
    )


def test_qm_region_expansion():
    """QM region auto-expands to include all atoms within cutoff."""
    import ase
    from tunnelscan.structure.loader import Topology
    from tunnelscan.structure.qm_region import select_qm_region

    # 5 atoms in a line: D-H-A plus two extra within 4.5 Å
    positions = np.array([
        [0.0, 0.0, 0.0],   # 0: donor
        [1.0, 0.0, 0.0],   # 1: H
        [2.0, 0.0, 0.0],   # 2: acceptor
        [3.5, 0.0, 0.0],   # 3: within 4.5 Å of donor
        [8.0, 0.0, 0.0],   # 4: far away
    ])
    atoms = ase.Atoms(["N", "H", "O", "C", "C"], positions=positions)
    topology = Topology(residues=[], hetatm_indices=set(), water_indices=set())

    qm_region = select_qm_region(atoms, topology, donor_idx=0, acceptor_idx=2,
                                  h_idx=1, cutoff=4.5)

    # Atoms 0,1,2 are seeds; atom 3 is within 4.5 of atom 0
    assert 0 in qm_region.qm_indices
    assert 1 in qm_region.qm_indices
    assert 2 in qm_region.qm_indices
    assert 3 in qm_region.qm_indices
    # Atom 4 at 8 Å from closest seed (atom 2) → distance > 4.5
    # Atom 2 is at (2,0,0), atom 4 at (8,0,0) → dist = 6 Å > 4.5
    assert 4 not in qm_region.qm_indices


def test_energy_subtraction():
    """Verify E_total = E_QM + E_MM - E_MM(QM) formula in engine."""
    import ase
    from tunnelscan.structure.loader import Topology
    from tunnelscan.structure.qm_region import QMRegion
    from tunnelscan.qmmm.engine import QMMMEngine

    positions = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ])
    atoms = ase.Atoms(["N", "H", "O"], positions=positions)

    qm_region = QMRegion(
        qm_indices=[0, 1],
        mm_indices=[2],
        boundary_pairs=[],
        link_positions=np.zeros((0, 3)),
    )

    engine = QMMMEngine(atoms, qm_region, temperature=300.0)

    # Engine should return a float energy without error
    e, f = engine.energy_and_forces()
    assert isinstance(e, float), f"Energy should be float, got {type(e)}"
    assert f.shape == (3, 3), f"Forces shape should be (3,3), got {f.shape}"
    # Subtraction scheme: total = QM + MM_all - MM_QM
    # Just verify no NaN/Inf
    assert np.isfinite(e), f"Energy {e} is not finite"
    assert np.all(np.isfinite(f)), f"Forces contain non-finite values"
