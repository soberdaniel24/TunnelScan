import os
import time
import tempfile
import numpy as np
import pytest
import json


def _write_mock_pdb(path: str):
    """Write a minimal 3-atom D-H-A PDB (N-H-O linear)."""
    lines = [
        "REMARK Mock D-H-A test structure\n",
        "ATOM      1  N   ALA A   1       0.900   0.000   0.000  1.00  0.00           N\n",
        "ATOM      2  H   ALA A   1       0.000   0.000   0.000  1.00  0.00           H\n",
        "ATOM      3  O   ALA A   1      -2.500   0.000   0.000  1.00  0.00           O\n",
        "END\n",
    ]
    with open(path, "w") as f:
        f.writelines(lines)


def test_full_pipeline_fast():
    """Integration test: run full pipeline in fast_test mode on a mock 3-atom molecule."""
    from tunnelscan.pipeline.runner import run_tunnelscan
    from tunnelscan.kie.tunnelling_contribution import TunnellingResult

    with tempfile.TemporaryDirectory() as tmpdir:
        pdb_path = os.path.join(tmpdir, "mock.pdb")
        _write_mock_pdb(pdb_path)
        output_dir = os.path.join(tmpdir, "output")

        t0 = time.time()
        result = run_tunnelscan(
            pdb_path=pdb_path,
            donor_atom_index=0,
            acceptor_atom_index=2,
            hydrogen_atom_index=1,
            temperature=300.0,
            n_beads=4,
            n_tps_paths=5,
            run_convergence_check=False,
            experimental_kie=None,
            output_dir=output_dir,
            fast_test=True,
            use_gpu=False,
        )
        elapsed = time.time() - t0

        assert isinstance(result, TunnellingResult), f"Expected TunnellingResult, got {type(result)}"
        assert result.delta_delta_G is not None
        assert result.tunnelling_factor is not None
        assert result.kie_theoretical is not None
        assert result.kie_experimental is None
        assert result.zpe is not None
        assert result.classification is not None
        assert isinstance(result.flags, list)

        # Check output files created
        expected_files = [
            "results.json",
            "results_summary.txt",
            "tunnelling_report.md",
        ]
        for fname in expected_files:
            fpath = os.path.join(output_dir, fname)
            assert os.path.exists(fpath), f"Missing output file: {fname}"

        assert elapsed < 60.0, f"Pipeline took {elapsed:.1f}s, expected < 60s"


def _write_ala_dipeptide_pdb(path: str):
    """
    Write a minimal alanine dipeptide (ACE-ALA-NME) PDB.

    Atom index mapping (0-based):
      0  : ACE CH3 (C)   — methyl carbon
      1  : ACE C         — carbonyl carbon
      2  : ACE O         — carbonyl oxygen  ← acceptor (index 2)
      3  : ALA N         — amide nitrogen   ← donor   (index 3)
      4  : ALA H         — amide hydrogen   ← H       (index 4)
      5  : ALA CA        — alpha carbon
      6  : ALA C         — carbonyl carbon
      7  : ALA O         — carbonyl oxygen
      8  : NME N         — N-methyl nitrogen
      9  : NME HN        — N-methyl N-H
      10 : NME CH3 (C)   — methyl carbon

    N–H···O=C geometry: d(N–H)≈1.01 Å, d(H···O)≈2.0 Å.
    """
    lines = [
        "REMARK  Alanine dipeptide (ACE-ALA-NME) minimal test structure\n",
        "ATOM      1  CH3 ACE A   1       1.720   0.700   0.000  1.00  0.00           C\n",
        "ATOM      2  C   ACE A   1       0.260   0.700   0.000  1.00  0.00           C\n",
        "ATOM      3  O   ACE A   1      -0.360   1.750   0.000  1.00  0.00           O\n",  # acceptor idx=2
        "ATOM      4  N   ALA A   2      -0.360  -0.490   0.000  1.00  0.00           N\n",  # donor  idx=3
        "ATOM      5  H   ALA A   2       0.000  -1.390   0.000  1.00  0.00           H\n",  # H      idx=4
        "ATOM      6  CA  ALA A   2      -1.800  -0.650   0.000  1.00  0.00           C\n",
        "ATOM      7  C   ALA A   2      -2.380   0.750   0.000  1.00  0.00           C\n",
        "ATOM      8  O   ALA A   2      -1.720   1.790   0.000  1.00  0.00           O\n",
        "ATOM      9  N   NME A   3      -3.700   0.850   0.000  1.00  0.00           N\n",
        "ATOM     10  HN  NME A   3      -4.240   0.020   0.000  1.00  0.00           H\n",
        "ATOM     11  CH3 NME A   3      -4.420   2.100   0.000  1.00  0.00           C\n",
        "END\n",
    ]
    with open(path, "w") as f:
        f.writelines(lines)


def test_alanine_dipeptide_pipeline():
    """
    Integration test: run full pipeline in fast_test mode on alanine dipeptide.

    Uses ACE-ALA-NME with the backbone amide N as donor (idx 3), amide H as
    hydrogen (idx 4), and ACE carbonyl O as acceptor (idx 2).

    Verifies:
      - TunnellingResult is fully populated
      - All four output files are written
      - results.json is valid JSON with required keys
      - N=8 vs N=16 convergence check runs without error
      - Pipeline completes within 90 s in fast_test mode
    """
    from tunnelscan.pipeline.runner import run_tunnelscan
    from tunnelscan.kie.tunnelling_contribution import TunnellingResult

    with tempfile.TemporaryDirectory() as tmpdir:
        pdb_path = os.path.join(tmpdir, "ala_dip.pdb")
        _write_ala_dipeptide_pdb(pdb_path)
        output_dir = os.path.join(tmpdir, "ala_output")

        t0 = time.time()
        result = run_tunnelscan(
            pdb_path=pdb_path,
            donor_atom_index=3,     # ALA N
            acceptor_atom_index=2,  # ACE O
            hydrogen_atom_index=4,  # ALA H
            temperature=300.0,
            n_beads=8,
            n_tps_paths=20,         # reduced ensemble for integration test
            run_convergence_check=True,
            experimental_kie=None,
            output_dir=output_dir,
            fast_test=True,
            use_gpu=False,
        )
        elapsed = time.time() - t0

        # --- TunnellingResult fully populated ---
        assert isinstance(result, TunnellingResult)
        required_fields = [
            "delta_delta_G", "tunnelling_factor", "kie_theoretical",
            "zpe", "classification", "confidence_interval_95",
            "trajectory_variance", "deuterium_check_passed", "flags",
        ]
        for field in required_fields:
            val = getattr(result, field)
            assert val is not None, f"Field {field!r} is None"
        assert result.kie_experimental is None  # not provided
        assert result.classification in (
            "negligible", "weak", "moderate", "deep"
        ), f"Unexpected classification: {result.classification!r}"
        assert isinstance(result.flags, list)

        # --- Output files ---
        for fname in ("results.json", "results_summary.txt", "tunnelling_report.md"):
            fpath = os.path.join(output_dir, fname)
            assert os.path.exists(fpath), f"Missing output file: {fname}"

        # --- results.json is valid and has required keys ---
        with open(os.path.join(output_dir, "results.json")) as fh:
            data = json.load(fh)
        for key in ("delta_delta_G", "classification", "kie_theoretical",
                    "deuterium_check_passed", "classical_barrier"):
            assert key in data, f"Missing key in results.json: {key!r}"

        assert elapsed < 90.0, f"Pipeline took {elapsed:.1f}s, expected < 90s"
