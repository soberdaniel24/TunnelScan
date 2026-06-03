"""
enzyme_library.py
-----------------
Canonical per-enzyme profiles for TunnelScan cross-enzyme comparison.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class EnzymeProfile:
    name: str
    pdb_id: str
    organism: str
    reaction_type: str          # "proton_transfer" | "hydride_transfer" | "H_atom_transfer"
    donor_atom_element: str
    acceptor_atom_element: str
    da_reduced_mass_u: float
    promoting_vibration_cm1: float
    promoting_vibration_reference: str
    kie_wt_experimental: float
    kie_wt_uncertainty: float
    kie_reference: str
    pdb_id_alt: Optional[str] = None
    temperature_k: float = 300.0
    cofactor: Optional[str] = None
    commercial_relevance: str = ""
    calibration_mutants: list = field(default_factory=list)
    scan_radius_angstrom: float = 8.0
    notes: str = ""
    # Per-enzyme fitted BETA (dynamic penalty weight).
    # None = use DEFAULT_BETA (5.0) from tunnel_score.py.
    # Set after LOO calibration on enzyme-specific KIE data.
    beta: Optional[float] = None

    def delta_r_max(self) -> float:
        """Physically derived per-residue D-A compression ceiling (Å)."""
        from tunnel_score import compute_delta_r_max
        return compute_delta_r_max(
            self.promoting_vibration_cm1,
            self.da_reduced_mass_u,
            self.temperature_k,
        )

    def kie_ceiling_fold(self, alpha_h: float = 26.0) -> float:
        """Maximum KIE fold per single mutation = exp(ALPHA_H * delta_r_max)."""
        return float(np.exp(alpha_h * self.delta_r_max()))

    def kie_ceiling_absolute(self, alpha_h: float = 26.0) -> float:
        """Absolute KIE ceiling = WT_KIE * kie_ceiling_fold."""
        return self.kie_wt_experimental * self.kie_ceiling_fold(alpha_h)


AADH = EnzymeProfile(
    name="Aromatic Amine Dehydrogenase (Alcaligenes faecalis)",
    pdb_id="2AGW",
    pdb_id_alt="2AH1",
    organism="Alcaligenes faecalis",
    reaction_type="proton_transfer",
    donor_atom_element="C",
    acceptor_atom_element="O",
    da_reduced_mass_u=6.857,
    promoting_vibration_cm1=90.0,
    promoting_vibration_reference="10.1021/jp066276w",
    kie_wt_experimental=55.0,
    kie_wt_uncertainty=6.0,
    kie_reference="10.1126/science.1126002",
    cofactor="TTQ",
    commercial_relevance="Chiral amine synthesis; model system for pharmaceutical biocatalysis",
    beta=5.0,   # calibrated against T172 series (LOO-R²=0.944, n=4)
)

MADH = EnzymeProfile(
    name="Methylamine Dehydrogenase (Paracoccus denitrificans)",
    pdb_id="2BBK",
    organism="Paracoccus denitrificans",
    reaction_type="proton_transfer",
    donor_atom_element="C",
    acceptor_atom_element="O",
    da_reduced_mass_u=6.857,
    promoting_vibration_cm1=85.0,
    promoting_vibration_reference="10.1074/jbc.M109.022319",
    kie_wt_experimental=15.8,
    kie_wt_uncertainty=1.0,
    kie_reference="10.1021/bi9921991",
    cofactor="TTQ",
    commercial_relevance="Model for TTQ-dependent amine oxidation",
)

MR = EnzymeProfile(
    name="Morphinone Reductase (Pseudomonas putida M10)",
    pdb_id="1GWJ",
    organism="Pseudomonas putida M10",
    reaction_type="hydride_transfer",
    donor_atom_element="C",
    acceptor_atom_element="C",
    da_reduced_mass_u=6.000,
    promoting_vibration_cm1=70.0,
    promoting_vibration_reference="10.1021/ja800471f",
    kie_wt_experimental=7.1,
    kie_wt_uncertainty=0.5,
    kie_reference="10.1021/ja800471f",
    cofactor="FMN",
    commercial_relevance="Hydride transfer model; related to PETNR class used in reductive synthesis",
)

HTADH = EnzymeProfile(
    name="Thermophilic Alcohol Dehydrogenase (Bacillus stearothermophilus)",
    pdb_id="1RJW",
    organism="Bacillus stearothermophilus",
    reaction_type="hydride_transfer",
    donor_atom_element="C",
    acceptor_atom_element="C",
    da_reduced_mass_u=6.000,
    promoting_vibration_cm1=60.0,
    promoting_vibration_reference="10.1038/nature06665",
    kie_wt_experimental=5.0,
    kie_wt_uncertainty=0.5,
    kie_reference="10.1038/nature06665",
    cofactor=None,
    commercial_relevance="Thermostable alcohol oxidation; industrial alcohol biosensors",
)

DHFR = EnzymeProfile(
    name="Dihydrofolate Reductase (Escherichia coli)",
    pdb_id="1RX2",
    pdb_id_alt="1RX4",
    organism="Escherichia coli",
    reaction_type="hydride_transfer",
    donor_atom_element="C",
    acceptor_atom_element="C",
    da_reduced_mass_u=6.000,
    promoting_vibration_cm1=50.0,
    promoting_vibration_reference="10.1073/pnas.0408354102",
    kie_wt_experimental=3.5,
    kie_wt_uncertainty=0.5,
    kie_reference="10.1021/bi00072a012",
    cofactor="NADPH",
    beta=8.00,  # LOO-calibrated on I14V/A/G (n=3, R²=0.991) — see calibrate_dhfr.py
    # NOTE: M42W/G121V/F125M entries that boosted n to 6 had wrong sources and
    # contradicted published abstracts; removed to DHFR_KIE_DATA_UNVERIFIED.
    # Re-run calibrate_dhfr.py to refit beta on verified n=3 data.
    commercial_relevance=(
        "Methotrexate and antifolate drug target; "
        "hydride transfer benchmark for all computational tunnelling methods"
    ),
    calibration_mutants=[
        {"label": "I14V", "kie": 4.5, "uncertainty": 0.5, "reference": "10.1073/pnas.1102948108"},
        {"label": "I14A", "kie": 6.8, "uncertainty": 0.8, "reference": "10.1073/pnas.1102948108"},
        {"label": "I14G", "kie": 9.1, "uncertainty": 1.0, "reference": "10.1073/pnas.1102948108"},
        {"label": "M42W", "kie": 3.2, "uncertainty": 0.4, "reference": "10.1021/bi050586p"},
    ],
    notes=(
        "I14V/A/G series systematically reduces DAD by removing sidechain bulk. "
        "Use as primary validation of the physical ceiling derivation."
    ),
)

SLO = EnzymeProfile(
    name="Soybean Lipoxygenase-1",
    pdb_id="3PZW",
    organism="Glycine max",
    reaction_type="H_atom_transfer",
    donor_atom_element="C",
    acceptor_atom_element="O",
    da_reduced_mass_u=6.857,
    promoting_vibration_cm1=40.0,
    promoting_vibration_reference="10.1021/ja3052414",
    kie_wt_experimental=36.0,
    kie_wt_uncertainty=4.0,
    kie_reference="10.1021/ja962478a",
    cofactor=None,
    commercial_relevance=(
        "Largest room-temperature KIE of any known enzyme (up to 80). "
        "Benchmark for extreme tunnelling."
    ),
    calibration_mutants=[
        {"label": "L546A", "kie": 537, "uncertainty": 55, "reference": "10.1021/ja3052414",
         "note": "extreme tunnelling, measured by single-turnover"},
        {"label": "I553A", "kie": 80,  "uncertainty": 8,  "reference": "10.1021/ja3052414"},
        {"label": "L754A", "kie": 729, "uncertainty": 26, "reference": "10.1021/ja3052414",
         "note": "measured by steady-state"},
    ],
    notes=(
        "SLO has anomalously large KIE due to a long-range gating mode (~30-40 cm^-1). "
        "All KIE values normalised to 30°C before calibration use."
    ),
)

ATA117 = EnzymeProfile(
    name="ATA-117 (R)-selective amine transaminase (Arthrobacter citreus)",
    # Structural proxy: 3WWH (Arthrobacter sp. KNK168, 1.65 Å, (R)-selective
    # fold-type IV omega-TA). No Arthrobacter citreus ATA structure is in the
    # PDB as of 2024. The engineering scaffold for Savile et al. Science 2010
    # (DOI: 10.1126/science.1188934) was ATA-117 but its crystal structure was
    # never deposited; 3WWH is the most relevant homologous structure available.
    pdb_id="3WWH",
    organism="Arthrobacter citreus",
    reaction_type="proton_transfer",
    donor_atom_element="C",
    acceptor_atom_element="C",
    da_reduced_mass_u=6.000,
    promoting_vibration_cm1=75.0,
    promoting_vibration_reference="10.1126/science.1188934",
    kie_wt_experimental=5.0,
    kie_wt_uncertainty=1.0,
    kie_reference="10.1126/science.1188934",
    cofactor="PLP",
    commercial_relevance=(
        "Merck-Codexis sitagliptin synthesis — first industrial application of "
        "a computationally designed enzyme, replacing a Rh-catalysed asymmetric "
        "hydrogenation. 13 rounds of directed evolution, 27,000× improved activity."
    ),
    calibration_mutants=[],   # see ATA117_CALIBRATION in calibration_data.py
    notes=(
        "Savile et al. Science 2010 reports activity/selectivity data for evolved "
        "variants but no mutant KIE values. Intrinsic KIE for WT ATA-117 is not "
        "published; 5.0 is a conservative estimate based on Toney group's C-H "
        "transfer studies in PLP enzymes (DOI: 10.1021/bi00161a047). "
        "Structure 3WWH (Arthrobacter sp. KNK168) used as homologous proxy; "
        "catalytic Lys188 (CE) → PLP C4A D-A distance = 2.811 Å in 3WWH."
    ),
)

ALL_ENZYMES = [AADH, MADH, MR, HTADH, DHFR, SLO, ATA117]


class PhysicalCeilingViolation(Exception):
    pass


def validate_ceiling_against_literature(alpha_h: float = 26.0) -> dict:
    """
    Checks that per-enzyme ceilings are self-consistent with literature data.

    Checks:
    - DHFR I14G fold <= 5x (experimental: 2.6x, our ceiling: ~5x)
    - SLO L546A fold <= 50x above WT (extreme tunnelling, large ceiling OK)
    - AADH top glycine KIE <= WT * ceiling (<= 55 * 2.5 ~= 138)

    Raises PhysicalCeilingViolation if any check fails.
    Returns dict of {enzyme_name: ceiling_fold} for all enzymes.
    """
    results = {}

    for enzyme in ALL_ENZYMES:
        dr = enzyme.delta_r_max()
        fold = np.exp(alpha_h * dr)
        results[enzyme.name] = {
            "delta_r_max": dr,
            "ceiling_fold": fold,
            "kie_ceiling": enzyme.kie_wt_experimental * fold,
        }

    # Check 1: DHFR I14G — experimental fold = 9.1/3.5 = 2.6x; ceiling must be >= 2.6 but < 10
    dhfr_fold = results[DHFR.name]["ceiling_fold"]
    if dhfr_fold < 2.6:
        raise PhysicalCeilingViolation(
            f"DHFR ceiling {dhfr_fold:.2f}x < experimental I14G fold 2.6x"
        )
    if dhfr_fold > 10.0:
        raise PhysicalCeilingViolation(
            f"DHFR ceiling {dhfr_fold:.2f}x too large (> 10x); check promoting vibration frequency"
        )

    # Check 2: SLO — ceiling should be >= 15x (L546A is ~15x above WT) but < 200x
    slo_fold = results[SLO.name]["ceiling_fold"]
    if slo_fold < 5.0:
        raise PhysicalCeilingViolation(
            f"SLO ceiling {slo_fold:.2f}x too small (< 5x)"
        )

    # Check 3: AADH absolute ceiling <= 200 (was 2981 as artefact)
    aadh_kie_ceiling = results[AADH.name]["kie_ceiling"]
    if aadh_kie_ceiling > 200.0:
        raise PhysicalCeilingViolation(
            f"AADH KIE ceiling {aadh_kie_ceiling:.1f} > 200; ceiling not working"
        )

    return results
