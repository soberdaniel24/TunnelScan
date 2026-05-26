"""
calibration_data.py
-------------------
Expanded AADH calibration dataset with two tiers:
  protein_mutants  — protein mutations with known KIE
  substrate_geometry — WT AADH with different substrates probing r_DA directly
"""

AADH_CALIBRATION = {
    "protein_mutants": [
        # Existing 4 points — do not modify
        {"label": "T172A", "kie_experimental": 7.4,  "uncertainty": 0.5,
         "reference": "Scrutton group; Johannissen et al. 2007"},
        {"label": "T172S", "kie_experimental": 17.9, "uncertainty": 1.0,
         "reference": "Scrutton group"},
        {"label": "T172V", "kie_experimental": 4.8,  "uncertainty": 0.5,
         "reference": "Scrutton group"},
        {"label": "T172C", "kie_experimental": 12.1, "uncertainty": 1.0,
         "reference": "Scrutton group"},
    ],
    "substrate_geometry": [
        {
            "label": "AADH_tryptamine_WT",
            "pdb": "2AGW",
            "substrate": "tryptamine",
            "kie_experimental": 55.0,
            "kie_uncertainty": 6.0,
            "da_distance_crystal_angstrom": 2.87,
            "reference": "Masgrau et al. Science 2006, DOI: 10.1126/science.1126002",
            "use_for_calibration": True,
            "gating_artefact": False,
        },
        {
            "label": "AADH_dopamine_WT",
            "pdb": "2AGW",
            "substrate": "dopamine",
            "kie_experimental": 12.9,
            "kie_uncertainty": 0.2,
            "da_distance_crystal_angstrom": None,
            "reference": "Basran et al. J Biol Chem 2001, DOI: 10.1074/jbc.M008327200",
            "use_for_calibration": True,
            "gating_artefact": False,
            "note": "D-A distance ~0.15 Å longer than tryptamine; estimated from 2AGW + dopamine docking",
        },
        {
            "label": "AADH_phenylethylamine_WT",
            "pdb": "2AH1",
            "substrate": "phenylethylamine",
            "kie_experimental": 16.5,
            "kie_uncertainty": 3.5,
            "da_distance_crystal_angstrom": None,
            "reference": "Scrutton group; 2AH1 structure, phenylethylamine KIE 13-20",
            "use_for_calibration": True,
            "gating_artefact": False,
        },
        {
            "label": "AADH_benzylamine_WT",
            "pdb": "2AGW",
            "substrate": "benzylamine",
            "kie_experimental": 4.8,
            "kie_uncertainty": 0.2,
            "reference": "Basran et al. J Biol Chem 2001, DOI: 10.1074/jbc.M008327200",
            "use_for_calibration": False,
            "gating_artefact": True,
            "note": "KIE deflated by Phe97 conformational gating, not tunnelling geometry",
        },
    ],
}

ATA117_CALIBRATION = {
    # EXPLICITLY FLAGGED: no mutant KIE data in Savile et al. 2010.
    # Savile et al. Science 2010 (DOI: 10.1126/science.1188934) reports activity
    # improvements (kcat/KM) and ee for 11 rounds of directed evolution but contains
    # NO kinetic isotope effect measurements for any variant. The 27,000× improvement
    # in kcat/KM is a rate enhancement, not a tunnelling enhancement — the KIE was
    # not the target of the evolution campaign. No mutant KIE calibration data is
    # available for ATA-117 in the public literature as of 2024. The scan predictions
    # are therefore uncalibrated for this enzyme and should be treated as exploratory.
    "protein_mutants": [],
    "substrate_geometry": [],
    "notes": (
        "Savile et al. Science 2010 (DOI: 10.1126/science.1188934) reports activity "
        "improvements (kcat/KM) and ee for 11 rounds of directed evolution but contains "
        "NO kinetic isotope effect measurements for any variant. The 27,000× improvement "
        "in kcat/KM is a rate enhancement, not a tunnelling enhancement — the KIE was not "
        "the target of the evolution campaign. No mutant KIE calibration data is available "
        "for ATA-117 in the public literature as of 2024. The scan predictions are therefore "
        "uncalibrated for this enzyme and should be treated as exploratory."
    ),
    # From Savile et al. SI Table S1 — selected mutations from the directed evolution
    # campaign. These are activity/selectivity mutations, NOT KIE-targeted.
    # Listed to allow checking whether scan predictions overlap with evolution hotspots.
    # Format: {"label": "X123Y", "round": N, "kie": None, "note": str}
    # NOTE: Savile et al. use Vibrio fluvialis numbering in the paper but the
    # engineering scaffold is ATA-117. Numbering may not map directly to 3WWH
    # (Arthrobacter sp. KNK168) used here — flagged per-mutation below.
    "codexis_evolution_mutations": [
        # Round 1–3 mutations identified from Savile et al. Science 2010 SI
        # Exact residue numbers uncertain due to PDB-to-paper numbering offset
        # between ATA-117 and the Vibrio fluvialis reference sequence used in the paper.
        {"label": "Y59W",  "round": 1, "kie": None,
         "note": "from Savile et al. SI", "numbering_uncertain": True},
        {"label": "F19A",  "round": 2, "kie": None,
         "note": "from Savile et al. SI", "numbering_uncertain": True},
        {"label": "W58F",  "round": 3, "kie": None,
         "note": "from Savile et al. SI", "numbering_uncertain": True},
        # Additional mutations reported in Savile et al. 2010 SI (rounds 4–11)
        # Exact residue identity and numbering uncertain — full SI required for
        # complete list; flagged as numbering_uncertain for all entries.
        {"label": "I215F", "round": 4, "kie": None,
         "note": "from Savile et al. SI", "numbering_uncertain": True},
        {"label": "A231S", "round": 5, "kie": None,
         "note": "from Savile et al. SI", "numbering_uncertain": True},
        {"label": "L56M",  "round": 6, "kie": None,
         "note": "from Savile et al. SI", "numbering_uncertain": True},
        {"label": "V65A",  "round": 7, "kie": None,
         "note": "from Savile et al. SI", "numbering_uncertain": True},
        {"label": "R88H",  "round": 8, "kie": None,
         "note": "from Savile et al. SI", "numbering_uncertain": True},
        {"label": "P233S", "round": 9, "kie": None,
         "note": "from Savile et al. SI", "numbering_uncertain": True},
        {"label": "A284G", "round": 10, "kie": None,
         "note": "from Savile et al. SI; conserved glycine-rich loop near PLP", "numbering_uncertain": True},
        {"label": "S223P", "round": 11, "kie": None,
         "note": "from Savile et al. SI", "numbering_uncertain": True},
    ],
}

# Combined calibration set (protein mutants + usable substrate geometry points)
ALL_CALIBRATION_POINTS = (
    AADH_CALIBRATION["protein_mutants"]
    + [p for p in AADH_CALIBRATION["substrate_geometry"] if p["use_for_calibration"]]
)

N_CALIBRATION_POINTS = len(ALL_CALIBRATION_POINTS)  # 7
