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

# Combined calibration set (protein mutants + usable substrate geometry points)
ALL_CALIBRATION_POINTS = (
    AADH_CALIBRATION["protein_mutants"]
    + [p for p in AADH_CALIBRATION["substrate_geometry"] if p["use_for_calibration"]]
)

N_CALIBRATION_POINTS = len(ALL_CALIBRATION_POINTS)  # 7
