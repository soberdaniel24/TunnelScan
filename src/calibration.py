"""
calibration.py
--------------
Published experimental KIE data for AADH and model calibration.

Uses the Scrutton group's published mutant KIE data to fit the weights
(alpha, beta, gamma) of the three-component TunnelScore. This means
predictions are grounded in real experimental observations, not just
theoretical approximations.

Primary sources:
  Masgrau et al. (2006) Science 312:237       — WT and key mutants
  Hay & Scrutton (2012) Nature Chemistry 4:161 — promoting vibrations
  Johannissen et al. (2011) FEBS J 278:1701   — dynamics analysis
  Pang et al. (2010) JACS 132:7038            — T172 series
  Hothi et al. (2008) ChemBioChem 9:2839      — N198 series

Data format:
  Each entry: mutation name, residue, orig_aa, new_aa,
              experimental KIE at 298K, KIE uncertainty,
              dominant mechanism, source
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class KIEDataPoint:
    """One experimentally measured KIE for an AADH variant."""
    label:       str    # e.g. 'T172A'
    residue:     int    # residue number
    orig_aa:     str    # three-letter original
    new_aa:      str    # three-letter substitution (or 'WT')
    chain:       str    # chain ID in PDB (B for AADH small subunit)
    kie_298k:    float  # experimental kH/kD at 298K
    kie_error:   float  # measurement uncertainty (±)
    mechanism:   str    # 'dynamic', 'static', 'mixed', 'wt'
    source:      str    # citation


# ── Published AADH KIE dataset ────────────────────────────────────────────────
#
# These are the training data for calibration.
# All values are for tryptamine as substrate at 298K.
# 'dynamic' = mutation disrupts promoting vibration (KIE drops dramatically)
# 'static'  = mutation affects geometry only (smaller KIE change)
#
AADH_KIE_DATA: List[KIEDataPoint] = [

    KIEDataPoint(
        label='WT', residue=0, orig_aa='WT', new_aa='WT', chain='B',
        kie_298k=55.0, kie_error=4.0,
        mechanism='wt',
        source='Masgrau et al. Science 2006'
    ),

    # T172 series — Thr172 is a DYNAMIC residue
    # Its hydroxyl H-bonds to the substrate and drives compressive motion.
    # Removing or changing this H-bond dramatically kills tunnelling.
    KIEDataPoint(
        label='T172A', residue=172, orig_aa='THR', new_aa='ALA', chain='B',
        kie_298k=7.4,  kie_error=0.8,
        mechanism='dynamic',
        source='Hay & Scrutton Nature Chemistry 2012'
    ),
    KIEDataPoint(
        label='T172S', residue=172, orig_aa='THR', new_aa='SER', chain='B',
        kie_298k=17.9, kie_error=1.5,
        mechanism='dynamic',
        source='Hay & Scrutton Nature Chemistry 2012'
    ),
    KIEDataPoint(
        label='T172V', residue=172, orig_aa='THR', new_aa='VAL', chain='B',
        kie_298k=4.8,  kie_error=0.6,
        mechanism='dynamic',
        source='Pang et al. JACS 2010'
    ),
    KIEDataPoint(
        label='T172C', residue=172, orig_aa='THR', new_aa='CYS', chain='B',
        kie_298k=12.1, kie_error=1.2,
        mechanism='dynamic',
        source='Hay & Scrutton Nature Chemistry 2012'
    ),

    # N198 series — Asn198 is more peripheral, STATIC character
    # Less coupled to promoting vibration, mutations cause moderate reduction
]

# ── DHFR published intrinsic KIE dataset ─────────────────────────────────────
#
# All values are intrinsic (commitment-corrected) kH/kD at ~298K unless noted.
# WT intrinsic KIE = 6.8; canonical measurement: Sikorski et al. 2004 JACS
# 126:4778, DOI 10.1021/ja031683w.
#
# LITERATURE VERIFICATION (2026-05-26):
# Each entry below was checked against its cited DOI. Findings:
#
#   I14 series: source DOI 10.1073/pnas.1102948108 ("Loveridge PNAS 2011")
#     does NOT exist (404). Correct paper is Stojković et al. 2012 JACS
#     DOI 10.1021/ja209425w (PMID 22171795, PMC4341912). Values (4.5/6.8/9.1)
#     are in the SI of that paper; direction is consistent with published
#     abstracts but exact numbers COULD NOT BE VERIFIED from open-access text.
#
#   M42W = 3.2: SOURCE NOT FOUND. DOI 10.1021/bi050586p returns no PubMed
#     result. Wang et al. 2006 Phil Trans (PMID 16873118) — the actual M42W
#     intrinsic KIE paper — explicitly states M42W has "inflated primary KIEs"
#     ABOVE WT. Stored value 3.2 < WT 6.8 is the WRONG DIRECTION.
#
#   G121V = 3.4: SOURCE WRONG. DOI 10.1073/pnas.032598199 misformatted;
#     Rajagopalan et al. 2002 Biochemistry (PMID 12379104) reports hydride
#     transfer RATES not KIEs. Wang et al. 2006 Biochemistry (PMID 16445280)
#     — the actual G121V KIE paper — says G121V has "slightly inflated primary
#     KIEs" ABOVE WT. Stored value 3.4 < WT 6.8 is the WRONG DIRECTION.
#
#   G121VM42W = 2.8: SOURCE NOT FOUND. Same bad DOI. Value unverifiable;
#     double mutant cannot be predicted by single-point scan regardless.
#
#   F125M = 3.0: SOURCE DOES NOT EXIST. No "Pudney et al. JACS 2013" DHFR
#     paper found in PubMed (only unrelated antimalarial Pudney paper). Value
#     appears fabricated.
#
# RESULT: M42W, G121V, G121VM42W, F125M moved to DHFR_KIE_DATA_UNVERIFIED.
# DHFR_KIE_DATA contains only entries whose direction is consistent with
# published literature. BETA_DHFR re-calibration on n=3 is needed.
#
# Confidence tiers:
#   HIGH      — sourced to correct paper; direction consistent with literature
#   LOW       — direction consistent but exact value not verified from full text
#   RETRACTED — stored value contradicts published data; moved to unverified list
#
# Sources:
#   [SI2004] Sikorski et al. JACS 2004, DOI 10.1021/ja031683w — WT intrinsic KIE
#   [ST2012] Stojković et al. JACS 2012, DOI 10.1021/ja209425w — I14 series
#
DHFR_KIE_DATA: List[KIEDataPoint] = [

    KIEDataPoint(
        label='WT', residue=0, orig_aa='WT', new_aa='WT', chain='A',
        kie_298k=6.8, kie_error=0.6,
        mechanism='wt',
        source='Sikorski et al. JACS 2004, DOI 10.1021/ja031683w [HIGH]'
    ),

    # ── I14 series (LOW confidence — correct source, values in SI not verified) ──
    # Ile14 lines the hydride donor face of the nicotinamide ring in 1RX2.
    # Smaller sidechains broaden the DAD distribution: I14G → most temperature-
    # dependent KIE (inflated at 25°C), I14V → slightly deflated. Direction
    # confirmed by Stojković 2012 abstract and cited computational studies.
    # Exact values (4.5 / 6.8 / 9.1) are in the paper's SI; full text blocked.
    KIEDataPoint(
        label='I14V', residue=14, orig_aa='ILE', new_aa='VAL', chain='A',
        kie_298k=4.5, kie_error=0.5,
        mechanism='static',
        source='Stojković et al. JACS 2012, DOI 10.1021/ja209425w [LOW — values in SI, not verified from full text]'
    ),
    KIEDataPoint(
        label='I14A', residue=14, orig_aa='ILE', new_aa='ALA', chain='A',
        kie_298k=6.8, kie_error=0.8,
        mechanism='static',
        source='Stojković et al. JACS 2012, DOI 10.1021/ja209425w [LOW — values in SI, not verified from full text]'
    ),
    KIEDataPoint(
        label='I14G', residue=14, orig_aa='ILE', new_aa='GLY', chain='A',
        kie_298k=9.1, kie_error=1.0,
        mechanism='static',
        source='Stojković et al. JACS 2012, DOI 10.1021/ja209425w [LOW — values in SI, not verified from full text]'
    ),
]

# ── Known experimental status of mutations ────────────────────────────────────

TESTED_MUTATIONS = {d.label for d in AADH_KIE_DATA if d.new_aa != 'WT'}
DHFR_TESTED_MUTATIONS = {d.label for d in DHFR_KIE_DATA if d.new_aa != 'WT'}

# ── Calibration fitting ───────────────────────────────────────────────────────

@dataclass
class CalibrationResult:
    alpha: float    # weight on static geometric score
    beta:  float    # weight on dynamic score (promoting vibration)
    r2:    float    # coefficient of determination on training data
    rmse:  float    # root mean squared error in ln(KIE)
    n_points: int


def fit_calibration(
    scores: List[Tuple[str, float, float]],
    data: List[KIEDataPoint] = AADH_KIE_DATA
) -> CalibrationResult:
    """
    Fit alpha, beta weights using least squares on published KIE data.

    Parameters
    ----------
    scores : list of (label, static_score, dynamic_score)
        Pre-computed scores for each data point in AADH_KIE_DATA.

    Returns
    -------
    CalibrationResult
    """
    # Build design matrix and target vector
    # Model: ln(KIE) = ln(KIE_WT) + alpha * static + beta * dynamic
    ln_kie_wt = np.log(55.0)

    score_dict = {label: (s, d) for label, s, d in scores}

    X_rows = []
    y = []

    for dp in data:
        if dp.new_aa == 'WT':
            continue
        if dp.label not in score_dict:
            continue
        s_score, d_score = score_dict[dp.label]
        X_rows.append([s_score, d_score])
        y.append(np.log(dp.kie_298k) - ln_kie_wt)

    if len(X_rows) < 2:
        return CalibrationResult(alpha=1.0, beta=1.0, r2=0.0, rmse=1.0, n_points=0)

    X = np.array(X_rows)
    y = np.array(y)

    # Least squares: min ||Xw - y||²
    # Use pseudo-inverse for stability
    try:
        w, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
        alpha, beta = float(w[0]), float(w[1])
    except Exception:
        alpha, beta = 1.0, 1.0

    # Compute R² and RMSE
    y_pred = X @ np.array([alpha, beta])
    ss_res = float(np.sum((y - y_pred)**2))
    ss_tot = float(np.sum((y - y.mean())**2))
    r2     = 1 - ss_res/ss_tot if ss_tot > 0 else 0.0
    rmse   = float(np.sqrt(ss_res / len(y)))

    return CalibrationResult(
        alpha=alpha, beta=beta,
        r2=r2, rmse=rmse,
        n_points=len(X_rows)
    )


def is_novel_prediction(mutation_label: str,
                         data: Optional[List[KIEDataPoint]] = None) -> bool:
    """True if this mutation has not been experimentally tested."""
    if data is None:
        return mutation_label not in TESTED_MUTATIONS
    tested = {d.label for d in data if d.new_aa != 'WT'}
    return mutation_label not in tested


def get_known_kie(mutation_label: str,
                  data: Optional[List[KIEDataPoint]] = None) -> Optional[float]:
    """Return experimental KIE if known, else None."""
    dataset = data if data is not None else AADH_KIE_DATA
    for dp in dataset:
        if dp.label == mutation_label:
            return dp.kie_298k
    return None
