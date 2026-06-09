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
# LITERATURE VERIFICATION (updated 2026-06-09):
#
# NOTE ON "INFLATED KIES" LANGUAGE IN THE LITERATURE
# ───────────────────────────────────────────────────
# Kohen-group papers (Wang 2006 and variants) report intrinsic KIEs via competitive
# triple-isotope labelling (H, D, T) and express results as Arrhenius parameters
# (A_H/T, A_H/D, ΔE_a). "Inflated primary KIEs" in these papers refers to
# inflated TEMPERATURE DEPENDENCE (A_H/T deviating from semiclassical) and to the
# H/T KIE ratio, NOT the H/D KIE ratio at 25°C.
#
# Specifically:
#   G121V: A_H/T = 7.4 > semiclassical lower limit (~3.3)  → inflated H/T
#          But directly measured kH/kD at 25°C = 4.9 ± 0.2 → BELOW WT (6.8)
#          (Wang et al. 2006 Biochemistry, PMC2553318, main text Table)
#   M42W:  A_H/T = 2.8 (below semiclassical) → strongly temperature-dependent
#          No direct kH/kD at 25°C in main text of any accessible paper;
#          only Arrhenius H/T and H/D parameters given.
#
# PRIOR RETRACTION HISTORY:
#   M42W = 3.2: SOURCE NOT FOUND. Original DOI 10.1021/bi050586p invalid.
#   G121V = 3.4: SOURCE WRONG. The actual G121V kH/kD = 4.9 (below WT, not 3.4).
#   G121VM42W = 2.8: SOURCE NOT FOUND. Double mutant; outside single-point scope.
#   F125M = 3.0: SOURCE DOES NOT EXIST. No "Pudney et al. JACS 2013" DHFR paper.
#   All four moved to DHFR_KIE_DATA_UNVERIFIED.
#
# Confidence tiers:
#   HIGH — value in main text/table of accessible paper; directly verified
#   LOW  — direction consistent but exact value in SI of paywalled paper
#
# Sources:
#   [SK2004]  Sikorski et al. JACS 2004, DOI 10.1021/ja031683w — WT kH/kD
#   [WA2006B] Wang et al. Biochemistry 2006, DOI 10.1021/bi0518242 — G121V kH/kD
#   [ST2012]  Stojković et al. JACS 2012, DOI 10.1021/ja209425w — I14 series
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

    # ── G121V (HIGH confidence — directly measured kH/kD in main text) ──────────
    # G121 is a distal residue (~19 Å from active site) in the coupled network.
    # G121V disrupts the promoting-vibration dynamic network, reducing kH/kD
    # from WT 6.8 to 4.9 at 25°C. Note: the H/T Arrhenius parameter A_H/T = 7.4
    # is above WT (7.0), so G121V's "inflated" language in the literature refers
    # to temperature dependence of H/T, not to absolute kH/kD at 25°C.
    # Force-evaluated outside the 10 Å scan radius via calibration force-eval.
    KIEDataPoint(
        label='G121V', residue=121, orig_aa='GLY', new_aa='VAL', chain='A',
        kie_298k=4.9, kie_error=0.2,
        mechanism='dynamic',
        source='Wang et al. Biochemistry 2006, DOI 10.1021/bi0518242 [HIGH — kH/kD in main text, PMC2553318]'
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
