"""
multi_mutation.py
-----------------
Predicts the combined effect of multiple simultaneous mutations.

Physical basis:
  Single mutation predictions assume all other residues are wild-type.
  Real enzyme engineering often combines multiple mutations. The combined
  effect is not simply additive because:

  1. GEOMETRIC INTERACTIONS: Two mutations that both compress the D-A
     distance have diminishing returns — you can only compress so far
     before steric clashes dominate. Effect is sub-additive.

  2. DYNAMIC INTERACTIONS: A static mutation (opens geometry) paired
     with a dynamic mutation (enhances promoting vibration) can be
     synergistic — each addresses a different physical bottleneck.

  3. ANTAGONISM: Two dynamic mutations affecting the same normal mode
     are sub-additive. Two mutations that both rigidify the active site
     damp the promoting vibration more than either alone.

  4. EPISTASIS: Rare cases where mutation B changes the structural
     context enough that mutation A's effect is qualitatively different.

Algorithm:
  For each pair (mut_i, mut_j) from top N singles:

  combined_delta = static_i + static_j                  (geometric, additive)
                 + dynamic_interaction(i, j)            (mode coupling)
                 + static_saturation_penalty(i, j)      (diminishing returns)

  dynamic_interaction(i, j):
    if both static-dominated: 0 (no coupling)
    if one static + one dynamic: +synergy_bonus
    if both dynamic, same mode: -overlap_penalty
    if both dynamic, different modes: small positive

  static_saturation_penalty:
    if both mutations compress D-A (negative da_change):
      penalty = -0.3 × min(|da_i|, |da_j|) × ALPHA_H
    else: 0

Validation:
  The Scrutton group has published a handful of double mutants for AADH.
  These provide a ground truth for the interaction model.
  Known double mutants: T172A/N198A (experimental KIE ~5.1)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tunnel_score import MutationScore
from calibration import AADH_KIE_DATA


ALPHA_H = 26.0


@dataclass
class DoubleMutantScore:
    """Prediction for a double mutant."""
    label:              str        # e.g. 'L380G/I374G'
    mut_a:              MutationScore
    mut_b:              MutationScore

    # Combined components
    static_combined:    float
    dynamic_combined:   float
    interaction_delta:  float      # synergy (+) or antagonism (-)
    total_delta:        float

    predicted_kie:      float
    fold_vs_wt:         float
    fold_vs_best_single: float     # how much better than the best single mut

    interaction_type:   str        # 'synergistic' | 'additive' | 'antagonistic'
    confidence:         float

    experimental_kie:   Optional[float] = None
    is_novel:           bool = True

    def summary(self) -> str:
        tag = f'[exp={self.experimental_kie:.0f}]' if self.experimental_kie else ''
        return (
            f"{self.label:<22} KIE={self.predicted_kie:>7.1f}  "
            f"fold_vs_best={self.fold_vs_best_single:>+5.2f}x  "
            f"{self.interaction_type:<14} "
            f"conf={self.confidence:.2f}  {tag}"
        )


def _geometric_saturation(da_a: float, da_b: float) -> float:
    """
    Penalty when two mutations both compress D-A distance.
    Physical basis: diminishing returns — each Angstrom of compression
    is exponentially more valuable, so two mutations compressing in the
    same direction have sub-additive geometric effects.
    Returns negative penalty to subtract from combined delta.
    """
    if da_a >= 0 or da_b >= 0:
        return 0.0  # only applies when both compress
    # Both negative (both compress)
    # Penalty = fraction of the smaller compression that gets 'used up'
    overlap = min(abs(da_a), abs(da_b))
    return float(-0.3 * overlap * ALPHA_H)


def _dynamic_interaction(a: MutationScore, b: MutationScore) -> float:
    """
    Interaction between dynamic components of two mutations.

    Static + Dynamic → synergistic: static opens geometry, dynamic
    enhances the amplitude of the promoting vibration — different
    physical bottlenecks, additive benefits.

    Dynamic + Dynamic (same residue vicinity) → sub-additive: both
    mutations affect the same normal mode, diminishing returns.

    Dynamic + Dynamic (different residue vicinity) → roughly additive.
    """
    a_is_dynamic = abs(a.dynamic_delta) > abs(a.static_delta) * 0.5
    b_is_dynamic = abs(b.dynamic_delta) > abs(b.static_delta) * 0.5
    a_is_static  = abs(a.static_delta)  > abs(a.dynamic_delta) * 2.0
    b_is_static  = abs(b.static_delta)  > abs(b.dynamic_delta) * 2.0

    # Static + Dynamic pairing — synergistic
    if (a_is_static and b_is_dynamic) or (a_is_dynamic and b_is_static):
        # Synergy bonus: up to 15% of the smaller dynamic component
        dyn_component = min(abs(a.dynamic_delta), abs(b.dynamic_delta))
        return float(+0.15 * dyn_component)

    # Dynamic + Dynamic — check for mode overlap via residue proximity
    if a_is_dynamic and b_is_dynamic:
        # Residues close together likely affect same mode → sub-additive
        # Use axis_distance as a proxy for spatial separation
        spatial_sep = abs(a.axis_distance - b.axis_distance)
        if spatial_sep < 3.0:
            # Close together — significant overlap
            overlap_penalty = -0.25 * min(abs(a.dynamic_delta),
                                          abs(b.dynamic_delta))
            return float(overlap_penalty)
        else:
            # Far apart — different modes, roughly additive
            return 0.0

    return 0.0


def score_double_mutant(
    a: MutationScore,
    b: MutationScore,
    wt_kie: float = 11.3,
    beta: float = 3.0
) -> DoubleMutantScore:
    """
    Predict the KIE for the double mutant combining mutations a and b.

    Parameters
    ----------
    a, b : MutationScore
        Single mutant predictions from TunnelScorer.score_mutation().
    wt_kie : float
        Wild-type KIE (Bell-correction baseline, default 11.3).
    beta : float
        Dynamic penalty weight (should match TunnelScorer beta).

    Returns
    -------
    DoubleMutantScore
    """
    label = f"{a.label}/{b.label}"

    # Static components are largely additive (geometric effects sum)
    static_combined = a.static_delta + b.static_delta

    # Apply geometric saturation penalty
    sat_penalty = _geometric_saturation(a.da_change, b.da_change)
    static_combined += sat_penalty

    # Dynamic components with interaction term
    dynamic_raw = a.dynamic_delta + b.dynamic_delta
    interaction  = _dynamic_interaction(a, b)
    dynamic_combined = dynamic_raw + interaction

    # Breathing components — additive (different residues, independent)
    breath_combined = a.breathing_delta + b.breathing_delta

    # Total delta
    total_delta = (static_combined
                   + beta * dynamic_combined
                   + breath_combined)

    predicted_kie = float(np.exp(np.clip(np.log(wt_kie) + total_delta, 0.0, 8.0)))
    fold_vs_wt    = predicted_kie / wt_kie

    # Compare to best single mutant
    best_single_kie = max(a.predicted_kie, b.predicted_kie)
    fold_vs_best    = predicted_kie / best_single_kie

    # Classify interaction
    if interaction > 0.05:
        interaction_type = 'synergistic'
    elif interaction < -0.05:
        interaction_type = 'antagonistic'
    elif sat_penalty < -0.3:
        interaction_type = 'sub-additive'
    else:
        interaction_type = 'additive'

    # Confidence: lower than single mutant confidence
    confidence = float(np.sqrt(a.confidence * b.confidence)) * 0.8

    # Check for known experimental double mutants
    exp_kie = _get_known_double_mutant_kie(a.label, b.label)

    return DoubleMutantScore(
        label=label,
        mut_a=a,
        mut_b=b,
        static_combined=static_combined,
        dynamic_combined=dynamic_combined,
        interaction_delta=interaction,
        total_delta=total_delta,
        predicted_kie=predicted_kie,
        fold_vs_wt=fold_vs_wt,
        fold_vs_best_single=fold_vs_best,
        interaction_type=interaction_type,
        confidence=confidence,
        experimental_kie=exp_kie,
        is_novel=(exp_kie is None)
    )


def _get_known_double_mutant_kie(label_a: str, label_b: str) -> Optional[float]:
    """Return experimental KIE for known double mutants."""
    known = {
        frozenset(['T172A', 'N156A']): 5.1,
    }
    key = frozenset([label_a, label_b])
    return known.get(key)


def scan_double_mutants(
    single_scores: List[MutationScore],
    top_n: int = 20,
    wt_kie: float = 11.3,
    beta: float = 3.0,
    min_confidence: float = 0.1
) -> List[DoubleMutantScore]:
    """
    Score all double mutant combinations from the top N single mutants.

    Parameters
    ----------
    single_scores : list of MutationScore
        All single mutant predictions, sorted by predicted KIE.
    top_n : int
        Number of top singles to combine (produces top_n*(top_n-1)/2 pairs).
    wt_kie : float
        Wild-type KIE baseline.
    beta : float
        Dynamic weight parameter.
    min_confidence : float
        Minimum confidence threshold for inclusion.

    Returns
    -------
    List of DoubleMutantScore sorted by predicted KIE descending.
    """
    # Take top N novel singles, but ensure dynamic mutations are included
    # by adding top 5 dynamic-dominated mutations even if outside top_n
    top_by_kie = [s for s in single_scores
                  if s.is_novel and s.confidence >= min_confidence][:top_n]
    top_dynamic = [s for s in single_scores
                   if s.is_novel and s.dominant_mechanism in ('dynamic','breathing')
                   and s.confidence >= min_confidence][:5]
    seen = {s.label for s in top_by_kie}
    candidates = top_by_kie + [s for s in top_dynamic if s.label not in seen]

    double_scores = []
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            a, b = candidates[i], candidates[j]

            # Skip pairs on the same residue
            if a.residue_number == b.residue_number:
                continue

            dm = score_double_mutant(a, b, wt_kie=wt_kie, beta=beta)
            double_scores.append(dm)

    double_scores.sort(key=lambda x: x.predicted_kie, reverse=True)
    return double_scores


def print_double_mutant_report(
    double_scores: List[DoubleMutantScore],
    wt_kie: float = 11.3,
    top_n: int = 20
):
    """Print a formatted report of double mutant predictions."""
    print(f"\n{'='*70}")
    print(f"  DOUBLE MUTANT PREDICTIONS")
    print(f"{'='*70}")
    print(f"  {len(double_scores)} combinations scored")
    print(f"  WT KIE baseline: {wt_kie:.1f}")

    synergistic = [d for d in double_scores if d.interaction_type == 'synergistic']
    print(f"  Synergistic pairs: {len(synergistic)}")
    print()

    # Sort by fold_vs_best for meaningful ranking
    by_fold = sorted(double_scores, key=lambda x: x.fold_vs_best_single, reverse=True)

    print(f"  {'Double mutant':<24} {'fold>best':>10} {'Δinteract':>10} "
          f"{'type':<16} {'conf':>6}")
    print(f"  {'-'*68}")

    for dm in by_fold[:top_n]:
        print(f"  {dm.label:<24} {dm.fold_vs_best_single:>+9.2f}x "
              f"{dm.interaction_delta:>+9.3f}  "
              f"{dm.interaction_type:<16} {dm.confidence:>5.2f}")

    if synergistic:
        print(f"\n  ★ Synergistic pairs (different mechanisms — highest experimental priority):")
        for dm in sorted(synergistic, key=lambda x: x.fold_vs_best_single, reverse=True)[:10]:
            a_mech = dm.mut_a.dominant_mechanism
            b_mech = dm.mut_b.dominant_mechanism
            print(f"    {dm.label:<28} fold>{dm.fold_vs_best_single:+.2f}x  "
                  f"{a_mech}+{b_mech}  interaction={dm.interaction_delta:+.3f}")
    else:
        print(f"\n  No synergistic pairs found in top {top_n} combinations.")
        print(f"  Top pairs are sub-additive (both static-dominated — diminishing returns)")

    print(f"{'='*70}\n")
