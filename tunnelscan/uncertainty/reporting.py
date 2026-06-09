from __future__ import annotations
import math


def format_result(result, bootstrap_H: dict, bootstrap_D: dict,
                  variance_report) -> str:
    ci_lo, ci_hi = result.confidence_interval_95
    n_paths = len(bootstrap_H.get("per_trajectory_barriers", []))

    lines = [
        "=" * 60,
        "TUNNELSCAN RESULTS",
        "=" * 60,
        f"Classification:         {result.classification}",
        f"ΔΔG (classical - H):    {result.delta_delta_G:.3f} kcal/mol",
        f"Tunnelling factor:      {result.tunnelling_factor:.3f}",
        f"ZPE (centroid):         {_fmt(result.zpe)} kcal/mol",
        f"KIE (theoretical):      {result.kie_theoretical:.3f}",
    ]

    if result.kie_experimental is not None:
        lines.append(f"KIE (experimental):     {result.kie_experimental:.3f}")
        lines.append(f"KIE agreement:          {'YES' if result.kie_agreement else 'NO'}")

    lines += [
        "",
        "Bootstrap Statistics (H):",
        f"  Barrier mean:  {bootstrap_H.get('mean', float('nan')):.3f} kcal/mol",
        f"  Barrier std:   {bootstrap_H.get('std', float('nan')):.3f} kcal/mol",
        f"  95% CI:        [{_fmt(ci_lo)}, {_fmt(ci_hi)}] kcal/mol",
        "",
        "Bootstrap Statistics (D):",
        f"  Barrier mean:  {bootstrap_D.get('mean', float('nan')):.3f} kcal/mol",
        f"  Barrier std:   {bootstrap_D.get('std', float('nan')):.3f} kcal/mol",
        "",
        "Variance Analysis:",
        f"  Classical std:    {variance_report.classical_std:.3f} kcal/mol",
        f"  Centroid std:     {variance_report.centroid_std:.3f} kcal/mol",
        f"  Enzyme type:      {variance_report.enzyme_type}",
        f"  Multi-pathway:    {variance_report.multi_pathway}",
        "",
        f"Deuterium check:      {'PASSED' if result.deuterium_check_passed else 'FAILED'}",
        f"N paths:              {n_paths}",
    ]

    if n_paths < 20:
        lines.append("  [RELIABILITY FLAG: n_paths < 20 — results may be unreliable]")

    if result.flags:
        lines.append("")
        lines.append("Flags:")
        for flag in result.flags:
            lines.append(f"  ! {flag}")

    lines.append("=" * 60)
    return "\n".join(lines)


def _fmt(x) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "N/A"
    return f"{x:.3f}"
