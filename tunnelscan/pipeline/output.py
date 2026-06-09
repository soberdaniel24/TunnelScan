from __future__ import annotations
import json
import math
import os
import numpy as np


def _nan_safe(obj):
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if math.isnan(v) or math.isinf(v) else v
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (list, tuple)):
        return type(obj)(_nan_safe(x) for x in obj)
    if isinstance(obj, dict):
        return {k: _nan_safe(v) for k, v in obj.items()}
    return obj


def save_results(output_dir: str, result, fe_classical, fe_H, fe_D,
                 variance_report, bootstrap_H: dict, bootstrap_D: dict):
    os.makedirs(output_dir, exist_ok=True)

    # results.json
    data = {
        "delta_delta_G": _nan_safe(result.delta_delta_G),
        "tunnelling_factor": _nan_safe(result.tunnelling_factor),
        "kie_theoretical": _nan_safe(result.kie_theoretical),
        "kie_experimental": _nan_safe(result.kie_experimental),
        "kie_agreement": result.kie_agreement,
        "zpe": _nan_safe(result.zpe),
        "classification": result.classification,
        "confidence_interval_95": _nan_safe(list(result.confidence_interval_95)),
        "trajectory_variance": _nan_safe(result.trajectory_variance),
        "deuterium_check_passed": result.deuterium_check_passed,
        "flags": result.flags,
        "bootstrap_H": _nan_safe(bootstrap_H),
        "bootstrap_D": _nan_safe(bootstrap_D),
        "classical_barrier": _nan_safe(fe_classical.barrier_height),
        "centroid_H_barrier": _nan_safe(fe_H.barrier_height),
        "centroid_D_barrier": _nan_safe(fe_D.barrier_height),
        "variance": {
            "classical_std": _nan_safe(variance_report.classical_std),
            "centroid_std": _nan_safe(variance_report.centroid_std),
            "enzyme_type": variance_report.enzyme_type,
            "multi_pathway": variance_report.multi_pathway,
        },
    }
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(data, f, indent=2)

    # results_summary.txt
    from tunnelscan.uncertainty.reporting import format_result
    summary = format_result(result, bootstrap_H, bootstrap_D, variance_report)
    with open(os.path.join(output_dir, "results_summary.txt"), "w") as f:
        f.write(summary)

    # Plots
    try:
        plot_free_energy_profiles(fe_classical, fe_H, output_dir)
    except Exception:
        pass
    try:
        plot_trajectory_variance(variance_report, output_dir)
    except Exception:
        pass

    # tunnelling_report.md
    with open(os.path.join(output_dir, "tunnelling_report.md"), "w") as f:
        f.write(_make_markdown_report(result, fe_classical, fe_H, fe_D,
                                       bootstrap_H, bootstrap_D, variance_report))


def plot_free_energy_profiles(fe_classical, fe_H, output_dir: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(fe_classical.xi_values, fe_classical.free_energy,
            "b-o", label=f"Classical (ΔG‡={fe_classical.barrier_height:.2f})")
    ax.plot(fe_H.xi_values, fe_H.free_energy,
            "r-s", label=f"Centroid H (ΔG‡={fe_H.barrier_height:.2f})")
    ax.set_xlabel("ξ = d(DH) - d(HA) (Å)")
    ax.set_ylabel("Free energy (kcal/mol)")
    ax.set_title("Free energy profiles")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "free_energy_profiles.png"), dpi=150)
    plt.close(fig)


def plot_trajectory_variance(variance_report, output_dir: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    if len(variance_report.classical_barriers) > 0:
        ax.hist(variance_report.classical_barriers, bins="auto", alpha=0.6,
                label=f"Classical (σ={variance_report.classical_std:.2f})", color="blue")
    if len(variance_report.centroid_barriers) > 0:
        ax.hist(variance_report.centroid_barriers, bins="auto", alpha=0.6,
                label=f"Centroid (σ={variance_report.centroid_std:.2f})", color="red")
    ax.set_xlabel("Barrier height (kcal/mol)")
    ax.set_ylabel("Count")
    ax.set_title("Per-trajectory barrier distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "trajectory_variance.png"), dpi=150)
    plt.close(fig)


def _make_markdown_report(result, fe_classical, fe_H, fe_D,
                           bootstrap_H, bootstrap_D, variance_report) -> str:
    ci_lo, ci_hi = result.confidence_interval_95
    lines = [
        "# TunnelScan Tunnelling Report",
        "",
        "## Summary",
        f"- **Classification**: {result.classification}",
        f"- **ΔΔG (classical − H centroid)**: {result.delta_delta_G:.3f} kcal/mol",
        f"- **Tunnelling factor**: {result.tunnelling_factor:.3f}",
        f"- **ZPE**: {result.zpe:.3f} kcal/mol" if not math.isnan(result.zpe) else "- **ZPE**: N/A",
        "",
        "## Barriers",
        f"| Method | ΔG‡ (kcal/mol) |",
        f"|--------|----------------|",
        f"| Classical | {fe_classical.barrier_height:.3f} |",
        f"| Centroid H | {fe_H.barrier_height:.3f} |",
        f"| Centroid D | {fe_D.barrier_height:.3f} |",
        "",
        "## KIE",
        f"- Theoretical KIE: {result.kie_theoretical:.3f}",
    ]
    if result.kie_experimental is not None:
        lines.append(f"- Experimental KIE: {result.kie_experimental:.3f}")
        lines.append(f"- Agreement: {'Yes' if result.kie_agreement else 'No'}")
    lines += [
        "",
        "## Statistics",
        f"- 95% CI (H barrier): [{ci_lo:.3f}, {ci_hi:.3f}] kcal/mol" if not math.isnan(ci_lo) else "- 95% CI: N/A",
        f"- Deuterium check: {'PASSED' if result.deuterium_check_passed else 'FAILED'}",
        f"- Enzyme type: {variance_report.enzyme_type}",
        f"- Multi-pathway: {variance_report.multi_pathway}",
        "",
    ]
    if result.flags:
        lines.append("## Flags")
        for flag in result.flags:
            lines.append(f"- `{flag}`")
    return "\n".join(lines)
