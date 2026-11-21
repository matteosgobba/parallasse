"""Render the exploratory-versus-replication inference comparisons."""

import argparse
from pathlib import Path

from render_figures import BLUE, ORANGE, export, style

# The shared renderer selects the headless backend before pyplot is imported.
# isort: split

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from parallasse.heldout import read_json, verify_hashes
from parallasse.heldout_metrics import MODELS

OLD = Path("results/task0007/20261111-mixture-transfer-01")
NEW = Path("results/task0008/20261111-replication-01")


def load_summaries():
    previous = read_json(Path("results/task0007/validation.json"))
    verify_hashes(
        {str(OLD / "summary.json"): previous["evidence_sha256"][str(OLD / "summary.json")]}
    )
    run = read_json(NEW / "run.json")
    if run["status"] != "completed":
        raise ValueError("Replication must be complete before presentation")
    verify_hashes(run["output_sha256"])
    return read_json(OLD / "summary.json"), read_json(NEW / "summary.json")


def bias_figure(reports):
    fig, ax = plt.subplots(figsize=(8.2, 2.75))
    fig.subplots_adjust(left=0.15, right=0.98, bottom=0.23, top=0.77)
    for report, offset, color, marker in zip(
        reports, (-0.12, 0.12), (BLUE, ORANGE), ("o", "D"), strict=True
    ):
        items = report["by_fraction"]["0.5"]["models"]
        for i, model in enumerate(MODELS):
            row = items[model]
            lo, hi = [
                row[k] * 100
                for k in ("absolute_bias_bootstrap_95_lo", "absolute_bias_bootstrap_95_hi")
            ]
            ax.hlines(i + offset, lo, hi, color=color, lw=1.9)
            ax.plot(row["mean_absolute_bias"] * 100, i + offset, marker, color=color, markersize=5)
    ax.set_yticks(range(4), MODELS)
    ax.set(ylim=(3.5, -0.5), xlabel="Absolute quark-fraction bias (percentage points)")
    ax.grid(axis="x")
    ax.tick_params(axis="y", length=0)
    ax.legend(
        handles=[
            Line2D([], [], color=c, marker=m, label=n)
            for c, m, n in ((BLUE, "o", "Exploratory"), (ORANGE, "D", "New-event replication"))
        ],
        loc="upper center",
        bbox_to_anchor=(0.53, 1.42),
        ncols=2,
        frameon=False,
    )
    return fig


def pid_figure(reports):
    fig, ax = plt.subplots(figsize=(8.2, 2.55))
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.25, top=0.78)
    limits = []
    for report, offset, color, marker in zip(
        reports, (-0.12, 0.12), (BLUE, ORANGE), ("o", "D"), strict=True
    ):
        for i, fraction in enumerate((0.25, 0.5, 0.75)):
            row = report["by_fraction"][str(fraction)]["contrasts"]["PFN-ID_minus_PFN-K"]
            lo, hi = [row[k] * 100 for k in ("bootstrap_95_lo", "bootstrap_95_hi")]
            limits.extend((lo, hi))
            ax.hlines(i + offset, lo, hi, color=color, lw=1.9)
            ax.plot(row["mean"] * 100, i + offset, marker, color=color, markersize=5)
    ax.axvline(0, color="#718393", ls="--", lw=1)
    ax.axhspan(0.6, 1.4, color="#edf4f8", zorder=0)
    ax.set_yticks(range(3), ["25% quarks", "50% (primary)", "75% quarks"])
    ax.set(
        ylim=(2.5, -0.5),
        xlim=(min(0, min(limits)) - 0.4, max(limits) + 0.4),
        xlabel="PFN-ID minus PFN-K absolute bias (percentage points)",
    )
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x")
    ax.legend(
        handles=[
            Line2D([], [], color=c, marker=m, label=n)
            for c, m, n in ((BLUE, "o", "Exploratory"), (ORANGE, "D", "New-event replication"))
        ],
        loc="upper center",
        bbox_to_anchor=(0.53, 1.42),
        ncols=2,
        frameon=False,
    )
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/figures"))
    args = parser.parse_args()
    reports = load_summaries()
    style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    export(bias_figure(reports), args.output_dir, "04_fraction_bias_replication")
    export(pid_figure(reports), args.output_dir, "05_pid_bias_replication")
    print("Rendered inference bias and paired PID contrasts in PNG and SVG.")


if __name__ == "__main__":
    main()
