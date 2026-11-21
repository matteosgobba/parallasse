"""Render the three report figures from the immutable TASK-0004 artifacts."""

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("work/matplotlib").resolve()))
os.environ["MPLBACKEND"] = "Agg"

import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MultipleLocator, PercentFormatter  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from parallasse.heldout import read_json, verify_hashes  # noqa: E402
from parallasse.heldout_metrics import KEYS, MODELS  # noqa: E402

RUN = Path("results/task0004/20261110-heldout-01")
COHORTS = ("test", "target")
LABELS = ("Pythia", "Herwig")
INK = "#172b3b"
MUTED = "#526779"
GRID = "#e2e9ef"
BLUE = "#216181"
ORANGE = "#ae4c26"
MODEL_STYLES = (
    ("#34495b", "-"),
    ("#7854a0", "--"),
    ("#007c7b", "-."),
    (ORANGE, ":"),
)


def style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": "#bccbd6",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "grid.color": GRID,
            "grid.linewidth": 0.65,
            "axes.axisbelow": True,
            "legend.fontsize": 9,
            "figure.facecolor": "white",
            "svg.fonttype": "none",
            "svg.hashsalt": "parallasse-heldout-20261110",
            "savefig.dpi": 300,
        }
    )


def export(fig, output: Path, name: str) -> None:
    for extension in ("png", "svg"):
        metadata = {"Date": None} if extension == "svg" else {"Software": "Parallasse"}
        fig.savefig(output / f"{name}.{extension}", metadata=metadata)
    plt.close(fig)


def roc_figure(report: dict, archives: dict):
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.45), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.095, right=0.968, bottom=0.16, top=0.87, wspace=0.14)
    grid = np.linspace(0, 1, 2001)
    for ax, cohort, name in zip(axes, COHORTS, LABELS, strict=True):
        ref = archives[cohort]
        with np.load(ref["path"], allow_pickle=False) as data:
            assert tuple(data["keys"]) == KEYS
            y, scores = data["labels"], data["scores"]
        ax.plot([0, 1], [0, 1], color="#a8b6c3", lw=1, zorder=0)
        for i, (model, (color, linestyle)) in enumerate(zip(MODELS, MODEL_STYLES, strict=True)):
            rocs, areas = [], []
            for seed in range(3):
                fpr, tpr, _ = roc_curve(y, scores[:, i * 3 + seed])
                areas.append(np.trapezoid(tpr, fpr))
                rocs.append(np.interp(grid, fpr, tpr))
            auc = report["cohort_metrics"][cohort][model]["auc"]["mean"]
            np.testing.assert_allclose(np.mean(areas), auc, atol=1e-12, rtol=0)
            ax.plot(
                grid,
                np.mean(rocs, axis=0),
                color=color,
                linestyle=linestyle,
                lw=1.8,
                label=f"{model}  {auc:.4f}",
            )
        ax.set(
            xlabel="Gluon acceptance (FPR)",
            xlim=(0, 1),
            ylim=(0, 1),
        )
        ax.set_title(f"{name}  |  {len(y):,} jets", loc="left", pad=9)
        ax.xaxis.set_major_locator(MultipleLocator(0.25))
        ax.yaxis.set_major_locator(MultipleLocator(0.25))
        ax.grid(True)
        ax.legend(
            loc="lower right",
            frameon=True,
            facecolor="white",
            edgecolor=GRID,
            title="Mean ROC-AUC",
            title_fontsize=8.5,
            borderpad=0.65,
        )
    axes[0].set_ylabel("Quark acceptance (TPR)")
    return fig


def acceptance_figure(report: dict):
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 2.58), sharey=True)
    fig.subplots_adjust(left=0.125, right=0.982, bottom=0.23, top=0.75, wspace=0.16)
    handles = [
        Line2D([], [], color=color, marker=marker, lw=1.5, markersize=5, label=name)
        for name, color, marker in zip(LABELS, (BLUE, ORANGE), ("o", "D"), strict=True)
    ]
    handles.append(Line2D([], [], color=MUTED, ls="--", lw=1, label="10% calibration target"))
    fig.legend(
        handles=handles,
        loc="upper center",
        ncols=3,
        frameon=False,
        bbox_to_anchor=(0.53, 0.995),
        columnspacing=1.6,
    )
    for ax, metric, title, limits, tick in zip(
        axes,
        ("fpr", "tpr"),
        ("Gluon acceptance (FPR)", "Quark acceptance (TPR)"),
        ((8, 23), (61, 70)),
        (4, 2),
        strict=True,
    ):
        for row, model in enumerate(MODELS):
            for cohort, offset, color, marker in zip(
                COHORTS,
                (-0.14, 0.14),
                (BLUE, ORANGE),
                ("o", "D"),
                strict=True,
            ):
                item = report["cohort_metrics"][cohort][model][metric]
                mean, lo, hi = (
                    item[key] * 100 for key in ("mean", "bootstrap_95_lo", "bootstrap_95_hi")
                )
                assert limits[0] <= lo <= hi <= limits[1]
                ax.hlines(row + offset, lo, hi, color=color, lw=1.7)
                ax.plot(mean, row + offset, marker, color=color, markersize=4.9)
        ax.set(xlim=limits, ylim=(3.52, -0.52), xlabel=title, yticks=range(len(MODELS)))
        ax.xaxis.set_major_locator(MultipleLocator(tick))
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x")
    axes[0].axvline(10, color=MUTED, ls="--", lw=1, zorder=0)
    axes[0].set_yticks(range(len(MODELS)), MODELS)
    return fig


def pid_figure(report: dict, seed_rows: list[dict]):
    by_key = {
        (row["cohort"], row["model"], int(row["seed"])): float(row["auc"]) for row in seed_rows
    }
    gains = [
        np.array(
            [
                by_key[cohort, "PFN-ID", seed] - by_key[cohort, "PFN-K", seed]
                for seed in (17, 29, 43)
            ]
        )
        for cohort in COHORTS
    ]
    points = [report["auc_contrasts"][cohort]["PFN-ID_minus_PFN-K"] for cohort in COHORTS]
    points.append(report["pid_auc_gain_pythia_minus_herwig"])
    fig, ax = plt.subplots(figsize=(8.2, 2.40))
    fig.subplots_adjust(left=0.19, right=0.982, bottom=0.24, top=0.73)
    for row, (item, seeds, color) in enumerate(
        zip(
            points,
            [*gains, gains[0] - gains[1]],
            (BLUE, ORANGE, "#435669"),
            strict=True,
        )
    ):
        np.testing.assert_allclose(item["mean"], np.mean(seeds), atol=1e-12, rtol=0)
        mean, lo, hi = (item[key] * 100 for key in ("mean", "bootstrap_95_lo", "bootstrap_95_hi"))
        ax.hlines(row - 0.10, lo, hi, color=color, lw=2.1)
        ax.plot(mean, row - 0.10, "o", color=color, markersize=5.2)
        ax.scatter(
            seeds * 100,
            np.full(3, row + 0.16),
            s=20,
            facecolors="none",
            edgecolors=color,
            linewidths=0.9,
        )
    ax.set(xlim=(-0.42, 2.31), ylim=(2.52, -0.48), xlabel="AUC difference (percentage points)")
    ax.set_yticks(range(3), ["Pythia", "Herwig", "Pythia - Herwig"])
    ax.tick_params(axis="y", length=0)
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.grid(axis="x")
    ax.axvline(0, color=MUTED, ls="--", lw=1, zorder=0)
    ax.axhline(1.50, color=GRID, lw=0.8, zorder=0)
    fig.legend(
        handles=[
            Line2D(
                [], [], color=INK, marker="o", markersize=5, lw=1.8, label="Mean and 95% interval"
            ),
            Line2D(
                [],
                [],
                color=INK,
                marker="o",
                markersize=4.5,
                markerfacecolor="white",
                lw=0,
                label="Individual seeds",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.56, 0.995),
        frameon=False,
        ncols=2,
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/figures"))
    args = parser.parse_args()
    validation = read_json(Path("results/task0004/validation.json"))
    verify_hashes(validation["evidence_sha256"])
    run, report = (read_json(RUN / name) for name in ("run.json", "summary.json"))
    if run["status"] != "completed" or report["score_artifacts"] != run["score_artifacts"]:
        raise ValueError("Expected the completed, matching evaluation evidence")
    with (RUN / "per_seed.csv").open(newline="") as handle:
        seed_rows = list(csv.DictReader(handle))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    style()
    export(roc_figure(report, run["score_artifacts"]), args.output_dir, "01_roc_by_generator")
    export(acceptance_figure(report), args.output_dir, "02_frozen_cut_acceptance")
    export(pid_figure(report, seed_rows), args.output_dir, "03_particle_identity_gain")
    print("Rendered three figures in PNG and SVG from verified held-out evidence.")


if __name__ == "__main__":
    main()
