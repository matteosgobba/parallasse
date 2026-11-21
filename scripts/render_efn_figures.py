"""Render the five-family EFN classification and inference comparison."""

import argparse
import csv
from pathlib import Path

from render_figures import export, style

# The shared renderer selects the headless backend before pyplot is imported.
# isort: split
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.metrics import roc_curve

from parallasse.efn_metrics import KEYS, MODELS
from parallasse.heldout import read_json, verify_hashes
from parallasse.mixtures import infer_fraction

RUN = Path("results/task0009/20261111-efn-02")
COLORS = ("#34495b", "#7854a0", "#007c7b", "#ae4c26", "#b22d62")
LINES = ("-", "--", "-.", ":", "-")


def load_evidence():
    run = read_json(RUN / "run.json")
    if run["status"] != "completed":
        raise ValueError("The EFN study must be complete before presentation")
    verify_hashes(run["output_sha256"])
    for name in ("source_scores", "target_scores"):
        ref = run["archives"][name]
        verify_hashes({ref["path"]: ref["sha256"]})
    return run, read_json(RUN / "summary.json")


def roc_figure(run, summary):
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.35), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.095, right=0.968, bottom=0.17, top=0.87, wspace=0.14)
    grid = np.linspace(0, 1, 2001)
    for ax, cohort, label in zip(axes, ("source", "target"), ("Pythia", "Herwig"), strict=True):
        with np.load(run["archives"][f"{cohort}_scores"]["path"]) as data:
            labels, scores = data["labels"], data["scores"]
            assert tuple(data["keys"]) == KEYS
        ax.plot([0, 1], [0, 1], color="#a8b6c3", lw=1, zorder=0)
        for model_index, (model, color, line) in enumerate(zip(MODELS, COLORS, LINES, strict=True)):
            curves, areas = [], []
            for seed in range(3):
                fpr, tpr, _ = roc_curve(labels, scores[:, model_index * 3 + seed])
                curves.append(np.interp(grid, fpr, tpr))
                areas.append(np.trapezoid(tpr, fpr))
            auc = summary["classification"][cohort]["models"][model]["auc"]["mean"]
            np.testing.assert_allclose(auc, np.mean(areas), atol=1e-12, rtol=0)
            ax.plot(
                grid,
                np.mean(curves, axis=0),
                color=color,
                ls=line,
                lw=2.2 if model == "EFN" else 1.5,
                label=f"{model}  {auc:.4f}",
            )
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Gluon acceptance (FPR)")
        ax.set_title(f"{label} | {len(labels):,} fresh jets", loc="left", pad=9)
        ax.grid(True)
        ax.legend(
            loc="lower right",
            title="Mean ROC-AUC",
            fontsize=8.6,
            title_fontsize=8.4,
            frameon=True,
            edgecolor="#dbe5ec",
            facecolor="white",
        )
    axes[0].set_ylabel("Quark acceptance (TPR)")
    return fig


def tradeoff_figure(summary):
    with (RUN / "target_classification.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert tuple(f"{r['model']}-{r['seed']}" for r in rows) == KEYS
    auc_seeds = np.array([float(r["auc"]) for r in rows]).reshape(5, 3)
    source = read_json(RUN / "source_summary.json")
    eq, eg = [
        np.array(source[key]) for key in ("response_quark_acceptance", "response_gluon_acceptance")
    ]
    rates = np.array([0.5 * (float(r["tpr"]) + float(r["fpr"])) for r in rows])
    bias_seeds = np.abs(infer_fraction(rates, eq, eg) - 0.5).reshape(5, 3) * 100
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.15), sharey=True)
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.19, top=0.80, wspace=0.14)
    for panel, (ax, metric, seeds, scale) in enumerate(
        zip(axes, ("auc", "bias"), (auc_seeds, bias_seeds), (1, 100), strict=True)
    ):
        ax.axhspan(3.5, 4.5, color="#f9edf2", zorder=0)
        for i, (model, color) in enumerate(zip(MODELS, COLORS, strict=True)):
            item = (
                summary["classification"]["target"]["models"][model]["auc"]
                if metric == "auc"
                else summary["by_fraction"]["0.5"]["models"][model]["absolute_bias"]
            )
            lo, hi = [item[k] * scale for k in ("bootstrap_95_lo", "bootstrap_95_hi")]
            ax.hlines(i - 0.1, lo, hi, color=color, lw=2)
            ax.plot(item["mean"] * scale, i - 0.1, "o", color=color, markersize=5)
            ax.scatter(
                seeds[i],
                np.full(3, i + 0.16),
                facecolors="white",
                edgecolors=color,
                s=18,
                lw=0.8,
                zorder=3,
            )
        ax.set_yticks(range(5), MODELS)
        ax.set_ylim(4.6, -0.5)
        ax.grid(axis="x")
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel(
            "Herwig ROC-AUC" if panel == 0 else "Absolute fraction bias (percentage points)"
        )
    fig.legend(
        handles=[
            Line2D(
                [],
                [],
                color="#172b3b",
                marker="o",
                lw=1.8,
                label="Seed mean and marginal 95% interval",
            ),
            Line2D(
                [],
                [],
                color="#172b3b",
                marker="o",
                markerfacecolor="white",
                lw=0,
                label="Individual seeds",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.54, 0.995),
        ncols=2,
        frameon=False,
        fontsize=8.7,
    )
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/figures"))
    args = parser.parse_args()
    run, summary = load_evidence()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    style()
    export(roc_figure(run, summary), args.output_dir, "06_efn_roc_by_generator")
    export(tradeoff_figure(summary), args.output_dir, "07_efn_auc_and_inference_bias")
    print("Rendered five-family ROC curves and AUC/inference-bias comparison in PNG and SVG.")


if __name__ == "__main__":
    main()
