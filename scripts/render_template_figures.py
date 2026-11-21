"""Render primary results and all 216 conditional coverage cells from verified tables."""

import argparse
import json
from pathlib import Path

from render_figures import export, style

# The shared renderer selects Agg and a project-local cache before pyplot import.
# isort: split
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from parallasse.template_presentation import (
    KEYS,
    LABELS,
    METHODS,
    PRESENTATION,
    load_evidence,
    presentation_data,
    sha256,
)

COLORS = ("#526779", "#b16a32", "#007c7b")
MARKERS = ("s", "^", "o")


def primary_figure(data):
    fig, axes = plt.subplots(1, 3, figsize=(8.8, 3.7), sharey=True)
    fig.subplots_adjust(left=0.15, right=0.98, top=0.77, bottom=0.20, wspace=0.27)
    rows = {(r["key"], r["method"]): r for r in data["primary"]}
    for ax in axes:
        ax.axhspan(2.5, 5.5, color="#f0f5f7", zorder=0)
        ax.set_ylim(5.55, -0.55)
        ax.set_yticks(range(6), KEYS, fontsize=9.7)
        ax.grid(axis="x")
        ax.tick_params(axis="y", length=0)
    for j, (method, color, marker) in enumerate(zip(METHODS, COLORS, MARKERS, strict=True)):
        values = [rows[key, method] for key in KEYS]
        y = np.arange(6) + (j - 1) * 0.22
        coverage = np.array([r["coverage"] for r in values]) * 100
        bounds = np.array([r["coverage_mc95"] for r in values]) * 100
        axes[0].errorbar(
            coverage,
            y,
            xerr=np.array([coverage - bounds[:, 0], bounds[:, 1] - coverage]),
            fmt=marker,
            color=color,
            ms=4.5,
            lw=1.1,
            capsize=2,
        )
        axes[1].scatter([r["bias"] * 100 for r in values], y, c=color, marker=marker, s=21)
        axes[2].scatter([r["median_span"] * 100 for r in values], y, c=color, marker=marker, s=21)
    axes[0].axvline(93, color="#b14d3b", ls="--", lw=1, zorder=1)
    axes[0].set(xlim=(-3, 103), xticks=[0, 25, 50, 75, 100], xlabel="Coverage (%)")
    axes[1].axvline(0, color="#afbac2", lw=0.8)
    axes[1].set(xlim=(-0.7, 21), xticks=[0, 5, 10, 15, 20], xlabel="Signed bias (pp)")
    axes[2].set(xlim=(3.2, 5.9), xticks=[3.5, 4.5, 5.5], xlabel="Median span (pp)")
    for ax, title, subtitle in zip(
        axes,
        ("Truth containment", "Mean estimation error", "Interval usefulness"),
        ("Dashed: 93% primary floor", "Points: 1000-draw means", "Primary ceiling: 25 pp"),
        strict=True,
    ):
        ax.set_title(title, fontsize=10, pad=24)
        ax.text(
            0.5, 1.045, subtitle, transform=ax.transAxes, ha="center", fontsize=8.1, color="#526779"
        )
    fig.legend(
        [Line2D([], [], color=c, marker=m, lw=0) for c, m in zip(COLORS, MARKERS, strict=True)],
        [LABELS[m] for m in METHODS],
        loc="upper center",
        ncols=3,
        frameon=False,
        bbox_to_anchor=(0.55, 1.015),
        fontsize=9.5,
    )
    return fig


def coverage_grid(data):
    fig, axes = plt.subplots(2, 3, figsize=(8.8, 5.7))
    fig.subplots_adjust(left=0.13, right=0.89, top=0.86, bottom=0.13, hspace=0.25, wspace=0.14)
    labels = [
        f"{f:.2f}\n{n // 1000}k"
        for n, f in [
            (1000, 0.25),
            (1000, 0.5),
            (1000, 0.75),
            (10000, 0.25),
            (10000, 0.5),
            (10000, 0.75),
        ]
    ]
    for row, generator in enumerate(("pythia", "herwig")):
        for col, method in enumerate(METHODS):
            ax = axes[row, col]
            values = np.array(data["coverage_grids"][generator][method]) * 100
            mesh = ax.imshow(values, vmin=0, vmax=100, cmap="cividis", aspect="auto")
            ax.set_yticks(range(6), KEYS if col == 0 else [], fontsize=9.3)
            ax.set_xticks(range(6), labels if row == 1 else [], fontsize=9.2)
            ax.tick_params(length=0)
            ax.axvline(2.5, color="white", lw=1.4)
            if col == 0:
                ax.text(
                    -0.47,
                    0.5,
                    generator.title(),
                    transform=ax.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=11,
                    fontweight="bold",
                )
            if row == 0:
                ax.set_title(LABELS[method], fontsize=10.5, pad=10)
            for i in range(6):
                for j in range(6):
                    color = "white" if values[i, j] < 48 else "#172b3b"
                    ax.text(
                        j,
                        i,
                        f"{values[i, j]:.1f}",
                        ha="center",
                        va="center",
                        color=color,
                        fontsize=8.8,
                    )
                    if values[i, j] < 93:
                        ax.add_patch(
                            Rectangle(
                                (j - 0.48, i - 0.48),
                                0.96,
                                0.96,
                                fill=False,
                                edgecolor="#d9625d",
                                lw=1.1,
                            )
                        )
            if row == 1:
                ax.add_patch(Rectangle((3.5, -0.5), 1, 6, fill=False, edgecolor="#2baeb8", lw=2))
    cax = fig.add_axes((0.91, 0.17, 0.015, 0.65))
    bar = fig.colorbar(mesh, cax=cax)
    bar.set_label("Conditional coverage (%)", fontsize=9, labelpad=5)
    bar.ax.tick_params(labelsize=9)
    fig.text(
        0.51,
        0.035,
        "Column labels: true f / mixture N; 1000 attempts per cell",
        ha="center",
        fontsize=9.2,
        color="#526779",
    )
    fig.text(
        0.51,
        0.95,
        "Red border: observed coverage <93%; cyan column: original Herwig primary",
        ha="center",
        fontsize=9.2,
        color="#526779",
    )
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/figures"))
    args = parser.parse_args()
    _, audit = load_evidence()
    data = presentation_data(audit)
    style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    export(primary_figure(data), args.output_dir, "08_template_primary")
    export(coverage_grid(data), args.output_dir, "09_template_coverage_grid")
    (PRESENTATION / "display_data.json").write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n"
    )
    paths = [
        args.output_dir / f"{name}.{extension}"
        for name in ("08_template_primary", "09_template_coverage_grid")
        for extension in ("png", "svg")
    ]
    (PRESENTATION / "figure_sha256.json").write_text(
        json.dumps({str(p): sha256(p) for p in paths}, indent=2, sort_keys=True) + "\n"
    )
    print("Rendered primary coverage/bias/span and every conditional coverage cell in PNG/SVG.")


if __name__ == "__main__":
    main()
