"""Assemble the original research note and a two-page replication addendum."""

import argparse
import csv
import tempfile
from pathlib import Path
from statistics import mean

import reportlab
from build_report import (
    ACCENT,
    CAPTION,
    GRID,
    MARGIN,
    MUTED,
    SMALL,
    WIDTH,
    Note,
    make_style,
    read_json,
    verify,
)
from pypdf import PdfReader, PdfWriter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

OLD = Path("results/task0007/20261111-mixture-transfer-01")
NEW = Path("results/task0008/20261111-replication-01")
CORE = Path("output/pdf/parallasse_report.pdf")
MODELS = ("XGB", "MLP", "PFN-K", "PFN-ID")
CODE = make_style("Courier", 7.2, 10.8)


class Addendum(Note):
    def page(self, number, descriptor):
        self.pdf.setStrokeColor(GRID)
        self.pdf.setLineWidth(0.7)
        self.pdf.line(MARGIN, 787, WIDTH - MARGIN, 787)
        self.pdf.line(MARGIN, 49, WIDTH - MARGIN, 49)
        self.line("PARALLASSE  /  REPLICATION ADDENDUM", MARGIN, 801, "Parallasse-Bold", 8, ACCENT)
        self.pdf.setFont("Parallasse", 8)
        self.pdf.setFillColor(MUTED)
        self.pdf.drawRightString(WIDTH - MARGIN, 801, descriptor)
        self.line("MATTEO SGOBBA  |  11 NOVEMBER 2026", MARGIN, 34, size=7.8, color=MUTED)
        self.pdf.drawRightString(WIDTH - MARGIN, 34, f"{number:02d} / 02")


def comparison_text(contrast):
    return (
        f"{contrast['mean'] * 100:+.2f} "
        f"[{contrast['bootstrap_95_lo'] * 100:+.2f}, {contrast['bootstrap_95_hi'] * 100:+.2f}]"
    )


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def coverage_range(rows, mode, primary=False):
    selected = [
        float(r["coverage_95"]) * 100
        for r in rows
        if r["mode"] == mode
        and (not primary or (float(r["fraction"]) == 0.5 and int(r["sample_size"]) == 10000))
    ]
    if min(selected) == max(selected):
        return f"{min(selected):.1f}%"
    return f"{min(selected):.1f}-{max(selected):.1f}%"


def inference_page(note, old, new, audit, classification, figures):
    note.page(1, "MIXTURE INFERENCE")
    y = note.title("From tagging to composition")
    y = note.paragraph(
        "A tagger can discriminate quarks from gluons while biasing a composition estimate. "
        "The extension tests how constituent identity (PID) affects that measurement under "
        "a change from Pythia to Herwig.",
        y,
        gap=12,
    )
    y = note.paragraph(
        "At each original strict score cut, measure source quark and gluon acceptances "
        "<b>e_q</b> and <b>e_g</b>. From the accepted fraction <b>r</b> in a mixture, infer its "
        "quark probability using the inverse below. Pseudoexperiment class counts fluctuate "
        "binomially. Point estimates retain values outside [0, 1].",
        y,
    )
    y = note.paragraph("f_hat = (r - e_g) / (e_q - e_g)", y, CODE, gap=14)
    y = note.heading("True quark fraction 50%, mixture size 10,000", y)
    table = [["Model", "Exploratory estimate", "Replication estimate", "Replica Herwig AUC"]]
    for model in MODELS:
        a, b = [r["by_fraction"]["0.5"]["models"][model] for r in (old, new)]
        auc = mean(float(row["auc"]) for row in classification if row["model"] == model)
        table.append(
            [
                model,
                f"{(a['mean_signed_bias'] + 0.5) * 100:.2f}%",
                f"{(b['mean_signed_bias'] + 0.5) * 100:.2f}%",
                f"{auc:.4f}",
            ]
        )
    y = note.table(table, y, (0.16, 0.28, 0.28, 0.28), compact=True)
    y = note.figure(
        figures / "04_fraction_bias_replication.png",
        y,
        "<b>Figure 4.</b> Absolute expected fraction bias, averaged over three per-seed biases. "
        "Bars are descriptive 95% paired class-bootstrap intervals. All four models and "
        "the predeclared primary mixture are shown.",
    )
    counts = audit["retained_counts"]
    total = sum(d["rows"] for d in audit["datasets"].values())
    y = note.heading("New events with audited identities", y)
    y = note.paragraph(
        f"The replica uses QG_jets_1.npz and QG_jets_herwig_1.npz from the same public "
        f"light-quark releases. An exact constituent-identity audit covered {total:,} old and "
        f"new records. New source removals were {counts['source_removed']:,} and new target "
        f"removals were {counts['target_removed']:,}. File and row identifiers are persisted.",
        y,
    )
    note.paragraph(
        f"The new source response and source pseudo-data groups each contain "
        f"{counts['response_quarks']:,} quarks and {counts['response_gluons']:,} gluons. "
        f"The target has {counts['target_quarks']:,} quarks and {counts['target_gluons']:,} "
        "gluons. The exploratory response used 5,000 per class. The larger replica response "
        "improves precision. The twelve checkpoints and original numerical cuts "
        "define both studies.",
        y,
        CAPTION,
    )
    note.pdf.showPage()


def interpretation_page(note, old, new, source_rows, target_rows, figures):
    note.page(2, "REPLICATION AND UNCERTAINTY")
    y = note.title("What the new events show")
    a, b = [r["by_fraction"]["0.5"]["contrasts"]["PFN-ID_minus_PFN-K"] for r in (old, new)]
    outcome = "met" if new["replication_criterion_met"] else "not met"
    y = note.paragraph(
        f"The primary PID-minus-kinematics absolute-bias contrast is "
        f"<b>{comparison_text(a)}</b> percentage points in exploration and "
        f"<b>{comparison_text(b)}</b> in replication. The predeclared positive-contrast "
        f"criterion is <b>{outcome}</b>. Replica seed contrasts range from "
        f"{b['seed_min'] * 100:+.2f} to {b['seed_max'] * 100:+.2f} points.",
        y,
    )
    y = note.figure(
        figures / "05_pid_bias_replication.png",
        y,
        "<b>Figure 5.</b> Positive values mean more absolute inference bias with PID. "
        "The shaded row is the primary 50% mixture. Other fractions are secondary. "
        "Expected bias at fixed source response is independent of mixture size.",
    )
    old_rows = read_csv(OLD / "source_metrics.csv") + read_csv(OLD / "target_metrics.csv")
    all_new = source_rows + target_rows
    y = note.heading("Coverage with its sampling design", y)
    y = note.table(
        [
            ["Design", "Exploratory", "Replication"],
            [
                "Repeated source, full grid",
                coverage_range(old_rows, "source_repeated_response"),
                coverage_range(all_new, "source_repeated_response"),
            ],
            [
                "Fixed source, primary",
                coverage_range(old_rows, "pythia_fixed_response", True),
                coverage_range(all_new, "pythia_fixed_response", True),
            ],
            [
                "Fixed Herwig, primary",
                coverage_range(old_rows, "herwig_fixed_response", True),
                coverage_range(all_new, "herwig_fixed_response", True),
            ],
        ],
        y,
        (0.48, 0.26, 0.26),
        compact=True,
    )
    y = note.paragraph(
        "The 95% delta intervals propagate independent binomial uncertainty from mixture "
        "acceptance and both source class efficiencies. Repeated-source controls redraw both "
        "measurements from a common empirical law. Fixed-response coverage conditions on the "
        "actual measured source response. Each cell has 1,000 paired repetitions. Bias intervals "
        "use 500 class-row bootstraps. Source and target draws are independent, "
        "and model/seed columns are paired.",
        y,
        SMALL,
        gap=10,
    )
    y = note.paragraph(
        "Replication used new records at the same particle-level generator settings and "
        "kinematic selection. Identity checks cover exact constituent records. Original collision "
        "identifiers are unavailable. The uncertainty analysis conditions on trained checkpoints "
        "and numerical cuts. Finite source offsets contribute to conditional bias and coverage.",
        y,
        SMALL,
        gap=12,
    )
    y = note.heading("Reproduce from the project root", y)
    y = note.paragraph(
        ".venv/bin/python scripts/replicate_mixture_inference.py --stage audit "
        "--run-id NEW_ID<br/>"
        ".venv/bin/python scripts/replicate_mixture_inference.py --stage source "
        "--run-id NEW_ID<br/>"
        ".venv/bin/python scripts/replicate_mixture_inference.py --stage target "
        "--run-id NEW_ID<br/>"
        ".venv/bin/python scripts/render_replication_figures.py<br/>"
        "python3 scripts/build_replication_report.py",
        y,
        CODE,
        gap=10,
    )
    note.paragraph(
        "The executed run is 20261111-replication-01. Manifests, row IDs, immutable code snapshots "
        "and SHA-256 values are indexed by README.md. PDF dependencies are pinned in "
        "reports/requirements-replication.txt. Dataset attributions and methodological references "
        "are on core-study page 4. New files belong to "
        '<link href="https://doi.org/10.5281/zenodo.3164691" color="#216181">Pythia v1</link> '
        'and <link href="https://doi.org/10.5281/zenodo.3066475" color="#216181">Herwig v1</link>.',
        y,
        SMALL,
    )
    note.pdf.showPage()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", type=Path, default=Path("reports/figures"))
    parser.add_argument("--output", type=Path, default=Path("output/pdf/parallasse_report_v2.pdf"))
    args = parser.parse_args()
    original = read_json("results/task0005/validation.json")["artifact_sha256"]
    verify(CORE, original[str(CORE)])
    evidence = read_json("results/task0007/validation.json")["evidence_sha256"]
    verify(OLD / "summary.json", evidence[str(OLD / "summary.json")])
    for name in ("source_metrics.csv", "target_metrics.csv"):
        verify(OLD / name, evidence[str(OLD / name)])
    run = read_json(NEW / "run.json")
    if run["status"] != "completed":
        raise ValueError("Replication must be complete before report assembly")
    for path, digest in run["output_sha256"].items():
        verify(path, digest)
    old, new = read_json(OLD / "summary.json"), read_json(NEW / "summary.json")
    fonts = Path(reportlab.__file__).parent / "fonts"
    pdfmetrics.registerFont(TTFont("Parallasse", str(fonts / "Vera.ttf")))
    pdfmetrics.registerFont(TTFont("Parallasse-Bold", str(fonts / "VeraBd.ttf")))
    pdfmetrics.registerFontFamily("Parallasse", normal="Parallasse", bold="Parallasse-Bold")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="replication-report-") as directory:
        fragment = Path(directory) / "addendum.pdf"
        note = Addendum(fragment)
        inference_page(
            note,
            old,
            new,
            read_json(NEW / "audit.json"),
            read_csv(NEW / "target_classification.csv"),
            args.figures,
        )
        interpretation_page(
            note,
            old,
            new,
            read_csv(NEW / "source_metrics.csv"),
            read_csv(NEW / "target_metrics.csv"),
            args.figures,
        )
        note.pdf.save()
        writer = PdfWriter()
        writer.append(PdfReader(CORE), outline_item="Core classification study, 10 November 2026")
        writer.append(
            PdfReader(fragment), outline_item="Mixture inference and replication, 11 November 2026"
        )
        writer.add_metadata(
            {
                "/Title": "Parallasse | Classification and mixture-inference replication",
                "/Author": "Matteo Sgobba",
                "/Subject": (
                    "Core study with independent-event replication addendum, 11 November 2026"
                ),
            }
        )
        with args.output.open("wb") as handle:
            writer.write(handle)
    print(f"Wrote six-page research note with replication addendum to {args.output}")


if __name__ == "__main__":
    main()
