"""Build the compact four-page five-family report after the EFN study."""

import argparse
import csv
from pathlib import Path
from statistics import mean

import reportlab
from build_report import (
    ACCENT,
    CAPTION,
    SMALL,
    Note,
    card_row,
    make_style,
    read_json,
    verify,
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

RUN = Path("results/task0009/20261111-efn-02")
MODELS = ("XGB", "MLP", "PFN-K", "PFN-ID", "EFN")
CODE = make_style("Courier", 7.5, 10.8)


def csv_rows(name):
    with (RUN / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def interval(item, scale=1):
    return (
        f"{item['mean'] * scale:+.2f} "
        f"[{item['bootstrap_95_lo'] * scale:+.2f}, {item['bootstrap_95_hi'] * scale:+.2f}]"
    )


def primary(summary):
    return summary["by_fraction"]["0.5"]["contrasts"]["EFN_minus_PFN-K"]


def overview(note, summary, audit):
    note.page(1, "EFN STUDY DESIGN")
    note.line("PHYSICS CONSTRAINTS AND GENERATOR TRANSFER", 53, 756, "Parallasse-Bold", 8.7, ACCENT)
    note.line("Parallasse", 53, 705, "Parallasse-Bold", 45)
    y = note.paragraph(
        "Energy flow, discrimination and inference bias",
        689,
        make_style("Parallasse", 12.1, 17),
        gap=8,
    )
    y = note.paragraph(
        "Matteo Sgobba | Independent scientific ML study | 12 November 2026", y, SMALL, gap=16
    )
    y = note.paragraph(
        "Does an infrared- and collinear-constrained Energy Flow Network (EFN) improve "
        "quark-fraction inference when a classifier trained on Pythia is used on Herwig? "
        "This bounded comparison measures classification, operating-point transfer and "
        "composition bias on a new common set of events.",
        y,
        gap=14,
    )
    item = summary["by_fraction"]["0.5"]["models"]["EFN"]["absolute_bias"]
    effect = primary(summary)
    y = card_row(
        note,
        [
            (
                f"{summary['classification']['target']['models']['EFN']['auc']['mean']:.4f}",
                "EFN Herwig ROC-AUC",
                "Mean over three seeds",
            ),
            (
                f"{item['mean'] * 100:.2f} pp",
                "EFN absolute fraction bias",
                "True quark fraction 50%",
            ),
            (
                f"{effect['mean'] * 100:+.2f} pp",
                "EFN minus PFN-K bias",
                f"95% [{effect['bootstrap_95_lo'] * 100:+.2f}, "
                f"{effect['bootstrap_95_hi'] * 100:+.2f}]",
            ),
        ],
        y,
    )
    y = note.heading("Energy-weighted learning from particle directions", y)
    y = note.paragraph(
        "The particle encoder consumes centered rapidity and azimuth. Its outputs are "
        "weighted by each constituent's relative transverse momentum z, summed, and passed "
        "to the jet head. Species labels are unused. The shared preprocessing fixes the "
        "pT-weighted and circular angular centers.",
        y,
    )
    y = note.paragraph(
        "score = sigmoid(F(sum_i z_i * Phi(delta_y_i, delta_phi_i)))", y, CODE, gap=14
    )
    y = note.heading("Source-only learning and fresh evaluation", y)
    counts = audit["retained_counts"]
    y = note.table(
        [
            ["Role", "Generator", "Jets", "Use"],
            ["Train", "Pythia", "60,000", "Original source cohort"],
            ["Tune", "Pythia", "10,000", "Source-only stopping"],
            ["Cut calibration", "Pythia", "5,000 gluons", "EFN strict 10% background cut"],
            [
                "Response",
                "Pythia",
                f"{counts['response_quarks'] + counts['response_gluons']:,}",
                "Class acceptance measurement",
            ],
            [
                "Source mixtures",
                "Pythia",
                f"{counts['population_quarks'] + counts['population_gluons']:,}",
                "Disjoint source pseudo-data pool",
            ],
            [
                "Target mixtures",
                "Herwig",
                f"{counts['target_retained']:,}",
                "Fresh transfer evaluation",
            ],
        ],
        y,
        (0.24, 0.16, 0.17, 0.43),
        compact=True,
    )
    note.paragraph(
        "Evaluation uses QG_jets_2.npz and QG_jets_herwig_2.npz. The identity audit covers "
        "400,000 prior and 200,000 new exact constituent records. No record removal was needed. "
        "Both new files have 50,000 quarks and 50,000 gluons. The source half split is "
        "stratified and persisted before prediction.",
        y,
        CAPTION,
    )
    note.pdf.showPage()


def discrimination(note, summary, figures):
    note.page(2, "CLASSIFICATION")
    y = note.title("Five models, one fresh evaluation")
    y = note.paragraph(
        "XGBoost and MLP share six aggregate inputs. PFN-K and PFN-ID have matched particle "
        "architectures, with identity switched off or on. EFN uses the same [32, 32, 64] "
        "particle widths and [64, 32] head widths. It has 9,537 parameters, compared with "
        "9,953 per PFN, and a different energy-dependence constraint.",
        y,
    )
    y = note.figure(
        figures / "06_efn_roc_by_generator.png",
        y,
        "<b>Figure 1.</b> Lines average interpolated per-seed TPR at shared FPR. Legend "
        "AUCs average exact per-seed areas. Every checkpoint sees the same fresh rows.",
    )
    y = note.heading("Held-out discrimination at a glance", y)
    table = [["Model", "Pythia AUC", "Herwig AUC", "AUC loss"]]
    for model in MODELS:
        values = [
            summary["classification"][cohort]["models"][model]["auc"]["mean"]
            for cohort in ("source", "target")
        ]
        gap = summary["classification"]["auc_gap_source_minus_target"][model]["mean"]
        table.append([model, f"{values[0]:.4f}", f"{values[1]:.4f}", f"{gap:.4f}"])
    y = note.table(table, y, (0.22, 0.26, 0.26, 0.26), compact=True)
    a, b = [
        summary["classification"][cohort]["EFN_minus_PFN-K_auc"] for cohort in ("source", "target")
    ]
    y = note.paragraph(
        f"EFN minus PFN-K AUC is {interval(a, 100)} percentage points on Pythia and "
        f"{interval(b, 100)} on Herwig. Brackets are paired 95% event-bootstrap percentiles. "
        "The source and target event streams are independent.",
        y,
        CAPTION,
        gap=12,
    )
    note.paragraph(
        "The selected EFN run uses one three-epoch pilot and full source fits at "
        "seeds 17, 29 and 43. "
        "All neural models use a 30-epoch ceiling, patience 5, batch size 256 and Adam "
        "(learning rate 0.001, weight decay 0.0001). Common-epoch minibatch order matches "
        "the original PFNs. Highest tuning AUC selects the checkpoint, with first ties retained.",
        y,
    )
    note.pdf.showPage()


def inference(note, summary, target_rows, source, figures):
    note.page(3, "COMPOSITION INFERENCE")
    y = note.title("What happens to a measured fraction")
    y = note.paragraph(
        "At each fixed numerical cut, measure Pythia quark and gluon efficiencies e_q and "
        "e_g on 25,000 jets per class. Invert the mixture acceptance r using "
        "<b>f_hat = (r - e_g)/(e_q - e_g)</b>. The primary comparison uses true quark "
        "probability 50% and mixture size 10,000. Estimates and intervals retain "
        "values outside [0, 1].",
        y,
    )
    y = note.figure(
        figures / "07_efn_auc_and_inference_bias.png",
        y,
        "<b>Figure 2.</b> Filled markers and bars show per-seed metric means and marginal "
        "95% intervals at fixed checkpoints and cuts. Open markers are the three individual "
        "training seeds. Absolute bias is formed per seed before averaging.",
    )
    y = note.heading("Primary mixture results", y)
    table = [["Model", "Expected quark %", "Bias (pp)", "RMSE (pp)", "Coverage hits / 1,000"]]
    for model in MODELS:
        bias = summary["by_fraction"]["0.5"]["models"][model]["signed_bias"]["mean"]
        selected = [
            r
            for r in target_rows
            if r["model"] == model
            and float(r["fraction"]) == 0.5
            and int(r["sample_size"]) == 10000
        ]
        rmse = mean(float(r["rmse"]) for r in selected) * 100
        hits = [int(r["coverage_hits"]) for r in selected]
        text = str(min(hits)) if min(hits) == max(hits) else f"{min(hits)}-{max(hits)}"
        table.append(
            [model, f"{(bias + 0.5) * 100:.2f}%", f"{bias * 100:+.2f}", f"{rmse:.2f}", text]
        )
    y = note.table(table, y, (0.15, 0.25, 0.17, 0.17, 0.26), compact=True)
    item = primary(summary)
    y = note.paragraph(
        f"The primary absolute-bias contrast is <b>{interval(item, 100)} percentage points</b>. "
        f"The observed seed contrasts range from {item['seed_min'] * 100:+.2f} to "
        f"{item['seed_max'] * 100:+.2f} points. The interval crosses zero, and the declared "
        "negative-upper-bound improvement criterion is unmet.",
        y,
        gap=12,
    )
    note.paragraph(
        f"Repeated-source coverage spans {source['control_coverage_min'] * 100:.1f}-"
        f"{source['control_coverage_max'] * 100:.1f}% across all 90 checkpoint/cell cases. "
        "Each cell has 1,000 repetitions. The complete 25%, 50%, 75% fraction grid and both "
        "1,000/10,000 mixture sizes are preserved in the results. Count and source-response "
        "precision enter the delta intervals. Bias intervals resample source-response and "
        "target class rows in 500 paired bootstraps.",
        y,
        CAPTION,
    )
    note.pdf.showPage()


def methods(note, fits):
    note.page(4, "CHECKS, SCOPE AND REPRODUCTION")
    y = note.title("Physical checks and scientific limits")
    records = fits["full"]
    checks = [r["invariance"] for r in records]
    y = note.paragraph(
        "The selected EFN checkpoints pass preprocessing-aware tests on 64 source probes per "
        "seed. Splitting a constituent into two collinear parts preserves its total momentum. "
        "An added particle is taken to vanishing relative pT. The largest observed probability "
        "differences across seeds are shown below.",
        y,
    )
    table = [
        ["Check", "Largest probability difference"],
        [
            "Constituent permutation",
            f"{max(c['permutation_max_abs_probability_delta'] for c in checks):.2e}",
        ],
        [
            "Extra arbitrary padding",
            f"{max(c['extra_padding_max_abs_probability_delta'] for c in checks):.2e}",
        ],
        [
            "Collinear 37% / 63% split",
            f"{max(c['collinear_max_abs_probability_delta'] for c in checks):.2e}",
        ],
        [
            "Soft particle, relative pT 1e-8",
            f"{max(c['soft_limit'][-1]['max_abs_probability_delta'] for c in checks):.2e}",
        ],
        ["Constituent PID changes", "0.00e+00"],
    ]
    y = note.table(table, y, (0.59, 0.41), compact=True)
    y = note.paragraph(
        "An initial source-only attempt collapsed at seed 17, with tuning AUC falling to 0.5. "
        "Its pilot, three full fits, predictions and code are archived. Before calibration "
        "or new-file performance, "
        "initialization was changed to He-uniform hidden kernels, zero biases and a Xavier "
        "output kernel. Layer widths, data, optimizer, seeds and budget were retained. "
        "Both attempts together used two EFN pilots and six full fits.",
        y,
        SMALL,
        gap=12,
    )
    y = note.paragraph(
        "Within this compact configuration and budget, the physical constraints coexist with "
        "lower discrimination and an unresolved bias difference relative to PFN-K. The "
        "comparison changes input constraints, parameter count and initialization. The study "
        "uses two particle-level generators in one kinematic window. Exact-record identities "
        "are audited, while original collision IDs are unavailable. Marginal intervals condition "
        "on the selected checkpoints and numerical cuts. Seed variation is reported separately.",
        y,
        SMALL,
        gap=12,
    )
    y = note.heading("Reproduce from the project root", y)
    y = note.paragraph(
        ".venv/bin/python scripts/run_efn_study.py --stage prepare --run-id NEW_ID<br/>"
        ".venv/bin/python scripts/run_efn_study.py --stage fit --run-id NEW_ID<br/>"
        ".venv/bin/python scripts/run_efn_study.py --stage calibrate --run-id NEW_ID<br/>"
        ".venv/bin/python scripts/run_efn_study.py --stage source --run-id NEW_ID<br/>"
        ".venv/bin/python scripts/run_efn_study.py --stage target --run-id NEW_ID<br/>"
        ".venv/bin/python scripts/render_efn_figures.py<br/>"
        "python3 scripts/build_efn_report.py",
        y,
        CODE,
        gap=10,
    )
    y = note.paragraph(
        "Recorded execution 20261111-efn-02. Source snapshots, checkpoints, cuts, audited IDs "
        "and draw archives are indexed by README.md. The PDF builder uses the ReportLab "
        "version in reports/requirements-report.txt.",
        y,
        SMALL,
        gap=12,
    )
    y = note.heading("Primary sources and data attribution", y)
    for reference in (
        "Komiske, Metodiev and Thaler. Energy Flow Networks. "
        '<link href="https://arxiv.org/abs/1810.05165" color="#216181">arXiv:1810.05165</link>.',
        "Bright-Thonney, Nachman and Thaler. Safe but Incalculable. "
        '<link href="https://arxiv.org/abs/2311.07652" color="#216181">arXiv:2311.07652</link>.',
        "Initialization conventions from "
        '<link href="https://energyflow.network/docs/archs/" color="#216181">EnergyFlow</link> '
        'and <link href="https://keras.io/api/layers/core_layers/dense/" '
        'color="#216181">Keras Dense</link>.',
        "Public light-quark samples, CC BY 4.0. "
        '<link href="https://doi.org/10.5281/zenodo.3164691" color="#216181">Pythia v1</link> '
        "(Komiske, Metodiev, Thaler) and "
        '<link href="https://doi.org/10.5281/zenodo.3066475" color="#216181">Herwig v1</link> '
        "(Pathak, Komiske, Metodiev, Schwartz).",
    ):
        y = note.paragraph(reference, y, SMALL, gap=4)
    note.pdf.showPage()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", type=Path, default=Path("reports/figures"))
    parser.add_argument("--output", type=Path, default=Path("output/pdf/parallasse_report_v3.pdf"))
    args = parser.parse_args()
    run = read_json(RUN / "run.json")
    if run["status"] != "completed":
        raise ValueError("The EFN study must be complete before report creation")
    for path, digest in run["output_sha256"].items():
        verify(path, digest)
    summary = read_json(RUN / "summary.json")
    fonts = Path(reportlab.__file__).parent / "fonts"
    pdfmetrics.registerFont(TTFont("Parallasse", str(fonts / "Vera.ttf")))
    pdfmetrics.registerFont(TTFont("Parallasse-Bold", str(fonts / "VeraBd.ttf")))
    pdfmetrics.registerFontFamily("Parallasse", normal="Parallasse", bold="Parallasse-Bold")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    note = Note(args.output)
    note.pdf.setTitle("Parallasse | Energy flow, discrimination and inference bias")
    note.pdf.setSubject(
        "Five-family comparison with EFN on fresh particle-level Pythia and Herwig events"
    )
    overview(note, summary, read_json(RUN / "audit.json"))
    discrimination(note, summary, args.figures)
    inference(
        note,
        summary,
        csv_rows("target_mixtures.csv"),
        read_json(RUN / "source_summary.json"),
        args.figures,
    )
    methods(note, read_json(RUN / "source_fits.json"))
    note.pdf.save()
    print(f"Wrote compact four-page EFN research report to {args.output}")


if __name__ == "__main__":
    main()
