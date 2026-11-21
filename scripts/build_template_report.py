"""Build the four-page score-template research note with ReportLab 4.4.9."""

import argparse
import sys
from pathlib import Path

import reportlab
from build_report import ACCENT, CAPTION, SMALL, Note, card_row, make_style
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from parallasse.template_presentation import (  # noqa: E402
    KEYS,
    PRESENTATION,
    SIMULATION,
    load_evidence,
    presentation_data,
    read_json,
    sha256,
)

CODE = make_style("Courier", 7.4, 10.8)
COMPACT = make_style("Parallasse", 9.8, 13.4)


def overview(note, data):
    note.page(1, "SCORE-TEMPLATE INFERENCE")
    note.line(
        "COMPOSITION, SIMULATION SENSITIVITY AND COVERAGE", 53, 756, "Parallasse-Bold", 8.7, ACCENT
    )
    note.line("Parallasse", 53, 705, "Parallasse-Bold", 45)
    y = note.paragraph(
        "Score-template inference under simulation variation",
        689,
        make_style("Parallasse", 12.1, 17),
        gap=8,
    )
    y = note.paragraph(
        "Matteo Sgobba | Independent scientific ML study | 16 November 2026", y, SMALL, gap=14
    )
    y = note.paragraph(
        "Can a classifier's full score distribution give useful quark-fraction intervals "
        "when the simulation changes? Six frozen source-trained PFNs compare a cut estimator, "
        "source-only score templates and a model with explicit Pythia/Herwig variation.",
        y,
        gap=14,
    )
    primary = [r for r in data["primary"] if r["method"] == SIMULATION]
    coverage = [r["coverage"] * 100 for r in primary]
    spans = [r["median_span"] * 100 for r in primary]
    y = card_row(
        note,
        [
            (
                f"{min(coverage):.1f}-{max(coverage):.1f}%",
                "Herwig primary coverage",
                "1000 attempts per checkpoint",
            ),
            (
                f"{min(spans):.2f}-{max(spans):.2f} pp",
                "Median interval span",
                "Primary ceiling: 25 pp",
            ),
            ("90.9 / 91.0%", "Secondary minima", "Pythia / Herwig coverage"),
        ],
        y,
    )
    y = note.heading("Three procedures, one frozen score bank", y)
    y = note.paragraph(
        "Cut inversion retains f_hat = (r - e_g)/(e_q - e_g) and its unbounded delta interval. "
        "The source-only profile uses eight fixed score bins and Pythia class shapes. "
        "The simulation-aware profile interpolates both class shapes with one free "
        "alpha in [0,1]; f is bounded to [0,1]. Four normalized finite class templates "
        "enter as independent fixed-total multinomial auxiliary measurements.",
        y,
        COMPACT,
    )
    y = note.paragraph(
        "p_b(f, alpha) = f*q_b(alpha) + (1-f)*g_b(alpha)<br/>"
        "q(alpha) = (1-alpha)*q_P + alpha*q_H; likewise for g",
        y,
        CODE,
        gap=12,
    )
    y = note.heading("Disjoint records before final prediction", y)
    y = note.table(
        [
            ["Generator", "Template role", "Evaluation role", "Purpose"],
            ["Pythia", "25k q + 25k g", "25k q + 25k g", "Source response / endpoint"],
            ["Herwig", "25k q + 25k g", "25k q + 25k g", "Modelling / endpoint"],
        ],
        y,
        (0.18, 0.23, 0.23, 0.36),
        compact=True,
    )
    y = note.paragraph(
        "Public light-quark files QG_jets_3.npz / QG_jets_herwig_3.npz: particle-level "
        "jets at pT 500-550 GeV, |y| &lt;1.7. An 800k-record exact-identity audit retained "
        "all 200k new records with no overlap/removals. Original collision IDs are unavailable.",
        y,
        CAPTION,
        gap=12,
    )
    y = note.heading("Predeclared decision and estimand", y)
    note.paragraph(
        "Primary: Herwig, generating quark probability f=0.50, N=10,000. Every PFN-K/PFN-ID "
        "checkpoint at seeds 17, 29, 43 must have observed coverage &gt;=93%, failures &lt;=1%, "
        "median outer span &lt;=0.25 and a positive lower paired descriptive coverage-contrast "
        "endpoint versus cut inversion. Class counts are binomial, not fixed to realized f. "
        "No classifier retraining, target-derived binning or cut recalibration occurs.",
        y,
        SMALL,
    )
    note.pdf.showPage()


def primary_page(note, audit, data, figures):
    note.page(2, "PRIMARY HERWIG RESULT")
    y = note.title("Coverage recovered within the model family")
    y = note.paragraph(
        "At f=0.50 and N=10,000, all six simulation-aware checkpoints meet the frozen "
        "criterion. Both source-only comparators have zero observed coverage hits in "
        "1000 attempts per checkpoint; this is not proof of zero population coverage.",
        y,
        COMPACT,
    )
    y = note.figure(
        figures / "08_template_primary.png",
        y,
        "<b>Figure 1.</b> Each row is a separate checkpoint. Left: observed coverage and "
        "per-cell Wilson MC95 bars. Middle/right: signed mean bias and median outer span, "
        "without marginal error bars. All values use 1000 successful draws; no seed ensemble. "
        "The 25 pp usefulness ceiling is stated above the span panel, outside its zoomed axis.",
    )
    y = note.heading("Simulation-aware estimates and intervals", y)
    table = [["Checkpoint", "Hits / 1000", "MC95 (%)", "Med. span pp", "Bias pp", "RMSE pp"]]
    for key in KEYS:
        r = next(r for r in data["primary"] if r["key"] == key and r["method"] == SIMULATION)
        lo, hi = r["coverage_mc95"]
        table.append(
            [
                key,
                str(r["coverage_hits"]),
                f"{100 * lo:.2f}-{100 * hi:.2f}",
                f"{100 * r['median_span']:.3f}",
                f"{100 * r['bias']:+.3f}",
                f"{100 * r['rmse']:.3f}",
            ]
        )
    y = note.table(table, y, (0.22, 0.16, 0.23, 0.17, 0.11, 0.11), compact=True)
    contrasts = [
        r["contrasts"][SIMULATION]
        for r in audit["contrasts"]
        if r["generator"] == "herwig" and r["sample_size"] == 10000 and r["fraction"] == 0.5
    ]
    low = min(c["paired_mc95"][0] for c in contrasts) * 100
    y = note.paragraph(
        f"Coverage gains are +96.5 to +97.7 pp; every paired descriptive MC95 lower endpoint "
        f"is positive (smallest +{low:.1f} pp). The 2000 paired resamples use the same "
        "pseudoexperiment indices across methods/checkpoints. They describe conditional "
        "Monte Carlo precision, not retraining uncertainty or simultaneous significance.",
        y,
        CAPTION,
        gap=12,
    )
    note.paragraph(
        "Cut bias is +5.55 to +10.36 pp and source-template bias +10.37 to +19.09 pp. "
        "The simulation-aware bias remains +0.55 to +0.81 pp: it is not unbiased. Its "
        "5.16-5.47 pp median spans are wider than cut intervals (3.62-3.85 pp). Herwig "
        "templates are modelling inputs; this gain does not demonstrate transfer to an "
        "unseen generator or a better classifier.",
        y,
        COMPACT,
    )
    note.pdf.showPage()


def conditional_page(note, data, figures):
    note.page(3, "COMPLETE GRID AND LIMITATIONS")
    y = note.title("Conditional coverage is not uniform")
    y = note.paragraph(
        "All 216 method/generator/fraction/checkpoint/N cells are shown below. Every cell "
        "has 1000 attempts. The secondary failures remain part of the result, alongside "
        "the successful original primary comparison.",
        y,
        COMPACT,
    )
    y = note.figure(
        figures / "09_template_coverage_grid.png",
        y,
        "<b>Figure 2.</b> Fixed-template, disjoint-pool conditional coverage (%), with no "
        "seed pooling. The red border is a descriptive &lt;93% reference, not a simultaneous "
        "cell-selection test. Cyan marks the original Herwig primary column for all methods. "
        "Full per-cell Wilson intervals, bias, RMSE and span tails are in the linked evidence.",
    )
    y = note.heading("Two low secondary cells, retained without retuning", y)
    rows = [["Endpoint / checkpoint", "f / N", "Hits / 1k", "MC95 (%)", "Span pp"]]
    for r in data["secondary_minima"]:
        lo, hi = r["coverage_mc95"]
        rows.append(
            [
                f"{r['generator'].title()} / {r['key']}",
                f"{r['fraction']:.2f} / 10k",
                str(r["coverage_hits"]),
                f"{100 * lo:.2f}-{100 * hi:.2f}",
                f"{100 * r['median_span']:.3f}",
            ]
        )
    y = note.table(rows, y, (0.34, 0.16, 0.13, 0.23, 0.14), compact=True)
    y = note.paragraph(
        "The Pythia cell has bias -1.352 pp; the Herwig cell +1.164 pp. These findings "
        "do not replace the primary cell, and no cutoff, seed or method was changed after "
        "opening final outcomes. At N=10,000, source-only template coverage on Herwig is "
        "zero observed hits in all 18 cells despite narrow intervals.",
        y,
        SMALL,
    )
    note.paragraph(
        "Common-law controls redraw auxiliary template measurements and pseudo-data from "
        "the same empirical law. Final evaluation holds actual templates fixed and draws "
        "from disjoint populations. Passing the first design does not guarantee conditional "
        "coverage in the second. The full grid answers both questions; the method is not "
        "uniformly reliable on these fixed pools.",
        y,
        SMALL,
    )
    note.pdf.showPage()


def audit_page(note, validation):
    note.page(4, "EVIDENCE, HISTORY AND REPRODUCTION")
    y = note.title("What was checked, and what remains limited")
    y = note.paragraph(
        "The scope is two particle-level generators in one kinematic window. A common "
        "alpha is a declared sensitivity model without a generator prior, not a complete "
        "detector systematic. Training-seed variation, classifier/cut selection and true "
        "event-population uncertainty are distinct from the reported conditional MC intervals.",
        y,
        COMPACT,
        gap=12,
    )
    y = note.heading("Transparent freezes and unsuccessful findings", y)
    y = note.paragraph(
        "The original v0.9 source gate failed: PFN-ID-43 at f=.25/N=1000 covered 987/1000, "
        "above its frozen 984-hit maximum. That gate stays false. A user-authorized "
        "prospective v0.10 revision replaced the upper veto with minimum-coverage "
        "qualification and usefulness checks, then used fresh reserved MC streams. "
        "All 36 source and 108 modelling controls qualified before either final role opened.",
        y,
        SMALL,
    )
    y = note.paragraph(
        "Control qualification uses an exact one-sided lower coverage bound &gt;=93% "
        "with error 0.05/144 (957 hits/1000), plus failure/span ceilings. This differs "
        "from the empirical primary floor. LR calibration uses rank 973/1000 and maximum "
        "over declared truth cells, independently of validation. Their error budgets are "
        "separate; no combined exact-95% guarantee is asserted.",
        y,
        SMALL,
    )
    y = note.paragraph(
        "An earlier source-only optimizer failure and its stable-KL/SLSQP repair remain "
        "archived. The earlier EFN source-only initialization repair is also preserved; "
        "its primary EFN-minus-PFN-K absolute-bias contrast +0.60 [-0.14,+1.35] pp did "
        "not meet its improvement criterion. None of these histories is replaced by this note.",
        y,
        SMALL,
        gap=12,
    )
    y = note.heading("Verified evidence at a glance", y)
    y = note.table(
        [
            ["Evidence", "Observed result"],
            ["Complete final grid", "216k attempts; zero unresolved failures"],
            ["Independent numerical checks", "144 fits / 288 endpoints; max LR residual 2.772e-4"],
            ["Other independent checks", "216 summaries, 72 contrasts, all 72k cut intervals"],
            ["Geometry / boundaries", "0 disconnected sets; 21,805 / 72k alpha boundaries"],
        ],
        y,
        (0.35, 0.65),
        compact=True,
    )
    y = note.paragraph(
        f"All 144k profile scans stabilized at 33 nodes; finite scans are not a theorem "
        f"excluding every narrow island. Original cut estimates remain unbounded. "
        f"121 tests preceded the unchanged executable freeze. Final-stage CPU was 3.689 h; "
        f"cumulative campaign {validation['cumulative_campaign_cpu_seconds'] / 3600:.3f} h "
        "includes the old attempt, under the same 24 h ceiling. All future outcome-driven "
        "method changes are exploratory; new confirmation requires fresh records/design.",
        y,
        SMALL,
        gap=12,
    )
    y = note.heading("Rebuild the presentation, without rerunning inference", y)
    y = note.paragraph(
        ".venv/bin/python scripts/render_template_figures.py<br/>"
        "$PARALLASSE_REPORT_PYTHON scripts/build_template_report.py",
        y,
        CODE,
        gap=7,
    )
    y = note.paragraph(
        "The second command uses ReportLab 4.4.9; the verified runtime path, figure input "
        "mapping, byte-replay hashes and all evidence links are recorded in README.md and "
        "results/task0015/. Scientific tables are indexed by results/task0014/. "
        "Completed studies reject reentry. Earlier notes remain unchanged.",
        y,
        SMALL,
        gap=12,
    )
    y = note.heading("Sources and attribution", y)
    for text in (
        "Komiske, Metodiev, Thaler. Energy Flow Networks. "
        '<link href="https://arxiv.org/abs/1810.05165" color="#216181">arXiv:1810.05165</link>.',
        "Public light-quark samples, CC BY 4.0: "
        '<link href="https://doi.org/10.5281/zenodo.3164691" color="#216181">Pythia v1</link> '
        "(Komiske, Metodiev, Thaler); "
        '<link href="https://doi.org/10.5281/zenodo.3066475" color="#216181">Herwig v1</link> '
        "(Pathak, Komiske, Metodiev, Schwartz).",
        "Related resilience study: Gambhir, LeBlanc, Zhou. "
        '<link href="https://arxiv.org/abs/2509.19431v2" color="#216181">'
        "arXiv:2509.19431v2</link>. "
        "This bounded project makes no novelty or CERN-affiliation claim.",
    ):
        y = note.paragraph(text, y, SMALL, gap=4)
    note.pdf.showPage()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", type=Path, default=Path("reports/figures"))
    parser.add_argument("--output", type=Path, default=Path("output/pdf/parallasse_report_v4.pdf"))
    args = parser.parse_args()
    if reportlab.Version != "4.4.9":
        raise ValueError("Use the optional pinned ReportLab 4.4.9 report runtime")
    validation, audit = load_evidence()
    data = presentation_data(audit)
    if data != read_json(PRESENTATION / "display_data.json"):
        raise ValueError("Figure display mapping differs from the verified tables")
    figure_hashes = read_json(PRESENTATION / "figure_sha256.json")
    for path, digest in figure_hashes.items():
        if sha256(path) != digest:
            raise ValueError(f"Figure changed after rendering: {path}")
    for name in ("08_template_primary.png", "09_template_coverage_grid.png"):
        expected = [digest for path, digest in figure_hashes.items() if Path(path).name == name]
        if len(expected) != 1 or sha256(args.figures / name) != expected[0]:
            raise ValueError(f"Selected PDF figure differs from the verified export: {name}")
    fonts = Path(reportlab.__file__).parent / "fonts"
    pdfmetrics.registerFont(TTFont("Parallasse", str(fonts / "Vera.ttf")))
    pdfmetrics.registerFont(TTFont("Parallasse-Bold", str(fonts / "VeraBd.ttf")))
    pdfmetrics.registerFontFamily("Parallasse", normal="Parallasse", bold="Parallasse-Bold")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    note = Note(args.output)
    note.pdf.setTitle("Parallasse | Score-template inference under simulation variation")
    note.pdf.setSubject("Primary Herwig coverage gains and secondary conditional coverage gaps")
    overview(note, data)
    primary_page(note, audit, data, args.figures)
    conditional_page(note, data, args.figures)
    audit_page(note, validation)
    note.pdf.save()
    print(f"Wrote four-page score-template research note to {args.output}")


if __name__ == "__main__":
    main()
