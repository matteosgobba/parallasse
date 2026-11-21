"""Build the four-page Parallasse research note with ReportLab 4.4.9."""

import argparse
import hashlib
import json
from pathlib import Path

import reportlab
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle

RUN = Path("results/task0004/20261110-heldout-01")
MODELS = ("XGB", "MLP", "PFN-K", "PFN-ID")
WIDTH, HEIGHT = A4
MARGIN = 53
CONTENT = WIDTH - 2 * MARGIN
INK = colors.HexColor("#172b3b")
MUTED = colors.HexColor("#506477")
ACCENT = colors.HexColor("#216181")
ORANGE = colors.HexColor("#a94623")
GRID = colors.HexColor("#dbe5ec")
PALE = colors.HexColor("#edf4f8")
WARM = colors.HexColor("#f9f1e9")


def make_style(font, size, leading, color=INK):
    return ParagraphStyle(
        "parallasse",
        fontName=font,
        fontSize=size,
        leading=leading,
        textColor=color,
        splitLongWords=False,
    )


BODY = make_style("Parallasse", 10.7, 15.0)
DECK = make_style("Parallasse", 12.1, 17.0, MUTED)
CAPTION = make_style("Parallasse", 9.3, 12.6, MUTED)
SMALL = make_style("Parallasse", 8.8, 11.7, MUTED)
CODE = make_style("Courier", 8.7, 13.2)


def read_json(path):
    return json.loads(Path(path).read_text())


def verify(path, expected):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError(f"Evidence hash mismatch: {path}")


class Note:
    def __init__(self, path):
        self.pdf = canvas.Canvas(
            str(path), pagesize=A4, invariant=1, pageCompression=1, initialFontName="Parallasse"
        )
        self.pdf.setTitle("Parallasse | Quark/gluon tagging across event generators")
        self.pdf.setAuthor("Matteo Sgobba")
        self.pdf.setSubject("Fixed-source calibration and particle identity under generator shift")
        self.bottom = 64

    def line(self, text, x, y, font="Parallasse", size=10, color=INK):
        self.pdf.setFont(font, size)
        self.pdf.setFillColor(color)
        self.pdf.drawString(x, y, text)

    def page(self, number, descriptor):
        self.pdf.setStrokeColor(GRID)
        self.pdf.setLineWidth(0.7)
        self.pdf.line(MARGIN, 787, WIDTH - MARGIN, 787)
        self.pdf.line(MARGIN, 49, WIDTH - MARGIN, 49)
        self.line("PARALLASSE  /  RESEARCH NOTE", MARGIN, 801, "Parallasse-Bold", 8, ACCENT)
        self.pdf.setFont("Parallasse", 8)
        self.pdf.setFillColor(MUTED)
        self.pdf.drawRightString(WIDTH - MARGIN, 801, descriptor)
        self.line("MATTEO SGOBBA  |  NOVEMBER 2026", MARGIN, 34, size=7.8, color=MUTED)
        self.pdf.drawRightString(WIDTH - MARGIN, 34, f"{number:02d} / 04")

    def paragraph(self, text, y, style=BODY, x=MARGIN, width=CONTENT, gap=10):
        p = Paragraph(text, style)
        _, height = p.wrap(width, 1000)
        if y - height < self.bottom:
            raise ValueError(f"Paragraph exceeds content area: {text[:55]}")
        p.drawOn(self.pdf, x, y - height)
        return y - height - gap

    def title(self, title, y=756):
        width = pdfmetrics.stringWidth(title, "Parallasse-Bold", 27)
        self.line(title, MARGIN, y, "Parallasse-Bold", min(27, 27 * CONTENT / width))
        return y - 17

    def heading(self, title, y):
        self.line(title, MARGIN, y, "Parallasse-Bold", 12.8, ACCENT)
        return y - 10

    def table(self, rows, y, fractions, compact=False):
        table = Table(
            rows,
            colWidths=[CONTENT * f for f in fractions],
            rowHeights=[24 if compact else 25] * len(rows),
        )
        table.setStyle(
            TableStyle(
                [
                    ("FONTNAME", (0, 0), (-1, -1), "Parallasse"),
                    ("FONTNAME", (0, 0), (-1, 0), "Parallasse-Bold"),
                    ("FONTNAME", (0, 1), (0, -1), "Parallasse-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8.8 if compact else 9.4),
                    ("TEXTCOLOR", (0, 0), (-1, -1), INK),
                    ("BACKGROUND", (0, 0), (-1, 0), PALE),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.white, colors.HexColor("#f8fafc")],
                    ),
                    ("LINEBELOW", (0, 0), (-1, -1), 0.4, GRID),
                    ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 7),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ]
            )
        )
        _, height = table.wrap(CONTENT, 1000)
        if y - height < self.bottom:
            raise ValueError("Table exceeds content area")
        table.drawOn(self.pdf, MARGIN, y - height)
        return y - height - 12

    def figure(self, path, y, caption):
        image = ImageReader(str(path))
        px_w, px_h = image.getSize()
        height = CONTENT * px_h / px_w
        if y - height < self.bottom:
            raise ValueError(f"Figure exceeds content area: {path}")
        self.pdf.drawImage(image, MARGIN, y - height, width=CONTENT, height=height)
        return self.paragraph(caption, y - height - 4, CAPTION, gap=16)

    def callout(self, title, text, y, fill=PALE):
        p = Paragraph(text, BODY)
        _, content_height = p.wrap(CONTENT - 30, 1000)
        height = content_height + 43
        if y - height < self.bottom:
            raise ValueError("Callout exceeds content area")
        self.pdf.setFillColor(fill)
        self.pdf.roundRect(MARGIN, y - height, CONTENT, height, 8, fill=1, stroke=0)
        self.line(title, MARGIN + 15, y - 21, "Parallasse-Bold", 10.5, ACCENT)
        p.drawOn(self.pdf, MARGIN + 15, y - 30 - content_height)
        return y - height - 18


def pct(value):
    return f"{value:.2%}"


def interval(item):
    return f"[{item['bootstrap_95_lo']:+.4f}, {item['bootstrap_95_hi']:+.4f}]"


def card_row(note, cards, top):
    gap = 9
    w = (CONTENT - 2 * gap) / 3
    for i, (value, label, detail) in enumerate(cards):
        x = MARGIN + i * (w + gap)
        note.pdf.setFillColor(WARM if i == 1 else PALE)
        note.pdf.roundRect(x, top - 75, w, 75, 7, fill=1, stroke=0)
        note.line(value, x + 11, top - 28, "Parallasse-Bold", 16, ORANGE if i == 1 else ACCENT)
        note.line(label, x + 11, top - 47, "Parallasse-Bold", 9.3)
        note.line(detail, x + 11, top - 62, size=8.1, color=MUTED)
    return top - 96


def overview(note, report):
    h = report["cohort_metrics"]["target"]
    gaps = [x["mean"] for x in report["auc_gap_pythia_minus_herwig"].values()]
    fprs = [x["fpr"]["mean"] for x in h.values()]
    pid = [report["auc_contrasts"][key]["PFN-ID_minus_PFN-K"]["mean"] for key in ("test", "target")]
    note.page(1, "STUDY DESIGN")
    note.line("FROZEN SOURCE-TO-TARGET EVALUATION", MARGIN, 756, "Parallasse-Bold", 8.7, ACCENT)
    note.line("Parallasse", MARGIN, 705, "Parallasse-Bold", 45)
    y = note.paragraph("Quark/gluon tagging across event generators", 689, DECK, gap=8)
    y = note.paragraph(
        "Matteo Sgobba  |  Independent scientific ML study  |  10 November 2026", y, SMALL, gap=16
    )
    y = note.paragraph(
        "A jet tagger trained on one simulator can still rank quarks above gluons on another, "
        "yet accept many more gluons at the same numerical cut. This study separates ranking, "
        "threshold transfer, and the effect of adding particle identity (PID).",
        y,
        BODY,
        gap=15,
    )
    y = card_row(
        note,
        [
            (
                f"{min(gaps) * 100:.2f}-{max(gaps) * 100:.2f} pp",
                "AUC loss on Herwig",
                "Across four model configurations",
            ),
            (
                f"{min(fprs) * 100:.1f}-{max(fprs) * 100:.1f}%",
                "Herwig gluon acceptance",
                "Cuts calibrated to 10% on Pythia",
            ),
            (f"{pid[0] * 100:.2f} to {pid[1] * 100:.2f} pp", "PID gain in AUC", "Pythia to Herwig"),
        ],
        y,
    )
    y = note.heading("A single training domain, two final tests", y)
    y = note.table(
        [
            ["Cohort", "Generator", "Jets", "Role"],
            ["Train", "Pythia", "60,000", "Fit models and preprocessing"],
            ["Tune", "Pythia", "10,000", "Choose stopping checkpoint"],
            ["Calibration", "Pythia", "10,000", "Set cuts using 5,000 gluons"],
            ["Final source test", "Pythia", "20,000", "Measure in-domain performance"],
            ["Final target test", "Herwig", "100,000", "Measure generator transfer"],
        ],
        y,
        (0.24, 0.16, 0.15, 0.45),
    )
    y = note.paragraph(
        "Both samples contain equal numbers of light-quark and gluon jets. The public "
        "particle-level "
        "samples have nominal jet transverse momentum 500-550 GeV and absolute rapidity below 1.7. "
        "An exact constituent-level audit found no duplicate jets or cross-generator overlap. "
        "Herwig was reserved for final scoring.",
        y,
    )
    note.paragraph(
        f"The final tests contain {report['class_counts']['test']['quarks']:,} and "
        f"{report['class_counts']['target']['quarks']:,} quarks, respectively, with the "
        "same numbers of gluons. Quark is the positive class. Both cohorts consist of simulated "
        "particle-level jets.",
        y,
        CAPTION,
    )
    note.pdf.showPage()


def discrimination(note, report, figures):
    note.page(2, "MODELS AND DISCRIMINATION")
    y = note.title("Discrimination under generator shift")
    y = note.paragraph(
        "Four configurations were fit on identical source rows at seeds 17, 29, and 43. "
        "Checkpoint selection used source tuning only.",
        y,
        DECK,
    )
    y = note.paragraph(
        "<b>XGBoost and MLP.</b> Both use six PID-blind summaries covering multiplicity, "
        "transverse-momentum "
        "dispersion, three radial moments, and maximum constituent momentum fraction. The MLP "
        "standardizes them using training rows only.",
        y,
    )
    y = note.paragraph(
        "<b>Particle Flow Networks (PFN-K and PFN-ID).</b> A shared encoder builds representations "
        "of real constituents for summation, "
        "using relative transverse momentum and centered angles. Both allocate 9,953 "
        "parameters and "
        "share initialization and batch order within each seed. PFN-K uses one constant identity "
        "token. PFN-ID receives categorical species. Padding is excluded before encoding.",
        y,
    )
    y = note.figure(
        figures / "01_roc_by_generator.png",
        y,
        "<b>Figure 1.</b> ROC on each final test. Lines average interpolated per-seed TPR "
        "at common FPR. Legend AUCs average the exact per-seed areas.",
    )
    y = note.heading("Held-out results at a glance", y)
    rows = [
        ["Model", "Pythia AUC", "Herwig AUC", "AUC loss", "Pythia TPR / FPR", "Herwig TPR / FPR"]
    ]
    for model in MODELS:
        p, h = (report["cohort_metrics"][cohort][model] for cohort in ("test", "target"))
        gap = report["auc_gap_pythia_minus_herwig"][model]["mean"]
        rows.append(
            [
                model,
                f"{p['auc']['mean']:.4f}",
                f"{h['auc']['mean']:.4f}",
                f"{gap:.4f}",
                f"{pct(p['tpr']['mean'])} / {pct(p['fpr']['mean'])}",
                f"{pct(h['tpr']['mean'])} / {pct(h['fpr']['mean'])}",
            ]
        )
    y = note.table(rows, y, (0.117, 0.125, 0.125, 0.115, 0.259, 0.259), compact=True)
    note.paragraph(
        "All entries average metrics over three seeds. AUC loss is Pythia minus Herwig. "
        "TPR and FPR are quark and gluon acceptance at the checkpoint-specific frozen cut. "
        "Exact accepted/total counts and seed ranges are in the evaluation artifacts.",
        y,
        CAPTION,
    )
    note.pdf.showPage()


def operating_point(note, report, figures):
    note.page(3, "OPERATING POINT AND ABLATION")
    y = note.title("Gluon acceptance rises on Herwig")
    y = note.paragraph(
        "For each checkpoint, set a strict score cut on Pythia calibration gluons. Each "
        "cut accepted "
        "500 of 5,000 there. The cut was frozen before either final test and applied unchanged.",
        y,
        DECK,
    )
    y = note.figure(
        figures / "02_frozen_cut_acceptance.png",
        y,
        "<b>Figure 2.</b> Mean test acceptance with 95% conditional event-bootstrap "
        "intervals. The dashed line is the nominal calibration target. Herwig accepts "
        "more gluons for every configuration.",
    )
    y = note.heading("Particle identity helps less on Herwig", y - 4)
    y = note.figure(
        figures / "03_particle_identity_gain.png",
        y,
        "<b>Figure 3.</b> PFN-ID minus PFN-K in AUC percentage points. The bottom row "
        "subtracts the Herwig gain from the Pythia gain. Open points show the three "
        "individual seeds. Filled points and bars show the mean and conditional 95% interval.",
    )
    change = report["pid_auc_gain_pythia_minus_herwig"]
    note.callout(
        "AUC gain and threshold behaviour are different results",
        f"The PID gain is {change['mean'] * 100:.2f} AUC percentage points larger on Pythia. "
        "On Herwig, PFN-ID has "
        "the largest mean quark acceptance, but also the largest gluon acceptance at its "
        "transferred cut. The transferred cuts compare models at different background acceptances.",
        y,
        WARM,
    )
    note.pdf.showPage()


def interpretation(note, report):
    note.page(4, "INTERPRETATION AND REPRODUCTION")
    y = note.title("What changes across generators")
    p, h = (report["auc_contrasts"][key]["PFN-K_minus_XGB"] for key in ("test", "target"))
    change = report["pid_auc_gain_pythia_minus_herwig"]
    y = note.paragraph(
        f"<b>Ranking changes modestly.</b> PFN-K minus XGBoost changes from {p['mean']:+.4f} "
        f"{interval(p)} on Pythia to {h['mean']:+.4f} {interval(h)} on Herwig. On Herwig, "
        "the difference varies in sign across seeds. XGBoost and the matched-input MLP have "
        "close AUCs on both tests.",
        y,
    )
    y = note.paragraph(
        f"<b>Particle identity has a domain-dependent effect.</b> The Pythia minus Herwig "
        f"difference in PID gain is {change['mean']:+.4f} {interval(change)}. The attenuation has "
        "the same direction across the three seeds, even though the small Herwig gain itself "
        "varies in sign.",
        y,
    )
    y = note.paragraph(
        "<b>Uncertainty and scope.</b> Brackets are descriptive 95% percentile intervals from 500 "
        "class-stratified event resamples per cohort. The same resampled rows are shared across "
        "models and seeds within a cohort. Resampling is independent across generators. "
        "The intervals "
        "condition on the trained models and frozen cuts. They exclude calibration and retraining "
        "uncertainty. The intervals describe individual contrasts and metrics. The study covers "
        "one particle-level sample per generator.",
        y,
        gap=15,
    )
    y = note.heading("Reproduce the presentation from frozen scores", y)
    y = note.paragraph(
        "Run <b>20261110-heldout-01</b> records row IDs, data and model hashes, numerical cuts, "
        "score arrays, counts, and bootstrap draws. From the project root, the following commands "
        "verify the code and rebuild the presentation. The PDF builder uses a Python environment "
        "with ReportLab 4.4.9 (see reports/requirements-report.txt).",
        y,
        gap=6,
    )
    y = note.paragraph(
        ".venv/bin/python -m pytest -q<br/>"
        ".venv/bin/python scripts/render_figures.py<br/>"
        "python scripts/build_report.py",
        y,
        CODE,
        gap=14,
    )
    y = note.heading("Data and methodological sources", y)
    refs = (
        (
            "P. Komiske, E. Metodiev and J. Thaler. Pythia8 Quark and Gluon Jets for Energy Flow, "
            'v1. <link href="https://doi.org/10.5281/zenodo.3164691" color="#216181">'
            "doi:10.5281/zenodo.3164691</link>."
        ),
        (
            "A. Pathak, P. Komiske, E. Metodiev and M. Schwartz. Herwig7.1 Quark and Gluon Jets, "
            'v1. <link href="https://doi.org/10.5281/zenodo.3066475" color="#216181">'
            "doi:10.5281/zenodo.3066475</link>. Both datasets carry a CC BY 4.0 licence."
        ),
        (
            "P. Komiske, E. Metodiev and J. Thaler. Energy Flow Networks: Deep Sets for "
            "Particle Jets. "
            '<link href="https://arxiv.org/abs/1810.05165" color="#216181">arXiv:1810.05165</link>.'
        ),
        (
            "R. Gambhir, M. LeBlanc and Y. Zhou. The Pareto Frontier of Resilient Jet Tagging. "
            '<link href="https://arxiv.org/abs/2509.19431" color="#216181">arXiv:2509.19431</link>.'
        ),
    )
    for reference in refs:
        y = note.paragraph(reference, y, SMALL, gap=5)
    note.pdf.showPage()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", type=Path, default=Path("reports/figures"))
    parser.add_argument("--output", type=Path, default=Path("output/pdf/parallasse_report.pdf"))
    args = parser.parse_args()
    evidence = read_json("results/task0004/validation.json")["evidence_sha256"]
    for name in ("summary.json", "cut_freeze.json", "run.json", "per_seed.csv"):
        path = RUN / name
        verify(path, evidence[str(path)])
    report = read_json(RUN / "summary.json")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fonts = Path(reportlab.__file__).parent / "fonts"
    pdfmetrics.registerFont(TTFont("Parallasse", str(fonts / "Vera.ttf")))
    pdfmetrics.registerFont(TTFont("Parallasse-Bold", str(fonts / "VeraBd.ttf")))
    pdfmetrics.registerFontFamily("Parallasse", normal="Parallasse", bold="Parallasse-Bold")
    note = Note(args.output)
    overview(note, report)
    discrimination(note, report, args.figures)
    operating_point(note, report, args.figures)
    interpretation(note, report)
    note.pdf.save()
    print(f"Wrote four-page research note: {args.output}")


if __name__ == "__main__":
    main()
