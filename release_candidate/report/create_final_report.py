"""Build the release-candidate PDF from its frozen validation evidence."""
from pathlib import Path
import json

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, Paragraph, PageBreak, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
METRICS = json.loads((ROOT / "evidence/validation_metrics.json").read_text(encoding="utf-8"))
REMARKS = json.loads((ROOT / "experiments/remarks_experiment_audit.json").read_text(encoding="utf-8"))
OUTPUT = ROOT / "report/SSU_DataDash_submission_candidate.pdf"

NAVY = colors.HexColor("#18324B")
TEAL = colors.HexColor("#147D80")
CORAL = colors.HexColor("#D35E3D")
PALE = colors.HexColor("#EEF3F4")
INK = colors.HexColor("#243443")
MUTED = colors.HexColor("#536575")
LINE = colors.HexColor("#D7DFE4")


def page_chrome(canvas, doc):
    canvas.saveState()
    width, height = letter
    canvas.setStrokeColor(LINE)
    canvas.setLineWidth(0.6)
    canvas.line(0.6 * inch, height - 0.46 * inch, width - 0.6 * inch, height - 0.46 * inch)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(0.6 * inch, height - 0.35 * inch, "SSU DATA DASH 2026  |  TORONTO FOURCAST")
    canvas.drawRightString(width - 0.6 * inch, 0.33 * inch,
                           f"Candidate - video pending  |  Page {doc.page}")
    canvas.restoreState()


def build():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleX", parent=styles["Title"], fontName="Helvetica-Bold",
                              fontSize=19, leading=23, textColor=NAVY, alignment=TA_LEFT, spaceAfter=4))
    styles.add(ParagraphStyle(name="DeckX", parent=styles["Normal"], fontName="Helvetica",
                              fontSize=8.5, leading=11, textColor=MUTED, spaceAfter=6))
    styles.add(ParagraphStyle(name="HeadX", parent=styles["Heading2"], fontName="Helvetica-Bold",
                              fontSize=11, leading=13, textColor=NAVY, spaceBefore=5, spaceAfter=2,
                              keepWithNext=True))
    styles.add(ParagraphStyle(name="BodyX", parent=styles["BodyText"], fontName="Helvetica",
                              fontSize=8.2, leading=10.3, textColor=INK, spaceAfter=3))
    styles.add(ParagraphStyle(name="SmallX", parent=styles["BodyText"], fontName="Helvetica",
                              fontSize=7.3, leading=8.7, textColor=MUTED, spaceAfter=2))
    styles.add(ParagraphStyle(name="CellX", parent=styles["BodyText"], fontName="Helvetica",
                              fontSize=7.1, leading=8.2, textColor=INK))
    styles.add(ParagraphStyle(name="CellHeadX", parent=styles["BodyText"], fontName="Helvetica-Bold",
                              fontSize=7.1, leading=8.2, textColor=colors.white))
    styles.add(ParagraphStyle(name="CalloutX", parent=styles["BodyText"], fontName="Helvetica",
                              fontSize=7.6, leading=9.3, textColor=INK, backColor=PALE,
                              borderColor=LINE, borderWidth=0.4, borderPadding=5, spaceBefore=2, spaceAfter=4))
    doc = SimpleDocTemplate(str(OUTPUT), pagesize=letter, leftMargin=0.6 * inch,
                            rightMargin=0.6 * inch, topMargin=0.62 * inch, bottomMargin=0.53 * inch,
                            title="SSU Data Dash 2026 - Toronto Fourcast",
                            author="Toronto Fourcast: Yan Cui, Jiayi Guan, Shengde Wang, Ziang Chen")
    p = lambda text, style="BodyX": Paragraph(text, styles[style])

    off = METRICS["off_market"]
    aware = METRICS["list_aware"]
    off_remarks = REMARKS["tracks"]["off_market"]["confirmation_mar_to_may"]
    aware_remarks = REMARKS["tracks"]["list_aware"]["confirmation_mar_to_may"]
    def delta_label(result):
        delta = result["delta_pp_candidate_minus_baseline"]
        ci = result["paired_home_bootstrap_95ci"]
        return f"{delta:+.2f} pp ({ci[0]:+.2f} to {ci[1]:+.2f})"

    remarks_table = [
        [p("Track", "CellHeadX"), p("Overall: delta (95% paired CI)", "CellHeadX"),
         p("Confirmation months: delta (95% paired CI)", "CellHeadX")],
        [p("Off-market", "CellX"), p(delta_label(REMARKS["tracks"]["off_market"]["overall"]), "CellX"),
         p(delta_label(off_remarks), "CellX")],
        [p("List-aware", "CellX"), p(delta_label(REMARKS["tracks"]["list_aware"]["overall"]), "CellX"),
         p(delta_label(aware_remarks), "CellX")],
    ]
    rt = Table(remarks_table, colWidths=[1.0 * inch, 2.7 * inch, 2.7 * inch], repeatRows=1)
    rt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
        ("GRID", (0, 0), (-1, -1), 0.35, LINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))

    story = [
        Spacer(1, 0.02 * inch),
        p("SSU Data Dash 2026: Toronto Home-Price Prediction", "TitleX"),
        p("TORONTO FOURCAST  |  Yan Cui  |  Jiayi Guan  |  Shengde Wang  |  Ziang Chen", "DeckX"),
        p("Five-minute video: <b>pending team recording</b>  |  Code and both prediction files: "
          "<link href='https://github.com/billchenziang33/toronto-home-price-prediction-2026/tree/7cffe186bd9f0a2e8fc22f3897d52b3dbaaa8df8' color='#147D80'>"
          "public frozen source at commit 7cffe18</link>", "SmallX"),
        p("Summary", "HeadX"),
        p("We trained separate models for the off-market and list-aware tracks. The published structured-feature model "
          "reached MdAPE of <b>10.60%</b> and <b>4.36%</b>, respectively, in forward-only validation. The list-aware "
          "model can use asking price; the off-market model cannot. These are validation scores, not hidden-test results."),
        p("Data, features, and model", "HeadX"),
        p("The challenge data are synthetic. Inputs include home area, bedrooms, washrooms, parking, coordinates, "
          "property type, house style, municipality, community, and listing month. Both tracks use a "
          "HistGradientBoostingRegressor trained on log sale price, with categorical inputs handled natively. "
          "Only the list-aware model receives list price. We used no outside data and did not inspect hidden sale prices."),
        p("Forward validation and scores", "HeadX"),
        p("Each of six monthly folds trains on sales from earlier months and predicts the next month. The validation "
          "covers December 2025 through May 2026 and 10,069 homes per track. Intervals below resample homes; they do "
          "not capture shared month-to-month market shocks."),
    ]
    table = [
        [p("Track", "CellHeadX"), p("Starter MdAPE", "CellHeadX"),
         p("Frozen MdAPE (95% CI)", "CellHeadX"), p("Frozen MAPE (95% CI)", "CellHeadX")],
        [p("Off-market", "CellX"), p("20.8%", "CellX"),
         p(f"{off['mdape']:.2f}% ({off['mdape_95ci'][0]:.2f}-{off['mdape_95ci'][1]:.2f})", "CellX"),
         p(f"{off['mape']:.2f}% ({off['mape_95ci'][0]:.2f}-{off['mape_95ci'][1]:.2f})", "CellX")],
        [p("List-aware", "CellX"), p("7.3%", "CellX"),
         p(f"{aware['mdape']:.2f}% ({aware['mdape_95ci'][0]:.2f}-{aware['mdape_95ci'][1]:.2f})", "CellX"),
         p(f"{aware['mape']:.2f}% ({aware['mape_95ci'][0]:.2f}-{aware['mape_95ci'][1]:.2f})", "CellX")],
    ]
    t = Table(table, colWidths=[1.05 * inch, 1.05 * inch, 2.15 * inch, 2.12 * inch], repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
        ("GRID", (0, 0), (-1, -1), 0.35, LINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story += [t, Spacer(1, 3),
              p("The starter scores shown are reproduced on the same saved walk-forward homes. MdAPE is the primary "
                "competition score; MAPE is the tie-breaker.", "SmallX"),
              p("Where the model misses", "HeadX"),
              p("Off-market error is highest in the cheapest and most expensive sold-price deciles. List-aware error "
                "is lower overall but also rises in the top decile. The lines show median absolute percentage error; "
                "the shaded bands show the middle half of homes in each decile."),
              Image(str(ROOT / "evidence/error_by_price_range.png"), width=6.85 * inch, height=3.55 * inch),
              p("Figure 1. Price-decile errors from the same six-month frozen validation predictions used for the table.", "SmallX"),
              PageBreak(),
              p("Remarks experiment and version choice", "HeadX"),
              p("The remarks experiment and frozen model have exact row, target, and baseline-prediction matches. "
                "The earlier bootstrap estimated the median of per-home error differences, while the reported point "
                "estimate subtracted two MdAPEs. We corrected it by resampling paired homes and recalculating both "
                "medians separately. Negative deltas favor remarks; every interval below includes zero."),
              rt,
              p("We also removed the earlier categorical-feature gain claim. The archived feature_v1/frozen prediction "
                "pair does not reproduce its point estimates or intervals, and its exact baseline specification is "
                "not preserved. The claim is omitted from the candidate materials.", "CalloutX"),
              p("Limits and outside data", "HeadX"),
              p("The synthetic community names and coordinates are not a real Toronto map. Hidden-period behavior is "
                "unknown. Home-level intervals do not account for correlated errors within a month. Outside data used: "
                "none. A next experiment would test price-tail calibration with the same forward-only validation."),
              p("Reproducibility", "HeadX"),
              p("Frozen source commit: 7cffe186bd9f0a2e8fc22f3897d52b3dbaaa8df8. With Python 3.10+ and the listed "
                "requirements, run <font name='Courier'>python run.py validate</font>, "
                "<font name='Courier'>python run.py report</font>, <font name='Courier'>python run.py predict</font>, "
                "and <font name='Courier'>python run.py check</font>. The package includes the exact published CSVs "
                "and the saved validation predictions used for this report."),
              p("Before submission, insert the team's recorded YouTube link here and confirm that the unlisted video "
                "opens without a login.", "SmallX")]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc.build(story, onFirstPage=page_chrome, onLaterPages=page_chrome)
    print(OUTPUT)


if __name__ == "__main__":
    build()
