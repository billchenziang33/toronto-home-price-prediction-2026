"""Build the human-readable PDF draft from verified validation artifacts.

Uses ReportLab from the Codex bundled Python runtime. The competition model
requirements remain in requirements.txt; this helper only formats the report.
"""
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "pdf" / "SSU_DataDash_submission_draft.pdf"
CHART = ROOT / "figures" / "error_by_price_range.png"

NAVY = colors.HexColor("#18324B")
BLUE = colors.HexColor("#276DB4")
ORANGE = colors.HexColor("#E66C3C")
PALE = colors.HexColor("#EEF3F7")
INK = colors.HexColor("#243443")
MUTED = colors.HexColor("#536575")


def page_chrome(canvas, doc):
    canvas.saveState()
    width, height = letter
    canvas.setStrokeColor(colors.HexColor("#D8E0E7"))
    canvas.setLineWidth(0.6)
    canvas.line(0.65 * inch, height - 0.48 * inch, width - 0.65 * inch, height - 0.48 * inch)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(0.65 * inch, height - 0.37 * inch, "SSU DATA DASH 2026  |  TORONTO HOME-PRICE PREDICTION")
    canvas.drawRightString(width - 0.65 * inch, 0.36 * inch, f"Draft  |  Page {doc.page}")
    canvas.restoreState()


def build():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="TitleCustom", parent=styles["Title"], fontName="Helvetica-Bold",
        fontSize=20, leading=24, textColor=NAVY, alignment=TA_LEFT,
        spaceAfter=5,
    ))
    styles.add(ParagraphStyle(
        name="Deck", parent=styles["Normal"], fontName="Helvetica",
        fontSize=9, leading=12, textColor=MUTED, spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        name="SectionCustom", parent=styles["Heading2"], fontName="Helvetica-Bold",
        fontSize=12, leading=14, textColor=NAVY, spaceBefore=8, spaceAfter=3,
        keepWithNext=True,
    ))
    styles.add(ParagraphStyle(
        name="BodyCustom", parent=styles["BodyText"], fontName="Helvetica",
        fontSize=8.7, leading=11.3, textColor=INK, spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        name="SmallCustom", parent=styles["BodyText"], fontName="Helvetica",
        fontSize=7.5, leading=9.2, textColor=MUTED, spaceAfter=3,
    ))
    styles.add(ParagraphStyle(
        name="Cell", parent=styles["BodyText"], fontName="Helvetica",
        fontSize=7.2, leading=8.7, textColor=INK,
    ))
    styles.add(ParagraphStyle(
        name="CellHead", parent=styles["BodyText"], fontName="Helvetica-Bold",
        fontSize=7.2, leading=8.7, textColor=colors.white,
    ))
    styles.add(ParagraphStyle(
        name="CodeCustom", parent=styles["Code"], fontName="Courier",
        fontSize=7.4, leading=9.3, leftIndent=5, textColor=INK,
        backColor=PALE, borderPadding=5, spaceBefore=2, spaceAfter=3,
    ))

    doc = SimpleDocTemplate(
        str(OUTPUT), pagesize=letter,
        rightMargin=0.65 * inch, leftMargin=0.65 * inch,
        topMargin=0.68 * inch, bottomMargin=0.58 * inch,
        title="SSU Data Dash 2026: Toronto Home-Price Prediction",
        author="Toronto Fourcast: Yan Cui, Jiayi Guan, Shengde Wang, Ziang Chen",
    )
    p = lambda text, style="BodyCustom": Paragraph(text, styles[style])
    story = [
        Spacer(1, 0.03 * inch),
        p("SSU Data Dash 2026: Toronto Home-Price Prediction", "TitleCustom"),
        p("TORONTO FOURCAST  |  Yan Cui · Jiayi Guan · Shengde Wang · Ziang Chen", "Deck"),
        p("Five-minute video: [pending recording]  &nbsp;&nbsp; Code and predictions: <link href='https://github.com/billchenziang33/toronto-home-price-prediction-2026' color='#167A9E'>public GitHub repository</link>", "SmallCustom"),
        p("Summary", "SectionCustom"),
        p("We trained separate gradient-boosted models for the off-market and list-aware questions. Across six monthly walk-forward validations from December 2025 through May 2026, the models reached MdAPE of <b>10.60%</b> off-market and <b>4.36%</b> list-aware on 10,069 homes per track. These scores are below the starter benchmarks of 20.8% and 7.3%, respectively. Community, municipality, and house style helped distinguish homes with similar sizes and coordinates; remaining off-market errors are largest for the lowest- and highest-priced homes."),
        p("Data and features", "SectionCustom"),
        p("We used the provided 150,000 synthetic sales from January 2021 through May 2026. Numeric features are square-footage midpoint, bedrooms, washrooms, parking, latitude, longitude, and months since January 2021. We added property type, house style, municipality, and community as categorical features. Off-market never uses list price; list-aware adds it. Garage and lot dimensions were tested but showed no reliable paired-validation gain. Listing remarks and third-party AVM estimates are absent from the prediction files. We used <b>no outside data</b>."),
        p("Models and validation", "SectionCustom"),
        p("Both tracks use scikit-learn's histogram gradient-boosting regressor trained on log sale price. It handles categories natively. Each fold learns category values from that fold's training sales; unseen categories become missing values. The list-aware model receives list price because it is available at prediction time. The off-market model does not."),
        p("For each month from December 2025 to May 2026, the model trained on sales before that month and predicted that month's sales. We did not use hidden test prices. The intervals below bootstrap individual homes and do not account for errors that share the same month's market conditions."),
    ]

    results = [
        [p("Track", "CellHead"), p("Starter MdAPE / MAPE", "CellHead"), p("Current MdAPE (95% CI)", "CellHead"), p("Current MAPE (95% CI)", "CellHead"), p("MdAPE change", "CellHead")],
        [p("Off-market", "Cell"), p("20.8% / 27.4%", "Cell"), p("10.60% (10.38-10.85)", "Cell"), p("13.40% (13.17-13.65)", "Cell"), p("-10.2 pp", "Cell")],
        [p("List-aware", "Cell"), p("7.3% / 8.1%", "Cell"), p("4.36% (4.26-4.46)", "Cell"), p("5.57% (5.47-5.66)", "Cell"), p("-2.9 pp", "Cell")],
    ]
    table = Table(results, colWidths=[0.85 * inch, 1.12 * inch, 1.45 * inch, 1.45 * inch, 0.9 * inch], repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("BACKGROUND", (0, 1), (-1, -1), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#C9D4DE")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.extend([
        p("Results", "SectionCustom"), table,
        Spacer(1, 3),
        p("Compared with our numeric and property-type feature model on the same homes, adding the three categories reduced MdAPE by 3.80 percentage points off-market (paired 95% bootstrap interval: 3.52 to 4.12) and 0.19 points list-aware (0.14 to 0.24). All six months improved on both tracks."),
        p("Where errors remain", "SectionCustom"),
    ])

    chart = Image(str(CHART), width=6.85 * inch, height=3.85 * inch)
    chart.hAlign = "CENTER"
    story.extend([
        chart,
        p("Absolute percentage error by shared sold-price decile. The line is the median; the shaded band is the middle 50%.", "SmallCustom"),
        p("Off-market MdAPE is 13.10% in the cheapest decile and 16.55% in the most expensive, compared with about 8% to 12% through the middle. List-aware MdAPE rises to 5.79% in the top decile from 3.67% in the second and third deciles. Off-market Co-op Apt has 14.33% MdAPE, but that estimate uses only 48 validation homes. These segments describe the synthetic validation set and do not establish why individual homes were mispriced."),
        p("Limitations and outside data", "SectionCustom"),
        p("Validation covers six months and may differ from the hidden June-September 2026 distribution. Home-level bootstrap intervals understate uncertainty if errors correlate within months. The dataset is synthetic: community names and coordinates do not map to real Toronto locations, so these results are not a live-market valuation service. We would next test time-aware community comparables and segment calibration using the same forward-only validation. <b>Outside data used: none.</b>"),
        p("Reproducing the predictions", "SectionCustom"),
        p("From a clean clone: <font name='Courier'>python -m pip install -r requirements.txt</font>, then run <font name='Courier'>python run.py validate</font>, <font name='Courier'>python run.py report</font>, <font name='Courier'>python run.py predict</font>, and <font name='Courier'>python run.py check</font>. The check command verifies the two required CSVs. Add the recorded, tested YouTube link before submitting."),
    ])

    doc.build(story, onFirstPage=page_chrome, onLaterPages=page_chrome)
    print(OUTPUT)


if __name__ == "__main__":
    build()
