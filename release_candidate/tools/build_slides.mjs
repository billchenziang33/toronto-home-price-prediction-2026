import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const runtimeModules = process.env.RUNTIME_NODE_MODULES ??
  "C:/Users/admin/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules";
process.env.RUNTIME_NODE_MODULES ??= runtimeModules;
const skillDir = process.env.CODEX_PRESENTATIONS_SKILL_DIR ??
  "C:/Users/admin/.codex/plugins/cache/openai-primary-runtime/presentations/26.1013.11914/skills/presentations";
const moduleFile = path.join(runtimeModules, "@oai/artifact-tool/dist/artifact_tool.mjs");
const { FileBlob, Presentation, PresentationFile } = await import(pathToFileURL(moduleFile).href);
const { finalizePresentation } = await import(pathToFileURL(path.join(skillDir, "container_tools/artifact_tool_utils.mjs")).href);

const W = 1280;
const H = 720;
const FONT = "Arial";
const C = {
  ink: "#18324B",
  text: "#243443",
  muted: "#536575",
  teal: "#147D80",
  coral: "#D35E3D",
  gold: "#E7B85C",
  paper: "#F7F6F1",
  white: "#FFFFFF",
  line: "#D7DFE4",
  paleTeal: "#E6F1F0",
  paleCoral: "#F8ECE7",
};

async function readImageBlob(imagePath) {
  const bytes = await fs.readFile(imagePath);
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
}
const outputPath = path.join(root, "slides/toronto_fourcast_final.pptx");
const buildDir = path.join(root, "build/slides");

const presentation = Presentation.create({ slideSize: { width: W, height: H } });

function addText(slide, name, text, x, y, w, h, size, color = C.text, bold = false, align = "left") {
  const shape = slide.shapes.add({
    geometry: "textbox",
    name,
    position: { left: x, top: y, width: w, height: h },
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  shape.text = text;
  shape.text.style = {
    fontSize: size,
    typeface: FONT,
    color,
    bold,
    alignment: align,
    verticalAlignment: "middle",
    wrap: "square",
    autoFit: "shrinkText",
    insets: 0,
  };
  return shape;
}

function addRect(slide, name, x, y, w, h, fill, lineFill = "none", lineWidth = 0) {
  return slide.shapes.add({
    geometry: "rect",
    name,
    position: { left: x, top: y, width: w, height: h },
    fill,
    line: { style: "solid", fill: lineFill, width: lineWidth },
  });
}

function addLine(slide, name, x, y, w, color = C.line, width = 2) {
  return slide.shapes.add({
    geometry: "line",
    name,
    position: { left: x, top: y, width: w, height: 0 },
    fill: "none",
    line: { style: "solid", fill: color, width },
  });
}

function addBase(slide, title, page) {
  slide.background.fill = C.paper;
  addText(slide, `title-${page}`, title, 76, 50, 1110, 78, 42, C.ink, true);
  addLine(slide, `rule-${page}`, 76, 144, 1128, C.line, 1.5);
  addText(slide, `team-${page}`, "TORONTO FOURCAST", 76, 674, 320, 20, 13, C.muted, true);
  addText(slide, `page-${page}`, `${page} / 6`, 1132, 674, 72, 20, 13, C.muted, false, "right");
}

function notes(slide, narration, source = "Official challenge source: https://github.com/erwinzhang7/SSU-Data-Dash") {
  slide.speakerNotes.text = `${narration}\n\nSource: ${source}`;
}

// 1. The two scoring tracks have different information available at prediction time.
{
  const slide = presentation.slides.add();
  slide.background.fill = C.paper;
  addText(slide, "cover-team", "TORONTO FOURCAST", 76, 58, 520, 26, 16, C.teal, true);
  addText(slide, "cover-title", "Two home-price\nquestions", 76, 126, 680, 170, 64, C.ink, true);
  addText(slide, "cover-subtitle", "Toronto home-price prediction challenge", 78, 314, 710, 40, 27, C.muted);
  addLine(slide, "cover-divider", 78, 402, 1122, C.line, 2);
  addText(slide, "offmarket-label", "OFF-MARKET", 78, 438, 300, 36, 23, C.teal, true);
  addText(slide, "offmarket-desc", "Estimate before listing\nNo asking price available", 78, 486, 470, 94, 30, C.text);
  addText(slide, "listaware-label", "LIST-AWARE", 676, 438, 300, 36, 23, C.coral, true);
  addText(slide, "listaware-desc", "Estimate after listing\nAsking price is an input", 676, 486, 470, 94, 30, C.text);
  addText(slide, "cover-foot", "Separate tracks keep the information boundary clear.", 78, 608, 1050, 32, 19, C.muted);
  notes(slide,
    "A home estimate can answer two different questions. Before the owner lists, there is no asking price to guide the estimate. After listing, the model can also use the asking price. We kept those tasks separate because an estimate that sees list price cannot answer the off-market question.",
    "Official handout and rubric: https://github.com/erwinzhang7/SSU-Data-Dash/blob/main/handout.md and https://github.com/erwinzhang7/SSU-Data-Dash/blob/main/rubric.md");
}

// 2. A forward-only timeline prevents future-sale leakage.
{
  const slide = presentation.slides.add();
  addBase(slide, "The test moves forward in time", 2);
  addText(slide, "timeline-intro", "Train on earlier sales. Predict the next month.", 80, 178, 1080, 42, 27, C.text);
  addText(slide, "history-label", "TRAINING HISTORY", 80, 262, 320, 28, 16, C.teal, true);
  addText(slide, "history-dates", "Earlier sales only\nwithin each fold", 80, 300, 350, 86, 21, C.text);
  addText(slide, "validation-label", "SIX WALK-FORWARD MONTHS", 476, 262, 390, 28, 16, C.coral, true);
  addText(slide, "validation-dates", "Dec 2025 - May 2026\n10,069 homes per track", 476, 300, 350, 86, 21, C.text);
  addText(slide, "test-label", "HIDDEN TEST", 932, 262, 260, 28, 16, C.ink, true);
  addText(slide, "test-dates", "Later hidden sales\nSale prices unavailable", 932, 300, 320, 86, 21, C.text);

  const y = 472;
  addLine(slide, "timeline-history", 88, y, 374, C.teal, 8);
  addLine(slide, "timeline-validation", 462, y, 404, C.coral, 8);
  addLine(slide, "timeline-test", 866, y, 324, C.ink, 8);
  const monthXs = [482, 550, 618, 686, 754, 822];
  for (let i = 0; i < monthXs.length; i++) {
    slide.shapes.add({
      geometry: "ellipse", name: `validation-month-${i + 1}`,
      position: { left: monthXs[i] - 7, top: y - 8, width: 16, height: 16 },
      fill: C.paper, line: { style: "solid", fill: C.coral, width: 3 },
    });
  }
  addText(slide, "timeline-training-foot", "Each fold uses only sales available before that month.", 80, 532, 1000, 34, 20, C.muted);
  addText(slide, "timeline-caveat", "Synthetic transaction records. Community labels and coordinates do not map to real Toronto addresses.", 80, 590, 1100, 36, 18, C.muted);
  notes(slide,
    "The challenge provides synthetic Toronto-style transaction data. We tested the model with six walk-forward monthly windows, for 10,069 homes per track overall. In each window, we trained on sales before the month and predicted sales in that month. We did not use hidden sale prices or outside records.",
    "Official starter README and handout: https://github.com/erwinzhang7/SSU-Data-Dash and https://github.com/erwinzhang7/SSU-Data-Dash/blob/main/handout.md");
}

// 3. Separate models share property information but preserve the list-price boundary.
{
  const slide = presentation.slides.add();
  addBase(slide, "The two models keep inputs separate", 3);
  addText(slide, "shared-label", "BOTH TRACKS", 82, 184, 360, 28, 16, C.teal, true);
  addText(slide, "shared-features", "Home area, rooms, parking\nProperty type, house style\nMunicipality, community\nCoordinates and listing month", 82, 232, 500, 190, 28, C.text);
  addText(slide, "model-method", "Histogram gradient boosting\ntrained on log sale price", 82, 470, 520, 78, 23, C.muted, true);
  addLine(slide, "feature-divider", 640, 184, 1, C.line, 1);
  addText(slide, "offmarket-only-label", "OFF-MARKET", 704, 204, 350, 30, 19, C.teal, true);
  addText(slide, "offmarket-only-copy", "No list-price input", 704, 246, 470, 54, 31, C.text);
  addText(slide, "listaware-only-label", "LIST-AWARE", 704, 362, 350, 30, 19, C.coral, true);
  addText(slide, "listaware-only-copy", "Adds the asking price", 704, 404, 470, 54, 31, C.text);
  addLine(slide, "category-finding-rule", 704, 506, 460, C.line, 1.5);
  addText(slide, "frozen-version", "FROZEN RELEASE\nPublished structured model\nRemarks kept in experiment archive", 704, 526, 480, 94, 19, C.ink, true);
  notes(slide,
    "Both models use home size, rooms, parking, location coordinates, property type, house style, municipality, community, and listing time. The list-aware model also receives list price; the off-market model never sees it. Both models predict log sale price with histogram gradient boosting. The frozen release is the previously published structured-feature version. Remarks predictions remain in a separate experiment archive.",
    "Frozen feature and model definitions: https://github.com/billchenziang33/toronto-home-price-prediction-2026/blob/7cffe186bd9f0a2e8fc22f3897d52b3dbaaa8df8/features.py and https://github.com/billchenziang33/toronto-home-price-prediction-2026/blob/7cffe186bd9f0a2e8fc22f3897d52b3dbaaa8df8/models.py");
}

// 4. The headline metrics compare each track with its own starter baseline.
{
  const slide = presentation.slides.add();
  addBase(slide, "The frozen model beats both starters", 4);
  addText(slide, "results-subhead", "MdAPE is the competition score; MAPE is the tie-breaker.", 80, 174, 1080, 36, 21, C.muted);
  const table = slide.tables.add({
    rows: 3, columns: 4, left: 80, top: 238, width: 1120, height: 220,
    values: [
      ["Track", "Starter MdAPE", "Frozen MdAPE", "Frozen MAPE"],
      ["Off-market", "20.8%", "10.60% (10.38–10.85)", "13.40% (13.17–13.65)"],
      ["List-aware", "7.3%", "4.36% (4.26–4.46)", "5.57% (5.47–5.66)"],
    ],
  });
  table.styleOptions = { headerRow: true, bandedRows: true };
  table.borders.assign({ style: "solid", fill: C.line, width: 1 });
  for (let col = 0; col < 4; col++) {
    const cell = table.getCell(0, col);
    cell.fill = C.ink;
    cell.text.style = { fontSize: 20, typeface: FONT, color: C.white, bold: true, verticalAlignment: "middle", insets: 10 };
  }
  for (let row = 1; row < 3; row++) {
    for (let col = 0; col < 4; col++) {
      const cell = table.getCell(row, col);
      cell.fill = row === 1 ? C.white : C.paleTeal;
      cell.text.style = { fontSize: 21, typeface: FONT, color: C.text, bold: col === 0, verticalAlignment: "middle", insets: 10 };
    }
  }
  addText(slide, "results-ci-note", "Parentheses show 95% home-bootstrap intervals. 10,069 validation homes per track.", 82, 472, 1110, 34, 18, C.muted);
  addText(slide, "results-warning", "Validation results only. Hidden-test performance is not yet known.", 82, 546, 1100, 44, 25, C.coral, true);
  notes(slide,
    "On the same walk-forward validation, off-market MdAPE fell from the starter's 20.8 percent to 10.60 percent. List-aware MdAPE fell from 7.3 percent to 4.36 percent. The table reports the published structured model and its reproduced starter comparison. These are validation results, not a claim about the hidden test.",
    "Official starter baselines and rubric: https://github.com/erwinzhang7/SSU-Data-Dash/blob/main/handout.md and https://github.com/erwinzhang7/SSU-Data-Dash/blob/main/rubric.md");
}

// 5. Price-decile errors expose the segments where review is most useful.
{
  const slide = presentation.slides.add();
  addBase(slide, "Error rises at the price extremes", 5);
  addText(slide, "error-chart-subhead", "Median absolute percentage error by shared sold-price decile", 80, 174, 1000, 32, 20, C.muted);
  slide.images.add({
    blob: await readImageBlob(path.join(root, "evidence/error_by_price_range.png")),
    contentType: "image/png",
    alt: "Median absolute percentage error by shared sold-price decile for both validation tracks",
    fit: "contain",
    position: { left: 72, top: 218, width: 835, height: 396 },
  });
  addText(slide, "off-market-tail", "OFF-MARKET\n13.10% lowest decile\n16.55% highest decile", 930, 292, 270, 130, 22, C.teal, true);
  addLine(slide, "tail-divider", 930, 452, 260, C.line, 1.5);
  addText(slide, "list-aware-tail", "LIST-AWARE\n5.79% highest decile", 930, 478, 270, 82, 22, C.coral, true);
  addText(slide, "tail-sample-note", "Price segments use the same frozen walk-forward predictions as the headline scores.", 82, 624, 1080, 28, 17, C.muted);
  notes(slide,
    "The error chart shows a remaining weakness. Off-market MdAPE is 13.10 percent in the cheapest price decile and 16.55 percent in the most expensive, compared with roughly 8 to 12 percent through the middle. List-aware MdAPE rises to 5.79 percent in the top decile. These slices come from the same frozen validation predictions as the headline scores.",
    "Computed from the frozen six-month predictions archived at evidence/validation_predictions_frozen.csv.");
}

// 6. State the practical limit and the next validation question.
{
  const slide = presentation.slides.add();
  addBase(slide, "Use the result within its limits", 6);
  addText(slide, "takeaway-main", "A stronger first estimate\nfor this synthetic benchmark", 80, 186, 650, 116, 39, C.ink, true);
  addText(slide, "limits-label", "LIMITS", 80, 360, 220, 28, 16, C.coral, true);
  addText(slide, "limits-copy", "Synthetic communities and coordinates do not map to Toronto.\nSix validation months may differ from the hidden period.\nHome-bootstrap intervals do not capture shared monthly shocks.\nCorrected remarks intervals include zero on both tracks.", 80, 398, 940, 164, 21, C.text);
  addLine(slide, "next-step-divider", 80, 580, 1120, C.line, 1.5);
  addText(slide, "next-step", "Next test: price-tail calibration using forward-only predictions.", 80, 604, 1100, 48, 20, C.teal, true);
  notes(slide,
    "Our results are useful within this synthetic benchmark, but the community names and coordinates do not map to real Toronto. The hidden period may behave differently from the six validation months, and home-level bootstrap intervals do not capture shared month shocks. The organizer says remarks reflect market dynamics, so we tested them separately. After correcting the paired MdAPE calculation, both confirmation-period intervals include zero. We therefore kept the published structured model frozen and retained the remarks predictions as an experiment. A next step is to test price-tail calibration with the same forward-only validation. Thank you.",
    "Official handout: https://github.com/erwinzhang7/SSU-Data-Dash/blob/main/handout.md. Organizer reply on remarks: https://discord.com/channels/1536459393493762058/1558195702700441700/1558522942113980459. Corrected analysis: experiments/remarks_experiment_audit.json.");
}

await fs.mkdir(buildDir, { recursive: true });
await fs.mkdir(path.dirname(outputPath), { recursive: true });
for (const [index, slide] of presentation.slides.items.entries()) {
  const png = await presentation.export({ slide, format: "png", scale: 1 });
  await fs.writeFile(path.join(buildDir, `slide-${String(index + 1).padStart(2, "0")}.png`), new Uint8Array(await png.arrayBuffer()));
}

const requirements = {
  explicitTotalSlideCount: 6,
  requiredNativeTableOwnerSlides: [4],
  requiredNativeChartOwnerSlides: [],
};
const candidateDir = path.join(root, "build/slide_finalizer");
await fs.mkdir(candidateDir, { recursive: true });
await fs.mkdir(path.dirname(outputPath), { recursive: true });
const candidatePath = path.join(candidateDir, "candidate.pptx");
await (await PresentationFile.exportPptx(presentation)).save(candidatePath);

const result = await finalizePresentation({
  ...requirements,
  workspaceDir: root,
  candidatePath,
  finalPath: outputPath,
  pythonExecutable: process.env.RUNTIME_PYTHON ??
    "C:/Users/admin/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe",
  integrityValidatorPath: path.join(skillDir, "container_tools/inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(skillDir, "container_tools/inspect_presentation_layout_geometry.py"),
  layoutArgs: ["--expected-slide-size-emu", "12192000,6858000", "--validate-heading-fit", "--require-native-table-slide", "4"],
  fontPolicy: { basis: "design", families: [FONT] },
  verifyArtifactToolImport: true,
  receiptPath: path.join(candidateDir, "validation.json"),
});
console.log(JSON.stringify(result, null, 2));

const finalPresentation = await PresentationFile.importPptx(await FileBlob.load(outputPath));
for (const [index, slide] of finalPresentation.slides.items.entries()) {
  const png = await finalPresentation.export({ slide, format: "png", scale: 1 });
  await fs.writeFile(path.join(buildDir, `final-slide-${String(index + 1).padStart(2, "0")}.png`),
    new Uint8Array(await png.arrayBuffer()));
}
