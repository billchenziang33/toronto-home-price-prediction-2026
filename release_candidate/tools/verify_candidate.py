"""Verify the release candidate without training a model or reading hidden labels."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
COMMIT = "7cffe186bd9f0a2e8fc22f3897d52b3dbaaa8df8"
KEYS = ["ml_num", "track", "month"]
NS = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"FAIL: {message}")


def interval(values: np.ndarray, statistic, repeats: int = 1000) -> list[float]:
    rng = np.random.default_rng(0)
    samples = [statistic(rng.choice(values, size=len(values))) for _ in range(repeats)]
    return [round(float(value), 2) for value in np.percentile(samples, [2.5, 97.5])]


def paired_delta(baseline: np.ndarray, candidate: np.ndarray) -> tuple[float, list[float]]:
    rng = np.random.default_rng(0)
    samples = []
    for _ in range(1000):
        indices = rng.integers(0, len(baseline), size=len(baseline))
        samples.append(np.median(candidate[indices]) - np.median(baseline[indices]))
    ci = [round(float(value), 3) for value in np.percentile(samples, [2.5, 97.5])]
    return round(float(np.median(candidate) - np.median(baseline)), 3), ci


def verify_metrics() -> None:
    predictions = pd.read_csv(ROOT / "evidence/validation_predictions_frozen.csv")
    expected_months = ["2025-12", "2026-01", "2026-02", "2026-03", "2026-04", "2026-05"]
    require(len(predictions) == 20138, "saved validation prediction row count changed")
    require(sorted(predictions.month.unique()) == expected_months, "validation months changed")
    reported = json.loads((ROOT / "evidence/validation_metrics.json").read_text(encoding="utf-8"))
    for track, rows in predictions.groupby("track", sort=True):
        require(len(rows) == 10069, f"{track} validation count changed")
        signed = 100 * (rows.predicted_price - rows.sold_price) / rows.sold_price
        absolute = signed.abs().to_numpy()
        actual = {
            "n": len(rows),
            "mape": round(float(absolute.mean()), 2),
            "mape_95ci": interval(absolute, np.mean),
            "mdape": round(float(np.median(absolute)), 2),
            "mdape_95ci": interval(absolute, np.median),
            "median_signed_error": round(float(np.median(signed)), 2),
        }
        require(actual == reported[track], f"{track} metrics do not match saved predictions")
    print("PASS validation metrics: exact match to 20,138 saved homes")


def verify_remarks() -> None:
    baseline = pd.read_csv(ROOT / "evidence/validation_predictions_frozen.csv")
    experiment = pd.read_csv(ROOT / "experiments/remarks_text_predictions.csv")
    frozen = baseline.set_index(KEYS).sort_index()
    paired = experiment.set_index(KEYS).sort_index()
    require(frozen.index.equals(paired.index), "remarks rows do not match frozen validation")
    require(np.array_equal(frozen.sold_price.to_numpy(), paired.sold_price.to_numpy()),
            "remarks target values do not match frozen validation")
    require(np.array_equal(frozen.predicted_price.to_numpy(), paired.baseline_price.to_numpy()),
            "remarks baseline prices do not match frozen predictions")
    actual = {}
    for track in ["off_market", "list_aware"]:
        track_rows = paired[paired.index.get_level_values("track") == track]
        groups = {
            "overall": track_rows,
            "selection_dec_to_feb": track_rows[
                track_rows.index.get_level_values("month").isin(["2025-12", "2026-01", "2026-02"])
            ],
            "confirmation_mar_to_may": track_rows[
                track_rows.index.get_level_values("month").isin(["2026-03", "2026-04", "2026-05"])
            ],
        }
        actual[track] = {}
        for name, rows in groups.items():
            base_ape = (100 * (rows.baseline_price - rows.sold_price).abs() / rows.sold_price).to_numpy()
            remarks_ape = (100 * (rows.remarks_price - rows.sold_price).abs() / rows.sold_price).to_numpy()
            delta, ci = paired_delta(base_ape, remarks_ape)
            actual[track][name] = {
                "n": len(rows),
                "baseline_mdape": round(float(np.median(base_ape)), 3),
                "remarks_mdape": round(float(np.median(remarks_ape)), 3),
                "delta_pp_candidate_minus_baseline": delta,
                "paired_home_bootstrap_95ci": ci,
            }
    expected = json.loads((ROOT / "experiments/remarks_experiment_audit.json").read_text(encoding="utf-8"))
    require(actual == expected["tracks"], "corrected remarks audit does not match saved paired predictions")
    print("PASS remarks audit: exact pairing and corrected difference-of-medians bootstrap")


def verify_csvs() -> None:
    expected = {"off_market": 3974, "list_aware": 4064}
    for track, count in expected.items():
        path = ROOT / "submissions" / f"submission_{track}.csv"
        frame = pd.read_csv(path, dtype={"ml_num": str})
        require(list(frame.columns) == ["ml_num", "predicted_price"], f"{path.name} columns changed")
        require(len(frame) == count, f"{path.name} row count changed")
        require(frame.ml_num.is_unique, f"{path.name} has duplicate IDs")
        price = pd.to_numeric(frame.predicted_price, errors="coerce").to_numpy()
        require(np.isfinite(price).all() and (price > 0).all(), f"{path.name} has invalid prices")
    print("PASS submission files: both CSV schemas, counts, IDs, and prices")


def verify_source_commit(source_repo: Path | None) -> None:
    if source_repo is None or not (source_repo / ".git").exists():
        print("SKIP Git-object comparison: source repository was not supplied")
        return
    files = [
        "data.py", "features.py", "models.py", "run.py", "check_submission.py",
        "requirements.txt", "LICENSE", "pytest.ini", "tests/test_starter.py",
        "submissions/submission_off_market.csv", "submissions/submission_list_aware.csv",
    ]
    for relative in files:
        expected = subprocess.check_output(["git", "-C", str(source_repo), "show", f"{COMMIT}:{relative}"])
        actual = (ROOT / relative).read_bytes()
        require(actual == expected, f"{relative} is not byte-for-byte equal to the frozen Git object")
    print(f"PASS source snapshot: code and both CSVs match commit {COMMIT}")


def verify_deck_and_text() -> None:
    deck = ROOT / "slides/toronto_fourcast_final.pptx"
    with zipfile.ZipFile(deck) as archive:
        slide_names = sorted(name for name in archive.namelist()
                             if re.fullmatch(r"ppt/slides/slide\d+\.xml", name))
        require(len(slide_names) == 6, "slide deck is not six slides")
        slide_text = []
        for name in slide_names:
            root = ET.fromstring(archive.read(name))
            slide_text.extend(node.text or "" for node in root.findall(".//a:t", NS))
    text = " ".join(slide_text)
    require("10.60%" in text and "4.36%" in text, "deck headline metrics missing")
    require("3.80" not in text and "0.19" not in text, "unsupported category-gain claim remains in deck")
    require("Remarks" in text and "Corrected remarks intervals include zero" in text,
            "deck does not explain the remarks decision")
    pdf_text = ""
    try:
        from pypdf import PdfReader
        reader = PdfReader(ROOT / "report/SSU_DataDash_submission_candidate.pdf")
        require(len(reader.pages) <= 4, "PDF exceeds four pages")
        pdf_text = " ".join(page.extract_text() or "" for page in reader.pages)
        require("pending team recording" in pdf_text.lower(), "PDF is missing the pending video marker")
        require("Yan Cui" in pdf_text and "Jiayi Guan" in pdf_text and "Shengde Wang" in pdf_text and "Ziang Chen" in pdf_text,
                "PDF team list is incomplete")
    except ImportError:
        print("NOTE PDF page and content audit requires pypdf; PDF render was checked separately")
    if pdf_text:
        require("3.80" not in pdf_text and "0.19" not in pdf_text,
                "unsupported category-gain claim remains in PDF")
    print("PASS deck: six slides with frozen metrics and corrected remarks decision")


def verify_hash_manifest() -> None:
    path = ROOT / "SHA256SUMS.json"
    if not path.exists():
        print("SKIP SHA256 manifest: not generated yet")
        return
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for record in manifest["files"]:
        target = ROOT / record["path"]
        require(target.is_file(), f"manifest file missing: {record['path']}")
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        require(digest == record["sha256"], f"hash mismatch: {record['path']}")
    print(f"PASS hashes: {len(manifest['files'])} packaged files")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-repo", type=Path, default=ROOT.parent,
                        help="worktree containing the frozen Git commit (optional for Desktop copy)")
    args = parser.parse_args()
    source = args.source_repo.resolve() if args.source_repo else None
    verify_metrics()
    verify_remarks()
    verify_csvs()
    verify_source_commit(source)
    verify_deck_and_text()
    verify_hash_manifest()
    print("Candidate verification complete. No model was trained.")


if __name__ == "__main__":
    main()
