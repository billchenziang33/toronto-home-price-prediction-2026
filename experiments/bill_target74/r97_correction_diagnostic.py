"""Reproduce a descriptive, no-fit correction-magnitude audit for R97."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "outputs/bill_target74/bill_target74_unbounded_20261024/batches/r97m_20261025T210147Z"
PAIRED = RUN / "paired_predictions.csv"
MANIFEST = RUN / "run_manifest.json"
OUTPUT = ROOT / "docs/model_evidence/2026-10-28/list_aware_r97_correction_diagnostic.json"
MONTHS = {
    "2023-03", "2023-07", "2023-11", "2024-03",
    "2024-06", "2024-09", "2025-01", "2025-03",
}
GROUPS = ("zero", "0_to_1pct_log", "1_to_3pct_log", "3_to_5pct_log")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def correction_group(value: float) -> str:
    if value == 0.0:
        return "zero"
    if value <= 0.01:
        return "0_to_1pct_log"
    if value <= 0.03:
        return "1_to_3pct_log"
    if value <= 0.05:
        return "3_to_5pct_log"
    raise ValueError(f"absolute correction exceeds the registered 0.05 cap: {value}")


def main() -> dict:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    source_hash = sha256(PAIRED)
    if source_hash != manifest.get("paired_predictions_sha256"):
        raise RuntimeError("R97 paired-prediction file does not match its run manifest")

    values: dict[str, list[tuple[float, float]]] = {name: [] for name in GROUPS}
    months_seen: set[str] = set()
    with PAIRED.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "month", "track", "sold_price", "base_prediction",
            "candidate_prediction", "correction_log",
        }
        if not required.issubset(reader.fieldnames or ()):
            raise RuntimeError("R97 paired predictions are missing required columns")
        for row in reader:
            if row["track"] != "list_aware":
                continue
            month = row["month"]
            if month not in MONTHS:
                raise RuntimeError(f"unexpected development month: {month}")
            months_seen.add(month)
            actual = float(row["sold_price"])
            parent = float(row["base_prediction"])
            candidate = float(row["candidate_prediction"])
            correction = abs(float(row["correction_log"]))
            if actual <= 0 or not all(map(math.isfinite, (actual, parent, candidate, correction))):
                raise RuntimeError("invalid actual, prediction, or correction in a saved development row")
            group = correction_group(correction)
            parent_ape = 100.0 * abs(parent - actual) / actual
            candidate_ape = 100.0 * abs(candidate - actual) / actual
            values[group].append((parent_ape, candidate_ape))

    row_count = sum(map(len, values.values()))
    if months_seen != MONTHS or row_count != 15_642:
        raise RuntimeError(f"unexpected R97 coverage: {sorted(months_seen)}, rows={row_count}")

    groups = {}
    for name, pairs in values.items():
        parent_apes = [pair[0] for pair in pairs]
        candidate_apes = [pair[1] for pair in pairs]
        groups[name] = {
            "n": len(pairs),
            "parent_mdape_percent": statistics.median(parent_apes) if pairs else None,
            "candidate_mdape_percent": statistics.median(candidate_apes) if pairs else None,
            "parent_mape_percent": statistics.fmean(parent_apes) if pairs else None,
            "candidate_mape_percent": statistics.fmean(candidate_apes) if pairs else None,
            "row_improved": sum(candidate < parent for parent, candidate in pairs),
            "row_worsened": sum(candidate > parent for parent, candidate in pairs),
            "row_unchanged": sum(candidate == parent for parent, candidate in pairs),
        }

    result = {
        "analysis": "descriptive post-hoc correction-magnitude grouping; no fits",
        "run_id": manifest["run_id"],
        "data_sha256": manifest["source_identity"]["data_sha256"],
        "paired_predictions_sha256": source_hash,
        "track": "list_aware",
        "months": sorted(months_seen),
        "rows": row_count,
        "groups": groups,
        "reserve_months_labels_opened": False,
        "hidden_labels_opened": False,
        "promotion_or_tuning_decision": False,
        "caveat": "Development months have been reused in search. These candidate-defined groups do not establish a causal or future benefit and must not be used to tune a correction threshold.",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, ensure_ascii=False))
