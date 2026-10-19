from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/model_research_next/month_audit.json"
MONTH_PATTERN = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")
DATA_PATH = ROOT / "data/.hf_cache/hub/datasets--Baliyang--SSU_data_dash/snapshots/96a258439e2a85f650699be8339e52d8ac5614b7/listings.parquet"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ledger_months() -> tuple[set[str], list[str]]:
    months: set[str] = set()
    files = []
    roots = [ROOT / "outputs", ROOT / "model_candidate_v2", ROOT / "release_candidate"]
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.jsonl"):
            if "ledger" not in path.name.lower():
                continue
            if path.is_relative_to(ROOT / "outputs/model_research_next"):
                continue
            files.append(str(path.relative_to(ROOT)))
            for raw in path.read_text(encoding="utf-8").splitlines():
                if not raw.strip():
                    continue
                try:
                    record = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                month = str(record.get("month", ""))
                status = str(record.get("status", "")).lower()
                if MONTH_PATTERN.fullmatch(month) and status not in {"cache_hit", "skipped", "not_started"}:
                    months.add(month)
    return months, sorted(set(files))


def _artifact_months() -> tuple[set[str], list[str]]:
    months: set[str] = set()
    files = []
    roots = [ROOT / "outputs", ROOT / "model_candidate_v2", ROOT / "release_candidate"]
    excluded = ROOT / "outputs/model_research_next"
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.csv"):
            if path.is_relative_to(excluded):
                continue
            try:
                frame = pd.read_csv(path, usecols=["month"])
            except (OSError, ValueError, pd.errors.ParserError):
                continue
            values = frame.month.dropna().astype(str)
            found = {value for value in values.unique() if MONTH_PATTERN.fullmatch(value)}
            if found:
                months.update(found)
                files.append(str(path.relative_to(ROOT)))
    return months, sorted(files)


def _protocol_months() -> set[str]:
    months: set[str] = set()
    for path in (ROOT / "experiments").rglob("protocol.json"):
        if path.parent.name == "model_research_next":
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for key in ("previously_evaluated_months", "development_months", "screen_months",
                    "confirmation_months", "oof_seed_months", "retrospective_benchmark_months"):
            values = payload.get(key, [])
            if isinstance(values, list):
                months.update(str(value) for value in values if MONTH_PATTERN.fullmatch(str(value)))
    return months


def build_audit() -> dict:
    data_hash = _sha256(DATA_PATH)
    expected_hash = "18cf87fa122b544018fccbc15040403b0fed37080c44a75d1b55deca96d2c4bf"
    if data_hash != expected_hash:
        raise ValueError(f"training source hash changed: {data_hash}")
    ledger_months, ledger_files = _ledger_months()
    artifact_months, artifact_files = _artifact_months()
    protocol_months = _protocol_months()
    used = ledger_months | artifact_months | protocol_months

    # Only dates and IDs are read here, and the cutoff excludes the first
    # previously used fresh-fold month (2024-10). No labels are inspected.
    rows = pd.read_parquet(
        DATA_PATH,
        columns=["ml_num", "date_sold"],
        filters=[("date_sold", "<", pd.Timestamp("2024-10-01"))],
    )
    rows["date_sold"] = pd.to_datetime(rows.date_sold, errors="coerce")
    rows = rows.dropna(subset=["date_sold"])
    rows["month"] = rows.date_sold.dt.to_period("M").astype(str)
    candidates = []
    for month, heldout in rows.groupby("month", sort=True):
        if not ("2021-01" <= month <= "2024-09") or month in used:
            continue
        cutoff = pd.Period(month, freq="M").start_time
        train_count = int((rows.date_sold < cutoff).sum())
        candidates.append({
            "month": month,
            "train_rows": train_count,
            "evaluation_rows": int(len(heldout)),
            "unique_evaluation_ids": int(heldout.ml_num.astype(str).nunique()),
            "unused_by_scanned_evidence": True,
        })
    return {
        "data_sha256": data_hash,
        "source_filter_exclusive": "2024-10-01",
        "source_columns_read": ["ml_num", "date_sold"],
        "sale_price_labels_read": False,
        "hidden_files_read": False,
        "prior_month_evidence": {
            "ledger_months": sorted(ledger_months),
            "artifact_months": sorted(artifact_months),
            "protocol_months": sorted(protocol_months),
            "used_months_union": sorted(used),
            "ledger_files": ledger_files,
            "prediction_artifact_files_with_month_column": artifact_files,
        },
        "unused_candidate_folds": candidates,
        "candidate_month_count": len(candidates),
    }


def main() -> None:
    result = build_audit()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(OUTPUT),
        "data_sha256": result["data_sha256"],
        "scanned_ledger_count": len(result["prior_month_evidence"]["ledger_files"]),
        "prediction_files_with_month_column": len(result["prior_month_evidence"]["prediction_artifact_files_with_month_column"]),
        "unused_folds": result["unused_candidate_folds"],
    }, indent=2))


if __name__ == "__main__":
    main()
