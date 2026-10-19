from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd
import pyarrow.parquet as parquet

from .candidates import DEVELOPMENT_MONTHS, FINAL_MONTHS, validate_fold_split


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = Path(__file__).with_name("protocol.json")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _months_from_ledger(path: Path) -> set[str]:
    months: set[str] = set()
    if not path.exists():
        return months
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        record = json.loads(raw_line)
        month = record.get("month")
        if (month and re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", str(month))
                and record.get("status") in {"completed", "running", "failed", "timed_out"}):
            months.add(str(month))
    return months


def _months_from_prediction_files(paths: list[Path]) -> tuple[set[str], int]:
    months: set[str] = set()
    scanned: set[Path] = set()
    for root in paths:
        if not root.exists():
            continue
        for path in root.rglob("*.csv"):
            if path.is_relative_to(ROOT / "outputs/model_research_fresh"):
                continue
            try:
                header = pd.read_csv(path, nrows=0)
                if "month" not in header.columns:
                    continue
                values = pd.read_csv(path, usecols=["month"])["month"].dropna().astype(str)
                months.update(value for value in values.unique()
                              if re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", value))
                scanned.add(path)
            except (OSError, ValueError, pd.errors.ParserError):
                continue
    return months, len(scanned)


def build_audit(final_stage: bool = False) -> dict:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    source = ROOT / protocol["source_data"]
    source_hash = sha256_file(source)
    if source_hash != protocol["data_sha256"]:
        raise ValueError(f"training data hash changed: {source_hash}")

    ledger_months = set()
    ledger_paths = [
        ROOT / "outputs/model_optimization_v1/fit_ledger.jsonl",
        ROOT / "outputs/model_candidate_v2/trial_ledger.jsonl",
    ]
    for path in ledger_paths:
        ledger_months.update(_months_from_ledger(path))
    artifact_months, prediction_files_count = _months_from_prediction_files([
        ROOT / "outputs",
        ROOT / "release_candidate",
        ROOT / "model_candidate_v2",
    ])
    previously_evaluated = ledger_months | artifact_months | set(protocol["previously_evaluated_months"])
    validate_fold_split(DEVELOPMENT_MONTHS, FINAL_MONTHS, previously_evaluated)

    parquet_metadata = parquet.ParquetFile(source).metadata
    latest_needed = "2025-09-01" if final_stage else "2025-04-01"
    # The development audit excludes all records from April 2025 onward, so
    # May/August final labels are not read before the freeze manifest exists.
    allowed_rows = pd.read_parquet(
        source,
        columns=["ml_num", "date_sold", "sold_price", "list_price"],
        filters=[("date_sold", "<", pd.Timestamp(latest_needed))],
    )
    allowed_rows["date_sold"] = pd.to_datetime(allowed_rows["date_sold"])
    selected_months = list(DEVELOPMENT_MONTHS) + (list(FINAL_MONTHS) if final_stage else [])
    fold_counts = {}
    for month in selected_months:
        heldout = allowed_rows.loc[allowed_rows.date_sold.dt.to_period("M").astype(str).eq(month)]
        train = allowed_rows.loc[allowed_rows.date_sold < pd.Period(month, freq="M").start_time]
        if heldout.empty or train.empty or heldout.ml_num.astype(str).duplicated().any():
            raise ValueError(f"invalid fold rows or duplicate IDs: {month}")
        fold_counts[month] = {"train_rows": len(train), "evaluation_rows": len(heldout)}

    return {
        "data_path": protocol["source_data"],
        "data_sha256": source_hash,
        "parquet_metadata_rows": parquet_metadata.num_rows,
        "parquet_schema": str(parquet.ParquetFile(source).schema_arrow),
        "previous_months_from_ledgers": sorted(ledger_months),
        "previous_months_from_prediction_artifacts": sorted(artifact_months),
        "previously_evaluated_months_union": sorted(previously_evaluated),
        "prediction_artifact_files_scanned_count": prediction_files_count,
        "development_months": list(DEVELOPMENT_MONTHS),
        "final_months": list(FINAL_MONTHS),
        "final_stage": final_stage,
        "allowed_data_cutoff_exclusive": latest_needed,
        "fold_counts": fold_counts,
        "quality": {
            "allowed_rows": len(allowed_rows),
            "unique_ml_num": bool(allowed_rows.ml_num.astype(str).is_unique),
            "missing_sale_date": int(allowed_rows.date_sold.isna().sum()),
            "missing_sale_price": int(allowed_rows.sold_price.isna().sum()),
            "nonpositive_sale_price": int((pd.to_numeric(allowed_rows.sold_price, errors="coerce") <= 0).sum()),
        },
    }


def main() -> None:
    output = ROOT / "outputs/model_research_fresh/month_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    result = build_audit()
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
