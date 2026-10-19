from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.model_optimization_v2.folds import build_month_fold
from experiments.model_optimization_v2.worker import fit_component


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def execute(job_path: Path) -> dict:
    job = json.loads(job_path.read_text(encoding="utf-8"))
    stage = job["stage"]
    month = job["month"]
    if stage == "development" and month not in {"2024-10", "2024-12", "2025-01", "2025-03"}:
        raise ValueError("development worker cannot open labels outside the frozen development months")
    if stage == "final" and month not in {"2025-05", "2025-08"}:
        raise ValueError("final worker cannot open labels outside the frozen final months")

    data_path = ROOT / job["data_path"]
    cutoff = "2025-04-01" if stage == "development" else "2025-09-01"
    listings = pd.read_parquet(data_path, filters=[("date_sold", "<", pd.Timestamp(cutoff))])
    listings["date_sold"] = pd.to_datetime(listings["date_sold"])
    train, homes, truth = build_month_fold(listings, month, job["track"])
    if pd.to_datetime(train.date_sold).max() >= pd.Period(month, freq="M").start_time:
        raise RuntimeError("fold training rows cross the held-out cutoff")
    if homes.ml_num.astype(str).duplicated().any() or truth.ml_num.astype(str).duplicated().any():
        raise RuntimeError("fold contains duplicate listing IDs")
    if not np.array_equal(homes.ml_num.astype(str).to_numpy(), truth.ml_num.astype(str).to_numpy()):
        raise RuntimeError("answer-hidden homes and labels do not align")

    spec = job["spec"]
    started = time.perf_counter()
    result = fit_component(
        train=train,
        homes=homes,
        track=job["track"],
        method_id=spec["method_id"],
        month=month,
        model_family=spec["model_family"],
        model_config=spec["model_config"],
        target_override=spec.get("target_override"),
    )
    fit_seconds = time.perf_counter() - started
    raw = np.asarray(result["predictions"], dtype=float)
    if raw.shape != (len(homes),) or not np.isfinite(raw).all() or (raw <= 0).any():
        raise ValueError("learner returned invalid dollar-price predictions")

    frame = pd.DataFrame({
        "ml_num": homes.ml_num.astype(str).to_numpy(),
        "track": job["track"],
        "month": month,
        "sold_price": truth.sold_price.to_numpy(dtype=float),
        "raw_prediction": raw,
    })
    prediction_path = ROOT / job["prediction_path"]
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = prediction_path.with_suffix(prediction_path.suffix + ".tmp")
    frame.to_csv(temp_path, index=False)
    os.replace(temp_path, prediction_path)

    metadata = {
        "component_id": job["component_id"],
        "track": job["track"],
        "month": month,
        "stage": stage,
        "job_fingerprint": job["job_fingerprint"],
        "data_sha256": job["data_sha256"],
        "code_sha256": job["code_sha256"],
        "model_family": spec["model_family"],
        "model_config": spec["model_config"],
        "method_id": spec["method_id"],
        "target_override": spec.get("target_override"),
        "target": result["target"],
        "actual_objective": result["actual_objective"],
        "train_rows": result["train_rows"],
        "train_ids_sha256": result["train_ids_sha256"],
        "evaluation_rows": len(frame),
        "evaluation_ids_sha256": hashlib.sha256(
            "\n".join(sorted(frame.ml_num)).encode("utf-8")
        ).hexdigest(),
        "feature_columns": result["feature_columns"],
        "fit_seconds": fit_seconds,
        "prediction_sha256": sha256_file(prediction_path),
        "sample_weight_mean": result["sample_weight_mean"],
    }
    metadata_path = prediction_path.with_suffix(".json")
    temp_metadata = metadata_path.with_suffix(".json.tmp")
    temp_metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    os.replace(temp_metadata, metadata_path)
    return metadata


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m experiments.model_research_fresh.worker JOB.json")
    result = execute(Path(sys.argv[1]))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
