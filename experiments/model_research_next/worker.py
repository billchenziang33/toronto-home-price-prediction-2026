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

from data import hide_answers
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
    month, track = job["month"], job["track"]
    if month not in job["allowed_months"] or track not in {"off_market", "list_aware"}:
        raise ValueError("worker job is outside the frozen fold/track plan")
    data_path = ROOT / job["data_path"]
    month_end = (pd.Period(month, freq="M") + 1).start_time
    # Read only the training history and this held-out month. Later labels are
    # never materialized in a development fold.
    listings = pd.read_parquet(data_path, filters=[("date_sold", "<", month_end)])
    listings["date_sold"] = pd.to_datetime(listings.date_sold, errors="coerce")
    train, homes, truth = build_month_fold(listings, month, track)
    cutoff = pd.Period(month, freq="M").start_time
    if pd.to_datetime(train.date_sold).max() >= cutoff:
        raise RuntimeError("training rows cross the outer evaluation cutoff")
    if not homes.ml_num.astype(str).is_unique or not truth.ml_num.astype(str).is_unique:
        raise RuntimeError("evaluation fold contains duplicate listing IDs")
    if not np.array_equal(homes.ml_num.astype(str).to_numpy(), truth.ml_num.astype(str).to_numpy()):
        raise RuntimeError("answer-hidden homes and labels do not align")

    spec = job["spec"]
    started = time.perf_counter()
    result = fit_component(
        train=train,
        homes=homes,
        track=track,
        method_id=spec["method_id"],
        month=month,
        model_family=spec["model_family"],
        model_config=spec["model_config"],
        target_override=spec.get("target_override"),
    )
    fit_seconds = time.perf_counter() - started
    predictions = np.asarray(result["predictions"], dtype=float)
    if predictions.shape != (len(homes),) or not np.isfinite(predictions).all() or (predictions <= 0).any():
        raise ValueError("learner returned non-finite or non-positive dollar predictions")

    keys = homes.ml_num.astype(str).to_numpy()
    heldout = listings.loc[listings.date_sold.dt.to_period("M").astype(str).eq(month)].copy()
    heldout["ml_num"] = heldout.ml_num.astype(str)
    heldout = heldout.set_index("ml_num").loc[keys]
    frame = pd.DataFrame({
        "ml_num": keys,
        "track": track,
        "month": month,
        "date_sold": heldout.date_sold.to_numpy(),
        "sold_price": truth.sold_price.to_numpy(dtype=float),
        "raw_prediction": predictions,
    })
    for column in ("house_type_name", "house_style", "municipality", "community"):
        if column in homes.columns:
            frame[column] = homes[column].to_numpy()
    output_path = ROOT / job["prediction_path"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    frame.to_csv(temp_path, index=False)
    os.replace(temp_path, output_path)
    return {
        "component_id": job["component_id"],
        "track": track,
        "month": month,
        "stage": job["stage"],
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
        "evaluation_rows": len(frame),
        "train_ids_sha256": result["train_ids_sha256"],
        "evaluation_ids_sha256": hashlib.sha256("\n".join(sorted(keys)).encode("utf-8")).hexdigest(),
        "feature_columns": result["feature_columns"],
        "sample_weight_mean": result["sample_weight_mean"],
        "fit_seconds": fit_seconds,
        "prediction_path": str(output_path.relative_to(ROOT)),
        "prediction_sha256": sha256_file(output_path),
    }


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m experiments.model_research_next.worker JOB.json")
    print(json.dumps(execute(Path(sys.argv[1]))))


if __name__ == "__main__":
    main()
