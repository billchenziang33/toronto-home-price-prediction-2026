"""One-process-per-fit worker; invoked by run.py with a JSON job description."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .adapters import PROJECT_ROOT, PriceModel, build_fold
from .metrics import score


def _hash_ids(values) -> str:
    import hashlib
    return hashlib.sha256("\n".join(sorted(map(str, values))).encode("utf-8")).hexdigest()


def execute(job_path: Path) -> dict:
    job = json.loads(job_path.read_text(encoding="utf-8"))
    data_path = PROJECT_ROOT / job["data_path"]
    listings = pd.read_parquet(data_path)
    month = job["month"]
    date_sold = pd.to_datetime(listings.date_sold)
    heldout = listings.loc[date_sold.dt.to_period("M").astype(str) == month].copy()
    train, homes = build_fold(listings, month, job["track"])
    if len(heldout) != len(homes) or not np.array_equal(heldout.ml_num.to_numpy(), homes.ml_num.to_numpy()):
        raise RuntimeError("held-out labels and prediction rows do not preserve the same IDs and order")

    started = time.perf_counter()
    model = PriceModel(job["track"], job["model_family"], job["model_config"], job["target"])
    model.fit(train)
    fit_seconds = time.perf_counter() - started
    predicted = model.predict(homes)
    baseline_path = job.get("baseline_prediction_path")
    if baseline_path:
        base_frame = pd.read_csv(PROJECT_ROOT / baseline_path, dtype={"ml_num": str})
        baseline = base_frame.set_index("ml_num").reindex(homes.ml_num.astype(str))
        if baseline.predicted_price.isna().any():
            raise RuntimeError("frozen HGBR development baseline is missing one or more held-out IDs")
        baseline_price = baseline.predicted_price.to_numpy(dtype=float)
    else:
        baseline_price = predicted.copy()

    rows = pd.DataFrame({
        "ml_num": homes.ml_num.astype(str).to_numpy(), "track": job["track"], "month": month,
        "model_id": job["trial_id"], "model_family": job["model_family"], "target": job["target"],
        "house_type_name": heldout.house_type_name.astype(str).to_numpy(),
        "sold_price": heldout.sold_price.to_numpy(dtype=float),
        "predicted_price": predicted, "baseline_price": baseline_price,
    })
    score_result = score(rows.predicted_price.to_numpy(), rows.sold_price.to_numpy())
    rows_temp = Path(job["prediction_path"] + ".tmp")
    rows_temp.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(rows_temp, index=False)
    os.replace(rows_temp, job["prediction_path"])

    transformed = model.transformer.transform(homes, model.model_family)
    unknown_counts = {column: int(transformed[column].isna().sum())
                      for column in model.transformer.categorical_columns() if column in transformed}
    result = {
        "trial_id": job["trial_id"], "track": job["track"], "month": month,
        "model_family": job["model_family"], "target": job["target"],
        "model_config": job["model_config"], "cutoff_exclusive": pd.Period(month, freq="M").start_time.isoformat(),
        "train_rows": len(train), "evaluation_rows": len(homes), "train_ids_sha256": _hash_ids(train.ml_num),
        "evaluation_ids_sha256": _hash_ids(homes.ml_num), "feature_columns": model.transformer.columns,
        "categorical_columns": model.transformer.categorical_columns(), "unknown_category_counts": unknown_counts,
        "fit_seconds": fit_seconds, "prediction_and_write_seconds": time.perf_counter() - started,
        "metrics": score_result, "prediction_path": job["prediction_path"],
        "fit_parameters": model.estimator.get_params(deep=False),
        "versions": {name: __import__(name).__version__ for name in ["numpy", "pandas", "sklearn"]},
    }
    result_path = Path(job["result_path"])
    result_temp = result_path.with_suffix(result_path.suffix + ".tmp")
    result_temp.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(result_temp, result_path)
    return result


def main() -> int:
    if len(sys.argv) != 3 or sys.argv[1] != "--job":
        print("usage: python -m experiments.model_optimization_v1.fit_worker --job PATH", file=sys.stderr)
        return 2
    result = execute(Path(sys.argv[2]))
    print(json.dumps({"trial_id": result["trial_id"], "fit_seconds": result["fit_seconds"],
                      "metrics": result["metrics"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
