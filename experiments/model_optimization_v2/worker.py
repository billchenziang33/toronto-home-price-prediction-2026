from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .features import build_features
from .learners import (inverse_price, make_learner, recency_weights, select_training_rows,
                      training_target, validate_actual_objective, method_ids)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _id_hash(values) -> str:
    return hashlib.sha256("\n".join(sorted(map(str, values))).encode("utf-8")).hexdigest()


def _normalize_categories(train_x: pd.DataFrame, eval_x: pd.DataFrame, family: str):
    categories = {}
    for column in train_x.select_dtypes(include=["object", "category", "string"]).columns:
        values = train_x[column].astype("string").fillna("Unknown")
        vocab = values.drop_duplicates().astype(str).tolist()
        categories[column] = vocab
        if family == "catboost":
            train_x[column] = values.astype(str)
            eval_x[column] = eval_x[column].astype("string").fillna("Unknown").astype(str)
        else:
            train_x[column] = pd.Categorical(values, categories=vocab)
            eval_x[column] = pd.Categorical(eval_x[column].astype("string").fillna("Unknown"), categories=vocab)
    for column in train_x.columns:
        if column not in categories:
            train_x[column] = pd.to_numeric(train_x[column], errors="coerce")
            eval_x[column] = pd.to_numeric(eval_x[column], errors="coerce")
    return train_x, eval_x, categories


def _sklearn_pipeline(method_id: str, train_x: pd.DataFrame):
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, SplineTransformer

    categorical = list(train_x.select_dtypes(include=["object", "category", "string"]).columns)
    numeric = [column for column in train_x.columns if column not in categorical]
    if method_id == "M16":
        spline = [column for column in numeric if column not in {"months", "list_price"}]
        linear = [column for column in numeric if column not in spline]
        transformers = []
        if spline:
            transformers.append(("spline", Pipeline([("impute", SimpleImputer(strategy="median")),
                ("basis", SplineTransformer(n_knots=5, degree=3, knots="quantile", include_bias=False))]), spline))
        if linear:
            transformers.append(("linear", SimpleImputer(strategy="median"), linear))
        if categorical:
            transformers.append(("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=True), categorical))
        transform = ColumnTransformer(transformers, sparse_threshold=1.0)
        return Pipeline([("features", transform), ("ridge", Ridge(alpha=10.0, solver="lsqr"))])
    if method_id == "M17":
        categorical_cardinality = sum(train_x[column].nunique(dropna=False) for column in categorical)
        estimated_sparse_bytes = len(train_x) * max(len(categorical), 1) * 8 + len(train_x) * max(len(numeric), 1) * 8
        if categorical_cardinality > 512 or estimated_sparse_bytes > 1_000_000_000:
            raise ValueError("M17 sparse one-hot estimate exceeds the 512-column or 1GB guard")
        transformers = []
        if numeric:
            transformers.append(("numeric", SimpleImputer(strategy="median"), numeric))
        if categorical:
            transformers.append(("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=True), categorical))
        transform = ColumnTransformer(transformers, sparse_threshold=1.0)
        model = ExtraTreesRegressor(n_estimators=200, min_samples_leaf=20, max_features=0.8,
                                    random_state=0, n_jobs=4)
        return Pipeline([("features", transform), ("extra_trees", model)])
    raise ValueError(f"no scikit-learn pipeline for {method_id}")


def fit_component(train: pd.DataFrame, homes: pd.DataFrame, track: str, method_id: str,
                  month: str, model_family: str, model_config: dict,
                  reference_history: pd.DataFrame | None = None,
                  target_override: str | None = None) -> dict:
    """Fit one model component on a fold and return unrounded dollar predictions."""
    methods = method_ids(method_id)
    selected = select_training_rows(train, month, method_id)
    origin = None
    if "M07" in methods:
        origin = (float(pd.to_numeric(selected.latitude, errors="coerce").min()),
                  float(pd.to_numeric(selected.longitude, errors="coerce").min()))
    history = reference_history if reference_history is not None else selected
    cutoff = pd.Period(month, freq="M").start_time
    train_x = build_features(selected.drop(columns=["sold_price", "date_sold"], errors="ignore"),
                             track, method_id, history=history, cutoff=cutoff, grid_origin=origin)
    eval_x = build_features(homes, track, method_id, history=history, cutoff=cutoff, grid_origin=origin)
    if list(train_x.columns) != list(eval_x.columns):
        raise ValueError("training/evaluation feature schemas differ")
    train_x, eval_x, categories = _normalize_categories(train_x.copy(), eval_x.copy(), model_family)
    target, target_name = training_target(selected, track, method_id, target_override=target_override)
    weights = recency_weights(selected, month, method_id)
    started = time.perf_counter()
    sklearn_methods = methods.intersection({"M16", "M17"})
    if len(sklearn_methods) > 1:
        raise ValueError("M16 and M17 are alternative learner families, not a composable feature bundle")
    if sklearn_methods:
        learner_method = next(iter(sklearn_methods))
        if len(methods) > 1:
            raise ValueError(f"{learner_method} must be evaluated as a standalone learner")
        estimator = _sklearn_pipeline(learner_method, train_x)
        estimator.fit(train_x, target)
        actual_objective = f"{learner_method.lower()} fixed configuration"
    else:
        params = dict(model_config)
        estimator = make_learner(model_family, params)
        fit_kwargs = {}
        if model_family == "catboost":
            fit_kwargs["cat_features"] = list(categories)
        if "M01" in methods:
            fit_kwargs["sample_weight"] = weights
        estimator.fit(train_x, target, **fit_kwargs)
        expected = str(params.get("objective", params.get("loss_function", "regression")))
        actual_objective = validate_actual_objective(estimator, model_family, expected)
    fit_seconds = time.perf_counter() - started
    transformed = np.asarray(estimator.predict(eval_x), dtype=float)
    predictions = inverse_price(transformed, homes, track, method_id, target_override=target_override)
    return {"estimator": estimator, "predictions": predictions, "fit_seconds": fit_seconds,
            "actual_objective": actual_objective, "target": target_name,
            "train_rows": len(selected), "train_ids_sha256": _id_hash(selected.ml_num),
            "feature_columns": list(train_x.columns), "category_vocabulary": categories,
            "grid_origin": origin, "sample_weight_mean": float(weights.mean()),
            "prediction_rows": len(homes)}


def execute(job_path: Path) -> dict:
    job = json.loads(job_path.read_text(encoding="utf-8"))
    data_path = Path(job["data_path"])
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path
    listings = pd.read_parquet(data_path)
    from .folds import build_month_fold
    train, homes, truth = build_month_fold(listings, job["month"], job["track"])
    if not np.array_equal(homes.ml_num.astype(str).to_numpy(), truth.ml_num.astype(str).to_numpy()):
        raise RuntimeError("prediction homes and held-out labels do not align")
    result = fit_component(train, homes, job["track"], job["method_id"], job["month"],
                           job["model_family"], job["model_config"],
                           target_override=job.get("target_override"))
    source = listings.loc[pd.to_datetime(listings.date_sold).dt.to_period("M").astype(str).eq(job["month"])]
    if source.ml_num.astype(str).duplicated().any():
        raise RuntimeError("evaluation fold has duplicate listing ids")
    source = source.set_index(source.ml_num.astype(str)).loc[homes.ml_num.astype(str)]
    frame = pd.DataFrame({"ml_num": homes.ml_num.astype(str), "track": job["track"],
                          "month": job["month"], "sold_price": truth.sold_price.to_numpy(dtype=float),
                          "predicted_price": result["predictions"],
                          "house_type_name": source.house_type_name.astype("string").fillna("<missing>").astype(str).to_numpy(),
                          "date_sold": pd.to_datetime(source.date_sold).to_numpy(),
                          "date_listed": pd.to_datetime(source.date_listed).to_numpy(),
                          "latitude": pd.to_numeric(source.latitude, errors="coerce").to_numpy(dtype=float),
                          "longitude": pd.to_numeric(source.longitude, errors="coerce").to_numpy(dtype=float)})
    prediction_path = Path(job["prediction_path"])
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = prediction_path.with_suffix(prediction_path.suffix + ".tmp")
    frame.to_csv(temp_path, index=False)
    os.replace(temp_path, prediction_path)
    if job.get("model_path"):
        model_path = Path(job["model_path"])
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with model_path.open("wb") as stream:
            pickle.dump(result["estimator"], stream, protocol=pickle.HIGHEST_PROTOCOL)
    payload = {key: value for key, value in result.items() if key not in {"estimator", "predictions"}}
    payload.update({"method_id": job["method_id"], "track": job["track"], "month": job["month"],
                    "evaluation_rows": len(homes), "evaluation_ids_sha256": _id_hash(homes.ml_num),
                    "prediction_path": str(prediction_path),
                    "prediction_sha256": hashlib.sha256(prediction_path.read_bytes()).hexdigest()})
    return payload


def main() -> int:
    if len(sys.argv) != 3 or sys.argv[1] != "--job":
        print("usage: python -m experiments.model_optimization_v2.worker --job PATH", file=sys.stderr)
        return 2
    result = execute(Path(sys.argv[2]))
    print(json.dumps(result, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
