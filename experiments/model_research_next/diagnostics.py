from __future__ import annotations

import numpy as np
import pandas as pd


_KEYS = ["month", "ml_num"]


def _validate_prediction_frame(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    required = set(_KEYS + ["sold_price", "predicted_price"])
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing paired prediction columns: {sorted(missing)}")
    result = frame[_KEYS + ["sold_price", "predicted_price"]].copy()
    result["month"] = result["month"].astype(str)
    result["ml_num"] = result["ml_num"].astype(str)
    if result.duplicated(_KEYS).any():
        raise ValueError(f"{label} contains duplicate month/home keys")
    for column in ("sold_price", "predicted_price"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
        values = result[column].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError(f"{label} {column} values must be finite and positive")
    return result


def compute_paired_residuals(parent: pd.DataFrame, candidate: pd.DataFrame,
                             features: pd.DataFrame | None = None) -> pd.DataFrame:
    """Align two prediction sets by month and listing ID, then attach diagnostics."""
    parent_rows = _validate_prediction_frame(parent, "parent")
    candidate_rows = _validate_prediction_frame(candidate, "candidate")
    left = parent_rows.set_index(_KEYS).sort_index()
    right = candidate_rows.set_index(_KEYS).sort_index()
    if not left.index.equals(right.index):
        raise ValueError("parent and candidate month/home keys do not align")
    if not np.array_equal(left.sold_price.to_numpy(), right.sold_price.to_numpy()):
        raise ValueError("parent and candidate sale prices do not align")

    result = left.reset_index().rename(columns={
        "predicted_price": "parent_prediction",
    })
    result["candidate_prediction"] = right.predicted_price.to_numpy(dtype=float)
    actual = result.sold_price.to_numpy(dtype=float)
    result["parent_ape_percent"] = 100.0 * np.abs(result.parent_prediction.to_numpy(dtype=float) - actual) / actual
    result["candidate_ape_percent"] = 100.0 * np.abs(result.candidate_prediction.to_numpy(dtype=float) - actual) / actual
    result["parent_signed_error_percent"] = 100.0 * (result.parent_prediction.to_numpy(dtype=float) - actual) / actual
    result["candidate_signed_error_percent"] = 100.0 * (result.candidate_prediction.to_numpy(dtype=float) - actual) / actual

    if features is not None:
        if "ml_num" not in features.columns:
            raise ValueError("features must contain ml_num")
        feature_rows = features.copy()
        feature_rows["ml_num"] = feature_rows.ml_num.astype(str)
        if feature_rows.ml_num.duplicated().any():
            raise ValueError("features contain duplicate listing IDs")
        collisions = (set(feature_rows.columns) - {"ml_num"}).intersection(result.columns)
        if collisions:
            raise ValueError(f"feature columns collide with paired predictions: {sorted(collisions)}")
        result = result.merge(feature_rows, on="ml_num", how="left", validate="many_to_one", indicator=True)
        if not result["_merge"].eq("both").all():
            missing_ids = result.loc[result["_merge"].ne("both"), "ml_num"].head(5).tolist()
            raise ValueError(f"features do not cover every paired listing; examples: {missing_ids}")
        result = result.drop(columns="_merge")
    return result


def _score(group: pd.DataFrame) -> dict:
    parent_ape = group.parent_ape_percent.to_numpy(dtype=float)
    candidate_ape = group.candidate_ape_percent.to_numpy(dtype=float)
    parent_mdape = float(np.median(parent_ape))
    candidate_mdape = float(np.median(candidate_ape))
    parent_mape = float(np.mean(parent_ape))
    candidate_mape = float(np.mean(candidate_ape))
    return {
        "n": int(len(group)),
        "parent_mdape_percent": parent_mdape,
        "candidate_mdape_percent": candidate_mdape,
        "relative_mdape_improvement": (parent_mdape - candidate_mdape) / parent_mdape if parent_mdape else 0.0,
        "parent_mape_percent": parent_mape,
        "candidate_mape_percent": candidate_mape,
        "relative_mape_change": (candidate_mape - parent_mape) / parent_mape if parent_mape else 0.0,
        "parent_median_signed_error_percent": float(np.median(group.parent_signed_error_percent)),
        "candidate_median_signed_error_percent": float(np.median(group.candidate_signed_error_percent)),
    }


def summarize_residuals(rows: pd.DataFrame, group_columns: tuple[str, ...] = (),
                        min_group_size: int = 100) -> dict:
    """Summarize paired errors overall and by predeclared descriptive segments."""
    needed = {"parent_ape_percent", "candidate_ape_percent",
              "parent_signed_error_percent", "candidate_signed_error_percent", *group_columns}
    missing = needed - set(rows.columns)
    if missing:
        raise ValueError(f"residual rows are missing summary columns: {sorted(missing)}")
    if min_group_size < 1:
        raise ValueError("min_group_size must be positive")
    if not len(rows):
        raise ValueError("residual summary requires at least one paired row")
    report = {"overall": _score(rows), "by_group": {}}
    for column in group_columns:
        groups = {}
        for key, group in rows.groupby(column, dropna=False, sort=True):
            if len(group) < min_group_size:
                continue
            label = "<missing>" if pd.isna(key) else str(key)
            groups[label] = _score(group)
        report["by_group"][column] = groups
    return report
