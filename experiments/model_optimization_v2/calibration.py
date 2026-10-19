from __future__ import annotations

import numpy as np
import pandas as pd


def _past_oof(oof: pd.DataFrame, cutoff: str | pd.Timestamp) -> pd.DataFrame:
    required = {"ml_num", "date_sold", "sold_price", "predicted_price", "house_type_name",
                "latitude", "longitude"}
    missing = required - set(oof.columns)
    if missing:
        raise ValueError(f"OOF calibration history is missing {sorted(missing)}")
    rows = oof.copy()
    rows["date_sold"] = pd.to_datetime(rows.date_sold)
    rows = rows.loc[rows.date_sold < pd.Timestamp(cutoff)].copy()
    if rows.empty:
        raise ValueError("no strictly past out-of-fold rows are available for calibration")
    actual = pd.to_numeric(rows.sold_price, errors="coerce").to_numpy(float)
    prediction = pd.to_numeric(rows.predicted_price, errors="coerce").to_numpy(float)
    if not np.isfinite(actual).all() or not np.isfinite(prediction).all() or (actual <= 0).any() or (prediction <= 0).any():
        raise ValueError("OOF calibration prices must be finite and positive")
    rows["ratio"] = actual / prediction
    rows["log_residual"] = np.log(rows.ratio)
    return rows


def fit_multiplier(oof: pd.DataFrame, cutoff: str | pd.Timestamp) -> dict:
    rows = _past_oof(oof, cutoff)
    factor = float(np.clip(np.median(rows.ratio), 0.95, 1.05))
    return {"factor": factor, "n": int(len(rows)), "max_oof_date": str(rows.date_sold.max().date())}


def apply_multiplier_calibration(predictions, params: dict) -> np.ndarray:
    result = np.asarray(predictions, dtype=float) * float(params["factor"])
    if not np.isfinite(result).all() or (result <= 0).any():
        raise ValueError("global calibration produced invalid predictions")
    return result


def fit_type_multiplier(oof: pd.DataFrame, cutoff: str | pd.Timestamp,
                        pseudocount: float = 200.0, min_group: int = 100) -> dict:
    rows = _past_oof(oof, cutoff)
    global_factor = float(np.clip(np.median(rows.ratio), 0.95, 1.05))
    groups = {}
    for house_type, group in rows.groupby("house_type_name", dropna=False):
        n = len(group)
        key = "<missing>" if pd.isna(house_type) else str(house_type)
        if n < min_group:
            continue
        observed = float(np.mean(group.ratio))
        factor = (n * observed + pseudocount * global_factor) / (n + pseudocount)
        groups[key] = {"factor": factor, "n": int(n)}
    return {"global_factor": global_factor, "groups": groups, "n": int(len(rows)),
            "max_oof_date": str(rows.date_sold.max().date()), "pseudocount": pseudocount,
            "min_group": min_group}


def apply_type_multiplier(predictions, homes: pd.DataFrame, params: dict) -> np.ndarray:
    types = homes.house_type_name.astype("string").fillna("<missing>").astype(str).to_numpy()
    factors = np.asarray([params["groups"].get(t, {}).get("factor", params["global_factor"])
                          for t in types], dtype=float)
    result = np.asarray(predictions, dtype=float) * factors
    if not np.isfinite(result).all() or (result <= 0).any():
        raise ValueError("type calibration produced invalid predictions")
    return result


def apply_local_residual(predictions, homes: pd.DataFrame, oof: pd.DataFrame,
                         cutoff: str | pd.Timestamp, neighbors: int = 30,
                         days: int = 365, pseudocount: float = 50.0) -> tuple[np.ndarray, dict]:
    rows = _past_oof(oof, cutoff)
    rows["latitude"] = pd.to_numeric(rows.latitude, errors="coerce")
    rows["longitude"] = pd.to_numeric(rows.longitude, errors="coerce")
    q = homes.copy()
    q["date_listed"] = pd.to_datetime(q.date_listed)
    q["latitude"] = pd.to_numeric(q.latitude, errors="coerce")
    q["longitude"] = pd.to_numeric(q.longitude, errors="coerce")
    scale_lat = float(rows.latitude.quantile(0.75) - rows.latitude.quantile(0.25))
    scale_lon = float(rows.longitude.quantile(0.75) - rows.longitude.quantile(0.25))
    if not np.isfinite(scale_lat) or scale_lat <= 0:
        scale_lat = 1.0
    if not np.isfinite(scale_lon) or scale_lon <= 0:
        scale_lon = 1.0
    factors = np.ones(len(q), dtype=float)
    coverage = 0
    for idx, row in enumerate(q.itertuples(index=False)):
        row_dict = row._asdict()
        eligible = rows.loc[(rows.date_sold < row_dict["date_listed"]) &
                            (rows.date_sold >= row_dict["date_listed"] - pd.Timedelta(days=days)) &
                            (rows.house_type_name.fillna("<missing>").astype(str).eq(
                                "<missing>" if pd.isna(row_dict["house_type_name"]) else str(row_dict["house_type_name"])))].copy()
        if not len(eligible) or pd.isna(row_dict["latitude"]) or pd.isna(row_dict["longitude"]):
            continue
        eligible = eligible.dropna(subset=["latitude", "longitude"])
        if eligible.empty:
            continue
        dlat = (eligible.latitude.to_numpy(float) - float(row_dict["latitude"])) / scale_lat
        dlon = (eligible.longitude.to_numpy(float) - float(row_dict["longitude"])) / scale_lon
        distance = np.hypot(dlat, dlon)
        nearest = np.argsort(distance, kind="stable")[:neighbors]
        n = len(nearest)
        correction = float(np.mean(eligible.log_residual.to_numpy(float)[nearest]))
        factors[idx] = np.exp(correction * n / (n + pseudocount))
        coverage += 1
    result = np.asarray(predictions, dtype=float) * factors
    if not np.isfinite(result).all() or (result <= 0).any():
        raise ValueError("local residual calibration produced invalid predictions")
    return result, {"n_oof": int(len(rows)), "max_oof_date": str(rows.date_sold.max().date()),
                   "coverage": coverage, "coverage_rate": coverage / max(len(q), 1),
                   "neighbors": neighbors, "days": days, "pseudocount": pseudocount}
