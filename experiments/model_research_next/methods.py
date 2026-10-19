from __future__ import annotations

import numpy as np
import pandas as pd


def _month_ordinal(month: str) -> int:
    try:
        year_text, month_text = month.split("-")
        year, number = int(year_text), int(month_text)
        if not 1 <= number <= 12:
            raise ValueError
        return year * 12 + number
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid YYYY-MM month: {month!r}") from exc


def validate_fold_plan(screen_months, confirmation_months, final_months,
                       previously_evaluated) -> None:
    groups = {
        "screen": tuple(screen_months),
        "confirmation": tuple(confirmation_months),
        "final": tuple(final_months),
    }
    used = set(previously_evaluated)
    all_months = []
    for name, months in groups.items():
        if not months:
            raise ValueError(f"{name} months must be nonempty")
        ordinals = [_month_ordinal(month) for month in months]
        if ordinals != sorted(set(ordinals)):
            raise ValueError(f"{name} months must be unique and chronological")
        reused = set(months).intersection(used)
        if reused:
            raise ValueError(f"months were previously evaluated: {sorted(reused)}")
        all_months.extend(months)
    if len(all_months) != len(set(all_months)):
        raise ValueError("screen, confirmation, and final months must be disjoint")
    if _month_ordinal(groups["confirmation"][0]) <= _month_ordinal(groups["screen"][-1]):
        raise ValueError("confirmation months must follow all screen months")
    if _month_ordinal(groups["final"][0]) <= _month_ordinal(groups["confirmation"][-1]):
        raise ValueError("final months must follow all confirmation months")


def fit_oof_calibration(oof: pd.DataFrame, cutoff: str | pd.Timestamp,
                        group_columns: tuple[str, ...] = (), pseudocount: float = 200.0,
                        min_group: int = 100, clip: tuple[float, float] = (0.95, 1.05),
                        max_months: int | None = None, min_oof_months: int = 1) -> dict:
    required = {"ml_num", "date_sold", "sold_price", "predicted_price", *group_columns}
    missing = required - set(oof.columns)
    if missing:
        raise ValueError(f"OOF calibration is missing columns: {sorted(missing)}")
    if (pseudocount < 0 or min_group < 1 or min_oof_months < 1
            or (max_months is not None and max_months < min_oof_months)
            or len(clip) != 2 or not 0 < clip[0] <= clip[1]):
        raise ValueError("invalid shrinkage or clipping parameters")
    rows = oof.copy()
    rows["date_sold"] = pd.to_datetime(rows.date_sold, errors="coerce")
    rows = rows.loc[rows.date_sold < pd.Timestamp(cutoff)].copy()
    if rows.empty:
        raise ValueError("no strictly past OOF predictions are available")
    oof_months = []
    if max_months is not None or min_oof_months > 1:
        if "month" not in rows.columns:
            raise ValueError("month is required for a rolling or multi-month OOF calibration")
        rows["month"] = rows.month.astype(str)
        oof_months = sorted(rows.month.unique())
        if len(oof_months) < min_oof_months:
            raise ValueError(f"OOF calibration requires at least {min_oof_months} past months")
        if max_months is not None:
            oof_months = oof_months[-max_months:]
            rows = rows.loc[rows.month.isin(oof_months)].copy()
    else:
        if "month" in rows.columns:
            oof_months = sorted(rows.month.astype(str).unique())
    if rows.duplicated(["ml_num", "date_sold"]).any():
        raise ValueError("OOF calibration contains duplicate listing/date rows")
    actual = pd.to_numeric(rows.sold_price, errors="coerce").to_numpy(dtype=float)
    predicted = pd.to_numeric(rows.predicted_price, errors="coerce").to_numpy(dtype=float)
    if (not np.isfinite(actual).all() or not np.isfinite(predicted).all()
            or (actual <= 0).any() or (predicted <= 0).any()):
        raise ValueError("OOF calibration prices must be finite and positive")
    rows["log_ratio"] = np.log(actual / predicted)
    global_log = float(np.median(rows.log_ratio.to_numpy(dtype=float)))
    global_factor = float(np.clip(np.exp(global_log), clip[0], clip[1]))
    factors = {}
    if group_columns:
        for key, group in rows.groupby(list(group_columns), dropna=False, sort=True):
            if len(group) < min_group:
                continue
            key_tuple = key if isinstance(key, tuple) else (key,)
            encoded = "|".join("<missing>" if pd.isna(value) else str(value) for value in key_tuple)
            observed_log = float(np.median(group.log_ratio.to_numpy(dtype=float)))
            shrink = len(group) / (len(group) + pseudocount) if pseudocount else 1.0
            factor = float(np.clip(np.exp(global_log + shrink * (observed_log - global_log)), clip[0], clip[1]))
            factors[encoded] = {"factor": factor, "n": int(len(group))}
    return {
        "global_factor": global_factor,
        "group_factors": factors,
        "group_columns": list(group_columns),
        "n_oof": int(len(rows)),
        "max_oof_date": str(rows.date_sold.max().date()),
        "oof_months": oof_months,
        "cutoff": str(pd.Timestamp(cutoff).date()),
        "pseudocount": float(pseudocount),
        "min_group": int(min_group),
        "max_months": max_months,
        "min_oof_months": int(min_oof_months),
        "clip": [float(clip[0]), float(clip[1])],
    }


def apply_oof_calibration(predictions, homes: pd.DataFrame, fitted: dict) -> np.ndarray:
    prediction = np.asarray(predictions, dtype=float)
    if prediction.ndim != 1 or len(prediction) != len(homes) or not np.isfinite(prediction).all() or (prediction <= 0).any():
        raise ValueError("predictions must be a finite positive vector aligned with homes")
    columns = fitted["group_columns"]
    missing = set(columns) - set(homes.columns)
    if missing:
        raise ValueError(f"homes are missing calibration groups: {sorted(missing)}")
    factors = np.full(len(homes), float(fitted["global_factor"]), dtype=float)
    if columns:
        for idx, values in enumerate(homes[columns].itertuples(index=False, name=None)):
            key = "|".join("<missing>" if pd.isna(value) else str(value) for value in values)
            factors[idx] = fitted["group_factors"].get(key, {}).get("factor", fitted["global_factor"])
    result = prediction * factors
    if not np.isfinite(result).all() or (result <= 0).any():
        raise ValueError("OOF calibration produced invalid prices")
    return result


def _prediction_band(values, edges: tuple[float, ...] | list[float]) -> np.ndarray:
    prediction = np.asarray(values, dtype=float)
    if prediction.ndim != 1 or not np.isfinite(prediction).all() or (prediction <= 0).any():
        raise ValueError("predicted-price bands require a finite positive vector")
    labels = [f"P{index + 1}" for index in range(len(edges) + 1)]
    return pd.cut(prediction, bins=[-np.inf, *edges, np.inf], labels=labels,
                  include_lowest=True).astype(str)


def fit_oof_prediction_band_calibration(
    oof: pd.DataFrame,
    cutoff: str | pd.Timestamp,
    group_columns: tuple[str, ...] = (),
    n_bands: int = 4,
    pseudocount: float = 200.0,
    min_group: int = 100,
    clip: tuple[float, float] = (0.95, 1.05),
    max_months: int | None = None,
    min_oof_months: int = 1,
) -> dict:
    required = {"date_sold", "predicted_price"}
    missing = required - set(oof.columns)
    if missing:
        raise ValueError(f"prediction-band calibration is missing columns: {sorted(missing)}")
    if n_bands < 2:
        raise ValueError("prediction-band calibration requires at least two requested bands")
    past = oof.copy()
    past["date_sold"] = pd.to_datetime(past.date_sold, errors="coerce")
    past = past.loc[past.date_sold < pd.Timestamp(cutoff)].copy()
    if past.empty:
        raise ValueError("no strictly past OOF predictions are available")
    if max_months is not None or min_oof_months > 1:
        if "month" not in past.columns:
            raise ValueError("month is required for a rolling or multi-month OOF calibration")
        months = sorted(past.month.astype(str).unique())
        if len(months) < min_oof_months:
            raise ValueError(f"OOF calibration requires at least {min_oof_months} past months")
        if max_months is not None:
            months = months[-max_months:]
            past = past.loc[past.month.astype(str).isin(months)].copy()
    predictions = pd.to_numeric(past.predicted_price, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(predictions).all() or (predictions <= 0).any():
        raise ValueError("OOF predicted prices must be finite and positive")
    quantiles = np.linspace(0.0, 1.0, n_bands + 1)[1:-1]
    edges = tuple(float(value) for value in np.unique(np.quantile(predictions, quantiles)))
    if any(not np.isfinite(value) for value in edges):
        raise ValueError("prediction-band boundaries must be finite")
    prepared = oof.copy()
    prepared["predicted_price_band"] = _prediction_band(prepared.predicted_price, edges)
    columns = (*group_columns, "predicted_price_band")
    fitted = fit_oof_calibration(
        prepared,
        cutoff=cutoff,
        group_columns=columns,
        pseudocount=pseudocount,
        min_group=min_group,
        clip=clip,
        max_months=max_months,
        min_oof_months=min_oof_months,
    )
    fitted["band_edges"] = list(edges)
    fitted["requested_bands"] = int(n_bands)
    return fitted


def apply_oof_prediction_band_calibration(predictions, homes: pd.DataFrame, fitted: dict) -> np.ndarray:
    prepared = homes.copy()
    prepared["predicted_price_band"] = _prediction_band(predictions, fitted["band_edges"])
    return apply_oof_calibration(predictions, prepared, fitted)


def fit_oof_blend_weight(oof: pd.DataFrame, weights: tuple[float, ...] = tuple(np.linspace(0.0, 1.0, 21))) -> float:
    required = {"month", "ml_num", "sold_price", "parent_prediction", "challenger_prediction"}
    missing = required - set(oof.columns)
    if missing:
        raise ValueError(f"OOF blend is missing columns: {sorted(missing)}")
    if oof.duplicated(["month", "ml_num"]).any():
        raise ValueError("OOF blend contains duplicate month/home keys")
    actual = pd.to_numeric(oof.sold_price, errors="coerce").to_numpy(dtype=float)
    parent = pd.to_numeric(oof.parent_prediction, errors="coerce").to_numpy(dtype=float)
    challenger = pd.to_numeric(oof.challenger_prediction, errors="coerce").to_numpy(dtype=float)
    if (not np.isfinite(actual).all() or not np.isfinite(parent).all() or not np.isfinite(challenger).all()
            or (actual <= 0).any() or (parent <= 0).any() or (challenger <= 0).any()):
        raise ValueError("OOF blend inputs must be finite positive dollar prices")
    grid = tuple(float(weight) for weight in weights)
    if not grid or any(not 0.0 <= weight <= 1.0 for weight in grid):
        raise ValueError("blend weights must be a nonempty grid in [0, 1]")
    scores = []
    for weight in grid:
        blended = (1.0 - weight) * parent + weight * challenger
        ape = np.abs(blended - actual) / actual
        scores.append((float(np.median(ape)), weight))
    return min(scores, key=lambda item: (item[0], item[1]))[1]
