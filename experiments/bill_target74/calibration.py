from __future__ import annotations

import numpy as np
import pandas as pd


_GROUP_COLUMNS = ("house_type_name", "municipality")
_NUMERIC_COLUMNS = ("house_area", "bedrooms", "washrooms")
_HISTORY_COLUMNS = {
    "date_sold", "house_type_name", "municipality", "house_area", "bedrooms",
    "washrooms", "sold_price", "parent_prediction",
}
_QUERY_COLUMNS = {
    "house_type_name", "municipality", "house_area", "bedrooms", "washrooms",
    "parent_prediction",
}


def _numeric_features(frame: pd.DataFrame) -> np.ndarray:
    from experiments.model_optimization_v2.learners import _area

    area = np.asarray(pd.to_numeric(_area(frame), errors="coerce"), dtype=float)
    area[~np.isfinite(area) | (area <= 0.0)] = np.nan
    log_area = np.full(len(frame), np.nan, dtype=float)
    valid_area = np.isfinite(area)
    log_area[valid_area] = np.log(area[valid_area])
    bedrooms = pd.to_numeric(frame["bedrooms"], errors="coerce").to_numpy(dtype=float)
    washrooms = pd.to_numeric(frame["washrooms"], errors="coerce").to_numpy(dtype=float)
    bedrooms[~np.isfinite(bedrooms)] = np.nan
    washrooms[~np.isfinite(washrooms)] = np.nan
    return np.column_stack((log_area, bedrooms, washrooms))


def _structure_distance(history: pd.DataFrame, query: pd.Series) -> np.ndarray:
    history_numeric = _numeric_features(history)
    query_numeric = _numeric_features(query.to_frame().T)
    if not np.isfinite(query_numeric[0]).any():
        raise ValueError("query has no usable structural measurements")

    medians = np.zeros(history_numeric.shape[1], dtype=float)
    for column in range(history_numeric.shape[1]):
        values = history_numeric[:, column]
        finite = values[np.isfinite(values)]
        medians[column] = float(np.median(finite)) if len(finite) else 0.0

    history_missing = ~np.isfinite(history_numeric)
    query_missing = ~np.isfinite(query_numeric)
    history_filled = np.where(history_missing, medians, history_numeric)
    query_filled = np.where(query_missing, medians, query_numeric)

    means = history_filled.mean(axis=0)
    scales = history_filled.std(axis=0)
    scales[~np.isfinite(scales) | (scales < 1e-8)] = 1.0
    history_scaled = (history_filled - means) / scales
    query_scaled = (query_filled - means) / scales

    numeric_distance = np.mean((history_scaled - query_scaled[0]) ** 2, axis=1)
    missing_distance = np.mean((history_missing != query_missing[0]), axis=1)
    return np.sqrt(numeric_distance + missing_distance)


def build_oof_neighbor_features(
    history_oof: pd.DataFrame,
    queries: pd.DataFrame,
    cutoff: str | pd.Timestamp,
    *,
    window_months: int = 24,
    half_life_months: float = 12.0,
    max_neighbors: int = 30,
    min_neighbors: int = 10,
) -> pd.DataFrame:
    """Estimate a shrunk log-price correction from strictly past OOF residuals.

    ``history_oof`` must contain predictions made before each historical sale
    month. ``queries`` must have hidden sale dates and prices removed. This
    function filters again at the outer prediction cutoff before calculating
    any residual contribution.
    """
    missing_history = sorted(_HISTORY_COLUMNS - set(history_oof.columns))
    missing_queries = sorted(_QUERY_COLUMNS - set(queries.columns))
    if missing_history:
        raise ValueError(f"past OOF history is missing columns: {missing_history}")
    if missing_queries:
        raise ValueError(f"queries are missing required columns: {missing_queries}")
    leaked = {"sold_price", "date_sold"}.intersection(queries.columns)
    if leaked:
        raise ValueError(f"query outcome columns are forbidden: {sorted(leaked)}")
    if window_months < 1 or half_life_months <= 0.0 or max_neighbors < 1:
        raise ValueError("window, half-life, and neighbor count must be positive")
    if min_neighbors < 1 or min_neighbors > max_neighbors:
        raise ValueError("min_neighbors must be between 1 and max_neighbors")

    cutoff_at = pd.Timestamp(cutoff)
    if cutoff_at.day != 1 or cutoff_at != cutoff_at.normalize():
        raise ValueError("cutoff must be the first day of a prediction month")

    output = pd.DataFrame({
        "neighbor_count": np.zeros(len(queries), dtype=np.int64),
        "effective_n": np.zeros(len(queries), dtype=float),
        "weighted_log_residual": np.zeros(len(queries), dtype=float),
        "correction_log": np.zeros(len(queries), dtype=float),
        "covered": np.zeros(len(queries), dtype=bool),
    }, index=queries.index)
    if queries.empty or history_oof.empty:
        return output

    history = history_oof.copy()
    history["date_sold"] = pd.to_datetime(history.date_sold, errors="coerce")
    sold = pd.to_numeric(history.sold_price, errors="coerce").to_numpy(dtype=float)
    parent = pd.to_numeric(history.parent_prediction, errors="coerce").to_numpy(dtype=float)
    valid_residual = (history.date_sold.notna().to_numpy() & np.isfinite(sold) & (sold > 0.0)
                      & np.isfinite(parent) & (parent > 0.0))
    history["_log_residual"] = np.nan
    history.loc[valid_residual, "_log_residual"] = np.log(sold[valid_residual] / parent[valid_residual])

    lower_bound = cutoff_at - pd.DateOffset(months=window_months)
    history_dates = pd.to_datetime(history.date_sold, errors="coerce")
    eligible = (valid_residual & (history_dates < cutoff_at).to_numpy()
                & (history_dates >= lower_bound).to_numpy())
    history = history.loc[eligible].copy()
    if history.empty:
        return output

    history_group = history[list(_GROUP_COLUMNS)].astype("string").fillna("")
    query_group = queries[list(_GROUP_COLUMNS)].astype("string").fillna("")
    history = history.assign(_group_type=history_group.iloc[:, 0].to_numpy(),
                             _group_municipality=history_group.iloc[:, 1].to_numpy())

    for position, (_, query) in enumerate(queries.iterrows()):
        type_name = str(query_group.iloc[position, 0])
        municipality = str(query_group.iloc[position, 1])
        if not type_name or not municipality:
            continue
        same_group = history.loc[(history._group_type == type_name)
                                 & (history._group_municipality == municipality)].copy()
        if same_group.empty:
            continue

        distance = _structure_distance(same_group, query)
        nearest = np.argsort(distance, kind="stable")[:max_neighbors]
        neighbors = same_group.iloc[nearest]
        distance = distance[nearest]
        dates = pd.to_datetime(neighbors.date_sold).to_numpy(dtype="datetime64[ns]")
        age_months = ((cutoff_at.to_datetime64() - dates) / np.timedelta64(1, "D")) / 30.44
        weights = (1.0 / (distance + 0.25)) * np.power(0.5, age_months / half_life_months)
        residuals = neighbors._log_residual.to_numpy(dtype=float)
        total_weight = float(weights.sum())
        if not np.isfinite(total_weight) or total_weight <= 0.0:
            continue

        effective_n = total_weight**2 / float(np.square(weights).sum())
        weighted_residual = float(np.dot(weights, residuals) / total_weight)
        count = len(neighbors)
        clipped = float(np.clip(weighted_residual, -0.10, 0.10))
        correction = 0.25 * effective_n / (effective_n + 30.0) * clipped
        output.iloc[position] = (count, effective_n, weighted_residual, correction,
                                 count >= min_neighbors)

    output["neighbor_count"] = output.neighbor_count.astype(int)
    output["covered"] = output.covered.astype(bool)
    return output


def apply_oof_correction(parent_prediction: np.ndarray, features: pd.DataFrame) -> np.ndarray:
    """Apply the preregistered capped/shrunk log correction to parent prices."""
    parent = np.asarray(parent_prediction, dtype=float)
    if parent.ndim != 1 or len(parent) != len(features):
        raise ValueError("parent predictions and OOF features must align row by row")
    if not np.isfinite(parent).all() or (parent <= 0.0).any():
        raise ValueError("parent predictions must be finite positive prices")
    if {"correction_log", "covered"} - set(features.columns):
        raise ValueError("OOF correction features are incomplete")
    correction = pd.to_numeric(features.correction_log, errors="coerce").to_numpy(dtype=float)
    covered = features.covered.to_numpy(dtype=bool)
    if not np.isfinite(correction).all():
        raise ValueError("OOF log corrections must be finite")
    return parent * np.exp(np.where(covered, correction, 0.0))
