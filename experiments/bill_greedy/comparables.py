from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


FEATURE_COLUMNS = (
    "nn_res_mean10", "nn_dist_mean10_km", "nn_age_mean10_months",
    "nn_res_idw10", "nn_res_med10", "nn_res_med30", "nn_n30",
)


def _query_cutoffs(queries: pd.DataFrame, cutoff, training_rows: bool) -> pd.Series:
    if training_rows:
        if "date_sold" not in queries:
            raise ValueError("training comparables require query sale dates")
        return pd.to_datetime(queries.date_sold, errors="raise").dt.to_period("M").dt.start_time
    if cutoff is None:
        raise ValueError("evaluation comparables require a fixed cutoff")
    if isinstance(cutoff, pd.Series):
        if not cutoff.index.equals(queries.index):
            raise ValueError("row-specific cutoffs must align with query rows")
        return pd.to_datetime(cutoff, errors="raise")
    return pd.Series(pd.Timestamp(cutoff), index=queries.index)


def nearest_comparable_features(sales: pd.DataFrame, queries: pd.DataFrame, *,
                                cutoff: str | pd.Timestamp | pd.Series | None = None,
                                training_rows: bool = False,
                                window_months: int = 24,
                                k_short: int = 10,
                                k_long: int = 30) -> pd.DataFrame:
    """Build past-only same-type nearby-sale residuals for each query.

    A comparable sale's residual is its log price per square foot minus the
    median for its property type in the previous complete calendar month. The
    neighbor pool is restricted to sales strictly before each query cutoff and
    inside the trailing window. Query outcomes are never read.
    """
    if window_months < 1 or k_short < 1 or k_long < k_short:
        raise ValueError("window and neighbor counts must be positive and ordered")
    required_sales = {"date_sold", "sold_price", "house_area", "house_type_name", "latitude", "longitude"}
    required_queries = {"house_type_name", "latitude", "longitude"}
    if missing := required_sales - set(sales.columns):
        raise ValueError(f"sales are missing comparable fields: {sorted(missing)}")
    if missing := required_queries - set(queries.columns):
        raise ValueError(f"queries are missing comparable fields: {sorted(missing)}")

    query_cutoffs = _query_cutoffs(queries, cutoff, training_rows)
    if query_cutoffs.isna().any():
        raise ValueError("query cutoffs must be valid dates")

    output = np.full((len(queries), len(FEATURE_COLUMNS)), np.nan, dtype=float)
    output[:, -1] = 0.0
    if queries.empty:
        return pd.DataFrame(output, columns=FEATURE_COLUMNS, index=queries.index)

    max_cutoff = query_cutoffs.max()
    sale_dates = pd.to_datetime(sales.date_sold, errors="coerce")
    sale_price = pd.to_numeric(sales.sold_price, errors="coerce").to_numpy(dtype=float)
    from experiments.model_optimization_v2.learners import _area
    area = _area(sales)
    log_ppsf = np.full(len(sales), np.nan, dtype=float)
    valid_price_area = np.isfinite(sale_price) & (sale_price > 0) & np.isfinite(area) & (area > 0)
    log_ppsf[valid_price_area] = np.log(sale_price[valid_price_area] / area[valid_price_area])
    sale_types = sales.house_type_name.astype("string").fillna("").astype(str).to_numpy()
    sale_lat = pd.to_numeric(sales.latitude, errors="coerce").to_numpy(dtype=float)
    sale_lon = pd.to_numeric(sales.longitude, errors="coerce").to_numpy(dtype=float)
    sale_x = sale_lon * 111.32 * np.cos(np.radians(43.7))
    sale_y = sale_lat * 110.57

    eligible = (sale_dates.notna().to_numpy() & (sale_dates.to_numpy() < max_cutoff.to_datetime64())
                & np.isfinite(log_ppsf) & np.isfinite(sale_x) & np.isfinite(sale_y)
                & (sale_types != ""))
    sale_months = sale_dates.dt.to_period("M")
    month_rows = pd.DataFrame({"type": sale_types, "month": sale_months,
                               "log_ppsf": log_ppsf}).loc[eligible]
    monthly_median = month_rows.groupby(["type", "month"], observed=True).log_ppsf.median()
    lagged_type_index = np.asarray([
        monthly_median.get((type_name, month - 1), np.nan)
        if type_name and not pd.isna(month) else np.nan
        for type_name, month in zip(sale_types, sale_months)
    ], dtype=float)
    residual = log_ppsf - lagged_type_index

    query_types = queries.house_type_name.astype("string").fillna("").astype(str).to_numpy()
    query_lat = pd.to_numeric(queries.latitude, errors="coerce").to_numpy(dtype=float)
    query_lon = pd.to_numeric(queries.longitude, errors="coerce").to_numpy(dtype=float)
    query_x = query_lon * 111.32 * np.cos(np.radians(43.7))
    query_y = query_lat * 110.57
    query_xy = np.column_stack([query_x, query_y])

    for cutoff_at in pd.unique(query_cutoffs):
        query_positions = np.flatnonzero(query_cutoffs.to_numpy() == cutoff_at)
        cutoff_at = pd.Timestamp(cutoff_at)
        lower_bound = cutoff_at - pd.DateOffset(months=window_months)
        for type_name in pd.unique(query_types[query_positions]):
            if not type_name:
                continue
            q_positions = query_positions[query_types[query_positions] == type_name]
            q_positions = q_positions[np.isfinite(query_xy[q_positions]).all(axis=1)]
            if not len(q_positions):
                continue
            pool_mask = (eligible & (sale_types == type_name)
                         & (sale_dates.to_numpy() < cutoff_at.to_datetime64())
                         & (sale_dates.to_numpy() >= lower_bound.to_datetime64())
                         & np.isfinite(residual))
            pool_positions = np.flatnonzero(pool_mask)
            if not len(pool_positions):
                continue
            pool_xy = np.column_stack([sale_x[pool_positions], sale_y[pool_positions]])
            tree = cKDTree(pool_xy)
            neighbor_count = min(k_long, len(pool_positions))
            distances, neighbor_indices = tree.query(query_xy[q_positions], k=neighbor_count)
            distances = np.asarray(distances, dtype=float).reshape(len(q_positions), neighbor_count)
            neighbor_indices = np.asarray(neighbor_indices, dtype=np.int64).reshape(len(q_positions), neighbor_count)
            selected_residuals = residual[pool_positions][neighbor_indices]
            selected_dates = sale_dates.iloc[pool_positions].to_numpy()[neighbor_indices]
            ages = ((cutoff_at.to_datetime64() - selected_dates) / np.timedelta64(1, "D")) / 30.44
            output[q_positions, -1] = neighbor_count
            if neighbor_count < k_short:
                continue
            top_dist = distances[:, :k_short]
            top_res = selected_residuals[:, :k_short]
            top_age = ages[:, :k_short]
            weights = 1.0 / (top_dist + 0.05)
            output[q_positions, 0] = top_res.mean(axis=1)
            output[q_positions, 1] = top_dist.mean(axis=1)
            output[q_positions, 2] = top_age.mean(axis=1)
            output[q_positions, 3] = np.average(top_res, axis=1, weights=weights)
            output[q_positions, 4] = np.median(top_res, axis=1)
            if neighbor_count >= k_long:
                output[q_positions, 5] = np.median(selected_residuals, axis=1)

    result = pd.DataFrame(output, columns=FEATURE_COLUMNS, index=queries.index)
    return result
