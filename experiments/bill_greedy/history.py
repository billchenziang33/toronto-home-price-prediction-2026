from __future__ import annotations

import numpy as np
import pandas as pd


def add_asof_history(sales: pd.DataFrame, queries: pd.DataFrame, *, key: str,
                     prefix: str, cutoff_column: str = "cutoff",
                     decay_half_life_months: float | None = None,
                     include_residual_std: bool = False) -> pd.DataFrame:
    """Add same-key sales summaries strictly before each query cutoff."""
    required_sales = {key, "date_sold", "lp", "res"}
    required_queries = {key, cutoff_column}
    if required_sales - set(sales) or required_queries - set(queries):
        raise ValueError("history inputs are missing key, cutoff, or sale fields")
    if not prefix.endswith("_"):
        raise ValueError("history feature prefix must end in an underscore")

    original_index = queries.index
    left = queries[[key, cutoff_column]].copy().reset_index(drop=True)
    left[cutoff_column] = pd.to_datetime(left[cutoff_column], errors="coerce").astype("datetime64[ns]")
    if left[cutoff_column].isna().any():
        raise ValueError("history query cutoffs must be valid timestamps")
    left["_query_order"] = np.arange(len(left), dtype=np.int64)

    right = sales[[key, "date_sold", "lp", "res"]].copy()
    right["date_sold"] = pd.to_datetime(right.date_sold, errors="coerce").astype("datetime64[ns]")
    right["lp"] = pd.to_numeric(right.lp, errors="coerce")
    right["res"] = pd.to_numeric(right.res, errors="coerce")
    right = right.loc[right.date_sold.notna() & right.lp.notna()].copy()
    right = right.sort_values("date_sold", kind="mergesort")

    if right.empty:
        result = pd.DataFrame(index=original_index)
        result[f"{prefix}n"] = 0.0
        for feature in ("res_n", "res_mean", "last_lp", "last_res", "age"):
            result[f"{prefix}{feature}"] = np.nan
        result[f"{prefix}res_n"] = 0.0
        if include_residual_std:
            result[f"{prefix}res_std"] = np.nan
        if decay_half_life_months is not None:
            result[f"{prefix}res_mean_decay"] = np.nan
        return result

    grouped = right.groupby(key, dropna=False, sort=False)
    right[f"{prefix}n"] = grouped.cumcount() + 1
    residual_valid = right.res.notna().astype(np.int64)
    right[f"{prefix}res_n"] = residual_valid.groupby(right[key], dropna=False, sort=False).cumsum()
    residual_sum = right.res.fillna(0.0).groupby(right[key], dropna=False, sort=False).cumsum()
    right[f"{prefix}res_mean"] = residual_sum.div(right[f"{prefix}res_n"].replace(0, np.nan))
    if include_residual_std:
        residual_sq_sum = right.res.fillna(0.0).pow(2).groupby(right[key], dropna=False, sort=False).cumsum()
        mean_square = residual_sq_sum.div(right[f"{prefix}res_n"].replace(0, np.nan))
        right[f"{prefix}res_std"] = (mean_square - right[f"{prefix}res_mean"].pow(2)).clip(lower=0).pow(0.5)
    right[f"{prefix}last_lp"] = right.lp
    right[f"{prefix}last_res"] = right.res
    right["_last_date"] = right.date_sold

    right = right.rename(columns={"date_sold": "_sale_date"})
    right = right.drop(columns=["lp"])
    left = left.sort_values(cutoff_column, kind="mergesort")
    right = right.sort_values("_sale_date", kind="mergesort")
    merged = pd.merge_asof(
        left,
        right,
        left_on=cutoff_column,
        right_on="_sale_date",
        by=key,
        direction="backward",
        allow_exact_matches=False,
    )
    merged[f"{prefix}age"] = (merged[cutoff_column] - merged["_last_date"]).dt.total_seconds() / (86400.0 * 30.44)
    feature_columns = [f"{prefix}{name}" for name in ("n", "res_n", "res_mean", "last_lp", "last_res", "age")]
    merged[f"{prefix}n"] = merged[f"{prefix}n"].fillna(0.0)
    merged[f"{prefix}res_n"] = merged[f"{prefix}res_n"].fillna(0.0)
    result = merged.sort_values("_query_order", kind="mergesort")[feature_columns]
    result.index = original_index
    if include_residual_std:
        result[f"{prefix}res_std"] = merged.sort_values("_query_order", kind="mergesort")[f"{prefix}res_std"].to_numpy()
    if decay_half_life_months is not None:
        if not np.isfinite(decay_half_life_months) or decay_half_life_months <= 0:
            raise ValueError("residual decay half-life must be positive")
        decay_rate = np.log(2.0) / (decay_half_life_months * 30.44)
        records = {}
        for group_key, rows in right.groupby(key, dropna=False, sort=False):
            date_days = (rows._sale_date.astype("int64").to_numpy(dtype=np.float64) / 86_400_000_000_000.0)
            base = float(date_days[0])
            weights = np.exp(decay_rate * (date_days - base))
            valid = rows.res.notna().to_numpy()
            weighted_n = np.concatenate(([0.0], np.cumsum(weights * valid)))
            weighted_sum = np.concatenate(([0.0], np.cumsum(weights * rows.res.fillna(0).to_numpy(dtype=float))))
            records[group_key] = (rows._sale_date.to_numpy(dtype="datetime64[ns]"), weighted_n, weighted_sum)
        decayed = np.full(len(left), np.nan, dtype=float)
        ordered_queries = left.sort_values("_query_order", kind="mergesort")
        for position, row in enumerate(ordered_queries.itertuples(index=False)):
            values = row._asdict()
            item = records.get(values[key])
            if item is None:
                continue
            dates, weighted_n, weighted_sum = item
            end = int(np.searchsorted(dates, np.datetime64(values[cutoff_column]), side="left"))
            if weighted_n[end] > 0:
                decayed[position] = weighted_sum[end] / weighted_n[end]
        result[f"{prefix}res_mean_decay"] = decayed
    return result


def _coordinate_key(frame: pd.DataFrame) -> pd.Series:
    lat = pd.to_numeric(frame.latitude, errors="coerce").round(6)
    lon = pd.to_numeric(frame.longitude, errors="coerce").round(6)
    valid = lat.notna() & lon.notna()
    result = lat.astype("string") + "_" + lon.astype("string")
    return result.where(valid, pd.NA)


def _unit_key(frame: pd.DataFrame, coordinate_key: pd.Series) -> pd.Series:
    pieces = pd.DataFrame({
        "coordinate": coordinate_key.astype("string"),
        "type": frame.house_type_name.astype("string"),
        "bedrooms": frame.bedrooms.astype("string"),
        "washrooms": frame.washrooms.astype("string"),
        "area": frame.house_area.astype("string"),
    }, index=frame.index).fillna("<missing>")
    key = pieces.astype(str).agg("|".join, axis=1).astype("string")
    return key.where(coordinate_key.notna(), pd.NA)


def _sale_history(sales: pd.DataFrame) -> pd.DataFrame:
    required = {"date_sold", "sold_price", "house_area", "house_type_name"}
    if required - set(sales):
        raise ValueError(f"sale history is missing fields: {sorted(required - set(sales))}")
    from model_candidate_v2.features import square_feet

    rows = sales.copy()
    rows["date_sold"] = pd.to_datetime(rows.date_sold, errors="coerce").astype("datetime64[ns]")
    price = pd.to_numeric(rows.sold_price, errors="coerce")
    area = pd.to_numeric(square_feet(rows.house_area), errors="coerce")
    valid = rows.date_sold.notna() & np.isfinite(price) & (price > 0) & np.isfinite(area) & (area > 0)
    rows = rows.loc[valid].copy()
    price = price.loc[valid].to_numpy(dtype=float)
    area = area.loc[valid].to_numpy(dtype=float)
    rows["lp"] = np.log(price)
    rows["lppsf"] = rows.lp.to_numpy(dtype=float) - np.log(area)
    rows["_month"] = rows.date_sold.dt.to_period("M")
    index = rows.groupby(["house_type_name", "_month"], dropna=False).lppsf.median().rename("_lag_index").reset_index()
    index["_month"] = index["_month"] + 1
    rows = rows.merge(index, on=["house_type_name", "_month"], how="left", sort=False)
    rows["res"] = rows.lppsf - rows._lag_index
    rows["b_key"] = _coordinate_key(rows)
    rows["u_key"] = _unit_key(rows, rows.b_key)
    return rows


def _lagged_type_market_index(sales: pd.DataFrame, queries: pd.DataFrame,
                              query_cutoff: pd.Series) -> np.ndarray:
    """Return the same-type median log price per square foot from the prior month."""
    history = sales.copy()
    history["_market_type"] = history.house_type_name.astype("string").fillna("__missing_house_type__")
    monthly = history.groupby(["_market_type", "_month"], dropna=False, sort=False).lppsf.median()
    cutoffs = pd.Series(pd.to_datetime(np.asarray(query_cutoff)), index=queries.index)
    prior_months = cutoffs.dt.to_period("M") - 1
    query_types = queries.house_type_name.astype("string").fillna("__missing_house_type__")
    return np.asarray([monthly.get((str(type_name), month), np.nan)
                       for type_name, month in zip(query_types, prior_months)], dtype=float)


def _hierarchical_market_index_features(sales: pd.DataFrame, queries: pd.DataFrame,
                                        query_cutoff: pd.Series, *,
                                        window_months: int = 12,
                                        shrinkage_count: float = 20.0) -> pd.DataFrame:
    """Build a prior-year log-price-per-sqft index with geographic fallbacks."""
    required = {"community", "municipality", "house_type_name"}
    if required - set(sales) or required - set(queries):
        raise ValueError(f"hierarchical market index requires fields: {sorted(required)}")
    if window_months < 1 or not np.isfinite(shrinkage_count) or shrinkage_count <= 0:
        raise ValueError("market-index window and shrinkage count must be positive")

    feature_names = ["market_local_lppsf", "market_local_n", "market_municipality_lppsf",
                     "market_municipality_n", "market_type_lppsf", "market_type_n",
                     "market_lppsf_shrunk", "market_fallback_level"]
    output = pd.DataFrame(np.nan, index=queries.index, columns=feature_names, dtype=float)
    for name in ("market_local_n", "market_municipality_n", "market_type_n"):
        output[name] = 0.0
    output["market_fallback_level"] = 3.0

    history = sales.copy()
    history["_market_type"] = history.house_type_name.astype("string").str.strip()
    query_rows = queries[["community", "municipality", "house_type_name"]].copy()
    query_rows["_market_type"] = query_rows.house_type_name.astype("string").str.strip()
    cutoffs = pd.Series(pd.to_datetime(np.asarray(query_cutoff), errors="coerce"), index=queries.index)
    if cutoffs.isna().any():
        raise ValueError("market-index query cutoffs must be valid timestamps")
    cutoffs = cutoffs.dt.to_period("M")

    for period, row_indexes in cutoffs.groupby(cutoffs, sort=False).groups.items():
        cutoff = period.start_time
        window_start = cutoff - pd.DateOffset(months=window_months)
        available = history.loc[(history.date_sold >= window_start) & (history.date_sold < cutoff)]
        query = query_rows.loc[row_indexes]
        local_stats = (available.dropna(subset=["community", "_market_type"])
                       .groupby(["community", "_market_type"], sort=False).lppsf
                       .agg(["median", "size"]))
        municipality_stats = (available.dropna(subset=["municipality", "_market_type"])
                              .groupby(["municipality", "_market_type"], sort=False).lppsf
                              .agg(["median", "size"]))
        type_stats = available.groupby("_market_type", sort=False).lppsf.agg(["median", "size"])

        local_keys = pd.MultiIndex.from_frame(query[["community", "_market_type"]])
        municipality_keys = pd.MultiIndex.from_frame(query[["municipality", "_market_type"]])
        local = local_stats.reindex(local_keys).reset_index(drop=True)
        municipality = municipality_stats.reindex(municipality_keys).reset_index(drop=True)
        type_values = type_stats.reindex(query["_market_type"].to_numpy()).reset_index(drop=True)
        local.index = query.index
        municipality.index = query.index
        type_values.index = query.index

        local_n = local["size"].fillna(0.0).to_numpy(dtype=float)
        municipality_n = municipality["size"].fillna(0.0).to_numpy(dtype=float)
        type_n = type_values["size"].fillna(0.0).to_numpy(dtype=float)
        local_value = local["median"].to_numpy(dtype=float)
        municipality_value = municipality["median"].to_numpy(dtype=float)
        type_value = type_values["median"].to_numpy(dtype=float)

        fallback = np.where(np.isfinite(municipality_value), municipality_value, type_value)
        estimate = np.where(np.isfinite(local_value), local_value, fallback)
        local_with_fallback = np.isfinite(local_value) & np.isfinite(fallback)
        alpha = local_n / (local_n + shrinkage_count)
        estimate[local_with_fallback] = (alpha[local_with_fallback] * local_value[local_with_fallback]
                                         + (1.0 - alpha[local_with_fallback]) * fallback[local_with_fallback])
        fallback_level = np.where(np.isfinite(local_value), 0.0,
                                  np.where(np.isfinite(municipality_value), 1.0,
                                           np.where(np.isfinite(type_value), 2.0, 3.0)))

        output.loc[query.index, "market_local_lppsf"] = local_value
        output.loc[query.index, "market_local_n"] = local_n
        output.loc[query.index, "market_municipality_lppsf"] = municipality_value
        output.loc[query.index, "market_municipality_n"] = municipality_n
        output.loc[query.index, "market_type_lppsf"] = type_value
        output.loc[query.index, "market_type_n"] = type_n
        output.loc[query.index, "market_lppsf_shrunk"] = estimate
        output.loc[query.index, "market_fallback_level"] = fallback_level
    return output


def hierarchical_sale_list_ratio_delta(sales: pd.DataFrame, queries: pd.DataFrame, *,
                                       cutoff: str | pd.Timestamp,
                                       window_months: int = 12,
                                       shrinkage_count: float = 20.0) -> pd.DataFrame:
    """Return a strictly past-only local log(sold/list) offset from the type median."""
    required = {"date_sold", "sold_price", "list_price", "community", "municipality", "house_type_name"}
    if required - set(sales) or {"community", "municipality", "house_type_name"} - set(queries):
        raise ValueError("sale/list history inputs are missing required columns")
    if window_months < 1 or not np.isfinite(shrinkage_count) or shrinkage_count <= 0:
        raise ValueError("history window and shrinkage count must be positive")

    cutoff_period = pd.Period(pd.Timestamp(cutoff), freq="M")
    cutoff_start = cutoff_period.start_time
    window_start = cutoff_start - pd.DateOffset(months=window_months)
    history = sales.copy()
    history["date_sold"] = pd.to_datetime(history.date_sold, errors="coerce")
    history["sold_price"] = pd.to_numeric(history.sold_price, errors="coerce")
    history["list_price"] = pd.to_numeric(history.list_price, errors="coerce")
    valid = (history.date_sold.ge(window_start) & history.date_sold.lt(cutoff_start)
             & np.isfinite(history.sold_price) & history.sold_price.gt(0)
             & np.isfinite(history.list_price) & history.list_price.gt(0))
    history = history.loc[valid].copy()
    history["_type"] = history.house_type_name.astype("string").str.strip().fillna("__missing_type__")
    history["_community"] = history.community.astype("string").str.strip().fillna("__missing_community__")
    history["_municipality"] = history.municipality.astype("string").str.strip().fillna("__missing_municipality__")
    history["_log_ratio"] = np.log(history.sold_price / history.list_price)

    query = queries[["community", "municipality", "house_type_name"]].copy()
    query["_type"] = query.house_type_name.astype("string").str.strip().fillna("__missing_type__")
    query["_community"] = query.community.astype("string").str.strip().fillna("__missing_community__")
    query["_municipality"] = query.municipality.astype("string").str.strip().fillna("__missing_municipality__")

    output = pd.DataFrame(index=queries.index)
    if history.empty:
        for name in ("ratio_type_median", "ratio_municipality_median", "ratio_local_median",
                     "ratio_shrunk", "ratio_delta"):
            output[name] = np.nan
        for name in ("ratio_type_n", "ratio_municipality_n", "ratio_local_n"):
            output[name] = 0.0
        output["ratio_fallback_level"] = 3.0
        return output

    type_stats = history.groupby("_type", sort=False)._log_ratio.agg(["median", "size"])
    municipality_stats = history.groupby(["_municipality", "_type"], sort=False)._log_ratio.agg(["median", "size"])
    local_stats = history.groupby(["_community", "_type"], sort=False)._log_ratio.agg(["median", "size"])
    type_values = type_stats.reindex(query["_type"].to_numpy()).reset_index(drop=True)
    municipality_keys = pd.MultiIndex.from_frame(query[["_municipality", "_type"]])
    local_keys = pd.MultiIndex.from_frame(query[["_community", "_type"]])
    municipality_values = municipality_stats.reindex(municipality_keys).reset_index(drop=True)
    local_values = local_stats.reindex(local_keys).reset_index(drop=True)

    type_n = type_values["size"].fillna(0.0).to_numpy(dtype=float)
    municipality_n = municipality_values["size"].fillna(0.0).to_numpy(dtype=float)
    local_n = local_values["size"].fillna(0.0).to_numpy(dtype=float)
    type_median = type_values["median"].to_numpy(dtype=float)
    municipality_median = municipality_values["median"].to_numpy(dtype=float)
    local_median = local_values["median"].to_numpy(dtype=float)

    municipality_shrunk = municipality_median.copy()
    can_shrink_municipality = np.isfinite(municipality_median) & np.isfinite(type_median)
    municipality_alpha = municipality_n / (municipality_n + shrinkage_count)
    municipality_shrunk[can_shrink_municipality] = (
        municipality_alpha[can_shrink_municipality] * municipality_median[can_shrink_municipality]
        + (1.0 - municipality_alpha[can_shrink_municipality]) * type_median[can_shrink_municipality])
    municipality_shrunk[~np.isfinite(municipality_shrunk)] = type_median[~np.isfinite(municipality_shrunk)]

    fallback = np.where(np.isfinite(municipality_shrunk), municipality_shrunk, type_median)
    estimate = fallback.copy()
    can_shrink_local = np.isfinite(local_median) & np.isfinite(fallback)
    local_alpha = local_n / (local_n + shrinkage_count)
    estimate[can_shrink_local] = (local_alpha[can_shrink_local] * local_median[can_shrink_local]
                                  + (1.0 - local_alpha[can_shrink_local]) * fallback[can_shrink_local])
    delta = estimate - type_median
    fallback_level = np.where(np.isfinite(local_median), 0.0,
                              np.where(np.isfinite(municipality_median), 1.0,
                                       np.where(np.isfinite(type_median), 2.0, 3.0)))

    output["ratio_type_median"] = type_median
    output["ratio_type_n"] = type_n
    output["ratio_municipality_median"] = municipality_median
    output["ratio_municipality_n"] = municipality_n
    output["ratio_local_median"] = local_median
    output["ratio_local_n"] = local_n
    output["ratio_shrunk"] = estimate
    output["ratio_delta"] = delta
    output["ratio_fallback_level"] = fallback_level
    return output


def building_history_features(sales: pd.DataFrame, queries: pd.DataFrame, *, track: str,
                              cutoff: str | pd.Timestamp | pd.Series | None = None,
                              training_rows: bool = False,
                              residual_shrink: float | None = None,
                              decay_half_life_months: float | None = None,
                              unit_history_reliability_pseudocount: float | None = None,
                              unit_history_reliability_half_life_months: float | None = None,
                              unit_history_fallback_to_building: bool = False,
                              include_residual_std: bool = False,
                              include_area_range: bool = False,
                              include_property_extras: bool = False,
                              include_market_index: bool = False,
                              include_hierarchical_market_index: bool = False) -> pd.DataFrame:
    """Build strict as-of coordinate/unit history features for a fold.

    Training observations use their own sale-month start. Evaluation observations
    use one supplied fold/origin cutoff; their answer fields are never read.
    """
    if track not in {"off_market", "list_aware"}:
        raise ValueError(f"unknown track: {track}")
    reliability_parameters = (unit_history_reliability_pseudocount,
                              unit_history_reliability_half_life_months)
    if any(value is not None for value in reliability_parameters):
        if any(value is None for value in reliability_parameters):
            raise ValueError("unit-history reliability requires both pseudocount and half-life")
        if (not np.isfinite(unit_history_reliability_pseudocount)
                or unit_history_reliability_pseudocount <= 0
                or not np.isfinite(unit_history_reliability_half_life_months)
                or unit_history_reliability_half_life_months <= 0):
            raise ValueError("unit-history reliability parameters must be finite and positive")
    if unit_history_fallback_to_building and unit_history_reliability_pseudocount is None:
        raise ValueError("unit-history building fallback requires reliability parameters")
    if training_rows:
        if "date_sold" not in queries:
            raise ValueError("training-row history requires sale dates to derive month-start cutoffs")
        query_cutoff = pd.to_datetime(queries.date_sold, errors="coerce").dt.to_period("M").dt.start_time
    elif cutoff is None:
        raise ValueError("evaluation history requires a fixed fold/origin cutoff")
    elif isinstance(cutoff, pd.Series):
        query_cutoff = pd.to_datetime(cutoff, errors="coerce")
    else:
        query_cutoff = pd.Series(pd.Timestamp(cutoff), index=queries.index)

    query_rows = queries.copy()
    query_rows["cutoff"] = np.asarray(query_cutoff)
    sale_rows = _sale_history(sales)
    max_cutoff = pd.to_datetime(query_rows.cutoff).max()
    sale_rows = sale_rows.loc[sale_rows.date_sold < max_cutoff].copy()
    query_rows["b_key"] = _coordinate_key(query_rows)
    query_rows["u_key"] = _unit_key(query_rows, query_rows.b_key)

    # Missing coordinates must never cause unrelated homes to share history.
    for key in ("b_key", "u_key"):
        bad_sales = sale_rows[key].isna()
        bad_queries = query_rows[key].isna()
        sale_rows.loc[bad_sales, key] = [f"__no_history_sale_{i}" for i in np.flatnonzero(bad_sales)]
        query_rows.loc[bad_queries, key] = [f"__no_history_query_{i}" for i in np.flatnonzero(bad_queries)]

    building = add_asof_history(sale_rows, query_rows, key="b_key", prefix="b_",
                                decay_half_life_months=decay_half_life_months,
                                include_residual_std=include_residual_std)
    unit = add_asof_history(sale_rows, query_rows, key="u_key", prefix="u_",
                           decay_half_life_months=decay_half_life_months,
                           include_residual_std=include_residual_std)
    if unit_history_reliability_pseudocount is not None:
        count = pd.to_numeric(unit["u_res_n"], errors="coerce").fillna(0.0).clip(lower=0.0)
        age = pd.to_numeric(unit["u_age"], errors="coerce")
        reliability = count / (count + unit_history_reliability_pseudocount)
        freshness = np.power(2.0, -age / unit_history_reliability_half_life_months)
        reliability = (reliability * freshness).where(age.notna(), 0.0).fillna(0.0)
        for column in ("u_res_mean", "u_last_res"):
            local = pd.to_numeric(unit[column], errors="coerce")
            if unit_history_fallback_to_building:
                building_prior = pd.to_numeric(building["b_res_mean"], errors="coerce")
                both_available = local.notna() & building_prior.notna()
                blended = local * reliability + building_prior * (1.0 - reliability)
                unit[column] = blended.where(both_available, local.combine_first(building_prior))
            else:
                unit[column] = local * reliability
    if residual_shrink is not None:
        if not np.isfinite(residual_shrink) or residual_shrink <= 0:
            raise ValueError("residual shrinkage pseudocount must be positive")
        for prefix, frame in (("b_", building), ("u_", unit)):
            count = pd.to_numeric(frame[f"{prefix}res_n"], errors="coerce").fillna(0.0)
            frame[f"{prefix}res_mean"] = frame[f"{prefix}res_mean"] * count / (count + residual_shrink)
    if decay_half_life_months is not None:
        for prefix, frame in (("b_", building), ("u_", unit)):
            frame[f"{prefix}res_mean"] = frame[f"{prefix}res_mean_decay"]
            frame.drop(columns=[f"{prefix}res_mean_decay"], inplace=True)
    unit_columns = ["u_n", "u_res_n", "u_res_mean", "u_last_lp", "u_last_res", "u_age"]
    if include_residual_std:
        unit_columns.append("u_res_std")
    unit = unit[unit_columns]
    result = pd.concat([building, unit], axis=1)
    if include_market_index:
        result["type_prev_month_log_ppsf"] = _lagged_type_market_index(sale_rows, queries, query_cutoff)
    if include_hierarchical_market_index:
        result = pd.concat([result, _hierarchical_market_index_features(sale_rows, queries, query_cutoff)], axis=1)
    if include_area_range:
        parts = queries.house_area.astype("string").str.extract(r"^\s*(\d+(?:\.\d+)?)\s*(?:-\s*(\d+(?:\.\d+)?))?")
        low = pd.to_numeric(parts[0], errors="coerce")
        high = pd.to_numeric(parts[1], errors="coerce").fillna(low)
        width = (high - low).clip(lower=0.0)
        midpoint = (high + low) / 2.0
        result["area_range_width"] = width
        result["area_is_range"] = (width > 0).astype("int8")
        result["area_relative_range_width"] = width / midpoint.where(midpoint > 0)
    if include_property_extras:
        for column in ("lot_front", "lot_depth", "garage"):
            if column not in queries.columns:
                raise ValueError(f"property-extra candidate requires {column}")
            result[column] = pd.to_numeric(queries[column], errors="coerce")
        result["lot_area"] = result["lot_front"] * result["lot_depth"]
    forbidden = {"sold_price", "date_sold", "list_price", "days_on_market",
                 "external_avm_estimate", "external_avm_estimate_date"}
    if forbidden.intersection(result.columns):
        raise AssertionError("answer or post-sale columns reached history features")
    if track == "off_market" and "list_price" in result:
        raise AssertionError("off-market history features must not include list_price")
    result.index = queries.index
    return result
