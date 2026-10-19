from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_base_spec = importlib.util.spec_from_file_location("v2_base_features", PROJECT_ROOT / "model_candidate_v2" / "features.py")
if _base_spec is None or _base_spec.loader is None:
    raise RuntimeError("cannot load frozen v2 base features")
_base_features = importlib.util.module_from_spec(_base_spec)
_base_spec.loader.exec_module(_base_features)


def build_features(homes: pd.DataFrame, track: str, method_id: str, *, history: pd.DataFrame | None = None,
                   cutoff: str | pd.Timestamp | None = None, grid_origin: tuple[float, float] | None = None) -> pd.DataFrame:
    if track not in {"off_market", "list_aware"}:
        raise ValueError(f"unknown track: {track}")
    methods = [] if method_id == "C0" else method_id.split("+")
    valid_methods = {f"M{i:02d}" for i in range(1, 21)}
    if any(method not in valid_methods for method in methods) or len(methods) != len(set(methods)):
        raise ValueError(f"unknown or repeated method in: {method_id}")
    result = (_base_features.off_market_features(homes) if track == "off_market"
              else _base_features.list_aware_features(homes)).copy()
    for method in methods:
        if method == "M03":
            month = pd.to_datetime(homes.date_listed).dt.month.astype(float)
            result["listed_month_sin"] = np.sin(2.0 * np.pi * (month - 1.0) / 12.0)
            result["listed_month_cos"] = np.cos(2.0 * np.pi * (month - 1.0) / 12.0)
            result["listed_quarter"] = pd.Categorical(((month - 1) // 3 + 1).astype(int).astype(str))
        elif method == "M04":
            parts = homes.house_area.astype("string").str.extract(r"^\s*(\d+(?:\.\d+)?)\s*(?:-\s*(\d+(?:\.\d+)?))?")
            low = pd.to_numeric(parts[0], errors="coerce")
            high = pd.to_numeric(parts[1], errors="coerce").fillna(low)
            area = (low + high) / 2.0
            width = (high - low).clip(lower=0.0)
            result["log_area"] = np.log(area.where(area > 0))
            result["area_is_range"] = (width > 0).astype(int)
            result["area_range_width"] = width
            result["area_relative_range_width"] = width / area.where(area > 0)
        elif method == "M05":
            area = pd.to_numeric(result["square_feet"], errors="coerce")
            bedrooms = pd.to_numeric(homes.bedrooms, errors="coerce")
            washrooms = pd.to_numeric(homes.washrooms, errors="coerce")
            parking = pd.to_numeric(homes.parking_total, errors="coerce")
            result["area_per_bedroom"] = area / bedrooms.where(bedrooms > 0)
            result["zero_bedrooms"] = (bedrooms <= 0).astype(int)
            result["washrooms_per_bedroom"] = washrooms / bedrooms.where(bedrooms > 0)
            result["parking_per_area"] = parking / area.where(area > 0)
            log_area = np.log(area.where(area > 0))
            for column in [c for c in result.columns if c.startswith("is_") and not c.endswith("_log_area")]:
                result[f"{column}_log_area"] = result[column].astype(float) * log_area
        elif method == "M06":
            result = result.drop(columns=[c for c in result.columns if c.startswith("is_")])
            result["house_type_name"] = homes.house_type_name.fillna("Unknown").astype("string")
        elif method == "M07":
            if grid_origin is None:
                lat = pd.to_numeric(homes.latitude, errors="coerce")
                lon = pd.to_numeric(homes.longitude, errors="coerce")
                if lat.notna().sum() == 0 or lon.notna().sum() == 0:
                    raise ValueError("M07 requires finite training coordinates")
                grid_origin = (float(lat.min()), float(lon.min()))
            lat_bin = np.floor((pd.to_numeric(homes.latitude, errors="coerce") - grid_origin[0]) / 0.02)
            lon_bin = np.floor((pd.to_numeric(homes.longitude, errors="coerce") - grid_origin[1]) / 0.02)
            result["grid_cell"] = (lat_bin.fillna(-1).astype(int).astype(str) + ":" +
                                   lon_bin.fillna(-1).astype(int).astype(str)).astype("string")
            result["type_grid"] = homes.house_type_name.fillna("Unknown").astype("string") + ":" + result.grid_cell
        elif method in {"M08", "M09"}:
            continue
    market_methods = {"M08", "M09"}.intersection(methods)
    if market_methods:
        if history is None or cutoff is None:
            raise ValueError("M08/M09 require a past-only sales history and cutoff")
        extras = asof_market_features(homes, history, cutoff)
        extras.index = homes.index
        selected = []
        if "M08" in market_methods:
            selected.extend(["community_type_log_ppsf_mean", "community_type_count",
                             "municipality_type_log_ppsf_mean", "municipality_type_count",
                             "community_type_log_ppsf_shrunk_50"])
        if "M09" in market_methods:
            selected.extend(["type_log_ppsf_mean_90d", "type_log_ppsf_count_90d",
                             "type_log_ppsf_mean_365d", "type_log_ppsf_count_365d"])
        result = pd.concat([result, extras[selected]], axis=1)
    forbidden = {"sold_price", "date_sold", "days_on_market", "external_avm_estimate", "external_avm_estimate_date"}
    if track == "off_market":
        forbidden.add("list_price")
    if forbidden.intersection(result.columns):
        raise ValueError(f"restricted fields reached model features: {sorted(forbidden.intersection(result.columns))}")
    return result

def _key(row, columns):
    return tuple("<missing>" if pd.isna(row[c]) else str(row[c]) for c in columns)


def _prefix_group(history: pd.DataFrame, group_columns: list[str]):
    records = {}
    for key, rows in history.groupby(group_columns, dropna=False, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        rows = rows.sort_values("date_sold", kind="mergesort")
        dates = pd.to_datetime(rows.date_sold).to_numpy(dtype="datetime64[ns]")
        values = rows.log_ppsf.to_numpy(dtype=float)
        ids = rows.ml_num.astype(str).to_numpy()
        sums = np.concatenate(([0.0], np.cumsum(values)))
        records[key] = (dates, values, ids, sums)
    return records


def _stats(records, key, query_time, query_id, window_days=None):
    item = records.get(key)
    if item is None:
        return 0.0, 0
    dates, values, ids, sums = item
    end = int(np.searchsorted(dates, np.datetime64(query_time), side="left"))
    start = 0 if window_days is None else int(np.searchsorted(
        dates, np.datetime64(query_time) - np.timedelta64(window_days, "D"), side="left"))
    total = float(sums[end] - sums[start])
    count = end - start
    if count and query_id:
        matching = np.flatnonzero(ids[start:end] == str(query_id))
        if len(matching):
            offset = start + int(matching[0])
            total -= float(values[offset])
            count -= 1
    return (total / count if count else 0.0), count


def asof_market_features(queries: pd.DataFrame, history: pd.DataFrame,
                         outer_cutoff: str | pd.Timestamp) -> pd.DataFrame:
    """Build strict-past community/type and type-momentum features per listing row."""
    required_q = {"ml_num", "date_listed", "community", "municipality", "house_type_name"}
    required_h = {"ml_num", "date_sold", "sold_price", "community", "municipality", "house_type_name"}
    if required_q - set(queries) or required_h - set(history) or not ({"area", "house_area"} & set(history)):
        raise ValueError("as-of inputs are missing required columns")
    cutoff = pd.Timestamp(outer_cutoff)
    valid = history.copy()
    valid["date_sold"] = pd.to_datetime(valid.date_sold)
    valid["sold_price"] = pd.to_numeric(valid.sold_price, errors="coerce")
    if "area" in valid:
        valid["area"] = pd.to_numeric(valid.area, errors="coerce")
    else:
        valid["area"] = pd.to_numeric(_base_features.square_feet(valid.house_area), errors="coerce")
    valid = valid.loc[(valid.date_sold < cutoff) & (valid.sold_price > 0) & (valid.area > 0)].copy()
    valid["log_ppsf"] = np.log(valid.sold_price / valid.area)
    group_cols = ["community", "municipality", "house_type_name"]
    community = _prefix_group(valid, group_cols)
    municipality = _prefix_group(valid, ["municipality", "house_type_name"])
    property_type = _prefix_group(valid, ["house_type_name"])

    q = queries.reset_index(drop=True).copy()
    q["date_listed"] = pd.to_datetime(q.date_listed)
    result = {name: [] for name in ["community_type_log_ppsf_mean", "community_type_count",
             "municipality_type_log_ppsf_mean", "municipality_type_count", "community_type_log_ppsf_shrunk_50",
             "type_log_ppsf_mean_90d", "type_log_ppsf_count_90d", "type_log_ppsf_mean_365d", "type_log_ppsf_count_365d"]}
    for row in q.itertuples(index=False):
        values = row._asdict()
        t = min(pd.Timestamp(values["date_listed"]), cutoff)
        ident = str(values["ml_num"])
        c_key = _key(values, group_cols)
        m_key = _key(values, ["municipality", "house_type_name"])
        t_key = _key(values, ["house_type_name"])
        c_mean, c_n = _stats(community, c_key, t, ident)
        m_mean, m_n = _stats(municipality, m_key, t, ident)
        mean90, n90 = _stats(property_type, t_key, t, ident, 90)
        mean365, n365 = _stats(property_type, t_key, t, ident, 365)
        shrunk = (c_mean * c_n + m_mean * 50.0) / (c_n + 50.0) if c_n + 50 else m_mean
        for name, value in [("community_type_log_ppsf_mean", c_mean), ("community_type_count", c_n),
                            ("municipality_type_log_ppsf_mean", m_mean), ("municipality_type_count", m_n),
                            ("community_type_log_ppsf_shrunk_50", shrunk), ("type_log_ppsf_mean_90d", mean90),
                            ("type_log_ppsf_count_90d", n90), ("type_log_ppsf_mean_365d", mean365),
                            ("type_log_ppsf_count_365d", n365)]:
            result[name].append(value)
    return pd.DataFrame(result)
