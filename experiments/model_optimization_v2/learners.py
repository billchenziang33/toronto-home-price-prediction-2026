from __future__ import annotations

import numpy as np
import pandas as pd


def method_ids(method_id: str) -> set[str]:
    methods = set() if method_id == "C0" else set(method_id.split("+"))
    valid = {f"M{i:02d}" for i in range(1, 21)}
    if len(methods) != (0 if method_id == "C0" else len(method_id.split("+"))) or methods - valid:
        raise ValueError(f"unknown or repeated method in: {method_id}")
    return methods


def _area(frame: pd.DataFrame) -> np.ndarray:
    if "square_feet" in frame:
        area = pd.to_numeric(frame["square_feet"], errors="coerce").to_numpy(dtype=float)
    else:
        from .features import _base_features
        area = pd.to_numeric(_base_features.square_feet(frame["house_area"]), errors="coerce").to_numpy(dtype=float)
    return area


def training_target(train: pd.DataFrame, track: str, method_id: str, target_override: str | None = None):
    methods = method_ids(method_id)
    price = pd.to_numeric(train.sold_price, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(price).all() or (price <= 0).any():
        raise ValueError("training sale prices must be finite and positive")
    if target_override is not None:
        if target_override == "log_sold_price":
            return np.log(price), target_override
        if target_override == "log_sold_over_list":
            listed = pd.to_numeric(train.list_price, errors="coerce").to_numpy(dtype=float)
            if not np.isfinite(listed).all() or (listed <= 0).any():
                raise ValueError("list-aware target requires finite positive list prices")
            return np.log(price / listed), target_override
        raise ValueError(f"unsupported target override: {target_override}")
    target_methods = methods.intersection({"M12", "M13"})
    if len(target_methods) > 1:
        raise ValueError("M12 and M13 define incompatible target transformations")
    if "M13" in methods:
        return price, "dollar_price"
    if "M12" in methods:
        if track != "off_market":
            raise ValueError("M12 is only eligible for off-market")
        area = _area(train)
        if not np.isfinite(area).all() or (area <= 0).any():
            raise ValueError("M12 requires finite positive area for every training row")
        return np.log(price / area), "log_price_per_area"
    if track == "list_aware":
        listed = pd.to_numeric(train.list_price, errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(listed).all() or (listed <= 0).any():
            raise ValueError("list-aware target requires finite positive list prices")
        return np.log(price / listed), "log_sold_over_list"
    return np.log(price), "log_sold_price"


def inverse_price(prediction, homes: pd.DataFrame, track: str, method_id: str,
                  target_override: str | None = None):
    methods = method_ids(method_id)
    pred = np.asarray(prediction, dtype=float)
    if not np.isfinite(pred).all():
        raise ValueError("transformed predictions must be finite")
    if target_override is not None:
        if target_override == "log_sold_price":
            price = np.exp(pred)
        elif target_override == "log_sold_over_list":
            listed = pd.to_numeric(homes.list_price, errors="coerce").to_numpy(dtype=float)
            if not np.isfinite(listed).all() or (listed <= 0).any():
                raise ValueError("list-aware inverse target requires finite positive list price")
            price = listed * np.exp(pred)
        else:
            raise ValueError(f"unsupported target override: {target_override}")
        if not np.isfinite(price).all() or (price <= 0).any():
            raise ValueError("inverse target produced a non-positive or non-finite price")
        return price
    target_methods = methods.intersection({"M12", "M13"})
    if len(target_methods) > 1:
        raise ValueError("M12 and M13 define incompatible target transformations")
    if "M13" in methods:
        price = pred
    elif "M12" in methods:
        area = _area(homes)
        if not np.isfinite(area).all() or (area <= 0).any():
            raise ValueError("M12 requires finite positive area at prediction time")
        price = np.exp(pred) * area
    elif track == "list_aware":
        listed = pd.to_numeric(homes.list_price, errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(listed).all() or (listed <= 0).any():
            raise ValueError("list-aware inverse target requires finite positive list price")
        price = listed * np.exp(pred)
    else:
        price = np.exp(pred)
    if not np.isfinite(price).all() or (price <= 0).any():
        raise ValueError("inverse target produced a non-positive or non-finite price")
    return price


def recency_weights(train: pd.DataFrame, month: str, method_id: str) -> np.ndarray:
    methods = method_ids(method_id)
    cutoff = pd.Period(month, freq="M").start_time
    dates = pd.to_datetime(train.date_sold)
    age = (cutoff - dates).dt.total_seconds().to_numpy(dtype=float) / 86400.0
    if (age < 0).any():
        raise ValueError("recency weights cannot use sales at or after the cutoff")
    if "M01" not in methods:
        return np.ones(len(train), dtype=float)
    weights = np.power(2.0, -age / 365.0)
    return weights / weights.mean()


def select_training_rows(train: pd.DataFrame, month: str, method_id: str) -> pd.DataFrame:
    methods = method_ids(method_id)
    cutoff = pd.Period(month, freq="M").start_time
    dates = pd.to_datetime(train.date_sold)
    selected = train.loc[dates < cutoff]
    if "M02" in methods:
        selected = selected.loc[pd.to_datetime(selected.date_sold) >= cutoff - pd.DateOffset(months=24)]
    if selected.empty:
        raise ValueError("method produced an empty training fold")
    return selected.copy()


def make_learner(family: str, config: dict):
    params = dict(config)
    if family == "lightgbm":
        from lightgbm import LGBMRegressor
        params.setdefault("n_jobs", 4)
        params.setdefault("random_state", 0)
        params.setdefault("verbosity", -1)
        params.setdefault("deterministic", True)
        params.setdefault("force_col_wise", True)
        return LGBMRegressor(**params)
    if family == "catboost":
        from catboost import CatBoostRegressor
        params.setdefault("thread_count", 4)
        params.setdefault("random_seed", 0)
        params.setdefault("verbose", False)
        params.setdefault("allow_writing_files", False)
        return CatBoostRegressor(**params)
    if family == "xgboost":
        from xgboost import XGBRegressor
        params.setdefault("n_jobs", 4)
        params.setdefault("random_state", 0)
        params.setdefault("tree_method", "hist")
        params.setdefault("enable_categorical", True)
        return XGBRegressor(**params)
    if family == "extra_trees":
        from sklearn.ensemble import ExtraTreesRegressor
        params.setdefault("n_jobs", 4)
        return ExtraTreesRegressor(**params)
    if family == "hgbr":
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(**params)
    raise ValueError(f"unsupported learner family: {family}")


def validate_actual_objective(estimator, family: str, expected: str) -> str:
    """Fail closed when the fitted or declared learner objective differs."""
    if family == "lightgbm":
        actual = getattr(estimator, "objective_", None)
        if actual is None:
            actual = estimator.get_params(deep=False).get("objective")
    elif family == "catboost":
        actual = estimator.get_all_params().get("loss_function") if hasattr(estimator, "get_all_params") else None
        if actual is None:
            actual = estimator.get_params().get("loss_function")
    elif family == "xgboost":
        actual = estimator.get_params(deep=False).get("objective")
    else:
        actual = expected
    normal = {"regression": "regression", "regression_l1": "regression_l1", "huber": "huber",
              "mape": "mape", "reg:squarederror": "reg:squarederror"}
    actual = normal.get(str(actual).lower(), str(actual).lower())
    expected = normal.get(expected.lower(), expected.lower())
    if actual != expected:
        raise ValueError(f"learner objective mismatch: expected {expected}, got {actual}")
    return actual
