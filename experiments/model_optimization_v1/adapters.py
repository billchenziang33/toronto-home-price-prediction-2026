"""Frozen-feature adapters for HGBR, CatBoost, and LightGBM model trials."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FROZEN_FEATURES_PATH = PROJECT_ROOT / "release_candidate" / "features.py"
_spec = importlib.util.spec_from_file_location("optimization_frozen_features", FROZEN_FEATURES_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"cannot load frozen feature module: {FROZEN_FEATURES_PATH}")
_frozen_features = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_frozen_features)

FORBIDDEN_COMMON = {"ml_num", "sold_price", "date_sold", "days_on_market",
                    "external_avm_estimate", "external_avm_estimate_date"}
TARGETS = {"log_sold_price", "log_sold_price_mae", "log_sold_over_list"}


def _feature_function(track: str):
    if track == "off_market":
        return _frozen_features.off_market_features
    if track == "list_aware":
        return _frozen_features.list_aware_features
    raise ValueError(f"unknown track: {track}")


class FrozenFeatureTransformer:
    """Fit category vocabularies on the training fold and preserve evaluation row order."""
    def __init__(self, track: str):
        self.track = track
        self.category_values: dict[str, list[Any]] = {}
        self.columns: list[str] | None = None

    def _make_features(self, homes: pd.DataFrame) -> pd.DataFrame:
        return _feature_function(self.track)(homes)

    def fit(self, train: pd.DataFrame) -> "FrozenFeatureTransformer":
        X = self._make_features(train).copy()
        self.columns = list(X.columns)
        self.category_values = {
            column: X[column].dropna().unique().tolist()
            for column in X.select_dtypes(include=["object", "category"]).columns
        }
        self._check_columns(X)
        return self

    def transform(self, homes: pd.DataFrame, model_family: str = "hgbr") -> pd.DataFrame:
        if self.columns is None:
            raise RuntimeError("transformer must be fitted on the training fold")
        X = self._make_features(homes).copy()
        self._check_columns(X)
        if list(X.columns) != self.columns or len(X) != len(homes):
            raise ValueError("frozen feature shape or order changed")
        for column, categories in self.category_values.items():
            if model_family == "catboost":
                X[column] = X[column].fillna("Unknown").astype(str)
            else:
                X[column] = pd.Categorical(X[column], categories=categories)
        if model_family not in {"hgbr", "catboost", "lightgbm"}:
            raise ValueError(f"unsupported model family: {model_family}")
        return X

    def categorical_columns(self) -> list[str]:
        return list(self.category_values)

    def _check_columns(self, X: pd.DataFrame) -> None:
        forbidden = set(FORBIDDEN_COMMON)
        if self.track == "off_market":
            forbidden.add("list_price")
        found = forbidden.intersection(X.columns)
        if found:
            raise ValueError(f"forbidden prediction inputs found: {sorted(found)}")


def make_target(train: pd.DataFrame, target: str) -> np.ndarray:
    if target not in TARGETS:
        raise ValueError(f"unsupported target: {target}")
    price = pd.to_numeric(train["sold_price"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(price).all() or (price <= 0).any():
        raise ValueError("training sold_price must be finite and positive")
    if target == "log_sold_over_list":
        if "list_price" not in train:
            raise ValueError("log_sold_over_list is only valid for list-aware data")
        listed = pd.to_numeric(train["list_price"], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(listed).all() or (listed <= 0).any():
            raise ValueError("training list_price must be finite and positive for ratio target")
        return np.log(price / listed)
    return np.log(price)


def inverse_target(prediction: np.ndarray, homes: pd.DataFrame, target: str) -> np.ndarray:
    z = np.asarray(prediction, dtype=float)
    if not np.isfinite(z).all():
        raise ValueError("model returned non-finite transformed predictions")
    if target == "log_sold_over_list":
        if "list_price" not in homes:
            raise ValueError("list_price is required to invert the ratio target")
        listed = pd.to_numeric(homes["list_price"], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(listed).all() or (listed <= 0).any():
            raise ValueError("prediction list_price must be finite and positive")
        result = listed * np.exp(z)
    elif target in {"log_sold_price", "log_sold_price_mae"}:
        result = np.exp(z)
    else:
        raise ValueError(f"unsupported target: {target}")
    if not np.isfinite(result).all() or (result <= 0).any():
        raise ValueError("inverse-transformed price must be finite and positive")
    return result


def make_estimator(model_family: str, config: dict, target: str = "log_sold_price"):
    params = dict(config)
    if model_family == "hgbr":
        params.setdefault("random_state", 0)
        params.setdefault("categorical_features", "from_dtype")
        if target != "log_sold_price" or "max_iter" in params:
            params.setdefault("early_stopping", False)
        if target == "log_sold_price_mae":
            params["loss"] = "absolute_error"
        else:
            params.setdefault("loss", "squared_error")
        return HistGradientBoostingRegressor(**params)
    if model_family == "catboost":
        try:
            from catboost import CatBoostRegressor
        except ImportError as error:
            raise RuntimeError("CatBoost is not installed in the optimization environment") from error
        params.setdefault("random_seed", 0)
        params.setdefault("thread_count", 4)
        params.setdefault("verbose", False)
        params.setdefault("allow_writing_files", False)
        params.setdefault("has_time", True)
        params.setdefault("loss_function", "MAE" if target == "log_sold_price_mae" else "RMSE")
        return CatBoostRegressor(**params)
    if model_family == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as error:
            raise RuntimeError("LightGBM is not installed in the optimization environment") from error
        params.setdefault("random_state", 0)
        params.setdefault("n_jobs", 4)
        params.setdefault("verbosity", -1)
        params.setdefault("deterministic", True)
        params.setdefault("force_col_wise", True)
        params.setdefault("objective", "regression_l1" if target == "log_sold_price_mae" else "regression")
        return LGBMRegressor(**params)
    if model_family == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as error:
            raise RuntimeError("XGBoost is not installed in the optimization environment") from error
        params.setdefault("random_state", 0)
        params.setdefault("n_jobs", 4)
        params.setdefault("tree_method", "hist")
        return XGBRegressor(**params)
    raise ValueError(f"unsupported model family: {model_family}")


class PriceModel:
    """Small fit/predict wrapper that stores fold-fitted category information."""
    def __init__(self, track: str, model_family: str, config: dict, target: str = "log_sold_price"):
        self.track, self.model_family, self.config, self.target = track, model_family, dict(config), target
        self.transformer = FrozenFeatureTransformer(track)
        self.estimator = make_estimator(model_family, self.config, target)

    def fit(self, train: pd.DataFrame) -> "PriceModel":
        fit_rows = train
        if self.model_family == "catboost" and {"date_sold", "ml_num"}.issubset(train.columns):
            fit_rows = train.sort_values(["date_sold", "ml_num"], kind="mergesort")
        self.transformer.fit(fit_rows)
        X = self.transformer.transform(fit_rows, self.model_family)
        y = make_target(fit_rows, self.target)
        if self.model_family == "catboost":
            self.estimator.set_params(cat_features=self.transformer.categorical_columns())
        self.estimator.fit(X, y)
        return self

    def predict(self, homes: pd.DataFrame) -> np.ndarray:
        prohibited = FORBIDDEN_COMMON - {"ml_num"}
        if self.track == "off_market":
            prohibited.add("list_price")
        leaked = prohibited.intersection(homes.columns)
        if leaked:
            raise ValueError(f"prediction rows contain unavailable fields: {sorted(leaked)}")
        X = self.transformer.transform(homes, self.model_family)
        return inverse_target(self.estimator.predict(X), homes, self.target)


def build_fold(listings: pd.DataFrame, month: str, track: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split one walk-forward month and hide sale outcomes before prediction."""
    from data import hide_answers

    if "date_sold" not in listings:
        raise ValueError("listings must contain date_sold")
    dates = pd.to_datetime(listings["date_sold"])
    start = pd.Period(month, freq="M").start_time
    train = listings.loc[dates < start].copy()
    heldout = listings.loc[dates.dt.to_period("M").astype(str) == month].copy()
    if train.empty or heldout.empty:
        raise ValueError(f"fold {month} has no training or held-out rows")
    if (pd.to_datetime(train.date_sold) >= start).any():
        raise AssertionError("walk-forward training rows include the evaluation month or later")
    homes = hide_answers(heldout, track)
    if "sold_price" in homes or "date_sold" in homes or (track == "off_market" and "list_price" in homes):
        raise AssertionError("target or restricted feature was not hidden before prediction")
    return train, homes

