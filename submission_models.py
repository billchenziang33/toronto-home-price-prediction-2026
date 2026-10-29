"""Reproducible submission routes backed by the frozen research candidates.

The research candidates were built from two registered, strictly-past OOF
pipelines. This module refits their learner components on the caller's training
rows and reapplies only the saved, safe OOF post-processing history. No hidden
labels or reserve-month labels are read here.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import lru_cache
from importlib import metadata
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
ASSET_DIR = ROOT / "model_assets"
OUTPUT_DIR = ROOT / "outputs" / "submission"
FIT_LEDGER = OUTPUT_DIR / "fit_ledger.jsonl"
_FIT_LEDGER_LOCK = threading.Lock()
REPRODUCIBILITY_FILES = (
    ".gitattributes", "models.py", "submission_models.py", "run.py", "data.py", "requirements.txt",
    "model_candidate_v2/features.py", "model_candidate_v2/winner_config.json",
    "model_candidate_v2/models/frozen_list_aware.json",
    "experiments/bill_greedy/runner.py", "experiments/bill_greedy/history.py",
    "experiments/bill_greedy/comparables.py",
    "experiments/model_optimization_v2/worker.py",
    "experiments/model_optimization_v2/features.py",
    "experiments/model_optimization_v2/pipeline.py",
    "experiments/model_optimization_v2/learners.py",
    "experiments/bill_target74/campaign_protocol.json",
    "experiments/bill_target74/runner.py",
    "experiments/bill_target74/submission_pipeline.py",
    "experiments/bill_target74/desktop_friendly_runtime.py",
    "experiments/bill_target74/desktop_friendly_policy.json",
    "experiments/bill_target74/r04_ablation.py",
    "experiments/bill_target74/r19_expert_weight.py",
    "experiments/bill_target74/r22_remarks_tfidf.py",
    "experiments/bill_target74/r32_comparable_r22.py",
    "experiments/bill_target74/r35_fixed_origin.py",
    "experiments/bill_target74/r40_past_oof_r26_blend.py",
    "experiments/bill_target74/r48_groupwise_past_oof_r22_r32_blend.py",
    "experiments/bill_target74/r49_parent_price_tercile_past_oof_blend.py",
    "experiments/bill_target74/r50_type_price_interaction_past_oof_blend.py",
    "experiments/bill_target74/r52_r48_r50_adaptive_meta_blend.py",
    "experiments/bill_target74/r02_structure_matched_current_parent.py",
    "experiments/bill_target74/r93_spatial_oof_calibration.py",
    "experiments/bill_target74/r97_past_oof_catboost_residual_meta.py",
    "experiments/bill_target74/r151_offmarket_r93_radius15km_clip035_vs_r22.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(path: Path) -> str:
    """Hash text files independently of the checkout's newline convention."""
    content = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(content).hexdigest()


def _append_fit_record(record: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with _FIT_LEDGER_LOCK, FIT_LEDGER.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, sort_keys=True, default=str) + "\n")


def _run_submission_fit(context: dict, fit_fn):
    """Run one learner fit with an independent ledger and the desktop wait hook."""
    runtime = None
    if os.environ.get("BILL_DESKTOP_FRIENDLY_ACTIVE") == "1":
        from experiments.bill_target74 import desktop_friendly_runtime as runtime
        runtime.wait_before_fit(
            f"submission:{context['route']}:{context['component']}"
        )
    fit_id = context.get("fit_id", f"submission:{context['route']}:{context['component']}:{context['cutoff']}")
    event = {
        "event": "fit_start", "fit_id": fit_id,
        "run_id": context.get("run_id"),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "route": context["route"], "component": context["component"],
        "track": context["track"], "cutoff": context["cutoff"],
        "training_rows": context["training_rows"],
        "evaluation_rows": context["evaluation_rows"],
        "desktop_friendly_runtime": runtime is not None,
    }
    _append_fit_record(event)
    started = time.perf_counter()
    try:
        fitted = fit_fn()
    except BaseException as exc:
        _append_fit_record({
            **event, "event": "fit_complete", "status": "failed",
            "seconds": time.perf_counter() - started,
            "error": f"{type(exc).__name__}: {exc}",
        })
        raise
    seconds = time.perf_counter() - started
    if isinstance(fitted, dict):
        fitted["fit_seconds"] = seconds
        details = {key: fitted.get(key) for key in (
            "prediction_sha256", "model_family", "model_config", "feature_columns",
        )}
    else:
        details = {}
    _append_fit_record({
        **event, "event": "fit_complete", "status": "complete",
        "seconds": seconds, **details,
    })
    return fitted


@contextmanager
def _submission_fit_recorder(context: dict) -> Iterator[None]:
    """Use a separate ledger and the desktop-friendly wait hook for each fit."""
    from experiments.bill_greedy import runner as greedy

    previous = greedy._fit_record

    def record_fit(fit_id: str, fit_fn):
        return _run_submission_fit({**context, "fit_id": fit_id}, fit_fn)

    greedy._fit_record = record_fit
    try:
        yield
    finally:
        greedy._fit_record = previous


def _predict_history_component(*, route: str, component: str, track: str,
                               train: pd.DataFrame, homes: pd.DataFrame,
                               cutoff: pd.Timestamp, variant: str,
                               variant_spec: dict, fit_rows: pd.DataFrame | None = None,
                               base_variant: str | None = None,
                               run_id: str | None = None) -> np.ndarray:
    from experiments.bill_greedy import runner as greedy
    from experiments.bill_target74 import r22_remarks_tfidf as r22

    original_spec = greedy.VARIANTS.get(variant)
    greedy.VARIANTS[variant] = variant_spec
    fit_rows = train if fit_rows is None else fit_rows
    context = {
        "route": route, "component": component, "track": track,
        "cutoff": cutoff.isoformat(), "training_rows": len(fit_rows),
        "evaluation_rows": len(homes), "run_id": run_id,
    }
    transformer, restore = r22._install_text_features(fit_rows, homes, track)
    try:
        with _submission_fit_recorder(context):
            fitted = greedy._fit_history_candidate(
                cutoff.strftime("%Y-%m"), track, train, homes, cutoff=cutoff,
                fit_rows=fit_rows, variant=variant, base_variant=base_variant,
                fit_tag=f"submission-{route}-{component}-{cutoff:%Y%m}",
            )
        prediction = np.asarray(fitted["predictions"], dtype=float).copy()
        if prediction.shape != (len(homes),) or not np.isfinite(prediction).all() or (prediction <= 0).any():
            raise RuntimeError(f"{route}/{component} produced invalid prices")
        del fitted, transformer
        return prediction
    finally:
        restore()
        if original_spec is None:
            greedy.VARIANTS.pop(variant, None)
        else:
            greedy.VARIANTS[variant] = original_spec


def _fit_hgbr(train: pd.DataFrame, homes: pd.DataFrame, cutoff: pd.Timestamp,
              route: str, run_id: str | None = None) -> np.ndarray:
    from experiments.bill_greedy import runner as greedy

    context = {"route": route, "component": "shared_hgbr", "track": "list_aware",
               "cutoff": cutoff.isoformat(), "training_rows": len(train),
               "evaluation_rows": len(homes), "run_id": run_id}
    with _submission_fit_recorder(context):
        fitted = greedy._fit_baseline_component(
            cutoff.strftime("%Y-%m"), "list_aware", train, homes, "HGBR",
            fit_tag=f"submission-{route}-HGBR-{cutoff:%Y%m}",
        )
    prediction = np.asarray(fitted["predictions"], dtype=float).copy()
    if prediction.shape != (len(homes),) or not np.isfinite(prediction).all() or (prediction <= 0).any():
        raise RuntimeError(f"{route}/HGBR produced invalid prices")
    del fitted
    return prediction


def _safe_asset(filename: str) -> pd.DataFrame:
    manifest_path = ASSET_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("access_scope", {}).get("reserved_month_labels_opened") is not False:
        raise RuntimeError("safe OOF manifest does not attest reserved-label isolation")
    if manifest.get("access_scope", {}).get("hidden_test_labels_opened") is not False:
        raise RuntimeError("safe OOF manifest does not attest hidden-label isolation")
    candidates = []
    for route in manifest.get("routes", {}).values():
        for record in (route.get("asset"),):
            if record and record.get("path") == filename:
                candidates.append(record)
        for record in route.get("meta_assets", {}).values():
            if record.get("path") == filename:
                candidates.append(record)
        for record in route.get("route_oof_assets", {}).values():
            if record.get("path") == filename:
                candidates.append(record)
    if len(candidates) != 1:
        raise RuntimeError(f"manifest does not identify exactly one {filename} asset")
    path = ASSET_DIR / filename
    if not path.is_file() or sha256(path) != candidates[0].get("sha256"):
        raise RuntimeError(f"safe OOF asset failed its manifest hash: {filename}")
    return pd.read_csv(path, dtype={"ml_num": str})


@lru_cache(maxsize=1)
def _score_safe_inputs() -> dict:
    """Load safe OOF inputs once, verify their hashes and row scopes."""
    from experiments.bill_target74 import r40_past_oof_r26_blend as weight_selection
    from experiments.bill_target74 import r49_parent_price_tercile_past_oof_blend as price_source

    r47 = _safe_asset("off_market_r47_weight_history.csv")
    r48_r50 = _safe_asset("off_market_r48_r50_weight_history.csv")
    r151 = _safe_asset("off_market_r151_safe_oof.csv")
    r151_route = _safe_asset("off_market_r151_candidate_oof.csv")
    r97 = _safe_asset("list_aware_r97_safe_oof.csv")
    reserved = {"2025-05", "2025-08"}
    for name, frame in (("R47", r47), ("R48/R50", r48_r50), ("R151", r151),
                        ("R151 route", r151_route), ("R97", r97)):
        if set(frame.month.astype(str)).intersection(reserved):
            raise RuntimeError(f"{name} safe OOF history includes a reserved month")
        if frame.duplicated(["month", "ml_num"]).any():
            raise RuntimeError(f"{name} safe OOF history has duplicate month/ID rows")
    r47["house_type_name"] = r47.house_type_name.astype("string").fillna("__MISSING__").astype(str)
    r47["parent_price_tercile"] = ""
    for month, index in r47.groupby(r47.month.astype(str), sort=False).groups.items():
        groups, _ = price_source._price_groups(r47.loc[index, "parent_prediction"].to_numpy(float))
        r47.loc[index, "parent_price_tercile"] = groups
    r47["date_sold"] = pd.to_datetime(r47.date_sold, errors="raise")
    r48_r50["month"] = r48_r50.month.astype(str)
    r151["date_sold"] = pd.to_datetime(r151.date_sold, errors="raise")
    r151["month"] = r151.month.astype(str)
    r151_route["month"] = r151_route.month.astype(str)
    r97["date_sold"] = pd.to_datetime(r97.date_sold, errors="raise")
    r97["month"] = r97.month.astype(str)
    r97["date_listed"] = pd.to_datetime(r97.date_listed, errors="raise")
    return {"r47": r47, "r48_r50": r48_r50, "r151": r151,
            "r151_route": r151_route, "r97": r97,
            "weight_selection": weight_selection}


def _blend_by_row_weight(parent: np.ndarray, variant: np.ndarray,
                         weights: np.ndarray) -> np.ndarray:
    parent = np.asarray(parent, dtype=float)
    variant = np.asarray(variant, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if (parent.ndim != 1 or variant.shape != parent.shape or weights.shape != parent.shape
            or not np.isfinite(parent).all() or not np.isfinite(variant).all()
            or not np.isfinite(weights).all() or (parent <= 0).any()
            or (variant <= 0).any() or (weights < 0).any() or (weights > 1).any()):
        raise ValueError("R22/R32 prices and per-home weights must be aligned, finite, and valid")
    return np.round((1.0 - weights) * parent + weights * variant)


def _r48_candidate(history: pd.DataFrame, r22: np.ndarray, r32: np.ndarray,
                   homes: pd.DataFrame, cutoff: pd.Timestamp,
                   selectors) -> tuple[np.ndarray, np.ndarray]:
    from experiments.bill_target74 import runner as campaign

    card = campaign.PROTOCOL["cards"]["R48_groupwise_past_oof_r22_r32_blend"]
    config = card["weight_selection"]
    min_global = int(config["minimum_global_past_oof_rows"])
    min_group = int(config["minimum_group_past_oof_rows"])
    mape_guard = float(config["past_mape_regression_max_percent"])
    global_pick = selectors.select_weight(
        history, cutoff, minimum_rows=min_global,
        max_mape_regression_percent=mape_guard,
    )
    global_weight = float(global_pick["weight"])
    current_types = homes.house_type_name.astype("string").fillna("__MISSING__").astype(str).to_numpy()
    weights = np.full(len(homes), global_weight, dtype=float)
    for house_type in sorted(set(current_types)):
        prior = history.loc[history.house_type_name.astype(str).eq(house_type)].copy()
        if len(prior) < min_group:
            continue
        pick = selectors.select_weight(
            prior, cutoff, minimum_rows=min_group,
            max_mape_regression_percent=mape_guard,
        )
        if pick["reason"] == "selected_from_strictly_past_oof":
            weights[current_types == house_type] = float(pick["weight"])
    r48 = _blend_by_row_weight(r22, r32, weights)
    return r48, weights


def _r50_candidate(history: pd.DataFrame, r22: np.ndarray, r32: np.ndarray,
                   homes: pd.DataFrame, cutoff: pd.Timestamp,
                   selectors) -> tuple[np.ndarray, np.ndarray]:
    from experiments.bill_target74 import runner as campaign
    from experiments.bill_target74 import r50_type_price_interaction_past_oof_blend as r50

    config = campaign.PROTOCOL["cards"]["R50_type_price_interaction_past_oof_blend"]["weight_selection"]
    min_global = int(config["minimum_global_past_oof_rows"])
    min_group = int(config["minimum_house_type_past_oof_rows"])
    mape_guard = float(config["past_mape_regression_max_percent"])
    global_pick = selectors.select_weight(
        history, cutoff, minimum_rows=min_global,
        max_mape_regression_percent=mape_guard,
    )
    global_weight = float(global_pick["weight"])
    type_values = homes.house_type_name.astype("string").fillna("__MISSING__").astype(str).to_numpy()
    groups, _thresholds = r50.price_source._price_groups(r22)
    type_picks = {}
    for house_type in sorted(set(type_values)):
        prior = history.loc[history.house_type_name.astype(str).eq(house_type)].copy()
        type_picks[house_type] = r50._pick_group_weight(
            prior, cutoff, min_group, mape_guard, global_weight,
            "fallback_to_global_insufficient_or_ineligible_house_type",
        )
    weights = np.empty(len(homes), dtype=float)
    for house_type, tercile in sorted(set(zip(type_values, groups))):
        prior = history.loc[
            history.house_type_name.astype(str).eq(house_type)
            & history.parent_price_tercile.astype(str).eq(str(tercile))
        ].copy()
        type_pick = type_picks[house_type]
        picked = r50._pick_group_weight(
            prior, cutoff, min_group, mape_guard, float(type_pick["weight"]),
            "fallback_to_house_type_insufficient_or_ineligible_interaction",
        )
        weights[(type_values == house_type) & (groups == tercile)] = float(picked["weight"])
    return _blend_by_row_weight(r22, r32, weights), weights


class CandidateRouteModel:
    track: str
    route: str

    def __init__(self, track: str, route: str):
        self.track = track
        self.route = route
        self.run_id = uuid.uuid4().hex
        self.train: pd.DataFrame | None = None
        self._safe: dict | None = None

    def fit(self, train: pd.DataFrame):
        if "sold_price" not in train or "date_sold" not in train:
            raise ValueError("fit requires past sales with sold_price and date_sold")
        self.train = train.copy()
        self.train["date_sold"] = pd.to_datetime(self.train.date_sold, errors="raise")
        if self.train.sold_price.isna().any() or (self.train.sold_price <= 0).any():
            raise ValueError("training sold prices must be present and positive")
        self._safe = _score_safe_inputs()
        return self

    def _check_cutoff(self, cutoff: pd.Timestamp, homes: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp]:
        if self.train is None:
            raise RuntimeError("call fit(train) before predict")
        cutoff = pd.Timestamp(cutoff).normalize()
        if not cutoff.day == 1:
            raise ValueError("candidate routes require a target-month-start cutoff")
        if (self.train.date_sold >= cutoff).any():
            raise ValueError(f"training rows include a sale at or after cutoff {cutoff.date()}")
        if not homes.ml_num.astype(str).is_unique:
            raise ValueError("prediction listing IDs must be unique")
        if self.track == "off_market" and "list_price" in homes.columns:
            raise ValueError("off-market query features must not include list_price")
        return homes.copy().reset_index(drop=True), cutoff

    def predict(self, homes: pd.DataFrame) -> np.ndarray:
        if self.train is None:
            raise RuntimeError("call fit(train) before predict")
        cutoff = self.train.date_sold.max().to_period("M").to_timestamp() + pd.offsets.MonthBegin(1)
        return self.predict_at(homes, cutoff)

    def predict_at(self, homes: pd.DataFrame, cutoff: pd.Timestamp) -> np.ndarray:
        predictions = self.predict_routes(homes, cutoff)
        return predictions[self.route]

    def predict_routes(self, homes: pd.DataFrame, cutoff: pd.Timestamp) -> dict[str, np.ndarray]:
        raise NotImplementedError


class OffMarketR151Model(CandidateRouteModel):
    def __init__(self, selected_route: str = "R151"):
        if selected_route not in {"R22", "R52", "R151"}:
            raise ValueError(f"unknown off-market route: {selected_route!r}")
        super().__init__("off_market", selected_route)

    def predict_routes(self, homes: pd.DataFrame, cutoff: pd.Timestamp) -> dict[str, np.ndarray]:
        from experiments.bill_greedy import runner as greedy
        from experiments.bill_target74 import r151_offmarket_r93_radius15km_clip035_vs_r22 as r151
        from experiments.bill_target74 import r52_r48_r50_adaptive_meta_blend as r52
        from experiments.bill_target74 import r22_remarks_tfidf as r22
        from experiments.bill_target74 import r32_comparable_r22 as r32_module
        from experiments.model_optimization_v2.pipeline import blend_prices

        homes, cutoff = self._check_cutoff(cutoff, homes)
        train = self.train.copy().reset_index(drop=True)
        month = cutoff.strftime("%Y-%m")
        r22_spec = {}
        r22_base = _predict_history_component(
            route="R151", component="R22_base", track=self.track,
            train=train, homes=homes, cutoff=cutoff, variant=r22.VARIANT,
            variant_spec=r22_spec, fit_rows=train, base_variant="G06_lot_garage",
            run_id=self.run_id,
        )
        detached_train = greedy._specialist_mask(train, "Detached")
        detached_homes = greedy._specialist_mask(homes, "Detached")
        r22_expert = _predict_history_component(
            route="R151", component="R22_detached_expert", track=self.track,
            train=train, homes=homes.loc[detached_homes].copy(), cutoff=cutoff,
            variant=r22.VARIANT, variant_spec=r22_spec,
            fit_rows=train.loc[detached_train].copy(), base_variant="G06_lot_garage",
            run_id=self.run_id,
        )
        r22_prediction = greedy._blend_specialist_predictions(
            np.round(r22_base), np.round(r22_expert), homes, "Detached", 0.75,
        )

        r32_prediction = {}
        for component, fit_rows, eval_rows in (
            ("R32_base", train, homes),
            ("R32_detached_expert", train.loc[detached_train].copy(),
             homes.loc[detached_homes].copy()),
        ):
            r32_prediction[component] = _predict_history_component(
                route="R151", component=component, track=self.track,
                train=train, homes=eval_rows, cutoff=cutoff,
                variant=r32_module.VARIANT, variant_spec=r32_module.VARIANT_SPEC,
                fit_rows=fit_rows, base_variant="G06_lot_garage", run_id=self.run_id,
            )
        r32_candidate = greedy._blend_specialist_predictions(
            np.round(r32_prediction["R32_base"]),
            np.round(r32_prediction["R32_detached_expert"]),
            homes, "Detached", 0.75,
        )

        safe = self._safe
        cutoff_month = cutoff.strftime("%Y-%m")
        r47_history = safe["r47"].loc[safe["r47"].date_sold.lt(cutoff)].copy()
        r48, _r48_weights = _r48_candidate(
            r47_history, r22_prediction, r32_candidate, homes, cutoff,
            safe["weight_selection"],
        )
        r50, _r50_weights = _r50_candidate(
            r47_history, r22_prediction, r32_candidate, homes, cutoff,
            safe["weight_selection"],
        )
        r48_r50 = safe["r48_r50"]
        prior_r52 = r48_r50.loc[r48_r50.month.astype(str).lt(cutoff_month)].copy()
        if len(prior_r52) >= r52.MIN_PAST_ROWS:
            r50_share, _share_details = r52._select_share(prior_r52)
        else:
            r50_share = 0.0
        r52_prediction = np.round((1.0 - r50_share) * r48 + r50_share * r50)
        correction, _diagnostics = r151._fit_correction(
            safe["r151"],
            homes[["house_type_name", "latitude", "longitude"]].copy(),
            cutoff,
        )
        r151_prediction = np.rint(r52_prediction * np.exp(correction))
        return {"R22": r22_prediction, "R52": r52_prediction, "R151": r151_prediction}


class ListAwareR97Model(CandidateRouteModel):
    def __init__(self):
        super().__init__("list_aware", "R97")

    def predict_routes(self, homes: pd.DataFrame, cutoff: pd.Timestamp) -> dict[str, np.ndarray]:
        from catboost import CatBoostRegressor
        from experiments.bill_greedy import runner as greedy
        from experiments.bill_target74 import r22_remarks_tfidf as r22
        from experiments.bill_target74 import r35_fixed_origin as r35
        from experiments.bill_target74 import r97_past_oof_catboost_residual_meta as r97
        from experiments.model_optimization_v2.pipeline import blend_prices

        homes, cutoff = self._check_cutoff(cutoff, homes)
        train = self.train.copy().reset_index(drop=True)
        detached_train = greedy._specialist_mask(train, "Detached")
        detached_homes = greedy._specialist_mask(homes, "Detached")

        base = _predict_history_component(
            route="R97", component="R22_core", track=self.track,
            train=train, homes=homes, cutoff=cutoff, variant=r22.VARIANT,
            variant_spec={}, fit_rows=train, run_id=self.run_id,
        )
        frozen = _fit_hgbr(train, homes, cutoff, "R97", run_id=self.run_id)
        r22_candidate = blend_prices(base, frozen, challenger_weight=0.75)

        expert = _predict_history_component(
            route="R97", component="R35_detached_expert", track=self.track,
            train=train, homes=homes.loc[detached_homes].copy(), cutoff=cutoff,
            variant=r22.VARIANT, variant_spec={},
            fit_rows=train.loc[detached_train].copy(), run_id=self.run_id,
        )
        r35_expert = blend_prices(expert, frozen[detached_homes], challenger_weight=0.75)
        r35_prediction = greedy._overlay_specialist_predictions(
            r22_candidate, homes, r35_expert, "Detached",
        )

        safe_oof = self._safe["r97"]
        past = safe_oof.loc[safe_oof.date_sold.lt(cutoff)].copy()
        query_frame = homes.copy()
        query_frame["base_prediction"] = r35_prediction
        query_x, categorical = r97._build_features(query_frame, "list_aware")
        correction = np.zeros(len(homes), dtype=float)
        if len(past) >= r97.MIN_HISTORY_ROWS:
            history_x, _ = r97._build_features(past, "list_aware")
            raw_target = np.log(
                past.sold_price.to_numpy(float) / past.base_prediction.to_numpy(float)
            )
            target = np.clip(raw_target, -r97.RESIDUAL_TARGET_CLIP, r97.RESIDUAL_TARGET_CLIP)
            context = {"route": "R97", "component": "CatBoost_residual_meta",
                       "track": self.track, "cutoff": cutoff.isoformat(),
                       "training_rows": len(past), "evaluation_rows": len(homes),
                       "run_id": self.run_id}
            def fit_meta_model():
                meta_model = CatBoostRegressor(**r97.CATBOOST_CONFIG)
                meta_model.fit(history_x, target, cat_features=categorical)
                return {
                    "predictions": np.asarray(meta_model.predict(query_x), dtype=float),
                    "model_family": "catboost",
                    "model_config": r97.CATBOOST_CONFIG,
                    "feature_columns": list(history_x.columns),
                }
            fitted_meta = _run_submission_fit(context, fit_meta_model)
            raw_correction = fitted_meta["predictions"]
            if not np.isfinite(raw_correction).all():
                raise RuntimeError("R97 residual meta-model produced non-finite corrections")
            correction = r97.CORRECTION_SHRINKAGE * np.clip(
                raw_correction, -r97.PREDICTION_RESIDUAL_CLIP,
                r97.PREDICTION_RESIDUAL_CLIP,
            )
        candidate = np.rint(r35_prediction * np.exp(correction))
        if not np.isfinite(candidate).all() or (candidate <= 0).any():
            raise RuntimeError("R97 produced invalid prices")
        return {"R35": np.asarray(r35_prediction, dtype=float),
                "R97": np.asarray(candidate, dtype=float)}


def model_manifest() -> dict:
    manifest = json.loads((ASSET_DIR / "manifest.json").read_text(encoding="utf-8"))
    source_files = {name: sha256_text(ROOT / name) for name in REPRODUCIBILITY_FILES}
    fit_ledger = {}
    if FIT_LEDGER.is_file():
        fit_ledger = {"path": str(FIT_LEDGER.relative_to(ROOT)).replace("\\", "/"),
                      "sha256": sha256_text(FIT_LEDGER),
                      "events": sum(1 for line in FIT_LEDGER.read_text(encoding="utf-8").splitlines() if line.strip())}
    return {
        "training_data_sha256": manifest["training_data_sha256"],
        "oof_manifest_sha256": sha256_text(ASSET_DIR / "manifest.json"),
        "assets": {
            path.name: sha256_text(path)
            for path in sorted(ASSET_DIR.glob("*.csv"))
        },
        "source_files_sha256": source_files,
        "fit_ledger": fit_ledger,
        "routes": {
            "off_market": {"selected": "R52", "comparators": ["R22", "R151"],
                           "selection_basis": "lowest MdAPE in retrospective standard validation",
                           "registered_comparator": "R22"},
            "list_aware": {"selected": "R97", "comparator": "R35",
                           "minimum_meta_history_rows": 4000},
        },
        "python": os.sys.version.split()[0],
        "dependencies": {
            name: metadata.version(name)
            for name in ("numpy", "pandas", "scikit-learn", "lightgbm", "catboost", "scipy")
        },
        "reserved_month_labels_read": False,
        "hidden_test_labels_read": False,
    }
