"""R67: current-parent screen for reliability-weighted unit residual fallback."""
from __future__ import annotations

import json
import platform
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

from data import hide_answers
from experiments.bill_greedy import history, runner as greedy
from experiments.bill_target74 import r03_catboost as helpers
from experiments.bill_target74 import r04_ablation as budget
from experiments.bill_target74 import r22_remarks_tfidf as r22
from experiments.bill_target74 import r58_past_log_residual_slope_calibration as r58
from experiments.bill_target74 import r65_cold_start_current_parents as r65
from experiments.bill_target74 import runner as campaign
from experiments.model_optimization_v2.folds import build_month_fold
from experiments.model_optimization_v2.pipeline import blend_prices


CARD = "R67_current_parent_unit_to_building_residual_fallback"
VARIANT = "BILL_R67_unit_to_building_fallback"
RELIABILITY = {
    "unit_history_reliability_pseudocount": 5.0,
    "unit_history_reliability_half_life_months": 24.0,
    "unit_history_fallback_to_building": True,
}
PARENTS = {"off_market": r58.R52_RUN, "list_aware": r58.R35_DEV_RUN}
SCREEN_MONTHS = tuple(campaign.PROTOCOL["screen_months"])
EXPECTED_FIT_STARTS = 6
OFF_MARKET_BASE = "G06_lot_garage"
OFF_MARKET_EXPERT_WEIGHT = 0.75
LIST_AWARE_CORE_WEIGHT = 0.75


def r67_fit_id(run_id: str, track: str, month: str, component: str) -> str:
    if not run_id or any(char in run_id for char in "\\/:"):
        raise ValueError("run_id must be a non-empty path-safe identifier")
    expected = {"off_market": {"base", "expert"}, "list_aware": {"detached_core"}}
    if track not in expected or month not in SCREEN_MONTHS or component not in expected[track]:
        raise ValueError(f"unregistered R67 fit: {track}/{month}/{component}")
    return f"{campaign.PROTOCOL['campaign_id']}:{CARD}:{track}:{month}:{component}:{run_id}"


def _save_protocol() -> None:
    campaign.PROTOCOL_PATH.write_text(
        json.dumps(campaign.PROTOCOL, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _verified_inputs() -> dict:
    return r65._verified_inputs()


def _parent_hashes(inputs: dict) -> dict:
    result = {}
    for track, item in inputs.items():
        if not set(SCREEN_MONTHS).issubset(set(item[4].month.astype(str))):
            raise RuntimeError(f"{track} current-parent artifact lacks a registered screen month")
        result[track] = {
            "run_id": item[0]["run_id"],
            "result_sha256": helpers._sha256(item[1]),
            "paired_predictions_sha256": helpers._sha256(item[2]),
            "manifest_sha256": helpers._sha256(item[1].parent / "run_manifest.json"),
        }
    return result


def _preflight(inputs: dict) -> dict:
    """Validate only the two registered development folds before any fit."""
    previous = greedy.VARIANTS.get(VARIANT)
    if previous is not None:
        raise RuntimeError(f"temporary R67 variant key already exists: {VARIANT}")
    greedy.VARIANTS[VARIANT] = dict(RELIABILITY)
    support = {}
    try:
        for month in SCREEN_MONTHS:
            listings, _, _, _, _ = greedy._load_fold(month)
            cutoff = pd.Period(month, freq="M").start_time
            if pd.to_datetime(listings.date_sold, errors="raise").max() >= cutoff + pd.offsets.MonthBegin(1):
                raise RuntimeError(f"R67 fold includes sales at or after the next-month cutoff: {month}")
            for track in PARENTS:
                base_variant = OFF_MARKET_BASE if track == "off_market" else None
                feature_options, include_comparables = greedy._history_feature_options(
                    base_variant=base_variant, variant=VARIANT
                )
                expected_options = dict(RELIABILITY)
                if track == "off_market":
                    expected_options["include_property_extras"] = True
                if include_comparables or feature_options != expected_options:
                    raise RuntimeError(
                        f"R67 {track} feature options differ from its registered architecture: {feature_options}"
                    )
                train, homes, truth = r65._prepare_track_fold(
                    listings, month, track, build_month_fold, hide_answers
                )
                ids = homes.ml_num.astype(str).tolist()
                aligned_parent = r65._aligned_parent_rows(inputs, track, month, ids, truth)
                # Run the exact strict-as-of transform without fitting, and check feature alignment.
                train_history = history.building_history_features(
                    train, train, track=track, training_rows=True, **feature_options
                )
                eval_history = history.building_history_features(
                    train, homes, track=track, cutoff=cutoff, **feature_options
                )
                if list(train_history.columns) != list(eval_history.columns):
                    raise RuntimeError(f"R67 history columns differ for {month}/{track}")
                required = {"u_res_mean", "u_last_res", "b_res_mean"}
                if not required.issubset(train_history.columns) or not required.issubset(eval_history.columns):
                    raise RuntimeError(f"R67 fallback features are missing for {month}/{track}")
                item = {
                    "training_rows": int(len(train)), "evaluation_rows": int(len(homes)),
                    "parent_rows_aligned": int(len(aligned_parent)),
                    "history_feature_columns": int(len(train_history.columns)),
                    "training_u_res_mean_present": int(train_history.u_res_mean.notna().sum()),
                    "training_b_res_mean_present": int(train_history.b_res_mean.notna().sum()),
                    "evaluation_u_res_mean_present": int(eval_history.u_res_mean.notna().sum()),
                    "evaluation_b_res_mean_present": int(eval_history.b_res_mean.notna().sum()),
                }
                if track == "off_market":
                    detached_train = train.loc[greedy._specialist_mask(train, "Detached")]
                    detached_eval = homes.loc[greedy._specialist_mask(homes, "Detached")]
                    if detached_train.empty or detached_eval.empty:
                        raise RuntimeError(f"R67 off-market Detached expert lacks rows for {month}")
                    item["detached_training_rows"] = int(len(detached_train))
                    item["detached_evaluation_rows"] = int(len(detached_eval))
                else:
                    detached_train = train.loc[greedy._specialist_mask(train, "Detached")]
                    detached_eval = homes.loc[greedy._specialist_mask(homes, "Detached")]
                    if detached_train.empty or detached_eval.empty:
                        raise RuntimeError(f"R67 list-aware Detached core lacks rows for {month}")
                    cache = greedy.OUTPUT / "frozen_hgbr" / "list_aware" / f"{month}.csv"
                    if not cache.is_file():
                        raise FileNotFoundError(f"R67 frozen HGBR cache is missing: {cache}")
                    full_hgbr = helpers._verified_hgbr_cache(greedy, month, track, train, homes)
                    if len(full_hgbr) != len(homes):
                        raise RuntimeError(f"R67 frozen HGBR rows do not align with the full {month} fold")
                    item["detached_training_rows"] = int(len(detached_train))
                    item["detached_evaluation_rows"] = int(len(detached_eval))
                    item["full_hgbr_cache_rows"] = int(len(full_hgbr))
                support[f"{track}/{month}"] = item
    finally:
        if previous is None:
            greedy.VARIANTS.pop(VARIANT, None)
        else:
            greedy.VARIANTS[VARIANT] = previous
    return support


def register_card() -> dict:
    existing = campaign.PROTOCOL["cards"].get(CARD)
    if existing:
        if existing.get("status") == "registered" and existing.get("source_sha256") == helpers._sha256(Path(__file__)):
            return existing
        if existing.get("status") == "screen_failed_runtime" and existing.get("fit_starts") == 0:
            source = budget._source_identity()
            inputs = _verified_inputs()
            hashes = _parent_hashes(inputs)
            support = _preflight(inputs)
            cache_hashes = {
                month: helpers._sha256(greedy.OUTPUT / "frozen_hgbr" / "list_aware" / f"{month}.csv")
                for month in SCREEN_MONTHS
            }
            if (source["data_sha256"] != existing.get("data_sha256")
                    or hashes != existing.get("parent_artifact_hashes")
                    or cache_hashes != existing.get("frozen_hgbr_cache_hashes")):
                raise RuntimeError("R67 zero-fit setup recovery stopped: data, parent, or cache inputs changed")
            previous_attempt = {
                "run_id": existing.get("run_id"), "failure_path": existing.get("failure_path"),
                "failure_reason": existing.get("failure_reason"), "fit_starts": 0,
                "source_sha256": existing.get("source_sha256"),
            }
            existing.setdefault("zero_fit_setup_failures", []).append(previous_attempt)
            existing.update({
                "status": "registered", "source_sha256": helpers._sha256(Path(__file__)),
                "preflight_support": support, "amended_before_first_fit": True,
                "amendment_reason": "remove a redundant in-run preflight that collided with the already-installed temporary feature variant",
            })
            for key in ("run_id", "fit_starts", "failure_path", "failure_reason"):
                existing.pop(key, None)
            _save_protocol()
            campaign._append({
                "event": "card_amended_before_first_fit", "card": CARD,
                "previous_run_id": previous_attempt["run_id"],
                "previous_failure": previous_attempt["failure_reason"],
                "fit_starts": 0, "source_sha256": existing["source_sha256"],
                "reason": existing["amendment_reason"],
                "reserved_months_labels_opened": False, "hidden_labels_opened": False,
            })
            campaign._write_progress(
                "r67_amended_before_first_fit",
                next_action="run the amended R67 screen under a new run ID; prior attempt recorded zero fits",
                r67_card=CARD,
            )
            return existing
        raise RuntimeError(f"R67 is already {existing.get('status')}; inspect its result before registering again")

    source = budget._source_identity()
    inputs = _verified_inputs()
    hashes = _parent_hashes(inputs)
    support = _preflight(inputs)
    cache_hashes = {
        month: helpers._sha256(greedy.OUTPUT / "frozen_hgbr" / "list_aware" / f"{month}.csv")
        for month in SCREEN_MONTHS
    }
    registration = {
        "status": "registered", "tracks": list(PARENTS), "screen_months": list(SCREEN_MONTHS),
        "expected_fit_starts": EXPECTED_FIT_STARTS, "parents": PARENTS,
        "parent_artifact_hashes": hashes, "frozen_hgbr_cache_hashes": cache_hashes,
        "method": (
            "Change only strict-as-of unit residual feature construction: blend u_res_mean and u_last_res with "
            "same-coordinate b_res_mean using w=n/(n+5)*2^(-age/24), falling back to whichever value is available. "
            "Keep R22 fold-scoped remarks TF-IDF, LightGBM family/configuration, objectives, G06 base, Detached "
            "75/25 off-market expert route, and R35 Detached-only list-aware 75/25 frozen-HGBR blend unchanged. "
            "Rerun the off-market R22 base/expert on both screen folds; refit only the R35 Detached core for list-aware."
        ),
        "difference_from_prior_work": (
            "Historical R11 tested the same fixed reliability fallback against the original G19/G01 setup and without "
            "R22 remarks TF-IDF. R67 retests only that fallback under current R52/R35 parent architectures; its parent "
            "rows and frozen HGBR caches are hash-pinned before fitting."
        ),
        "reliability": RELIABILITY,
        "primary_parent": {"off_market": "R52 candidate_prediction", "list_aware": "R35 candidate_prediction"},
        "secondary_reference": "R22 validated parent_prediction on the same rows",
        "preflight_support": support,
        "screen_gate": {
            "minimum_relative_pooled_mdape_improvement_percent": campaign.PROTOCOL["gates"]["screen_relative_mdape_improvement_percent"],
            "both_screen_months_must_improve": True,
            "maximum_mape_regression_percent": campaign.PROTOCOL["gates"]["screen_mape_regression_max_percent"],
        },
        "source_sha256": helpers._sha256(Path(__file__)), "data_sha256": source["data_sha256"],
        "reserve_validation_eligible": False,
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    }
    campaign.PROTOCOL["cards"][CARD] = registration
    _save_protocol()
    campaign._append({
        "event": "card_registered", "card": CARD, "tracks": list(PARENTS),
        "screen_months": list(SCREEN_MONTHS), "expected_fit_starts": EXPECTED_FIT_STARTS,
        "parents": PARENTS, "parent_artifact_hashes": hashes,
        "source_sha256": registration["source_sha256"], "data_sha256": source["data_sha256"],
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    campaign._write_progress(
        "r67_registered", next_action="verify registered parent/cache hashes, then run the six R67 screen fits",
        r67_card=CARD,
    )
    return registration


def _report(result: dict) -> str:
    lines = [
        "# R67：当前父模型 unit-to-building residual fallback 筛选", "",
        f"Campaign：`{result['campaign_id']}`；run：`{result['run_id']}`。",
        "唯一机制变化：仅将严格 as-of 的 `u_res_mean`、`u_last_res` 按固定可靠性权重回退至同坐标 `b_res_mean`；权重 `w=n/(n+5) × 2^(-age/24)`。",
        "保留 R22 remarks TF-IDF、LightGBM 配置与目标、off-market G06/Detached 75/25 路由，以及 list-aware R35 Detached 专家和冻结 HGBR 25% 混合。",
        "", "## 双月 pooled 结果", "",
        "| 赛道 | 父→候选 MdAPE | 相对改善 | MAPE变化 | 筛选通过 |",
        "|---|---:|---:|---:|---|",
    ]
    for track, score in result["tracks"].items():
        lines.append(
            f"| {track} | {score['parent']['mdape_percent']:.5f}% → {score['candidate']['mdape_percent']:.5f}% | "
            f"{score['relative_mdape_improvement_percent']:+.3f}% | "
            f"{score['mape_relative_change_percent']:+.3f}% | {'是' if score['qualifies_screen'] else '否'} |"
        )
    lines.extend(["", "## 逐月 MdAPE", "", "| 赛道 | 月份 | 父 MdAPE | 候选 MdAPE | 相对改善 |",
                  "|---|---|---:|---:|---:|"])
    for track, score in result["tracks"].items():
        for month, values in score["monthly"].items():
            lines.append(f"| {track} | {month} | {values['parent']['mdape_percent']:.5f}% | "
                         f"{values['candidate']['mdape_percent']:.5f}% | "
                         f"{values['relative_mdape_improvement_percent']:+.3f}% |")
    lines.extend([
        "", "## 解释与记录", "",
        "Screen gate：pooled MdAPE 相对改善至少 0.25%、两个筛选月都改善、MAPE 相对退化不超过 1%。",
        "仅过筛赛道可申请八个月开发确认；筛选结果不会自动替换父模型。2025-05/08 与隐藏标签未读取。",
        "历史 R11 的 G19/G01 结果来自不同父结构且不含 R22 TF-IDF，不能当作 R67 的直接比较。",
        "", "- 逐房配对预测：`paired_predictions.csv`",
        "- 单次 fit 与组件参数：`component_manifest.json`",
        "- 数据/父结果/源码哈希：`run_manifest.json`",
        "- 事件与每次 fit：campaign `fits.jsonl`", "",
    ])
    return "\n".join(lines)


def run_screen() -> dict:
    card = campaign.PROTOCOL["cards"].get(CARD)
    if not card or card.get("status") != "registered":
        raise RuntimeError("R67 must be preregistered before any training")
    if card.get("source_sha256") != helpers._sha256(Path(__file__)):
        raise RuntimeError("R67 source changed after registration")
    source = budget._source_identity()
    inputs = _verified_inputs()
    hashes = _parent_hashes(inputs)
    if hashes != card["parent_artifact_hashes"]:
        raise RuntimeError("R67 parent artifacts changed after registration")
    cache_hashes = {
        month: helpers._sha256(greedy.OUTPUT / "frozen_hgbr" / "list_aware" / f"{month}.csv")
        for month in SCREEN_MONTHS
    }
    if cache_hashes != card["frozen_hgbr_cache_hashes"]:
        raise RuntimeError("R67 frozen HGBR cache changed after registration")
    support = _preflight(inputs)
    if support != card["preflight_support"]:
        raise RuntimeError("R67 preflight support changed after registration")

    started = datetime.now(timezone.utc)
    run_id = "bill_target74_r67_unit_building_fallback_" + started.strftime("%Y%m%dT%H%M%SZ")
    out = campaign.CAMPAIGN_ROOT / "batches" / run_id
    out.mkdir(parents=True, exist_ok=False)
    context: dict[str, str] = {}
    fit_details: list[dict] = []
    components: list[dict] = []
    rows_by_track: dict[str, list[pd.DataFrame]] = {track: [] for track in PARENTS}
    campaign._append({
        "event": "card_start", "card": CARD, "run_id": run_id,
        "screen_months": list(SCREEN_MONTHS), "expected_fit_starts": EXPECTED_FIT_STARTS,
        "parents": PARENTS, "parent_artifact_hashes": hashes,
        "reliability": RELIABILITY, "data_sha256": source["data_sha256"],
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })

    previous_variant = greedy.VARIANTS.get(VARIANT)
    if previous_variant is not None:
        raise RuntimeError(f"temporary R67 variant key already exists: {VARIANT}")
    old_fit_record = greedy._fit_record
    old_hgbr = greedy._frozen_hgbr_predictions
    greedy.VARIANTS[VARIANT] = dict(RELIABILITY)

    def record_fit(parent_fit_id: str, fit_fn):
        limits = campaign.PROTOCOL["limits"]
        if campaign.limit_reached(budget._fit_count(), limits["exploration_fit_starts"]):
            raise RuntimeError("R67 exploration fit-start limit reached")
        elapsed = budget._campaign_elapsed_seconds()
        if campaign.limit_exceeded(elapsed, None if limits["exploration_minutes"] is None
                                   else limits["exploration_minutes"] * 60):
            raise RuntimeError("R67 exploration time limit reached")
        batch_elapsed = (datetime.now(timezone.utc) - started).total_seconds()
        if campaign.limit_exceeded(batch_elapsed, None if limits["batch_minutes"] is None
                                   else limits["batch_minutes"] * 60):
            raise RuntimeError("R67 batch time limit reached before fit start")
        fit_id = r67_fit_id(run_id, context["track"], context["month"], context["component"])
        fit_rows = dict(context)
        campaign._append({
            "event": "fit_start", "card": CARD, "run_id": run_id, "fit_id": fit_id,
            "parent_fit_id": parent_fit_id, **fit_rows,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "data_sha256": source["data_sha256"], "reliability": RELIABILITY,
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        tick = time.perf_counter()
        try:
            fitted = fit_fn()
        except Exception as exc:
            campaign._append({"event": "fit_complete", "card": CARD, "run_id": run_id,
                              "fit_id": fit_id, "status": "failed",
                              "seconds": time.perf_counter() - tick,
                              "error": f"{type(exc).__name__}: {exc}"})
            raise
        seconds = time.perf_counter() - tick
        try:
            predictions = np.asarray(fitted["predictions"], dtype=float)
            expected_n = int(context["evaluation_rows"])
            if predictions.shape != (expected_n,) or not np.isfinite(predictions).all() or (predictions <= 0).any():
                raise RuntimeError(f"R67 fit produced invalid or misaligned predictions: {fit_id}")
            prediction_path = out / f"component_{len(fit_details):02d}.csv"
            pd.DataFrame({"ml_num": context["evaluation_ids"],
                          "predicted_price": predictions}).to_csv(prediction_path, index=False)
        except Exception as exc:
            campaign._append({"event": "fit_complete", "card": CARD, "run_id": run_id,
                              "fit_id": fit_id, "status": "failed_after_fit",
                              "seconds": seconds, "error": f"{type(exc).__name__}: {exc}"})
            raise
        detail = {
            "fit_id": fit_id, "parent_fit_id": parent_fit_id, **fit_rows,
            "seconds": seconds, "status": "complete", "model_family": fitted["model_family"],
            "model_config": fitted["model_config"], "target": fitted["target"],
            "actual_objective": fitted["actual_objective"],
            "training_rows": fitted["training_rows"], "evaluation_rows": fitted["evaluation_rows"],
            "feature_columns": fitted["feature_columns"],
            "prediction_sha256": fitted["prediction_sha256"],
            "saved_predictions": {"path": prediction_path.name, "rows": expected_n,
                                  "sha256": helpers._sha256(prediction_path)},
        }
        fit_details.append(detail)
        campaign._append({"event": "fit_complete", "card": CARD, "run_id": run_id, **detail})
        if campaign.limit_exceeded(seconds, limits["single_fit_seconds"]):
            raise TimeoutError(f"R67 fit exceeded {limits['single_fit_seconds']} seconds")
        return fitted

    greedy._fit_record = record_fit
    greedy._frozen_hgbr_predictions = lambda month, track, train, homes: helpers._verified_hgbr_cache(
        greedy, month, track, train, homes
    )
    try:
        # The parent, ID, strict-as-of, and full-fold cache checks ran immediately
        # before this run was opened, and are pinned to the registered card.
        for month in SCREEN_MONTHS:
            listings, _, _, _, _ = greedy._load_fold(month)
            cutoff = pd.Period(month, freq="M").start_time
            for track in PARENTS:
                train, homes, truth = r65._prepare_track_fold(
                    listings, month, track, build_month_fold, hide_answers
                )
                ids = homes.ml_num.astype(str).tolist()
                aligned_parent = r65._aligned_parent_rows(inputs, track, month, ids, truth)
                parent = pd.DataFrame({
                    "ml_num": ids, "sold_price": truth.sold_price.to_numpy(float),
                    "predicted_price": aligned_parent.candidate_prediction.to_numpy(float),
                })
                if track == "off_market":
                    context.clear(); context.update({
                        "track": track, "month": month, "component": "base",
                        "training_rows": int(len(train)), "evaluation_rows": int(len(homes)),
                        "evaluation_ids": ids, "date_cutoff_exclusive": cutoff.isoformat(),
                    })
                    transformer, restore = r22._install_text_features(train, homes, track)
                    try:
                        base = greedy._fit_history_candidate(
                            month, track, train, homes, base_variant=OFF_MARKET_BASE,
                            variant=VARIANT, fit_tag=f"R67-base-{month}", fit_rows=train,
                        )
                    finally:
                        restore()
                    components.append({
                        "track": track, "month": month, "component": "base",
                        "model_family": base["model_family"], "model_config": base["model_config"],
                        "target": base["target"], "actual_objective": base["actual_objective"],
                        "feature_columns": base["feature_columns"],
                        "training_rows": base["training_rows"],
                        "text_feature_count": len(transformer.columns_ or []),
                    })

                    detached_train = train.loc[greedy._specialist_mask(train, "Detached")].copy()
                    detached_homes = homes.loc[greedy._specialist_mask(homes, "Detached")].copy()
                    context.clear(); context.update({
                        "track": track, "month": month, "component": "expert",
                        "training_rows": int(len(detached_train)),
                        "evaluation_rows": int(len(detached_homes)),
                        "evaluation_ids": detached_homes.ml_num.astype(str).tolist(),
                        "date_cutoff_exclusive": cutoff.isoformat(),
                    })
                    transformer, restore = r22._install_text_features(detached_train, detached_homes, track)
                    try:
                        expert = greedy._fit_history_candidate(
                            month, track, train, detached_homes, base_variant=OFF_MARKET_BASE,
                            variant=VARIANT, fit_tag=f"R67-expert-{month}", fit_rows=detached_train,
                        )
                    finally:
                        restore()
                    components.append({
                        "track": track, "month": month, "component": "expert",
                        "model_family": expert["model_family"], "model_config": expert["model_config"],
                        "target": expert["target"], "actual_objective": expert["actual_objective"],
                        "feature_columns": expert["feature_columns"],
                        "training_rows": expert["training_rows"],
                        "text_feature_count": len(transformer.columns_ or []),
                    })
                    candidate_prediction = greedy._blend_specialist_predictions(
                        np.round(base["predictions"]), np.round(expert["predictions"]),
                        homes, "Detached", OFF_MARKET_EXPERT_WEIGHT,
                    )
                else:
                    detached_train = train.loc[greedy._specialist_mask(train, "Detached")].copy()
                    detached_homes = homes.loc[greedy._specialist_mask(homes, "Detached")].copy()
                    detached_mask = greedy._specialist_mask(homes, "Detached")
                    context.clear(); context.update({
                        "track": track, "month": month, "component": "detached_core",
                        "training_rows": int(len(detached_train)),
                        "evaluation_rows": int(len(detached_homes)),
                        "evaluation_ids": detached_homes.ml_num.astype(str).tolist(),
                        "date_cutoff_exclusive": cutoff.isoformat(),
                    })
                    transformer, restore = r22._install_text_features(detached_train, detached_homes, track)
                    try:
                        core = greedy._fit_history_candidate(
                            month, track, train, detached_homes, variant=VARIANT,
                            fit_tag=f"R67-list-detached-core-{month}", fit_rows=detached_train,
                        )
                    finally:
                        restore()
                    components.append({
                        "track": track, "month": month, "component": "detached_core",
                        "model_family": core["model_family"], "model_config": core["model_config"],
                        "target": core["target"], "actual_objective": core["actual_objective"],
                        "feature_columns": core["feature_columns"],
                        "training_rows": core["training_rows"],
                        "text_feature_count": len(transformer.columns_ or []),
                    })
                    # Validate HGBR against the full fold first, then slice to Detached.
                    full_hgbr = helpers._verified_hgbr_cache(greedy, month, track, train, homes)
                    if len(full_hgbr) != len(homes):
                        raise RuntimeError(f"R67 frozen HGBR row count differs for {month}")
                    detached_candidate = blend_prices(
                        core["predictions"], full_hgbr[detached_mask],
                        challenger_weight=LIST_AWARE_CORE_WEIGHT,
                    )
                    candidate_prediction = parent.predicted_price.to_numpy(float).copy()
                    candidate_prediction[detached_mask] = detached_candidate

                candidate = pd.DataFrame({
                    "ml_num": ids, "sold_price": truth.sold_price.to_numpy(float),
                    "predicted_price": candidate_prediction,
                })
                paired = helpers._candidate_frame(month, track, truth, parent, candidate)
                paired["validated_R22_parent_prediction"] = aligned_parent.parent_prediction.to_numpy(float)
                paired["primary_parent"] = PARENTS[track]
                rows_by_track[track].append(paired)
                campaign._append({
                    "event": "screen_fold_complete", "card": CARD, "run_id": run_id,
                    "track": track, "month": month, "parent_id": PARENTS[track],
                    "parent_artifact_sha256": hashes[track]["paired_predictions_sha256"],
                    "date_cutoff_exclusive": cutoff.isoformat(),
                    "reserved_months_labels_opened": False, "hidden_labels_opened": False,
                })
        if len(fit_details) != EXPECTED_FIT_STARTS:
            raise RuntimeError(f"R67 expected {EXPECTED_FIT_STARTS} fits, observed {len(fit_details)}")
    except Exception as exc:
        failure_path = out / "failure_checkpoint.json"
        failure = {
            "run_id": run_id, "card": CARD, "error": f"{type(exc).__name__}: {exc}",
            "fit_starts": len(fit_details), "fit_details": fit_details,
            "completed_components": components,
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        }
        failure_path.write_text(json.dumps(failure, indent=2, default=str), encoding="utf-8")
        campaign._append({"event": "card_failed", "card": CARD, "run_id": run_id,
                          "error": failure["error"], "fit_starts": len(fit_details),
                          "failure_path": str(failure_path.relative_to(campaign.ROOT)).replace("\\", "/"),
                          "reserved_months_labels_opened": False, "hidden_labels_opened": False})
        card.update({"status": "screen_failed_runtime", "run_id": run_id,
                     "fit_starts": len(fit_details),
                     "failure_path": str(failure_path.relative_to(campaign.ROOT)).replace("\\", "/"),
                     "failure_reason": failure["error"]})
        _save_protocol()
        campaign._write_progress("r67_runtime_failed", next_action="inspect the R67 failure checkpoint and fit ledger before any retry",
                                 latest_error=failure["error"])
        raise
    finally:
        greedy._fit_record = old_fit_record
        greedy._frozen_hgbr_predictions = old_hgbr
        if previous_variant is None:
            greedy.VARIANTS.pop(VARIANT, None)
        else:
            greedy.VARIANTS[VARIANT] = previous_variant

    paired_all = pd.concat([frame for frames in rows_by_track.values() for frame in frames], ignore_index=True)
    paired_path = out / "paired_predictions.csv"
    paired_all.to_csv(paired_path, index=False)
    tracks = {track: helpers._track_result(pd.concat(frames, ignore_index=True))
              for track, frames in rows_by_track.items()}
    result = {
        "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id, "card": CARD,
        "screen_months": list(SCREEN_MONTHS), "parents": PARENTS,
        "reliability": RELIABILITY, "tracks": tracks,
        "fit_starts": len(fit_details), "reserved_months_labels_opened": False,
        "hidden_labels_opened": False, "data_sha256": source["data_sha256"],
    }
    results_path = out / "r67_screen_results.json"
    results_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    components_path = out / "component_manifest.json"
    components_path.write_text(json.dumps(components, indent=2, default=str), encoding="utf-8")
    code_paths = [
        campaign.PROTOCOL_PATH, Path(__file__), Path(r22.__file__), Path(r58.__file__), Path(r65.__file__),
        Path(budget.__file__), Path(helpers.__file__), Path(campaign.__file__),
        Path(greedy.__file__), Path(history.__file__),
        campaign.ROOT / "experiments/model_optimization_v2/worker.py",
        campaign.ROOT / "experiments/model_optimization_v2/learners.py",
        campaign.ROOT / "experiments/model_optimization_v2/pipeline.py",
        campaign.ROOT / "model_candidate_v2/features.py",
        campaign.ROOT / "model_candidate_v2/winner_config.json",
    ]
    code_paths = list(dict.fromkeys(code_paths))
    run_manifest = {
        "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
        "source_identity": source, "parents": PARENTS,
        "parent_artifact_hashes": hashes, "frozen_hgbr_cache_hashes": cache_hashes,
        "reliability": RELIABILITY,
        "code_sha256": {str(path.relative_to(campaign.ROOT)).replace("\\", "/"): helpers._sha256(path)
                        for path in code_paths},
        "runtime": {"python": platform.python_version(), "lightgbm": metadata.version("lightgbm"),
                    "numpy": np.__version__, "pandas": pd.__version__,
                    "scikit_learn": metadata.version("scikit-learn")},
        "fit_details": fit_details, "components_sha256": helpers._sha256(components_path),
        "paired_predictions_sha256": helpers._sha256(paired_path),
        "results_sha256": helpers._sha256(results_path),
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    }
    manifest_path = out / "run_manifest.json"
    manifest_path.write_text(json.dumps(run_manifest, indent=2, default=str), encoding="utf-8")
    report_path = out / "research_report.md"
    report_path.write_text(_report(result), encoding="utf-8")
    passed = {track: score["qualifies_screen"] for track, score in tracks.items()}
    card.update({
        "status": "screen_passed" if any(passed.values()) else "screen_failed",
        "run_id": run_id, "fit_starts": len(fit_details),
        "result_path": str(results_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        "paired_predictions_path": str(paired_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        "manifest_path": str(manifest_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        "qualifies_by_track": passed,
    })
    _save_protocol()
    campaign._append({"event": "card_complete", "card": CARD, "run_id": run_id,
                      "fit_starts": len(fit_details),
                      "result_path": str(results_path.relative_to(campaign.ROOT)).replace("\\", "/"),
                      "qualifies_by_track": passed,
                      "reserved_months_labels_opened": False, "hidden_labels_opened": False})
    next_action = (
        "run eight-month development confirmation for: " + ", ".join(track for track, ok in passed.items() if ok)
        if any(passed.values()) else "R67 failed its two-month screen; retain R22/R35 and register a distinct single-mechanism hypothesis"
    )
    campaign._write_progress("r67_screen_complete", next_action=next_action,
                             r67_result=str(results_path.relative_to(campaign.ROOT)).replace("\\", "/"),
                             r67_fit_starts=len(fit_details))
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="R67 current-parent unit-to-building residual fallback")
    parser.add_argument("--register", action="store_true", help="register and preflight the card without learner fits")
    args = parser.parse_args()
    result = register_card() if args.register else run_screen()
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str), flush=True)


if __name__ == "__main__":
    main()
