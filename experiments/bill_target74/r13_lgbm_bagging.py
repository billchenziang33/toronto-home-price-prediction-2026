from __future__ import annotations

import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from importlib import metadata

import numpy as np
import pandas as pd

from experiments.bill_target74 import r03_catboost as helpers
from experiments.bill_target74 import r04_ablation as budget
from experiments.bill_target74 import runner as campaign

ROOT = campaign.ROOT

SEEDS = (101, 202, 303)
SAMPLING_FRACTION = 0.8
CARD = "R13_seed_ensemble_lightgbm_row_feature_bagging"
COMPONENTS = {"off_market": ("base", "expert"), "list_aware": ("core",)}


def bagging_overrides(seed: int) -> dict[str, float | int]:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    return {
        "feature_fraction": SAMPLING_FRACTION,
        "bagging_fraction": SAMPLING_FRACTION,
        "bagging_freq": 1,
        "random_state": seed,
        "bagging_seed": seed,
        "feature_fraction_seed": seed,
        "data_random_seed": seed,
    }


def geometric_mean_price_predictions(members: list[np.ndarray]) -> np.ndarray:
    if not members:
        raise ValueError("at least one prediction member is required")
    vectors = [np.asarray(member, dtype=float) for member in members]
    shape = vectors[0].shape
    if len(shape) != 1 or any(vector.shape != shape for vector in vectors):
        raise ValueError("prediction members must be aligned one-dimensional vectors")
    if any(not np.isfinite(vector).all() or (vector <= 0.0).any() for vector in vectors):
        raise ValueError("prediction members must contain finite positive prices")
    result = np.exp(np.mean(np.log(np.stack(vectors, axis=0)), axis=0))
    if not np.isfinite(result).all() or (result <= 0.0).any():
        raise ValueError("geometric ensemble produced invalid prices")
    return result


def r13_fit_id(run_id: str, track: str, month: str, component: str, seed: int) -> str:
    if not run_id or any(char in run_id for char in "\\/"):
        raise ValueError("run_id must be path-safe")
    if track not in COMPONENTS:
        raise ValueError(f"unknown track: {track!r}")
    if month not in campaign.PROTOCOL["screen_months"]:
        raise ValueError(f"month is not a registered screen fold: {month!r}")
    if component not in COMPONENTS[track]:
        raise ValueError(f"component is not valid for {track}: {component!r}")
    if seed not in SEEDS:
        raise ValueError(f"seed is not in the frozen R13 ensemble: {seed!r}")
    return f"{campaign.PROTOCOL['campaign_id']}:{CARD}:{track}:{month}:{component}:seed{seed}:{run_id}"


def _sha256(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _report(result: dict) -> str:
    lines = [
        "# R13：固定采样率的 LightGBM 种子集成筛选", "",
        f"Campaign：`{result['campaign_id']}`；run：`{result['run_id']}`。",
        "唯一新机制：每个父模型组件固定使用 80% feature_fraction、80% bagging_fraction、bagging_freq=1，按 101/202/303 三个 seed 拟合，并在 log-price/log-ratio 预测空间取几何均值。Off-market 保持 G19 的 G06 base + Detached expert 与 75/25 路由；list-aware 保持 G01 的 75% LightGBM + 25% 冻结 HGBR 混合。",
        "筛选只使用 2023-03 和 2024-06；未重选 seed、采样率或融合权重。", "",
        "## 双月配对结果", "",
        "| 赛道 | 父 MdAPE | 候选 MdAPE | 相对改善 | MAPE 变化 | 两月均改善 | 通过筛选 |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for track, score in result["tracks"].items():
        both_screen_months_improved = all(
            month_result["relative_mdape_improvement_percent"] > 0.0
            for month_result in score["monthly"].values()
        )
        lines.append(
            f"| {track} | {score['parent']['mdape_percent']:.5f}% | "
            f"{score['candidate']['mdape_percent']:.5f}% | "
            f"{score['relative_mdape_improvement_percent']:+.3f}% | "
            f"{score['mape_relative_change_percent']:+.3f}% | "
            f"{'是' if both_screen_months_improved else '否'} | "
            f"{'是' if score['qualifies_screen'] else '否'} |"
        )
    lines.extend(["", "## 逐月 MdAPE", "", "| 赛道 | 月份 | 父 MdAPE | 候选 MdAPE | 相对改善 |",
                  "|---|---|---:|---:|---:|"])
    for track, score in result["tracks"].items():
        for month, monthly in score["monthly"].items():
            lines.append(
                f"| {track} | {month} | {monthly['parent']['mdape_percent']:.5f}% | "
                f"{monthly['candidate']['mdape_percent']:.5f}% | "
                f"{monthly['relative_mdape_improvement_percent']:+.3f}% |"
            )
    lines.extend([
        "", "## 实验边界", "",
        f"实际 learner fits：{result['fit_starts']}；固定 seed：{result['seeds']}；采样率：{result['sampling_fraction']:.2f}。",
        "任何一条赛道未达到 pooled MdAPE 相对改善至少 0.25%、两个筛选月都改善、MAPE 退化不超过 1% 的门槛，都不进入完整开发确认。父模型不因筛选结果自动变化。",
        "Reserved 2025-05/08 与隐藏标签未读取。", "",
        "- 逐房配对预测：`paired_predictions.csv`",
        "- 每个组件/seed 的模型配置和预测哈希：`component_manifest.json`",
        "- 数据、父模型和源码哈希：`run_manifest.json`",
    ])
    return "\n".join(lines) + "\n"


def _assert_exploration_budget_open() -> None:
    limits = campaign.PROTOCOL["limits"]
    if campaign.limit_reached(budget._fit_count(), limits["exploration_fit_starts"]):
        raise RuntimeError("R13 exploration fit-start reserve reached")
    if campaign.limit_exceeded(budget._campaign_elapsed_seconds(),
                               None if limits["exploration_minutes"] is None
                               else limits["exploration_minutes"] * 60):
        raise RuntimeError("R13 exploration wall-clock reserve reached")


def run_r13_screen() -> dict:
    from data import hide_answers
    from experiments.bill_greedy import runner as greedy
    from experiments.model_optimization_v2.pipeline import blend_prices

    # Refuse closed campaigns before creating a batch folder or ledger events.
    _assert_exploration_budget_open()
    parents = dict(campaign.PROTOCOL["parent_by_track"])
    source = budget._source_identity()
    started = datetime.now(timezone.utc)
    run_id = "bill_target74_r13_lgbm_bagging_" + started.strftime("%Y%m%dT%H%M%SZ")
    out = campaign.CAMPAIGN_ROOT / "batches" / run_id
    out.mkdir(parents=True, exist_ok=False)
    limits = campaign.PROTOCOL["limits"]
    campaign._append({"event": "card_start", "card": CARD, "run_id": run_id,
                      "fit_start_count": 0, "parents": parents,
                      "screen_months": list(campaign.PROTOCOL["screen_months"]),
                      "seeds": list(SEEDS), "sampling_fraction": SAMPLING_FRACTION,
                      "expected_fit_starts": sum(map(len, COMPONENTS.values())) * len(SEEDS)
                      * len(campaign.PROTOCOL["screen_months"]),
                      "reserved_months_labels_opened": False, "hidden_labels_opened": False})

    old_fit_record = greedy._fit_record
    old_hgbr = greedy._frozen_hgbr_predictions
    added_variants: list[str] = []
    context: dict[str, str | int] = {}
    fit_details: list[dict] = []
    component_manifest: list[dict] = []
    rows_by_track: dict[str, list[pd.DataFrame]] = {track: [] for track in parents}
    try:
        for seed in SEEDS:
            variant = f"R13_bagged_seed_{seed}"
            if variant in greedy.VARIANTS:
                raise RuntimeError(f"R13 variant already exists in the shared runner: {variant}")
            greedy.VARIANTS[variant] = {
                "model_config_overrides": {
                    track: bagging_overrides(seed) for track in parents
                }
            }
            added_variants.append(variant)

        def record_fit(parent_fit_id: str, fit_fn):
            if campaign.limit_reached(budget._fit_count(), limits["exploration_fit_starts"]):
                raise RuntimeError("R13 exploration fit-start reserve reached")
            if campaign.limit_exceeded(budget._campaign_elapsed_seconds(),
                                       None if limits["exploration_minutes"] is None
                                       else limits["exploration_minutes"] * 60):
                raise RuntimeError("R13 exploration wall-clock reserve reached")
            if campaign.limit_exceeded((datetime.now(timezone.utc) - started).total_seconds(),
                                       None if limits["batch_minutes"] is None
                                       else limits["batch_minutes"] * 60):
                raise RuntimeError("R13 batch wall-clock cap reached before fit start")
            fit_id = r13_fit_id(run_id, str(context["track"]), str(context["month"]),
                                str(context["component"]), int(context["seed"]))
            card_context = dict(context)
            campaign._append({"event": "fit_start", "card": CARD, "run_id": run_id,
                              "fit_id": fit_id, "parent_fit_id": parent_fit_id,
                              "data_sha256": source["data_sha256"], **card_context})
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
            detail = {"fit_id": fit_id, "parent_fit_id": parent_fit_id, **card_context,
                      "seconds": seconds,
                      "status": "complete" if not campaign.limit_exceeded(
                          seconds, limits["single_fit_seconds"]) else "over_limit",
                      "model_config": fitted["model_config"], "target": fitted["target"],
                      "feature_columns": fitted["feature_columns"],
                      "training_rows": fitted["training_rows"],
                      "prediction_sha256": fitted["prediction_sha256"]}
            fit_details.append(detail)
            campaign._append({"event": "fit_complete", "card": CARD, "run_id": run_id, **detail})
            if detail["status"] != "complete":
                raise TimeoutError(f"R13 fit exceeded {limits['single_fit_seconds']} seconds")
            return fitted

        greedy._fit_record = record_fit
        greedy._frozen_hgbr_predictions = lambda month, track, train, homes: helpers._verified_hgbr_cache(
            greedy, month, track, train, homes)

        for month in campaign.PROTOCOL["screen_months"]:
            listings, train, _, truth, heldout = greedy._load_fold(month)
            cutoff = (pd.Period(month, freq="M") + 1).start_time
            if pd.to_datetime(listings.date_sold, errors="raise").max() >= cutoff:
                raise RuntimeError(f"R13 fold includes sales at or after cutoff: {month}")
            for track, parent_id in parents.items():
                homes = hide_answers(heldout, track)
                parent, parent_path = budget._check_parent_artifact(
                    greedy, month, track, homes, truth, parent_id)
                member_predictions: dict[str, list[np.ndarray]] = {
                    name: [] for name in COMPONENTS[track]
                }
                if track == "off_market":
                    fit_jobs = [
                        ("base", train, homes, "G06_lot_garage"),
                        ("expert", train.loc[greedy._specialist_mask(train, "Detached")].copy(),
                         homes.loc[greedy._specialist_mask(homes, "Detached")].copy(), "G06_lot_garage"),
                    ]
                    if fit_jobs[1][1].empty or fit_jobs[1][2].empty:
                        raise RuntimeError(f"R13 Detached expert is empty for {month}")
                else:
                    fit_jobs = [("core", train, homes, None)]

                for component, fit_rows, eval_rows, base_variant in fit_jobs:
                    for seed in SEEDS:
                        variant = f"R13_bagged_seed_{seed}"
                        context.clear()
                        context.update({"track": track, "month": month,
                                        "component": component, "seed": seed})
                        fitted = greedy._fit_history_candidate(
                            month, track, train, eval_rows,
                            base_variant=base_variant, variant=variant,
                            fit_tag=f"R13-{component}-{month}-seed{seed}",
                            fit_rows=fit_rows)
                        member_predictions[component].append(fitted["predictions"])
                        component_manifest.append({
                            "track": track, "month": month, "component": component,
                            "seed": seed, "parent_id": parent_id,
                            "model_family": fitted["model_family"],
                            "model_config": fitted["model_config"], "target": fitted["target"],
                            "training_rows": fitted["training_rows"],
                            "evaluation_rows": fitted["evaluation_rows"],
                            "feature_columns": fitted["feature_columns"],
                            "prediction_sha256": fitted["prediction_sha256"],
                        })

                if track == "off_market":
                    base_prediction = np.round(geometric_mean_price_predictions(member_predictions["base"]))
                    expert_prediction = np.round(geometric_mean_price_predictions(member_predictions["expert"]))
                    candidate_prediction = greedy._blend_specialist_predictions(
                        base_prediction, expert_prediction, homes, "Detached", 0.75)
                else:
                    core_prediction = geometric_mean_price_predictions(member_predictions["core"])
                    frozen = greedy._frozen_hgbr_predictions(month, track, train, homes)
                    candidate_prediction = blend_prices(core_prediction, frozen, challenger_weight=0.75)

                candidate = pd.DataFrame({
                    "ml_num": homes.ml_num.astype(str).to_numpy(),
                    "sold_price": truth.sold_price.to_numpy(float),
                    "predicted_price": candidate_prediction,
                })
                rows_by_track[track].append(helpers._candidate_frame(month, track, truth, parent, candidate))
                campaign._append({"event": "screen_fold_complete", "card": CARD, "run_id": run_id,
                                  "track": track, "month": month, "parent_id": parent_id,
                                  "parent_csv_sha256": helpers._sha256(parent_path),
                                  "date_cutoff_exclusive": cutoff.isoformat()})
    except Exception as exc:
        campaign._append({"event": "card_failed", "card": CARD, "run_id": run_id,
                          "error": f"{type(exc).__name__}: {exc}",
                          "fit_starts": len(fit_details),
                          "reserved_months_labels_opened": False,
                          "hidden_labels_opened": False})
        campaign._write_progress("r13_bagging_failed",
                                 next_action="inspect R13 failure record; do not rerun without checking fit ledger",
                                 latest_error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        greedy._fit_record = old_fit_record
        greedy._frozen_hgbr_predictions = old_hgbr
        for variant in added_variants:
            greedy.VARIANTS.pop(variant, None)

    paired = pd.concat([frame for track_frames in rows_by_track.values() for frame in track_frames],
                       ignore_index=True)
    paired_path = out / "paired_predictions.csv"
    paired.to_csv(paired_path, index=False)
    tracks = {
        track: helpers._track_result(pd.concat(frames, ignore_index=True))
        for track, frames in rows_by_track.items()
    }
    result = {
        "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id, "card": CARD,
        "parents": parents, "screen_months": list(campaign.PROTOCOL["screen_months"]),
        "seeds": list(SEEDS), "sampling_fraction": SAMPLING_FRACTION,
        "fit_starts": len(fit_details), "tracks": tracks,
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        "data_sha256": source["data_sha256"],
    }
    result_path = out / "r13_screen_results.json"
    result_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    (out / "component_manifest.json").write_text(
        json.dumps(component_manifest, indent=2, default=str), encoding="utf-8")
    (out / "fit_details.json").write_text(json.dumps(fit_details, indent=2, default=str), encoding="utf-8")
    source_files = [campaign.PROTOCOL_PATH, campaign.PACKAGE / "r13_lgbm_bagging.py",
                    campaign.PACKAGE / "r03_catboost.py", campaign.PACKAGE / "r04_ablation.py",
                    ROOT / "experiments/bill_greedy/runner.py", ROOT / "experiments/bill_greedy/history.py",
                    ROOT / "experiments/model_optimization_v2/learners.py",
                    ROOT / "experiments/model_optimization_v2/pipeline.py",
                    ROOT / "model_candidate_v2/features.py", ROOT / "model_candidate_v2/winner_config.json"]
    manifest = {
        "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
        "source_identity": source, "parents": parents,
        "code_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): _sha256(path)
                        for path in source_files},
        "runtime": {"python": platform.python_version(), "lightgbm": metadata.version("lightgbm"),
                    "numpy": np.__version__, "pandas": pd.__version__},
        "screen_months": list(campaign.PROTOCOL["screen_months"]),
        "seeds": list(SEEDS), "sampling_fraction": SAMPLING_FRACTION,
        "paired_predictions_sha256": _sha256(paired_path),
        "result_sha256": _sha256(result_path),
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out / "research_report.md").write_text(_report(result), encoding="utf-8")
    campaign._append({"event": "card_complete", "card": CARD, "run_id": run_id,
                      "fit_starts": len(fit_details), "result_path": str(result_path.relative_to(ROOT)),
                      "qualifies_by_track": {track: score["qualifies_screen"]
                                             for track, score in tracks.items()},
                      "reserved_months_labels_opened": False, "hidden_labels_opened": False})
    campaign._write_progress("r13_bagging_screen_complete",
                             next_action="confirm only qualifying R13 tracks under the frozen development gates",
                             r13_result=str(result_path.relative_to(ROOT)))
    return result
