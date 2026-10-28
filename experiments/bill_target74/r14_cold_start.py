"""Routing primitives for a strict as-of coordinate cold-start candidate."""
from __future__ import annotations

import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.bill_target74 import r03_catboost as report_helpers
from experiments.bill_target74 import r04_ablation as budget
from experiments.bill_target74 import runner as campaign


ROOT = campaign.ROOT
PARENTS = dict(campaign.PROTOCOL["parent_by_track"])
SCREEN_MONTHS = tuple(campaign.PROTOCOL["screen_months"])
CARD = "R14_asof_cold_start_route"

OFF_MARKET_DETACHED_WEIGHT = 0.75
LIST_AWARE_CORE_WEIGHT = 0.75


def training_cold_start_mask(train: pd.DataFrame, track: str) -> np.ndarray:
    """Select rows with no building sale before their own sale-month cutoff."""
    from experiments.bill_greedy.history import building_history_features

    history = building_history_features(train, train, track=track, training_rows=True)
    return cold_start_mask(train, history)


def evaluation_cold_start_mask(train: pd.DataFrame, homes: pd.DataFrame, *,
                               track: str, cutoff: str | pd.Timestamp) -> np.ndarray:
    """Select evaluation homes with no building sale before the fixed fold cutoff."""
    from experiments.bill_greedy.history import building_history_features

    history = building_history_features(train, homes, track=track, cutoff=cutoff)
    return cold_start_mask(homes, history)


def r14_fit_id(run_id: str, track: str, month: str, component: str) -> str:
    if not run_id or any(char in run_id for char in "\\/"):
        raise ValueError("run_id must be path-safe")
    if track not in PARENTS:
        raise ValueError(f"unknown track: {track!r}")
    if month not in SCREEN_MONTHS:
        raise ValueError(f"month is not a registered screen fold: {month!r}")
    components = {"off_market": {"base", "expert"}, "list_aware": {"core"}}
    if component not in components[track]:
        raise ValueError(f"component is not valid for {track}: {component!r}")
    return f"{campaign.PROTOCOL['campaign_id']}:{CARD}:{track}:{month}:{component}:{run_id}"


def _price_vector(values, label: str) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or not np.isfinite(vector).all() or (vector <= 0.0).any():
        raise ValueError(f"{label} must be a one-dimensional vector of finite positive prices")
    return vector


def cold_start_mask(rows: pd.DataFrame, history_features: pd.DataFrame) -> np.ndarray:
    """Select finite-coordinate rows with no prior building history.

    ``history_features`` must come from ``building_history_features`` for the
    same rows and the appropriate per-sale-month or evaluation cutoff. That
    helper defines the coordinate key at six decimal places and excludes sales
    on the cutoff date.
    """
    required_rows = {"latitude", "longitude"}
    missing_rows = required_rows - set(rows.columns)
    if missing_rows:
        raise ValueError(f"cold-start rows are missing coordinates: {sorted(missing_rows)}")
    if "b_n" not in history_features:
        raise ValueError("cold-start history features must include b_n")
    if not rows.index.equals(history_features.index):
        raise ValueError("rows and history features must align by index and order")

    latitude = pd.to_numeric(rows["latitude"], errors="coerce").to_numpy(dtype=float)
    longitude = pd.to_numeric(rows["longitude"], errors="coerce").to_numpy(dtype=float)
    valid_coordinates = np.isfinite(latitude) & np.isfinite(longitude)
    prior_sales = pd.to_numeric(history_features["b_n"], errors="coerce").to_numpy(dtype=float)
    return valid_coordinates & np.isfinite(prior_sales) & (prior_sales == 0.0)


def off_market_cold_predictions(cold_homes: pd.DataFrame, *, base_predictions,
                                detached_expert_predictions) -> np.ndarray:
    """Apply the frozen G19 base/Detached route inside the cold-start group."""
    if "house_type_name" not in cold_homes:
        raise ValueError("off-market cold homes must include house_type_name")
    base = _price_vector(base_predictions, "base predictions")
    expert = _price_vector(detached_expert_predictions, "Detached expert predictions")
    if base.shape != (len(cold_homes),):
        raise ValueError("base predictions must align with cold homes")
    detached = cold_homes["house_type_name"].astype("string").eq("Detached").fillna(False).to_numpy(dtype=bool)
    if expert.shape != (int(detached.sum()),):
        raise ValueError("Detached expert predictions must align with cold Detached homes")

    result = np.round(base)
    result[detached] = np.round(
        (1.0 - OFF_MARKET_DETACHED_WEIGHT) * result[detached]
        + OFF_MARKET_DETACHED_WEIGHT * np.round(expert)
    )
    return result


def list_aware_cold_predictions(*, core_predictions, frozen_hgbr_predictions) -> np.ndarray:
    """Apply the frozen G01 75% core / 25% HGBR mix inside the cold route."""
    from experiments.model_optimization_v2.pipeline import blend_prices

    core = _price_vector(core_predictions, "list-aware core predictions")
    hgbr = _price_vector(frozen_hgbr_predictions, "frozen HGBR predictions")
    if core.shape != hgbr.shape:
        raise ValueError("list-aware core and frozen HGBR predictions must align")
    return blend_prices(core, hgbr, challenger_weight=LIST_AWARE_CORE_WEIGHT)


def replace_routed_predictions(parent_predictions, routed_predictions,
                               route_mask) -> np.ndarray:
    """Replace route rows while preserving every other parent prediction."""
    parent = _price_vector(parent_predictions, "parent predictions")
    routed = _price_vector(routed_predictions, "routed predictions")
    mask = np.asarray(route_mask)
    if parent.ndim != 1 or routed.ndim != 1:
        raise ValueError("parent and routed predictions must be one-dimensional")
    if mask.dtype != np.dtype(bool) or mask.shape != parent.shape:
        raise ValueError("route mask must be a boolean vector aligned with parent predictions")
    if routed.shape != (int(mask.sum()),):
        raise ValueError("routed predictions must align with the selected route rows")
    result = parent.copy()
    result[mask] = routed
    return result


def _source_identity() -> dict:
    branch = subprocess.run(["git", "branch", "--show-current"], cwd=ROOT,
                            capture_output=True, text=True, check=True).stdout.strip()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True, check=True).stdout.strip()
    if branch != "bill" or commit != campaign.PROTOCOL["parent_commit"]:
        raise RuntimeError(f"expected bill at {campaign.PROTOCOL['parent_commit']}, found {branch} at {commit}")
    if campaign._sha256(campaign.DATA) != campaign.PROTOCOL["data_sha256"]:
        raise RuntimeError("cached listings parquet hash differs from the campaign data hash")
    return {"branch": branch, "commit": commit, "data_sha256": campaign.PROTOCOL["data_sha256"]}


def _report(result: dict) -> str:
    lines = [
        "# R14：严格 as-of building-coordinate 冷启动专家筛选", "",
        f"Campaign：`{result['campaign_id']}`；run：`{result['run_id']}`。",
        "训练只使用各行自身成交月开始时没有历史 building-coordinate 成交且坐标有效的房源；评估路由只使用评估月 cutoff 前可见的历史。",
        "Off-market 在冷启动路由内重训 G19 的 G06 base 与 Detached expert，保留 75/25 专家混合；list-aware 在同一路由内重训 G01 core，并保留冻结 HGBR 25% 混合。路由外逐房沿用父预测。",
        "此结构不同于 G35/G37：候选模型的 base/core 与专家均只在严格 as-of 冷启动训练人群上拟合，而非将单个 G06 冷启动专家混入全体父版。", "",
        "## 双月总体结果", "",
        "| 赛道 | 父 MdAPE | 候选 MdAPE | 相对改善 | MAPE 变化 | 通过筛选 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for track, score in result["tracks"].items():
        lines.append(
            f"| {track} | {score['parent']['mdape_percent']:.5f}% | "
            f"{score['candidate']['mdape_percent']:.5f}% | "
            f"{score['relative_mdape_improvement_percent']:+.3f}% | "
            f"{score['mape_relative_change_percent']:+.3f}% | "
            f"{'是' if score['qualifies_screen'] else '否'} |"
        )
    lines.extend(["", "## 逐月结果与冷启动覆盖", "",
                  "| 赛道 | 月份 | 父 MdAPE | 候选 MdAPE | 相对改善 | 冷启动评估数 | 覆盖率 |",
                  "|---|---|---:|---:|---:|---:|---:|"])
    for track, score in result["tracks"].items():
        for month, monthly in score["monthly"].items():
            coverage = result["route_coverage"][track][month]
            lines.append(
                f"| {track} | {month} | {monthly['parent']['mdape_percent']:.5f}% | "
                f"{monthly['candidate']['mdape_percent']:.5f}% | "
                f"{monthly['relative_mdape_improvement_percent']:+.3f}% | "
                f"{coverage['cold_rows']}/{coverage['evaluation_rows']} | {coverage['coverage_percent']:.2f}% |"
            )
    lines.extend([
        "", "## 实验边界", "",
        f"真实 learner fit starts：{result['fit_starts']}；保留折和隐藏标签未读取。",
        "双月晋级门槛：pooled MdAPE 相对改善至少 0.25%、两个筛选月都改善、MAPE 退化不超过 1%。",
        "筛选结果不自动改变父模型；只有过筛赛道才进入八个月开发确认。", "",
        "- 逐房配对预测：`paired_predictions.csv`",
        "- 组件训练行数、模型参数和特征：`component_manifest.json`",
        "- 数据、父模型和源码哈希：`run_manifest.json`",
        "- Campaign 事件账本：`../../fits.jsonl`",
    ])
    return "\n".join(lines) + "\n"


def run_r14_screen() -> dict:
    from data import hide_answers
    from experiments.bill_greedy import runner as greedy
    from experiments.model_optimization_v2.folds import build_month_fold

    source = _source_identity()
    started = datetime.now(timezone.utc)
    run_id = "bill_target74_r14_cold_start_" + started.strftime("%Y%m%dT%H%M%SZ")
    out = campaign.CAMPAIGN_ROOT / "batches" / run_id
    out.mkdir(parents=True, exist_ok=False)
    campaign._append({"event": "card_start", "card": CARD, "run_id": run_id,
                      "screen_months": list(SCREEN_MONTHS), "parents": PARENTS,
                      "expected_fit_starts": 6,
                      "training_route": "valid coordinates and b_n == 0 at each training row sale-month start",
                      "evaluation_route": "valid coordinates and b_n == 0 at each screen-month start",
                      "reserved_months_labels_opened": False, "hidden_labels_opened": False})

    old_fit_record = greedy._fit_record
    old_hgbr = greedy._frozen_hgbr_predictions
    context: dict[str, str] = {}
    fit_details: list[dict] = []
    components: list[dict] = []
    rows_by_track: dict[str, list[pd.DataFrame]] = {track: [] for track in PARENTS}
    route_coverage: dict[str, dict] = {track: {} for track in PARENTS}

    def record_fit(parent_fit_id: str, fit_fn):
        fit_id = r14_fit_id(run_id, context["track"], context["month"], context["component"])
        fit_context = dict(context)
        campaign._append({"event": "fit_start", "card": CARD, "run_id": run_id,
                          "fit_id": fit_id, "parent_fit_id": parent_fit_id,
                          "data_sha256": source["data_sha256"], **fit_context})
        timer = time.perf_counter()
        try:
            fitted = fit_fn()
        except Exception as exc:
            campaign._append({"event": "fit_complete", "card": CARD, "run_id": run_id,
                              "fit_id": fit_id, "status": "failed",
                              "seconds": time.perf_counter() - timer,
                              "error": f"{type(exc).__name__}: {exc}"})
            raise
        detail = {"fit_id": fit_id, "parent_fit_id": parent_fit_id, **fit_context,
                  "seconds": time.perf_counter() - timer, "status": "complete",
                  "feature_columns": fitted["feature_columns"],
                  "model_family": fitted["model_family"], "model_config": fitted["model_config"],
                  "target": fitted["target"], "training_rows": fitted["training_rows"],
                  "evaluation_rows": fitted["evaluation_rows"],
                  "prediction_sha256": fitted["prediction_sha256"]}
        fit_details.append(detail)
        campaign._append({"event": "fit_complete", "card": CARD, "run_id": run_id, **detail})
        return fitted

    greedy._fit_record = record_fit
    greedy._frozen_hgbr_predictions = lambda month, track, train, homes: report_helpers._verified_hgbr_cache(
        greedy, month, track, train, homes)
    try:
        for month in SCREEN_MONTHS:
            listings, _, _, _, _ = greedy._load_fold(month)
            cutoff = pd.Period(month, freq="M").start_time
            if pd.to_datetime(listings.date_sold, errors="raise").max() >= cutoff + pd.offsets.MonthBegin(1):
                raise RuntimeError(f"fold input includes sales at or after next-month cutoff: {month}")
            for track, parent_id in PARENTS.items():
                train, homes, truth = build_month_fold(listings, month, track)
                heldout = listings.loc[pd.to_datetime(listings.date_sold).dt.to_period("M").astype(str).eq(month)].copy()
                heldout["ml_num"] = heldout.ml_num.astype(str)
                heldout = heldout.set_index("ml_num").loc[homes.ml_num.astype(str)].reset_index()
                homes = hide_answers(heldout, track)
                if not np.array_equal(homes.ml_num.astype(str).to_numpy(), truth.ml_num.astype(str).to_numpy()):
                    raise RuntimeError(f"heldout rows and labels differ in {month}/{track}")

                parent, parent_path = budget._check_parent_artifact(
                    greedy, month, track, homes, truth, parent_id)
                train_mask = training_cold_start_mask(train, track)
                eval_mask = evaluation_cold_start_mask(train, homes, track=track, cutoff=cutoff)
                cold_train = train.loc[train_mask].copy()
                cold_homes = homes.loc[eval_mask].copy()
                if cold_train.empty or cold_homes.empty:
                    raise RuntimeError(f"cold-start route is empty for {month}/{track}")
                route_coverage[track][month] = {
                    "training_rows": int(len(train)), "cold_training_rows": int(len(cold_train)),
                    "evaluation_rows": int(len(homes)), "cold_rows": int(len(cold_homes)),
                    "coverage_percent": float(100.0 * len(cold_homes) / len(homes)),
                }

                if track == "off_market":
                    context.clear(); context.update({"track": track, "month": month, "component": "base"})
                    base = greedy._fit_history_candidate(
                        month, track, train, cold_homes, base_variant="G06_lot_garage",
                        fit_tag=f"R14-cold-base-{month}", fit_rows=cold_train)
                    components.append({"track": track, "month": month, "component": "base",
                                       "parent_id": parent_id, "training_rows": base["training_rows"],
                                       "model_family": base["model_family"], "model_config": base["model_config"],
                                       "target": base["target"], "feature_columns": base["feature_columns"]})

                    cold_detached = cold_train.loc[greedy._specialist_mask(cold_train, "Detached")].copy()
                    detached_eval_mask = greedy._specialist_mask(cold_homes, "Detached")
                    detached_homes = cold_homes.loc[detached_eval_mask].copy()
                    if cold_detached.empty or detached_homes.empty:
                        raise RuntimeError(f"cold Detached expert has no data in {month}")
                    context.clear(); context.update({"track": track, "month": month, "component": "expert"})
                    expert = greedy._fit_history_candidate(
                        month, track, train, detached_homes, base_variant="G06_lot_garage",
                        fit_tag=f"R14-cold-expert-{month}", fit_rows=cold_detached)
                    components.append({"track": track, "month": month, "component": "expert",
                                       "parent_id": parent_id, "training_rows": expert["training_rows"],
                                       "model_family": expert["model_family"], "model_config": expert["model_config"],
                                       "target": expert["target"], "feature_columns": expert["feature_columns"]})
                    cold_prediction = off_market_cold_predictions(
                        cold_homes, base_predictions=base["predictions"],
                        detached_expert_predictions=expert["predictions"])
                else:
                    context.clear(); context.update({"track": track, "month": month, "component": "core"})
                    core = greedy._fit_history_candidate(
                        month, track, train, cold_homes, fit_tag=f"R14-cold-core-{month}",
                        fit_rows=cold_train)
                    components.append({"track": track, "month": month, "component": "core",
                                       "parent_id": parent_id, "training_rows": core["training_rows"],
                                       "model_family": core["model_family"], "model_config": core["model_config"],
                                       "target": core["target"], "feature_columns": core["feature_columns"]})
                    frozen_hgbr = report_helpers._verified_hgbr_cache(greedy, month, track, train, homes)
                    cold_prediction = list_aware_cold_predictions(
                        core_predictions=core["predictions"],
                        frozen_hgbr_predictions=frozen_hgbr[eval_mask])

                candidate_prediction = replace_routed_predictions(
                    parent.predicted_price.to_numpy(float), cold_prediction, eval_mask)
                candidate = pd.DataFrame({"ml_num": homes.ml_num.astype(str).to_numpy(),
                                          "sold_price": truth.sold_price.to_numpy(float),
                                          "predicted_price": candidate_prediction})
                rows_by_track[track].append(report_helpers._candidate_frame(
                    month, track, truth, parent, candidate))
                campaign._append({"event": "screen_fold_complete", "card": CARD, "run_id": run_id,
                                  "track": track, "month": month, "parent_id": parent_id,
                                  "parent_csv_sha256": report_helpers._sha256(parent_path),
                                  "cold_training_rows": int(len(cold_train)),
                                  "cold_evaluation_rows": int(len(cold_homes)),
                                  "date_cutoff_exclusive": cutoff.isoformat()})
    except Exception as exc:
        campaign._append({"event": "card_failed", "card": CARD, "run_id": run_id,
                          "error": f"{type(exc).__name__}: {exc}", "fit_starts": len(fit_details)})
        campaign._write_progress("r14_failed", next_action="inspect R14 failure and saved fits before retry",
                                 latest_error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        greedy._fit_record = old_fit_record
        greedy._frozen_hgbr_predictions = old_hgbr

    paired_all = pd.concat([frame for frames in rows_by_track.values() for frame in frames], ignore_index=True)
    paired_path = out / "paired_predictions.csv"
    paired_all.to_csv(paired_path, index=False)
    tracks = {track: report_helpers._track_result(pd.concat(frames, ignore_index=True))
              for track, frames in rows_by_track.items()}
    result = {"campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
              "card": CARD, "screen_months": list(SCREEN_MONTHS), "parents": PARENTS,
              "tracks": tracks, "route_coverage": route_coverage,
              "fit_starts": len(fit_details), "reserved_months_labels_opened": False,
              "hidden_labels_opened": False, "data_sha256": source["data_sha256"]}
    results_path = out / "r14_screen_results.json"
    results_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    (out / "component_manifest.json").write_text(json.dumps(components, indent=2, default=str), encoding="utf-8")
    (out / "run_manifest.json").write_text(json.dumps({
        "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
        "source_identity": source, "parents": PARENTS,
        "code_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): report_helpers._sha256(path)
                        for path in [campaign.PROTOCOL_PATH, campaign.PACKAGE / "r14_cold_start.py",
                                     campaign.PACKAGE / "r04_ablation.py", campaign.PACKAGE / "r03_catboost.py",
                                     ROOT / "experiments/bill_greedy/runner.py",
                                     ROOT / "experiments/bill_greedy/history.py",
                                     ROOT / "experiments/model_optimization_v2/learners.py",
                                     ROOT / "model_candidate_v2/features.py",
                                     ROOT / "model_candidate_v2/winner_config.json"]},
        "parent_prediction_hashes": {
            f"{track}/{month}": report_helpers._sha256(campaign._row_prediction_path(month, track))
            for track in PARENTS for month in SCREEN_MONTHS},
        "paired_predictions_sha256": report_helpers._sha256(paired_path),
        "results_sha256": report_helpers._sha256(results_path),
        "runtime": {"python": platform.python_version(), "lightgbm": metadata.version("lightgbm"),
                    "numpy": np.__version__, "pandas": pd.__version__},
        "fit_details": fit_details,
    }, indent=2), encoding="utf-8")
    (out / "research_report.md").write_text(_report(result), encoding="utf-8")
    passed = {track: score["qualifies_screen"] for track, score in tracks.items()}
    campaign._append({"event": "card_complete", "card": CARD, "run_id": run_id,
                      "fit_starts": len(fit_details), "result_path": str(results_path.relative_to(ROOT)),
                      "qualifies_by_track": passed,
                      "reserved_months_labels_opened": False, "hidden_labels_opened": False})
    campaign._write_progress("r14_cold_start_screen_complete",
                             next_action="confirm only R14 tracks that pass the frozen development gate",
                             r14_result=str(results_path.relative_to(ROOT)))
    return result
