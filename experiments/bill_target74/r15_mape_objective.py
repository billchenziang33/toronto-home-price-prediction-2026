from __future__ import annotations

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
TRACK = "list_aware"
PARENT = campaign.PROTOCOL["parent_by_track"][TRACK]
SCREEN_MONTHS = tuple(campaign.PROTOCOL["screen_months"])
CARD = "R15_list_aware_direct_dollar_mape"
VARIANT = "R15_list_aware_direct_dollar_mape"


def mape_variant(parent_config: dict) -> dict:
    """Copy the frozen G01 learner config and change only its loss objective."""
    if parent_config.get("objective") != "regression":
        raise ValueError("R15 requires the frozen G01 regression objective as its parent")
    config = dict(parent_config)
    config["objective"] = "mape"
    return config


def r15_fit_id(run_id: str, month: str) -> str:
    if not run_id or any(char in run_id for char in "\\/"):
        raise ValueError("run_id must be path-safe")
    if month not in SCREEN_MONTHS:
        raise ValueError(f"month is not a registered screen fold: {month!r}")
    return f"{campaign.PROTOCOL['campaign_id']}:{CARD}:{TRACK}:{month}:core:{run_id}"


def _report(result: dict) -> str:
    score = result["tracks"][TRACK]
    lines = [
        "# R15：List-aware 直接美元 MAPE 目标复测", "",
        f"Campaign：`{result['campaign_id']}`；run：`{result['run_id']}`。",
        "候选只改变 G01 LightGBM core 的 objective 为 `mape`，训练目标使用美元成交价；特征、LightGBM 参数（除 objective）、HGBR 25% 混合、路由与预测取整保持不变。",
        "筛选仅使用 2023-03 与 2024-06；off-market 未运行此卡。", "",
        "## 双月总体结果", "",
        "| 赛道 | 父 MdAPE | 候选 MdAPE | 相对改善 | MAPE 变化 | 通过筛选 |",
        "|---|---:|---:|---:|---:|---|",
        f"| {TRACK} | {score['parent']['mdape_percent']:.5f}% | "
        f"{score['candidate']['mdape_percent']:.5f}% | "
        f"{score['relative_mdape_improvement_percent']:+.3f}% | "
        f"{score['mape_relative_change_percent']:+.3f}% | "
        f"{'是' if score['qualifies_screen'] else '否'} |",
        "", "## 逐月结果", "",
        "| 月份 | 父 MdAPE | 候选 MdAPE | 相对改善 |", "|---|---:|---:|---:|",
    ]
    for month, monthly in score["monthly"].items():
        lines.append(f"| {month} | {monthly['parent']['mdape_percent']:.5f}% | "
                     f"{monthly['candidate']['mdape_percent']:.5f}% | "
                     f"{monthly['relative_mdape_improvement_percent']:+.3f}% |")
    lines.extend([
        "", "## 实验边界", "",
        f"真实 learner fit starts：{result['fit_starts']}；保留折和隐藏标签未读取。",
        "筛选门槛：pooled MdAPE 相对改善至少 0.25%、两个月都改善、MAPE 退化不超过 1%。不通过则淘汰此 objective，不改父模型。", "",
        "- 逐房配对预测：`paired_predictions.csv`",
        "- 参数、实际 objective、训练目标和哈希：`run_manifest.json`",
        "- Campaign 事件账本：`../../fits.jsonl`",
    ])
    return "\n".join(lines) + "\n"


def run_r15_screen() -> dict:
    from data import hide_answers
    from experiments.bill_greedy import runner as greedy
    from experiments.model_optimization_v2.folds import build_month_fold
    from experiments.model_optimization_v2.pipeline import blend_prices

    source = budget._source_identity()
    winner = json.loads((ROOT / "model_candidate_v2/winner_config.json").read_text(encoding="utf-8"))
    base_config = dict(winner["locked"][TRACK]["model_config"])
    variant_config = mape_variant(base_config)
    started = datetime.now(timezone.utc)
    run_id = "bill_target74_r15_mape_" + started.strftime("%Y%m%dT%H%M%SZ")
    out = campaign.CAMPAIGN_ROOT / "batches" / run_id
    out.mkdir(parents=True, exist_ok=False)
    campaign._append({"event": "card_start", "card": CARD, "run_id": run_id,
                      "screen_months": list(SCREEN_MONTHS), "parent": PARENT,
                      "expected_fit_starts": 2, "variant_config": variant_config,
                      "target": "dollar_price", "reserved_months_labels_opened": False,
                      "hidden_labels_opened": False})

    old_variant = greedy.VARIANTS.get(VARIANT)
    old_fit_record = greedy._fit_record
    old_hgbr = greedy._frozen_hgbr_predictions
    greedy.VARIANTS[VARIANT] = {"target_method": "M13",
                                "model_config_overrides": {TRACK: variant_config}}
    context: dict[str, str] = {}
    fit_details: list[dict] = []
    paired_frames = []

    def record_fit(parent_fit_id: str, fit_fn):
        month = context["month"]
        fit_id = r15_fit_id(run_id, month)
        campaign._append({"event": "fit_start", "card": CARD, "run_id": run_id,
                          "fit_id": fit_id, "parent_fit_id": parent_fit_id,
                          "track": TRACK, "month": month, "component": "core",
                          "model_config": variant_config, "data_sha256": source["data_sha256"]})
        timer = time.perf_counter()
        try:
            fitted = fit_fn()
        except Exception as exc:
            campaign._append({"event": "fit_complete", "card": CARD, "run_id": run_id,
                              "fit_id": fit_id, "status": "failed",
                              "seconds": time.perf_counter() - timer,
                              "error": f"{type(exc).__name__}: {exc}"})
            raise
        detail = {"fit_id": fit_id, "parent_fit_id": parent_fit_id, "track": TRACK,
                  "month": month, "component": "core", "seconds": time.perf_counter() - timer,
                  "status": "complete", "model_family": fitted["model_family"],
                  "model_config": fitted["model_config"], "actual_objective": fitted["actual_objective"],
                  "target": fitted["target"], "training_rows": fitted["training_rows"],
                  "feature_columns": fitted["feature_columns"],
                  "prediction_sha256": fitted["prediction_sha256"]}
        fit_details.append(detail)
        campaign._append({"event": "fit_complete", "card": CARD, "run_id": run_id, **detail})
        return fitted

    greedy._fit_record = record_fit
    greedy._frozen_hgbr_predictions = lambda month, track, train, homes: helpers._verified_hgbr_cache(
        greedy, month, track, train, homes)
    try:
        for month in SCREEN_MONTHS:
            listings, *_ = greedy._load_fold(month)
            train, fold_homes, truth = build_month_fold(listings, month, TRACK)
            heldout = listings.loc[pd.to_datetime(listings.date_sold).dt.to_period("M").astype(str).eq(month)].copy()
            heldout["ml_num"] = heldout.ml_num.astype(str)
            heldout = heldout.set_index("ml_num").loc[fold_homes.ml_num.astype(str)].reset_index()
            homes = hide_answers(heldout, TRACK)
            parent, parent_path = budget._check_parent_artifact(greedy, month, TRACK, homes, truth, PARENT)

            context.clear(); context.update({"month": month})
            core = greedy._fit_history_candidate(
                month, TRACK, train, homes, variant=VARIANT, fit_tag=f"R15-mape-{month}")
            frozen_hgbr = helpers._verified_hgbr_cache(greedy, month, TRACK, train, homes)
            candidate_prediction = blend_prices(core["predictions"], frozen_hgbr, challenger_weight=0.75)
            candidate = pd.DataFrame({"ml_num": homes.ml_num.astype(str).to_numpy(),
                                      "sold_price": truth.sold_price.to_numpy(float),
                                      "predicted_price": candidate_prediction})
            paired_frames.append(helpers._candidate_frame(month, TRACK, truth, parent, candidate))
            campaign._append({"event": "screen_fold_complete", "card": CARD, "run_id": run_id,
                              "track": TRACK, "month": month, "parent_id": PARENT,
                              "parent_csv_sha256": helpers._sha256(parent_path),
                              "target": core["target"], "actual_objective": core["actual_objective"]})
    except Exception as exc:
        campaign._append({"event": "card_failed", "card": CARD, "run_id": run_id,
                          "error": f"{type(exc).__name__}: {exc}", "fit_starts": len(fit_details)})
        campaign._write_progress("r15_failed", next_action="inspect R15 fit and failure records before retry",
                                 latest_error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        greedy._fit_record = old_fit_record
        greedy._frozen_hgbr_predictions = old_hgbr
        if old_variant is None:
            greedy.VARIANTS.pop(VARIANT, None)
        else:
            greedy.VARIANTS[VARIANT] = old_variant

    paired = pd.concat(paired_frames, ignore_index=True)
    paired_path = out / "paired_predictions.csv"
    paired.to_csv(paired_path, index=False)
    tracks = {TRACK: helpers._track_result(paired)}
    result = {"campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
              "card": CARD, "parents": {TRACK: PARENT}, "screen_months": list(SCREEN_MONTHS),
              "tracks": tracks, "fit_starts": len(fit_details), "target": "dollar_price",
              "model_config": variant_config, "reserved_months_labels_opened": False,
              "hidden_labels_opened": False, "data_sha256": source["data_sha256"]}
    result_path = out / "r15_screen_results.json"
    result_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    manifest = {
        "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
        "source_identity": source, "parents": {TRACK: PARENT},
        "model_config": variant_config, "target": "dollar_price", "fit_details": fit_details,
        "code_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): helpers._sha256(path)
                        for path in [campaign.PROTOCOL_PATH, campaign.PACKAGE / "r15_mape_objective.py",
                                     campaign.PACKAGE / "r03_catboost.py", campaign.PACKAGE / "r04_ablation.py",
                                     ROOT / "experiments/bill_greedy/runner.py",
                                     ROOT / "experiments/bill_greedy/history.py",
                                     ROOT / "experiments/model_optimization_v2/learners.py",
                                     ROOT / "model_candidate_v2/features.py",
                                     ROOT / "model_candidate_v2/winner_config.json"]},
        "parent_prediction_hashes": {month: helpers._sha256(campaign._row_prediction_path(month, TRACK))
                                     for month in SCREEN_MONTHS},
        "paired_predictions_sha256": helpers._sha256(paired_path),
        "result_sha256": helpers._sha256(result_path),
        "runtime": {"python": platform.python_version(), "lightgbm": metadata.version("lightgbm"),
                    "numpy": np.__version__, "pandas": pd.__version__},
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out / "research_report.md").write_text(_report(result), encoding="utf-8")
    passed = tracks[TRACK]["qualifies_screen"]
    campaign._append({"event": "card_complete", "card": CARD, "run_id": run_id,
                      "fit_starts": len(fit_details), "result_path": str(result_path.relative_to(ROOT)),
                      "qualifies_by_track": {TRACK: passed},
                      "reserved_months_labels_opened": False, "hidden_labels_opened": False})
    campaign._write_progress("r15_mape_screen_complete",
                             next_action=("confirm list-aware R15 under eight-month development gates" if passed
                                          else "R15 did not qualify; register the next distinct evidence-based card"),
                             r15_result=str(result_path.relative_to(ROOT)))
    return result
