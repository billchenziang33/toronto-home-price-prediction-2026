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
from experiments.model_optimization_v2.pipeline import blend_prices

ROOT = campaign.ROOT
SCREEN_MONTHS = tuple(campaign.PROTOCOL["screen_months"])
PARENTS = dict(campaign.PROTOCOL["parent_by_track"])
VARIANT = "BILL_R11_unit_to_building_fallback"
CARD = "R11_unit_to_building_residual_fallback"
RELIABILITY = {
    "unit_history_reliability_pseudocount": 5.0,
    "unit_history_reliability_half_life_months": 24.0,
    "unit_history_fallback_to_building": True,
}


def reliability_variant_spec() -> dict:
    return dict(RELIABILITY)


def r11_fit_id(run_id: str, track: str, month: str, component: str) -> str:
    if track not in PARENTS:
        raise ValueError(f"unknown track {track!r}")
    if month not in SCREEN_MONTHS:
        raise ValueError(f"month {month!r} is not a registered screen fold")
    expected = {"base", "expert"} if track == "off_market" else {"core"}
    if component not in expected:
        raise ValueError(f"component {component!r} is not registered for {track}")
    return f"{campaign.PROTOCOL['campaign_id']}:R11_unit_to_building:{track}:{month}:{component}:{run_id}"


def _report(result: dict) -> str:
    lines = [
        "# BILL <7% / <4% 持续研究：R11 unit-to-building residual fallback", "",
        f"Campaign：`{result['campaign_id']}`；run：`{result['run_id']}`。",
        "唯一变化：仅将 unit-history 的 `u_res_mean` 与 `u_last_res` 按固定可靠性权重回退到同坐标 building residual；其余模型、训练目标、特征列、父模型路由、冻结 HGBR 混合与价格取整保持不变。",
        "权重 `w = n/(n+5) × 2^(-age/24)`，其中 `n=u_res_n`，`age=u_age`（月）。两类历史均严格 as-of：两者都有值时用 `w×unit + (1-w)×building`，缺一时使用可用的另一层，两者都缺失时保留缺失。参数未使用筛选月调节。",
        "这与历史 G30 的社区×房型售/挂牌比修正不同；本卡的信号是同坐标售出价格残差。",
        "", "## 双月 pooled 结果", "",
        "| 赛道 | 父→候选 MdAPE | 相对改善 | MAPE变化 | 父→候选 P90 APE | 父→候选 APE>20% | 筛选通过 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for track, score in result["tracks"].items():
        lines.append(
            f"| {track} | {score['parent']['mdape_percent']:.5f}% → "
            f"{score['candidate']['mdape_percent']:.5f}% | "
            f"{score['relative_mdape_improvement_percent']:+.3f}% | "
            f"{score['mape_relative_change_percent']:+.3f}% | "
            f"{score['parent']['p90_ape_percent']:.3f}% → {score['candidate']['p90_ape_percent']:.3f}% | "
            f"{score['parent']['ape_over_20_percent']:.3f}% → {score['candidate']['ape_over_20_percent']:.3f}% | "
            f"{'是' if score['qualifies_screen'] else '否'} |"
        )
    lines.extend(["", "## 逐月 MdAPE", "", "| 赛道 | 月份 | 父 MdAPE | 候选 MdAPE | 相对改善 |",
                  "|---|---|---:|---:|---:|"])
    for track, score in result["tracks"].items():
        for month, values in score["monthly"].items():
            lines.append(f"| {track} | {month} | {values['parent']['mdape_percent']:.5f}% | "
                         f"{values['candidate']['mdape_percent']:.5f}% | "
                         f"{values['relative_mdape_improvement_percent']:+.3f}% |")
    lines.extend([
        "", "## 账本与边界", "",
        f"真实 learner fit starts：{result['fit_starts']}；筛选月份：`{', '.join(SCREEN_MONTHS)}`。",
        "只比较通过当前 campaign 冻结父预测的两个月；未通过的轨道不扩展到八个月确认。预留月份与隐藏标签未读取。",
        "历史父预测 sidecar 与当前工作区代码哈希存在已记录差异；父预测仅作冻结比较器。",
        "", "- 逐房配对预测：`paired_predictions.csv`",
        "- 组件与特征配置：`component_manifest.json`",
        "- 运行来源及逐 fit 记录：`run_manifest.json` / campaign `fits.jsonl`", "",
    ])
    return "\n".join(lines)


def run_r11_screen() -> dict:
    from data import hide_answers
    from experiments.bill_greedy import runner as greedy

    source = helpers._source_identity()
    started = datetime.now(timezone.utc)
    run_id = "bill_target74_r11_unit_to_building_" + started.strftime("%Y%m%dT%H%M%SZ")
    out = campaign.CAMPAIGN_ROOT / "batches" / run_id
    out.mkdir(parents=True, exist_ok=False)
    if VARIANT in greedy.VARIANTS:
        raise RuntimeError(f"temporary R11 variant key already exists: {VARIANT}")
    previous_variant = greedy.VARIANTS.get(VARIANT)
    greedy.VARIANTS[VARIANT] = reliability_variant_spec()
    old_fit_record = greedy._fit_record
    old_hgbr = greedy._frozen_hgbr_predictions
    context: dict[str, str] = {}
    fit_details: list[dict] = []
    component_manifest: list[dict] = []
    rows_by_track: dict[str, list[pd.DataFrame]] = {track: [] for track in PARENTS}
    campaign._append({"event": "card_start", "card": CARD, "run_id": run_id,
                      "screen_months": list(SCREEN_MONTHS), "fit_start_count": 0,
                      "variant": VARIANT, "reliability": RELIABILITY,
                      "parent_by_track": PARENTS})

    def record_fit(parent_fit_id: str, fit_fn):
        limits = campaign.PROTOCOL["limits"]
        if campaign.limit_reached(budget._fit_count(), limits["exploration_fit_starts"]):
            raise RuntimeError("R11 exploration fit-start reserve reached")
        if campaign.limit_exceeded(budget._campaign_elapsed_seconds(),
                                   None if limits["exploration_minutes"] is None
                                   else limits["exploration_minutes"] * 60):
            raise RuntimeError("R11 campaign exploration wall-clock reserve reached")
        if campaign.limit_exceeded((datetime.now(timezone.utc) - started).total_seconds(),
                                   None if limits["batch_minutes"] is None
                                   else limits["batch_minutes"] * 60):
            raise RuntimeError("R11 batch wall-clock cap reached before fit start")
        fit_id = r11_fit_id(run_id, context["track"], context["month"], context["component"])
        fit_context = dict(context)
        campaign._append({"event": "fit_start", "card": CARD, "run_id": run_id,
                          "fit_id": fit_id, "parent_fit_id": parent_fit_id, **fit_context,
                          "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                          "data_sha256": source["data_sha256"], "reliability": RELIABILITY})
        tick = time.perf_counter()
        try:
            fitted = fit_fn()
        except Exception as exc:
            campaign._append({"event": "fit_complete", "card": CARD, "run_id": run_id,
                              "fit_id": fit_id, "status": "failed", "seconds": time.perf_counter() - tick,
                              "error": f"{type(exc).__name__}: {exc}"})
            raise
        seconds = time.perf_counter() - tick
        status = "complete" if not campaign.limit_exceeded(seconds, limits["single_fit_seconds"]) else "over_limit"
        detail = {"fit_id": fit_id, "parent_fit_id": parent_fit_id, **fit_context,
                  "seconds": seconds, "status": status,
                  "feature_columns": fitted["feature_columns"], "target": fitted["target"],
                  "actual_objective": fitted["actual_objective"], "model_config": fitted["model_config"],
                  "prediction_sha256": fitted["prediction_sha256"],
                  "training_rows": fitted["training_rows"], "reliability": RELIABILITY}
        fit_details.append(detail)
        campaign._append({"event": "fit_complete", "card": CARD, "run_id": run_id, **detail})
        if status != "complete":
            raise TimeoutError(f"fit exceeded {limits['single_fit_seconds']} seconds")
        return fitted

    greedy._fit_record = record_fit
    greedy._frozen_hgbr_predictions = lambda month, track, train, homes: helpers._verified_hgbr_cache(
        greedy, month, track, train, homes)
    try:
        for month in SCREEN_MONTHS:
            listings, train, _, truth, heldout = greedy._load_fold(month)
            cutoff = (pd.Period(month, freq="M") + 1).start_time
            if pd.to_datetime(listings.date_sold, errors="raise").max() >= cutoff:
                raise RuntimeError(f"fold input includes sales at or after next-month cutoff: {month}")
            for track, parent_id in PARENTS.items():
                homes = hide_answers(heldout, track)
                parent, parent_path = budget._check_parent_artifact(greedy, month, track, homes, truth, parent_id)
                if track == "off_market":
                    context.clear(); context.update({"track": track, "month": month, "component": "base"})
                    base = greedy._fit_history_candidate(
                        month, track, train, homes, variant=VARIANT, base_variant="G06_lot_garage",
                        fit_tag=f"R11hier-base-{month}")
                    component_manifest.append({"track": track, "month": month, "component": "base",
                                              "parent_id": parent_id, "model_family": base["model_family"],
                                              "model_config": base["model_config"], "target": base["target"],
                                              "actual_objective": base["actual_objective"],
                                              "feature_columns": base["feature_columns"],
                                              "training_rows": base["training_rows"], "reliability": RELIABILITY})
                    detached_train = train.loc[greedy._specialist_mask(train, "Detached")].copy()
                    detached_mask = greedy._specialist_mask(homes, "Detached")
                    if detached_train.empty or not detached_mask.any():
                        raise RuntimeError(f"Detached expert has no training/evaluation rows in {month}")
                    context.clear(); context.update({"track": track, "month": month, "component": "expert"})
                    expert = greedy._fit_history_candidate(
                        month, track, train, homes.loc[detached_mask].copy(), variant=VARIANT,
                        base_variant="G06_lot_garage", fit_tag=f"R11hier-expert-{month}",
                        fit_rows=detached_train)
                    component_manifest.append({"track": track, "month": month, "component": "expert",
                                              "parent_id": parent_id, "model_family": expert["model_family"],
                                              "model_config": expert["model_config"], "target": expert["target"],
                                              "actual_objective": expert["actual_objective"],
                                              "feature_columns": expert["feature_columns"],
                                              "training_rows": expert["training_rows"], "reliability": RELIABILITY})
                    candidate_prediction = greedy._blend_specialist_predictions(
                        np.round(base["predictions"]), np.round(expert["predictions"]), homes,
                        "Detached", 0.75)
                else:
                    context.clear(); context.update({"track": track, "month": month, "component": "core"})
                    core = greedy._fit_history_candidate(
                        month, track, train, homes, variant=VARIANT, fit_tag=f"R11hier-core-{month}")
                    component_manifest.append({"track": track, "month": month, "component": "core",
                                              "parent_id": parent_id, "model_family": core["model_family"],
                                              "model_config": core["model_config"], "target": core["target"],
                                              "actual_objective": core["actual_objective"],
                                              "feature_columns": core["feature_columns"],
                                              "training_rows": core["training_rows"], "reliability": RELIABILITY})
                    frozen_hgbr = helpers._verified_hgbr_cache(greedy, month, track, train, homes)
                    candidate_prediction = blend_prices(core["predictions"], frozen_hgbr, challenger_weight=0.75)

                candidate = pd.DataFrame({"ml_num": homes.ml_num.astype(str).to_numpy(),
                                          "sold_price": truth.sold_price.to_numpy(float),
                                          "predicted_price": candidate_prediction})
                paired = helpers._candidate_frame(month, track, truth, parent, candidate)
                rows_by_track[track].append(paired)
                campaign._append({"event": "screen_fold_complete", "card": CARD, "run_id": run_id,
                                  "track": track, "month": month, "parent_id": parent_id,
                                  "parent_csv_sha256": helpers._sha256(parent_path),
                                  "target_method": "C0", "reliability": RELIABILITY,
                                  "date_cutoff_exclusive": cutoff.isoformat()})
    except Exception as exc:
        campaign._append({"event": "card_failed", "card": CARD, "run_id": run_id,
                          "error": f"{type(exc).__name__}: {exc}", "fit_starts": len(fit_details)})
        campaign._write_progress("r11_failed", next_action="inspect R11 failure record before any retry",
                                 latest_error=f"{type(exc).__name__}: {exc}")
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
    result = {"campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id, "card": CARD,
              "variant": VARIANT, "reliability": RELIABILITY, "target_method": "C0",
              "screen_months": list(SCREEN_MONTHS), "parents": PARENTS, "tracks": tracks,
              "fit_starts": len(fit_details), "reserved_months_labels_opened": False,
              "hidden_labels_opened": False, "data_sha256": source["data_sha256"]}
    results_path = out / "r11_screen_results.json"
    results_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    (out / "component_manifest.json").write_text(
        json.dumps(component_manifest, indent=2, default=str), encoding="utf-8")
    source_files = [campaign.PROTOCOL_PATH, campaign.PACKAGE / "r11_unit_to_building.py",
                    campaign.PACKAGE / "r04_ablation.py", campaign.PACKAGE / "r03_catboost.py",
                    ROOT / "experiments/bill_greedy/runner.py", ROOT / "experiments/bill_greedy/history.py",
                    ROOT / "experiments/model_optimization_v2/worker.py",
                    ROOT / "experiments/model_optimization_v2/learners.py",
                    ROOT / "experiments/model_optimization_v2/pipeline.py",
                    ROOT / "model_candidate_v2/features.py", ROOT / "model_candidate_v2/winner_config.json"]
    run_manifest = {"campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
                    "source_identity": source, "reliability": RELIABILITY,
                    "code_sha256": {str(path.relative_to(ROOT)): helpers._sha256(path) for path in source_files},
                    "runtime": {"python": platform.python_version(), "lightgbm": metadata.version("lightgbm"),
                                "numpy": np.__version__, "pandas": pd.__version__,
                                "scikit_learn": metadata.version("scikit-learn")},
                    "parent_artifacts": {
                        f"{track}/{month}": {"path": str(campaign._row_prediction_path(month, track).relative_to(ROOT)),
                                              "csv_sha256": helpers._sha256(campaign._row_prediction_path(month, track)),
                                              "sidecar_sha256": helpers._sha256(campaign._row_prediction_path(month, track).with_suffix(".json"))}
                        for track in PARENTS for month in SCREEN_MONTHS},
                    "frozen_hgbr_artifacts": {
                        month: {"path": str((greedy.OUTPUT / "frozen_hgbr/list_aware" / f"{month}.csv").relative_to(ROOT)),
                                "csv_sha256": helpers._sha256(greedy.OUTPUT / "frozen_hgbr/list_aware" / f"{month}.csv")}
                        for month in SCREEN_MONTHS},
                    "paired_prediction_csv_sha256": helpers._sha256(paired_path),
                    "results_sha256": helpers._sha256(results_path), "fit_starts": len(fit_details),
                    "fit_details": fit_details}
    (out / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    (out / "research_report.md").write_text(_report(result), encoding="utf-8")
    passed = {track: score["qualifies_screen"] for track, score in tracks.items()}
    if any(passed.values()):
        next_action = "run eight-month development confirmation for: " + ", ".join(
            track for track, qualified in passed.items() if qualified)
    else:
        next_action = "R11 unit-to-building fallback did not qualify; register the next distinct evidence-based method"
    campaign._append({"event": "card_complete", "card": CARD, "run_id": run_id,
                      "fit_starts": len(fit_details), "qualifies_by_track": passed,
                      "result_path": str(results_path.relative_to(ROOT)).replace("\\", "/"),
                      "paired_prediction_csv_sha256": run_manifest["paired_prediction_csv_sha256"]})
    campaign._write_progress("r11_screen_complete", next_action=next_action,
                             r11_result=str(results_path.relative_to(ROOT)).replace("\\", "/"))
    return result


def main() -> None:
    result = run_r11_screen()
    for track, score in result["tracks"].items():
        print(f"{track}: parent={score['parent']['mdape_percent']:.5f}% "
              f"candidate={score['candidate']['mdape_percent']:.5f}% "
              f"relative_gain={score['relative_mdape_improvement_percent']:+.3f}% "
              f"MAPE_change={score['mape_relative_change_percent']:+.3f}% "
              f"screen_pass={score['qualifies_screen']}", flush=True)
    print(f"report={campaign.CAMPAIGN_ROOT / 'batches' / result['run_id'] / 'research_report.md'}", flush=True)


if __name__ == "__main__":
    main()
