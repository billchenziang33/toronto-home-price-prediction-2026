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
from experiments.model_optimization_v2.pipeline import blend_prices

ROOT = campaign.ROOT
SCREEN_MONTHS = tuple(campaign.PROTOCOL["screen_months"])
PARENTS = dict(campaign.PROTOCOL["parent_by_track"])
FEATURE_VARIANT = "G17_nearest_comparables"
CARD = "R07_current_parent_comparable_features"


def r07_fit_id(run_id: str, track: str, month: str, component: str) -> str:
    if track not in {"off_market", "list_aware"}:
        raise ValueError(f"unknown track {track!r}")
    if month not in SCREEN_MONTHS:
        raise ValueError(f"month {month!r} is not a registered screen fold")
    if component not in {"base", "expert", "core"}:
        raise ValueError(f"unknown component {component!r}")
    return (f"{campaign.PROTOCOL['campaign_id']}:R07:{FEATURE_VARIANT}:"
            f"{track}:{month}:{component}:{run_id}")


def list_aware_comparable_prediction(core_prediction, frozen_hgbr_prediction) -> np.ndarray:
    core = np.asarray(core_prediction, dtype=float)
    frozen = np.asarray(frozen_hgbr_prediction, dtype=float)
    if core.shape != frozen.shape or core.ndim != 1:
        raise ValueError("comparable core and frozen HGBR predictions must be aligned vectors")
    if (not np.isfinite(core).all() or not np.isfinite(frozen).all()
            or (core <= 0.0).any() or (frozen <= 0.0).any()):
        raise ValueError("comparable core and frozen HGBR predictions must be finite positive prices")
    return blend_prices(core, frozen, challenger_weight=0.75)


def _report(result: dict) -> str:
    lines = [
        "# BILL <7% / <4% 持续研究：R07 当前父模型结构下的可比房源特征筛选", "",
        f"Campaign：`{result['campaign_id']}`；run：`{result['run_id']}`。",
        f"唯一新增机制：strict-as-of 同房型、邻近成交残差特征（`{FEATURE_VARIANT}`）。",
        "Off-market 在当前 G19 结构的 G06 base 与 Detached expert 上都加入可比房源特征，再按固定 75/25 specialist/base 组合；list-aware 在 LightGBM core 加特征，并保留 G01 固定的 75% LightGBM + 25% HGBR 组合。",
        "较早 G17 结果使用不同 off-market 父结构，list-aware 结果未按本卡保留同一 HGBR 混合，因此本卡是结构匹配的重测；旧 G17 结果仍作为历史负向证据，不覆盖。",
        "", "## 双月合并结果", "",
        "| 赛道 | 父 MdAPE | 候选 MdAPE | 相对改善 | MAPE变化 | 筛选通过 |",
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
    lines.extend(["", "## 逐月结果", "", "| 赛道 | 月份 | 父 MdAPE | 候选 MdAPE | 相对改善 |",
                  "|---|---|---:|---:|---:|"])
    for track, score in result["tracks"].items():
        for month, monthly in score["monthly"].items():
            lines.append(f"| {track} | {month} | {monthly['parent']['mdape_percent']:.5f}% | "
                         f"{monthly['candidate']['mdape_percent']:.5f}% | "
                         f"{monthly['relative_mdape_improvement_percent']:+.3f}% |")
    lines.extend(["", "## 边界与判定", "",
                  f"真实 learner fit starts：{result['fit_starts']}；筛选月份：{', '.join(SCREEN_MONTHS)}。",
                  "每个评估折仅使用截止时点之前的训练成交；候选未改变父版，未通过的赛道不扩展到八个月确认。预留月和隐藏标签未读取。",
                  "历史父预测 sidecar 代码哈希差异仍需在比较中披露。",
                  "", "- 逐房配对预测：`paired_predictions.csv`",
                  "- 组件特征/参数：`component_manifest.json`",
                  "- 代码、数据、父预测与 fit 记录：`run_manifest.json` / campaign `fits.jsonl`", ""])
    return "\n".join(lines)


def run_r07_comparable_screen() -> dict:
    from data import hide_answers
    from experiments.bill_greedy import runner as greedy
    from experiments.model_optimization_v2 import worker

    source = budget._source_identity()
    started = datetime.now(timezone.utc)
    run_id = "bill_target74_r07_comparable_" + started.strftime("%Y%m%dT%H%M%SZ")
    out = campaign.CAMPAIGN_ROOT / "batches" / run_id
    out.mkdir(parents=True, exist_ok=False)
    campaign._append({"event": "card_start", "card": CARD, "run_id": run_id,
                      "screen_months": list(SCREEN_MONTHS), "feature_variant": FEATURE_VARIANT,
                      "fit_start_count": 0, "parent_by_track": PARENTS,
                      "note": "R07 uses current parents and preserves their routing/HGBR blend"})

    old_fit_record = greedy._fit_record
    old_hgbr = greedy._frozen_hgbr_predictions
    context: dict[str, str] = {}
    fit_details: list[dict] = []
    component_manifest: list[dict] = []
    rows_by_track: dict[str, list[pd.DataFrame]] = {track: [] for track in PARENTS}
    try:
        def record_fit(parent_fit_id: str, fit_fn):
            limits = campaign.PROTOCOL["limits"]
            if campaign.limit_reached(budget._fit_count(), limits["exploration_fit_starts"]):
                raise RuntimeError("R07 exploration fit-start reserve reached")
            if campaign.limit_exceeded(budget._campaign_elapsed_seconds(),
                                       None if limits["exploration_minutes"] is None
                                       else limits["exploration_minutes"] * 60):
                raise RuntimeError("R07 campaign exploration wall-clock reserve reached")
            if campaign.limit_exceeded((datetime.now(timezone.utc) - started).total_seconds(),
                                       None if limits["batch_minutes"] is None
                                       else limits["batch_minutes"] * 60):
                raise RuntimeError("R07 batch wall-clock cap reached before fit start")
            fit_id = r07_fit_id(run_id, context["track"], context["month"], context["component"])
            card_context = dict(context)
            campaign._append({"event": "fit_start", "card": CARD, "run_id": run_id,
                              "fit_id": fit_id, "parent_fit_id": parent_fit_id, **card_context,
                              "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                              "data_sha256": source["data_sha256"]})
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
            detail = {"fit_id": fit_id, "parent_fit_id": parent_fit_id, **card_context,
                      "seconds": seconds, "status": status,
                      "feature_columns": fitted["feature_columns"],
                      "prediction_sha256": fitted["prediction_sha256"],
                      "training_rows": fitted["training_rows"]}
            fit_details.append(detail)
            campaign._append({"event": "fit_complete", "card": CARD, "run_id": run_id, **detail})
            if status != "complete":
                raise TimeoutError(f"fit exceeded {limits['single_fit_seconds']} seconds")
            return fitted

        greedy._fit_record = record_fit
        greedy._frozen_hgbr_predictions = lambda month, track, train, homes: helpers._verified_hgbr_cache(
            greedy, month, track, train, homes)
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
                        month, track, train, homes, base_variant="G06_lot_garage",
                        variant=FEATURE_VARIANT, fit_tag=f"R07comp-base-{month}")
                    component_manifest.append({"track": track, "month": month, "component": "base",
                                              "parent_id": parent_id, "model_family": base["model_family"],
                                              "model_config": base["model_config"], "target": base["target"],
                                              "feature_columns": base["feature_columns"],
                                              "training_rows": base["training_rows"]})
                    base_prediction = np.round(base["predictions"])
                    detached_train = train.loc[greedy._specialist_mask(train, "Detached")].copy()
                    detached_mask = greedy._specialist_mask(homes, "Detached")
                    if detached_train.empty or not detached_mask.any():
                        raise RuntimeError(f"Detached expert has no training/evaluation rows in {month}")
                    context.clear(); context.update({"track": track, "month": month, "component": "expert"})
                    expert = greedy._fit_history_candidate(
                        month, track, train, homes.loc[detached_mask].copy(),
                        base_variant="G06_lot_garage", variant=FEATURE_VARIANT,
                        fit_tag=f"R07comp-expert-{month}", fit_rows=detached_train)
                    component_manifest.append({"track": track, "month": month, "component": "expert",
                                              "parent_id": parent_id, "model_family": expert["model_family"],
                                              "model_config": expert["model_config"], "target": expert["target"],
                                              "feature_columns": expert["feature_columns"],
                                              "training_rows": expert["training_rows"]})
                    candidate_prediction = greedy._blend_specialist_predictions(
                        base_prediction, np.round(expert["predictions"]), homes, "Detached", 0.75)
                else:
                    context.clear(); context.update({"track": track, "month": month, "component": "core"})
                    core = greedy._fit_history_candidate(
                        month, track, train, homes, variant=FEATURE_VARIANT,
                        fit_tag=f"R07comp-core-{month}")
                    component_manifest.append({"track": track, "month": month, "component": "core",
                                              "parent_id": parent_id, "model_family": core["model_family"],
                                              "model_config": core["model_config"], "target": core["target"],
                                              "feature_columns": core["feature_columns"],
                                              "training_rows": core["training_rows"]})
                    frozen = helpers._verified_hgbr_cache(greedy, month, track, train, homes)
                    candidate_prediction = list_aware_comparable_prediction(core["predictions"], frozen)

                candidate = pd.DataFrame({"ml_num": homes.ml_num.astype(str).to_numpy(),
                                          "sold_price": truth.sold_price.to_numpy(float),
                                          "predicted_price": candidate_prediction})
                paired = helpers._candidate_frame(month, track, truth, parent, candidate)
                rows_by_track[track].append(paired)
                campaign._append({"event": "screen_fold_complete", "card": CARD, "run_id": run_id,
                                  "track": track, "month": month, "parent_id": parent_id,
                                  "parent_csv_sha256": helpers._sha256(parent_path),
                                  "feature_variant": FEATURE_VARIANT,
                                  "date_cutoff_exclusive": cutoff.isoformat()})
    except Exception as exc:
        campaign._append({"event": "card_failed", "card": CARD, "run_id": run_id,
                          "error": f"{type(exc).__name__}: {exc}",
                          "fit_starts": len(fit_details)})
        campaign._write_progress("r07_failed", next_action="inspect the R07 failure record before any retry",
                                 latest_error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        greedy._fit_record = old_fit_record
        greedy._frozen_hgbr_predictions = old_hgbr

    paired_all = pd.concat([frame for frames in rows_by_track.values() for frame in frames], ignore_index=True)
    paired_path = out / "paired_predictions.csv"
    paired_all.to_csv(paired_path, index=False)
    tracks = {track: helpers._track_result(pd.concat(frames, ignore_index=True))
              for track, frames in rows_by_track.items()}
    result = {"campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id, "card": CARD,
              "feature_variant": FEATURE_VARIANT, "screen_months": list(SCREEN_MONTHS),
              "parents": PARENTS, "tracks": tracks, "fit_starts": len(fit_details),
              "reserved_months_labels_opened": False, "hidden_labels_opened": False,
              "data_sha256": source["data_sha256"]}
    results_path = out / "r07_screen_results.json"
    results_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    (out / "component_manifest.json").write_text(json.dumps(component_manifest, indent=2, default=str), encoding="utf-8")
    source_files = [campaign.PROTOCOL_PATH, campaign.PACKAGE / "r07_comparable_current_parent.py",
                    campaign.PACKAGE / "r04_ablation.py", campaign.PACKAGE / "r03_catboost.py",
                    ROOT / "experiments/bill_greedy/runner.py", ROOT / "experiments/bill_greedy/history.py",
                    ROOT / "experiments/bill_greedy/comparables.py",
                    ROOT / "experiments/model_optimization_v2/worker.py",
                    ROOT / "experiments/model_optimization_v2/learners.py",
                    ROOT / "experiments/model_optimization_v2/pipeline.py",
                    ROOT / "model_candidate_v2/features.py", ROOT / "model_candidate_v2/winner_config.json"]
    manifest = {"campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
                "source_identity": source,
                "code_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): helpers._sha256(path)
                                for path in source_files},
                "runtime": {"python": platform.python_version(), "lightgbm": metadata.version("lightgbm"),
                            "catboost": metadata.version("catboost"), "numpy": np.__version__,
                            "pandas": pd.__version__, "scikit_learn": metadata.version("scikit-learn")},
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
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out / "research_report.md").write_text(_report(result), encoding="utf-8")
    passed = {track: score["qualifies_screen"] for track, score in tracks.items()}
    next_action = ("run eight-month development confirmation for: " + ", ".join(track for track, ok in passed.items() if ok)
                   if any(passed.values()) else
                   "R07 comparable feature screen did not qualify; test the registered relative-error-aligned robust loss")
    campaign._append({"event": "card_complete", "card": CARD, "run_id": run_id,
                      "fit_starts": len(fit_details), "qualifies_by_track": passed,
                      "result_path": str(results_path.relative_to(ROOT)).replace("\\", "/"),
                      "paired_prediction_csv_sha256": manifest["paired_prediction_csv_sha256"]})
    campaign._write_progress("r07_screen_complete", next_action=next_action,
                             r07_result=str(results_path.relative_to(ROOT)).replace("\\", "/"))
    return result


def main() -> None:
    result = run_r07_comparable_screen()
    for track, score in result["tracks"].items():
        print(f"{track}: parent={score['parent']['mdape_percent']:.5f}% "
              f"candidate={score['candidate']['mdape_percent']:.5f}% "
              f"relative_gain={score['relative_mdape_improvement_percent']:+.3f}% "
              f"MAPE_change={score['mape_relative_change_percent']:+.3f}% "
              f"screen_pass={score['qualifies_screen']}", flush=True)
    print(f"report={campaign.CAMPAIGN_ROOT / 'batches' / result['run_id'] / 'research_report.md'}", flush=True)


if __name__ == "__main__":
    main()
