"""R30: test fixed LightGBM quantile loss on top of verified R22 text features."""
from __future__ import annotations

import json
import shutil
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from experiments.bill_target74.r19_expert_weight import _sha256, _write_json


CARD = "R30_r22_quantile_objective"
VARIANT = CARD
SCREEN_MONTHS = ("2023-03", "2024-06")
TRACKS = ("off_market", "list_aware")
VARIANT_SPEC = {
    "model_config_overrides": {
        "off_market": {"objective": "quantile", "alpha": 0.5},
        "list_aware": {"objective": "quantile", "alpha": 0.5},
    }
}


def screen_fit_plan() -> list[tuple[str, str, str]]:
    return [
        (month, track, component)
        for month in SCREEN_MONTHS
        for track, components in (("off_market", ("base", "expert")),
                                  ("list_aware", ("core",)))
        for component in components
    ]


def _fit_id(run_id: str, track: str, month: str, component: str) -> str:
    return f"bill_target74:{CARD}:screen:{track}:{month}:{component}:{run_id}"


def _source_files(campaign) -> list:
    return [
        campaign.PROTOCOL_PATH,
        campaign.PACKAGE / "r30_quantile_r22.py",
        campaign.PACKAGE / "r22_remarks_tfidf.py",
        campaign.PACKAGE / "r23_word_char_tfidf.py",
        campaign.PACKAGE / "r03_catboost.py",
        campaign.PACKAGE / "r04_ablation.py",
        campaign.PACKAGE / "runner.py",
        campaign.ROOT / "model_candidate_v2/features.py",
        campaign.ROOT / "experiments/bill_greedy/runner.py",
        campaign.ROOT / "experiments/bill_greedy/history.py",
        campaign.ROOT / "experiments/model_optimization_v2/learners.py",
    ]


def _report(result: dict) -> str:
    lines = [
        "# R30：R22 词级文本特征 + LightGBM Quantile 损失", "",
        f"Campaign：`{result['campaign_id']}`；run：`{result['run_id']}`。",
        f"比较对象：哈希验证的 R22 run `{result['parent_candidate']}`。",
        "R22 的 256 词项训练折 TF-IDF、结构/历史特征、模型参数、训练目标、路由和融合不变；仅将 LightGBM objective 固定为 quantile、alpha=0.5。",
        "每个 fold 的词表/IDF 只在训练行 remarks 上拟合；候选训练目标仍是 R22 的 log-price / log(sold/list)。", "",
        "## 双月筛选", "",
        "| 赛道 | R22 MdAPE | R30 MdAPE | 相对改善 | MAPE变化 | 两月均改善 | 通过 |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for track, score in result["tracks"].items():
        both = all(score["monthly"][month]["relative_mdape_improvement_percent"] > 0.0
                   for month in SCREEN_MONTHS)
        lines.append(
            f"| {track} | {score['parent']['mdape_percent']:.5f}% | "
            f"{score['candidate']['mdape_percent']:.5f}% | "
            f"{score['relative_mdape_improvement_percent']:+.3f}% | "
            f"{score['mape_relative_change_percent']:+.3f}% | "
            f"{'是' if both else '否'} | {'是' if score['qualifies_screen'] else '否'} |"
        )
    lines.extend(["", "## 各筛选月", "",
                  "| 赛道 | 月份 | R22 MdAPE | R30 MdAPE | 相对改善 |",
                  "|---|---|---:|---:|---:|"])
    for track, score in result["tracks"].items():
        for month, values in score["monthly"].items():
            lines.append(f"| {track} | {month} | {values['parent']['mdape_percent']:.5f}% | "
                         f"{values['candidate']['mdape_percent']:.5f}% | "
                         f"{values['relative_mdape_improvement_percent']:+.3f}% |")
    lines.extend(["", "历史 R08/R16 Q50 结果未使用 R22 文本；本轮只测预先声明的量化目标交互。保留月份与隐藏标签未读取。", ""])
    return "\n".join(lines)


def run_r30_screen() -> dict:
    from data import hide_answers
    from experiments.bill_target74 import r03_catboost as helpers
    from experiments.bill_target74 import r04_ablation as budget
    from experiments.bill_target74 import runner as campaign
    from experiments.bill_target74 import r22_remarks_tfidf as r22
    from experiments.bill_target74 import r23_word_char_tfidf as shared
    from experiments.bill_target74.r03_catboost import _verified_hgbr_cache
    from experiments.bill_greedy import runner as greedy
    from experiments.model_optimization_v2.pipeline import blend_prices

    if campaign.PROTOCOL["cards"].get(CARD, {}).get("status") != "registered":
        raise RuntimeError("R30 must be preregistered before fitting")
    parent_result, parent_path, parent_manifest = r22._verified_screen_result(campaign)
    if not all(parent_result["tracks"][track]["qualifies_screen"] for track in TRACKS):
        raise RuntimeError("R30 requires both tracks to qualify in the verified R22 screen")
    parent_rows = pd.read_csv(parent_path.parent / "paired_predictions.csv", dtype={"ml_num": str})
    source = budget._source_identity()
    run_id = "bill_target74_r30_quantile_r22_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = campaign.CAMPAIGN_ROOT / "batches" / run_id
    output.mkdir(parents=True, exist_ok=False)
    plan = screen_fit_plan()
    campaign._append({
        "event": "card_start", "card": CARD, "run_id": run_id,
        "screen_months": list(SCREEN_MONTHS), "tracks": list(TRACKS),
        "parent_candidate": parent_result["run_id"], "model_family": "lightgbm",
        "model_config_overrides": VARIANT_SPEC["model_config_overrides"],
        "expected_fit_starts": len(plan), "data_sha256": source["data_sha256"],
        "parent_result_sha256": _sha256(parent_path),
        "parent_predictions_sha256": parent_manifest["paired_predictions_sha256"],
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    fit_details, component_manifest = [], []
    rows_by_track = {track: [] for track in TRACKS}
    old_fit_record = greedy._fit_record
    old_hgbr = greedy._frozen_hgbr_predictions
    old_variant = greedy.VARIANTS.get(VARIANT)
    greedy.VARIANTS[VARIANT] = VARIANT_SPEC
    fit_context: dict[str, str] = {}
    fit_start_count = 0

    def record_fit(_fit_id_original: str, fit_fn):
        nonlocal fit_start_count
        if not fit_context:
            raise RuntimeError("R30 fit context is not initialized")
        track, month, component = (fit_context[key] for key in ("track", "month", "component"))
        fit_id = _fit_id(run_id, track, month, component)
        fit_start_count += 1
        campaign._append({"event": "fit_start", "card": CARD, "stage": "screen",
                          "run_id": run_id, "fit_id": fit_id, "track": track,
                          "month": month, "component": component,
                          "data_sha256": source["data_sha256"]})
        started = time.perf_counter()
        try:
            fitted = fit_fn()
        except Exception as exc:
            campaign._append({"event": "fit_complete", "card": CARD, "stage": "screen",
                              "run_id": run_id, "fit_id": fit_id, "track": track,
                              "month": month, "component": component, "status": "failed",
                              "seconds": time.perf_counter() - started,
                              "error": f"{type(exc).__name__}: {exc}"})
            raise
        detail = {"fit_id": fit_id, "track": track, "month": month,
                  "component": component, "seconds": time.perf_counter() - started,
                  "model_family": fitted["model_family"], "model_config": fitted["model_config"],
                  "target": fitted["target"], "actual_objective": fitted.get("actual_objective"),
                  "training_rows": fitted["training_rows"], "evaluation_rows": fitted["evaluation_rows"],
                  "tfidf_feature_count": sum(name.startswith("remarks_tfidf_")
                                              for name in fitted["feature_columns"]),
                  "prediction_sha256": fitted["prediction_sha256"]}
        fit_details.append(detail)
        campaign._append({"event": "fit_complete", "card": CARD, "stage": "screen",
                          "run_id": run_id, **detail, "status": "complete"})
        return fitted

    greedy._fit_record = record_fit
    greedy._frozen_hgbr_predictions = lambda month, track, train, homes: _verified_hgbr_cache(
        greedy, month, track, train, homes)
    try:
        for month in SCREEN_MONTHS:
            _, train, _, truth, heldout = greedy._load_fold(month)
            cutoff = pd.Period(month, freq="M").start_time
            if (pd.to_datetime(train.date_sold, errors="raise") >= cutoff).any():
                raise RuntimeError(f"R30 training rows include sales at or after cutoff {month}")
            for track in TRACKS:
                homes = hide_answers(heldout, track)
                reference = shared.r22_reference_frame(parent_rows, truth, month, track)
                if track == "off_market":
                    specialist = greedy._specialist_mask(train, "Detached")
                    eval_specialist = greedy._specialist_mask(homes, "Detached")
                    fit_jobs = [("base", train, homes),
                                ("expert", train.loc[specialist].copy(),
                                 homes.loc[eval_specialist].copy())]
                else:
                    fit_jobs = [("core", train, homes)]
                fitted_components = {}
                for component, fit_rows, eval_rows in fit_jobs:
                    transformer, restore = r22._install_text_features(fit_rows, eval_rows, track)
                    fit_context.update(track=track, month=month, component=component)
                    try:
                        if track == "off_market":
                            fitted = greedy._fit_history_candidate(
                                month, track, train, eval_rows, base_variant="G06_lot_garage",
                                variant=VARIANT, fit_tag=f"R30-{component}-{month}", fit_rows=fit_rows)
                        else:
                            fitted = greedy._fit_history_candidate(
                                month, track, train, eval_rows, variant=VARIANT,
                                fit_tag=f"R30-core-{month}", fit_rows=fit_rows)
                    finally:
                        restore()
                    fitted_components[component] = fitted
                    component_manifest.append({
                        "track": track, "month": month, "component": component,
                        "model_family": fitted["model_family"], "model_config": fitted["model_config"],
                        "target": fitted["target"], "actual_objective": fitted.get("actual_objective"),
                        "training_rows": fitted["training_rows"], "evaluation_rows": fitted["evaluation_rows"],
                        "tfidf_feature_count": len(transformer.columns_ or []),
                        "vocabulary_terms": transformer.vectorizer.get_feature_names_out().tolist(),
                        "prediction_sha256": fitted["prediction_sha256"],
                    })
                if track == "off_market":
                    candidate_prediction = greedy._blend_specialist_predictions(
                        np.round(fitted_components["base"]["predictions"]),
                        np.round(fitted_components["expert"]["predictions"]),
                        homes, "Detached", 0.75)
                else:
                    frozen = greedy._frozen_hgbr_predictions(month, track, train, homes)
                    candidate_prediction = blend_prices(fitted_components["core"]["predictions"],
                                                        frozen, challenger_weight=0.75)
                candidate = pd.DataFrame({"ml_num": homes.ml_num.astype(str).to_numpy(),
                                          "sold_price": truth.sold_price.to_numpy(float),
                                          "predicted_price": np.asarray(candidate_prediction, dtype=float)})
                parent = reference[["ml_num", "sold_price", "predicted_price"]].copy()
                paired = helpers._candidate_frame(month, track, truth, parent, candidate)
                rows_by_track[track].append(paired)
                campaign._append({"event": "screen_fold_complete", "card": CARD,
                                  "run_id": run_id, "track": track, "month": month,
                                  "reference_run_id": parent_result["run_id"],
                                  "date_cutoff_exclusive": cutoff.isoformat(),
                                  "reserved_months_labels_opened": False,
                                  "hidden_labels_opened": False})
    except Exception as exc:
        failure = {"run_id": run_id, "card": CARD,
                   "error": f"{type(exc).__name__}: {exc}",
                   "fit_starts": fit_start_count, "fit_details": fit_details,
                   "completed_components": [f"{r['track']}/{r['month']}/{r['component']}"
                                             for r in component_manifest],
                   "reserved_months_labels_opened": False, "hidden_labels_opened": False}
        _write_json(output / "failure_checkpoint.json", failure)
        campaign._append({"event": "card_failed", "card": CARD, "run_id": run_id,
                          "fit_starts": fit_start_count, "error": failure["error"]})
        campaign._write_progress("r30_screen_failed",
                                 next_action="audit R30 failure checkpoint and fit ledger before any retry",
                                 r30_failure=str((output / "failure_checkpoint.json").relative_to(
                                     campaign.ROOT)).replace("\\", "/"))
        raise
    finally:
        greedy._fit_record, greedy._frozen_hgbr_predictions = old_fit_record, old_hgbr
        if old_variant is None:
            greedy.VARIANTS.pop(VARIANT, None)
        else:
            greedy.VARIANTS[VARIANT] = old_variant

    if fit_start_count != len(plan):
        raise RuntimeError(f"R30 expected {len(plan)} fit starts; observed {fit_start_count}")
    paired = pd.concat([frame for groups in rows_by_track.values() for frame in groups], ignore_index=True)
    paired_path = output / "paired_predictions.csv"
    paired.to_csv(paired_path, index=False)
    tracks = {track: shared._track_result(pd.concat(rows, ignore_index=True))
              for track, rows in rows_by_track.items()}
    result = {"campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
              "card": CARD, "parent_candidate": parent_result["run_id"],
              "screen_months": list(SCREEN_MONTHS), "fit_starts": fit_start_count,
              "tracks": tracks, "fit_details": fit_details,
              "model_config_overrides": VARIANT_SPEC["model_config_overrides"],
              "component_manifest": component_manifest,
              "reserved_months_labels_opened": False, "hidden_labels_opened": False,
              "data_sha256": source["data_sha256"]}
    result_path = output / "r30_screen_results.json"
    _write_json(result_path, result)
    provenance = output / "provenance"
    provenance.mkdir(exist_ok=True)
    shutil.copy2(campaign.PACKAGE / "r30_quantile_r22.py", provenance / "r30_quantile_r22_source.py")
    shutil.copy2(campaign.PROTOCOL_PATH, provenance / "r30_preregistered_campaign_protocol.json")
    manifest = {"campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
                "source_identity": source,
                "code_sha256": {str(path.relative_to(campaign.ROOT)).replace("\\", "/"): _sha256(path)
                                for path in _source_files(campaign)},
                "data_sha256": source["data_sha256"], "fit_starts": fit_start_count,
                "fit_details": fit_details, "component_manifest": component_manifest,
                "paired_predictions_sha256": _sha256(paired_path),
                "result_sha256": _sha256(result_path),
                "parent_run_id": parent_result["run_id"],
                "parent_result_sha256": _sha256(parent_path),
                "parent_predictions_sha256": parent_manifest["paired_predictions_sha256"],
                "reserved_months_labels_opened": False, "hidden_labels_opened": False}
    _write_json(output / "run_manifest.json", manifest)
    (output / "research_report.md").write_text(_report(result), encoding="utf-8")
    qualifies = {track: score["qualifies_screen"] for track, score in tracks.items()}
    campaign._append({"event": "card_complete", "card": CARD, "run_id": run_id,
                      "fit_starts": fit_start_count,
                      "result_path": str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
                      "qualifies_by_track": qualifies,
                      "reserved_months_labels_opened": False, "hidden_labels_opened": False})
    next_action = ("register R30 development confirmation for screen-qualified tracks"
                   if any(qualifies.values()) else
                   "R30 quantile-plus-R22-text interaction failed screen; retain R22 and register a distinct method")
    campaign._write_progress("r30_screen_complete", next_action=next_action,
                             r30_result=str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
                             latest_fit_count=fit_start_count)
    return result


def main() -> None:
    from experiments.bill_target74 import runner as campaign

    parser_result = None
    if campaign.PROTOCOL["cards"].get(CARD, {}).get("status") != "registered":
        raise RuntimeError("R30 must be preregistered before fitting")
    parser_result = run_r30_screen()
    for track, score in parser_result["tracks"].items():
        print(f"{track}: R22={score['parent']['mdape_percent']:.5f}% "
              f"R30={score['candidate']['mdape_percent']:.5f}% "
              f"gain={score['relative_mdape_improvement_percent']:+.3f}% "
              f"screen_pass={score['qualifies_screen']}", flush=True)
    print(f"run_id={parser_result['run_id']}", flush=True)


if __name__ == "__main__":
    main()
