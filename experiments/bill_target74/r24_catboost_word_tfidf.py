"""R24: compare CatBoost using the verified R22 train-fold word TF-IDF features."""
from __future__ import annotations

import json
import shutil
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from experiments.bill_target74.r19_expert_weight import _sha256, _write_json


CARD = "R24_catboost_with_r22_word_tfidf"
VARIANT = CARD
SCREEN_MONTHS = ("2023-03", "2024-06")
TRACKS = ("off_market", "list_aware")


def screen_fit_plan() -> list[tuple[str, str, str]]:
    return [
        (month, track, component)
        for month in SCREEN_MONTHS
        for track, components in (("off_market", ("base", "expert")),
                                  ("list_aware", ("core",)))
        for component in components
    ]


def _install_catboost_word_features(fit_rows: pd.DataFrame, eval_rows: pd.DataFrame,
                                    track: str, *, transformer=None):
    """Fit R22 word vocabulary on fold training rows and transform CatBoost's sorted copy."""
    from model_candidate_v2 import features as frozen_features
    from experiments.bill_target74.r22_remarks_tfidf import FoldTfidf

    if track not in TRACKS:
        raise ValueError(f"unknown track: {track!r}")
    if "remarks" not in fit_rows or "remarks" not in eval_rows:
        raise ValueError("training and evaluation rows must both contain remarks")
    if "ml_num" not in fit_rows or "ml_num" not in eval_rows:
        raise ValueError("training and evaluation rows must have listing IDs")
    fit_ids = fit_rows.ml_num.astype(str).tolist()
    eval_ids = eval_rows.ml_num.astype(str).tolist()
    if len(set(fit_ids)) != len(fit_ids) or len(set(eval_ids)) != len(eval_ids):
        raise ValueError("training and evaluation listing IDs must be unique")
    fit_id_set, eval_id_set = set(fit_ids), set(eval_ids)
    if fit_id_set.intersection(eval_id_set):
        raise ValueError("training and evaluation listing IDs must be disjoint")

    attribute = "off_market_features" if track == "off_market" else "list_aware_features"
    original = getattr(frozen_features, attribute)
    transformer = transformer or FoldTfidf()
    fitted_training_vocabulary = False

    def wrapped(homes):
        nonlocal fitted_training_vocabulary
        base = original(homes)
        row_ids = homes.ml_num.astype(str).tolist()
        row_id_set = set(row_ids)
        if row_id_set == fit_id_set:
            if not fitted_training_vocabulary:
                text = transformer.fit_transform(homes.remarks, index=base.index)
                fitted_training_vocabulary = True
            else:
                text = transformer.transform(homes.remarks, index=base.index)
        elif row_id_set == eval_id_set:
            if not fitted_training_vocabulary:
                raise RuntimeError("CatBoost evaluation transform was requested before training fit")
            text = transformer.transform(homes.remarks, index=base.index)
        else:
            raise RuntimeError("unexpected listing rows passed to the fold-scoped text feature builder")
        return pd.concat([base, text], axis=1)

    setattr(frozen_features, attribute, wrapped)

    def restore() -> None:
        setattr(frozen_features, attribute, original)

    return transformer, restore


def _fit_id(run_id: str, track: str, month: str, component: str) -> str:
    return f"bill_target74:{CARD}:{track}:{month}:{component}:{run_id}"


def _report(result: dict) -> str:
    lines = [
        "# R24：CatBoost + R22 训练折词级 TF-IDF", "",
        f"Campaign：`{result['campaign_id']}`；run：`{result['run_id']}`。",
        f"增量比较父模型：通过哈希验证的 R22 run `{result['parent_candidate']}`。",
        "与 R22 相比只替换学习器为固定的 CatBoost 配置；词级 TF-IDF、结构化/历史特征、目标、分组路由与预测融合保持不变。",
        "R03 测过不含 R22 remarks 的 CatBoost；本轮只检验 CatBoost 与训练折文本特征的组合是否优于 R22。",
        "", "## 双月增量筛选", "",
        "| 赛道 | R22 MdAPE | R24 MdAPE | 相对改善 | MAPE变化 | 通过 |",
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
    lines.extend(["", "## 各筛选月", "", "| 赛道 | 月份 | R22 MdAPE | R24 MdAPE | 相对改善 |",
                  "|---|---|---:|---:|---:|"])
    for track, score in result["tracks"].items():
        for month, values in score["monthly"].items():
            lines.append(f"| {track} | {month} | {values['parent']['mdape_percent']:.5f}% | "
                         f"{values['candidate']['mdape_percent']:.5f}% | "
                         f"{values['relative_mdape_improvement_percent']:+.3f}% |")
    lines.extend(["", "保留月份和隐藏标签未读取。筛选通过也只允许进入八个月开发确认，不代表严格目标已经达到。", ""])
    return "\n".join(lines)


def run_r24_screen() -> dict:
    from data import hide_answers
    from experiments.bill_target74 import r03_catboost as helpers
    from experiments.bill_target74 import r04_ablation as budget
    from experiments.bill_target74 import runner as campaign
    from experiments.bill_target74 import r22_remarks_tfidf as r22
    from experiments.bill_target74 import r23_word_char_tfidf as shared
    from experiments.bill_target74.r03_catboost import CATBOOST_CONFIG, _verified_hgbr_cache
    from experiments.bill_greedy import runner as greedy

    if campaign.PROTOCOL["cards"].get(CARD, {}).get("status") != "registered":
        raise RuntimeError("R24 must be registered in campaign_protocol.json before fitting")
    r22_result, r22_result_path, r22_manifest = r22._verified_screen_result(campaign)
    if not all(r22_result["tracks"][track]["qualifies_screen"] for track in TRACKS):
        raise RuntimeError("R24 requires both tracks to have passed the verified R22 screen")
    r22_rows_path = r22_result_path.parent / "paired_predictions.csv"
    r22_rows = pd.read_csv(r22_rows_path, dtype={"ml_num": str})
    source = budget._source_identity()
    run_id = "bill_target74_r24_catboost_word_tfidf_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = campaign.CAMPAIGN_ROOT / "batches" / run_id
    output.mkdir(parents=True, exist_ok=False)
    campaign._append({
        "event": "card_start", "card": CARD, "run_id": run_id,
        "screen_months": list(SCREEN_MONTHS), "tracks": list(TRACKS),
        "parent_candidate": r22_result["run_id"], "model_family": "catboost",
        "model_config": CATBOOST_CONFIG, "expected_fit_starts": 6,
        "text_features": {"word_max_features": 256, "min_df": 25, "max_df": 0.95,
                           "ngram_range": [1, 2], "stop_words": "english",
                           "vocabulary_policy": "fit training rows only"},
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    fit_details, components, rows_by_track = [], [], {track: [] for track in TRACKS}
    old_fit_record, old_hgbr = greedy._fit_record, greedy._frozen_hgbr_predictions
    had_variant, old_variant = VARIANT in greedy.VARIANTS, greedy.VARIANTS.get(VARIANT)
    greedy.VARIANTS[VARIANT] = {"model_family": "catboost", "model_config": dict(CATBOOST_CONFIG)}
    fit_context: dict[str, str] = {}
    fit_start_count = 0

    def record_fit(_original_fit_id: str, fit_fn):
        nonlocal fit_start_count
        if not fit_context:
            raise RuntimeError("R24 fit context is not initialized")
        track, month, component = (fit_context[key] for key in ("track", "month", "component"))
        fit_id = _fit_id(run_id, track, month, component)
        fit_start_count += 1
        campaign._append({"event": "fit_start", "card": CARD, "run_id": run_id,
                          "fit_id": fit_id, "track": track, "month": month,
                          "component": component, "data_sha256": source["data_sha256"]})
        started = time.perf_counter()
        try:
            fitted = fit_fn()
        except Exception as exc:
            campaign._append({"event": "fit_complete", "card": CARD, "run_id": run_id,
                              "fit_id": fit_id, "track": track, "month": month,
                              "component": component, "status": "failed",
                              "seconds": time.perf_counter() - started,
                              "error": f"{type(exc).__name__}: {exc}"})
            raise
        seconds = time.perf_counter() - started
        detail = {"fit_id": fit_id, "track": track, "month": month, "component": component,
                  "seconds": seconds, "model_family": fitted["model_family"],
                  "model_config": fitted["model_config"], "target": fitted["target"],
                  "actual_objective": fitted.get("actual_objective"),
                  "training_rows": fitted["training_rows"], "evaluation_rows": fitted["evaluation_rows"],
                  "feature_columns": fitted["feature_columns"],
                  "text_feature_count": sum(name.startswith("remarks_tfidf_")
                                             for name in fitted["feature_columns"]),
                  "prediction_sha256": fitted["prediction_sha256"]}
        fit_details.append(detail)
        campaign._append({"event": "fit_complete", "card": CARD, "run_id": run_id,
                          "fit_id": fit_id, "track": track, "month": month,
                          "component": component, "status": "complete", "seconds": seconds,
                          "prediction_sha256": fitted["prediction_sha256"],
                          "training_rows": fitted["training_rows"],
                          "model_family": fitted["model_family"],
                          "actual_objective": fitted.get("actual_objective")})
        return fitted

    greedy._fit_record = record_fit
    greedy._frozen_hgbr_predictions = lambda month, track, train, homes: _verified_hgbr_cache(
        greedy, month, track, train, homes)
    try:
        for month in SCREEN_MONTHS:
            _, train, _, truth, heldout = greedy._load_fold(month)
            cutoff = pd.Period(month, freq="M").start_time
            if (pd.to_datetime(train.date_sold, errors="raise") >= cutoff).any():
                raise RuntimeError(f"R24 training data contain sales at or after cutoff: {month}")
            for track in TRACKS:
                homes = hide_answers(heldout, track)
                reference = shared.r22_reference_frame(r22_rows, truth, month, track)
                if track == "off_market":
                    specialist = greedy._specialist_mask(train, "Detached")
                    eval_specialist = greedy._specialist_mask(homes, "Detached")
                    fit_jobs = [
                        ("base", train, homes),
                        ("expert", train.loc[specialist].copy(), homes.loc[eval_specialist].copy()),
                    ]
                    fitted_components = {}
                    for component, fit_rows, eval_rows in fit_jobs:
                        transformer, restore = r24_installer(fit_rows, eval_rows, track)
                        fit_context.update(track=track, month=month, component=component)
                        try:
                            fitted = greedy._fit_history_candidate(
                                month, track, train, eval_rows, base_variant="G06_lot_garage",
                                variant=VARIANT, fit_tag=f"R24-{component}-{month}", fit_rows=fit_rows,
                            )
                        finally:
                            restore()
                        fitted_components[component] = fitted
                        components.append({"track": track, "month": month, "component": component,
                                           "prediction_sha256": fitted["prediction_sha256"],
                                           "feature_columns": fitted["feature_columns"],
                                           "word_vocabulary": transformer.vectorizer.get_feature_names_out().tolist()})
                    base_prediction = np.round(fitted_components["base"]["predictions"])
                    expert_prediction = np.round(fitted_components["expert"]["predictions"])
                    candidate_prediction = greedy._blend_specialist_predictions(
                        base_prediction, expert_prediction, homes, "Detached", 0.75)
                else:
                    transformer, restore = r24_installer(train, homes, track)
                    fit_context.update(track=track, month=month, component="core")
                    try:
                        core = greedy._fit_history_candidate(
                            month, track, train, homes, variant=VARIANT,
                            fit_tag=f"R24-core-{month}", fit_rows=train,
                        )
                    finally:
                        restore()
                    frozen = greedy._frozen_hgbr_predictions(month, track, train, homes)
                    from experiments.model_optimization_v2.pipeline import blend_prices
                    candidate_prediction = blend_prices(core["predictions"], frozen, challenger_weight=0.75)
                    components.append({"track": track, "month": month, "component": "core",
                                       "prediction_sha256": core["prediction_sha256"],
                                       "feature_columns": core["feature_columns"],
                                       "word_vocabulary": transformer.vectorizer.get_feature_names_out().tolist()})

                candidate = pd.DataFrame({"ml_num": homes.ml_num.astype(str).to_numpy(),
                                          "sold_price": truth.sold_price.to_numpy(float),
                                          "predicted_price": np.asarray(candidate_prediction, dtype=float)})
                parent = reference[["ml_num", "sold_price", "predicted_price"]].copy()
                paired = helpers._candidate_frame(month, track, truth, parent, candidate)
                if "initial_parent_prediction" in reference:
                    paired["initial_parent_prediction"] = reference.initial_parent_prediction.to_numpy(float)
                rows_by_track[track].append(paired)
                campaign._append({"event": "screen_fold_complete", "card": CARD,
                                  "run_id": run_id, "track": track, "month": month,
                                  "reference_run_id": r22_result["run_id"],
                                  "date_cutoff_exclusive": cutoff.isoformat()})
    except Exception as exc:
        failure = {"run_id": run_id, "card": CARD, "error": f"{type(exc).__name__}: {exc}",
                   "fit_starts": fit_start_count, "fit_details": fit_details,
                   "reserved_months_labels_opened": False, "hidden_labels_opened": False}
        _write_json(output / "failure_checkpoint.json", failure)
        campaign._append({"event": "card_failed", "card": CARD, "run_id": run_id,
                          "fit_starts": fit_start_count, "error": failure["error"]})
        campaign._write_progress("r24_catboost_word_tfidf_failed",
                                 next_action="inspect R24 checkpoint and ledger before retry",
                                 r24_failure=str((output / "failure_checkpoint.json").relative_to(campaign.ROOT)).replace("\\", "/"))
        raise
    finally:
        greedy._fit_record, greedy._frozen_hgbr_predictions = old_fit_record, old_hgbr
        if had_variant:
            greedy.VARIANTS[VARIANT] = old_variant
        else:
            greedy.VARIANTS.pop(VARIANT, None)

    paired = pd.concat([frame for groups in rows_by_track.values() for frame in groups], ignore_index=True)
    paired_path = output / "paired_predictions.csv"
    paired.to_csv(paired_path, index=False)
    tracks = {track: shared._track_result(pd.concat(rows, ignore_index=True))
              for track, rows in rows_by_track.items()}
    result = {"campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
              "card": CARD, "parent_candidate": r22_result["run_id"],
              "screen_months": list(SCREEN_MONTHS), "fit_starts": fit_start_count,
              "tracks": tracks, "fit_details": fit_details,
              "reserved_months_labels_opened": False, "hidden_labels_opened": False,
              "data_sha256": source["data_sha256"]}
    result_path = output / "r24_screen_results.json"
    _write_json(result_path, result)
    source_files = [campaign.PROTOCOL_PATH, campaign.PACKAGE / "r24_catboost_word_tfidf.py",
                    campaign.PACKAGE / "r22_remarks_tfidf.py",
                    campaign.PACKAGE / "r23_word_char_tfidf.py",
                    campaign.PACKAGE / "r03_catboost.py", campaign.PACKAGE / "r04_ablation.py",
                    campaign.PACKAGE / "runner.py", campaign.ROOT / "model_candidate_v2/features.py",
                    campaign.ROOT / "experiments/bill_greedy/runner.py",
                    campaign.ROOT / "experiments/bill_greedy/history.py",
                    campaign.ROOT / "experiments/model_optimization_v2/learners.py"]
    provenance = output / "provenance"
    provenance.mkdir(exist_ok=True)
    shutil.copy2(campaign.PACKAGE / "r24_catboost_word_tfidf.py",
                 provenance / "r24_catboost_word_tfidf_source.py")
    manifest = {
        "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
        "source_identity": source,
        "code_sha256": {str(path.relative_to(campaign.ROOT)).replace("\\", "/"): _sha256(path)
                        for path in source_files},
        "data_sha256": source["data_sha256"], "fit_starts": fit_start_count,
        "fit_details": fit_details, "component_manifest": components,
        "paired_predictions_sha256": _sha256(paired_path),
        "result_sha256": _sha256(result_path),
        "r22_parent_run_id": r22_result["run_id"],
        "r22_parent_result_sha256": _sha256(r22_result_path),
        "r22_parent_predictions_sha256": r22_manifest["paired_predictions_sha256"],
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    }
    _write_json(output / "run_manifest.json", manifest)
    (output / "research_report.md").write_text(_report(result), encoding="utf-8")
    qualifies = {track: score["qualifies_screen"] for track, score in tracks.items()}
    campaign._append({"event": "card_complete", "card": CARD, "run_id": run_id,
                      "fit_starts": fit_start_count,
                      "result_path": str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
                      "qualifies_by_track": qualifies,
                      "reserved_months_labels_opened": False, "hidden_labels_opened": False})
    next_action = ("run R24 eight-month development confirmation only for screen-qualified tracks"
                   if any(qualifies.values()) else
                   "R24 CatBoost with R22 remarks failed the incremental screen; keep R22 and register a distinct method")
    campaign._write_progress("r24_catboost_word_tfidf_screen_complete",
                             next_action=next_action,
                             r24_result=str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
                             latest_fit_count=fit_start_count)
    return result


def r24_installer(fit_rows, eval_rows, track: str):
    return _install_catboost_word_features(fit_rows, eval_rows, track)


def main() -> None:
    result = run_r24_screen()
    print(json.dumps({"run_id": result["run_id"], "fit_starts": result["fit_starts"],
                      "tracks": result["tracks"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
