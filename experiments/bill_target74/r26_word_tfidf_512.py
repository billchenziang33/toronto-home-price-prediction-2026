"""R26: test a wider train-fold word TF-IDF vocabulary against verified R22."""
from __future__ import annotations

import json
import shutil
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from experiments.bill_target74.r19_expert_weight import _sha256, _write_json


CARD = "R26_word_tfidf_512"
VARIANT = CARD
SCREEN_MONTHS = ("2023-03", "2024-06")
TRACKS = ("off_market", "list_aware")
MAX_FEATURES = 512


def screen_fit_plan() -> list[tuple[str, str, str]]:
    return [
        (month, track, component)
        for month in SCREEN_MONTHS
        for track, components in (("off_market", ("base", "expert")),
                                  ("list_aware", ("core",)))
        for component in components
    ]


def _fit_id(run_id: str, track: str, month: str, component: str) -> str:
    return f"bill_target74:{CARD}:{track}:{month}:{component}:{run_id}"


def _install_wide_word_features(fit_rows: pd.DataFrame, eval_rows: pd.DataFrame,
                                track: str):
    """Fit the 512-term R22-compatible TF-IDF only on this component's train fold."""
    from model_candidate_v2 import features as frozen_features
    from experiments.bill_target74.r22_remarks_tfidf import FoldTfidf

    if track not in TRACKS:
        raise ValueError(f"unknown track: {track!r}")
    if "remarks" not in fit_rows or "remarks" not in eval_rows:
        raise ValueError("training and evaluation rows must both contain remarks")
    attribute = "off_market_features" if track == "off_market" else "list_aware_features"
    original = getattr(frozen_features, attribute)
    transformer = FoldTfidf(max_features=MAX_FEATURES)
    fit_identity, eval_identity = id(fit_rows), id(eval_rows)
    fitted_training_vocabulary = False

    def wrapped(homes):
        nonlocal fitted_training_vocabulary
        base = original(homes)
        if id(homes) == fit_identity:
            if not fitted_training_vocabulary:
                text = transformer.fit_transform(homes.remarks, index=base.index)
                fitted_training_vocabulary = True
            else:
                text = transformer.transform(homes.remarks, index=base.index)
        elif id(homes) == eval_identity:
            if not fitted_training_vocabulary:
                raise RuntimeError("evaluation transform was requested before training vocabulary fit")
            text = transformer.transform(homes.remarks, index=base.index)
        else:
            raise RuntimeError("unexpected rows passed to the fold-scoped TF-IDF feature builder")
        return pd.concat([base, text], axis=1)

    setattr(frozen_features, attribute, wrapped)

    def restore() -> None:
        setattr(frozen_features, attribute, original)

    return transformer, restore


def _report(result: dict) -> str:
    lines = [
        "# R26：扩大 R22 词级 TF-IDF 词表至 512 项", "",
        f"Campaign：`{result['campaign_id']}`；run：`{result['run_id']}`。",
        f"增量父模型：哈希验证的 R22 run `{result['parent_candidate']}`。",
        "只把词级 unigram/bigram TF-IDF 的 `max_features` 从 256 提高到 512；LightGBM、其他特征、训练目标、路由和融合保持不变。",
        "每个组件的词表和 IDF 只在该组件训练折 remarks 上拟合。", "",
        "## 双月增量筛选", "",
        "| 赛道 | R22 MdAPE | R26 MdAPE | 相对改善 | MAPE变化 | 通过 |",
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
    lines.extend(["", "## 各筛选月", "", "| 赛道 | 月份 | R22 MdAPE | R26 MdAPE | 相对改善 |",
                  "|---|---|---:|---:|---:|"])
    for track, score in result["tracks"].items():
        for month, values in score["monthly"].items():
            lines.append(f"| {track} | {month} | {values['parent']['mdape_percent']:.5f}% | "
                         f"{values['candidate']['mdape_percent']:.5f}% | "
                         f"{values['relative_mdape_improvement_percent']:+.3f}% |")
    lines.extend(["", "保留月份与隐藏标签未读取。只有通过筛选的轨道才可进入八个月开发确认。", ""])
    return "\n".join(lines)


def _verified_r26_screen_result(campaign, expected_run_id: str | None = None):
    progress = json.loads(campaign.STATE.read_text(encoding="utf-8"))
    relative = progress.get("r26_result")
    if not relative:
        raise RuntimeError("campaign progress does not identify the completed R26 screen")
    result_path = campaign.ROOT / relative
    if not result_path.is_file():
        raise FileNotFoundError(f"R26 screen result is missing: {result_path}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    output = result_path.parent
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    paired_path = output / "paired_predictions.csv"
    if (result.get("campaign_id") != campaign.PROTOCOL["campaign_id"]
            or result.get("card") != CARD or manifest.get("run_id") != result.get("run_id")):
        raise RuntimeError("R26 screen identity does not match the active campaign")
    if expected_run_id and result.get("run_id") != expected_run_id:
        raise RuntimeError(f"requested R26 screen {expected_run_id!r} differs from saved screen")
    if (manifest.get("result_sha256") != _sha256(result_path)
            or manifest.get("paired_predictions_sha256") != _sha256(paired_path)
            or manifest.get("fit_starts") != result.get("fit_starts")):
        raise RuntimeError("R26 screen result or paired predictions fail manifest verification")
    source_copy = output / "provenance" / "r26_word_tfidf_512_source.py"
    protocol_copy = output / "provenance" / "r26_preregistered_campaign_protocol.json"
    code_hashes = manifest.get("code_sha256", {})
    if (_sha256(source_copy) != code_hashes.get("experiments/bill_target74/r26_word_tfidf_512.py")
            or _sha256(protocol_copy) != code_hashes.get("experiments/bill_target74/campaign_protocol.json")):
        raise RuntimeError("R26 screen source or preregistered protocol snapshot fails manifest verification")
    return result, result_path, manifest


def _r22_development_parent(rows: pd.DataFrame, truth: pd.DataFrame, month: str,
                            track: str) -> tuple[pd.DataFrame, np.ndarray]:
    required = {"month", "track", "ml_num", "sold_price", "parent_prediction", "candidate_prediction"}
    if missing := required - set(rows.columns):
        raise ValueError(f"R22 development rows are missing columns: {sorted(missing)}")
    selected = rows.loc[rows.month.eq(month) & rows.track.eq(track)].copy()
    if selected.ml_num.astype(str).duplicated().any() or len(selected) != len(truth):
        raise RuntimeError(f"R22 development rows are duplicated or incomplete for {month}/{track}")
    ids = truth.ml_num.astype(str).to_numpy()
    indexed = selected.assign(ml_num=selected.ml_num.astype(str)).set_index("ml_num")
    aligned = indexed.reindex(ids)
    if (aligned.candidate_prediction.isna().any()
            or not np.array_equal(aligned.sold_price.to_numpy(float), truth.sold_price.to_numpy(float))):
        raise RuntimeError(f"R22 development candidate does not align with {month}/{track} truth")
    candidate = aligned.candidate_prediction.to_numpy(float)
    if not np.isfinite(candidate).all() or (candidate <= 0.0).any():
        raise RuntimeError(f"R22 development candidate has invalid prices for {month}/{track}")
    parent = pd.DataFrame({"ml_num": ids, "sold_price": truth.sold_price.to_numpy(float),
                           "predicted_price": candidate})
    initial = aligned.parent_prediction.to_numpy(float)
    return parent, initial


def run_r26_development_confirmation(screen_run_id: str | None = None) -> dict:
    """Confirm the screen-qualified off-market R26 route on all eight frozen months."""
    from data import hide_answers
    from experiments.bill_target74 import r03_catboost as helpers
    from experiments.bill_target74 import r04_ablation as budget
    from experiments.bill_target74 import runner as campaign
    from experiments.bill_target74 import r22_remarks_tfidf as r22
    from experiments.bill_target74 import r23_word_char_tfidf as shared
    from experiments.bill_target74.r03_catboost import _verified_hgbr_cache
    from experiments.bill_greedy import runner as greedy

    screen, screen_path, screen_manifest = _verified_r26_screen_result(campaign, screen_run_id)
    eligible_tracks = tuple(track for track in TRACKS
                            if screen.get("tracks", {}).get(track, {}).get("qualifies_screen") is True)
    if eligible_tracks != ("off_market",):
        raise RuntimeError(f"R26 development authorization expected only off-market, found {eligible_tracks}")
    r22_dev, r22_dev_path, r22_dev_manifest = r22._verified_development_result(campaign)
    if not r22_dev.get("tracks", {}).get("off_market", {}).get("qualifies_development"):
        raise RuntimeError("verified R22 off-market development result is not a qualified comparison parent")
    r22_dev_rows = pd.read_csv(r22_dev_path.parent / "paired_predictions.csv", dtype={"ml_num": str})
    source = budget._source_identity()
    run_id = "bill_target74_r26_word_tfidf_512_development_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = campaign.CAMPAIGN_ROOT / "batches" / run_id
    output.mkdir(parents=True, exist_ok=False)
    plan = [(month, "off_market", component)
            for month in r22.DEVELOPMENT_MONTHS for component in ("base", "expert")]
    if len(plan) != 16:
        raise RuntimeError(f"R26 development plan must contain 16 fits; found {len(plan)}")
    campaign._append({"event": "card_start", "card": CARD, "stage": "development_confirmation",
                      "run_id": run_id, "screen_run_id": screen["run_id"],
                      "parent_development_run_id": r22_dev["run_id"],
                      "tracks": list(eligible_tracks), "development_months": list(r22.DEVELOPMENT_MONTHS),
                      "expected_fit_starts": len(plan), "max_features": MAX_FEATURES,
                      "reserved_months_labels_opened": False, "hidden_labels_opened": False})
    fit_details, component_manifest, paired_rows = [], [], []
    old_fit_record = greedy._fit_record
    fit_context: dict[str, object] = {}
    fit_start_count = 0

    def record_fit(_original_fit_id: str, fit_fn):
        nonlocal fit_start_count
        if not fit_context:
            raise RuntimeError("R26 development fit context is not initialized")
        month, component = str(fit_context["month"]), str(fit_context["component"])
        fit_id = (f"bill_target74:{CARD}:development_confirmation:off_market:"
                  f"{month}:{component}:{run_id}")
        fit_start_count += 1
        campaign._append({"event": "fit_start", "card": CARD,
                          "stage": "development_confirmation", "run_id": run_id,
                          "fit_id": fit_id, "track": "off_market", "month": month,
                          "component": component, "data_sha256": source["data_sha256"]})
        started = time.perf_counter()
        try:
            fitted = fit_fn()
        except Exception as exc:
            campaign._append({"event": "fit_complete", "card": CARD,
                              "stage": "development_confirmation", "run_id": run_id,
                              "fit_id": fit_id, "track": "off_market", "month": month,
                              "component": component, "status": "failed",
                              "seconds": time.perf_counter() - started,
                              "error": f"{type(exc).__name__}: {exc}"})
            raise
        seconds = time.perf_counter() - started
        saved = r22._save_component_predictions(output, "off_market", month, component,
                                                fit_context["evaluation_ids"], fitted["predictions"])
        detail = {"fit_id": fit_id, "track": "off_market", "month": month,
                  "component": component, "seconds": seconds,
                  "model_family": fitted["model_family"], "model_config": fitted["model_config"],
                  "target": fitted["target"], "actual_objective": fitted.get("actual_objective"),
                  "training_rows": fitted["training_rows"],
                  "evaluation_rows": fitted["evaluation_rows"],
                  "tfidf_feature_count": sum(name.startswith("remarks_tfidf_")
                                              for name in fitted["feature_columns"]),
                  "prediction_sha256": fitted["prediction_sha256"], "saved_predictions": saved}
        fit_details.append(detail)
        campaign._append({"event": "fit_complete", "card": CARD,
                          "stage": "development_confirmation", "run_id": run_id,
                          **detail, "status": "complete"})
        return fitted

    greedy._fit_record = record_fit
    try:
        for month in r22.DEVELOPMENT_MONTHS:
            _, train, _, truth, heldout = greedy._load_fold(month)
            cutoff = pd.Period(month, freq="M").start_time
            if (pd.to_datetime(train.date_sold, errors="raise") >= cutoff).any():
                raise RuntimeError(f"R26 development training data include sales at or after cutoff: {month}")
            homes = hide_answers(heldout, "off_market")
            parent, initial_prediction = _r22_development_parent(
                r22_dev_rows, truth, month, "off_market")
            specialist = greedy._specialist_mask(train, "Detached")
            eval_specialist = greedy._specialist_mask(homes, "Detached")
            fitted_components = {}
            for component, fit_rows, eval_rows in (
                    ("base", train, homes),
                    ("expert", train.loc[specialist].copy(), homes.loc[eval_specialist].copy())):
                transformer, restore = _install_wide_word_features(fit_rows, eval_rows, "off_market")
                fit_context.update(month=month, component=component,
                                   evaluation_ids=eval_rows.ml_num.astype(str).tolist())
                try:
                    fitted = greedy._fit_history_candidate(
                        month, "off_market", train, eval_rows, base_variant="G06_lot_garage",
                        variant=VARIANT, fit_tag=f"R26-development-{component}-{month}",
                        fit_rows=fit_rows,
                    )
                finally:
                    restore()
                fitted_components[component] = fitted
                component_manifest.append({
                    "track": "off_market", "month": month, "component": component,
                    "model_family": fitted["model_family"], "model_config": fitted["model_config"],
                    "target": fitted["target"], "training_rows": fitted["training_rows"],
                    "evaluation_rows": fitted["evaluation_rows"],
                    "tfidf_feature_count": len(transformer.columns_ or []),
                    "vocabulary_terms": transformer.vectorizer.get_feature_names_out().tolist(),
                    "prediction_sha256": fitted["prediction_sha256"],
                })
            candidate_prediction = greedy._blend_specialist_predictions(
                np.round(fitted_components["base"]["predictions"]),
                np.round(fitted_components["expert"]["predictions"]), homes, "Detached", 0.75)
            candidate = pd.DataFrame({"ml_num": homes.ml_num.astype(str).to_numpy(),
                                      "sold_price": truth.sold_price.to_numpy(float),
                                      "predicted_price": np.asarray(candidate_prediction, dtype=float)})
            paired = helpers._candidate_frame(month, "off_market", truth, parent, candidate)
            paired["initial_parent_prediction"] = initial_prediction
            paired_rows.append(paired)
            campaign._append({"event": "development_fold_complete", "card": CARD,
                              "stage": "development_confirmation", "run_id": run_id,
                              "track": "off_market", "month": month,
                              "parent_candidate_run_id": r22_dev["run_id"],
                              "date_cutoff_exclusive": cutoff.isoformat(),
                              "reserved_months_labels_opened": False,
                              "hidden_labels_opened": False})
            pd.concat(paired_rows, ignore_index=True).to_csv(output / "partial_off_market.csv", index=False)
    except Exception as exc:
        failure = {"run_id": run_id, "card": CARD, "stage": "development_confirmation",
                   "error": f"{type(exc).__name__}: {exc}", "fit_starts": fit_start_count,
                   "fit_details": fit_details,
                   "completed_components": [f"{r['month']}/{r['component']}" for r in component_manifest],
                   "reserved_months_labels_opened": False, "hidden_labels_opened": False}
        _write_json(output / "failure_checkpoint.json", failure)
        campaign._append({"event": "card_failed", "card": CARD,
                          "stage": "development_confirmation", "run_id": run_id,
                          "fit_starts": fit_start_count, "error": failure["error"]})
        campaign._write_progress("r26_development_confirmation_failed",
                                 next_action="audit R26 development checkpoint and ledger before any retry",
                                 r26_development_failure=str(
                                     (output / "failure_checkpoint.json").relative_to(campaign.ROOT)
                                 ).replace("\\", "/"))
        raise
    finally:
        greedy._fit_record = old_fit_record

    if fit_start_count != len(plan):
        raise RuntimeError(f"R26 development expected {len(plan)} fit starts; observed {fit_start_count}")
    paired_frame = pd.concat(paired_rows, ignore_index=True)
    paired_path = output / "paired_predictions.csv"
    paired_frame.to_csv(paired_path, index=False)
    gates = campaign.PROTOCOL["gates"]
    target = campaign.PROTOCOL["initial_development_metrics"]["off_market"]["target_mdape_strictly_below"]
    scored = r22.score_development_confirmation(
        paired_frame, months=tuple(r22.DEVELOPMENT_MONTHS), target_mdape_strictly_below=target,
        bootstrap_replicates=2000, bootstrap_seed=20261024,
        minimum_relative_gain_percent=gates["development_relative_mdape_improvement_percent"],
        minimum_months_improved=gates["development_months_improved_min"],
        maximum_worst_month_regression_percent=gates["development_worst_month_regression_max_percent"],
        maximum_mape_regression_percent=gates["development_mape_regression_max_percent"],
    )
    result = {"campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
              "card": CARD, "stage": "development_confirmation",
              "screen_run_id": screen["run_id"], "parent_development_run_id": r22_dev["run_id"],
              "track": "off_market", "development_months": list(r22.DEVELOPMENT_MONTHS),
              "max_features": MAX_FEATURES, "fit_starts": fit_start_count,
              "tracks": {"off_market": scored}, "fit_details": fit_details,
              "component_manifest": component_manifest,
              "reserved_months_labels_opened": False, "hidden_labels_opened": False,
              "data_sha256": source["data_sha256"]}
    result_path = output / "r26_development_results.json"
    _write_json(result_path, result)
    source_files = [campaign.PROTOCOL_PATH, campaign.PACKAGE / "r26_word_tfidf_512.py",
                    campaign.PACKAGE / "r22_remarks_tfidf.py",
                    campaign.PACKAGE / "r23_word_char_tfidf.py",
                    campaign.PACKAGE / "r03_catboost.py", campaign.PACKAGE / "r04_ablation.py",
                    campaign.PACKAGE / "runner.py", campaign.ROOT / "model_candidate_v2/features.py",
                    campaign.ROOT / "experiments/bill_greedy/runner.py",
                    campaign.ROOT / "experiments/bill_greedy/history.py",
                    campaign.ROOT / "experiments/model_optimization_v2/learners.py"]
    provenance = output / "provenance"
    provenance.mkdir(parents=True, exist_ok=True)
    shutil.copy2(campaign.PACKAGE / "r26_word_tfidf_512.py",
                 provenance / "r26_word_tfidf_512_source.py")
    shutil.copy2(campaign.PROTOCOL_PATH, provenance / "r26_development_campaign_protocol.json")
    manifest = {"campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
                "stage": "development_confirmation", "source_identity": source,
                "screen_run_id": screen["run_id"], "screen_result_sha256": _sha256(screen_path),
                "screen_paired_predictions_sha256": screen_manifest["paired_predictions_sha256"],
                "parent_development_run_id": r22_dev["run_id"],
                "parent_development_result_sha256": _sha256(r22_dev_path),
                "parent_development_predictions_sha256": r22_dev_manifest["paired_predictions_sha256"],
                "code_sha256": {str(path.relative_to(campaign.ROOT)).replace("\\", "/"): _sha256(path)
                                for path in source_files},
                "data_sha256": source["data_sha256"], "fit_starts": fit_start_count,
                "fit_details": fit_details, "component_manifest": component_manifest,
                "paired_predictions_sha256": _sha256(paired_path),
                "result_sha256": _sha256(result_path),
                "reserved_months_labels_opened": False, "hidden_labels_opened": False}
    _write_json(output / "run_manifest.json", manifest)
    monthly = scored["monthly"]
    lines = ["# R26：off-market 八个月开发确认", "",
             f"运行：`{run_id}`；筛选运行：`{screen['run_id']}`；配对父版：R22 `{r22_dev['run_id']}`。",
             f"固定 512 词项 TF-IDF；完成 {fit_start_count} fits。", "",
             "| 父 MdAPE | R26 MdAPE | 相对改善 | MAPE变化 | 改善月份 | 最差单月退化 | Bootstrap 95%上界 | 开发门槛 | 目标 |",
             "|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    lines.append(
        f"| {scored['parent']['mdape_percent']:.5f}% | {scored['candidate']['mdape_percent']:.5f}% | "
        f"{scored['relative_mdape_improvement_percent']:+.3f}% | "
        f"{scored['mape_relative_change_percent']:+.3f}% | {scored['months_improved']}/8 | "
        f"{scored['worst_month_regression_percent']:.3f}% | "
        f"{scored['bootstrap_mdape_difference_percent']['upper95']:+.5f} pp | "
        f"{'通过' if scored['qualifies_development'] else '未通过'} | "
        f"{'达到' if scored['target_met'] else '未达到'} <{target:.2f}% |"
    )
    lines.extend(["", "## 逐月改善", "", "| 月份 | R22 MdAPE | R26 MdAPE | 相对改善 |",
                  "|---|---:|---:|---:|"])
    for month, values in monthly.items():
        lines.append(f"| {month} | {values['parent']['mdape_percent']:.5f}% | "
                     f"{values['candidate']['mdape_percent']:.5f}% | "
                     f"{values['relative_mdape_improvement_percent']:+.3f}% |")
    lines.extend(["", "保留 2025-05/08 与隐藏标签未读取；若开发门槛通过，仅允许进入固定起点确认。", ""])
    (output / "research_report.md").write_text("\n".join(lines), encoding="utf-8")
    campaign._append({"event": "card_complete", "card": CARD,
                      "stage": "development_confirmation", "run_id": run_id,
                      "fit_starts": fit_start_count,
                      "result_path": str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
                      "qualifies_by_track": {"off_market": scored["qualifies_development"]},
                      "targets_met_by_track": {"off_market": scored["target_met"]},
                      "reserved_months_labels_opened": False, "hidden_labels_opened": False})
    next_action = ("register R26 fixed-origin confirmation for off-market"
                   if scored["qualifies_development"] else
                   "R26 failed eight-month development gate; keep R22 and register a distinct method")
    campaign._write_progress("r26_development_confirmation_complete", next_action=next_action,
                             r26_development_result=str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
                             latest_fit_count=fit_start_count)
    return result


def run_r26_screen() -> dict:
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
        raise RuntimeError("R26 must be registered in campaign_protocol.json before fitting")
    r22_result, r22_result_path, r22_manifest = r22._verified_screen_result(campaign)
    if not all(r22_result["tracks"][track]["qualifies_screen"] for track in TRACKS):
        raise RuntimeError("R26 requires both tracks to have passed the verified R22 screen")
    r22_rows_path = r22_result_path.parent / "paired_predictions.csv"
    r22_rows = pd.read_csv(r22_rows_path, dtype={"ml_num": str})
    source = budget._source_identity()
    run_id = "bill_target74_r26_word_tfidf_512_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = campaign.CAMPAIGN_ROOT / "batches" / run_id
    output.mkdir(parents=True, exist_ok=False)
    campaign._append({
        "event": "card_start", "card": CARD, "run_id": run_id,
        "screen_months": list(SCREEN_MONTHS), "tracks": list(TRACKS),
        "parent_candidate": r22_result["run_id"], "model_family": "lightgbm",
        "text_features": {"max_features": MAX_FEATURES, "min_df": 25, "max_df": 0.95,
                           "ngram_range": [1, 2], "stop_words": "english",
                           "vocabulary_policy": "training-fold-only"},
        "expected_fit_starts": len(screen_fit_plan()),
        "data_sha256": source["data_sha256"],
        "r22_parent_result_sha256": _sha256(r22_result_path),
        "r22_parent_predictions_sha256": r22_manifest["paired_predictions_sha256"],
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    fit_details, component_manifest = [], []
    rows_by_track = {track: [] for track in TRACKS}
    old_fit_record, old_hgbr = greedy._fit_record, greedy._frozen_hgbr_predictions
    fit_context: dict[str, str] = {}
    fit_start_count = 0

    def record_fit(_original_fit_id: str, fit_fn):
        nonlocal fit_start_count
        if not fit_context:
            raise RuntimeError("R26 fit context is not initialized")
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
                  "tfidf_feature_count": sum(name.startswith("remarks_tfidf_")
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
                raise RuntimeError(f"R26 training data include sales at or after cutoff: {month}")
            for track in TRACKS:
                homes = hide_answers(heldout, track)
                reference = shared.r22_reference_frame(r22_rows, truth, month, track)
                if track == "off_market":
                    specialist = greedy._specialist_mask(train, "Detached")
                    eval_specialist = greedy._specialist_mask(homes, "Detached")
                    fit_jobs = [("base", train, homes),
                                ("expert", train.loc[specialist].copy(),
                                 homes.loc[eval_specialist].copy())]
                    fitted_components = {}
                    for component, fit_rows, eval_rows in fit_jobs:
                        transformer, restore = _install_wide_word_features(fit_rows, eval_rows, track)
                        fit_context.update(track=track, month=month, component=component)
                        try:
                            fitted = greedy._fit_history_candidate(
                                month, track, train, eval_rows, base_variant="G06_lot_garage",
                                variant=VARIANT, fit_tag=f"R26-{component}-{month}", fit_rows=fit_rows,
                            )
                        finally:
                            restore()
                        fitted_components[component] = fitted
                        component_manifest.append({
                            "track": track, "month": month, "component": component,
                            "model_family": fitted["model_family"], "model_config": fitted["model_config"],
                            "target": fitted["target"], "training_rows": fitted["training_rows"],
                            "evaluation_rows": fitted["evaluation_rows"],
                            "tfidf_feature_count": sum(name.startswith("remarks_tfidf_")
                                                        for name in fitted["feature_columns"]),
                            "vocabulary_terms": transformer.vectorizer.get_feature_names_out().tolist(),
                            "prediction_sha256": fitted["prediction_sha256"],
                        })
                    base_prediction = np.round(fitted_components["base"]["predictions"])
                    expert_prediction = np.round(fitted_components["expert"]["predictions"])
                    candidate_prediction = greedy._blend_specialist_predictions(
                        base_prediction, expert_prediction, homes, "Detached", 0.75)
                else:
                    transformer, restore = _install_wide_word_features(train, homes, track)
                    fit_context.update(track=track, month=month, component="core")
                    try:
                        core = greedy._fit_history_candidate(
                            month, track, train, homes, variant=VARIANT,
                            fit_tag=f"R26-core-{month}", fit_rows=train,
                        )
                    finally:
                        restore()
                    frozen = greedy._frozen_hgbr_predictions(month, track, train, homes)
                    candidate_prediction = blend_prices(core["predictions"], frozen, challenger_weight=0.75)
                    component_manifest.append({
                        "track": track, "month": month, "component": "core",
                        "model_family": core["model_family"], "model_config": core["model_config"],
                        "target": core["target"], "training_rows": core["training_rows"],
                        "evaluation_rows": core["evaluation_rows"],
                        "tfidf_feature_count": sum(name.startswith("remarks_tfidf_")
                                                    for name in core["feature_columns"]),
                        "vocabulary_terms": transformer.vectorizer.get_feature_names_out().tolist(),
                        "prediction_sha256": core["prediction_sha256"],
                    })

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
        failure = {"run_id": run_id, "card": CARD,
                   "error": f"{type(exc).__name__}: {exc}",
                   "fit_starts": fit_start_count, "fit_details": fit_details,
                   "completed_components": [f"{r['track']}/{r['month']}/{r['component']}"
                                             for r in component_manifest],
                   "reserved_months_labels_opened": False, "hidden_labels_opened": False}
        _write_json(output / "failure_checkpoint.json", failure)
        campaign._append({"event": "card_failed", "card": CARD, "run_id": run_id,
                          "fit_starts": fit_start_count, "error": failure["error"]})
        campaign._write_progress("r26_word_tfidf_512_failed",
                                 next_action="inspect R26 checkpoint and ledger before any retry",
                                 r26_failure=str((output / "failure_checkpoint.json").relative_to(campaign.ROOT)).replace("\\", "/"))
        raise
    finally:
        greedy._fit_record, greedy._frozen_hgbr_predictions = old_fit_record, old_hgbr

    if fit_start_count != len(screen_fit_plan()):
        raise RuntimeError(f"R26 expected {len(screen_fit_plan())} fit starts; observed {fit_start_count}")
    paired = pd.concat([frame for groups in rows_by_track.values() for frame in groups], ignore_index=True)
    paired_path = output / "paired_predictions.csv"
    paired.to_csv(paired_path, index=False)
    tracks = {track: shared._track_result(pd.concat(rows, ignore_index=True))
              for track, rows in rows_by_track.items()}
    result = {"campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
              "card": CARD, "parent_candidate": r22_result["run_id"],
              "screen_months": list(SCREEN_MONTHS), "fit_starts": fit_start_count,
              "tracks": tracks, "fit_details": fit_details,
              "text_features": {"max_features": MAX_FEATURES, "min_df": 25,
                                 "max_df": 0.95, "ngram_range": [1, 2]},
              "component_manifest": component_manifest,
              "reserved_months_labels_opened": False, "hidden_labels_opened": False,
              "data_sha256": source["data_sha256"]}
    result_path = output / "r26_screen_results.json"
    _write_json(result_path, result)
    source_files = [campaign.PROTOCOL_PATH, campaign.PACKAGE / "r26_word_tfidf_512.py",
                    campaign.PACKAGE / "r22_remarks_tfidf.py",
                    campaign.PACKAGE / "r23_word_char_tfidf.py",
                    campaign.PACKAGE / "r03_catboost.py", campaign.PACKAGE / "r04_ablation.py",
                    campaign.PACKAGE / "runner.py", campaign.ROOT / "model_candidate_v2/features.py",
                    campaign.ROOT / "experiments/bill_greedy/runner.py",
                    campaign.ROOT / "experiments/bill_greedy/history.py",
                    campaign.ROOT / "experiments/model_optimization_v2/learners.py"]
    provenance = output / "provenance"
    provenance.mkdir(exist_ok=True)
    shutil.copy2(campaign.PACKAGE / "r26_word_tfidf_512.py",
                 provenance / "r26_word_tfidf_512_source.py")
    shutil.copy2(campaign.PROTOCOL_PATH, provenance / "r26_preregistered_campaign_protocol.json")
    manifest = {"campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
                "source_identity": source,
                "code_sha256": {str(path.relative_to(campaign.ROOT)).replace("\\", "/"): _sha256(path)
                                for path in source_files},
                "data_sha256": source["data_sha256"], "fit_starts": fit_start_count,
                "fit_details": fit_details, "component_manifest": component_manifest,
                "paired_predictions_sha256": _sha256(paired_path),
                "result_sha256": _sha256(result_path),
                "parent_run_id": r22_result["run_id"],
                "parent_result_sha256": _sha256(r22_result_path),
                "parent_predictions_sha256": r22_manifest["paired_predictions_sha256"],
                "reserved_months_labels_opened": False, "hidden_labels_opened": False}
    _write_json(output / "run_manifest.json", manifest)
    (output / "research_report.md").write_text(_report(result), encoding="utf-8")
    qualifies = {track: score["qualifies_screen"] for track, score in tracks.items()}
    campaign._append({"event": "card_complete", "card": CARD, "run_id": run_id,
                      "fit_starts": fit_start_count,
                      "result_path": str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
                      "qualifies_by_track": qualifies,
                      "reserved_months_labels_opened": False, "hidden_labels_opened": False})
    next_action = ("prepare R26 development confirmation for screen-qualified tracks"
                   if any(qualifies.values()) else
                   "R26 wider word vocabulary failed the incremental screen; keep R22 and register a new distinct method")
    campaign._write_progress("r26_word_tfidf_512_screen_complete", next_action=next_action,
                             r26_result=str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
                             latest_fit_count=fit_start_count)
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run R26 screen or its screen-qualified development confirmation")
    parser.add_argument("--development", action="store_true",
                        help="run the frozen eight-month development confirmation for off-market only")
    parser.add_argument("--screen-run-id", default=None,
                        help="require this saved R26 screen run ID for development confirmation")
    args = parser.parse_args()
    result = (run_r26_development_confirmation(args.screen_run_id) if args.development
              else run_r26_screen())
    print(json.dumps({"run_id": result["run_id"], "fit_starts": result["fit_starts"],
                      "stage": result.get("stage", "screen"),
                      "tracks": {track: {"parent_mdape_percent": score["parent"]["mdape_percent"],
                                         "candidate_mdape_percent": score["candidate"]["mdape_percent"],
                                         "relative_mdape_improvement_percent": score[
                                             "relative_mdape_improvement_percent"],
                                         "mape_relative_change_percent": score[
                                             "mape_relative_change_percent"],
                                         "qualifies": score.get("qualifies_development",
                                                                 score.get("qualifies_screen")),
                                         "target_met": score.get("target_met"),
                                         "monthly": {month: values["relative_mdape_improvement_percent"]
                                                     for month, values in score["monthly"].items()}}
                                 for track, score in result["tracks"].items()}},
                     indent=2))


if __name__ == "__main__":
    main()
