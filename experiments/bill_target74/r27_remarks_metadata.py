"""R27: retain R22 train-fold word TF-IDF and add fixed row-local remark metadata."""
from __future__ import annotations

import json
import re
import shutil
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from experiments.bill_target74.r19_expert_weight import _sha256, _write_json


CARD = "R27_remarks_metadata"
VARIANT = CARD
SCREEN_MONTHS = ("2023-03", "2024-06")
TRACKS = ("off_market", "list_aware")
METADATA_FEATURES = (
    "remarks_log_char_count",
    "remarks_log_word_count",
    "remarks_unique_word_ratio",
    "remarks_mean_word_length",
    "remarks_log_numeric_token_count",
    "remarks_uppercase_ratio",
    "remarks_log_emphasis_count",
    "remarks_empty",
)


def screen_fit_plan() -> list[tuple[str, str, str]]:
    return [
        (month, track, component)
        for month in SCREEN_MONTHS
        for track, components in (("off_market", ("base", "expert")),
                                  ("list_aware", ("core",)))
        for component in components
    ]


def _fit_id(run_id: str, stage: str, track: str, month: str, component: str) -> str:
    return f"bill_target74:{CARD}:{stage}:{track}:{month}:{component}:{run_id}"


def remark_metadata(remarks, *, index=None) -> pd.DataFrame:
    """Create deterministic row-local statistics; no labels or fitted corpus state."""
    from experiments.bill_target74.r22_remarks_tfidf import FoldTfidf

    texts = FoldTfidf._texts(remarks)
    alpha_tokens = texts.map(lambda value: re.findall(r"(?u)\b[a-zA-Z]{2,}\b", value.lower()))
    word_counts = alpha_tokens.map(len).to_numpy(dtype=np.float64)
    char_counts = texts.str.len().to_numpy(dtype=np.float64)
    unique_ratios = np.fromiter(
        (len(set(tokens)) / len(tokens) if tokens else 0.0 for tokens in alpha_tokens),
        dtype=np.float64,
        count=len(texts),
    )
    mean_word_lengths = np.fromiter(
        (sum(map(len, tokens)) / len(tokens) if tokens else 0.0 for tokens in alpha_tokens),
        dtype=np.float64,
        count=len(texts),
    )
    numeric_counts = texts.str.count(r"\d+").to_numpy(dtype=np.float64)
    alphabetic_counts = texts.str.count(r"[A-Za-z]").to_numpy(dtype=np.float64)
    uppercase_counts = texts.str.count(r"[A-Z]").to_numpy(dtype=np.float64)
    emphasis_counts = texts.str.count(r"[!$%#?]").to_numpy(dtype=np.float64)
    values = np.column_stack((
        np.log1p(char_counts),
        np.log1p(word_counts),
        unique_ratios,
        mean_word_lengths,
        np.log1p(numeric_counts),
        np.divide(uppercase_counts, alphabetic_counts,
                  out=np.zeros_like(uppercase_counts), where=alphabetic_counts > 0),
        np.log1p(emphasis_counts),
        (word_counts == 0).astype(np.float64),
    ))
    if values.shape != (len(texts), len(METADATA_FEATURES)) or not np.isfinite(values).all():
        raise ValueError("remark metadata must be a finite row-aligned feature matrix")
    return pd.DataFrame(values, index=index, columns=METADATA_FEATURES)


def _install_r27_features(fit_rows: pd.DataFrame, eval_rows: pd.DataFrame, track: str):
    """Patch one frozen R22 feature builder for a single fold, then restore it."""
    from model_candidate_v2 import features as frozen_features
    from experiments.bill_target74.r22_remarks_tfidf import FoldTfidf

    if track not in TRACKS:
        raise ValueError(f"unknown track: {track!r}")
    if "remarks" not in fit_rows or "remarks" not in eval_rows:
        raise ValueError("training and evaluation rows must both contain remarks")
    attribute = "off_market_features" if track == "off_market" else "list_aware_features"
    original = getattr(frozen_features, attribute)
    transformer = FoldTfidf()
    fitted_training_vocabulary = False

    def wrapped(homes):
        nonlocal fitted_training_vocabulary
        base = original(homes)
        if homes is fit_rows:
            if fitted_training_vocabulary:
                text = transformer.transform(homes.remarks, index=base.index)
            else:
                text = transformer.fit_transform(homes.remarks, index=base.index)
                fitted_training_vocabulary = True
        elif homes is eval_rows:
            if not fitted_training_vocabulary:
                raise RuntimeError("evaluation remarks requested before training-fold TF-IDF fit")
            text = transformer.transform(homes.remarks, index=base.index)
        else:
            raise RuntimeError("unexpected rows passed to the R27 fold-scoped feature builder")
        metadata = remark_metadata(homes.remarks, index=base.index)
        if set(metadata.columns) & set(base.columns) or set(metadata.columns) & set(text.columns):
            raise RuntimeError("R27 remark feature names collide with an existing feature")
        return pd.concat([base, text, metadata], axis=1)

    setattr(frozen_features, attribute, wrapped)

    def restore() -> None:
        setattr(frozen_features, attribute, original)

    return transformer, restore


def _report(result: dict) -> str:
    lines = [
        "# R27：在 R22 词级 TF-IDF 上追加逐房源描述统计", "",
        f"Campaign：`{result['campaign_id']}`；run：`{result['run_id']}`。",
        f"增量父候选：哈希验证的 R22 run `{result['parent_candidate']}`。",
        "保留 R22 的 256 项训练折词级 TF-IDF、LightGBM、结构/历史特征、目标、路由和融合；只添加 8 个固定、逐房源的 remarks 统计。",
        "统计量只从本行文本计算，不用标签、验证集语料统计或额外数据。", "",
        "## 双月增量筛选", "",
        "| 赛道 | R22 MdAPE | R27 MdAPE | 相对改善 | MAPE变化 | 两月均改善 | 通过 |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for track, score in result["tracks"].items():
        both_months_improved = all(
            values["relative_mdape_improvement_percent"] > 0.0
            for values in score["monthly"].values()
        )
        lines.append(
            f"| {track} | {score['parent']['mdape_percent']:.5f}% | "
            f"{score['candidate']['mdape_percent']:.5f}% | "
            f"{score['relative_mdape_improvement_percent']:+.3f}% | "
            f"{score['mape_relative_change_percent']:+.3f}% | "
            f"{'是' if both_months_improved else '否'} | "
            f"{'是' if score['qualifies_screen'] else '否'} |"
        )
    lines.extend(["", "## 各筛选月", "",
                  "| 赛道 | 月份 | R22 MdAPE | R27 MdAPE | 相对改善 |",
                  "|---|---|---:|---:|---:|"])
    for track, score in result["tracks"].items():
        for month, values in score["monthly"].items():
            lines.append(
                f"| {track} | {month} | {values['parent']['mdape_percent']:.5f}% | "
                f"{values['candidate']['mdape_percent']:.5f}% | "
                f"{values['relative_mdape_improvement_percent']:+.3f}% |"
            )
    lines.extend(["", "保留月份和隐藏标签未读取。只有通过筛选的赛道才进入八个月开发确认。", ""])
    return "\n".join(lines)


def _source_files(campaign) -> list:
    return [
        campaign.PROTOCOL_PATH,
        campaign.PACKAGE / "r27_remarks_metadata.py",
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


def run_r27_screen() -> dict:
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
        raise RuntimeError("R27 must be preregistered before fitting")
    r22_result, r22_result_path, r22_manifest = r22._verified_screen_result(campaign)
    if not all(r22_result["tracks"][track]["qualifies_screen"] for track in TRACKS):
        raise RuntimeError("R27 requires the verified R22 screen to qualify both tracks")
    r22_rows = pd.read_csv(r22_result_path.parent / "paired_predictions.csv", dtype={"ml_num": str})
    source = budget._source_identity()
    run_id = "bill_target74_r27_remarks_metadata_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = campaign.CAMPAIGN_ROOT / "batches" / run_id
    output.mkdir(parents=True, exist_ok=False)
    campaign._append({
        "event": "card_start", "card": CARD, "run_id": run_id,
        "screen_months": list(SCREEN_MONTHS), "tracks": list(TRACKS),
        "parent_candidate": r22_result["run_id"], "model_family": "lightgbm",
        "tfidf_features": {"max_features": 256, "min_df": 25, "max_df": 0.95,
                            "ngram_range": [1, 2], "stop_words": "english",
                            "vocabulary_policy": "training-fold-only"},
        "metadata_features": list(METADATA_FEATURES),
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
            raise RuntimeError("R27 fit context is not initialized")
        track, month, component = (fit_context[key] for key in ("track", "month", "component"))
        fit_id = _fit_id(run_id, "screen", track, month, component)
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
        seconds = time.perf_counter() - started
        detail = {"fit_id": fit_id, "track": track, "month": month,
                  "component": component, "seconds": seconds,
                  "model_family": fitted["model_family"], "model_config": fitted["model_config"],
                  "target": fitted["target"], "actual_objective": fitted.get("actual_objective"),
                  "training_rows": fitted["training_rows"], "evaluation_rows": fitted["evaluation_rows"],
                  "tfidf_feature_count": sum(name.startswith("remarks_tfidf_")
                                              for name in fitted["feature_columns"]),
                  "metadata_feature_count": sum(name in METADATA_FEATURES
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
                raise RuntimeError(f"R27 training rows include sales at or after cutoff {month}")
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
                        transformer, restore = _install_r27_features(fit_rows, eval_rows, track)
                        fit_context.update(track=track, month=month, component=component)
                        try:
                            fitted = greedy._fit_history_candidate(
                                month, track, train, eval_rows, base_variant="G06_lot_garage",
                                variant=VARIANT, fit_tag=f"R27-{component}-{month}", fit_rows=fit_rows)
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
                            "metadata_feature_names": [name for name in METADATA_FEATURES
                                                       if name in fitted["feature_columns"]],
                            "prediction_sha256": fitted["prediction_sha256"],
                        })
                    candidate_prediction = greedy._blend_specialist_predictions(
                        np.round(fitted_components["base"]["predictions"]),
                        np.round(fitted_components["expert"]["predictions"]),
                        homes, "Detached", 0.75)
                else:
                    transformer, restore = _install_r27_features(train, homes, track)
                    fit_context.update(track=track, month=month, component="core")
                    try:
                        core = greedy._fit_history_candidate(
                            month, track, train, homes, variant=VARIANT,
                            fit_tag=f"R27-core-{month}", fit_rows=train)
                    finally:
                        restore()
                    frozen = greedy._frozen_hgbr_predictions(month, track, train, homes)
                    candidate_prediction = blend_prices(core["predictions"], frozen,
                                                        challenger_weight=0.75)
                    component_manifest.append({
                        "track": track, "month": month, "component": "core",
                        "model_family": core["model_family"], "model_config": core["model_config"],
                        "target": core["target"], "training_rows": core["training_rows"],
                        "evaluation_rows": core["evaluation_rows"],
                        "tfidf_feature_count": sum(name.startswith("remarks_tfidf_")
                                                    for name in core["feature_columns"]),
                        "metadata_feature_names": [name for name in METADATA_FEATURES
                                                   if name in core["feature_columns"]],
                        "prediction_sha256": core["prediction_sha256"],
                    })
                candidate = pd.DataFrame({"ml_num": homes.ml_num.astype(str).to_numpy(),
                                          "sold_price": truth.sold_price.to_numpy(float),
                                          "predicted_price": np.asarray(candidate_prediction, dtype=float)})
                parent = reference[["ml_num", "sold_price", "predicted_price"]].copy()
                paired = helpers._candidate_frame(month, track, truth, parent, candidate)
                rows_by_track[track].append(paired)
                campaign._append({"event": "screen_fold_complete", "card": CARD,
                                  "run_id": run_id, "track": track, "month": month,
                                  "reference_run_id": r22_result["run_id"],
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
        campaign._write_progress("r27_screen_failed",
                                 next_action="audit R27 failure checkpoint and ledger before any retry",
                                 r27_failure=str((output / "failure_checkpoint.json").relative_to(
                                     campaign.ROOT)).replace("\\", "/"))
        raise
    finally:
        greedy._fit_record, greedy._frozen_hgbr_predictions = old_fit_record, old_hgbr

    if fit_start_count != len(screen_fit_plan()):
        raise RuntimeError(f"R27 expected {len(screen_fit_plan())} fits; observed {fit_start_count}")
    paired = pd.concat([frame for groups in rows_by_track.values() for frame in groups], ignore_index=True)
    paired_path = output / "paired_predictions.csv"
    paired.to_csv(paired_path, index=False)
    tracks = {track: shared._track_result(pd.concat(rows, ignore_index=True))
              for track, rows in rows_by_track.items()}
    result = {"campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
              "card": CARD, "parent_candidate": r22_result["run_id"],
              "screen_months": list(SCREEN_MONTHS), "fit_starts": fit_start_count,
              "tracks": tracks, "fit_details": fit_details,
              "metadata_features": list(METADATA_FEATURES),
              "tfidf_features": {"max_features": 256, "min_df": 25, "max_df": 0.95,
                                  "ngram_range": [1, 2]},
              "component_manifest": component_manifest,
              "reserved_months_labels_opened": False, "hidden_labels_opened": False,
              "data_sha256": source["data_sha256"]}
    result_path = output / "r27_screen_results.json"
    _write_json(result_path, result)
    provenance = output / "provenance"
    provenance.mkdir(exist_ok=True)
    shutil.copy2(campaign.PACKAGE / "r27_remarks_metadata.py",
                 provenance / "r27_remarks_metadata_source.py")
    shutil.copy2(campaign.PROTOCOL_PATH, provenance / "r27_preregistered_campaign_protocol.json")
    manifest = {"campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
                "source_identity": source,
                "code_sha256": {str(path.relative_to(campaign.ROOT)).replace("\\", "/"): _sha256(path)
                                for path in _source_files(campaign)},
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
    next_action = ("register R27 eight-month development confirmation for qualified tracks"
                   if any(qualifies.values()) else
                   "R27 remark metadata failed screen; preserve R22 and register a distinct method")
    campaign._write_progress("r27_screen_complete", next_action=next_action,
                             r27_result=str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
                             latest_fit_count=fit_start_count)
    return result


def recover_r27_screen_finalization(run_id: str) -> dict:
    """Finalize scored screen artifacts after a report-only failure, without refitting."""
    from experiments.bill_target74 import runner as campaign

    output = campaign.CAMPAIGN_ROOT / "batches" / run_id
    result_path = output / "r27_screen_results.json"
    manifest_path = output / "run_manifest.json"
    paired_path = output / "paired_predictions.csv"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (result.get("run_id") != run_id or result.get("card") != CARD
            or manifest.get("run_id") != run_id
            or manifest.get("result_sha256") != _sha256(result_path)
            or manifest.get("paired_predictions_sha256") != _sha256(paired_path)
            or manifest.get("fit_starts") != result.get("fit_starts")):
        raise RuntimeError("saved R27 screen artifacts fail manifest verification")
    source_copy = output / "provenance" / "r27_remarks_metadata_source.py"
    protocol_copy = output / "provenance" / "r27_preregistered_campaign_protocol.json"
    hashes = manifest.get("code_sha256", {})
    if (_sha256(source_copy) != hashes.get("experiments/bill_target74/r27_remarks_metadata.py")
            or _sha256(protocol_copy) != hashes.get("experiments/bill_target74/campaign_protocol.json")):
        raise RuntimeError("saved R27 source or protocol snapshot fails manifest verification")
    parent_dir = campaign.CAMPAIGN_ROOT / "batches" / result["parent_candidate"]
    parent_result = parent_dir / "r22_screen_results.json"
    parent_manifest = json.loads((parent_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("parent_result_sha256") != _sha256(parent_result)
            or manifest.get("parent_predictions_sha256") != parent_manifest.get("paired_predictions_sha256")):
        raise RuntimeError("saved R27 parent artifacts fail R22 hash verification")
    ledger = [json.loads(line) for line in campaign.LEDGER.read_text(encoding="utf-8").splitlines() if line]
    run_events = [row for row in ledger if row.get("run_id") == run_id]
    starts = [row for row in run_events if row.get("event") == "fit_start"]
    completions = [row for row in run_events if row.get("event") == "fit_complete"]
    if (len(starts) != result.get("fit_starts") or len(completions) != len(starts)
            or any(row.get("status") != "complete" for row in completions)):
        raise RuntimeError("R27 ledger does not show a successful completion for every fit")
    if not any(row.get("event") == "finalization_failure" for row in run_events):
        campaign._append({"event": "finalization_failure", "card": CARD, "run_id": run_id,
                          "stage": "report", "fit_starts": len(starts),
                          "error": "report expected an unregistered both_screen_months_improved field after fits and scores were saved"})
    (output / "research_report.md").write_text(_report(result), encoding="utf-8")
    qualifies = {track: score["qualifies_screen"] for track, score in result["tracks"].items()}
    if not any(row.get("event") == "card_complete" for row in run_events):
        campaign._append({"event": "card_complete", "card": CARD, "run_id": run_id,
                          "fit_starts": len(starts),
                          "result_path": str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
                          "qualifies_by_track": qualifies, "recovered_without_refit": True,
                          "reserved_months_labels_opened": False, "hidden_labels_opened": False})
    if not any(row.get("event") == "finalization_recovered" for row in run_events):
        campaign._append({"event": "finalization_recovered", "card": CARD, "run_id": run_id,
                          "stage": "report", "recovered_without_refit": True,
                          "result_sha256": _sha256(result_path),
                          "paired_predictions_sha256": _sha256(paired_path)})
    next_action = ("register R27 eight-month development confirmation for qualified tracks"
                   if any(qualifies.values()) else
                   "R27 remark metadata failed screen; preserve R22 and register a distinct method")
    campaign._write_progress("r27_screen_complete", next_action=next_action,
                             r27_result=str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
                             latest_fit_count=len(starts))
    return result


def _verified_r27_screen(campaign, expected_run_id: str | None = None):
    progress = json.loads(campaign.STATE.read_text(encoding="utf-8"))
    relative = progress.get("r27_result")
    if not relative:
        raise RuntimeError("campaign progress does not identify a completed R27 screen")
    path = campaign.ROOT / relative
    result = json.loads(path.read_text(encoding="utf-8"))
    manifest = json.loads((path.parent / "run_manifest.json").read_text(encoding="utf-8"))
    paired_path = path.parent / "paired_predictions.csv"
    if expected_run_id and result.get("run_id") != expected_run_id:
        raise RuntimeError("saved R27 screen run ID differs from the requested run")
    if (result.get("card") != CARD or manifest.get("run_id") != result.get("run_id")
            or manifest.get("result_sha256") != _sha256(path)
            or manifest.get("paired_predictions_sha256") != _sha256(paired_path)
            or manifest.get("fit_starts") != result.get("fit_starts")):
        raise RuntimeError("R27 screen artifacts do not match the manifest")
    source_copy = path.parent / "provenance" / "r27_remarks_metadata_source.py"
    protocol_copy = path.parent / "provenance" / "r27_preregistered_campaign_protocol.json"
    expected_code = manifest.get("code_sha256", {})
    if (_sha256(source_copy) != expected_code.get("experiments/bill_target74/r27_remarks_metadata.py")
            or _sha256(protocol_copy) != expected_code.get("experiments/bill_target74/campaign_protocol.json")):
        raise RuntimeError("R27 screen provenance snapshots fail their code-hash checks")
    return result, path, manifest


def _r22_parent(rows: pd.DataFrame, truth: pd.DataFrame, month: str, track: str):
    from experiments.bill_target74.r26_word_tfidf_512 import _r22_development_parent
    return _r22_development_parent(rows, truth, month, track)


def run_r27_development_confirmation(screen_run_id: str | None = None) -> dict:
    """Confirm only screen-qualified tracks over the frozen eight-month window."""
    from data import hide_answers
    from experiments.bill_target74 import r03_catboost as helpers
    from experiments.bill_target74 import r04_ablation as budget
    from experiments.bill_target74 import runner as campaign
    from experiments.bill_target74 import r22_remarks_tfidf as r22
    from experiments.bill_target74 import r23_word_char_tfidf as shared
    from experiments.bill_target74.r03_catboost import _verified_hgbr_cache
    from experiments.bill_greedy import runner as greedy
    from experiments.model_optimization_v2.pipeline import blend_prices

    screen, screen_path, screen_manifest = _verified_r27_screen(campaign, screen_run_id)
    eligible_tracks = tuple(track for track in TRACKS
                            if screen["tracks"][track]["qualifies_screen"] is True)
    if not eligible_tracks:
        raise RuntimeError("R27 has no screen-qualified track to confirm")
    r22_dev, r22_dev_path, r22_dev_manifest = r22._verified_development_result(campaign)
    r22_dev_rows = pd.read_csv(r22_dev_path.parent / "paired_predictions.csv", dtype={"ml_num": str})
    source = budget._source_identity()
    run_id = "bill_target74_r27_remarks_metadata_development_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = campaign.CAMPAIGN_ROOT / "batches" / run_id
    output.mkdir(parents=True, exist_ok=False)
    plan = [(month, track, component)
            for month in r22.DEVELOPMENT_MONTHS for track in eligible_tracks
            for component in (("base", "expert") if track == "off_market" else ("core",))]
    campaign._append({"event": "card_start", "card": CARD,
                      "stage": "development_confirmation", "run_id": run_id,
                      "screen_run_id": screen["run_id"],
                      "parent_development_run_id": r22_dev["run_id"],
                      "tracks": list(eligible_tracks),
                      "development_months": list(r22.DEVELOPMENT_MONTHS),
                      "expected_fit_starts": len(plan), "metadata_features": list(METADATA_FEATURES),
                      "reserved_months_labels_opened": False, "hidden_labels_opened": False})
    fit_details, component_manifest, paired_rows = [], [], []
    old_fit_record, old_hgbr = greedy._fit_record, greedy._frozen_hgbr_predictions
    fit_context: dict[str, str] = {}
    fit_start_count = 0

    def record_fit(_original_fit_id: str, fit_fn):
        nonlocal fit_start_count
        if not fit_context:
            raise RuntimeError("R27 development fit context is not initialized")
        track, month, component = (fit_context[key] for key in ("track", "month", "component"))
        fit_id = _fit_id(run_id, "development_confirmation", track, month, component)
        fit_start_count += 1
        campaign._append({"event": "fit_start", "card": CARD,
                          "stage": "development_confirmation", "run_id": run_id,
                          "fit_id": fit_id, "track": track, "month": month,
                          "component": component, "data_sha256": source["data_sha256"]})
        started = time.perf_counter()
        try:
            fitted = fit_fn()
        except Exception as exc:
            campaign._append({"event": "fit_complete", "card": CARD,
                              "stage": "development_confirmation", "run_id": run_id,
                              "fit_id": fit_id, "track": track, "month": month,
                              "component": component, "status": "failed",
                              "seconds": time.perf_counter() - started,
                              "error": f"{type(exc).__name__}: {exc}"})
            raise
        seconds = time.perf_counter() - started
        saved = r22._save_component_predictions(output, track, month, component,
                                                fit_context["evaluation_ids"], fitted["predictions"])
        detail = {"fit_id": fit_id, "track": track, "month": month,
                  "component": component, "seconds": seconds,
                  "model_family": fitted["model_family"], "model_config": fitted["model_config"],
                  "target": fitted["target"], "actual_objective": fitted.get("actual_objective"),
                  "training_rows": fitted["training_rows"], "evaluation_rows": fitted["evaluation_rows"],
                  "tfidf_feature_count": sum(name.startswith("remarks_tfidf_")
                                              for name in fitted["feature_columns"]),
                  "metadata_feature_count": sum(name in METADATA_FEATURES
                                                 for name in fitted["feature_columns"]),
                  "prediction_sha256": fitted["prediction_sha256"], "saved_predictions": saved}
        fit_details.append(detail)
        campaign._append({"event": "fit_complete", "card": CARD,
                          "stage": "development_confirmation", "run_id": run_id,
                          **detail, "status": "complete"})
        return fitted

    greedy._fit_record = record_fit
    greedy._frozen_hgbr_predictions = lambda month, track, train, homes: _verified_hgbr_cache(
        greedy, month, track, train, homes)
    try:
        for month in r22.DEVELOPMENT_MONTHS:
            _, train, _, truth, heldout = greedy._load_fold(month)
            cutoff = pd.Period(month, freq="M").start_time
            if (pd.to_datetime(train.date_sold, errors="raise") >= cutoff).any():
                raise RuntimeError(f"R27 development training rows violate the cutoff for {month}")
            for track in eligible_tracks:
                homes = hide_answers(heldout, track)
                parent, _ = _r22_parent(r22_dev_rows, truth, month, track)
                fitted_components = {}
                if track == "off_market":
                    specialist = greedy._specialist_mask(train, "Detached")
                    eval_specialist = greedy._specialist_mask(homes, "Detached")
                    fit_jobs = [("base", train, homes),
                                ("expert", train.loc[specialist].copy(),
                                 homes.loc[eval_specialist].copy())]
                else:
                    fit_jobs = [("core", train, homes)]
                for component, fit_rows, eval_rows in fit_jobs:
                    transformer, restore = _install_r27_features(fit_rows, eval_rows, track)
                    fit_context.update(track=track, month=month, component=component,
                                       evaluation_ids=eval_rows.ml_num.astype(str).tolist())
                    try:
                        if component in {"base", "expert"}:
                            fitted = greedy._fit_history_candidate(
                                month, track, train, eval_rows, base_variant="G06_lot_garage",
                                variant=VARIANT, fit_tag=f"R27-development-{component}-{month}",
                                fit_rows=fit_rows)
                        else:
                            fitted = greedy._fit_history_candidate(
                                month, track, train, eval_rows, variant=VARIANT,
                                fit_tag=f"R27-development-core-{month}", fit_rows=fit_rows)
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
                        "metadata_feature_names": [name for name in METADATA_FEATURES
                                                   if name in fitted["feature_columns"]],
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
                paired = helpers._candidate_frame(month, track, truth, parent, candidate)
                paired_rows.append(paired)
                campaign._append({"event": "development_fold_complete", "card": CARD,
                                  "stage": "development_confirmation", "run_id": run_id,
                                  "track": track, "month": month,
                                  "parent_candidate_run_id": r22_dev["run_id"],
                                  "date_cutoff_exclusive": cutoff.isoformat(),
                                  "reserved_months_labels_opened": False,
                                  "hidden_labels_opened": False})
                pd.concat(paired_rows, ignore_index=True).to_csv(output / "partial_paired_predictions.csv",
                                                                 index=False)
    except Exception as exc:
        failure = {"run_id": run_id, "card": CARD,
                   "stage": "development_confirmation", "error": f"{type(exc).__name__}: {exc}",
                   "fit_starts": fit_start_count, "fit_details": fit_details,
                   "completed_components": [f"{r['track']}/{r['month']}/{r['component']}"
                                             for r in component_manifest],
                   "reserved_months_labels_opened": False, "hidden_labels_opened": False}
        _write_json(output / "failure_checkpoint.json", failure)
        campaign._append({"event": "card_failed", "card": CARD,
                          "stage": "development_confirmation", "run_id": run_id,
                          "fit_starts": fit_start_count, "error": failure["error"]})
        campaign._write_progress("r27_development_confirmation_failed",
                                 next_action="audit R27 development checkpoint before any retry",
                                 r27_development_failure=str((output / "failure_checkpoint.json").relative_to(
                                     campaign.ROOT)).replace("\\", "/"))
        raise
    finally:
        greedy._fit_record, greedy._frozen_hgbr_predictions = old_fit_record, old_hgbr

    if fit_start_count != len(plan):
        raise RuntimeError(f"R27 development expected {len(plan)} fits; observed {fit_start_count}")
    paired_frame = pd.concat(paired_rows, ignore_index=True)
    paired_path = output / "paired_predictions.csv"
    paired_frame.to_csv(paired_path, index=False)
    gates = campaign.PROTOCOL["gates"]
    scored = {}
    for track in eligible_tracks:
        target = campaign.PROTOCOL["initial_development_metrics"][track]["target_mdape_strictly_below"]
        scored[track] = r22.score_development_confirmation(
            paired_frame.loc[paired_frame.track.eq(track)].copy(), months=tuple(r22.DEVELOPMENT_MONTHS),
            target_mdape_strictly_below=target, bootstrap_replicates=2000,
            bootstrap_seed=20261024,
            minimum_relative_gain_percent=gates["development_relative_mdape_improvement_percent"],
            minimum_months_improved=gates["development_months_improved_min"],
            maximum_worst_month_regression_percent=gates["development_worst_month_regression_max_percent"],
            maximum_mape_regression_percent=gates["development_mape_regression_max_percent"],
        )
    result = {"campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
              "card": CARD, "stage": "development_confirmation",
              "screen_run_id": screen["run_id"], "parent_development_run_id": r22_dev["run_id"],
              "tracks": scored, "eligible_tracks": list(eligible_tracks),
              "development_months": list(r22.DEVELOPMENT_MONTHS),
              "metadata_features": list(METADATA_FEATURES),
              "fit_starts": fit_start_count, "fit_details": fit_details,
              "component_manifest": component_manifest,
              "reserved_months_labels_opened": False, "hidden_labels_opened": False,
              "data_sha256": source["data_sha256"]}
    result_path = output / "r27_development_results.json"
    _write_json(result_path, result)
    provenance = output / "provenance"
    provenance.mkdir(parents=True, exist_ok=True)
    shutil.copy2(campaign.PACKAGE / "r27_remarks_metadata.py",
                 provenance / "r27_remarks_metadata_source.py")
    shutil.copy2(campaign.PROTOCOL_PATH, provenance / "r27_development_preregistered_campaign_protocol.json")
    manifest = {"campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
                "stage": "development_confirmation", "source_identity": source,
                "screen_run_id": screen["run_id"], "screen_result_sha256": _sha256(screen_path),
                "screen_paired_predictions_sha256": screen_manifest["paired_predictions_sha256"],
                "parent_development_run_id": r22_dev["run_id"],
                "parent_development_result_sha256": _sha256(r22_dev_path),
                "parent_development_predictions_sha256": r22_dev_manifest["paired_predictions_sha256"],
                "code_sha256": {str(path.relative_to(campaign.ROOT)).replace("\\", "/"): _sha256(path)
                                for path in _source_files(campaign)},
                "data_sha256": source["data_sha256"], "fit_starts": fit_start_count,
                "fit_details": fit_details, "component_manifest": component_manifest,
                "paired_predictions_sha256": _sha256(paired_path),
                "result_sha256": _sha256(result_path),
                "reserved_months_labels_opened": False, "hidden_labels_opened": False}
    _write_json(output / "run_manifest.json", manifest)
    lines = ["# R27：八个月开发确认", "",
             f"运行：`{run_id}`；筛选：`{screen['run_id']}`；父候选 R22：`{r22_dev['run_id']}`。",
             f"追加固定逐行文本统计：{', '.join(METADATA_FEATURES)}。", "",
             "| 赛道 | R22 MdAPE | R27 MdAPE | 相对改善 | MAPE变化 | 改善月数 | CI 95%上界 | 开发门槛 | 目标 |",
             "|---|---:|---:|---:|---:|---:|---:|---|---|"]
    for track, score in scored.items():
        target = campaign.PROTOCOL["initial_development_metrics"][track]["target_mdape_strictly_below"]
        lines.append(f"| {track} | {score['parent']['mdape_percent']:.5f}% | "
                     f"{score['candidate']['mdape_percent']:.5f}% | "
                     f"{score['relative_mdape_improvement_percent']:+.3f}% | "
                     f"{score['mape_relative_change_percent']:+.3f}% | "
                     f"{score['months_improved']}/8 | "
                     f"{score['bootstrap_mdape_difference_percent']['upper95']:+.5f} pp | "
                     f"{'通过' if score['qualifies_development'] else '未通过'} | "
                     f"{'达到' if score['target_met'] else '未达到'} <{target:.2f}% |")
    lines.extend(["", "保留 2025-05/08 与隐藏标签未读取。", ""])
    (output / "research_report.md").write_text("\n".join(lines), encoding="utf-8")
    qualifies = {track: score["qualifies_development"] for track, score in scored.items()}
    targets = {track: score["target_met"] for track, score in scored.items()}
    campaign._append({"event": "card_complete", "card": CARD,
                      "stage": "development_confirmation", "run_id": run_id,
                      "fit_starts": fit_start_count,
                      "result_path": str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
                      "qualifies_by_track": qualifies, "targets_met_by_track": targets,
                      "reserved_months_labels_opened": False, "hidden_labels_opened": False})
    campaign._write_progress("r27_development_confirmation_complete",
                             next_action=("register fixed-origin confirmation for R27 development-qualified tracks"
                                          if any(qualifies.values()) else
                                          "R27 failed development gates; keep R22 and register a distinct method"),
                             r27_development_result=str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
                             latest_fit_count=fit_start_count)
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run the preregistered R27 remark-metadata experiment")
    parser.add_argument("--development", action="store_true",
                        help="run eight-month confirmation for only screen-qualified tracks")
    parser.add_argument("--screen-run-id", default=None,
                        help="expected completed screen run ID for development confirmation")
    parser.add_argument("--recover-screen-run-id", default=None,
                        help="recover final report and ledger state from saved screen artifacts without refitting")
    args = parser.parse_args()
    if args.recover_screen_run_id:
        result = recover_r27_screen_finalization(args.recover_screen_run_id)
    elif args.development:
        result = run_r27_development_confirmation(args.screen_run_id)
    else:
        result = run_r27_screen()
    for track, score in result["tracks"].items():
        if args.development:
            print(f"{track}: R22={score['parent']['mdape_percent']:.5f}% "
                  f"R27={score['candidate']['mdape_percent']:.5f}% "
                  f"gain={score['relative_mdape_improvement_percent']:+.3f}% "
                  f"dev_pass={score['qualifies_development']} target_met={score['target_met']}",
                  flush=True)
        else:
            print(f"{track}: R22={score['parent']['mdape_percent']:.5f}% "
                  f"R27={score['candidate']['mdape_percent']:.5f}% "
                  f"gain={score['relative_mdape_improvement_percent']:+.3f}% "
                  f"screen_pass={score['qualifies_screen']}", flush=True)
    print(f"run_id={result['run_id']}", flush=True)


if __name__ == "__main__":
    main()
