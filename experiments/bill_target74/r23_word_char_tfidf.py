"""R23: add train-fold character n-grams to the verified R22 word TF-IDF model."""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from experiments.bill_target74.r19_expert_weight import _sha256, _write_json


CARD = "R23_train_fold_word_char_tfidf_remarks"
VARIANT = "R23_train_fold_word_char_tfidf_remarks"
SCREEN_MONTHS = ("2023-03", "2024-06")
TRACKS = ("off_market", "list_aware")
WORD_MAX_FEATURES = 256
CHAR_MAX_FEATURES = 128
MIN_DF = 25
MAX_DF = 0.95


class FoldWordCharTfidf:
    """Fit independent word and character TF-IDF vocabularies on training remarks."""

    def __init__(self, *, word_max_features: int = WORD_MAX_FEATURES,
                 char_max_features: int = CHAR_MAX_FEATURES,
                 min_df: int | float = MIN_DF, max_df: int | float = MAX_DF):
        common = {"min_df": min_df, "max_df": max_df, "sublinear_tf": True,
                  "dtype": np.float32}
        self.word_vectorizer = TfidfVectorizer(
            **common, max_features=word_max_features, ngram_range=(1, 2),
            stop_words="english", token_pattern=r"(?u)\b[a-zA-Z]{2,}\b",
        )
        self.char_vectorizer = TfidfVectorizer(
            **common, max_features=char_max_features, analyzer="char_wb",
            ngram_range=(3, 5),
        )
        self.columns_: list[str] | None = None

    @staticmethod
    def _texts(values) -> pd.Series:
        texts = pd.Series(values, copy=False).fillna("").astype(str)
        if texts.ndim != 1:
            raise ValueError("remarks must be a one-dimensional sequence")
        return texts

    def _frame(self, word_matrix, char_matrix, *, index) -> pd.DataFrame:
        word_columns = [f"remarks_word_tfidf_{i:03d}" for i in range(word_matrix.shape[1])]
        char_columns = [f"remarks_char_tfidf_{i:03d}" for i in range(char_matrix.shape[1])]
        columns = word_columns + char_columns
        frame = pd.DataFrame(
            np.concatenate([word_matrix.toarray(), char_matrix.toarray()], axis=1),
            index=index, columns=columns,
        )
        if not np.isfinite(frame.to_numpy()).all():
            raise ValueError("word/character TF-IDF features contain non-finite values")
        return frame

    def fit_transform(self, remarks, *, index=None) -> pd.DataFrame:
        texts = self._texts(remarks).tolist()
        word_matrix = self.word_vectorizer.fit_transform(texts)
        char_matrix = self.char_vectorizer.fit_transform(texts)
        frame = self._frame(word_matrix, char_matrix, index=index)
        self.columns_ = frame.columns.tolist()
        return frame

    def transform(self, remarks, *, index=None) -> pd.DataFrame:
        if self.columns_ is None:
            raise RuntimeError("training remarks must fit the TF-IDF vocabularies before transform")
        texts = self._texts(remarks).tolist()
        word_matrix = self.word_vectorizer.transform(texts)
        char_matrix = self.char_vectorizer.transform(texts)
        frame = self._frame(word_matrix, char_matrix, index=index)
        if frame.columns.tolist() != self.columns_:
            raise RuntimeError("evaluation TF-IDF columns differ from the training vocabulary")
        return frame


def screen_fit_plan() -> list[tuple[str, str, str]]:
    return [
        (month, track, component)
        for month in SCREEN_MONTHS
        for track, components in (("off_market", ("base", "expert")),
                                  ("list_aware", ("core",)))
        for component in components
    ]


def r22_reference_frame(r22_rows: pd.DataFrame, truth: pd.DataFrame,
                        month: str, track: str) -> pd.DataFrame:
    """Align verified R22 candidate predictions to the exact R23 evaluation rows."""
    required = {"month", "track", "ml_num", "sold_price", "candidate_prediction"}
    missing = required - set(r22_rows.columns)
    if missing:
        raise ValueError(f"R22 prediction rows are missing columns: {sorted(missing)}")
    selected = r22_rows.loc[r22_rows.month.eq(month) & r22_rows.track.eq(track)].copy()
    if selected.ml_num.astype(str).duplicated().any():
        raise RuntimeError(f"R22 prediction ids are duplicated for {month}/{track}")
    expected_ids = truth.ml_num.astype(str).to_numpy()
    indexed = selected.assign(ml_num=selected.ml_num.astype(str)).set_index("ml_num")
    aligned = indexed.reindex(expected_ids)
    if aligned.candidate_prediction.isna().any() or len(selected) != len(truth):
        raise RuntimeError(f"R22 prediction rows do not cover the R23 fold for {month}/{track}")
    if not np.array_equal(aligned.sold_price.to_numpy(float), truth.sold_price.to_numpy(float)):
        raise RuntimeError(f"R22 labels do not align with R23 truth for {month}/{track}")
    result = pd.DataFrame({
        "ml_num": expected_ids,
        "sold_price": truth.sold_price.to_numpy(float),
        "predicted_price": aligned.candidate_prediction.to_numpy(float),
    })
    if ((result.predicted_price <= 0).any()
            or not np.isfinite(result.predicted_price.to_numpy(float)).all()):
        raise RuntimeError(f"R22 reference predictions are invalid for {month}/{track}")
    if "parent_prediction" in aligned:
        result["initial_parent_prediction"] = aligned.parent_prediction.to_numpy(float)
    return result


def qualifies_incremental_screen(relative_gain: float, monthly_gains: list[float],
                                 mape_change: float) -> bool:
    """Apply the frozen screen gates against R22, the current validated candidate."""
    from experiments.bill_target74 import runner as campaign

    gates = campaign.PROTOCOL["gates"]
    return bool(
        relative_gain >= gates["screen_relative_mdape_improvement_percent"]
        and all(value > 0.0 for value in monthly_gains)
        and mape_change <= gates["screen_mape_regression_max_percent"]
    )


def _install_text_features(fit_rows: pd.DataFrame, eval_rows: pd.DataFrame, track: str):
    from model_candidate_v2 import features as frozen_features

    if "remarks" not in fit_rows or "remarks" not in eval_rows:
        raise ValueError("training and evaluation rows must both contain listing remarks")
    if track not in TRACKS:
        raise ValueError(f"unknown track: {track!r}")
    attribute = "off_market_features" if track == "off_market" else "list_aware_features"
    original = getattr(frozen_features, attribute)
    transformer = FoldWordCharTfidf()
    fit_identity, eval_identity = id(fit_rows), id(eval_rows)

    def wrapped(homes):
        base = original(homes)
        if id(homes) == fit_identity:
            text = transformer.fit_transform(homes.remarks, index=base.index)
        elif id(homes) == eval_identity:
            text = transformer.transform(homes.remarks, index=base.index)
        else:
            raise RuntimeError("unexpected row frame passed to the fold-scoped text feature builder")
        return pd.concat([base, text], axis=1)

    setattr(frozen_features, attribute, wrapped)

    def restore() -> None:
        setattr(frozen_features, attribute, original)

    return transformer, restore


def _fit_id(run_id: str, track: str, month: str, component: str) -> str:
    return f"bill_target74:{CARD}:{track}:{month}:{component}:{run_id}"


def _track_result(rows: pd.DataFrame) -> dict:
    from experiments.bill_target74 import runner as campaign

    parent = campaign._metrics(rows.sold_price.to_numpy(float), rows.parent_prediction.to_numpy(float))
    candidate = campaign._metrics(rows.sold_price.to_numpy(float), rows.candidate_prediction.to_numpy(float))
    relative = campaign._relative_gain(parent["mdape_percent"], candidate["mdape_percent"])
    mape_change = 100.0 * (candidate["mape_percent"] - parent["mape_percent"]) / parent["mape_percent"]
    monthly = {}
    for month in SCREEN_MONTHS:
        part = rows.loc[rows.month.eq(month)]
        pm = campaign._metrics(part.sold_price.to_numpy(float), part.parent_prediction.to_numpy(float))
        cm = campaign._metrics(part.sold_price.to_numpy(float), part.candidate_prediction.to_numpy(float))
        monthly[month] = {
            "parent": pm,
            "candidate": cm,
            "relative_mdape_improvement_percent": campaign._relative_gain(
                pm["mdape_percent"], cm["mdape_percent"]),
        }
    monthly_gains = [monthly[month]["relative_mdape_improvement_percent"] for month in SCREEN_MONTHS]
    initial_parent = None
    if "initial_parent_prediction" in rows:
        initial_metric = campaign._metrics(
            rows.sold_price.to_numpy(float), rows.initial_parent_prediction.to_numpy(float)
        )
        initial_parent = {
            "metrics": initial_metric,
            "relative_mdape_improvement_percent": campaign._relative_gain(
                initial_metric["mdape_percent"], candidate["mdape_percent"]),
        }
    return {
        "parent": parent,
        "candidate": candidate,
        "initial_parent": initial_parent,
        "relative_mdape_improvement_percent": relative,
        "mape_relative_change_percent": mape_change,
        "monthly": monthly,
        "qualifies_screen": qualifies_incremental_screen(relative, monthly_gains, mape_change),
    }


def _report(result: dict) -> str:
    lines = [
        "# R23：在 R22 训练折词级 TF-IDF 上追加字符 n-gram", "",
        f"Campaign：`{result['campaign_id']}`；run：`{result['run_id']}`。",
        f"筛选父候选：经过 manifest 与预测文件哈希验证的 R22 run `{result['parent_candidate']}`。",
        "R23 只增加在各自训练折拟合的最多 128 个 `char_wb` 3–5 gram 特征；原 256 个词级 TF-IDF、表格特征、模型配置、目标和融合权重保持不变。",
        "门槛比较 R23 对 R22 的增量效果，不能把 R22 相对 G19/G01 的既有收益归给 R23。",
        "", "## 增量筛选结果", "",
        "| 赛道 | R22 MdAPE | R23 MdAPE | 相对改善 | MAPE变化 | 两月都改善 | 通过 |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for track, score in result["tracks"].items():
        both = all(row["relative_mdape_improvement_percent"] > 0
                   for row in score["monthly"].values())
        lines.append(
            f"| {track} | {score['parent']['mdape_percent']:.5f}% | "
            f"{score['candidate']['mdape_percent']:.5f}% | "
            f"{score['relative_mdape_improvement_percent']:+.3f}% | "
            f"{score['mape_relative_change_percent']:+.3f}% | "
            f"{'是' if both else '否'} | {'是' if score['qualifies_screen'] else '否'} |"
        )
    lines.extend(["", "保留月份与隐藏标签均未读取。筛选通过只允许进入下一阶段开发确认，不代表达到 `<7% / <4%` 目标。", ""])
    return "\n".join(lines)


def finalize_saved_run(run_id: str) -> dict:
    """Recover post-fit report/progress finalization without starting any new fits."""
    from experiments.bill_target74 import runner as campaign
    from experiments.bill_target74 import r22_remarks_tfidf as r22

    output = campaign.CAMPAIGN_ROOT / "batches" / run_id
    result_path = output / "r23_screen_results.json"
    paired_path = output / "paired_predictions.csv"
    manifest_path = output / "run_manifest.json"
    snapshot_path = output / "provenance" / "r23_word_char_tfidf_source.py"
    for path in (result_path, paired_path, manifest_path, snapshot_path):
        if not path.is_file():
            raise FileNotFoundError(f"R23 recovery artifact is missing: {path}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (result.get("run_id") != run_id or result.get("card") != CARD
            or manifest.get("run_id") != run_id or manifest.get("campaign_id") != campaign.PROTOCOL["campaign_id"]):
        raise RuntimeError("R23 recovery artifacts do not identify the requested run")
    if (manifest.get("result_sha256") != _sha256(result_path)
            or manifest.get("paired_predictions_sha256") != _sha256(paired_path)
            or manifest.get("code_sha256", {}).get("experiments/bill_target74/r23_word_char_tfidf.py")
            != _sha256(snapshot_path)):
        raise RuntimeError("R23 recovery artifact hashes do not match the fit-time manifest")
    if result.get("fit_starts") != manifest.get("fit_starts"):
        raise RuntimeError("R23 fit counts differ between result and manifest")
    r22_result, r22_path, r22_manifest = r22._verified_screen_result(campaign)
    if (result.get("parent_candidate") != r22_result.get("run_id")
            or manifest.get("r22_parent_run_id") != r22_result.get("run_id")
            or manifest.get("r22_parent_result_sha256") != _sha256(r22_path)
            or manifest.get("r22_parent_predictions_sha256") != r22_manifest.get("paired_predictions_sha256")):
        raise RuntimeError("R23 did not use the verified R22 reference recorded at fit time")

    events = [json.loads(line) for line in campaign.LEDGER.read_text(encoding="utf-8").splitlines()]
    run_events = [event for event in events if event.get("run_id") == run_id]
    fit_starts = [event for event in run_events if event.get("event") == "fit_start"]
    fit_completes = [event for event in run_events if event.get("event") == "fit_complete"]
    if (len(fit_starts) != result["fit_starts"]
            or len(fit_completes) != len(fit_starts)
            or any(event.get("status") != "complete" for event in fit_completes)):
        raise RuntimeError("R23 fit ledger does not show every started fit completed")
    completed_ids = {event.get("fit_id") for event in fit_completes}
    if {event.get("fit_id") for event in fit_starts} != completed_ids:
        raise RuntimeError("R23 started and completed fit IDs differ")
    if any(event.get("event") == "card_failed" for event in run_events):
        raise RuntimeError("R23 has a recorded fit-stage failure; refusing report-only recovery")
    if not any(event.get("event") == "card_complete" for event in run_events):
        (output / "research_report.md").write_text(_report(result), encoding="utf-8")
        campaign._append({"event": "card_finalization_recovered", "card": CARD,
                          "run_id": run_id, "fit_starts": len(fit_starts),
                          "reason": "report field mismatch after all fit and score artifacts were saved",
                          "new_fits_started": 0})
        campaign._append({"event": "card_complete", "card": CARD, "run_id": run_id,
                          "fit_starts": len(fit_starts),
                          "result_path": str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
                          "qualifies_by_track": {track: value["qualifies_screen"]
                                                 for track, value in result["tracks"].items()},
                          "reserved_months_labels_opened": False, "hidden_labels_opened": False,
                          "recovered_finalization": True})
        qualifies = {track: value["qualifies_screen"] for track, value in result["tracks"].items()}
        next_action = ("run R23 eight-month development confirmation only for screen-qualified tracks"
                       if any(qualifies.values()) else
                       "R23 character n-grams did not beat validated R22; keep R22 and register a distinct mechanism")
        campaign._write_progress(
            "r23_word_char_tfidf_screen_complete", next_action=next_action,
            r23_result=str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
            latest_fit_count=len(fit_starts),
        )
    return result


def run_r23_screen() -> dict:
    from data import hide_answers
    from experiments.bill_target74 import r03_catboost as helpers
    from experiments.bill_target74 import r04_ablation as budget
    from experiments.bill_target74 import runner as campaign
    from experiments.bill_target74 import r22_remarks_tfidf as r22
    from experiments.bill_target74.r03_catboost import _verified_hgbr_cache
    from experiments.bill_greedy import runner as greedy

    protocol_card = campaign.PROTOCOL["cards"].get(CARD)
    if not protocol_card or protocol_card.get("status") != "registered":
        raise RuntimeError("R23 must be preregistered in campaign_protocol.json before fitting")
    r22_result, r22_result_path, r22_manifest = r22._verified_screen_result(campaign)
    qualified = [track for track in TRACKS
                 if r22_result["tracks"][track]["qualifies_screen"]]
    if len(qualified) != len(TRACKS):
        raise RuntimeError("R23 combined two-track design requires both R22 screen tracks")
    r22_rows_path = r22_result_path.parent / "paired_predictions.csv"
    r22_rows = pd.read_csv(r22_rows_path, dtype={"ml_num": str})
    source = budget._source_identity()
    run_id = "bill_target74_r23_word_char_tfidf_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = campaign.CAMPAIGN_ROOT / "batches" / run_id
    output.mkdir(parents=True, exist_ok=False)
    campaign._append({
        "event": "card_start", "card": CARD, "run_id": run_id,
        "screen_months": list(SCREEN_MONTHS), "tracks": list(TRACKS),
        "parent_candidate": r22_result["run_id"],
        "features": {"word_max_features": WORD_MAX_FEATURES, "char_max_features": CHAR_MAX_FEATURES,
                     "word_ngram_range": [1, 2], "char_ngram_range": [3, 5],
                     "char_analyzer": "char_wb", "min_df": MIN_DF, "max_df": MAX_DF},
        "expected_fit_starts": 6, "reserved_months_labels_opened": False,
        "hidden_labels_opened": False,
    })
    fit_details, component_manifest = [], []
    rows_by_track = {track: [] for track in TRACKS}
    old_fit_record, old_hgbr = greedy._fit_record, greedy._frozen_hgbr_predictions
    fit_context: dict[str, str] = {}
    fit_start_count = 0

    def record_fit(_original_fit_id: str, fit_fn):
        nonlocal fit_start_count
        if not fit_context:
            raise RuntimeError("R23 fit context was not initialized")
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
                  "training_rows": fitted["training_rows"], "evaluation_rows": fitted["evaluation_rows"],
                  "feature_columns": fitted["feature_columns"],
                  "word_feature_count": sum(name.startswith("remarks_word_tfidf_")
                                             for name in fitted["feature_columns"]),
                  "char_feature_count": sum(name.startswith("remarks_char_tfidf_")
                                             for name in fitted["feature_columns"]),
                  "prediction_sha256": fitted["prediction_sha256"]}
        fit_details.append(detail)
        campaign._append({"event": "fit_complete", "card": CARD, "run_id": run_id,
                          "fit_id": fit_id, "track": track, "month": month,
                          "component": component, "status": "complete", "seconds": seconds,
                          "prediction_sha256": fitted["prediction_sha256"],
                          "training_rows": fitted["training_rows"]})
        return fitted

    greedy._fit_record = record_fit
    greedy._frozen_hgbr_predictions = lambda month, track, train, homes: _verified_hgbr_cache(
        greedy, month, track, train, homes)
    try:
        for month in SCREEN_MONTHS:
            _, train, _, truth, heldout = greedy._load_fold(month)
            cutoff = pd.Period(month, freq="M").start_time
            if (pd.to_datetime(train.date_sold, errors="raise") >= cutoff).any():
                raise RuntimeError(f"R23 training rows include sales at or after cutoff: {month}")
            for track in TRACKS:
                homes = hide_answers(heldout, track)
                reference = r22_reference_frame(r22_rows, truth, month, track)
                if track == "off_market":
                    specialist = greedy._specialist_mask(train, "Detached")
                    eval_specialist = greedy._specialist_mask(homes, "Detached")
                    fit_jobs = [
                        ("base", train, homes),
                        ("expert", train.loc[specialist].copy(), homes.loc[eval_specialist].copy()),
                    ]
                    fitted_components = {}
                    for component, fit_rows, eval_rows in fit_jobs:
                        transformer, restore = _install_text_features(fit_rows, eval_rows, track)
                        fit_context.update(track=track, month=month, component=component)
                        try:
                            fitted = greedy._fit_history_candidate(
                                month, track, train, eval_rows, base_variant="G06_lot_garage",
                                variant=VARIANT, fit_tag=f"R23-{component}-{month}", fit_rows=fit_rows,
                            )
                        finally:
                            restore()
                        fitted_components[component] = fitted
                        component_manifest.append({
                            "track": track, "month": month, "component": component,
                            "word_vocabulary": transformer.word_vectorizer.get_feature_names_out().tolist(),
                            "char_vocabulary": transformer.char_vectorizer.get_feature_names_out().tolist(),
                            "prediction_sha256": fitted["prediction_sha256"],
                        })
                    base_prediction = np.round(fitted_components["base"]["predictions"])
                    expert_prediction = np.round(fitted_components["expert"]["predictions"])
                    candidate_prediction = greedy._blend_specialist_predictions(
                        base_prediction, expert_prediction, homes, "Detached", 0.75)
                else:
                    transformer, restore = _install_text_features(train, homes, track)
                    fit_context.update(track=track, month=month, component="core")
                    try:
                        core = greedy._fit_history_candidate(
                            month, track, train, homes, variant=VARIANT,
                            fit_tag=f"R23-core-{month}", fit_rows=train,
                        )
                    finally:
                        restore()
                    frozen = greedy._frozen_hgbr_predictions(month, track, train, homes)
                    from experiments.model_optimization_v2.pipeline import blend_prices
                    candidate_prediction = blend_prices(core["predictions"], frozen, challenger_weight=0.75)
                    component_manifest.append({
                        "track": track, "month": month, "component": "core",
                        "word_vocabulary": transformer.word_vectorizer.get_feature_names_out().tolist(),
                        "char_vocabulary": transformer.char_vectorizer.get_feature_names_out().tolist(),
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
                campaign._append({"event": "screen_fold_complete", "card": CARD, "run_id": run_id,
                                  "track": track, "month": month,
                                  "reference_run_id": r22_result["run_id"],
                                  "date_cutoff_exclusive": cutoff.isoformat()})
    except Exception as exc:
        failure = {"run_id": run_id, "card": CARD, "error": f"{type(exc).__name__}: {exc}",
                   "fit_starts": fit_start_count, "fit_details": fit_details,
                   "completed_components": [f"{row['track']}/{row['month']}/{row['component']}"
                                             for row in component_manifest],
                   "reserved_months_labels_opened": False, "hidden_labels_opened": False}
        _write_json(output / "failure_checkpoint.json", failure)
        campaign._append({"event": "card_failed", "card": CARD, "run_id": run_id,
                          "fit_starts": fit_start_count, "error": failure["error"]})
        campaign._write_progress("r23_word_char_tfidf_failed",
                                 next_action="inspect R23 checkpoint and fit ledger before retry",
                                 r23_failure=str((output / "failure_checkpoint.json").relative_to(campaign.ROOT)).replace("\\", "/"))
        raise
    finally:
        greedy._fit_record, greedy._frozen_hgbr_predictions = old_fit_record, old_hgbr

    paired = pd.concat([frame for track_frames in rows_by_track.values() for frame in track_frames],
                       ignore_index=True)
    paired_path = output / "paired_predictions.csv"
    paired.to_csv(paired_path, index=False)
    tracks = {track: _track_result(pd.concat(rows, ignore_index=True))
              for track, rows in rows_by_track.items()}
    result = {"campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
              "card": CARD, "parent_candidate": r22_result["run_id"],
              "parents": campaign.PROTOCOL["parent_by_track"],
              "screen_months": list(SCREEN_MONTHS), "fit_starts": fit_start_count,
              "tracks": tracks, "fit_details": fit_details,
              "reserved_months_labels_opened": False, "hidden_labels_opened": False,
              "data_sha256": source["data_sha256"]}
    result_path = output / "r23_screen_results.json"
    _write_json(result_path, result)
    source_files = [campaign.PROTOCOL_PATH, campaign.PACKAGE / "r23_word_char_tfidf.py",
                    campaign.PACKAGE / "r22_remarks_tfidf.py", campaign.PACKAGE / "r13_lgbm_bagging.py",
                    campaign.PACKAGE / "r03_catboost.py", campaign.PACKAGE / "r04_ablation.py",
                    campaign.PACKAGE / "runner.py", campaign.ROOT / "model_candidate_v2/features.py",
                    campaign.ROOT / "experiments/bill_greedy/runner.py",
                    campaign.ROOT / "experiments/bill_greedy/history.py",
                    campaign.ROOT / "experiments/model_optimization_v2/learners.py"]
    manifest = {
        "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
        "source_identity": source,
        "code_sha256": {str(path.relative_to(campaign.ROOT)).replace("\\", "/"): _sha256(path)
                        for path in source_files},
        "data_sha256": source["data_sha256"], "fit_starts": fit_start_count,
        "fit_details": fit_details, "component_manifest": component_manifest,
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
                      "result_path": str(result_path.relative_to(campaign.ROOT)),
                      "qualifies_by_track": qualifies, "reserved_months_labels_opened": False,
                      "hidden_labels_opened": False})
    next_action = ("run R23 eight-month development confirmation only for screen-qualified tracks"
                   if any(qualifies.values()) else
                   "R23 character n-grams did not beat validated R22; keep R22 and register a distinct mechanism")
    campaign._write_progress(
        "r23_word_char_tfidf_screen_complete", next_action=next_action,
        r23_result=str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        latest_fit_count=len(fit_details),
    )
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run or finalize the R23 word/character TF-IDF screen")
    parser.add_argument("--recover-finalization", metavar="RUN_ID",
                        help="write report and completion state from already-saved fit artifacts; never refit")
    args = parser.parse_args()
    if args.recover_finalization:
        result = finalize_saved_run(args.recover_finalization)
    else:
        result = run_r23_screen()
    print(json.dumps({"run_id": result["run_id"], "fit_starts": result["fit_starts"],
                      "tracks": result["tracks"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
