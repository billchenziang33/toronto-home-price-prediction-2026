"""R25: append train-fold-only latent semantic features to R22 word TF-IDF."""
from __future__ import annotations

import json
import shutil
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

from experiments.bill_target74.r19_expert_weight import _sha256, _write_json


CARD = "R25_train_fold_tfidf_svd"
VARIANT = CARD
SCREEN_MONTHS = ("2023-03", "2024-06")
TRACKS = ("off_market", "list_aware")
SVD_COMPONENTS = 32
SVD_ITERATIONS = 5
SVD_SEED = 0


class FoldTfidfSvd:
    """Append SVD factors fitted only on a component's training-fold remarks."""

    def __init__(self):
        from experiments.bill_target74.r22_remarks_tfidf import FoldTfidf

        self.word = FoldTfidf()
        self.svd = TruncatedSVD(n_components=SVD_COMPONENTS, n_iter=SVD_ITERATIONS,
                                random_state=SVD_SEED, algorithm="randomized")
        self.columns_: list[str] | None = None

    @staticmethod
    def _texts(values) -> pd.Series:
        texts = pd.Series(values, copy=False).fillna("").astype(str)
        if texts.ndim != 1:
            raise ValueError("remarks must be a one-dimensional sequence")
        return texts

    @staticmethod
    def _append(word: pd.DataFrame, latent_values: np.ndarray, *, index) -> pd.DataFrame:
        latent = np.asarray(latent_values, dtype=np.float32)
        if latent.ndim != 2 or latent.shape != (len(word), SVD_COMPONENTS):
            raise ValueError("SVD factors must have one row and 32 components per listing")
        columns = [f"remarks_svd_{position:02d}" for position in range(SVD_COMPONENTS)]
        frame = pd.concat([word, pd.DataFrame(latent, index=index, columns=columns)], axis=1)
        if not np.isfinite(frame.to_numpy(dtype=np.float32)).all():
            raise ValueError("TF-IDF/SVD features contain non-finite values")
        return frame

    def fit_transform(self, remarks, *, index=None) -> pd.DataFrame:
        texts = self._texts(remarks)
        word = self.word.fit_transform(texts, index=index)
        matrix = self.word.vectorizer.transform(texts.tolist())
        if matrix.shape[0] <= SVD_COMPONENTS or matrix.shape[1] < SVD_COMPONENTS:
            raise ValueError(f"training TF-IDF shape {matrix.shape} cannot support {SVD_COMPONENTS} SVD factors")
        self.svd.fit(matrix)
        latent = self.svd.transform(matrix)
        frame = self._append(word, latent, index=word.index)
        self.columns_ = frame.columns.tolist()
        return frame

    def transform(self, remarks, *, index=None) -> pd.DataFrame:
        if self.columns_ is None:
            raise RuntimeError("training-fold TF-IDF and SVD must be fitted before held-out transform")
        texts = self._texts(remarks)
        word = self.word.transform(texts, index=index)
        matrix = self.word.vectorizer.transform(texts.tolist())
        latent = self.svd.transform(matrix)
        frame = self._append(word, latent, index=word.index)
        if frame.columns.tolist() != self.columns_:
            raise RuntimeError("held-out TF-IDF/SVD columns differ from the training-fold columns")
        return frame

    @property
    def explained_variance_ratio_(self) -> list[float]:
        if not hasattr(self.svd, "explained_variance_ratio_"):
            raise RuntimeError("SVD has not been fitted")
        return self.svd.explained_variance_ratio_.astype(float).tolist()


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


def _install_svd_features(fit_rows: pd.DataFrame, eval_rows: pd.DataFrame, track: str):
    """Patch one frozen feature builder and restore it after the component fit."""
    from model_candidate_v2 import features as frozen_features

    if track not in TRACKS:
        raise ValueError(f"unknown track: {track!r}")
    if "remarks" not in fit_rows or "remarks" not in eval_rows:
        raise ValueError("training and evaluation rows must both contain remarks")
    attribute = "off_market_features" if track == "off_market" else "list_aware_features"
    original = getattr(frozen_features, attribute)
    transformer = FoldTfidfSvd()
    fit_identity, eval_identity = id(fit_rows), id(eval_rows)
    fitted_training_transform = False

    def wrapped(homes):
        nonlocal fitted_training_transform
        base = original(homes)
        if id(homes) == fit_identity:
            if not fitted_training_transform:
                text = transformer.fit_transform(homes.remarks, index=base.index)
                fitted_training_transform = True
            else:
                text = transformer.transform(homes.remarks, index=base.index)
        elif id(homes) == eval_identity:
            if not fitted_training_transform:
                raise RuntimeError("held-out transform was requested before fitting train-fold SVD")
            text = transformer.transform(homes.remarks, index=base.index)
        else:
            raise RuntimeError("unexpected row frame passed to the fold-scoped TF-IDF/SVD builder")
        return pd.concat([base, text], axis=1)

    setattr(frozen_features, attribute, wrapped)

    def restore() -> None:
        setattr(frozen_features, attribute, original)

    return transformer, restore


def _report(result: dict) -> str:
    lines = [
        "# R25：R22 词级 TF-IDF + 训练折内 SVD", "",
        f"Campaign：`{result['campaign_id']}`；run：`{result['run_id']}`。",
        f"配对父候选：哈希验证的 R22 run `{result['parent_candidate']}`。",
        "R22 词级 TF-IDF、LightGBM 配置、结构化/历史特征、目标、路由和推理融合保持不变；只附加训练折内拟合的 32 个 SVD 文本因子。",
        "筛选月份的 remarks 仅用冻结训练折 vectorizer 与 SVD 变换；没有用筛选标签拟合词表、IDF 或 SVD 基底。", "",
        "## 双月增量筛选", "",
        "| 赛道 | R22 MdAPE | R25 MdAPE | 相对改善 | MAPE变化 | 通过 |",
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
    lines.extend(["", "## 各筛选月", "", "| 赛道 | 月份 | R22 MdAPE | R25 MdAPE | 相对改善 |",
                  "|---|---|---:|---:|---:|"])
    for track, score in result["tracks"].items():
        for month, values in score["monthly"].items():
            lines.append(f"| {track} | {month} | {values['parent']['mdape_percent']:.5f}% | "
                         f"{values['candidate']['mdape_percent']:.5f}% | "
                         f"{values['relative_mdape_improvement_percent']:+.3f}% |")
    lines.extend(["", "保留月份和隐藏标签未读取。筛选通过只允许进入八个月开发确认，不代表目标已达到。", ""])
    return "\n".join(lines)


def run_r25_screen() -> dict:
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
        raise RuntimeError("R25 must be registered in campaign_protocol.json before fitting")
    if screen_fit_plan() != [
            (month, track, component)
            for month in SCREEN_MONTHS
            for track, components in (("off_market", ("base", "expert")),
                                      ("list_aware", ("core",)))
            for component in components]:
        raise RuntimeError("R25 screen fit plan changed after preregistration")
    r22_result, r22_result_path, r22_manifest = r22._verified_screen_result(campaign)
    if not all(r22_result["tracks"][track]["qualifies_screen"] for track in TRACKS):
        raise RuntimeError("R25 requires both tracks to have passed the verified R22 screen")
    r22_rows_path = r22_result_path.parent / "paired_predictions.csv"
    r22_rows = pd.read_csv(r22_rows_path, dtype={"ml_num": str})
    source = budget._source_identity()
    run_id = "bill_target74_r25_tfidf_svd_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = campaign.CAMPAIGN_ROOT / "batches" / run_id
    output.mkdir(parents=True, exist_ok=False)
    campaign._append({
        "event": "card_start", "card": CARD, "run_id": run_id,
        "screen_months": list(SCREEN_MONTHS), "tracks": list(TRACKS),
        "parent_candidate": r22_result["run_id"], "model_family": "lightgbm",
        "text_features": {"tfidf_max_features": 256, "tfidf_min_df": 25,
                           "tfidf_max_df": 0.95, "word_ngram_range": [1, 2],
                           "svd_algorithm": "randomized", "svd_components": SVD_COMPONENTS,
                           "svd_iterations": SVD_ITERATIONS, "svd_seed": SVD_SEED,
                           "fit_scope": "training remarks only"},
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
            raise RuntimeError("R25 fit context is not initialized")
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
                  "svd_feature_count": sum(name.startswith("remarks_svd_")
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
                raise RuntimeError(f"R25 training data include sales at or after cutoff: {month}")
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
                        transformer, restore = _install_svd_features(fit_rows, eval_rows, track)
                        fit_context.update(track=track, month=month, component=component)
                        try:
                            fitted = greedy._fit_history_candidate(
                                month, track, train, eval_rows, base_variant="G06_lot_garage",
                                variant=VARIANT, fit_tag=f"R25-{component}-{month}", fit_rows=fit_rows,
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
                            "svd_feature_count": sum(name.startswith("remarks_svd_")
                                                     for name in fitted["feature_columns"]),
                            "svd_explained_variance_ratio": transformer.explained_variance_ratio_,
                            "tfidf_vocabulary": transformer.word.vectorizer.get_feature_names_out().tolist(),
                            "prediction_sha256": fitted["prediction_sha256"],
                        })
                    base_prediction = np.round(fitted_components["base"]["predictions"])
                    expert_prediction = np.round(fitted_components["expert"]["predictions"])
                    candidate_prediction = greedy._blend_specialist_predictions(
                        base_prediction, expert_prediction, homes, "Detached", 0.75)
                else:
                    transformer, restore = _install_svd_features(train, homes, track)
                    fit_context.update(track=track, month=month, component="core")
                    try:
                        core = greedy._fit_history_candidate(
                            month, track, train, homes, variant=VARIANT,
                            fit_tag=f"R25-core-{month}", fit_rows=train,
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
                        "svd_feature_count": sum(name.startswith("remarks_svd_")
                                                 for name in core["feature_columns"]),
                        "svd_explained_variance_ratio": transformer.explained_variance_ratio_,
                        "tfidf_vocabulary": transformer.word.vectorizer.get_feature_names_out().tolist(),
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
        campaign._write_progress("r25_tfidf_svd_failed",
                                 next_action="inspect R25 checkpoint and ledger before any retry",
                                 r25_failure=str((output / "failure_checkpoint.json").relative_to(campaign.ROOT)).replace("\\", "/"))
        raise
    finally:
        greedy._fit_record, greedy._frozen_hgbr_predictions = old_fit_record, old_hgbr

    if fit_start_count != len(screen_fit_plan()):
        raise RuntimeError(f"R25 expected {len(screen_fit_plan())} fit starts; observed {fit_start_count}")
    paired = pd.concat([frame for groups in rows_by_track.values() for frame in groups], ignore_index=True)
    paired_path = output / "paired_predictions.csv"
    paired.to_csv(paired_path, index=False)
    tracks = {track: shared._track_result(pd.concat(rows, ignore_index=True))
              for track, rows in rows_by_track.items()}
    result = {"campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
              "card": CARD, "parent_candidate": r22_result["run_id"],
              "screen_months": list(SCREEN_MONTHS), "fit_starts": fit_start_count,
              "tracks": tracks, "fit_details": fit_details,
              "component_manifest": component_manifest,
              "svd": {"algorithm": "randomized", "n_components": SVD_COMPONENTS,
                      "n_iter": SVD_ITERATIONS, "random_state": SVD_SEED},
              "reserved_months_labels_opened": False, "hidden_labels_opened": False,
              "data_sha256": source["data_sha256"]}
    result_path = output / "r25_screen_results.json"
    _write_json(result_path, result)
    source_files = [campaign.PROTOCOL_PATH, campaign.PACKAGE / "r25_tfidf_svd.py",
                    campaign.PACKAGE / "r22_remarks_tfidf.py",
                    campaign.PACKAGE / "r23_word_char_tfidf.py",
                    campaign.PACKAGE / "r03_catboost.py", campaign.PACKAGE / "r04_ablation.py",
                    campaign.PACKAGE / "runner.py", campaign.ROOT / "model_candidate_v2/features.py",
                    campaign.ROOT / "experiments/bill_greedy/runner.py",
                    campaign.ROOT / "experiments/bill_greedy/history.py",
                    campaign.ROOT / "experiments/model_optimization_v2/learners.py"]
    provenance = output / "provenance"
    provenance.mkdir(exist_ok=True)
    shutil.copy2(campaign.PACKAGE / "r25_tfidf_svd.py",
                 provenance / "r25_tfidf_svd_source.py")
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
    next_action = ("design R25 development confirmation only for screen-qualified tracks"
                   if any(qualifies.values()) else
                   "R25 SVD factors failed the incremental screen; preserve R22 and register a distinct method")
    campaign._write_progress("r25_tfidf_svd_screen_complete", next_action=next_action,
                             r25_result=str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
                             latest_fit_count=fit_start_count)
    return result


def main() -> None:
    result = run_r25_screen()
    print(json.dumps({"run_id": result["run_id"], "fit_starts": result["fit_starts"],
                      "tracks": {track: {"parent_mdape_percent": score["parent"]["mdape_percent"],
                                         "candidate_mdape_percent": score["candidate"]["mdape_percent"],
                                         "relative_mdape_improvement_percent": score[
                                             "relative_mdape_improvement_percent"],
                                         "mape_relative_change_percent": score[
                                             "mape_relative_change_percent"],
                                         "qualifies_screen": score["qualifies_screen"],
                                         "monthly": {month: values["relative_mdape_improvement_percent"]
                                                     for month, values in score["monthly"].items()}}
                                 for track, score in result["tracks"].items()}},
                     indent=2))


if __name__ == "__main__":
    main()
