"""R164: R97 CatBoost residual learner augmented with fold-local listing text."""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from experiments.bill_target74 import r04_ablation as provenance
from experiments.bill_target74 import r22_remarks_tfidf as r22
from experiments.bill_target74 import r97_past_oof_catboost_residual_meta as r97
from experiments.bill_target74 import r154_listaware_lgbm_quantile_residual as r154
from experiments.bill_target74 import r161_listaware_seed_oof_catboost as r161
from experiments.bill_target74 import runner as campaign


CARD = "R164_listaware_text_residual"
AUDITOR_PATH = Path(__file__).with_name("r164_independent_audit.py")
TRACK = "list_aware"
SCREEN_MONTHS = tuple(campaign.PROTOCOL["screen_months"])
DEVELOPMENT_MONTHS = tuple(campaign.PROTOCOL["development_months"])
SEED_MONTHS = tuple(r154.SEED_MONTHS)
HISTORY_WINDOW_POLICY = "all_available_strict_past"
MIN_HISTORY_ROWS = 3500
RESIDUAL_TARGET_CLIP = 0.15
PREDICTION_RESIDUAL_CLIP = 0.10
CORRECTION_SHRINKAGE = 0.50
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20261164
R97_RUN_ID = r161.R97_RUN_ID
R97_DIR = campaign.CAMPAIGN_ROOT / "batches" / R97_RUN_ID
CATBOOST_CONFIG = dict(r97.CATBOOST_CONFIG)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _keys_sha256(frame: pd.DataFrame) -> str:
    keys = sorted(zip(frame.month.astype(str), frame.ml_num.astype(str)))
    payload = "\n".join(f"{month}|{ml_num}" for month, ml_num in keys).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _vocabulary_sha256(vocabulary: dict[str, int]) -> str:
    normalized = {str(token): int(index) for token, index in vocabulary.items()}
    payload = json.dumps(normalized, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(campaign.ROOT.resolve())).replace("\\", "/")


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _source_files() -> list[Path]:
    paths = [Path(__file__), AUDITOR_PATH, Path(r22.__file__), Path(r97.__file__), Path(r154.__file__),
             Path(r161.__file__), campaign.PACKAGE / "r03_catboost.py",
             campaign.PACKAGE / "r04_ablation.py", campaign.PACKAGE / "runner.py",
             campaign.ROOT / "experiments/bill_greedy/runner.py",
             campaign.ROOT / "experiments/bill_greedy/history.py",
             campaign.ROOT / "model_candidate_v2/features.py"]
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("R164 runner, auditor, or pinned source file is missing")
    return paths


def _code_hashes() -> dict[str, str]:
    return {_relative(path): _sha(path) for path in _source_files()}


def _current_best() -> tuple[pd.DataFrame, dict]:
    paired, identity = r161._r97_current_best()
    paired = paired.loc[paired.track.eq(TRACK)].copy()
    if paired.duplicated(["month", "ml_num"]).any():
        raise RuntimeError("R164 R97 current-best rows are duplicated")
    return paired, identity


def _read_months(stage: str) -> tuple[str, ...]:
    return r154._read_months(stage)


def _load_frame_with_remarks(inputs: dict, stage: str) -> pd.DataFrame:
    frame = r154._load_frame(inputs, stage)
    months = _read_months(stage)
    if set(months) & set(campaign.PROTOCOL["reserved_months"]):
        raise RuntimeError("R164 safe remarks filter intersects reserved months")
    filters = []
    for month in months:
        start = pd.Timestamp(month + "-01")
        filters.append([("date_sold", ">=", start),
                        ("date_sold", "<", start + pd.offsets.MonthBegin(1))])
    remarks = pd.read_parquet(campaign.DATA, columns=["ml_num", "date_sold", "remarks"], filters=filters)
    remarks["date_sold"] = pd.to_datetime(remarks.date_sold, errors="raise")
    remarks["month"] = remarks.date_sold.dt.to_period("M").astype(str)
    remarks["ml_num"] = remarks.ml_num.astype(str)
    if (set(remarks.month.astype(str)) != set(months)
            or remarks.duplicated(["month", "ml_num"]).any()
            or set(remarks.month.astype(str)) & set(campaign.PROTOCOL["reserved_months"])):
        raise RuntimeError("R164 remarks rows have unexpected safe-month coverage or duplicate keys")
    remarks = remarks[["month", "ml_num", "remarks"]]
    frame["month"] = frame.month.astype(str)
    frame["ml_num"] = frame.ml_num.astype(str)
    joined = frame.merge(remarks, on=["month", "ml_num"], how="left",
                         validate="one_to_one", indicator=True)
    if not joined._merge.eq("both").all() or len(joined) != len(frame):
        raise RuntimeError("R164 safe remarks do not align with OOF rows")
    joined["remarks"] = joined.remarks.fillna("").astype(str)
    return joined.drop(columns="_merge")


def _history_for(frame: pd.DataFrame, month: str) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    cutoff = pd.Timestamp(month + "-01")
    dates = pd.to_datetime(frame.date_sold, errors="raise")
    history = frame.loc[dates.lt(cutoff)].copy()
    if history.empty:
        raise RuntimeError(f"R164 has no strictly past history for {month}")
    window_start = pd.to_datetime(history.date_sold, errors="raise").min()
    if (pd.to_datetime(history.date_sold).ge(cutoff).any()
            or history.duplicated(["month", "ml_num"]).any()):
        raise RuntimeError(f"R164 strict-past history invalid for {month}")
    return history, window_start, cutoff


def _preflight(stage: str) -> tuple[dict, pd.DataFrame, pd.DataFrame, dict, dict, dict]:
    inputs = r154._input_artifacts()
    frames = {name: _load_frame_with_remarks(inputs, name) for name in ("screen", "development")}
    frame = frames[stage]
    best, best_identity = _current_best()
    support = {}
    for scope, scope_frame in frames.items():
        if scope_frame.duplicated(["month", "ml_num"]).any():
            raise RuntimeError(f"R164 {scope} safe frame contains duplicate month/listing keys")
        scope_months = set(scope_frame.month.astype(str))
        if (scope_months & set(campaign.PROTOCOL["reserved_months"])
                or scope_months != set(_read_months(scope))):
            raise RuntimeError(f"R164 {scope} safe frame has an invalid month scope")
        target_months = SCREEN_MONTHS if scope == "screen" else DEVELOPMENT_MONTHS
        support[scope] = {}
        for month in target_months:
            history, window_start, cutoff = _history_for(scope_frame, month)
            text = r22.FoldTfidf()
            text_features = text.fit_transform(history.remarks)
            support[scope][month] = {"rows": int(len(history)),
                                     "months": sorted(history.month.astype(str).unique().tolist()),
                                     "window_start_inclusive": window_start.isoformat(),
                                     "cutoff_exclusive": cutoff.isoformat(),
                                     "training_oof_keys_sha256": _keys_sha256(history),
                                     "remarks_training_rows": int(len(history)),
                                     "remarks_feature_count": int(text_features.shape[1]),
                                     "remarks_vocabulary_sha256": _vocabulary_sha256(text.vocabulary_)}
            if len(history) < MIN_HISTORY_ROWS:
                raise RuntimeError(f"R164 has {len(history)} rows for {month}; needs {MIN_HISTORY_ROWS}")
    return inputs, frame, best, best_identity, support, provenance._source_identity()


def _card_spec(inputs: dict, best_identity: dict, source: dict, support: dict) -> dict:
    return {
        "status": "registered", "track": TRACK,
        "screen_months": list(SCREEN_MONTHS), "development_months": list(DEVELOPMENT_MONTHS),
        "seed_months": list(SEED_MONTHS),
        "history_policy": HISTORY_WINDOW_POLICY,
        "minimum_past_oof_rows": MIN_HISTORY_ROWS,
        "screen_fit_starts": len(SCREEN_MONTHS),
        "development_fit_starts": len(DEVELOPMENT_MONTHS),
        "baseline_by_track": {TRACK: "validated R35 list-aware OOF"},
        "current_best_comparator": best_identity,
        "method": ("Fit the R97 CatBoost MAE residual learner on R44 seed OOF and R35 safe development "
                   "OOF rows strictly earlier than each target month. Add R22 FoldTfidf remarks features, "
                   "fitting the vocabulary on that fold's training rows only. Preserve the R97 learner, "
                   "numeric/categorical features, target, and correction."),
        "difference_from_prior_work": ("R97's residual meta-learner excluded remarks; R22/R35 used remarks "
                                        "only in the first-stage price model. R164 tests fold-local text "
                                        "features in the residual learner itself. R161 tested expanded "
                                        "history without residual text."),
        "parent_oof_inputs": inputs,
        "current_best_artifacts": best_identity,
        "source_preflight": {"strict_past_support_by_target_month": support},
        "catboost_config": CATBOOST_CONFIG,
        "residual_target": "clip(log(sold_price / R35_prediction), -0.15, 0.15)",
        "correction": {"formula": "round(R35_prediction * exp(0.50 * clip(predicted_log_residual, -0.10, 0.10)))",
                       "residual_target_clip": RESIDUAL_TARGET_CLIP,
                       "prediction_residual_clip": PREDICTION_RESIDUAL_CLIP,
                       "shrinkage": CORRECTION_SHRINKAGE,
                       "rounding": "nearest dollar"},
        "features": {"builder": "r97._build_features + r22.FoldTfidf",
                     "categorical_features": list(r97.CATEGORICAL_FEATURES),
                     "remarks_tfidf": {"max_features": r22.MAX_FEATURES, "min_df": r22.MIN_DF,
                                        "max_df": r22.MAX_DF, "ngram_range": list(r22.NGRAM_RANGE),
                                        "stop_words": "english", "sublinear_tf": True,
                                        "fit_scope": "strict-past training remarks only"},
                     "forbidden": ["sold_price", "date_sold", "external_avm_estimate"]},
        "screen_gates": {"minimum_relative_mdape_improvement_percent": 0.25,
                          "both_months_improved": True,
                          "maximum_mape_regression_percent": 1.0,
                          "paired_bootstrap_upper95_below_zero": True,
                          "comparators": ["R35 validated parent", "R97 current-best development candidate"],
                          "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                          "bootstrap_seed": BOOTSTRAP_SEED},
        "development_gates": dict(campaign.PROTOCOL["gates"]),
        "strict_development_mdape_target_percent": 4.0,
        "data_sha256": source["data_sha256"], "parent_commit": source["commit"],
        "source_sha256": _code_hashes(),
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        "reserve_validation_eligible": False,
    }


def _register(inputs: dict, best_identity: dict, source: dict, support: dict) -> dict:
    hashes = _code_hashes()
    existing = campaign.PROTOCOL["cards"].get(CARD)
    if existing is not None:
        if (existing.get("source_sha256") != hashes
                or existing.get("parent_oof_inputs") != inputs
                or existing.get("current_best_artifacts") != best_identity):
            raise RuntimeError("R164 frozen source or input identity changed after registration")
        return existing
    card = _card_spec(inputs, best_identity, source, support)
    campaign.PROTOCOL["cards"][CARD] = card
    campaign.PROTOCOL_PATH.write_text(json.dumps(campaign.PROTOCOL, indent=2, ensure_ascii=False) + "\n",
                                      encoding="utf-8")
    campaign._append({"event": "card_registered", "card": CARD,
                      "source_sha256": hashes[_relative(Path(__file__))],
                      "parent_oof_inputs": inputs, "current_best_artifacts": best_identity,
                      "history_policy": HISTORY_WINDOW_POLICY,
                      "strict_past_support_by_target_month": support,
                      "reserved_months_labels_opened": False, "hidden_labels_opened": False})
    campaign._write_progress("r164_registered",
                             next_action="run R164 list-aware CatBoost screen with fold-local remarks TF-IDF",
                             r164_card=CARD)
    return card


def _metric(actual: np.ndarray, prediction: np.ndarray) -> dict:
    ape = np.abs(prediction - actual) / actual * 100.0
    return {"n": int(len(actual)), "mdape_percent": float(np.median(ape)),
            "mape_percent": float(np.mean(ape))}


def _bootstrap(rows: pd.DataFrame, reference: str) -> dict:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    groups = [part for _, part in rows.groupby("month", sort=True)]
    differences = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    for replicate in range(BOOTSTRAP_REPLICATES):
        actual_parts, before_parts, after_parts = [], [], []
        for part in groups:
            indices = rng.integers(0, len(part), size=len(part))
            actual_parts.append(part.sold_price.to_numpy(float)[indices])
            before_parts.append(part[reference].to_numpy(float)[indices])
            after_parts.append(part.candidate_prediction.to_numpy(float)[indices])
        actual = np.concatenate(actual_parts)
        before = np.concatenate(before_parts)
        after = np.concatenate(after_parts)
        differences[replicate] = (np.median(np.abs(after - actual) / actual * 100.0)
                                  - np.median(np.abs(before - actual) / actual * 100.0))
    return {"lower95": float(np.quantile(differences, 0.025)),
            "upper95": float(np.quantile(differences, 0.975)),
            "replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED}


def _score(rows: pd.DataFrame, reference: str, months: tuple[str, ...], stage: str) -> dict:
    actual = rows.sold_price.to_numpy(float)
    before = rows[reference].to_numpy(float)
    after = rows.candidate_prediction.to_numpy(float)
    values = np.column_stack([actual, before, after])
    if not np.isfinite(values).all() or (values <= 0.0).any():
        raise RuntimeError("R164 score input contains invalid prices")
    benchmark, candidate = _metric(actual, before), _metric(actual, after)
    gain = 100.0 * (benchmark["mdape_percent"] - candidate["mdape_percent"]) / benchmark["mdape_percent"]
    mape_change = 100.0 * (candidate["mape_percent"] - benchmark["mape_percent"]) / benchmark["mape_percent"]
    monthly, gains = {}, []
    for month in months:
        part = rows.loc[rows.month.astype(str).eq(month)]
        b = _metric(part.sold_price.to_numpy(float), part[reference].to_numpy(float))
        c = _metric(part.sold_price.to_numpy(float), part.candidate_prediction.to_numpy(float))
        month_gain = 100.0 * (b["mdape_percent"] - c["mdape_percent"]) / b["mdape_percent"]
        gains.append(month_gain)
        monthly[month] = {"benchmark": b, "candidate": c,
                          "relative_mdape_improvement_percent": float(month_gain)}
    bootstrap = _bootstrap(rows, reference)
    gates = campaign.PROTOCOL["gates"]
    if stage == "screen":
        checks = {"minimum_relative_mdape_improvement": gain >= gates["screen_relative_mdape_improvement_percent"],
                  "both_screen_months_improved": all(value > 0.0 for value in gains),
                  "mape_guard": mape_change <= gates["screen_mape_regression_max_percent"],
                  "paired_bootstrap_upper95_below_zero": bootstrap["upper95"] < 0.0}
    else:
        checks = {"development_relative_gain": gain >= gates["development_relative_mdape_improvement_percent"],
                  "months_improved": sum(value > 0.0 for value in gains) >= gates["development_months_improved_min"],
                  "worst_month_guard": max([0.0] + [-value for value in gains])
                  <= gates["development_worst_month_regression_max_percent"],
                  "mape_guard": mape_change <= gates["development_mape_regression_max_percent"],
                  "paired_bootstrap_upper95_below_zero": bootstrap["upper95"] < 0.0,
                  "strict_listaware_target": candidate["mdape_percent"] < 4.0}
    return {"n": int(len(rows)), "benchmark": benchmark, "candidate": candidate,
            "relative_mdape_improvement_percent": float(gain),
            "mape_relative_change_percent": float(mape_change),
            "months_improved": int(sum(value > 0.0 for value in gains)),
            "worst_month_regression_percent": float(max([0.0] + [-value for value in gains])),
            "paired_bootstrap_mdape_difference_pp": bootstrap,
            "monthly": monthly, "gate_checks": checks,
            "qualifies": all(checks.values())}


def _fit_fold(history: pd.DataFrame, query: pd.DataFrame, run_id: str, stage: str,
              output: Path, source: dict, details: list[dict], month: str) -> pd.DataFrame:
    history, window_start, cutoff = _history_for(history, month)
    if len(history) < MIN_HISTORY_ROWS:
        raise RuntimeError(f"R164 requires {MIN_HISTORY_ROWS} rows in {month}; found {len(history)}")
    if query.empty or query.ml_num.duplicated().any():
        raise RuntimeError(f"R164 query rows are missing or duplicated for {month}")
    train_x, categorical = r97._build_features(history, TRACK)
    query_x, query_categorical = r97._build_features(query, TRACK)
    tfidf = r22.FoldTfidf()
    train_text = tfidf.fit_transform(history.remarks, index=train_x.index)
    query_text = tfidf.transform(query.remarks, index=query_x.index)
    train_x = pd.concat([train_x, train_text], axis=1)
    query_x = pd.concat([query_x, query_text], axis=1)
    if (categorical != query_categorical or list(train_x.columns) != list(query_x.columns)
            or {"sold_price", "date_sold", "remarks"} & set(train_x.columns)
            or train_text.shape[1] != query_text.shape[1]):
        raise RuntimeError(f"R164 feature schema or target leakage check failed for {month}")
    vocabulary_sha = _vocabulary_sha256(tfidf.vocabulary_)
    target_raw = np.log(history.sold_price.to_numpy(float) / history.base_prediction.to_numpy(float))
    if not np.isfinite(target_raw).all():
        raise RuntimeError(f"R164 has non-finite residual targets for {month}")
    target = np.clip(target_raw, -RESIDUAL_TARGET_CLIP, RESIDUAL_TARGET_CLIP)
    fit_id = f"{campaign.PROTOCOL['campaign_id']}:{CARD}:{TRACK}:{month}:{stage}:{run_id}"
    start_event = {"event": "fit_start", "card": CARD, "run_id": run_id,
                   "stage": stage, "fit_id": fit_id, "track": TRACK, "month": month,
                   "training_oof_rows": len(history),
                   "training_oof_months": sorted(history.month.astype(str).unique().tolist()),
                   "training_oof_keys_sha256": _keys_sha256(history),
                   "query_keys_sha256": _keys_sha256(query),
                   "window_start_inclusive": window_start.isoformat(),
                   "history_cutoff_exclusive": cutoff.isoformat(),
                   "remarks_feature_count": int(train_text.shape[1]),
                   "remarks_vocabulary_sha256": vocabulary_sha,
                   "data_sha256": source["data_sha256"],
                   "reserved_months_labels_opened": False, "hidden_labels_opened": False}
    campaign._append(start_event)
    started = datetime.now(timezone.utc)
    try:
        model = CatBoostRegressor(**CATBOOST_CONFIG)
        model.fit(train_x, target, cat_features=categorical)
        raw_prediction = np.asarray(model.predict(query_x), dtype=float)
        correction = CORRECTION_SHRINKAGE * np.clip(
            raw_prediction, -PREDICTION_RESIDUAL_CLIP, PREDICTION_RESIDUAL_CLIP)
        base = query.base_prediction.to_numpy(float)
        prediction = np.rint(base * np.exp(correction))
        if not np.isfinite(prediction).all() or (prediction <= 0.0).any():
            raise RuntimeError(f"R164 generated invalid predictions for {month}")
        prediction_path = output / "fold_predictions" / f"{month}.csv"
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame({"month": month, "track": TRACK,
                              "ml_num": query.ml_num.astype(str).to_numpy(),
                              "sold_price": query.sold_price.to_numpy(float),
                              "base_prediction": base,
                              "candidate_prediction": prediction,
                              "correction_log": correction,
                              "raw_predicted_log_residual": raw_prediction,
                              "history_oof_rows": int(len(history))})
        frame.to_csv(prediction_path, index=False)
        detail = {**start_event, "fit_seconds": (datetime.now(timezone.utc) - started).total_seconds(),
                  "model_family": "catboost", "model_config": CATBOOST_CONFIG,
                  "feature_columns": list(train_x.columns), "categorical_features": categorical,
                  "remarks_feature_count": int(train_text.shape[1]),
                  "remarks_vocabulary_sha256": vocabulary_sha,
                  "remarks_tfidf_config": {"max_features": r22.MAX_FEATURES, "min_df": r22.MIN_DF,
                                           "max_df": r22.MAX_DF, "ngram_range": list(r22.NGRAM_RANGE)},
                  "residual_target_clip": RESIDUAL_TARGET_CLIP,
                  "prediction_clip": PREDICTION_RESIDUAL_CLIP,
                  "correction_shrinkage": CORRECTION_SHRINKAGE,
                  "prediction_file": _relative(prediction_path),
                  "prediction_file_sha256": _sha(prediction_path)}
        details.append(detail)
        campaign._append({"event": "fit_complete", "card": CARD, "run_id": run_id,
                          "stage": stage, "fit_id": fit_id, "track": TRACK, "month": month,
                          "status": "complete", "seconds": detail["fit_seconds"],
                          "training_oof_rows": len(history),
                          "training_oof_keys_sha256": _keys_sha256(history),
                          "query_keys_sha256": _keys_sha256(query),
                          "remarks_feature_count": int(train_text.shape[1]),
                          "remarks_vocabulary_sha256": vocabulary_sha,
                          "prediction_file_sha256": detail["prediction_file_sha256"],
                          "reserved_months_labels_opened": False, "hidden_labels_opened": False})
        return frame
    except BaseException as exc:
        campaign._append({"event": "fit_complete", "card": CARD, "run_id": run_id,
                          "stage": stage, "fit_id": fit_id, "track": TRACK, "month": month,
                          "status": "failed", "error": f"{type(exc).__name__}: {exc}",
                          "reserved_months_labels_opened": False, "hidden_labels_opened": False})
        raise


def _paired(frame: pd.DataFrame, best: pd.DataFrame, fold_paths: dict,
            months: tuple[str, ...]) -> pd.DataFrame:
    parts = []
    for month in months:
        candidate = pd.read_csv(fold_paths[month], dtype={"ml_num": str})
        parent = frame.loc[frame.month.astype(str).eq(month), ["month", "ml_num", "sold_price", "base_prediction"]].copy()
        current = best.loc[best.month.astype(str).eq(month), ["ml_num", "sold_price", "candidate_prediction"]].copy()
        parent["ml_num"], current["ml_num"] = parent.ml_num.astype(str), current.ml_num.astype(str)
        if (parent.ml_num.duplicated().any() or current.ml_num.duplicated().any()
                or set(parent.ml_num) != set(candidate.ml_num.astype(str))
                or set(current.ml_num) != set(candidate.ml_num.astype(str))):
            raise RuntimeError(f"R164 predictions do not align for {month}")
        parent = parent.set_index("ml_num").reindex(candidate.ml_num.astype(str))
        current = current.set_index("ml_num").reindex(candidate.ml_num.astype(str))
        if (parent.isna().any().any() or current.isna().any().any()
                or not np.array_equal(parent.sold_price.to_numpy(float), candidate.sold_price.to_numpy(float))
                or not np.array_equal(current.sold_price.to_numpy(float), candidate.sold_price.to_numpy(float))):
            raise RuntimeError(f"R164 labels or parent predictions do not align for {month}")
        parts.append(pd.DataFrame({"month": month, "track": TRACK,
                                   "ml_num": candidate.ml_num.astype(str),
                                   "sold_price": candidate.sold_price.astype(float),
                                   "baseline_prediction": parent.base_prediction.to_numpy(float),
                                   "current_best_prediction": current.candidate_prediction.to_numpy(float),
                                   "candidate_prediction": candidate.candidate_prediction.to_numpy(float),
                                   "correction_log": candidate.correction_log.to_numpy(float),
                                   "history_oof_rows": candidate.history_oof_rows.astype(int)}))
    result = pd.concat(parts, ignore_index=True)
    if result.duplicated(["month", "track", "ml_num"]).any():
        raise RuntimeError("R164 paired predictions contain duplicate keys")
    return result


def _write_result(stage: str, run_id: str, output: Path, source: dict, inputs: dict,
                  best_identity: dict, support: dict, details: list[dict],
                  paired: pd.DataFrame, months: tuple[str, ...]) -> dict:
    paired_path = output / "paired_predictions.csv"
    paired.to_csv(paired_path, index=False)
    baseline = _score(paired, "baseline_prediction", months, stage)
    current_best = _score(paired, "current_best_prediction", months, stage)
    qualifies = baseline["qualifies"] and current_best["qualifies"] if stage == "screen" else baseline["qualifies"]
    result = {"campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
              "card": CARD, "stage": stage, "track": TRACK, "months": list(months),
              "registered_fit_starts": len(details), "baseline_scores": baseline,
              "current_best_scores": current_best, "qualifies": bool(qualifies),
              "strict_target_mdape_percent": 4.0 if stage == "development" else None,
              "fit_details": details, "strict_past_support": support,
              "data_sha256": source["data_sha256"], "input_artifacts": inputs,
              "current_best_artifacts": best_identity,
              "reserved_months_labels_opened": False, "hidden_labels_opened": False}
    result_path = output / ("screen_results.json" if stage == "screen" else "development_results.json")
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    provenance_dir = output / "provenance"
    provenance_dir.mkdir(exist_ok=True)
    source_hashes, snapshots = {}, {}
    for path in _source_files() + [campaign.PROTOCOL_PATH]:
        rel, digest = _relative(path), _sha(path)
        snapshot_name = hashlib.sha256(rel.encode("utf-8")).hexdigest()[:12] + "_" + path.name + ".snapshot"
        snapshot = provenance_dir / snapshot_name
        shutil.copy2(path, snapshot)
        source_hashes[rel], snapshots[rel] = digest, _relative(snapshot)
    manifest = {"campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
                "card": CARD, "stage": stage, "source_identity": source,
                "source_sha256": source_hashes, "source_snapshot_paths": snapshots,
                "input_artifacts": inputs, "current_best_artifacts": best_identity,
                "strict_past_support": support, "registered_fit_starts": len(details),
                "fit_details": details, "paired_predictions_sha256": _sha(paired_path),
                "result_sha256": _sha(result_path),
                "reserved_months_labels_opened": False, "hidden_labels_opened": False}
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                                               encoding="utf-8")
    report_lines = ["# R164 list-aware CatBoost with fold-local remarks TF-IDF", "",
                    f"Run `{run_id}`; stage `{stage}`; strict-past OOF rows only.",
                    "R44/R35 safe OOF rows were eligible only when sold before the target month.",
                    "R97's CatBoost configuration, existing features, target, and correction stayed fixed; only fold-local remarks TF-IDF features were added.",
                    "No reserved-month or hidden labels were opened.", ""]
    for label, score in (("R35 validated parent", baseline), ("R97 current best", current_best)):
        report_lines.extend([f"## Versus {label}", "",
                             f"Benchmark MdAPE: {score['benchmark']['mdape_percent']:.6f}%",
                             f"Candidate MdAPE: {score['candidate']['mdape_percent']:.6f}%",
                             f"Relative MdAPE improvement: {score['relative_mdape_improvement_percent']:+.4f}%",
                             f"MAPE relative change: {score['mape_relative_change_percent']:+.4f}%",
                             f"Improved months: {score['months_improved']}/{len(months)}",
                             f"Bootstrap upper 95%: {score['paired_bootstrap_mdape_difference_pp']['upper95']:+.6f} pp",
                             f"Gate passed: {score['qualifies']}", ""])
        for month, detail in score["monthly"].items():
            report_lines.append(f"- {month}: {detail['benchmark']['mdape_percent']:.5f}% → {detail['candidate']['mdape_percent']:.5f}%")
        report_lines.append("")
    (output / "research_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    card = campaign.PROTOCOL["cards"][CARD]
    card["last_run_id"] = run_id
    card["last_result_path"] = _relative(result_path)
    card["screen_preliminary_pass"] = bool(qualifies) if stage == "screen" else card.get("screen_preliminary_pass", False)
    card["status"] = "screen_pending_independent_audit" if stage == "screen" else "development_pending_independent_audit"
    if stage == "screen":
        card["screen_passed_tracks"] = [TRACK] if qualifies else []
    campaign.PROTOCOL_PATH.write_text(json.dumps(campaign.PROTOCOL, indent=2, ensure_ascii=False) + "\n",
                                      encoding="utf-8")
    campaign._append({"event": "card_complete", "card": CARD, "stage": stage,
                      "run_id": run_id, "fit_starts": len(details),
                      "result_path": _relative(result_path), "qualifies": bool(qualifies),
                      "baseline_gate": baseline["qualifies"], "current_best_gate": current_best["qualifies"],
                      "reserved_months_labels_opened": False, "hidden_labels_opened": False})
    campaign._write_progress(f"r164_{stage}_pending_audit",
                             next_action="independently audit R164 before any development or promotion",
                             r164_card=CARD, r164_run_id=run_id,
                             r164_result=_relative(result_path), r164_preliminary_pass=bool(qualifies))
    return result


def run(stage: str = "screen") -> dict:
    if stage not in {"screen", "development"}:
        raise ValueError("stage must be screen or development")
    inputs, frame, best, best_identity, support, source = _preflight(stage)
    card = _register(inputs, best_identity, source, support)
    if stage == "screen" and card.get("status") != "registered":
        raise RuntimeError(f"R164 screen is already run or pending audit: {card.get('status')}")
    if stage == "development" and (card.get("status") != "screen_passed"
                                    or card.get("screen_passed_tracks") != [TRACK]):
        raise RuntimeError("R164 development requires an independently audited passing screen")
    months = SCREEN_MONTHS if stage == "screen" else DEVELOPMENT_MONTHS
    run_id = f"r164{'s' if stage == 'screen' else 'd'}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    output = campaign.CAMPAIGN_ROOT / "batches" / run_id
    output.mkdir(parents=True, exist_ok=False)
    (output / "fold_predictions").mkdir()
    campaign._append({"event": "card_start", "card": CARD, "run_id": run_id,
                      "stage": stage, "months": list(months), "track": TRACK,
                      "expected_fit_starts": len(months), "data_sha256": source["data_sha256"],
                      "reserved_months_labels_opened": False, "hidden_labels_opened": False})
    details, fold_paths, query_rows = [], {}, []
    try:
        for month in months:
            history, window_start, cutoff = _history_for(frame, month)
            query = frame.loc[frame.month.astype(str).eq(month)].copy().reset_index(drop=True)
            if query.empty or pd.to_datetime(history.date_sold).ge(cutoff).any():
                raise RuntimeError(f"R164 strict-past query or cutoff failed for {month}")
            prediction_frame = _fit_fold(history, query, run_id, stage, output, source, details, month)
            fold_paths[month] = output / "fold_predictions" / f"{month}.csv"
            query_rows.append(query)
            campaign._append({"event": "fold_complete", "card": CARD, "run_id": run_id,
                              "stage": stage, "track": TRACK, "month": month,
                              "rows": len(prediction_frame), "history_rows": len(history),
                              "window_start_inclusive": window_start.isoformat(),
                              "history_cutoff_exclusive": cutoff.isoformat(),
                              "reserved_months_labels_opened": False, "hidden_labels_opened": False})
    except BaseException as exc:
        checkpoint = {"card": CARD, "run_id": run_id, "stage": stage,
                      "error": f"{type(exc).__name__}: {exc}", "fit_details": details,
                      "reserved_months_labels_opened": False, "hidden_labels_opened": False}
        (output / "failure_checkpoint.json").write_text(json.dumps(checkpoint, indent=2, ensure_ascii=False) + "\n",
                                                        encoding="utf-8")
        campaign._append({"event": "card_failed", "card": CARD, "run_id": run_id,
                          "stage": stage, "fit_starts": len(details), "error": checkpoint["error"],
                          "reserved_months_labels_opened": False, "hidden_labels_opened": False})
        campaign._write_progress(f"r164_{stage}_failed",
                                 next_action="preserve R164 checkpoint and audit its completed fits; do not rerun the same screen blindly",
                                 r164_card=CARD, r164_run_id=run_id)
        raise
    scoring_frame = pd.concat(query_rows, ignore_index=True)
    paired = _paired(scoring_frame, best, fold_paths, months)
    return _write_result(stage, run_id, output, source, inputs, best_identity,
                         support, details, paired, months)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("screen", "development"), default="screen")
    args = parser.parse_args()
    outcome = run(args.stage)
    print(json.dumps({"run_id": outcome["run_id"], "stage": outcome["stage"],
                      "qualifies": outcome["qualifies"],
                      "baseline_mdape": outcome["baseline_scores"]["candidate"]["mdape_percent"],
                      "current_best_mdape": outcome["current_best_scores"]["candidate"]["mdape_percent"],
                      "fit_starts": outcome["registered_fit_starts"]}, ensure_ascii=False), flush=True)
