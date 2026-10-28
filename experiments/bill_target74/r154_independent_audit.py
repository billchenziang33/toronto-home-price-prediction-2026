"""Independently rebuild, refit, and audit R154 list-aware quantile residual folds."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from experiments.bill_target74 import r97_past_oof_catboost_residual_meta as r97


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "experiments/bill_target74/campaign_protocol.json"
PROGRESS_PATH = ROOT / "outputs/bill_target74/bill_target74_unbounded_20261024/progress.json"
LEDGER_PATH = ROOT / "outputs/bill_target74/bill_target74_unbounded_20261024/fits.jsonl"
CARD_KEY = "R154_listaware_lgbm_quantile_residual_r35"
TRACK = "list_aware"
R35_RUN_ID = "bill_target74_r35_house_type_text_development_20261025T040058Z"
R35_SCREEN_RUN_ID = "bill_target74_r35_house_type_text_20261025T035638Z"
R44_RUN_ID = "bill_target74_r44_past_oof_r35_blend_20261025T055003Z"
R44_CARD = "R44_past_oof_r35_blend"
SEED_MONTHS = ("2022-06", "2022-11")
SCREEN_MONTHS = ("2023-03", "2024-06")
RESIDUAL_TARGET_CLIP = 0.15
PREDICTION_RESIDUAL_CLIP = 0.10
CORRECTION_SHRINKAGE = 0.50
MIN_TRAINING_OOF_ROWS = 3500
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20261074
MODEL_CONFIG = {
    "objective": "quantile", "alpha": 0.5, "n_estimators": 500,
    "learning_rate": 0.03, "num_leaves": 15, "max_depth": 4,
    "min_child_samples": 50, "reg_lambda": 20.0, "max_bin": 63,
    "random_state": 20261074, "n_jobs": 1, "deterministic": True,
    "force_col_wise": True, "verbosity": -1,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def feature_matrices(history: pd.DataFrame, query: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train_x, categorical = r97._build_features(
        history.drop(columns=["sold_price", "date_sold", "month", "ml_num"], errors="ignore"), TRACK,
    )
    query_x, query_categorical = r97._build_features(
        query.drop(columns=["sold_price", "date_sold", "month", "ml_num"], errors="ignore"), TRACK,
    )
    if categorical != query_categorical or list(train_x.columns) != list(query_x.columns):
        raise RuntimeError("R154 audit train/query feature schemas differ")
    if {"sold_price", "date_sold"} & (set(train_x.columns) | set(query_x.columns)):
        raise RuntimeError("R154 audit found target labels in model features")
    for column in categorical:
        levels = sorted(train_x[column].astype("string").fillna("__MISSING__").unique().astype(str).tolist())
        train_x[column] = pd.Categorical(train_x[column].astype(str), categories=levels)
        query_x[column] = pd.Categorical(query_x[column].astype(str), categories=levels)
    return train_x, query_x, categorical


def refit_predict(history: pd.DataFrame, query: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict]:
    train_x, query_x, categorical = feature_matrices(history, query)
    target_raw = np.log(history.sold_price.to_numpy(float) / history.base_prediction.to_numpy(float))
    target = np.clip(target_raw, -RESIDUAL_TARGET_CLIP, RESIDUAL_TARGET_CLIP)
    model = LGBMRegressor(**MODEL_CONFIG)
    model.fit(train_x, target, categorical_feature=categorical)
    raw = np.asarray(model.predict(query_x), dtype=float)
    correction = CORRECTION_SHRINKAGE * np.clip(raw, -PREDICTION_RESIDUAL_CLIP, PREDICTION_RESIDUAL_CLIP)
    if not np.isfinite(raw).all() or not np.isfinite(correction).all():
        raise RuntimeError("R154 audit refit returned non-finite residuals")
    return raw, correction, {"feature_columns": list(train_x.columns), "categorical_features": categorical,
                             "effective_config": {k: v for k, v in model.get_params().items() if k in MODEL_CONFIG}}


def safe_frame(protocol: dict, inputs: dict, stage: str) -> pd.DataFrame:
    targets = SCREEN_MONTHS if stage == "screen" else tuple(protocol["development_months"])
    reads = tuple(sorted(set(SEED_MONTHS) | {month for month in protocol["development_months"] if month <= max(targets)}))
    if set(reads) & set(protocol["reserved_months"]):
        raise RuntimeError("R154 audit filters include a reserved month")
    data_path = ROOT / protocol["data_path"]
    if sha256(data_path) != protocol["data_sha256"]:
        raise RuntimeError("R154 audit data hash differs from protocol")
    filters = []
    for month in reads:
        start = pd.Timestamp(month + "-01")
        filters.append([("date_sold", ">=", start), ("date_sold", "<", start + pd.offsets.MonthBegin(1))])
    columns = ["ml_num", "date_sold", "date_listed", "sold_price", "list_price", "municipality",
               "community", "house_type_name", "house_style", "garage", "house_area", "bedrooms",
               "washrooms", "parking_total", "latitude", "longitude"]
    metadata = pd.read_parquet(data_path, columns=columns, filters=filters)
    metadata["date_sold"] = pd.to_datetime(metadata.date_sold, errors="raise")
    metadata["date_listed"] = pd.to_datetime(metadata.date_listed, errors="raise")
    metadata["month"] = metadata.date_sold.dt.to_period("M").astype(str)
    metadata["ml_num"] = metadata.ml_num.astype(str)
    metadata["house_type_name"] = metadata.house_type_name.astype("string").fillna("__MISSING__").astype(str)
    metadata["source_sold_price"] = pd.to_numeric(metadata.sold_price, errors="raise")
    if set(metadata.month.unique()) != set(reads) or metadata.duplicated(["month", "ml_num"]).any():
        raise RuntimeError("R154 audit metadata month coverage or key uniqueness failed")

    r35 = pd.read_csv(ROOT / inputs["r35"]["predictions_path"], dtype={"ml_num": str})
    r35 = r35.loc[r35.track.eq(TRACK) & r35.month.astype(str).isin(reads),
                  ["month", "ml_num", "sold_price", "candidate_prediction"]].copy()
    r35["month"], r35["ml_num"] = r35.month.astype(str), r35.ml_num.astype(str)
    r35["base_prediction"] = pd.to_numeric(r35.candidate_prediction, errors="raise")
    r35 = r35.drop(columns="candidate_prediction")
    dev = r35.loc[r35.month.isin(protocol["development_months"])].merge(
        metadata.loc[metadata.month.isin(protocol["development_months"])],
        on=["month", "ml_num"], how="outer", validate="one_to_one", indicator=True,
        suffixes=("", "_source"),
    )
    if (not dev._merge.eq("both").all()
            or not np.array_equal(dev.sold_price.to_numpy(float), dev.source_sold_price.to_numpy(float))):
        raise RuntimeError("R154 audit R35 keys/labels differ from safe metadata")
    dev = dev.drop(columns=["_merge", "source_sold_price"])

    seed = pd.read_csv(ROOT / inputs["r44"]["seed_predictions_path"], dtype={"ml_num": str})
    seed = seed.loc[seed.track.eq(TRACK), ["month", "ml_num", "date_sold", "sold_price", "variant_prediction"]].copy()
    seed["month"], seed["ml_num"] = seed.month.astype(str), seed.ml_num.astype(str)
    seed["date_sold"] = pd.to_datetime(seed.date_sold, errors="raise")
    seed["base_prediction"] = pd.to_numeric(seed.variant_prediction, errors="raise")
    seed = seed.drop(columns="variant_prediction")
    seed_rows = seed.merge(metadata, on=["month", "ml_num"], how="left", validate="one_to_one",
                           indicator=True, suffixes=("_seed", "_source"))
    if (not seed_rows._merge.eq("both").all()
            or not np.array_equal(seed_rows.date_sold_seed.to_numpy("datetime64[ns]"), seed_rows.date_sold_source.to_numpy("datetime64[ns]"))
            or not np.array_equal(seed_rows.sold_price_seed.to_numpy(float), seed_rows.source_sold_price.to_numpy(float))):
        raise RuntimeError("R154 audit R44 seed keys/dates/labels differ from safe metadata")
    for column in ["date_listed", "list_price", "municipality", "community", "house_type_name", "house_style",
                   "garage", "house_area", "bedrooms", "washrooms", "parking_total", "latitude", "longitude"]:
        if f"{column}_source" in seed_rows:
            seed_rows[column] = seed_rows[f"{column}_source"]
    seed_rows = seed_rows.rename(columns={"date_sold_seed": "date_sold", "sold_price_seed": "sold_price"})
    keep = ["month", "ml_num", "date_sold", "date_listed", "sold_price", "base_prediction", "list_price",
            "municipality", "community", "house_type_name", "house_style", "garage", "house_area",
            "bedrooms", "washrooms", "parking_total", "latitude", "longitude"]
    frame = pd.concat([seed_rows[keep], dev[keep]], ignore_index=True)
    if set(frame.month.unique()) != set(reads) or frame.duplicated(["month", "ml_num"]).any():
        raise RuntimeError("R154 audit combined seed/R35 coverage failed")
    return frame.sort_values(["month", "ml_num"], kind="mergesort").reset_index(drop=True)


def score(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    ape = 100.0 * np.abs(predicted - actual) / actual
    return {"n": int(len(ape)), "mdape_percent": float(np.median(ape)), "mape_percent": float(np.mean(ape))}


def rescore(paired: pd.DataFrame, months: list[str]) -> dict:
    actual = paired.sold_price.to_numpy(float)
    parent = paired.parent_prediction.to_numpy(float)
    candidate = paired.candidate_prediction.to_numpy(float)
    pa, ca = 100.0 * np.abs(parent - actual) / actual, 100.0 * np.abs(candidate - actual) / actual
    pmetrics, cmetrics = score(actual, parent), score(actual, candidate)
    gain = 100.0 * (pmetrics["mdape_percent"] - cmetrics["mdape_percent"]) / pmetrics["mdape_percent"]
    mape_change = 100.0 * (cmetrics["mape_percent"] - pmetrics["mape_percent"]) / pmetrics["mape_percent"]
    monthly, groups = {}, []
    for month in months:
        idx = np.flatnonzero(paired.month.astype(str).eq(month).to_numpy())
        groups.append(idx)
        p, c = score(actual[idx], parent[idx]), score(actual[idx], candidate[idx])
        monthly[month] = {"n": len(idx), "parent": p, "candidate": c,
                          "relative_mdape_improvement_percent": 100.0 * (p["mdape_percent"] - c["mdape_percent"]) / p["mdape_percent"]}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    diffs = np.empty(BOOTSTRAP_REPLICATES)
    for i in range(BOOTSTRAP_REPLICATES):
        a, b = [], []
        for idx in groups:
            sample = idx[rng.integers(0, len(idx), size=len(idx))]
            a.append(pa[sample]); b.append(ca[sample])
        diffs[i] = float(np.median(np.concatenate(b)) - np.median(np.concatenate(a)))
    lo, hi = map(float, np.quantile(diffs, [0.025, 0.975]))
    monthly_gains = [v["relative_mdape_improvement_percent"] for v in monthly.values()]
    months_improved = sum(v > 0 for v in monthly_gains)
    return {"n": len(paired), "months": months, "parent": pmetrics, "candidate": cmetrics,
            "relative_mdape_improvement_percent": float(gain), "mape_relative_change_percent": float(mape_change),
            "monthly": monthly, "months_improved": int(months_improved),
            "worst_month_regression_percent": float(max(0.0, -min(monthly_gains))),
            "bootstrap_mdape_difference_percent": {"lower95": lo, "upper95": hi},
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "qualifies_development": bool(gain >= (0.25 if len(months) == 2 else 0.5)
                                           and months_improved >= (len(months) if len(months) == 2 else 6)
                                           and max(0.0, -min(monthly_gains)) <= (0.0 if len(months) == 2 else 5.0)
                                           and mape_change <= 1.0 and hi < 0.0),
            "target_mdape_strictly_below": 4.0, "target_met": bool(cmetrics["mdape_percent"] < 4.0)}


def main() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    progress = json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))
    card = protocol["cards"][CARD_KEY]
    if protocol.get("reserve_validation_eligible") is not False:
        raise RuntimeError("R154 reserve validation must remain ineligible")
    if protocol["budget_policy"] != "unbounded_until_objective_and_validation_complete" or any(
        v is not None for k, v in protocol["limits"].items() if k != "learner_threads"
    ):
        raise RuntimeError("R154 campaign budget policy is not the registered uncapped policy")
    if card.get("lightgbm_version") != lgb.__version__:
        raise RuntimeError("R154 LightGBM runtime version differs from registration")
    code_ok = all(sha256(ROOT / path) == digest for path, digest in card["code_sha256"].items())
    if not code_ok:
        raise RuntimeError("R154 registered source/auditor hashes changed")
    for name, fields in card["input_artifacts"].items():
        hash_fields = {"result_path": "result_sha256", "predictions_path": "predictions_sha256",
                       "manifest_path": "manifest_sha256", "seed_predictions_path": "seed_predictions_sha256",
                       "paired_predictions_path": "paired_predictions_sha256"}
        for field, digest_key in hash_fields.items():
            if field in fields and sha256(ROOT / fields[field]) != fields[digest_key]:
                raise RuntimeError(f"R154 registered input hash mismatch: {name}/{field}")

    stages = []
    for stage in ("screen", "development"):
        result_key = f"{stage}_result_path"
        if result_key in card:
            stages.append((stage, ROOT / card[result_key], card[f"{stage}_run_id"]))
    if not stages:
        raise RuntimeError("R154 has no completed stage to audit")
    ledger = [json.loads(line) for line in LEDGER_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    stage_audits = {}
    for stage, result_path, run_id in stages:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        pair_path, manifest_path = result_path.parent / "paired_predictions.csv", result_path.parent / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        months = list(SCREEN_MONTHS if stage == "screen" else protocol["development_months"])
        reads = list(sorted(set(SEED_MONTHS) | {m for m in protocol["development_months"] if m <= max(months)}))
        if (result.get("run_id") != run_id or result.get("card") != CARD_KEY or result.get("stage") != stage
                or result.get("months") != months or result.get("filtered_source_months") != reads
                or result.get("input_artifacts") != card["input_artifacts"]
                or result.get("reserved_months_labels_opened") is not False or result.get("hidden_labels_opened") is not False
                or manifest.get("run_id") != run_id or manifest.get("card") != CARD_KEY
                or manifest.get("stage") != stage or manifest.get("code_sha256") != card["code_sha256"]
                or manifest.get("data_sha256") != protocol["data_sha256"]
                or manifest.get("input_artifacts") != card["input_artifacts"]
                or manifest.get("filtered_source_months") != reads
                or manifest.get("result_sha256") != sha256(result_path)
                or manifest.get("paired_predictions_sha256") != sha256(pair_path)
                or manifest.get("reserved_months_labels_opened") is not False
                or manifest.get("hidden_labels_opened") is not False):
            raise RuntimeError(f"R154 {stage} result/manifest/hash/access verification failed")

        frame = safe_frame(protocol, card["input_artifacts"], stage)
        saved = pd.read_csv(pair_path, dtype={"ml_num": str})
        if saved.duplicated(["month", "ml_num"]).any() or set(saved.month.astype(str).unique()) != set(months):
            raise RuntimeError(f"R154 {stage} saved rows have duplicate keys or wrong months")
        rebuilt_parts, model_info_by_month = [], {}
        for month in months:
            cutoff = pd.Timestamp(month + "-01")
            history = frame.loc[pd.to_datetime(frame.date_sold).lt(cutoff)].copy()
            history = history.sort_values(["month", "ml_num"], kind="mergesort").reset_index(drop=True)
            query = frame.loc[frame.month.eq(month)].copy().sort_values("ml_num", kind="mergesort").reset_index(drop=True)
            if (card.get("min_training_oof_rows") != MIN_TRAINING_OOF_ROWS
                    or len(history) < MIN_TRAINING_OOF_ROWS
                    or not pd.to_datetime(history.date_sold).lt(cutoff).all()):
                raise RuntimeError("R154 audit detected insufficient or non-past training history")
            raw, correction, info = refit_predict(history, query)
            base = query.base_prediction.to_numpy(float)
            candidate = np.rint(base * np.exp(correction))
            replay = pd.DataFrame({"month": month, "ml_num": query.ml_num.astype(str),
                                   "sold_price": query.sold_price.to_numpy(float),
                                   "parent_prediction": base, "base_prediction": base,
                                   "raw_residual_prediction": raw, "correction_log": correction,
                                   "candidate_prediction": candidate})
            rebuilt_parts.append(replay)
            model_info_by_month[month] = {"history_rows": int(len(history)),
                                          "history_months": sorted(history.month.astype(str).unique().tolist()),
                                          **info}
        rebuilt = pd.concat(rebuilt_parts, ignore_index=True)
        joined = rebuilt.merge(saved, on=["month", "ml_num"], how="outer", validate="one_to_one",
                               suffixes=("_audit", "_saved"), indicator=True)
        if not joined._merge.eq("both").all() or len(joined) != len(saved):
            raise RuntimeError(f"R154 {stage} keys differ from the audit replay")
        for col in ("sold_price", "parent_prediction", "base_prediction", "candidate_prediction"):
            if not np.array_equal(joined[f"{col}_audit"].to_numpy(float), joined[f"{col}_saved"].to_numpy(float)):
                raise RuntimeError(f"R154 {stage} {col} differs from deterministic LightGBM refit")
        raw_delta = float(np.max(np.abs(joined.raw_residual_prediction_audit.to_numpy(float)
                                        - joined.raw_residual_prediction_saved.to_numpy(float))))
        correction_delta = float(np.max(np.abs(joined.correction_log_audit.to_numpy(float)
                                               - joined.correction_log_saved.to_numpy(float))))
        if max(raw_delta, correction_delta) > 1e-12:
            raise RuntimeError(f"R154 {stage} learner replay mismatch: {raw_delta}, {correction_delta}")
        metrics_input = joined.rename(columns={"sold_price_audit": "sold_price",
                                               "parent_prediction_audit": "parent_prediction",
                                               "candidate_prediction_audit": "candidate_prediction"})
        metrics = rescore(metrics_input, months)
        expected = result["scores"]
        numeric_deltas = [abs(float(metrics[key]) - float(expected[key])) for key in
                          ("relative_mdape_improvement_percent", "mape_relative_change_percent", "months_improved",
                           "worst_month_regression_percent")]
        numeric_deltas.extend(abs(float(metrics[key][sub]) - float(expected[key][sub]))
                              for key in ("parent", "candidate") for sub in ("mdape_percent", "mape_percent"))
        if max(numeric_deltas) > 1e-12:
            raise RuntimeError("R154 independent rescoring differs from result file")
        events = [e for e in ledger if e.get("card") == CARD_KEY and e.get("run_id") == run_id]
        starts = [e for e in events if e.get("event") == "fit_start"]
        completes = [e for e in events if e.get("event") == "fit_complete"]
        if len(starts) != len(months) or len(completes) != len(months):
            raise RuntimeError("R154 fit starts/completions do not reconcile")
        for event in starts:
            if (event.get("reserved_months_labels_opened") is not False
                    or event.get("hidden_labels_opened") is not False
                    or event.get("query_label_columns_passed") != []
                    or any(m >= event["month"] for m in event.get("training_oof_months", []))):
                raise RuntimeError("R154 ledger reveals unsafe query labels or history")
        audit = {
            "card_key": CARD_KEY, "stage": stage, "run_id": run_id,
            "registered_code_hashes_match": code_ok, "registered_input_hashes_match": True,
            "lightgbm_version": lgb.__version__, "model_config": MODEL_CONFIG,
            "safe_month_filter_only": reads, "seed_months": list(SEED_MONTHS),
            "target_months": months, "rows": int(len(joined)),
            "query_target_labels_excluded": True, "strict_past_history_verified": True,
            "deterministic_refit_prediction_max_abs_difference": float(max(raw_delta, correction_delta)),
            "candidate_predictions_match_exactly": True, "parent_predictions_match_exactly": True,
            "independent_score_max_abs_difference": float(max(numeric_deltas)),
            "fit_starts": len(starts), "fit_completes": len(completes),
            "campaign_total_fit_starts": progress["fit_starts"],
            "reserve_or_hidden_access_by_r154": False,
            "campaign_had_prior_reserve_access_incident": bool(protocol.get("reserved_months_labels_opened")),
            "reserve_validation_eligible": False,
            "pooled_parent": metrics["parent"], "pooled_candidate": metrics["candidate"],
            "monthly": metrics["monthly"], "qualifies": result["qualifies"],
            "target_below_4_percent_met": metrics["target_met"],
            "fold_replay": model_info_by_month,
        }
        audit["audit_script_sha256"] = sha256(Path(__file__))
        (result_path.parent / "independent_audit.json").write_text(
            json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
        )
        stage_audits[stage] = audit
    print(json.dumps({"audited_stages": list(stage_audits),
                      "audit_paths": [str((ROOT / card[f"{stage}_result_path"]).parent / "independent_audit.json")
                                      for stage in stage_audits]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
