from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from experiments.bill_target74 import r03_catboost as r03
from experiments.bill_target74 import r04_ablation as provenance
from experiments.bill_target74 import r97_past_oof_catboost_residual_meta as r97
from experiments.bill_target74 import r154_listaware_lgbm_quantile_residual as r154
from experiments.bill_target74 import runner as campaign


CARD = "R161_listaware_r97_catboost_seed_oof_history"
AUDITOR_PATH = Path(__file__).with_name("r161_independent_audit.py")
TRACK = "list_aware"
SCREEN_MONTHS = tuple(campaign.PROTOCOL["screen_months"])
DEVELOPMENT_MONTHS = tuple(campaign.PROTOCOL["development_months"])
SEED_MONTHS = tuple(r154.SEED_MONTHS)
MIN_HISTORY_ROWS = 3500
RESIDUAL_TARGET_CLIP = 0.15
PREDICTION_RESIDUAL_CLIP = 0.10
CORRECTION_SHRINKAGE = 0.50
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20261161
R97_RUN_ID = "r97m_20261025T210147Z"
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


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(campaign.ROOT.resolve())).replace("\\", "/")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _source_files() -> list[Path]:
    return [Path(__file__), AUDITOR_PATH, Path(r97.__file__), Path(r154.__file__),
            campaign.PACKAGE / "r03_catboost.py", campaign.PACKAGE / "r04_ablation.py",
            campaign.PACKAGE / "runner.py", campaign.ROOT / "experiments/bill_greedy/runner.py",
            campaign.ROOT / "experiments/bill_greedy/history.py",
            campaign.ROOT / "model_candidate_v2/features.py"]


def _code_hashes() -> dict[str, str]:
    return {_relative(path): _sha(path) for path in _source_files()}


def _r97_current_best() -> tuple[pd.DataFrame, dict]:
    result_path = R97_DIR / "development_results.json"
    paired_path = R97_DIR / "paired_predictions.csv"
    manifest_path = R97_DIR / "run_manifest.json"
    if not all(path.is_file() for path in (result_path, paired_path, manifest_path)):
        raise FileNotFoundError("R161 requires the saved R97 development comparison")
    result, manifest = _read_json(result_path), _read_json(manifest_path)
    if (result.get("run_id") != R97_RUN_ID or result.get("card") != r97.CARD
            or result.get("campaign_id") != campaign.PROTOCOL["campaign_id"]
            or result.get("data_sha256") != campaign.PROTOCOL["data_sha256"]
            or manifest.get("run_id") != R97_RUN_ID
            or manifest.get("paired_predictions_sha256") != _sha(paired_path)
            or manifest.get("development_results_sha256") != _sha(result_path)
            or manifest.get("source_identity", {}).get("data_sha256") != campaign.PROTOCOL["data_sha256"]
            or result.get("reserved_months_labels_opened") is not False
            or result.get("hidden_labels_opened") is not False
            or manifest.get("reserved_months_labels_opened") is not False
            or manifest.get("hidden_labels_opened") is not False):
        raise RuntimeError("R97 current-best artifact identity, hashes, or safe-label flags failed")
    paired = pd.read_csv(paired_path, dtype={"ml_num": str})
    paired = paired.loc[paired.track.eq(TRACK),
                        ["month", "track", "ml_num", "sold_price", "base_prediction", "candidate_prediction"]].copy()
    paired["month"] = paired.month.astype(str)
    paired["ml_num"] = paired.ml_num.astype(str)
    if (paired.empty or paired.duplicated(["month", "ml_num"]).any()
            or set(paired.month) != set(DEVELOPMENT_MONTHS)):
        raise RuntimeError("R97 current-best paired predictions are incomplete or duplicated")
    return paired, {"result_path": _relative(result_path), "result_sha256": _sha(result_path),
                    "paired_predictions_path": _relative(paired_path),
                    "paired_predictions_sha256": _sha(paired_path),
                    "manifest_path": _relative(manifest_path), "manifest_sha256": _sha(manifest_path),
                    "run_id": R97_RUN_ID, "card": r97.CARD}


def _preflight(stage: str) -> tuple[dict, pd.DataFrame, pd.DataFrame, dict, dict]:
    inputs = r154._input_artifacts()
    frame = r154._load_frame(inputs, stage)
    best, best_identity = _r97_current_best()
    if frame.duplicated(["month", "ml_num"]).any():
        raise RuntimeError("combined R44/R35 OOF input contains duplicate month/listing keys")
    if set(frame.month.astype(str)) & set(campaign.PROTOCOL["reserved_months"]):
        raise RuntimeError("R161 input frame contains a reserved month")
    expected_months = set((*SEED_MONTHS, *[m for m in DEVELOPMENT_MONTHS
                                           if m <= (max(SCREEN_MONTHS) if stage == "screen"
                                                    else max(DEVELOPMENT_MONTHS))]))
    if set(frame.month.astype(str)) != expected_months:
        raise RuntimeError(f"R161 read unexpected month scope: {sorted(frame.month.astype(str).unique())}")
    if "track" in frame.columns and not frame.track.eq(TRACK).all():
        raise RuntimeError("R161 safe input unexpectedly contains another track")
    history_counts = {}
    for month in SCREEN_MONTHS if stage == "screen" else DEVELOPMENT_MONTHS:
        cutoff = pd.Timestamp(month + "-01")
        history = frame.loc[pd.to_datetime(frame.date_sold).lt(cutoff)]
        if (pd.to_datetime(history.date_sold) >= cutoff).any():
            raise RuntimeError(f"R161 history crosses the strict cutoff for {month}")
        history_counts[month] = {"rows": int(len(history)),
                                 "months": sorted(history.month.astype(str).unique().tolist())}
    if stage == "screen" and history_counts.get("2023-03", {}).get("rows", 0) < MIN_HISTORY_ROWS:
        raise RuntimeError("2023-03 lacks the registered 3,500 strict-past OOF rows")
    if stage == "screen" and history_counts.get("2024-06", {}).get("rows", 0) < MIN_HISTORY_ROWS:
        raise RuntimeError("2024-06 lacks the registered 3,500 strict-past OOF rows")
    return inputs, frame, best, best_identity, history_counts


def _card_spec(inputs: dict, best_identity: dict, source: dict,
               history_counts: dict) -> dict:
    return {
        "status": "registered", "track": TRACK,
        "screen_months": list(SCREEN_MONTHS), "development_months": list(DEVELOPMENT_MONTHS),
        "seed_months": list(SEED_MONTHS),
        "expected_screen_fit_starts": len(SCREEN_MONTHS),
        "expected_development_fit_starts": len(DEVELOPMENT_MONTHS),
        "baseline_by_track": {TRACK: "validated R35 list-aware OOF"},
        "current_best_comparator": best_identity,
        "minimum_past_oof_rows": MIN_HISTORY_ROWS,
        "method": ("Fit the frozen R97 CatBoost MAE residual learner on R44 R35 seed OOF rows "
                   "and R35 safe development OOF rows strictly earlier than each target fold. "
                   "Use R97 features, clipped log residual target, 0.50 correction shrinkage and "
                   "the fixed R97 prediction clip; only expand the verified history source."),
        "difference_from_prior_work": ("R97 used CatBoost MAE but had no pre-2023 seed OOF rows, "
                                       "so its first two folds remained unchanged. R154 used the "
                                       "same R44/R35 history source and R97 feature schema but "
                                       "LightGBM Quantile, which failed; R161 tests CatBoost MAE "
                                       "on that history and the fixed 3,500-row support floor."),
        "parent_oof_inputs": inputs,
        "r97_current_best_artifacts": best_identity,
        "source_preflight": {"strict_past_rows_by_screen_month": history_counts},
        "catboost_config": CATBOOST_CONFIG,
        "residual_target": "clip(log(sold_price / R35_prediction), -0.15, 0.15)",
        "correction": {"formula": "round(R35_prediction * exp(0.50 * clip(predicted_log_residual, -0.10, 0.10)))",
                       "residual_target_clip": RESIDUAL_TARGET_CLIP,
                       "prediction_residual_clip": PREDICTION_RESIDUAL_CLIP,
                       "shrinkage": CORRECTION_SHRINKAGE,
                       "rounding": "nearest dollar"},
        "features": {"builder": "r97._build_features", "categorical_features": list(r97.CATEGORICAL_FEATURES),
                     "forbidden": ["sold_price", "date_sold", "external_avm_estimate", "remarks"]},
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


def _register(inputs: dict, best_identity: dict, source: dict,
              history_counts: dict) -> dict:
    hashes = _code_hashes()
    existing = campaign.PROTOCOL["cards"].get(CARD)
    if existing is not None:
        if (existing.get("source_sha256") != hashes
                or existing.get("parent_oof_inputs") != inputs
                or existing.get("r97_current_best_artifacts") != best_identity):
            raise RuntimeError("R161 frozen source or input identities changed after registration")
        return existing
    card = _card_spec(inputs, best_identity, source, history_counts)
    campaign.PROTOCOL["cards"][CARD] = card
    campaign.PROTOCOL_PATH.write_text(json.dumps(campaign.PROTOCOL, indent=2, ensure_ascii=False) + "\n",
                                      encoding="utf-8")
    campaign._append({"event": "card_registered", "card": CARD,
                      "source_sha256": hashes[_relative(Path(__file__))],
                      "parent_oof_inputs": inputs, "r97_current_best_artifacts": best_identity,
                      "strict_past_rows_by_screen_month": history_counts,
                      "expected_screen_fit_starts": len(SCREEN_MONTHS),
                      "reserved_months_labels_opened": False, "hidden_labels_opened": False})
    campaign._write_progress("r161_registered",
                             next_action="run R161 fixed list-aware CatBoost seed-OOF screen",
                             r161_card=CARD)
    return card


def _metric(actual: np.ndarray, prediction: np.ndarray) -> dict:
    ape = np.abs(prediction - actual) / actual * 100.0
    return {"n": int(len(actual)), "mdape_percent": float(np.median(ape)),
            "mape_percent": float(np.mean(ape))}


def _bootstrap_upper(rows: pd.DataFrame, reference: str, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    groups = [part for _, part in rows.groupby("month", sort=True)]
    diffs = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    for replicate in range(BOOTSTRAP_REPLICATES):
        before_parts, after_parts = [], []
        for part in groups:
            ix = rng.integers(0, len(part), size=len(part))
            actual = part.sold_price.to_numpy(float)[ix]
            before = part[reference].to_numpy(float)[ix]
            after = part.candidate_prediction.to_numpy(float)[ix]
            before_parts.append(np.abs(before - actual) / actual * 100.0)
            after_parts.append(np.abs(after - actual) / actual * 100.0)
        diffs[replicate] = np.median(np.concatenate(after_parts)) - np.median(np.concatenate(before_parts))
    return {"lower95": float(np.quantile(diffs, .025)),
            "upper95": float(np.quantile(diffs, .975)),
            "replicates": BOOTSTRAP_REPLICATES, "seed": seed}


def _score(paired: pd.DataFrame, reference: str, months: tuple[str, ...], stage: str) -> dict:
    rows = paired.copy()
    actual = rows.sold_price.to_numpy(float)
    ref = rows[reference].to_numpy(float)
    candidate = rows.candidate_prediction.to_numpy(float)
    if not np.isfinite(np.column_stack([actual, ref, candidate])).all() \
            or (np.column_stack([actual, ref, candidate]) <= 0.0).any():
        raise RuntimeError("R161 scoring received non-finite or non-positive prices")
    benchmark, challenger = _metric(actual, ref), _metric(actual, candidate)
    gain = 100.0 * (benchmark["mdape_percent"] - challenger["mdape_percent"]) / benchmark["mdape_percent"]
    mape_change = 100.0 * (challenger["mape_percent"] - benchmark["mape_percent"]) / benchmark["mape_percent"]
    monthly = {}
    month_gains = []
    for month in months:
        part = rows.loc[rows.month.astype(str).eq(month)]
        before = _metric(part.sold_price.to_numpy(float), part[reference].to_numpy(float))
        after = _metric(part.sold_price.to_numpy(float), part.candidate_prediction.to_numpy(float))
        month_gain = 100.0 * (before["mdape_percent"] - after["mdape_percent"]) / before["mdape_percent"]
        month_gains.append(month_gain)
        monthly[month] = {"benchmark": before, "candidate": after,
                          "relative_mdape_improvement_percent": month_gain}
    bootstrap = _bootstrap_upper(rows, reference, BOOTSTRAP_SEED)
    gates = campaign.PROTOCOL["gates"]
    if stage == "screen":
        checks = {"minimum_relative_mdape_improvement": gain >= gates["screen_relative_mdape_improvement_percent"],
                  "both_screen_months_improved": all(value > 0.0 for value in month_gains),
                  "mape_guard": mape_change <= gates["screen_mape_regression_max_percent"],
                  "paired_bootstrap_upper95_below_zero": bootstrap["upper95"] < 0.0}
    else:
        target = 4.0
        worst = max([0.0] + [-value for value in month_gains])
        checks = {"development_relative_gain": gain >= gates["development_relative_mdape_improvement_percent"],
                  "months_improved": sum(value > 0 for value in month_gains) >= gates["development_months_improved_min"],
                  "worst_month_guard": worst <= gates["development_worst_month_regression_max_percent"],
                  "mape_guard": mape_change <= gates["development_mape_regression_max_percent"],
                  "paired_bootstrap_upper95_below_zero": bootstrap["upper95"] < 0.0,
                  "strict_listaware_target": challenger["mdape_percent"] < target}
    return {"n": len(rows), "benchmark": benchmark, "candidate": challenger,
            "relative_mdape_improvement_percent": float(gain),
            "mape_relative_change_percent": float(mape_change),
            "months_improved": int(sum(value > 0.0 for value in month_gains)),
            "worst_month_regression_percent": float(max([0.0] + [-value for value in month_gains])),
            "paired_bootstrap_mdape_difference_pp": bootstrap, "monthly": monthly,
            "gate_checks": checks, "qualifies": all(checks.values())}


def _fit_fold(history: pd.DataFrame, query: pd.DataFrame, run_id: str, stage: str,
              output: Path, source: dict, fit_details: list[dict], month: str) -> pd.DataFrame:
    cutoff = pd.Timestamp(month + "-01")
    history_dates = pd.to_datetime(history.date_sold, errors="raise")
    if history.empty or history_dates.ge(cutoff).any():
        raise RuntimeError(f"R161 history is empty or violates strict cutoff for {month}")
    if len(history) < MIN_HISTORY_ROWS:
        raise RuntimeError(f"R161 requires {MIN_HISTORY_ROWS} strict-past rows for {month}; found {len(history)}")
    if history.duplicated(["month", "ml_num"]).any() or query.ml_num.duplicated().any():
        raise RuntimeError(f"R161 has duplicate history/query IDs for {month}")
    train_x, cats = r97._build_features(history, TRACK)
    query_x, query_cats = r97._build_features(query, TRACK)
    if (cats != query_cats or list(train_x.columns) != list(query_x.columns)
            or {"sold_price", "date_sold"} & set(train_x.columns)):
        raise RuntimeError(f"R161 train/query feature schema or target exclusion failed for {month}")
    raw = np.log(history.sold_price.to_numpy(float) / history.base_prediction.to_numpy(float))
    if not np.isfinite(raw).all():
        raise RuntimeError(f"R161 residual target is non-finite for {month}")
    target = np.clip(raw, -RESIDUAL_TARGET_CLIP, RESIDUAL_TARGET_CLIP)
    fit_id = f"{campaign.PROTOCOL['campaign_id']}:{CARD}:{TRACK}:{month}:{stage}:{run_id}"
    campaign._append({"event": "fit_start", "card": CARD, "run_id": run_id,
                      "stage": stage, "fit_id": fit_id, "track": TRACK, "month": month,
                      "training_oof_rows": len(history),
                      "training_oof_keys_sha256": _keys_sha256(history),
                      "query_keys_sha256": _keys_sha256(query),
                      "training_oof_months": sorted(history.month.astype(str).unique().tolist()),
                      "history_cutoff_exclusive": cutoff.isoformat(),
                      "data_sha256": source["data_sha256"],
                      "reserved_months_labels_opened": False, "hidden_labels_opened": False})
    started = datetime.now(timezone.utc)
    try:
        model = CatBoostRegressor(**CATBOOST_CONFIG)
        model.fit(train_x, target, cat_features=cats)
        raw_prediction = np.asarray(model.predict(query_x), dtype=float)
        correction = CORRECTION_SHRINKAGE * np.clip(raw_prediction,
                                                   -PREDICTION_RESIDUAL_CLIP,
                                                   PREDICTION_RESIDUAL_CLIP)
        base = query.base_prediction.to_numpy(float)
        prediction = np.rint(base * np.exp(correction))
        if not np.isfinite(prediction).all() or (prediction <= 0.0).any():
            raise RuntimeError(f"R161 predictions are invalid for {month}")
        destination = output / "fold_predictions" / f"{month}.csv"
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame({"month": month, "track": TRACK,
                              "ml_num": query.ml_num.astype(str).to_numpy(),
                              "sold_price": query.sold_price.to_numpy(float),
                              "base_prediction": base,
                              "candidate_prediction": prediction,
                              "correction_log": correction,
                              "raw_predicted_log_residual": raw_prediction,
                              "history_oof_rows": len(history)})
        frame.to_csv(destination, index=False)
        detail = {"fit_id": fit_id, "track": TRACK, "stage": stage, "month": month,
                  "fit_started_utc": started.isoformat(),
                  "fit_seconds": (datetime.now(timezone.utc) - started).total_seconds(),
                  "training_oof_rows": int(len(history)),
                  "training_oof_keys_sha256": _keys_sha256(history),
                  "query_keys_sha256": _keys_sha256(query),
                  "training_oof_months": sorted(history.month.astype(str).unique().tolist()),
                  "history_cutoff_exclusive": cutoff.isoformat(),
                  "evaluation_rows": int(len(query)), "target": "clipped_log_sold_over_R35_prediction",
                  "model_family": "catboost", "model_config": CATBOOST_CONFIG,
                  "feature_columns": list(train_x.columns), "categorical_features": cats,
                  "residual_target_clip": RESIDUAL_TARGET_CLIP,
                  "prediction_clip": PREDICTION_RESIDUAL_CLIP,
                  "correction_shrinkage": CORRECTION_SHRINKAGE,
                  "prediction_file": _relative(destination),
                  "prediction_file_sha256": _sha(destination)}
        fit_details.append(detail)
        campaign._append({"event": "fit_complete", "card": CARD, "run_id": run_id,
                          "stage": stage, "fit_id": fit_id, "track": TRACK, "month": month,
                          "status": "complete", "seconds": detail["fit_seconds"],
                          "training_oof_rows": len(history),
                          "prediction_file_sha256": detail["prediction_file_sha256"],
                          "reserved_months_labels_opened": False, "hidden_labels_opened": False})
        return frame
    except BaseException as exc:
        campaign._append({"event": "fit_complete", "card": CARD, "run_id": run_id,
                          "stage": stage, "fit_id": fit_id, "track": TRACK, "month": month,
                          "status": "failed", "error": f"{type(exc).__name__}: {exc}",
                          "reserved_months_labels_opened": False, "hidden_labels_opened": False})
        raise


def _make_paired(frame: pd.DataFrame, best: pd.DataFrame,
                 months: tuple[str, ...]) -> pd.DataFrame:
    parts = []
    for month in months:
        candidate = pd.read_csv(frame.attrs["fold_paths"][month], dtype={"ml_num": str})
        base_rows = frame.loc[frame.month.astype(str).eq(month), ["month", "ml_num", "sold_price", "base_prediction"]].copy()
        best_rows = best.loc[best.month.astype(str).eq(month), ["ml_num", "sold_price", "candidate_prediction"]].copy()
        best_rows = best_rows.rename(columns={"candidate_prediction": "current_best_prediction"})
        base_rows["ml_num"] = base_rows.ml_num.astype(str)
        if (base_rows.ml_num.duplicated().any() or best_rows.ml_num.duplicated().any()
                or set(candidate.ml_num.astype(str)) != set(base_rows.ml_num)
                or set(best_rows.ml_num.astype(str)) != set(base_rows.ml_num)):
            raise RuntimeError(f"R161 row keys do not align for {month}")
        base_rows = base_rows.set_index("ml_num").reindex(candidate.ml_num.astype(str))
        best_rows = best_rows.set_index("ml_num").reindex(candidate.ml_num.astype(str))
        candidate_ids = candidate.ml_num.astype(str).to_numpy()
        if (base_rows.isna().any().any() or best_rows.isna().any().any()
                or not np.array_equal(base_rows.sold_price.to_numpy(float), candidate.sold_price.to_numpy(float))
                or not np.array_equal(best_rows.sold_price.to_numpy(float), candidate.sold_price.to_numpy(float))):
            raise RuntimeError(f"R161 heldout labels mismatch for {month}")
        parts.append(pd.DataFrame({"month": month, "track": TRACK, "ml_num": candidate_ids,
                                   "sold_price": candidate.sold_price.to_numpy(float),
                                   "baseline_prediction": base_rows.base_prediction.to_numpy(float),
                                   "current_best_prediction": best_rows.current_best_prediction.to_numpy(float),
                                   "candidate_prediction": candidate.candidate_prediction.to_numpy(float),
                                   "correction_log": candidate.correction_log.to_numpy(float),
                                   "history_oof_rows": candidate.history_oof_rows.to_numpy(int)}))
    paired = pd.concat(parts, ignore_index=True)
    if paired.duplicated(["month", "track", "ml_num"]).any():
        raise RuntimeError("R161 paired rows contain duplicate keys")
    return paired


def _write_artifacts(stage: str, run_id: str, output: Path, source: dict,
                     inputs: dict, best_identity: dict, fit_details: list[dict],
                     paired: pd.DataFrame, months: tuple[str, ...]) -> dict:
    paired_path = output / "paired_predictions.csv"
    paired.to_csv(paired_path, index=False)
    baseline = _score(paired, "baseline_prediction", months, stage)
    current_best = _score(paired, "current_best_prediction", months, stage)
    qualifies = baseline["qualifies"] and current_best["qualifies"] if stage == "screen" else baseline["qualifies"]
    result = {"campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
              "card": CARD, "stage": stage, "track": TRACK,
              "months": list(months), "registered_fit_starts": len(fit_details),
              "baseline_scores": baseline, "current_best_scores": current_best,
              "qualifies": bool(qualifies),
              "strict_target_mdape_percent": 4.0 if stage == "development" else None,
              "fit_details": fit_details, "data_sha256": source["data_sha256"],
              "input_artifacts": inputs, "current_best_artifacts": best_identity,
              "reserved_months_labels_opened": False, "hidden_labels_opened": False}
    result_path = output / ("screen_results.json" if stage == "screen" else "development_results.json")
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    provenance_dir = output / "provenance"
    provenance_dir.mkdir(exist_ok=True)
    source_hashes, source_snapshots = {}, {}
    source_paths = _source_files() + [campaign.PROTOCOL_PATH]
    for path in source_paths:
        rel = _relative(path)
        digest = _sha(path)
        snapshot_name = hashlib.sha256(rel.encode("utf-8")).hexdigest()[:12] + "_" + path.name + ".snapshot"
        snapshot = provenance_dir / snapshot_name
        shutil.copy2(path, snapshot)
        source_hashes[rel] = digest
        source_snapshots[rel] = _relative(snapshot)

    manifest = {"campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
                "card": CARD, "stage": stage, "source_identity": source,
                "source_sha256": source_hashes, "source_snapshot_paths": source_snapshots,
                "input_artifacts": inputs, "r97_current_best_artifacts": best_identity,
                "registered_fit_starts": len(fit_details), "fit_details": fit_details,
                "paired_predictions_sha256": _sha(paired_path),
                "result_sha256": _sha(result_path),
                "reserved_months_labels_opened": False, "hidden_labels_opened": False}
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report = ["# R161 list-aware CatBoost with expanded seed OOF history", "",
              f"Run: `{run_id}`; stage: `{stage}`; track: `{TRACK}`.",
              "Each target fold uses only earlier R44 seed OOF and R35 development OOF rows. The 2022 seed OOF base is R44's saved R35 candidate prediction.",
              "The registered R97 CatBoost MAE learner, feature schema, residual clipping, correction shrinkage and screen gates remained fixed.",
              "No reserved-month or hidden labels were opened.", ""]
    for label, score in (("R35 validated baseline", baseline), ("R97 current best", current_best)):
        report.extend([f"## Compared with {label}", "",
                       f"Baseline MdAPE: {score['benchmark']['mdape_percent']:.6f}%",
                       f"Candidate MdAPE: {score['candidate']['mdape_percent']:.6f}%",
                       f"Relative MdAPE improvement: {score['relative_mdape_improvement_percent']:+.4f}%",
                       f"MAPE relative change: {score['mape_relative_change_percent']:+.4f}%",
                       f"Months improved: {score['months_improved']}/{len(months)}",
                       f"Paired-bootstrap upper 95%: {score['paired_bootstrap_mdape_difference_pp']['upper95']:+.6f} pp",
                       f"Gate: {score['qualifies']}", ""])
        for month, detail in score["monthly"].items():
            report.append(f"- {month}: {detail['benchmark']['mdape_percent']:.5f}% → {detail['candidate']['mdape_percent']:.5f}%")
        report.append("")
    (output / "research_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

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
                      "run_id": run_id, "fit_starts": len(fit_details),
                      "result_path": _relative(result_path), "qualifies": bool(qualifies),
                      "baseline_gate": baseline["qualifies"],
                      "current_best_gate": current_best["qualifies"],
                      "reserved_months_labels_opened": False, "hidden_labels_opened": False})
    campaign._write_progress(f"r161_{stage}_pending_audit",
                             next_action="independently audit R161 before any development or promotion",
                             r161_card=CARD, r161_run_id=run_id,
                             r161_result=_relative(result_path), r161_preliminary_pass=bool(qualifies))
    return result


def run(stage: str = "screen") -> dict:
    if stage not in {"screen", "development"}:
        raise ValueError("stage must be screen or development")
    source = provenance._source_identity()
    inputs, frame, best, best_identity, history_counts = _preflight(stage)
    card = _register(inputs, best_identity, source, history_counts)
    if stage == "screen" and card.get("status") != "registered":
        raise RuntimeError("R161 screen has already run or is pending audit")
    if stage == "development" and (card.get("status") != "screen_passed"
                                    or card.get("screen_passed_tracks") != [TRACK]):
        raise RuntimeError("R161 development requires an independently audited passing screen")
    months = SCREEN_MONTHS if stage == "screen" else DEVELOPMENT_MONTHS
    run_id = f"r161{'s' if stage == 'screen' else 'd'}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    output = campaign.CAMPAIGN_ROOT / "batches" / run_id
    output.mkdir(parents=True, exist_ok=False)
    (output / "fold_predictions").mkdir()
    campaign._append({"event": "card_start", "card": CARD, "run_id": run_id,
                      "stage": stage, "months": list(months), "track": TRACK,
                      "expected_fit_starts": len(months), "data_sha256": source["data_sha256"],
                      "reserved_months_labels_opened": False, "hidden_labels_opened": False})
    details: list[dict] = []
    fold_paths = {}
    fold_rows = []
    try:
        for month in months:
            cutoff = pd.Timestamp(month + "-01")
            history = frame.loc[pd.to_datetime(frame.date_sold).lt(cutoff)].copy()
            query = frame.loc[frame.month.astype(str).eq(month)].copy().reset_index(drop=True)
            if query.empty or query.ml_num.duplicated().any():
                raise RuntimeError(f"R161 query rows are missing or duplicated for {month}")
            if (pd.to_datetime(history.date_sold) >= cutoff).any() \
                    or history.month.astype(str).isin(campaign.PROTOCOL["reserved_months"]).any():
                raise RuntimeError(f"R161 history crosses target or reserve cutoff for {month}")
            prediction_frame = _fit_fold(history, query, run_id, stage, output,
                                         source, details, month)
            fold_path = output / "fold_predictions" / f"{month}.csv"
            fold_paths[month] = fold_path
            fold_rows.append(query)
            campaign._append({"event": "fold_complete", "card": CARD, "run_id": run_id,
                              "stage": stage, "track": TRACK, "month": month,
                              "rows": len(prediction_frame), "history_rows": len(history),
                              "reserved_months_labels_opened": False, "hidden_labels_opened": False})
    except BaseException as exc:
        checkpoint = {"card": CARD, "run_id": run_id, "stage": stage,
                      "error": f"{type(exc).__name__}: {exc}", "fit_details": details,
                      "reserved_months_labels_opened": False, "hidden_labels_opened": False}
        (output / "failure_checkpoint.json").write_text(
            json.dumps(checkpoint, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        campaign._append({"event": "card_failed", "card": CARD, "run_id": run_id,
                          "stage": stage, "fit_starts": len(details),
                          "error": checkpoint["error"],
                          "reserved_months_labels_opened": False, "hidden_labels_opened": False})
        campaign._write_progress(f"r161_{stage}_failed",
                                 next_action="inspect R161 failure checkpoint and preserve completed fit artifacts",
                                 r161_card=CARD, r161_run_id=run_id)
        raise

    scoring_frame = pd.concat(fold_rows, ignore_index=True)
    scoring_frame.attrs["fold_paths"] = {month: str(path) for month, path in fold_paths.items()}
    paired = _make_paired(scoring_frame, best, months)
    return _write_artifacts(stage, run_id, output, source, inputs,
                            best_identity, details, paired, months)


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
