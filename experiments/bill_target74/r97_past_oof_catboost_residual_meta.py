"""R97: strict-past CatBoost model for residual correction on current OOF parents."""
from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from experiments.bill_target74 import r02_structure_matched_current_parent as r02
from experiments.bill_target74 import r93_spatial_oof_calibration as common
from experiments.bill_target74 import runner as campaign


CARD = "R97_past_oof_catboost_residual_meta"
TRACKS = common.TRACKS
MONTHS = common.MONTHS
MIN_HISTORY_ROWS = 4000
RESIDUAL_TARGET_CLIP = 0.15
PREDICTION_RESIDUAL_CLIP = 0.10
CORRECTION_SHRINKAGE = 0.50
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20261031

CATEGORICAL_FEATURES = [
    "house_type_name", "municipality", "community", "house_style", "garage",
]
NUMERIC_FEATURES = [
    "log_parent_prediction", "log_house_area", "bedrooms", "washrooms",
    "parking_total", "latitude", "longitude", "date_listed_month_index",
]
CATBOOST_CONFIG = {
    "loss_function": "MAE",
    "iterations": 500,
    "depth": 4,
    "learning_rate": 0.03,
    "l2_leaf_reg": 20.0,
    "random_strength": 1.0,
    "random_seed": 20261025,
    "thread_count": 4,
    "allow_writing_files": False,
    "verbose": False,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return str(path.relative_to(campaign.ROOT)).replace("\\", "/")


def _source_identity() -> dict:
    branch = subprocess.run(
        ["git", "branch", "--show-current"], cwd=campaign.ROOT,
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=campaign.ROOT,
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    if branch != "bill" or commit != campaign.PROTOCOL["parent_commit"]:
        raise RuntimeError(
            f"expected bill at {campaign.PROTOCOL['parent_commit']}, found {branch} at {commit}"
        )
    if _sha256(campaign.DATA) != campaign.PROTOCOL["data_sha256"]:
        raise RuntimeError("cached listings parquet hash differs from the campaign data hash")
    return {"branch": branch, "commit": commit, "data_sha256": campaign.PROTOCOL["data_sha256"]}


def _registration_spec(source: dict, parent_info: dict) -> dict:
    return {
        "status": "registered",
        "method": (
            "For each target development month, fit a CatBoost MAE learner to strictly earlier "
            "R52/R35 out-of-fold log residuals. Predict a bounded multiplicative correction from "
            "pre-sale structure, location, listing month, and parent prediction; use list-price "
            "features only on list-aware. Leave a fold unchanged until at least 4,000 earlier OOF rows exist."
        ),
        "difference_from_prior_work": (
            "R58 used a univariate linear median slope; R93 used a fixed local coordinate-neighbor median; "
            "R95 used municipality-by-house-type medians; R02 used nearest structural neighbors. "
            "R97 learns nonlinear multivariate conditional residuals with a fixed CatBoost MAE model, "
            "using only safe prior-month OOF rows."
        ),
        "target": "clip(log(sold_price / parent_prediction), -0.15, 0.15)",
        "parents": {
            "off_market": "R52 strict-past experimental OOF prediction",
            "list_aware": "R35 validated OOF prediction",
        },
        "parent_oof_source": parent_info,
        "development_months": list(MONTHS),
        "minimum_past_oof_rows": MIN_HISTORY_ROWS,
        "features": {
            "all_tracks": [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES],
            "list_aware_only": ["log_list_price", "log_list_to_parent"],
            "forbidden": ["sold_price", "date_sold", "external_avm_estimate", "remarks"],
            "date_sold_use": "month filtering, temporal cutoff, and scoring only; never a model feature",
        },
        "correction": {
            "formula": "round(parent_prediction * exp(0.50 * clip(predicted_log_residual, -0.10, 0.10)))",
            "prediction_clip": PREDICTION_RESIDUAL_CLIP,
            "shrinkage": CORRECTION_SHRINKAGE,
            "rounding": "nearest dollar",
        },
        "catboost_config": CATBOOST_CONFIG,
        "expected_fit_starts": 12,
        "development_gate": dict(campaign.PROTOCOL["gates"]),
        "paired_bootstrap": {"replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED},
        "data_sha256": source["data_sha256"],
        "parent_commit": source["commit"],
        "source_sha256": _sha256(Path(__file__)),
        "reserved_months_labels_opened": False,
        "hidden_labels_opened": False,
    }


def register_card() -> dict:
    source = _source_identity()
    parent_info = r02._verify_r93()
    card = _registration_spec(source, parent_info)
    existing = campaign.PROTOCOL["cards"].get(CARD)
    if existing:
        if existing.get("source_sha256") == card["source_sha256"]:
            if existing.get("status") != "registered":
                raise RuntimeError(f"{CARD} already has status {existing.get('status')!r}")
            return existing
        ledger = [
            json.loads(line) for line in campaign.LEDGER.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ] if campaign.LEDGER.exists() else []
        started = [row for row in ledger if row.get("card") == CARD and row.get("event") == "fit_start"]
        if started or existing.get("status") != "registered":
            raise RuntimeError("cannot amend R97 after a learner fit started or the card was completed")
        old_hash = existing.get("source_sha256")
        card["registration_amendments"] = list(existing.get("registration_amendments", [])) + [{
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "reason": "Before any fit, remove duplicate label columns from the metadata join and bind only the five parent prediction columns; method and parameters unchanged.",
            "previous_source_sha256": old_hash,
            "current_source_sha256": card["source_sha256"],
            "fit_starts": 0,
            "failed_setup_run_ids": [
                row.get("run_id") for row in ledger
                if row.get("card") == CARD and row.get("event") in {"card_failed", "card_setup_failed"}
            ],
        }]
        campaign.PROTOCOL["cards"][CARD] = card
        campaign.PROTOCOL_PATH.write_text(
            json.dumps(campaign.PROTOCOL, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        campaign._append({
            "event": "card_registration_amended", "card": CARD, "fit_starts": 0,
            "old_source_sha256": old_hash, "new_source_sha256": card["source_sha256"],
            "reason": "fix duplicate sold_price merge columns before fitting; method unchanged",
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        campaign._write_progress(
            "r97_registered", next_action="Run the amended strict-past CatBoost residual meta-learner",
            r97_card=CARD,
        )
        return card
    campaign.PROTOCOL["cards"][CARD] = card
    campaign.PROTOCOL_PATH.write_text(
        json.dumps(campaign.PROTOCOL, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    campaign._append({
        "event": "card_registered", "card": CARD, "fit_starts": 0,
        "source_sha256": card["source_sha256"], "data_sha256": source["data_sha256"],
        "parent_oof_source": parent_info,
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    campaign._write_progress(
        "r97_registered", next_action="Run the registered strict-past CatBoost residual meta-learner",
        r97_card=CARD,
    )
    return card


def _load_safe_metadata() -> pd.DataFrame:
    if _sha256(campaign.DATA) != campaign.PROTOCOL["data_sha256"]:
        raise RuntimeError("cached listings data hash differs from campaign protocol")
    filters = []
    for month in MONTHS:
        start = pd.Timestamp(month + "-01")
        filters.append([
            ("date_sold", ">=", start),
            ("date_sold", "<", start + pd.offsets.MonthBegin(1)),
        ])
    columns = [
        "ml_num", "date_sold", "date_listed", "sold_price", "list_price",
        "municipality", "community", "house_type_name", "house_style", "garage",
        "house_area", "bedrooms", "washrooms", "parking_total", "latitude", "longitude",
    ]
    frame = pd.read_parquet(campaign.DATA, columns=columns, filters=filters)
    frame["date_sold"] = pd.to_datetime(frame.date_sold, errors="raise")
    frame["date_listed"] = pd.to_datetime(frame.date_listed, errors="raise")
    frame["month"] = frame.date_sold.dt.to_period("M").astype(str)
    observed = sorted(frame.month.unique().tolist())
    if observed != sorted(MONTHS):
        raise RuntimeError(f"safe development filters returned unexpected months: {observed}")
    if set(observed).intersection(campaign.PROTOCOL["reserved_months"]):
        raise RuntimeError("reserved-month rows appeared in exact development filters")
    frame["ml_num"] = frame.ml_num.astype(str)
    if frame.duplicated(["month", "ml_num"]).any():
        raise RuntimeError("safe metadata contains duplicate month/listing keys")
    frame["source_sold_price"] = pd.to_numeric(frame.sold_price, errors="raise")
    frame = frame.drop(columns="sold_price")
    return frame


def _build_features(frame: pd.DataFrame, track: str) -> tuple[pd.DataFrame, list[str]]:
    parent = pd.to_numeric(frame.base_prediction, errors="raise").to_numpy(float)
    if not np.isfinite(parent).all() or (parent <= 0.0).any():
        raise RuntimeError(f"invalid parent predictions in {track} feature batch")
    listed = pd.to_datetime(frame.date_listed, errors="raise")
    area = pd.to_numeric(frame.house_area, errors="coerce").to_numpy(float)
    if (np.isfinite(area) & (area <= 0.0)).any():
        raise RuntimeError("non-positive house area encountered")
    listed_month = (listed.dt.year.to_numpy(int) * 12 + listed.dt.month.to_numpy(int)).astype(float)
    features = pd.DataFrame(index=frame.index)
    features["log_parent_prediction"] = np.log(parent)
    features["log_house_area"] = np.log(area)
    for column in ("bedrooms", "washrooms", "parking_total", "latitude", "longitude"):
        features[column] = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
    features["date_listed_month_index"] = listed_month
    for column in CATEGORICAL_FEATURES:
        features[column] = frame[column].astype("string").fillna("__MISSING__").astype(str)
    if track == "list_aware":
        list_price = pd.to_numeric(frame.list_price, errors="raise").to_numpy(float)
        if not np.isfinite(list_price).all() or (list_price <= 0.0).any():
            raise RuntimeError("invalid list price in list-aware feature batch")
        features["log_list_price"] = np.log(list_price)
        features["log_list_to_parent"] = np.log(list_price / parent)
    elif track == "off_market" and "list_price" in features.columns:
        raise RuntimeError("list price unexpectedly entered off-market model features")
    ordered = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    if track == "list_aware":
        ordered += ["log_list_price", "log_list_to_parent"]
    return features.loc[:, ordered], list(CATEGORICAL_FEATURES)


def _save_checkpoint(output: Path, current: pd.DataFrame, feature_names: list[str],
                     diagnostics: dict) -> dict:
    keep = [
        "month", "track", "ml_num", "sold_price", "base_prediction", "candidate_prediction",
        "correction_log", "history_oof_rows",
    ]
    path = output / "fold_predictions" / f"{current.track.iloc[0]}_{current.month.iloc[0]}.csv"
    current.loc[:, keep].to_csv(path, index=False)
    digest = _sha256(path)
    checkpoint = {
        "path": _relative(path), "sha256": digest, "rows": int(len(current)),
        "feature_columns": feature_names, **diagnostics,
    }
    campaign._append({
        "event": "fold_checkpoint_saved", "card": CARD, "run_id": output.name,
        "track": str(current.track.iloc[0]), "month": str(current.month.iloc[0]),
        "rows": int(len(current)), "path": _relative(path), "sha256": digest,
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    return checkpoint


def _score_track(frame: pd.DataFrame) -> dict:
    actual = frame.sold_price.to_numpy(float)
    base = frame.base_prediction.to_numpy(float)
    candidate = frame.candidate_prediction.to_numpy(float)
    base_metrics = common._metrics(actual, base)
    candidate_metrics = common._metrics(actual, candidate)
    monthly = {}
    for month, group in frame.groupby("month", sort=True):
        before = common._metrics(group.sold_price.to_numpy(float), group.base_prediction.to_numpy(float))
        after = common._metrics(group.sold_price.to_numpy(float), group.candidate_prediction.to_numpy(float))
        monthly[str(month)] = {
            "n": int(len(group)), "base": before, "candidate": after,
            "relative_mdape_improvement_percent": 100.0 * (
                before["mdape_percent"] - after["mdape_percent"]
            ) / before["mdape_percent"],
        }
    gain = 100.0 * (base_metrics["mdape_percent"] - candidate_metrics["mdape_percent"]) / base_metrics["mdape_percent"]
    mape_change = 100.0 * (candidate_metrics["mape_percent"] - base_metrics["mape_percent"]) / base_metrics["mape_percent"]
    monthly_gains = [item["relative_mdape_improvement_percent"] for item in monthly.values()]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    base_ape = 100.0 * np.abs(base - actual) / actual
    candidate_ape = 100.0 * np.abs(candidate - actual) / actual
    differences = np.empty(BOOTSTRAP_REPLICATES)
    for replicate in range(BOOTSTRAP_REPLICATES):
        index = rng.integers(0, len(actual), size=len(actual))
        differences[replicate] = np.median(candidate_ape[index]) - np.median(base_ape[index])
    bootstrap = {
        "lower95": float(np.quantile(differences, .025)),
        "upper95": float(np.quantile(differences, .975)),
        "replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED,
    }
    gates = campaign.PROTOCOL["gates"]
    target = 7.0 if frame.track.iloc[0] == "off_market" else 4.0
    checks = {
        "relative_mdape_improvement": gain >= gates["development_relative_mdape_improvement_percent"],
        "months_improved": sum(value > 0 for value in monthly_gains) >= gates["development_months_improved_min"],
        "worst_month_regression": max([0.0] + [-value for value in monthly_gains])
        <= gates["development_worst_month_regression_max_percent"],
        "mape_regression": mape_change <= gates["development_mape_regression_max_percent"],
        "bootstrap_upper_below_zero": bootstrap["upper95"] < 0.0,
        "strict_target": candidate_metrics["mdape_percent"] < target,
    }
    return {
        "base": base_metrics, "candidate": candidate_metrics,
        "relative_mdape_improvement_percent": float(gain),
        "mape_relative_change_percent": float(mape_change),
        "months_improved": int(sum(value > 0 for value in monthly_gains)),
        "worst_month_regression_percent": float(max([0.0] + [-value for value in monthly_gains])),
        "paired_bootstrap_mdape_difference_pp": bootstrap, "monthly": monthly,
        "gate_checks": checks, "qualifies_development": all(checks.values()),
        "target_mdape_strictly_below": target,
    }


def run() -> dict:
    card = campaign.PROTOCOL["cards"].get(CARD)
    if not card or card.get("status") != "registered":
        raise RuntimeError("R97 must be preregistered before training")
    if card.get("source_sha256") != _sha256(Path(__file__)):
        raise RuntimeError("R97 source changed after preregistration")
    if card.get("data_sha256") != _sha256(campaign.DATA):
        raise RuntimeError("R97 data hash differs from registration")
    source = _source_identity()
    r93_info = r02._verify_r93()
    result_path = campaign.ROOT / r93_info["result_path"]
    prediction_path = campaign.ROOT / r93_info["predictions_path"]
    manifest_path = campaign.ROOT / r93_info["manifest_path"]
    parent_rows = pd.read_csv(prediction_path, dtype={"ml_num": str})
    required_parent = {"month", "track", "ml_num", "sold_price", "base_prediction"}
    if required_parent - set(parent_rows.columns):
        raise RuntimeError("R93 paired predictions do not include the registered parent columns")
    parent_rows = parent_rows.loc[parent_rows.track.isin(TRACKS), [
        "month", "track", "ml_num", "sold_price", "base_prediction",
    ]].copy()
    parent_rows["month"] = parent_rows.month.astype(str)
    parent_rows["ml_num"] = parent_rows.ml_num.astype(str)
    if len(parent_rows) != len(TRACKS) * 15642 or parent_rows.duplicated(["month", "track", "ml_num"]).any():
        raise RuntimeError("R93 parent rows are incomplete or duplicated")
    if set(parent_rows.month.unique()) != set(MONTHS):
        raise RuntimeError("R93 rows do not cover exactly the eight safe development months")

    metadata = _load_safe_metadata()
    rows = parent_rows.merge(metadata, on=["month", "ml_num"], how="left", validate="many_to_one")
    if rows.date_sold.isna().any() or rows.date_listed.isna().any() or rows.source_sold_price.isna().any():
        raise RuntimeError("safe source metadata does not cover every OOF prediction row")
    if not np.array_equal(rows.sold_price.to_numpy(float), rows.source_sold_price.to_numpy(float)):
        raise RuntimeError("R93 labels differ from exact-month-filtered source labels")
    if set(rows.month.unique()).intersection(campaign.PROTOCOL["reserved_months"]):
        raise RuntimeError("reserved labels unexpectedly entered R97 rows")
    rows["base_prediction"] = rows.base_prediction.astype(float)
    rows["sold_price"] = rows.sold_price.astype(float)
    rows["track"] = rows.track.astype(str)
    rows = rows.sort_values(["track", "month", "ml_num"], kind="mergesort").reset_index(drop=True)

    run_id = "r97m_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = campaign.CAMPAIGN_ROOT / "batches" / run_id
    try:
        output.mkdir(parents=True, exist_ok=False)
        (output / "fold_predictions").mkdir()
        provenance = output / "provenance"
        provenance.mkdir()
        shutil.copy2(campaign.PROTOCOL_PATH, provenance / "registered_campaign_protocol.json")
        shutil.copy2(result_path, provenance / "r93_development_results.json")
        shutil.copy2(prediction_path, provenance / "r93_paired_predictions.csv")
        shutil.copy2(manifest_path, provenance / "r93_run_manifest.json")
    except BaseException as exc:
        campaign._append({
            "event": "card_setup_failed", "card": CARD, "run_id": run_id,
            "fit_starts": 0, "error": f"{type(exc).__name__}: {exc}",
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        campaign._write_progress(
            "r97_setup_failed", next_action="Inspect the R97 setup failure and retry only after a zero-fit correction",
            latest_error=f"{type(exc).__name__}: {exc}", r97_run_id=run_id,
        )
        raise
    campaign._append({
        "event": "card_start", "card": CARD, "run_id": run_id,
        "development_months": list(MONTHS), "expected_fit_starts": card["expected_fit_starts"],
        "r93_predictions_sha256": r93_info["predictions_sha256"],
        "data_sha256": card["data_sha256"], "reserved_months_labels_opened": False,
        "hidden_labels_opened": False,
    })

    fit_details: list[dict] = []
    fold_hashes: dict[str, dict] = {}
    output_frames: list[pd.DataFrame] = []
    try:
        for track in TRACKS:
            track_rows = rows.loc[rows.track.eq(track)].copy()
            for month in MONTHS:
                current = track_rows.loc[track_rows.month.eq(month)].copy().reset_index(drop=True)
                cutoff = pd.Timestamp(month + "-01")
                history = track_rows.loc[pd.to_datetime(track_rows.date_sold).lt(cutoff)].copy()
                if history.empty or (pd.to_datetime(history.date_sold) >= cutoff).any():
                    if not history.empty:
                        raise RuntimeError(f"R97 history crossed strict cutoff for {track}/{month}")
                query_x, categorical = _build_features(current, track)
                correction_log = np.zeros(len(current), dtype=float)
                fit_detail = {
                    "track": track, "month": month, "history_oof_rows": int(len(history)),
                    "history_months": sorted(history.month.unique().tolist()),
                    "history_cutoff_exclusive": cutoff.isoformat(),
                    "model_fit": False, "status": "no_sufficient_history",
                }
                if len(history) >= MIN_HISTORY_ROWS:
                    history_x, _ = _build_features(history, track)
                    raw_target = np.log(
                        history.sold_price.to_numpy(float) / history.base_prediction.to_numpy(float)
                    )
                    if not np.isfinite(raw_target).all():
                        raise RuntimeError(f"R97 residual target is non-finite for {track}/{month}")
                    train_target = np.clip(raw_target, -RESIDUAL_TARGET_CLIP, RESIDUAL_TARGET_CLIP)
                    fit_id = f"{campaign.PROTOCOL['campaign_id']}:{CARD}:{track}:{month}:{run_id}"
                    fit_detail["fit_id"] = fit_id
                    fit_detail["residual_target_clip"] = RESIDUAL_TARGET_CLIP
                    fit_detail["catboost_config"] = CATBOOST_CONFIG
                    campaign._append({
                        "event": "fit_start", "card": CARD, "run_id": run_id, "fit_id": fit_id,
                        "track": track, "month": month, "training_oof_rows": int(len(history)),
                        "training_oof_months": sorted(history.month.unique().tolist()),
                        "history_cutoff_exclusive": cutoff.isoformat(),
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "data_sha256": card["data_sha256"], "reserved_months_labels_opened": False,
                        "hidden_labels_opened": False,
                    })
                    model = CatBoostRegressor(**CATBOOST_CONFIG)
                    started = time.perf_counter()
                    try:
                        model.fit(history_x, train_target, cat_features=categorical)
                        raw_correction = np.asarray(model.predict(query_x), dtype=float)
                        if not np.isfinite(raw_correction).all():
                            raise RuntimeError("CatBoost generated non-finite residual predictions")
                        correction_log = CORRECTION_SHRINKAGE * np.clip(
                            raw_correction, -PREDICTION_RESIDUAL_CLIP, PREDICTION_RESIDUAL_CLIP
                        )
                        fit_detail.update({
                            "model_fit": True, "status": "complete",
                            "seconds": float(time.perf_counter() - started),
                            "prediction_correction_median": float(np.median(raw_correction)),
                            "prediction_correction_p90_abs": float(np.quantile(np.abs(raw_correction), .90)),
                        })
                    except BaseException as exc:
                        elapsed = float(time.perf_counter() - started)
                        campaign._append({
                            "event": "fit_complete", "card": CARD, "run_id": run_id,
                            "fit_id": fit_id, "track": track, "month": month,
                            "status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                            "seconds": elapsed, "error": f"{type(exc).__name__}: {exc}",
                            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
                        })
                        raise
                    fit_detail["effective_config"] = model.get_all_params()
                    fit_details.append(fit_detail)
                    campaign._append({
                        "event": "fit_complete", "card": CARD, "run_id": run_id,
                        "fit_id": fit_id, "track": track, "month": month,
                        "status": "complete", "seconds": fit_detail["seconds"],
                        "training_oof_rows": int(len(history)),
                        "prediction_correction_median": fit_detail["prediction_correction_median"],
                        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
                    })
                candidate = np.rint(current.base_prediction.to_numpy(float) * np.exp(correction_log))
                if not np.isfinite(candidate).all() or (candidate <= 0.0).any():
                    raise RuntimeError(f"R97 generated invalid predictions for {track}/{month}")
                current["candidate_prediction"] = candidate.astype(float)
                current["correction_log"] = correction_log
                current["history_oof_rows"] = int(len(history))
                saved = _save_checkpoint(output, current, list(query_x.columns), fit_detail)
                fold_hashes[f"{track}/{month}"] = saved
                output_frames.append(current[[
                    "month", "track", "ml_num", "sold_price", "base_prediction",
                    "candidate_prediction", "correction_log", "history_oof_rows",
                ]].copy())

        paired = pd.concat(output_frames, ignore_index=True)
        if len(paired) != len(TRACKS) * 15642 or paired.duplicated(["month", "track", "ml_num"]).any():
            raise RuntimeError("R97 paired prediction coverage or listing-key uniqueness is invalid")
        if len(fit_details) != card["expected_fit_starts"]:
            raise RuntimeError(
                f"R97 expected {card['expected_fit_starts']} model fits, observed {len(fit_details)}"
            )
        scores = {
            track: _score_track(paired.loc[paired.track.eq(track)].copy())
            for track in TRACKS
        }
        passed = {track: score["qualifies_development"] for track, score in scores.items()}
        result = {
            "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id, "card": CARD,
            "stage": "strict_past_walk_forward_development",
            "development_months": list(MONTHS), "tracks": scores,
            "fit_starts": len(fit_details), "fit_details": fit_details,
            "fold_prediction_hashes": fold_hashes, "r93_source": r93_info,
            "data_sha256": card["data_sha256"],
            "correction": card["correction"], "catboost_config": CATBOOST_CONFIG,
            "reserved_months_labels_opened": False, "reserve_validation_eligible": False,
            "hidden_labels_opened": False,
        }
        paired_path = output / "paired_predictions.csv"
        paired.to_csv(paired_path, index=False)
        result_path = output / "development_results.json"
        result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        card.update({
            "status": "development_complete", "run_id": run_id,
            "result_path": _relative(result_path), "fit_starts": len(fit_details),
            "qualifies_by_track": passed,
            "development_metrics": {
                track: {
                    "base_mdape_percent": score["base"]["mdape_percent"],
                    "candidate_mdape_percent": score["candidate"]["mdape_percent"],
                    "relative_mdape_improvement_percent": score["relative_mdape_improvement_percent"],
                    "mape_relative_change_percent": score["mape_relative_change_percent"],
                    "paired_bootstrap_upper95_pp": score["paired_bootstrap_mdape_difference_pp"]["upper95"],
                    "qualifies_development": score["qualifies_development"],
                }
                for track, score in scores.items()
            },
        })
        campaign.PROTOCOL_PATH.write_text(
            json.dumps(campaign.PROTOCOL, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        source_files = [
            Path(__file__), campaign.PROTOCOL_PATH, campaign.PACKAGE / "r02_structure_matched_current_parent.py",
            campaign.PACKAGE / "r93_spatial_oof_calibration.py", campaign.PACKAGE / "runner.py",
        ]
        manifest = {
            "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
            "source_identity": source,
            "source_sha256": {_relative(path): _sha256(path) for path in source_files if path.is_file()},
            "runtime": {
                "python": platform.python_version(), "catboost": metadata.version("catboost"),
                "numpy": np.__version__, "pandas": pd.__version__,
            },
            "r93_source": r93_info,
            "metadata_query": {"columns": metadata.columns.tolist(), "months": list(MONTHS)},
            "feature_columns": {
                "all_tracks": [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES],
                "list_aware_only": ["log_list_price", "log_list_to_parent"],
            },
            "catboost_config": CATBOOST_CONFIG, "fit_starts": len(fit_details),
            "fit_details": fit_details, "fold_prediction_hashes": fold_hashes,
            "paired_predictions_sha256": _sha256(paired_path),
            "result_sha256": _sha256(result_path),
        }
        (output / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        report = [
            "# R97: past-OOF CatBoost residual meta-learner", "",
            f"Campaign `{campaign.PROTOCOL['campaign_id']}`; run `{run_id}`.",
            "Training labels are strictly earlier OOF rows; the current month contributes only for scoring.",
            "`date_sold` is used for filtering and cutoff checks only. Reserved and hidden labels remain unopened.",
            "", "## Development results", "",
            "| Track | Base MdAPE | Candidate MdAPE | Relative MdAPE gain | MAPE change | Months improved | Bootstrap upper95 pp | Gate |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
        for track, score in scores.items():
            report.append(
                f"| {track} | {score['base']['mdape_percent']:.5f}% | "
                f"{score['candidate']['mdape_percent']:.5f}% | "
                f"{score['relative_mdape_improvement_percent']:+.3f}% | "
                f"{score['mape_relative_change_percent']:+.3f}% | {score['months_improved']}/8 | "
                f"{score['paired_bootstrap_mdape_difference_pp']['upper95']:+.5f} | "
                f"{'PASS' if score['qualifies_development'] else 'FAIL'} |"
            )
        report.extend([
            "", f"Actual CatBoost fit starts: {len(fit_details)} (expected 12).",
            "Promotion requires every registered development gate; this run never changes the parent model.",
            "", "Artifacts: `paired_predictions.csv`, `development_results.json`, `run_manifest.json`, "
            "and per-fold files under `fold_predictions/`.", "",
        ])
        report_path = output / "research_report.md"
        report_path.write_text("\n".join(report), encoding="utf-8")
        next_action = (
            "R97 development-qualified tracks require fixed-origin confirmation: "
            + ", ".join(track for track, ok in passed.items() if ok)
            if any(passed.values()) else
            "R97 failed development gates; retain R22/R35 and register a distinct method"
        )
        campaign._append({
            "event": "card_complete", "card": CARD, "run_id": run_id,
            "fit_starts": len(fit_details), "qualifies_by_track": passed,
            "result_path": _relative(result_path),
            "paired_predictions_sha256": manifest["paired_predictions_sha256"],
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        campaign._write_progress(
            "r97_development_complete", next_action=next_action,
            r97_run_id=run_id, r97_result=_relative(result_path), r97_fit_starts=len(fit_details),
        )
        return result
    except BaseException as exc:
        campaign._append({
            "event": "card_failed", "card": CARD, "run_id": run_id,
            "error": f"{type(exc).__name__}: {exc}",
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        campaign._write_progress(
            "r97_failed", next_action="Inspect the R97 durable fold checkpoints before recovery",
            latest_error=f"{type(exc).__name__}: {exc}", r97_run_id=run_id,
        )
        raise


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["register", "run"])
    args = parser.parse_args()
    if args.action == "register":
        card = register_card()
        print(f"Registered {CARD}; expected fits={card['expected_fit_starts']}; source={card['source_sha256']}", flush=True)
        return
    result = run()
    for track, score in result["tracks"].items():
        print(
            f"{track}: base={score['base']['mdape_percent']:.5f}% "
            f"candidate={score['candidate']['mdape_percent']:.5f}% "
            f"gain={score['relative_mdape_improvement_percent']:+.3f}% "
            f"MAPE_change={score['mape_relative_change_percent']:+.3f}% "
            f"months_improved={score['months_improved']}/8 "
            f"bootstrap_upper95={score['paired_bootstrap_mdape_difference_pp']['upper95']:+.5f} "
            f"qualifies={score['qualifies_development']}", flush=True,
        )
    print(f"report={campaign.CAMPAIGN_ROOT / 'batches' / result['run_id'] / 'research_report.md'}", flush=True)


if __name__ == "__main__":
    main()
