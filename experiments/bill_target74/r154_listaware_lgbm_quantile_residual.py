"""R154: LightGBM median log-residual meta-learner on R35 list-aware OOFs."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from experiments.bill_target74 import r22_remarks_tfidf as r22
from experiments.bill_target74 import r93_spatial_oof_calibration as r93
from experiments.bill_target74 import r97_past_oof_catboost_residual_meta as r97
from experiments.bill_target74 import runner as campaign


CARD = "R154_listaware_lgbm_quantile_residual_r35"
TRACK = "list_aware"
R35_CARD = "R35_listaware_detached_r22_text_expert"
R35_RUN_ID = "bill_target74_r35_house_type_text_development_20261025T040058Z"
R35_SCREEN_RUN_ID = "bill_target74_r35_house_type_text_20261025T035638Z"
R44_CARD = "R44_past_oof_r35_blend"
R44_RUN_ID = "bill_target74_r44_past_oof_r35_blend_20261025T055003Z"
SEED_MONTHS = ("2022-06", "2022-11")
MONTHS = tuple(campaign.PROTOCOL["development_months"])
SCREEN_MONTHS = tuple(campaign.PROTOCOL["screen_months"])
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
AUDITOR_PATH = Path(__file__).with_name("r154_independent_audit.py")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(campaign.ROOT.resolve())).replace("\\", "/")


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _code_hashes() -> dict[str, str]:
    paths = [Path(__file__), AUDITOR_PATH, Path(r97.__file__), Path(r93.__file__),
             Path(r22.__file__), Path(campaign.__file__)]
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("R154 runner, auditor, or pinned source is missing")
    return {_relative(path): _sha256(path) for path in paths}


def _input_artifacts() -> dict:
    progress = _json(campaign.STATE)
    r35_result = campaign.ROOT / progress["r35_development_result"]
    r35_pair, r35_manifest = r35_result.parent / "paired_predictions.csv", r35_result.parent / "run_manifest.json"
    r35_data, r35_meta = _json(r35_result), _json(r35_manifest)
    r35_card = campaign.PROTOCOL["cards"].get(R35_CARD, {})
    if (not all(path.is_file() for path in (r35_result, r35_pair, r35_manifest))
            or r35_data.get("run_id") != R35_RUN_ID or r35_data.get("card") != R35_CARD
            or r35_data.get("campaign_id") != campaign.PROTOCOL["campaign_id"]
            or r35_data.get("data_sha256") != campaign.PROTOCOL["data_sha256"]
            or r35_data.get("reserved_months_labels_opened") is not False
            or r35_data.get("hidden_labels_opened") is not False
            or r35_meta.get("run_id") != R35_RUN_ID
            or r35_meta.get("campaign_id") != campaign.PROTOCOL["campaign_id"]
            or r35_meta.get("source_identity", {}).get("commit") != campaign.PROTOCOL["parent_commit"]
            or r35_meta.get("source_identity", {}).get("data_sha256") != campaign.PROTOCOL["data_sha256"]
            or r35_meta.get("result_sha256") != _sha256(r35_result)
            or r35_meta.get("paired_predictions_sha256") != _sha256(r35_pair)
            or r35_meta.get("reserved_months_labels_opened") is not False
            or r35_meta.get("hidden_labels_opened") is not False
            or r35_card.get("status") != "fixed_origin_confirmation_passed"
            or r35_card.get("development_run_id") != R35_RUN_ID):
        raise RuntimeError("R154 R35 validated parent identity, hashes, or access flags failed")

    r44_root = campaign.CAMPAIGN_ROOT / "batches" / R44_RUN_ID
    seed_path, r44_result, r44_manifest = (r44_root / "past_oof_predictions.csv",
                                            r44_root / "screen_results.json",
                                            r44_root / "run_manifest.json")
    r44_pair = r44_root / "paired_predictions.csv"
    if not all(path.is_file() for path in (seed_path, r44_result, r44_manifest, r44_pair)):
        raise FileNotFoundError("R154 requires complete R44 seed OOF artifacts")
    r44_data, r44_meta = _json(r44_result), _json(r44_manifest)
    r44_card = campaign.PROTOCOL["cards"].get(R44_CARD, {})
    safe_months = set(r44_meta.get("safe_oof_months_read", []))
    allowed = set(SEED_MONTHS) | set(MONTHS)
    if (r44_data.get("run_id") != R44_RUN_ID or r44_data.get("card") != R44_CARD
            or r44_data.get("campaign_id") != campaign.PROTOCOL["campaign_id"]
            or r44_data.get("data_sha256") != campaign.PROTOCOL["data_sha256"]
            or set(r44_data.get("seed_months", [])) != set(SEED_MONTHS)
            or r44_data.get("reserved_months_labels_opened") is not False
            or r44_data.get("hidden_labels_opened") is not False
            or r44_meta.get("run_id") != R44_RUN_ID
            or r44_meta.get("campaign_id") != campaign.PROTOCOL["campaign_id"]
            or r44_meta.get("parent_run_id") != R35_SCREEN_RUN_ID
            or r44_meta.get("source_identity", {}).get("commit") != campaign.PROTOCOL["parent_commit"]
            or r44_meta.get("source_identity", {}).get("data_sha256") != campaign.PROTOCOL["data_sha256"]
            or r44_meta.get("seed_oof_sha256") != _sha256(seed_path)
            or r44_meta.get("paired_predictions_sha256") != _sha256(r44_pair)
            or r44_meta.get("result_sha256") != _sha256(r44_result)
            or r44_meta.get("reserved_months_labels_opened") is not False
            or r44_meta.get("hidden_labels_opened") is not False
            or safe_months & set(campaign.PROTOCOL["reserved_months"])
            or not safe_months.issubset(allowed)
            or r44_card.get("parent_run_id") != R35_SCREEN_RUN_ID
            or r44_card.get("variant_run_id") != R35_SCREEN_RUN_ID):
        raise RuntimeError("R154 R44 seed identity, parent route, hashes, or safe flags failed")
    return {
        "r35": {
            "card": R35_CARD, "run_id": R35_RUN_ID,
            "result_path": _relative(r35_result), "result_sha256": _sha256(r35_result),
            "predictions_path": _relative(r35_pair), "predictions_sha256": _sha256(r35_pair),
            "manifest_path": _relative(r35_manifest), "manifest_sha256": _sha256(r35_manifest),
        },
        "r44": {
            "card": R44_CARD, "run_id": R44_RUN_ID,
            "seed_predictions_path": _relative(seed_path), "seed_predictions_sha256": _sha256(seed_path),
            "result_path": _relative(r44_result), "result_sha256": _sha256(r44_result),
            "manifest_path": _relative(r44_manifest), "manifest_sha256": _sha256(r44_manifest),
            "paired_predictions_path": _relative(r44_pair), "paired_predictions_sha256": _sha256(r44_pair),
            "seed_months": list(SEED_MONTHS),
        },
    }


def _read_months(stage: str) -> tuple[str, ...]:
    targets = SCREEN_MONTHS if stage == "screen" else MONTHS
    last_month = max(targets)
    return tuple(sorted(set(SEED_MONTHS) | {month for month in MONTHS if month <= last_month}))


def _load_frame(inputs: dict, stage: str) -> pd.DataFrame:
    read_months = _read_months(stage)
    if set(read_months) & set(campaign.PROTOCOL["reserved_months"]):
        raise RuntimeError("R154 read filter intersects reserved months")
    if _sha256(campaign.DATA) != campaign.PROTOCOL["data_sha256"]:
        raise RuntimeError("R154 source parquet hash differs from campaign")
    filters = []
    for month in read_months:
        start = pd.Timestamp(month + "-01")
        filters.append([("date_sold", ">=", start),
                        ("date_sold", "<", start + pd.offsets.MonthBegin(1))])
    columns = ["ml_num", "date_sold", "date_listed", "sold_price", "list_price",
               "municipality", "community", "house_type_name", "house_style", "garage",
               "house_area", "bedrooms", "washrooms", "parking_total", "latitude", "longitude"]
    metadata = pd.read_parquet(campaign.DATA, columns=columns, filters=filters)
    metadata["date_sold"] = pd.to_datetime(metadata.date_sold, errors="raise")
    metadata["date_listed"] = pd.to_datetime(metadata.date_listed, errors="raise")
    metadata["month"] = metadata.date_sold.dt.to_period("M").astype(str)
    metadata["ml_num"] = metadata.ml_num.astype(str)
    metadata["house_type_name"] = metadata.house_type_name.astype("string").fillna("__MISSING__").astype(str)
    if set(metadata.month.unique()) != set(read_months) or metadata.duplicated(["month", "ml_num"]).any():
        raise RuntimeError("R154 safe metadata has unexpected coverage or duplicate keys")
    if set(metadata.month.unique()) & set(campaign.PROTOCOL["reserved_months"]):
        raise RuntimeError("reserved month returned by R154 exact date filters")
    metadata["source_sold_price"] = pd.to_numeric(metadata.sold_price, errors="raise")

    r35_path = campaign.ROOT / inputs["r35"]["predictions_path"]
    r35 = pd.read_csv(r35_path, dtype={"ml_num": str})
    required = {"month", "track", "ml_num", "sold_price", "candidate_prediction"}
    if required - set(r35.columns):
        raise RuntimeError("R35 OOF file lacks list-aware prediction columns")
    r35 = r35.loc[r35.track.eq(TRACK) & r35.month.astype(str).isin(read_months),
                  ["month", "ml_num", "sold_price", "candidate_prediction"]].copy()
    r35["month"], r35["ml_num"] = r35.month.astype(str), r35.ml_num.astype(str)
    r35["sold_price"] = pd.to_numeric(r35.sold_price, errors="raise")
    r35["base_prediction"] = pd.to_numeric(r35.candidate_prediction, errors="raise")
    r35.drop(columns="candidate_prediction", inplace=True)
    dev = r35.loc[r35.month.isin(MONTHS)].merge(
        metadata.loc[metadata.month.isin(MONTHS)], on=["month", "ml_num"],
        how="outer", validate="one_to_one", indicator=True, suffixes=("", "_source"),
    )
    if not dev._merge.eq("both").all() or not np.array_equal(dev.sold_price.to_numpy(float), dev.source_sold_price.to_numpy(float)):
        raise RuntimeError("R35 safe OOF rows do not exactly align with source keys/labels")
    if not np.isfinite(dev.base_prediction.to_numpy(float)).all() or (dev.base_prediction.to_numpy(float) <= 0).any():
        raise RuntimeError("R35 base predictions are invalid")
    dev.drop(columns=["_merge", "source_sold_price"], inplace=True)

    seed = pd.read_csv(campaign.ROOT / inputs["r44"]["seed_predictions_path"], dtype={"ml_num": str})
    required_seed = {"month", "track", "ml_num", "date_sold", "sold_price", "variant_prediction"}
    if required_seed - set(seed.columns):
        raise RuntimeError("R44 seed file lacks required columns")
    seed = seed.loc[seed.track.eq(TRACK), ["month", "ml_num", "date_sold", "sold_price", "variant_prediction"]].copy()
    seed["month"], seed["ml_num"] = seed.month.astype(str), seed.ml_num.astype(str)
    seed["date_sold"] = pd.to_datetime(seed.date_sold, errors="raise")
    seed["sold_price"] = pd.to_numeric(seed.sold_price, errors="raise")
    seed["base_prediction"] = pd.to_numeric(seed.variant_prediction, errors="raise")
    if set(seed.month.unique()) != set(SEED_MONTHS) or seed.duplicated(["month", "ml_num"]).any():
        raise RuntimeError("R44 seed months or keys are invalid")
    seed.drop(columns="variant_prediction", inplace=True)
    seed_rows = seed.merge(metadata, on=["month", "ml_num"], how="left", validate="one_to_one",
                           indicator=True, suffixes=("_seed", "_source"))
    if (not seed_rows._merge.eq("both").all()
            or not np.array_equal(seed_rows.date_sold_seed.to_numpy("datetime64[ns]"), seed_rows.date_sold_source.to_numpy("datetime64[ns]"))
            or not np.array_equal(seed_rows.sold_price_seed.to_numpy(float), seed_rows.source_sold_price.to_numpy(float))):
        raise RuntimeError("R44 seed dates or labels do not match exact source metadata")
    seed_rows["date_listed"] = pd.to_datetime(seed_rows.date_listed, errors="raise")
    for column in ["list_price", "municipality", "community", "house_type_name", "house_style", "garage",
                   "house_area", "bedrooms", "washrooms", "parking_total", "latitude", "longitude"]:
        source_column = f"{column}_source"
        if source_column in seed_rows.columns:
            seed_rows[column] = seed_rows[source_column]
    seed_rows = seed_rows.rename(columns={"date_sold_seed": "date_sold", "sold_price_seed": "sold_price"})
    keep = ["month", "ml_num", "date_sold", "date_listed", "sold_price", "base_prediction", "list_price",
            "municipality", "community", "house_type_name", "house_style", "garage", "house_area",
            "bedrooms", "washrooms", "parking_total", "latitude", "longitude"]
    frame = pd.concat([seed_rows[keep], dev[keep]], ignore_index=True)
    if frame.duplicated(["month", "ml_num"]).any() or set(frame.month.unique()) != set(read_months):
        raise RuntimeError("R154 combined safe OOF frame has invalid keys or month coverage")
    frame = frame.sort_values(["month", "ml_num"], kind="mergesort").reset_index(drop=True)
    return frame


def _feature_matrices(history: pd.DataFrame, query: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    forbidden = {"sold_price", "date_sold"}
    train_input = history.drop(columns=["sold_price", "date_sold", "month", "ml_num"], errors="ignore")
    query_input = query.drop(columns=["sold_price", "date_sold", "month", "ml_num"], errors="ignore")
    train_x, categorical = r97._build_features(train_input, TRACK)
    query_x, query_categorical = r97._build_features(query_input, TRACK)
    if categorical != query_categorical or list(train_x.columns) != list(query_x.columns):
        raise RuntimeError("R154 train/query feature schemas differ")
    if forbidden & set(train_x.columns) or forbidden & set(query_x.columns):
        raise RuntimeError("R154 target sale date/price leaked into feature matrix")
    for column in categorical:
        categories = sorted(train_x[column].astype("string").fillna("__MISSING__").unique().astype(str).tolist())
        train_x[column] = pd.Categorical(train_x[column].astype(str), categories=categories)
        query_x[column] = pd.Categorical(query_x[column].astype(str), categories=categories)
    return train_x, query_x, categorical


def _fit_fold(history: pd.DataFrame, query: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict]:
    train_x, query_x, categorical = _feature_matrices(history, query)
    actual = history.sold_price.to_numpy(float)
    base = history.base_prediction.to_numpy(float)
    if (not np.isfinite(actual).all() or (actual <= 0).any()
            or not np.isfinite(base).all() or (base <= 0).any()):
        raise RuntimeError("R154 training residual inputs are not positive finite prices")
    target_raw = np.log(actual / base)
    target = np.clip(target_raw, -RESIDUAL_TARGET_CLIP, RESIDUAL_TARGET_CLIP)
    model = LGBMRegressor(**MODEL_CONFIG)
    model.fit(train_x, target, categorical_feature=categorical)
    raw_residual = np.asarray(model.predict(query_x), dtype=float)
    if not np.isfinite(raw_residual).all():
        raise RuntimeError("R154 LightGBM returned non-finite residual predictions")
    correction = CORRECTION_SHRINKAGE * np.clip(
        raw_residual, -PREDICTION_RESIDUAL_CLIP, PREDICTION_RESIDUAL_CLIP,
    )
    detail = {
        "effective_config": {key: value for key, value in model.get_params().items()
                             if key in MODEL_CONFIG},
        "feature_columns": list(train_x.columns), "categorical_features": categorical,
        "target_clip": RESIDUAL_TARGET_CLIP, "prediction_clip": PREDICTION_RESIDUAL_CLIP,
        "correction_shrinkage": CORRECTION_SHRINKAGE,
        "raw_residual_median": float(np.median(raw_residual)),
        "raw_residual_abs_p90": float(np.quantile(np.abs(raw_residual), 0.9)),
    }
    return raw_residual, correction, detail


def register_card() -> dict:
    existing = campaign.PROTOCOL["cards"].get(CARD)
    inputs = _input_artifacts()
    hashes = _code_hashes()
    if existing:
        if existing.get("status") != "registered":
            raise RuntimeError(f"{CARD} already has status {existing.get('status')}")
        if existing.get("code_sha256") != hashes or existing.get("input_artifacts") != inputs:
            raise RuntimeError("R154 code or registered input hashes changed")
        return existing
    frame = _load_frame(inputs, "screen")
    source = {
        "safe_history_rows": int(len(frame)), "safe_months": sorted(frame.month.astype(str).unique().tolist()),
        "rows_by_month": {str(k): int(v) for k, v in frame.month.value_counts().sort_index().items()},
        "target_rows": {month: int(frame.month.eq(month).sum()) for month in SCREEN_MONTHS},
        "read_months": list(_read_months("screen")),
    }
    card = {
        "status": "registered", "track": TRACK, "screen_months": list(SCREEN_MONTHS),
        "development_months": list(MONTHS), "seed_months": list(SEED_MONTHS),
        "expected_screen_learner_fit_starts": len(SCREEN_MONTHS),
        "expected_development_learner_fit_starts": len(MONTHS),
        "min_training_oof_rows": MIN_TRAINING_OOF_ROWS,
        "comparison_parent": {"card": R35_CARD, "run_id": R35_RUN_ID},
        "base_candidate": {"card": R35_CARD, "run_id": R35_RUN_ID},
        "method": (
            "For each target month, fit LightGBM objective quantile alpha=0.5 to clipped "
            "log(sold_price/R35_prediction) residuals from only earlier R44 seed OOF and R35 "
            "development OOF rows. Reuse R97's pre-sale numeric/categorical feature schema "
            "including list_price and list-to-parent ratio. Apply 0.50 * clip(predicted residual, "
            "+/-0.10), then exp-multiply R35 and round to nearest dollar."
        ),
        "difference_from_prior_work": (
            "R97/R109 use CatBoost MAE on a past-OOF residual target; R154 uses deterministic "
            "LightGBM quantile alpha 0.5 on the same fixed R97 feature/target/correction design. "
            "R147/R148 changed the primary listing-price learner target, not the second-stage "
            "OOF residual learner. R153 tested spatial residual correction and failed."
        ),
        "model_config": MODEL_CONFIG, "residual_target_clip": RESIDUAL_TARGET_CLIP,
        "prediction_residual_clip": PREDICTION_RESIDUAL_CLIP,
        "correction_shrinkage": CORRECTION_SHRINKAGE,
        "feature_builder": "r97._build_features; list-aware only; excludes sold_price/date_sold",
        "screen_gates": {
            "relative_mdape_gain_min_percent": campaign.PROTOCOL["gates"]["screen_relative_mdape_improvement_percent"],
            "both_screen_months_improve": True,
            "mape_regression_max_percent": campaign.PROTOCOL["gates"]["screen_mape_regression_max_percent"],
            "bootstrap_upper95_below_zero": True,
        },
        "development_gates": dict(campaign.PROTOCOL["gates"]),
        "strict_development_mdape_target_percent": 4.0,
        "source_preflight": source,
        "source_data_filter_policy": (
            "Exact date_sold filters for R44 seed months and safe R35 development months through "
            "each stage's last target month. Never read reserved 2025-05/08 or hidden labels."
        ),
        "input_artifacts": inputs, "code_sha256": hashes,
        "lightgbm_version": lgb.__version__, "data_sha256": campaign.PROTOCOL["data_sha256"],
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        "reserve_validation_eligible": False,
    }
    campaign.PROTOCOL["cards"][CARD] = card
    campaign.PROTOCOL_PATH.write_text(json.dumps(campaign.PROTOCOL, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    campaign._append({
        "event": "card_registered", "card": CARD, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "expected_screen_learner_fit_starts": len(SCREEN_MONTHS), "learner_training_fit_starts": 0,
        "source_sha256": hashes[_relative(Path(__file__))], "input_artifacts": inputs,
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    campaign._write_progress("r154_registered", next_action="run R154 fixed list-aware LightGBM median-residual screen", r154_card=CARD)
    return card


def _stage_started(stage: str) -> bool:
    if not campaign.LEDGER.is_file():
        return False
    with campaign.LEDGER.open("r", encoding="utf-8") as stream:
        return any(event.get("card") == CARD and event.get("stage") == stage and event.get("event") == "fit_start"
                   for line in stream if line.strip() for event in [json.loads(line)])


def amend_registration_after_zero_fit_failure(reason: str) -> dict:
    card = campaign.PROTOCOL["cards"].get(CARD)
    if not card or card.get("status") != "registered":
        raise RuntimeError("only a still-registered R154 card can receive a zero-fit amendment")
    if not reason.strip():
        raise ValueError("an explicit pre-fit failure reason is required")
    if _stage_started("screen") or _stage_started("development"):
        raise RuntimeError("R154 has a fit_start event; its registration cannot be amended")
    inputs = _input_artifacts()
    if card.get("input_artifacts") != inputs:
        raise RuntimeError("R154 upstream artifacts changed; do not amend this registration")
    events = [json.loads(line) for line in campaign.LEDGER.read_text(encoding="utf-8").splitlines()
              if line.strip()]
    failures = [event for event in events if event.get("card") == CARD and event.get("event") == "card_failed"
                and event.get("stage") == "screen"]
    if not failures:
        raise RuntimeError("R154 has no recorded screen setup failure to amend")
    failed_run = failures[-1]["run_id"]
    checkpoint_path = campaign.CAMPAIGN_ROOT / "batches" / failed_run / "failure_checkpoint.json"
    checkpoint = _json(checkpoint_path)
    if checkpoint.get("fit_starts_completed") != 0 or checkpoint.get("run_id") != failed_run:
        raise RuntimeError("R154 failed screen is not a verified zero-fit setup failure")
    hashes = _code_hashes()
    previous_hashes = card.get("code_sha256")
    if previous_hashes == hashes:
        raise RuntimeError("R154 registered hashes already match; inspect the failure instead")
    amended_at = datetime.now(timezone.utc).isoformat()
    reason_record = reason.strip()
    card["registration_revision"] = int(card.get("registration_revision", 1)) + 1
    card["min_training_oof_rows"] = MIN_TRAINING_OOF_ROWS
    card.setdefault("registration_amendments", []).append({
        "timestamp_utc": amended_at, "reason": reason_record, "fit_starts": 0,
        "failed_run_id": failed_run, "previous_code_sha256": previous_hashes,
        "current_code_sha256": hashes, "min_training_oof_rows": MIN_TRAINING_OOF_ROWS,
        "input_artifacts_unchanged": True,
    })
    card["code_sha256"] = hashes
    campaign.PROTOCOL_PATH.write_text(json.dumps(campaign.PROTOCOL, indent=2, ensure_ascii=False) + "\n",
                                      encoding="utf-8")
    campaign._append({
        "event": "registration_amended", "card": CARD, "timestamp_utc": amended_at,
        "reason": reason_record, "fit_starts": 0, "failed_run_id": failed_run,
        "code_sha256": hashes, "input_artifacts_unchanged": True,
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    campaign._write_progress("r154_registration_amended",
                             next_action="rerun R154 fixed screen under the amended 3,500-row support floor",
                             r154_card=CARD, r154_failed_run_id=failed_run, r154_fit_starts=0)
    return card


def _run(stage: str) -> dict:
    card = campaign.PROTOCOL["cards"].get(CARD)
    if stage == "screen" and (not card or card.get("status") != "registered"):
        raise RuntimeError("R154 screen must be registered and may run only once")
    if stage == "development" and (not card or card.get("status") != "screen_passed"):
        raise RuntimeError("R154 development requires a passed fixed screen")
    if card.get("code_sha256") != _code_hashes() or card.get("input_artifacts") != _input_artifacts():
        raise RuntimeError("R154 preregistered code or source inputs changed")
    if card.get("min_training_oof_rows") != MIN_TRAINING_OOF_ROWS:
        raise RuntimeError("R154 registered minimum training support differs from the runner")
    if _stage_started(stage):
        raise RuntimeError(f"R154 {stage} already has fit starts; do not repeat fixed folds")
    frame = _load_frame(card["input_artifacts"], stage)
    months = list(SCREEN_MONTHS if stage == "screen" else MONTHS)
    run_id = ("r154s_" if stage == "screen" else "r154d_") + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = campaign.CAMPAIGN_ROOT / "batches" / run_id
    output.mkdir(parents=True, exist_ok=False)
    provenance = output / "provenance"
    provenance.mkdir()
    for name, fields in {
        "r35": ("result_path", "predictions_path", "manifest_path"),
        "r44": ("result_path", "seed_predictions_path", "manifest_path", "paired_predictions_path"),
    }.items():
        for field in fields:
            source = campaign.ROOT / card["input_artifacts"][name][field]
            shutil.copy2(source, provenance / f"{name}_{field}{source.suffix}")
    try:
        paired_parts, fit_details = [], []
        for month in months:
            cutoff = pd.Timestamp(month + "-01")
            history = frame.loc[pd.to_datetime(frame.date_sold, errors="raise").lt(cutoff)].copy()
            history = history.sort_values(["month", "ml_num"], kind="mergesort").reset_index(drop=True)
            query = frame.loc[frame.month.eq(month)].copy().sort_values("ml_num", kind="mergesort").reset_index(drop=True)
            if len(history) < MIN_TRAINING_OOF_ROWS or not pd.to_datetime(history.date_sold).lt(cutoff).all():
                raise RuntimeError(f"R154 {month} history failed the strict cutoff/support check")
            if query.empty or query.duplicated(["month", "ml_num"]).any():
                raise RuntimeError(f"R154 query rows are empty or duplicated for {month}")
            fit_id = f"{campaign.PROTOCOL['campaign_id']}:{CARD}:{stage}:{TRACK}:{month}:{run_id}"
            history_months = sorted(history.month.astype(str).unique().tolist())
            campaign._append({
                "event": "fit_start", "card": CARD, "stage": stage, "run_id": run_id,
                "fit_id": fit_id, "track": TRACK, "month": month,
                "training_oof_rows": int(len(history)), "training_oof_months": history_months,
                "history_cutoff_exclusive": cutoff.isoformat(),
                "query_rows": int(len(query)), "query_feature_columns": [
                    "log_parent_prediction", "log_house_area", "bedrooms", "washrooms", "parking_total",
                    "latitude", "longitude", "date_listed_month_index", "house_type_name", "municipality",
                    "community", "house_style", "garage", "log_list_price", "log_list_to_parent",
                ],
                "query_label_columns_passed": [], "model_config": MODEL_CONFIG,
                "reserved_months_labels_opened": False, "hidden_labels_opened": False,
            })
            started = time.perf_counter()
            raw_residual, correction, model_info = _fit_fold(history, query)
            base = query.base_prediction.to_numpy(float)
            candidate = np.rint(base * np.exp(correction))
            if not np.isfinite(candidate).all() or (candidate <= 0).any():
                raise RuntimeError(f"R154 generated invalid predictions for {month}")
            rows = pd.DataFrame({
                "month": month, "track": TRACK, "ml_num": query.ml_num.astype(str),
                "sold_price": query.sold_price.to_numpy(float), "parent_prediction": base,
                "base_prediction": base, "raw_residual_prediction": raw_residual,
                "correction_log": correction, "candidate_prediction": candidate,
                "history_oof_rows": len(history), "history_months": ",".join(history_months),
            })
            rows["parent_ape_percent"] = 100.0 * np.abs(rows.sold_price - rows.parent_prediction) / rows.sold_price
            rows["candidate_ape_percent"] = 100.0 * np.abs(rows.sold_price - rows.candidate_prediction) / rows.sold_price
            paired_parts.append(rows)
            detail = {
                "fit_id": fit_id, "month": month, "track": TRACK, "status": "complete",
                "seconds": time.perf_counter() - started, "training_oof_rows": int(len(history)),
                "training_oof_months": history_months, "history_cutoff_exclusive": cutoff.isoformat(),
                "query_rows": int(len(query)), "learner_training_fit": True,
                "residual_target_clip": RESIDUAL_TARGET_CLIP,
                "prediction_residual_clip": PREDICTION_RESIDUAL_CLIP,
                "correction_shrinkage": CORRECTION_SHRINKAGE, **model_info,
            }
            fit_details.append(detail)
            campaign._append({
                "event": "fit_complete", "card": CARD, "stage": stage, "run_id": run_id,
                **detail, "reserved_months_labels_opened": False, "hidden_labels_opened": False,
            })
        paired = pd.concat(paired_parts, ignore_index=True)
        if paired.duplicated(["month", "ml_num"]).any():
            raise RuntimeError("R154 produced duplicate paired prediction keys")
        score = r22.score_development_confirmation(
            paired[["month", "ml_num", "sold_price", "parent_prediction", "candidate_prediction"]],
            months=tuple(months), target_mdape_strictly_below=4.0,
            bootstrap_replicates=BOOTSTRAP_REPLICATES, bootstrap_seed=BOOTSTRAP_SEED,
            minimum_relative_gain_percent=(campaign.PROTOCOL["gates"]["screen_relative_mdape_improvement_percent"]
                                           if stage == "screen" else campaign.PROTOCOL["gates"]["development_relative_mdape_improvement_percent"]),
            minimum_months_improved=(len(months) if stage == "screen" else campaign.PROTOCOL["gates"]["development_months_improved_min"]),
            maximum_worst_month_regression_percent=(0.0 if stage == "screen" else campaign.PROTOCOL["gates"]["development_worst_month_regression_max_percent"]),
            maximum_mape_regression_percent=(campaign.PROTOCOL["gates"]["screen_mape_regression_max_percent"]
                                             if stage == "screen" else campaign.PROTOCOL["gates"]["development_mape_regression_max_percent"]),
        )
        if stage == "screen":
            checks = {
                "relative_mdape_gain": score["relative_mdape_improvement_percent"] >= campaign.PROTOCOL["gates"]["screen_relative_mdape_improvement_percent"],
                "both_screen_months_improved": score["months_improved"] == len(months),
                "mape_guard": score["mape_relative_change_percent"] <= campaign.PROTOCOL["gates"]["screen_mape_regression_max_percent"],
                "bootstrap_upper95_below_zero": score["bootstrap_mdape_difference_percent"]["upper95"] < 0,
            }
        else:
            checks = {"development_relative_gates": score["qualifies_development"],
                      "strict_listaware_target_below_4_percent": score["target_met"]}
        qualifies = all(checks.values())
        result = {
            "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
            "card": CARD, "stage": stage, "track": TRACK, "months": months,
            "scores": score, "gate_checks": checks, "qualifies": bool(qualifies),
            "learner_training_fit_starts": len(fit_details), "fit_details": fit_details,
            "input_artifacts": card["input_artifacts"], "filtered_source_months": list(_read_months(stage)),
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
            "reserve_validation_eligible": False,
        }
        pair_path, result_path = output / "paired_predictions.csv", output / ("screen_results.json" if stage == "screen" else "development_results.json")
        paired.to_csv(pair_path, index=False)
        result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        manifest = {
            "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
            "card": CARD, "stage": stage, "track": TRACK, "data_sha256": card["data_sha256"],
            "code_sha256": card["code_sha256"], "input_artifacts": card["input_artifacts"],
            "filtered_source_months": list(_read_months(stage)),
            "paired_predictions_sha256": _sha256(pair_path), "result_sha256": _sha256(result_path),
            "learner_training_fit_starts": len(fit_details),
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        }
        manifest_path = output / "run_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if stage == "screen":
            card.update({"status": "screen_passed" if qualifies else "screen_failed",
                         "screen_run_id": run_id, "screen_result_path": _relative(result_path),
                         "screen_result_sha256": _sha256(result_path),
                         "screen_paired_predictions_path": _relative(pair_path),
                         "screen_paired_predictions_sha256": _sha256(pair_path),
                         "screen_manifest_path": _relative(manifest_path),
                         "screen_manifest_sha256": _sha256(manifest_path),
                         "screen_metrics": score, "screen_gate_checks": checks})
            next_action = ("run R154 eight-month development confirmation; screen passed" if qualifies
                           else "retain R35; R154 fixed screen failed; register a distinct residual method")
        else:
            card.update({"status": "development_complete", "development_run_id": run_id,
                         "development_result_path": _relative(result_path),
                         "development_result_sha256": _sha256(result_path),
                         "development_paired_predictions_path": _relative(pair_path),
                         "development_paired_predictions_sha256": _sha256(pair_path),
                         "development_manifest_path": _relative(manifest_path),
                         "development_manifest_sha256": _sha256(manifest_path),
                         "development_metrics": score, "development_gate_checks": checks})
            next_action = ("review fixed-origin confirmation eligibility; reserve remains ineligible" if qualifies
                           else "retain R35; R154 development did not meet all registered gates")
        campaign.PROTOCOL_PATH.write_text(json.dumps(campaign.PROTOCOL, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        campaign._append({"event": "card_complete", "card": CARD, "stage": stage, "run_id": run_id,
                          "fit_starts": len(fit_details), "qualifies": bool(qualifies),
                          "result_path": _relative(result_path), "result_sha256": _sha256(result_path),
                          "paired_predictions_sha256": _sha256(pair_path),
                          "reserved_months_labels_opened": False, "hidden_labels_opened": False})
        campaign._write_progress(f"r154_{stage}_{'passed' if qualifies else 'failed'}",
                                 next_action=next_action, r154_run_id=run_id, r154_result=_relative(result_path))
        return result
    except Exception as exc:
        checkpoint = {"run_id": run_id, "card": CARD, "stage": stage,
                      "fit_starts_completed": len(fit_details) if "fit_details" in locals() else 0,
                      "error": f"{type(exc).__name__}: {exc}",
                      "reserved_months_labels_opened": False, "hidden_labels_opened": False}
        (output / "failure_checkpoint.json").write_text(json.dumps(checkpoint, indent=2) + "\n", encoding="utf-8")
        campaign._append({"event": "card_failed", "card": CARD, "stage": stage, "run_id": run_id,
                          "error": checkpoint["error"],
                          "reserved_months_labels_opened": False, "hidden_labels_opened": False})
        campaign._write_progress(f"r154_{stage}_failed_runtime",
                                 next_action="audit R154 failure checkpoint before any retry", r154_run_id=run_id)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Register or run R154 LightGBM median residual screen")
    parser.add_argument("--register-only", action="store_true")
    parser.add_argument("--amend-registration", action="store_true")
    parser.add_argument("--reason", default="")
    parser.add_argument("--screen", action="store_true")
    parser.add_argument("--development", action="store_true")
    args = parser.parse_args()
    if sum((args.register_only, args.amend_registration, args.screen, args.development)) != 1:
        parser.error("choose exactly one of --register-only, --amend-registration, --screen, or --development")
    if args.register_only:
        print(json.dumps(register_card(), indent=2, ensure_ascii=False))
    elif args.amend_registration:
        print(json.dumps(amend_registration_after_zero_fit_failure(args.reason), indent=2, ensure_ascii=False))
    else:
        _run("screen" if args.screen else "development")


if __name__ == "__main__":
    main()
