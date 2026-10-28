"""R109: extend the R97 list-aware residual learner with verified early R35 OOF seeds."""
from __future__ import annotations

import hashlib
import json
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from experiments.bill_target74 import r02_structure_matched_current_parent as parent_verify
from experiments.bill_target74 import r44_past_oof_r35_shrinkage as r44
from experiments.bill_target74 import r93_spatial_oof_calibration as common
from experiments.bill_target74 import r97_past_oof_catboost_residual_meta as r97
from experiments.bill_target74 import runner as campaign


CARD = "R109_seeded_past_oof_catboost_residual_meta"
R44_RUN_ID = "bill_target74_r44_past_oof_r35_blend_20261025T055003Z"
SCREEN_MONTHS = ("2023-03", "2024-06")
SEED_MONTHS = ("2022-06", "2022-11")
SAFE_MONTHS = tuple(sorted((*SEED_MONTHS, *r97.MONTHS)))
MIN_HISTORY_ROWS = 3500
EXPECTED_FIT_STARTS = 2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return str(path.relative_to(campaign.ROOT)).replace("\\", "/")


def _r44_paths() -> tuple[Path, Path, Path, Path]:
    root = campaign.CAMPAIGN_ROOT / "batches" / R44_RUN_ID
    return (root / "past_oof_predictions.csv", root / "screen_results.json",
            root / "run_manifest.json", root / "paired_predictions.csv")


def _verify_r44_seed() -> tuple[pd.DataFrame, dict]:
    seed_path, result_path, manifest_path, paired_path = _r44_paths()
    if not all(path.is_file() for path in (seed_path, result_path, manifest_path, paired_path)):
        raise FileNotFoundError("R109 requires the complete R44 seed OOF artifact set")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    protocol_card = campaign.PROTOCOL["cards"].get("R44_past_oof_r35_blend", {})
    if (manifest.get("run_id") != R44_RUN_ID
            or manifest.get("campaign_id") != campaign.PROTOCOL["campaign_id"]
            or manifest.get("source_identity", {}).get("commit") != campaign.PROTOCOL["parent_commit"]
            or manifest.get("source_identity", {}).get("data_sha256") != campaign.PROTOCOL["data_sha256"]
            or manifest.get("seed_oof_sha256") != _sha256(seed_path)
            or manifest.get("paired_predictions_sha256") != _sha256(paired_path)
            or manifest.get("result_sha256") != _sha256(result_path)
            or manifest.get("reserved_months_labels_opened") is not False
            or manifest.get("hidden_labels_opened") is not False
            or result.get("reserved_months_labels_opened") is not False
            or result.get("hidden_labels_opened") is not False):
        raise RuntimeError("R44 seed identity, hashes, or label-access flags failed verification")
    validated_parent = r44._verified_parent()
    if (protocol_card.get("variant_run_id") != R35_RUN_ID
            or manifest.get("parent_run_id") != R35_RUN_ID
            or validated_parent["screen"]["run_id"] != R35_RUN_ID):
        raise RuntimeError("R44 seed route is not bound to the validated R35 variant")
    if set(manifest.get("safe_oof_months_read", [])) & set(campaign.PROTOCOL["reserved_months"]):
        raise RuntimeError("R44 manifest lists an ineligible reserve month")
    if set(result.get("seed_months", [])) != set(SEED_MONTHS):
        raise RuntimeError("R44 seed months differ from the registered safe OOF months")
    if protocol_card.get("parent_run_id") != R35_RUN_ID:
        raise RuntimeError("R44 registered parent no longer points to R35")

    rows = pd.read_csv(seed_path, dtype={"ml_num": str})
    required = {"month", "track", "ml_num", "date_sold", "sold_price", "variant_prediction"}
    if required - set(rows.columns):
        raise RuntimeError(f"R44 seed OOF file lacks {sorted(required - set(rows.columns))}")
    rows = rows.loc[rows.track.eq("list_aware")].copy()
    rows["month"] = rows.month.astype(str)
    rows["date_sold"] = pd.to_datetime(rows.date_sold, errors="raise")
    rows["base_prediction"] = pd.to_numeric(rows.variant_prediction, errors="raise")
    rows = rows[["month", "ml_num", "date_sold", "sold_price", "base_prediction"]]
    if (set(rows.month.unique()) != set(SEED_MONTHS)
            or rows.duplicated(["month", "ml_num"]).any()
            or not np.isfinite(rows[["sold_price", "base_prediction"]].to_numpy(float)).all()
            or (rows[["sold_price", "base_prediction"]].to_numpy(float) <= 0).any()):
        raise RuntimeError("R44 seed OOF rows have invalid coverage, IDs, or prices")
    info = {
        "run_id": R44_RUN_ID,
        "seed_predictions_path": _relative(seed_path),
        "seed_predictions_sha256": _sha256(seed_path),
        "result_path": _relative(result_path),
        "result_sha256": _sha256(result_path),
        "manifest_path": _relative(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "paired_predictions_sha256": _sha256(paired_path),
        "seed_months": list(SEED_MONTHS),
        "rows": int(len(rows)),
        "reserved_months_labels_opened": False,
        "hidden_labels_opened": False,
    }
    return rows, info


def _read_safe_metadata() -> pd.DataFrame:
    if set(SAFE_MONTHS) & set(campaign.PROTOCOL["reserved_months"]):
        raise RuntimeError("R109 safe-month list intersects an ineligible reserve month")
    columns = [
        "ml_num", "date_sold", "date_listed", "sold_price", "list_price",
        "municipality", "community", "house_type_name", "house_style", "garage",
        "house_area", "bedrooms", "washrooms", "parking_total", "latitude", "longitude",
    ]
    parts = []
    for month in SAFE_MONTHS:
        start = pd.Timestamp(month + "-01")
        stop = start + pd.offsets.MonthBegin(1)
        part = pd.read_parquet(
            campaign.DATA, columns=columns,
            filters=[[('date_sold', '>=', start), ('date_sold', '<', stop)]],
        )
        part["date_sold"] = pd.to_datetime(part.date_sold, errors="raise")
        part["date_listed"] = pd.to_datetime(part.date_listed, errors="raise")
        part["month"] = part.date_sold.dt.to_period("M").astype(str)
        if len(part) == 0 or set(part.month.unique()) != {month}:
            raise RuntimeError(f"R109 exact date filter returned unexpected rows for {month}")
        part["ml_num"] = part.ml_num.astype(str)
        if part.duplicated(["month", "ml_num"]).any():
            raise RuntimeError(f"R109 source metadata has duplicate keys in {month}")
        parts.append(part)
    frame = pd.concat(parts, ignore_index=True)
    if set(frame.month.unique()) != set(SAFE_MONTHS):
        raise RuntimeError("R109 metadata includes a month outside its exact safe filters")
    return frame


def _prepare_inputs() -> dict:
    source = r97._source_identity()
    seed, seed_info = _verify_r44_seed()
    r93_info = parent_verify._verify_r93()
    r93_path = campaign.ROOT / r93_info["predictions_path"]
    r93 = pd.read_csv(r93_path, dtype={"ml_num": str})
    required = {"month", "track", "ml_num", "sold_price", "base_prediction"}
    if required - set(r93.columns):
        raise RuntimeError(f"R93 parent OOF file lacks {sorted(required - set(r93.columns))}")
    r93 = r93.loc[r93.track.eq("list_aware")].copy()
    r93["month"] = r93.month.astype(str)
    if set(r93.month.unique()) != set(r97.MONTHS) or r93.duplicated(["month", "ml_num"]).any():
        raise RuntimeError("R93 list-aware parent OOF rows do not match the eight safe months")
    metadata = _read_safe_metadata()
    metadata["source_sold_price"] = pd.to_numeric(metadata.sold_price, errors="raise")
    r93 = r93.merge(metadata, on=["month", "ml_num"], how="left", validate="one_to_one",
                    suffixes=("", "_source"))
    if r93.date_sold.isna().any() or r93.date_listed.isna().any():
        raise RuntimeError("R93 OOF rows failed to join exact-month metadata")
    if not np.array_equal(r93.sold_price.to_numpy(float), r93.source_sold_price.to_numpy(float)):
        raise RuntimeError("R93 parent OOF labels differ from exact-month source labels")
    if (not np.isfinite(r93.base_prediction.to_numpy(float)).all()
            or (r93.base_prediction.to_numpy(float) <= 0).any()):
        raise RuntimeError("R93 parent predictions are invalid")

    seed_meta = seed.merge(metadata, on=["month", "ml_num"], how="left", validate="one_to_one",
                           suffixes=("", "_source"))
    if seed_meta.date_listed.isna().any() or seed_meta.date_sold_source.isna().any():
        raise RuntimeError("R44 seeds failed to join exact-month source metadata")
    if (not np.array_equal(seed_meta.date_sold.to_numpy("datetime64[ns]"),
                           seed_meta.date_sold_source.to_numpy("datetime64[ns]"))
            or not np.array_equal(seed_meta.sold_price.to_numpy(float),
                                  seed_meta.source_sold_price.to_numpy(float))):
        raise RuntimeError("R44 seed dates or labels differ from exact-month source metadata")
    seed_meta = seed_meta.drop(columns=["date_sold_source", "sold_price_source", "source_sold_price"])

    history = pd.concat([
        seed_meta[["month", "ml_num", "date_sold", "sold_price", "base_prediction",
                   "date_listed", "list_price", "municipality", "community", "house_type_name",
                   "house_style", "garage", "house_area", "bedrooms", "washrooms",
                   "parking_total", "latitude", "longitude"]],
        r93[["month", "ml_num", "date_sold", "sold_price", "base_prediction",
             "date_listed", "list_price", "municipality", "community", "house_type_name",
             "house_style", "garage", "house_area", "bedrooms", "washrooms",
             "parking_total", "latitude", "longitude"]],
    ], ignore_index=True)
    if history.duplicated(["month", "ml_num"]).any():
        raise RuntimeError("R109 combined OOF history contains duplicate month/listing keys")

    current = r93.loc[r93.month.isin(SCREEN_MONTHS)].copy()
    if set(current.month.unique()) != set(SCREEN_MONTHS):
        raise RuntimeError("R109 screen parent rows do not cover both registered months")
    fit_plan = {}
    for month in SCREEN_MONTHS:
        cutoff = pd.Timestamp(month + "-01")
        prior = history.loc[pd.to_datetime(history.date_sold).lt(cutoff)].copy()
        if len(prior) < MIN_HISTORY_ROWS:
            raise RuntimeError(f"R109 has only {len(prior)} past OOF rows for {month}")
        if not pd.to_datetime(prior.date_sold).lt(cutoff).all():
            raise RuntimeError(f"R109 history crossed the strict cutoff for {month}")
        query = current.loc[current.month.eq(month)].copy()
        fit_plan[month] = {"history_rows": int(len(prior)),
                           "history_months": sorted(prior.month.astype(str).unique().tolist()),
                           "query_rows": int(len(query)), "cutoff_exclusive": cutoff.isoformat()}
    return {
        "source": source, "seed": seed, "seed_info": seed_info, "r93_info": r93_info,
        "history": history, "current": current, "fit_plan": fit_plan,
        "metadata_months": list(SAFE_MONTHS),
    }


def _registration_spec(inputs: dict) -> dict:
    return {
        "status": "registered",
        "method": (
            "R97 list-aware CatBoost MAE residual meta-learner, now trained on the verified R44 "
            "2022-06/2022-11 R35 seed OOF rows plus strictly earlier R93/R35 development OOF rows. "
            "Use the unchanged R97 feature set, residual clipping, shrinkage, and CatBoost config; "
            "fit only 2023-03 and 2024-06 screen cutoffs."
        ),
        "difference_from_prior_work": (
            "R97 had no correction in 2023-03 and 2023-07 because its R93-only past history had fewer "
            "than 4,000 rows. R109 adds hash-verified 2022 R35 seed OOF history and lowers the fixed "
            "minimum support to 3,500 so March 2023 can train from 3,962 legitimate earlier OOF rows."
        ),
        "track": "list_aware",
        "parent": "R35 validated list-aware candidate; R93 base_prediction on the safe OOF folds",
        "parent_run_id": "bill_target74_r35_house_type_text_20261025T035638Z",
        "seed_oof_source": inputs["seed_info"],
        "parent_oof_source": inputs["r93_info"],
        "seed_months": list(SEED_MONTHS),
        "screen_months": list(SCREEN_MONTHS),
        "per_screen_cutoff_history": inputs["fit_plan"],
        "minimum_past_oof_rows": MIN_HISTORY_ROWS,
        "fit_plan": {"expected_fit_starts": EXPECTED_FIT_STARTS,
                     "fits": [{"month": m, **inputs["fit_plan"][m]} for m in SCREEN_MONTHS]},
        "target": "clip(log(sold_price / R35_oof_prediction), -0.15, 0.15)",
        "features": {
            "same_as": "R97_past_oof_catboost_residual_meta",
            "list_aware_only": ["log_list_price", "log_list_to_parent"],
            "forbidden": ["sold_price", "date_sold", "external_avm_estimate", "remarks"],
            "history_rule": "Only OOF labels with date_sold strictly before the screen month start may train its fit.",
        },
        "correction": r97.CORRECTION_SHRINKAGE,
        "residual_target_clip": r97.RESIDUAL_TARGET_CLIP,
        "prediction_residual_clip": r97.PREDICTION_RESIDUAL_CLIP,
        "catboost_config": r97.CATBOOST_CONFIG,
        "screen_gate": {
            "pooled_relative_mdape_improvement_percent_min": campaign.PROTOCOL["gates"]["screen_relative_mdape_improvement_percent"],
            "both_months_must_improve": campaign.PROTOCOL["gates"]["screen_both_months_improve"],
            "mape_regression_max_percent": campaign.PROTOCOL["gates"]["screen_mape_regression_max_percent"],
        },
        "data_sha256": inputs["source"]["data_sha256"],
        "parent_commit": inputs["source"]["commit"],
        "source_sha256": _sha256(Path(__file__)),
        "reserved_months_labels_opened": False,
        "hidden_labels_opened": False,
    }


def register_card(inputs: dict) -> dict:
    card = _registration_spec(inputs)
    existing = campaign.PROTOCOL["cards"].get(CARD)
    if existing:
        if existing.get("source_sha256") != card["source_sha256"]:
            raise RuntimeError("R109 is already registered with a different source; create a new card")
        if existing.get("status") != "registered":
            raise RuntimeError(f"R109 already has status {existing.get('status')!r}")
        return existing
    campaign.PROTOCOL["cards"][CARD] = card
    campaign.PROTOCOL_PATH.write_text(
        json.dumps(campaign.PROTOCOL, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    campaign._append({
        "event": "card_registered", "card": CARD, "fit_starts": 0,
        "source_sha256": card["source_sha256"], "data_sha256": card["data_sha256"],
        "seed_oof_source": inputs["seed_info"], "parent_oof_source": inputs["r93_info"],
        "screen_months": list(SCREEN_MONTHS), "expected_fit_starts": EXPECTED_FIT_STARTS,
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    campaign._write_progress(
        "r109_registered", next_action="Run the registered R109 seeded list-aware residual screen",
        r109_card=CARD,
    )
    return card


def _score(current: pd.DataFrame) -> dict:
    y = current.sold_price.to_numpy(float)
    base = current.base_prediction.to_numpy(float)
    candidate = current.candidate_prediction.to_numpy(float)
    before = common._metrics(y, base)
    after = common._metrics(y, candidate)
    monthly = {}
    for month, group in current.groupby("month", sort=True):
        bm = common._metrics(group.sold_price.to_numpy(float), group.base_prediction.to_numpy(float))
        cm = common._metrics(group.sold_price.to_numpy(float), group.candidate_prediction.to_numpy(float))
        monthly[str(month)] = {
            "n": int(len(group)), "parent": bm, "candidate": cm,
            "relative_mdape_improvement_percent": float(
                100.0 * (bm["mdape_percent"] - cm["mdape_percent"]) / bm["mdape_percent"]
            ),
        }
    gain = 100.0 * (before["mdape_percent"] - after["mdape_percent"]) / before["mdape_percent"]
    mape_change = 100.0 * (after["mape_percent"] - before["mape_percent"]) / before["mape_percent"]
    both = all(row["relative_mdape_improvement_percent"] > 0 for row in monthly.values())
    gates = campaign.PROTOCOL["gates"]
    checks = {
        "pooled_relative_mdape_improvement": gain >= gates["screen_relative_mdape_improvement_percent"],
        "both_screen_months_improve": both if gates["screen_both_months_improve"] else True,
        "mape_regression": mape_change <= gates["screen_mape_regression_max_percent"],
    }
    return {
        "parent": before, "candidate": after,
        "relative_mdape_improvement_percent": float(gain),
        "mape_relative_change_percent": float(mape_change),
        "monthly": monthly, "gate_checks": checks,
        "qualifies_screen": all(checks.values()),
    }


def run_screen(inputs: dict, card: dict) -> dict:
    run_id = "r109m_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = campaign.CAMPAIGN_ROOT / "batches" / run_id
    output.mkdir(parents=True, exist_ok=False)
    (output / "fold_predictions").mkdir()
    provenance = output / "provenance"
    provenance.mkdir()
    seed_path, r44_result_path, r44_manifest_path, _ = _r44_paths()
    r93_paths = parent_verify._r93_paths()
    snapshots = [
        (campaign.PROTOCOL_PATH, "registered_campaign_protocol.json"),
        (seed_path, "r44_past_oof_predictions.csv"),
        (r44_result_path, "r44_screen_results.json"),
        (r44_manifest_path, "r44_run_manifest.json"),
        (_r44_paths()[3], "r44_paired_predictions.csv"),
        (r93_paths[0], "r93_development_results.json"),
        (r93_paths[1], "r93_paired_predictions.csv"),
        (r93_paths[2], "r93_run_manifest.json"),
    ]
    for source_path, snapshot_name in snapshots:
        shutil.copy2(source_path, provenance / snapshot_name)
    shutil.copy2(Path(__file__), provenance / (Path(__file__).name + ".source"))

    campaign._append({
        "event": "card_start", "card": CARD, "run_id": run_id,
        "screen_months": list(SCREEN_MONTHS), "expected_fit_starts": EXPECTED_FIT_STARTS,
        "seed_rows": inputs["seed_info"]["rows"], "fit_plan": inputs["fit_plan"],
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    outputs = []
    fit_details = []
    try:
        for month in SCREEN_MONTHS:
            cutoff = pd.Timestamp(month + "-01")
            history = inputs["history"].loc[
                pd.to_datetime(inputs["history"].date_sold).lt(cutoff)
            ].copy().reset_index(drop=True)
            current = inputs["current"].loc[inputs["current"].month.eq(month)].copy()
            current = current.sort_values("ml_num", kind="mergesort").reset_index(drop=True)
            history = history.sort_values(["month", "ml_num"], kind="mergesort").reset_index(drop=True)
            if len(history) < MIN_HISTORY_ROWS or not pd.to_datetime(history.date_sold).lt(cutoff).all():
                raise RuntimeError(f"R109 history failed the strict cutoff for {month}")
            if current.empty or current.duplicated(["month", "ml_num"]).any():
                raise RuntimeError(f"R109 current rows are invalid for {month}")
            history_x, categorical = r97._build_features(history, "list_aware")
            query_x, query_categorical = r97._build_features(current, "list_aware")
            if categorical != query_categorical or list(history_x.columns) != list(query_x.columns):
                raise RuntimeError(f"R109 feature schemas differ for {month}")
            raw_target = np.log(history.sold_price.to_numpy(float) / history.base_prediction.to_numpy(float))
            if not np.isfinite(raw_target).all():
                raise RuntimeError(f"R109 residual target is non-finite for {month}")
            train_target = np.clip(raw_target, -r97.RESIDUAL_TARGET_CLIP, r97.RESIDUAL_TARGET_CLIP)
            fit_id = f"{campaign.PROTOCOL['campaign_id']}:{CARD}:list_aware:{month}:{run_id}"
            detail = {
                "track": "list_aware", "month": month, "fit_id": fit_id,
                "training_oof_rows": int(len(history)),
                "training_oof_months": sorted(history.month.astype(str).unique().tolist()),
                "history_cutoff_exclusive": cutoff.isoformat(),
                "query_rows": int(len(current)), "feature_columns": list(query_x.columns),
                "categorical_features": list(categorical), "catboost_config": r97.CATBOOST_CONFIG,
            }
            campaign._append({
                "event": "fit_start", "card": CARD, "run_id": run_id, "fit_id": fit_id,
                "track": "list_aware", "month": month, "training_oof_rows": int(len(history)),
                "training_oof_months": detail["training_oof_months"],
                "history_cutoff_exclusive": cutoff.isoformat(),
                "data_sha256": card["data_sha256"],
                "reserved_months_labels_opened": False, "hidden_labels_opened": False,
            })
            model = CatBoostRegressor(**r97.CATBOOST_CONFIG)
            started = time.perf_counter()
            try:
                model.fit(history_x, train_target, cat_features=categorical)
                raw_correction = np.asarray(model.predict(query_x), dtype=float)
                if not np.isfinite(raw_correction).all():
                    raise RuntimeError("R109 CatBoost returned non-finite corrections")
            except BaseException as exc:
                campaign._append({
                    "event": "fit_complete", "card": CARD, "run_id": run_id,
                    "fit_id": fit_id, "track": "list_aware", "month": month,
                    "status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                    "seconds": float(time.perf_counter() - started),
                    "error": f"{type(exc).__name__}: {exc}",
                    "reserved_months_labels_opened": False, "hidden_labels_opened": False,
                })
                raise
            correction = r97.CORRECTION_SHRINKAGE * np.clip(
                raw_correction, -r97.PREDICTION_RESIDUAL_CLIP, r97.PREDICTION_RESIDUAL_CLIP
            )
            current["candidate_prediction"] = np.rint(
                current.base_prediction.to_numpy(float) * np.exp(correction)
            )
            current["correction_log"] = correction
            current["history_oof_rows"] = int(len(history))
            current["history_months"] = ",".join(detail["training_oof_months"])
            detail.update({
                "status": "complete", "seconds": float(time.perf_counter() - started),
                "model_fit": True, "effective_config": model.get_all_params(),
                "correction_median": float(np.median(raw_correction)),
                "correction_p90_abs": float(np.quantile(np.abs(raw_correction), .90)),
                "prediction_sha256": "",
            })
            fit_details.append(detail)
            campaign._append({
                "event": "fit_complete", "card": CARD, "run_id": run_id,
                "fit_id": fit_id, "track": "list_aware", "month": month,
                "status": "complete", "seconds": detail["seconds"],
                "training_oof_rows": int(len(history)),
                "reserved_months_labels_opened": False, "hidden_labels_opened": False,
            })
            keep = ["month", "track", "ml_num", "sold_price", "base_prediction",
                    "candidate_prediction", "correction_log", "history_oof_rows", "history_months"]
            fold = output / "fold_predictions" / f"list_aware_{month}.csv"
            current.loc[:, keep].to_csv(fold, index=False)
            detail["prediction_sha256"] = _sha256(fold)
            outputs.append(current.loc[:, keep].copy())

        paired = pd.concat(outputs, ignore_index=True)
        if (len(paired) != sum(inputs["fit_plan"][m]["query_rows"] for m in SCREEN_MONTHS)
                or paired.duplicated(["month", "track", "ml_num"]).any()
                or len(fit_details) != EXPECTED_FIT_STARTS):
            raise RuntimeError("R109 screen prediction coverage or fit count is invalid")
        result = {
            "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
            "card": CARD, "stage": "registered_two_month_screen",
            "track": "list_aware", "screen_months": list(SCREEN_MONTHS),
            "seed_oof_source": inputs["seed_info"], "parent_oof_source": inputs["r93_info"],
            "fit_starts": len(fit_details), "fit_details": fit_details,
            "metrics": _score(paired), "paired_rows": int(len(paired)),
            "reserved_months_labels_opened": False,
            "reserve_validation_eligible": False, "hidden_labels_opened": False,
        }
        result_path = output / "screen_results.json"
        paired_path = output / "paired_predictions.csv"
        result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        paired.to_csv(paired_path, index=False)
        manifest = {
            "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
            "card": CARD, "source_identity": inputs["source"],
            "source_sha256": {"experiments/bill_target74/r109_seeded_past_oof_catboost_residual_meta.py": _sha256(Path(__file__))},
            "input_sha256": {
                "R44_seed_oof": inputs["seed_info"]["seed_predictions_sha256"],
                "R44_manifest": inputs["seed_info"]["manifest_sha256"],
                "R44_result": inputs["seed_info"]["result_sha256"],
                "R93_predictions": inputs["r93_info"]["predictions_sha256"],
                "R93_result": inputs["r93_info"]["result_sha256"],
                "data": inputs["source"]["data_sha256"],
            },
            "r44_seed_run_id": R44_RUN_ID, "r93_parent_run_id": inputs["r93_info"]["run_id"],
            "parent_run_id": card["parent_run_id"], "screen_months": list(SCREEN_MONTHS),
            "safe_oof_months_read": list(SAFE_MONTHS), "minimum_past_oof_rows": MIN_HISTORY_ROWS,
            "fit_starts": len(fit_details), "fit_details": fit_details,
            "runtime": {"python": platform.python_version(), "catboost": metadata.version("catboost"),
                        "numpy": np.__version__, "pandas": pd.__version__},
            "paired_predictions_sha256": _sha256(paired_path),
            "screen_results_sha256": _sha256(result_path),
            "reserved_months_labels_opened": False,
            "reserve_validation_eligible": False, "hidden_labels_opened": False,
        }
        manifest_path = output / "run_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        status = "screen_pass" if result["metrics"]["qualifies_screen"] else "screen_failed"
        campaign.PROTOCOL["cards"][CARD]["status"] = status
        campaign.PROTOCOL["cards"][CARD]["run_id"] = run_id
        campaign.PROTOCOL["cards"][CARD]["result_path"] = _relative(result_path)
        campaign.PROTOCOL["cards"][CARD]["paired_predictions_path"] = _relative(paired_path)
        campaign.PROTOCOL["cards"][CARD]["result_sha256"] = _sha256(result_path)
        campaign.PROTOCOL["cards"][CARD]["paired_predictions_sha256"] = _sha256(paired_path)
        campaign.PROTOCOL_PATH.write_text(
            json.dumps(campaign.PROTOCOL, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        campaign._append({
            "event": "card_complete", "card": CARD, "run_id": run_id,
            "status": status, "fit_starts": len(fit_details),
            "qualifies_screen": result["metrics"]["qualifies_screen"],
            "result_path": _relative(result_path),
            "result_sha256": _sha256(result_path),
            "paired_predictions_sha256": _sha256(paired_path),
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        next_action = (
            "R109 passed screen; run only the registered eight-month list-aware development confirmation"
            if status == "screen_pass" else
            "R109 failed screen; retain R35 and register a distinct list-aware method"
        )
        campaign._write_progress(
            f"r109_{status}", next_action=next_action, r109_run_id=run_id,
            r109_result=_relative(result_path), r109_fit_starts=len(fit_details),
        )
        return result
    except BaseException as exc:
        campaign.PROTOCOL["cards"][CARD]["status"] = "screen_failed_runtime"
        campaign.PROTOCOL["cards"][CARD]["run_id"] = run_id
        campaign.PROTOCOL_PATH.write_text(
            json.dumps(campaign.PROTOCOL, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        campaign._append({
            "event": "card_failed", "card": CARD, "run_id": run_id,
            "status": "screen_failed_runtime", "fit_starts": len(fit_details),
            "fit_completions": sum(d.get("status") == "complete" for d in fit_details),
            "error": f"{type(exc).__name__}: {exc}",
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        campaign._write_progress(
            "r109_screen_failed_runtime",
            next_action="Inspect R109 runtime failure and preserve all started fit records",
            r109_run_id=run_id, r109_error=f"{type(exc).__name__}: {exc}",
        )
        raise


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in {"--preflight", "--run"}:
        raise SystemExit("usage: python -m experiments.bill_target74.r109_seeded_past_oof_catboost_residual_meta --preflight|--run")
    inputs = _prepare_inputs()
    preview = {
        "card": CARD, "seed_rows": inputs["seed_info"]["rows"],
        "seed_months": list(SEED_MONTHS), "fit_plan": inputs["fit_plan"],
        "safe_months": list(SAFE_MONTHS), "reserved_or_hidden_opened": False,
    }
    if sys.argv[1] == "--preflight":
        print(json.dumps(preview, indent=2, ensure_ascii=False))
        return
    card = register_card(inputs)
    result = run_screen(inputs, card)
    print(json.dumps({"run_id": result["run_id"], "fit_starts": result["fit_starts"],
                      "metrics": result["metrics"], "reserved_months_labels_opened": False,
                      "hidden_labels_opened": False}, indent=2, ensure_ascii=False))


R35_RUN_ID = "bill_target74_r35_house_type_text_20261025T035638Z"

if __name__ == "__main__":
    main()
