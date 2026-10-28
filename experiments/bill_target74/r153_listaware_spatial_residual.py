"""R153: apply the fixed R151 spatial residual correction to list-aware R35 OOFs."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from experiments.bill_target74 import r22_remarks_tfidf as r22
from experiments.bill_target74 import r93_spatial_oof_calibration as r93
from experiments.bill_target74 import runner as campaign


CARD = "R153_listaware_r35_spatial_residual_clip035"
TRACK = "list_aware"
R35_CARD = "R35_listaware_detached_r22_text_expert"
R35_RUN_ID = "bill_target74_r35_house_type_text_development_20261025T040058Z"
R44_CARD = "R44_past_oof_r35_blend"
R44_RUN_ID = "bill_target74_r44_past_oof_r35_blend_20261025T055003Z"
R44_SEED_MONTHS = ("2022-06", "2022-11")
MONTHS = tuple(campaign.PROTOCOL["development_months"])
SCREEN_MONTHS = tuple(campaign.PROTOCOL["screen_months"])
MAX_DISTANCE_KM = 1.5
MAX_NEIGHBORS = 50
MIN_NEIGHBORS = 15
MIN_TYPE_HISTORY = 100
DISTANCE_BANDWIDTH_KM = 0.75
RECENCY_HALF_LIFE_MONTHS = 12.0
SHRINKAGE_PSEUDOCOUNT = 20.0
MAX_ABS_LOG_DELTA = 0.035
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20261073
AUDITOR_PATH = Path(__file__).with_name("r153_independent_audit.py")


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
    paths = [Path(__file__), AUDITOR_PATH, Path(r93.__file__), Path(r22.__file__), Path(campaign.__file__)]
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("R153 requires its runner, auditor, and pinned helper sources")
    return {_relative(path): _sha256(path) for path in paths}


def _r35_paths() -> tuple[Path, Path, Path]:
    progress = _json(campaign.STATE)
    result = campaign.ROOT / progress["r35_development_result"]
    return result, result.parent / "paired_predictions.csv", result.parent / "run_manifest.json"


def _r44_paths() -> tuple[Path, Path, Path, Path]:
    root = campaign.CAMPAIGN_ROOT / "batches" / R44_RUN_ID
    return (root / "past_oof_predictions.csv", root / "screen_results.json",
            root / "run_manifest.json", root / "paired_predictions.csv")


def _verified_inputs() -> dict:
    protocol = campaign.PROTOCOL
    r35_result_path, r35_predictions_path, r35_manifest_path = _r35_paths()
    r35_result, r35_manifest = _json(r35_result_path), _json(r35_manifest_path)
    r35_card = protocol["cards"].get(R35_CARD, {})
    if not all(path.is_file() for path in (r35_result_path, r35_predictions_path, r35_manifest_path)):
        raise FileNotFoundError("R153 requires complete R35 development artifacts")
    if (r35_result.get("run_id") != R35_RUN_ID
            or r35_result.get("card") != R35_CARD
            or r35_result.get("campaign_id") != protocol["campaign_id"]
            or r35_result.get("data_sha256") != protocol["data_sha256"]
            or r35_result.get("reserved_months_labels_opened") is not False
            or r35_result.get("hidden_labels_opened") is not False
            or r35_manifest.get("campaign_id") != protocol["campaign_id"]
            or r35_manifest.get("run_id") != R35_RUN_ID
            or r35_manifest.get("source_identity", {}).get("commit") != protocol["parent_commit"]
            or r35_manifest.get("source_identity", {}).get("data_sha256") != protocol["data_sha256"]
            or r35_manifest.get("result_sha256") != _sha256(r35_result_path)
            or r35_manifest.get("paired_predictions_sha256") != _sha256(r35_predictions_path)
            or r35_manifest.get("reserved_months_labels_opened") is not False
            or r35_manifest.get("hidden_labels_opened") is not False
            or r35_card.get("status") != "fixed_origin_confirmation_passed"
            or r35_card.get("development_run_id") != R35_RUN_ID):
        raise RuntimeError("R35 identity, hashes, validation status, or label-access flags failed")

    seed_path, r44_result_path, r44_manifest_path, r44_paired_path = _r44_paths()
    if not all(path.is_file() for path in (seed_path, r44_result_path, r44_manifest_path, r44_paired_path)):
        raise FileNotFoundError("R153 requires complete R44 seed OOF artifacts")
    r44_result, r44_manifest = _json(r44_result_path), _json(r44_manifest_path)
    r44_card = protocol["cards"].get(R44_CARD, {})
    allowed_months = set(R44_SEED_MONTHS) | set(MONTHS)
    read_months = set(r44_manifest.get("safe_oof_months_read", []))
    if (r44_result.get("run_id") != R44_RUN_ID
            or r44_result.get("card") != R44_CARD
            or r44_result.get("campaign_id") != protocol["campaign_id"]
            or r44_result.get("data_sha256") != protocol["data_sha256"]
            or set(r44_result.get("seed_months", [])) != set(R44_SEED_MONTHS)
            or r44_result.get("reserved_months_labels_opened") is not False
            or r44_result.get("hidden_labels_opened") is not False
            or r44_manifest.get("campaign_id") != protocol["campaign_id"]
            or r44_manifest.get("run_id") != R44_RUN_ID
            or r44_manifest.get("parent_run_id") != R35_SCREEN_RUN_ID
            or r44_manifest.get("source_identity", {}).get("commit") != protocol["parent_commit"]
            or r44_manifest.get("source_identity", {}).get("data_sha256") != protocol["data_sha256"]
            or r44_manifest.get("seed_oof_sha256") != _sha256(seed_path)
            or r44_manifest.get("paired_predictions_sha256") != _sha256(r44_paired_path)
            or r44_manifest.get("result_sha256") != _sha256(r44_result_path)
            or r44_manifest.get("reserved_months_labels_opened") is not False
            or r44_manifest.get("hidden_labels_opened") is not False
            or not read_months.issubset(allowed_months)
            or set(r44_manifest.get("safe_oof_months_read", [])) & set(protocol["reserved_months"])
            or r44_card.get("parent_run_id") != R35_SCREEN_RUN_ID
            or r44_card.get("variant_run_id") != R35_SCREEN_RUN_ID):
        raise RuntimeError("R44 seed identity, hashes, parent route, safe months, or access flags failed")

    return {
        "r35": {
            "card": R35_CARD, "run_id": R35_RUN_ID,
            "result_path": _relative(r35_result_path), "result_sha256": _sha256(r35_result_path),
            "predictions_path": _relative(r35_predictions_path),
            "predictions_sha256": _sha256(r35_predictions_path),
            "manifest_path": _relative(r35_manifest_path), "manifest_sha256": _sha256(r35_manifest_path),
        },
        "r44": {
            "card": R44_CARD, "run_id": R44_RUN_ID,
            "seed_predictions_path": _relative(seed_path), "seed_predictions_sha256": _sha256(seed_path),
            "result_path": _relative(r44_result_path), "result_sha256": _sha256(r44_result_path),
            "manifest_path": _relative(r44_manifest_path), "manifest_sha256": _sha256(r44_manifest_path),
            "paired_predictions_path": _relative(r44_paired_path),
            "paired_predictions_sha256": _sha256(r44_paired_path),
            "seed_months": list(R44_SEED_MONTHS),
        },
    }


R35_SCREEN_RUN_ID = "bill_target74_r35_house_type_text_20261025T035638Z"


def _read_months(stage: str) -> tuple[str, ...]:
    targets = SCREEN_MONTHS if stage == "screen" else MONTHS
    last_month = max(targets)
    return tuple(sorted(set(R44_SEED_MONTHS) | {month for month in MONTHS if month <= last_month}))


def _read_safe_frame(inputs: dict, stage: str) -> pd.DataFrame:
    read_months = _read_months(stage)
    if set(read_months) & set(campaign.PROTOCOL["reserved_months"]):
        raise RuntimeError("R153 exact month list intersects reserved months")
    if _sha256(campaign.DATA) != campaign.PROTOCOL["data_sha256"]:
        raise RuntimeError("current listings parquet hash differs from the registered campaign")
    filters = []
    for month in read_months:
        start = pd.Timestamp(month + "-01")
        filters.append([("date_sold", ">=", start),
                        ("date_sold", "<", start + pd.offsets.MonthBegin(1))])
    columns = ["ml_num", "date_sold", "sold_price", "house_type_name", "latitude", "longitude"]
    metadata = pd.read_parquet(campaign.DATA, columns=columns, filters=filters)
    metadata["date_sold"] = pd.to_datetime(metadata.date_sold, errors="raise")
    metadata["month"] = metadata.date_sold.dt.to_period("M").astype(str)
    metadata["ml_num"] = metadata.ml_num.astype(str)
    metadata["house_type_name"] = metadata.house_type_name.astype("string").fillna("__MISSING__").astype(str)
    for column in ("sold_price", "latitude", "longitude"):
        metadata[column] = pd.to_numeric(metadata[column], errors="coerce")
    if set(metadata.month.unique()) != set(read_months):
        raise RuntimeError("exact safe metadata filter returned unexpected month coverage")
    if set(metadata.month.unique()) & set(campaign.PROTOCOL["reserved_months"]):
        raise RuntimeError("reserved-month rows appeared in R153 metadata")
    if metadata.duplicated(["month", "ml_num"]).any():
        raise RuntimeError("safe metadata has duplicate month/listing keys")

    r35_path = campaign.ROOT / inputs["r35"]["predictions_path"]
    r35 = pd.read_csv(r35_path, dtype={"ml_num": str})
    required = {"month", "track", "ml_num", "sold_price", "candidate_prediction"}
    if required - set(r35.columns):
        raise RuntimeError("R35 predictions lack the required list-aware OOF columns")
    r35 = r35.loc[r35.track.eq(TRACK) & r35.month.astype(str).isin(MONTHS),
                  ["month", "ml_num", "sold_price", "candidate_prediction"]].copy()
    r35["month"] = r35.month.astype(str)
    r35["ml_num"] = r35.ml_num.astype(str)
    r35["sold_price"] = pd.to_numeric(r35.sold_price, errors="raise")
    r35["base_prediction"] = pd.to_numeric(r35.candidate_prediction, errors="raise")
    r35.drop(columns="candidate_prediction", inplace=True)
    r35 = r35.loc[r35.month.isin(read_months)].copy()
    dev_meta = metadata.loc[metadata.month.isin(MONTHS)].copy()
    dev = r35.rename(columns={"sold_price": "sold_price_r35"}).merge(
        dev_meta, on=["month", "ml_num"], how="outer", validate="one_to_one", indicator=True,
        suffixes=("", "_source"),
    )
    if not dev._merge.eq("both").all():
        raise RuntimeError("R35 list-aware OOF keys do not exactly match their safe metadata rows")
    dev.drop(columns="_merge", inplace=True)
    if not np.array_equal(dev.sold_price.to_numpy(float), dev.sold_price_r35.to_numpy(float)):
        raise RuntimeError("R35 OOF labels differ from exact safe-month source labels")
    base = dev.base_prediction.to_numpy(float)
    if not np.isfinite(base).all() or (base <= 0).any():
        raise RuntimeError("R35 list-aware base predictions are not finite and positive")
    dev = dev[["month", "ml_num", "date_sold", "sold_price", "base_prediction",
               "house_type_name", "latitude", "longitude"]].copy()

    seed_path = campaign.ROOT / inputs["r44"]["seed_predictions_path"]
    seed = pd.read_csv(seed_path, dtype={"ml_num": str})
    seed_required = {"month", "track", "ml_num", "date_sold", "sold_price", "variant_prediction"}
    if seed_required - set(seed.columns):
        raise RuntimeError("R44 seed file lacks required R35 list-aware prediction columns")
    seed = seed.loc[seed.track.eq(TRACK), list(seed_required)].copy()
    seed["month"] = seed.month.astype(str)
    seed["ml_num"] = seed.ml_num.astype(str)
    seed["date_sold"] = pd.to_datetime(seed.date_sold, errors="raise")
    seed["sold_price"] = pd.to_numeric(seed.sold_price, errors="raise")
    seed["base_prediction"] = pd.to_numeric(seed.variant_prediction, errors="raise")
    if (set(seed.month.unique()) != set(R44_SEED_MONTHS)
            or seed.duplicated(["month", "ml_num"]).any()
            or not np.isfinite(seed.base_prediction.to_numpy(float)).all()
            or (seed.base_prediction.to_numpy(float) <= 0).any()):
        raise RuntimeError("R44 list-aware seed rows have invalid coverage, keys, or predictions")
    seed.drop(columns="variant_prediction", inplace=True)
    seed_meta = seed.merge(
        metadata, on=["month", "ml_num"], how="left", validate="one_to_one",
        suffixes=("_seed", "_source"), indicator=True,
    )
    if not seed_meta._merge.eq("both").all():
        raise RuntimeError("R44 seed OOF keys do not exist in the exact safe metadata rows")
    if (not np.array_equal(seed_meta.date_sold_seed.to_numpy("datetime64[ns]"),
                           seed_meta.date_sold_source.to_numpy("datetime64[ns]"))
            or not np.array_equal(seed_meta.sold_price_seed.to_numpy(float),
                                  seed_meta.sold_price_source.to_numpy(float))):
        raise RuntimeError("R44 seed dates or labels differ from exact source metadata")
    seed_meta = seed_meta.rename(columns={
        "date_sold_source": "date_sold", "sold_price_source": "sold_price",
        "house_type_name_source": "house_type_name", "latitude_source": "latitude",
        "longitude_source": "longitude",
    })
    seed_meta.drop(columns=["_merge", "date_sold_seed", "sold_price_seed",
                            "house_type_name_seed", "latitude_seed", "longitude_seed"],
                   errors="ignore", inplace=True)
    seed_frame = seed_meta[["month", "ml_num", "date_sold", "sold_price", "base_prediction",
                            "house_type_name", "latitude", "longitude"]].copy()
    frame = pd.concat([seed_frame, dev], ignore_index=True)
    if frame.duplicated(["month", "ml_num"]).any():
        raise RuntimeError("R44 seeds and R35 OOF rows contain duplicate keys")
    if set(frame.month.unique()) != set(read_months):
        raise RuntimeError("R153 combined safe history has unexpected month coverage")
    return frame.sort_values(["month", "ml_num"], kind="mergesort").reset_index(drop=True)


def _fit_correction(history: pd.DataFrame, queries: pd.DataFrame,
                    cutoff: pd.Timestamp) -> tuple[np.ndarray, pd.DataFrame]:
    queries = queries.reset_index(drop=True).copy()
    required_history = {"date_sold", "house_type_name", "latitude", "longitude", "sold_price", "base_prediction"}
    required_query = {"house_type_name", "latitude", "longitude"}
    if required_history - set(history.columns) or required_query - set(queries.columns):
        raise ValueError("R153 history or query inputs are incomplete")
    if {"date_sold", "sold_price", "base_prediction"}.intersection(queries.columns):
        raise ValueError("R153 query calibration inputs must not include target dates, labels, or predictions")
    history = history.loc[pd.to_datetime(history.date_sold, errors="raise").lt(cutoff)].copy()
    if not history.empty and (pd.to_datetime(history.date_sold, errors="raise") >= cutoff).any():
        raise RuntimeError("R153 history contains a target-month or future sale")
    actual = history.sold_price.to_numpy(float)
    prediction = history.base_prediction.to_numpy(float)
    if (not np.isfinite(actual).all() or (actual <= 0).any()
            or not np.isfinite(prediction).all() or (prediction <= 0).any()):
        raise ValueError("R153 past R35 OOF residual inputs must be finite and positive")
    history["log_residual"] = np.log(actual / prediction)
    history["house_type_name"] = history.house_type_name.astype("string").fillna("__MISSING__").astype(str)
    queries["house_type_name"] = queries.house_type_name.astype("string").fillna("__MISSING__").astype(str)
    correction = np.zeros(len(queries), dtype=float)
    records: list[dict | None] = [None] * len(queries)

    for property_type, query_index in queries.groupby("house_type_name", sort=False).groups.items():
        q_positions = np.asarray(list(query_index), dtype=int)
        type_history = history.loc[history.house_type_name.eq(property_type)].copy()
        if len(type_history) < MIN_TYPE_HISTORY:
            for pos in q_positions:
                records[pos] = {"row_index": int(pos), "neighbor_count": 0,
                                "effective_neighbor_count": 0.0, "type_center_log_residual": 0.0,
                                "local_center_log_residual": 0.0, "correction_log": 0.0,
                                "fallback_reason": "insufficient_type_history"}
            continue
        type_weights = r93._recency_weights(type_history.date_sold, cutoff)
        type_residual = type_history.log_residual.to_numpy(float)
        type_center = r93._weighted_median(type_residual, type_weights)
        valid_history = (
            np.isfinite(type_history.latitude.to_numpy(float))
            & np.isfinite(type_history.longitude.to_numpy(float))
            & type_history.latitude.between(40.0, 50.0).to_numpy()
            & type_history.longitude.between(-90.0, -60.0).to_numpy()
        )
        local_history = type_history.loc[valid_history].copy().reset_index(drop=True)
        if len(local_history) < MIN_NEIGHBORS:
            for pos in q_positions:
                records[pos] = {"row_index": int(pos), "neighbor_count": 0,
                                "effective_neighbor_count": 0.0,
                                "type_center_log_residual": float(type_center),
                                "local_center_log_residual": float(type_center),
                                "correction_log": 0.0,
                                "fallback_reason": "insufficient_localizable_history"}
            continue

        tree = cKDTree(r93._project_coordinates(
            local_history.latitude.to_numpy(float), local_history.longitude.to_numpy(float)
        ))
        residual = local_history.log_residual.to_numpy(float)
        weights_past = r93._recency_weights(local_history.date_sold, cutoff)
        query_rows = queries.iloc[q_positions]
        query_valid = (
            np.isfinite(query_rows.latitude.to_numpy(float))
            & np.isfinite(query_rows.longitude.to_numpy(float))
            & query_rows.latitude.between(40.0, 50.0).to_numpy()
            & query_rows.longitude.between(-90.0, -60.0).to_numpy()
        )
        if not query_valid.any():
            for pos in q_positions:
                records[pos] = {"row_index": int(pos), "neighbor_count": 0,
                                "effective_neighbor_count": 0.0,
                                "type_center_log_residual": float(type_center),
                                "local_center_log_residual": float(type_center),
                                "correction_log": 0.0,
                                "fallback_reason": "missing_or_invalid_query_coordinates"}
            continue
        query_coordinates = r93._project_coordinates(
            query_rows.loc[query_valid, "latitude"].to_numpy(float),
            query_rows.loc[query_valid, "longitude"].to_numpy(float),
        )
        neighbors = min(MAX_NEIGHBORS, len(local_history))
        distances, indices = tree.query(
            query_coordinates, k=neighbors, distance_upper_bound=MAX_DISTANCE_KM, workers=1,
        )
        if neighbors == 1:
            distances, indices = distances[:, None], indices[:, None]
        valid_positions = np.flatnonzero(query_valid)
        for local_row, original_pos in enumerate(q_positions):
            if not query_valid[local_row]:
                records[original_pos] = {"row_index": int(original_pos), "neighbor_count": 0,
                                         "effective_neighbor_count": 0.0,
                                         "type_center_log_residual": float(type_center),
                                         "local_center_log_residual": float(type_center),
                                         "correction_log": 0.0,
                                         "fallback_reason": "missing_or_invalid_query_coordinates"}
                continue
            lookup = int(np.searchsorted(valid_positions, local_row))
            row_distance, row_index = distances[lookup], indices[lookup]
            keep = np.isfinite(row_distance) & (row_index < len(local_history))
            neighbor_count = int(keep.sum())
            effective_count, local_center, reason = 0.0, float(type_center), None
            if neighbor_count >= MIN_NEIGHBORS:
                selected = row_index[keep].astype(int)
                distance = row_distance[keep]
                weights = weights_past[selected] * np.exp(-0.5 * np.square(distance / DISTANCE_BANDWIDTH_KM))
                if not np.isfinite(weights).all() or weights.sum() <= 0:
                    reason = "invalid_neighbor_weights"
                else:
                    effective_count = float(weights.sum() ** 2 / np.square(weights).sum())
                    local_center = r93._weighted_median(residual[selected], weights)
                    delta = (local_center - type_center) * (
                        effective_count / (effective_count + SHRINKAGE_PSEUDOCOUNT)
                    )
                    correction[original_pos] = float(np.clip(delta, -MAX_ABS_LOG_DELTA, MAX_ABS_LOG_DELTA))
            else:
                reason = "insufficient_spatial_neighbors"
            records[original_pos] = {
                "row_index": int(original_pos), "neighbor_count": neighbor_count,
                "effective_neighbor_count": effective_count,
                "type_center_log_residual": float(type_center),
                "local_center_log_residual": float(local_center),
                "correction_log": float(correction[original_pos]), "fallback_reason": reason,
            }

    diagnostics = pd.DataFrame(records)
    if len(diagnostics) != len(queries) or any(record is None for record in records) or not np.isfinite(correction).all():
        raise RuntimeError("R153 failed to create complete finite calibration diagnostics")
    return correction, diagnostics


def register_card() -> dict:
    existing = campaign.PROTOCOL["cards"].get(CARD)
    inputs = _verified_inputs()
    hashes = _code_hashes()
    if existing:
        if existing.get("status") != "registered":
            raise RuntimeError(f"{CARD} is already {existing.get('status')}; inspect its runs")
        if existing.get("code_sha256") != hashes or existing.get("input_artifacts") != inputs:
            raise RuntimeError("R153 code or input hashes changed after preregistration")
        return existing

    preflight = _read_safe_frame(inputs, "screen")
    card = {
        "status": "registered", "stage": "fixed_screen_then_eight_month_development",
        "tracks": [TRACK], "screen_months": list(SCREEN_MONTHS),
        "development_months": list(MONTHS), "seed_months": list(R44_SEED_MONTHS),
        "expected_screen_calibration_fit_starts": len(SCREEN_MONTHS),
        "expected_development_calibration_fit_starts": len(MONTHS),
        "learner_training_fit_starts": 0,
        "comparison_parent": {"card": R35_CARD, "run_id": R35_RUN_ID},
        "base_candidate": {"card": R35_CARD, "run_id": R35_RUN_ID},
        "method": (
            "Use the saved R35 list-aware OOF prediction as both comparison parent and candidate base. "
            "For each target month, derive log(sold_price/R35_prediction) only from R44 R35 seed OOF "
            "rows and R35 safe development OOF rows strictly earlier than the target. Within the same "
            "house type, use up to 50 nearest homes within 1.5 km. Weight residuals by a Gaussian "
            "distance kernel with 0.75 km bandwidth and a 12-month recency half-life. Subtract the "
            "recency-weighted whole-type residual median, shrink by effective neighbor count/(effective "
            "neighbor count+20), clip to +/-0.035 log points, multiply the R35 base by exp(correction), "
            "and round to the nearest dollar. Require 100 past rows of the type and 15 local neighbors; "
            "missing coordinates or insufficient support leaves the R35 prediction unchanged. No target "
            "month labels are passed to calibration and no learner is retrained."
        ),
        "difference_from_prior_work": (
            "R151 passed its fixed off-market development gates using a related R93 spatial correction, "
            "but applied it to R52 and compared against R22. R153 keeps that correction fixed and tests "
            "it on the distinct validated R35 list-aware route, with R35 compared against itself. R44 "
            "supplies verified early R35 list-aware OOF seeds for the first development month."
        ),
        "parameters": {
            "earth_radius_km": r93.EARTH_KM,
            "latitude_projection_center": r93.LATITUDE_CENTER,
            "max_distance_km": MAX_DISTANCE_KM, "max_neighbors": MAX_NEIGHBORS,
            "min_neighbors": MIN_NEIGHBORS, "min_type_history_rows": MIN_TYPE_HISTORY,
            "distance_gaussian_bandwidth_km": DISTANCE_BANDWIDTH_KM,
            "recency_half_life_months": RECENCY_HALF_LIFE_MONTHS,
            "shrinkage_pseudocount": SHRINKAGE_PSEUDOCOUNT,
            "max_abs_log_delta": MAX_ABS_LOG_DELTA,
        },
        "source_preflight": {
            "safe_history_rows": int(len(preflight)),
            "safe_months": sorted(preflight.month.unique().tolist()),
            "rows_by_month": {str(k): int(v) for k, v in preflight.month.value_counts().sort_index().items()},
            "screen_target_rows": {month: int(preflight.month.eq(month).sum()) for month in SCREEN_MONTHS},
            "screen_read_months": list(_read_months("screen")),
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        },
        "source_data_filter_policy": (
            "Read only exact date_sold month filters for R44 seed months and the R35 safe development "
            "months needed by each stage. The screen reads through 2024-06; development reads exactly "
            "the eight registered safe development months. Never read 2025-05/08 or hidden labels."
        ),
        "screen_gates": {
            "relative_mdape_gain_min_percent": campaign.PROTOCOL["gates"]["screen_relative_mdape_improvement_percent"],
            "both_screen_months_improve": True,
            "mape_regression_max_percent": campaign.PROTOCOL["gates"]["screen_mape_regression_max_percent"],
            "bootstrap_upper95_below_zero": True,
        },
        "development_gates": dict(campaign.PROTOCOL["gates"]),
        "strict_development_mdape_target_percent": 4.0,
        "input_artifacts": inputs, "code_sha256": hashes,
        "upstream_code_hashes": {
            "r35": _json(campaign.ROOT / inputs["r35"]["manifest_path"]).get("code_sha256", {}),
            "r44": _json(campaign.ROOT / inputs["r44"]["manifest_path"]).get("code_sha256", {}),
        },
        "data_sha256": campaign.PROTOCOL["data_sha256"],
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        "reserve_validation_eligible": False,
        "registration_amendments": [],
    }
    campaign.PROTOCOL["cards"][CARD] = card
    campaign.PROTOCOL_PATH.write_text(json.dumps(campaign.PROTOCOL, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    campaign._append({
        "event": "card_registered", "card": CARD, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "expected_screen_calibration_fit_starts": len(SCREEN_MONTHS),
        "expected_development_calibration_fit_starts": len(MONTHS),
        "learner_training_fit_starts": 0, "source_sha256": hashes[_relative(Path(__file__))],
        "input_artifacts": inputs, "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    campaign._write_progress("r153_registered", next_action="run R153 fixed two-month list-aware spatial residual screen", r153_card=CARD)
    return card


def _stage_started(stage: str) -> bool:
    if not campaign.LEDGER.is_file():
        return False
    with campaign.LEDGER.open("r", encoding="utf-8") as stream:
        return any(
            event.get("card") == CARD and event.get("stage") == stage and event.get("event") == "fit_start"
            for line in stream if line.strip() for event in [json.loads(line)]
        )


def _run(stage: str) -> dict:
    card = campaign.PROTOCOL["cards"].get(CARD)
    if stage == "screen" and (not card or card.get("status") != "registered"):
        raise RuntimeError("R153 must be registered and its screen has not already run")
    if stage == "development" and (not card or card.get("status") != "screen_passed"):
        raise RuntimeError("R153 development requires the fixed screen to pass")
    if card.get("code_sha256") != _code_hashes() or card.get("input_artifacts") != _verified_inputs():
        raise RuntimeError("R153 registered source or input hashes changed")
    if _stage_started(stage):
        raise RuntimeError(f"R153 {stage} already has a fit_start event; do not rerun fixed folds")

    inputs = card["input_artifacts"]
    frame = _read_safe_frame(inputs, stage)
    target_months = SCREEN_MONTHS if stage == "screen" else MONTHS
    run_id = ("r153s_" if stage == "screen" else "r153d_") + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = campaign.CAMPAIGN_ROOT / "batches" / run_id
    output.mkdir(parents=True, exist_ok=False)
    provenance = output / "provenance"
    provenance.mkdir()
    for name, item in inputs.items():
        file_items = ([("result_path", "result"), ("predictions_path", "paired_predictions"),
                       ("manifest_path", "manifest")] if name == "r35" else
                      [("result_path", "result"), ("seed_predictions_path", "seed_oof"),
                       ("manifest_path", "manifest"), ("paired_predictions_path", "paired_predictions")])
        for field, label in file_items:
            src = campaign.ROOT / item[field]
            shutil.copy2(src, provenance / f"{name}_{label}{src.suffix}")

    fit_details: list[dict] = []
    outputs: list[pd.DataFrame] = []
    try:
        for month in target_months:
            cutoff = pd.Timestamp(month + "-01")
            history = frame.loc[frame.month.lt(month)].copy()
            query = frame.loc[frame.month.eq(month)].copy().reset_index(drop=True)
            if query.empty:
                raise RuntimeError(f"R153 has no exact safe query rows for {month}")
            if not history.empty and (pd.to_datetime(history.date_sold, errors="raise") >= cutoff).any():
                raise RuntimeError(f"R153 history is not strictly before {month}")
            fit_id = f"{campaign.PROTOCOL['campaign_id']}:{CARD}:{stage}:{TRACK}:{month}:{run_id}"
            history_months = sorted(history.month.astype(str).unique().tolist())
            campaign._append({
                "event": "fit_start", "card": CARD, "stage": stage, "run_id": run_id,
                "fit_id": fit_id, "track": TRACK, "month": month, "calibration_only": True,
                "learner_training_fit": False, "history_months": history_months,
                "history_rows": int(len(history)), "cutoff_exclusive": cutoff.isoformat(),
                "query_columns": ["house_type_name", "latitude", "longitude"],
                "query_label_columns_passed": [],
                "reserved_months_labels_opened": False, "hidden_labels_opened": False,
            })
            started = time.perf_counter()
            query_features = query[["house_type_name", "latitude", "longitude"]].copy()
            corrections, diagnostics = _fit_correction(history, query_features, cutoff)
            base = query.base_prediction.to_numpy(float)
            candidate = np.rint(base * np.exp(corrections))
            if not np.isfinite(candidate).all() or (candidate <= 0).any():
                raise RuntimeError(f"R153 generated invalid predictions for {month}")
            paired = pd.DataFrame({
                "month": query.month.astype(str), "track": TRACK, "ml_num": query.ml_num.astype(str),
                "sold_price": query.sold_price.to_numpy(float), "parent_prediction": base,
                "candidate_prediction": candidate, "base_prediction": base,
                "correction_log": corrections,
                "neighbor_count": diagnostics.neighbor_count.to_numpy(int),
                "effective_neighbor_count": diagnostics.effective_neighbor_count.to_numpy(float),
                "type_center_log_residual": diagnostics.type_center_log_residual.to_numpy(float),
                "local_center_log_residual": diagnostics.local_center_log_residual.to_numpy(float),
                "fallback_reason": diagnostics.fallback_reason,
            })
            paired["parent_ape_percent"] = 100.0 * np.abs(paired.sold_price - paired.parent_prediction) / paired.sold_price
            paired["candidate_ape_percent"] = 100.0 * np.abs(paired.sold_price - paired.candidate_prediction) / paired.sold_price
            outputs.append(paired)
            detail = {
                "fit_id": fit_id, "track": TRACK, "month": month, "status": "complete",
                "seconds": time.perf_counter() - started, "calibration_only": True,
                "learner_training_fit": False, "history_months": history_months,
                "history_rows": int(len(history)), "history_cutoff_exclusive": cutoff.isoformat(),
                "corrected_rows": int(np.count_nonzero(corrections)),
                "coverage_percent": float(100.0 * np.mean(corrections != 0.0)),
                "mean_neighbor_count": float(diagnostics.neighbor_count.mean()),
                "median_effective_neighbor_count": float(diagnostics.effective_neighbor_count.median()),
            }
            fit_details.append(detail)
            campaign._append({
                "event": "fit_complete", "card": CARD, "stage": stage, "run_id": run_id,
                **detail, "reserved_months_labels_opened": False, "hidden_labels_opened": False,
            })

        paired = pd.concat(outputs, ignore_index=True)
        if paired.duplicated(["month", "ml_num"]).any():
            raise RuntimeError("R153 produced duplicate month/listing predictions")
        if stage == "screen":
            score = r22.score_development_confirmation(
                paired[["month", "ml_num", "sold_price", "parent_prediction", "candidate_prediction"]],
                months=SCREEN_MONTHS, target_mdape_strictly_below=4.0,
                bootstrap_replicates=BOOTSTRAP_REPLICATES, bootstrap_seed=BOOTSTRAP_SEED,
                minimum_relative_gain_percent=campaign.PROTOCOL["gates"]["screen_relative_mdape_improvement_percent"],
                minimum_months_improved=len(SCREEN_MONTHS), maximum_worst_month_regression_percent=0.0,
                maximum_mape_regression_percent=campaign.PROTOCOL["gates"]["screen_mape_regression_max_percent"],
            )
            checks = {
                "relative_mdape_gain": score["relative_mdape_improvement_percent"] >= campaign.PROTOCOL["gates"]["screen_relative_mdape_improvement_percent"],
                "both_screen_months_improved": score["months_improved"] == len(SCREEN_MONTHS),
                "mape_guard": score["mape_relative_change_percent"] <= campaign.PROTOCOL["gates"]["screen_mape_regression_max_percent"],
                "bootstrap_upper95_below_zero": score["bootstrap_mdape_difference_percent"]["upper95"] < 0.0,
            }
        else:
            score = r22.score_development_confirmation(
                paired[["month", "ml_num", "sold_price", "parent_prediction", "candidate_prediction"]],
                months=MONTHS, target_mdape_strictly_below=4.0,
                bootstrap_replicates=BOOTSTRAP_REPLICATES, bootstrap_seed=BOOTSTRAP_SEED,
            )
            checks = {
                "development_relative_gates": score["qualifies_development"],
                "strict_listaware_target_below_4_percent": score["target_met"],
            }
        qualifies = all(checks.values())
        result = {
            "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
            "card": CARD, "stage": stage, "track": TRACK, "months": list(target_months),
            "scores": score, "gate_checks": checks, "qualifies": bool(qualifies),
            "calibration_fit_starts": len(fit_details), "learner_training_fit_starts": 0,
            "fit_details": fit_details, "input_artifacts": inputs,
            "filtered_source_months": list(_read_months(stage)),
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
            "reserve_validation_eligible": False,
        }
        pair_path = output / "paired_predictions.csv"
        result_path = output / ("screen_results.json" if stage == "screen" else "development_results.json")
        paired.to_csv(pair_path, index=False)
        result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        manifest = {
            "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
            "card": CARD, "stage": stage, "track": TRACK, "data_sha256": card["data_sha256"],
            "code_sha256": card["code_sha256"], "input_artifacts": inputs,
            "filtered_source_months": list(_read_months(stage)),
            "provenance_copy_files": sorted(path.name for path in provenance.iterdir()),
            "paired_predictions_sha256": _sha256(pair_path), "result_sha256": _sha256(result_path),
            "calibration_fit_starts": len(fit_details), "learner_training_fit_starts": 0,
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        }
        manifest_path = output / "run_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if stage == "screen":
            card.update({
                "status": "screen_passed" if qualifies else "screen_failed",
                "screen_run_id": run_id, "screen_result_path": _relative(result_path),
                "screen_result_sha256": _sha256(result_path),
                "screen_paired_predictions_path": _relative(pair_path),
                "screen_paired_predictions_sha256": _sha256(pair_path),
                "screen_manifest_path": _relative(manifest_path),
                "screen_manifest_sha256": _sha256(manifest_path),
                "screen_metrics": score, "screen_gate_checks": checks,
                "qualifies_screen": bool(qualifies),
            })
            next_action = ("run R153 fixed eight-month development confirmation; screen passed"
                           if qualifies else "retain R35; R153 fixed screen failed; register a distinct method")
        else:
            card.update({
                "status": "development_complete", "development_run_id": run_id,
                "development_result_path": _relative(result_path),
                "development_result_sha256": _sha256(result_path),
                "development_paired_predictions_path": _relative(pair_path),
                "development_paired_predictions_sha256": _sha256(pair_path),
                "development_manifest_path": _relative(manifest_path),
                "development_manifest_sha256": _sha256(manifest_path),
                "development_metrics": score, "development_gate_checks": checks,
                "qualifies_development": bool(qualifies),
            })
            next_action = ("review eligible fixed-origin confirmation; reserve validation remains ineligible"
                           if qualifies else "retain R35; R153 development did not meet all registered gates")
        campaign.PROTOCOL_PATH.write_text(json.dumps(campaign.PROTOCOL, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        campaign._append({
            "event": "card_complete", "card": CARD, "stage": stage, "run_id": run_id,
            "fit_starts": len(fit_details), "learner_training_fit_starts": 0,
            "qualifies": bool(qualifies), "result_path": _relative(result_path),
            "result_sha256": _sha256(result_path), "paired_predictions_sha256": _sha256(pair_path),
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        campaign._write_progress(
            f"r153_{stage}_{'passed' if qualifies else 'failed'}", next_action=next_action,
            r153_run_id=run_id, r153_result=_relative(result_path), r153_learner_training_fit_starts=0,
        )
        return result
    except Exception as exc:
        checkpoint = {
            "run_id": run_id, "card": CARD, "stage": stage,
            "fit_starts_completed": len(fit_details), "fit_start_events_exist": _stage_started(stage),
            "error": f"{type(exc).__name__}: {exc}",
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        }
        (output / "failure_checkpoint.json").write_text(json.dumps(checkpoint, indent=2) + "\n", encoding="utf-8")
        campaign._append({
            "event": "card_failed", "card": CARD, "stage": stage, "run_id": run_id,
            "fit_starts": len(fit_details), "error": checkpoint["error"],
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        campaign._write_progress(
            f"r153_{stage}_failed_runtime", next_action="audit R153 failure checkpoint before any retry",
            r153_run_id=run_id,
        )
        raise


def amend_registration_after_zero_fit_failure(reason: str) -> dict:
    card = campaign.PROTOCOL["cards"].get(CARD)
    if not card or card.get("status") != "registered":
        raise RuntimeError("only a registered R153 card with a zero-fit setup failure can be amended")
    if not reason.strip():
        raise ValueError("an explicit setup-failure reason is required")
    if _stage_started("screen") or _stage_started("development"):
        raise RuntimeError("R153 has a fit_start event; its registered code cannot be amended")
    inputs = _verified_inputs()
    if card.get("input_artifacts") != inputs:
        raise RuntimeError("R153 upstream artifacts changed; do not amend this registration")
    hashes = _code_hashes()
    if card.get("code_sha256") == hashes:
        raise RuntimeError("R153 registered hashes already match; inspect the setup failure instead")
    card.setdefault("registration_amendments", []).append({
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "reason": reason,
        "fit_starts": 0, "previous_code_sha256": card.get("code_sha256"), "current_code_sha256": hashes,
    })
    card["code_sha256"] = hashes
    campaign.PROTOCOL_PATH.write_text(json.dumps(campaign.PROTOCOL, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    campaign._append({
        "event": "registration_amended", "card": CARD,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "reason": reason,
        "fit_starts": 0, "code_sha256": hashes,
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    campaign._write_progress("r153_registration_amended", next_action="run R153 fixed two-month list-aware spatial residual screen", r153_card=CARD)
    return card


def main() -> None:
    parser = argparse.ArgumentParser(description="Register or run R153 list-aware R35 spatial residual screen")
    parser.add_argument("--register-only", action="store_true")
    parser.add_argument("--screen", action="store_true")
    parser.add_argument("--development", action="store_true")
    parser.add_argument("--amend-registration", action="store_true")
    parser.add_argument("--reason", default="")
    args = parser.parse_args()
    if sum((args.register_only, args.screen, args.development, args.amend_registration)) != 1:
        parser.error("choose exactly one of --register-only, --screen, --development, or --amend-registration")
    if args.register_only:
        print(json.dumps(register_card(), indent=2, ensure_ascii=False))
    elif args.amend_registration:
        print(json.dumps(amend_registration_after_zero_fit_failure(args.reason), indent=2, ensure_ascii=False))
    else:
        _run("screen" if args.screen else "development")


if __name__ == "__main__":
    main()
