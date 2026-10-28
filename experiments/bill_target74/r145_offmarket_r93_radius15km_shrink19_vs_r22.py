"""R145: R93 spatial-only residual calibration with shrinkage pseudocount 19."""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from experiments.bill_target74 import r22_remarks_tfidf as r22
from experiments.bill_target74 import r93_spatial_oof_calibration as r93
from experiments.bill_target74 import runner as campaign


CARD = "R145_offmarket_r93_radius15km_shrink19_vs_r22"
TRACK = "off_market"
MONTHS = tuple(campaign.PROTOCOL["development_months"])
SCREEN_MONTHS = tuple(campaign.PROTOCOL["screen_months"])
MAX_DISTANCE_KM = 1.5
MAX_NEIGHBORS = r93.MAX_NEIGHBORS
MIN_NEIGHBORS = r93.MIN_NEIGHBORS
MIN_TYPE_HISTORY = r93.MIN_TYPE_HISTORY
DISTANCE_BANDWIDTH_KM = r93.DISTANCE_BANDWIDTH_KM
RECENCY_HALF_LIFE_MONTHS = r93.RECENCY_HALF_LIFE_MONTHS
SHRINKAGE_PSEUDOCOUNT = 19.0
MAX_ABS_LOG_DELTA = r93.MAX_ABS_LOG_DELTA
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20261062
R22_CARD = "R22_train_fold_tfidf_remarks"
R52_CARD = "R52_r48_r50_strict_past_meta_blend"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return str(path.relative_to(campaign.ROOT)).replace("\\", "/")


def _code_hashes() -> dict[str, str]:
    paths = [Path(__file__), Path(r93.__file__), Path(r22.__file__), Path(campaign.__file__)]
    return {_relative(path): _sha256(path) for path in paths}


def _verified_input(name: str, progress_key: str, expected_card: str) -> dict:
    progress = json.loads(campaign.STATE.read_text(encoding="utf-8"))
    result_path = campaign.ROOT / progress[progress_key]
    predictions_path = result_path.parent / "paired_predictions.csv"
    manifest_path = result_path.parent / "run_manifest.json"
    if not all(path.is_file() for path in (result_path, predictions_path, manifest_path)):
        raise FileNotFoundError(f"incomplete saved {name} artifacts")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if result.get("reserved_months_labels_opened") is not False:
        raise RuntimeError(f"{name} source run did not attest safe reserve access")
    if result.get("hidden_labels_opened") is not False:
        raise RuntimeError(f"{name} source run did not attest hidden-label isolation")
    if result.get("data_sha256") != campaign.PROTOCOL["data_sha256"]:
        raise RuntimeError(f"{name} source data hash differs from the campaign")
    if manifest.get("paired_predictions_sha256") != _sha256(predictions_path):
        raise RuntimeError(f"{name} prediction hash differs from its manifest")
    if manifest.get("result_sha256") != _sha256(result_path):
        raise RuntimeError(f"{name} result hash differs from its manifest")
    if result.get("card") != expected_card:
        raise RuntimeError(f"unexpected {name} card identity")
    return {
        "card": expected_card, "run_id": result["run_id"],
        "result_path": result_path, "result_sha256": _sha256(result_path),
        "predictions_path": predictions_path, "predictions_sha256": _sha256(predictions_path),
        "manifest_path": manifest_path, "manifest_sha256": _sha256(manifest_path),
    }


def _inputs() -> dict:
    return {
        "r52": _verified_input("r52", "r52_result", R52_CARD),
        "r22": _verified_input("r22", "r22_development_result", R22_CARD),
    }


def _input_hashes(inputs: dict) -> dict:
    return {
        name: {
            "card": entry["card"], "run_id": entry["run_id"],
            "result_sha256": entry["result_sha256"],
            "paired_predictions_sha256": entry["predictions_sha256"],
            "manifest_sha256": entry["manifest_sha256"],
        }
        for name, entry in inputs.items()
    }


def _read_safe_frame(inputs: dict, stage: str) -> pd.DataFrame:
    target_months = SCREEN_MONTHS if stage == "screen" else MONTHS
    last_month = max(target_months)
    read_months = tuple(month for month in MONTHS if month <= last_month)
    listings_path = campaign.DATA
    if _sha256(listings_path) != campaign.PROTOCOL["data_sha256"]:
        raise RuntimeError("current listings parquet hash differs from the registered campaign")
    filters = []
    for month in read_months:
        start = pd.Timestamp(month + "-01")
        filters.append([
            ("date_sold", ">=", start),
            ("date_sold", "<", start + pd.offsets.MonthBegin(1)),
        ])
    columns = [
        "ml_num", "date_sold", "sold_price", "house_type_name", "latitude", "longitude",
        "house_area", "bedrooms", "washrooms",
    ]
    metadata = pd.read_parquet(listings_path, columns=columns, filters=filters)
    metadata["date_sold"] = pd.to_datetime(metadata.date_sold, errors="raise")
    metadata["month"] = metadata.date_sold.dt.to_period("M").astype(str)
    if set(metadata.month.unique()) != set(read_months):
        raise RuntimeError("filtered source rows do not match the requested safe month set")
    if set(metadata.month.unique()).intersection(campaign.PROTOCOL["reserved_months"]):
        raise RuntimeError("reserved-month rows appeared in R145's exact safe-date filter")
    if metadata.ml_num.astype(str).duplicated().any():
        raise RuntimeError("safe development metadata contains duplicate listing IDs")
    metadata["ml_num"] = metadata.ml_num.astype(str)
    metadata["house_type_name"] = (
        metadata.house_type_name.astype("string").fillna("__MISSING__").astype(str)
    )
    for column in ("sold_price", "latitude", "longitude", "house_area", "bedrooms", "washrooms"):
        metadata[column] = pd.to_numeric(metadata[column], errors="coerce")

    pieces = {}
    for name, prediction_column in (("r52", "prediction_r52"), ("r22", "prediction_r22")):
        frame = pd.read_csv(inputs[name]["predictions_path"], dtype={"ml_num": str})
        required = {"month", "track", "ml_num", "sold_price", "candidate_prediction"}
        if missing := required - set(frame.columns):
            raise RuntimeError(f"{name} OOF predictions lack columns: {sorted(missing)}")
        frame = frame.loc[frame.track.eq(TRACK) & frame.month.astype(str).isin(read_months),
                          ["month", "ml_num", "sold_price", "candidate_prediction"]].copy()
        frame["month"] = frame.month.astype(str)
        frame["ml_num"] = frame.ml_num.astype(str)
        if frame.duplicated(["month", "ml_num"]).any():
            raise RuntimeError(f"{name} OOF predictions contain duplicate keys")
        frame.rename(columns={
            "sold_price": f"sold_price_{name}", "candidate_prediction": prediction_column,
        }, inplace=True)
        pieces[name] = frame
    paired = pieces["r52"].merge(
        pieces["r22"], on=["month", "ml_num"], how="outer",
        validate="one_to_one", indicator=True,
    )
    if not paired._merge.eq("both").all():
        raise RuntimeError("R52 and R22 OOF predictions do not have identical safe keys")
    paired.drop(columns="_merge", inplace=True)
    paired = paired.merge(
        metadata, on=["month", "ml_num"], how="outer",
        validate="one_to_one", indicator=True,
    )
    if not paired._merge.eq("both").all():
        raise RuntimeError("R52/R22 OOF predictions do not align with safe metadata keys")
    paired.drop(columns="_merge", inplace=True)
    if len(paired) != int(metadata.shape[0]):
        raise RuntimeError("R52/R22 OOF rows do not align one-to-one with safe metadata")
    for name in ("r52", "r22"):
        if not np.array_equal(paired.sold_price.to_numpy(float), paired[f"sold_price_{name}"].to_numpy(float)):
            raise RuntimeError(f"safe labels do not match saved {name.upper()} labels")
        prediction = paired[f"prediction_{name}"].to_numpy(float)
        if not np.isfinite(prediction).all() or (prediction <= 0).any():
            raise RuntimeError(f"R145 {name.upper()} predictions are not finite and positive")
    return paired.loc[paired.month.isin(read_months)].sort_values(
        ["month", "ml_num"], kind="mergesort").reset_index(drop=True)


def _fit_correction(history: pd.DataFrame, queries: pd.DataFrame,
                    cutoff: pd.Timestamp) -> tuple[np.ndarray, pd.DataFrame]:
    queries = queries.reset_index(drop=True)
    required_history = {
        "date_sold", "house_type_name", "latitude", "longitude", "sold_price", "prediction_r52",
    }
    required_query = {"house_type_name", "latitude", "longitude"}
    if required_history - set(history.columns) or required_query - set(queries.columns):
        raise ValueError("R145 history or query inputs are incomplete")
    if {"date_sold", "sold_price", "prediction_r52", "prediction_r22"}.intersection(queries.columns):
        raise ValueError("R145 query-side calibration inputs must not include dates, labels, or predictions")
    history = history.loc[pd.to_datetime(history.date_sold, errors="raise").lt(cutoff)].copy()
    if not history.empty and (pd.to_datetime(history.date_sold, errors="raise") >= cutoff).any():
        raise RuntimeError("R145 history contains a target-month or future sale")
    actual = history.sold_price.to_numpy(float)
    prediction = history.prediction_r52.to_numpy(float)
    if (not np.isfinite(actual).all() or (actual <= 0).any()
            or not np.isfinite(prediction).all() or (prediction <= 0).any()):
        raise ValueError("R145 past R52 OOF residual inputs must be finite and positive")
    history["log_residual"] = np.log(actual / prediction)
    history["house_type_name"] = (
        history.house_type_name.astype("string").fillna("__MISSING__").astype(str)
    )
    queries["house_type_name"] = (
        queries.house_type_name.astype("string").fillna("__MISSING__").astype(str)
    )
    correction = np.zeros(len(queries), dtype=float)
    records = [None] * len(queries)
    if history.empty:
        diagnostics = pd.DataFrame({
            "row_index": np.arange(len(queries), dtype=int),
            "neighbor_count": np.zeros(len(queries), dtype=int),
            "effective_neighbor_count": np.zeros(len(queries), dtype=float),
            "type_center_log_residual": np.zeros(len(queries), dtype=float),
            "correction_log": correction,
            "fallback_reason": ["no_strictly_past_oof_history"] * len(queries),
        })
        return correction, diagnostics

    for property_type, query_index in queries.groupby("house_type_name", sort=False).groups.items():
        q_positions = np.asarray(list(query_index), dtype=int)
        type_history = history.loc[history.house_type_name.eq(property_type)].copy()
        if len(type_history) < MIN_TYPE_HISTORY:
            for pos in q_positions:
                records[pos] = {
                    "row_index": int(pos), "neighbor_count": 0,
                    "effective_neighbor_count": 0.0, "type_center_log_residual": 0.0,
                    "correction_log": 0.0, "fallback_reason": "insufficient_type_history",
                }
            continue
        type_weights = r93._recency_weights(type_history.date_sold, cutoff)
        type_residual = type_history.log_residual.to_numpy(float)
        type_center = r93._weighted_median(type_residual, type_weights)
        h_valid = (
            np.isfinite(type_history.latitude.to_numpy(float))
            & np.isfinite(type_history.longitude.to_numpy(float))
            & type_history.latitude.between(40.0, 50.0).to_numpy()
            & type_history.longitude.between(-90.0, -60.0).to_numpy()
        )
        local_history = type_history.loc[h_valid].copy().reset_index(drop=True)
        if len(local_history) < MIN_NEIGHBORS:
            for pos in q_positions:
                records[pos] = {
                    "row_index": int(pos), "neighbor_count": 0,
                    "effective_neighbor_count": 0.0,
                    "type_center_log_residual": float(type_center),
                    "correction_log": 0.0,
                    "fallback_reason": "insufficient_localizable_history",
                }
            continue
        coordinates = r93._project_coordinates(
            local_history.latitude.to_numpy(float), local_history.longitude.to_numpy(float)
        )
        tree = cKDTree(coordinates)
        group_residual = local_history.log_residual.to_numpy(float)
        group_weights = r93._recency_weights(local_history.date_sold, cutoff)

        q_rows = queries.iloc[q_positions]
        q_valid = (
            np.isfinite(q_rows.latitude.to_numpy(float))
            & np.isfinite(q_rows.longitude.to_numpy(float))
            & q_rows.latitude.between(40.0, 50.0).to_numpy()
            & q_rows.longitude.between(-90.0, -60.0).to_numpy()
        )
        if not q_valid.any():
            for pos in q_positions:
                records[pos] = {
                    "row_index": int(pos), "neighbor_count": 0,
                    "effective_neighbor_count": 0.0,
                    "type_center_log_residual": float(type_center),
                    "correction_log": 0.0,
                    "fallback_reason": "missing_or_invalid_query_coordinates",
                }
            continue
        query_coords = r93._project_coordinates(
            q_rows.loc[q_valid, "latitude"].to_numpy(float),
            q_rows.loc[q_valid, "longitude"].to_numpy(float),
        )
        k = min(MAX_NEIGHBORS, len(local_history))
        distances, indices = tree.query(
            query_coords, k=k, distance_upper_bound=MAX_DISTANCE_KM, workers=1,
        )
        if k == 1:
            distances = distances[:, None]
            indices = indices[:, None]
        valid_query_positions = np.flatnonzero(q_valid)
        for local_row, original_pos in enumerate(q_positions):
            if not q_valid[local_row]:
                records[original_pos] = {
                    "row_index": int(original_pos), "neighbor_count": 0,
                    "effective_neighbor_count": 0.0,
                    "type_center_log_residual": float(type_center),
                    "correction_log": 0.0,
                    "fallback_reason": "missing_or_invalid_query_coordinates",
                }
                continue
            lookup = int(np.searchsorted(valid_query_positions, local_row))
            row_distances = distances[lookup]
            row_indices = indices[lookup]
            keep = np.isfinite(row_distances) & (row_indices < len(local_history))
            neighbor_count = int(keep.sum())
            effective_count = 0.0
            local_center = type_center
            delta = 0.0
            fallback = None
            if neighbor_count >= MIN_NEIGHBORS:
                neighbor_idx = row_indices[keep].astype(int)
                distance = row_distances[keep]
                weights = group_weights[neighbor_idx] * np.exp(
                    -0.5 * np.square(distance / DISTANCE_BANDWIDTH_KM)
                )
                if not np.isfinite(weights).all() or weights.sum() <= 0.0:
                    fallback = "invalid_neighbor_weights"
                else:
                    effective_count = float(weights.sum() ** 2 / np.square(weights).sum())
                    local_center = r93._weighted_median(group_residual[neighbor_idx], weights)
                    delta = (local_center - type_center) * (
                        effective_count / (effective_count + SHRINKAGE_PSEUDOCOUNT)
                    )
                    delta = float(np.clip(delta, -MAX_ABS_LOG_DELTA, MAX_ABS_LOG_DELTA))
                    correction[original_pos] = delta
            else:
                fallback = "insufficient_spatial_neighbors"
            records[original_pos] = {
                "row_index": int(original_pos),
                "neighbor_count": neighbor_count, "effective_neighbor_count": effective_count,
                "type_center_log_residual": float(type_center),
                "local_center_log_residual": float(local_center),
                "correction_log": float(correction[original_pos]), "fallback_reason": fallback,
            }
    diagnostics = pd.DataFrame(records)
    if len(diagnostics) != len(queries) or any(record is None for record in records) or not np.isfinite(correction).all():
        raise RuntimeError("R145 failed to create complete finite calibration diagnostics")
    return correction, diagnostics


def amend_registration_after_zero_fit_setup_failure() -> dict:
    """Record the failed preflight and amend only code hashes after a zero-fit fix."""
    existing = campaign.PROTOCOL["cards"].get(CARD)
    if not existing or existing.get("status") != "registered":
        raise RuntimeError("R145 can only amend an existing registered card")
    input_hashes = _input_hashes(_inputs())
    if existing.get("input_artifacts") != input_hashes:
        raise RuntimeError("R145 inputs changed; this is not a source-only setup amendment")

    events = []
    with campaign.LEDGER.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                event = json.loads(line)
                if event.get("card") == CARD:
                    events.append(event)
    failed_runs = {
        event.get("run_id") for event in events
        if event.get("event") == "card_failed" and event.get("fit_starts", 0) == 0
    }
    fit_runs = {
        event.get("run_id") for event in events
        if event.get("event") == "fit_start"
    }
    if failed_runs & fit_runs:
        raise RuntimeError("R145 setup amendment found a failed run with learner/calibration fits")

    if not failed_runs:
        raise RuntimeError("R145 amendment requires an existing zero-fit setup failure in the ledger")

    old_code_hashes = existing.get("code_sha256", {})
    new_code_hashes = _code_hashes()
    script_path = _relative(Path(__file__))
    if old_code_hashes == new_code_hashes:
        return existing
    existing["code_sha256"] = new_code_hashes
    existing.setdefault("registration_amendments", []).append({
        "at_utc": datetime.now(timezone.utc).isoformat(),
        "old_source_sha256": old_code_hashes.get(script_path),
        "new_source_sha256": new_code_hashes.get(script_path),
        "reason": "repair R145 after a recorded zero-fit setup failure; preserve the registered R52/R22 inputs and method",
        "failed_setup_run_ids": sorted(run_id for run_id in failed_runs if run_id),
        "hypothesis_parameters_gates_and_inputs_unchanged": True,
    })
    campaign.PROTOCOL_PATH.write_text(
        json.dumps(campaign.PROTOCOL, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    campaign._append({
        "event": "card_registration_amended", "card": CARD,
        "old_source_sha256": old_code_hashes.get(script_path),
        "new_source_sha256": new_code_hashes.get(script_path),
        "reason": "repair R145 after a recorded zero-fit setup failure; preserve the registered R52/R22 inputs and method",
        "fit_starts_in_failed_attempts": 0,
        "hypothesis_parameters_gates_and_inputs_unchanged": True,
    })
    campaign._write_progress(
        "r145_registration_amended",
        next_action="retry the R145 fixed screen after the zero-fit setup correction",
        latest_error=None,
        r145_setup_failure_runs=sorted(run_id for run_id in failed_runs if run_id),
    )
    return existing


def register_card() -> dict:
    existing = campaign.PROTOCOL["cards"].get(CARD)
    code_hashes = _code_hashes()
    if existing:
        if existing.get("status") != "registered":
            raise RuntimeError(f"{CARD} is already {existing.get('status')}; inspect its run")
        if existing.get("code_sha256") != code_hashes:
            raise RuntimeError("R145 code changed after preregistration")
        if existing.get("input_artifacts") != _input_hashes(_inputs()):
            raise RuntimeError("R145 input artifacts changed after preregistration")
        return existing
    inputs = _inputs()
    preflight = _read_safe_frame(inputs, "screen")
    card = {
        "status": "registered", "stage": "fixed_screen_then_eight_month_development",
        "tracks": [TRACK], "screen_months": list(SCREEN_MONTHS),
        "development_months": list(MONTHS),
        "expected_calibration_fit_starts": len(SCREEN_MONTHS) + len(MONTHS),
        "learner_training_fit_starts": 0,
        "comparison_parent": {"card": R22_CARD, "run_id": inputs["r22"]["run_id"]},
        "base_candidate": {"card": R52_CARD, "run_id": inputs["r52"]["run_id"]},
        "method": (
            "Use the saved R52 OOF prediction as the candidate base and the R22 OOF prediction as "
            "the comparison parent. For each target month, derive log(sold_price/R52_prediction) "
            "only from earlier safe R52 OOF rows. Within the same house type, use up to 50 nearest "
            "homes within a 1.5 km radius. Weight their residuals by a Gaussian distance kernel "
            "with 0.75 km bandwidth and a 12-month recency half-life. Subtract the recency-weighted "
            "whole-type residual median, shrink by effective neighbor count/(effective neighbor "
            "count+19), clip to +/-0.03 log points, multiply the R52 base by exp(correction), then "
            "round to the nearest dollar. Require 100 past rows of the type and 15 local neighbors; "
            "missing query coordinates or insufficient support leaves the R52 prediction unchanged. "
            "The only R93 parameter changed is the shrinkage pseudocount, from 25 to 19; the radius "
            "remains 1.5 km. No target-month label is passed to calibration and no learner is retrained."
        ),
        "difference_from_prior_work": (
            "R93 applies this spatial-only correction to the R52 candidate with a 1.5 km radius and "
            "25-row shrinkage pseudocount. R145 keeps the radius fixed and lowers the pseudocount to "
            "19, comparing the resulting R52 candidate with "
            "row-aligned validated R22 predictions. This screen comparison was independently "
            "rescored from the saved R93 candidate and is now preregistered against R22."
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
            "safe_rows": int(len(preflight)),
            "safe_months": sorted(preflight.month.unique().tolist()),
            "rows_by_month": {str(k): int(v) for k, v in preflight.month.value_counts().sort_index().items()},
            "screen_target_rows": {
                month: int(preflight.month.eq(month).sum()) for month in SCREEN_MONTHS
            },
            "reserved_months_labels_opened": False,
            "hidden_labels_opened": False,
        },
        "source_data_filter_policy": "Read only exact safe development-month date_sold filters; screen stage reads only months through 2024-06; development stage reads only the fixed eight safe development months. Never read 2025-05/08 or hidden labels.",
        "screen_gates": {
            "relative_mdape_gain_min_percent": campaign.PROTOCOL["gates"]["screen_relative_mdape_improvement_percent"],
            "both_screen_months_improve": True,
            "mape_regression_max_percent": campaign.PROTOCOL["gates"]["screen_mape_regression_max_percent"],
            "bootstrap_upper95_below_zero": True,
        },
        "development_gates": dict(campaign.PROTOCOL["gates"]),
        "input_artifacts": _input_hashes(inputs), "code_sha256": code_hashes,
        "data_sha256": campaign.PROTOCOL["data_sha256"],
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    }
    campaign.PROTOCOL["cards"][CARD] = card
    campaign.PROTOCOL_PATH.write_text(
        json.dumps(campaign.PROTOCOL, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    campaign._append({
        "event": "card_registered", "card": CARD,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "expected_calibration_fit_starts": card["expected_calibration_fit_starts"],
        "learner_training_fit_starts": 0, "source_sha256": code_hashes[_relative(Path(__file__))],
        "input_artifacts": card["input_artifacts"],
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    campaign._write_progress(
        "r145_registered", next_action="run R145 fixed two-month R52/R22 screen at 1.5 km with shrinkage pseudocount 19",
        r145_card=CARD,
    )
    return card


def _run(stage: str) -> dict:
    card = campaign.PROTOCOL["cards"].get(CARD)
    if not card or card.get("status") not in ("registered", "screen_passed"):
        raise RuntimeError("R145 must be registered; development requires a passed fixed screen")
    if card.get("code_sha256") != _code_hashes():
        raise RuntimeError("R145 code/dependency hashes changed after registration")
    if stage == "screen" and card["status"] != "registered":
        raise RuntimeError("R145 screen already ran; do not rerun its fixed folds")
    if stage == "development" and card["status"] != "screen_passed":
        raise RuntimeError("R145 development requires a passed screen")
    inputs = _inputs()
    if _input_hashes(inputs) != card["input_artifacts"]:
        raise RuntimeError("R145 input artifacts changed after preregistration")
    frame = _read_safe_frame(inputs, stage)
    target_months = SCREEN_MONTHS if stage == "screen" else MONTHS
    run_id = ("r145s_" if stage == "screen" else "r145d_") + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = campaign.CAMPAIGN_ROOT / "batches" / run_id
    output.mkdir(parents=True, exist_ok=False)
    (output / "provenance").mkdir()
    provenance_copies = []
    for name, item in inputs.items():
        for kind in ("result_path", "manifest_path"):
            source_path = item[kind]
            destination = output / "provenance" / f"{name}_{source_path.name}"
            destination.write_bytes(source_path.read_bytes())
            provenance_copies.append(_relative(destination))

    fit_details = []
    outputs = []
    try:
        for month in target_months:
            cutoff = pd.Timestamp(month + "-01")
            history = frame.loc[frame.month.lt(month)].copy()
            query = frame.loc[frame.month.eq(month)].copy()
            if query.empty:
                raise RuntimeError(f"R145 has no exact safe query rows for {month}")
            if not history.empty and (pd.to_datetime(history.date_sold, errors="raise") >= cutoff).any():
                raise RuntimeError(f"R145 history is not strictly before {month}")
            fit_id = f"{campaign.PROTOCOL['campaign_id']}:{CARD}:{stage}:{TRACK}:{month}:{run_id}"
            campaign._append({
                "event": "fit_start", "card": CARD, "stage": stage, "run_id": run_id,
                "fit_id": fit_id, "track": TRACK, "month": month,
                "calibration_only": True, "learner_training_fit": False,
                "history_months": sorted(history.month.unique().tolist()),
                "history_rows": len(history), "cutoff_exclusive": cutoff.isoformat(),
                "reserved_months_labels_opened": False, "hidden_labels_opened": False,
            })
            started = time.perf_counter()
            # Calibration sees only features and labels strictly before the target month.
            query_for_calibration = query[["house_type_name", "latitude", "longitude"]].copy()
            corrections, diagnostics = _fit_correction(history, query_for_calibration, cutoff)
            candidate = np.rint(query.prediction_r52.to_numpy(float) * np.exp(corrections))
            if not np.isfinite(candidate).all() or (candidate <= 0).any():
                raise RuntimeError(f"R145 generated invalid predictions for {month}")
            paired = pd.DataFrame({
                "month": query.month, "track": TRACK, "ml_num": query.ml_num,
                "sold_price": query.sold_price,
                "parent_prediction": query.prediction_r22,
                "candidate_prediction": candidate,
                "correction_log": corrections,
                "neighbor_count": diagnostics.neighbor_count.to_numpy(int),
                "effective_neighbor_count": diagnostics.effective_neighbor_count.to_numpy(float),
                "type_center_log_residual": diagnostics.get(
                    "type_center_log_residual", pd.Series(np.zeros(len(query)))
                ).fillna(0.0).to_numpy(float),
            })
            paired["parent_ape_percent"] = 100.0 * np.abs(paired.sold_price - paired.parent_prediction) / paired.sold_price
            paired["candidate_ape_percent"] = 100.0 * np.abs(paired.sold_price - paired.candidate_prediction) / paired.sold_price
            outputs.append(paired)
            detail = {
                "fit_id": fit_id, "track": TRACK, "month": month,
                "status": "complete", "seconds": time.perf_counter() - started,
                "calibration_only": True, "learner_training_fit": False,
                "history_months": sorted(history.month.unique().tolist()),
                "history_rows": int(len(history)),
                "history_cutoff_exclusive": cutoff.isoformat(),
                "corrected_rows": int(np.count_nonzero(corrections)),
                "coverage_percent": float(100.0 * np.mean(np.asarray(corrections) != 0.0)),
                "mean_neighbor_count": float(diagnostics.neighbor_count.mean()),
                "median_effective_neighbor_count": float(diagnostics.effective_neighbor_count.median()),
            }
            fit_details.append(detail)
            campaign._append({
                "event": "fit_complete", "card": CARD, "stage": stage,
                "run_id": run_id, **detail,
                "reserved_months_labels_opened": False, "hidden_labels_opened": False,
            })

        paired = pd.concat(outputs, ignore_index=True)
        if paired.duplicated(["month", "ml_num"]).any():
            raise RuntimeError("R145 produced duplicate month/listing predictions")
        if stage == "screen":
            score = r22.score_development_confirmation(
                paired[["month", "ml_num", "sold_price", "parent_prediction", "candidate_prediction"]],
                months=SCREEN_MONTHS, target_mdape_strictly_below=7.0,
                bootstrap_replicates=BOOTSTRAP_REPLICATES, bootstrap_seed=BOOTSTRAP_SEED,
                minimum_relative_gain_percent=campaign.PROTOCOL["gates"]["screen_relative_mdape_improvement_percent"],
                minimum_months_improved=2, maximum_worst_month_regression_percent=0.0,
                maximum_mape_regression_percent=campaign.PROTOCOL["gates"]["screen_mape_regression_max_percent"],
            )
            checks = {
                "relative_mdape_gain": score["relative_mdape_improvement_percent"] >= campaign.PROTOCOL["gates"]["screen_relative_mdape_improvement_percent"],
                "both_months_improved": score["months_improved"] == 2,
                "mape_guard": score["mape_relative_change_percent"] <= campaign.PROTOCOL["gates"]["screen_mape_regression_max_percent"],
                "bootstrap_upper_below_zero": score["bootstrap_mdape_difference_percent"]["upper95"] < 0.0,
            }
            qualifies = all(checks.values())
        else:
            score = r22.score_development_confirmation(
                paired[["month", "ml_num", "sold_price", "parent_prediction", "candidate_prediction"]],
                months=MONTHS, target_mdape_strictly_below=7.0,
                bootstrap_replicates=BOOTSTRAP_REPLICATES, bootstrap_seed=BOOTSTRAP_SEED,
            )
            checks = {"development_gates": score["qualifies_development"],
                      "strict_mdape_target": score["target_met"]}
            qualifies = all(checks.values())

        result = {
            "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
            "card": CARD, "stage": stage, "track": TRACK,
            "months": list(target_months), "scores": score,
            "gate_checks": checks, "qualifies": bool(qualifies),
            "calibration_fit_starts": len(fit_details), "learner_training_fit_starts": 0,
            "fit_details": fit_details, "input_artifacts": card["input_artifacts"],
            "filtered_source_months": sorted(frame.month.unique().tolist()),
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
            "reserve_validation_eligible": False,
        }
        pair_path = output / "paired_predictions.csv"
        result_path = output / ("screen_results.json" if stage == "screen" else "development_results.json")
        paired.to_csv(pair_path, index=False)
        result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        manifest = {
            "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
            "card": CARD, "stage": stage, "data_sha256": card["data_sha256"],
            "code_sha256": card["code_sha256"], "input_artifacts": card["input_artifacts"],
            "provenance_copies": provenance_copies,
            "filtered_source_months": sorted(frame.month.unique().tolist()),
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
        campaign.PROTOCOL_PATH.write_text(
            json.dumps(campaign.PROTOCOL, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        campaign._append({
            "event": "card_complete", "card": CARD, "stage": stage, "run_id": run_id,
            "fit_starts": len(fit_details), "learner_training_fit_starts": 0,
            "qualifies": bool(qualifies), "result_path": _relative(result_path),
            "result_sha256": _sha256(result_path),
            "paired_predictions_sha256": _sha256(pair_path),
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        next_action = (
            "run R145 eight-month development confirmation; fixed screen passed"
            if stage == "screen" and qualifies else
            "retain R22 and register a different spatial/structural mechanism; R145 screen failed"
            if stage == "screen" else
            "proceed to eligible fixed-origin confirmation only if R145 meets strict development target and gates"
            if qualifies else
            "retain R22; R145 did not reach strict off-market target and development gates"
        )
        campaign._write_progress(
            f"r145_{stage}_{'passed' if qualifies else 'failed'}",
            next_action=next_action, r145_run_id=run_id, r145_result=_relative(result_path),
            r145_learner_training_fit_starts=0,
        )
        return result
    except Exception as exc:
        campaign._append({
            "event": "card_failed", "card": CARD, "stage": stage, "run_id": run_id,
            "fit_starts": len(fit_details), "error": f"{type(exc).__name__}: {exc}",
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        campaign._write_progress(
            f"r145_{stage}_failed_runtime", next_action="audit R145 failure checkpoint before any retry",
            r145_run_id=run_id,
        )
        raise


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Register or run R145 R93 1.5 km correction with shrinkage pseudocount 19")
    parser.add_argument("--register-only", action="store_true")
    parser.add_argument("--amend-registration", action="store_true")
    parser.add_argument("--screen", action="store_true")
    parser.add_argument("--development", action="store_true")
    args = parser.parse_args()
    if sum((args.register_only, args.amend_registration, args.screen, args.development)) != 1:
        parser.error("choose exactly one of --register-only, --amend-registration, --screen, or --development")
    if args.register_only:
        print(json.dumps(register_card(), indent=2, ensure_ascii=False))
    elif args.amend_registration:
        print(json.dumps(amend_registration_after_zero_fit_setup_failure(), indent=2, ensure_ascii=False))
    else:
        if CARD not in campaign.PROTOCOL["cards"]:
            raise RuntimeError("register R145 before running it")
        _run("screen" if args.screen else "development")


if __name__ == "__main__":
    main()
