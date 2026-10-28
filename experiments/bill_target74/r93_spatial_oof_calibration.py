"""R93: strict-past spatial residual-difference calibration for both tracks."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from experiments.bill_target74 import runner as campaign


CARD = "R93_strict_past_spatial_residual_difference"
TRACKS = ("off_market", "list_aware")
MONTHS = tuple(campaign.PROTOCOL["development_months"])
PARENTS = {
    "off_market": {
        "run_id": "bill_target74_r52_r48_r50_adaptive_20261025T074051Z",
        "result": Path("outputs/bill_target74/bill_target74_unbounded_20261024/batches")
        / "bill_target74_r52_r48_r50_adaptive_20261025T074051Z/development_results.json",
    },
    "list_aware": {
        "run_id": "bill_target74_r35_house_type_text_development_20261025T040058Z",
        "result": Path("outputs/bill_target74/bill_target74_unbounded_20261024/batches")
        / "bill_target74_r35_house_type_text_development_20261025T040058Z/development_results.json",
    },
}
EARTH_KM = 6371.0088
LATITUDE_CENTER = 43.7
MAX_DISTANCE_KM = 1.5
MAX_NEIGHBORS = 50
MIN_NEIGHBORS = 15
MIN_TYPE_HISTORY = 100
DISTANCE_BANDWIDTH_KM = 0.75
RECENCY_HALF_LIFE_MONTHS = 12.0
SHRINKAGE_PSEUDOCOUNT = 25.0
MAX_ABS_LOG_DELTA = 0.03
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20261027


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return str(path.relative_to(campaign.ROOT)).replace("\\", "/")


def _parent_paths(track: str) -> tuple[Path, Path, Path]:
    result_path = campaign.ROOT / PARENTS[track]["result"]
    prediction_path = result_path.parent / "paired_predictions.csv"
    manifest_path = result_path.parent / "run_manifest.json"
    return result_path, prediction_path, manifest_path


def _verify_parent(track: str) -> dict:
    result_path, prediction_path, manifest_path = _parent_paths(track)
    if not all(path.is_file() for path in (result_path, prediction_path, manifest_path)):
        raise FileNotFoundError(f"incomplete registered parent artifacts for {track}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if result.get("run_id") != PARENTS[track]["run_id"]:
        raise RuntimeError(f"unexpected {track} parent run id")
    if (result.get("data_sha256") != campaign.PROTOCOL["data_sha256"]
            or manifest.get("data_sha256", campaign.PROTOCOL["data_sha256"])
            != campaign.PROTOCOL["data_sha256"]):
        raise RuntimeError(f"{track} parent data hash mismatch")
    if result.get("reserved_months_labels_opened") is not False:
        raise RuntimeError(f"{track} parent reports reserve-label access")
    if result.get("hidden_labels_opened") is not False:
        raise RuntimeError(f"{track} parent reports hidden-label access")
    if manifest.get("paired_predictions_sha256") != _sha256(prediction_path):
        raise RuntimeError(f"{track} parent paired-prediction hash mismatch")
    return {
        "run_id": result["run_id"],
        "result_path": _relative(result_path),
        "result_sha256": _sha256(result_path),
        "predictions_path": _relative(prediction_path),
        "predictions_sha256": _sha256(prediction_path),
        "manifest_path": _relative(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
    }


def register_card() -> dict:
    from experiments.bill_target74 import r04_ablation as identity

    existing = campaign.PROTOCOL["cards"].get(CARD)
    if existing:
        if existing.get("status") == "registered":
            if existing.get("source_sha256") == _sha256(Path(__file__)):
                return existing
            raise RuntimeError("R93 source changed after registration; inspect before amendment")
        raise RuntimeError(f"R93 already has status {existing.get('status')!r}")

    source = identity._source_identity()
    parents = {track: _verify_parent(track) for track in TRACKS}
    registration = {
        "status": "registered",
        "tracks": list(TRACKS),
        "development_months": list(MONTHS),
        "expected_calibration_fit_starts": (len(MONTHS) - 1) * len(TRACKS),
        "parents": parents,
        "method": (
            "For each development month, use only earlier safe out-of-fold predictions and labels. "
            "Within the same house type, estimate a recency-weighted median log error among up to 50 "
            "nearest past OOF homes within 1.5 km. Subtract the recency-weighted whole-type median "
            "log error so only the local residual difference is applied. Shrink by effective neighbor "
            "count/(effective neighbor count+25), clip to +/-0.03 log points, then multiply the base "
            "prediction by exp(correction) and round to the nearest dollar. The first month, missing "
            "coordinates, groups with fewer than 100 past rows, or queries with fewer than 15 local "
            "neighbors keep the base prediction unchanged."
        ),
        "difference_from_prior_work": (
            "R12 used house-type x predicted-price-bin residual corrections; R58/R59 fit a global "
            "price-dependent residual calibration and worsened both tracks. R93 uses spatially local "
            "past-OOF residual differences within house type, subtracts the type-wide temporal bias, "
            "and changes neither the base learner nor its blend."
        ),
        "parameters": {
            "earth_radius_km": EARTH_KM,
            "latitude_projection_center": LATITUDE_CENTER,
            "maximum_distance_km": MAX_DISTANCE_KM,
            "maximum_neighbors": MAX_NEIGHBORS,
            "minimum_neighbors": MIN_NEIGHBORS,
            "minimum_type_history_rows": MIN_TYPE_HISTORY,
            "distance_gaussian_bandwidth_km": DISTANCE_BANDWIDTH_KM,
            "recency_half_life_months": RECENCY_HALF_LIFE_MONTHS,
            "shrinkage_pseudocount": SHRINKAGE_PSEUDOCOUNT,
            "maximum_absolute_log_delta": MAX_ABS_LOG_DELTA,
        },
        "development_gate": dict(campaign.PROTOCOL["gates"]),
        "screen_months": list(campaign.PROTOCOL["screen_months"]),
        "data_sha256": source["data_sha256"],
        "parent_commit": source["commit"],
        "source_sha256": _sha256(Path(__file__)),
        "reserved_months_labels_opened": False,
        "hidden_labels_opened": False,
    }
    campaign.PROTOCOL["cards"][CARD] = registration
    campaign.PROTOCOL_PATH.write_text(
        json.dumps(campaign.PROTOCOL, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    campaign._append({
        "event": "card_registered", "card": CARD, "tracks": list(TRACKS),
        "parents": {track: parents[track]["run_id"] for track in TRACKS},
        "development_months": list(MONTHS),
        "expected_calibration_fit_starts": registration["expected_calibration_fit_starts"],
        "data_sha256": source["data_sha256"], "source_sha256": registration["source_sha256"],
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    campaign._write_progress(
        "r93_registered",
        next_action="run R93 strict-past spatial residual-difference calibration on locked development folds",
        r93_card=CARD,
    )
    return registration


def _load_development_metadata() -> pd.DataFrame:
    data_path = campaign.DATA
    if _sha256(data_path) != campaign.PROTOCOL["data_sha256"]:
        raise RuntimeError("cached listings parquet hash differs from the frozen campaign data")
    month_filters = []
    for month in MONTHS:
        start = pd.Timestamp(month + "-01")
        month_filters.append([
            ("date_sold", ">=", start),
            ("date_sold", "<", start + pd.offsets.MonthBegin(1)),
        ])
    metadata = pd.read_parquet(
        data_path,
        columns=["ml_num", "date_sold", "sold_price", "house_type_name", "latitude", "longitude"],
        filters=month_filters,
    )
    metadata["date_sold"] = pd.to_datetime(metadata.date_sold, errors="raise")
    metadata["month"] = metadata.date_sold.dt.to_period("M").astype(str)
    observed = sorted(metadata.month.unique().tolist())
    if observed != sorted(MONTHS):
        raise RuntimeError(f"date-filtered metadata returned unexpected months: {observed}")
    if set(observed).intersection(campaign.PROTOCOL["reserved_months"]):
        raise RuntimeError("reserved-month rows appeared in the exact development-fold filter")
    if metadata.ml_num.astype(str).duplicated().any():
        raise RuntimeError("development metadata contains duplicate listing IDs")
    metadata["ml_num"] = metadata.ml_num.astype(str)
    metadata["source_sold_price"] = pd.to_numeric(metadata.sold_price, errors="raise")
    metadata = metadata.drop(columns=["sold_price"])
    metadata["house_type_name"] = (
        metadata.house_type_name.astype("string").fillna("__MISSING__").astype(str)
    )
    metadata["latitude"] = pd.to_numeric(metadata.latitude, errors="coerce")
    metadata["longitude"] = pd.to_numeric(metadata.longitude, errors="coerce")
    return metadata


def _load_parent_rows(track: str) -> pd.DataFrame:
    _, prediction_path, _ = _parent_paths(track)
    rows = pd.read_csv(prediction_path, dtype={"ml_num": str})
    if "track" in rows:
        rows = rows.loc[rows.track.eq(track)].copy()
    required = {"month", "ml_num", "sold_price", "candidate_prediction"}
    missing = sorted(required - set(rows.columns))
    if missing:
        raise RuntimeError(f"{track} parent predictions are missing columns {missing}")
    rows = rows[["month", "ml_num", "sold_price", "candidate_prediction"]].copy()
    rows["month"] = rows.month.astype(str)
    rows["ml_num"] = rows.ml_num.astype(str)
    if rows.duplicated(["month", "ml_num"]).any():
        raise RuntimeError(f"{track} parent has duplicate month/listing rows")
    if set(rows.month.unique()) != set(MONTHS) or len(rows) != 15642:
        raise RuntimeError(f"{track} parent does not cover exactly the eight development folds")
    if (not np.isfinite(rows.sold_price.to_numpy(float)).all()
            or (rows.sold_price.to_numpy(float) <= 0).any()
            or not np.isfinite(rows.candidate_prediction.to_numpy(float)).all()
            or (rows.candidate_prediction.to_numpy(float) <= 0).any()):
        raise RuntimeError(f"{track} parent contains invalid sale prices or predictions")
    return rows


def _project_coordinates(latitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
    latitude = np.asarray(latitude, dtype=float)
    longitude = np.asarray(longitude, dtype=float)
    x = longitude * 111.32 * np.cos(np.deg2rad(LATITUDE_CENTER))
    y = latitude * 111.32
    return np.column_stack([x, y])


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    sorted_weights = weights[order]
    total = float(sorted_weights.sum())
    if total <= 0.0:
        raise ValueError("weighted median requires positive total weight")
    index = int(np.searchsorted(np.cumsum(sorted_weights), total * 0.5, side="left"))
    return float(sorted_values[min(index, len(sorted_values) - 1)])


def _recency_weights(dates: pd.Series, cutoff: pd.Timestamp) -> np.ndarray:
    age_months = np.maximum(
        0.0,
        (cutoff - pd.to_datetime(dates, errors="raise")).dt.total_seconds().to_numpy(float)
        / (30.4375 * 24.0 * 60.0 * 60.0),
    )
    return np.exp(-np.log(2.0) * age_months / RECENCY_HALF_LIFE_MONTHS)


def _fit_local_corrections(history: pd.DataFrame, queries: pd.DataFrame,
                           cutoff: pd.Timestamp) -> tuple[np.ndarray, pd.DataFrame]:
    required_history = {
        "date_sold", "house_type_name", "latitude", "longitude",
        "sold_price", "candidate_prediction",
    }
    required_query = {
        "house_type_name", "latitude", "longitude", "candidate_prediction",
    }
    if required_history - set(history.columns) or required_query - set(queries.columns):
        raise ValueError("R93 history or query inputs are incomplete")
    if {"sold_price", "date_sold"}.intersection(queries.columns):
        raise ValueError("R93 query-side calibration inputs must not include outcomes or sale dates")
    history = history.loc[pd.to_datetime(history.date_sold).lt(cutoff)].copy()
    if history.empty:
        raise ValueError("R93 calibrator requires strictly earlier OOF rows")
    actual = history.sold_price.to_numpy(float)
    prediction = history.candidate_prediction.to_numpy(float)
    if (not np.isfinite(actual).all() or (actual <= 0).any()
            or not np.isfinite(prediction).all() or (prediction <= 0).any()):
        raise ValueError("R93 past-OOF residual inputs must be finite and positive")
    history["log_residual"] = np.log(actual / prediction)
    history["house_type_name"] = (
        history.house_type_name.astype("string").fillna("__MISSING__").astype(str)
    )
    queries = queries.copy()
    queries["house_type_name"] = (
        queries.house_type_name.astype("string").fillna("__MISSING__").astype(str)
    )
    output = np.zeros(len(queries), dtype=float)
    records = []
    for property_type, query_index in queries.groupby("house_type_name", sort=False).groups.items():
        query_positions = np.asarray(list(query_index), dtype=int)
        group = history.loc[history.house_type_name.eq(property_type)].copy()
        if len(group) < MIN_TYPE_HISTORY:
            records.extend({"row_index": int(i), "neighbor_count": 0,
                            "effective_neighbor_count": 0.0, "global_center": 0.0,
                            "local_center": 0.0, "correction_log": 0.0}
                           for i in query_positions)
            continue
        valid_hist = (
            np.isfinite(group.latitude.to_numpy(float))
            & np.isfinite(group.longitude.to_numpy(float))
            & group.latitude.between(40.0, 50.0).to_numpy()
            & group.longitude.between(-90.0, -60.0).to_numpy()
        )
        group = group.loc[valid_hist].copy()
        if len(group) < MIN_NEIGHBORS:
            records.extend({"row_index": int(i), "neighbor_count": 0,
                            "effective_neighbor_count": 0.0, "global_center": 0.0,
                            "local_center": 0.0, "correction_log": 0.0}
                           for i in query_positions)
            continue
        coords = _project_coordinates(group.latitude.to_numpy(float), group.longitude.to_numpy(float))
        tree = cKDTree(coords)
        q = queries.iloc[query_positions]
        q_lat = q.latitude.to_numpy(float)
        q_lon = q.longitude.to_numpy(float)
        q_valid = (
            np.isfinite(q_lat) & np.isfinite(q_lon)
            & (q_lat >= 40.0) & (q_lat <= 50.0)
            & (q_lon >= -90.0) & (q_lon <= -60.0)
        )
        if not q_valid.any():
            records.extend({"row_index": int(i), "neighbor_count": 0,
                            "effective_neighbor_count": 0.0, "global_center": 0.0,
                            "local_center": 0.0, "correction_log": 0.0}
                           for i in query_positions)
            continue
        query_coords = _project_coordinates(q_lat[q_valid], q_lon[q_valid])
        k = min(MAX_NEIGHBORS, len(group))
        distances, indices = tree.query(
            query_coords, k=k, distance_upper_bound=MAX_DISTANCE_KM, workers=1,
        )
        if k == 1:
            distances = distances[:, None]
            indices = indices[:, None]
        group_residual = group.log_residual.to_numpy(float)
        group_weights = _recency_weights(group.date_sold, cutoff)
        global_center = _weighted_median(group_residual, group_weights)
        valid_query_positions = np.flatnonzero(q_valid)
        for local_row, original_query_position in enumerate(query_positions):
            if not q_valid[local_row]:
                records.append({"row_index": int(original_query_position), "neighbor_count": 0,
                                "effective_neighbor_count": 0.0, "global_center": global_center,
                                "local_center": global_center, "correction_log": 0.0})
                continue
            valid_index = int(np.searchsorted(valid_query_positions, local_row))
            row_distances = distances[valid_index]
            row_indices = indices[valid_index]
            keep = np.isfinite(row_distances) & (row_indices < len(group))
            neighbor_count = int(keep.sum())
            correction = 0.0
            local_center = global_center
            effective_count = 0.0
            if neighbor_count >= MIN_NEIGHBORS:
                neighbor_idx = row_indices[keep].astype(int)
                distance = row_distances[keep]
                neighbor_age_weights = group_weights[neighbor_idx]
                distance_weights = np.exp(-0.5 * np.square(distance / DISTANCE_BANDWIDTH_KM))
                weights = neighbor_age_weights * distance_weights
                if weights.sum() > 0:
                    effective_count = float(weights.sum() ** 2 / np.square(weights).sum())
                    local_center = _weighted_median(group_residual[neighbor_idx], weights)
                    correction = (local_center - global_center) * (
                        effective_count / (effective_count + SHRINKAGE_PSEUDOCOUNT)
                    )
                    correction = float(np.clip(correction, -MAX_ABS_LOG_DELTA, MAX_ABS_LOG_DELTA))
            output[original_query_position] = correction
            records.append({
                "row_index": int(original_query_position),
                "neighbor_count": neighbor_count,
                "effective_neighbor_count": effective_count,
                "global_center": global_center,
                "local_center": local_center,
                "correction_log": correction,
            })
    diagnostics = pd.DataFrame(records).set_index("row_index").reindex(range(len(queries)))
    if diagnostics.isna().any().any() or not np.isfinite(output).all():
        raise RuntimeError("R93 failed to create finite correction diagnostics for each query")
    return output, diagnostics


def _metrics(actual: np.ndarray, prediction: np.ndarray) -> dict:
    ape = 100.0 * np.abs(prediction - actual) / actual
    return {
        "n": int(len(ape)),
        "mdape_percent": float(np.median(ape)),
        "mape_percent": float(np.mean(ape)),
        "p90_ape_percent": float(np.quantile(ape, 0.90)),
        "ape_over_20_percent": float(np.mean(ape > 20.0) * 100.0),
    }


def _bootstrap_mdape_difference(actual: np.ndarray, base: np.ndarray, candidate: np.ndarray) -> dict:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    base_ape = 100.0 * np.abs(base - actual) / actual
    candidate_ape = 100.0 * np.abs(candidate - actual) / actual
    differences = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    for replicate in range(BOOTSTRAP_REPLICATES):
        index = rng.integers(0, len(actual), size=len(actual))
        differences[replicate] = np.median(candidate_ape[index]) - np.median(base_ape[index])
    return {
        "lower95": float(np.quantile(differences, 0.025)),
        "upper95": float(np.quantile(differences, 0.975)),
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
    }


def _score_track(frame: pd.DataFrame) -> dict:
    actual = frame.sold_price.to_numpy(float)
    base = frame.base_prediction.to_numpy(float)
    candidate = frame.candidate_prediction.to_numpy(float)
    base_metric = _metrics(actual, base)
    candidate_metric = _metrics(actual, candidate)
    monthly = {}
    for month, group in frame.groupby("month", sort=True):
        before = _metrics(group.sold_price.to_numpy(float), group.base_prediction.to_numpy(float))
        after = _metrics(group.sold_price.to_numpy(float), group.candidate_prediction.to_numpy(float))
        monthly[str(month)] = {
            "n": len(group), "base": before, "candidate": after,
            "relative_mdape_improvement_percent": 100.0 * (
                before["mdape_percent"] - after["mdape_percent"]
            ) / before["mdape_percent"],
        }
    gain = 100.0 * (base_metric["mdape_percent"] - candidate_metric["mdape_percent"]) / base_metric["mdape_percent"]
    mape_change = 100.0 * (candidate_metric["mape_percent"] - base_metric["mape_percent"]) / base_metric["mape_percent"]
    months_improved = sum(x["relative_mdape_improvement_percent"] > 0 for x in monthly.values())
    worst_regression = max(
        [0.0] + [-x["relative_mdape_improvement_percent"] for x in monthly.values()]
    )
    bootstrap = _bootstrap_mdape_difference(actual, base, candidate)
    gates = campaign.PROTOCOL["gates"]
    target = 7.0 if frame.track.iloc[0] == "off_market" else 4.0
    gate_checks = {
        "relative_mdape_improvement": gain >= gates["development_relative_mdape_improvement_percent"],
        "months_improved": months_improved >= gates["development_months_improved_min"],
        "worst_month_regression": worst_regression <= gates["development_worst_month_regression_max_percent"],
        "mape_regression": mape_change <= gates["development_mape_regression_max_percent"],
        "bootstrap_upper_below_zero": bootstrap["upper95"] < 0.0,
        "strict_target": candidate_metric["mdape_percent"] < target,
    }
    return {
        "base": base_metric,
        "candidate": candidate_metric,
        "relative_mdape_improvement_percent": gain,
        "mape_relative_change_percent": mape_change,
        "months_improved": months_improved,
        "worst_month_regression_percent": worst_regression,
        "paired_bootstrap_mdape_difference_pp": bootstrap,
        "monthly": monthly,
        "gate_checks": gate_checks,
        "qualifies_development": all(gate_checks.values()),
        "target_mdape_strictly_below": target,
    }


def run() -> dict:
    from experiments.bill_target74 import r04_ablation as identity

    card = campaign.PROTOCOL["cards"].get(CARD)
    if not card or card.get("status") != "registered":
        raise RuntimeError("R93 must be registered before execution")
    if card.get("source_sha256") != _sha256(Path(__file__)):
        raise RuntimeError("R93 source hash changed after preregistration")
    source = identity._source_identity()
    if source["data_sha256"] != card["data_sha256"]:
        raise RuntimeError("R93 source data differs from registration")
    parents = {track: _verify_parent(track) for track in TRACKS}
    metadata = _load_development_metadata()
    run_started = datetime.now(timezone.utc)
    run_id = "r93s_" + run_started.strftime("%Y%m%dT%H%M%SZ")
    output = campaign.CAMPAIGN_ROOT / "batches" / run_id
    output.mkdir(parents=True, exist_ok=False)
    provenance = output / "provenance"
    provenance.mkdir()
    shutil.copy2(campaign.PROTOCOL_PATH, provenance / "registered_campaign_protocol.json")
    for track in TRACKS:
        result_path, prediction_path, manifest_path = _parent_paths(track)
        shutil.copy2(result_path, provenance / f"{track}_parent_result.json")
        shutil.copy2(manifest_path, provenance / f"{track}_parent_run_manifest.json")
        (provenance / f"{track}_parent_predictions.sha256").write_text(
            parents[track]["predictions_sha256"] + "\n", encoding="utf-8"
        )
    campaign._append({
        "event": "card_start", "card": CARD, "run_id": run_id,
        "tracks": list(TRACKS), "development_months": list(MONTHS),
        "expected_calibration_fit_starts": (len(MONTHS) - 1) * len(TRACKS),
        "data_sha256": source["data_sha256"],
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })

    output_frames = []
    fit_details = []
    fit_start_count = 0
    try:
        for track in TRACKS:
            parent_rows = _load_parent_rows(track)
            parent_rows = parent_rows.merge(
                metadata, on=["month", "ml_num"], how="left", validate="one_to_one",
            )
            if parent_rows.date_sold.isna().any():
                raise RuntimeError(f"{track} parent rows failed metadata alignment")
            if not np.array_equal(
                parent_rows.sold_price.to_numpy(float), parent_rows.source_sold_price.to_numpy(float)
            ):
                raise RuntimeError(f"{track} parent labels differ from the date-filtered source rows")
            parent_rows = parent_rows.drop(columns=["source_sold_price"])

            history_parts = []
            for month in MONTHS:
                current = parent_rows.loc[parent_rows.month.eq(month)].copy().reset_index(drop=True)
                if current.empty:
                    raise RuntimeError(f"{track} has no rows for development fold {month}")
                cutoff = pd.Timestamp(month + "-01")
                base = current.candidate_prediction.to_numpy(float)
                if history_parts:
                    history = pd.concat(history_parts, ignore_index=True)
                    if pd.to_datetime(history.date_sold).max() >= cutoff:
                        raise RuntimeError(f"{track}/{month} past-OOF history crossed its cutoff")
                    queries = current[[
                        "house_type_name", "latitude", "longitude", "candidate_prediction",
                    ]].copy()
                    fit_id = f"{campaign.PROTOCOL['campaign_id']}:{CARD}:{track}:{month}:{run_id}"
                    fit_start_count += 1
                    campaign._append({
                        "event": "fit_start", "card": CARD, "run_id": run_id,
                        "fit_id": fit_id, "stage": "calibration", "track": track,
                        "month": month, "training_oof_rows": len(history),
                        "query_rows": len(queries), "history_cutoff_exclusive": cutoff.isoformat(),
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "data_sha256": source["data_sha256"],
                    })
                    try:
                        correction, diagnostics = _fit_local_corrections(history, queries, cutoff)
                        candidate = np.rint(base * np.exp(correction))
                        if not np.isfinite(candidate).all() or (candidate <= 0).any():
                            raise RuntimeError("R93 produced invalid predictions")
                    except Exception as exc:
                        campaign._append({
                            "event": "fit_complete", "card": CARD, "run_id": run_id,
                            "fit_id": fit_id, "stage": "calibration", "track": track,
                            "month": month, "status": "failed",
                            "error": f"{type(exc).__name__}: {exc}",
                        })
                        raise
                    detail = {
                        "fit_id": fit_id, "track": track, "month": month,
                        "status": "complete", "training_oof_rows": len(history),
                        "query_rows": len(queries), "history_cutoff_exclusive": cutoff.isoformat(),
                        "corrected_rows": int(np.count_nonzero(correction)),
                        "median_abs_log_correction": float(np.median(np.abs(correction))),
                        "p95_abs_log_correction": float(np.quantile(np.abs(correction), 0.95)),
                        "mean_neighbor_count": float(diagnostics.neighbor_count.mean()),
                        "median_effective_neighbor_count": float(
                            diagnostics.effective_neighbor_count.median()
                        ),
                    }
                    fit_details.append(detail)
                    campaign._append({
                        "event": "fit_complete", "card": CARD, "run_id": run_id,
                        "stage": "calibration", **detail,
                        "reserved_months_labels_opened": False,
                        "hidden_labels_opened": False,
                    })
                else:
                    correction = np.zeros(len(current), dtype=float)
                    candidate = base.copy()
                    diagnostics = pd.DataFrame({
                        "neighbor_count": np.zeros(len(current), dtype=int),
                        "effective_neighbor_count": np.zeros(len(current), dtype=float),
                        "correction_log": np.zeros(len(current), dtype=float),
                    })
                current["base_prediction"] = base
                current["candidate_prediction"] = candidate
                current["correction_log"] = correction
                current["neighbor_count"] = diagnostics.neighbor_count.to_numpy(int)
                current["effective_neighbor_count"] = diagnostics.effective_neighbor_count.to_numpy(float)
                current["base_ape_percent"] = (
                    100.0 * np.abs(current.base_prediction - current.sold_price) / current.sold_price
                )
                current["candidate_ape_percent"] = (
                    100.0 * np.abs(current.candidate_prediction - current.sold_price) / current.sold_price
                )
                current["track"] = track
                output_frames.append(current[[
                    "month", "track", "ml_num", "sold_price", "base_prediction",
                    "candidate_prediction", "base_ape_percent", "candidate_ape_percent",
                    "house_type_name", "latitude", "longitude", "correction_log",
                    "neighbor_count", "effective_neighbor_count",
                ]])
                history_parts.append(current[[
                    "month", "date_sold", "house_type_name", "latitude", "longitude",
                    "sold_price", "candidate_prediction",
                ]].rename(columns={"candidate_prediction": "candidate_prediction"}))

        paired = pd.concat(output_frames, ignore_index=True)
        if paired.duplicated(["month", "track", "ml_num"]).any():
            raise RuntimeError("R93 output has duplicate month/track/listing rows")
        if set(paired.month.unique()) != set(MONTHS):
            raise RuntimeError("R93 output does not cover every registered development month")
        if set(paired.track.unique()) != set(TRACKS) or len(paired) != 31284:
            raise RuntimeError("R93 output does not cover exactly both development tracks")
        if set(paired.month.unique()).intersection(campaign.PROTOCOL["reserved_months"]):
            raise RuntimeError("R93 output includes a reserved month")
        tracks = {
            track: _score_track(paired.loc[paired.track.eq(track)].copy())
            for track in TRACKS
        }
        result = {
            "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
            "card": CARD, "stage": "strict_past_walk_forward_development",
            "development_months": list(MONTHS), "fit_starts": fit_start_count,
            "tracks": tracks, "fit_details": fit_details,
            "parents": {track: parents[track] for track in TRACKS},
            "spatial_parameters": campaign.PROTOCOL["cards"][CARD]["parameters"],
            "data_sha256": source["data_sha256"],
            "reserved_months_labels_opened": False,
            "reserve_validation_eligible": False,
            "hidden_labels_opened": False,
        }
        paired_path = output / "paired_predictions.csv"
        paired.to_csv(paired_path, index=False)
        result_path = output / "development_results.json"
        result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        code_paths = [
            Path(__file__), campaign.PROTOCOL_PATH,
            campaign.PACKAGE / "r04_ablation.py", campaign.PACKAGE / "runner.py",
        ]
        manifest = {
            "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
            "source_identity": source,
            "source_sha256": {
                _relative(path): _sha256(path) for path in code_paths if path.is_file()
            },
            "parents": parents,
            "data_sha256": source["data_sha256"],
            "fit_starts": fit_start_count, "fit_details": fit_details,
            "paired_predictions_sha256": _sha256(paired_path),
            "result_sha256": _sha256(result_path),
            "reserved_months_labels_opened": False,
            "reserve_validation_eligible": False,
            "hidden_labels_opened": False,
        }
        manifest_path = output / "run_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        report_lines = [
            "# R93：严格过去 OOF 空间残差差值校正", "",
            f"运行：`{run_id}`；数据 SHA-256：`{source['data_sha256']}`。",
            "基线：off-market R52 当前最佳实验候选；list-aware 已验证 R35。",
            "每个月的局部校正仅使用严格早于该月的 OOF 误差；邻居为同房型、1.5 km 内最近 50 套，"
            "以 12 个月半衰期和 0.75 km 高斯距离核加权。校正使用本地中位 log 残差减去同房型全局中位，"
            "按有效邻居数收缩并限制在 ±0.03 log 点。", "",
            "| Track | Base MdAPE | Candidate MdAPE | 相对改善 | MAPE变化 | 月份改善 | bootstrap upper95 (pp) | 达到目标 | 晋级 |",
            "|---|---:|---:|---:|---:|---:|---:|---|---|",
        ]
        for track, score in tracks.items():
            target_met = score["candidate"]["mdape_percent"] < score["target_mdape_strictly_below"]
            report_lines.append(
                f"| {track} | {score['base']['mdape_percent']:.5f}% | "
                f"{score['candidate']['mdape_percent']:.5f}% | "
                f"{score['relative_mdape_improvement_percent']:+.3f}% | "
                f"{score['mape_relative_change_percent']:+.3f}% | "
                f"{score['months_improved']}/8 | "
                f"{score['paired_bootstrap_mdape_difference_pp']['upper95']:+.5f} | "
                f"{'是' if target_met else '否'} | "
                f"{'是' if score['qualifies_development'] else '否'} |"
            )
        report_lines.extend([
            "", "## 逐月 MdAPE", "",
            "| Track | Month | Base | Candidate | 相对改善 |", "|---|---|---:|---:|---:|",
        ])
        for track, score in tracks.items():
            for month, monthly in score["monthly"].items():
                report_lines.append(
                    f"| {track} | {month} | {monthly['base']['mdape_percent']:.5f}% | "
                    f"{monthly['candidate']['mdape_percent']:.5f}% | "
                    f"{monthly['relative_mdape_improvement_percent']:+.3f}% |"
                )
        report_lines.extend([
            "", f"Calibration fit starts：{fit_start_count}（第一折无过去 OOF 样本，不拟合）。",
            "仅读取八个预登记开发月份的坐标、房型和已保存 OOF 预测；保留月与隐藏标签未读取。",
            "保留月验证资格仍为不可用，不能将本轮开发结果描述为独立最终测试。", "",
        ])
        report_path = output / "research_report.md"
        report_path.write_text("\n".join(report_lines), encoding="utf-8")
        passed = {track: score["qualifies_development"] for track, score in tracks.items()}
        campaign._append({
            "event": "card_complete", "card": CARD, "run_id": run_id,
            "fit_starts": fit_start_count, "qualifies_by_track": passed,
            "result_path": _relative(result_path),
            "paired_predictions_sha256": manifest["paired_predictions_sha256"],
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        campaign._write_progress(
            "r93_development_complete",
            next_action=(
                "R93 passed development for: " + ", ".join(t for t, ok in passed.items() if ok)
                if any(passed.values()) else
                "R93 spatial residual calibration failed development; register a distinct evidence-based hypothesis"
            ),
            r93_result=_relative(result_path), r93_run_id=run_id,
            r93_fit_starts=fit_start_count,
        )
        return result
    except Exception as exc:
        checkpoint = {
            "run_id": run_id, "card": CARD,
            "error": f"{type(exc).__name__}: {exc}", "fit_starts": fit_start_count,
            "completed_calibrations": fit_details,
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        }
        (output / "failure_checkpoint.json").write_text(
            json.dumps(checkpoint, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        campaign._append({
            "event": "card_failed", "card": CARD, "run_id": run_id,
            "fit_starts": fit_start_count, "error": checkpoint["error"],
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        campaign._write_progress(
            "r93_failed", next_action="audit the R93 failure checkpoint and ledger before any retry",
            r93_failure=_relative(output / "failure_checkpoint.json"), latest_error=checkpoint["error"],
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register", action="store_true", help="preregister the locked R93 hypothesis")
    args = parser.parse_args()
    if args.register:
        registration = register_card()
        print(json.dumps({"card": CARD, "status": registration["status"],
                          "expected_calibration_fit_starts": registration["expected_calibration_fit_starts"]},
                         ensure_ascii=False), flush=True)
        return
    result = run()
    for track, score in result["tracks"].items():
        print(
            f"{track}: base={score['base']['mdape_percent']:.5f}% "
            f"candidate={score['candidate']['mdape_percent']:.5f}% "
            f"relative_gain={score['relative_mdape_improvement_percent']:+.3f}% "
            f"MAPE_change={score['mape_relative_change_percent']:+.3f}% "
            f"months={score['months_improved']}/8 "
            f"bootstrap_upper95={score['paired_bootstrap_mdape_difference_pp']['upper95']:+.5f} "
            f"qualifies={score['qualifies_development']}", flush=True,
        )
    print(f"report={campaign.CAMPAIGN_ROOT / 'batches' / result['run_id'] / 'research_report.md'}", flush=True)


if __name__ == "__main__":
    main()
