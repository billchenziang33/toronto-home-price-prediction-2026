"""Independently reconstruct the R143 safe-month spatial corrections and audit."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "experiments/bill_target74/campaign_protocol.json"
PROGRESS_PATH = ROOT / "outputs/bill_target74/bill_target74_unbounded_20261024/progress.json"
LEDGER_PATH = ROOT / "outputs/bill_target74/bill_target74_unbounded_20261024/fits.jsonl"
CARD_KEY = "R143_offmarket_r93_radius175km_vs_r22"  # Historical suffix typo; registered radius is 1.625 km.
TRACK = "off_market"
RADIUS_KM = 1.625
MAX_NEIGHBORS = 50
MIN_NEIGHBORS = 15
MIN_TYPE_HISTORY = 100
DISTANCE_BANDWIDTH_KM = 0.75
RECENCY_HALF_LIFE_MONTHS = 12.0
SHRINKAGE_PSEUDOCOUNT = 25.0
MAX_ABS_LOG_DELTA = 0.03
LATITUDE_CENTER = 43.7


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values, kind="mergesort")
    values, weights = values[order], weights[order]
    threshold = float(weights.sum()) * 0.5
    return float(values[min(int(np.searchsorted(np.cumsum(weights), threshold, side="left")), len(values) - 1)])


def recency_weights(dates: pd.Series, cutoff: pd.Timestamp) -> np.ndarray:
    seconds_per_month = 30.4375 * 24.0 * 60.0 * 60.0
    age = np.maximum(0.0, (cutoff - pd.to_datetime(dates, errors="raise")).dt.total_seconds().to_numpy() / seconds_per_month)
    return np.exp(-np.log(2.0) * age / RECENCY_HALF_LIFE_MONTHS)


def project(latitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
    longitude_km = longitude * 111.32 * np.cos(np.deg2rad(LATITUDE_CENTER))
    latitude_km = latitude * 111.32
    return np.column_stack((longitude_km, latitude_km))


def reconstruct(frame: pd.DataFrame, months: list[str]) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for month in months:
        cutoff = pd.Timestamp(month + "-01")
        history = frame.loc[frame.month.lt(month)].copy()
        query = frame.loc[frame.month.eq(month)].copy().reset_index(drop=True)
        correction = np.zeros(len(query), dtype=float)
        neighbors = np.zeros(len(query), dtype=int)
        effective = np.zeros(len(query), dtype=float)
        type_center_out = np.zeros(len(query), dtype=float)
        for house_type, positions in query.groupby("house_type_name", sort=False).groups.items():
            positions = np.asarray(list(positions), dtype=int)
            typed = history.loc[history.house_type_name.eq(house_type)].copy()
            if len(typed) < MIN_TYPE_HISTORY:
                continue
            type_residual = np.log(typed.sold_price.to_numpy(float) / typed.prediction_r52.to_numpy(float))
            type_weights = recency_weights(typed.date_sold, cutoff)
            center = weighted_median(type_residual, type_weights)
            type_center_out[positions] = center
            valid_history = (
                np.isfinite(typed.latitude.to_numpy(float))
                & np.isfinite(typed.longitude.to_numpy(float))
                & typed.latitude.between(40.0, 50.0).to_numpy()
                & typed.longitude.between(-90.0, -60.0).to_numpy()
            )
            local = typed.loc[valid_history].copy().reset_index(drop=True)
            if len(local) < MIN_NEIGHBORS:
                continue
            tree = cKDTree(project(local.latitude.to_numpy(float), local.longitude.to_numpy(float)))
            local_residual = np.log(local.sold_price.to_numpy(float) / local.prediction_r52.to_numpy(float))
            local_recency = recency_weights(local.date_sold, cutoff)
            for pos in positions:
                row = query.iloc[pos]
                if not (np.isfinite(row.latitude) and np.isfinite(row.longitude)
                        and 40.0 <= row.latitude <= 50.0 and -90.0 <= row.longitude <= -60.0):
                    continue
                distances, indices = tree.query(
                    project(np.asarray([row.latitude]), np.asarray([row.longitude]))[0],
                    k=min(MAX_NEIGHBORS, len(local)), distance_upper_bound=RADIUS_KM, workers=1,
                )
                valid = np.isfinite(distances) & (indices < len(local))
                count = int(valid.sum())
                neighbors[pos] = count
                if count < MIN_NEIGHBORS:
                    continue
                distances = np.asarray(distances)[valid]
                indices = np.asarray(indices)[valid].astype(int)
                weights = local_recency[indices] * np.exp(-0.5 * np.square(distances / DISTANCE_BANDWIDTH_KM))
                if not np.isfinite(weights).all() or weights.sum() <= 0.0:
                    continue
                effective[pos] = float(weights.sum() ** 2 / np.square(weights).sum())
                local_center = weighted_median(local_residual[indices], weights)
                correction[pos] = np.clip(
                    (local_center - center) * effective[pos] / (effective[pos] + SHRINKAGE_PSEUDOCOUNT),
                    -MAX_ABS_LOG_DELTA, MAX_ABS_LOG_DELTA,
                )
        query["audit_correction"] = correction
        query["audit_neighbor_count"] = neighbors
        query["audit_effective_neighbor_count"] = effective
        query["audit_type_center"] = type_center_out
        query["audit_candidate_prediction"] = np.rint(query.prediction_r52.to_numpy(float) * np.exp(correction))
        pieces.append(query)
    return pd.concat(pieces, ignore_index=True)


def score(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    ape = np.abs(actual - prediction) / actual * 100.0
    return {"mdape_percent": float(np.median(ape)), "mape_percent": float(np.mean(ape))}


def main() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    progress = json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))
    card = protocol["cards"][CARD_KEY]
    if card["parameters"]["max_distance_km"] != RADIUS_KM:
        raise RuntimeError("R143 registered radius is not 1.625 km")
    if protocol["budget_policy"] != "unbounded_until_objective_and_validation_complete":
        raise RuntimeError("active campaign budget policy is not unbounded")
    if any(value is not None for key, value in protocol["limits"].items() if key != "learner_threads"):
        raise RuntimeError("active campaign contains a finite fit/time limit")
    incident = protocol.get("reserve_label_access_incident", {})
    if (not protocol["reserved_months_labels_opened"] or protocol["reserve_validation_eligible"]
            or incident.get("incident_id") != "reserve_full_parquet_preflight_20261024"):
        raise RuntimeError("campaign reserve-access incident state is inconsistent with its protocol")

    result_path = ROOT / progress["r143_result"]
    result = json.loads(result_path.read_text(encoding="utf-8"))
    batch = result_path.parent
    prediction_path = batch / "paired_predictions.csv"
    manifest_path = batch / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    code_hashes = card["code_sha256"]
    code_hashes_match = all(sha256(ROOT / path) == digest for path, digest in code_hashes.items())
    if not code_hashes_match or manifest["code_sha256"] != code_hashes:
        raise RuntimeError("registered or manifest source hashes do not match current files")
    if sha256(result_path) != manifest["result_sha256"] or sha256(prediction_path) != manifest["paired_predictions_sha256"]:
        raise RuntimeError("result or paired-prediction hash does not match the run manifest")
    if manifest["data_sha256"] != protocol["data_sha256"]:
        raise RuntimeError("R143 run data hash does not match the active campaign")
    if result["reserved_months_labels_opened"] or result["hidden_labels_opened"]:
        raise RuntimeError("R143 run reports reserve or hidden-label access")

    inputs = card["input_artifacts"]
    parent_frames = {}
    for name, progress_key in (("r52", "r52_result"), ("r22", "r22_development_result")):
        parent_result_path = ROOT / progress[progress_key]
        parent_manifest_path = parent_result_path.parent / "run_manifest.json"
        parent_prediction_path = parent_result_path.parent / "paired_predictions.csv"
        parent_result = json.loads(parent_result_path.read_text(encoding="utf-8"))
        expected = inputs[name]
        if (sha256(parent_result_path) != expected["result_sha256"]
                or sha256(parent_prediction_path) != expected["paired_predictions_sha256"]
                or sha256(parent_manifest_path) != expected["manifest_sha256"]):
            raise RuntimeError(f"R143 {name.upper()} OOF provenance does not match preregistration")
        if parent_result["reserved_months_labels_opened"] or parent_result["hidden_labels_opened"]:
            raise RuntimeError(f"R143 {name.upper()} source reports reserved or hidden access")
        source = pd.read_csv(parent_prediction_path, dtype={"ml_num": str})
        source = source.loc[source.track.eq(TRACK), ["month", "ml_num", "sold_price", "candidate_prediction"]].copy()
        source["month"] = source.month.astype(str)
        if source.duplicated(["month", "ml_num"]).any():
            raise RuntimeError(f"duplicate {name.upper()} parent rows")
        parent_frames[name] = source

    months = list(protocol["development_months"])
    data_path = ROOT / protocol["data_path"]
    if sha256(data_path) != protocol["data_sha256"]:
        raise RuntimeError("source parquet hash differs from the campaign")
    filters = []
    for month in months:
        start = pd.Timestamp(month + "-01")
        filters.append([("date_sold", ">=", start), ("date_sold", "<", start + pd.offsets.MonthBegin(1))])
    metadata = pd.read_parquet(data_path, columns=[
        "ml_num", "date_sold", "sold_price", "house_type_name", "latitude", "longitude",
    ], filters=filters)
    metadata["date_sold"] = pd.to_datetime(metadata.date_sold, errors="raise")
    metadata["month"] = metadata.date_sold.dt.to_period("M").astype(str)
    metadata["ml_num"] = metadata.ml_num.astype(str)
    metadata["house_type_name"] = metadata.house_type_name.astype("string").fillna("__MISSING__").astype(str)
    for column in ("sold_price", "latitude", "longitude"):
        metadata[column] = pd.to_numeric(metadata[column], errors="coerce")
    if set(metadata.month.unique()) != set(months) or set(months).intersection(protocol["reserved_months"]):
        raise RuntimeError("audit filter included an unexpected or reserved month")
    if metadata.duplicated(["month", "ml_num"]).any():
        raise RuntimeError("safe metadata contains duplicate month/listing keys")

    r52 = parent_frames["r52"].rename(columns={"sold_price": "sold_price_r52", "candidate_prediction": "prediction_r52"})
    r22 = parent_frames["r22"].rename(columns={"sold_price": "sold_price_r22", "candidate_prediction": "prediction_r22"})
    frame = r52.merge(r22, on=["month", "ml_num"], how="outer", validate="one_to_one", indicator=True)
    if not frame._merge.eq("both").all():
        raise RuntimeError("R52 and R22 source keys are not exactly aligned")
    frame = frame.drop(columns="_merge").merge(metadata, on=["month", "ml_num"], how="outer", validate="one_to_one", indicator=True)
    if not frame._merge.eq("both").all():
        raise RuntimeError("parent OOF rows do not align with safe source metadata")
    frame = frame.drop(columns="_merge").sort_values(["month", "ml_num"], kind="mergesort").reset_index(drop=True)
    if not np.array_equal(frame.sold_price.to_numpy(float), frame.sold_price_r52.to_numpy(float)):
        raise RuntimeError("safe sale labels differ from R52 OOF labels")
    if not np.array_equal(frame.sold_price.to_numpy(float), frame.sold_price_r22.to_numpy(float)):
        raise RuntimeError("safe sale labels differ from R22 OOF labels")
    reconstructed = reconstruct(frame, months)

    saved = pd.read_csv(prediction_path, dtype={"ml_num": str})
    joined = reconstructed.merge(saved, on=["month", "ml_num"], how="outer", validate="one_to_one", suffixes=("_audit", "_saved"), indicator=True)
    if not joined._merge.eq("both").all() or len(joined) != len(saved) or saved.duplicated(["month", "ml_num"]).any():
        raise RuntimeError("independent candidate keys do not match the saved paired predictions")
    correction_delta = float(np.max(np.abs(joined.audit_correction.to_numpy(float) - joined.correction_log.to_numpy(float))))
    center_delta = float(np.max(np.abs(joined.audit_type_center.to_numpy(float) - joined.type_center_log_residual.to_numpy(float))))
    if not np.array_equal(joined.audit_neighbor_count.to_numpy(int), joined.neighbor_count.to_numpy(int)):
        raise RuntimeError("independent neighbor counts do not match saved diagnostics")
    if correction_delta > 1e-12 or center_delta > 1e-12:
        raise RuntimeError("independently reconstructed corrections differ from saved diagnostics")
    if not np.array_equal(joined.audit_candidate_prediction.to_numpy(float), joined.candidate_prediction.to_numpy(float)):
        raise RuntimeError("independently reconstructed predictions differ from saved candidates")
    if not np.array_equal(joined.prediction_r22.to_numpy(float), joined.parent_prediction.to_numpy(float)):
        raise RuntimeError("saved parent predictions do not match R22 OOF")
    if not np.array_equal(joined.sold_price_audit.to_numpy(float), joined.sold_price_saved.to_numpy(float)):
        raise RuntimeError("saved labels do not match safe source metadata")

    monthly = {}
    for month in months:
        part = joined.loc[joined.month.eq(month)]
        monthly[month] = {
            "n": int(len(part)),
            "parent": score(part.sold_price_audit.to_numpy(float), part.parent_prediction.to_numpy(float)),
            "candidate": score(part.sold_price_audit.to_numpy(float), part.audit_candidate_prediction.to_numpy(float)),
        }
    pooled_parent = score(joined.sold_price_audit.to_numpy(float), joined.parent_prediction.to_numpy(float))
    pooled_candidate = score(joined.sold_price_audit.to_numpy(float), joined.audit_candidate_prediction.to_numpy(float))
    expected = result["scores"]
    metric_delta = max(
        abs(pooled_parent["mdape_percent"] - expected["parent"]["mdape_percent"]),
        abs(pooled_candidate["mdape_percent"] - expected["candidate"]["mdape_percent"]),
        abs(pooled_parent["mape_percent"] - expected["parent"]["mape_percent"]),
        abs(pooled_candidate["mape_percent"] - expected["candidate"]["mape_percent"]),
    )
    if metric_delta > 1e-12:
        raise RuntimeError("independent pooled metrics differ from registered development results")

    with LEDGER_PATH.open("r", encoding="utf-8") as stream:
        ledger = [json.loads(line) for line in stream if line.strip()]
    run_ids = {card["screen_run_id"], card["development_run_id"]}
    events = [event for event in ledger if event.get("card") == CARD_KEY and event.get("run_id") in run_ids]
    starts = [event for event in events if event.get("event") == "fit_start"]
    completes = [event for event in events if event.get("event") == "fit_complete"]
    if len(starts) != 10 or len(completes) != 10:
        raise RuntimeError("R143 screen/development fit starts and completions do not reconcile")
    if any(event.get("reserved_months_labels_opened") or event.get("hidden_labels_opened") for event in events):
        raise RuntimeError("R143 ledger reports reserved or hidden-label access")
    for event in starts:
        if any(history_month >= event["month"] for history_month in event.get("history_months", [])):
            raise RuntimeError("R143 fit ledger includes same-month or future OOF history")

    audit = {
        "screen_run_id": card["screen_run_id"], "development_run_id": card["development_run_id"],
        "card_key": CARD_KEY, "card_key_suffix_note": "Historical key suffix says radius175km; registered parameter and source constant are 1.625 km.",
        "registered_radius_km": card["parameters"]["max_distance_km"], "rows": int(len(joined)),
        "unique_keys": True, "safe_months_only": months, "r143_reserved_or_hidden_labels_opened": False,
        "campaign_reserved_labels_previously_opened": True, "reserve_validation_eligible": False,
        "r52_r22_parent_keys_align": True, "labels_align_with_metadata": True,
        "strict_past_history_verified_from_ledger": True,
        "max_abs_correction_difference": correction_delta, "neighbor_counts_match_exactly": True,
        "max_abs_type_center_difference": center_delta, "candidate_predictions_match_exactly": True,
        "parent_predictions_match_exactly": True, "pooled_metric_max_difference": metric_delta,
        "pooled_parent": pooled_parent, "pooled_candidate": pooled_candidate, "monthly": monthly,
        "result_hash_matches_manifest": True, "paired_predictions_hash_matches_manifest": True,
        "registered_source_hashes_match": code_hashes_match, "parent_input_hashes_match_registration": True,
        "fit_starts": len(starts), "fit_completes": len(completes),
        "campaign_fit_starts": progress["fit_starts"], "limits": protocol["limits"],
        "development_relative_gates_pass": result["scores"]["qualifies_development"],
        "strict_target_below_7_percent_met": result["scores"]["target_met"],
        "promotion_decision": "retain R22; R143 reduces error but remains at or above strict 7.00% target",
    }
    audit["audit_script_sha256"] = sha256(Path(__file__))
    audit_path = batch / "independent_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"audit_path": str(audit_path), "rows": len(joined), "fit_starts": len(starts),
                      "pooled_candidate_mdape": pooled_candidate["mdape_percent"],
                      "strict_target_met": expected["target_met"], "correction_delta": correction_delta,
                      "neighbor_counts_match": True}, indent=2))


if __name__ == "__main__":
    main()
