"""Independently reconstruct and audit R153 list-aware spatial residual outputs."""
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
CARD_KEY = "R153_listaware_r35_spatial_residual_clip035"
TRACK = "list_aware"
R35_CARD = "R35_listaware_detached_r22_text_expert"
R35_RUN_ID = "bill_target74_r35_house_type_text_development_20261025T040058Z"
R44_CARD = "R44_past_oof_r35_blend"
R44_RUN_ID = "bill_target74_r44_past_oof_r35_blend_20261025T055003Z"
R35_SCREEN_RUN_ID = "bill_target74_r35_house_type_text_20261025T035638Z"
SEED_MONTHS = ("2022-06", "2022-11")
SCREEN_MONTHS = ("2023-03", "2024-06")
RADIUS_KM = 1.5
MAX_NEIGHBORS = 50
MIN_NEIGHBORS = 15
MIN_TYPE_HISTORY = 100
DISTANCE_BANDWIDTH_KM = 0.75
RECENCY_HALF_LIFE_MONTHS = 12.0
SHRINKAGE_PSEUDOCOUNT = 20.0
MAX_ABS_LOG_DELTA = 0.035
LATITUDE_CENTER = 43.7
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20261073


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
    idx = int(np.searchsorted(np.cumsum(weights), threshold, side="left"))
    return float(values[min(idx, len(values) - 1)])


def recency_weights(dates: pd.Series, cutoff: pd.Timestamp) -> np.ndarray:
    seconds_per_month = 30.4375 * 24.0 * 60.0 * 60.0
    age = np.maximum(0.0, (cutoff - pd.to_datetime(dates, errors="raise")).dt.total_seconds().to_numpy(float)
                     / seconds_per_month)
    return np.exp(-np.log(2.0) * age / RECENCY_HALF_LIFE_MONTHS)


def project(latitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
    x = np.asarray(longitude, dtype=float) * 111.32 * np.cos(np.deg2rad(LATITUDE_CENTER))
    y = np.asarray(latitude, dtype=float) * 111.32
    return np.column_stack((x, y))


def reconstruct_one(history: pd.DataFrame, queries: pd.DataFrame,
                    cutoff: pd.Timestamp) -> pd.DataFrame:
    history = history.loc[pd.to_datetime(history.date_sold, errors="raise").lt(cutoff)].copy()
    if not history.empty and (pd.to_datetime(history.date_sold, errors="raise") >= cutoff).any():
        raise RuntimeError("auditor found same-month or future R153 history")
    history["residual"] = np.log(history.sold_price.to_numpy(float) / history.base_prediction.to_numpy(float))
    queries = queries.reset_index(drop=True)
    corrections = np.zeros(len(queries), dtype=float)
    records: list[dict | None] = [None] * len(queries)
    for property_type, query_indices in queries.groupby("house_type_name", sort=False).groups.items():
        positions = np.asarray(list(query_indices), dtype=int)
        typed = history.loc[history.house_type_name.eq(property_type)].copy()
        if len(typed) < MIN_TYPE_HISTORY:
            for pos in positions:
                records[pos] = {"neighbor_count": 0, "effective_neighbor_count": 0.0,
                                "type_center_log_residual": 0.0, "local_center_log_residual": 0.0,
                                "correction_log": 0.0, "fallback_reason": "insufficient_type_history"}
            continue
        all_type_weights = recency_weights(typed.date_sold, cutoff)
        type_residuals = typed.residual.to_numpy(float)
        type_center = weighted_median(type_residuals, all_type_weights)
        valid_history = (
            np.isfinite(typed.latitude.to_numpy(float)) & np.isfinite(typed.longitude.to_numpy(float))
            & typed.latitude.between(40.0, 50.0).to_numpy()
            & typed.longitude.between(-90.0, -60.0).to_numpy()
        )
        local = typed.loc[valid_history].reset_index(drop=True)
        if len(local) < MIN_NEIGHBORS:
            for pos in positions:
                records[pos] = {"neighbor_count": 0, "effective_neighbor_count": 0.0,
                                "type_center_log_residual": type_center,
                                "local_center_log_residual": type_center, "correction_log": 0.0,
                                "fallback_reason": "insufficient_localizable_history"}
            continue
        tree = cKDTree(project(local.latitude.to_numpy(float), local.longitude.to_numpy(float)))
        local_residual = local.residual.to_numpy(float)
        local_recency = recency_weights(local.date_sold, cutoff)
        qtype = queries.iloc[positions]
        qvalid = (
            np.isfinite(qtype.latitude.to_numpy(float)) & np.isfinite(qtype.longitude.to_numpy(float))
            & qtype.latitude.between(40.0, 50.0).to_numpy()
            & qtype.longitude.between(-90.0, -60.0).to_numpy()
        )
        if not qvalid.any():
            for pos in positions:
                records[pos] = {"neighbor_count": 0, "effective_neighbor_count": 0.0,
                                "type_center_log_residual": type_center,
                                "local_center_log_residual": type_center, "correction_log": 0.0,
                                "fallback_reason": "missing_or_invalid_query_coordinates"}
            continue
        coords = project(qtype.loc[qvalid, "latitude"].to_numpy(float), qtype.loc[qvalid, "longitude"].to_numpy(float))
        k = min(MAX_NEIGHBORS, len(local))
        distances, indices = tree.query(coords, k=k, distance_upper_bound=RADIUS_KM, workers=1)
        if k == 1:
            distances, indices = distances[:, None], indices[:, None]
        valid_positions = np.flatnonzero(qvalid)
        for local_row, original_pos in enumerate(positions):
            if not qvalid[local_row]:
                records[original_pos] = {"neighbor_count": 0, "effective_neighbor_count": 0.0,
                                         "type_center_log_residual": type_center,
                                         "local_center_log_residual": type_center, "correction_log": 0.0,
                                         "fallback_reason": "missing_or_invalid_query_coordinates"}
                continue
            lookup = int(np.searchsorted(valid_positions, local_row))
            dist, idx = distances[lookup], indices[lookup]
            keep = np.isfinite(dist) & (idx < len(local))
            n = int(keep.sum())
            eff, local_center, reason = 0.0, type_center, None
            if n >= MIN_NEIGHBORS:
                chosen, chosen_dist = idx[keep].astype(int), dist[keep]
                weights = local_recency[chosen] * np.exp(-0.5 * np.square(chosen_dist / DISTANCE_BANDWIDTH_KM))
                if not np.isfinite(weights).all() or weights.sum() <= 0:
                    reason = "invalid_neighbor_weights"
                else:
                    eff = float(weights.sum() ** 2 / np.square(weights).sum())
                    local_center = weighted_median(local_residual[chosen], weights)
                    raw = (local_center - type_center) * eff / (eff + SHRINKAGE_PSEUDOCOUNT)
                    corrections[original_pos] = float(np.clip(raw, -MAX_ABS_LOG_DELTA, MAX_ABS_LOG_DELTA))
            else:
                reason = "insufficient_spatial_neighbors"
            records[original_pos] = {"neighbor_count": n, "effective_neighbor_count": eff,
                                     "type_center_log_residual": float(type_center),
                                     "local_center_log_residual": float(local_center),
                                     "correction_log": float(corrections[original_pos]),
                                     "fallback_reason": reason}
    if any(record is None for record in records):
        raise RuntimeError("auditor did not reconstruct every R153 correction")
    return pd.DataFrame(records)


def score(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    ape = 100.0 * np.abs(np.asarray(prediction, float) - np.asarray(actual, float)) / np.asarray(actual, float)
    return {"n": int(len(ape)), "mdape_percent": float(np.median(ape)), "mape_percent": float(np.mean(ape))}


def score_paired(paired: pd.DataFrame, months: list[str], gates: dict) -> dict:
    actual = paired.sold_price.to_numpy(float)
    parent = paired.parent_prediction.to_numpy(float)
    candidate = paired.candidate_prediction.to_numpy(float)
    parent_ape = 100.0 * np.abs(parent - actual) / actual
    candidate_ape = 100.0 * np.abs(candidate - actual) / actual
    parent_metrics, candidate_metrics = score(actual, parent), score(actual, candidate)
    relative_gain = 100.0 * (parent_metrics["mdape_percent"] - candidate_metrics["mdape_percent"]) / parent_metrics["mdape_percent"]
    mape_change = 100.0 * (candidate_metrics["mape_percent"] - parent_metrics["mape_percent"]) / parent_metrics["mape_percent"]
    monthly, indices_by_month = {}, []
    for month in months:
        idx = np.flatnonzero(paired.month.astype(str).eq(month).to_numpy())
        indices_by_month.append(idx)
        p, c = score(actual[idx], parent[idx]), score(actual[idx], candidate[idx])
        gain = 100.0 * (p["mdape_percent"] - c["mdape_percent"]) / p["mdape_percent"]
        monthly[month] = {"n": int(len(idx)), "parent": p, "candidate": c,
                          "relative_mdape_improvement_percent": float(gain)}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    differences = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    for replicate in range(BOOTSTRAP_REPLICATES):
        sampled_parent, sampled_candidate = [], []
        for idx in indices_by_month:
            picked = idx[rng.integers(0, len(idx), size=len(idx))]
            sampled_parent.append(parent_ape[picked])
            sampled_candidate.append(candidate_ape[picked])
        differences[replicate] = float(np.median(np.concatenate(sampled_candidate))
                                       - np.median(np.concatenate(sampled_parent)))
    lower, upper = map(float, np.quantile(differences, [0.025, 0.975]))
    monthly_gains = [value["relative_mdape_improvement_percent"] for value in monthly.values()]
    improved = sum(value > 0.0 for value in monthly_gains)
    worst = max(0.0, -min(monthly_gains))
    minimum_gain = gates["screen_relative_mdape_improvement_percent"] if len(months) == 2 else gates["development_relative_mdape_improvement_percent"]
    minimum_months = len(months) if len(months) == 2 else gates["development_months_improved_min"]
    maximum_worst = 0.0 if len(months) == 2 else gates["development_worst_month_regression_max_percent"]
    maximum_mape = gates["screen_mape_regression_max_percent"] if len(months) == 2 else gates["development_mape_regression_max_percent"]
    qualifies = bool(relative_gain >= minimum_gain and improved >= minimum_months
                     and worst <= maximum_worst and mape_change <= maximum_mape and upper < 0.0)
    result = {"n": int(len(paired)), "months": months, "parent": parent_metrics,
              "candidate": candidate_metrics, "relative_mdape_improvement_percent": float(relative_gain),
              "mape_relative_change_percent": float(mape_change), "monthly": monthly,
              "months_improved": int(improved), "worst_month_regression_percent": float(worst),
              "bootstrap_mdape_difference_percent": {"lower95": lower, "upper95": upper},
              "bootstrap_replicates": BOOTSTRAP_REPLICATES, "qualifies_development": qualifies,
              "target_mdape_strictly_below": 4.0,
              "target_met": bool(candidate_metrics["mdape_percent"] < 4.0)}
    return result


def expected_months(protocol: dict, stage: str) -> tuple[str, ...]:
    targets = SCREEN_MONTHS if stage == "screen" else tuple(protocol["development_months"])
    last = max(targets)
    return tuple(sorted(set(SEED_MONTHS) | {m for m in protocol["development_months"] if m <= last}))


def load_safe_frame(protocol: dict, inputs: dict, stage: str) -> pd.DataFrame:
    months = expected_months(protocol, stage)
    if set(months) & set(protocol["reserved_months"]):
        raise RuntimeError("audit month filters include a reserved month")
    data_path = ROOT / protocol["data_path"]
    if sha256(data_path) != protocol["data_sha256"]:
        raise RuntimeError("audit source parquet hash differs from protocol")
    filters = []
    for month in months:
        start = pd.Timestamp(month + "-01")
        filters.append([("date_sold", ">=", start), ("date_sold", "<", start + pd.offsets.MonthBegin(1))])
    metadata = pd.read_parquet(data_path, columns=[
        "ml_num", "date_sold", "sold_price", "house_type_name", "latitude", "longitude",
    ], filters=filters)
    metadata["month"] = pd.to_datetime(metadata.date_sold, errors="raise").dt.to_period("M").astype(str)
    metadata["date_sold"] = pd.to_datetime(metadata.date_sold, errors="raise")
    metadata["ml_num"] = metadata.ml_num.astype(str)
    metadata["house_type_name"] = metadata.house_type_name.astype("string").fillna("__MISSING__").astype(str)
    for column in ("sold_price", "latitude", "longitude"):
        metadata[column] = pd.to_numeric(metadata[column], errors="coerce")
    if set(metadata.month.unique()) != set(months) or metadata.duplicated(["month", "ml_num"]).any():
        raise RuntimeError("audit exact safe metadata has unexpected months or duplicate keys")

    r35_path = ROOT / inputs["r35"]["predictions_path"]
    r35 = pd.read_csv(r35_path, dtype={"ml_num": str})
    r35 = r35.loc[r35.track.eq(TRACK) & r35.month.astype(str).isin(months),
                  ["month", "ml_num", "sold_price", "candidate_prediction"]].copy()
    r35["month"], r35["ml_num"] = r35.month.astype(str), r35.ml_num.astype(str)
    r35.rename(columns={"sold_price": "sold_price_r35", "candidate_prediction": "base_prediction"}, inplace=True)
    dev_meta = metadata.loc[metadata.month.isin(protocol["development_months"])].copy()
    dev = r35.merge(dev_meta, on=["month", "ml_num"], how="outer", validate="one_to_one", indicator=True)
    if not dev._merge.eq("both").all() or not np.array_equal(dev.sold_price.to_numpy(float), dev.sold_price_r35.to_numpy(float)):
        raise RuntimeError("audit found R35 predictions or labels misaligned with safe metadata")
    dev.drop(columns=["_merge", "sold_price_r35"], inplace=True)
    dev = dev[["month", "ml_num", "date_sold", "sold_price", "base_prediction",
               "house_type_name", "latitude", "longitude"]]

    seed = pd.read_csv(ROOT / inputs["r44"]["seed_predictions_path"], dtype={"ml_num": str})
    seed = seed.loc[seed.track.eq(TRACK), ["month", "ml_num", "date_sold", "sold_price", "variant_prediction"]].copy()
    seed["month"], seed["ml_num"] = seed.month.astype(str), seed.ml_num.astype(str)
    seed["date_sold"] = pd.to_datetime(seed.date_sold, errors="raise")
    seed.rename(columns={"sold_price": "sold_price_seed", "variant_prediction": "base_prediction",
                         "date_sold": "date_sold_seed"}, inplace=True)
    seed_meta = seed.merge(metadata, on=["month", "ml_num"], how="left", validate="one_to_one",
                           indicator=True, suffixes=("_seed", "_source"))
    if (not seed_meta._merge.eq("both").all()
            or not np.array_equal(seed_meta.date_sold_seed.to_numpy("datetime64[ns]"),
                                  seed_meta.date_sold.to_numpy("datetime64[ns]"))
            or not np.array_equal(seed_meta.sold_price_seed.to_numpy(float), seed_meta.sold_price.to_numpy(float))):
        raise RuntimeError("audit found R44 seed rows misaligned with safe source metadata")
    seed_frame = seed_meta[["month", "ml_num", "date_sold", "sold_price", "base_prediction",
                            "house_type_name", "latitude", "longitude"]].copy()
    frame = pd.concat([seed_frame, dev], ignore_index=True)
    if frame.duplicated(["month", "ml_num"]).any() or set(frame.month.unique()) != set(months):
        raise RuntimeError("audit combined seed and R35 frames have invalid coverage or duplicate keys")
    return frame.sort_values(["month", "ml_num"], kind="mergesort").reset_index(drop=True)


def verify_inputs(protocol: dict, progress: dict, card: dict) -> dict:
    r35_result_path = ROOT / progress["r35_development_result"]
    r35_prediction_path, r35_manifest_path = r35_result_path.parent / "paired_predictions.csv", r35_result_path.parent / "run_manifest.json"
    r44_root = ROOT / "outputs/bill_target74/bill_target74_unbounded_20261024/batches" / R44_RUN_ID
    r44_seed_path, r44_result_path = r44_root / "past_oof_predictions.csv", r44_root / "screen_results.json"
    r44_manifest_path, r44_pair_path = r44_root / "run_manifest.json", r44_root / "paired_predictions.csv"
    paths = {
        "r35": {"result_path": r35_result_path, "predictions_path": r35_prediction_path,
                "manifest_path": r35_manifest_path},
        "r44": {"seed_predictions_path": r44_seed_path, "result_path": r44_result_path,
                "manifest_path": r44_manifest_path, "paired_predictions_path": r44_pair_path},
    }
    for name, fields in paths.items():
        for field, path in fields.items():
            relative = str(path.relative_to(ROOT)).replace("\\", "/")
            expected = card["input_artifacts"][name]
            key_path = field
            if relative != expected[key_path] or not path.is_file():
                raise RuntimeError(f"R153 registered {name} input path changed or is missing: {relative}")
            hash_key = {"result_path": "result_sha256", "predictions_path": "predictions_sha256",
                        "manifest_path": "manifest_sha256", "seed_predictions_path": "seed_predictions_sha256",
                        "paired_predictions_path": "paired_predictions_sha256"}[field]
            if sha256(path) != expected[hash_key]:
                raise RuntimeError(f"R153 registered {name} input hash mismatch: {field}")
    r35 = json.loads(r35_result_path.read_text(encoding="utf-8"))
    m35 = json.loads(r35_manifest_path.read_text(encoding="utf-8"))
    r44 = json.loads(r44_result_path.read_text(encoding="utf-8"))
    m44 = json.loads(r44_manifest_path.read_text(encoding="utf-8"))
    if (r35.get("run_id") != R35_RUN_ID or r35.get("card") != R35_CARD
            or r35.get("data_sha256") != protocol["data_sha256"]
            or r35.get("reserved_months_labels_opened") is not False or r35.get("hidden_labels_opened") is not False
            or m35.get("campaign_id") != protocol["campaign_id"] or m35.get("run_id") != R35_RUN_ID
            or m35.get("result_sha256") != sha256(r35_result_path)
            or m35.get("paired_predictions_sha256") != sha256(r35_prediction_path)
            or m35.get("reserved_months_labels_opened") is not False or m35.get("hidden_labels_opened") is not False):
        raise RuntimeError("R153 R35 parent verification failed")
    if (r44.get("run_id") != R44_RUN_ID or r44.get("card") != R44_CARD
            or set(r44.get("seed_months", [])) != set(SEED_MONTHS)
            or r44.get("reserved_months_labels_opened") is not False or r44.get("hidden_labels_opened") is not False
            or m44.get("campaign_id") != protocol["campaign_id"] or m44.get("run_id") != R44_RUN_ID
            or m44.get("parent_run_id") != R35_SCREEN_RUN_ID
            or m44.get("seed_oof_sha256") != sha256(r44_seed_path)
            or m44.get("result_sha256") != sha256(r44_result_path)
            or m44.get("reserved_months_labels_opened") is not False or m44.get("hidden_labels_opened") is not False
            or set(m44.get("safe_oof_months_read", [])) & set(protocol["reserved_months"])
            or set(m44.get("safe_oof_months_read", [])) - (set(SEED_MONTHS) | set(protocol["development_months"]))):
        raise RuntimeError("R153 R44 seed verification failed")
    if (protocol["cards"][R35_CARD].get("status") != "fixed_origin_confirmation_passed"
            or protocol["cards"][R35_CARD].get("development_run_id") != R35_RUN_ID
            or protocol["cards"][R44_CARD].get("parent_run_id") != R35_SCREEN_RUN_ID
            or protocol["cards"][R44_CARD].get("variant_run_id") != R35_SCREEN_RUN_ID):
        raise RuntimeError("R153 source route is no longer the validated R35 list-aware route")
    return {"r35": r35, "r44": r44, "r35_manifest": m35, "r44_manifest": m44}


def main() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    progress = json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))
    card = protocol["cards"][CARD_KEY]
    if (protocol["budget_policy"] != "unbounded_until_objective_and_validation_complete"
            or any(value is not None for key, value in protocol["limits"].items() if key != "learner_threads")):
        raise RuntimeError("active campaign budget no longer matches the uncapped policy")
    if protocol.get("reserve_validation_eligible") is not False:
        raise RuntimeError("campaign reserve-validation eligibility changed unexpectedly")
    expected_code = card["code_sha256"]
    code_match = all(sha256(ROOT / path) == digest for path, digest in expected_code.items())
    if not code_match:
        raise RuntimeError("R153 registered source/auditor hashes do not match current files")
    verified = verify_inputs(protocol, progress, card)
    if card.get("input_artifacts") != {
        "r35": {key: value for key, value in card["input_artifacts"]["r35"].items()},
        "r44": {key: value for key, value in card["input_artifacts"]["r44"].items()},
    }:
        raise RuntimeError("R153 input artifact registration structure is invalid")

    stages = []
    for stage in ("screen", "development"):
        result_field = f"{stage}_result_path"
        if result_field in card:
            stages.append((stage, ROOT / card[result_field], card[f"{stage}_run_id"]))
    if not stages:
        raise RuntimeError("R153 has no completed stage to audit")

    ledger = [json.loads(line) for line in LEDGER_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    stage_audits = {}
    for stage, result_path, run_id in stages:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        batch = result_path.parent
        pair_path, manifest_path = batch / "paired_predictions.csv", batch / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        target_months = SCREEN_MONTHS if stage == "screen" else list(protocol["development_months"])
        expected_reads = list(expected_months(protocol, stage))
        if (result.get("run_id") != run_id or result.get("card") != CARD_KEY or result.get("stage") != stage
                or result.get("campaign_id") != protocol["campaign_id"]
                or result.get("months") != target_months
                or result.get("filtered_source_months") != expected_reads
                or result.get("input_artifacts") != card["input_artifacts"]
                or result.get("reserved_months_labels_opened") is not False
                or result.get("hidden_labels_opened") is not False
                or manifest.get("run_id") != run_id or manifest.get("card") != CARD_KEY
                or manifest.get("stage") != stage or manifest.get("data_sha256") != protocol["data_sha256"]
                or manifest.get("code_sha256") != expected_code
                or manifest.get("input_artifacts") != card["input_artifacts"]
                or manifest.get("filtered_source_months") != expected_reads
                or manifest.get("reserved_months_labels_opened") is not False
                or manifest.get("hidden_labels_opened") is not False
                or manifest.get("result_sha256") != sha256(result_path)
                or manifest.get("paired_predictions_sha256") != sha256(pair_path)):
            raise RuntimeError(f"R153 {stage} manifest/result integrity or access flags failed")
        if set(expected_reads) & set(protocol["reserved_months"]):
            raise RuntimeError("R153 stage includes a reserved month in its filter")

        frame = load_safe_frame(protocol, card["input_artifacts"], stage)
        saved = pd.read_csv(pair_path, dtype={"ml_num": str})
        if saved.duplicated(["month", "ml_num"]).any() or set(saved.month.astype(str).unique()) != set(target_months):
            raise RuntimeError(f"R153 {stage} predictions have duplicate keys or incorrect month coverage")
        rebuilt_parts = []
        for month in target_months:
            cutoff = pd.Timestamp(month + "-01")
            history = frame.loc[frame.month.lt(month)].copy()
            query = frame.loc[frame.month.eq(month)].copy().reset_index(drop=True)
            features = query[["house_type_name", "latitude", "longitude"]].copy()
            diagnostics = reconstruct_one(history, features, cutoff)
            rebuilt = query[["month", "ml_num", "sold_price", "base_prediction"]].copy()
            rebuilt["correction_log"] = diagnostics.correction_log.to_numpy(float)
            rebuilt["candidate_prediction"] = np.rint(
                rebuilt.base_prediction.to_numpy(float) * np.exp(rebuilt.correction_log.to_numpy(float))
            )
            rebuilt["neighbor_count"] = diagnostics.neighbor_count.to_numpy(int)
            rebuilt["effective_neighbor_count"] = diagnostics.effective_neighbor_count.to_numpy(float)
            rebuilt["type_center_log_residual"] = diagnostics.type_center_log_residual.to_numpy(float)
            rebuilt["local_center_log_residual"] = diagnostics.local_center_log_residual.to_numpy(float)
            rebuilt["fallback_reason"] = diagnostics.fallback_reason
            rebuilt_parts.append(rebuilt)
        rebuilt = pd.concat(rebuilt_parts, ignore_index=True)
        joined = rebuilt.merge(saved, on=["month", "ml_num"], how="outer", validate="one_to_one",
                               suffixes=("_audit", "_saved"), indicator=True)
        if not joined._merge.eq("both").all() or len(joined) != len(saved):
            raise RuntimeError(f"R153 {stage} saved rows do not match independently rebuilt safe OOF rows")
        if not np.array_equal(joined.sold_price_audit.to_numpy(float), joined.sold_price_saved.to_numpy(float)):
            raise RuntimeError(f"R153 {stage} labels differ from safe source metadata")
        if not np.array_equal(joined.base_prediction_audit.to_numpy(float), joined.parent_prediction.to_numpy(float)):
            raise RuntimeError(f"R153 {stage} comparison parent is not exact R35")
        if not np.array_equal(joined.base_prediction_audit.to_numpy(float), joined.base_prediction_saved.to_numpy(float)):
            raise RuntimeError(f"R153 {stage} saved base is not exact R35")
        if not np.array_equal(joined.candidate_prediction_audit.to_numpy(float), joined.candidate_prediction_saved.to_numpy(float)):
            raise RuntimeError(f"R153 {stage} saved corrected predictions differ from reconstruction")
        if not np.array_equal(joined.neighbor_count_audit.to_numpy(int), joined.neighbor_count_saved.to_numpy(int)):
            raise RuntimeError(f"R153 {stage} saved neighbor counts differ from reconstruction")
        tolerances = [
            float(np.max(np.abs(joined.correction_log_audit.to_numpy(float) - joined.correction_log_saved.to_numpy(float)))),
            float(np.max(np.abs(joined.effective_neighbor_count_audit.to_numpy(float) - joined.effective_neighbor_count_saved.to_numpy(float)))),
            float(np.max(np.abs(joined.type_center_log_residual_audit.to_numpy(float) - joined.type_center_log_residual_saved.to_numpy(float)))),
            float(np.max(np.abs(joined.local_center_log_residual_audit.to_numpy(float) - joined.local_center_log_residual_saved.to_numpy(float)))),
        ]
        if max(tolerances) > 1e-12 or not joined.fallback_reason_audit.fillna("").eq(joined.fallback_reason_saved.fillna("")).all():
            raise RuntimeError(f"R153 {stage} calibration diagnostics differ from independent reconstruction")

        score_months = list(target_months)
        scoring = joined.rename(columns={"sold_price_audit": "sold_price"})
        scoring["parent_prediction"] = scoring.base_prediction_audit
        scoring["candidate_prediction"] = scoring.candidate_prediction_audit
        metrics = score_paired(scoring, score_months, protocol["gates"])
        expected_scores = result["scores"]
        metric_keys = ("n", "parent", "candidate", "relative_mdape_improvement_percent",
                       "mape_relative_change_percent", "monthly", "months_improved",
                       "worst_month_regression_percent", "bootstrap_mdape_difference_percent",
                       "bootstrap_replicates")
        metric_delta = 0.0
        for key in metric_keys:
            if key not in expected_scores:
                raise RuntimeError(f"R153 {stage} score is missing {key}")
            if key in ("parent", "candidate", "monthly", "bootstrap_mdape_difference_percent"):
                # These structures contain only numeric values and month names.
                def flatten(value, prefix=""):
                    if isinstance(value, dict):
                        out = {}
                        for subkey, subvalue in value.items():
                            out.update(flatten(subvalue, f"{prefix}/{subkey}"))
                        return out
                    if isinstance(value, list):
                        return {prefix: value}
                    return {prefix: float(value)} if isinstance(value, (int, float, np.number)) else {prefix: value}
                left, right = flatten(metrics[key]), flatten(expected_scores[key])
                if left.keys() != right.keys():
                    raise RuntimeError(f"R153 {stage} score shape differs for {key}")
                for name in left:
                    if isinstance(left[name], (int, float)) and isinstance(right[name], (int, float)):
                        metric_delta = max(metric_delta, abs(float(left[name]) - float(right[name])))
                    elif left[name] != right[name]:
                        raise RuntimeError(f"R153 {stage} score values differ for {key}/{name}")
            elif isinstance(expected_scores[key], (int, float, np.number)):
                metric_delta = max(metric_delta, abs(float(metrics[key]) - float(expected_scores[key])))
            elif metrics[key] != expected_scores[key]:
                raise RuntimeError(f"R153 {stage} score values differ for {key}")
        if metric_delta > 1e-12:
            raise RuntimeError(f"R153 {stage} independent metrics differ by {metric_delta}")

        events = [event for event in ledger if event.get("card") == CARD_KEY and event.get("run_id") == run_id]
        starts = [event for event in events if event.get("event") == "fit_start"]
        completes = [event for event in events if event.get("event") == "fit_complete"]
        if len(starts) != len(target_months) or len(completes) != len(target_months):
            raise RuntimeError(f"R153 {stage} fit starts/completions do not reconcile")
        for event in starts:
            if event.get("reserved_months_labels_opened") is not False or event.get("hidden_labels_opened") is not False:
                raise RuntimeError("R153 fit start does not explicitly attest safe label access")
            if event.get("query_label_columns_passed") != [] or set(event.get("query_columns", [])) != {"house_type_name", "latitude", "longitude"}:
                raise RuntimeError("R153 fit start does not prove query labels were excluded")
            if any(history_month >= event["month"] for history_month in event.get("history_months", [])):
                raise RuntimeError("R153 fit ledger includes same-month or future history")
        stage_audits[stage] = {
            "run_id": run_id, "rows": int(len(joined)), "months": target_months,
            "filtered_source_months": expected_reads, "fit_starts": len(starts),
            "fit_completes": len(completes), "all_fit_query_labels_excluded": True,
            "strict_past_history_verified": True, "correction_max_abs_difference": tolerances[0],
            "effective_count_max_abs_difference": tolerances[1],
            "type_center_max_abs_difference": tolerances[2],
            "local_center_max_abs_difference": tolerances[3],
            "neighbor_counts_match_exactly": True, "candidate_predictions_match_exactly": True,
            "pooled_metric_max_difference": metric_delta,
            "pooled_parent": metrics["parent"], "pooled_candidate": metrics["candidate"],
            "monthly": metrics["monthly"], "screen_or_development_qualifies": result["qualifies"],
            "strict_listaware_target_below_4_percent_met": metrics["target_met"],
        }
        audit = {
            "card_key": CARD_KEY, "stage": stage, "run_id": run_id,
            "registered_code_hashes_match": code_match,
            "registered_input_hashes_match": True,
            "r35_validated_parent_identity": R35_RUN_ID,
            "r44_verified_seed_identity": R44_RUN_ID,
            "seed_months": list(SEED_MONTHS), "safe_month_filter_only": expected_reads,
            "reserved_or_hidden_label_access_by_r153": False,
            "campaign_had_prior_reserve_access_incident": bool(protocol.get("reserved_months_labels_opened")),
            "reserve_validation_eligible": False,
            "fit_starts_total_campaign": progress["fit_starts"],
            "campaign_budget_policy": protocol["budget_policy"], "campaign_limits": protocol["limits"],
            "audit": stage_audits[stage],
        }
        audit["audit_script_sha256"] = sha256(Path(__file__))
        audit_path = batch / "independent_audit.json"
        audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps({"audited_stages": stage_audits, "registered_code_hashes_match": code_match,
                      "reserve_validation_eligible": False}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
