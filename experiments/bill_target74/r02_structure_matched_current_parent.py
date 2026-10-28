"""R02 current-parent execution: safe-month strict-past structure-matched residual calibration."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.bill_target74 import calibration, r93_spatial_oof_calibration as common
from experiments.bill_target74 import runner as campaign


CARD = "R02_structure_matched_past_oof"
TRACKS = common.TRACKS
MONTHS = common.MONTHS
R93_RUN = "r93s_20261025T201838Z"
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20261031
WINDOW_MONTHS = 24
HALF_LIFE_MONTHS = 12.0
MAX_NEIGHBORS = 30
MIN_NEIGHBORS = 10
MAX_ABS_LOG_RESIDUAL = 0.10
SHRINKAGE_PSEUDOCOUNT = 30.0
CORRECTION_MULTIPLIER = 0.25


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return str(path.relative_to(campaign.ROOT)).replace("\\", "/")


def _r93_paths() -> tuple[Path, Path, Path]:
    result = campaign.CAMPAIGN_ROOT / "batches" / R93_RUN / "development_results.json"
    return result, result.parent / "paired_predictions.csv", result.parent / "run_manifest.json"


def _verify_r93() -> dict:
    result_path, pairs_path, manifest_path = _r93_paths()
    if not all(p.is_file() for p in (result_path, pairs_path, manifest_path)):
        raise FileNotFoundError("R02 current-parent execution requires complete saved R93 artifacts")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (result.get("data_sha256") != campaign.PROTOCOL["data_sha256"]
            or result.get("reserved_months_labels_opened") is not False
            or result.get("hidden_labels_opened") is not False):
        raise RuntimeError("R93 identity, data hash or label-access flags are invalid")
    if manifest.get("paired_predictions_sha256") != _sha256(pairs_path) or manifest.get("result_sha256") != _sha256(result_path):
        raise RuntimeError("R93 artifact hashes differ from its manifest")
    return {
        "run_id": result["run_id"], "result_path": _relative(result_path),
        "result_sha256": _sha256(result_path), "predictions_path": _relative(pairs_path),
        "predictions_sha256": _sha256(pairs_path), "manifest_path": _relative(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
    }


def register_card() -> dict:
    from experiments.bill_target74 import r04_ablation as identity

    card = campaign.PROTOCOL["cards"].get(CARD)
    if not card:
        raise RuntimeError("R02 original preregistration is missing")
    source_hash = _sha256(Path(__file__))
    existing = card.get("safe_current_execution")
    if existing:
        if card.get("status") != "registered":
            raise RuntimeError(f"R02 current execution already has status {card.get('status')!r}")
        if existing.get("source_sha256") == source_hash and card.get("data_sha256"):
            return card
        ledger = [json.loads(line) for line in campaign.LEDGER.read_text(encoding="utf-8").splitlines()]
        started = {row.get("fit_id") for row in ledger
                   if row.get("card") == CARD and row.get("event") == "fit_start"}
        closed = {row.get("fit_id") for row in ledger
                  if row.get("card") == CARD and row.get("event") in {"fit_complete", "fit_attempt_reconciled"}}
        completed_card = any(row.get("card") == CARD and row.get("event") == "card_complete" for row in ledger)
        if completed_card or started - closed:
            raise RuntimeError("cannot amend the R02 implementation with a completed or unresolved calibration fit")
        source = identity._source_identity()
        old_hash = existing.get("source_sha256")
        existing["source_sha256"] = source_hash
        existing["data_sha256"] = source["data_sha256"]
        card["data_sha256"] = source["data_sha256"]
        card["parent_commit"] = source["commit"]
        card.setdefault("registration_amendments", []).append({
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "reason": "Correct result-row parent column naming and persist each completed fold before final scoring; the structural-neighbor method and parameters are unchanged.",
            "previous_current_execution_source_sha256": old_hash,
            "current_execution_source_sha256": source_hash,
            "fit_starts": 0, "fit_starts_in_prior_failed_attempts": len(started),
        })
        campaign.PROTOCOL_PATH.write_text(
            json.dumps(campaign.PROTOCOL, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        campaign._append({
            "event": "card_registration_amended", "card": CARD, "fit_starts": 0,
            "fit_starts_in_prior_failed_attempts": len(started),
            "old_source_sha256": old_hash, "new_source_sha256": source_hash,
            "reason": "fix parent prediction result column naming and add durable per-fold prediction checkpoints; structural method unchanged",
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        return card
    if (card.get("half_life_months") != HALF_LIFE_MONTHS
            or card.get("max_neighbors") != MAX_NEIGHBORS
            or card.get("min_neighbors") != MIN_NEIGHBORS
            or card.get("window_months") != WINDOW_MONTHS):
        raise RuntimeError("R02 registered structural-neighbor parameters differ from implementation")
    r93 = _verify_r93()
    source = identity._source_identity()
    execution = {
        "status": "registered", "source_sha256": source_hash,
        "parents": {"off_market": "R52 experimental OOF parent", "list_aware": "R35 validated OOF parent"},
        "r93_oof_source": r93, "development_months": list(MONTHS),
        "query_source_policy": "read only exact eight safe development-month parquet filters; do not materialize full listings parquet",
        "history_source_policy": "use strictly earlier saved R52/R35 OOF predictions in the safe development window only; no 2022 seed from a different parent",
        "method": (
            "Within the same municipality and house type, compute structural distance from log(square feet), bedrooms, "
            "and washrooms using historical OOF-group scaling and missingness. Among up to 30 nearest past OOF homes "
            "from the previous 24 months, weight log(sold_price/base_prediction) by 1/(distance+0.25) and a 12-month "
            "half-life. If at least 10 neighbors exist, clip the weighted residual to +/-0.10, shrink by "
            "0.25*n_eff/(n_eff+30), multiply the base prediction by exp(correction), and round to nearest dollar. "
            "The first development month has no same-parent OOF history and remains unchanged."
        ),
        "parameters": {
            "window_months": WINDOW_MONTHS, "half_life_months": HALF_LIFE_MONTHS,
            "max_neighbors": MAX_NEIGHBORS, "min_neighbors": MIN_NEIGHBORS,
            "max_abs_log_residual": MAX_ABS_LOG_RESIDUAL,
            "shrinkage_pseudocount": SHRINKAGE_PSEUDOCOUNT,
            "correction_multiplier": CORRECTION_MULTIPLIER,
            "distance_features": ["log(square_feet)", "bedrooms", "washrooms"],
            "group_columns": ["municipality", "house_type_name"],
        },
        "data_sha256": source["data_sha256"], "parent_commit": source["commit"],
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    }
    old_source = card.get("source_sha256")
    card["safe_current_execution"] = execution
    card["data_sha256"] = source["data_sha256"]
    card["parent_commit"] = source["commit"]
    card.setdefault("registration_amendments", []).append({
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "reason": (
            "Activate the already registered structural-neighbor calibration on current R52/R35 OOF parents, "
            "using only the eight exact safe development-month filters. The legacy R02 runner reads the full parquet "
            "and uses G19/G01, so it is not invoked. Structural distance, recency, neighbor count, shrinkage and "
            "correction parameters remain as preregistered."
        ),
        "previous_source_sha256": old_source, "current_execution_source_sha256": source_hash,
        "fit_starts": 0, "method_parameters_unchanged": True,
    })
    campaign.PROTOCOL_PATH.write_text(
        json.dumps(campaign.PROTOCOL, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    campaign._append({
        "event": "card_registration_amended", "card": CARD, "fit_starts": 0,
        "old_source_sha256": old_source, "new_source_sha256": source_hash,
        "reason": execution["query_source_policy"] + "; rebase onto current OOF parents; preserve registered structural method",
        "method_parameters_unchanged": True, "reserved_months_labels_opened": False,
        "hidden_labels_opened": False,
    })
    campaign._append({
        "event": "card_current_execution_registered", "card": CARD, "source_sha256": source_hash,
        "expected_calibration_fit_starts": (len(MONTHS) - 1) * len(TRACKS),
        "r93_predictions_sha256": r93["predictions_sha256"],
        "data_sha256": source["data_sha256"], "reserved_months_labels_opened": False,
        "hidden_labels_opened": False,
    })
    campaign._write_progress(
        "r02_current_parent_registered", r02_current_source_sha256=source_hash,
        next_action="run safe R02 structural-neighbor calibration on current R52/R35 OOF rows",
    )
    return card


def _load_safe_metadata() -> pd.DataFrame:
    if _sha256(campaign.DATA) != campaign.PROTOCOL["data_sha256"]:
        raise RuntimeError("cached listings data hash differs from protocol")
    filters = []
    for month in MONTHS:
        start = pd.Timestamp(month + "-01")
        filters.append([("date_sold", ">=", start), ("date_sold", "<", start + pd.offsets.MonthBegin(1))])
    metadata = pd.read_parquet(
        campaign.DATA,
        columns=["ml_num", "date_sold", "sold_price", "municipality", "house_type_name",
                 "house_area", "bedrooms", "washrooms"],
        filters=filters,
    )
    metadata["date_sold"] = pd.to_datetime(metadata.date_sold, errors="raise")
    metadata["month"] = metadata.date_sold.dt.to_period("M").astype(str)
    observed = sorted(metadata.month.unique().tolist())
    if observed != sorted(MONTHS) or set(observed).intersection(campaign.PROTOCOL["reserved_months"]):
        raise RuntimeError(f"R02 exact development filter returned unexpected months: {observed}")
    metadata["ml_num"] = metadata.ml_num.astype(str)
    if metadata.ml_num.duplicated().any():
        raise RuntimeError("R02 source metadata contains duplicate listing IDs")
    metadata["source_sold_price"] = pd.to_numeric(metadata.sold_price, errors="raise")
    metadata = metadata.drop(columns="sold_price")
    for column in ("municipality", "house_type_name"):
        metadata[column] = metadata[column].astype("string").fillna("__MISSING__").astype(str)
    return metadata


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
            "n": len(group), "base": before, "candidate": after,
            "relative_mdape_improvement_percent": 100.0 * (before["mdape_percent"] - after["mdape_percent"]) / before["mdape_percent"],
        }
    gain = 100.0 * (base_metrics["mdape_percent"] - candidate_metrics["mdape_percent"]) / base_metrics["mdape_percent"]
    mape_change = 100.0 * (candidate_metrics["mape_percent"] - base_metrics["mape_percent"]) / base_metrics["mape_percent"]
    monthly_gains = [x["relative_mdape_improvement_percent"] for x in monthly.values()]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    base_ape = 100.0 * np.abs(base - actual) / actual
    candidate_ape = 100.0 * np.abs(candidate - actual) / actual
    differences = np.empty(BOOTSTRAP_REPLICATES)
    for i in range(BOOTSTRAP_REPLICATES):
        idx = rng.integers(0, len(actual), size=len(actual))
        differences[i] = np.median(candidate_ape[idx]) - np.median(base_ape[idx])
    bootstrap = {"lower95": float(np.quantile(differences, .025)),
                 "upper95": float(np.quantile(differences, .975)),
                 "replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED}
    gates = campaign.PROTOCOL["gates"]
    target = 7.0 if frame.track.iloc[0] == "off_market" else 4.0
    checks = {
        "relative_mdape_improvement": gain >= gates["development_relative_mdape_improvement_percent"],
        "months_improved": sum(x > 0 for x in monthly_gains) >= gates["development_months_improved_min"],
        "worst_month_regression": max([0.0] + [-x for x in monthly_gains]) <= gates["development_worst_month_regression_max_percent"],
        "mape_regression": mape_change <= gates["development_mape_regression_max_percent"],
        "bootstrap_upper_below_zero": bootstrap["upper95"] < 0.0,
        "strict_target": candidate_metrics["mdape_percent"] < target,
    }
    return {
        "base": base_metrics, "candidate": candidate_metrics,
        "relative_mdape_improvement_percent": float(gain), "mape_relative_change_percent": float(mape_change),
        "months_improved": int(sum(x > 0 for x in monthly_gains)),
        "worst_month_regression_percent": float(max([0.0] + [-x for x in monthly_gains])),
        "paired_bootstrap_mdape_difference_pp": bootstrap, "monthly": monthly,
        "neighbor_coverage_percent": float(100 * frame.covered.mean()),
        "median_neighbors": float(frame.neighbor_count.median()),
        "gate_checks": checks, "qualifies_development": all(checks.values()),
        "target_mdape_strictly_below": target,
    }


def run() -> dict:
    card = campaign.PROTOCOL["cards"].get(CARD, {})
    execution = card.get("safe_current_execution")
    if not execution or execution.get("status") != "registered":
        raise RuntimeError("R02 safe current-parent execution must be registered before running")
    if execution.get("source_sha256") != _sha256(Path(__file__)):
        raise RuntimeError("R02 source hash changed after registration")
    if _sha256(campaign.DATA) != card["data_sha256"]:
        raise RuntimeError("R02 data hash differs from registration")
    r93_info = _verify_r93()
    rows = pd.read_csv(campaign.ROOT / r93_info["predictions_path"], dtype={"ml_num": str})
    metadata = _load_safe_metadata().rename(columns={"house_type_name": "source_house_type_name"})
    rows = rows.merge(metadata, on=["month", "ml_num"], how="left", validate="many_to_one")
    if rows.date_sold.isna().any() or rows.source_sold_price.isna().any():
        raise RuntimeError("R02 safe metadata did not cover every R93 prediction row")
    if not np.array_equal(rows.sold_price.to_numpy(float), rows.source_sold_price.to_numpy(float)):
        raise RuntimeError("R02 OOF labels do not match exact-month-filtered source labels")
    source_type = rows.source_house_type_name.astype("string").fillna("__MISSING__").astype(str).to_numpy()
    r93_type = rows.house_type_name.astype("string").fillna("__MISSING__").astype(str).to_numpy()
    if not np.array_equal(source_type, r93_type):
        raise RuntimeError("R02 OOF house types differ from safe source metadata")
    rows["house_type_name"] = r93_type
    rows["month"] = rows.month.astype(str)
    rows["track"] = rows.track.astype(str)
    rows["parent_prediction"] = rows.base_prediction.to_numpy(float)
    run_id = "r02s_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = campaign.CAMPAIGN_ROOT / "batches" / run_id
    output.mkdir(parents=True, exist_ok=False)
    provenance = output / "provenance"
    provenance.mkdir()
    fold_dir = output / "fold_predictions"
    fold_dir.mkdir()
    shutil.copy2(campaign.PROTOCOL_PATH, provenance / "registered_campaign_protocol.json")
    result_path, _, manifest_path = _r93_paths()
    shutil.copy2(result_path, provenance / "r93_development_results.json")
    shutil.copy2(manifest_path, provenance / "r93_run_manifest.json")
    campaign._append({
        "event": "card_start", "card": CARD, "run_id": run_id, "tracks": list(TRACKS),
        "development_months": list(MONTHS), "expected_calibration_fit_starts": (len(MONTHS) - 1) * len(TRACKS),
        "r93_predictions_sha256": r93_info["predictions_sha256"], "data_sha256": card["data_sha256"],
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    fit_details, output_frames, fold_prediction_hashes = [], [], {}
    try:
        for track in TRACKS:
            track_rows = rows.loc[rows.track.eq(track)].copy()
            for month_index, month in enumerate(MONTHS):
                current = track_rows.loc[track_rows.month.eq(month)].copy().reset_index(drop=True)
                cutoff = pd.Timestamp(month + "-01")
                history = track_rows.loc[pd.to_datetime(track_rows.date_sold).lt(cutoff)].copy()
                if not history.empty and (pd.to_datetime(history.date_sold) >= cutoff).any():
                    raise RuntimeError(f"R02 history crossed strict cutoff for {track}/{month}")
                query = current[["house_type_name", "municipality", "house_area", "bedrooms", "washrooms", "parent_prediction"]].copy()
                if {"sold_price", "date_sold"}.intersection(query.columns):
                    raise RuntimeError("R02 query accidentally contains outcomes")
                fit_id = None
                if month_index > 0:
                    fit_id = f"{campaign.PROTOCOL['campaign_id']}:{CARD}:{track}:{month}:{run_id}"
                    campaign._append({
                        "event": "fit_start", "card": CARD, "run_id": run_id, "fit_id": fit_id,
                        "stage": "strict_past_structure_neighbor_calibration", "track": track, "month": month,
                        "training_oof_rows": len(history), "training_oof_months": sorted(history.month.unique().tolist()),
                        "history_cutoff_exclusive": cutoff.isoformat(), "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "data_sha256": card["data_sha256"], "reserved_months_labels_opened": False,
                        "hidden_labels_opened": False,
                    })
                try:
                    features = calibration.build_oof_neighbor_features(
                        history[["date_sold", "house_type_name", "municipality", "house_area", "bedrooms", "washrooms", "sold_price", "parent_prediction"]],
                        query, cutoff, window_months=WINDOW_MONTHS, half_life_months=HALF_LIFE_MONTHS,
                        max_neighbors=MAX_NEIGHBORS, min_neighbors=MIN_NEIGHBORS,
                    )
                    candidate = np.rint(calibration.apply_oof_correction(current.parent_prediction.to_numpy(float), features))
                    if not np.isfinite(candidate).all() or (candidate <= 0.0).any():
                        raise RuntimeError(f"R02 generated invalid predictions for {track}/{month}")
                except Exception as exc:
                    if fit_id:
                        campaign._append({
                            "event": "fit_complete", "card": CARD, "run_id": run_id, "fit_id": fit_id,
                            "stage": "strict_past_structure_neighbor_calibration", "track": track,
                            "month": month, "status": "failed", "error": f"{type(exc).__name__}: {exc}",
                            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
                        })
                    raise
                if fit_id:
                    detail = {
                        "fit_id": fit_id, "track": track, "month": month, "status": "complete",
                        "training_oof_rows": len(history), "training_oof_months": sorted(history.month.unique().tolist()),
                        "covered_rows": int(features.covered.sum()), "coverage_percent": float(100 * features.covered.mean()),
                        "median_neighbors": float(features.neighbor_count.median()),
                        "p90_neighbors": float(features.neighbor_count.quantile(.9)),
                        "median_effective_neighbors": float(features.effective_n.median()),
                    }
                    fit_details.append(detail)
                    campaign._append({
                        "event": "fit_complete", "card": CARD, "run_id": run_id,
                        "stage": "strict_past_structure_neighbor_calibration", **detail,
                        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
                    })
                current["candidate_prediction"] = candidate
                current["covered"] = features.covered.to_numpy(bool)
                current["neighbor_count"] = features.neighbor_count.to_numpy(int)
                current["effective_n"] = features.effective_n.to_numpy(float)
                current["weighted_log_residual"] = features.weighted_log_residual.to_numpy(float)
                current["correction_log"] = features.correction_log.to_numpy(float)
                current["base_prediction"] = current.parent_prediction.to_numpy(float)
                fold_frame = current[[
                    "month", "track", "ml_num", "sold_price", "base_prediction", "candidate_prediction",
                    "house_type_name", "municipality", "house_area", "bedrooms", "washrooms", "covered",
                    "neighbor_count", "effective_n", "weighted_log_residual", "correction_log", "date_sold",
                ]].copy()
                fold_path = fold_dir / f"{track}_{month}.csv"
                fold_frame.to_csv(fold_path, index=False)
                fold_prediction_hashes[f"{track}/{month}"] = {
                    "path": _relative(fold_path), "sha256": _sha256(fold_path), "rows": int(len(fold_frame)),
                }
                campaign._append({
                    "event": "fold_checkpoint_saved", "card": CARD, "run_id": run_id,
                    "track": track, "month": month, "rows": int(len(fold_frame)),
                    "path": _relative(fold_path), "sha256": fold_prediction_hashes[f"{track}/{month}"]["sha256"],
                    "reserved_months_labels_opened": False, "hidden_labels_opened": False,
                })
                output_frames.append(fold_frame)
        paired = pd.concat(output_frames, ignore_index=True)
        if len(paired) != 31284 or paired.duplicated(["month", "track", "ml_num"]).any():
            raise RuntimeError("R02 row coverage or listing-key uniqueness is invalid")
        scores = {track: _score_track(paired.loc[paired.track.eq(track)].copy()) for track in TRACKS}
        result = {
            "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id, "card": CARD,
            "stage": "strict_past_walk_forward_development", "development_months": list(MONTHS),
            "fit_starts": len(fit_details), "tracks": scores, "fit_details": fit_details,
            "parent_by_track": execution["parents"], "r93_source": r93_info,
            "fold_prediction_hashes": fold_prediction_hashes,
            "data_sha256": card["data_sha256"], "reserved_months_labels_opened": False,
            "reserve_validation_eligible": False, "hidden_labels_opened": False,
        }
        pairs_path = output / "paired_predictions.csv"
        paired.to_csv(pairs_path, index=False)
        results_path = output / "development_results.json"
        results_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        source_paths = [Path(__file__), Path(calibration.__file__), Path(common.__file__),
                        campaign.PROTOCOL_PATH, campaign.PACKAGE / "runner.py"]
        manifest = {
            "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
            "source_identity": {"data_sha256": card["data_sha256"], "parent_commit": execution["parents"]},
            "source_sha256": {_relative(p): _sha256(p) for p in source_paths if p.is_file()},
            "r93_source": r93_info, "fit_starts": len(fit_details), "fit_details": fit_details,
            "fold_prediction_hashes": fold_prediction_hashes,
            "paired_predictions_sha256": _sha256(pairs_path), "result_sha256": _sha256(results_path),
            "reserved_months_labels_opened": False, "reserve_validation_eligible": False,
            "hidden_labels_opened": False,
        }
        manifest_path = output / "run_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        report_lines = [
            "# R02：严格过去结构相似房源残差校正", "",
            f"运行：`{run_id}`；以 R52 off-market / R35 list-aware OOF 为基础，评估八个安全开发月。",
            "同市镇、同房型内按 log(square feet)、卧室数和卫生间数做结构匹配；只用截止日前 24 个月的 OOF 残差。",
            "校正参数固定为预登记值，不使用当前月标签选邻居、权重或参数。", "",
            "| Track | Base MdAPE | Candidate MdAPE | 相对改善 | MAPE变化 | 月份改善 | 覆盖率 | bootstrap upper95 (pp) | <目标 | 晋级 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
        ]
        for track, score in scores.items():
            report_lines.append(
                f"| {track} | {score['base']['mdape_percent']:.5f}% | {score['candidate']['mdape_percent']:.5f}% | "
                f"{score['relative_mdape_improvement_percent']:+.3f}% | {score['mape_relative_change_percent']:+.3f}% | "
                f"{score['months_improved']}/8 | {score['neighbor_coverage_percent']:.1f}% | "
                f"{score['paired_bootstrap_mdape_difference_pp']['upper95']:+.5f} | "
                f"{'是' if score['candidate']['mdape_percent'] < score['target_mdape_strictly_below'] else '否'} | "
                f"{'是' if score['qualifies_development'] else '否'} |"
            )
        report_lines += ["", "## 逐月 MdAPE", "", "| Track | Month | Base | Candidate | Relative improvement | Coverage |", "|---|---|---:|---:|---:|---:|"]
        for track, score in scores.items():
            for month, detail in score["monthly"].items():
                coverage = paired.loc[paired.track.eq(track) & paired.month.eq(month), "covered"].mean() * 100
                report_lines.append(
                    f"| {track} | {month} | {detail['base']['mdape_percent']:.5f}% | "
                    f"{detail['candidate']['mdape_percent']:.5f}% | "
                    f"{detail['relative_mdape_improvement_percent']:+.3f}% | {coverage:.1f}% |"
                )
        report_lines += [
            "", "## 限制", "",
            "输入与评分只使用登记的八个安全开发月。查询特征不含 sale date 或 sold price；历史 OOF 残差必须早于每个预测月。",
            "2025-05/08 保留标签和隐藏标签未读取，且 2025-05/08 不再具备独立保留集验证资格。未通过的候选不替换 R22/R35。", "",
        ]
        report_path = output / "research_report.md"
        report_path.write_text("\n".join(report_lines), encoding="utf-8")
        passed = {track: score["qualifies_development"] for track, score in scores.items()}
        campaign._append({
            "event": "card_complete", "card": CARD, "run_id": run_id, "fit_starts": len(fit_details),
            "qualifies_by_track": passed, "result_path": _relative(results_path),
            "paired_predictions_sha256": manifest["paired_predictions_sha256"],
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        campaign._write_progress(
            "r02_current_parent_complete", r02_current_result=_relative(results_path), r02_current_run_id=run_id,
            r02_current_fit_starts=len(fit_details),
            next_action=("run fixed-origin confirmation for: " + ", ".join(t for t, ok in passed.items() if ok)
                         if any(passed.values()) else "R02 structure-matched calibration did not pass development; register a distinct mechanism"),
        )
        return result
    except KeyboardInterrupt as exc:
        ledger_rows = [json.loads(line) for line in campaign.LEDGER.read_text(encoding="utf-8").splitlines()]
        started_count = sum(x.get("event") == "fit_start" and x.get("run_id") == run_id for x in ledger_rows)
        checkpoint = {
            "run_id": run_id, "card": CARD, "fit_starts": len(fit_details),
            "completed_calibrations": fit_details,
            "error": f"KeyboardInterrupt: {exc or 'operator interrupted the run'}",
            "fold_prediction_hashes": fold_prediction_hashes,
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        }
        failure_path = output / "failure_checkpoint.json"
        failure_path.write_text(json.dumps(checkpoint, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        campaign._append({
            "event": "card_interrupted", "card": CARD, "run_id": run_id,
            "fit_starts": started_count,
            "failure_path": _relative(failure_path), "error": checkpoint["error"],
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        campaign._write_progress(
            "r02_current_parent_interrupted", r02_current_failure=_relative(failure_path),
            next_action="audit interrupted R02 checkpoint before retry", latest_error=checkpoint["error"],
        )
        raise
    except Exception as exc:
        checkpoint = {
            "run_id": run_id, "card": CARD, "fit_starts": len(fit_details),
            "completed_calibrations": fit_details, "error": f"{type(exc).__name__}: {exc}",
            "fold_prediction_hashes": fold_prediction_hashes,
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        }
        (output / "failure_checkpoint.json").write_text(json.dumps(checkpoint, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        campaign._append({
            "event": "card_failed", "card": CARD, "run_id": run_id, "fit_starts": len(fit_details),
            "error": checkpoint["error"], "reserved_months_labels_opened": False,
            "hidden_labels_opened": False,
        })
        campaign._write_progress(
            "r02_current_parent_failed", r02_current_failure=_relative(output / "failure_checkpoint.json"),
            next_action="audit safe R02 failure checkpoint and ledger before any recovery", latest_error=checkpoint["error"],
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register", action="store_true")
    args = parser.parse_args()
    if args.register:
        card = register_card()
        print(json.dumps({"card": CARD, "status": card["safe_current_execution"]["status"],
                          "expected_calibration_fit_starts": (len(MONTHS) - 1) * len(TRACKS)}, ensure_ascii=False), flush=True)
        return
    result = run()
    for track, score in result["tracks"].items():
        print(
            f"{track}: base={score['base']['mdape_percent']:.5f}% "
            f"candidate={score['candidate']['mdape_percent']:.5f}% "
            f"gain={score['relative_mdape_improvement_percent']:+.3f}% "
            f"MAPE_change={score['mape_relative_change_percent']:+.3f}% "
            f"months={score['months_improved']}/8 coverage={score['neighbor_coverage_percent']:.1f}% "
            f"bootstrap_upper95={score['paired_bootstrap_mdape_difference_pp']['upper95']:+.5f} "
            f"qualifies={score['qualifies_development']}", flush=True,
        )
    print(f"report={campaign.CAMPAIGN_ROOT / 'batches' / result['run_id'] / 'research_report.md'}", flush=True)


if __name__ == "__main__":
    main()
