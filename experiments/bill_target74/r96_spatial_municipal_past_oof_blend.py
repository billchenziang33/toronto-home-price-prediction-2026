"""R96: strict-past per-house-type blend of spatial and municipality residual corrections."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.bill_target74 import r93_spatial_oof_calibration as common
from experiments.bill_target74 import runner as campaign


CARD = "R96_past_oof_spatial_municipality_correction_blend"
TRACKS = common.TRACKS
MONTHS = common.MONTHS
MIN_GROUP_HISTORY_ROWS = 500
MIN_GROUP_HISTORY_MONTHS = 2
ALPHA_GRID = tuple(round(float(x), 1) for x in np.linspace(0.0, 1.0, 11))
MAX_ABS_COMBINED_LOG_CORRECTION = 0.05
MAPE_GUARD_PERCENT = 1.0
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20261030
R94_RUN = "r94g_20261025T202220Z"
R95_RUN = "r95m_20261025T202752Z"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return str(path.relative_to(campaign.ROOT)).replace("\\", "/")


def _run_paths(run_id: str) -> tuple[Path, Path, Path]:
    result = campaign.CAMPAIGN_ROOT / "batches" / run_id / "development_results.json"
    return result, result.parent / "paired_predictions.csv", result.parent / "run_manifest.json"


def _verify_run(run_id: str, expected_card: str) -> dict:
    result_path, pairs_path, manifest_path = _run_paths(run_id)
    if not all(p.is_file() for p in (result_path, pairs_path, manifest_path)):
        raise FileNotFoundError(f"R96 requires complete saved artifacts for {run_id}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (result.get("card") != expected_card
            or result.get("data_sha256") != campaign.PROTOCOL["data_sha256"]
            or result.get("reserved_months_labels_opened") is not False
            or result.get("hidden_labels_opened") is not False):
        raise RuntimeError(f"R96 input identity, data hash or label-access flags are invalid: {run_id}")
    if (manifest.get("paired_predictions_sha256") != _sha256(pairs_path)
            or manifest.get("result_sha256") != _sha256(result_path)):
        raise RuntimeError(f"R96 input artifact hashes differ from manifest: {run_id}")
    return {
        "run_id": run_id, "result_path": _relative(result_path),
        "result_sha256": _sha256(result_path), "predictions_path": _relative(pairs_path),
        "predictions_sha256": _sha256(pairs_path), "manifest_path": _relative(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
    }


def register_card() -> dict:
    from experiments.bill_target74 import r04_ablation as identity

    existing = campaign.PROTOCOL["cards"].get(CARD)
    current_hash = _sha256(Path(__file__))
    if existing:
        if existing.get("status") == "registered" and existing.get("source_sha256") == current_hash:
            return existing
        raise RuntimeError(f"R96 already has status {existing.get('status')!r}")
    r94_source = _verify_run(R94_RUN, "R94_past_oof_house_type_spatial_strength")
    r95_source = _verify_run(R95_RUN, "R95_municipality_type_past_oof_residual_difference")
    source = identity._source_identity()
    registration = {
        "status": "registered", "tracks": list(TRACKS), "development_months": list(MONTHS),
        "expected_calibration_fit_starts": (len(MONTHS) - 1) * len(TRACKS),
        "parents": {"off_market": "R52 experimental OOF parent", "list_aware": "R35 validated OOF parent"},
        "sources": {"R94_spatial": r94_source, "R95_municipality": r95_source},
        "method": (
            "For each outer development month and track, blend the already strict-past R94 spatial log correction "
            "with the R95 municipality x house-type log correction. For each query house type, select municipality "
            "weight alpha from 0.0..1.0 by 0.1 using only earlier OOF rows; require at least 500 past rows across "
            "two earlier months. Minimize past MdAPE subject to no more than 1% MAPE regression versus R94 spatial "
            "correction alone; ties favor lower MAPE then smaller alpha. Apply spatial_log + alpha*municipality_log, "
            "clip the combined correction to +/-0.05, multiply the original base prediction by exp(correction), "
            "then round to the nearest dollar. Unsupported groups use alpha=0."
        ),
        "difference_from_prior_work": (
            "R94 tunes the spatial-correction strength by house type and R95 independently applies a municipality "
            "residual difference. R96 tests whether the municipality signal adds value on top of the R94 spatial "
            "candidate using strict-past OOF selection; it does not tune on the target month or refit base learners."
        ),
        "parameters": {
            "minimum_group_history_rows": MIN_GROUP_HISTORY_ROWS,
            "minimum_group_history_months": MIN_GROUP_HISTORY_MONTHS,
            "alpha_grid": list(ALPHA_GRID),
            "maximum_abs_combined_log_correction": MAX_ABS_COMBINED_LOG_CORRECTION,
            "past_mape_guard_percent": MAPE_GUARD_PERCENT,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "development_gate": dict(campaign.PROTOCOL["gates"]),
        "data_sha256": source["data_sha256"], "parent_commit": source["commit"],
        "source_sha256": current_hash, "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    }
    campaign.PROTOCOL["cards"][CARD] = registration
    campaign.PROTOCOL_PATH.write_text(
        json.dumps(campaign.PROTOCOL, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    campaign._append({
        "event": "card_registered", "card": CARD, "tracks": list(TRACKS),
        "development_months": list(MONTHS), "expected_calibration_fit_starts": registration["expected_calibration_fit_starts"],
        "r94_predictions_sha256": r94_source["predictions_sha256"],
        "r95_predictions_sha256": r95_source["predictions_sha256"],
        "data_sha256": source["data_sha256"], "source_sha256": current_hash,
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    campaign._write_progress(
        "r96_registered", r96_card=CARD,
        next_action="run R96 strict-past blend of saved R94 spatial and R95 municipality corrections",
    )
    return registration


def _load_safe_metadata() -> pd.DataFrame:
    if _sha256(campaign.DATA) != campaign.PROTOCOL["data_sha256"]:
        raise RuntimeError("cached listings data hash differs from protocol")
    filters = []
    for month in MONTHS:
        start = pd.Timestamp(month + "-01")
        filters.append([("date_sold", ">=", start), ("date_sold", "<", start + pd.offsets.MonthBegin(1))])
    metadata = pd.read_parquet(
        campaign.DATA,
        columns=["ml_num", "date_sold", "sold_price", "municipality", "house_type_name"],
        filters=filters,
    )
    metadata["date_sold"] = pd.to_datetime(metadata.date_sold, errors="raise")
    metadata["month"] = metadata.date_sold.dt.to_period("M").astype(str)
    observed = sorted(metadata.month.unique().tolist())
    if observed != sorted(MONTHS) or set(observed).intersection(campaign.PROTOCOL["reserved_months"]):
        raise RuntimeError(f"R96 exact development filter returned unexpected months: {observed}")
    metadata["ml_num"] = metadata.ml_num.astype(str)
    if metadata.ml_num.duplicated().any():
        raise RuntimeError("R96 source metadata has duplicate listing IDs")
    metadata["source_sold_price"] = pd.to_numeric(metadata.sold_price, errors="raise")
    metadata = metadata.drop(columns="sold_price")
    for column in ("municipality", "house_type_name"):
        metadata[column] = metadata[column].astype("string").fillna("__MISSING__").astype(str)
    return metadata


def _choose_alpha(history: pd.DataFrame) -> tuple[float, dict]:
    if len(history) < MIN_GROUP_HISTORY_ROWS or history.month.nunique() < MIN_GROUP_HISTORY_MONTHS:
        return 0.0, {"fallback": True, "history_rows": int(len(history)),
                     "history_months": int(history.month.nunique()), "selected_alpha": 0.0,
                     "reason": "insufficient_strict_past_group_history"}
    actual = history.sold_price.to_numpy(float)
    base = history.base_prediction.to_numpy(float)
    spatial = history.spatial_correction_log.to_numpy(float)
    municipal = history.municipality_correction_log.to_numpy(float)
    spatial_only = np.rint(base * np.exp(np.clip(spatial, -MAX_ABS_COMBINED_LOG_CORRECTION,
                                                  MAX_ABS_COMBINED_LOG_CORRECTION)))
    reference = common._metrics(actual, spatial_only)
    eligible = []
    for alpha in ALPHA_GRID:
        combined = np.clip(spatial + alpha * municipal,
                           -MAX_ABS_COMBINED_LOG_CORRECTION, MAX_ABS_COMBINED_LOG_CORRECTION)
        prediction = np.rint(base * np.exp(combined))
        score = common._metrics(actual, prediction)
        mape_change = 100.0 * (score["mape_percent"] - reference["mape_percent"]) / reference["mape_percent"]
        if mape_change <= MAPE_GUARD_PERCENT:
            eligible.append((score["mdape_percent"], score["mape_percent"], alpha, mape_change))
    if not eligible:
        return 0.0, {"fallback": True, "history_rows": int(len(history)),
                     "history_months": int(history.month.nunique()), "selected_alpha": 0.0,
                     "reason": "all_alpha_values_failed_past_mape_guard"}
    mdape, mape, alpha, mape_change = min(eligible, key=lambda item: (item[0], item[1], item[2]))
    return float(alpha), {
        "fallback": False, "history_rows": int(len(history)), "history_months": int(history.month.nunique()),
        "spatial_only_past_mdape_percent": reference["mdape_percent"],
        "selected_past_mdape_percent": mdape, "selected_past_mape_percent": mape,
        "selected_past_mape_relative_change_percent": float(mape_change), "selected_alpha": float(alpha),
    }


def _track_score(frame: pd.DataFrame) -> dict:
    actual = frame.sold_price.to_numpy(float)
    base = frame.base_prediction.to_numpy(float)
    spatial = frame.spatial_prediction.to_numpy(float)
    candidate = frame.candidate_prediction.to_numpy(float)
    base_metrics = common._metrics(actual, base)
    candidate_metrics = common._metrics(actual, candidate)
    spatial_metrics = common._metrics(actual, spatial)
    monthly = {}
    for month, group in frame.groupby("month", sort=True):
        month_actual = group.sold_price.to_numpy(float)
        month_base = common._metrics(month_actual, group.base_prediction.to_numpy(float))
        month_spatial = common._metrics(month_actual, group.spatial_prediction.to_numpy(float))
        month_candidate = common._metrics(month_actual, group.candidate_prediction.to_numpy(float))
        monthly[str(month)] = {
            "n": len(group), "base": month_base, "spatial": month_spatial, "candidate": month_candidate,
            "relative_mdape_improvement_percent": 100.0 * (month_base["mdape_percent"] - month_candidate["mdape_percent"]) / month_base["mdape_percent"],
        }
    gain = 100.0 * (base_metrics["mdape_percent"] - candidate_metrics["mdape_percent"]) / base_metrics["mdape_percent"]
    mape_change = 100.0 * (candidate_metrics["mape_percent"] - base_metrics["mape_percent"]) / base_metrics["mape_percent"]
    monthly_gains = [m["relative_mdape_improvement_percent"] for m in monthly.values()]
    gates = campaign.PROTOCOL["gates"]
    target = 7.0 if frame.track.iloc[0] == "off_market" else 4.0
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    base_ape = 100.0 * np.abs(base - actual) / actual
    candidate_ape = 100.0 * np.abs(candidate - actual) / actual
    spatial_ape = 100.0 * np.abs(spatial - actual) / actual
    diff = np.empty(BOOTSTRAP_REPLICATES)
    spatial_diff = np.empty(BOOTSTRAP_REPLICATES)
    for i in range(BOOTSTRAP_REPLICATES):
        ix = rng.integers(0, len(actual), size=len(actual))
        diff[i] = np.median(candidate_ape[ix]) - np.median(base_ape[ix])
        spatial_diff[i] = np.median(candidate_ape[ix]) - np.median(spatial_ape[ix])
    bootstrap = {"lower95": float(np.quantile(diff, .025)), "upper95": float(np.quantile(diff, .975)),
                 "replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED}
    bootstrap_vs_spatial = {"lower95": float(np.quantile(spatial_diff, .025)),
                            "upper95": float(np.quantile(spatial_diff, .975)),
                            "replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED}
    checks = {
        "relative_mdape_improvement": gain >= gates["development_relative_mdape_improvement_percent"],
        "months_improved": sum(g > 0 for g in monthly_gains) >= gates["development_months_improved_min"],
        "worst_month_regression": max([0.0] + [-g for g in monthly_gains]) <= gates["development_worst_month_regression_max_percent"],
        "mape_regression": mape_change <= gates["development_mape_regression_max_percent"],
        "bootstrap_upper_below_zero": bootstrap["upper95"] < 0.0,
        "strict_target": candidate_metrics["mdape_percent"] < target,
    }
    return {
        "base": base_metrics, "r94_spatial": spatial_metrics, "candidate": candidate_metrics,
        "relative_mdape_improvement_percent": float(gain), "mape_relative_change_percent": float(mape_change),
        "months_improved": int(sum(g > 0 for g in monthly_gains)),
        "worst_month_regression_percent": float(max([0.0] + [-g for g in monthly_gains])),
        "paired_bootstrap_mdape_difference_pp": bootstrap,
        "paired_bootstrap_vs_r94_spatial_mdape_difference_pp": bootstrap_vs_spatial,
        "monthly": monthly, "gate_checks": checks, "qualifies_development": all(checks.values()),
        "target_mdape_strictly_below": target,
    }


def run() -> dict:
    card = campaign.PROTOCOL["cards"].get(CARD)
    if not card or card.get("status") != "registered":
        raise RuntimeError("R96 must be registered before execution")
    if card.get("source_sha256") != _sha256(Path(__file__)):
        raise RuntimeError("R96 source hash changed after registration")
    if _sha256(campaign.DATA) != card["data_sha256"]:
        raise RuntimeError("R96 data hash differs from registration")
    r94_info = _verify_run(R94_RUN, "R94_past_oof_house_type_spatial_strength")
    r95_info = _verify_run(R95_RUN, "R95_municipality_type_past_oof_residual_difference")
    r94 = pd.read_csv(campaign.ROOT / r94_info["predictions_path"], dtype={"ml_num": str})
    r95 = pd.read_csv(campaign.ROOT / r95_info["predictions_path"], dtype={"ml_num": str})
    keys = ["month", "track", "ml_num"]
    if r94.duplicated(keys).any() or r95.duplicated(keys).any():
        raise RuntimeError("R96 inputs contain duplicate month/track/listing keys")
    r95 = r95[keys + ["base_prediction", "sold_price", "correction_log", "house_type_name"]].rename(
        columns={"correction_log": "municipality_correction_log", "house_type_name": "r95_house_type_name"}
    )
    rows = r94.merge(r95, on=keys, how="inner", validate="one_to_one", suffixes=("", "_r95"))
    if len(rows) != 31284 or len(rows) != len(r94) or len(rows) != len(r95):
        raise RuntimeError("R94/R95 prediction rows do not align exactly")
    if not np.array_equal(rows.sold_price.to_numpy(float), rows.sold_price_r95.to_numpy(float)):
        raise RuntimeError("R94/R95 labels differ on aligned rows")
    if not np.array_equal(rows.base_prediction.to_numpy(float), rows.base_prediction_r95.to_numpy(float)):
        raise RuntimeError("R94/R95 parent predictions differ on aligned rows")
    type_a = rows.house_type_name.astype("string").fillna("__MISSING__").astype(str).to_numpy()
    type_b = rows.r95_house_type_name.astype("string").fillna("__MISSING__").astype(str).to_numpy()
    if not np.array_equal(type_a, type_b):
        raise RuntimeError("R94/R95 house-type assignments differ")
    rows["house_type_name"] = type_a
    rows["month"] = rows.month.astype(str)
    metadata = _load_safe_metadata()
    metadata = metadata.rename(columns={"house_type_name": "source_house_type_name"})
    rows = rows.merge(metadata, on=["month", "ml_num"], how="left", validate="many_to_one")
    if rows.date_sold.isna().any() or rows.source_sold_price.isna().any():
        raise RuntimeError("R96 safe source metadata did not cover every paired prediction")
    if not np.array_equal(rows.sold_price.to_numpy(float), rows.source_sold_price.to_numpy(float)):
        raise RuntimeError("R96 paired labels differ from exact-month-filtered source labels")
    source_type = rows.source_house_type_name.astype("string").fillna("__MISSING__").astype(str).to_numpy()
    if not np.array_equal(type_a, source_type):
        raise RuntimeError("R94/R95 house types differ from exact-month-filtered source metadata")
    rows["track"] = rows.track.astype(str)
    rows["spatial_correction_log"] = rows.applied_correction_log.to_numpy(float)
    rows["municipality_correction_log"] = rows.municipality_correction_log.to_numpy(float)
    if (not np.isfinite(rows.spatial_correction_log.to_numpy(float)).all()
            or not np.isfinite(rows.municipality_correction_log.to_numpy(float)).all()):
        raise RuntimeError("R96 input correction vectors must be finite")

    run_id = "r96b_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = campaign.CAMPAIGN_ROOT / "batches" / run_id
    output.mkdir(parents=True, exist_ok=False)
    provenance = output / "provenance"
    provenance.mkdir()
    shutil_inputs = [campaign.PROTOCOL_PATH]
    for rel in [r94_info["result_path"], r94_info["manifest_path"], r95_info["result_path"], r95_info["manifest_path"]]:
        shutil_inputs.append(campaign.ROOT / rel)
    import shutil
    for path in shutil_inputs:
        shutil.copy2(path, provenance / path.name)
    campaign._append({
        "event": "card_start", "card": CARD, "run_id": run_id, "tracks": list(TRACKS),
        "development_months": list(MONTHS), "expected_calibration_fit_starts": (len(MONTHS) - 1) * len(TRACKS),
        "r94_predictions_sha256": r94_info["predictions_sha256"], "r95_predictions_sha256": r95_info["predictions_sha256"],
        "data_sha256": card["data_sha256"], "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    fit_details, frames = [], []
    try:
        for track in TRACKS:
            track_rows = rows.loc[rows.track.eq(track)].copy()
            for month_index, month in enumerate(MONTHS):
                current = track_rows.loc[track_rows.month.eq(month)].copy().reset_index(drop=True)
                cutoff = pd.Timestamp(month + "-01")
                history = track_rows.loc[pd.to_datetime(track_rows.date_sold).lt(cutoff)].copy()
                if not history.empty and (pd.to_datetime(history.date_sold) >= cutoff).any():
                    raise RuntimeError(f"R96 history crossed strict cutoff for {track}/{month}")
                alpha_by_type, selection_by_type = {}, {}
                if month_index == 0:
                    alpha_by_type = {str(x): 0.0 for x in current.house_type_name.unique()}
                    selection_by_type = {str(x): {"fallback": True, "history_rows": 0,
                                                  "history_months": 0, "selected_alpha": 0.0,
                                                  "reason": "first_development_month_has_no_past_oof"}
                                        for x in current.house_type_name.unique()}
                else:
                    fit_id = f"{campaign.PROTOCOL['campaign_id']}:{CARD}:{track}:{month}:{run_id}"
                    campaign._append({
                        "event": "fit_start", "card": CARD, "run_id": run_id, "fit_id": fit_id,
                        "stage": "past_oof_blend_weight_selection", "track": track, "month": month,
                        "training_oof_rows": len(history), "training_oof_months": sorted(history.month.unique().tolist()),
                        "history_cutoff_exclusive": cutoff.isoformat(), "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "data_sha256": card["data_sha256"], "reserved_months_labels_opened": False,
                        "hidden_labels_opened": False,
                    })
                    for house_type in current.house_type_name.unique():
                        group_history = history.loc[history.house_type_name.eq(house_type)]
                        alpha, details = _choose_alpha(group_history)
                        alpha_by_type[str(house_type)] = alpha
                        selection_by_type[str(house_type)] = details
                    fit_detail = {
                        "fit_id": fit_id, "track": track, "month": month, "status": "complete",
                        "training_oof_rows": len(history), "training_oof_months": sorted(history.month.unique().tolist()),
                        "selected_alpha_by_house_type": alpha_by_type,
                        "selection_details_by_house_type": selection_by_type,
                    }
                    fit_details.append(fit_detail)
                    campaign._append({
                        "event": "fit_complete", "card": CARD, "run_id": run_id,
                        "stage": "past_oof_blend_weight_selection", **fit_detail,
                        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
                    })
                alpha = current.house_type_name.map(alpha_by_type).fillna(0.0).to_numpy(float)
                spatial = current.spatial_correction_log.to_numpy(float)
                municipal = current.municipality_correction_log.to_numpy(float)
                combined = np.clip(spatial + alpha * municipal,
                                   -MAX_ABS_COMBINED_LOG_CORRECTION, MAX_ABS_COMBINED_LOG_CORRECTION)
                candidate = np.rint(current.base_prediction.to_numpy(float) * np.exp(combined))
                if not np.isfinite(candidate).all() or (candidate <= 0.0).any():
                    raise RuntimeError(f"R96 produced invalid predictions for {track}/{month}")
                current["selected_alpha"] = alpha
                current["spatial_prediction"] = np.rint(
                    current.base_prediction.to_numpy(float) * np.exp(np.clip(spatial, -MAX_ABS_COMBINED_LOG_CORRECTION,
                                                                                MAX_ABS_COMBINED_LOG_CORRECTION))
                )
                current["candidate_prediction"] = candidate
                current["combined_correction_log"] = combined
                frames.append(current[[
                    "month", "track", "ml_num", "sold_price", "base_prediction", "spatial_prediction",
                    "candidate_prediction", "selected_alpha", "spatial_correction_log",
                    "municipality_correction_log", "combined_correction_log", "house_type_name", "municipality",
                    "date_sold",
                ]])
        paired = pd.concat(frames, ignore_index=True)
        if len(paired) != 31284 or paired.duplicated(["month", "track", "ml_num"]).any():
            raise RuntimeError("R96 row coverage or listing-key uniqueness is invalid")
        scores = {track: _track_score(paired.loc[paired.track.eq(track)].copy()) for track in TRACKS}
        result = {
            "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id, "card": CARD,
            "stage": "strict_past_walk_forward_development", "development_months": list(MONTHS),
            "fit_starts": len(fit_details), "tracks": scores, "fit_details": fit_details,
            "r94_source": r94_info, "r95_source": r95_info, "data_sha256": card["data_sha256"],
            "reserved_months_labels_opened": False, "reserve_validation_eligible": False,
            "hidden_labels_opened": False,
        }
        pairs_path = output / "paired_predictions.csv"
        paired.to_csv(pairs_path, index=False)
        result_path = output / "development_results.json"
        result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        source_files = [Path(__file__), Path(common.__file__), campaign.PROTOCOL_PATH, campaign.PACKAGE / "runner.py"]
        manifest = {
            "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
            "source_identity": {"data_sha256": card["data_sha256"], "parent_commit": card["parent_commit"]},
            "source_sha256": {_relative(p): _sha256(p) for p in source_files if p.is_file()},
            "r94_source": r94_info, "r95_source": r95_info, "fit_starts": len(fit_details),
            "fit_details": fit_details, "paired_predictions_sha256": _sha256(pairs_path),
            "result_sha256": _sha256(result_path), "reserved_months_labels_opened": False,
            "reserve_validation_eligible": False, "hidden_labels_opened": False,
        }
        manifest_path = output / "run_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        report_lines = [
            "# R96：严格过去 OOF 空间 × 市镇校正混合", "",
            f"运行：`{run_id}`；输入为经哈希核验的 R94 与 R95 八折 OOF 结果。",
            "每个赛道、房型和月份，只用严格早于预测月的历史 OOF 行选择 municipality correction 权重。",
            "选择目标为历史 MdAPE，MAPE 相对 R94 spatial-only 的退化不超过 1%；校正限幅 ±0.05。", "",
            "| Track | Base MdAPE | R94 spatial | R96 candidate | 相对改善 | MAPE变化 | 月份改善 | bootstrap upper95 (pp) | <目标 | 晋级 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
        ]
        for track, score in scores.items():
            report_lines.append(
                f"| {track} | {score['base']['mdape_percent']:.5f}% | {score['r94_spatial']['mdape_percent']:.5f}% | "
                f"{score['candidate']['mdape_percent']:.5f}% | {score['relative_mdape_improvement_percent']:+.3f}% | "
                f"{score['mape_relative_change_percent']:+.3f}% | {score['months_improved']}/8 | "
                f"{score['paired_bootstrap_mdape_difference_pp']['upper95']:+.5f} | "
                f"{'是' if score['candidate']['mdape_percent'] < score['target_mdape_strictly_below'] else '否'} | "
                f"{'是' if score['qualifies_development'] else '否'} |"
            )
        report_lines += ["", "## Alpha 选择", ""]
        for fit in fit_details:
            report_lines.append(f"- {fit['track']} / {fit['month']}: {fit['selected_alpha_by_house_type']}")
        report_lines += [
            "", "## 限制", "",
            "本轮只使用登记的八个安全开发月和保存的 R94/R95 OOF 预测。2025-05/08 预留标签与隐藏标签未读取；预留集仍不具备独立验证资格。",
            "失败的校正组合不替换 R22/R35。固定起点检查只对通过开发门槛的赛道执行。", "",
        ]
        report_path = output / "research_report.md"
        report_path.write_text("\n".join(report_lines), encoding="utf-8")
        passed = {track: score["qualifies_development"] for track, score in scores.items()}
        campaign._append({
            "event": "card_complete", "card": CARD, "run_id": run_id, "fit_starts": len(fit_details),
            "qualifies_by_track": passed, "result_path": _relative(result_path),
            "paired_predictions_sha256": manifest["paired_predictions_sha256"],
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        campaign._write_progress(
            "r96_development_complete", r96_result=_relative(result_path), r96_run_id=run_id,
            r96_fit_starts=len(fit_details),
            next_action=("run fixed-origin confirmation for: " + ", ".join(t for t, ok in passed.items() if ok)
                         if any(passed.values()) else "R96 did not pass development; investigate a distinct route"),
        )
        return result
    except Exception as exc:
        checkpoint = {
            "run_id": run_id, "card": CARD, "fit_starts": len(fit_details),
            "completed_calibrations": fit_details, "error": f"{type(exc).__name__}: {exc}",
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        }
        (output / "failure_checkpoint.json").write_text(
            json.dumps(checkpoint, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        campaign._append({
            "event": "card_failed", "card": CARD, "run_id": run_id, "fit_starts": len(fit_details),
            "error": checkpoint["error"], "reserved_months_labels_opened": False,
            "hidden_labels_opened": False,
        })
        campaign._write_progress(
            "r96_failed", r96_failure=_relative(output / "failure_checkpoint.json"),
            next_action="audit R96 failure checkpoint and ledger before retry", latest_error=checkpoint["error"],
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register", action="store_true")
    args = parser.parse_args()
    if args.register:
        card = register_card()
        print(json.dumps({"card": CARD, "status": card["status"],
                          "expected_calibration_fit_starts": card["expected_calibration_fit_starts"]}, ensure_ascii=False), flush=True)
        return
    result = run()
    for track, score in result["tracks"].items():
        print(
            f"{track}: base={score['base']['mdape_percent']:.5f}% "
            f"spatial={score['r94_spatial']['mdape_percent']:.5f}% "
            f"candidate={score['candidate']['mdape_percent']:.5f}% "
            f"gain={score['relative_mdape_improvement_percent']:+.3f}% "
            f"MAPE_change={score['mape_relative_change_percent']:+.3f}% "
            f"months={score['months_improved']}/8 "
            f"bootstrap_upper95={score['paired_bootstrap_mdape_difference_pp']['upper95']:+.5f} "
            f"qualifies={score['qualifies_development']}", flush=True,
        )
    print(f"report={campaign.CAMPAIGN_ROOT / 'batches' / result['run_id'] / 'research_report.md'}", flush=True)


if __name__ == "__main__":
    main()
