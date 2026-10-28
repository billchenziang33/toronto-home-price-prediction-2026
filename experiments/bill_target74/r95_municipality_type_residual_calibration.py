"""R95: strict-past municipality x house-type residual-difference calibration."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.bill_target74 import r93_spatial_oof_calibration as common
from experiments.bill_target74 import runner as campaign


CARD = "R95_municipality_type_past_oof_residual_difference"
TRACKS = common.TRACKS
MONTHS = common.MONTHS
MIN_GROUP_HISTORY_ROWS = 500
MIN_GROUP_HISTORY_MONTHS = 2
RECENCY_HALF_LIFE_MONTHS = 12.0
SHRINKAGE_PSEUDOCOUNT = 500.0
MAX_ABS_LOG_CORRECTION = 0.03
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20261029
R93_RESULT_REL = Path(
    "outputs/bill_target74/bill_target74_unbounded_20261024/batches/"
    "r93s_20261025T201838Z/development_results.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return str(path.relative_to(campaign.ROOT)).replace("\\", "/")


def _r93_paths() -> tuple[Path, Path, Path]:
    result_path = campaign.ROOT / R93_RESULT_REL
    return result_path, result_path.parent / "paired_predictions.csv", result_path.parent / "run_manifest.json"


def _verify_r93() -> dict:
    result_path, predictions_path, manifest_path = _r93_paths()
    if not all(path.is_file() for path in (result_path, predictions_path, manifest_path)):
        raise FileNotFoundError("R95 requires complete saved R93 artifacts")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (result.get("card") != common.CARD
            or result.get("data_sha256") != campaign.PROTOCOL["data_sha256"]
            or result.get("reserved_months_labels_opened") is not False
            or result.get("hidden_labels_opened") is not False):
        raise RuntimeError("R93 identity, data hash or label-access flags are invalid")
    if (manifest.get("paired_predictions_sha256") != _sha256(predictions_path)
            or manifest.get("result_sha256") != _sha256(result_path)):
        raise RuntimeError("R93 output hashes differ from its manifest")
    return {
        "run_id": result["run_id"],
        "result_path": _relative(result_path), "result_sha256": _sha256(result_path),
        "predictions_path": _relative(predictions_path), "predictions_sha256": _sha256(predictions_path),
        "manifest_path": _relative(manifest_path), "manifest_sha256": _sha256(manifest_path),
    }


def register_card() -> dict:
    from experiments.bill_target74 import r04_ablation as identity

    existing = campaign.PROTOCOL["cards"].get(CARD)
    if existing:
        if existing.get("status") == "registered" and existing.get("source_sha256") == _sha256(Path(__file__)):
            return existing
        raise RuntimeError(f"R95 already has status {existing.get('status')!r}")
    source = identity._source_identity()
    r93_identity = _verify_r93()
    registration = {
        "status": "registered",
        "tracks": list(TRACKS),
        "development_months": list(MONTHS),
        "expected_calibration_fit_starts": (len(MONTHS) - 1) * len(TRACKS),
        "parents": {
            "off_market": "R52 off-market experimental OOF predictions",
            "list_aware": "R35 list-aware validated OOF predictions",
        },
        "r93_source": r93_identity,
        "method": (
            "For each fold, use only earlier safe OOF rows to estimate a recency-weighted median "
            "log(sold_price/base_prediction) for each municipality x house-type group. Subtract the "
            "recency-weighted all-municipality median for the same house type, shrink the difference "
            "by effective group rows/(effective group rows+500), clip to +/-0.03 log points, and "
            "multiply the base price by exp(correction) before dollar rounding. Require at least 500 "
            "past group rows across at least two earlier OOF months; otherwise leave that group unchanged."
        ),
        "difference_from_prior_work": (
            "R21 selected municipality-specific shares for an off-market Detached expert; it did not "
            "calibrate residual bias. R93 used continuous nearest-coordinate residual smoothing. R95 "
            "uses discrete municipality x house-type OOF residual differences and keeps the base model "
            "and all model predictions unchanged outside eligible groups."
        ),
        "parameters": {
            "minimum_group_history_rows": MIN_GROUP_HISTORY_ROWS,
            "minimum_group_history_months": MIN_GROUP_HISTORY_MONTHS,
            "recency_half_life_months": RECENCY_HALF_LIFE_MONTHS,
            "shrinkage_pseudocount": SHRINKAGE_PSEUDOCOUNT,
            "maximum_absolute_log_correction": MAX_ABS_LOG_CORRECTION,
        },
        "development_gate": dict(campaign.PROTOCOL["gates"]),
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
        "development_months": list(MONTHS), "r93_predictions_sha256": r93_identity["predictions_sha256"],
        "expected_calibration_fit_starts": registration["expected_calibration_fit_starts"],
        "data_sha256": source["data_sha256"], "source_sha256": registration["source_sha256"],
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    campaign._write_progress(
        "r95_registered",
        next_action="run R95 strict-past municipality x house-type residual-difference calibration",
        r95_card=CARD,
    )
    return registration


def _load_safe_metadata() -> pd.DataFrame:
    if _sha256(campaign.DATA) != campaign.PROTOCOL["data_sha256"]:
        raise RuntimeError("cached listings data hash differs from protocol")
    filters = []
    for month in MONTHS:
        start = pd.Timestamp(month + "-01")
        filters.append([("date_sold", ">=", start),
                        ("date_sold", "<", start + pd.offsets.MonthBegin(1))])
    metadata = pd.read_parquet(
        campaign.DATA,
        columns=["ml_num", "date_sold", "sold_price", "municipality", "house_type_name"],
        filters=filters,
    )
    metadata["date_sold"] = pd.to_datetime(metadata.date_sold, errors="raise")
    metadata["month"] = metadata.date_sold.dt.to_period("M").astype(str)
    seen = sorted(metadata.month.unique().tolist())
    if seen != sorted(MONTHS) or set(seen).intersection(campaign.PROTOCOL["reserved_months"]):
        raise RuntimeError(f"R95 exact development filter returned unexpected months: {seen}")
    if metadata.ml_num.astype(str).duplicated().any():
        raise RuntimeError("R95 metadata has duplicate listing IDs")
    metadata["ml_num"] = metadata.ml_num.astype(str)
    metadata["source_sold_price"] = pd.to_numeric(metadata.sold_price, errors="raise")
    metadata = metadata.drop(columns=["sold_price"])
    for column in ("municipality", "house_type_name"):
        metadata[column] = metadata[column].astype("string").fillna("__MISSING__").astype(str)
    return metadata


def _fit_group_corrections(history: pd.DataFrame, query: pd.DataFrame,
                           cutoff: pd.Timestamp) -> tuple[np.ndarray, pd.DataFrame]:
    if {"date_sold", "sold_price"}.intersection(query.columns):
        raise ValueError("R95 query inputs must not contain outcomes or sale dates")
    history = history.loc[pd.to_datetime(history.date_sold).lt(cutoff)].copy()
    if history.empty:
        raise ValueError("R95 requires strictly earlier OOF history")
    actual = history.sold_price.to_numpy(float)
    base = history.base_prediction.to_numpy(float)
    if (not np.isfinite(actual).all() or (actual <= 0).any()
            or not np.isfinite(base).all() or (base <= 0).any()):
        raise ValueError("R95 past OOF inputs must be finite and positive")
    history["log_residual"] = np.log(actual / base)
    history["municipality"] = history.municipality.astype("string").fillna("__MISSING__").astype(str)
    history["house_type_name"] = history.house_type_name.astype("string").fillna("__MISSING__").astype(str)
    query = query.copy()
    query["municipality"] = query.municipality.astype("string").fillna("__MISSING__").astype(str)
    query["house_type_name"] = query.house_type_name.astype("string").fillna("__MISSING__").astype(str)
    correction = np.zeros(len(query), dtype=float)
    details = []
    for property_type, query_indices in query.groupby("house_type_name", sort=False).groups.items():
        positions = np.asarray(list(query_indices), dtype=int)
        type_history = history.loc[history.house_type_name.eq(property_type)].copy()
        if len(type_history) < MIN_GROUP_HISTORY_ROWS:
            details.extend({"row_index": int(index), "history_rows": len(type_history),
                            "history_months": int(type_history.month.nunique()), "correction_log": 0.0}
                           for index in positions)
            continue
        cutoff_weights = common._recency_weights(type_history.date_sold, cutoff)
        type_center = common._weighted_median(type_history.log_residual.to_numpy(float), cutoff_weights)
        type_municipality = type_history.groupby("municipality", sort=False).groups
        query_municipality = query.iloc[positions].municipality.to_numpy(str)
        for municipality in pd.unique(query_municipality):
            group_positions = positions[query_municipality == municipality]
            key_history = type_history.loc[type_history.municipality.eq(municipality)].copy()
            group_count = len(key_history)
            group_months = int(key_history.month.nunique())
            delta = 0.0
            if group_count >= MIN_GROUP_HISTORY_ROWS and group_months >= MIN_GROUP_HISTORY_MONTHS:
                weights = common._recency_weights(key_history.date_sold, cutoff)
                local_center = common._weighted_median(key_history.log_residual.to_numpy(float), weights)
                effective_count = float(weights.sum() ** 2 / np.square(weights).sum())
                delta = (local_center - type_center) * effective_count / (
                    effective_count + SHRINKAGE_PSEUDOCOUNT
                )
                delta = float(np.clip(delta, -MAX_ABS_LOG_CORRECTION, MAX_ABS_LOG_CORRECTION))
            else:
                effective_count = 0.0
            correction[group_positions] = delta
            details.extend({
                "row_index": int(index), "history_rows": group_count,
                "history_months": group_months, "effective_history_rows": effective_count,
                "type_global_center": type_center, "correction_log": delta,
            } for index in group_positions)
    diagnostic = pd.DataFrame(details).set_index("row_index").reindex(range(len(query)))
    required_diagnostic = diagnostic[["history_rows", "history_months", "correction_log"]]
    if required_diagnostic.isna().any().any() or not np.isfinite(correction).all():
        raise RuntimeError("R95 failed to create finite corrections for all query rows")
    return correction, diagnostic


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
            "relative_mdape_improvement_percent": 100.0 * (
                before["mdape_percent"] - after["mdape_percent"]
            ) / before["mdape_percent"],
        }
    gain = 100.0 * (base_metrics["mdape_percent"] - candidate_metrics["mdape_percent"]) / base_metrics["mdape_percent"]
    mape_change = 100.0 * (candidate_metrics["mape_percent"] - base_metrics["mape_percent"]) / base_metrics["mape_percent"]
    improvements = [x["relative_mdape_improvement_percent"] for x in monthly.values()]
    months_improved = sum(value > 0 for value in improvements)
    worst_regression = max([0.0] + [-value for value in improvements])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    base_ape = 100.0 * np.abs(base - actual) / actual
    candidate_ape = 100.0 * np.abs(candidate - actual) / actual
    diffs = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    for i in range(BOOTSTRAP_REPLICATES):
        idx = rng.integers(0, len(actual), size=len(actual))
        diffs[i] = np.median(candidate_ape[idx]) - np.median(base_ape[idx])
    bootstrap = {
        "lower95": float(np.quantile(diffs, 0.025)),
        "upper95": float(np.quantile(diffs, 0.975)),
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
    }
    gates = campaign.PROTOCOL["gates"]
    target = 7.0 if frame.track.iloc[0] == "off_market" else 4.0
    gate_checks = {
        "relative_mdape_improvement": gain >= gates["development_relative_mdape_improvement_percent"],
        "months_improved": months_improved >= gates["development_months_improved_min"],
        "worst_month_regression": worst_regression <= gates["development_worst_month_regression_max_percent"],
        "mape_regression": mape_change <= gates["development_mape_regression_max_percent"],
        "bootstrap_upper_below_zero": bootstrap["upper95"] < 0.0,
        "strict_target": candidate_metrics["mdape_percent"] < target,
    }
    return {
        "base": base_metrics, "candidate": candidate_metrics,
        "relative_mdape_improvement_percent": gain, "mape_relative_change_percent": mape_change,
        "months_improved": months_improved, "worst_month_regression_percent": worst_regression,
        "paired_bootstrap_mdape_difference_pp": bootstrap, "monthly": monthly,
        "gate_checks": gate_checks, "qualifies_development": all(gate_checks.values()),
        "target_mdape_strictly_below": target,
    }


def run() -> dict:
    from experiments.bill_target74 import r04_ablation as identity

    card = campaign.PROTOCOL["cards"].get(CARD)
    if not card or card.get("status") != "registered":
        raise RuntimeError("R95 must be registered before execution")
    if card.get("source_sha256") != _sha256(Path(__file__)):
        raise RuntimeError("R95 source hash changed after registration")
    source = identity._source_identity()
    if source["data_sha256"] != card["data_sha256"]:
        raise RuntimeError("R95 data hash differs from registration")
    r93_identity = _verify_r93()
    r93_rows = pd.read_csv(campaign.ROOT / r93_identity["predictions_path"], dtype={"ml_num": str})
    metadata = _load_safe_metadata()
    # Each development listing has one row per track in R93, so the safe
    # metadata table is unique by month/listing but the predictions are not.
    source_metadata = metadata.rename(columns={"house_type_name": "source_house_type_name"})
    r93_rows = r93_rows.merge(
        source_metadata, on=["month", "ml_num"], how="left", validate="many_to_one"
    )
    if r93_rows.date_sold.isna().any() or r93_rows.source_sold_price.isna().any():
        raise RuntimeError("R95 R93 prediction rows failed exact development metadata alignment")
    if not np.array_equal(r93_rows.sold_price.to_numpy(float), r93_rows.source_sold_price.to_numpy(float)):
        raise RuntimeError("R93 labels differ from date-filtered development source labels")
    r93_type = r93_rows.house_type_name.astype("string").fillna("__MISSING__").astype(str)
    source_type = r93_rows.source_house_type_name.astype("string").fillna("__MISSING__").astype(str)
    if not np.array_equal(r93_type.to_numpy(), source_type.to_numpy()):
        raise RuntimeError("R93 house types differ from date-filtered development metadata")
    r93_rows = r93_rows.drop(columns=["source_sold_price", "source_house_type_name"])
    run_id = "r95m_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = campaign.CAMPAIGN_ROOT / "batches" / run_id
    output.mkdir(parents=True, exist_ok=False)
    provenance = output / "provenance"
    provenance.mkdir()
    shutil.copy2(campaign.PROTOCOL_PATH, provenance / "registered_campaign_protocol.json")
    result_path, _, manifest_path = _r93_paths()
    shutil.copy2(result_path, provenance / "r93_development_results.json")
    shutil.copy2(manifest_path, provenance / "r93_run_manifest.json")
    campaign._append({
        "event": "card_start", "card": CARD, "run_id": run_id,
        "tracks": list(TRACKS), "development_months": list(MONTHS),
        "expected_calibration_fit_starts": (len(MONTHS) - 1) * len(TRACKS),
        "r93_predictions_sha256": r93_identity["predictions_sha256"],
        "data_sha256": source["data_sha256"],
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    fit_details, fit_start_count, output_frames = [], 0, []
    try:
        for track in TRACKS:
            track_rows = r93_rows.loc[r93_rows.track.eq(track)].copy()
            track_rows["month"] = track_rows.month.astype(str)
            for month_index, month in enumerate(MONTHS):
                current = track_rows.loc[track_rows.month.eq(month)].copy().reset_index(drop=True)
                history = track_rows.loc[track_rows.month.isin(MONTHS[:month_index])].copy()
                cutoff = pd.Timestamp(month + "-01")
                if not history.empty and pd.to_datetime(history.date_sold).max() >= cutoff:
                    raise RuntimeError(f"R95 history crossed its strict cutoff for {track}/{month}")
                query = current[["municipality", "house_type_name", "base_prediction"]].copy()
                correction = np.zeros(len(current), dtype=float)
                diagnostic = pd.DataFrame({
                    "history_rows": np.zeros(len(current), dtype=int),
                    "history_months": np.zeros(len(current), dtype=int),
                    "correction_log": np.zeros(len(current), dtype=float),
                })
                if month_index > 0:
                    fit_start_count += 1
                    fit_id = f"{campaign.PROTOCOL['campaign_id']}:{CARD}:{track}:{month}:{run_id}"
                    campaign._append({
                        "event": "fit_start", "card": CARD, "run_id": run_id,
                        "fit_id": fit_id, "stage": "calibration", "track": track,
                        "month": month, "training_oof_rows": len(history),
                        "training_oof_months": sorted(history.month.unique().tolist()),
                        "history_cutoff_exclusive": cutoff.isoformat(),
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "data_sha256": source["data_sha256"],
                    })
                    try:
                        correction, diagnostic = _fit_group_corrections(history, query, cutoff)
                        candidate = np.rint(current.base_prediction.to_numpy(float) * np.exp(correction))
                        if not np.isfinite(candidate).all() or (candidate <= 0).any():
                            raise RuntimeError("R95 generated invalid predictions")
                    except Exception as exc:
                        campaign._append({
                            "event": "fit_complete", "card": CARD, "run_id": run_id,
                            "fit_id": fit_id, "stage": "calibration", "track": track,
                            "month": month, "status": "failed",
                            "error": f"{type(exc).__name__}: {exc}",
                        })
                        raise
                    fit_detail = {
                        "fit_id": fit_id, "track": track, "month": month,
                        "status": "complete", "training_oof_rows": len(history),
                        "training_oof_months": sorted(history.month.unique().tolist()),
                        "corrected_rows": int(np.count_nonzero(correction)),
                        "mean_neighbor_rows": float(diagnostic.history_rows.mean()),
                        "median_log_correction_abs": float(np.median(np.abs(correction))),
                        "group_support": diagnostic.groupby(
                            [query.house_type_name, query.municipality], dropna=False
                        ).agg({"history_rows": "first", "history_months": "first",
                               "correction_log": "first"}).reset_index().to_dict(orient="records"),
                    }
                    fit_details.append(fit_detail)
                    campaign._append({
                        "event": "fit_complete", "card": CARD, "run_id": run_id,
                        "stage": "calibration", **fit_detail,
                        "reserved_months_labels_opened": False,
                        "hidden_labels_opened": False,
                    })
                else:
                    candidate = current.base_prediction.to_numpy(float).copy()
                current["candidate_prediction"] = candidate
                current["correction_log"] = correction
                current["history_rows"] = diagnostic.history_rows.to_numpy(int)
                current["history_months"] = diagnostic.history_months.to_numpy(int)
                current["base_ape_percent"] = (
                    100.0 * np.abs(current.base_prediction - current.sold_price) / current.sold_price
                )
                current["candidate_ape_percent"] = (
                    100.0 * np.abs(current.candidate_prediction - current.sold_price) / current.sold_price
                )
                output_frames.append(current[[
                    "month", "track", "ml_num", "sold_price", "base_prediction",
                    "candidate_prediction", "base_ape_percent", "candidate_ape_percent",
                    "municipality", "house_type_name", "correction_log", "history_rows", "history_months",
                ]])
        paired = pd.concat(output_frames, ignore_index=True)
        if len(paired) != 31284 or paired.duplicated(["month", "track", "ml_num"]).any():
            raise RuntimeError("R95 row coverage or listing ID uniqueness is invalid")
        tracks = {track: _score_track(paired.loc[paired.track.eq(track)].copy()) for track in TRACKS}
        result = {
            "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
            "card": CARD, "stage": "strict_past_walk_forward_development",
            "development_months": list(MONTHS), "fit_starts": fit_start_count,
            "tracks": tracks, "fit_details": fit_details, "r93_source": r93_identity,
            "parameters": campaign.PROTOCOL["cards"][CARD]["parameters"],
            "data_sha256": source["data_sha256"],
            "reserved_months_labels_opened": False,
            "reserve_validation_eligible": False,
            "hidden_labels_opened": False,
        }
        paired_path = output / "paired_predictions.csv"
        paired.to_csv(paired_path, index=False)
        result_file = output / "development_results.json"
        result_file.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        source_paths = [Path(__file__), Path(common.__file__),
                        campaign.PROTOCOL_PATH, campaign.PACKAGE / "runner.py"]
        manifest = {
            "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
            "source_identity": source,
            "source_sha256": {_relative(p): _sha256(p) for p in source_paths if p.is_file()},
            "r93_source": r93_identity, "fit_starts": fit_start_count,
            "fit_details": fit_details,
            "paired_predictions_sha256": _sha256(paired_path),
            "result_sha256": _sha256(result_file),
            "reserved_months_labels_opened": False,
            "reserve_validation_eligible": False,
            "hidden_labels_opened": False,
        }
        (output / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        lines = [
            "# R95：严格过去 OOF 市镇 × 房型残差差值校正", "",
            f"运行：`{run_id}`；输入为哈希核验的 R93 八折 OOF 预测。",
            "每个外层月仅使用此前的 OOF 标签与预测；市镇 × 房型组过去至少 500 套、至少两个 OOF 月才校正。",
            "校正为该组中位 log 残差减去同房型全市镇中位，按有效样本收缩并限制在 ±0.03。", "",
            "| Track | Base MdAPE | Candidate MdAPE | 相对改善 | MAPE变化 | 月份改善 | bootstrap upper95 (pp) | 达到目标 | 晋级 |",
            "|---|---:|---:|---:|---:|---:|---:|---|---|",
        ]
        for track, score in tracks.items():
            lines.append(
                f"| {track} | {score['base']['mdape_percent']:.5f}% | "
                f"{score['candidate']['mdape_percent']:.5f}% | "
                f"{score['relative_mdape_improvement_percent']:+.3f}% | "
                f"{score['mape_relative_change_percent']:+.3f}% | "
                f"{score['months_improved']}/8 | "
                f"{score['paired_bootstrap_mdape_difference_pp']['upper95']:+.5f} | "
                f"{'是' if score['candidate']['mdape_percent'] < score['target_mdape_strictly_below'] else '否'} | "
                f"{'是' if score['qualifies_development'] else '否'} |"
            )
        lines.extend(["", "## 逐月 MdAPE", "", "| Track | Month | Base | Candidate | 相对改善 |", "|---|---|---:|---:|---:|"])
        for track, score in tracks.items():
            for month, monthly in score["monthly"].items():
                lines.append(
                    f"| {track} | {month} | {monthly['base']['mdape_percent']:.5f}% | "
                    f"{monthly['candidate']['mdape_percent']:.5f}% | "
                    f"{monthly['relative_mdape_improvement_percent']:+.3f}% |"
                )
        lines.extend([
            "", f"Calibration fit starts：{fit_start_count}。",
            "只读取八个安全开发月的登记字段；2025-05/08 和隐藏标签未读取，保留集验证仍不可用。", "",
        ])
        report = output / "research_report.md"
        report.write_text("\n".join(lines), encoding="utf-8")
        passed = {track: score["qualifies_development"] for track, score in tracks.items()}
        campaign._append({
            "event": "card_complete", "card": CARD, "run_id": run_id,
            "fit_starts": fit_start_count, "qualifies_by_track": passed,
            "result_path": _relative(result_file),
            "paired_predictions_sha256": manifest["paired_predictions_sha256"],
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        campaign._write_progress(
            "r95_development_complete",
            next_action=(
                "R95 passed development for: " + ", ".join(t for t, ok in passed.items() if ok)
                if any(passed.values()) else
                "R95 municipality x type calibration failed development; register a distinct evidence-based hypothesis"
            ),
            r95_result=_relative(result_file), r95_run_id=run_id, r95_fit_starts=fit_start_count,
        )
        return result
    except Exception as exc:
        checkpoint = {
            "run_id": run_id, "card": CARD, "fit_starts": fit_start_count,
            "completed_calibrations": fit_details,
            "error": f"{type(exc).__name__}: {exc}",
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
            "r95_failed", next_action="audit R95 failure checkpoint and ledger before any retry",
            r95_failure=_relative(output / "failure_checkpoint.json"), latest_error=checkpoint["error"],
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register", action="store_true", help="preregister the R95 method")
    args = parser.parse_args()
    if args.register:
        card = register_card()
        print(json.dumps({"card": CARD, "status": card["status"],
                          "expected_calibration_fit_starts": card["expected_calibration_fit_starts"]},
                         ensure_ascii=False), flush=True)
        return
    result = run()
    for track, score in result["tracks"].items():
        print(
            f"{track}: base={score['base']['mdape_percent']:.5f}% "
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
