"""R136: apply strict-past spatial residual correction to R126 off-market OOF prices."""
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

from experiments.bill_target74 import runner as campaign


CARD = "R136_offmarket_r126_spatial_residual"
TRACK = "off_market"
BASE_CARD = "R126_R125_offmarket_cyclic_month_development_confirmation"
SCREEN_MONTHS = tuple(campaign.PROTOCOL["screen_months"])
DEVELOPMENT_MONTHS = tuple(campaign.PROTOCOL["development_months"])
TARGET_MDAPE = 7.0
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20261136


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return str(path.relative_to(campaign.ROOT)).replace("\\", "/")


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n",
                    encoding="utf-8")


def _source_files() -> list[Path]:
    return [
        Path(__file__),
        campaign.PACKAGE / "r93_spatial_oof_calibration.py",
        campaign.PACKAGE / "r125_cyclic_listing_month_features.py",
        campaign.PACKAGE / "r126_r125_offmarket_development_confirmation.py",
        campaign.PACKAGE / "r22_remarks_tfidf.py",
        campaign.PACKAGE / "r04_ablation.py",
        campaign.PACKAGE / "runner.py",
        campaign.ROOT / "experiments/bill_greedy/runner.py",
        campaign.ROOT / "experiments/bill_greedy/history.py",
        campaign.ROOT / "experiments/model_optimization_v2/pipeline.py",
        campaign.ROOT / "experiments/model_optimization_v2/folds.py",
    ]


def _code_hashes() -> dict[str, str]:
    return {_relative(path): _sha256(path) for path in _source_files()}


def _r22_paths() -> dict[str, Path]:
    progress = json.loads(campaign.STATE.read_text(encoding="utf-8"))
    result_paths = {
        "screen": campaign.ROOT / progress["r22_result"],
        "development": campaign.ROOT / progress["r22_development_result"],
    }
    return {
        stage: path
        for stage, result in result_paths.items()
        for path in (result, result.parent / "paired_predictions.csv", result.parent / "run_manifest.json")
    }


def _verify_artifacts(result_path: Path, predictions_path: Path, manifest_path: Path,
                      expected_run_id: str, *, development: bool = False) -> tuple[dict, pd.DataFrame, dict]:
    if not all(path.is_file() for path in (result_path, predictions_path, manifest_path)):
        raise FileNotFoundError("a registered R22/R126 result, prediction, or manifest is missing")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if result.get("run_id") != expected_run_id or manifest.get("run_id") != expected_run_id:
        raise RuntimeError("parent run ID differs from its registered artifact identity")
    if (manifest.get("result_sha256") != _sha256(result_path)
            or manifest.get("paired_predictions_sha256") != _sha256(predictions_path)):
        raise RuntimeError("parent result or paired-prediction hash does not match its manifest")
    if (result.get("reserved_months_labels_opened") is not False
            or result.get("hidden_labels_opened") is not False):
        raise RuntimeError("R22/R126 parent reports reserve or hidden-label access")
    if result.get("data_sha256") != campaign.PROTOCOL["data_sha256"]:
        raise RuntimeError("parent result data hash differs from the active campaign")
    rows = pd.read_csv(predictions_path, dtype={"ml_num": str})
    rows = rows.loc[rows.track.eq(TRACK)].copy()
    rows["month"] = rows.month.astype(str)
    rows["ml_num"] = rows.ml_num.astype(str)
    expected_months = set(DEVELOPMENT_MONTHS if development else SCREEN_MONTHS)
    if set(rows.month.unique()) != expected_months:
        raise RuntimeError("parent predictions do not cover the registered safe months")
    if rows.duplicated(["month", "ml_num"]).any():
        raise RuntimeError("parent predictions contain duplicate month/listing IDs")
    return result, rows, manifest


def _verified_inputs() -> dict:
    from experiments.bill_target74 import r93_spatial_oof_calibration as spatial

    progress = json.loads(campaign.STATE.read_text(encoding="utf-8"))
    screen_result_path = campaign.ROOT / progress["r22_result"]
    dev_result_path = campaign.ROOT / progress["r22_development_result"]
    screen_result, r22_screen, screen_manifest = _verify_artifacts(
        screen_result_path, screen_result_path.parent / "paired_predictions.csv",
        screen_result_path.parent / "run_manifest.json", "bill_target74_r22_remarks_tfidf_20261024T192106Z",
    )
    dev_result, r22_dev, dev_manifest = _verify_artifacts(
        dev_result_path, dev_result_path.parent / "paired_predictions.csv",
        dev_result_path.parent / "run_manifest.json", "bill_target74_r22_development_20261024T193459Z",
        development=True,
    )
    base_card = campaign.PROTOCOL["cards"].get(BASE_CARD)
    if not base_card or base_card.get("status") != "development_complete":
        raise RuntimeError("R136 requires the completed R126 off-market development route")
    base_result_path = campaign.ROOT / base_card["result_path"]
    base_predictions_path = campaign.ROOT / base_card["paired_predictions_path"]
    base_manifest_path = campaign.ROOT / base_card["manifest_path"]
    base_result, base_rows, base_manifest = _verify_artifacts(
        base_result_path, base_predictions_path, base_manifest_path,
        base_card["run_id"], development=True,
    )
    expected_parent_hashes = {
        "result": base_card["r22_development_result_sha256"],
        "predictions": base_card["r22_development_paired_predictions_sha256"],
    }
    if (expected_parent_hashes["result"] != _sha256(dev_result_path)
            or expected_parent_hashes["predictions"] != _sha256(dev_result_path.parent / "paired_predictions.csv")):
        raise RuntimeError("R126's recorded R22 development parent hashes no longer match")
    if base_card.get("data_sha256") != campaign.PROTOCOL["data_sha256"]:
        raise RuntimeError("R126 data identity differs from the active campaign")

    r22_columns = r22_dev[["month", "ml_num", "sold_price", "candidate_prediction"]].rename(
        columns={"candidate_prediction": "verified_r22_prediction"}
    )
    joined = base_rows.merge(
        r22_columns, on=["month", "ml_num", "sold_price"], how="left", validate="one_to_one",
    )
    if joined.verified_r22_prediction.isna().any() or len(joined) != 15642:
        raise RuntimeError("R126 development rows do not align one-to-one with R22")
    if "parent_prediction" not in joined or not np.array_equal(
        joined.verified_r22_prediction.to_numpy(float), joined.parent_prediction.to_numpy(float),
    ):
        raise RuntimeError("R126 saved parent predictions differ from the verified R22 development parent")
    joined = joined.drop(columns=["verified_r22_prediction"])

    metadata = spatial._load_development_metadata()
    joined = joined.merge(metadata, on=["month", "ml_num"], how="left", validate="one_to_one")
    if joined.date_sold.isna().any() or joined.source_sold_price.isna().any():
        raise RuntimeError("R126 development rows failed safe-fold metadata alignment")
    if not np.array_equal(joined.sold_price.to_numpy(float), joined.source_sold_price.to_numpy(float)):
        raise RuntimeError("R126 OOF labels differ from exact safe-month source labels")
    return {
        "screen_result_path": screen_result_path,
        "screen_predictions_path": screen_result_path.parent / "paired_predictions.csv",
        "screen_manifest_path": screen_result_path.parent / "run_manifest.json",
        "screen_result": screen_result, "r22_screen": r22_screen,
        "screen_manifest": screen_manifest,
        "development_result_path": dev_result_path,
        "development_predictions_path": dev_result_path.parent / "paired_predictions.csv",
        "development_manifest_path": dev_result_path.parent / "run_manifest.json",
        "development_result": dev_result, "r22_development": r22_dev,
        "development_manifest": dev_manifest,
        "base_result_path": base_result_path,
        "base_predictions_path": base_predictions_path,
        "base_manifest_path": base_manifest_path,
        "base_result": base_result, "base_rows": joined,
        "base_manifest": base_manifest,
        "metadata": metadata,
    }


def _parent_artifact_hashes(inputs: dict) -> dict[str, str]:
    paths = (
        "screen_result_path", "screen_predictions_path", "screen_manifest_path",
        "development_result_path", "development_predictions_path", "development_manifest_path",
        "base_result_path", "base_predictions_path", "base_manifest_path",
    )
    return {key: _sha256(inputs[key]) for key in paths}


def register_card() -> dict:
    from experiments.bill_target74 import r04_ablation as identity

    existing = campaign.PROTOCOL["cards"].get(CARD)
    code_hashes = _code_hashes()
    if existing:
        if existing.get("status") != "registered":
            raise RuntimeError(f"{CARD} is already {existing.get('status')}; inspect its saved run")
        if existing.get("code_sha256") != code_hashes:
            raise RuntimeError("R136 source changed after preregistration")
        inputs = _verified_inputs()
        if existing.get("parent_artifact_hashes") != _parent_artifact_hashes(inputs):
            raise RuntimeError("R136 parent artifacts changed after preregistration")
        return existing

    inputs = _verified_inputs()
    source = identity._source_identity()
    if source.get("data_sha256") != campaign.PROTOCOL["data_sha256"]:
        raise RuntimeError("active data hash differs from the campaign registration")
    route_support = {
        month: {
            "rows": int(inputs["base_rows"].month.eq(month).sum()),
            "history_rows_before_cutoff": int(
                pd.to_datetime(inputs["base_rows"].date_sold).lt(pd.Timestamp(month + "-01")).sum()
            ),
        }
        for month in SCREEN_MONTHS
    }
    card = {
        "status": "registered",
        "stage": "fixed_two_month_screen_then_eight_month_development",
        "track": TRACK,
        "screen_months": list(SCREEN_MONTHS),
        "development_months": list(DEVELOPMENT_MONTHS),
        "expected_screen_calibration_fit_starts": sum(v["history_rows_before_cutoff"] > 0 for v in route_support.values()),
        "expected_development_calibration_fit_starts": max(0, len(DEVELOPMENT_MONTHS) - 1),
        "comparison_parent": "R22_train_fold_tfidf_remarks",
        "base_candidate": BASE_CARD,
        "parent_run_ids": {
            "r22_screen": inputs["screen_result"]["run_id"],
            "r22_development": inputs["development_result"]["run_id"],
            "r126_development": inputs["base_result"]["run_id"],
        },
        "parent_artifact_hashes": _parent_artifact_hashes(inputs),
        "method": (
            "Use R126's off-market cyclic listing-month OOF predictions as both the base and the "
            "strictly past residual history, and compare with hash-verified R22 predictions on the "
            "same safe rows. For each target month, within house type, compute a 12-month recency- "
            "and distance-weighted median log(sold_price/R126_prediction) among up to 50 earlier "
            "OOF homes within 1.5 km. Subtract the type-wide recency-weighted median; shrink by "
            "n_eff/(n_eff+25), clip to +/-0.03 log points, multiply the R126 price by exp(delta), "
            "and round to dollars. No history means no correction."
        ),
        "single_change": "Apply the fixed R93 spatial residual-difference correction to R126 predictions using only earlier R126 OOF rows",
        "parameters": {
            "maximum_distance_km": 1.5,
            "maximum_neighbors": 50,
            "minimum_neighbors": 15,
            "minimum_type_history_rows": 100,
            "distance_gaussian_bandwidth_km": 0.75,
            "recency_half_life_months": 12.0,
            "shrinkage_pseudocount": 25.0,
            "maximum_absolute_log_delta": 0.03,
        },
        "route_support": route_support,
        "screen_gates": {
            "relative_mdape_gain_min_percent": campaign.PROTOCOL["gates"]["screen_relative_mdape_improvement_percent"],
            "both_screen_months_improve": True,
            "mape_regression_max_percent": campaign.PROTOCOL["gates"]["screen_mape_regression_max_percent"],
            "paired_bootstrap_upper95_below_zero": True,
        },
        "development_gates": dict(campaign.PROTOCOL["gates"]),
        "target_mdape_strictly_below_percent": TARGET_MDAPE,
        "source_identity": source,
        "code_sha256": code_hashes,
        "data_sha256": campaign.PROTOCOL["data_sha256"],
        "reserved_months_labels_opened": False,
        "hidden_labels_opened": False,
    }
    campaign.PROTOCOL["cards"][CARD] = card
    _write_json(campaign.PROTOCOL_PATH, campaign.PROTOCOL)
    campaign._append({
        "event": "card_registered", "card": CARD,
        "expected_screen_calibration_fit_starts": card["expected_screen_calibration_fit_starts"],
        "expected_development_calibration_fit_starts": card["expected_development_calibration_fit_starts"],
        "data_sha256": card["data_sha256"], "code_sha256": code_hashes,
        "parent_artifact_hashes": card["parent_artifact_hashes"],
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    campaign._write_progress(
        "r136_registered",
        next_action="run the preregistered R136 off-market R126 spatial residual screen",
        r136_card=CARD,
    )
    return card


def _score(rows: pd.DataFrame, months: tuple[str, ...], stage: str) -> tuple[dict, dict]:
    from experiments.bill_target74 import r22_remarks_tfidf as r22

    frame = rows[["month", "ml_num", "sold_price", "parent_prediction", "candidate_prediction"]].copy()
    common = {
        "months": months,
        "target_mdape_strictly_below": TARGET_MDAPE,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }
    if stage == "screen":
        score = r22.score_development_confirmation(
            frame,
            **common,
            minimum_relative_gain_percent=campaign.PROTOCOL["gates"]["screen_relative_mdape_improvement_percent"],
            minimum_months_improved=len(months),
            maximum_worst_month_regression_percent=0.0,
            maximum_mape_regression_percent=campaign.PROTOCOL["gates"]["screen_mape_regression_max_percent"],
        )
        return score, {
            "relative_mdape_gain": score["relative_mdape_improvement_percent"] >= campaign.PROTOCOL["gates"]["screen_relative_mdape_improvement_percent"],
            "both_screen_months_improved": score["months_improved"] == len(months),
            "mape_guard": score["mape_relative_change_percent"] <= campaign.PROTOCOL["gates"]["screen_mape_regression_max_percent"],
            "paired_bootstrap_upper95_below_zero": score["bootstrap_mdape_difference_percent"]["upper95"] < 0.0,
        }
    score = r22.score_development_confirmation(frame, **common)
    return score, {
        "development_gates": bool(score["qualifies_development"]),
        "strict_mdape_target": bool(score["target_met"]),
    }


def run(stage: str) -> dict:
    from experiments.bill_target74 import r04_ablation as identity
    from experiments.bill_target74 import r93_spatial_oof_calibration as spatial

    card = campaign.PROTOCOL["cards"].get(CARD)
    needed = "registered" if stage == "screen" else "screen_passed"
    if not card or card.get("status") != needed:
        raise RuntimeError(f"R136 {stage} requires card status {needed}")
    if card.get("code_sha256") != _code_hashes():
        raise RuntimeError("R136 source or dependency hash changed after preregistration")
    inputs = _verified_inputs()
    if card.get("parent_artifact_hashes") != _parent_artifact_hashes(inputs):
        raise RuntimeError("R136 parent artifacts changed after preregistration")
    source = identity._source_identity()
    if source.get("data_sha256") != campaign.PROTOCOL["data_sha256"]:
        raise RuntimeError("active data identity differs from the preregistered R136 data")

    months = SCREEN_MONTHS if stage == "screen" else DEVELOPMENT_MONTHS
    run_id = ("r136s_" if stage == "screen" else "r136d_") + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = campaign.CAMPAIGN_ROOT / "batches" / run_id
    output.mkdir(parents=True, exist_ok=False)
    provenance = output / "provenance"
    provenance.mkdir()
    for key in ("screen_result_path", "screen_predictions_path", "screen_manifest_path",
                "development_result_path", "development_predictions_path", "development_manifest_path",
                "base_result_path", "base_predictions_path", "base_manifest_path"):
        shutil.copy2(inputs[key], provenance / (key + inputs[key].suffix))
    snapshots = {}
    code_hashes = _code_hashes()
    for index, path in enumerate(_source_files()):
        name = f"src_{index:02d}{path.suffix}.source"
        shutil.copy2(path, provenance / name)
        snapshots[_relative(path)] = name

    calibration_count = 0
    fit_details: list[dict] = []
    paired_parts: list[pd.DataFrame] = []
    route_details: dict[str, dict] = {}
    campaign._append({
        "event": "card_start", "card": CARD, "stage": stage, "run_id": run_id,
        "track": TRACK, "months": list(months),
        "expected_fit_starts": card[f"expected_{stage}_calibration_fit_starts"],
        "base_candidate": BASE_CARD,
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    base = inputs["base_rows"].copy()
    base["month"] = base.month.astype(str)
    base["ml_num"] = base.ml_num.astype(str)
    r22_screen = inputs["r22_screen"].copy()
    r22_screen["month"] = r22_screen.month.astype(str)
    r22_screen["ml_num"] = r22_screen.ml_num.astype(str)
    r22_dev = inputs["r22_development"].copy()
    r22_dev["month"] = r22_dev.month.astype(str)
    r22_dev["ml_num"] = r22_dev.ml_num.astype(str)
    r22_lookup = r22_screen if stage == "screen" else r22_dev
    r22_lookup = r22_lookup[["month", "ml_num", "sold_price", "candidate_prediction"]].rename(
        columns={"candidate_prediction": "verified_r22_prediction"}
    )

    try:
        for month in months:
            current = base.loc[base.month.eq(month)].copy().reset_index(drop=True)
            if current.empty:
                raise RuntimeError(f"R136 has no safe R126 rows for {month}")
            parent = r22_lookup.loc[r22_lookup.month.eq(month)].copy()
            current = current.merge(parent, on=["month", "ml_num", "sold_price"], how="left", validate="one_to_one")
            if current.verified_r22_prediction.isna().any() or len(current) != len(parent):
                raise RuntimeError(f"R136 R22 parent rows do not exactly align for {month}")
            if not np.array_equal(
                current.parent_prediction.to_numpy(float),
                current.verified_r22_prediction.to_numpy(float),
            ):
                raise RuntimeError(f"R136 saved R126 parent does not match direct R22 rows for {month}")
            cutoff = pd.Timestamp(month + "-01")
            history = base.loc[pd.to_datetime(base.date_sold).lt(cutoff), [
                "date_sold", "house_type_name", "latitude", "longitude", "sold_price", "candidate_prediction",
            ]].copy()
            query = current[[
                "house_type_name", "latitude", "longitude", "candidate_prediction",
            ]].copy()
            if {"sold_price", "date_sold"}.intersection(query.columns):
                raise RuntimeError("R136 query data must not contain current-fold outcomes or sale dates")
            correction = np.zeros(len(current), dtype=float)
            diagnostics = pd.DataFrame({
                "neighbor_count": np.zeros(len(current), dtype=int),
                "effective_neighbor_count": np.zeros(len(current), dtype=float),
                "correction_log": np.zeros(len(current), dtype=float),
            })
            if not history.empty:
                fit_id = f"{campaign.PROTOCOL['campaign_id']}:{CARD}:{stage}:{TRACK}:{month}:{run_id}"
                calibration_count += 1
                fit_started = time.perf_counter()
                campaign._append({
                    "event": "fit_start", "card": CARD, "stage": stage,
                    "run_id": run_id, "fit_id": fit_id, "track": TRACK,
                    "month": month, "component": "r126_spatial_residual_calibration",
                    "training_oof_rows": int(len(history)), "query_rows": int(len(query)),
                    "history_cutoff_exclusive": cutoff.isoformat(),
                    "data_sha256": source["data_sha256"],
                    "reserved_months_labels_opened": False, "hidden_labels_opened": False,
                })
                try:
                    if pd.to_datetime(history.date_sold).max() >= cutoff:
                        raise RuntimeError(f"R136 OOF history crosses the {month} cutoff")
                    correction, diagnostics = spatial._fit_local_corrections(history, query, cutoff)
                except Exception as exc:
                    campaign._append({
                        "event": "fit_complete", "card": CARD, "stage": stage,
                        "run_id": run_id, "fit_id": fit_id, "track": TRACK,
                        "month": month, "status": "failed",
                        "seconds": time.perf_counter() - fit_started,
                        "error": f"{type(exc).__name__}: {exc}",
                        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
                    })
                    raise
                detail = {
                    "fit_id": fit_id, "track": TRACK, "month": month,
                    "training_oof_rows": int(len(history)), "query_rows": int(len(query)),
                    "seconds": time.perf_counter() - fit_started,
                    "corrected_rows": int(np.count_nonzero(correction)),
                    "mean_abs_log_correction": float(np.mean(np.abs(correction))),
                    "max_abs_log_correction": float(np.max(np.abs(correction))),
                    "neighbor_count_min": int(diagnostics.neighbor_count.min()),
                    "neighbor_count_median": float(diagnostics.neighbor_count.median()),
                    "neighbor_count_max": int(diagnostics.neighbor_count.max()),
                }
                fit_details.append(detail)
                campaign._append({
                    "event": "fit_complete", "card": CARD, "stage": stage,
                    "run_id": run_id, "status": "complete", **detail,
                    "reserved_months_labels_opened": False, "hidden_labels_opened": False,
                })

            base_values = current.candidate_prediction.to_numpy(float)
            candidate_values = np.rint(base_values * np.exp(correction))
            if not np.isfinite(candidate_values).all() or (candidate_values <= 0).any():
                raise RuntimeError(f"R136 generated invalid predictions for {month}")
            paired = pd.DataFrame({
                "month": month, "track": TRACK,
                "ml_num": current.ml_num.astype(str),
                "sold_price": current.sold_price.to_numpy(float),
                "parent_prediction": current.parent_prediction.to_numpy(float),
                "base_prediction": base_values,
                "candidate_prediction": candidate_values,
            })
            paired_parts.append(paired)
            route_details[month] = {
                "rows": int(len(current)),
                "history_rows": int(len(history)),
                "history_cutoff_exclusive": cutoff.isoformat(),
                "base_prediction_is_r126": True,
                "r22_parent_aligned": True,
                "unchanged_from_base": int(np.count_nonzero(correction == 0.0)),
                "corrected_rows": int(np.count_nonzero(correction)),
                "parent_metrics": campaign._metrics(paired.sold_price, paired.parent_prediction),
                "base_metrics": campaign._metrics(paired.sold_price, paired.base_prediction),
                "candidate_metrics": campaign._metrics(paired.sold_price, paired.candidate_prediction),
            }
            campaign._append({
                "event": "screen_fold_complete" if stage == "screen" else "development_fold_complete",
                "card": CARD, "stage": stage, "run_id": run_id, "track": TRACK,
                "month": month, "rows": len(current), "history_rows": len(history),
                "history_cutoff_exclusive": cutoff.isoformat(),
                "reserved_months_labels_opened": False, "hidden_labels_opened": False,
            })
            pd.concat(paired_parts, ignore_index=True).to_csv(
                output / "partial_paired_predictions.csv", index=False,
            )
    except Exception as exc:
        failure = {
            "run_id": run_id, "card": CARD, "stage": stage,
            "error": f"{type(exc).__name__}: {exc}",
            "fit_starts": calibration_count, "fit_details": fit_details,
            "route_details": route_details,
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        }
        _write_json(output / "failure_checkpoint.json", failure)
        campaign._append({
            "event": "card_failed", "card": CARD, "stage": stage,
            "run_id": run_id, "fit_starts": calibration_count,
            "error": failure["error"],
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        campaign.PROTOCOL["cards"][CARD].update({
            "status": f"{stage}_failed_runtime", "run_id": run_id,
            "failure_path": _relative(output / "failure_checkpoint.json"),
            "fit_starts": calibration_count,
        })
        _write_json(campaign.PROTOCOL_PATH, campaign.PROTOCOL)
        campaign._write_progress(
            f"r136_{stage}_failed_runtime",
            next_action=f"audit R136 {stage} failure checkpoint before any retry",
            r136_run_id=run_id,
        )
        raise

    expected = card[f"expected_{stage}_calibration_fit_starts"]
    if calibration_count != expected:
        raise RuntimeError(f"R136 expected {expected} calibration fits, recorded {calibration_count}")
    paired = pd.concat(paired_parts, ignore_index=True)
    if paired.duplicated(["month", "track", "ml_num"]).any():
        raise RuntimeError("R136 produced duplicate paired prediction keys")
    paired_path = output / "paired_predictions.csv"
    paired.to_csv(paired_path, index=False)
    score, gate_checks = _score(paired, tuple(months), stage)
    passed = all(gate_checks.values()) if stage == "screen" else bool(
        score["qualifies_development"] and score["target_met"]
    )
    result = {
        "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
        "card": CARD, "stage": stage, "track": TRACK,
        "comparison_parent": "R22", "base_candidate": BASE_CARD,
        "months": list(months), "fit_starts": calibration_count,
        "tracks": {TRACK: score}, "gate_checks": gate_checks,
        "qualifies_by_track": {TRACK: passed},
        "calibration_fit_details": fit_details,
        "route_diagnostics": route_details,
        "data_sha256": source["data_sha256"],
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    }
    result_path = output / ("screen_results.json" if stage == "screen" else "development_results.json")
    _write_json(result_path, result)
    manifest = {
        "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
        "data_sha256": source["data_sha256"], "source_identity": source,
        "code_sha256": code_hashes, "source_snapshots": snapshots,
        "parent_artifact_hashes": card["parent_artifact_hashes"],
        "fit_starts": calibration_count, "fit_details": fit_details,
        "paired_predictions_sha256": _sha256(paired_path),
        "result_sha256": _sha256(result_path),
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    }
    _write_json(output / "run_manifest.json", manifest)
    report = [
        "# R136 off-market R126 spatial residual calibration", "",
        f"Run `{run_id}`; parent R22; base candidate R126.",
        f"Stage: {stage}; months: {', '.join(months)}; calibration starts: {calibration_count}.",
        f"Parent MdAPE: {score['parent']['mdape_percent']:.6f}%.",
        f"Candidate MdAPE: {score['candidate']['mdape_percent']:.6f}%.",
        f"Relative MdAPE gain: {score['relative_mdape_improvement_percent']:+.4f}%.",
        f"MAPE relative change: {score['mape_relative_change_percent']:+.4f}%.",
        f"Gate checks: `{json.dumps(gate_checks, ensure_ascii=False)}`.",
        "Every calibration query omitted the target month's outcome and sale date; history was strictly earlier OOF.",
        "Reserved-month and hidden labels were not opened.",
        f"Artifacts: `{_relative(output)}`.", "",
    ]
    (output / "research_report.md").write_text("\n".join(report), encoding="utf-8")
    campaign._append({
        "event": "card_complete", "card": CARD, "stage": stage,
        "run_id": run_id, "fit_starts": calibration_count,
        "result_path": _relative(result_path),
        "qualifies_by_track": {TRACK: passed},
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    card = campaign.PROTOCOL["cards"][CARD]
    card.update({
        "status": ("screen_passed" if passed else "screen_failed") if stage == "screen" else "development_complete",
        "run_id": run_id,
        f"{stage}_fit_starts": calibration_count,
        f"{stage}_result_path": _relative(result_path),
        f"{stage}_result_sha256": _sha256(result_path),
        f"{stage}_paired_predictions_path": _relative(paired_path),
        f"{stage}_paired_predictions_sha256": _sha256(paired_path),
        f"{stage}_manifest_path": _relative(output / "run_manifest.json"),
        f"{stage}_manifest_sha256": _sha256(output / "run_manifest.json"),
        f"{stage}_metrics": score,
        f"{stage}_gate_checks": gate_checks,
        "qualifies_screen": passed if stage == "screen" else card.get("qualifies_screen", False),
        "qualifies_development": passed if stage == "development" else card.get("qualifies_development", False),
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    _write_json(campaign.PROTOCOL_PATH, campaign.PROTOCOL)
    if stage == "screen":
        status = "r136_screen_passed" if passed else "r136_screen_failed"
        next_action = "R136 passed the fixed screen; run eight-month development" if passed else "R136 failed the fixed screen; retain current parents and select another route"
    else:
        status = "r136_development_passed" if passed else "r136_development_failed"
        next_action = "R136 met the strict target; run locked fixed-origin process" if passed else "R136 did not meet the strict target and development gates"
    campaign._write_progress(
        status, next_action=next_action,
        r136_run_id=run_id, r136_result=_relative(result_path),
        r136_calibration_fit_starts=calibration_count,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="R136 strict-past spatial correction over R126 off-market OOF")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--register-only", action="store_true")
    group.add_argument("--screen", action="store_true")
    group.add_argument("--development", action="store_true")
    args = parser.parse_args()
    if args.register_only:
        print(json.dumps(register_card(), indent=2, ensure_ascii=False))
        return
    result = run("screen" if args.screen else "development")
    score = result["tracks"][TRACK]
    print(
        f"R136 {result['stage']}: parent={score['parent']['mdape_percent']:.6f}% "
        f"candidate={score['candidate']['mdape_percent']:.6f}% "
        f"relative_gain={score['relative_mdape_improvement_percent']:+.4f}% "
        f"passed={result['qualifies_by_track'][TRACK]} run={result['run_id']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
