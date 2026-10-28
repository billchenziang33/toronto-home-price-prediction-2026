"""R59: resume R58 calibration after correcting a post-rounding bound check."""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor

from experiments.bill_target74 import r04_ablation as budget
from experiments.bill_target74 import r19_expert_weight as common
from experiments.bill_target74 import r22_remarks_tfidf as scoring
from experiments.bill_target74 import r58_past_log_residual_slope_calibration as r58
from experiments.bill_target74 import runner as campaign


CARD = "R59_past_log_residual_calibration_rounding_fix"
R58_RUN = "bill_target74_r58_past_log_residual_calibration_20261025T085200Z"
TRACKS = r58.TRACKS
MONTHS = r58.MONTHS
REUSED_FIT_KEY = ("off_market", "2023-07")
EXPECTED_REUSED_FITS = 1
EXPECTED_NEW_FITS = r58.EXPECTED_CALIBRATION_FITS - EXPECTED_REUSED_FITS


def _json_digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _r58_evidence() -> tuple[dict, dict, dict, dict, str]:
    events = [json.loads(line) for line in campaign.LEDGER.read_text(encoding="utf-8").splitlines()]
    selected = [
        row for row in events
        if row.get("card") == r58.CARD and row.get("run_id") == R58_RUN
    ]
    start = [row for row in selected if row.get("event") == "card_start"]
    fit_starts = [row for row in selected if row.get("event") == "fit_start"]
    fit_done = [row for row in selected if row.get("event") == "fit_complete"]
    failures = [row for row in selected if row.get("event") == "card_failed"]
    if len(start) != 1 or len(fit_starts) != 1 or len(fit_done) != 1 or len(failures) != 1:
        raise RuntimeError("R58 ledger does not contain the expected single-fit failure sequence")
    if fit_done[0].get("status") != "complete" or fit_starts[0].get("fit_id") != fit_done[0].get("fit_id"):
        raise RuntimeError("R58's reusable fit is not a matching completed fit")
    if (fit_done[0].get("track"), fit_done[0].get("month")) != REUSED_FIT_KEY:
        raise RuntimeError("R58 completed fit is not the preregistered off-market 2023-07 fit")
    if any(row.get("reserved_months_labels_opened") or row.get("hidden_labels_opened") for row in selected):
        raise RuntimeError("R58 ledger records restricted-label access")
    evidence = {"card_start": start[0], "fit_start": fit_starts[0],
                "fit_complete": fit_done[0], "card_failed": failures[0]}
    return start[0], fit_starts[0], fit_done[0], failures[0], _json_digest(evidence)


def _save_protocol() -> None:
    campaign.PROTOCOL_PATH.write_text(
        json.dumps(campaign.PROTOCOL, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def register_card() -> dict:
    existing = campaign.PROTOCOL["cards"].get(CARD)
    if existing:
        if existing.get("status") == "registered" and existing.get("source_sha256") == common._sha256(Path(__file__)):
            return existing
        raise RuntimeError(f"R59 is already {existing.get('status')}; inspect before rerunning")

    r35 = campaign.PROTOCOL["cards"].get("R35_listaware_detached_r22_text_expert")
    r52 = campaign.PROTOCOL["cards"].get("R52_r48_r50_strict_past_meta_blend")
    r58_card = campaign.PROTOCOL["cards"].get(r58.CARD)
    if not r35 or r35.get("status") != "fixed_origin_confirmation_passed":
        raise RuntimeError("R59 requires the verified R35 list-aware development candidate")
    if not r52 or r52.get("run_id") != r58.R52_RUN:
        raise RuntimeError("R59 requires the registered R52 off-market development candidate")
    if not r58_card or r58_card.get("status") != "registered":
        raise RuntimeError("R59 expects the failed R58 card to remain registered for audit")

    r58_start, old_fit_start, reused_fit, failure, evidence_sha256 = _r58_evidence()
    if "post-rounding" not in failure.get("error", "") and "outside its preregistered bound" not in failure.get("error", ""):
        raise RuntimeError("R58 failure does not match the diagnosed rounding-bound assertion")
    if old_fit_start.get("component") != "past_log_residual_median_regression":
        raise RuntimeError("R58 completed fit used an unexpected calibration component")

    source = budget._source_identity()
    parent_files = {
        "off_market": (campaign.ROOT / r58.R52_RESULT,
                       campaign.ROOT / r58.R52_RESULT.parent / "paired_predictions.csv"),
        "list_aware": (campaign.ROOT / r58.R35_DEV_RESULT,
                       campaign.ROOT / r58.R35_DEV_RESULT.parent / "paired_predictions.csv"),
    }
    parent_hashes = {
        track: {"result_sha256": common._sha256(paths[0]), "paired_predictions_sha256": common._sha256(paths[1])}
        for track, paths in parent_files.items()
    }
    if r58_start.get("input_result_sha256") != {k: v["result_sha256"] for k, v in parent_hashes.items()}:
        raise RuntimeError("R58 did not use the currently registered parent results")
    if r58_start.get("input_paired_sha256") != {k: v["paired_predictions_sha256"] for k, v in parent_hashes.items()}:
        raise RuntimeError("R58 did not use the currently registered parent predictions")

    registration = {
        "status": "registered",
        "tracks": list(TRACKS),
        "development_months": list(MONTHS),
        "expected_fit_starts": EXPECTED_NEW_FITS,
        "new_calibration_fit_starts": EXPECTED_NEW_FITS,
        "reused_completed_fit_starts": 0,
        "calibration_folds": r58.EXPECTED_CALIBRATION_FITS,
        "reused_calibration_folds": EXPECTED_REUSED_FITS,
        "parents": {
            "off_market": "R52 strict-past meta-blend candidate; same rows and predictions as R58",
            "list_aware": "R35 Detached-text candidate; same rows and predictions as R58",
        },
        "parent_run_ids": {"off_market": r58.R52_RUN, "list_aware": r58.R35_DEV_RUN},
        "parent_artifact_hashes": parent_hashes,
        "method": (
            "Resume the preregistered R58 strict-past median regression on log(sold_price / base_prediction) "
            "versus centered log(base_prediction), using only earlier safe development months. Reuse, without "
            "refitting, the single completed R58 off_market/2023-07 QuantileRegressor fit after verifying its "
            "append-only ledger events and parent hashes. Fit only the remaining 13 calibration folds with "
            "quantile=0.5, alpha=0.01, solver=highs. Clip the applied correction before exponentiation to "
            "[-0.05,+0.05], then round predictions to the nearest dollar. Validate the bound on the clipped "
            "pre-rounding correction; dollar rounding can move the final ratio by a few 1e-7 log points."
        ),
        "calibration": {
            "estimator": "sklearn.linear_model.QuantileRegressor",
            "quantile": 0.5, "alpha": r58.QUANTILE_ALPHA, "solver": "highs",
            "minimum_past_rows": r58.MIN_PAST_ROWS,
            "max_abs_log_correction_before_rounding": r58.MAX_ABS_LOG_CORRECTION,
            "prediction_rounding": "nearest dollar after bounded correction",
        },
        "r58_failure": {
            "run_id": R58_RUN,
            "failure": failure["error"],
            "completed_fit_id": reused_fit["fit_id"],
            "completed_fit_track_month": [reused_fit["track"], reused_fit["month"]],
            "ledger_event_bundle_sha256": evidence_sha256,
            "failure_checkpoint_underreported_fit_starts": True,
            "fit_ledger_is_authoritative": True,
        },
        "development_gate": {
            "minimum_relative_mdape_improvement_percent": 0.5,
            "minimum_months_improved": 6,
            "maximum_worst_month_regression_percent": 5.0,
            "maximum_mape_regression_percent": 1.0,
            "paired_bootstrap_replicates": r58.BOOTSTRAP_REPLICATES,
            "paired_bootstrap_seed": r58.BOOTSTRAP_SEED,
            "paired_bootstrap_upper95_below_zero": True,
            "strict_target_mdape_percent": {"off_market": 7.0, "list_aware": 4.0},
        },
        "policy": (
            "Read only registered safe development predictions for 2023-03 through 2025-03. Each calibration "
            "fold uses earlier months only. Never read 2025-05/08 or hidden labels. Preserve R22/R35 as validated "
            "parents until a separate fixed-origin confirmation passes."
        ),
        "source_sha256": common._sha256(Path(__file__)),
        "data_sha256": source["data_sha256"],
        "reserve_validation_eligible": False,
        "reserved_months_labels_opened": False,
        "hidden_labels_opened": False,
    }
    campaign.PROTOCOL["cards"][CARD] = registration
    _save_protocol()
    campaign._append({
        "event": "card_registered", "card": CARD,
        "tracks": list(TRACKS), "development_months": list(MONTHS),
        "expected_fit_starts": EXPECTED_NEW_FITS,
        "reused_fit_id": reused_fit["fit_id"],
        "source_sha256": registration["source_sha256"], "data_sha256": source["data_sha256"],
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    campaign._write_progress(
        "r59_registered",
        next_action="resume R58's preregistered past-only calibration, reusing its completed fit and correcting the post-rounding assertion",
        r59_card=CARD,
    )
    return registration


def _load_r58_reuse(card: dict) -> tuple[dict, str]:
    r58_start, _, fit, failure, evidence_sha256 = _r58_evidence()
    expected = card["r58_failure"]
    if evidence_sha256 != expected["ledger_event_bundle_sha256"]:
        raise RuntimeError("R58 append-only ledger evidence changed after R59 registration")
    if fit["fit_id"] != expected["completed_fit_id"]:
        raise RuntimeError("R58's completed fit ID differs from the R59 registration")
    if failure["error"] != expected["failure"]:
        raise RuntimeError("R58's recorded failure differs from the R59 registration")
    if r58_start.get("input_result_sha256") != {k: v["result_sha256"] for k, v in card["parent_artifact_hashes"].items()}:
        raise RuntimeError("R58 result artifact hashes differ from the R59 registration")
    if r58_start.get("input_paired_sha256") != {k: v["paired_predictions_sha256"] for k, v in card["parent_artifact_hashes"].items()}:
        raise RuntimeError("R58 paired-prediction hashes differ from the R59 registration")
    return fit, evidence_sha256


def _calibrate_track(
    *, run_id: str, track: str, rows: pd.DataFrame, reused_fit: dict,
) -> tuple[pd.DataFrame, list[dict], list[dict]]:
    ordered = rows.copy()
    ordered["month_period"] = pd.PeriodIndex(ordered.month.astype(str), freq="M")
    output_parts, new_fits, reused_fits = [], [], []
    months_done = []

    for month in MONTHS:
        target_period = pd.Period(month, freq="M")
        current = ordered.loc[ordered.month.eq(month)].copy()
        past = ordered.loc[ordered.month_period < target_period].copy()
        base = current.candidate_prediction.to_numpy(dtype=float)
        intercept = slope = center = 0.0
        calibrated = False
        fit_id = None
        fit_source = "none"
        raw_correction = np.zeros(len(base), dtype=float)
        applied_correction = np.zeros(len(base), dtype=float)

        if len(past) >= r58.MIN_PAST_ROWS:
            if (track, month) == REUSED_FIT_KEY:
                intercept = float(reused_fit["intercept"])
                slope = float(reused_fit["slope"])
                center = float(reused_fit["prediction_center_log_price"])
                fit_id = reused_fit["fit_id"]
                fit_source = "reused_r58_completed_fit"
                expected_months = sorted(past.month.astype(str).unique())
                if (reused_fit.get("training_rows") != len(past)
                        or reused_fit.get("training_months") != expected_months
                        or reused_fit.get("training_cutoff_exclusive") != target_period.start_time.isoformat()):
                    raise RuntimeError("R58 reusable fit training window does not match the strict-past fold")
                reused_fits.append({
                    "fit_id": fit_id, "track": track, "month": month,
                    "training_rows": len(past), "source_run_id": R58_RUN,
                    "source_card": r58.CARD, "status": "reused_without_refit",
                })
            else:
                train_prediction = past.candidate_prediction.to_numpy(dtype=float)
                train_actual = past.sold_price.to_numpy(dtype=float)
                center = float(np.median(np.log(train_prediction)))
                x = (np.log(train_prediction) - center).reshape(-1, 1)
                residual = np.log(train_actual / train_prediction)
                fit_id = f"{campaign.PROTOCOL['campaign_id']}:{CARD}:calibration:{track}:{month}:{run_id}"
                training_months = sorted(past.month.astype(str).unique())
                cutoff = target_period.start_time
                campaign._append({
                    "event": "fit_start", "card": CARD, "stage": "calibration",
                    "run_id": run_id, "fit_id": fit_id, "track": track, "month": month,
                    "component": "past_log_residual_median_regression",
                    "training_rows": int(len(past)), "evaluation_rows": int(len(current)),
                    "training_months": training_months,
                    "training_cutoff_exclusive": cutoff.isoformat(),
                    "quantile": 0.5, "alpha": r58.QUANTILE_ALPHA,
                    "data_sha256": campaign.PROTOCOL["data_sha256"],
                    "reserved_months_labels_opened": False, "hidden_labels_opened": False,
                })
                started = time.perf_counter()
                try:
                    model = QuantileRegressor(
                        quantile=0.5, alpha=r58.QUANTILE_ALPHA,
                        fit_intercept=True, solver="highs",
                    )
                    model.fit(x, residual)
                    intercept, slope = float(model.intercept_), float(model.coef_[0])
                    detail = {
                        "fit_id": fit_id, "track": track, "month": month,
                        "training_rows": int(len(past)), "evaluation_rows": int(len(current)),
                        "training_months": training_months,
                        "training_cutoff_exclusive": cutoff.isoformat(),
                        "quantile": 0.5, "alpha": r58.QUANTILE_ALPHA, "solver": "highs",
                        "intercept": intercept, "slope": slope,
                        "prediction_center_log_price": center,
                        "max_abs_log_correction_before_rounding": r58.MAX_ABS_LOG_CORRECTION,
                        "seconds": time.perf_counter() - started, "status": "complete",
                    }
                    new_fits.append(detail)
                    campaign._append({
                        "event": "fit_complete", "card": CARD, "stage": "calibration",
                        "run_id": run_id, **detail,
                        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
                    })
                except Exception as exc:
                    campaign._append({
                        "event": "fit_complete", "card": CARD, "stage": "calibration",
                        "run_id": run_id, "fit_id": fit_id, "track": track, "month": month,
                        "status": "failed", "seconds": time.perf_counter() - started,
                        "error": f"{type(exc).__name__}: {exc}",
                    })
                    raise
                fit_source = "new_r59_fit"

            raw_correction = intercept + slope * (np.log(base) - center)
            applied_correction = np.clip(
                raw_correction, -r58.MAX_ABS_LOG_CORRECTION, r58.MAX_ABS_LOG_CORRECTION
            )
            if not np.isfinite(applied_correction).all() or (
                np.max(np.abs(applied_correction)) > r58.MAX_ABS_LOG_CORRECTION + 1e-12
            ):
                raise RuntimeError("R59 clipped pre-rounding log correction exceeds its registered bound")
            calibrated_predictions = np.round(base * np.exp(applied_correction))
            if not np.isfinite(calibrated_predictions).all() or (calibrated_predictions <= 0).any():
                raise ValueError("R59 calibration produced invalid predictions")
            calibrated = True
        else:
            calibrated_predictions = base.copy()

        current["base_prediction"] = base
        current["candidate_prediction"] = calibrated_predictions
        current["calibration_intercept"] = intercept
        current["calibration_slope"] = slope
        current["calibration_center_log_price"] = center
        current["calibration_training_rows"] = int(len(past))
        current["calibration_fit_id"] = fit_id
        current["calibration_fit_source"] = fit_source
        current["calibration_applied"] = calibrated
        current["max_abs_applied_log_correction_before_rounding"] = (
            float(np.max(np.abs(applied_correction))) if len(applied_correction) else 0.0
        )
        current["max_abs_rounded_log_delta"] = (
            float(np.max(np.abs(np.log(calibrated_predictions / base)))) if calibrated else 0.0
        )
        output_parts.append(current.drop(columns="month_period"))
        months_done.append(month)

    if months_done != list(MONTHS):
        raise RuntimeError(f"R59 evaluated unexpected months for {track}: {months_done}")
    return pd.concat(output_parts, ignore_index=True), new_fits, reused_fits


def run() -> dict:
    card = campaign.PROTOCOL["cards"].get(CARD)
    if not card or card.get("status") != "registered":
        raise RuntimeError("R59 must be preregistered before reading development predictions")
    if card.get("source_sha256") != common._sha256(Path(__file__)):
        raise RuntimeError("R59 source changed after preregistration")
    source = budget._source_identity()
    if source["data_sha256"] != card["data_sha256"]:
        raise RuntimeError("R59 source data hash differs from the preregistered identity")
    reused_fit, reuse_evidence_sha256 = _load_r58_reuse(card)

    inputs = {
        "off_market": r58._verified_input(campaign, r58.R52_RESULT, r58.R52_RUN, "off_market"),
        "list_aware": r58._verified_input(campaign, r58.R35_DEV_RESULT, r58.R35_DEV_RUN, "list_aware"),
    }
    for track, item in inputs.items():
        expected = card["parent_artifact_hashes"][track]
        if common._sha256(item[1]) != expected["result_sha256"] or common._sha256(item[2]) != expected["paired_predictions_sha256"]:
            raise RuntimeError(f"R59 {track} parent hashes changed after registration")

    run_id = "bill_target74_r59_past_log_residual_rounding_fix_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = campaign.CAMPAIGN_ROOT / "batches" / run_id
    output.mkdir(parents=True, exist_ok=False)
    provenance = output / "provenance"
    provenance.mkdir()
    for track, (_, result_path, paired_path, manifest, _) in inputs.items():
        prefix = "r52" if track == "off_market" else "r35_development"
        shutil.copy2(result_path, provenance / f"{prefix}_development_results.json")
        shutil.copy2(paired_path, provenance / f"{prefix}_development_paired_predictions.csv")
        shutil.copy2(result_path.parent / "run_manifest.json", provenance / f"{prefix}_development_run_manifest.json")
    reuse_snapshot = {
        "run_id": R58_RUN, "card": r58.CARD, "fit": reused_fit,
        "ledger_event_bundle_sha256": reuse_evidence_sha256,
        "reuse_policy": "reuse completed coefficients; do not start a duplicate fit",
    }
    common._write_json(provenance / "r58_reused_fit_evidence.json", reuse_snapshot)

    campaign._append({
        "event": "card_start", "card": CARD, "run_id": run_id,
        "tracks": list(TRACKS), "development_months": list(MONTHS),
        "expected_fit_starts": EXPECTED_NEW_FITS,
        "new_calibration_fit_starts": EXPECTED_NEW_FITS,
        "reused_calibration_fit_starts": 0,
        "reused_calibration_folds": EXPECTED_REUSED_FITS,
        "reused_fit_id": reused_fit["fit_id"],
        "r58_ledger_event_bundle_sha256": reuse_evidence_sha256,
        "parent_run_ids": card["parent_run_ids"],
        "parent_artifact_hashes": card["parent_artifact_hashes"],
        "data_sha256": source["data_sha256"],
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    campaign._append({
        "event": "fit_reuse", "card": CARD, "run_id": run_id,
        "fit_id": reused_fit["fit_id"], "source_card": r58.CARD,
        "source_run_id": R58_RUN, "track": reused_fit["track"], "month": reused_fit["month"],
        "ledger_event_bundle_sha256": reuse_evidence_sha256,
        "fit_starts": 0, "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })

    new_fits, reused_fits, paired_parts, tracks = [], [], [], {}
    try:
        for track in TRACKS:
            rows = inputs[track][4]
            corrected, fits, reused = _calibrate_track(
                run_id=run_id, track=track, rows=rows, reused_fit=reused_fit,
            )
            new_fits.extend(fits)
            reused_fits.extend(reused)
            comparison = corrected[[
                "month", "track", "ml_num", "sold_price", "base_prediction", "candidate_prediction",
                "calibration_intercept", "calibration_slope", "calibration_center_log_price",
                "calibration_training_rows", "calibration_fit_id", "calibration_fit_source",
                "calibration_applied", "max_abs_applied_log_correction_before_rounding",
                "max_abs_rounded_log_delta",
            ]].copy().rename(columns={"base_prediction": "parent_prediction"})
            score = scoring.score_development_confirmation(
                comparison, months=MONTHS,
                target_mdape_strictly_below=7.0 if track == "off_market" else 4.0,
                bootstrap_replicates=r58.BOOTSTRAP_REPLICATES,
                bootstrap_seed=r58.BOOTSTRAP_SEED,
                minimum_relative_gain_percent=0.5,
                minimum_months_improved=6,
                maximum_worst_month_regression_percent=5.0,
                maximum_mape_regression_percent=1.0,
            )
            tracks[track] = score
            paired_parts.append(comparison)
    except Exception as exc:
        common._write_json(output / "failure_checkpoint.json", {
            "run_id": run_id, "card": CARD,
            "error": f"{type(exc).__name__}: {exc}", "fit_starts": len(new_fits),
            "new_fit_details": new_fits, "reused_fit_details": reused_fits,
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        card.update({"status": "failed", "run_id": run_id, "fit_starts": len(new_fits),
                     "error": f"{type(exc).__name__}: {exc}"})
        _save_protocol()
        campaign._append({
            "event": "card_failed", "card": CARD, "run_id": run_id,
            "fit_starts": len(new_fits), "reused_fit_starts": 0,
            "error": f"{type(exc).__name__}: {exc}",
        })
        campaign._write_progress(
            "r59_development_failed", next_action="inspect R59 checkpoint and ledger before a separate preregistered follow-up",
            r59_failure=str((output / "failure_checkpoint.json").relative_to(campaign.ROOT)).replace("\\", "/"),
        )
        raise

    if len(new_fits) != EXPECTED_NEW_FITS or len(reused_fits) != EXPECTED_REUSED_FITS:
        raise RuntimeError(
            f"R59 expected {EXPECTED_NEW_FITS} new fits plus {EXPECTED_REUSED_FITS} reused fit; "
            f"got {len(new_fits)} new and {len(reused_fits)} reused"
        )
    paired = pd.concat(paired_parts, ignore_index=True)
    paired_path = output / "paired_predictions.csv"
    paired.to_csv(paired_path, index=False)
    result = {
        "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id, "card": CARD,
        "parent_run_ids": card["parent_run_ids"], "development_months": list(MONTHS),
        "fit_starts": len(new_fits), "new_calibration_fit_starts": len(new_fits),
        "reused_calibration_fit_starts": 0, "reused_calibration_folds": len(reused_fits),
        "fit_details": new_fits, "reused_fit_details": reused_fits,
        "tracks": tracks, "calibration": card["calibration"],
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        "reserve_validation_eligible": False, "data_sha256": source["data_sha256"],
    }
    result_path = output / "development_results.json"
    common._write_json(result_path, result)

    source_files = list(dict.fromkeys([*r58._source_files(campaign), Path(__file__)]))
    code_hashes = {str(p.relative_to(campaign.ROOT)).replace("\\", "/"): common._sha256(p) for p in source_files}
    for index, path in enumerate(source_files):
        shutil.copy2(path, provenance / f"src_{index:02d}_{path.name}.source")
    manifest = {
        "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
        "source_identity": source, "code_sha256": code_hashes,
        "data_sha256": source["data_sha256"], "fit_starts": len(new_fits),
        "new_fit_details": new_fits, "reused_fit_details": reused_fits,
        "reused_fit_ledger_event_bundle_sha256": reuse_evidence_sha256,
        "parent_artifacts": {
            track: {"run_id": item[0]["run_id"], "result_sha256": common._sha256(item[1]),
                    "paired_predictions_sha256": common._sha256(item[2])}
            for track, item in inputs.items()
        },
        "result_sha256": common._sha256(result_path),
        "paired_predictions_sha256": common._sha256(paired_path),
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    }
    manifest_path = output / "run_manifest.json"
    common._write_json(manifest_path, manifest)

    lines = [
        "# R59: R58 calibration resumed with the pre-rounding bound check", "",
        f"Campaign: `{result['campaign_id']}`; run: `{run_id}`.",
        f"New fit starts: {len(new_fits)}; reused R58 fit starts: 0; reused calibration folds: {len(reused_fits)}.",
        "The ±0.05 bound is checked on the clipped correction before exponentiation and dollar rounding.",
        "The prior R58 2023-07 coefficients were verified in the append-only ledger and reused without refitting.",
        "No 2025-05/08 or hidden labels were accessed.", "",
        "| Track | Base MdAPE | Candidate MdAPE | Relative gain | MAPE change | Months improved | Bootstrap upper95 (pp) | Goal met | Gate |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for track, score in tracks.items():
        lines.append(
            f"| {track} | {score['parent']['mdape_percent']:.5f}% | "
            f"{score['candidate']['mdape_percent']:.5f}% | "
            f"{score['relative_mdape_improvement_percent']:+.3f}% | "
            f"{score['mape_relative_change_percent']:+.3f}% | {score['months_improved']}/8 | "
            f"{score['bootstrap_mdape_difference_percent']['upper95']:+.5f} | "
            f"{'yes' if score['target_met'] else 'no'} | "
            f"{'pass' if score['qualifies_development'] else 'fail'} |"
        )
    (output / "research_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    qualifies = {track: score["qualifies_development"] for track, score in tracks.items()}
    card.update({
        "status": "development_passed" if any(qualifies.values()) else "development_failed",
        "run_id": run_id, "fit_starts": len(new_fits),
        "new_calibration_fit_starts": len(new_fits), "reused_calibration_folds": len(reused_fits),
        "result_path": str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        "paired_predictions_path": str(paired_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        "manifest_path": str(manifest_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        "qualifies_by_track": qualifies, "development_metrics": tracks,
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    _save_protocol()
    campaign._append({
        "event": "card_complete", "card": CARD, "run_id": run_id,
        "fit_starts": len(new_fits), "reused_fit_starts": 0,
        "reused_calibration_folds": len(reused_fits), "result_path": card["result_path"],
        "qualifies_by_track": qualifies, "development_metrics": tracks,
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    passing = [track for track, passed in qualifies.items() if passed]
    status = "r59_development_passed" if passing else "r59_development_failed"
    next_action = (
        f"prepare a separate safe fixed-origin confirmation for R59 tracks {passing}; retain validated parents until it passes"
        if passing else "R59 failed development gates; retain R22/R35 and preregister a distinct hypothesis"
    )
    campaign._write_progress(
        status, next_action=next_action, r59_run_id=run_id,
        r59_result=card["result_path"], r59_fit_starts=len(new_fits),
        r59_reused_calibration_folds=len(reused_fits),
    )
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Register or run R59 past-only calibration continuation")
    parser.add_argument("--register", action="store_true", help="register the frozen R59 experiment")
    args = parser.parse_args()
    output = register_card() if args.register else run()
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
