"""R58: strict-past log-residual median calibration of the current track candidates."""
from __future__ import annotations

import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor

from experiments.bill_target74.r19_expert_weight import _sha256, _write_json


CARD = "R58_past_log_residual_slope_calibration"
TRACKS = ("off_market", "list_aware")
MONTHS = ("2023-03", "2023-07", "2023-11", "2024-03",
          "2024-06", "2024-09", "2025-01", "2025-03")
R52_RUN = "bill_target74_r52_r48_r50_adaptive_20261025T074051Z"
R35_DEV_RUN = "bill_target74_r35_house_type_text_development_20261025T040058Z"
R52_RESULT = Path("outputs/bill_target74/bill_target74_unbounded_20261024/batches") / R52_RUN / "development_results.json"
R35_DEV_RESULT = Path("outputs/bill_target74/bill_target74_unbounded_20261024/batches") / R35_DEV_RUN / "development_results.json"
MIN_PAST_ROWS = 1000
QUANTILE_ALPHA = 0.01
MAX_ABS_LOG_CORRECTION = 0.05
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20261026
EXPECTED_CALIBRATION_FITS = (len(MONTHS) - 1) * len(TRACKS)


def _source_files(campaign) -> list[Path]:
    return [
        campaign.PROTOCOL_PATH,
        Path(__file__),
        campaign.PACKAGE / "r04_ablation.py",
        campaign.PACKAGE / "r19_expert_weight.py",
        campaign.PACKAGE / "r22_remarks_tfidf.py",
        campaign.PACKAGE / "r52_r48_r50_adaptive_meta_blend.py",
        campaign.PACKAGE / "runner.py",
        campaign.ROOT / "experiments/bill_greedy/runner.py",
        campaign.ROOT / "experiments/model_optimization_v2/learners.py",
        campaign.ROOT / str(R52_RESULT).replace("\\", "/"),
        campaign.ROOT / str(R35_DEV_RESULT).replace("\\", "/"),
    ]


def register_card() -> dict:
    from experiments.bill_target74 import r04_ablation as budget
    from experiments.bill_target74 import runner as campaign

    existing = campaign.PROTOCOL["cards"].get(CARD)
    if existing:
        if existing.get("status") == "registered" and existing.get("source_sha256") == _sha256(Path(__file__)):
            return existing
        raise RuntimeError(f"R58 is already {existing.get('status')}; inspect before rerunning")

    r35 = campaign.PROTOCOL["cards"].get("R35_listaware_detached_r22_text_expert")
    r52 = campaign.PROTOCOL["cards"].get("R52_r48_r50_strict_past_meta_blend")
    if not r35 or r35.get("status") != "fixed_origin_confirmation_passed":
        raise RuntimeError("R58 requires the verified R35 list-aware development candidate")
    if not r52 or r52.get("status") != "development_failed" or r52.get("run_id") != R52_RUN:
        raise RuntimeError("R58 requires the recorded R52 best experimental off-market candidate")

    source = budget._source_identity()
    registration = {
        "status": "registered",
        "tracks": list(TRACKS),
        "development_months": list(MONTHS),
        "expected_fit_starts": EXPECTED_CALIBRATION_FITS,
        "base_model_fit_starts": 0,
        "calibration_fit_starts": EXPECTED_CALIBRATION_FITS,
        "parents": {
            "off_market": "R52 strict-past meta-blend candidate on identical safe development rows",
            "list_aware": "R35 Detached-text specialist candidate on identical safe development rows",
        },
        "parent_run_ids": {"off_market": R52_RUN, "list_aware": R35_DEV_RUN},
        "method": (
            "For each track and each development month after the first, fit a univariate median "
            "regression of log(sold_price / base_prediction) on centered log(base_prediction), "
            "using only earlier registered development months. Use sklearn QuantileRegressor "
            "with quantile=0.5, alpha=0.01, solver=highs; center at the past median log prediction; "
            "clip the applied log correction to [-0.05, +0.05], exponentiate, then round to the "
            "nearest dollar. The first month and any target with fewer than 1,000 past rows use the "
            "base prediction unchanged. Fit separate calibrators for the two tracks."
        ),
        "difference_from_prior_work": (
            "R12 used discrete house-type × prediction-quartile median corrections on the older "
            "G19/G01 parents. R47-R54 selected model-blend weights using past OOF. R58 instead "
            "fits one continuous, regularized median slope to the current R52/R35 log residuals, "
            "with a strict past-month cutoff and a fixed correction bound. No base learner is retrained."
        ),
        "calibration": {
            "estimator": "sklearn.linear_model.QuantileRegressor",
            "quantile": 0.5,
            "alpha": QUANTILE_ALPHA,
            "solver": "highs",
            "minimum_past_rows": MIN_PAST_ROWS,
            "max_abs_log_correction": MAX_ABS_LOG_CORRECTION,
        },
        "development_gate": {
            "minimum_relative_mdape_improvement_percent": 0.5,
            "minimum_months_improved": 6,
            "maximum_worst_month_regression_percent": 5.0,
            "maximum_mape_regression_percent": 1.0,
            "paired_bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "paired_bootstrap_seed": BOOTSTRAP_SEED,
            "paired_bootstrap_upper95_below_zero": True,
            "strict_target_mdape_percent": {"off_market": 7.0, "list_aware": 4.0},
        },
        "policy": (
            "Verify source, result, manifest, paired-row, and data hashes before scoring. For each "
            "month use only earlier safe OOF rows to fit its calibration; do not use current-month "
            "labels for correction selection. Read only 2023-03 through 2025-03 registered safe "
            "development predictions. Never read 2025-05/08 or hidden labels. This reused development "
            "window is not an independent test. Keep R22/R35 as validated parents unless later "
            "fixed-origin confirmation also passes."
        ),
        "source_sha256": _sha256(Path(__file__)),
        "data_sha256": source["data_sha256"],
        "reserve_validation_eligible": False,
        "reserved_months_labels_opened": False,
        "hidden_labels_opened": False,
    }
    campaign.PROTOCOL["cards"][CARD] = registration
    campaign.PROTOCOL_PATH.write_text(
        json.dumps(campaign.PROTOCOL, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    campaign._append({
        "event": "card_registered", "card": CARD,
        "tracks": list(TRACKS), "development_months": list(MONTHS),
        "expected_fit_starts": EXPECTED_CALIBRATION_FITS,
        "source_sha256": registration["source_sha256"], "data_sha256": source["data_sha256"],
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    campaign._write_progress(
        "r58_registered",
        next_action="verify R52/R35 development artifacts and run the preregistered past-only median residual calibration",
        r58_card=CARD,
    )
    return registration


def _verified_input(campaign, result_relative: Path, expected_run: str, expected_track: str):
    result_path = campaign.ROOT / result_relative
    paired_path = result_path.parent / "paired_predictions.csv"
    manifest_path = result_path.parent / "run_manifest.json"
    if not all(path.is_file() for path in (result_path, paired_path, manifest_path)):
        raise FileNotFoundError(f"missing {expected_track} result, paired predictions, or run manifest")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (result.get("run_id") != expected_run or manifest.get("run_id") != expected_run
            or result.get("campaign_id") != campaign.PROTOCOL["campaign_id"]
            or manifest.get("campaign_id") != campaign.PROTOCOL["campaign_id"]):
        raise RuntimeError(f"{expected_track} inputs do not match the preregistered run")
    source_data = manifest.get("data_sha256") or manifest.get("source_identity", {}).get("data_sha256")
    if (manifest.get("result_sha256") != _sha256(result_path)
            or manifest.get("paired_predictions_sha256") != _sha256(paired_path)
            or source_data != campaign.PROTOCOL["data_sha256"]):
        raise RuntimeError(f"{expected_track} input hashes or source data identity do not match")
    if result.get("reserved_months_labels_opened") or result.get("hidden_labels_opened"):
        raise RuntimeError(f"{expected_track} input records restricted-label access")
    rows = pd.read_csv(paired_path, dtype={"ml_num": str})
    required = {"month", "track", "ml_num", "sold_price", "candidate_prediction"}
    if missing := required - set(rows.columns):
        raise ValueError(f"{expected_track} paired predictions lack {sorted(missing)}")
    rows = rows.loc[rows.track.eq(expected_track) & rows.month.isin(MONTHS)].copy()
    if rows.duplicated(["month", "track", "ml_num"]).any():
        raise RuntimeError(f"{expected_track} rows are not unique by month and listing")
    if set(rows.month.astype(str).unique()) != set(MONTHS):
        raise RuntimeError(f"{expected_track} rows do not exactly cover the registered safe months")
    values = rows[["sold_price", "candidate_prediction"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0.0).any():
        raise ValueError(f"{expected_track} inputs include invalid sale prices or predictions")
    return result, result_path, paired_path, manifest, rows


def _calibrate_track(
    *, campaign, run_id: str, track: str, rows: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict]]:
    ordered = rows.copy()
    ordered["month_period"] = pd.PeriodIndex(ordered.month.astype(str), freq="M")
    output_parts = []
    fits = []
    months_done = []
    for month in MONTHS:
        target_period = pd.Period(month, freq="M")
        current = ordered.loc[ordered.month.eq(month)].copy()
        past = ordered.loc[ordered.month_period < target_period].copy()
        base = current.candidate_prediction.to_numpy(dtype=float)
        correction_intercept = 0.0
        correction_slope = 0.0
        center = 0.0
        calibrated = False
        fit_id = None

        if len(past) >= MIN_PAST_ROWS:
            train_prediction = past.candidate_prediction.to_numpy(dtype=float)
            train_actual = past.sold_price.to_numpy(dtype=float)
            x_center = float(np.median(np.log(train_prediction)))
            x = (np.log(train_prediction) - x_center).reshape(-1, 1)
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
                "quantile": 0.5, "alpha": QUANTILE_ALPHA,
                "data_sha256": campaign.PROTOCOL["data_sha256"],
                "reserved_months_labels_opened": False, "hidden_labels_opened": False,
            })
            started = time.perf_counter()
            try:
                model = QuantileRegressor(
                    quantile=0.5, alpha=QUANTILE_ALPHA, fit_intercept=True, solver="highs"
                )
                model.fit(x, residual)
                correction_intercept = float(model.intercept_)
                correction_slope = float(model.coef_[0])
                center = x_center
                raw_correction = correction_intercept + correction_slope * (
                    np.log(base) - center
                )
                applied_correction = np.clip(
                    raw_correction, -MAX_ABS_LOG_CORRECTION, MAX_ABS_LOG_CORRECTION
                )
                calibrated_predictions = np.round(base * np.exp(applied_correction))
                if not np.isfinite(calibrated_predictions).all() or (calibrated_predictions <= 0).any():
                    raise ValueError("calibration produced invalid predictions")
                calibrated = True
                detail = {
                    "fit_id": fit_id, "track": track, "month": month,
                    "training_rows": int(len(past)), "evaluation_rows": int(len(current)),
                    "training_months": training_months, "training_cutoff_exclusive": cutoff.isoformat(),
                    "quantile": 0.5, "alpha": QUANTILE_ALPHA, "solver": "highs",
                    "intercept": correction_intercept, "slope": correction_slope,
                    "prediction_center_log_price": center,
                    "max_abs_log_correction": MAX_ABS_LOG_CORRECTION,
                    "seconds": time.perf_counter() - started, "status": "complete",
                }
                fits.append(detail)
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
        else:
            calibrated_predictions = base.copy()

        current["base_prediction"] = base
        current["candidate_prediction"] = calibrated_predictions
        current["calibration_intercept"] = correction_intercept
        current["calibration_slope"] = correction_slope
        current["calibration_center_log_price"] = center
        current["calibration_training_rows"] = int(len(past))
        current["calibration_fit_id"] = fit_id
        current["calibration_applied"] = calibrated
        output_parts.append(current.drop(columns="month_period"))
        months_done.append(month)

        if calibrated:
            deltas = np.log(calibrated_predictions / base)
            if np.max(np.abs(deltas)) > MAX_ABS_LOG_CORRECTION + 1e-12:
                raise RuntimeError("R58 applied a correction outside its preregistered bound")

    if months_done != list(MONTHS):
        raise RuntimeError(f"R58 evaluated unexpected months for {track}: {months_done}")
    return pd.concat(output_parts, ignore_index=True), fits


def run() -> dict:
    from experiments.bill_target74 import r04_ablation as budget
    from experiments.bill_target74 import r22_remarks_tfidf as scoring
    from experiments.bill_target74 import runner as campaign

    card = campaign.PROTOCOL["cards"].get(CARD)
    if not card or card.get("status") != "registered":
        raise RuntimeError("R58 must be preregistered before reading development results")
    if card.get("source_sha256") != _sha256(Path(__file__)):
        raise RuntimeError("R58 source changed after preregistration")
    source = budget._source_identity()
    if source["data_sha256"] != card["data_sha256"]:
        raise RuntimeError("R58 source data changed after preregistration")

    inputs = {
        "off_market": _verified_input(campaign, R52_RESULT, R52_RUN, "off_market"),
        "list_aware": _verified_input(campaign, R35_DEV_RESULT, R35_DEV_RUN, "list_aware"),
    }
    run_id = "bill_target74_r58_past_log_residual_calibration_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = campaign.CAMPAIGN_ROOT / "batches" / run_id
    output.mkdir(parents=True, exist_ok=False)
    provenance = output / "provenance"
    provenance.mkdir()
    for track, (_, result_path, paired_path, manifest, _) in inputs.items():
        prefix = "r52" if track == "off_market" else "r35_development"
        shutil.copy2(result_path, provenance / f"{prefix}_development_results.json")
        shutil.copy2(paired_path, provenance / f"{prefix}_development_paired_predictions.csv")
        shutil.copy2(result_path.parent / "run_manifest.json", provenance / f"{prefix}_development_run_manifest.json")

    campaign._append({
        "event": "card_start", "card": CARD, "run_id": run_id,
        "tracks": list(TRACKS), "development_months": list(MONTHS),
        "expected_fit_starts": card["expected_fit_starts"],
        "calibration_fit_starts": EXPECTED_CALIBRATION_FITS,
        "parent_run_ids": card["parent_run_ids"],
        "input_result_sha256": {track: _sha256(item[1]) for track, item in inputs.items()},
        "input_paired_sha256": {track: _sha256(item[2]) for track, item in inputs.items()},
        "data_sha256": source["data_sha256"],
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })

    tracks = {}
    fit_details = []
    paired_parts = []
    try:
        for track in TRACKS:
            _, _, _, _, rows = inputs[track]
            corrected, fits = _calibrate_track(
                campaign=campaign, run_id=run_id, track=track, rows=rows
            )
            fit_details.extend(fits)
            comparison = corrected[[
                "month", "track", "ml_num", "sold_price", "base_prediction", "candidate_prediction",
                "calibration_intercept", "calibration_slope", "calibration_center_log_price",
                "calibration_training_rows", "calibration_fit_id", "calibration_applied",
            ]].copy()
            comparison = comparison.rename(columns={"base_prediction": "parent_prediction"})
            score = scoring.score_development_confirmation(
                comparison,
                months=MONTHS,
                target_mdape_strictly_below=7.0 if track == "off_market" else 4.0,
                bootstrap_replicates=BOOTSTRAP_REPLICATES,
                bootstrap_seed=BOOTSTRAP_SEED,
                minimum_relative_gain_percent=0.5,
                minimum_months_improved=6,
                maximum_worst_month_regression_percent=5.0,
                maximum_mape_regression_percent=1.0,
            )
            tracks[track] = score
            paired_parts.append(comparison)
    except Exception as exc:
        _write_json(output / "failure_checkpoint.json", {
            "run_id": run_id, "card": CARD,
            "error": f"{type(exc).__name__}: {exc}", "fit_starts": len(fit_details),
            "fit_details": fit_details,
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        campaign._append({
            "event": "card_failed", "card": CARD, "run_id": run_id,
            "fit_starts": len(fit_details), "error": f"{type(exc).__name__}: {exc}",
        })
        campaign._write_progress(
            "r58_development_failed", next_action="inspect the R58 failure checkpoint and ledger before retry",
            r58_failure=str((output / "failure_checkpoint.json").relative_to(campaign.ROOT)).replace("\\", "/"),
        )
        raise

    if len(fit_details) != EXPECTED_CALIBRATION_FITS:
        raise RuntimeError(f"R58 expected {EXPECTED_CALIBRATION_FITS} calibration fits; got {len(fit_details)}")
    paired = pd.concat(paired_parts, ignore_index=True)
    paired_path = output / "paired_predictions.csv"
    paired.to_csv(paired_path, index=False)
    result = {
        "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id, "card": CARD,
        "parent_run_ids": card["parent_run_ids"], "development_months": list(MONTHS),
        "fit_starts": len(fit_details), "base_model_fit_starts": 0,
        "calibration_fit_starts": len(fit_details), "fit_details": fit_details,
        "tracks": tracks, "calibration": card["calibration"],
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        "reserve_validation_eligible": False, "data_sha256": source["data_sha256"],
    }
    result_path = output / "development_results.json"
    _write_json(result_path, result)
    source_files = _source_files(campaign)
    code_hashes = {
        str(path.relative_to(campaign.ROOT)).replace("\\", "/"): _sha256(path)
        for path in source_files
    }
    for index, path in enumerate(source_files):
        shutil.copy2(path, provenance / f"src_{index:02d}_{path.name}.source")
    manifest = {
        "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
        "source_identity": source, "code_sha256": code_hashes,
        "data_sha256": source["data_sha256"], "fit_starts": len(fit_details),
        "fit_details": fit_details,
        "parent_artifacts": {
            track: {"run_id": item[0]["run_id"], "result_sha256": _sha256(item[1]),
                    "paired_predictions_sha256": _sha256(item[2])}
            for track, item in inputs.items()
        },
        "result_sha256": _sha256(result_path), "paired_predictions_sha256": _sha256(paired_path),
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    }
    manifest_path = output / "run_manifest.json"
    _write_json(manifest_path, manifest)

    lines = [
        "# R58: strict-past log-residual median calibration", "",
        f"Campaign: `{result['campaign_id']}`; run: `{run_id}`.",
        "Each month is calibrated only from earlier safe development months. No base learner was retrained.",
        "The correction is clipped to ±0.05 log points and rounded to the nearest dollar.",
        "The first month has no past OOF support and uses the base prediction unchanged.",
        "No 2025-05/08 or hidden labels were accessed.", "",
        "| Track | Base MdAPE | Calibrated MdAPE | Relative gain | MAPE change | Months improved | Bootstrap upper95 (pp) | Goal met | Gate |",
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
        "run_id": run_id, "fit_starts": len(fit_details), "calibration_fit_starts": len(fit_details),
        "result_path": str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        "paired_predictions_path": str(paired_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        "manifest_path": str(manifest_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        "qualifies_by_track": qualifies, "development_metrics": tracks,
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    campaign.PROTOCOL_PATH.write_text(
        json.dumps(campaign.PROTOCOL, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    campaign._append({
        "event": "card_complete", "card": CARD, "run_id": run_id,
        "fit_starts": len(fit_details), "result_path": card["result_path"],
        "qualifies_by_track": qualifies, "development_metrics": tracks,
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    passing = [track for track, passed in qualifies.items() if passed]
    status = "r58_development_passed" if passing else "r58_development_failed"
    next_action = (
        f"prepare a separate safe fixed-origin confirmation for R58 tracks {passing}; keep R22/R35 as validated parents until confirmation"
        if passing else
        "R58 failed development gates; retain R22/R35 and preregister a distinct hypothesis"
    )
    campaign._write_progress(
        status, next_action=next_action, r58_run_id=run_id,
        r58_result=card["result_path"], r58_fit_starts=len(fit_details),
    )
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Register or run R58 strict-past residual calibration")
    parser.add_argument("--register-only", action="store_true",
                        help="preregister without loading paired development rows or fitting calibrators")
    args = parser.parse_args()
    registration = register_card()
    if args.register_only:
        print(json.dumps(registration, indent=2, ensure_ascii=False))
        return
    print(json.dumps(run(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
