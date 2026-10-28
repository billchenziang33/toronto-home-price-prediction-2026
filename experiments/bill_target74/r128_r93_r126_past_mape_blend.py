"""R128: strictly past-selected MAPE blend of R93 spatial and R126 cyclic OOF."""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.bill_target74 import r22_remarks_tfidf as r22
from experiments.bill_target74 import runner as campaign


CARD = "R128_offmarket_r93_r126_past_mape_blend"
TRACK = "off_market"
MONTHS = tuple(campaign.PROTOCOL["development_months"])
SCREEN_MONTHS = tuple(campaign.PROTOCOL["screen_months"])
GRID = tuple(float(value) for value in np.round(np.arange(0.0, 1.0001, 0.05), 2))
MIN_HISTORY_MONTHS = 2
MIN_HISTORY_ROWS = 3000
MAPE_GUARD_PERCENT = float(campaign.PROTOCOL["gates"]["screen_mape_regression_max_percent"])
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20261061
INPUT_CARDS = {
    "r93": "R93_strict_past_spatial_residual_difference",
    "r126": "R126_R125_offmarket_cyclic_month_development_confirmation",
    "r22": "R22_train_fold_tfidf_remarks",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return str(path.relative_to(campaign.ROOT)).replace("\\", "/")


def _resolve_inputs() -> dict:
    progress = json.loads(campaign.STATE.read_text(encoding="utf-8"))
    keys = {"r93": "r93_result", "r126": "r126_result", "r22": "r22_development_result"}
    resolved = {}
    for name, key in keys.items():
        result_path = campaign.ROOT / progress[key]
        paired_path = result_path.parent / "paired_predictions.csv"
        manifest_path = result_path.parent / "run_manifest.json"
        if not all(path.is_file() for path in (result_path, paired_path, manifest_path)):
            raise FileNotFoundError(f"incomplete saved {name} artifacts")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if result.get("reserved_months_labels_opened") is not False:
            raise RuntimeError(f"{name} result does not attest that its run avoided reserved labels")
        if result.get("hidden_labels_opened") is not False:
            raise RuntimeError(f"{name} result does not attest that its run avoided hidden labels")
        if result.get("data_sha256") != campaign.PROTOCOL["data_sha256"]:
            raise RuntimeError(f"{name} result data hash differs from the active campaign")
        if manifest.get("paired_predictions_sha256") != _sha256(paired_path):
            raise RuntimeError(f"{name} prediction file hash differs from its manifest")
        if manifest.get("result_sha256") != _sha256(result_path):
            raise RuntimeError(f"{name} result file hash differs from its manifest")
        resolved[name] = {
            "card": INPUT_CARDS[name],
            "run_id": result["run_id"],
            "result_path": result_path,
            "result_sha256": _sha256(result_path),
            "predictions_path": paired_path,
            "predictions_sha256": _sha256(paired_path),
            "manifest_path": manifest_path,
            "manifest_sha256": _sha256(manifest_path),
        }
    return resolved


def _code_hashes() -> dict[str, str]:
    return {
        _relative(Path(__file__)): _sha256(Path(__file__)),
        _relative(Path(r22.__file__)): _sha256(Path(r22.__file__)),
        _relative(Path(campaign.__file__)): _sha256(Path(campaign.__file__)),
    }


def _validate_pairs(inputs: dict) -> pd.DataFrame:
    frames = {}
    for name, meta in inputs.items():
        frame = pd.read_csv(meta["predictions_path"], dtype={"ml_num": str})
        required = {"month", "track", "ml_num", "sold_price", "candidate_prediction"}
        if missing := required - set(frame.columns):
            raise RuntimeError(f"{name} OOF predictions are missing columns: {sorted(missing)}")
        frame = frame.loc[frame.track.eq(TRACK),
                          ["month", "ml_num", "sold_price", "candidate_prediction"]].copy()
        frame["month"] = frame.month.astype(str)
        frame["ml_num"] = frame.ml_num.astype(str)
        if frame.duplicated(["month", "ml_num"]).any():
            raise RuntimeError(f"{name} predictions contain duplicate month/listing keys")
        if set(frame.month.unique()) != set(MONTHS) or len(frame) != 15642:
            raise RuntimeError(f"{name} predictions do not cover the exact safe development window")
        frames[name] = frame.rename(columns={
            "sold_price": f"sold_price_{name}",
            "candidate_prediction": {"r93": "prediction_r93", "r126": "prediction_r126",
                                     "r22": "prediction_r22"}[name],
        })

    merged = frames["r22"]
    merged = merged.merge(frames["r93"], on=["month", "ml_num"], validate="one_to_one")
    merged = merged.merge(frames["r126"], on=["month", "ml_num"], validate="one_to_one")
    actuals = [merged[f"sold_price_{name}"].to_numpy(float) for name in ("r22", "r93", "r126")]
    if not all(np.array_equal(actuals[0], other) for other in actuals[1:]):
        raise RuntimeError("R22, R93, and R126 labels differ on aligned listing keys")
    predictions = merged[["prediction_r22", "prediction_r93", "prediction_r126"]].to_numpy(float)
    if not np.isfinite(predictions).all() or (predictions <= 0).any():
        raise RuntimeError("an input OOF prediction is not finite and positive")
    if set(merged.month.unique()).intersection(campaign.PROTOCOL["reserved_months"]):
        raise RuntimeError("reserved-month rows appeared in the R128 safe development inputs")
    merged["sold_price"] = merged.sold_price_r22.astype(float)
    merged = merged.sort_values(["month", "ml_num"], kind="mergesort").reset_index(drop=True)
    return merged


def register_card() -> dict:
    existing = campaign.PROTOCOL["cards"].get(CARD)
    current_code_hashes = _code_hashes()
    if existing:
        if existing.get("status") != "registered":
            raise RuntimeError(f"{CARD} is already {existing.get('status')}; inspect before retry")
        if existing.get("code_sha256") != current_code_hashes:
            raise RuntimeError("R128 code changed after registration")
        if existing.get("input_artifacts") != _input_hashes(_resolve_inputs()):
            raise RuntimeError("R128 inputs changed after registration")
        return existing

    inputs = _resolve_inputs()
    _validate_pairs(inputs)
    artifact_hashes = _input_hashes(inputs)
    card = {
        "status": "registered",
        "stage": "fixed_two_month_screen_then_eight_month_development",
        "tracks": [TRACK],
        "screen_months": list(SCREEN_MONTHS),
        "development_months": list(MONTHS),
        "expected_calibration_fit_starts": len(SCREEN_MONTHS) + len(MONTHS),
        "learner_training_fit_starts": 0,
        "comparison_parent": "R22 current validated off-market OOF predictions",
        "candidate_components": {
            "r93": inputs["r93"]["run_id"],
            "r126": inputs["r126"]["run_id"],
        },
        "method": (
            "For each target development month, select a single dollar-space blend weight on "
            "R126 among 0.00..1.00 in 0.05 increments, using only earlier safe development OOF "
            "rows. Minimize past pooled MAPE subject to past pooled MdAPE no worse than the R93 "
            "candidate and past pooled MAPE no more than 1.00% relatively worse than R22; ties "
            "favor lower MdAPE then smaller R126 weight. Require at least two earlier months and "
            "3000 earlier rows; otherwise use R93 (weight 0). Blend rounded-dollar R93/R126 "
            "predictions and round the result to the nearest dollar. No model learner is refit."
        ),
        "difference_from_prior_work": (
            "R93 uses a past-only spatial residual-difference calibration on an R52 base; R126 "
            "adds cyclic listing-month features to an R22 base. R128 tests their strictly past-" 
            "selected blend using MAPE as the inner weight objective; a prior read-only MdAPE-" 
            "selected blend diagnostic did not beat standalone R93 and is not repeated."
        ),
        "weight_grid": list(GRID),
        "minimum_past_months": MIN_HISTORY_MONTHS,
        "minimum_past_rows": MIN_HISTORY_ROWS,
        "past_mape_guard_percent_relative_to_r22": MAPE_GUARD_PERCENT,
        "screen_gates": {
            "relative_mdape_gain_min_percent": campaign.PROTOCOL["gates"]["screen_relative_mdape_improvement_percent"],
            "both_screen_months_improve": campaign.PROTOCOL["gates"]["screen_both_months_improve"],
            "mape_regression_max_percent": MAPE_GUARD_PERCENT,
            "paired_bootstrap_upper95_below_zero": True,
        },
        "development_gates": dict(campaign.PROTOCOL["gates"]),
        "input_artifacts": artifact_hashes,
        "code_sha256": current_code_hashes,
        "data_sha256": campaign.PROTOCOL["data_sha256"],
        "reserved_months_labels_opened": False,
        "hidden_labels_opened": False,
    }
    campaign.PROTOCOL["cards"][CARD] = card
    campaign.PROTOCOL_PATH.write_text(
        json.dumps(campaign.PROTOCOL, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    campaign._append({
        "event": "card_registered", "card": CARD,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "expected_calibration_fit_starts": card["expected_calibration_fit_starts"],
        "learner_training_fit_starts": 0, "source_sha256": current_code_hashes[_relative(Path(__file__))],
        "input_artifacts": artifact_hashes,
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    campaign._write_progress(
        "r128_registered",
        next_action="run the registered fixed 2023-03 and 2024-06 past-MAPE blend screen",
        r128_card=CARD,
    )
    return card


def _input_hashes(inputs: dict) -> dict:
    return {
        name: {
            "card": item["card"], "run_id": item["run_id"],
            "result_sha256": item["result_sha256"],
            "paired_predictions_sha256": item["predictions_sha256"],
            "manifest_sha256": item["manifest_sha256"],
        }
        for name, item in inputs.items()
    }


def _select_weight(frame: pd.DataFrame, month: str) -> tuple[float, dict]:
    history = frame.loc[frame.month.lt(month)]
    detail = {
        "target_month": month,
        "past_rows": int(len(history)),
        "past_months": sorted(history.month.unique().tolist()),
    }
    if history.month.nunique() < MIN_HISTORY_MONTHS or len(history) < MIN_HISTORY_ROWS:
        return 0.0, {**detail, "selected_r126_weight": 0.0,
                      "fallback_reason": "insufficient_strictly_past_support"}

    y = history.sold_price.to_numpy(float)
    p22 = history.prediction_r22.to_numpy(float)
    p93 = history.prediction_r93.to_numpy(float)
    p126 = history.prediction_r126.to_numpy(float)
    r93_mdape = campaign._metrics(y, p93)["mdape_percent"]
    r22_mape = campaign._metrics(y, p22)["mape_percent"]
    feasible = []
    candidates = []
    for weight in GRID:
        prediction = np.rint((1.0 - weight) * p93 + weight * p126)
        metrics = campaign._metrics(y, prediction)
        mape_change = 100.0 * (metrics["mape_percent"] - r22_mape) / r22_mape
        row = (metrics["mape_percent"], metrics["mdape_percent"], weight, mape_change)
        candidates.append(row)
        if metrics["mdape_percent"] <= r93_mdape + 1e-12 and mape_change <= MAPE_GUARD_PERCENT + 1e-12:
            feasible.append(row)
    if not feasible:
        return 0.0, {**detail, "selected_r126_weight": 0.0,
                      "fallback_reason": "no_feasible_weight_on_past_rows",
                      "past_r93_mdape_percent": float(r93_mdape),
                      "past_r22_mape_percent": float(r22_mape)}
    best = min(feasible, key=lambda value: (value[0], value[1], value[2]))
    return float(best[2]), {
        **detail, "selected_r126_weight": float(best[2]), "fallback_reason": None,
        "selected_past_mape_percent": float(best[0]),
        "selected_past_mdape_percent": float(best[1]),
        "selected_past_mape_change_vs_r22_percent": float(best[3]),
        "past_r93_mdape_percent": float(r93_mdape),
        "past_r22_mape_percent": float(r22_mape),
    }


def _run(stage: str) -> dict:
    card = campaign.PROTOCOL["cards"].get(CARD)
    if not card or card.get("status") not in ("registered", "screen_passed"):
        raise RuntimeError("R128 must be registered, and its screen must pass before development")
    if card.get("code_sha256") != _code_hashes():
        raise RuntimeError("R128 code/dependency hash changed after registration")
    if stage == "screen" and card["status"] != "registered":
        raise RuntimeError("R128 screen has already run; do not rerun on the same folds")
    if stage == "development" and card["status"] != "screen_passed":
        raise RuntimeError("R128 development confirmation requires a passed screen")

    inputs = _resolve_inputs()
    if _input_hashes(inputs) != card["input_artifacts"]:
        raise RuntimeError("R128 source OOF inputs changed after preregistration")
    frame = _validate_pairs(inputs)
    selected_months = SCREEN_MONTHS if stage == "screen" else MONTHS
    run_id = "r128s_" if stage == "screen" else "r128d_"
    run_id += datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = campaign.CAMPAIGN_ROOT / "batches" / run_id
    output.mkdir(parents=True, exist_ok=False)
    (output / "provenance").mkdir()
    shutil_files = []
    for name, item in inputs.items():
        for kind in ("result_path", "manifest_path"):
            source_path = item[kind]
            destination = output / "provenance" / f"{name}_{source_path.name}"
            destination.write_bytes(source_path.read_bytes())
            shutil_files.append(_relative(destination))

    fit_details = []
    outputs = []
    try:
        for month in selected_months:
            fit_id = f"{campaign.PROTOCOL['campaign_id']}:{CARD}:{stage}:{TRACK}:{month}:{run_id}"
            campaign._append({
                "event": "fit_start", "card": CARD, "stage": stage, "run_id": run_id,
                "fit_id": fit_id, "track": TRACK, "month": month,
                "calibration_only": True, "learner_training_fit": False,
                "data_sha256": card["data_sha256"],
                "reserved_months_labels_opened": False, "hidden_labels_opened": False,
            })
            started = time.perf_counter()
            weight, selection = _select_weight(frame, month)
            target = frame.loc[frame.month.eq(month)].copy()
            prediction = np.rint(
                (1.0 - weight) * target.prediction_r93.to_numpy(float)
                + weight * target.prediction_r126.to_numpy(float)
            )
            if not np.isfinite(prediction).all() or (prediction <= 0).any():
                raise RuntimeError(f"R128 generated invalid predictions for {month}")
            row = pd.DataFrame({
                "month": target.month, "track": TRACK, "ml_num": target.ml_num,
                "sold_price": target.sold_price, "parent_prediction": target.prediction_r22,
                "r93_prediction": target.prediction_r93,
                "r126_prediction": target.prediction_r126,
                "r126_weight": weight, "candidate_prediction": prediction,
            })
            row["parent_ape_percent"] = 100.0 * np.abs(row.sold_price - row.parent_prediction) / row.sold_price
            row["candidate_ape_percent"] = 100.0 * np.abs(row.sold_price - row.candidate_prediction) / row.sold_price
            outputs.append(row)
            detail = {
                "fit_id": fit_id, "track": TRACK, "month": month, "status": "complete",
                "seconds": time.perf_counter() - started, "calibration_only": True,
                "learner_training_fit": False, **selection,
            }
            fit_details.append(detail)
            campaign._append({
                "event": "fit_complete", "card": CARD, "stage": stage, "run_id": run_id,
                **detail, "reserved_months_labels_opened": False, "hidden_labels_opened": False,
            })

        paired = pd.concat(outputs, ignore_index=True)
        if paired.duplicated(["month", "ml_num"]).any():
            raise RuntimeError("R128 generated duplicate month/listing predictions")
        if stage == "screen":
            score = r22.score_development_confirmation(
                paired[["month", "ml_num", "sold_price", "parent_prediction", "candidate_prediction"]],
                months=SCREEN_MONTHS, target_mdape_strictly_below=7.0,
                bootstrap_replicates=BOOTSTRAP_REPLICATES, bootstrap_seed=BOOTSTRAP_SEED,
                minimum_relative_gain_percent=campaign.PROTOCOL["gates"]["screen_relative_mdape_improvement_percent"],
                minimum_months_improved=2, maximum_worst_month_regression_percent=0.0,
                maximum_mape_regression_percent=MAPE_GUARD_PERCENT,
            )
            checks = {
                "relative_mdape_gain": score["relative_mdape_improvement_percent"] >= campaign.PROTOCOL["gates"]["screen_relative_mdape_improvement_percent"],
                "both_months_improved": score["months_improved"] == 2,
                "mape_guard": score["mape_relative_change_percent"] <= MAPE_GUARD_PERCENT,
                "bootstrap_upper_below_zero": score["bootstrap_mdape_difference_percent"]["upper95"] < 0.0,
            }
            qualifies = all(checks.values())
        else:
            score = r22.score_development_confirmation(
                paired[["month", "ml_num", "sold_price", "parent_prediction", "candidate_prediction"]],
                months=MONTHS, target_mdape_strictly_below=7.0,
                bootstrap_replicates=BOOTSTRAP_REPLICATES, bootstrap_seed=BOOTSTRAP_SEED,
            )
            checks = {
                "development_gates": score["qualifies_development"],
                "strict_mdape_target": score["target_met"],
            }
            qualifies = all(checks.values())

        result = {
            "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
            "card": CARD, "stage": stage, "track": TRACK,
            "months": list(selected_months), "scores": score,
            "gate_checks": checks, "qualifies": bool(qualifies),
            "calibration_fit_starts": len(fit_details), "learner_training_fit_starts": 0,
            "fit_details": fit_details, "input_artifacts": card["input_artifacts"],
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
            "reserve_validation_eligible": False,
        }
        pair_path = output / "paired_predictions.csv"
        result_path = output / ("screen_results.json" if stage == "screen" else "development_results.json")
        paired.to_csv(pair_path, index=False)
        result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        manifest = {
            "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id, "card": CARD,
            "stage": stage, "data_sha256": card["data_sha256"],
            "code_sha256": card["code_sha256"], "input_artifacts": card["input_artifacts"],
            "provenance_copies": shutil_files,
            "paired_predictions_sha256": _sha256(pair_path),
            "result_sha256": _sha256(result_path),
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
            "run R128 eight-month development confirmation; fixed screen passed"
            if stage == "screen" and qualifies else
            "keep R22 and register a distinct off-market method; R128 screen failed"
            if stage == "screen" else
            "proceed to eligible fixed-origin confirmation only if R128 meets the development target and gates"
            if qualifies else
            "retain R22; R128 did not reach the strict off-market target and development gates"
        )
        status = f"r128_{stage}_{'passed' if qualifies else 'failed'}"
        campaign._write_progress(
            status, next_action=next_action, r128_run_id=run_id,
            r128_result=_relative(result_path), r128_learner_training_fit_starts=0,
        )
        return result
    except Exception as exc:
        campaign._append({
            "event": "card_failed", "card": CARD, "stage": stage, "run_id": run_id,
            "fit_starts": len(fit_details), "error": f"{type(exc).__name__}: {exc}",
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        campaign._write_progress(
            f"r128_{stage}_failed_runtime", next_action="audit R128 checkpoint before any retry",
            r128_run_id=run_id,
        )
        raise


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Register or run the R128 past-MAPE blend")
    parser.add_argument("--register-only", action="store_true")
    parser.add_argument("--screen", action="store_true")
    parser.add_argument("--development", action="store_true")
    args = parser.parse_args()
    if sum((args.register_only, args.screen, args.development)) != 1:
        parser.error("choose exactly one of --register-only, --screen, or --development")
    if args.register_only:
        print(json.dumps(register_card(), indent=2, ensure_ascii=False))
    else:
        if CARD not in campaign.PROTOCOL["cards"]:
            raise RuntimeError("register R128 before running it")
        _run("screen" if args.screen else "development")


if __name__ == "__main__":
    main()
