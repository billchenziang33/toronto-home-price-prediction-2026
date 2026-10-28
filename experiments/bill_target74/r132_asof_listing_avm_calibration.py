"""R132: safely calibrate the supplied external AVM using past sales only."""
from __future__ import annotations

import json
import hashlib
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.bill_target74 import r22_remarks_tfidf as r22
from experiments.bill_target74 import runner as campaign
from experiments.bill_greedy import runner as greedy


CARD = "R132_asof_listing_avm_calibration"
TRACKS = ("off_market", "list_aware")
MONTHS = tuple(campaign.PROTOCOL["development_months"])
SCREEN_MONTHS = tuple(campaign.PROTOCOL["screen_months"])
PARENT_KEYS = {"off_market": "r22_development_result", "list_aware": "r35_development_result"}
PARENT_CARDS = {"off_market": "R22_train_fold_tfidf_remarks",
                "list_aware": "R35_listaware_detached_r22_text_expert"}
MIN_HISTORY_AVM_ROWS = 500
MIN_TYPE_HISTORY_ROWS = 100
TYPE_SHRINKAGE_PSEUDOCOUNT = 200.0
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20261132
TARGET_MDAPE = {"off_market": 7.0, "list_aware": 4.0}


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return str(path.relative_to(campaign.ROOT)).replace("\\", "/")


def _source_paths() -> list[Path]:
    return [
        Path(__file__), Path(campaign.__file__),
        Path(greedy.__file__), Path(r22.__file__),
        campaign.ROOT / "experiments/model_optimization_v2/folds.py",
    ]


def _source_hashes() -> dict[str, str]:
    hashes = {_relative(path): _sha256(path) for path in _source_paths()}
    # The card and its run status are written into this protocol during the
    # experiment, so hash the immutable campaign settings without mutable cards.
    protocol_context = {key: value for key, value in campaign.PROTOCOL.items() if key != "cards"}
    encoded = json.dumps(
        protocol_context, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    hashes["campaign_protocol_context"] = hashlib.sha256(encoded).hexdigest()
    return hashes


def _resolve_inputs() -> dict:
    progress = json.loads(campaign.STATE.read_text(encoding="utf-8"))
    resolved = {}
    for track, key in PARENT_KEYS.items():
        result_path = campaign.ROOT / progress[key]
        pair_path = result_path.parent / "paired_predictions.csv"
        manifest_path = result_path.parent / "run_manifest.json"
        if not all(path.is_file() for path in (result_path, pair_path, manifest_path)):
            raise FileNotFoundError(f"incomplete saved {track} parent artifacts")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for flag in ("reserved_months_labels_opened", "hidden_labels_opened"):
            if result.get(flag) is not False or manifest.get(flag) is not False:
                raise RuntimeError(f"{track} parent does not attest safe-label handling: {flag}")
        if result.get("data_sha256") != campaign.PROTOCOL["data_sha256"]:
            raise RuntimeError(f"{track} parent data hash differs from active campaign")
        if manifest.get("result_sha256") != _sha256(result_path):
            raise RuntimeError(f"{track} parent result hash differs from manifest")
        if manifest.get("paired_predictions_sha256") != _sha256(pair_path):
            raise RuntimeError(f"{track} parent prediction hash differs from manifest")
        frame = pd.read_csv(pair_path, dtype={"ml_num": str})
        required = {"month", "track", "ml_num", "sold_price", "candidate_prediction"}
        if missing := required - set(frame.columns):
            raise RuntimeError(f"{track} parent predictions lack columns: {sorted(missing)}")
        frame = frame.loc[frame.track.eq(track),
                          ["month", "ml_num", "sold_price", "candidate_prediction"]].copy()
        frame["month"] = frame.month.astype(str)
        frame["ml_num"] = frame.ml_num.astype(str)
        if frame.duplicated(["month", "ml_num"]).any():
            raise RuntimeError(f"{track} parent predictions contain duplicate keys")
        if len(frame) != 15642 or set(frame.month.unique()) != set(MONTHS):
            raise RuntimeError(f"{track} parent predictions do not cover the safe development window")
        if set(frame.month).intersection(campaign.PROTOCOL["reserved_months"]):
            raise RuntimeError(f"{track} parent predictions include a reserved month")
        if not np.isfinite(frame.candidate_prediction.to_numpy(float)).all() or (
                frame.candidate_prediction.to_numpy(float) <= 0).any():
            raise RuntimeError(f"{track} parent has invalid predictions")
        resolved[track] = {
            "card": PARENT_CARDS[track], "run_id": result["run_id"],
            "result_path": result_path, "paired_path": pair_path,
            "manifest_path": manifest_path, "frame": frame,
            "result_sha256": _sha256(result_path), "paired_predictions_sha256": _sha256(pair_path),
            "manifest_sha256": _sha256(manifest_path),
        }
    return resolved


def _input_hashes(inputs: dict) -> dict:
    return {
        track: {
            "card": item["card"], "run_id": item["run_id"],
            "result_sha256": item["result_sha256"],
            "paired_predictions_sha256": item["paired_predictions_sha256"],
            "manifest_sha256": item["manifest_sha256"],
        }
        for track, item in inputs.items()
    }


def register_card() -> dict:
    existing = campaign.PROTOCOL["cards"].get(CARD)
    if existing:
        if existing.get("status") != "registered":
            raise RuntimeError(f"{CARD} is already {existing.get('status')}; do not rerun")
        if existing.get("code_sha256") != _source_hashes():
            raise RuntimeError("R132 source changed after preregistration")
        inputs = _resolve_inputs()
        if existing.get("input_artifacts") != _input_hashes(inputs):
            raise RuntimeError("R132 parent inputs changed after preregistration")
        return existing

    inputs = _resolve_inputs()
    hashes = _source_hashes()
    card = {
        "status": "registered",
        "stage": "fixed_two_month_screen_then_eight_month_development",
        "tracks": list(TRACKS),
        "screen_months": list(SCREEN_MONTHS),
        "development_months": list(MONTHS),
        "expected_screen_calibration_fit_starts": len(SCREEN_MONTHS),
        "expected_development_calibration_fit_starts": len(MONTHS),
        "learner_training_fit_starts": 0,
        "comparison_parents": dict(PARENT_CARDS),
        "parent_runs": {track: inputs[track]["run_id"] for track in TRACKS},
        "method": (
            "For each target month, use only training rows sold before that month. An external AVM "
            "estimate is eligible only when positive and its estimate date is no later than the "
            "property's listing date. Fit a global median log(sold_price/AVM) correction from past "
            "eligible rows; for house types with at least 100 past eligible rows, shrink the type "
            "median toward the global median by n/(n+200). Require at least 500 past eligible AVM "
            "rows. For eligible heldout listings, replace the parent prediction with the calibrated "
            "AVM estimate rounded to the nearest dollar; for every ineligible listing, keep its "
            "validated parent prediction unchanged. The current-month sold price is used only for "
            "scoring and is never used to fit the calibration. No learner is trained."
        ),
        "difference_from_prior_work": (
            "R22 explicitly excluded external AVM inputs, and the active safe-fold audit found only "
            "14.109% of the eight-month development rows have an estimate dated by listing. R132 "
            "tests a causal, as-of-listing AVM fallback with past-only log-ratio calibration; estimates "
            "dated after listing are treated as unavailable. This does not use days_on_market or any "
            "post-listing estimate."
        ),
        "parameters": {
            "avm_availability_cutoff": "external_avm_estimate_date <= date_listed",
            "minimum_past_avm_rows": MIN_HISTORY_AVM_ROWS,
            "minimum_type_history_rows": MIN_TYPE_HISTORY_ROWS,
            "type_median_shrinkage_pseudocount": TYPE_SHRINKAGE_PSEUDOCOUNT,
            "eligible_row_action": "replace parent with calibrated AVM estimate; round to nearest dollar",
            "ineligible_row_action": "keep parent prediction unchanged",
            "learner_training_fit_starts": 0,
        },
        "screen_gates": {
            "relative_mdape_gain_min_percent": campaign.PROTOCOL["gates"]["screen_relative_mdape_improvement_percent"],
            "both_screen_months_improve": True,
            "mape_regression_max_percent": campaign.PROTOCOL["gates"]["screen_mape_regression_max_percent"],
            "paired_bootstrap_upper95_below_zero": True,
        },
        "development_gates": dict(campaign.PROTOCOL["gates"]),
        "target_mdape_strictly_below_percent": dict(TARGET_MDAPE),
        "input_artifacts": _input_hashes(inputs),
        "code_sha256": hashes,
        "data_sha256": campaign.PROTOCOL["data_sha256"],
        "reserved_months_labels_opened": False,
        "hidden_labels_opened": False,
        "reserve_validation_eligible": False,
    }
    campaign.PROTOCOL["cards"][CARD] = card
    campaign.PROTOCOL_PATH.write_text(
        json.dumps(campaign.PROTOCOL, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    campaign._append({
        "event": "card_registered", "card": CARD,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "expected_screen_calibration_fit_starts": len(SCREEN_MONTHS),
        "expected_development_calibration_fit_starts": len(MONTHS),
        "learner_training_fit_starts": 0,
        "source_sha256": hashes[_relative(Path(__file__))],
        "input_artifacts": card["input_artifacts"],
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    campaign._write_progress(
        "r132_registered",
        next_action="run the preregistered safe two-month as-of AVM screen",
        r132_card=CARD,
    )
    return card


def _asof_avm(rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    avm = pd.to_numeric(rows.external_avm_estimate, errors="coerce").to_numpy(float)
    estimate_date = pd.to_datetime(rows.external_avm_estimate_date, errors="coerce")
    listing_date = pd.to_datetime(rows.date_listed, errors="coerce")
    eligible = (np.isfinite(avm) & (avm > 0) & estimate_date.notna().to_numpy()
                & listing_date.notna().to_numpy()
                & (estimate_date.to_numpy() <= listing_date.to_numpy()))
    return eligible, avm, estimate_date.to_numpy()


def _fit_calibration(train: pd.DataFrame, target_month: str) -> tuple[float, dict]:
    cutoff = pd.Period(target_month, freq="M").start_time
    dates = pd.to_datetime(train.date_sold, errors="raise")
    if dates.empty or (dates >= cutoff).any():
        raise RuntimeError(f"R132 training rows are not strictly before {target_month}")
    if dates.dt.to_period("M").astype(str).isin(campaign.PROTOCOL["reserved_months"]).any():
        raise RuntimeError("R132 training rows include a reserved-month label")
    eligible, avm, _ = _asof_avm(train)
    history = train.loc[eligible, ["house_type_name", "sold_price"]].copy()
    history["avm"] = avm[eligible]
    history = history.loc[
        pd.to_numeric(history.sold_price, errors="coerce").gt(0)
        & pd.to_numeric(history.avm, errors="coerce").gt(0)
    ].copy()
    if len(history) < MIN_HISTORY_AVM_ROWS:
        raise RuntimeError(
            f"R132 has only {len(history)} past as-of AVM rows for {target_month}; "
            f"requires {MIN_HISTORY_AVM_ROWS}"
        )
    history["log_ratio"] = np.log(history.sold_price.astype(float) / history.avm.astype(float))
    global_correction = float(history.log_ratio.median())
    type_corrections = {}
    type_counts = {}
    for house_type, group in history.groupby("house_type_name", dropna=False, sort=True):
        count = len(group)
        type_counts[str(house_type)] = count
        if count >= MIN_TYPE_HISTORY_ROWS:
            local = float(group.log_ratio.median())
            strength = count / (count + TYPE_SHRINKAGE_PSEUDOCOUNT)
            type_corrections[str(house_type)] = float(
                global_correction + strength * (local - global_correction)
            )
    return global_correction, {
        "target_month": target_month,
        "training_rows_before_cutoff": int(len(train)),
        "past_asof_avm_rows": int(len(history)),
        "past_avm_coverage_percent": float(100.0 * len(history) / len(train)),
        "global_log_sold_to_avm_median": global_correction,
        "type_history_rows": type_counts,
        "type_log_corrections": type_corrections,
    }


def _score_track(paired: pd.DataFrame, track: str, months: tuple[str, ...], stage: str) -> tuple[dict, dict]:
    rows = paired.loc[paired.track.eq(track)].copy()
    frame = rows[["month", "ml_num", "sold_price", "parent_prediction", "candidate_prediction"]].copy()
    target = TARGET_MDAPE[track]
    if stage == "screen":
        score = r22.score_development_confirmation(
            frame, months=months, target_mdape_strictly_below=target,
            bootstrap_replicates=BOOTSTRAP_REPLICATES, bootstrap_seed=BOOTSTRAP_SEED,
            minimum_relative_gain_percent=campaign.PROTOCOL["gates"]["screen_relative_mdape_improvement_percent"],
            minimum_months_improved=len(months), maximum_worst_month_regression_percent=0.0,
            maximum_mape_regression_percent=campaign.PROTOCOL["gates"]["screen_mape_regression_max_percent"],
        )
        checks = {
            "relative_mdape_gain": score["relative_mdape_improvement_percent"] >= campaign.PROTOCOL["gates"]["screen_relative_mdape_improvement_percent"],
            "both_screen_months_improved": score["months_improved"] == len(months),
            "mape_guard": score["mape_relative_change_percent"] <= campaign.PROTOCOL["gates"]["screen_mape_regression_max_percent"],
            "paired_bootstrap_upper95_below_zero": score["bootstrap_mdape_difference_percent"]["upper95"] < 0.0,
        }
        return score, checks
    score = r22.score_development_confirmation(
        frame, months=months, target_mdape_strictly_below=target,
        bootstrap_replicates=BOOTSTRAP_REPLICATES, bootstrap_seed=BOOTSTRAP_SEED,
    )
    return score, {
        "development_gates": bool(score["qualifies_development"]),
        "strict_mdape_target": bool(score["target_met"]),
    }


def _run(stage: str) -> dict:
    card = campaign.PROTOCOL["cards"].get(CARD)
    allowed = {"registered"} if stage == "screen" else {"screen_passed"}
    if not card or card.get("status") not in allowed:
        raise RuntimeError(f"R132 {stage} is not allowed from status {None if not card else card.get('status')}")
    if card.get("code_sha256") != _source_hashes():
        raise RuntimeError("R132 source/dependency hashes changed after registration")
    inputs = _resolve_inputs()
    if _input_hashes(inputs) != card["input_artifacts"]:
        raise RuntimeError("R132 parent OOF inputs changed after registration")
    tracks = list(TRACKS) if stage == "screen" else list(card["screen_passed_tracks"])
    months = SCREEN_MONTHS if stage == "screen" else MONTHS
    run_id = f"r132{'s' if stage == 'screen' else 'd'}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    output = campaign.CAMPAIGN_ROOT / "batches" / run_id
    output.mkdir(parents=True, exist_ok=False)
    (output / "provenance").mkdir()
    copied = []
    for track, item in inputs.items():
        for name in ("result_path", "paired_path", "manifest_path"):
            source = item[name]
            dest = output / "provenance" / f"{track}_{source.name}"
            shutil.copy2(source, dest)
            copied.append(_relative(dest))
    fit_details, outputs = [], []
    try:
        for month in months:
            fit_id = f"{campaign.PROTOCOL['campaign_id']}:{CARD}:{stage}:shared_avm:{month}:{run_id}"
            campaign._append({
                "event": "fit_start", "card": CARD, "stage": stage, "run_id": run_id,
                "fit_id": fit_id, "track": "shared_asof_avm", "month": month,
                "calibration_only": True, "learner_training_fit": False,
                "data_sha256": card["data_sha256"],
                "reserved_months_labels_opened": False, "hidden_labels_opened": False,
            })
            started = time.perf_counter()
            _, train, _, truth, heldout = greedy._load_fold(month)
            cutoff = pd.Period(month, freq="M").start_time
            dates = pd.to_datetime(train.date_sold, errors="raise")
            heldout_months = set(pd.to_datetime(heldout.date_sold, errors="raise").dt.to_period("M").astype(str))
            if (dates >= cutoff).any() or dates.dt.to_period("M").astype(str).isin(
                    campaign.PROTOCOL["reserved_months"]).any():
                raise RuntimeError(f"R132 fold {month} violates its strict safe cutoff")
            if heldout_months != {month} or heldout_months.intersection(campaign.PROTOCOL["reserved_months"]):
                raise RuntimeError(f"R132 heldout rows are not exactly the safe month {month}")
            if (pd.to_datetime(heldout.date_sold, errors="raise") >= pd.Timestamp("2025-05-01")).any():
                raise RuntimeError("R132 attempted to load rows at or after the first reserved month")
            global_correction, calibration = _fit_calibration(train, month)
            eligible, avm, _ = _asof_avm(heldout)
            if not np.array_equal(heldout.ml_num.astype(str).to_numpy(), truth.ml_num.astype(str).to_numpy()):
                raise RuntimeError(f"R132 heldout IDs and safe-month labels differ for {month}")
            avm_prediction = avm * np.exp(global_correction)
            house_types = heldout.house_type_name.astype("string").fillna("<NA>").astype(str).to_numpy()
            for i, house_type in enumerate(house_types):
                avm_prediction[i] = avm[i] * np.exp(calibration["type_log_corrections"].get(house_type, global_correction))
            if not np.isfinite(avm_prediction[eligible]).all() or (avm_prediction[eligible] <= 0).any():
                raise RuntimeError(f"R132 generated invalid AVM predictions for {month}")
            calibration["eligible_evaluation_rows"] = int(eligible.sum())
            calibration["eligible_evaluation_rate_percent"] = float(100.0 * eligible.mean())
            fit_details.append({
                "fit_id": fit_id, "month": month, "status": "complete",
                "seconds": time.perf_counter() - started, "calibration_only": True,
                "learner_training_fit": False, **calibration,
            })
            campaign._append({
                "event": "fit_complete", "card": CARD, "stage": stage, "run_id": run_id,
                "fit_id": fit_id, "track": "shared_asof_avm", "month": month,
                "status": "complete", "seconds": fit_details[-1]["seconds"],
                "training_rows_before_cutoff": calibration["training_rows_before_cutoff"],
                "past_asof_avm_rows": calibration["past_asof_avm_rows"],
                "eligible_evaluation_rows": calibration["eligible_evaluation_rows"],
                "learner_training_fit": False,
                "reserved_months_labels_opened": False, "hidden_labels_opened": False,
            })

            held = heldout[["ml_num", "sold_price"]].copy()
            held["ml_num"] = held.ml_num.astype(str)
            held["month"] = month
            held["avm_eligible"] = eligible
            held["calibrated_avm_prediction"] = np.rint(avm_prediction)
            for track in tracks:
                parent = inputs[track]["frame"].loc[
                    inputs[track]["frame"].month.eq(month),
                    ["month", "ml_num", "sold_price", "candidate_prediction"],
                ].copy().rename(columns={"candidate_prediction": "parent_prediction"})
                parent["ml_num"] = parent.ml_num.astype(str)
                paired = parent.merge(held, on=["month", "ml_num", "sold_price"], how="left", validate="one_to_one")
                if len(paired) != len(parent) or paired.avm_eligible.isna().any():
                    raise RuntimeError(f"R132 parent and as-of AVM rows do not align for {track}/{month}")
                paired["candidate_prediction"] = paired.parent_prediction
                eligible_rows = paired.avm_eligible.astype(bool)
                paired.loc[eligible_rows, "candidate_prediction"] = paired.loc[
                    eligible_rows, "calibrated_avm_prediction"
                ]
                paired["track"] = track
                paired["avm_eligible"] = eligible_rows.astype(int)
                paired["avm_prediction_used"] = paired.calibrated_avm_prediction.where(eligible_rows)
                outputs.append(paired[[
                    "month", "track", "ml_num", "sold_price", "parent_prediction",
                    "candidate_prediction", "avm_eligible", "avm_prediction_used",
                ]])

        paired_all = pd.concat(outputs, ignore_index=True)
        if paired_all.duplicated(["month", "track", "ml_num"]).any():
            raise RuntimeError("R132 generated duplicate paired rows")
        scores, checks = {}, {}
        for track in tracks:
            scores[track], checks[track] = _score_track(
                paired_all.loc[paired_all.track.eq(track)], track, months, stage
            )
        passed = [track for track in tracks if all(checks[track].values())]
        result = {
            "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
            "card": CARD, "stage": stage, "tracks": tracks, "months": list(months),
            "scores": scores, "gate_checks": checks, "qualifies_by_track": {t: t in passed for t in tracks},
            "calibration_fit_starts": len(fit_details), "learner_training_fit_starts": 0,
            "fit_details": fit_details, "input_artifacts": card["input_artifacts"],
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
            "reserve_validation_eligible": False,
        }
        pair_path = output / "paired_predictions.csv"
        result_path = output / ("screen_results.json" if stage == "screen" else "development_results.json")
        paired_all.to_csv(pair_path, index=False)
        result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        manifest = {
            "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id, "card": CARD,
            "stage": stage, "data_sha256": card["data_sha256"], "code_sha256": card["code_sha256"],
            "input_artifacts": card["input_artifacts"], "provenance_copies": copied,
            "paired_predictions_sha256": _sha256(pair_path), "result_sha256": _sha256(result_path),
            "calibration_fit_starts": len(fit_details), "learner_training_fit_starts": 0,
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        }
        manifest_path = output / "run_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        report = [
            "# R132 as-of listing-date AVM calibration", "",
            "Only external estimates dated on or before the listing date were used. All other rows kept the validated parent prediction.",
            "No learner was trained; each fit ledger entry is a past-only median-ratio calibration.", "",
        ]
        for track in tracks:
            score = scores[track]
            report.extend([
                f"## {track}", "",
                f"Parent MdAPE: {score['parent']['mdape_percent']:.6f}%",
                f"Candidate MdAPE: {score['candidate']['mdape_percent']:.6f}%",
                f"Relative MdAPE gain: {score['relative_mdape_improvement_percent']:+.4f}%",
                f"MAPE relative change: {score['mape_relative_change_percent']:+.4f}%",
                f"Months improved: {score['months_improved']}/{len(months)}",
                f"Paired-bootstrap upper 95%: {score['bootstrap_mdape_difference_percent']['upper95']:+.5f} pp",
                f"Gates: {json.dumps(checks[track], ensure_ascii=False)}", "",
            ])
            for month, detail in score["monthly"].items():
                report.append(
                    f"- {month}: {detail['parent']['mdape_percent']:.5f}% → "
                    f"{detail['candidate']['mdape_percent']:.5f}%"
                )
            report.append("")
        report.append("No reserved-month or hidden labels were opened.")
        (output / "research_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

        if stage == "screen":
            card.update({
                "status": "screen_passed" if passed else "screen_failed",
                "screen_passed_tracks": passed, "screen_run_id": run_id,
                "screen_result_path": _relative(result_path), "screen_result_sha256": _sha256(result_path),
                "screen_paired_predictions_path": _relative(pair_path),
                "screen_paired_predictions_sha256": _sha256(pair_path),
                "screen_manifest_path": _relative(manifest_path),
                "screen_manifest_sha256": _sha256(manifest_path),
                "screen_metrics": scores, "screen_gate_checks": checks,
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
                "development_metrics": scores, "development_gate_checks": checks,
                "qualifies_development_by_track": {t: bool(scores[t]["qualifies_development"] and scores[t]["target_met"]) for t in tracks},
            })
        campaign.PROTOCOL_PATH.write_text(
            json.dumps(campaign.PROTOCOL, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        campaign._append({
            "event": "card_complete", "card": CARD, "stage": stage, "run_id": run_id,
            "fit_starts": len(fit_details), "learner_training_fit_starts": 0,
            "qualifies_by_track": {t: t in passed for t in tracks},
            "result_path": _relative(result_path), "result_sha256": _sha256(result_path),
            "paired_predictions_sha256": _sha256(pair_path),
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        next_action = (
            "run R132 eight-month safe development for " + ", ".join(passed)
            if stage == "screen" and passed else
            "retain R22/R35; R132 screen failed and register a distinct method"
            if stage == "screen" else
            "retain validated parents unless R132 meets the strict per-track development target and gates"
        )
        campaign._write_progress(
            f"r132_{stage}_{'passed' if passed else 'failed'}", next_action=next_action,
            r132_run_id=run_id, r132_result=_relative(result_path),
            r132_learner_training_fit_starts=0,
        )
        return result
    except Exception as exc:
        campaign._append({
            "event": "card_failed", "card": CARD, "stage": stage, "run_id": run_id,
            "fit_starts": len(fit_details), "error": f"{type(exc).__name__}: {exc}",
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        campaign._write_progress(
            f"r132_{stage}_failed_runtime", next_action="audit R132 artifacts before any retry",
            r132_run_id=run_id,
        )
        raise


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Register or run R132 as-of AVM calibration")
    parser.add_argument("--register-only", action="store_true")
    parser.add_argument("--screen", action="store_true")
    parser.add_argument("--development", action="store_true")
    args = parser.parse_args()
    if sum((args.register_only, args.screen, args.development)) != 1:
        parser.error("choose exactly one of --register-only, --screen, or --development")
    if args.register_only:
        print(json.dumps(register_card(), indent=2, ensure_ascii=False))
        return
    result = _run("screen" if args.screen else "development")
    print(json.dumps({
        "run_id": result["run_id"], "fit_starts": result["calibration_fit_starts"],
        "learner_training_fit_starts": 0, "qualifies_by_track": result["qualifies_by_track"],
        "scores": {track: {
            "parent_mdape": score["parent"]["mdape_percent"],
            "candidate_mdape": score["candidate"]["mdape_percent"],
            "relative_mdape_gain_percent": score["relative_mdape_improvement_percent"],
            "mape_relative_change_percent": score["mape_relative_change_percent"],
            "months_improved": score["months_improved"],
            "bootstrap_upper95_pp": score["bootstrap_mdape_difference_percent"]["upper95"],
            "gate_checks": result["gate_checks"][track],
        } for track, score in result["scores"].items()},
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
