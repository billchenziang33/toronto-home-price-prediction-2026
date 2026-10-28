"""R124: strictly-past house-type-by-direction shrinkage of R97 corrections."""
from __future__ import annotations

import json
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.bill_target74 import r116_r97_correction_direction_gate as r116
from experiments.bill_target74 import runner as campaign


CARD = "R124_r97_house_type_direction_past_oof_gate"
TRACKS = ("off_market", "list_aware")
MONTHS = tuple(campaign.PROTOCOL["development_months"])
GAMMA_GRID = (0.0, 0.25, 0.50, 0.75, 1.0)
MIN_HISTORY_ROWS = 500
MIN_HISTORY_MONTHS = 2
MAPE_REGRESSION_MAX_PERCENT = 1.0
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20261053
R93_RUN = "r93s_20261025T201838Z"
R97_RUN = "r97m_20261025T210147Z"


def _sha256(path: Path) -> str:
    return r116._sha256(path)


def _relative(path: Path) -> str:
    return str(path.relative_to(campaign.ROOT)).replace("\\", "/")


def _registration_spec() -> dict:
    source = r116._source_identity()
    inputs = r116._input_identity()
    return {
        "status": "registered",
        "tracks": list(TRACKS), "development_months": list(MONTHS),
        "parents": {"off_market": "R52 experimental OOF base", "list_aware": "R35 validated OOF base"},
        "best_comparators": {"off_market": "R93 spatial OOF candidate", "list_aware": "fixed R97 candidate"},
        "method": (
            "Reuse the saved walk-forward R97 log-residual correction. For each target development month, "
            "select gamma from {0,0.25,0.50,0.75,1.00} for each house_type_name x correction-sign group, "
            "using only strictly earlier R97 OOF rows. Minimize past MdAPE subject to past MAPE no more than "
            "1% above gamma=0. Require at least 500 earlier rows across at least two earlier months; if an "
            "interaction group lacks support, back off to house type, then correction sign, then gamma=0. "
            "Apply round(base_prediction * exp(gamma * correction_log))."
        ),
        "difference_from_prior_work": (
            "R116 selected correction strength globally by correction sign; R123 selected it by house type. "
            "R124 tests their untried house_type_name x correction-sign interaction with a predeclared support "
            "hierarchy. R113 magnitude bins, R94 spatial calibration, and R97's base residual learner remain "
            "unchanged. This experiment uses strict-past OOF rows and trains no new learner."
        ),
        "selection": {
            "group": "house_type_name x sign(correction_log), with house-type then sign-only backoff",
            "gamma_grid": list(GAMMA_GRID),
            "minimum_past_rows_per_group": MIN_HISTORY_ROWS,
            "minimum_past_months_per_group": MIN_HISTORY_MONTHS,
            "mape_regression_max_percent": MAPE_REGRESSION_MAX_PERCENT,
            "minimum_rows_and_months_per_level": {"rows": MIN_HISTORY_ROWS, "months": MIN_HISTORY_MONTHS},
            "fallback_order": ["house_type_x_correction_sign", "house_type", "correction_sign", "gamma_zero"],
            "tie_break": ["lower past MdAPE", "lower past MAPE", "smaller gamma"],
        },
        "inputs": inputs,
        "source_sha256": _sha256(Path(__file__)),
        "data_sha256": source["data_sha256"], "parent_commit": source["commit"],
        "expected_calibration_fit_starts": (len(MONTHS) - 1) * len(TRACKS),
        "learner_training_fit_starts": 0,
        "promotion_gates": campaign.PROTOCOL["gates"],
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    }


def register_card() -> dict:
    previous = campaign.PROTOCOL["cards"].get(CARD)
    if previous:
        if previous.get("status") != "registered":
            raise RuntimeError(f"{CARD} already exists with status {previous.get('status')!r}; inspect before rerunning")
        current_inputs = r116._input_identity()
        if previous.get("inputs") != current_inputs:
            raise RuntimeError("R124 registered input identity changed; refusing to reuse the card")
        ledger_rows = ([json.loads(line) for line in campaign.LEDGER.read_text(encoding="utf-8").splitlines()
                        if line.strip()] if campaign.LEDGER.is_file() else [])
        if any(row.get("event") == "fit_start" and row.get("card") == CARD for row in ledger_rows):
            raise RuntimeError("R124 has a fit_start; its preregistered source may not be changed")
        current_sha = _sha256(Path(__file__))
        if previous.get("source_sha256") != current_sha:
            previous_sha = previous.get("source_sha256")
            previous["source_sha256"] = current_sha
            campaign.PROTOCOL_PATH.write_text(
                json.dumps(campaign.PROTOCOL, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            campaign._append({
                "event": "preregistered_source_hash_updated_before_fit", "card": CARD,
                "previous_source_sha256": previous_sha, "source_sha256": current_sha,
                "reason": "made R124 registration idempotent before its first fit_start",
                "fit_starts": 0, "reserved_months_labels_opened": False,
                "hidden_labels_opened": False,
            })
        return previous
    card = _registration_spec()
    campaign.PROTOCOL["cards"][CARD] = card
    campaign.PROTOCOL_PATH.write_text(
        json.dumps(campaign.PROTOCOL, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    campaign._append({"event": "card_registered", "card": CARD,
                      "expected_calibration_fit_starts": card["expected_calibration_fit_starts"],
                      "learner_training_fit_starts": 0, "source_sha256": card["source_sha256"],
                      "inputs": card["inputs"], "data_sha256": card["data_sha256"],
                      "reserved_months_labels_opened": False, "hidden_labels_opened": False})
    campaign._write_progress("r124_registered", next_action="run R124 strictly-past house-type-by-direction residual calibration",
                             r124_card=CARD)
    return card


def _select_group(history: pd.DataFrame, house_type: str | None,
                  correction_sign: str | None) -> tuple[float | None, dict]:
    mask = pd.Series(True, index=history.index)
    if house_type is not None:
        mask &= history.house_type_name.eq(house_type)
    if correction_sign is not None:
        sign_mask = history.correction_log.gt(0.0) if correction_sign == "positive" else history.correction_log.le(0.0)
        mask &= sign_mask
    group = history.loc[mask]
    months = sorted(group.month.astype(str).unique().tolist())
    detail = {"house_type_name": house_type, "correction_sign": correction_sign,
              "past_rows": int(len(group)), "past_months": months}
    if len(group) < MIN_HISTORY_ROWS or len(months) < MIN_HISTORY_MONTHS:
        return None, {**detail, "gamma": None, "fallback": "insufficient_past_support"}

    actual = group.sold_price_r93.to_numpy(float)
    base = np.rint(group.base_prediction.to_numpy(float))
    delta = group.correction_log.to_numpy(float)
    baseline_mape = campaign._metrics(actual, base)["mape_percent"]
    feasible = []
    for gamma in GAMMA_GRID:
        prediction = np.rint(base * np.exp(gamma * delta))
        score = campaign._metrics(actual, prediction)
        mape_change = 100.0 * (score["mape_percent"] - baseline_mape) / baseline_mape
        if mape_change <= MAPE_REGRESSION_MAX_PERCENT + 1e-12:
            feasible.append((score["mdape_percent"], score["mape_percent"], gamma, mape_change))
    mdape, mape, gamma, mape_change = min(feasible, key=lambda item: item[:3])
    return gamma, {**detail, "gamma": gamma,
                   "selected_past_mdape_percent": float(mdape),
                   "selected_past_mape_percent": float(mape),
                   "selected_past_mape_change_percent": float(mape_change)}


def _select_gamma(history: pd.DataFrame, house_type: str,
                  correction_sign: str) -> tuple[float, dict]:
    levels = (("house_type_x_correction_sign", house_type, correction_sign),
              ("house_type", house_type, None),
              ("correction_sign", None, correction_sign))
    attempted = []
    for level, type_filter, sign_filter in levels:
        gamma, detail = _select_group(history, type_filter, sign_filter)
        attempted.append({"level": level, **detail})
        if gamma is not None:
            return gamma, {**detail, "selected_level": level,
                           "fallback": None if level == "house_type_x_correction_sign" else level,
                           "attempted_levels": attempted}
    return 0.0, {"house_type_name": house_type, "correction_sign": correction_sign,
                 "gamma": 0.0, "fallback": "insufficient_past_support_all_levels",
                 "attempted_levels": attempted}


def _score_track(frame: pd.DataFrame) -> dict:
    actual = frame.sold_price_r93.to_numpy(float)
    base = np.rint(frame.base_prediction.to_numpy(float))
    fixed = frame.candidate_prediction.to_numpy(float)
    candidate = frame.r124_prediction.to_numpy(float)
    track = str(frame.track.iloc[0])
    best = (frame.candidate_prediction_r93.to_numpy(float) if track == "off_market" else fixed)
    scores = {
        "parent": campaign._metrics(actual, base),
        "r97_fixed": campaign._metrics(actual, fixed),
        "r124": campaign._metrics(actual, candidate),
        "previous_best": campaign._metrics(actual, best),
    }
    monthly = {}
    for month, group in frame.groupby("month", sort=True):
        y = group.sold_price_r93.to_numpy(float)
        p = np.rint(group.base_prediction.to_numpy(float))
        c = group.r124_prediction.to_numpy(float)
        pm, cm = campaign._metrics(y, p), campaign._metrics(y, c)
        monthly[str(month)] = {"n": int(len(group)), "parent": pm, "candidate": cm,
                               "relative_mdape_improvement_percent": float(
                                   100 * (pm["mdape_percent"] - cm["mdape_percent"]) / pm["mdape_percent"])}

    gain = 100 * (scores["parent"]["mdape_percent"] - scores["r124"]["mdape_percent"]) / scores["parent"]["mdape_percent"]
    mape_change = 100 * (scores["r124"]["mape_percent"] - scores["parent"]["mape_percent"]) / scores["parent"]["mape_percent"]
    monthly_gains = [item["relative_mdape_improvement_percent"] for item in monthly.values()]
    base_ape = 100 * np.abs(base - actual) / actual
    candidate_ape = 100 * np.abs(candidate - actual) / actual
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    differences = np.empty(BOOTSTRAP_REPLICATES)
    for replicate in range(BOOTSTRAP_REPLICATES):
        index = rng.integers(0, len(actual), size=len(actual))
        differences[replicate] = np.median(candidate_ape[index]) - np.median(base_ape[index])
    bootstrap = {"lower95": float(np.quantile(differences, 0.025)),
                 "upper95": float(np.quantile(differences, 0.975)),
                 "replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED}
    gates = campaign.PROTOCOL["gates"]
    target = 7.0 if track == "off_market" else 4.0
    worst_regression = max([0.0] + [-value for value in monthly_gains])
    checks = {
        "relative_mdape_improvement": gain >= gates["development_relative_mdape_improvement_percent"],
        "months_improved": sum(value > 0 for value in monthly_gains) >= gates["development_months_improved_min"],
        "worst_month_regression": worst_regression <= gates["development_worst_month_regression_max_percent"],
        "mape_regression": mape_change <= gates["development_mape_regression_max_percent"],
        "bootstrap_upper_below_zero": bootstrap["upper95"] < 0.0,
        "strict_target": scores["r124"]["mdape_percent"] < target,
        "beats_previous_best": scores["r124"]["mdape_percent"] < scores["previous_best"]["mdape_percent"],
    }
    return {**scores, "relative_mdape_improvement_percent": float(gain),
            "mape_relative_change_percent": float(mape_change),
            "months_improved": int(sum(value > 0 for value in monthly_gains)),
            "worst_month_regression_percent": float(worst_regression),
            "paired_bootstrap_mdape_difference_pp": bootstrap, "monthly": monthly,
            "gate_checks": checks, "qualifies_development": all(checks.values()),
            "target_mdape_strictly_below": target}


def run() -> dict:
    card = campaign.PROTOCOL["cards"].get(CARD)
    if not card or card.get("status") != "registered":
        raise RuntimeError("R124 must be preregistered before scoring")
    if card.get("source_sha256") != _sha256(Path(__file__)):
        raise RuntimeError("R124 source changed after preregistration")
    source = r116._source_identity()
    inputs = r116._input_identity()
    if inputs != card["inputs"]:
        raise RuntimeError("R124 input identity differs from preregistration")

    # R116's loader verifies exact R93/R97 row alignment and hashes before exposing labels.
    rows = r116._load_inputs()
    if "house_type_name" not in rows:
        raise RuntimeError("R93 does not supply house_type_name for R124 calibration")
    rows["house_type_name"] = rows.house_type_name.fillna("<missing>").astype(str)
    rows["correction_sign"] = np.where(rows.correction_log.gt(0.0), "positive", "nonpositive")
    rows["r124_prediction"] = np.nan
    rows["selected_gamma"] = np.nan
    run_id = "r124c_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = campaign.CAMPAIGN_ROOT / "batches" / run_id
    output.mkdir(parents=True, exist_ok=False)
    provenance = output / "provenance"
    provenance.mkdir()
    shutil.copy2(Path(__file__), provenance / Path(__file__).name)
    shutil.copy2(campaign.PACKAGE / "r116_r97_correction_direction_gate.py", provenance)
    for name, identity in inputs.items():
        shutil.copy2(campaign.ROOT / identity["path"], provenance / f"{name}_paired_predictions.csv")
    (provenance / "registered_campaign_protocol.json").write_text(
        json.dumps(campaign.PROTOCOL, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    campaign._append({"event": "card_start", "card": CARD, "run_id": run_id,
                      "tracks": list(TRACKS), "development_months": list(MONTHS),
                      "expected_calibration_fit_starts": card["expected_calibration_fit_starts"],
                      "learner_training_fit_starts": 0, "inputs": inputs,
                      "reserved_months_labels_opened": False, "hidden_labels_opened": False})
    output_rows = []
    fit_details = []
    for track in TRACKS:
        track_rows = rows.loc[rows.track.eq(track)].copy()
        for month in MONTHS:
            current = track_rows.loc[track_rows.month.eq(month)].copy().reset_index(drop=True)
            history = track_rows.loc[track_rows.month.lt(month)].copy()
            if current.empty:
                raise RuntimeError(f"R124 has no OOF rows for {track}/{month}")
            cutoff = pd.Timestamp(month + "-01")
            if not history.empty and pd.to_datetime(history.month + "-01").max() >= cutoff:
                raise RuntimeError(f"R124 history crosses target cutoff {track}/{month}")
            if history.empty:
                gamma_by_group = {}
                support = {}
                for house_type in sorted(current.house_type_name.unique().tolist()):
                    for sign in ("nonpositive", "positive"):
                        key = f"{house_type}|{sign}"
                        gamma_by_group[key] = 0.0
                        support[key] = {"house_type_name": house_type, "correction_sign": sign,
                                        "past_rows": 0, "past_months": [], "gamma": 0.0,
                                        "fallback": "first_development_fold"}
            else:
                fit_id = f"{campaign.PROTOCOL['campaign_id']}:{CARD}:{track}:{month}:{run_id}"
                campaign._append({"event": "fit_start", "card": CARD, "run_id": run_id,
                                  "fit_id": fit_id, "stage": "past_oof_calibration", "track": track,
                                  "month": month, "training_oof_rows": int(len(history)),
                                  "query_rows": int(len(current)), "history_cutoff_exclusive": cutoff.isoformat(),
                                  "learner_training_fit": False, "reserved_months_labels_opened": False,
                                  "hidden_labels_opened": False})
                gamma_by_group, support = {}, {}
                for house_type in sorted(current.house_type_name.unique().tolist()):
                    for sign in ("nonpositive", "positive"):
                        key = f"{house_type}|{sign}"
                        gamma, detail = _select_gamma(history, str(house_type), sign)
                        gamma_by_group[key] = gamma
                        support[key] = detail

            gamma_values = np.array([
                gamma_by_group.get(f"{house_type}|{sign}", 0.0)
                for house_type, sign in zip(current.house_type_name, current.correction_sign)
            ], dtype=float)
            base = current.base_prediction.to_numpy(float)
            correction = current.correction_log.to_numpy(float)
            prediction = np.rint(base * np.exp(gamma_values * correction))
            if not np.isfinite(prediction).all() or (prediction <= 0).any():
                raise RuntimeError(f"R124 produced invalid prices for {track}/{month}")
            current["r124_prediction"] = prediction
            current["selected_gamma"] = gamma_values
            current["history_oof_rows"] = int(len(history))
            current["history_oof_months"] = len(history.month.unique())
            output_rows.append(current)
            if not history.empty:
                detail = {"fit_id": fit_id, "track": track, "month": month, "status": "complete",
                          "training_oof_rows": int(len(history)), "query_rows": int(len(current)),
                          "history_cutoff_exclusive": cutoff.isoformat(),
                          "gamma_by_house_type_and_sign": gamma_by_group,
                          "support_by_house_type_and_sign": support}
                fit_details.append(detail)
                campaign._append({"event": "fit_complete", "card": CARD, "run_id": run_id,
                                  "stage": "past_oof_calibration", **detail,
                                  "learner_training_fit": False, "reserved_months_labels_opened": False,
                                  "hidden_labels_opened": False})

    paired = pd.concat(output_rows, ignore_index=True)
    scores = {track: _score_track(paired.loc[paired.track.eq(track)].copy()) for track in TRACKS}
    result = {"campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id, "card": CARD,
              "stage": "strict_past_walk_forward_development", "development_months": list(MONTHS),
              "tracks": scores, "calibration_fit_starts": len(fit_details),
              "learner_training_fit_starts": 0, "fit_details": fit_details,
              "inputs": inputs, "reserved_months_labels_opened": False,
              "hidden_labels_opened": False}
    pair_path = output / "paired_predictions.csv"
    result_path = output / "development_results.json"
    paired.to_csv(pair_path, index=False)
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report = ["# R124 — strict-past R97 house-type × direction correction gate", "",
              f"Campaign `{result['campaign_id']}`; run `{run_id}`.",
              "For each development month, R124 selects correction shrinkage by house type and correction sign using only earlier R97 OOF rows. Groups without 500 past rows over two months back off to house type, then sign, then gamma=0. It trains no new learner.",
              "", "| Track | Base MdAPE | Fixed R97 | R124 | Previous best | MAPE change | Months improved | Pass |",
              "|---|---:|---:|---:|---:|---:|---:|---|"]
    for track, score in scores.items():
        report.append(f"| {track} | {score['parent']['mdape_percent']:.6f}% | {score['r97_fixed']['mdape_percent']:.6f}% | "
                      f"{score['r124']['mdape_percent']:.6f}% | {score['previous_best']['mdape_percent']:.6f}% | "
                      f"{score['mape_relative_change_percent']:+.3f}% | {score['months_improved']}/8 | "
                      f"{'yes' if score['qualifies_development'] else 'no'} |")
    report.extend(["", "This is development-only OOF evidence. R22/R35 remain the validated parents unless a separate preregistered confirmation passes.",
                   "No reserve or hidden labels were opened.", ""])
    report_path = output / "research_report.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    manifest = {"run_id": run_id, "card": CARD, "branch": source["branch"], "commit": source["commit"],
                "source_sha256": _sha256(Path(__file__)), "data_sha256": source["data_sha256"],
                "input_files": inputs, "result_path": _relative(result_path),
                "result_sha256": _sha256(result_path), "paired_predictions_path": _relative(pair_path),
                "paired_predictions_sha256": _sha256(pair_path), "report_path": _relative(report_path),
                "report_sha256": _sha256(report_path), "python": sys.version,
                "platform": platform.platform(), "packages": {name: __import__(name).__version__
                    for name in ("numpy", "pandas")}, "calibration_fit_starts": len(fit_details),
                "learner_training_fit_starts": 0, "reserved_months_labels_opened": False,
                "hidden_labels_opened": False}
    manifest_path = output / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    card.update({"status": "development_complete", "run_id": run_id,
                 "fit_starts": len(fit_details), "learner_training_fit_starts": 0,
                 "development_metrics": scores,
                 "qualifies_by_track": {track: value["qualifies_development"] for track, value in scores.items()},
                 "result_path": _relative(result_path), "result_sha256": _sha256(result_path),
                 "paired_predictions_path": _relative(pair_path), "paired_predictions_sha256": _sha256(pair_path),
                 "manifest_path": _relative(manifest_path), "manifest_sha256": _sha256(manifest_path),
                 "reserved_months_labels_opened": False, "hidden_labels_opened": False})
    campaign.PROTOCOL_PATH.write_text(json.dumps(campaign.PROTOCOL, indent=2, ensure_ascii=False) + "\n",
                                     encoding="utf-8")
    campaign._append({"event": "card_complete", "card": CARD, "run_id": run_id,
                      "fit_starts": len(fit_details), "learner_training_fit_starts": 0,
                      "qualifies_by_track": {track: value["qualifies_development"] for track, value in scores.items()},
                      "result_path": _relative(result_path), "result_sha256": _sha256(result_path),
                      "paired_predictions_sha256": _sha256(pair_path),
                      "reserved_months_labels_opened": False, "hidden_labels_opened": False})
    campaign._write_progress("r124_development_complete",
                             next_action="retain R22/R35; use R124 OOF results to choose a distinct preregistered next experiment",
                             r124_run_id=run_id, r124_result=_relative(result_path))
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Register or run R124 house-type-by-direction past-OOF calibration")
    parser.add_argument("--register-only", action="store_true")
    args = parser.parse_args()
    card = register_card()
    if args.register_only:
        print(json.dumps(card, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(run(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
