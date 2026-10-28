"""R158: strict-past median multiplicative correction for R35 list-aware predictions."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.bill_target74 import runner as campaign


CARD = "R158_listaware_r35_past_oof_median_ratio_shift"
TRACK = "list_aware"
PARENT_CARD = "R35_listaware_detached_r22_text_expert"
SCREEN_MONTHS = tuple(campaign.PROTOCOL["screen_months"])
DEVELOPMENT_MONTHS = tuple(campaign.PROTOCOL["development_months"])
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20261158
MIN_HISTORY_ROWS = 1000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return str(path.relative_to(campaign.ROOT)).replace("\\", "/")


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
                    encoding="utf-8")


def _source_files() -> list[Path]:
    package = campaign.PACKAGE
    return [Path(__file__), package / "r158_independent_audit.py",
            package / "r157_listaware_isotonic_calibration.py",
            package / "r157_independent_audit.py",
            package / "r156_listaware_price_band_experts.py",
            package / "r44_past_oof_r35_shrinkage.py",
            package / "r22_remarks_tfidf.py", package / "runner.py"]


def _code_hashes() -> dict[str, str]:
    return {_relative(path): _sha256(path) for path in _source_files()}


def _helpers():
    from experiments.bill_target74 import r157_listaware_isotonic_calibration as r157
    return r157


def _calibration_support(oof: pd.DataFrame, month: str) -> dict:
    history = oof.loc[oof.month.lt(month)].copy()
    if len(history) < MIN_HISTORY_ROWS:
        raise RuntimeError(f"R158 has fewer than {MIN_HISTORY_ROWS} past OOF rows for {month}")
    history_months = sorted(history.month.unique().tolist())
    if not history_months or any(past >= month for past in history_months):
        raise RuntimeError(f"R158 history violates strict-past cutoff for {month}")
    residual = np.log(history.sold_price_source.to_numpy(float)
                      / history.variant_prediction.to_numpy(float))
    if not np.isfinite(residual).all():
        raise RuntimeError(f"R158 log price-ratio residuals are invalid for {month}")
    membership = history[["month", "ml_num", "variant_prediction", "sold_price_source", "list_price"]]
    membership = membership.sort_values(["month", "ml_num"], kind="mergesort")
    membership_hash = hashlib.sha256(
        pd.util.hash_pandas_object(membership, index=False).to_numpy().tobytes()
    ).hexdigest()
    return {"history_rows": int(len(history)), "history_months": history_months,
            "training_membership_sha256": membership_hash,
            "unique_log_ratio_residuals": int(np.unique(residual).size),
            "median_log_residual": float(np.median(residual)),
            "residual_median_absolute_deviation": float(np.median(np.abs(residual - np.median(residual))))}


def _preflight_support() -> dict:
    r157 = _helpers()
    oof, queries, _ = r157._load_inputs("screen")
    return {month: {**_calibration_support(oof, month),
                    "evaluation_rows": int(queries.month.eq(month).sum())}
            for month in SCREEN_MONTHS}


def register_card() -> dict:
    existing = campaign.PROTOCOL["cards"].get(CARD)
    code_hashes = _code_hashes()
    limits = campaign.PROTOCOL.get("limits", {})
    limit_keys = ["campaign_fit_starts", "campaign_minutes", "exploration_fit_starts",
                  "exploration_minutes", "final_reserve_fit_starts", "final_reserve_minutes",
                  "batch_fit_starts", "batch_minutes", "single_fit_seconds"]
    if any(limits.get(key) is not None for key in limit_keys):
        raise RuntimeError("R158 requires the active unbounded campaign limits")
    r157 = _helpers()
    parent = r157._verified_parent()
    parent_hashes = r157._parent_artifact_hashes(parent)
    r44 = r157._verified_r44_source()
    from experiments.bill_target74 import r04_ablation as identity_helpers
    identity = identity_helpers._source_identity()
    if identity.get("data_sha256") != campaign.PROTOCOL["data_sha256"]:
        raise RuntimeError("R158 source data identity differs from active campaign")
    support = _preflight_support()
    input_hashes = {"r44_result": _sha256(r44[2]), "r44_manifest": _sha256(r44[3]),
                    "r44_past_oof_predictions": _sha256(r44[4])}
    if existing:
        if existing.get("status") != "registered":
            raise RuntimeError(f"{CARD} already has status {existing.get('status')}; inspect it before retry")
        if (existing.get("code_sha256") != code_hashes or existing.get("input_hashes") != input_hashes
                or existing.get("parent_artifact_hashes") != parent_hashes
                or existing.get("preflight_support") != support):
            raise RuntimeError("R158 registration identity changed")
        return existing

    card = {
        "status": "registered", "stage": "fixed_two_month_screen_then_eight_month_development",
        "track": TRACK, "tracks": [TRACK], "screen_months": list(SCREEN_MONTHS),
        "development_months": list(DEVELOPMENT_MONTHS), "expected_fit_starts": len(SCREEN_MONTHS),
        "development_fit_starts_if_screen_passes": len(DEVELOPMENT_MONTHS),
        "comparison_parent": PARENT_CARD, "parent_run_id": parent["screen"]["run_id"],
        "parent_development_run_id": parent["development"]["run_id"],
        "parent_artifact_hashes": parent_hashes, "history_source_card": "R44_past_oof_r35_blend",
        "history_source_policy": "Use only R35 OOF rows from strictly earlier safe sale months.",
        "history_target": "log(sold_price/R35_OOF_prediction)",
        "calibration": {"estimator": "single strict-past median log ratio residual",
                        "correction": "median(log(sold_price/R35_OOF_prediction))",
                        "price_reconstruction": "round(R35_prediction * exp(median_log_residual))",
                        "minimum_past_oof_rows": MIN_HISTORY_ROWS},
        "method_id": "R158_LISTAWARE_STRICT_PAST_MEDIAN_MULTIPLICATIVE_RATIO_SHIFT",
        "method": ("For each target month, estimate one additive log-price residual as the median of "
                   "strictly earlier R35 OOF log(sold_price/R35_prediction) values. Multiply every R35 "
                   "target prediction by exp(that median), round to dollars, and leave all other rows unchanged."),
        "why_distinct": ("R157 used a flexible monotone isotonic mapping from prediction ratio to sale/list ratio. "
                         "R158 uses a single robust location correction on log sold/predicted price ratios; "
                         "it has no learned slope, bins, features, or screen-selected hyperparameters."),
        "preflight_support": support, "input_hashes": input_hashes,
        "screen_gates": {
            "relative_mdape_gain_min_percent": campaign.PROTOCOL["gates"]["screen_relative_mdape_improvement_percent"],
            "both_screen_months_improved": True,
            "mape_regression_max_percent": campaign.PROTOCOL["gates"]["screen_mape_regression_max_percent"],
            "paired_bootstrap_upper95_below_zero": True},
        "development_gates": dict(campaign.PROTOCOL["gates"]), "target_mdape_strictly_below_percent": 4.0,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES, "bootstrap_seed": BOOTSTRAP_SEED,
        "source_identity": identity, "data_sha256": campaign.PROTOCOL["data_sha256"],
        "code_sha256": code_hashes, "reserved_months_labels_opened": False,
        "hidden_labels_opened": False, "reserve_validation_eligible": False}
    campaign.PROTOCOL["cards"][CARD] = card
    _write_json(campaign.PROTOCOL_PATH, campaign.PROTOCOL)
    campaign._append({"event": "card_registered", "card": CARD,
                      "expected_fit_starts": card["expected_fit_starts"],
                      "development_fit_starts_if_screen_passes": card["development_fit_starts_if_screen_passes"],
                      "parent_artifact_hashes": parent_hashes, "input_hashes": input_hashes,
                      "source_sha256": code_hashes[_relative(Path(__file__))],
                      "reserved_months_labels_opened": False, "hidden_labels_opened": False})
    campaign._write_progress("r158_registered", r158_card=CARD,
                             next_action="run the preregistered two-fit R158 median ratio-shift screen")
    return card


def run(stage: str) -> dict:
    card = campaign.PROTOCOL["cards"].get(CARD)
    expected_status = "registered" if stage == "screen" else "screen_passed"
    if not card or card.get("status") != expected_status:
        raise RuntimeError(f"R158 {stage} requires card status {expected_status!r}")
    if card.get("code_sha256") != _code_hashes():
        raise RuntimeError("R158 source hashes changed after registration")
    if card.get("data_sha256") != _sha256(campaign.DATA):
        raise RuntimeError("R158 dataset hash changed after registration")
    r157 = _helpers()
    parent = r157._verified_parent()
    if r157._parent_artifact_hashes(parent) != card["parent_artifact_hashes"]:
        raise RuntimeError("R158 R35 parent artifacts changed after registration")
    r44 = r157._verified_r44_source()
    input_hashes = {"r44_result": _sha256(r44[2]), "r44_manifest": _sha256(r44[3]),
                    "r44_past_oof_predictions": _sha256(r44[4])}
    if input_hashes != card["input_hashes"]:
        raise RuntimeError("R158 R44 OOF artifacts changed after registration")
    oof, queries, input_identity = r157._load_inputs(stage, parent, r44)
    months = SCREEN_MONTHS if stage == "screen" else DEVELOPMENT_MONTHS
    run_id = ("r158s_" if stage == "screen" else "r158d_") + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = campaign.CAMPAIGN_ROOT / "batches" / run_id
    output.mkdir(parents=True, exist_ok=False)
    campaign._append({"event": "card_start", "card": CARD, "run_id": run_id, "stage": stage,
                      "track": TRACK, "months": list(months), "expected_fit_starts": len(months),
                      "reserved_months_labels_opened": False, "hidden_labels_opened": False})
    parts, profiles = [], {}
    for month in months:
        history = oof.loc[oof.month.lt(month)].copy()
        current = queries.loc[queries.month.eq(month)].copy()
        support = _calibration_support(oof, month)
        if current.empty or support["history_rows"] != len(history):
            raise RuntimeError(f"R158 target or history support is inconsistent for {month}")
        correction = support["median_log_residual"]
        candidate = np.rint(current.parent_prediction.to_numpy(float) * np.exp(correction))
        if not np.isfinite(candidate).all() or (candidate <= 0).any():
            raise RuntimeError(f"R158 generated invalid sale-price predictions for {month}")
        fit_id = f"{campaign.PROTOCOL['campaign_id']}:{CARD}:{stage}:{month}:{run_id}"
        campaign._append({"event": "fit_start", "card": CARD, "run_id": run_id, "stage": stage,
                          "fit_id": fit_id, "track": TRACK, "month": month,
                          "model_family": "past_oof_median_log_ratio_calibration",
                          "expected_objective": "median_log_sold_over_r35_prediction",
                          "training_rows": support["history_rows"], "training_months": support["history_months"],
                          "training_cutoff_exclusive": f"{month}-01T00:00:00",
                          "training_membership_sha256": support["training_membership_sha256"],
                          "evaluation_rows": int(len(current)),
                          "reserved_months_labels_opened": False, "hidden_labels_opened": False})
        profile = {"fit_id": fit_id, **support}
        profiles[month] = profile
        campaign._append({"event": "fit_complete", "card": CARD, "run_id": run_id, "stage": stage,
                          "fit_id": fit_id, "track": TRACK, "month": month, "status": "complete",
                          "actual_objective": "median_log_sold_over_r35_prediction",
                          "training_rows": support["history_rows"], "training_months": support["history_months"],
                          "training_membership_sha256": support["training_membership_sha256"],
                          "evaluation_rows": int(len(current)),
                          "median_log_residual": correction,
                          "reserved_months_labels_opened": False, "hidden_labels_opened": False})
        parts.append(pd.DataFrame({"month": month, "track": TRACK, "ml_num": current.ml_num.to_numpy(str),
                                   "sold_price": current.sold_price.to_numpy(float),
                                   "list_price": current.list_price.to_numpy(float),
                                   "parent_prediction": current.parent_prediction.to_numpy(float),
                                   "candidate_prediction": candidate.astype(float),
                                   "median_log_ratio_correction": correction,
                                   "history_oof_rows": support["history_rows"],
                                   "history_months": "|".join(support["history_months"])}))
    paired = pd.concat(parts, ignore_index=True)
    if len(paired) != len(queries) or paired.duplicated(["month", "ml_num"]).any():
        raise RuntimeError("R158 paired predictions do not cover unique target rows")
    r157 = _helpers()
    score, checks = r157._score(paired, months, stage)
    passed = all(checks.values())
    status = ("screen_passed" if passed else "screen_failed") if stage == "screen" else (
        "development_complete" if passed else "development_failed")
    paired_path = output / "paired_predictions.csv"
    profiles_path = output / "calibration_profiles.json"
    result_path = output / ("screen_results.json" if stage == "screen" else "development_results.json")
    manifest_path = output / "run_manifest.json"
    paired.to_csv(paired_path, index=False)
    _write_json(profiles_path, profiles)
    result = {"campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id, "card": CARD,
              "stage": stage, "track": TRACK, "months": list(months), "fit_starts": len(months),
              "target_mdape_strictly_below_percent": 4.0, "tracks": {TRACK: score},
              "gate_checks": checks, "qualifies_by_track": {TRACK: passed},
              "reserved_months_labels_opened": False, "hidden_labels_opened": False,
              "reserve_validation_eligible": False,
              "safe_oof_months_read": input_identity["safe_oof_months_used"],
              "metadata_months_read": input_identity["metadata_months_read"]}
    _write_json(result_path, result)
    manifest = {"campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id, "card": CARD,
                "stage": stage, "months": list(months), "source_identity": card["source_identity"],
                "code_sha256": _code_hashes(), "data_sha256": card["data_sha256"],
                "parent_artifact_hashes": card["parent_artifact_hashes"], "input_identity": input_identity,
                "input_hashes": input_hashes, "fit_starts": len(months), "fit_details": profiles,
                "calibration_profiles_sha256": _sha256(profiles_path),
                "paired_predictions_sha256": _sha256(paired_path), "result_sha256": _sha256(result_path),
                "reserved_months_labels_opened": False, "hidden_labels_opened": False,
                "reserve_validation_eligible": False}
    _write_json(manifest_path, manifest)
    report = ["# R158 — strict-past median ratio shift", "", f"Run: {run_id}; stage: {stage}; status: {status}.",
              "Each target fold receives one median log(sold/R35 prediction) correction from strictly earlier R35 OOF rows.",
              f"Pooled MdAPE: {score['parent']['mdape_percent']:.7f}% -> {score['candidate']['mdape_percent']:.7f}%.",
              f"Relative MdAPE improvement: {score['relative_mdape_improvement_percent']:.4f}%; MAPE change: {score['mape_relative_change_percent']:.4f}%.",
              f"Gate checks: {checks}.", "Reserved and hidden labels were not opened."]
    (output / "research_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    card = campaign.PROTOCOL["cards"][CARD]
    card["status"] = status
    card[f"{stage}_run_id"] = run_id
    card[f"{stage}_fit_starts"] = len(months)
    card[f"{stage}_result_path"] = _relative(result_path)
    card[f"{stage}_result_sha256"] = _sha256(result_path)
    card[f"{stage}_paired_predictions_path"] = _relative(paired_path)
    card[f"{stage}_paired_predictions_sha256"] = _sha256(paired_path)
    card[f"{stage}_manifest_path"] = _relative(manifest_path)
    card[f"{stage}_manifest_sha256"] = _sha256(manifest_path)
    card[f"{stage}_metrics"] = score
    card[f"{stage}_gate_checks"] = checks
    card[f"{stage}_qualifies"] = passed
    _write_json(campaign.PROTOCOL_PATH, campaign.PROTOCOL)
    campaign._append({"event": "card_complete", "card": CARD, "run_id": run_id, "stage": stage,
                      "fit_starts": len(months), "qualifies_by_track": {TRACK: passed},
                      "result_path": _relative(result_path), "reserved_months_labels_opened": False,
                      "hidden_labels_opened": False})
    campaign._write_progress(f"r158_{stage}_{'passed' if passed else 'failed'}",
                             next_action=("independently audit R158 before deciding whether to run development"
                                          if stage == "screen" and passed else
                                          "retain R35 and register a distinct method after R158 fails screening"
                                          if stage == "screen" else
                                          "independently audit R158 development before any parent update"),
                             r158_card=CARD, r158_run_id=run_id,
                             r158_result=_relative(result_path), r158_fit_starts=len(months))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--register-only", action="store_true")
    parser.add_argument("--stage", choices=["screen", "development"])
    args = parser.parse_args()
    if args.register_only:
        print(json.dumps(register_card(), indent=2, ensure_ascii=False), flush=True)
    elif not args.stage:
        parser.error("--stage is required unless --register-only is used")
    else:
        print(json.dumps(run(args.stage), indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
