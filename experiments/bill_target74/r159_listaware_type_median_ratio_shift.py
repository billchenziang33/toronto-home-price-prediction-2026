"""R159: strict-past house-type median ratio correction for the R35 list-aware route."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.bill_target74 import runner as campaign


CARD = "R159_listaware_r35_type_past_oof_median_ratio_shift"
TRACK = "list_aware"
PARENT_CARD = "R35_listaware_detached_r22_text_expert"
SCREEN_MONTHS = tuple(campaign.PROTOCOL["screen_months"])
DEVELOPMENT_MONTHS = tuple(campaign.PROTOCOL["development_months"])
RESERVED_MONTHS = set(campaign.PROTOCOL["reserved_months"])
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20261159
MIN_GROUP_HISTORY_ROWS = 100


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return str(path.relative_to(campaign.ROOT)).replace("\\", "/")


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def _source_files() -> list[Path]:
    p = campaign.PACKAGE
    return [Path(__file__), p / "r159_independent_audit.py",
            p / "r157_listaware_isotonic_calibration.py", p / "r157_independent_audit.py",
            p / "r156_listaware_price_band_experts.py", p / "r44_past_oof_r35_shrinkage.py",
            p / "r22_remarks_tfidf.py", p / "runner.py"]


def _code_hashes() -> dict[str, str]:
    return {_relative(path): _sha256(path) for path in _source_files()}


def _helpers():
    from experiments.bill_target74 import r157_listaware_isotonic_calibration as r157
    return r157


def _safe_house_types(months: list[str]) -> pd.DataFrame:
    if set(months).intersection(RESERVED_MONTHS):
        raise RuntimeError("R159 attempted to read reserved-month house types")
    if _sha256(campaign.DATA) != campaign.PROTOCOL["data_sha256"]:
        raise RuntimeError("R159 data hash differs from campaign identity")
    parts = []
    for month in sorted(set(months)):
        start = pd.Timestamp(month + "-01")
        stop = start + pd.offsets.MonthBegin(1)
        part = pd.read_parquet(
            campaign.DATA, engine="pyarrow", columns=["ml_num", "date_sold", "house_type_name"],
            filters=[[ ("date_sold", ">=", start), ("date_sold", "<", stop) ]],
        )
        if part.empty:
            raise RuntimeError(f"R159 safe house-type filter returned no rows for {month}")
        part["date_sold"] = pd.to_datetime(part.date_sold, errors="raise")
        if set(part.date_sold.dt.to_period("M").astype(str).unique()) != {month}:
            raise RuntimeError(f"R159 house-type filter escaped {month}")
        part["month"] = month
        part["ml_num"] = part.ml_num.astype(str)
        part["house_type_name"] = part.house_type_name.fillna("__MISSING__").astype(str)
        parts.append(part[["month", "ml_num", "house_type_name"]])
    result = pd.concat(parts, ignore_index=True)
    if result.duplicated(["month", "ml_num"]).any():
        raise RuntimeError("R159 safe house-type metadata has duplicate listing keys")
    return result


def _load_inputs(stage: str, parent: dict | None = None, r44: tuple | None = None):
    r157 = _helpers()
    oof, queries, identity = r157._load_inputs(stage, parent, r44)
    types = _safe_house_types(identity["metadata_months_read"])
    for name, frame in (("OOF", oof), ("target", queries)):
        joined = frame.merge(types, on=["month", "ml_num"], how="left", validate="one_to_one")
        if joined.house_type_name.isna().any() or len(joined) != len(frame):
            raise RuntimeError(f"R159 {name} rows do not align with safe house-type metadata")
        if name == "OOF":
            oof = joined
        else:
            queries = joined
    identity = dict(identity)
    identity["house_type_months_read"] = identity["metadata_months_read"]
    identity["house_type_rows_read"] = int(len(types))
    identity["house_type_metadata_sha256"] = hashlib.sha256(
        pd.util.hash_pandas_object(types.sort_values(["month", "ml_num"], kind="mergesort"),
                                   index=False).to_numpy().tobytes()
    ).hexdigest()
    return oof, queries, identity


def _group_corrections(oof: pd.DataFrame, queries: pd.DataFrame, month: str) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    history = oof.loc[oof.month.lt(month)].copy()
    current = queries.loc[queries.month.eq(month)].copy()
    if history.empty or current.empty or any(past >= month for past in history.month.unique()):
        raise RuntimeError(f"R159 history or target violates strict-past support for {month}")
    history["log_residual"] = np.log(history.sold_price_source.to_numpy(float)
                                     / history.variant_prediction.to_numpy(float))
    if not np.isfinite(history.log_residual.to_numpy(float)).all():
        raise RuntimeError(f"R159 log-ratio residuals are invalid for {month}")
    membership = history[["month", "ml_num", "house_type_name", "variant_prediction",
                          "sold_price_source", "list_price"]].sort_values(
                              ["month", "ml_num"], kind="mergesort")
    membership_hash = hashlib.sha256(
        pd.util.hash_pandas_object(membership, index=False).to_numpy().tobytes()
    ).hexdigest()
    groups = {}
    for house_type in sorted(current.house_type_name.unique().tolist()):
        type_history = history.loc[history.house_type_name.eq(house_type), "log_residual"].to_numpy(float)
        if len(type_history) >= MIN_GROUP_HISTORY_ROWS:
            correction = float(np.median(type_history))
            mode = "type_median"
        else:
            correction = 0.0
            mode = "keep_parent_below_minimum_support"
        groups[str(house_type)] = {"training_rows": int(len(type_history)),
                                   "correction": correction, "mode": mode}
    return {"history_rows": int(len(history)), "history_months": sorted(history.month.unique().tolist()),
            "training_membership_sha256": membership_hash, "groups": groups}, history, current


def _preflight_support() -> dict:
    oof, queries, identity = _load_inputs("screen")
    if len(oof) != identity["metadata_rows_read"]:
        raise RuntimeError("R159 safe history count changed during type join")
    support = {month: {**_group_corrections(oof, queries, month)[0],
                       "evaluation_rows": int(queries.month.eq(month).sum()),
                       "evaluation_rows_by_type": queries.loc[queries.month.eq(month)]
                           .groupby("house_type_name").size().astype(int).to_dict()}
               for month in SCREEN_MONTHS}
    return support


def _score(paired: pd.DataFrame, months: tuple[str, ...], stage: str) -> tuple[dict, dict]:
    from experiments.bill_target74 import r22_remarks_tfidf as r22
    gates = campaign.PROTOCOL["gates"]
    frame = paired[["month", "ml_num", "sold_price", "parent_prediction", "candidate_prediction"]].copy()
    kwargs = ({"minimum_relative_gain_percent": gates["screen_relative_mdape_improvement_percent"],
               "minimum_months_improved": len(months), "maximum_worst_month_regression_percent": 0.0,
               "maximum_mape_regression_percent": gates["screen_mape_regression_max_percent"]}
              if stage == "screen" else
              {"minimum_relative_gain_percent": gates["development_relative_mdape_improvement_percent"],
               "minimum_months_improved": gates["development_months_improved_min"],
               "maximum_worst_month_regression_percent": gates["development_worst_month_regression_max_percent"],
               "maximum_mape_regression_percent": gates["development_mape_regression_max_percent"]})
    score = r22.score_development_confirmation(
        frame, months=months, target_mdape_strictly_below=4.0,
        bootstrap_replicates=BOOTSTRAP_REPLICATES, bootstrap_seed=BOOTSTRAP_SEED, **kwargs)
    if stage == "screen":
        checks = {"relative_mdape_gain": score["relative_mdape_improvement_percent"]
                  >= gates["screen_relative_mdape_improvement_percent"],
                  "both_screen_months_improved": score["months_improved"] == len(months),
                  "mape_guard": score["mape_relative_change_percent"] <= gates["screen_mape_regression_max_percent"],
                  "paired_bootstrap_upper95_below_zero": score["bootstrap_mdape_difference_percent"]["upper95"] < 0.0}
    else:
        checks = {"development_relative_gain": score["relative_mdape_improvement_percent"]
                  >= gates["development_relative_mdape_improvement_percent"],
                  "minimum_months_improved": score["months_improved"] >= gates["development_months_improved_min"],
                  "worst_month_guard": score["worst_month_regression_percent"]
                  <= gates["development_worst_month_regression_max_percent"],
                  "mape_guard": score["mape_relative_change_percent"] <= gates["development_mape_regression_max_percent"],
                  "paired_bootstrap_upper95_below_zero": score["bootstrap_mdape_difference_percent"]["upper95"] < 0.0,
                  "strict_target": bool(score["target_met"])}
    return score, checks


def register_card() -> dict:
    existing = campaign.PROTOCOL["cards"].get(CARD)
    code_hashes = _code_hashes()
    limits = campaign.PROTOCOL.get("limits", {})
    limit_keys = ["campaign_fit_starts", "campaign_minutes", "exploration_fit_starts", "exploration_minutes",
                  "final_reserve_fit_starts", "final_reserve_minutes", "batch_fit_starts", "batch_minutes",
                  "single_fit_seconds"]
    if any(limits.get(key) is not None for key in limit_keys):
        raise RuntimeError("R159 requires active unbounded campaign limits")
    r157 = _helpers()
    parent = r157._verified_parent()
    parent_hashes = r157._parent_artifact_hashes(parent)
    r44 = r157._verified_r44_source()
    from experiments.bill_target74 import r04_ablation as identity_helpers
    identity = identity_helpers._source_identity()
    support = _preflight_support()
    input_hashes = {"r44_result": _sha256(r44[2]), "r44_manifest": _sha256(r44[3]),
                    "r44_past_oof_predictions": _sha256(r44[4])}
    if existing:
        if existing.get("status") != "registered":
            raise RuntimeError(f"{CARD} already has status {existing.get('status')}; inspect it before retry")
        if (existing.get("code_sha256") != code_hashes or existing.get("input_hashes") != input_hashes
                or existing.get("parent_artifact_hashes") != parent_hashes
                or existing.get("preflight_support") != support):
            raise RuntimeError("R159 registration identity changed")
        return existing
    card = {
        "status": "registered", "stage": "fixed_two_month_screen_then_eight_month_development",
        "track": TRACK, "tracks": [TRACK], "screen_months": list(SCREEN_MONTHS),
        "development_months": list(DEVELOPMENT_MONTHS), "expected_fit_starts": len(SCREEN_MONTHS),
        "development_fit_starts_if_screen_passes": len(DEVELOPMENT_MONTHS),
        "comparison_parent": PARENT_CARD, "parent_run_id": parent["screen"]["run_id"],
        "parent_development_run_id": parent["development"]["run_id"],
        "parent_artifact_hashes": parent_hashes, "history_source_card": "R44_past_oof_r35_blend",
        "history_source_policy": "Use only verified list-aware R35 OOF labels from safe months strictly before each target month.",
        "method_id": "R159_LISTAWARE_STRICT_PAST_HOUSE_TYPE_MEDIAN_RATIO_SHIFT",
        "method": ("For each target month and target house type, compute the median strict-past OOF "
                   "log(sold_price/R35_prediction) for that type. Apply exp(median) to matching target "
                   "R35 predictions when at least 100 past OOF examples exist; otherwise preserve R35."),
        "minimum_group_history_rows": MIN_GROUP_HISTORY_ROWS,
        "fallback": "leave parent predictions unchanged for a target group with fewer than 100 strict-past OOF rows",
        "no_screen_tuning": True, "preflight_support": support, "input_hashes": input_hashes,
        "screen_gates": {"relative_mdape_gain_min_percent": campaign.PROTOCOL["gates"]["screen_relative_mdape_improvement_percent"],
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
    campaign._write_progress("r159_registered", r159_card=CARD,
                             next_action="run fixed two-fold R159 house-type median-ratio screen")
    return card


def run(stage: str) -> dict:
    card = campaign.PROTOCOL["cards"].get(CARD)
    expected_status = "registered" if stage == "screen" else "screen_passed"
    if not card or card.get("status") != expected_status:
        raise RuntimeError(f"R159 {stage} requires card status {expected_status!r}")
    if card.get("code_sha256") != _code_hashes() or card.get("data_sha256") != _sha256(campaign.DATA):
        raise RuntimeError("R159 source or data identity changed after registration")
    r157 = _helpers()
    parent = r157._verified_parent()
    if r157._parent_artifact_hashes(parent) != card["parent_artifact_hashes"]:
        raise RuntimeError("R159 R35 parent changed after registration")
    r44 = r157._verified_r44_source()
    input_hashes = {"r44_result": _sha256(r44[2]), "r44_manifest": _sha256(r44[3]),
                    "r44_past_oof_predictions": _sha256(r44[4])}
    if input_hashes != card["input_hashes"]:
        raise RuntimeError("R159 R44 OOF source changed after registration")
    oof, queries, input_identity = _load_inputs(stage, parent, r44)
    months = SCREEN_MONTHS if stage == "screen" else DEVELOPMENT_MONTHS
    run_id = ("r159s_" if stage == "screen" else "r159d_") + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = campaign.CAMPAIGN_ROOT / "batches" / run_id
    output.mkdir(parents=True, exist_ok=False)
    campaign._append({"event": "card_start", "card": CARD, "run_id": run_id, "stage": stage,
                      "track": TRACK, "months": list(months), "expected_fit_starts": len(months),
                      "reserved_months_labels_opened": False, "hidden_labels_opened": False})
    parts, profiles = [], {}
    for month in months:
        profile, history, current = _group_corrections(oof, queries, month)
        if any(past >= month for past in profile["history_months"]):
            raise RuntimeError(f"R159 history cutoff violation for {month}")
        correction_map = {group: values["correction"] for group, values in profile["groups"].items()}
        corrections = current.house_type_name.map(correction_map).fillna(0.0).to_numpy(float)
        candidate = np.rint(current.parent_prediction.to_numpy(float) * np.exp(corrections))
        if not np.isfinite(candidate).all() or (candidate <= 0.0).any():
            raise RuntimeError(f"R159 generated invalid predictions for {month}")
        fit_id = f"{campaign.PROTOCOL['campaign_id']}:{CARD}:{stage}:{month}:{run_id}"
        profile.update({"fit_id": fit_id, "evaluation_rows": int(len(current)),
                        "evaluation_rows_by_type": current.groupby("house_type_name").size().astype(int).to_dict()})
        profiles[month] = profile
        campaign._append({"event": "fit_start", "card": CARD, "run_id": run_id, "stage": stage,
                          "fit_id": fit_id, "track": TRACK, "month": month,
                          "model_family": "past_oof_house_type_median_ratio_calibration",
                          "expected_objective": "per_type_median_log_sold_over_r35_prediction",
                          "training_rows": profile["history_rows"], "training_months": profile["history_months"],
                          "training_cutoff_exclusive": f"{month}-01T00:00:00",
                          "training_membership_sha256": profile["training_membership_sha256"],
                          "evaluation_rows": int(len(current)), "group_corrections": profile["groups"],
                          "reserved_months_labels_opened": False, "hidden_labels_opened": False})
        campaign._append({"event": "fit_complete", "card": CARD, "run_id": run_id, "stage": stage,
                          "fit_id": fit_id, "track": TRACK, "month": month, "status": "complete",
                          "actual_objective": "per_type_median_log_sold_over_r35_prediction",
                          "training_rows": profile["history_rows"], "training_months": profile["history_months"],
                          "training_membership_sha256": profile["training_membership_sha256"],
                          "evaluation_rows": int(len(current)), "group_corrections": profile["groups"],
                          "reserved_months_labels_opened": False, "hidden_labels_opened": False})
        parts.append(pd.DataFrame({"month": month, "track": TRACK, "ml_num": current.ml_num.to_numpy(str),
                                   "house_type_name": current.house_type_name.to_numpy(str),
                                   "sold_price": current.sold_price.to_numpy(float),
                                   "list_price": current.list_price.to_numpy(float),
                                   "parent_prediction": current.parent_prediction.to_numpy(float),
                                   "candidate_prediction": candidate.astype(float),
                                   "log_ratio_correction": corrections,
                                   "history_oof_rows": profile["history_rows"],
                                   "history_months": "|".join(profile["history_months"])}))
    paired = pd.concat(parts, ignore_index=True)
    if len(paired) != len(queries) or paired.duplicated(["month", "ml_num"]).any():
        raise RuntimeError("R159 predictions do not cover the exact unique target rows")
    score, checks = _score(paired, months, stage)
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
              "reserve_validation_eligible": False, "safe_oof_months_read": input_identity["safe_oof_months_used"],
              "metadata_months_read": input_identity["metadata_months_read"],
              "house_type_months_read": input_identity["house_type_months_read"]}
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
    (output / "research_report.md").write_text(
        "\n".join(["# R159 — strict-past house-type median ratio shift", "",
                  f"Run: {run_id}; stage: {stage}; status: {status}.",
                  "Each target house type receives its past-only median log(sold/R35 prediction) when support is at least 100; sparse types keep R35 unchanged.",
                  f"Pooled MdAPE: {score['parent']['mdape_percent']:.7f}% -> {score['candidate']['mdape_percent']:.7f}%.",
                  f"Relative MdAPE improvement: {score['relative_mdape_improvement_percent']:.4f}%; MAPE change: {score['mape_relative_change_percent']:.4f}%.",
                  f"Gate checks: {checks}.", "Only safe, date-filtered months were read; reserve and hidden labels were not opened."]) + "\n",
        encoding="utf-8")
    card = campaign.PROTOCOL["cards"][CARD]
    card["status"] = status
    card[f"{stage}_run_id"] = run_id
    card[f"{stage}_fit_starts"] = len(months)
    for name, path in (("result", result_path), ("paired_predictions", paired_path), ("manifest", manifest_path)):
        card[f"{stage}_{name}_path"] = _relative(path)
        card[f"{stage}_{name}_sha256"] = _sha256(path)
    card[f"{stage}_metrics"] = score
    card[f"{stage}_gate_checks"] = checks
    card[f"{stage}_qualifies"] = passed
    _write_json(campaign.PROTOCOL_PATH, campaign.PROTOCOL)
    campaign._append({"event": "card_complete", "card": CARD, "run_id": run_id, "stage": stage,
                      "fit_starts": len(months), "qualifies_by_track": {TRACK: passed},
                      "result_path": _relative(result_path), "reserved_months_labels_opened": False,
                      "hidden_labels_opened": False})
    campaign._write_progress(f"r159_{stage}_{'passed' if passed else 'failed'}",
                             next_action=("independently audit R159 before any development stage"
                                          if stage == "screen" and passed else
                                          "retain R35; R159 screen failed, register a distinct method"
                                          if stage == "screen" else
                                          "independently audit R159 development before any parent update"),
                             r159_card=CARD, r159_run_id=run_id,
                             r159_result=_relative(result_path), r159_fit_starts=len(months))
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
