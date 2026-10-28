"""Independent post-run reconstruction and audit for R158."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.bill_target74 import runner as campaign


CARD = "R158_listaware_r35_past_oof_median_ratio_shift"
TRACK = "list_aware"
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20261158
SCREEN_MONTHS = tuple(campaign.PROTOCOL["screen_months"])
DEVELOPMENT_MONTHS = tuple(campaign.PROTOCOL["development_months"])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return str(path.relative_to(campaign.ROOT)).replace("\\", "/")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _same(actual, expected, path: str = "root") -> None:
    if isinstance(expected, dict):
        _require(isinstance(actual, dict) and set(actual) == set(expected), f"{path}: keys differ")
        for key in expected:
            _same(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        _require(isinstance(actual, list) and len(actual) == len(expected), f"{path}: lengths differ")
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            _same(left, right, f"{path}[{index}]")
    elif isinstance(expected, (float, int)) and not isinstance(expected, bool):
        _require(isinstance(actual, (float, int)) and np.isclose(
            float(actual), float(expected), rtol=1e-12, atol=1e-12), f"{path}: values differ")
    else:
        _require(actual == expected, f"{path}: {actual!r} != {expected!r}")


def _audit(run_id: str) -> dict:
    from experiments.bill_target74 import r04_ablation as identity_helpers
    from experiments.bill_target74 import r157_listaware_isotonic_calibration as r157
    from experiments.bill_target74 import r158_listaware_median_ratio_shift as r158
    from experiments.bill_target74 import r22_remarks_tfidf as r22

    _require(run_id.startswith(("r158s_", "r158d_")), "unexpected R158 run ID")
    stage = "screen" if run_id.startswith("r158s_") else "development"
    months = SCREEN_MONTHS if stage == "screen" else DEVELOPMENT_MONTHS
    root = campaign.CAMPAIGN_ROOT / "batches" / run_id
    result_path = root / ("screen_results.json" if stage == "screen" else "development_results.json")
    manifest_path = root / "run_manifest.json"
    paired_path = root / "paired_predictions.csv"
    profiles_path = root / "calibration_profiles.json"
    _require(all(path.is_file() for path in (result_path, manifest_path, paired_path, profiles_path)),
             "R158 run is missing a required artifact")
    card = campaign.PROTOCOL["cards"].get(CARD)
    _require(bool(card), "R158 preregistration card is missing")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(result.get("campaign_id") == campaign.PROTOCOL["campaign_id"], "campaign ID mismatch")
    _require(result.get("run_id") == manifest.get("run_id") == run_id, "R158 run ID mismatch")
    _require(result.get("card") == manifest.get("card") == CARD and result.get("stage") == stage
             and manifest.get("stage") == stage, "R158 card or stage mismatch")
    _require(result.get("months") == manifest.get("months") == list(months), "R158 month inventory mismatch")
    _require(result.get("fit_starts") == len(months) == manifest.get("fit_starts"), "R158 fit count mismatch")
    _require(result.get("reserved_months_labels_opened") is False
             and result.get("hidden_labels_opened") is False, "R158 result reports restricted-label access")
    _require(manifest.get("reserved_months_labels_opened") is False
             and manifest.get("hidden_labels_opened") is False
             and manifest.get("reserve_validation_eligible") is False, "R158 manifest flags are invalid")
    _require(result.get("reserve_validation_eligible") is False, "R158 result has reserve eligibility")
    _require(manifest.get("result_sha256") == _sha256(result_path), "R158 result hash mismatch")
    _require(manifest.get("paired_predictions_sha256") == _sha256(paired_path), "R158 paired hash mismatch")
    _require(manifest.get("calibration_profiles_sha256") == _sha256(profiles_path), "R158 profile hash mismatch")
    _require(manifest.get("code_sha256") == card.get("code_sha256") == r158._code_hashes(),
             "R158 source hashes differ from preregistration")
    for relative, expected in card["code_sha256"].items():
        path = campaign.ROOT / relative
        _require(path.is_file() and _sha256(path) == expected, f"R158 source hash mismatch: {relative}")
    _require(manifest.get("data_sha256") == card.get("data_sha256")
             == campaign.PROTOCOL["data_sha256"], "R158 data hash mismatch")
    _require(manifest.get("parent_artifact_hashes") == card.get("parent_artifact_hashes"),
             "R158 parent hashes differ from preregistration")
    identity = identity_helpers._source_identity()
    _require(manifest.get("source_identity") == card.get("source_identity") == identity,
             "R158 live source identity changed")

    r44 = r157._verified_r44_source()
    input_hashes = {"r44_result": _sha256(r44[2]), "r44_manifest": _sha256(r44[3]),
                    "r44_past_oof_predictions": _sha256(r44[4])}
    _require(input_hashes == card.get("input_hashes") == manifest.get("input_hashes"),
             "R158 R44 source hashes differ from preregistration")
    parent = r157._verified_parent()
    _require(r157._parent_artifact_hashes(parent) == card.get("parent_artifact_hashes"),
             "R35 parent artifacts changed")

    # The shared loader performs per-month date-filtered safe reads and verifies R44 labels,
    # prediction alignment, and R35 parent IDs. The audit below independently rebuilds R158.
    oof, queries, input_identity = r157._load_inputs(stage, parent, r44)
    _require(manifest.get("input_identity") == input_identity, "R158 input identity changed")
    _require(result.get("safe_oof_months_read") == input_identity["safe_oof_months_used"],
             "R158 result safe OOF month inventory differs")
    _require(result.get("metadata_months_read") == input_identity["metadata_months_read"],
             "R158 result safe metadata month inventory differs")
    _require(not set(input_identity["safe_oof_months_used"]).intersection(campaign.PROTOCOL["reserved_months"]),
             "R158 used reserved OOF history")

    paired = pd.read_csv(paired_path, dtype={"ml_num": str})
    profiles = json.loads(profiles_path.read_text(encoding="utf-8"))
    required = {"month", "track", "ml_num", "sold_price", "list_price", "parent_prediction",
                "candidate_prediction", "median_log_ratio_correction", "history_oof_rows", "history_months"}
    _require(required.issubset(paired.columns), "R158 paired file lacks expected columns")
    paired["month"] = paired.month.astype(str)
    paired["ml_num"] = paired.ml_num.astype(str)
    _require(set(paired.month.unique()) == set(months) and paired.track.eq(TRACK).all(),
             "R158 paired month or track mismatch")
    _require(not paired.duplicated(["month", "ml_num"]).any(), "R158 paired keys are duplicated")
    _require(set(profiles) == set(months), "R158 profile months mismatch")

    checked = {}
    for month in months:
        history = oof.loc[oof.month.lt(month)].copy()
        current = queries.loc[queries.month.eq(month)].copy()
        _require(not history.empty and not current.empty, f"R158 history/evaluation is empty for {month}")
        history_months = sorted(history.month.unique().tolist())
        _require(all(past < month for past in history_months), f"R158 history is not strictly past for {month}")
        residual = np.log(history.sold_price_source.to_numpy(float)
                          / history.variant_prediction.to_numpy(float))
        expected_offset = float(np.median(residual))
        membership = history[["month", "ml_num", "variant_prediction", "sold_price_source", "list_price"]]
        membership = membership.sort_values(["month", "ml_num"], kind="mergesort")
        membership_hash = hashlib.sha256(
            pd.util.hash_pandas_object(membership, index=False).to_numpy().tobytes()
        ).hexdigest()
        expected_candidate = np.rint(current.parent_prediction.to_numpy(float) * np.exp(expected_offset))
        target_rows = paired.loc[paired.month.eq(month)].copy().set_index("ml_num", drop=False)
        current = current.copy()
        current["ml_num"] = current.ml_num.astype(str)
        current = current.set_index("ml_num", drop=False)
        _require(set(target_rows.index) == set(current.index), f"R158 target IDs differ for {month}")
        target_rows = target_rows.loc[current.index]
        _require(np.array_equal(target_rows.sold_price.to_numpy(float), current.sold_price.to_numpy(float)),
                 f"R158 target labels differ for {month}")
        _require(np.array_equal(target_rows.list_price.to_numpy(float), current.list_price.to_numpy(float)),
                 f"R158 target list prices differ for {month}")
        _require(np.array_equal(target_rows.parent_prediction.to_numpy(float), current.parent_prediction.to_numpy(float)),
                 f"R158 parent predictions differ for {month}")
        _require(np.array_equal(target_rows.candidate_prediction.to_numpy(float), expected_candidate),
                 f"R158 candidate predictions do not reconstruct for {month}")
        _require(np.allclose(target_rows.median_log_ratio_correction.to_numpy(float), expected_offset,
                             rtol=0.0, atol=1e-12), f"R158 correction differs for {month}")
        _require(np.array_equal(target_rows.history_oof_rows.to_numpy(int), np.full(len(target_rows), len(history))),
                 f"R158 history row count differs for {month}")
        _require(set(target_rows.history_months.astype(str)) == {"|".join(history_months)},
                 f"R158 history month record differs for {month}")
        profile = profiles[month]
        _require(profile.get("fit_id") == f"{campaign.PROTOCOL['campaign_id']}:{CARD}:{stage}:{month}:{run_id}",
                 f"R158 fit ID differs for {month}")
        _require(profile.get("history_rows") == len(history)
                 and profile.get("history_months") == history_months
                 and profile.get("training_membership_sha256") == membership_hash,
                 f"R158 profile history provenance differs for {month}")
        _require(np.isclose(profile.get("median_log_residual"), expected_offset, rtol=0.0, atol=1e-12),
                 f"R158 saved median differs for {month}")
        checked[month] = {"rows": int(len(target_rows)), "history_rows": int(len(history)),
                          "history_months": history_months, "median_log_residual": expected_offset,
                          "strict_past_verified": True}
    _require(len(paired) == len(queries), "R158 paired predictions do not cover all safe target rows")

    ledger = [json.loads(line) for line in campaign.LEDGER.read_text(encoding="utf-8").splitlines() if line.strip()]
    starts = [item for item in ledger if item.get("event") == "fit_start" and item.get("card") == CARD
              and item.get("run_id") == run_id]
    completes = [item for item in ledger if item.get("event") == "fit_complete" and item.get("card") == CARD
                 and item.get("run_id") == run_id]
    _require(len(starts) == len(completes) == len(months), "R158 ledger fit counts differ")
    _require({item.get("fit_id") for item in starts} == {item.get("fit_id") for item in completes}
             == {profiles[month]["fit_id"] for month in months}, "R158 fit IDs do not reconcile")
    _require(all(item.get("status") == "complete" for item in completes), "R158 fit did not complete")
    _require(all(item.get("reserved_months_labels_opened") is False
                 and item.get("hidden_labels_opened") is False for item in starts + completes),
             "R158 ledger reports restricted-label access")

    score_frame = paired[["month", "ml_num", "sold_price", "parent_prediction", "candidate_prediction"]].copy()
    score = r22.score_development_confirmation(
        score_frame, months=months, target_mdape_strictly_below=4.0,
        bootstrap_replicates=BOOTSTRAP_REPLICATES, bootstrap_seed=BOOTSTRAP_SEED,
        **({
            "minimum_relative_gain_percent": campaign.PROTOCOL["gates"]["screen_relative_mdape_improvement_percent"],
            "minimum_months_improved": len(months), "maximum_worst_month_regression_percent": 0.0,
            "maximum_mape_regression_percent": campaign.PROTOCOL["gates"]["screen_mape_regression_max_percent"],
        } if stage == "screen" else {
            "minimum_relative_gain_percent": campaign.PROTOCOL["gates"]["development_relative_mdape_improvement_percent"],
            "minimum_months_improved": campaign.PROTOCOL["gates"]["development_months_improved_min"],
            "maximum_worst_month_regression_percent": campaign.PROTOCOL["gates"]["development_worst_month_regression_max_percent"],
            "maximum_mape_regression_percent": campaign.PROTOCOL["gates"]["development_mape_regression_max_percent"],
        }),
    )
    _same(result.get("tracks", {}).get(TRACK), score, "track_metrics")
    gates = campaign.PROTOCOL["gates"]
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
    _same(result.get("gate_checks"), checks, "gate_checks")
    passed = all(checks.values())
    _require(result.get("qualifies_by_track", {}).get(TRACK) is passed, "R158 qualification outcome differs")
    status = ("screen_passed" if passed else "screen_failed") if stage == "screen" else (
        "development_complete" if passed else "development_failed")
    _require(card.get("status") == status, "R158 protocol status differs from independent score")

    summary = {"status": "passed", "audit_tool_path": _relative(Path(__file__)),
               "audit_tool_sha256": _sha256(Path(__file__)), "run_id": run_id, "stage": stage,
               "rows": int(len(paired)), "fit_starts": len(starts), "parent_card": "R35_listaware_detached_r22_text_expert",
               "pooled_parent_mdape_percent": score["parent"]["mdape_percent"],
               "pooled_candidate_mdape_percent": score["candidate"]["mdape_percent"],
               "relative_mdape_improvement_percent": score["relative_mdape_improvement_percent"],
               "target_met": bool(score["target_met"]), "gate_checks": checks, "months": checked,
               "reserved_months_labels_opened": False, "hidden_labels_opened": False}
    audit_path = root / "independent_audit.json"
    audit_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    card[f"{stage}_audit_status"] = "passed"
    card[f"{stage}_audit_path"] = _relative(audit_path)
    card[f"{stage}_audit_sha256"] = _sha256(audit_path)
    campaign.PROTOCOL["cards"][CARD] = card
    campaign.PROTOCOL_PATH.write_text(json.dumps(campaign.PROTOCOL, indent=2, ensure_ascii=False) + "\n",
                                      encoding="utf-8")
    next_action = ("run R158 eight-month development only because the screen passed independent audit"
                   if stage == "screen" and passed else
                   "retain validated R35; R158 screen did not qualify for development"
                   if stage == "screen" else
                   "review independently audited R158 development results; fixed-origin confirmation remains required")
    campaign._write_progress(f"r158_{stage}_audit_passed", next_action=next_action,
                             r158_audit=_relative(audit_path))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Independently reconstruct and audit an R158 run")
    parser.add_argument("run_id")
    args = parser.parse_args()
    print(json.dumps(_audit(args.run_id), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
