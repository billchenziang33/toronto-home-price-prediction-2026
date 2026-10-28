"""Independent reconstruction audit for R160 listing-quarter calibration."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.bill_target74 import runner as campaign


CARD = "R160_listaware_r35_listing_quarter_median_residual"
TRACK = "list_aware"
SCREEN_MONTHS = tuple(campaign.PROTOCOL["screen_months"])
DEVELOPMENT_MONTHS = tuple(campaign.PROTOCOL["development_months"])
RESERVED_MONTHS = set(campaign.PROTOCOL["reserved_months"])
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20261160
MIN_GROUP_HISTORY_ROWS = 50
SHRINKAGE_PRIOR_ROWS = 200.0


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


def _audit(run_id: str) -> dict:
    from experiments.bill_target74 import r04_ablation as identity_helpers
    from experiments.bill_target74 import r157_listaware_isotonic_calibration as r157
    from experiments.bill_target74 import r160_listaware_listing_quarter_calibration as r160
    from experiments.bill_target74 import r22_remarks_tfidf as r22

    _require(run_id.startswith(("r160s_", "r160d_")), "unexpected R160 run ID")
    stage = "screen" if run_id.startswith("r160s_") else "development"
    months = SCREEN_MONTHS if stage == "screen" else DEVELOPMENT_MONTHS
    root = campaign.CAMPAIGN_ROOT / "batches" / run_id
    result_path = root / ("screen_results.json" if stage == "screen" else "development_results.json")
    manifest_path, paired_path = root / "run_manifest.json", root / "paired_predictions.csv"
    profile_path = root / "calibration_profiles.json"
    _require(all(path.is_file() for path in (result_path, manifest_path, paired_path, profile_path)),
             "R160 required artifact is missing")
    card = campaign.PROTOCOL["cards"].get(CARD)
    _require(bool(card), "R160 protocol card is missing")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(result.get("campaign_id") == campaign.PROTOCOL["campaign_id"], "campaign ID mismatch")
    _require(result.get("run_id") == manifest.get("run_id") == run_id, "run ID mismatch")
    _require(result.get("card") == manifest.get("card") == CARD and result.get("stage") == stage
             and manifest.get("stage") == stage, "R160 card or stage mismatch")
    _require(result.get("months") == manifest.get("months") == list(months), "R160 months mismatch")
    _require(result.get("fit_starts") == manifest.get("fit_starts") == len(months), "R160 fit count mismatch")
    for artifact in (result, manifest):
        _require(artifact.get("reserved_months_labels_opened") is False
                 and artifact.get("hidden_labels_opened") is False, "restricted-label access flag is not false")
        _require(artifact.get("reserve_validation_eligible") is False, "reserve validation eligibility must stay false")
    _require(manifest.get("result_sha256") == _sha256(result_path)
             and manifest.get("paired_predictions_sha256") == _sha256(paired_path)
             and manifest.get("calibration_profiles_sha256") == _sha256(profile_path),
             "artifact hash mismatch")
    _require(card.get("code_sha256") == manifest.get("code_sha256") == r160._code_hashes(),
             "R160 source hashes differ from registration")
    for relative, expected in card["code_sha256"].items():
        path = campaign.ROOT / relative
        _require(path.is_file() and _sha256(path) == expected, f"source hash mismatch: {relative}")
    _require(manifest.get("data_sha256") == card.get("data_sha256") == campaign.PROTOCOL["data_sha256"],
             "dataset hash mismatch")
    _require(manifest.get("source_identity") == card.get("source_identity")
             == identity_helpers._source_identity(), "source identity changed")

    r44 = r157._verified_r44_source()
    input_hashes = {"r44_result": _sha256(r44[2]), "r44_manifest": _sha256(r44[3]),
                    "r44_past_oof_predictions": _sha256(r44[4])}
    _require(input_hashes == card.get("input_hashes") == manifest.get("input_hashes"),
             "R44 input hashes differ")
    parent = r157._verified_parent()
    _require(r157._parent_artifact_hashes(parent) == card.get("parent_artifact_hashes"),
             "R35 parent artifacts changed")

    # The frozen loader uses exact safe-month date filters and joins listing dates by unique IDs.
    # This auditor independently reconstructs every R160 group correction and target prediction.
    oof, queries, input_identity = r160._load_inputs(stage, parent, r44)
    _require(manifest.get("input_identity") == input_identity, "R160 input identity differs")
    _require(result.get("safe_oof_months_read") == input_identity["safe_oof_months_used"]
             and result.get("metadata_months_read") == input_identity["metadata_months_read"]
             and result.get("listing_quarter_months_read") == input_identity["listing_quarter_months_read"],
             "safe-month inventory differs")
    _require(not set(input_identity["safe_oof_months_used"]).intersection(RESERVED_MONTHS),
             "R160 OOF inventory contains a reserved month")

    paired = pd.read_csv(paired_path, dtype={"ml_num": str})
    profiles = json.loads(profile_path.read_text(encoding="utf-8"))
    required = {"month", "track", "ml_num", "listing_quarter", "sold_price", "list_price",
                "parent_prediction", "candidate_prediction", "log_ratio_correction",
                "history_oof_rows", "history_months"}
    _require(required.issubset(paired.columns), "R160 paired predictions lack required fields")
    paired["month"] = paired.month.astype(str)
    paired["ml_num"] = paired.ml_num.astype(str)
    _require(set(paired.month.unique()) == set(months) and paired.track.eq(TRACK).all(),
             "paired month or track mismatch")
    _require(not paired.duplicated(["month", "ml_num"]).any(), "paired listing keys are duplicated")
    _require(set(profiles) == set(months), "R160 profile folds mismatch")
    checked = {}
    for month in months:
        history = oof.loc[oof.month.lt(month)].copy()
        current = queries.loc[queries.month.eq(month)].copy()
        _require(not history.empty and not current.empty, f"empty history or target fold: {month}")
        history_months = sorted(history.month.unique().tolist())
        _require(all(past < month for past in history_months)
                 and not set(history_months).intersection(RESERVED_MONTHS),
                 f"strict-past or reserved-month violation for {month}")
        history["log_residual"] = np.log(history.sold_price_source.to_numpy(float)
                                         / history.variant_prediction.to_numpy(float))
        members = history[["month", "ml_num", "listing_quarter", "variant_prediction",
                           "sold_price_source", "list_price"]].sort_values(
                               ["month", "ml_num"], kind="mergesort")
        membership_hash = hashlib.sha256(
            pd.util.hash_pandas_object(members, index=False).to_numpy().tobytes()
        ).hexdigest()
        groups, correction_map = {}, {}
        for quarter in sorted(current.listing_quarter.unique().tolist()):
            residuals = history.loc[history.listing_quarter.eq(quarter), "log_residual"].to_numpy(float)
            n = len(residuals)
            median = float(np.median(residuals)) if n else 0.0
            weight = float(n / (n + SHRINKAGE_PRIOR_ROWS)) if n else 0.0
            correction = float(weight * median) if n >= MIN_GROUP_HISTORY_ROWS else 0.0
            groups[str(quarter)] = {
                "training_rows": int(n), "median_log_residual": median,
                "shrinkage_weight": weight, "correction": correction,
                "mode": "quarter_median_shrunk" if n >= MIN_GROUP_HISTORY_ROWS else "keep_parent_below_minimum_support",
            }
            correction_map[str(quarter)] = correction
        row_corrections = current.listing_quarter.map(correction_map).fillna(0.0).to_numpy(float)
        rebuilt = np.rint(current.parent_prediction.to_numpy(float) * np.exp(row_corrections))
        actual = paired.loc[paired.month.eq(month)].copy().set_index("ml_num", drop=False)
        current["ml_num"] = current.ml_num.astype(str)
        current = current.set_index("ml_num", drop=False)
        _require(set(actual.index) == set(current.index), f"target IDs differ for {month}")
        actual = actual.loc[current.index]
        for column in ("sold_price", "list_price", "parent_prediction"):
            _require(np.array_equal(actual[column].to_numpy(float), current[column].to_numpy(float)),
                     f"paired {column} differs for {month}")
        _require(np.array_equal(actual.listing_quarter.astype(str).to_numpy(),
                                 current.listing_quarter.astype(str).to_numpy()),
                 f"listing quarter differs for {month}")
        _require(np.array_equal(actual.candidate_prediction.to_numpy(float), rebuilt),
                 f"candidate predictions do not reconstruct for {month}")
        _require(np.allclose(actual.log_ratio_correction.to_numpy(float), row_corrections,
                             rtol=0.0, atol=1e-12), f"corrections differ for {month}")
        _require(np.array_equal(actual.history_oof_rows.to_numpy(int), np.full(len(actual), len(history))),
                 f"history counts differ for {month}")
        _require(set(actual.history_months.astype(str)) == {"|".join(history_months)},
                 f"history month inventory differs for {month}")
        profile = profiles[month]
        _require(profile.get("history_rows") == len(history)
                 and profile.get("history_months") == history_months
                 and profile.get("training_membership_sha256") == membership_hash,
                 f"history membership mismatch for {month}")
        _require(profile.get("groups") == groups, f"quarter correction profile mismatch for {month}")
        _require(profile.get("fit_id") == f"{campaign.PROTOCOL['campaign_id']}:{CARD}:{stage}:{month}:{run_id}",
                 f"fit ID mismatch for {month}")
        checked[month] = {"rows": int(len(actual)), "history_rows": int(len(history)),
                          "history_months": history_months, "groups": groups,
                          "strict_past_verified": True}
    _require(len(paired) == len(queries), "paired prediction count differs from target set")

    ledger = [json.loads(line) for line in campaign.LEDGER.read_text(encoding="utf-8").splitlines() if line.strip()]
    starts = [row for row in ledger if row.get("event") == "fit_start" and row.get("card") == CARD
              and row.get("run_id") == run_id]
    ends = [row for row in ledger if row.get("event") == "fit_complete" and row.get("card") == CARD
            and row.get("run_id") == run_id]
    _require(len(starts) == len(ends) == len(months), "R160 fit ledger counts differ")
    _require({row.get("fit_id") for row in starts} == {row.get("fit_id") for row in ends}
             == {profiles[month]["fit_id"] for month in months}, "R160 fit IDs do not reconcile")
    _require(all(row.get("status") == "complete" for row in ends)
             and all(row.get("reserved_months_labels_opened") is False
                     and row.get("hidden_labels_opened") is False for row in starts + ends),
             "R160 ledger contains a failure or restricted-label access")

    gates = campaign.PROTOCOL["gates"]
    kwargs = ({"minimum_relative_gain_percent": gates["screen_relative_mdape_improvement_percent"],
               "minimum_months_improved": len(months), "maximum_worst_month_regression_percent": 0.0,
               "maximum_mape_regression_percent": gates["screen_mape_regression_max_percent"]}
              if stage == "screen" else
              {"minimum_relative_gain_percent": gates["development_relative_mdape_improvement_percent"],
               "minimum_months_improved": gates["development_months_improved_min"],
               "maximum_worst_month_regression_percent": gates["development_worst_month_regression_max_percent"],
               "maximum_mape_regression_percent": gates["development_mape_regression_max_percent"]})
    score = r22.score_development_confirmation(
        paired[["month", "ml_num", "sold_price", "parent_prediction", "candidate_prediction"]].copy(),
        months=months, target_mdape_strictly_below=4.0,
        bootstrap_replicates=BOOTSTRAP_REPLICATES, bootstrap_seed=BOOTSTRAP_SEED, **kwargs)
    _require(result.get("tracks", {}).get(TRACK) == score, "independently recomputed scores differ")
    if stage == "screen":
        checks = {
            "relative_mdape_gain": score["relative_mdape_improvement_percent"] >= gates["screen_relative_mdape_improvement_percent"],
            "both_screen_months_improved": score["months_improved"] == len(months),
            "mape_guard": score["mape_relative_change_percent"] <= gates["screen_mape_regression_max_percent"],
            "paired_bootstrap_upper95_below_zero": score["bootstrap_mdape_difference_percent"]["upper95"] < 0.0,
        }
    else:
        checks = {
            "development_relative_gain": score["relative_mdape_improvement_percent"] >= gates["development_relative_mdape_improvement_percent"],
            "minimum_months_improved": score["months_improved"] >= gates["development_months_improved_min"],
            "worst_month_guard": score["worst_month_regression_percent"] <= gates["development_worst_month_regression_max_percent"],
            "mape_guard": score["mape_relative_change_percent"] <= gates["development_mape_regression_max_percent"],
            "paired_bootstrap_upper95_below_zero": score["bootstrap_mdape_difference_percent"]["upper95"] < 0.0,
            "strict_target": bool(score["target_met"]),
        }
    _require(result.get("gate_checks") == checks, "R160 gate checks differ")
    passed = all(checks.values())
    _require(result.get("qualifies_by_track", {}).get(TRACK) is passed, "promotion decision differs")
    status = ("screen_passed" if passed else "screen_failed") if stage == "screen" else (
        "development_complete" if passed else "development_failed")
    _require(card.get("status") == status, "protocol status differs")
    audit_path = root / "independent_audit.json"
    summary = {"status": "passed", "audit_tool_path": _relative(Path(__file__)),
               "audit_tool_sha256": _sha256(Path(__file__)), "run_id": run_id, "stage": stage,
               "rows": int(len(paired)), "fit_starts": len(starts),
               "pooled_parent_mdape_percent": score["parent"]["mdape_percent"],
               "pooled_candidate_mdape_percent": score["candidate"]["mdape_percent"],
               "relative_mdape_improvement_percent": score["relative_mdape_improvement_percent"],
               "target_met": bool(score["target_met"]), "gate_checks": checks,
               "months": checked, "reserved_months_labels_opened": False, "hidden_labels_opened": False}
    audit_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    card[f"{stage}_audit_status"] = "passed"
    card[f"{stage}_audit_path"] = _relative(audit_path)
    card[f"{stage}_audit_sha256"] = _sha256(audit_path)
    campaign.PROTOCOL_PATH.write_text(json.dumps(campaign.PROTOCOL, indent=2, ensure_ascii=False) + "\n",
                                      encoding="utf-8")
    campaign._write_progress(f"r160_{stage}_audit_passed",
                             next_action=("run development only after a passing screen and audit"
                                          if stage == "screen" and passed else
                                          "retain R35; R160 screen failed, do not tune these folds"
                                          if stage == "screen" else
                                          "review audited R160 development; fixed-origin confirmation remains required"),
                             r160_audit=_relative(audit_path))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Independently reconstruct and audit R160")
    parser.add_argument("run_id")
    args = parser.parse_args()
    print(json.dumps(_audit(args.run_id), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
