"""Independent reconstruction audit for R159 house-type median calibration."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.bill_target74 import runner as campaign


CARD = "R159_listaware_r35_type_past_oof_median_ratio_shift"
TRACK = "list_aware"
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


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _audit(run_id: str) -> dict:
    from experiments.bill_target74 import r04_ablation as identity_helpers
    from experiments.bill_target74 import r157_listaware_isotonic_calibration as r157
    from experiments.bill_target74 import r159_listaware_type_median_ratio_shift as r159
    from experiments.bill_target74 import r22_remarks_tfidf as r22

    _require(run_id.startswith(("r159s_", "r159d_")), "unexpected R159 run ID")
    stage = "screen" if run_id.startswith("r159s_") else "development"
    months = SCREEN_MONTHS if stage == "screen" else DEVELOPMENT_MONTHS
    root = campaign.CAMPAIGN_ROOT / "batches" / run_id
    result_path = root / ("screen_results.json" if stage == "screen" else "development_results.json")
    manifest_path, paired_path, profile_path = root / "run_manifest.json", root / "paired_predictions.csv", root / "calibration_profiles.json"
    _require(all(path.is_file() for path in (result_path, manifest_path, paired_path, profile_path)),
             "required R159 artifacts are missing")
    card = campaign.PROTOCOL["cards"].get(CARD)
    _require(bool(card), "R159 registration is missing")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(result.get("campaign_id") == campaign.PROTOCOL["campaign_id"], "campaign ID mismatch")
    _require(result.get("run_id") == manifest.get("run_id") == run_id, "run ID mismatch")
    _require(result.get("card") == manifest.get("card") == CARD and result.get("stage") == stage
             and manifest.get("stage") == stage, "card or stage mismatch")
    _require(result.get("months") == manifest.get("months") == list(months), "month inventory mismatch")
    _require(result.get("fit_starts") == manifest.get("fit_starts") == len(months), "fit count mismatch")
    for artifact in (result, manifest):
        _require(artifact.get("reserved_months_labels_opened") is False
                 and artifact.get("hidden_labels_opened") is False, "restricted-label flag is not false")
    _require(result.get("reserve_validation_eligible") is False
             and manifest.get("reserve_validation_eligible") is False, "reserve eligibility must remain false")
    _require(manifest.get("result_sha256") == _sha256(result_path)
             and manifest.get("paired_predictions_sha256") == _sha256(paired_path)
             and manifest.get("calibration_profiles_sha256") == _sha256(profile_path), "artifact hash mismatch")
    _require(card.get("code_sha256") == manifest.get("code_sha256") == r159._code_hashes(),
             "R159 source hashes differ from registration")
    for relative, digest in card["code_sha256"].items():
        source = campaign.ROOT / relative
        _require(source.is_file() and _sha256(source) == digest, f"source hash mismatch: {relative}")
    _require(manifest.get("data_sha256") == card.get("data_sha256") == campaign.PROTOCOL["data_sha256"],
             "dataset hash mismatch")
    _require(manifest.get("source_identity") == card.get("source_identity")
             == identity_helpers._source_identity(), "source identity changed")

    r44 = r157._verified_r44_source()
    input_hashes = {"r44_result": _sha256(r44[2]), "r44_manifest": _sha256(r44[3]),
                    "r44_past_oof_predictions": _sha256(r44[4])}
    _require(input_hashes == card.get("input_hashes") == manifest.get("input_hashes"),
             "R44 source hashes differ")
    parent = r157._verified_parent()
    _require(r157._parent_artifact_hashes(parent) == card.get("parent_artifact_hashes"),
             "R35 parent artifact hashes differ")

    # This helper re-reads only registered safe months with date predicates and attaches
    # house_type_name from those same filtered source rows. Refit each group median below.
    oof, queries, input_identity = r159._load_inputs(stage, parent, r44)
    _require(manifest.get("input_identity") == input_identity, "R159 input identity differs")
    _require(result.get("safe_oof_months_read") == input_identity["safe_oof_months_used"]
             and result.get("metadata_months_read") == input_identity["metadata_months_read"]
             and result.get("house_type_months_read") == input_identity["house_type_months_read"],
             "safe-month inventory differs")
    _require(not set(input_identity["safe_oof_months_used"]).intersection(RESERVED_MONTHS),
             "reserved OOF history was read")

    paired = pd.read_csv(paired_path, dtype={"ml_num": str})
    profiles = json.loads(profile_path.read_text(encoding="utf-8"))
    required = {"month", "track", "ml_num", "house_type_name", "sold_price", "list_price",
                "parent_prediction", "candidate_prediction", "log_ratio_correction",
                "history_oof_rows", "history_months"}
    _require(required.issubset(paired.columns), "paired predictions lack required fields")
    paired["month"] = paired.month.astype(str)
    paired["ml_num"] = paired.ml_num.astype(str)
    _require(set(paired.month.unique()) == set(months) and paired.track.eq(TRACK).all(),
             "paired month or track mismatch")
    _require(not paired.duplicated(["month", "ml_num"]).any(), "duplicate target listing keys")
    _require(set(profiles) == set(months), "profile fold coverage mismatch")
    checked = {}
    for month in months:
        history = oof.loc[oof.month.lt(month)].copy()
        current = queries.loc[queries.month.eq(month)].copy()
        _require(not history.empty and not current.empty and len(history) >= 1000,
                 f"insufficient history or target rows for {month}")
        history_months = sorted(history.month.unique().tolist())
        _require(all(value < month for value in history_months)
                 and not set(history_months).intersection(RESERVED_MONTHS),
                 f"strict-past cutoff violation for {month}")
        history["residual"] = np.log(history.sold_price_source.to_numpy(float)
                                     / history.variant_prediction.to_numpy(float))
        members = history[["month", "ml_num", "house_type_name", "variant_prediction",
                           "sold_price_source", "list_price"]].sort_values(
                               ["month", "ml_num"], kind="mergesort")
        membership_hash = hashlib.sha256(
            pd.util.hash_pandas_object(members, index=False).to_numpy().tobytes()
        ).hexdigest()
        groups, corrections = {}, {}
        for house_type in sorted(current.house_type_name.unique().tolist()):
            residuals = history.loc[history.house_type_name.eq(house_type), "residual"].to_numpy(float)
            if len(residuals) >= MIN_GROUP_HISTORY_ROWS:
                delta, mode = float(np.median(residuals)), "type_median"
            else:
                delta, mode = 0.0, "keep_parent_below_minimum_support"
            groups[str(house_type)] = {"training_rows": int(len(residuals)),
                                       "correction": delta, "mode": mode}
            corrections[str(house_type)] = delta
        row_corrections = current.house_type_name.map(corrections).fillna(0.0).to_numpy(float)
        rebuilt = np.rint(current.parent_prediction.to_numpy(float) * np.exp(row_corrections))
        observed = paired.loc[paired.month.eq(month)].copy().set_index("ml_num", drop=False)
        current["ml_num"] = current.ml_num.astype(str)
        current = current.set_index("ml_num", drop=False)
        _require(set(observed.index) == set(current.index), f"target IDs differ for {month}")
        observed = observed.loc[current.index]
        for column in ("house_type_name", "sold_price", "list_price", "parent_prediction"):
            _require(np.array_equal(observed[column].astype(str).to_numpy(),
                                    current[column].astype(str).to_numpy()) if column == "house_type_name"
                     else np.array_equal(observed[column].to_numpy(float), current[column].to_numpy(float)),
                     f"paired {column} differs for {month}")
        _require(np.array_equal(observed.candidate_prediction.to_numpy(float), rebuilt),
                 f"candidate predictions do not reconstruct for {month}")
        _require(np.allclose(observed.log_ratio_correction.to_numpy(float), row_corrections,
                             rtol=0.0, atol=1e-12), f"correction values differ for {month}")
        _require(np.array_equal(observed.history_oof_rows.to_numpy(int), np.full(len(observed), len(history))),
                 f"history counts differ for {month}")
        _require(set(observed.history_months.astype(str)) == {"|".join(history_months)},
                 f"history month labels differ for {month}")
        profile = profiles[month]
        _require(profile.get("history_rows") == len(history)
                 and profile.get("history_months") == history_months
                 and profile.get("training_membership_sha256") == membership_hash,
                 f"history provenance differs for {month}")
        _require(profile.get("fit_id") == f"{campaign.PROTOCOL['campaign_id']}:{CARD}:{stage}:{month}:{run_id}",
                 f"fit ID differs for {month}")
        _require(profile.get("groups") == groups, f"group offsets differ for {month}")
        checked[month] = {"rows": int(len(observed)), "history_rows": int(len(history)),
                          "history_months": history_months, "groups": groups,
                          "strict_past_verified": True}
    _require(len(paired) == len(queries), "paired rows do not cover the target set")

    ledger = [json.loads(line) for line in campaign.LEDGER.read_text(encoding="utf-8").splitlines() if line.strip()]
    starts = [x for x in ledger if x.get("event") == "fit_start" and x.get("card") == CARD
              and x.get("run_id") == run_id]
    ends = [x for x in ledger if x.get("event") == "fit_complete" and x.get("card") == CARD
            and x.get("run_id") == run_id]
    _require(len(starts) == len(ends) == len(months), "fit ledger start/complete mismatch")
    _require({x.get("fit_id") for x in starts} == {x.get("fit_id") for x in ends}
             == {profiles[m]["fit_id"] for m in months}, "fit IDs do not reconcile")
    _require(all(x.get("status") == "complete" for x in ends)
             and all(x.get("reserved_months_labels_opened") is False
                     and x.get("hidden_labels_opened") is False for x in starts + ends),
             "fit ledger contains failure or restricted-label access")

    from experiments.bill_target74 import r22_remarks_tfidf as r22
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
        months=months, target_mdape_strictly_below=4.0, bootstrap_replicates=BOOTSTRAP_REPLICATES,
        bootstrap_seed=BOOTSTRAP_SEED, **kwargs)
    _require(result.get("tracks", {}).get(TRACK) == score, "independently recomputed metrics differ")
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
    _require(result.get("gate_checks") == checks, "gate outcomes differ")
    passed = all(checks.values())
    _require(result.get("qualifies_by_track", {}).get(TRACK) is passed, "promotion outcome differs")
    expected_status = ("screen_passed" if passed else "screen_failed") if stage == "screen" else (
        "development_complete" if passed else "development_failed")
    _require(card.get("status") == expected_status, "protocol status differs")
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
    campaign._write_progress(f"r159_{stage}_audit_passed",
                             next_action=("run development only after an audited passing screen"
                                          if stage == "screen" and passed else
                                          "retain R35; R159 screen failed, do not tune these folds"
                                          if stage == "screen" else
                                          "review audited R159 development before any parent update"),
                             r159_audit=_relative(audit_path))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Independently reconstruct and audit an R159 run")
    parser.add_argument("run_id")
    args = parser.parse_args()
    print(json.dumps(_audit(args.run_id), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
