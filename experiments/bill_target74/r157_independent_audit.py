"""Independent post-run audit for the R157 isotonic calibration experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from experiments.bill_target74 import runner as campaign


CARD = "R157_listaware_r35_isotonic_ratio_calibration"
PARENT_CARD = "R35_listaware_detached_r22_text_expert"
R44_CARD = "R44_past_oof_r35_blend"
TRACK = "list_aware"
RESERVED_MONTHS = set(campaign.PROTOCOL["reserved_months"])
SCREEN_MONTHS = tuple(campaign.PROTOCOL["screen_months"])
DEVELOPMENT_MONTHS = tuple(campaign.PROTOCOL["development_months"])
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20261157


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
            float(actual), float(expected), rtol=1e-12, atol=1e-12),
            f"{path}: {actual!r} != {expected!r}")
    else:
        _require(actual == expected, f"{path}: {actual!r} != {expected!r}")


def _safe_month_metadata(months: list[str], data_path: Path) -> pd.DataFrame:
    _require(not set(months).intersection(RESERVED_MONTHS), "audit requested reserved-month labels")
    _require(_sha256(data_path) == campaign.PROTOCOL["data_sha256"], "cached dataset hash mismatch")
    parts = []
    for month in sorted(set(months)):
        start = pd.Timestamp(month + "-01")
        stop = start + pd.offsets.MonthBegin(1)
        part = pd.read_parquet(
            data_path,
            engine="pyarrow",
            columns=["ml_num", "date_sold", "sold_price", "list_price"],
            filters=[[ ("date_sold", ">=", start), ("date_sold", "<", stop) ]],
        )
        _require(not part.empty, f"safe-month date filter returned no rows for {month}")
        part["date_sold"] = pd.to_datetime(part.date_sold, errors="raise")
        _require(set(part.date_sold.dt.to_period("M").astype(str).unique()) == {month},
                 f"safe-month date filter escaped {month}")
        part["month"] = month
        part["ml_num"] = part.ml_num.astype(str)
        part["sold_price"] = pd.to_numeric(part.sold_price, errors="raise")
        part["list_price"] = pd.to_numeric(part.list_price, errors="raise")
        parts.append(part[["month", "ml_num", "date_sold", "sold_price", "list_price"]])
    metadata = pd.concat(parts, ignore_index=True)
    _require(not metadata.duplicated(["month", "ml_num"]).any(), "duplicate safe-month listing keys")
    _require(np.isfinite(metadata[["sold_price", "list_price"]].to_numpy(float)).all(),
             "safe-month source contains non-finite sale/list prices")
    _require((metadata[["sold_price", "list_price"]] > 0).all().all(),
             "safe-month source contains non-positive sale/list prices")
    return metadata


def _audit(run_id: str) -> dict:
    from experiments.bill_target74 import r04_ablation as identity_helpers
    from experiments.bill_target74 import r156_listaware_price_band_experts as parent_helpers
    from experiments.bill_target74 import r22_remarks_tfidf as r22
    from experiments.bill_target74 import r157_listaware_isotonic_calibration as r157

    _require(run_id.startswith(("r157s_", "r157d_")), "unexpected R157 run ID")
    stage = "screen" if run_id.startswith("r157s_") else "development"
    months = SCREEN_MONTHS if stage == "screen" else DEVELOPMENT_MONTHS
    root = campaign.CAMPAIGN_ROOT / "batches" / run_id
    result_path = root / ("screen_results.json" if stage == "screen" else "development_results.json")
    manifest_path = root / "run_manifest.json"
    paired_path = root / "paired_predictions.csv"
    profiles_path = root / "calibration_profiles.json"
    _require(all(path.is_file() for path in (result_path, manifest_path, paired_path, profiles_path)),
             "R157 result, manifest, paired predictions, or calibration profiles are missing")

    card = campaign.PROTOCOL["cards"].get(CARD)
    _require(bool(card), "R157 preregistration card is missing")
    expected_statuses = {"screen": {"screen_passed", "screen_failed"},
                         "development": {"development_complete", "development_failed"}}
    _require(card.get("status") in expected_statuses[stage], "R157 card status does not match the run stage")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(result.get("campaign_id") == campaign.PROTOCOL["campaign_id"], "campaign ID mismatch")
    _require(result.get("run_id") == manifest.get("run_id") == run_id, "R157 run ID mismatch")
    _require(result.get("card") == manifest.get("card") == CARD, "R157 card mismatch")
    _require(result.get("stage") == manifest.get("stage") == stage, "R157 stage mismatch")
    _require(result.get("months") == manifest.get("months", list(months)) == list(months),
             "R157 month inventory mismatch")
    _require(result.get("fit_starts") == len(months) == manifest.get("fit_starts"), "R157 fit count mismatch")
    _require(result.get("reserved_months_labels_opened") is False
             and result.get("hidden_labels_opened") is False, "R157 result reports restricted-label access")
    _require(manifest.get("reserved_months_labels_opened") is False
             and manifest.get("hidden_labels_opened") is False
             and manifest.get("reserve_validation_eligible") is False,
             "R157 manifest reports restricted-label access or eligibility")
    _require(result.get("reserve_validation_eligible") is False, "R157 result has reserve eligibility")
    _require(manifest.get("result_sha256") == _sha256(result_path), "R157 result hash mismatch")
    _require(manifest.get("paired_predictions_sha256") == _sha256(paired_path),
             "R157 paired predictions hash mismatch")
    _require(manifest.get("calibration_profiles_sha256") == _sha256(profiles_path),
             "R157 calibration profiles hash mismatch")
    _require(card.get("code_sha256") == r157._code_hashes() == manifest.get("code_sha256"),
             "R157 preregistered/current/manifest source hashes differ")
    for relative, expected_hash in card["code_sha256"].items():
        source_path = campaign.ROOT / relative
        _require(source_path.is_file() and _sha256(source_path) == expected_hash,
                 f"R157 source hash mismatch: {relative}")
    _require(manifest.get("data_sha256") == card.get("data_sha256")
             == campaign.PROTOCOL["data_sha256"], "R157 data hash mismatch")
    _require(manifest.get("parent_artifact_hashes") == card.get("parent_artifact_hashes"),
             "R157 parent hashes differ from preregistration")

    parent = r157._verified_parent()
    _require(parent_helpers._parent_artifact_hashes(parent) == card["parent_artifact_hashes"],
             "live R35 parent artifacts changed")
    identity = identity_helpers._source_identity()
    _require(manifest.get("source_identity") == card.get("source_identity") == identity,
             "R157 source identity changed")
    _require(identity.get("data_sha256") == campaign.PROTOCOL["data_sha256"], "live dataset identity mismatch")

    r44_result, r44_manifest, r44_result_path, r44_manifest_path, oof_path = r157._verified_r44_source()
    input_hashes = {
        "r44_result": _sha256(r44_result_path),
        "r44_manifest": _sha256(r44_manifest_path),
        "r44_past_oof_predictions": _sha256(oof_path),
    }
    _require(input_hashes == card.get("input_hashes") == manifest.get("input_hashes"),
             "R157 R44 input hashes differ from registration")
    safe_oof_months = sorted(month for month in r44_result["safe_oof_months_read"] if month <= max(months))
    _require(not set(safe_oof_months).intersection(RESERVED_MONTHS), "R157 OOF history includes reserved months")
    _require(set(months).issubset(safe_oof_months), "R44 OOF inventory lacks target-fold predictions")
    oof = pd.read_csv(oof_path, dtype={"ml_num": str})
    oof["month"] = oof.month.astype(str)
    oof = oof.loc[oof.month.isin(safe_oof_months)].copy()
    _require(set(oof.month.unique()) == set(safe_oof_months), "R44 OOF rows do not match safe inventory")
    _require(not oof.duplicated(["month", "ml_num"]).any(), "R44 OOF rows have duplicate keys")
    _require(oof.track.dropna().eq(TRACK).all(), "R44 OOF history contains another prediction route")
    _require(np.isfinite(oof[["sold_price", "variant_prediction"]].to_numpy(float)).all(),
             "R44 OOF contains non-finite labels or predictions")

    metadata_months = sorted(set(safe_oof_months).union(months))
    metadata = _safe_month_metadata(metadata_months, campaign.DATA)
    oof = oof.merge(metadata[["month", "ml_num", "sold_price", "list_price"]],
                     on=["month", "ml_num"], how="left", validate="one_to_one",
                     suffixes=("_oof", "_source"))
    _require(oof.list_price.notna().all() and oof.sold_price_source.notna().all(),
             "R44 OOF rows lack filtered source metadata")
    _require(np.array_equal(oof.sold_price_oof.to_numpy(float), oof.sold_price_source.to_numpy(float)),
             "R44 OOF labels differ from filtered source labels")
    _require(manifest.get("input_identity", {}).get("safe_oof_months_used") == safe_oof_months,
             "R157 manifest OOF month inventory differs")
    _require(manifest.get("input_identity", {}).get("metadata_months_read") == metadata_months,
             "R157 manifest safe metadata inventory differs")
    _require(result.get("safe_oof_months_read") == safe_oof_months
             and result.get("metadata_months_read") == metadata_months,
             "R157 result safe-month inventory differs")

    parent_rows = parent["screen_rows"] if stage == "screen" else parent["development_rows"]
    parent_rows = parent_rows.loc[parent_rows.track.eq(TRACK),
                                  ["month", "ml_num", "sold_price", "candidate_prediction"]].copy()
    parent_rows["month"] = parent_rows.month.astype(str)
    parent_rows["ml_num"] = parent_rows.ml_num.astype(str)
    parent_rows = parent_rows.rename(columns={"sold_price": "sold_price_parent",
                                              "candidate_prediction": "parent_prediction"})
    source_metadata = metadata[["month", "ml_num", "sold_price", "list_price"]].rename(
        columns={"sold_price": "sold_price_source"}
    )
    parent_rows = parent_rows.merge(source_metadata, on=["month", "ml_num"],
                                    how="left", validate="one_to_one")
    _require(parent_rows.sold_price_source.notna().all() and parent_rows.list_price.notna().all(),
             "R35 target rows lack filtered source metadata")
    _require(np.array_equal(parent_rows.sold_price_parent.to_numpy(float),
                             parent_rows.sold_price_source.to_numpy(float)),
             "R35 target labels differ from filtered source labels")
    parent_rows = parent_rows.sort_values(["month", "ml_num"], kind="mergesort").reset_index(drop=True)

    paired = pd.read_csv(paired_path, dtype={"ml_num": str})
    required = {"month", "track", "ml_num", "sold_price", "list_price", "parent_prediction",
                "candidate_prediction", "calibrated_log_sale_list_ratio", "history_oof_rows", "history_months"}
    _require(required.issubset(paired.columns), "R157 paired file lacks required columns")
    paired["month"] = paired.month.astype(str)
    paired["ml_num"] = paired.ml_num.astype(str)
    _require(set(paired.month.unique()) == set(months) and paired.track.eq(TRACK).all(),
             "R157 paired month/track mismatch")
    _require(not paired.duplicated(["month", "ml_num"]).any(), "R157 paired keys are duplicated")
    _require(np.isfinite(paired[["sold_price", "list_price", "parent_prediction", "candidate_prediction",
                                 "calibrated_log_sale_list_ratio"]].to_numpy(float)).all(),
             "R157 paired predictions contain non-finite values")

    profiles = json.loads(profiles_path.read_text(encoding="utf-8"))
    _require(set(profiles) == set(months), "R157 calibration profile coverage mismatch")
    audit_months = {}
    for month in months:
        history = oof.loc[oof.month.lt(month)].copy()
        target = parent_rows.loc[parent_rows.month.eq(month)].copy()
        _require(len(history) >= int(card["calibration"]["minimum_past_oof_rows"]),
                 f"R157 history support below registered minimum for {month}")
        history_months = sorted(history.month.unique().tolist())
        _require(bool(history_months) and all(past < month for past in history_months),
                 f"R157 history is not strictly past for {month}")
        _require(not set(history_months).intersection(RESERVED_MONTHS),
                 f"R157 history contains reserved labels for {month}")
        _require(not target.empty, f"R35 target fold is empty for {month}")
        history["ml_num"] = history.ml_num.astype(str)
        membership = history[["month", "ml_num", "variant_prediction", "sold_price_source", "list_price"]]
        membership = membership.sort_values(["month", "ml_num"], kind="mergesort")
        membership_hash = hashlib.sha256(
            pd.util.hash_pandas_object(membership, index=False).to_numpy().tobytes()
        ).hexdigest()
        x_history = np.log(history.variant_prediction.to_numpy(float) / history.list_price.to_numpy(float))
        y_history = np.log(history.sold_price_source.to_numpy(float) / history.list_price.to_numpy(float))
        _require(target.sold_price_source.notna().all() and target.list_price.notna().all(),
                 f"R35 target fold lacks sale/list metadata for {month}")
        x_target = np.log(target.parent_prediction.to_numpy(float) / target.list_price.to_numpy(float))
        model = IsotonicRegression(increasing=True, out_of_bounds="clip", y_min=None, y_max=None)
        model.fit(x_history, y_history)
        calibrated = model.predict(x_target)
        reconstructed = np.rint(target.list_price.to_numpy(float) * np.exp(calibrated))
        month_paired = paired.loc[paired.month.eq(month)].copy()
        month_paired = month_paired.set_index("ml_num", drop=False)
        target = target.set_index("ml_num", drop=False)
        _require(set(month_paired.index) == set(target.index), f"R157 target IDs mismatch for {month}")
        target = target.loc[month_paired.index]
        _require(np.array_equal(month_paired.parent_prediction.to_numpy(float),
                                 target.parent_prediction.to_numpy(float)),
                 f"R157 parent predictions mismatch for {month}")
        _require(np.array_equal(month_paired.sold_price.to_numpy(float), target.sold_price_source.to_numpy(float)),
                 f"R157 target labels mismatch for {month}")
        _require(np.array_equal(month_paired.list_price.to_numpy(float), target.list_price.to_numpy(float)),
                 f"R157 list prices mismatch for {month}")
        _require(np.array_equal(month_paired.candidate_prediction.to_numpy(float), reconstructed),
                 f"R157 candidate predictions do not reconstruct for {month}")
        _require(np.allclose(month_paired.calibrated_log_sale_list_ratio.to_numpy(float), calibrated,
                             rtol=0.0, atol=1e-12), f"R157 calibrated ratios differ for {month}")
        _require(np.array_equal(month_paired.history_oof_rows.to_numpy(int),
                                 np.full(len(month_paired), len(history))),
                 f"R157 history row counts differ for {month}")
        _require(set(month_paired.history_months.astype(str)) == {"|".join(history_months)},
                 f"R157 history month record differs for {month}")
        profile = profiles[month]
        _require(profile.get("training_rows") == len(history)
                 and profile.get("training_months") == history_months
                 and profile.get("training_membership_sha256") == membership_hash,
                 f"R157 profile history provenance differs for {month}")
        _require(profile.get("fit_id") == f"{campaign.PROTOCOL['campaign_id']}:{CARD}:{stage}:{month}:{run_id}",
                 f"R157 profile fit ID differs for {month}")
        _require(profile.get("isotonic_threshold_count") == len(model.X_thresholds_),
                 f"R157 isotonic support differs for {month}")
        _require(np.allclose(profile.get("isotonic_x_thresholds", []), model.X_thresholds_, rtol=1e-12, atol=1e-12)
                 and np.allclose(profile.get("isotonic_y_thresholds", []), model.y_thresholds_, rtol=1e-12, atol=1e-12),
                 f"R157 isotonic thresholds do not reconstruct for {month}")
        audit_months[month] = {"evaluation_rows": int(len(target)), "history_rows": int(len(history)),
                               "history_months": history_months, "unique_history_x": int(np.unique(x_history).size),
                               "strict_past_verified": True}

    _require(len(paired) == len(parent_rows), "R157 paired rows do not cover the exact parent target rows")
    _require(set(paired.month.astype(str) + "|" + paired.ml_num.astype(str))
             == set(parent_rows.month.astype(str) + "|" + parent_rows.ml_num.astype(str)),
             "R157 paired row keys differ from R35 parent")
    ledger = [json.loads(line) for line in campaign.LEDGER.read_text(encoding="utf-8").splitlines() if line.strip()]
    starts = [entry for entry in ledger if entry.get("event") == "fit_start" and entry.get("card") == CARD
              and entry.get("run_id") == run_id]
    completes = [entry for entry in ledger if entry.get("event") == "fit_complete" and entry.get("card") == CARD
                 and entry.get("run_id") == run_id]
    _require(len(starts) == len(completes) == len(months), "R157 ledger starts/completions mismatch")
    _require({entry.get("fit_id") for entry in starts} == {entry.get("fit_id") for entry in completes}
             == {profiles[month]["fit_id"] for month in months}, "R157 fit IDs do not reconcile")
    _require(all(entry.get("status") == "complete" for entry in completes), "R157 fit did not complete")
    _require(all(entry.get("reserved_months_labels_opened") is False
                 and entry.get("hidden_labels_opened") is False for entry in starts + completes),
             "R157 fit ledger reports restricted-label access")
    for month in months:
        start = next(entry for entry in starts if entry.get("month") == month)
        _require(start.get("training_months") == audit_months[month]["history_months"]
                 and start.get("training_rows") == audit_months[month]["history_rows"]
                 and start.get("training_membership_sha256") == profiles[month]["training_membership_sha256"],
                 f"R157 fit ledger history differs for {month}")

    from experiments.bill_target74 import r157_listaware_isotonic_calibration as r157
    score_frame = paired[["month", "ml_num", "sold_price", "parent_prediction", "candidate_prediction"]].copy()
    fresh = r22.score_development_confirmation(
        score_frame,
        months=months,
        target_mdape_strictly_below=4.0,
        bootstrap_replicates=BOOTSTRAP_REPLICATES,
        bootstrap_seed=BOOTSTRAP_SEED,
        **({
            "minimum_relative_gain_percent": campaign.PROTOCOL["gates"]["screen_relative_mdape_improvement_percent"],
            "minimum_months_improved": len(months),
            "maximum_worst_month_regression_percent": 0.0,
            "maximum_mape_regression_percent": campaign.PROTOCOL["gates"]["screen_mape_regression_max_percent"],
        } if stage == "screen" else {
            "minimum_relative_gain_percent": campaign.PROTOCOL["gates"]["development_relative_mdape_improvement_percent"],
            "minimum_months_improved": campaign.PROTOCOL["gates"]["development_months_improved_min"],
            "maximum_worst_month_regression_percent": campaign.PROTOCOL["gates"]["development_worst_month_regression_max_percent"],
            "maximum_mape_regression_percent": campaign.PROTOCOL["gates"]["development_mape_regression_max_percent"],
        }),
    )
    _same(result.get("tracks", {}).get(TRACK), fresh, "track_metrics")
    if stage == "screen":
        gates = campaign.PROTOCOL["gates"]
        fresh_gates = {
            "relative_mdape_gain": fresh["relative_mdape_improvement_percent"]
                                   >= gates["screen_relative_mdape_improvement_percent"],
            "both_screen_months_improved": fresh["months_improved"] == len(months),
            "mape_guard": fresh["mape_relative_change_percent"] <= gates["screen_mape_regression_max_percent"],
            "paired_bootstrap_upper95_below_zero": fresh["bootstrap_mdape_difference_percent"]["upper95"] < 0.0,
        }
    else:
        gates = campaign.PROTOCOL["gates"]
        fresh_gates = {
            "development_relative_gain": fresh["relative_mdape_improvement_percent"]
                                         >= gates["development_relative_mdape_improvement_percent"],
            "minimum_months_improved": fresh["months_improved"] >= gates["development_months_improved_min"],
            "worst_month_guard": fresh["worst_month_regression_percent"]
                                 <= gates["development_worst_month_regression_max_percent"],
            "mape_guard": fresh["mape_relative_change_percent"]
                          <= gates["development_mape_regression_max_percent"],
            "paired_bootstrap_upper95_below_zero": fresh["bootstrap_mdape_difference_percent"]["upper95"] < 0.0,
            "strict_target": bool(fresh["target_met"]),
        }
    _same(result.get("gate_checks"), fresh_gates, "gate_checks")
    passed = all(fresh_gates.values())
    _require(result.get("qualifies_by_track", {}).get(TRACK) is passed, "R157 gate outcome mismatch")
    expected_status = ("screen_passed" if passed else "screen_failed") if stage == "screen" else (
        "development_complete" if passed else "development_failed")
    _require(card.get("status") == expected_status, "R157 protocol status conflicts with recomputed score")

    summary = {
        "status": "passed", "audit_tool_path": _relative(Path(__file__)),
        "audit_tool_sha256": _sha256(Path(__file__)), "run_id": run_id, "stage": stage,
        "rows": int(len(paired)), "fit_starts": len(starts),
        "history_source_run_id": r44_result["run_id"],
        "parent_card": PARENT_CARD, "parent_run_id": parent["screen"]["run_id"],
        "pooled_parent_mdape_percent": fresh["parent"]["mdape_percent"],
        "pooled_candidate_mdape_percent": fresh["candidate"]["mdape_percent"],
        "relative_mdape_improvement_percent": fresh["relative_mdape_improvement_percent"],
        "target_met": bool(fresh["target_met"]), "gate_checks": fresh_gates,
        "months": audit_months,
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    }
    audit_path = root / "independent_audit.json"
    audit_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    card[f"{stage}_audit_status"] = "passed"
    card[f"{stage}_audit_path"] = _relative(audit_path)
    card[f"{stage}_audit_sha256"] = _sha256(audit_path)
    campaign.PROTOCOL["cards"][CARD] = card
    campaign.PROTOCOL_PATH.write_text(json.dumps(campaign.PROTOCOL, indent=2, ensure_ascii=False) + "\n",
                                      encoding="utf-8")
    next_action = (
        "run R157 eight-month development only because the two-month screen passed independent audit"
        if stage == "screen" and passed else
        "retain validated R35; R157 screen did not qualify for development"
        if stage == "screen" else
        "review independently audited R157 development results; fixed-origin confirmation still required"
    )
    campaign._write_progress(f"r157_{stage}_audit_passed", next_action=next_action,
                             r157_audit=_relative(audit_path))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Independently audit R157 artifacts and strict-past calibration")
    parser.add_argument("run_id")
    args = parser.parse_args()
    print(json.dumps(_audit(args.run_id), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
