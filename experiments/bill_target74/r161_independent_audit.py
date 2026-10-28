"""Independent reconstruction and promotion audit for R161."""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from experiments.bill_target74 import r97_past_oof_catboost_residual_meta as r97
from experiments.bill_target74 import r154_listaware_lgbm_quantile_residual as r154
from experiments.bill_target74 import r161_listaware_seed_oof_catboost as r161
from experiments.bill_target74 import runner as campaign


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _keys_sha256(frame: pd.DataFrame) -> str:
    keys = sorted(zip(frame.month.astype(str), frame.ml_num.astype(str)))
    payload = "\n".join(f"{month}|{ml_num}" for month, ml_num in keys).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _same_number(left, right, atol: float = 1e-9) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=atol)


def _metric(actual: np.ndarray, prediction: np.ndarray) -> dict:
    ape = np.abs(prediction - actual) / actual * 100.0
    return {"n": int(len(actual)), "mdape_percent": float(np.median(ape)),
            "mape_percent": float(np.mean(ape))}


def _bootstrap(rows: pd.DataFrame, reference: str) -> dict:
    rng = np.random.default_rng(r161.BOOTSTRAP_SEED)
    groups = [part for _, part in rows.groupby("month", sort=True)]
    differences = np.empty(r161.BOOTSTRAP_REPLICATES, dtype=float)
    for replicate in range(r161.BOOTSTRAP_REPLICATES):
        actual_parts, before_parts, after_parts = [], [], []
        for part in groups:
            indices = rng.integers(0, len(part), size=len(part))
            actual_parts.append(part.sold_price.to_numpy(float)[indices])
            before_parts.append(part[reference].to_numpy(float)[indices])
            after_parts.append(part.candidate_prediction.to_numpy(float)[indices])
        actual = np.concatenate(actual_parts)
        before = np.concatenate(before_parts)
        after = np.concatenate(after_parts)
        differences[replicate] = (np.median(np.abs(after - actual) / actual * 100.0)
                                  - np.median(np.abs(before - actual) / actual * 100.0))
    return {"lower95": float(np.quantile(differences, 0.025)),
            "upper95": float(np.quantile(differences, 0.975)),
            "replicates": r161.BOOTSTRAP_REPLICATES, "seed": r161.BOOTSTRAP_SEED}


def _score(rows: pd.DataFrame, reference: str, months: tuple[str, ...], stage: str) -> dict:
    actual = rows.sold_price.to_numpy(float)
    reference_prediction = rows[reference].to_numpy(float)
    candidate_prediction = rows.candidate_prediction.to_numpy(float)
    if (not np.isfinite(np.column_stack([actual, reference_prediction, candidate_prediction])).all()
            or (np.column_stack([actual, reference_prediction, candidate_prediction]) <= 0.0).any()):
        raise RuntimeError("audit found invalid actual or prediction values")
    benchmark = _metric(actual, reference_prediction)
    candidate = _metric(actual, candidate_prediction)
    gain = 100.0 * (benchmark["mdape_percent"] - candidate["mdape_percent"]) / benchmark["mdape_percent"]
    mape_change = 100.0 * (candidate["mape_percent"] - benchmark["mape_percent"]) / benchmark["mape_percent"]
    monthly, month_gains = {}, []
    for month in months:
        part = rows.loc[rows.month.astype(str).eq(month)]
        before = _metric(part.sold_price.to_numpy(float), part[reference].to_numpy(float))
        after = _metric(part.sold_price.to_numpy(float), part.candidate_prediction.to_numpy(float))
        month_gain = 100.0 * (before["mdape_percent"] - after["mdape_percent"]) / before["mdape_percent"]
        month_gains.append(month_gain)
        monthly[month] = {"benchmark": before, "candidate": after,
                          "relative_mdape_improvement_percent": float(month_gain)}
    bootstrap = _bootstrap(rows, reference)
    gates = campaign.PROTOCOL["gates"]
    if stage == "screen":
        checks = {"minimum_relative_mdape_improvement": gain >= gates["screen_relative_mdape_improvement_percent"],
                  "both_screen_months_improved": all(value > 0.0 for value in month_gains),
                  "mape_guard": mape_change <= gates["screen_mape_regression_max_percent"],
                  "paired_bootstrap_upper95_below_zero": bootstrap["upper95"] < 0.0}
    else:
        checks = {"development_relative_gain": gain >= gates["development_relative_mdape_improvement_percent"],
                  "months_improved": sum(value > 0.0 for value in month_gains) >= gates["development_months_improved_min"],
                  "worst_month_guard": max([0.0] + [-value for value in month_gains])
                  <= gates["development_worst_month_regression_max_percent"],
                  "mape_guard": mape_change <= gates["development_mape_regression_max_percent"],
                  "paired_bootstrap_upper95_below_zero": bootstrap["upper95"] < 0.0,
                  "strict_listaware_target": candidate["mdape_percent"] < 4.0}
    return {"n": len(rows), "benchmark": benchmark, "candidate": candidate,
            "relative_mdape_improvement_percent": float(gain),
            "mape_relative_change_percent": float(mape_change),
            "months_improved": int(sum(value > 0.0 for value in month_gains)),
            "worst_month_regression_percent": float(max([0.0] + [-value for value in month_gains])),
            "paired_bootstrap_mdape_difference_pp": bootstrap, "monthly": monthly,
            "gate_checks": checks, "qualifies": all(checks.values())}


def _compare_scores(saved: dict, rebuilt: dict) -> bool:
    scalar_paths = [
        ("benchmark", "mdape_percent"), ("benchmark", "mape_percent"),
        ("candidate", "mdape_percent"), ("candidate", "mape_percent"),
    ]
    for top, key in scalar_paths:
        if not _same_number(saved[top][key], rebuilt[top][key]):
            return False
    for key in ("relative_mdape_improvement_percent", "mape_relative_change_percent",
                "worst_month_regression_percent"):
        if not _same_number(saved[key], rebuilt[key]):
            return False
    if (saved["n"] != rebuilt["n"] or saved["months_improved"] != rebuilt["months_improved"]
            or saved["qualifies"] != rebuilt["qualifies"]
            or saved["gate_checks"] != rebuilt["gate_checks"]):
        return False
    for key in ("lower95", "upper95"):
        if not _same_number(saved["paired_bootstrap_mdape_difference_pp"][key],
                            rebuilt["paired_bootstrap_mdape_difference_pp"][key], atol=1e-12):
            return False
    for month in rebuilt["monthly"]:
        for top in ("benchmark", "candidate"):
            for key in ("mdape_percent", "mape_percent"):
                if not _same_number(saved["monthly"][month][top][key],
                                    rebuilt["monthly"][month][top][key]):
                    return False
        if not _same_number(saved["monthly"][month]["relative_mdape_improvement_percent"],
                            rebuilt["monthly"][month]["relative_mdape_improvement_percent"]):
            return False
    return True


def _ledger_rows() -> list[dict]:
    return [json.loads(line) for line in campaign.LEDGER.read_text(encoding="utf-8").splitlines() if line.strip()]


def _record_reconstruction_fit(run_id: str, month: str, history: pd.DataFrame,
                               query: pd.DataFrame, expected: pd.DataFrame) -> dict:
    fit_id = f"{campaign.PROTOCOL['campaign_id']}:{r161.CARD}:{month}:audit:{run_id}"
    fit_event = {"event": "fit_start", "card": r161.CARD, "run_id": run_id,
                 "stage": "independent_audit", "fit_id": fit_id, "track": r161.TRACK,
                 "month": month, "training_oof_rows": len(history),
                 "training_oof_keys_sha256": _keys_sha256(history),
                 "query_keys_sha256": _keys_sha256(query),
                 "history_cutoff_exclusive": (pd.Timestamp(month + "-01")).isoformat(),
                 "reserved_months_labels_opened": False, "hidden_labels_opened": False}
    campaign._append(fit_event)
    started = datetime.now(timezone.utc)
    try:
        train_x, categorical = r97._build_features(history, r161.TRACK)
        query_x, query_categorical = r97._build_features(query, r161.TRACK)
        if (categorical != query_categorical or list(train_x.columns) != list(query_x.columns)
                or {"sold_price", "date_sold"} & set(train_x.columns)):
            raise RuntimeError(f"audit feature schema/leakage check failed for {month}")
        target_raw = np.log(history.sold_price.to_numpy(float) / history.base_prediction.to_numpy(float))
        target = np.clip(target_raw, -r161.RESIDUAL_TARGET_CLIP, r161.RESIDUAL_TARGET_CLIP)
        model = CatBoostRegressor(**r161.CATBOOST_CONFIG)
        model.fit(train_x, target, cat_features=categorical)
        raw_prediction = np.asarray(model.predict(query_x), dtype=float)
        correction = r161.CORRECTION_SHRINKAGE * np.clip(
            raw_prediction, -r161.PREDICTION_RESIDUAL_CLIP, r161.PREDICTION_RESIDUAL_CLIP)
        prediction = np.rint(query.base_prediction.to_numpy(float) * np.exp(correction))
        checks = {
            "raw_residual_prediction_max_abs_diff": float(np.max(np.abs(
                raw_prediction - expected.raw_predicted_log_residual.to_numpy(float)))),
            "correction_max_abs_diff": float(np.max(np.abs(
                correction - expected.correction_log.to_numpy(float)))),
            "candidate_prediction_max_abs_diff": float(np.max(np.abs(
                prediction - expected.candidate_prediction.to_numpy(float)))),
            "training_key_hash_matches": _keys_sha256(history) == fit_event["training_oof_keys_sha256"],
            "query_key_hash_matches": _keys_sha256(query) == fit_event["query_keys_sha256"],
        }
        checks["predictions_reconstructed"] = (
            checks["raw_residual_prediction_max_abs_diff"] <= 1e-10
            and checks["correction_max_abs_diff"] <= 1e-10
            and checks["candidate_prediction_max_abs_diff"] == 0.0)
        seconds = (datetime.now(timezone.utc) - started).total_seconds()
        campaign._append({"event": "fit_complete", "card": r161.CARD, "run_id": run_id,
                          "stage": "independent_audit", "fit_id": fit_id, "track": r161.TRACK,
                          "month": month, "status": "complete", "seconds": seconds,
                          "training_oof_rows": len(history), **checks,
                          "reserved_months_labels_opened": False, "hidden_labels_opened": False})
        return checks
    except BaseException as exc:
        campaign._append({"event": "fit_complete", "card": r161.CARD, "run_id": run_id,
                          "stage": "independent_audit", "fit_id": fit_id, "track": r161.TRACK,
                          "month": month, "status": "failed",
                          "error": f"{type(exc).__name__}: {exc}",
                          "reserved_months_labels_opened": False, "hidden_labels_opened": False})
        raise


def audit() -> dict:
    card = campaign.PROTOCOL["cards"].get(r161.CARD)
    if not card:
        raise RuntimeError("R161 has no preregistered protocol card")
    result_path = campaign.ROOT / card.get("last_result_path", "")
    if not result_path.is_file():
        raise FileNotFoundError("R161 result path is missing")
    result = _read_json(result_path)
    run_id, stage = result["run_id"], result["stage"]
    output = result_path.parent
    audit_path = output / "independent_audit.json"
    if card.get("status") in {"screen_passed", "screen_failed", "development_audited", "development_audit_failed"}:
        if not audit_path.is_file():
            raise RuntimeError("R161 is marked audited but its audit artifact is missing")
        return _read_json(audit_path)
    if card.get("status") not in {"screen_pending_independent_audit", "development_pending_independent_audit"}:
        raise RuntimeError(f"R161 is not pending an independent audit: {card.get('status')}")

    manifest_path = output / "run_manifest.json"
    paired_path = output / "paired_predictions.csv"
    manifest = _read_json(manifest_path)
    months = tuple(result["months"])
    expected_months = tuple(r161.SCREEN_MONTHS if stage == "screen" else r161.DEVELOPMENT_MONTHS)
    checks: dict[str, object] = {}
    checks["identity"] = (run_id == card.get("last_run_id")
                           and result.get("campaign_id") == campaign.PROTOCOL["campaign_id"]
                           and result.get("card") == r161.CARD
                           and result.get("track") == r161.TRACK
                           and stage in {"screen", "development"}
                           and months == expected_months
                           and manifest.get("run_id") == run_id
                           and manifest.get("card") == r161.CARD
                           and manifest.get("stage") == stage)
    checks["safe_label_flags"] = (result.get("reserved_months_labels_opened") is False
                                  and result.get("hidden_labels_opened") is False
                                  and manifest.get("reserved_months_labels_opened") is False
                                  and manifest.get("hidden_labels_opened") is False
                                  and campaign.PROTOCOL.get("hidden_labels_opened") is False)
    checks["result_hash"] = manifest.get("result_sha256") == r161._sha(result_path)
    checks["paired_hash"] = manifest.get("paired_predictions_sha256") == r161._sha(paired_path)
    checks["input_artifacts"] = manifest.get("input_artifacts") == card.get("parent_oof_inputs") == r154._input_artifacts()
    checks["current_best_identity"] = manifest.get("r97_current_best_artifacts") == card.get("r97_current_best_artifacts")
    checks["source_hashes"] = manifest.get("source_sha256") == card.get("source_sha256") | {
        "experiments/bill_target74/campaign_protocol.json": manifest.get("source_sha256", {}).get(
            "experiments/bill_target74/campaign_protocol.json")
    }
    source_snapshot_checks = {}
    for relative, expected_hash in manifest.get("source_sha256", {}).items():
        snapshot = campaign.ROOT / manifest.get("source_snapshot_paths", {}).get(relative, "")
        source_snapshot_checks[relative] = snapshot.is_file() and r161._sha(snapshot) == expected_hash
        if relative != r161._relative(campaign.PROTOCOL_PATH):
            current = campaign.ROOT / relative
            source_snapshot_checks[relative] = (source_snapshot_checks[relative]
                                                and current.is_file() and r161._sha(current) == expected_hash)
    checks["source_snapshots"] = bool(source_snapshot_checks) and all(source_snapshot_checks.values())
    checks["data_identity"] = (manifest.get("source_identity", {}).get("data_sha256")
                                == campaign.PROTOCOL["data_sha256"]
                                and manifest.get("source_identity", {}).get("branch") == "bill"
                                and manifest.get("source_identity", {}).get("commit") == campaign.PROTOCOL["parent_commit"])

    inputs = card["parent_oof_inputs"]
    frame = r154._load_frame(inputs, stage)
    best, best_identity = r161._r97_current_best()
    if best_identity != card["r97_current_best_artifacts"]:
        raise RuntimeError("R97 current-best comparator identity changed")
    if frame.duplicated(["month", "ml_num"]).any() or set(frame.month.astype(str)) & set(campaign.PROTOCOL["reserved_months"]):
        raise RuntimeError("independent audit safe OOF frame has duplicate keys or reserved months")
    read_months = set(frame.month.astype(str))
    expected_read_months = set((*r161.SEED_MONTHS, *[m for m in r161.DEVELOPMENT_MONTHS if m <= max(months)]))
    checks["safe_oof_scope"] = read_months == expected_read_months

    paired = pd.read_csv(paired_path, dtype={"ml_num": str})
    if paired.duplicated(["month", "track", "ml_num"]).any() or set(paired.month.astype(str)) != set(months):
        raise RuntimeError("R161 paired result contains duplicate or unexpected rows")
    fold_prediction_checks = {}
    reconstruction_checks = []
    for month in months:
        cutoff = pd.Timestamp(month + "-01")
        history = frame.loc[pd.to_datetime(frame.date_sold).lt(cutoff)].copy()
        query = frame.loc[frame.month.astype(str).eq(month)].copy().reset_index(drop=True)
        prediction_path = output / "fold_predictions" / f"{month}.csv"
        prediction_file = pd.read_csv(prediction_path, dtype={"ml_num": str})
        if (query.empty or query.ml_num.duplicated().any()
                or history.duplicated(["month", "ml_num"]).any()
                or pd.to_datetime(history.date_sold).ge(cutoff).any()
                or history.month.astype(str).isin(campaign.PROTOCOL["reserved_months"]).any()):
            raise RuntimeError(f"R161 strict-past or query-key audit failed for {month}")
        expected_fit = next((entry for entry in result.get("fit_details", []) if entry.get("month") == month), None)
        if expected_fit is None:
            raise RuntimeError(f"R161 fit details are missing {month}")
        fold_prediction_checks[month] = (
            r161._sha(prediction_path) == expected_fit.get("prediction_file_sha256")
            and len(prediction_file) == len(query)
            and set(prediction_file.ml_num.astype(str)) == set(query.ml_num.astype(str))
            and _keys_sha256(history) == expected_fit.get("training_oof_keys_sha256")
            and _keys_sha256(query) == expected_fit.get("query_keys_sha256")
            and len(history) == expected_fit.get("training_oof_rows")
            and sorted(history.month.astype(str).unique().tolist()) == expected_fit.get("training_oof_months")
            and len(history) >= r161.MIN_HISTORY_ROWS)
        if not fold_prediction_checks[month]:
            raise RuntimeError(f"R161 fold provenance or strict-past support audit failed for {month}")
        expected_rows = query.set_index("ml_num").reindex(prediction_file.ml_num.astype(str))
        if (expected_rows.isna().any().any()
                or not np.array_equal(expected_rows.sold_price.to_numpy(float), prediction_file.sold_price.to_numpy(float))
                or not np.array_equal(expected_rows.base_prediction.to_numpy(float), prediction_file.base_prediction.to_numpy(float))):
            raise RuntimeError(f"R161 prediction rows do not align to safe parent OOF rows for {month}")
        fold_rows = paired.loc[paired.month.astype(str).eq(month)].set_index("ml_num").reindex(prediction_file.ml_num.astype(str))
        best_month = best.loc[best.month.astype(str).eq(month)].set_index("ml_num").reindex(prediction_file.ml_num.astype(str))
        if (fold_rows.isna().any().any() or best_month.isna().any().any()
                or not np.array_equal(fold_rows.sold_price.to_numpy(float), prediction_file.sold_price.to_numpy(float))
                or not np.array_equal(fold_rows.baseline_prediction.to_numpy(float), prediction_file.base_prediction.to_numpy(float))
                or not np.array_equal(fold_rows.current_best_prediction.to_numpy(float), best_month.candidate_prediction.to_numpy(float))
                or not np.array_equal(fold_rows.candidate_prediction.to_numpy(float), prediction_file.candidate_prediction.to_numpy(float))):
            raise RuntimeError(f"R161 paired baseline/current-best/candidate alignment failed for {month}")
        reconstruction_checks.append(_record_reconstruction_fit(run_id, month, history, query, prediction_file))

    rebuilt_rows = paired.copy()
    rebuilt_baseline = _score(rebuilt_rows, "baseline_prediction", months, stage)
    rebuilt_best = _score(rebuilt_rows, "current_best_prediction", months, stage)
    checks["predictions_reconstructed"] = all(entry["predictions_reconstructed"] for entry in reconstruction_checks)
    checks["metrics_recomputed"] = (_compare_scores(result["baseline_scores"], rebuilt_baseline)
                                    and _compare_scores(result["current_best_scores"], rebuilt_best))
    checks["screen_result_qualification"] = (result.get("qualifies")
                                              == (rebuilt_baseline["qualifies"] and rebuilt_best["qualifies"]
                                                  if stage == "screen" else rebuilt_baseline["qualifies"]))
    ledger = _ledger_rows()
    runner_starts = [event for event in ledger if event.get("event") == "fit_start"
                     and event.get("card") == r161.CARD and event.get("run_id") == run_id
                     and event.get("stage") == stage]
    runner_completes = [event for event in ledger if event.get("event") == "fit_complete"
                        and event.get("card") == r161.CARD and event.get("run_id") == run_id
                        and event.get("stage") == stage and event.get("status") == "complete"]
    audit_starts = [event for event in ledger if event.get("event") == "fit_start"
                    and event.get("card") == r161.CARD and event.get("run_id") == run_id
                    and event.get("stage") == "independent_audit"]
    audit_completes = [event for event in ledger if event.get("event") == "fit_complete"
                       and event.get("card") == r161.CARD and event.get("run_id") == run_id
                       and event.get("stage") == "independent_audit" and event.get("status") == "complete"]
    checks["fit_ledger"] = (len(runner_starts) == len(months) == len(runner_completes)
                            and len(audit_starts) == len(months) == len(audit_completes)
                            and result.get("registered_fit_starts") == len(months))
    checks["no_access_flags_in_ledger"] = not any(
        event.get("card") == r161.CARD and event.get("run_id") == run_id
        and (event.get("reserved_months_labels_opened") is not False
             or event.get("hidden_labels_opened") is not False)
        for event in ledger if event.get("event") in {"fit_start", "fit_complete", "fold_complete", "card_start"})

    audit_ok = all(value is True for value in checks.values())
    result_ok = result.get("qualifies") is True
    audit_result = {"card": r161.CARD, "run_id": run_id, "stage": stage,
                    "audit_passed": bool(audit_ok), "promotion_gate_passed": bool(result_ok),
                    "checks": checks, "fold_prediction_checks": fold_prediction_checks,
                    "reconstruction_checks": reconstruction_checks,
                    "recomputed_scores": {"baseline": rebuilt_baseline, "current_best": rebuilt_best},
                    "inputs": inputs, "source_hashes": manifest.get("source_sha256"),
                    "paired_predictions_sha256": r161._sha(paired_path),
                    "reserved_months_labels_opened": False, "hidden_labels_opened": False}
    audit_path.write_text(json.dumps(audit_result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    audit_sha256 = r161._sha(audit_path)
    campaign._append({"event": "independent_audit_complete", "card": r161.CARD,
                      "run_id": run_id, "stage": stage, "audit_passed": bool(audit_ok),
                      "promotion_gate_passed": bool(result_ok), "audit_path": r161._relative(audit_path),
                      "audit_sha256": audit_sha256,
                      "reserved_months_labels_opened": False, "hidden_labels_opened": False})
    card["audit_path" if stage == "screen" else "development_audit_path"] = r161._relative(audit_path)
    card["audit_sha256" if stage == "screen" else "development_audit_sha256"] = audit_sha256
    if stage == "screen":
        card["status"] = "screen_passed" if audit_ok and result_ok else "screen_failed"
        card["screen_passed_tracks"] = [r161.TRACK] if audit_ok and result_ok else []
        progress_status = "r161_screen_audit_passed" if audit_ok else "r161_screen_audit_failed"
        next_action = ("run R161 eight-month development only if the screen and audit both pass"
                       if audit_ok and result_ok else
                       "retain current best; R161 screen did not pass the complete audited gate")
    else:
        card["status"] = "development_audited" if audit_ok and result_ok else "development_audit_failed"
        progress_status = "r161_development_audit_passed" if audit_ok else "r161_development_audit_failed"
        next_action = ("record R161 development result; do not promote without eligible fixed-origin confirmation"
                       if audit_ok and result_ok else
                       "retain validated R35; R161 development gate or audit failed")
    campaign.PROTOCOL_PATH.write_text(json.dumps(campaign.PROTOCOL, indent=2, ensure_ascii=False) + "\n",
                                      encoding="utf-8")
    campaign._write_progress(progress_status, next_action=next_action,
                             r161_card=r161.CARD, r161_run_id=run_id,
                             r161_audit_path=r161._relative(audit_path),
                             r161_audit_passed=bool(audit_ok), r161_promotion_gate_passed=bool(result_ok))
    return audit_result


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, ensure_ascii=False), flush=True)
