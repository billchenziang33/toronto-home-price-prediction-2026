"""Independent reconstruction audit for R163 lower-L2 CatBoost."""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from experiments.bill_target74 import r04_ablation as provenance
from experiments.bill_target74 import r97_past_oof_catboost_residual_meta as r97
from experiments.bill_target74 import r154_listaware_lgbm_quantile_residual as r154
from experiments.bill_target74 import r163_listaware_catboost_l2 as r163
from experiments.bill_target74 import runner as campaign


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _keys_sha256(frame: pd.DataFrame) -> str:
    keys = sorted(zip(frame.month.astype(str), frame.ml_num.astype(str)))
    return hashlib.sha256("\n".join(f"{m}|{i}" for m, i in keys).encode("utf-8")).hexdigest()


def _equal(left, right, atol: float = 1e-9) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=atol)


def _metric(actual: np.ndarray, prediction: np.ndarray) -> dict:
    ape = np.abs(prediction - actual) / actual * 100.0
    return {"n": int(len(actual)), "mdape_percent": float(np.median(ape)),
            "mape_percent": float(np.mean(ape))}


def _bootstrap(rows: pd.DataFrame, reference: str) -> dict:
    rng = np.random.default_rng(r163.BOOTSTRAP_SEED)
    groups = [part for _, part in rows.groupby("month", sort=True)]
    differences = np.empty(r163.BOOTSTRAP_REPLICATES, dtype=float)
    for replicate in range(r163.BOOTSTRAP_REPLICATES):
        actual_parts, before_parts, after_parts = [], [], []
        for part in groups:
            ix = rng.integers(0, len(part), size=len(part))
            actual_parts.append(part.sold_price.to_numpy(float)[ix])
            before_parts.append(part[reference].to_numpy(float)[ix])
            after_parts.append(part.candidate_prediction.to_numpy(float)[ix])
        actual, before, after = map(np.concatenate, (actual_parts, before_parts, after_parts))
        differences[replicate] = (np.median(np.abs(after - actual) / actual * 100.0)
                                  - np.median(np.abs(before - actual) / actual * 100.0))
    return {"lower95": float(np.quantile(differences, 0.025)),
            "upper95": float(np.quantile(differences, 0.975)),
            "replicates": r163.BOOTSTRAP_REPLICATES, "seed": r163.BOOTSTRAP_SEED}


def _score(rows: pd.DataFrame, reference: str, months: tuple[str, ...], stage: str) -> dict:
    actual = rows.sold_price.to_numpy(float)
    before = rows[reference].to_numpy(float)
    after = rows.candidate_prediction.to_numpy(float)
    matrix = np.column_stack([actual, before, after])
    if not np.isfinite(matrix).all() or (matrix <= 0.0).any():
        raise RuntimeError("R163 audit found non-finite or non-positive prices")
    benchmark, candidate = _metric(actual, before), _metric(actual, after)
    relative_gain = 100.0 * (benchmark["mdape_percent"] - candidate["mdape_percent"]) / benchmark["mdape_percent"]
    mape_change = 100.0 * (candidate["mape_percent"] - benchmark["mape_percent"]) / benchmark["mape_percent"]
    monthly, gains = {}, []
    for month in months:
        subset = rows.loc[rows.month.astype(str).eq(month)]
        b = _metric(subset.sold_price.to_numpy(float), subset[reference].to_numpy(float))
        c = _metric(subset.sold_price.to_numpy(float), subset.candidate_prediction.to_numpy(float))
        gain = 100.0 * (b["mdape_percent"] - c["mdape_percent"]) / b["mdape_percent"]
        gains.append(gain)
        monthly[month] = {"benchmark": b, "candidate": c,
                          "relative_mdape_improvement_percent": float(gain)}
    bootstrap = _bootstrap(rows, reference)
    gates = campaign.PROTOCOL["gates"]
    if stage == "screen":
        checks = {"minimum_relative_mdape_improvement": relative_gain >= gates["screen_relative_mdape_improvement_percent"],
                  "both_screen_months_improved": all(g > 0.0 for g in gains),
                  "mape_guard": mape_change <= gates["screen_mape_regression_max_percent"],
                  "paired_bootstrap_upper95_below_zero": bootstrap["upper95"] < 0.0}
    else:
        checks = {"development_relative_gain": relative_gain >= gates["development_relative_mdape_improvement_percent"],
                  "months_improved": sum(g > 0.0 for g in gains) >= gates["development_months_improved_min"],
                  "worst_month_guard": max([0.0] + [-g for g in gains])
                  <= gates["development_worst_month_regression_max_percent"],
                  "mape_guard": mape_change <= gates["development_mape_regression_max_percent"],
                  "paired_bootstrap_upper95_below_zero": bootstrap["upper95"] < 0.0,
                  "strict_listaware_target": candidate["mdape_percent"] < 4.0}
    return {"n": int(len(rows)), "benchmark": benchmark, "candidate": candidate,
            "relative_mdape_improvement_percent": float(relative_gain),
            "mape_relative_change_percent": float(mape_change),
            "months_improved": int(sum(g > 0.0 for g in gains)),
            "worst_month_regression_percent": float(max([0.0] + [-g for g in gains])),
            "paired_bootstrap_mdape_difference_pp": bootstrap, "monthly": monthly,
            "gate_checks": checks, "qualifies": all(checks.values())}


def _score_matches(saved: dict, rebuilt: dict) -> bool:
    for section in ("benchmark", "candidate"):
        for name in ("mdape_percent", "mape_percent"):
            if not _equal(saved[section][name], rebuilt[section][name]):
                return False
    for name in ("relative_mdape_improvement_percent", "mape_relative_change_percent",
                 "worst_month_regression_percent"):
        if not _equal(saved[name], rebuilt[name]):
            return False
    if (saved["n"] != rebuilt["n"] or saved["months_improved"] != rebuilt["months_improved"]
            or saved["gate_checks"] != rebuilt["gate_checks"] or saved["qualifies"] != rebuilt["qualifies"]):
        return False
    for key in ("lower95", "upper95"):
        if not _equal(saved["paired_bootstrap_mdape_difference_pp"][key],
                      rebuilt["paired_bootstrap_mdape_difference_pp"][key], atol=1e-12):
            return False
    for month in rebuilt["monthly"]:
        for section in ("benchmark", "candidate"):
            for name in ("mdape_percent", "mape_percent"):
                if not _equal(saved["monthly"][month][section][name], rebuilt["monthly"][month][section][name]):
                    return False
        if not _equal(saved["monthly"][month]["relative_mdape_improvement_percent"],
                      rebuilt["monthly"][month]["relative_mdape_improvement_percent"]):
            return False
    return True


def _ledger() -> list[dict]:
    return [json.loads(line) for line in campaign.LEDGER.read_text(encoding="utf-8").splitlines() if line.strip()]


def _reconstruct_fit(run_id: str, month: str, history: pd.DataFrame,
                     query: pd.DataFrame, saved: pd.DataFrame) -> dict:
    cutoff = pd.Timestamp(month + "-01")
    window_start = cutoff - pd.DateOffset(months=r163.ROLLING_WINDOW_MONTHS)
    fit_id = f"{campaign.PROTOCOL['campaign_id']}:{r163.CARD}:{month}:audit:{run_id}"
    campaign._append({"event": "fit_start", "card": r163.CARD, "run_id": run_id,
                      "stage": "independent_audit", "fit_id": fit_id, "track": r163.TRACK,
                      "month": month, "training_oof_rows": len(history),
                      "training_oof_keys_sha256": _keys_sha256(history),
                      "query_keys_sha256": _keys_sha256(query),
                      "window_start_inclusive": window_start.isoformat(),
                      "history_cutoff_exclusive": cutoff.isoformat(),
                      "reserved_months_labels_opened": False, "hidden_labels_opened": False})
    started = datetime.now(timezone.utc)
    try:
        train_x, categorical = r97._build_features(history, r163.TRACK)
        query_x, query_categorical = r97._build_features(query, r163.TRACK)
        if (categorical != query_categorical or list(train_x.columns) != list(query_x.columns)
                or {"sold_price", "date_sold"} & set(train_x.columns)):
            raise RuntimeError(f"R163 audit feature schema/leakage failed for {month}")
        raw_target = np.log(history.sold_price.to_numpy(float) / history.base_prediction.to_numpy(float))
        target = np.clip(raw_target, -r163.RESIDUAL_TARGET_CLIP, r163.RESIDUAL_TARGET_CLIP)
        model = CatBoostRegressor(**r163.CATBOOST_CONFIG)
        model.fit(train_x, target, cat_features=categorical)
        raw_prediction = np.asarray(model.predict(query_x), dtype=float)
        correction = r163.CORRECTION_SHRINKAGE * np.clip(
            raw_prediction, -r163.PREDICTION_RESIDUAL_CLIP, r163.PREDICTION_RESIDUAL_CLIP)
        prediction = np.rint(query.base_prediction.to_numpy(float) * np.exp(correction))
        checks = {"raw_residual_prediction_max_abs_diff": float(np.max(np.abs(
                        raw_prediction - saved.raw_predicted_log_residual.to_numpy(float)))),
                  "correction_max_abs_diff": float(np.max(np.abs(
                        correction - saved.correction_log.to_numpy(float)))),
                  "candidate_prediction_max_abs_diff": float(np.max(np.abs(
                        prediction - saved.candidate_prediction.to_numpy(float)))),
                  "training_oof_keys_sha256": _keys_sha256(history),
                  "query_keys_sha256": _keys_sha256(query),
                  "training_rows": int(len(history)),
                  "history_months": sorted(history.month.astype(str).unique().tolist()),
                  "window_start_inclusive": window_start.isoformat(),
                  "cutoff_exclusive": cutoff.isoformat()}
        checks["predictions_reconstructed"] = (
            checks["raw_residual_prediction_max_abs_diff"] <= 1e-10
            and checks["correction_max_abs_diff"] <= 1e-10
            and checks["candidate_prediction_max_abs_diff"] == 0.0)
        campaign._append({"event": "fit_complete", "card": r163.CARD, "run_id": run_id,
                          "stage": "independent_audit", "fit_id": fit_id, "track": r163.TRACK,
                          "month": month, "status": "complete",
                          "seconds": (datetime.now(timezone.utc) - started).total_seconds(),
                          **checks, "reserved_months_labels_opened": False,
                          "hidden_labels_opened": False})
        return checks
    except BaseException as exc:
        campaign._append({"event": "fit_complete", "card": r163.CARD, "run_id": run_id,
                          "stage": "independent_audit", "fit_id": fit_id, "track": r163.TRACK,
                          "month": month, "status": "failed", "error": f"{type(exc).__name__}: {exc}",
                          "reserved_months_labels_opened": False, "hidden_labels_opened": False})
        raise


def audit() -> dict:
    card = campaign.PROTOCOL["cards"].get(r163.CARD)
    if not card:
        raise RuntimeError("R163 is not registered")
    result_path = campaign.ROOT / card.get("last_result_path", "")
    if not result_path.is_file():
        raise FileNotFoundError("R163 result artifact is missing")
    result = _json(result_path)
    run_id, stage = result["run_id"], result["stage"]
    output = result_path.parent
    audit_path = output / "independent_audit.json"
    if card.get("status") in {"screen_passed", "screen_failed", "development_audited", "development_audit_failed"}:
        if not audit_path.is_file():
            raise RuntimeError("R163 is marked audited but its audit artifact is missing")
        return _json(audit_path)
    if card.get("status") not in {"screen_pending_independent_audit", "development_pending_independent_audit"}:
        raise RuntimeError(f"R163 is not pending audit: {card.get('status')}")

    manifest_path, paired_path = output / "run_manifest.json", output / "paired_predictions.csv"
    manifest = _json(manifest_path)
    months = tuple(result["months"])
    expected_months = tuple(r163.SCREEN_MONTHS if stage == "screen" else r163.DEVELOPMENT_MONTHS)
    checks: dict[str, object] = {}
    checks["identity"] = (run_id == card.get("last_run_id")
                           and result.get("campaign_id") == campaign.PROTOCOL["campaign_id"]
                           and result.get("card") == r163.CARD and result.get("track") == r163.TRACK
                           and stage in {"screen", "development"} and months == expected_months
                           and manifest.get("run_id") == run_id and manifest.get("card") == r163.CARD
                           and manifest.get("stage") == stage)
    checks["run_access_flags"] = (result.get("reserved_months_labels_opened") is False
                                  and result.get("hidden_labels_opened") is False
                                  and manifest.get("reserved_months_labels_opened") is False
                                  and manifest.get("hidden_labels_opened") is False
                                  and card.get("reserved_months_labels_opened") is False
                                  and card.get("hidden_labels_opened") is False)
    checks["result_hash"] = manifest.get("result_sha256") == r163._sha(result_path)
    checks["paired_hash"] = manifest.get("paired_predictions_sha256") == r163._sha(paired_path)
    inputs = r154._input_artifacts()
    checks["input_identity"] = manifest.get("input_artifacts") == card.get("parent_oof_inputs") == inputs
    checks["current_best_identity"] = manifest.get("current_best_artifacts") == card.get("current_best_artifacts")
    protocol_relative = r163._relative(campaign.PROTOCOL_PATH)
    expected_source_hashes = {**card.get("source_sha256", {}),
                              protocol_relative: manifest.get("source_sha256", {}).get(protocol_relative)}
    checks["source_identity"] = manifest.get("source_sha256") == expected_source_hashes
    snapshot_checks = {}
    for relative, digest in manifest.get("source_sha256", {}).items():
        snapshot = campaign.ROOT / manifest.get("source_snapshot_paths", {}).get(relative, "")
        snapshot_ok = snapshot.is_file() and r163._sha(snapshot) == digest
        if relative != protocol_relative:
            current = campaign.ROOT / relative
            snapshot_ok = snapshot_ok and current.is_file() and r163._sha(current) == digest
        snapshot_checks[relative] = snapshot_ok
    checks["source_snapshots"] = bool(snapshot_checks) and all(snapshot_checks.values())
    current_source = provenance._source_identity()
    checks["data_identity"] = (manifest.get("source_identity") == current_source
                                and current_source.get("branch") == "bill"
                                and current_source.get("data_sha256") == campaign.PROTOCOL["data_sha256"])

    frame = r154._load_frame(inputs, stage)
    best, best_identity = r163._current_best()
    if best_identity != card["current_best_artifacts"]:
        raise RuntimeError("R163 R97 comparator identity changed")
    safe_months = set(frame.month.astype(str))
    checks["safe_oof_scope"] = (safe_months == set(r163._read_months(stage))
                                and not safe_months.intersection(campaign.PROTOCOL["reserved_months"])
                                and not frame.duplicated(["month", "ml_num"]).any())
    recomputed_support = {}
    for scope in ("screen", "development"):
        scope_frame = r154._load_frame(inputs, scope)
        recomputed_support[scope] = {}
        targets = r163.SCREEN_MONTHS if scope == "screen" else r163.DEVELOPMENT_MONTHS
        for month in targets:
            cutoff = pd.Timestamp(month + "-01")
            window_start = cutoff - pd.DateOffset(months=r163.ROLLING_WINDOW_MONTHS)
            dates = pd.to_datetime(scope_frame.date_sold, errors="raise")
            history = scope_frame.loc[dates.lt(cutoff) & dates.ge(window_start)]
            recomputed_support[scope][month] = {
                "rows": int(len(history)),
                "months": sorted(history.month.astype(str).unique().tolist()),
                "window_start_inclusive": window_start.isoformat(),
                "cutoff_exclusive": cutoff.isoformat(),
                "training_oof_keys_sha256": _keys_sha256(history),
            }
    registered_support = card.get("source_preflight", {}).get("rolling_support_by_target_month")
    checks["registered_support_matches"] = (registered_support == recomputed_support
                                              and result.get("rolling_support") == recomputed_support
                                              and all(item["rows"] >= r163.MIN_HISTORY_ROWS
                                                      for scope in recomputed_support.values()
                                                      for item in scope.values()))
    paired = pd.read_csv(paired_path, dtype={"ml_num": str})
    if (paired.duplicated(["month", "track", "ml_num"]).any()
            or set(paired.month.astype(str)) != set(months)):
        raise RuntimeError("R163 paired artifact has duplicate or unexpected rows")

    fold_checks, reconstructions = {}, []
    for month in months:
        cutoff = pd.Timestamp(month + "-01")
        window_start = cutoff - pd.DateOffset(months=r163.ROLLING_WINDOW_MONTHS)
        dates = pd.to_datetime(frame.date_sold, errors="raise")
        history = frame.loc[dates.lt(cutoff) & dates.ge(window_start)].copy()
        query = frame.loc[frame.month.astype(str).eq(month)].copy().reset_index(drop=True)
        prediction_path = output / "fold_predictions" / f"{month}.csv"
        saved = pd.read_csv(prediction_path, dtype={"ml_num": str})
        detail = next((row for row in result.get("fit_details", []) if row.get("month") == month), None)
        if detail is None:
            raise RuntimeError(f"R163 fit details missing for {month}")
        strict_ok = (not history.empty and not query.empty
                     and pd.to_datetime(history.date_sold).ge(cutoff).sum() == 0
                     and pd.to_datetime(history.date_sold).lt(window_start).sum() == 0
                     and not history.duplicated(["month", "ml_num"]).any()
                     and not query.ml_num.duplicated().any()
                     and len(history) >= r163.MIN_HISTORY_ROWS)
        fold_checks[month] = (strict_ok
                              and r163._sha(prediction_path) == detail.get("prediction_file_sha256")
                              and _keys_sha256(history) == detail.get("training_oof_keys_sha256")
                              and _keys_sha256(query) == detail.get("query_keys_sha256")
                              and len(history) == detail.get("training_oof_rows")
                              and sorted(history.month.astype(str).unique().tolist()) == detail.get("training_oof_months")
                              and detail.get("window_start_inclusive") == window_start.isoformat()
                              and detail.get("history_cutoff_exclusive") == cutoff.isoformat()
                              and len(saved) == len(query)
                              and set(saved.ml_num.astype(str)) == set(query.ml_num.astype(str)))
        if not fold_checks[month]:
            raise RuntimeError(f"R163 strict-past/history/hash audit failed for {month}")
        query_aligned = query.set_index("ml_num").reindex(saved.ml_num.astype(str))
        paired_aligned = paired.loc[paired.month.astype(str).eq(month)].set_index("ml_num").reindex(saved.ml_num.astype(str))
        best_aligned = best.loc[best.month.astype(str).eq(month)].set_index("ml_num").reindex(saved.ml_num.astype(str))
        if (query_aligned.isna().any().any() or paired_aligned.isna().any().any() or best_aligned.isna().any().any()
                or not np.array_equal(query_aligned.sold_price.to_numpy(float), saved.sold_price.to_numpy(float))
                or not np.array_equal(query_aligned.base_prediction.to_numpy(float), saved.base_prediction.to_numpy(float))
                or not np.array_equal(paired_aligned.sold_price.to_numpy(float), saved.sold_price.to_numpy(float))
                or not np.array_equal(paired_aligned.baseline_prediction.to_numpy(float), saved.base_prediction.to_numpy(float))
                or not np.array_equal(paired_aligned.current_best_prediction.to_numpy(float), best_aligned.candidate_prediction.to_numpy(float))
                or not np.array_equal(paired_aligned.candidate_prediction.to_numpy(float), saved.candidate_prediction.to_numpy(float))):
            raise RuntimeError(f"R163 row-level parent/current-best/candidate alignment failed for {month}")
        reconstructions.append(_reconstruct_fit(run_id, month, history, query, saved))

    rebuilt_parent = _score(paired, "baseline_prediction", months, stage)
    rebuilt_current = _score(paired, "current_best_prediction", months, stage)
    checks["fold_provenance"] = all(fold_checks.values())
    checks["predictions_reconstructed"] = all(row["predictions_reconstructed"] for row in reconstructions)
    checks["metrics_recomputed"] = (_score_matches(result["baseline_scores"], rebuilt_parent)
                                    and _score_matches(result["current_best_scores"], rebuilt_current))
    expected_qualifies = (rebuilt_parent["qualifies"] and rebuilt_current["qualifies"]
                          if stage == "screen" else rebuilt_parent["qualifies"])
    checks["qualification"] = result.get("qualifies") is expected_qualifies

    events = _ledger()
    run_starts = [e for e in events if e.get("event") == "fit_start" and e.get("card") == r163.CARD
                  and e.get("run_id") == run_id and e.get("stage") == stage]
    run_done = [e for e in events if e.get("event") == "fit_complete" and e.get("card") == r163.CARD
                and e.get("run_id") == run_id and e.get("stage") == stage and e.get("status") == "complete"]
    audit_starts = [e for e in events if e.get("event") == "fit_start" and e.get("card") == r163.CARD
                    and e.get("run_id") == run_id and e.get("stage") == "independent_audit"]
    audit_done = [e for e in events if e.get("event") == "fit_complete" and e.get("card") == r163.CARD
                  and e.get("run_id") == run_id and e.get("stage") == "independent_audit"
                  and e.get("status") == "complete"]
    checks["fit_ledger"] = (len(run_starts) == len(months) == len(run_done)
                            and len(audit_starts) == len(months) == len(audit_done)
                            and result.get("registered_fit_starts") == len(months))
    checks["ledger_access_flags"] = not any(
        event.get("reserved_months_labels_opened") is True or event.get("hidden_labels_opened") is True
        for event in events if event.get("card") == r163.CARD and event.get("run_id") == run_id)

    audit_passed = all(value is True for value in checks.values())
    result_passed = result.get("qualifies") is True
    audit_result = {"card": r163.CARD, "run_id": run_id, "stage": stage,
                    "audit_passed": bool(audit_passed), "promotion_gate_passed": bool(result_passed),
                    "checks": checks, "fold_checks": fold_checks,
                    "reconstruction_checks": reconstructions,
                    "recomputed_scores": {"baseline": rebuilt_parent, "current_best": rebuilt_current},
                    "input_artifacts": inputs, "source_hashes": manifest.get("source_sha256"),
                    "paired_predictions_sha256": r163._sha(paired_path),
                    "reserved_months_labels_opened": False, "hidden_labels_opened": False}
    audit_path.write_text(json.dumps(audit_result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    audit_sha = r163._sha(audit_path)
    campaign._append({"event": "independent_audit_complete", "card": r163.CARD,
                      "run_id": run_id, "stage": stage, "audit_passed": bool(audit_passed),
                      "promotion_gate_passed": bool(result_passed),
                      "audit_path": r163._relative(audit_path), "audit_sha256": audit_sha,
                      "reserved_months_labels_opened": False, "hidden_labels_opened": False})
    if stage == "screen":
        card["status"] = "screen_passed" if audit_passed and result_passed else "screen_failed"
        card["screen_passed_tracks"] = [r163.TRACK] if audit_passed and result_passed else []
        progress_status = ("r163_screen_audit_passed" if audit_passed and result_passed
                           else "r163_screen_failed_audited" if audit_passed
                           else "r163_screen_audit_failed")
        next_action = ("run R163 eight-month development after audited passing screen"
                       if audit_passed and result_passed else
                       "retain R35/R97; R163 fixed rolling-window screen did not pass the complete audited gate")
    else:
        card["status"] = "development_audited" if audit_passed and result_passed else "development_audit_failed"
        progress_status = "r163_development_audit_passed" if audit_passed else "r163_development_audit_failed"
        next_action = ("record R163 development result; retain R35 until eligible confirmation"
                       if audit_passed and result_passed else
                       "retain R35 validated parent; R163 development gate or audit failed")
    card["audit_path" if stage == "screen" else "development_audit_path"] = r163._relative(audit_path)
    card["audit_sha256" if stage == "screen" else "development_audit_sha256"] = audit_sha
    card["audit_passed" if stage == "screen" else "development_audit_passed"] = bool(audit_passed)
    campaign.PROTOCOL_PATH.write_text(json.dumps(campaign.PROTOCOL, indent=2, ensure_ascii=False) + "\n",
                                      encoding="utf-8")
    campaign._write_progress(progress_status, next_action=next_action,
                             r163_card=r163.CARD, r163_run_id=run_id,
                             r163_audit_path=r163._relative(audit_path),
                             r163_audit_passed=bool(audit_passed),
                             r163_promotion_gate_passed=bool(result_passed))
    return audit_result


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, ensure_ascii=False), flush=True)
