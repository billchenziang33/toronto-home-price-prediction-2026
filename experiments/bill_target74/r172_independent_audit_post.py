"""Independent safe-fold reconstruction audit for R172's Detached front-depth-ratio screen."""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.bill_target74.r19_expert_weight import _sha256


def _deep_equal(left, right, atol: float = 1e-12) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_deep_equal(left[key], right[key], atol) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_deep_equal(a, b, atol) for a, b in zip(left, right))
    if isinstance(left, (int, float, np.number)) and isinstance(right, (int, float, np.number)):
        return bool(np.isclose(float(left), float(right), rtol=0.0, atol=atol, equal_nan=True))
    return left == right


def _ledger(campaign) -> list[dict]:
    return [json.loads(line) for line in campaign.LEDGER.read_text(encoding="utf-8").splitlines() if line.strip()]


def _install_audit_features(r22, fit_rows: pd.DataFrame, eval_rows: pd.DataFrame):
    """Rebuild the same fold-local word TF-IDF transform as R35/R172."""
    _, restore = r22._install_text_features(fit_rows, eval_rows, "list_aware")
    support = {
        "training": {"rows": int(len(fit_rows))},
        "evaluation": {"rows": int(len(eval_rows))},
    }
    return restore, support


def _parent_predictions(campaign, r172) -> tuple[dict, pd.DataFrame, Path, Path]:
    parent_card = campaign.PROTOCOL["cards"].get(r172.R35_CARD)
    if not parent_card or parent_card.get("status") != "fixed_origin_confirmation_passed":
        raise RuntimeError("R172 audit cannot verify a non-validated R35 parent")
    result_path = campaign.ROOT / parent_card["result_path"]
    paired_path = result_path.parent / "paired_predictions.csv"
    manifest_path = result_path.parent / "run_manifest.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    development_path = campaign.ROOT / parent_card["development_result_path"]
    development_paired_path = development_path.parent / "paired_predictions.csv"
    development_manifest_path = development_path.parent / "run_manifest.json"
    development_result = json.loads(development_path.read_text(encoding="utf-8"))
    development_manifest = json.loads(development_manifest_path.read_text(encoding="utf-8"))
    expected_artifacts = {
        "screen_result_sha256": _sha256(result_path),
        "screen_predictions_sha256": _sha256(paired_path),
        "screen_manifest_sha256": _sha256(manifest_path),
        "development_result_sha256": _sha256(development_path),
        "development_predictions_sha256": _sha256(development_paired_path),
        "development_manifest_sha256": _sha256(development_manifest_path),
    }
    if (result.get("card") != r172.R35_CARD
            or result.get("run_id") != parent_card.get("run_id")
            or result.get("campaign_id") != campaign.PROTOCOL["campaign_id"]
            or manifest.get("run_id") != result.get("run_id")
            or manifest.get("result_sha256") != _sha256(result_path)
            or manifest.get("paired_predictions_sha256") != _sha256(paired_path)
            or development_result.get("run_id") != parent_card.get("development_run_id")
            or development_manifest.get("result_sha256") != _sha256(development_path)
            or development_manifest.get("paired_predictions_sha256") != _sha256(development_paired_path)
            or campaign.PROTOCOL["cards"][r172.CARD].get("parent_artifact_hashes") != expected_artifacts
            or manifest.get("source_identity", {}).get("data_sha256") != campaign.PROTOCOL["data_sha256"]
            or result.get("reserved_months_labels_opened") is not False
            or result.get("hidden_labels_opened") is not False):
        raise RuntimeError("R172 audit R35 artifacts failed identity, hash, or access-flag checks")
    rows = pd.read_csv(paired_path, dtype={"ml_num": str})
    rows = rows.loc[rows.track.eq("list_aware") & rows.month.isin(r172.SCREEN_MONTHS)].copy()
    if rows.duplicated(["month", "ml_num"]).any() or set(rows.month.astype(str)) != set(r172.SCREEN_MONTHS):
        raise RuntimeError("R172 audit R35 rows do not uniquely cover both safe screen folds")
    return result, rows, result_path, paired_path


def _reconstruct_fold(campaign, r172, month: str, saved: pd.DataFrame,
                      parent: pd.DataFrame, fit_detail: dict) -> dict:
    from data import hide_answers
    from experiments.bill_target74 import r03_catboost as helpers
    from experiments.bill_target74 import r22_remarks_tfidf as r22
    from experiments.bill_greedy import runner as greedy
    from experiments.model_optimization_v2.pipeline import blend_prices

    _, train, _, truth, heldout = greedy._load_fold(month)
    cutoff = pd.Period(month, freq="M").start_time
    if (pd.to_datetime(train.date_sold, errors="raise") >= cutoff).any():
        raise RuntimeError(f"R172 audit found non-past training rows for {month}")
    homes = hide_answers(heldout, "list_aware")
    detached = greedy._specialist_mask(homes, "Detached")
    train_detached = greedy._specialist_mask(train, "Detached")
    fit_rows = train.loc[train_detached].copy()
    eval_rows = homes.loc[detached].copy()
    if len(fit_rows) < 500 or int(detached.sum()) < 100 or not (~detached).any():
        raise RuntimeError(f"R172 audit found insufficient route support for {month}")

    saved_index = saved.set_index(saved.ml_num.astype(str)).reindex(homes.ml_num.astype(str))
    parent_index = parent.loc[parent.month.astype(str).eq(month)].set_index("ml_num").reindex(homes.ml_num.astype(str))
    if (saved_index.isna().any().any() or parent_index.isna().any().any()
            or not np.array_equal(saved_index.sold_price.to_numpy(float), truth.sold_price.to_numpy(float))
            or not np.array_equal(parent_index.sold_price.to_numpy(float), truth.sold_price.to_numpy(float))
            or not np.array_equal(saved_index.parent_prediction.to_numpy(float), parent_index.candidate_prediction.to_numpy(float))):
        raise RuntimeError(f"R172 audit found row, label, or R35 parent misalignment for {month}")
    parent_prediction = parent_index.candidate_prediction.to_numpy(float)

    restore, feature_support = _install_audit_features(r22, fit_rows, eval_rows)
    old_fit_record = greedy._fit_record
    old_variant = greedy.VARIANTS.get(r172.VARIANT)
    greedy.VARIANTS[r172.VARIANT] = {
        "target_override": r172.TARGET_OVERRIDE,
        "model_config_overrides": r172.MODEL_OVERRIDES,
    }
    fit_id = f"{campaign.PROTOCOL['campaign_id']}:{r172.CARD}:independent_audit:list_aware:{month}:{fit_detail['fit_id']}"
    campaign._append({
        "event": "fit_start", "card": r172.CARD, "stage": "independent_audit",
        "run_id": fit_detail["run_id"], "fit_id": fit_id, "track": "list_aware",
        "month": month, "component": "detached_expert",
        "training_rows": int(len(fit_rows)), "evaluation_rows": int(len(eval_rows)),
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    started = time.perf_counter()

    def audit_fit(_parent_fit_id: str, fit_fn):
        fitted = fit_fn()
        model_config = fitted.get("model_config", {})
        estimator_fair_c = fitted.get("estimator").get_params(deep=False).get("fair_c")
        if (fitted.get("target") != "log_sold_over_list"
                or fitted.get("actual_objective") != "fair"
                or model_config.get("objective") != "fair"
                or model_config.get("fair_c") != 1.0
                or estimator_fair_c != 1.0):
            raise RuntimeError("R172 audit refit target or Fair-loss configuration differs from registration")
        return fitted

    greedy._fit_record = audit_fit
    try:
        fitted = greedy._fit_history_candidate(
            month, "list_aware", train, eval_rows, fit_rows=fit_rows,
            variant=r172.VARIANT, base_variant=None,
            fit_tag=f"R172-audit-{month}-Detached-fair-loss",
        )
    except BaseException as exc:
        campaign._append({
            "event": "fit_complete", "card": r172.CARD, "stage": "independent_audit",
            "run_id": fit_detail["run_id"], "fit_id": fit_id, "track": "list_aware",
            "month": month, "status": "failed", "seconds": time.perf_counter() - started,
            "error": f"{type(exc).__name__}: {exc}",
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        raise
    finally:
        greedy._fit_record = old_fit_record
        if old_variant is None:
            greedy.VARIANTS.pop(r172.VARIANT, None)
        else:
            greedy.VARIANTS[r172.VARIANT] = old_variant
        restore()

    if set(feature_support) != {"training", "evaluation"}:
        raise RuntimeError(f"R172 audit feature coverage was not recorded for {month}")
    if list(fitted.get("feature_columns", [])) != list(fit_detail.get("feature_columns", [])):
        raise RuntimeError(f"R172 audit feature column order differs for {month}")
    frozen_hgbr = helpers._verified_hgbr_cache(greedy, month, "list_aware", train, homes)
    specialist = blend_prices(fitted["predictions"], frozen_hgbr[detached], challenger_weight=r172.SPECIALIST_WEIGHT)
    rebuilt = parent_prediction.copy()
    rebuilt[detached] = specialist
    saved_candidate = saved_index.candidate_prediction.to_numpy(float)
    if not np.array_equal(rebuilt[~detached], parent_prediction[~detached]):
        raise RuntimeError(f"R172 audit found changed non-Detached predictions for {month}")
    max_prediction_diff = float(np.max(np.abs(rebuilt - saved_candidate)))
    if max_prediction_diff != 0.0:
        raise RuntimeError(f"R172 audit candidate prediction differs by {max_prediction_diff} for {month}")
    checks = {
        "month": month,
        "cutoff_exclusive": cutoff.isoformat(),
        "training_rows": int(len(fit_rows)),
        "evaluation_rows": int(len(homes)),
        "detached_training_rows": int(len(fit_rows)),
        "detached_evaluation_rows": int(detached.sum()),
        "feature_support": feature_support,
        "feature_columns_match": list(fitted["feature_columns"]) == list(fit_detail["feature_columns"]),
        "actual_objective": fitted.get("actual_objective"),
        "fair_c": fitted.get("model_config", {}).get("fair_c"),
        "prediction_max_abs_diff": max_prediction_diff,
        "predictions_reconstructed": True,
        "non_detached_predictions_unchanged": True,
    }
    campaign._append({
        "event": "fit_complete", "card": r172.CARD, "stage": "independent_audit",
        "run_id": fit_detail["run_id"], "fit_id": fit_id, "track": "list_aware",
        "month": month, "component": "detached_expert", "status": "complete",
        "seconds": time.perf_counter() - started, **checks,
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    return checks


def audit(reuse_prior_refits: bool = False) -> dict:
    from experiments.bill_target74 import r04_ablation as budget
    from experiments.bill_target74 import r22_remarks_tfidf as r22
    from experiments.bill_target74 import runner as campaign
    from experiments.bill_target74 import r172_listaware_fair_loss as r172

    card = campaign.PROTOCOL["cards"].get(r172.CARD)
    if not card or card.get("status") not in {"screen_failed", "screen_passed", "screen_audit_failed"}:
        raise RuntimeError(f"R172 has no completed screen to audit: {None if not card else card.get('status')}")
    result_path = campaign.ROOT / card["screen_result_path"]
    output = result_path.parent
    paired_path = output / "paired_predictions.csv"
    manifest_path = output / "run_manifest.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    paired = pd.read_csv(paired_path, dtype={"ml_num": str})
    parent_result, parent_rows, parent_path, parent_paired_path = _parent_predictions(campaign, r172)
    checks: dict[str, object] = {}
    checks["identity"] = (
        result.get("campaign_id") == campaign.PROTOCOL["campaign_id"]
        and result.get("run_id") == card.get("run_id")
        and result.get("card") == r172.CARD and result.get("stage") == "screen"
        and tuple(result.get("months", [])) == r172.SCREEN_MONTHS
        and manifest.get("campaign_id") == result.get("campaign_id")
        and manifest.get("run_id") == result.get("run_id")
        and manifest.get("parent_card") == r172.R35_CARD
    )
    checks["result_hash"] = manifest.get("result_sha256") == _sha256(result_path)
    checks["paired_hash"] = manifest.get("paired_predictions_sha256") == _sha256(paired_path)
    checks["parent_identity"] = (
        result.get("parent_run_id") == parent_result.get("run_id") == card.get("parent_run_id")
        and manifest.get("parent_artifact_hashes") == card.get("parent_artifact_hashes")
        and card.get("parent_artifact_hashes", {}).get("screen_result_sha256") == _sha256(parent_path)
        and card.get("parent_artifact_hashes", {}).get("screen_predictions_sha256") == _sha256(parent_paired_path)
    )
    checks["data_identity"] = (
        manifest.get("source_identity", {}).get("data_sha256") == campaign.PROTOCOL["data_sha256"]
        and manifest.get("source_identity") == budget._source_identity()
    )
    checks["parent_artifact_hashes"] = (
        manifest.get("parent_artifact_hashes") == card.get("parent_artifact_hashes")
    )
    source_checks = {}
    for relative, digest in manifest.get("code_sha256", {}).items():
        snapshot_name = manifest.get("source_snapshots", {}).get(relative)
        snapshot = output / "provenance" / (snapshot_name or "")
        snapshot_ok = snapshot.is_file() and _sha256(snapshot) == digest
        current = campaign.ROOT / relative
        snapshot_ok = snapshot_ok and current.is_file() and _sha256(current) == digest
        source_checks[relative] = bool(snapshot_ok)
    checks["source_snapshots"] = bool(source_checks) and all(source_checks.values())
    saved_limits = campaign.PROTOCOL.get("limits", {})
    checks["registered_protocol"] = (
        card.get("screen_months") == list(r172.SCREEN_MONTHS)
        and card.get("model_config_overrides") == r172.MODEL_OVERRIDES
        and card.get("code_sha256") == manifest.get("code_sha256")
        and all(value is None for key, value in saved_limits.items() if key != "learner_threads")
        and not (set(r172.SCREEN_MONTHS) & set(campaign.PROTOCOL.get("reserved_months", [])))
    )
    safe_months = tuple(result.get("months", []))
    checks["access_flags"] = (
        result.get("reserved_months_labels_opened") is False
        and result.get("hidden_labels_opened") is False
        and result.get("reserve_validation_eligible") is False
        and manifest.get("reserved_months_labels_opened") is False
        and manifest.get("hidden_labels_opened") is False
        and card.get("reserved_months_labels_opened") is False
        and card.get("hidden_labels_opened") is False
        and not (set(safe_months) & set(campaign.PROTOCOL["reserved_months"]))
    )
    checks["paired_rows"] = (
        not paired.duplicated(["month", "track", "ml_num"]).any()
        and set(paired.month.astype(str)) == set(r172.SCREEN_MONTHS)
        and paired.track.eq("list_aware").all()
        and len(paired) == len(parent_rows)
    )
    if not checks["paired_rows"]:
        raise RuntimeError("R172 audit paired predictions have duplicate or unexpected screen rows")
    safe_inputs = manifest.get("safe_input_artifacts", {})
    checks["safe_input_artifacts"] = set(safe_inputs) == set(r172.SCREEN_MONTHS)
    for month, info in safe_inputs.items():
        snapshot = output / "provenance" / info.get("snapshot", "")
        checks["safe_input_artifacts"] = bool(checks["safe_input_artifacts"] and snapshot.is_file()
                                              and _sha256(snapshot) == info.get("sha256")
                                              and info.get("original_path", "").endswith(
                                                  f"/frozen_hgbr/list_aware/{month}.csv"))
    fold_checks = []
    prior_audit_path = output / "independent_audit_initial_failed.json"
    prior_audit = None
    prior_fold_by_month = {}
    if reuse_prior_refits:
        prior_audit = json.loads(prior_audit_path.read_text(encoding="utf-8"))
        if (prior_audit.get("run_id") != result.get("run_id")
                or prior_audit.get("audit_passed") is not False
                or prior_audit.get("checks", {}).get("data_identity") is not False
                or any(value is not True for name, value in prior_audit.get("checks", {}).items()
                       if name != "data_identity")):
            raise RuntimeError("prior audit report is not the expected data-identity-only failure")
        prior_fold_by_month = {item.get("month"): item for item in prior_audit.get("fold_checks", [])}
        if set(prior_fold_by_month) != set(r172.SCREEN_MONTHS):
            raise RuntimeError("prior audit refits do not cover both screen months")

    for month in r172.SCREEN_MONTHS:
        saved_month = paired.loc[paired.month.astype(str).eq(month)].copy()
        parent_month = parent_rows.loc[parent_rows.month.astype(str).eq(month)].copy()
        detail = next((item for item in result.get("fit_details", []) if item.get("month") == month), None)
        if detail is None:
            raise RuntimeError(f"R172 fit detail missing for {month}")
        if reuse_prior_refits:
            fold = prior_fold_by_month[month]
            if (fold.get("predictions_reconstructed") is not True
                    or fold.get("prediction_max_abs_diff") != 0.0
                    or fold.get("actual_objective") != "fair"
                    or fold.get("fair_c") != 1.0
                    or fold.get("feature_columns_match") is not True
                    or fold.get("non_detached_predictions_unchanged") is not True):
                raise RuntimeError(f"prior independent refit evidence failed for {month}")
            fold_checks.append(fold)
        else:
            fold_checks.append(_reconstruct_fold(campaign, r172, month, saved_month, parent_month,
                                                {**detail, "run_id": result["run_id"]}))
    score_frame = paired[["month", "ml_num", "sold_price", "parent_prediction", "candidate_prediction"]].copy()
    rebuilt_score = r22.score_development_confirmation(
        score_frame,
        months=r172.SCREEN_MONTHS,
        target_mdape_strictly_below=r172.TARGET_MDAPE,
        bootstrap_replicates=r172.BOOTSTRAP_REPLICATES,
        bootstrap_seed=r172.BOOTSTRAP_SEED,
        minimum_relative_gain_percent=campaign.PROTOCOL["gates"]["screen_relative_mdape_improvement_percent"],
        minimum_months_improved=len(r172.SCREEN_MONTHS),
        maximum_worst_month_regression_percent=0.0,
        maximum_mape_regression_percent=campaign.PROTOCOL["gates"]["screen_mape_regression_max_percent"],
    )
    checks["metrics_recomputed"] = _deep_equal(result.get("tracks", {}).get("list_aware"), rebuilt_score)
    rebuilt_gate_checks = {
        "relative_mdape_gain": rebuilt_score["relative_mdape_improvement_percent"]
        >= campaign.PROTOCOL["gates"]["screen_relative_mdape_improvement_percent"],
        "both_screen_months_improved": rebuilt_score["months_improved"] == len(r172.SCREEN_MONTHS),
        "mape_guard": rebuilt_score["mape_relative_change_percent"]
        <= campaign.PROTOCOL["gates"]["screen_mape_regression_max_percent"],
        "paired_bootstrap_upper95_below_zero": rebuilt_score["bootstrap_mdape_difference_percent"]["upper95"] < 0.0,
    }
    checks["screen_gate_checks"] = result.get("gate_checks") == rebuilt_gate_checks
    checks["screen_qualification"] = result.get("qualifies_by_track", {}).get("list_aware") is all(
        rebuilt_gate_checks.values()
    )
    checks["candidate_fit_ledger"] = (
        result.get("fit_starts") == len(r172.SCREEN_MONTHS)
        and manifest.get("fit_starts") == result.get("fit_starts")
        and len(result.get("fit_details", [])) == len(r172.SCREEN_MONTHS)
        and all(item.get("target") == "log_sold_over_list"
                and item.get("actual_objective") == "fair"
                and item.get("fair_c") == 1.0
                and item.get("model_config", {}).get("objective") == "fair"
                and item.get("model_config", {}).get("fair_c") == 1.0
                for item in result.get("fit_details", []))
    )
    events = [event for event in _ledger(campaign)
              if event.get("card") == r172.CARD and event.get("run_id") == result.get("run_id")]
    screen_starts = [event for event in events if event.get("event") == "fit_start" and event.get("stage") == "screen"]
    screen_done = [event for event in events if event.get("event") == "fit_complete" and event.get("stage") == "screen" and event.get("status") == "complete"]
    audit_starts = [event for event in events if event.get("event") == "fit_start" and event.get("stage") == "independent_audit"]
    audit_done = [event for event in events if event.get("event") == "fit_complete" and event.get("stage") == "independent_audit" and event.get("status") == "complete"]
    checks["fit_ledger"] = (
        len(screen_starts) == len(screen_done) == len(audit_starts) == len(audit_done) == len(r172.SCREEN_MONTHS)
        and len({event.get("fit_id") for event in screen_starts}) == len(r172.SCREEN_MONTHS)
        and len({event.get("fit_id") for event in audit_starts}) == len(r172.SCREEN_MONTHS)
    )
    checks["ledger_access_flags"] = not any(
        event.get("reserved_months_labels_opened") is True or event.get("hidden_labels_opened") is True
        for event in events
    )
    audit_passed = all(value is True for value in checks.values()) and all(
        fold.get("predictions_reconstructed") is True for fold in fold_checks
    )
    promotion_gate = bool(audit_passed and all(rebuilt_gate_checks.values()))
    audit_result = {
        "card": r172.CARD, "run_id": result["run_id"], "stage": "screen",
        "audit_passed": bool(audit_passed), "promotion_gate_passed": promotion_gate,
        "checks": checks, "source_checks": source_checks, "fold_checks": fold_checks,
        "recomputed_score": rebuilt_score, "recomputed_gate_checks": rebuilt_gate_checks,
        "audit_tool_path": "experiments/bill_target74/r172_independent_audit_post.py",
        "audit_tool_sha256": _sha256(Path(__file__)),
        "reused_prior_refits": bool(reuse_prior_refits),
        "prior_audit_path": str(prior_audit_path.relative_to(campaign.ROOT)).replace("\\", "/")
        if reuse_prior_refits else None,
        "prior_audit_sha256": _sha256(prior_audit_path) if reuse_prior_refits else None,
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    }
    audit_path = output / ("independent_audit_corrected.json" if reuse_prior_refits else "independent_audit.json")
    audit_path.write_text(json.dumps(audit_result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    campaign._append({
        "event": "independent_audit_correction_complete" if reuse_prior_refits else "independent_audit_complete",
        "card": r172.CARD,
        "run_id": result["run_id"], "stage": "screen",
        "audit_passed": bool(audit_passed), "promotion_gate_passed": promotion_gate,
        "audit_path": str(audit_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        "audit_sha256": _sha256(audit_path),
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    card["audit_path"] = str(audit_path.relative_to(campaign.ROOT)).replace("\\", "/")
    card["audit_passed"] = bool(audit_passed)
    card["promotion_gate_passed"] = promotion_gate
    if not audit_passed:
        card["status"] = "screen_audit_failed"
        status = "r172_screen_audit_failed"
        next_action = "inspect R172 independent audit discrepancies; do not promote or retune the screen"
    elif promotion_gate:
        card["status"] = "screen_passed"
        status = "r172_screen_audited_passed"
        next_action = "run the registered R172 eight-month development comparison against R35 and current-best R97"
    else:
        card["status"] = "screen_failed"
        status = "r172_screen_audited_failed"
        next_action = "retain R35 and register a distinct list-aware hypothesis; do not retune R172 on its screen months"
    campaign.PROTOCOL_PATH.write_text(
        json.dumps(campaign.PROTOCOL, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    campaign._write_progress(status, next_action=next_action,
                             r172_run_id=result["run_id"], r172_audit_passed=bool(audit_passed),
                             r172_promotion_gate_passed=promotion_gate)
    return audit_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Independently verify the R172 Fair-loss screen")
    parser.add_argument(
        "--reuse-prior-refits", action="store_true",
        help="reuse completed refits from the prior audit report after correcting a non-model audit check",
    )
    args = parser.parse_args()
    print(json.dumps(audit(reuse_prior_refits=args.reuse_prior_refits), indent=2, ensure_ascii=False))
