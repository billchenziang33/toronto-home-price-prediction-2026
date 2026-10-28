"""Independent safe-fold reconstruction audit for R174's Detached front-depth-ratio screen."""
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
    """Rebuild the same fold-local word TF-IDF transform as R35/R174."""
    _, restore = r22._install_text_features(fit_rows, eval_rows, "list_aware")
    support = {
        "training": {"rows": int(len(fit_rows))},
        "evaluation": {"rows": int(len(eval_rows))},
    }
    return restore, support


def _parent_predictions(campaign, r174, months: tuple[str, ...]) -> tuple[dict, pd.DataFrame, Path, Path]:
    parent_card = campaign.PROTOCOL["cards"].get(r174.R35_CARD)
    if not parent_card or parent_card.get("status") != "fixed_origin_confirmation_passed":
        raise RuntimeError("R174 audit cannot verify a non-validated R35 parent")
    screen_path = campaign.ROOT / parent_card["result_path"]
    screen_rows_path = screen_path.parent / "paired_predictions.csv"
    screen_manifest_path = screen_path.parent / "run_manifest.json"
    development_path = campaign.ROOT / parent_card["development_result_path"]
    development_rows_path = development_path.parent / "paired_predictions.csv"
    development_manifest_path = development_path.parent / "run_manifest.json"
    screen = json.loads(screen_path.read_text(encoding="utf-8"))
    screen_manifest = json.loads(screen_manifest_path.read_text(encoding="utf-8"))
    development = json.loads(development_path.read_text(encoding="utf-8"))
    development_manifest = json.loads(development_manifest_path.read_text(encoding="utf-8"))
    expected = {
        "screen_result_sha256": _sha256(screen_path),
        "screen_predictions_sha256": _sha256(screen_rows_path),
        "screen_manifest_sha256": _sha256(screen_manifest_path),
        "development_result_sha256": _sha256(development_path),
        "development_predictions_sha256": _sha256(development_rows_path),
        "development_manifest_sha256": _sha256(development_manifest_path),
    }
    registered = campaign.PROTOCOL["cards"][r174.CARD].get("parent_artifact_hashes")
    if (screen.get("card") != r174.R35_CARD
            or screen.get("run_id") != parent_card.get("run_id")
            or development.get("run_id") != parent_card.get("development_run_id")
            or screen_manifest.get("result_sha256") != expected["screen_result_sha256"]
            or screen_manifest.get("paired_predictions_sha256") != expected["screen_predictions_sha256"]
            or development_manifest.get("result_sha256") != expected["development_result_sha256"]
            or development_manifest.get("paired_predictions_sha256") != expected["development_predictions_sha256"]
            or registered != expected
            or screen_manifest.get("source_identity", {}).get("data_sha256") != campaign.PROTOCOL["data_sha256"]
            or development_manifest.get("source_identity", {}).get("data_sha256") != campaign.PROTOCOL["data_sha256"]
            or screen.get("reserved_months_labels_opened") is not False
            or screen.get("hidden_labels_opened") is not False
            or development.get("reserved_months_labels_opened") is not False
            or development.get("hidden_labels_opened") is not False):
        raise RuntimeError("R174 audit R35 artifacts failed identity, hash, or access-flag checks")
    is_screen = set(months) == set(r174.SCREEN_MONTHS)
    selected_path = screen_rows_path if is_screen else development_rows_path
    rows = pd.read_csv(selected_path, dtype={"ml_num": str})
    rows = rows.loc[rows.track.eq("list_aware") & rows.month.astype(str).isin(months)].copy()
    if rows.duplicated(["month", "ml_num"]).any() or set(rows.month.astype(str)) != set(months):
        raise RuntimeError("R174 audit R35 rows do not uniquely cover the requested safe folds")
    return screen, rows, screen_path, screen_rows_path


def _reconstruct_fold(campaign, r174, month: str, saved: pd.DataFrame,
                      parent: pd.DataFrame, fit_detail: dict) -> dict:
    from data import hide_answers
    from experiments.bill_target74 import r03_catboost as helpers
    from experiments.bill_target74 import r22_remarks_tfidf as r22
    from experiments.bill_greedy import runner as greedy
    from experiments.model_optimization_v2.pipeline import blend_prices

    _, train, _, truth, heldout = greedy._load_fold(month)
    cutoff = pd.Period(month, freq="M").start_time
    if (pd.to_datetime(train.date_sold, errors="raise") >= cutoff).any():
        raise RuntimeError(f"R174 audit found non-past training rows for {month}")
    homes = hide_answers(heldout, "list_aware")
    detached = greedy._specialist_mask(homes, "Detached")
    train_detached = greedy._specialist_mask(train, "Detached")
    fit_rows = train.loc[train_detached].copy()
    eval_rows = homes.loc[detached].copy()
    if len(fit_rows) < 500 or int(detached.sum()) < 100 or not (~detached).any():
        raise RuntimeError(f"R174 audit found insufficient route support for {month}")

    saved_index = saved.set_index(saved.ml_num.astype(str)).reindex(homes.ml_num.astype(str))
    parent_index = parent.loc[parent.month.astype(str).eq(month)].set_index("ml_num").reindex(homes.ml_num.astype(str))
    if (saved_index.isna().any().any() or parent_index.isna().any().any()
            or not np.array_equal(saved_index.sold_price.to_numpy(float), truth.sold_price.to_numpy(float))
            or not np.array_equal(parent_index.sold_price.to_numpy(float), truth.sold_price.to_numpy(float))
            or not np.array_equal(saved_index.parent_prediction.to_numpy(float), parent_index.candidate_prediction.to_numpy(float))):
        raise RuntimeError(f"R174 audit found row, label, or R35 parent misalignment for {month}")
    parent_prediction = parent_index.candidate_prediction.to_numpy(float)

    restore, feature_support = _install_audit_features(r22, fit_rows, eval_rows)
    old_fit_record = greedy._fit_record
    old_variant = greedy.VARIANTS.get(r174.VARIANT)
    greedy.VARIANTS[r174.VARIANT] = {
        "target_override": r174.TARGET_OVERRIDE,
        "model_config_overrides": r174.MODEL_OVERRIDES,
    }
    fit_id = f"{campaign.PROTOCOL['campaign_id']}:{r174.CARD}:independent_audit:list_aware:{month}:{fit_detail['fit_id']}"
    campaign._append({
        "event": "fit_start", "card": r174.CARD, "stage": "independent_audit",
        "run_id": fit_detail["run_id"], "fit_id": fit_id, "track": "list_aware",
        "month": month, "component": r174.COMPONENT,
        "training_rows": int(len(fit_rows)), "evaluation_rows": int(len(eval_rows)),
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    started = time.perf_counter()

    def audit_fit(_parent_fit_id: str, fit_fn):
        fitted = fit_fn()
        model_config = fitted.get("model_config", {})
        estimator_tweedie_variance_power = fitted.get("estimator").get_params(deep=False).get("tweedie_variance_power")
        if (fitted.get("target") != "sold_over_list_ratio"
                or fitted.get("actual_objective") != "tweedie"
                or model_config.get("objective") != "tweedie"
                or model_config.get("tweedie_variance_power") != 1.5
                or estimator_tweedie_variance_power != 1.5):
            raise RuntimeError("R174 audit refit target or Tweedie-loss configuration differs from registration")
        return fitted

    greedy._fit_record = audit_fit
    try:
        fitted = greedy._fit_history_candidate(
            month, "list_aware", train, eval_rows, fit_rows=fit_rows,
            variant=r174.VARIANT, base_variant=None,
            fit_tag=f"R174-audit-{month}-Detached-ratio-tweedie",
        )
    except BaseException as exc:
        campaign._append({
            "event": "fit_complete", "card": r174.CARD, "stage": "independent_audit",
            "run_id": fit_detail["run_id"], "fit_id": fit_id, "track": "list_aware",
            "month": month, "status": "failed", "seconds": time.perf_counter() - started,
            "error": f"{type(exc).__name__}: {exc}",
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        raise
    finally:
        greedy._fit_record = old_fit_record
        if old_variant is None:
            greedy.VARIANTS.pop(r174.VARIANT, None)
        else:
            greedy.VARIANTS[r174.VARIANT] = old_variant
        restore()

    if set(feature_support) != {"training", "evaluation"}:
        raise RuntimeError(f"R174 audit feature coverage was not recorded for {month}")
    if list(fitted.get("feature_columns", [])) != list(fit_detail.get("feature_columns", [])):
        raise RuntimeError(f"R174 audit feature column order differs for {month}")
    frozen_hgbr = helpers._verified_hgbr_cache(greedy, month, "list_aware", train, homes)
    specialist = blend_prices(fitted["predictions"], frozen_hgbr[detached], challenger_weight=r174.SPECIALIST_WEIGHT)
    rebuilt = parent_prediction.copy()
    rebuilt[detached] = specialist
    saved_candidate = saved_index.candidate_prediction.to_numpy(float)
    if not np.array_equal(rebuilt[~detached], parent_prediction[~detached]):
        raise RuntimeError(f"R174 audit found changed non-Detached predictions for {month}")
    max_prediction_diff = float(np.max(np.abs(rebuilt - saved_candidate)))
    if max_prediction_diff != 0.0:
        raise RuntimeError(f"R174 audit candidate prediction differs by {max_prediction_diff} for {month}")
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
        "tweedie_variance_power": fitted.get("model_config", {}).get("tweedie_variance_power"),
        "prediction_max_abs_diff": max_prediction_diff,
        "predictions_reconstructed": True,
        "non_detached_predictions_unchanged": True,
    }
    campaign._append({
        "event": "fit_complete", "card": r174.CARD, "stage": "independent_audit",
        "run_id": fit_detail["run_id"], "fit_id": fit_id, "track": "list_aware",
        "month": month, "component": r174.COMPONENT, "status": "complete",
        "seconds": time.perf_counter() - started, **checks,
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    return checks


def audit() -> dict:
    from experiments.bill_target74 import r04_ablation as budget
    from experiments.bill_target74 import r22_remarks_tfidf as r22
    from experiments.bill_target74 import runner as campaign
    from experiments.bill_target74 import r174_listaware_detached_ratio_tweedie as r174

    card = campaign.PROTOCOL["cards"].get(r174.CARD)
    if not card:
        raise RuntimeError("R174 has no registered campaign card")
    if card.get("status") in {"screen_failed", "screen_passed", "screen_audit_failed"}:
        stage = "screen"
    elif card.get("status") in {"development_complete", "development_audit_failed", "development_audited"}:
        stage = "development"
    else:
        raise RuntimeError(f"R174 has no completed stage to audit: {card.get('status')}")
    months = r174.SCREEN_MONTHS if stage == "screen" else r174.DEVELOPMENT_MONTHS
    result_path = campaign.ROOT / card[f"{stage}_result_path"]
    output = result_path.parent
    paired_path = output / "paired_predictions.csv"
    manifest_path = output / "run_manifest.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    paired = pd.read_csv(paired_path, dtype={"ml_num": str})
    parent_result, parent_rows, parent_path, parent_paired_path = _parent_predictions(campaign, r174, months)
    checks: dict[str, object] = {}
    checks["identity"] = (
        result.get("campaign_id") == campaign.PROTOCOL["campaign_id"]
        and result.get("run_id") == card.get("run_id")
        and result.get("card") == r174.CARD and result.get("stage") == stage
        and tuple(result.get("months", [])) == tuple(months)
        and manifest.get("campaign_id") == result.get("campaign_id")
        and manifest.get("run_id") == result.get("run_id")
        and manifest.get("parent_card") == r174.R35_CARD
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
    checks["parent_artifact_hashes"] = manifest.get("parent_artifact_hashes") == card.get("parent_artifact_hashes")
    source_checks = {}
    expected_source_hashes = r174._code_hashes()
    checks["source_set"] = set(manifest.get("code_sha256", {})) == set(expected_source_hashes)
    for relative, digest in manifest.get("code_sha256", {}).items():
        snapshot_name = manifest.get("source_snapshots", {}).get(relative)
        snapshot = output / "provenance" / (snapshot_name or "")
        current = campaign.ROOT / relative
        source_checks[relative] = bool(
            digest == expected_source_hashes.get(relative)
            and current.is_file() and _sha256(current) == digest
            and snapshot.is_file() and _sha256(snapshot) == digest
        )
    checks["source_hashes"] = bool(source_checks) and all(source_checks.values())
    checks["data_hash"] = manifest.get("source_identity", {}).get("data_sha256") == campaign.PROTOCOL["data_sha256"]
    checks["access_flags"] = (
        result.get("reserved_months_labels_opened") is False
        and result.get("hidden_labels_opened") is False
        and result.get("reserve_validation_eligible") is False
        and manifest.get("reserved_months_labels_opened") is False
        and manifest.get("hidden_labels_opened") is False
        and card.get("reserved_months_labels_opened") is False
        and card.get("hidden_labels_opened") is False
        and not (set(months) & set(campaign.PROTOCOL["reserved_months"]))
    )
    checks["paired_rows"] = (
        not paired.duplicated(["month", "track", "ml_num"]).any()
        and set(paired.month.astype(str)) == set(months)
        and paired.track.eq("list_aware").all()
        and len(paired) == len(parent_rows)
    )
    if not checks["paired_rows"]:
        raise RuntimeError("R174 audit paired predictions have duplicate or unexpected rows")
    safe_inputs = manifest.get("safe_input_artifacts", {})
    checks["safe_input_artifacts"] = set(safe_inputs) == set(months)
    for month, info in safe_inputs.items():
        snapshot = output / "provenance" / info.get("snapshot", "")
        checks["safe_input_artifacts"] = bool(
            checks["safe_input_artifacts"] and snapshot.is_file()
            and _sha256(snapshot) == info.get("sha256")
            and info.get("original_path", "").endswith(f"/frozen_hgbr/list_aware/{month}.csv")
        )
    fold_checks = []
    for month in months:
        saved_month = paired.loc[paired.month.astype(str).eq(month)].copy()
        parent_month = parent_rows.loc[parent_rows.month.astype(str).eq(month)].copy()
        detail = next((item for item in result.get("fit_details", []) if item.get("month") == month), None)
        if detail is None:
            raise RuntimeError(f"R174 fit detail missing for {month}")
        fold_checks.append(_reconstruct_fold(campaign, r174, month, saved_month, parent_month,
                                             {**detail, "run_id": result["run_id"]}))

    score_frame = paired[["month", "ml_num", "sold_price", "parent_prediction", "candidate_prediction"]].copy()
    common = {
        "months": tuple(months),
        "target_mdape_strictly_below": r174.TARGET_MDAPE,
        "bootstrap_replicates": r174.BOOTSTRAP_REPLICATES,
        "bootstrap_seed": r174.BOOTSTRAP_SEED,
    }
    if stage == "screen":
        rebuilt_score = r22.score_development_confirmation(
            score_frame, **common,
            minimum_relative_gain_percent=campaign.PROTOCOL["gates"]["screen_relative_mdape_improvement_percent"],
            minimum_months_improved=len(months), maximum_worst_month_regression_percent=0.0,
            maximum_mape_regression_percent=campaign.PROTOCOL["gates"]["screen_mape_regression_max_percent"],
        )
        rebuilt_gate_checks = {
            "relative_mdape_gain": rebuilt_score["relative_mdape_improvement_percent"] >= campaign.PROTOCOL["gates"]["screen_relative_mdape_improvement_percent"],
            "both_screen_months_improved": rebuilt_score["months_improved"] == len(months),
            "mape_guard": rebuilt_score["mape_relative_change_percent"] <= campaign.PROTOCOL["gates"]["screen_mape_regression_max_percent"],
            "paired_bootstrap_upper95_below_zero": rebuilt_score["bootstrap_mdape_difference_percent"]["upper95"] < 0.0,
        }
    else:
        rebuilt_score = r22.score_development_confirmation(score_frame, **common)
        rebuilt_gate_checks = {
            "development_gates": bool(rebuilt_score["qualifies_development"]),
            "strict_mdape_target": bool(rebuilt_score["target_met"]),
        }
    checks["metrics_recomputed"] = _deep_equal(result.get("tracks", {}).get("list_aware"), rebuilt_score)
    checks["stage_gate_checks"] = result.get("gate_checks") == rebuilt_gate_checks
    promotion_gate = all(rebuilt_gate_checks.values())
    checks["stage_qualification"] = result.get("qualifies_by_track", {}).get("list_aware") is promotion_gate
    details = result.get("fit_details", [])
    checks["candidate_fit_ledger"] = (
        result.get("fit_starts") == len(months)
        and manifest.get("fit_starts") == result.get("fit_starts")
        and len(details) == len(months)
        and {item.get("month") for item in details} == set(months)
        and all(item.get("target") == r174.TARGET_OVERRIDE
                and item.get("actual_objective") == "tweedie"
                and item.get("tweedie_variance_power") == 1.5
                and item.get("model_config", {}).get("objective") == "tweedie"
                and item.get("model_config", {}).get("tweedie_variance_power") == 1.5
                for item in details)
    )
    events = [event for event in _ledger(campaign)
              if event.get("card") == r174.CARD and event.get("run_id") == result.get("run_id")]
    candidate_starts = [event for event in events if event.get("event") == "fit_start" and event.get("stage") == stage]
    candidate_done = [event for event in events if event.get("event") == "fit_complete" and event.get("stage") == stage and event.get("status") == "complete"]
    audit_starts = [event for event in events if event.get("event") == "fit_start" and event.get("stage") == "independent_audit"]
    audit_done = [event for event in events if event.get("event") == "fit_complete" and event.get("stage") == "independent_audit" and event.get("status") == "complete"]
    checks["fit_ledger"] = (
        len(candidate_starts) == len(candidate_done) == len(audit_starts) == len(audit_done) == len(months)
        and len({event.get("fit_id") for event in candidate_starts}) == len(months)
        and len({event.get("fit_id") for event in audit_starts}) == len(months)
        and {event.get("fit_id") for event in candidate_starts} == {event.get("fit_id") for event in candidate_done}
    )
    checks["ledger_access_flags"] = not any(
        event.get("reserved_months_labels_opened") is True or event.get("hidden_labels_opened") is True
        for event in events
    )
    audit_passed = all(value is True for value in checks.values()) and all(
        fold.get("predictions_reconstructed") is True for fold in fold_checks
    )
    promotion_gate = bool(audit_passed and promotion_gate)
    audit_result = {
        "card": r174.CARD, "run_id": result["run_id"], "stage": stage,
        "audit_passed": bool(audit_passed), "promotion_gate_passed": promotion_gate,
        "checks": checks, "source_checks": source_checks, "fold_checks": fold_checks,
        "recomputed_score": rebuilt_score, "recomputed_gate_checks": rebuilt_gate_checks,
        "audit_tool_path": "experiments/bill_target74/r174_independent_audit_post.py",
        "audit_tool_sha256": _sha256(Path(__file__)),
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    }
    audit_path = output / "independent_audit.json"
    audit_path.write_text(json.dumps(audit_result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    campaign._append({
        "event": "independent_audit_complete", "card": r174.CARD,
        "run_id": result["run_id"], "stage": stage,
        "audit_passed": bool(audit_passed), "promotion_gate_passed": promotion_gate,
        "audit_path": str(audit_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        "audit_sha256": _sha256(audit_path),
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    card[f"{stage}_audit_path"] = str(audit_path.relative_to(campaign.ROOT)).replace("\\", "/")
    card[f"{stage}_audit_passed"] = bool(audit_passed)
    card[f"{stage}_promotion_gate_passed"] = promotion_gate
    card["audit_path"] = card[f"{stage}_audit_path"]
    card["audit_passed"] = bool(audit_passed)
    card["promotion_gate_passed"] = promotion_gate
    if not audit_passed:
        card["status"] = f"{stage}_audit_failed"
        status = f"r174_{stage}_audit_failed"
        next_action = f"inspect R174 {stage} audit discrepancies; do not promote or retune this candidate"
    elif stage == "screen":
        card["status"] = "screen_passed" if promotion_gate else "screen_failed"
        status = "r174_screen_audited_passed" if promotion_gate else "r174_screen_audited_failed"
        next_action = (
            "run the frozen eight-month R174 development comparison against R35 and current-best R97"
            if promotion_gate else "retain R35 and register a distinct list-aware hypothesis; do not retune R174 on its screen months"
        )
    else:
        card["status"] = "development_audited"
        status = "r174_development_audited_passed" if promotion_gate else "r174_development_audited_failed"
        next_action = (
            "R174 passed development gates; prepare locked fixed-origin confirmation"
            if promotion_gate else "retain R35; R174 did not satisfy all development gates or the strict target"
        )
    campaign.PROTOCOL_PATH.write_text(json.dumps(campaign.PROTOCOL, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    campaign._write_progress(status, next_action=next_action,
                             r174_run_id=result["run_id"], r174_stage=stage,
                             r174_audit_passed=bool(audit_passed),
                             r174_promotion_gate_passed=promotion_gate)
    return audit_result




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Independently verify an R174 Tweedie safe-fold run")
    parser.parse_args()
    print(json.dumps(audit(), indent=2, ensure_ascii=False))
