"""Independent safe-fold reconstruction audit for R171's Detached 24-month window screen."""
from __future__ import annotations

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


def _install_audit_text(r22, fit_rows: pd.DataFrame, eval_rows: pd.DataFrame,
                         window_start: pd.Timestamp, cutoff: pd.Timestamp):
    """Rebuild only the fold-local remarks transformer for the audited window rows."""
    _, restore = r22._install_text_features(fit_rows, eval_rows, "list_aware")
    return restore, {
        "training_window_months": 24,
        "window_start_inclusive": str(window_start.date()),
        "cutoff_exclusive": str(cutoff.date()),
        "training_rows": int(len(fit_rows)),
        "evaluation_rows": int(len(eval_rows)),
    }


def _parent_predictions(campaign, r171) -> tuple[dict, pd.DataFrame, Path, Path]:
    parent_card = campaign.PROTOCOL["cards"].get(r171.PARENT_CARD)
    if not parent_card or parent_card.get("status") != "fixed_origin_confirmation_passed":
        raise RuntimeError("R171 audit cannot verify a non-validated R35 parent")
    result_path = campaign.ROOT / parent_card["result_path"]
    paired_path = result_path.parent / "paired_predictions.csv"
    manifest_path = result_path.parent / "run_manifest.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (result.get("card") != r171.PARENT_CARD
            or result.get("run_id") != parent_card.get("run_id")
            or result.get("campaign_id") != campaign.PROTOCOL["campaign_id"]
            or manifest.get("run_id") != result.get("run_id")
            or manifest.get("result_sha256") != _sha256(result_path)
            or manifest.get("paired_predictions_sha256") != _sha256(paired_path)
            or manifest.get("source_identity", {}).get("data_sha256") != campaign.PROTOCOL["data_sha256"]
            or result.get("reserved_months_labels_opened") is not False
            or result.get("hidden_labels_opened") is not False):
        raise RuntimeError("R171 audit R35 artifacts failed identity, hash, or access-flag checks")
    rows = pd.read_csv(paired_path, dtype={"ml_num": str})
    rows = rows.loc[rows.track.eq("list_aware") & rows.month.isin(r171.SCREEN_MONTHS)].copy()
    if rows.duplicated(["month", "ml_num"]).any() or set(rows.month.astype(str)) != set(r171.SCREEN_MONTHS):
        raise RuntimeError("R171 audit R35 rows do not uniquely cover both safe screen folds")
    return result, rows, result_path, paired_path


def _reconstruct_fold(campaign, r171, month: str, saved: pd.DataFrame,
                      parent: pd.DataFrame, fit_detail: dict) -> dict:
    from data import hide_answers
    from experiments.bill_target74 import r03_catboost as helpers
    from experiments.bill_target74 import r22_remarks_tfidf as r22
    from experiments.bill_greedy import runner as greedy
    from experiments.model_optimization_v2.pipeline import blend_prices

    _, train, _, truth, heldout = greedy._load_fold(month)
    cutoff = pd.Period(month, freq="M").start_time
    if (pd.to_datetime(train.date_sold, errors="raise") >= cutoff).any():
        raise RuntimeError(f"R171 audit found non-past training rows for {month}")
    homes = hide_answers(heldout, "list_aware")
    detached = greedy._specialist_mask(homes, "Detached")
    train_detached = greedy._specialist_mask(train, "Detached")
    fit_rows = train.loc[train_detached].copy()
    window_start = cutoff - pd.DateOffset(months=r171.WINDOW_MONTHS)
    train_dates = pd.to_datetime(fit_rows.date_sold, errors="raise")
    fit_rows = fit_rows.loc[train_dates.ge(window_start)].copy()
    eval_rows = homes.loc[detached].copy()
    if len(fit_rows) < 500 or int(detached.sum()) < 100 or not (~detached).any():
        raise RuntimeError(f"R171 audit found insufficient route support for {month}")

    saved_index = saved.set_index(saved.ml_num.astype(str)).reindex(homes.ml_num.astype(str))
    parent_index = parent.loc[parent.month.astype(str).eq(month)].set_index("ml_num").reindex(homes.ml_num.astype(str))
    if (saved_index.isna().any().any() or parent_index.isna().any().any()
            or not np.array_equal(saved_index.sold_price.to_numpy(float), truth.sold_price.to_numpy(float))
            or not np.array_equal(parent_index.sold_price.to_numpy(float), truth.sold_price.to_numpy(float))
            or not np.array_equal(saved_index.parent_prediction.to_numpy(float), parent_index.candidate_prediction.to_numpy(float))):
        raise RuntimeError(f"R171 audit found row, label, or R35 parent misalignment for {month}")
    parent_prediction = parent_index.candidate_prediction.to_numpy(float)

    if ((pd.to_datetime(fit_rows.date_sold, errors="raise") < window_start).any()
            or (pd.to_datetime(fit_rows.date_sold, errors="raise") >= cutoff).any()
            or len(fit_rows) != int(fit_detail.get("training_rows", -1))):
        raise RuntimeError(f"R171 audit found training rows outside its fixed rolling window for {month}")
    restore, window_detail = _install_audit_text(r22, fit_rows, eval_rows, window_start, cutoff)
    old_fit_record = greedy._fit_record
    old_variant = greedy.VARIANTS.get(r171.VARIANT)
    greedy.VARIANTS[r171.VARIANT] = {"model_config_overrides": r171.MODEL_OVERRIDES}
    fit_id = f"{campaign.PROTOCOL['campaign_id']}:{r171.CARD}:independent_audit:list_aware:{month}:{fit_detail['fit_id']}"
    campaign._append({
        "event": "fit_start", "card": r171.CARD, "stage": "independent_audit",
        "run_id": fit_detail["run_id"], "fit_id": fit_id, "track": "list_aware",
        "month": month, "component": "detached_expert",
        "training_rows": int(len(fit_rows)), "evaluation_rows": int(len(eval_rows)),
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    started = time.perf_counter()

    def audit_fit(_parent_fit_id: str, fit_fn):
        fitted = fit_fn()
        if fitted.get("target") != "log_sold_over_list":
            raise RuntimeError("R171 audit refit target differs from registration")
        return fitted

    greedy._fit_record = audit_fit
    try:
        fitted = greedy._fit_history_candidate(
            month, "list_aware", train, eval_rows, fit_rows=fit_rows,
            variant=r171.VARIANT, base_variant=None,
            fit_tag=f"R171-audit-{month}-Detached-24m-window",
        )
    except BaseException as exc:
        campaign._append({
            "event": "fit_complete", "card": r171.CARD, "stage": "independent_audit",
            "run_id": fit_detail["run_id"], "fit_id": fit_id, "track": "list_aware",
            "month": month, "status": "failed", "seconds": time.perf_counter() - started,
            "error": f"{type(exc).__name__}: {exc}",
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        raise
    finally:
        greedy._fit_record = old_fit_record
        if old_variant is None:
            greedy.VARIANTS.pop(r171.VARIANT, None)
        else:
            greedy.VARIANTS[r171.VARIANT] = old_variant
        restore()

    if list(fitted.get("feature_columns", [])) != list(fit_detail.get("feature_columns", [])):
        raise RuntimeError(f"R171 audit feature column order differs for {month}")
    frozen_hgbr = helpers._verified_hgbr_cache(greedy, month, "list_aware", train, homes)
    specialist = blend_prices(fitted["predictions"], frozen_hgbr[detached], challenger_weight=r171.CORE_WEIGHT)
    rebuilt = parent_prediction.copy()
    rebuilt[detached] = specialist
    saved_candidate = saved_index.candidate_prediction.to_numpy(float)
    if not np.array_equal(rebuilt[~detached], parent_prediction[~detached]):
        raise RuntimeError(f"R171 audit found changed non-Detached predictions for {month}")
    max_prediction_diff = float(np.max(np.abs(rebuilt - saved_candidate)))
    if max_prediction_diff != 0.0:
        raise RuntimeError(f"R171 audit candidate prediction differs by {max_prediction_diff} for {month}")
    checks = {
        "month": month,
        "cutoff_exclusive": cutoff.isoformat(),
        "training_rows": int(len(fit_rows)),
        "evaluation_rows": int(len(homes)),
        "detached_training_rows": int(len(fit_rows)),
        "detached_evaluation_rows": int(detached.sum()),
        "training_window": window_detail,
        "feature_columns_match": list(fitted["feature_columns"]) == list(fit_detail["feature_columns"]),
        "prediction_max_abs_diff": max_prediction_diff,
        "predictions_reconstructed": True,
        "non_detached_predictions_unchanged": True,
    }
    campaign._append({
        "event": "fit_complete", "card": r171.CARD, "stage": "independent_audit",
        "run_id": fit_detail["run_id"], "fit_id": fit_id, "track": "list_aware",
        "month": month, "component": "detached_expert", "status": "complete",
        "seconds": time.perf_counter() - started, **checks,
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    return checks


def audit() -> dict:
    from experiments.bill_target74 import r04_ablation as budget
    from experiments.bill_target74 import r23_word_char_tfidf as scoring
    from experiments.bill_target74 import runner as campaign
    from experiments.bill_target74 import r171_listaware_detached_24m_window as r171

    card = campaign.PROTOCOL["cards"].get(r171.CARD)
    if not card or card.get("status") != "screen_pending_independent_audit":
        raise RuntimeError(f"R171 is not pending independent audit: {None if not card else card.get('status')}")
    result_path = campaign.ROOT / card["last_result_path"]
    output = result_path.parent
    paired_path = output / "paired_predictions.csv"
    manifest_path = output / "run_manifest.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    paired = pd.read_csv(paired_path, dtype={"ml_num": str})
    parent_result, parent_rows, parent_path, parent_paired_path = _parent_predictions(campaign, r171)
    checks: dict[str, object] = {}
    checks["identity"] = (
        result.get("campaign_id") == campaign.PROTOCOL["campaign_id"]
        and result.get("run_id") == card.get("run_id") == card.get("last_run_id")
        and result.get("card") == r171.CARD and result.get("stage") == "screen"
        and tuple(result.get("screen_months", [])) == r171.SCREEN_MONTHS
        and manifest.get("campaign_id") == result.get("campaign_id")
        and manifest.get("run_id") == result.get("run_id")
        and manifest.get("card") == r171.CARD and manifest.get("stage") == "screen"
    )
    checks["result_hash"] = manifest.get("result_sha256") == _sha256(result_path)
    checks["paired_hash"] = manifest.get("paired_predictions_sha256") == _sha256(paired_path)
    checks["parent_identity"] = (
        result.get("parent_candidate") == parent_result.get("run_id") == card.get("parent_run_id")
        and manifest.get("parent_run_id") == parent_result.get("run_id")
        and manifest.get("parent_result_sha256") == _sha256(parent_path)
        and manifest.get("parent_predictions_sha256") == _sha256(parent_paired_path)
    )
    checks["data_identity"] = (
        manifest.get("data_sha256") == campaign.PROTOCOL["data_sha256"]
        and manifest.get("source_identity") == budget._source_identity()
    )
    protocol_relative = str(campaign.PROTOCOL_PATH.relative_to(campaign.ROOT)).replace("\\", "/")
    source_checks = {}
    for relative, digest in manifest.get("source_sha256", {}).items():
        snapshot = campaign.ROOT / manifest.get("source_snapshot_paths", {}).get(relative, "")
        snapshot_ok = snapshot.is_file() and _sha256(snapshot) == digest
        if relative != protocol_relative:
            current = campaign.ROOT / relative
            snapshot_ok = snapshot_ok and current.is_file() and _sha256(current) == digest
        source_checks[relative] = bool(snapshot_ok)
    checks["source_snapshots"] = bool(source_checks) and all(source_checks.values())
    protocol_snapshot = campaign.ROOT / manifest.get("source_snapshot_paths", {}).get(protocol_relative, "")
    if protocol_snapshot.is_file():
        saved_protocol = json.loads(protocol_snapshot.read_text(encoding="utf-8"))
        saved_card = saved_protocol.get("cards", {}).get(r171.CARD, {})
        saved_limits = saved_protocol.get("limits", {})
        checks["registered_protocol_snapshot"] = (
            saved_card.get("status") == "registered"
            and saved_card.get("source_sha256") == r171._sha256(Path(r171.__file__))
            and tuple(saved_card.get("screen_months", [])) == r171.SCREEN_MONTHS
            and saved_card.get("training_window_months") == r171.WINDOW_MONTHS
            and all(value is None for key, value in saved_limits.items() if key != "learner_threads")
            and not (set(r171.SCREEN_MONTHS) & set(saved_protocol.get("reserved_months", [])))
        )
    else:
        checks["registered_protocol_snapshot"] = False
    safe_months = tuple(result.get("screen_months", []))
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
        and set(paired.month.astype(str)) == set(r171.SCREEN_MONTHS)
        and paired.track.eq("list_aware").all()
        and len(paired) == len(parent_rows)
    )
    if not checks["paired_rows"]:
        raise RuntimeError("R171 audit paired predictions have duplicate or unexpected screen rows")
    fold_checks = []
    for month in r171.SCREEN_MONTHS:
        saved_month = paired.loc[paired.month.astype(str).eq(month)].copy()
        parent_month = parent_rows.loc[parent_rows.month.astype(str).eq(month)].copy()
        detail = next((item for item in result.get("fit_details", []) if item.get("month") == month), None)
        if detail is None:
            raise RuntimeError(f"R171 fit detail missing for {month}")
        fold_checks.append(_reconstruct_fold(campaign, r171, month, saved_month, parent_month,
                                            {**detail, "run_id": result["run_id"]}))
    rebuilt_score = scoring._track_result(paired)
    checks["metrics_recomputed"] = _deep_equal(result.get("tracks", {}).get("list_aware"), rebuilt_score)
    checks["candidate_fit_ledger"] = (
        result.get("fit_starts") == len(r171.SCREEN_MONTHS)
        and manifest.get("fit_starts") == result.get("fit_starts")
        and len(result.get("fit_details", [])) == len(r171.SCREEN_MONTHS)
        and all(item.get("target") == "log_sold_over_list"
                and item.get("training_window_months") == r171.WINDOW_MONTHS
                for item in result.get("fit_details", []))
    )
    events = [event for event in _ledger(campaign)
              if event.get("card") == r171.CARD and event.get("run_id") == result.get("run_id")]
    screen_starts = [event for event in events if event.get("event") == "fit_start" and event.get("stage") == "screen"]
    screen_done = [event for event in events if event.get("event") == "fit_complete" and event.get("stage") == "screen" and event.get("status") == "complete"]
    audit_starts = [event for event in events if event.get("event") == "fit_start" and event.get("stage") == "independent_audit"]
    audit_done = [event for event in events if event.get("event") == "fit_complete" and event.get("stage") == "independent_audit" and event.get("status") == "complete"]
    checks["fit_ledger"] = (
        len(screen_starts) == len(screen_done) == len(audit_starts) == len(audit_done) == len(r171.SCREEN_MONTHS)
        and len({event.get("fit_id") for event in screen_starts}) == len(r171.SCREEN_MONTHS)
        and len({event.get("fit_id") for event in audit_starts}) == len(r171.SCREEN_MONTHS)
    )
    checks["ledger_access_flags"] = not any(
        event.get("reserved_months_labels_opened") is True or event.get("hidden_labels_opened") is True
        for event in events
    )
    audit_passed = all(value is True for value in checks.values()) and all(
        fold.get("predictions_reconstructed") is True for fold in fold_checks
    )
    promotion_gate = bool(audit_passed and rebuilt_score.get("qualifies_screen") is True)
    audit_result = {
        "card": r171.CARD, "run_id": result["run_id"], "stage": "screen",
        "audit_passed": bool(audit_passed), "promotion_gate_passed": promotion_gate,
        "checks": checks, "source_checks": source_checks, "fold_checks": fold_checks,
        "recomputed_score": rebuilt_score,
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    }
    audit_path = output / "independent_audit.json"
    audit_path.write_text(json.dumps(audit_result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    campaign._append({
        "event": "independent_audit_complete", "card": r171.CARD,
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
        status = "r171_screen_audit_failed"
        next_action = "inspect R171 independent audit discrepancies; do not promote or retune the screen"
    elif promotion_gate:
        card["status"] = "screen_passed"
        status = "r171_screen_audited_passed"
        next_action = "run the registered R171 eight-month development comparison against R35 and current-best R97"
    else:
        card["status"] = "screen_failed"
        status = "r171_screen_audited_failed"
        next_action = "retain R35 and register a distinct list-aware hypothesis; do not retune R171 on its screen months"
    campaign.PROTOCOL_PATH.write_text(
        json.dumps(campaign.PROTOCOL, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    campaign._write_progress(status, next_action=next_action,
                             r171_run_id=result["run_id"], r171_audit_passed=bool(audit_passed),
                             r171_promotion_gate_passed=promotion_gate)
    return audit_result


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, ensure_ascii=False))
