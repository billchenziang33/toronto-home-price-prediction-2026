"""Execution-only compatibility replay; no new accuracy claim or promotion."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from datetime import datetime, timezone

from experiments.bill_target74 import desktop_friendly_runtime as runtime

CARD = "R173_desktop_friendly_scheduling_replay"
MONTHS = ("2023-03", "2024-06")


def _write(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run() -> dict:
    runtime.wait_before_fit("R173:before_imports")
    import pandas as pd
    from experiments.bill_target74 import runner as campaign
    from experiments.bill_target74 import r172_listaware_fair_loss as reference
    from experiments.bill_target74 import r172_independent_audit_post as refit
    from experiments.bill_target74 import r04_ablation as identity

    if CARD in campaign.PROTOCOL["cards"]:
        raise RuntimeError("R173 is already registered; inspect its state before starting another replay")
    source = identity._source_identity()
    parent = campaign.PROTOCOL["cards"][reference.CARD]
    if parent.get("audit_passed") is not True or parent.get("status") != "screen_failed":
        raise RuntimeError("the frozen R172 reference is not an audited closed screen")
    reference_path = campaign.ROOT / parent["screen_result_path"]
    reference_result = json.loads(reference_path.read_text(encoding="utf-8"))
    reference_paired = reference_path.parent / "paired_predictions.csv"
    reference_manifest = reference_path.parent / "run_manifest.json"
    original_manifest = json.loads(reference_manifest.read_text(encoding="utf-8"))
    reference_hashes = {str(path.relative_to(campaign.ROOT)).replace("\\", "/"):
                        reference._sha256(path) for path in
                        [reference_path, reference_paired, reference_manifest]}
    if (original_manifest.get("result_sha256") != reference._sha256(reference_path)
            or original_manifest.get("paired_predictions_sha256") != reference._sha256(reference_paired)
            or tuple(reference_result.get("months", [])) != MONTHS
            or set(MONTHS) & set(campaign.PROTOCOL["reserved_months"])):
        raise RuntimeError("reference hashes, months or safe-data constraints failed")
    paths = list(dict.fromkeys([Path(__file__), Path(runtime.__file__), runtime.POLICY_PATH,
                               Path(refit.__file__), *reference._source_files()]))
    hashes = {str(path.relative_to(campaign.ROOT)).replace("\\", "/"):
              reference._sha256(path) for path in paths}
    run_id = "r173r_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = campaign.CAMPAIGN_ROOT / "batches" / run_id
    provenance = output / "provenance"
    provenance.mkdir(parents=True, exist_ok=False)
    snapshots = {}
    for i, path in enumerate(paths):
        relative = str(path.relative_to(campaign.ROOT)).replace("\\", "/")
        name = f"src_{i:02d}{path.suffix}.source"
        shutil.copy2(path, provenance / name)
        snapshots[relative] = name
    _write(provenance / "registration_protocol.json", campaign.PROTOCOL)
    card = {
        "status": "registered", "run_id": run_id, "type": "execution_only_compatibility",
        "method": "Replay frozen R172 under the desktop-friendly runtime and require exact predictions; no model search or promotion.",
        "reference_card": reference.CARD, "reference_run_id": reference_result["run_id"],
        "reference_hashes": reference_hashes, "source_identity": source,
        "code_sha256": hashes, "months": list(MONTHS), "expected_fit_starts": 2,
        "resource_policy": runtime.POLICY, "success": "both safe folds have max_abs_prediction_diff=0",
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    }
    campaign.PROTOCOL["cards"][CARD] = card
    campaign.PROTOCOL["desktop_friendly_execution"] = {
        "policy_path": str(runtime.POLICY_PATH.relative_to(campaign.ROOT)).replace("\\", "/"),
        "runtime_module": "experiments.bill_target74.desktop_friendly_runtime",
        "policy": runtime.POLICY,
        "user_authorized": True,
        "required_for_future_training_and_audits": True,
    }
    _write(campaign.PROTOCOL_PATH, campaign.PROTOCOL)
    campaign._append({"event": "card_registered", "card": CARD, "run_id": run_id,
                      "expected_fit_starts": 2, "type": card["type"],
                      "code_sha256": hashes, "resource_policy": runtime.POLICY,
                      "reserved_months_labels_opened": False, "hidden_labels_opened": False})
    campaign._write_progress("r173_replay_running", r173_run_id=run_id,
                             next_action="verify desktop-friendly scheduling reproduces both frozen safe-fold predictions")
    saved = pd.read_csv(reference_paired, dtype={"ml_num": str})
    _, parent_rows, _, _ = refit._parent_predictions(campaign, reference)
    if (len(saved) != 4662 or saved.duplicated(["month", "track", "ml_num"]).any()
            or set(saved.month.astype(str)) != set(MONTHS)):
        raise RuntimeError("reference predictions do not cover the registered unique safe rows")
    original_append = campaign._append
    fold_checks = []

    def record(event: dict) -> None:
        if event.get("card") != reference.CARD or event.get("stage") != "independent_audit":
            raise RuntimeError("unexpected event from frozen reference reconstruction")
        if event.get("event") == "fit_start":
            runtime.wait_before_fit(f"R173:fit:{event['month']}")
        original_append({**event, "card": CARD, "run_id": run_id,
                         "stage": "scheduling_replay",
                         "fit_id": event["fit_id"].replace(reference.CARD, CARD),
                         "reference_card": reference.CARD,
                         "runtime_settings": runtime.native_settings()})

    try:
        for month in MONTHS:
            runtime.wait_before_fit(f"R173:before_fold:{month}")
            detail = next(row for row in reference_result["fit_details"] if row["month"] == month)
            campaign._append = record
            try:
                checked = refit._reconstruct_fold(
                    campaign, reference, month, saved.loc[saved.month.eq(month)],
                    parent_rows, {**detail, "run_id": run_id})
            finally:
                campaign._append = original_append
            fold_checks.append(checked)
            _write(output / "partial_replay_checks.json", {"run_id": run_id, "fold_checks": fold_checks})
            print(json.dumps({"month": month, "prediction_max_abs_diff": checked["prediction_max_abs_diff"]}), flush=True)
    except BaseException as exc:
        card["status"] = "replay_failed"
        _write(output / "failure_checkpoint.json", {"run_id": run_id, "fold_checks": fold_checks,
                                                    "error": f"{type(exc).__name__}: {exc}"})
        original_append({"event": "card_failed", "card": CARD, "run_id": run_id,
                         "error": f"{type(exc).__name__}: {exc}",
                         "reserved_months_labels_opened": False, "hidden_labels_opened": False})
        _write(campaign.PROTOCOL_PATH, campaign.PROTOCOL)
        campaign._write_progress("r173_replay_failed", r173_run_id=run_id)
        raise
    shutil.copy2(reference_paired, output / "paired_predictions.csv")
    result = {"card": CARD, "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
              "type": card["type"], "fit_starts": 2, "fold_checks": fold_checks,
              "reference_run_id": reference_result["run_id"],
              "reference_screen_metrics": reference_result["tracks"]["list_aware"],
              "exact_prediction_replay_passed": all(row["prediction_max_abs_diff"] == 0 for row in fold_checks),
              "accuracy_improvement_claimed": False, "promoted": False,
              "resource_policy": runtime.POLICY, "runtime_settings": runtime.native_settings(),
              "reserved_months_labels_opened": False, "hidden_labels_opened": False}
    _write(output / "replay_results.json", result)
    _write(output / "run_manifest.json", {
        "campaign_id": campaign.PROTOCOL["campaign_id"], "card": CARD, "run_id": run_id,
        "code_sha256": hashes, "source_snapshots": snapshots, "reference_hashes": reference_hashes,
        "source_identity": source, "result_sha256": reference._sha256(output / "replay_results.json"),
        "paired_predictions_sha256": reference._sha256(output / "paired_predictions.csv"),
        "reserved_months_labels_opened": False, "hidden_labels_opened": False})
    card.update(status="replay_pending_integrity_audit",
                result_path=str((output / "replay_results.json").relative_to(campaign.ROOT)).replace("\\", "/"))
    _write(campaign.PROTOCOL_PATH, campaign.PROTOCOL)
    original_append({"event": "card_complete", "card": CARD, "run_id": run_id,
                     "exact_prediction_replay_passed": result["exact_prediction_replay_passed"],
                     "reserved_months_labels_opened": False, "hidden_labels_opened": False})
    campaign._write_progress("r173_replay_pending_integrity_audit", r173_run_id=run_id,
                             next_action="run R173 no-refit artifact and ledger audit")
    return {"run_id": run_id, "exact_prediction_replay_passed": result["exact_prediction_replay_passed"],
            "fit_starts": 2, "result_path": card["result_path"]}


def audit(run_id: str) -> dict:
    from experiments.bill_target74 import runner as campaign
    from experiments.bill_target74.r172_listaware_fair_loss import _sha256

    card = campaign.PROTOCOL["cards"][CARD]
    if card["run_id"] != run_id or card["status"] != "replay_pending_integrity_audit":
        raise RuntimeError("R173 run is not pending its integrity audit")
    result_path = campaign.ROOT / card["result_path"]
    output = result_path.parent
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    events = [json.loads(line) for line in campaign.LEDGER.read_text(encoding="utf-8").splitlines() if line.strip()]
    events = [row for row in events if row.get("card") == CARD and row.get("run_id") == run_id]
    starts = [row for row in events if row["event"] == "fit_start"]
    completions = [row for row in events if row["event"] == "fit_complete"]
    checks = {
        "identity": result["run_id"] == manifest["run_id"] == run_id and result["card"] == CARD,
        "result_hash": _sha256(result_path) == manifest["result_sha256"],
        "paired_hash": _sha256(output / "paired_predictions.csv") == manifest["paired_predictions_sha256"],
        "reference_hashes": all(_sha256(campaign.ROOT / p) == h for p, h in manifest["reference_hashes"].items()),
        "source_and_snapshots": all(_sha256(campaign.ROOT / p) == h and
            _sha256(output / "provenance" / manifest["source_snapshots"][p]) == h
            for p, h in manifest["code_sha256"].items()),
        "safe_folds": {row["month"] for row in result["fold_checks"]} == set(MONTHS),
        "exact_predictions": len(result["fold_checks"]) == 2 and all(
            row["prediction_max_abs_diff"] == 0 and row["predictions_reconstructed"]
            and row["non_detached_predictions_unchanged"] for row in result["fold_checks"]),
        "two_completed_fits": len(starts) == len(completions) == 2
            and {row["fit_id"] for row in starts} == {row["fit_id"] for row in completions}
            and all(row.get("status") == "complete" for row in completions),
        "resource_settings": all(row["runtime_settings"]["priority_class"] == 0x4000
            and row["runtime_settings"]["logical_processors"] <= 4 for row in starts + completions),
        "unbounded_budget": all(v is None for k, v in campaign.PROTOCOL["limits"].items() if k != "learner_threads"),
        "safe_access_flags": all(row.get("reserved_months_labels_opened") is False
            and row.get("hidden_labels_opened") is False for row in [result, manifest, *events]),
        "no_promotion": result["promoted"] is False and result["accuracy_improvement_claimed"] is False,
    }
    audit_result = {"run_id": run_id, "audit_passed": all(checks.values()), "checks": checks,
                    "refits_added_by_this_audit": 0}
    _write(output / "integrity_audit.json", audit_result)
    card.update(status="replay_verified" if audit_result["audit_passed"] else "replay_audit_failed",
                audit_passed=audit_result["audit_passed"])
    _write(campaign.PROTOCOL_PATH, campaign.PROTOCOL)
    campaign._append({"event": "integrity_audit_complete", "card": CARD, **audit_result,
                      "reserved_months_labels_opened": False, "hidden_labels_opened": False})
    campaign._write_progress("r173_desktop_friendly_verified" if audit_result["audit_passed"] else "r173_audit_failed",
                             r173_audit_passed=audit_result["audit_passed"],
                             next_action="continue distinct list-aware model hypotheses via the desktop-friendly runtime; targets and promotion gates unchanged")
    if not audit_result["audit_passed"]:
        raise RuntimeError(f"R173 integrity audit failed: {checks}")
    return audit_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit")
    args = parser.parse_args()
    print(json.dumps(audit(args.audit) if args.audit else run(), indent=2), flush=True)
