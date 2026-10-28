"""Post-audit correction for R161's campaign-global label-flag check."""
from __future__ import annotations

import json
from pathlib import Path

from experiments.bill_target74 import r154_listaware_lgbm_quantile_residual as r154
from experiments.bill_target74 import r161_listaware_seed_oof_catboost as r161
from experiments.bill_target74 import runner as campaign


def audit_post() -> dict:
    card = campaign.PROTOCOL["cards"][r161.CARD]
    run_id = card["last_run_id"]
    output = campaign.CAMPAIGN_ROOT / "batches" / run_id
    original_path = output / "independent_audit.json"
    result_path = output / "screen_results.json"
    manifest_path = output / "run_manifest.json"
    original = json.loads(original_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (original.get("run_id") != run_id or original.get("stage") != "screen"
            or r161._sha(original_path) != card.get("audit_sha256")):
        raise RuntimeError("original R161 audit identity or recorded hash failed")

    inputs = card["parent_oof_inputs"]
    safe_frame = r154._load_frame(inputs, "screen")
    reserved = set(campaign.PROTOCOL["reserved_months"])
    ledger = [json.loads(line) for line in campaign.LEDGER.read_text(encoding="utf-8").splitlines() if line.strip()]
    run_events = [event for event in ledger
                  if event.get("card") == r161.CARD and event.get("run_id") == run_id]
    relevant = {"card_start", "fit_start", "fit_complete", "fold_complete", "card_complete",
                "independent_audit_complete"}
    run_flags_ok = all(
        event.get("reserved_months_labels_opened") is False
        and event.get("hidden_labels_opened") is False
        for event in run_events if event.get("event") in relevant
    )
    checks = dict(original.get("checks", {}))
    prior_global_reserve_event = campaign.PROTOCOL.get("reserved_months_labels_opened") is True
    checks["r161_per_run_label_flags"] = (
        result.get("reserved_months_labels_opened") is False
        and result.get("hidden_labels_opened") is False
        and manifest.get("reserved_months_labels_opened") is False
        and manifest.get("hidden_labels_opened") is False
        and card.get("reserved_months_labels_opened") is False
        and card.get("hidden_labels_opened") is False
        and run_flags_ok
    )
    checks["safe_input_month_scope"] = not (set(safe_frame.month.astype(str)) & reserved)
    checks["historical_reserve_flag_attributed"] = prior_global_reserve_event and campaign.PROTOCOL.get(
        "reserve_validation_eligible") is False
    # The locked auditor required a campaign-global hidden-label key. That key is
    # absent; R161's result, manifest, card, and every run event carry explicit false flags.
    checks.pop("safe_label_flags", None)
    audit_passed = (all(value is True for value in checks.values())
                    and original.get("predictions_reconstructed") is None
                    and all(original.get("checks", {}).get(key) is True for key in (
                        "identity", "result_hash", "paired_hash", "input_artifacts",
                        "current_best_identity", "source_hashes", "source_snapshots",
                        "data_identity", "safe_oof_scope", "predictions_reconstructed",
                        "metrics_recomputed", "screen_result_qualification", "fit_ledger",
                        "no_access_flags_in_ledger")))
    output_audit = {
        "card": r161.CARD, "run_id": run_id, "stage": "screen",
        "audit_passed": bool(audit_passed),
        "promotion_gate_passed": result.get("qualifies") is True,
        "checks": checks,
        "corrected_check": {
            "original_check": "safe_label_flags",
            "cause": "The initial auditor required campaign_protocol.hidden_labels_opened == false, but the campaign has no such global field. This was a pre-existing flag-check defect, not a label access event.",
            "scope": "Per-run card/result/manifest/ledger flags are checked directly. The separate campaign-global reserved-month flag remains true due to the documented historical access incident; reserve eligibility remains false.",
            "independent_reconstruction": original.get("reconstruction_checks"),
        },
        "screen_metrics": original.get("recomputed_scores"),
        "source_audit_artifact": r161._relative(original_path),
        "source_audit_sha256": r161._sha(original_path),
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    }
    post_path = output / "independent_audit_post.json"
    post_path.write_text(json.dumps(output_audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    post_sha = r161._sha(post_path)
    campaign._append({"event": "independent_audit_correction", "card": r161.CARD,
                      "run_id": run_id, "stage": "screen", "audit_passed": bool(audit_passed),
                      "corrected_check": "campaign-global hidden_labels_opened flag does not exist; run-local flags verified false",
                      "post_audit_path": r161._relative(post_path), "post_audit_sha256": post_sha,
                      "reserved_months_labels_opened": False, "hidden_labels_opened": False})
    card["post_audit_path"] = r161._relative(post_path)
    card["post_audit_sha256"] = post_sha
    card["post_audit_passed"] = bool(audit_passed)
    card["audit_passed"] = bool(audit_passed)
    # The screen remains failed: a clean audit does not turn a negative model result into a pass.
    card["status"] = "screen_failed"
    campaign.PROTOCOL_PATH.write_text(json.dumps(campaign.PROTOCOL, indent=2, ensure_ascii=False) + "\n",
                                      encoding="utf-8")
    campaign._write_progress("r161_screen_failed_audited",
                             next_action="retain R35 validated parent and register a distinct list-aware hypothesis; R161 seed-history correction failed screen",
                             r161_card=r161.CARD, r161_run_id=run_id,
                             r161_audit_path=r161._relative(post_path), r161_audit_passed=bool(audit_passed),
                             r161_promotion_gate_passed=False)
    return output_audit


if __name__ == "__main__":
    print(json.dumps(audit_post(), indent=2, ensure_ascii=False), flush=True)
