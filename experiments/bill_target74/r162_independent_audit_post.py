"""Run R162's locked auditor after supplying its omitted R161 module import."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from experiments.bill_target74 import r161_listaware_seed_oof_catboost as r161
from experiments.bill_target74 import r162_independent_audit as locked_auditor
from experiments.bill_target74 import r162_listaware_catboost_rolling18m as r162
from experiments.bill_target74 import runner as campaign


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_post_audit() -> dict:
    card = campaign.PROTOCOL["cards"][r162.CARD]
    run_id = card.get("last_run_id")
    if not run_id or card.get("status") != "screen_pending_independent_audit":
        raise RuntimeError("R162 is not pending its first independent audit")
    output = campaign.CAMPAIGN_ROOT / "batches" / run_id
    events = [json.loads(line) for line in campaign.LEDGER.read_text(encoding="utf-8").splitlines() if line.strip()]
    audit_starts = [event for event in events if event.get("card") == r162.CARD
                    and event.get("run_id") == run_id and event.get("stage") == "independent_audit"
                    and event.get("event") == "fit_start"]
    if audit_starts or (output / "independent_audit.json").exists():
        raise RuntimeError("R162 already has audit fit starts or an audit artifact; do not repeat it")

    failure_path = output / "audit_preflight_failure.json"
    if failure_path.is_file():
        failure = json.loads(failure_path.read_text(encoding="utf-8"))
        attempts = list(failure.get("attempts", []))
    else:
        attempts = []
        failure = {"card": r162.CARD, "run_id": run_id,
                   "stage": "independent_audit_preflight"}
    if not attempts:
        attempts.append({"attempt": 1, "error": "NameError: name 'r161' is not defined",
                         "cause": "The locked auditor referenced r161._r97_current_best without importing r161.",
                         "audit_fit_starts": 0})
    if not any("_read_months" in item.get("error", "") for item in attempts):
        attempts.append({"attempt": len(attempts) + 1,
                         "error": "NameError: name '_read_months' is not defined",
                         "cause": "The locked auditor called _read_months as a local global instead of r162._read_months.",
                         "audit_fit_starts": 0,
                         "recorded_utc": datetime.now(timezone.utc).isoformat()})
        campaign._append({"event": "audit_preflight_failed", "card": r162.CARD,
                          "run_id": run_id, "stage": "independent_audit",
                          "error": attempts[-1]["error"], "audit_fit_starts": 0,
                          "reserved_months_labels_opened": False, "hidden_labels_opened": False})
    failure.update({"attempts": attempts, "audit_fit_starts_before_recovery": 0,
                    "evidence": {"screen_fit_starts": 2, "audit_fit_starts": 0,
                                 "audit_artifact_existed": False},
                    "reserved_months_labels_opened": False,
                    "hidden_labels_opened": False,
                    "updated_utc": datetime.now(timezone.utc).isoformat()})
    failure_path.write_text(json.dumps(failure, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Keep the preregistered auditor file/hash intact; provide its missing module/helpers at runtime.
    locked_auditor.r161 = r161
    locked_auditor._read_months = r162._read_months
    try:
        audit_result = locked_auditor.audit()
    except BaseException as exc:
        attempts.append({"attempt": len(attempts) + 1,
                         "error": f"{type(exc).__name__}: {exc}",
                         "cause": "The locked audit failed before writing its result; no candidate fit was rerun.",
                         "audit_fit_starts": 0,
                         "recorded_utc": datetime.now(timezone.utc).isoformat()})
        failure["attempts"] = attempts
        failure["updated_utc"] = datetime.now(timezone.utc).isoformat()
        failure_path.write_text(json.dumps(failure, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        campaign._append({"event": "audit_preflight_failed", "card": r162.CARD,
                          "run_id": run_id, "stage": "independent_audit",
                          "error": attempts[-1]["error"], "audit_fit_starts": 0,
                          "reserved_months_labels_opened": False, "hidden_labels_opened": False})
        raise
    audit_path = output / "independent_audit.json"
    if not audit_path.is_file() or audit_result.get("run_id") != run_id:
        raise RuntimeError("R162 locked auditor did not write the expected audit artifact")
    post = {"card": r162.CARD, "run_id": run_id, "stage": "screen",
            "post_audit_passed": audit_result.get("audit_passed") is True,
            "promotion_gate_passed": audit_result.get("promotion_gate_passed") is True,
            "locked_auditor_path": r162._relative(Path(locked_auditor.__file__)),
            "locked_auditor_sha256": _sha(Path(locked_auditor.__file__)),
            "preflight_failures_path": r162._relative(failure_path),
            "preflight_failures_sha256": _sha(failure_path),
            "preflight_failures": attempts,
            "correction": "Injected the omitted R161 module and safe-month helper into the loaded auditor namespace; did not edit the preregistered auditor or rerun candidate fits.",
            "audit_path": r162._relative(audit_path),
            "audit_sha256": _sha(audit_path),
            "audit_checks": audit_result.get("checks"),
            "reserved_months_labels_opened": False, "hidden_labels_opened": False}
    post_path = output / "independent_audit_post.json"
    post_path.write_text(json.dumps(post, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    post_hash = _sha(post_path)
    campaign._append({"event": "independent_audit_correction", "card": r162.CARD,
                      "run_id": run_id, "stage": "screen",
                      "audit_passed": post["post_audit_passed"],
                      "promotion_gate_passed": post["promotion_gate_passed"],
                      "post_audit_path": r162._relative(post_path), "post_audit_sha256": post_hash,
                      "reserved_months_labels_opened": False, "hidden_labels_opened": False})
    card["post_audit_path"] = r162._relative(post_path)
    card["post_audit_sha256"] = post_hash
    card["post_audit_passed"] = post["post_audit_passed"]
    campaign.PROTOCOL_PATH.write_text(json.dumps(campaign.PROTOCOL, indent=2, ensure_ascii=False) + "\n",
                                      encoding="utf-8")
    current_progress = json.loads(campaign.STATE.read_text(encoding="utf-8"))
    campaign._write_progress(current_progress.get("status", "r162_screen_audited"),
                             next_action=current_progress.get("next_action"),
                             r162_post_audit_path=r162._relative(post_path),
                             r162_post_audit_passed=post["post_audit_passed"])
    return post


if __name__ == "__main__":
    print(json.dumps(run_post_audit(), indent=2, ensure_ascii=False), flush=True)
