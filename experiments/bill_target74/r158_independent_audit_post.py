"""Post-run audit for R158, accounting for its recorded bootstrap-seed deviation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from experiments.bill_target74 import runner as campaign
from experiments.bill_target74 import r158_independent_audit as registered_audit


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _audit(run_id: str) -> dict:
    # R158 delegates scoring to R157._score, whose fixed seed is 20261157.
    # The R158 card declared 20261158, but the registered runner did not pass that seed.
    actual_seed = 20261157
    registered_audit.BOOTSTRAP_SEED = actual_seed
    summary = registered_audit._audit(run_id)
    audit_path = campaign.CAMPAIGN_ROOT / "batches" / run_id / "independent_audit.json"
    summary["status"] = "passed_with_method_deviation"
    summary["audit_tool_path"] = str(Path(__file__).relative_to(campaign.ROOT)).replace("\\", "/")
    summary["audit_tool_sha256"] = _sha256(Path(__file__))
    summary["method_deviation"] = {
        "field": "bootstrap_seed",
        "registered_seed": 20261158,
        "actual_seed": actual_seed,
        "cause": "R158 called the R157 score helper, which hard-codes bootstrap seed 20261157.",
        "impact": "Point estimates and promotion decision are unchanged; both seed results have a positive candidate-minus-parent bootstrap interval and fail the screen gates.",
        "development_authorized": False,
    }
    audit_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    card = campaign.PROTOCOL["cards"]["R158_listaware_r35_past_oof_median_ratio_shift"]
    card["screen_audit_status"] = "passed_with_method_deviation"
    card["screen_audit_path"] = str(audit_path.relative_to(campaign.ROOT)).replace("\\", "/")
    card["screen_audit_sha256"] = _sha256(audit_path)
    card["screen_bootstrap_seed_actual"] = actual_seed
    card["screen_method_deviation"] = summary["method_deviation"]
    campaign.PROTOCOL["cards"]["R158_listaware_r35_past_oof_median_ratio_shift"] = card
    campaign.PROTOCOL_PATH.write_text(json.dumps(campaign.PROTOCOL, indent=2, ensure_ascii=False) + "\n",
                                      encoding="utf-8")
    campaign._write_progress(
        "r158_screen_audit_method_deviation",
        next_action="retain R35; R158 failed the fixed screen and its bootstrap seed differed from preregistration",
        r158_audit=str(audit_path.relative_to(campaign.ROOT)).replace("\\", "/"),
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-run R158 audit with the actual score seed")
    parser.add_argument("run_id")
    args = parser.parse_args()
    print(json.dumps(_audit(args.run_id), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
