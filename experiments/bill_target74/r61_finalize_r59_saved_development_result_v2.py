"""R61: finalize R59 from saved predictions with the score schema used by R59."""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.bill_target74 import r19_expert_weight as common
from experiments.bill_target74 import r22_remarks_tfidf as scoring
from experiments.bill_target74 import r58_past_log_residual_slope_calibration as r58
from experiments.bill_target74 import r59_past_log_residual_calibration_rounding_fix as r59
from experiments.bill_target74 import runner as campaign


CARD = "R61_finalize_r59_saved_development_result_v2"
R59_CARD = r59.CARD
R60_CARD = "R60_finalize_r59_saved_development_result"


def _save_protocol() -> None:
    campaign.PROTOCOL_PATH.write_text(
        json.dumps(campaign.PROTOCOL, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _events() -> tuple[str, list[dict]]:
    rows = [json.loads(line) for line in campaign.LEDGER.read_text(encoding="utf-8").splitlines()]
    starts = [x for x in rows if x.get("card") == R59_CARD and x.get("event") == "card_start"]
    if len(starts) != 1:
        raise RuntimeError("expected one R59 start event")
    run_id = starts[0]["run_id"]
    return run_id, [x for x in rows if x.get("card") == R59_CARD and x.get("run_id") == run_id]


def register_card() -> dict:
    existing = campaign.PROTOCOL["cards"].get(CARD)
    if existing:
        if existing.get("status") == "registered" and existing.get("source_sha256") == common._sha256(Path(__file__)):
            return existing
        raise RuntimeError(f"R61 is already {existing.get('status')}; inspect before rerunning")
    r59_card = campaign.PROTOCOL["cards"].get(R59_CARD)
    if not r59_card or r59_card.get("status") != "registered":
        raise RuntimeError("R61 requires R59's saved but unfinalized development outputs")
    if r59_card.get("source_sha256") != common._sha256(Path(r59.__file__)):
        raise RuntimeError("R59 source no longer matches its preregistered hash")

    run_id, rows = _events()
    out = campaign.CAMPAIGN_ROOT / "batches" / run_id
    result_path, paired_path = out / "development_results.json", out / "paired_predictions.csv"
    if not result_path.is_file() or not paired_path.is_file():
        raise FileNotFoundError("R59 result and paired-prediction artifacts must both exist")
    starts = [x for x in rows if x.get("event") == "fit_start"]
    done = [x for x in rows if x.get("event") == "fit_complete"]
    reuse = [x for x in rows if x.get("event") == "fit_reuse"]
    if len(starts) != r59.EXPECTED_NEW_FITS or len(done) != r59.EXPECTED_NEW_FITS:
        raise RuntimeError("R59 ledger does not show all preregistered fits completed")
    if any(x.get("status") != "complete" for x in done) or len(reuse) != 1:
        raise RuntimeError("R59 ledger contains a failed fit or missing reused fold")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if (result.get("fit_starts") != r59.EXPECTED_NEW_FITS
            or len(result.get("fit_details", [])) != r59.EXPECTED_NEW_FITS
            or len(result.get("reused_fit_details", [])) != 1):
        raise RuntimeError("R59 saved outputs do not match its ledger counts")
    if any(x.get("event") in {"card_complete", "card_failed"} for x in rows):
        raise RuntimeError("R59 already has a terminal event")

    r60_card = campaign.PROTOCOL["cards"].get(R60_CARD)
    if r60_card and r60_card.get("status") == "registered":
        r60_card.update({"status": "failed", "failure_reason": "score verification expected metrics not emitted by the registered score function"})
        campaign._append({
            "event": "card_failed", "card": R60_CARD,
            "error": "KeyError: p90_ape_percent is not part of score_development_confirmation output",
            "fit_starts": 0, "target_card": R59_CARD,
        })

    registration = {
        "status": "registered", "operation": "zero_fit_saved_artifact_verification_and_finalization",
        "target_card": R59_CARD, "target_run_id": run_id, "expected_fit_starts": 0,
        "saved_artifact_hashes": {
            "development_results_sha256": common._sha256(result_path),
            "paired_predictions_sha256": common._sha256(paired_path),
        },
        "observed_fit_starts": len(starts), "observed_fit_completions": len(done),
        "observed_reused_folds": len(reuse),
        "prior_finalizer_failure": "R60 checked p90_ape_percent, which this score result does not expose; no artifacts were changed.",
        "source_sha256": common._sha256(Path(__file__)),
        "data_sha256": campaign.PROTOCOL["data_sha256"],
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    }
    campaign.PROTOCOL["cards"][CARD] = registration
    _save_protocol()
    campaign._append({
        "event": "card_registered", "card": CARD, "target_card": R59_CARD,
        "target_run_id": run_id, "expected_fit_starts": 0,
        "source_sha256": registration["source_sha256"],
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    campaign._write_progress(
        "r61_registered", next_action="verify R59 saved outputs and finish short-path provenance packaging",
        r61_card=CARD,
    )
    return registration


def _verify(run_id: str, rows: list[dict], result: dict, paired: pd.DataFrame) -> dict:
    if paired.empty or paired.duplicated(["month", "track", "ml_num"]).any():
        raise RuntimeError("R59 paired predictions are empty or contain duplicate row keys")
    if set(paired.month.astype(str).unique()) != set(r58.MONTHS) or set(paired.track.astype(str).unique()) != set(r58.TRACKS):
        raise RuntimeError("R59 paired predictions do not cover the registered months and tracks")
    price_columns = ["sold_price", "parent_prediction", "candidate_prediction"]
    prices = paired[price_columns].to_numpy(dtype=float)
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise RuntimeError("R59 saved prices are invalid")

    parent_rows = {
        "off_market": r58._verified_input(campaign, r58.R52_RESULT, r58.R52_RUN, "off_market")[4],
        "list_aware": r58._verified_input(campaign, r58.R35_DEV_RESULT, r58.R35_DEV_RUN, "list_aware")[4],
    }
    recomputed = {}
    for track in r58.TRACKS:
        current = paired.loc[paired.track.eq(track)].copy()
        parent = parent_rows[track][["month", "track", "ml_num", "sold_price", "candidate_prediction"]].rename(
            columns={"candidate_prediction": "expected_parent_prediction"}
        )
        merged = current.merge(parent, on=["month", "track", "ml_num", "sold_price"], how="outer", indicator=True)
        if not merged._merge.eq("both").all():
            raise RuntimeError(f"R59 {track} rows or actual labels differ from its registered parent")
        if not np.array_equal(merged.parent_prediction.to_numpy(float), merged.expected_parent_prediction.to_numpy(float)):
            raise RuntimeError(f"R59 {track} parent predictions differ from the registered source")
        applied = current.loc[current.calibration_applied.astype(bool)]
        max_pre_round = float(applied.max_abs_applied_log_correction_before_rounding.max()) if len(applied) else 0.0
        if max_pre_round > r58.MAX_ABS_LOG_CORRECTION + 1e-12:
            raise RuntimeError(f"R59 {track} correction exceeds its registered bound before rounding")
        recomputed[track] = scoring.score_development_confirmation(
            current, months=r58.MONTHS,
            target_mdape_strictly_below=7.0 if track == "off_market" else 4.0,
            bootstrap_replicates=r58.BOOTSTRAP_REPLICATES,
            bootstrap_seed=r58.BOOTSTRAP_SEED,
            minimum_relative_gain_percent=0.5,
            minimum_months_improved=6,
            maximum_worst_month_regression_percent=5.0,
            maximum_mape_regression_percent=1.0,
        )
        stored = result["tracks"][track]
        for side in ("parent", "candidate"):
            for metric in ("n", "mdape_percent", "mape_percent"):
                if not np.isclose(stored[side][metric], recomputed[track][side][metric], rtol=0, atol=1e-12):
                    raise RuntimeError(f"R59 {track} {side}.{metric} does not reproduce from saved paired predictions")
        for key in ("relative_mdape_improvement_percent", "mape_relative_change_percent"):
            if not np.isclose(stored[key], recomputed[track][key], rtol=0, atol=1e-12):
                raise RuntimeError(f"R59 {track} {key} does not reproduce")
        if stored["qualifies_development"] != recomputed[track]["qualifies_development"]:
            raise RuntimeError(f"R59 {track} development gate does not reproduce")

    starts = {x["fit_id"] for x in rows if x.get("event") == "fit_start"}
    complete_rows = [x for x in rows if x.get("event") == "fit_complete"]
    complete = {x["fit_id"] for x in complete_rows if x.get("status") == "complete"}
    saved = {x["fit_id"] for x in result["fit_details"]}
    reuse = [x for x in rows if x.get("event") == "fit_reuse"]
    if len(starts) != r59.EXPECTED_NEW_FITS or starts != complete or starts != saved:
        raise RuntimeError("R59 new fit IDs differ between the ledger and result artifact")
    if len(reuse) != 1 or reuse[0].get("fit_starts") != 0:
        raise RuntimeError("R59 reused-fold ledger entry is missing or counted as a fit")
    if any(result["tracks"][t]["qualifies_development"] for t in r58.TRACKS):
        raise RuntimeError("R59 unexpectedly has a development-passing track")

    rounded_deltas = paired.loc[paired.calibration_applied.astype(bool), "max_abs_rounded_log_delta"]
    return {
        "new_fit_starts": len(starts), "reused_folds": len(reuse),
        "row_counts": {t: int(paired.track.eq(t).sum()) for t in r58.TRACKS},
        "max_abs_applied_correction_before_rounding": float(paired.max_abs_applied_log_correction_before_rounding.max()),
        "max_abs_rounded_log_delta": float(rounded_deltas.max()) if len(rounded_deltas) else 0.0,
        "recomputed_metrics_match": True,
        "recomputed_development_gates": {t: recomputed[t]["qualifies_development"] for t in r58.TRACKS},
    }


def run() -> dict:
    card = campaign.PROTOCOL["cards"].get(CARD)
    r59_card = campaign.PROTOCOL["cards"].get(R59_CARD)
    if not card or card.get("status") != "registered" or not r59_card or r59_card.get("status") != "registered":
        raise RuntimeError("R61 and R59 are not in their registered states")
    if card.get("source_sha256") != common._sha256(Path(__file__)):
        raise RuntimeError("R61 source changed after registration")
    if r59_card.get("source_sha256") != common._sha256(Path(r59.__file__)):
        raise RuntimeError("R59 source differs from its registration")
    run_id, rows = _events()
    out = campaign.CAMPAIGN_ROOT / "batches" / run_id
    result_path, paired_path = out / "development_results.json", out / "paired_predictions.csv"
    expected_hashes = card["saved_artifact_hashes"]
    if common._sha256(result_path) != expected_hashes["development_results_sha256"] or common._sha256(paired_path) != expected_hashes["paired_predictions_sha256"]:
        raise RuntimeError("R59 saved artifacts changed after R61 registration")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    paired = pd.read_csv(paired_path, dtype={"ml_num": str})
    verification = _verify(run_id, rows, result, paired)

    provenance = out / "provenance"
    sources = [*r58._source_files(campaign), Path(r59.__file__)]
    source_map = []
    for index, source in enumerate(sources):
        old_snapshot = provenance / f"src_{index:02d}_{source.name}.source"
        snapshot = old_snapshot if old_snapshot.is_file() else source
        digest = common._sha256(snapshot)
        if source == Path(r59.__file__) and digest != r59_card["source_sha256"]:
            raise RuntimeError("R59 provenance source hash differs from its registration")
        short = provenance / f"code{index:02d}.source"
        if snapshot != short:
            shutil.copy2(snapshot, short)
        source_map.append({
            "source_path": str(source.relative_to(campaign.ROOT)).replace("\\", "/"),
            "snapshot_path": str(short.relative_to(out)).replace("\\", "/"),
            "sha256": digest,
        })
    common._write_json(provenance / "source_map.json", source_map)

    report = [
        "# R59 development result — finalized from saved artifacts", "",
        f"Run: `{run_id}`. R61 used zero learner fits and did not alter predictions.",
        "R59 completed its scoring and saved outputs; only final provenance copying failed because the last Windows destination path was too long.",
        "R61 verified parent-row identity, fit IDs, prediction hashes, and reproduced the registered scores and gates.",
        "The ±0.05 bound applies to the clipped correction before dollar rounding. Rounding can produce a tiny larger log ratio.",
        "No 2025-05/08 or hidden labels were accessed.", "",
        "| Track | Parent MdAPE | Candidate MdAPE | Relative gain | MAPE change | Months improved | Bootstrap upper95 (pp) | Target met | Gate |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for track, score in result["tracks"].items():
        report.append(
            f"| {track} | {score['parent']['mdape_percent']:.5f}% | {score['candidate']['mdape_percent']:.5f}% | "
            f"{score['relative_mdape_improvement_percent']:+.3f}% | {score['mape_relative_change_percent']:+.3f}% | "
            f"{score['months_improved']}/8 | {score['bootstrap_mdape_difference_percent']['upper95']:+.5f} | "
            f"{'yes' if score['target_met'] else 'no'} | {'pass' if score['qualifies_development'] else 'fail'} |"
        )
    report.extend([
        "", f"New fits: {verification['new_fit_starts']}; reused calibration folds: {verification['reused_folds']}; "
        f"rows: {verification['row_counts']}.",
        f"Maximum applied correction before rounding: {verification['max_abs_applied_correction_before_rounding']:.12f}; "
        f"maximum rounded log delta: {verification['max_abs_rounded_log_delta']:.12f}.",
    ])
    report_path = out / "research_report.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    protocol_snapshot = provenance / "src_00_campaign_protocol.json.source"
    manifest = {
        "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
        "finalization_card": CARD, "finalizer_source_sha256": card["source_sha256"],
        "protocol_snapshot_sha256_at_r59_run": common._sha256(protocol_snapshot),
        "code_snapshots": source_map, "data_sha256": campaign.PROTOCOL["data_sha256"],
        "fit_starts": result["fit_starts"], "reused_calibration_folds": result["reused_calibration_folds"],
        "result_sha256": common._sha256(result_path), "paired_predictions_sha256": common._sha256(paired_path),
        "report_sha256": common._sha256(report_path), "verification": verification,
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    }
    manifest_path = out / "finalization_manifest.json"
    common._write_json(manifest_path, manifest)

    r58_card = campaign.PROTOCOL["cards"].get(r58.CARD)
    if r58_card and r58_card.get("status") == "registered":
        r58_card.update({
            "status": "failed", "run_id": r58_card.get("run_id") or r59.R58_RUN,
            "fit_starts": 1,
            "failure_reason": "post-rounding assertion compared rounded predictions with the pre-rounding correction bound",
            "finalized_by": CARD,
        })
    r59_card.update({
        "status": "development_failed", "run_id": run_id,
        "result_path": str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        "paired_predictions_path": str(paired_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        "manifest_path": str(manifest_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        "report_path": str(report_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        "qualifies_by_track": {t: x["qualifies_development"] for t, x in result["tracks"].items()},
        "development_metrics": result["tracks"], "finalized_by": CARD,
    })
    card.update({
        "status": "finalization_complete", "run_id": run_id, "fit_starts": 0,
        "manifest_path": str(manifest_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        "report_path": str(report_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        "verification": verification,
    })
    _save_protocol()
    campaign._append({
        "event": "card_start", "card": CARD, "run_id": run_id,
        "target_card": R59_CARD, "expected_fit_starts": 0,
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    campaign._append({
        "event": "card_complete", "card": CARD, "run_id": run_id,
        "target_card": R59_CARD, "fit_starts": 0,
        "verified_fit_starts": result["fit_starts"],
        "qualifies_by_track": r59_card["qualifies_by_track"],
        "manifest_path": str(manifest_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    campaign._write_progress(
        "r59_development_failed",
        next_action="R59 failed both development gates; retain R22/R35 and register a distinct hypothesis",
        r59_run_id=run_id, r59_fit_starts=result["fit_starts"],
        r59_reused_calibration_folds=result["reused_calibration_folds"],
        r61_finalization_manifest=str(manifest_path.relative_to(campaign.ROOT)).replace("\\", "/"),
    )
    return manifest


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Register or finalize R59's saved development result")
    parser.add_argument("--register", action="store_true")
    args = parser.parse_args()
    print(json.dumps(register_card() if args.register else run(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
