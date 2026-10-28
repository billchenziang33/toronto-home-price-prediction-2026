"""R60: verify and finalize R59's saved outputs after a Windows path-length failure."""
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


CARD = "R60_finalize_r59_saved_development_result"
R59_CARD = r59.CARD


def _save_protocol() -> None:
    campaign.PROTOCOL_PATH.write_text(
        json.dumps(campaign.PROTOCOL, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _r59_run_events() -> tuple[str, list[dict]]:
    events = [json.loads(line) for line in campaign.LEDGER.read_text(encoding="utf-8").splitlines()]
    starts = [row for row in events if row.get("card") == R59_CARD and row.get("event") == "card_start"]
    if len(starts) != 1:
        raise RuntimeError("expected exactly one R59 card_start event")
    run_id = starts[0]["run_id"]
    rows = [row for row in events if row.get("card") == R59_CARD and row.get("run_id") == run_id]
    return run_id, rows


def register_card() -> dict:
    existing = campaign.PROTOCOL["cards"].get(CARD)
    if existing:
        if existing.get("status") == "registered" and existing.get("source_sha256") == common._sha256(Path(__file__)):
            return existing
        raise RuntimeError(f"R60 is already {existing.get('status')}; inspect before rerunning")
    r59_card = campaign.PROTOCOL["cards"].get(R59_CARD)
    if not r59_card or r59_card.get("status") != "registered":
        raise RuntimeError("R60 requires R59's interrupted packaging state")
    if r59_card.get("source_sha256") != common._sha256(Path(r59.__file__)):
        raise RuntimeError("R59 registered source no longer matches its saved script")

    run_id, rows = _r59_run_events()
    out = campaign.CAMPAIGN_ROOT / "batches" / run_id
    result_path, paired_path = out / "development_results.json", out / "paired_predictions.csv"
    if not result_path.is_file() or not paired_path.is_file():
        raise FileNotFoundError("R59 saved development outputs are incomplete")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    starts = [row for row in rows if row.get("event") == "fit_start"]
    completes = [row for row in rows if row.get("event") == "fit_complete"]
    failed = [row for row in completes if row.get("status") != "complete"]
    reuse = [row for row in rows if row.get("event") == "fit_reuse"]
    if len(starts) != r59.EXPECTED_NEW_FITS or len(completes) != r59.EXPECTED_NEW_FITS or failed:
        raise RuntimeError("R59 ledger does not show all preregistered new fits completed successfully")
    if len(reuse) != 1 or reuse[0].get("fit_starts") != 0:
        raise RuntimeError("R59 must reuse exactly one completed calibration fold without refitting")
    if result.get("fit_starts") != r59.EXPECTED_NEW_FITS or len(result.get("reused_fit_details", [])) != 1:
        raise RuntimeError("R59 saved results do not match its registered fit and reuse counts")
    if any(row.get("event") in {"card_complete", "card_failed"} for row in rows):
        raise RuntimeError("R59 already has a terminal ledger event; do not finalize twice")

    registration = {
        "status": "registered", "operation": "zero_fit_artifact_verification_and_finalization",
        "target_card": R59_CARD, "target_run_id": run_id,
        "expected_fit_starts": 0,
        "saved_artifact_hashes": {
            "development_results_sha256": common._sha256(result_path),
            "paired_predictions_sha256": common._sha256(paired_path),
        },
        "observed_fit_starts": len(starts), "observed_fit_completions": len(completes),
        "observed_reused_folds": len(reuse),
        "failure_context": "R59 completed all fits and wrote results; copying the last provenance source failed because its destination exceeded Windows MAX_PATH.",
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
        "r60_registered", next_action="verify the saved R59 predictions, finish short-path provenance packaging, and close R59",
        r60_card=CARD,
    )
    return registration


def _verify_saved_result(run_id: str, rows: list[dict], result: dict, paired: pd.DataFrame) -> dict:
    r59_card = campaign.PROTOCOL["cards"][R59_CARD]
    out = campaign.CAMPAIGN_ROOT / "batches" / run_id
    reuse_events = [row for row in rows if row.get("event") == "fit_reuse"]
    if len(paired) == 0 or paired.duplicated(["month", "track", "ml_num"]).any():
        raise RuntimeError("R59 saved predictions contain no rows or duplicate prediction keys")
    if set(paired.month.astype(str).unique()) != set(r58.MONTHS):
        raise RuntimeError("R59 saved predictions do not cover exactly the registered safe months")
    if set(paired.track.astype(str).unique()) != set(r58.TRACKS):
        raise RuntimeError("R59 saved predictions do not cover both registered tracks")
    if not np.isfinite(paired[["sold_price", "parent_prediction", "candidate_prediction"]].to_numpy(float)).all():
        raise RuntimeError("R59 saved predictions contain non-finite values")
    if (paired[["sold_price", "parent_prediction", "candidate_prediction"]].to_numpy(float) <= 0).any():
        raise RuntimeError("R59 saved predictions contain non-positive prices")

    input_by_track = {
        "off_market": r58._verified_input(campaign, r58.R52_RESULT, r58.R52_RUN, "off_market")[4],
        "list_aware": r58._verified_input(campaign, r58.R35_DEV_RESULT, r58.R35_DEV_RUN, "list_aware")[4],
    }
    recomputed = {}
    for track in r58.TRACKS:
        candidate = paired.loc[paired.track.eq(track)].copy()
        parent = input_by_track[track]
        expected = parent[["month", "track", "ml_num", "sold_price", "candidate_prediction"]].rename(
            columns={"candidate_prediction": "expected_parent_prediction"}
        )
        joined = candidate.merge(expected, on=["month", "track", "ml_num", "sold_price"], how="outer", indicator=True)
        if not joined._merge.eq("both").all():
            raise RuntimeError(f"R59 {track} saved row IDs or labels do not match the registered parent")
        if not np.array_equal(joined.parent_prediction.to_numpy(float), joined.expected_parent_prediction.to_numpy(float)):
            raise RuntimeError(f"R59 {track} parent predictions do not match the registered source")

        applied = candidate.loc[candidate.calibration_applied.astype(bool)]
        if len(applied) and (
            applied.max_abs_applied_log_correction_before_rounding.max() > r58.MAX_ABS_LOG_CORRECTION + 1e-12
        ):
            raise RuntimeError(f"R59 {track} exceeded its bound before dollar rounding")
        recomputed[track] = scoring.score_development_confirmation(
            candidate, months=r58.MONTHS,
            target_mdape_strictly_below=7.0 if track == "off_market" else 4.0,
            bootstrap_replicates=r58.BOOTSTRAP_REPLICATES,
            bootstrap_seed=r58.BOOTSTRAP_SEED,
            minimum_relative_gain_percent=0.5,
            minimum_months_improved=6,
            maximum_worst_month_regression_percent=5.0,
            maximum_mape_regression_percent=1.0,
        )
        stored = result["tracks"][track]
        checked_metrics = ("mdape_percent", "mape_percent", "p90_ape_percent", "ape_over_20_percent")
        for side in ("parent", "candidate"):
            for metric in checked_metrics:
                if not np.isclose(stored[side][metric], recomputed[track][side][metric], rtol=0, atol=1e-12):
                    raise RuntimeError(f"R59 saved {track} {side}.{metric} does not reproduce from its paired predictions")
        for key in ("relative_mdape_improvement_percent", "mape_relative_change_percent"):
            if not np.isclose(stored[key], recomputed[track][key], rtol=0, atol=1e-12):
                raise RuntimeError(f"R59 saved {track} {key} does not reproduce")
        if stored["qualifies_development"] != recomputed[track]["qualifies_development"]:
            raise RuntimeError(f"R59 saved {track} development gate does not reproduce")

    if len(reuse_events) != 1 or len(result.get("fit_details", [])) != r59.EXPECTED_NEW_FITS:
        raise RuntimeError("R59 saved result fit ledger does not match its preregistration")
    starts = {row["fit_id"] for row in rows if row.get("event") == "fit_start"}
    completes = {
        row["fit_id"] for row in rows
        if row.get("event") == "fit_complete" and row.get("status") == "complete"
    }
    saved_fit_ids = {row["fit_id"] for row in result["fit_details"]}
    if starts != completes or starts != saved_fit_ids:
        raise RuntimeError("R59 ledger fit IDs do not match the saved result's completed calibrators")
    if result["tracks"]["off_market"]["qualifies_development"] or result["tracks"]["list_aware"]["qualifies_development"]:
        raise RuntimeError("unexpectedly, R59 has a passing track; finalization requires separate review")

    return {
        "recomputed_tracks": recomputed,
        "row_counts": {track: int(paired.track.eq(track).sum()) for track in r58.TRACKS},
        "max_pre_rounding_correction": float(paired.max_abs_applied_log_correction_before_rounding.max()),
        "max_post_rounding_log_delta": float(paired.max_abs_rounded_log_delta.max()),
        "fit_start_count": len(starts),
        "reuse_event_count": len(reuse_events),
        "parent_run_ids": r59_card["parent_run_ids"],
    }


def run() -> dict:
    card = campaign.PROTOCOL["cards"].get(CARD)
    if not card or card.get("status") != "registered":
        raise RuntimeError("R60 must be preregistered before finalization")
    if card.get("source_sha256") != common._sha256(Path(__file__)):
        raise RuntimeError("R60 finalizer source changed after preregistration")
    r59_card = campaign.PROTOCOL["cards"].get(R59_CARD)
    if not r59_card or r59_card.get("status") != "registered":
        raise RuntimeError("R59 is not in the expected interrupted packaging state")
    if r59_card.get("source_sha256") != common._sha256(Path(r59.__file__)):
        raise RuntimeError("R59 source no longer matches the preregistered source hash")

    run_id, rows = _r59_run_events()
    out = campaign.CAMPAIGN_ROOT / "batches" / run_id
    result_path, paired_path = out / "development_results.json", out / "paired_predictions.csv"
    if (common._sha256(result_path) != card["saved_artifact_hashes"]["development_results_sha256"]
            or common._sha256(paired_path) != card["saved_artifact_hashes"]["paired_predictions_sha256"]):
        raise RuntimeError("R59 saved development artifact hash changed after R60 registration")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    paired = pd.read_csv(paired_path, dtype={"ml_num": str})
    verification = _verify_saved_result(run_id, rows, result, paired)

    # Reuse the source snapshots copied by R59 and keep the final source destination short for Windows.
    provenance = out / "provenance"
    original_sources = [*r58._source_files(campaign), Path(r59.__file__)]
    code_map = []
    for index, source in enumerate(original_sources):
        existing_snapshot = provenance / f"src_{index:02d}_{source.name}.source"
        if existing_snapshot.is_file():
            snapshot = existing_snapshot
        else:
            if index != len(original_sources) - 1:
                raise FileNotFoundError(f"R59 provenance snapshot is missing: {existing_snapshot.name}")
            snapshot = source
        snapshot_hash = common._sha256(snapshot)
        if source == Path(r59.__file__) and snapshot_hash != r59_card["source_sha256"]:
            raise RuntimeError("R59 source hash differs from the registered source")
        short_name = f"code{index:02d}.source"
        short_snapshot = provenance / short_name
        if snapshot != short_snapshot:
            shutil.copy2(snapshot, short_snapshot)
        code_map.append({
            "short_path": f"provenance/{short_name}",
            "source_path": str(source.relative_to(campaign.ROOT)).replace("\\", "/"),
            "sha256": snapshot_hash,
        })
    common._write_json(provenance / "source_map.json", code_map)

    report_lines = [
        "# R59 development result — finalized from saved artifacts", "",
        f"Run: `{run_id}`. R60 performed zero learner fits and did not alter predictions.",
        "The R59 scoring pass completed before Windows rejected the last long provenance-copy path. "
        "R60 reproduced the saved scores from the paired prediction file and verified source, data, row, and fit-ledger identities.",
        "The ±0.05 correction bound is verified before dollar rounding; the rounded log delta can exceed it by a few 1e-7.",
        "No reserve-month or hidden labels were accessed.", "",
        "| Track | Parent MdAPE | Candidate MdAPE | Relative gain | MAPE change | Months improved | Bootstrap upper95 (pp) | Strict target | Development gate |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for track, score in result["tracks"].items():
        report_lines.append(
            f"| {track} | {score['parent']['mdape_percent']:.5f}% | "
            f"{score['candidate']['mdape_percent']:.5f}% | "
            f"{score['relative_mdape_improvement_percent']:+.3f}% | "
            f"{score['mape_relative_change_percent']:+.3f}% | {score['months_improved']}/8 | "
            f"{score['bootstrap_mdape_difference_percent']['upper95']:+.5f} | "
            f"{'yes' if score['target_met'] else 'no'} | "
            f"{'pass' if score['qualifies_development'] else 'fail'} |"
        )
    report_lines.extend([
        "", f"Fit ledger: {verification['fit_start_count']} new fits; "
        f"{verification['reuse_event_count']} previously completed fold reused with zero new fits.",
        f"Rows: {verification['row_counts']}.",
        f"Maximum applied correction before rounding: {verification['max_pre_rounding_correction']:.12f}; "
        f"maximum rounded log delta: {verification['max_post_rounding_log_delta']:.12f}.",
    ])
    report_path = out / "research_report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    source_snapshot = provenance / "src_00_campaign_protocol.json.source"
    code_hashes = {row["source_path"]: row["sha256"] for row in code_map}
    manifest = {
        "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
        "finalization_card": CARD,
        "verification_card_source_sha256": card["source_sha256"],
        "protocol_snapshot_sha256_at_r59_run": common._sha256(source_snapshot),
        "code_sha256_from_r59_snapshots": code_hashes,
        "code_snapshot_map": code_map,
        "data_sha256": campaign.PROTOCOL["data_sha256"],
        "fit_starts": result["fit_starts"],
        "reused_calibration_folds": result["reused_calibration_folds"],
        "parent_run_ids": result["parent_run_ids"],
        "development_results_sha256": common._sha256(result_path),
        "paired_predictions_sha256": common._sha256(paired_path),
        "research_report_sha256": common._sha256(report_path),
        "verification": verification,
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    }
    manifest_path = out / "finalization_manifest.json"
    common._write_json(manifest_path, manifest)

    # Close the protocol's stale registered state while preserving the original append-only events.
    r58_card = campaign.PROTOCOL["cards"].get(r58.CARD)
    if r58_card and r58_card.get("status") == "registered":
        r58_card.update({
            "status": "failed", "run_id": r58_card.get("run_id", "bill_target74_r58_past_log_residual_calibration_20261025T085200Z"),
            "fit_starts": 1, "failure_reason": "post-rounding assertion checked rounded prices against the pre-rounding correction bound",
            "finalized_by": CARD,
        })
    r59_card.update({
        "status": "development_failed", "run_id": run_id,
        "result_path": str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        "paired_predictions_path": str(paired_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        "manifest_path": str(manifest_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        "report_path": str(report_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        "qualifies_by_track": {track: data["qualifies_development"] for track, data in result["tracks"].items()},
        "development_metrics": result["tracks"], "finalized_by": CARD,
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    card.update({
        "status": "finalization_complete", "run_id": run_id,
        "fit_starts": 0, "target_artifact_hashes_verified": card["saved_artifact_hashes"],
        "manifest_path": str(manifest_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        "report_path": str(report_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        "verification": verification,
    })
    _save_protocol()
    campaign._append({
        "event": "card_complete", "card": CARD, "run_id": run_id,
        "target_card": R59_CARD, "fit_starts": 0,
        "verified_fit_starts": result["fit_starts"],
        "qualifies_by_track": r59_card["qualifies_by_track"],
        "result_path": str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        "manifest_path": str(manifest_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    campaign._write_progress(
        "r59_development_failed",
        next_action="R59 failed both development gates; keep R22/R35 as validated parents and register a distinct hypothesis",
        r59_run_id=run_id,
        r59_result=str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        r59_fit_starts=result["fit_starts"],
        r59_reused_calibration_folds=result["reused_calibration_folds"],
        r60_finalization_manifest=str(manifest_path.relative_to(campaign.ROOT)).replace("\\", "/"),
    )
    return manifest


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Register or finalize R59 saved development artifacts")
    parser.add_argument("--register", action="store_true")
    args = parser.parse_args()
    print(json.dumps(register_card() if args.register else run(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
