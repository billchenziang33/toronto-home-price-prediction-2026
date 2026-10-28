from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from data import hide_answers
from experiments.bill_target74 import r03_catboost as r03
from experiments.bill_target74 import r04_ablation as provenance
from experiments.bill_target74 import r106_native_house_type as r106
from experiments.bill_target74 import runner as campaign
from experiments.bill_greedy import runner as greedy


CARD = r106.CARD
RUN_ID = "r106m_20261027T222049Z"
R151_DIR = campaign.CAMPAIGN_ROOT / "batches" / "r151d_20261026T102852Z"
R97_DIR = campaign.CAMPAIGN_ROOT / "batches" / "r97m_20261025T210147Z"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def append(event: dict) -> None:
    if event.get("event") == "fit_reused":
        seen = [json.loads(line) for line in campaign.LEDGER.read_text(encoding="utf-8").splitlines()
                if line.strip()]
        if any(e.get("event") == "fit_reused" and e.get("run_id") == event.get("run_id")
               and e.get("original_fit_id") == event.get("original_fit_id") for e in seen):
            return
    campaign._append({**event, "reserved_months_labels_opened": False,
                      "hidden_labels_opened": False})


def artifact_predictions(directory: Path, track: str, month: str,
                         prediction_column: str) -> pd.DataFrame:
    path = directory / "paired_predictions.csv"
    frame = pd.read_csv(path, dtype={"ml_num": str})
    frame = frame.loc[frame.track.eq(track) & frame.month.astype(str).eq(month),
                      ["ml_num", "sold_price", prediction_column]].copy()
    if frame.empty or frame.ml_num.duplicated().any():
        raise RuntimeError(f"invalid comparator rows: {directory.name}/{track}/{month}")
    frame[prediction_column] = pd.to_numeric(frame[prediction_column], errors="raise")
    if not np.isfinite(frame[["sold_price", prediction_column]].to_numpy(float)).all():
        raise RuntimeError("non-finite comparator predictions")
    return frame


def component(path: Path, homes: pd.DataFrame, track: str, month: str,
              component_name: str, run_events: list[dict]) -> np.ndarray:
    frame = pd.read_csv(path, dtype={"ml_num": str})
    if frame.ml_num.duplicated().any() or set(frame.ml_num) != set(homes.ml_num.astype(str)):
        raise RuntimeError(f"component IDs mismatch: {track}/{month}/{component_name}")
    events = [e for e in run_events if e.get("event") == "fit_complete"
              and e.get("status") == "complete" and e.get("track") == track
              and e.get("month") == month and e.get("component") == component_name]
    if len(events) != 1 or sha(path) != events[0].get("prediction_file_sha256"):
        raise RuntimeError(f"component ledger/hash mismatch: {track}/{month}/{component_name}")
    values = frame.set_index("ml_num").loc[homes.ml_num.astype(str), "predicted_price"].to_numpy(float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise RuntimeError("component predictions must be finite positive prices")
    if events[0].get("run_id") == RUN_ID:
        append({"event": "fit_reused", "card": CARD, "run_id": RUN_ID,
                "original_fit_id": events[0].get("fit_id"), "track": track,
                "month": month, "component": component_name,
                "prediction_file_sha256": sha(path), "training_was_reused": True})
    return values


def paired_bootstrap_gate(paired: pd.DataFrame, track: str, months: tuple[str, ...],
                          comparator: str) -> dict:
    frame = paired.loc[paired.track.eq(track)].copy()
    frame["benchmark_prediction"] = frame[comparator]
    return r106._score(frame, track, months, "screen")


def verify_comparators() -> tuple[dict, dict, dict]:
    source = provenance._source_identity()
    r151_result = read_json(R151_DIR / "development_results.json")
    r151_manifest = read_json(R151_DIR / "run_manifest.json")
    r151_paired = R151_DIR / "paired_predictions.csv"
    if (r151_manifest.get("data_sha256") != source["data_sha256"]
            or r151_manifest.get("paired_predictions_sha256") != sha(r151_paired)
            or r151_manifest.get("result_sha256") != sha(R151_DIR / "development_results.json")
            or r151_manifest.get("reserved_months_labels_opened") is not False
            or r151_manifest.get("hidden_labels_opened") is not False):
        raise RuntimeError("R151 comparator provenance/integrity check failed")

    r97_result = read_json(R97_DIR / "development_results.json")
    r97_manifest = read_json(R97_DIR / "run_manifest.json")
    r97_paired = R97_DIR / "paired_predictions.csv"
    if (r97_result.get("data_sha256") != source["data_sha256"]
            or r97_manifest.get("source_identity", {}).get("data_sha256") != source["data_sha256"]
            or r97_manifest.get("paired_predictions_sha256") != sha(r97_paired)
            or r97_manifest.get("development_results_sha256") != sha(R97_DIR / "development_results.json")
            or r97_manifest.get("reserved_months_labels_opened") is not False
            or r97_manifest.get("hidden_labels_opened") is not False):
        raise RuntimeError("R97 comparator provenance/integrity check failed")
    return source, {"result": r151_result, "manifest": r151_manifest,
                    "paired_path": r151_paired}, {"result": r97_result,
                                                 "manifest": r97_manifest,
                                                 "paired_path": r97_paired}


def register_comparator_amendment(r151: dict, r97: dict) -> None:
    card = campaign.PROTOCOL["cards"][CARD]
    entry = {"reason": "supplement the historical registered screen baselines with the current best development candidates before evaluating the recovered R106 predictions",
             "historical_registered_baselines_retained": card["benchmark_by_track"],
             "current_best_comparators": {"off_market": "R151 candidate_prediction",
                                          "list_aware": "R97 candidate_prediction"},
             "r151_paired_predictions_sha256": r151["manifest"]["paired_predictions_sha256"],
             "r97_paired_predictions_sha256": r97["manifest"]["paired_predictions_sha256"],
             "screen_qualification_rule": "A track may proceed only if it passes the unchanged registered screen gate against both its historical baseline and the current-best development candidate.",
             "reserved_months_labels_opened": False, "hidden_labels_opened": False}
    if not any(x.get("reason") == entry["reason"] for x in card.get("registration_amendments", [])):
        card.setdefault("registration_amendments", []).append(entry)
        append({"event": "card_comparator_amended", "card": CARD, **entry})
    recovery_entry = {
        "reason": "resume the second interrupted R106 orchestration without repeating verified completed fits",
        "recovery_run_id": RUN_ID,
        "original_recovery_policy": "rerun all six registered screen fits under a new run_id",
        "amended_recovery_policy": "preserve five completed component predictions after ledger, hash and heldout-ID verification; mark the sole incomplete fit attempt abandoned; retry only that fit with a unique recovery fit ID; score the full registered six-task screen after reconstruction",
        "evidence": "five successful fit_complete events with matching prediction hashes, one unmatched fit_start with no prediction file, and no live Python learner process or execution session",
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    }
    if not any(x.get("reason") == recovery_entry["reason"]
               and x.get("recovery_run_id") == RUN_ID
               for x in card.get("registration_amendments", [])):
        card.setdefault("registration_amendments", []).append(recovery_entry)
        campaign.PROTOCOL_PATH.write_text(
            json.dumps(campaign.PROTOCOL, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        append({"event": "card_recovery_policy_amended", "card": CARD, **recovery_entry})
    elif campaign.PROTOCOL_PATH.is_file():
        campaign.PROTOCOL_PATH.write_text(
            json.dumps(campaign.PROTOCOL, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run() -> dict:
    card = campaign.PROTOCOL["cards"].get(CARD)
    source = provenance._source_identity()
    if (card is None or card.get("status") != "registered"
            or card.get("source_sha256") != sha(Path(r106.__file__))
            or card.get("data_sha256") != source["data_sha256"]):
        raise RuntimeError("R106 registration/source/data identity is not the expected interrupted screen")

    r151, r97 = None, None
    source, r151, r97 = verify_comparators()
    register_comparator_amendment(r151, r97)

    output = campaign.CAMPAIGN_ROOT / "batches" / RUN_ID
    if not output.is_dir() or (output / "results.json").exists():
        raise RuntimeError("the interrupted R106 output is missing or already finalized")
    ledger_events = [json.loads(line) for line in campaign.LEDGER.read_text(encoding="utf-8").splitlines()
                     if line.strip()]
    run_events = [e for e in ledger_events if e.get("run_id") == RUN_ID]
    starts = [e for e in run_events if e.get("event") == "fit_start"]
    completed = [e for e in run_events if e.get("event") == "fit_complete"
                 and e.get("status") == "complete"]
    incomplete = [e for e in starts if not any(c.get("fit_id") == e.get("fit_id") for c in completed)]
    expected_keys = {(track, month, component_name)
                     for month in r106.SCREEN_MONTHS
                     for track, component_names in (("off_market", ("base", "detached_expert")),
                                                     ("list_aware", ("core",)))
                     for component_name in component_names}
    completed_keys = {(e.get("track"), e.get("month"), e.get("component")) for e in completed}
    if (len(starts) != 6 or len(completed) != 5 or len(incomplete) != 1
            or completed_keys | {(e.get("track"), e.get("month"), e.get("component")) for e in incomplete}
            != expected_keys
            or not any(e.get("event") == "screen_fold_complete" and e.get("month") == "2023-03"
                       and e.get("track") == t for e in run_events for t in ("off_market", "list_aware"))
            or any(e.get("event") in {"card_complete", "card_failed"} for e in run_events)):
        raise RuntimeError("R106 interrupted-run ledger is not the expected five-completed/one-incomplete state")

    # Verify all five saved component files before resuming the one missing fit.
    for event in completed:
        path = output / "component_predictions" / event["track"] / event["month"] / f"{event['component']}.csv"
        if not path.is_file() or sha(path) != event.get("prediction_file_sha256"):
            raise RuntimeError(f"saved R106 component failed ledger/hash verification: {path}")

    old_hgbr = greedy._frozen_hgbr_predictions
    greedy._frozen_hgbr_predictions = lambda month, track, train, homes: r03._verified_hgbr_cache(
        greedy, month, track, train, homes)
    try:
        missing = incomplete[0]
        recovery_run_id = RUN_ID + "_resume1"
        all_events = [json.loads(line) for line in campaign.LEDGER.read_text(encoding="utf-8").splitlines()
                      if line.strip()]
        if not any(e.get("event") == "fit_abandoned" and e.get("fit_id") == missing.get("fit_id")
                   for e in all_events):
            append({"event": "fit_abandoned", "card": CARD, "run_id": RUN_ID,
                    "fit_id": missing.get("fit_id"), "track": missing.get("track"),
                    "month": missing.get("month"), "component": missing.get("component"),
                    "reason": "the prior execution session and Python learner process are no longer present; no completion event or prediction file exists"})
        if not any(e.get("event") == "card_recovery_start" and e.get("recovery_run_id") == recovery_run_id
                   for e in all_events):
            append({"event": "card_recovery_start", "card": CARD, "run_id": RUN_ID,
                    "recovery_run_id": recovery_run_id,
                    "reused_completed_components": len(completed),
                    "abandoned_incomplete_attempts": len(incomplete),
                    "retry_fit_tasks": 1})
        _, train, _, truth, heldout = greedy._load_fold("2024-06")
        homes = hide_answers(heldout, "list_aware")
        dates = pd.to_datetime(train.date_sold, errors="raise")
        cutoff = pd.Period("2024-06", freq="M").start_time
        if dates.ge(cutoff).any() or dates.dt.to_period("M").astype(str).isin(
                campaign.PROTOCOL["reserved_months"]).any():
            raise RuntimeError("strict-past training or reserve-month policy violated")
        details: list[dict] = []
        recovery_events = [e for e in all_events if e.get("run_id") == recovery_run_id
                           and e.get("event") == "fit_complete" and e.get("status") == "complete"
                           and e.get("track") == "list_aware" and e.get("month") == "2024-06"
                           and e.get("component") == "core"]
        recovery_path = output / "component_predictions" / "list_aware" / "2024-06" / "core.csv"
        if recovery_events:
            if (len(recovery_events) != 1 or not recovery_path.is_file()
                    or sha(recovery_path) != recovery_events[0].get("prediction_file_sha256")):
                raise RuntimeError("previous recovery fit does not match its ledger/hash; refusing duplicate fit")
        else:
            core = r106._fit("2024-06", "list_aware", train, train, homes, "core",
                             recovery_run_id, "screen", output, source, details)
    finally:
        greedy._frozen_hgbr_predictions = old_hgbr

    folds = []
    all_events = [json.loads(line) for line in campaign.LEDGER.read_text(encoding="utf-8").splitlines()
                  if line.strip()]
    recovery_run_id = RUN_ID + "_resume1"
    all_events = [e for e in all_events if e.get("run_id") in {RUN_ID, recovery_run_id}]
    current_best_columns = {"off_market": (r151["paired_path"], "candidate_prediction"),
                            "list_aware": (r97["paired_path"], "candidate_prediction")}
    original_benchmark = r106._benchmark(r106.SCREEN_MONTHS)
    for month in r106.SCREEN_MONTHS:
        _, train, _, truth, heldout = greedy._load_fold(month)
        cutoff = pd.Period(month, freq="M").start_time
        dates = pd.to_datetime(train.date_sold, errors="raise")
        if dates.ge(cutoff).any() or dates.dt.to_period("M").astype(str).isin(
                campaign.PROTOCOL["reserved_months"]).any():
            raise RuntimeError(f"strict-past training or reserve-month policy violated for {month}")
        ids = heldout.ml_num.astype(str).to_numpy()
        for track in ("off_market", "list_aware"):
            homes = hide_answers(heldout, track)
            if track == "off_market":
                base = component(output / "component_predictions" / track / month / "base.csv",
                                 homes, track, month, "base", run_events)
                route_eval = greedy._specialist_mask(homes, "Detached")
                expert_homes = homes.loc[route_eval].copy()
                expert = component(output / "component_predictions" / track / month / "detached_expert.csv",
                                   expert_homes, track, month, "detached_expert", all_events)
                candidate = greedy._blend_specialist_predictions(
                    np.round(base), np.round(expert), homes, "Detached", .75)
            else:
                path = output / "component_predictions" / track / month / "core.csv"
                core_values = component(path, homes, track, month, "core", all_events)
                frozen = r03._verified_hgbr_cache(greedy, month, track, train, homes)
                from experiments.model_optimization_v2.pipeline import blend_prices
                candidate = blend_prices(core_values, frozen, challenger_weight=.75)

            historical = original_benchmark.loc[
                original_benchmark.month.eq(month) & original_benchmark.track.eq(track),
                ["ml_num", "sold_price", "benchmark_prediction"]].copy()
            historical["ml_num"] = historical.ml_num.astype(str)
            historical = historical.set_index("ml_num").reindex(ids)
            if historical.isna().any().any() or not np.array_equal(
                    historical.sold_price.to_numpy(float), truth.sold_price.to_numpy(float)):
                raise RuntimeError(f"historical benchmark/heldout labels do not align for {track}/{month}")

            best_path, best_column = current_best_columns[track]
            best = artifact_predictions(best_path.parent, track, month, best_column)
            if set(best.ml_num) != set(ids):
                raise RuntimeError(f"current-best row IDs do not match {track}/{month}")
            best = best.set_index("ml_num").reindex(ids)
            if best.isna().any().any() or not np.array_equal(
                    best.sold_price.to_numpy(float), truth.sold_price.to_numpy(float)):
                raise RuntimeError(f"current-best comparator/heldout labels do not align for {track}/{month}")
            folds.append(pd.DataFrame({
                "month": month, "track": track, "ml_num": ids,
                "sold_price": truth.sold_price.to_numpy(float),
                "benchmark_prediction": historical.benchmark_prediction.to_numpy(float),
                "current_best_prediction": best[best_column].to_numpy(float),
                "candidate_prediction": np.asarray(candidate, dtype=float),
            }))

    paired = pd.concat(folds, ignore_index=True)
    if paired.duplicated(["month", "track", "ml_num"]).any():
        raise RuntimeError("duplicate R106 paired-prediction keys")
    paired_path = output / "paired_predictions.csv"
    paired.to_csv(paired_path, index=False)
    historical_scores = {t: r106._score(paired, t, r106.SCREEN_MONTHS, "screen")
                         for t in ("off_market", "list_aware")}
    current_scores = {t: paired_bootstrap_gate(paired, t, r106.SCREEN_MONTHS,
                                               "current_best_prediction")
                      for t in ("off_market", "list_aware")}
    passed = [t for t in historical_scores
              if historical_scores[t]["qualifies_screen"] and current_scores[t]["qualifies_screen"]]

    all_events = [json.loads(line) for line in campaign.LEDGER.read_text(encoding="utf-8").splitlines()
                  if line.strip()]
    recovery_completions = [e for e in all_events if e.get("run_id") == recovery_run_id
                            and e.get("event") == "fit_complete" and e.get("status") == "complete"
                            and e.get("track") == "list_aware" and e.get("month") == "2024-06"
                            and e.get("component") == "core"]
    if len(recovery_completions) != 1:
        raise RuntimeError("the resumed R106 fit must have exactly one complete ledger record")
    retry_detail = recovery_completions[0]
    details = [{"fit_id": retry_detail.get("fit_id"), "track": "list_aware",
                "month": "2024-06", "component": "core",
                "fit_seconds": retry_detail.get("seconds"),
                "training_rows": retry_detail.get("training_rows"),
                "prediction_file_sha256": retry_detail.get("prediction_file_sha256"),
                "recovered_fit": True}]
    details.extend({"fit_id": e.get("fit_id"), "track": e.get("track"),
                    "month": e.get("month"), "component": e.get("component"),
                    "fit_seconds": e.get("seconds"), "prediction_file_sha256": e.get("prediction_file_sha256"),
                    "reused_from_interrupted_run": True}
                   for e in completed)
    result = {"campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": RUN_ID,
              "recovery_run_id": recovery_run_id, "card": CARD, "stage": "screen",
              "months": list(r106.SCREEN_MONTHS), "tracks": ["off_market", "list_aware"],
              "registered_fit_tasks": 6, "reused_completed_fit_outputs": 5,
              "abandoned_fit_attempts": 1, "recovery_fit_starts": 1,
              "historical_registered_baseline_scores": historical_scores,
              "current_best_comparator_scores": current_scores,
              "screen_passed_tracks": passed, "fit_details": details,
              "reserved_months_labels_opened": False, "hidden_labels_opened": False,
              "data_sha256": source["data_sha256"]}
    results_path = output / "results.json"
    results_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    sources = [Path(r106.__file__), Path(__file__), campaign.PROTOCOL_PATH,
               campaign.PACKAGE / "r03_catboost.py", campaign.PACKAGE / "r04_ablation.py",
               campaign.PACKAGE / "runner.py", campaign.ROOT / "experiments/bill_greedy/runner.py",
               campaign.ROOT / "experiments/bill_greedy/history.py",
               campaign.ROOT / "model_candidate_v2/features.py",
               R151_DIR / "development_results.json", R151_DIR / "paired_predictions.csv",
               R151_DIR / "run_manifest.json", R97_DIR / "development_results.json",
               R97_DIR / "paired_predictions.csv", R97_DIR / "run_manifest.json"]
    provenance_dir = output / "provenance_recovery"
    provenance_dir.mkdir(exist_ok=True)
    source_hashes = {}
    for path in sources:
        key = str(path.relative_to(campaign.ROOT)).replace("\\", "/")
        source_hashes[key] = sha(path)
        shutil.copy2(path, provenance_dir / (path.name + ".snapshot"))
    manifest = {"campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": RUN_ID,
                "recovery_run_id": recovery_run_id, "source_identity": source,
                "source_sha256": source_hashes, "registered_fit_tasks": 6,
                "reused_completed_fit_outputs": 5, "abandoned_fit_attempts": 1,
                "recovery_fit_starts": 1, "paired_predictions_sha256": sha(paired_path),
                "results_sha256": sha(results_path),
                "benchmark_sha256": sha(r106.R94_DIR / "paired_predictions.csv"),
                "current_best_comparators": {
                    "off_market": {"run_id": r151["result"]["run_id"],
                                   "paired_predictions_sha256": r151["manifest"]["paired_predictions_sha256"]},
                    "list_aware": {"run_id": r97["result"]["run_id"],
                                   "paired_predictions_sha256": r97["manifest"]["paired_predictions_sha256"]}},
                "historical_successful_fit_ids": [e.get("fit_id") for e in completed],
                "abandoned_fit_id": incomplete[0].get("fit_id"),
                "recovery_fit_id": retry_detail.get("fit_id"),
                "reserved_months_labels_opened": False, "hidden_labels_opened": False}
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report = ["# R106 M06 Native House-Type Screen — Recovered Run", "",
              "This result resumes the interrupted six-task screen. Five completed component predictions were reused only after ledger/hash/ID verification. One incomplete list-aware component fit was marked abandoned and retried under a distinct recovery fit ID.", "",
              "All promotion gates below must pass against both the original registered baseline and the current best development candidate. No reserved or hidden labels were accessed.", ""]
    for label, scores in (("Historical registered baseline", historical_scores),
                          ("Current best development candidate", current_scores)):
        report.extend([f"## {label}", ""])
        for track, score in scores.items():
            report.extend([f"### {track}", "",
                           f"Comparator pooled MdAPE: {score['benchmark']['mdape_percent']:.6f}%",
                           f"R106 pooled MdAPE: {score['candidate']['mdape_percent']:.6f}%",
                           f"Relative MdAPE gain: {score['relative_mdape_improvement_percent']:+.4f}%",
                           f"MAPE relative change: {score['mape_relative_change_percent']:+.4f}%",
                           f"Improved screen months: {score['months_improved']}/2",
                           f"Gate: {score['qualifies_screen']}", ""])
            for month, details_row in score["monthly"].items():
                report.append(f"- {month}: {details_row['benchmark']['mdape_percent']:.5f}% → {details_row['candidate']['mdape_percent']:.5f}%")
            report.append("")
    report.append("No reserve or hidden labels were opened.")
    (output / "research_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    card = campaign.PROTOCOL["cards"][CARD]
    card["screen_passed_tracks"] = passed
    card["last_run_id"] = RUN_ID
    card["last_result_path"] = str(results_path.relative_to(campaign.ROOT)).replace("\\", "/")
    card["status"] = "screen_recovered_pending_audit"
    campaign.PROTOCOL_PATH.write_text(json.dumps(campaign.PROTOCOL, indent=2, ensure_ascii=False) + "\n",
                                      encoding="utf-8")
    append({"event": "card_recovered_complete", "card": CARD, "run_id": RUN_ID,
            "recovery_run_id": recovery_run_id, "registered_fit_tasks": 6,
            "reused_completed_fit_outputs": 5, "abandoned_fit_attempts": 1,
            "recovery_fit_starts": 1, "screen_passed_tracks": passed,
            "historical_scores": historical_scores, "current_best_scores": current_scores,
            "result_path": str(results_path.relative_to(campaign.ROOT)).replace("\\", "/")})
    next_action = "independently audit R106 screen before any development run"
    campaign._write_progress("r106_screen_recovered_pending_audit", next_action=next_action,
                             r106_run_id=RUN_ID,
                             r106_result=str(results_path.relative_to(campaign.ROOT)).replace("\\", "/"),
                             r106_reused_components=5, r106_abandoned_attempts=1,
                             r106_recovery_fit_starts=1, r106_screen_passed_tracks=passed)
    return result


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False), flush=True)
