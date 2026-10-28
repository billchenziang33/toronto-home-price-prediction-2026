from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.bill_target74 import runner as campaign


CARD = "R106_native_house_type"
RUN_ID = "r106m_20261027T222049Z"
RECOVERY_ID = RUN_ID + "_resume1"
OUTPUT = campaign.CAMPAIGN_ROOT / "batches" / RUN_ID
R94 = campaign.CAMPAIGN_ROOT / "batches" / "r94g_20261025T202220Z"
R151 = campaign.CAMPAIGN_ROOT / "batches" / "r151d_20261026T102852Z"
R97 = campaign.CAMPAIGN_ROOT / "batches" / "r97m_20261025T210147Z"
SAFE_MONTHS = ("2023-03", "2024-06")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jread(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def metric(actual: np.ndarray, prediction: np.ndarray) -> dict:
    ape = np.abs(prediction - actual) / actual * 100
    return {"n": int(len(actual)), "mdape_percent": float(np.median(ape)),
            "mape_percent": float(np.mean(ape))}


def gate(paired: pd.DataFrame, track: str, reference_column: str) -> dict:
    rows = paired.loc[paired.track.eq(track)]
    actual = rows.sold_price.to_numpy(float)
    reference = rows[reference_column].to_numpy(float)
    candidate = rows.candidate_prediction.to_numpy(float)
    before, after = metric(actual, reference), metric(actual, candidate)
    gain = 100 * (before["mdape_percent"] - after["mdape_percent"]) / before["mdape_percent"]
    mape_change = 100 * (after["mape_percent"] - before["mape_percent"]) / before["mape_percent"]
    monthly = {}
    for month in SAFE_MONTHS:
        part = rows.loc[rows.month.astype(str).eq(month)]
        b = metric(part.sold_price.to_numpy(float), part[reference_column].to_numpy(float))
        c = metric(part.sold_price.to_numpy(float), part.candidate_prediction.to_numpy(float))
        monthly[month] = {"benchmark": b, "candidate": c,
                          "relative_mdape_improvement_percent":
                              100 * (b["mdape_percent"] - c["mdape_percent"]) / b["mdape_percent"]}
    improved = sum(x["relative_mdape_improvement_percent"] > 0 for x in monthly.values())
    qualifies = gain >= .25 and improved == 2 and mape_change <= 1.0
    return {"n": len(rows), "benchmark": before, "candidate": after,
            "relative_mdape_improvement_percent": gain,
            "mape_relative_change_percent": mape_change,
            "months_improved": improved, "monthly": monthly,
            "qualifies_screen": qualifies}


def compare(actual: dict, expected: dict, path: str) -> None:
    for key in ("n", "mdape_percent", "mape_percent"):
        if key == "n":
            if actual[key] != expected[key]:
                raise RuntimeError(f"metric row count mismatch at {path}.{key}")
        elif not np.isclose(actual[key], expected[key], rtol=0, atol=1e-10):
            raise RuntimeError(f"metric mismatch at {path}.{key}: {actual[key]} vs {expected[key]}")


def audit() -> dict:
    results_path = OUTPUT / "results.json"
    paired_path = OUTPUT / "paired_predictions.csv"
    manifest_path = OUTPUT / "run_manifest.json"
    report_path = OUTPUT / "research_report.md"
    results, manifest = jread(results_path), jread(manifest_path)
    protocol = jread(campaign.PROTOCOL_PATH)
    card = protocol["cards"][CARD]
    paired = pd.read_csv(paired_path, dtype={"ml_num": str})
    expected_hashes = {"paired_predictions_sha256": sha(paired_path),
                       "results_sha256": sha(results_path)}
    for field, value in expected_hashes.items():
        if manifest.get(field) != value:
            raise RuntimeError(f"R106 manifest hash mismatch: {field}")
    if (results.get("data_sha256") != manifest.get("source_identity", {}).get("data_sha256")
            or results.get("data_sha256") != card.get("data_sha256")
            or results.get("reserved_months_labels_opened") is not False
            or results.get("hidden_labels_opened") is not False
            or manifest.get("reserved_months_labels_opened") is not False
            or manifest.get("hidden_labels_opened") is not False):
        raise RuntimeError("R106 data identity/reserve/hidden-label flags are invalid")
    if (set(paired.month.astype(str)) != set(SAFE_MONTHS)
            or set(paired.track) != {"off_market", "list_aware"}
            or paired.duplicated(["month", "track", "ml_num"]).any()
            or len(paired) != 9324):
        raise RuntimeError("R106 paired rows do not match the registered safe screen scope")
    for col in ("sold_price", "benchmark_prediction", "current_best_prediction", "candidate_prediction"):
        values = paired[col].to_numpy(float)
        if not np.isfinite(values).all() or (values <= 0).any():
            raise RuntimeError(f"invalid prices in R106 paired predictions: {col}")

    source_specs = {
        "off_market": (R94 / "paired_predictions.csv", "candidate_prediction",
                       R151 / "paired_predictions.csv", "candidate_prediction"),
        "list_aware": (R94 / "paired_predictions.csv", "base_prediction",
                       R97 / "paired_predictions.csv", "candidate_prediction"),
    }
    row_checks = []
    for track, (hist_path, hist_column, current_path, current_column) in source_specs.items():
        for month in SAFE_MONTHS:
            subset = paired.loc[paired.track.eq(track) & paired.month.astype(str).eq(month)].copy()
            historical = pd.read_csv(hist_path, dtype={"ml_num": str})
            historical = historical.loc[historical.track.eq(track)
                                        & historical.month.astype(str).eq(month),
                                        ["ml_num", "sold_price", hist_column]]
            current = pd.read_csv(current_path, dtype={"ml_num": str})
            current = current.loc[current.track.eq(track) & current.month.astype(str).eq(month),
                                  ["ml_num", "sold_price", current_column]]
            if (historical.ml_num.duplicated().any() or current.ml_num.duplicated().any()
                    or set(historical.ml_num) != set(subset.ml_num)
                    or set(current.ml_num) != set(subset.ml_num)):
                raise RuntimeError(f"row-key alignment failure: {track}/{month}")
            historical = historical.set_index("ml_num").reindex(subset.ml_num)
            current = current.set_index("ml_num").reindex(subset.ml_num)
            if (not np.array_equal(historical.sold_price.to_numpy(float), subset.sold_price.to_numpy(float))
                    or not np.array_equal(current.sold_price.to_numpy(float), subset.sold_price.to_numpy(float))
                    or not np.allclose(historical[hist_column].to_numpy(float),
                                       subset.benchmark_prediction.to_numpy(float), rtol=0, atol=1e-10)
                    or not np.allclose(current[current_column].to_numpy(float),
                                       subset.current_best_prediction.to_numpy(float), rtol=0, atol=1e-10)):
                raise RuntimeError(f"prediction or label mismatch: {track}/{month}")
            row_checks.append({"track": track, "month": month, "rows": len(subset),
                               "historical_comparator_source": str(hist_path.relative_to(campaign.ROOT)).replace("\\", "/"),
                               "current_best_source": str(current_path.relative_to(campaign.ROOT)).replace("\\", "/")})

    historical_scores, current_scores = {}, {}
    for track in ("off_market", "list_aware"):
        historical_scores[track] = gate(paired, track, "benchmark_prediction")
        current_scores[track] = gate(paired, track, "current_best_prediction")
    for label, recomputed, expected in (
        ("historical", historical_scores, results["historical_registered_baseline_scores"]),
        ("current_best", current_scores, results["current_best_comparator_scores"])):
        for track in recomputed:
            actual, saved = recomputed[track], expected[track]
            for key in ("n", "relative_mdape_improvement_percent", "mape_relative_change_percent",
                        "months_improved", "qualifies_screen"):
                if isinstance(actual[key], float):
                    if not np.isclose(actual[key], saved[key], rtol=0, atol=1e-10):
                        raise RuntimeError(f"score mismatch: {label}/{track}/{key}")
                elif actual[key] != saved[key]:
                    raise RuntimeError(f"score mismatch: {label}/{track}/{key}")
            compare(actual["benchmark"], saved["benchmark"], f"{label}/{track}/benchmark")
            compare(actual["candidate"], saved["candidate"], f"{label}/{track}/candidate")
            for month in SAFE_MONTHS:
                compare(actual["monthly"][month]["benchmark"], saved["monthly"][month]["benchmark"],
                        f"{label}/{track}/{month}/benchmark")
                compare(actual["monthly"][month]["candidate"], saved["monthly"][month]["candidate"],
                        f"{label}/{track}/{month}/candidate")

    qualified = [track for track in ("off_market", "list_aware")
                 if historical_scores[track]["qualifies_screen"]
                 and current_scores[track]["qualifies_screen"]]
    if qualified != results.get("screen_passed_tracks") or qualified:
        raise RuntimeError("screen pass list is inconsistent; expected no tracks to pass both comparators")
    events = [json.loads(line) for line in campaign.LEDGER.read_text(encoding="utf-8").splitlines()
              if line.strip()]
    root_events = [e for e in events if e.get("run_id") == RUN_ID]
    retry_events = [e for e in events if e.get("run_id") == RECOVERY_ID]
    starts = [e for e in root_events if e.get("event") == "fit_start"]
    completes = [e for e in root_events if e.get("event") == "fit_complete" and e.get("status") == "complete"]
    abandoned = [e for e in events if e.get("event") == "fit_abandoned"
                 and e.get("run_id") == RUN_ID]
    retry_starts = [e for e in retry_events if e.get("event") == "fit_start"]
    retry_completes = [e for e in retry_events if e.get("event") == "fit_complete"
                       and e.get("status") == "complete"]
    reused = [e for e in events if e.get("event") == "fit_reused" and e.get("run_id") == RUN_ID]
    if (len(starts) != 6 or len(completes) != 5 or len(abandoned) != 1
            or len(retry_starts) != 1 or len(retry_completes) != 1 or len(reused) != 5
            or len({e.get("original_fit_id") for e in reused}) != 5):
        raise RuntimeError("R106 recovery fit ledger reconciliation failed")
    ledger_starts = [e for e in root_events + retry_events if e.get("event") == "fit_start"]
    ledger_completes = [e for e in root_events + retry_events if e.get("event") == "fit_complete"]
    if (any(e.get("reserved_months_labels_opened") is not False
            or e.get("hidden_labels_opened") is not False for e in ledger_starts)
            or any(e.get("fit_id") not in {s.get("fit_id") for s in ledger_starts}
                   for e in ledger_completes)
            or any(e.get("reserved_months_labels_opened") is True
                   or e.get("hidden_labels_opened") is True for e in root_events + retry_events)):
        raise RuntimeError("R106 ledger has a reserve/hidden-label policy violation")
    for event in completes + retry_completes:
        path = OUTPUT / "component_predictions" / event["track"] / event["month"] / f"{event['component']}.csv"
        if not path.is_file() or sha(path) != event.get("prediction_file_sha256"):
            raise RuntimeError("R106 component prediction does not match its completion event")

    source_hashes = manifest.get("source_sha256", {})
    source_snapshot_paths = {}
    snapshot_dir = OUTPUT / "provenance_recovery"
    for rel, expected in source_hashes.items():
        path = campaign.ROOT / Path(rel)
        original_snapshot = snapshot_dir / (path.name + ".snapshot")
        if path.name == "campaign_protocol.json" and original_snapshot.is_file() \
                and sha(original_snapshot) == expected:
            source_to_copy = original_snapshot
        elif path.is_file() and sha(path) == expected:
            source_to_copy = path
        elif original_snapshot.is_file() and sha(original_snapshot) == expected:
            source_to_copy = original_snapshot
        else:
            raise RuntimeError(f"neither a source file nor its recorded snapshot matches: {rel}")
        unique_name = hashlib.sha256(rel.encode("utf-8")).hexdigest()[:12] + "_" + path.name + ".snapshot"
        unique_snapshot = snapshot_dir / unique_name
        if not unique_snapshot.is_file() or sha(unique_snapshot) != expected:
            shutil.copy2(source_to_copy, unique_snapshot)
        if sha(unique_snapshot) != expected:
            raise RuntimeError(f"source snapshot hash mismatch: {rel}")
        source_snapshot_paths[rel] = str(unique_snapshot.relative_to(OUTPUT)).replace("\\", "/")
    if source_hashes.get("experiments/bill_target74/r106_native_house_type.py") != card.get("source_sha256"):
        raise RuntimeError("protocol registration does not match the fitted R106 implementation")
    manifest["source_snapshot_paths"] = source_snapshot_paths
    manifest["snapshot_collision_repaired_from_hash_verified_sources"] = True
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not report_path.is_file():
        raise RuntimeError("R106 research report is missing")

    audit = {"status": "passed", "run_id": RUN_ID, "recovery_run_id": RECOVERY_ID,
             "independent_prediction_and_metric_reconstruction": True,
             "paired_rows": len(paired), "row_alignment_checks": row_checks,
             "historical_registered_baseline_scores": historical_scores,
             "current_best_comparator_scores": current_scores,
             "qualified_tracks": qualified,
             "fit_ledger": {"original_registered_attempts": len(starts),
                            "completed_original_attempts": len(completes),
                            "abandoned_attempts": len(abandoned),
                            "recovery_attempts": len(retry_starts),
                            "completed_recovery_attempts": len(retry_completes),
                            "reused_completed_artifacts": len(reused)},
             "reserved_months_labels_opened": False, "hidden_labels_opened": False,
             "verified_hashes": expected_hashes}
    audit_path = OUTPUT / "independent_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    card["status"] = "screen_failed"
    card["audit_path"] = str(audit_path.relative_to(campaign.ROOT)).replace("\\", "/")
    campaign.PROTOCOL_PATH.write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + "\n",
                                      encoding="utf-8")
    campaign._append({"event": "card_independent_audit_complete", "card": CARD,
                      "run_id": RUN_ID, "audit_path": str(audit_path.relative_to(campaign.ROOT)).replace("\\", "/"),
                      "status": "passed", "qualified_tracks": qualified,
                      "reserved_months_labels_opened": False, "hidden_labels_opened": False})
    campaign._write_progress("r106_screen_audit_passed",
                             next_action="retain current best candidates; R106 native house-type screen failed against registered and current-best gates",
                             r106_audit=str(audit_path.relative_to(campaign.ROOT)).replace("\\", "/"),
                             r106_qualified_tracks=qualified)
    return audit


if __name__ == "__main__":
    print(json.dumps(audit(), ensure_ascii=False), flush=True)
