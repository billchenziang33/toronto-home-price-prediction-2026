"""Post-run R166 audit wrapper with corrected run-relative snapshot resolution."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "experiments/bill_target74/campaign_protocol.json"
PROGRESS_PATH = ROOT / "outputs/bill_target74/bill_target74_unbounded_20261024/progress.json"
LEDGER_PATH = ROOT / "outputs/bill_target74/bill_target74_unbounded_20261024/fits.jsonl"
CARD_KEY = "R166_listaware_r97_shrink035"
TRACK = "list_aware"
SCREEN_MONTHS = ("2023-03", "2024-06")
DEVELOPMENT_MONTHS = ("2023-03", "2023-07", "2023-11", "2024-03",
                      "2024-06", "2024-09", "2025-01", "2025-03")
RESERVED_MONTHS = {"2025-05", "2025-08"}
ORIGINAL_SHRINKAGE = 0.50
CANDIDATE_SHRINKAGE = 0.35
SCREEN_BOOTSTRAP_SEED = 20261166
DEVELOPMENT_BOOTSTRAP_SEED = 20261167
BOOTSTRAP_REPLICATES = 2000


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(actual: np.ndarray, prediction: np.ndarray) -> dict:
    ape = np.abs(prediction - actual) / actual * 100.0
    return {"n": int(len(actual)), "mdape_percent": float(np.median(ape)),
            "mape_percent": float(np.mean(ape))}


def _score(frame: pd.DataFrame, reference: str, months: tuple[str, ...], stage: str) -> dict:
    actual = frame.sold_price.to_numpy(float)
    before = frame[reference].to_numpy(float)
    after = frame.candidate_prediction.to_numpy(float)
    parent_metrics = _metric(actual, before)
    candidate_metrics = _metric(actual, after)
    gain = (parent_metrics["mdape_percent"] - candidate_metrics["mdape_percent"])
    gain = 100.0 * gain / parent_metrics["mdape_percent"]
    mape_change = (candidate_metrics["mape_percent"] - parent_metrics["mape_percent"])
    mape_change = 100.0 * mape_change / parent_metrics["mape_percent"]
    month_diffs: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    monthly, month_gains = {}, []
    for month in months:
        subset = frame.loc[frame.month.eq(month)]
        y = subset.sold_price.to_numpy(float)
        b = subset[reference].to_numpy(float)
        c = subset.candidate_prediction.to_numpy(float)
        before_month, after_month = _metric(y, b), _metric(y, c)
        month_gain = 100.0 * (before_month["mdape_percent"] - after_month["mdape_percent"])
        month_gain /= before_month["mdape_percent"]
        month_gains.append(month_gain)
        monthly[month] = {"benchmark": before_month, "candidate": after_month,
                          "relative_mdape_improvement_percent": float(month_gain)}
        month_diffs[month] = (y, b, c)
    seed = SCREEN_BOOTSTRAP_SEED if stage == "screen" else DEVELOPMENT_BOOTSTRAP_SEED
    rng = np.random.default_rng(seed)
    deltas = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    names = sorted(month_diffs)
    for replicate in range(BOOTSTRAP_REPLICATES):
        y_parts, b_parts, c_parts = [], [], []
        for name in names:
            y, b, c = month_diffs[name]
            sample = rng.integers(0, len(y), len(y))
            y_parts.append(y[sample])
            b_parts.append(b[sample])
            c_parts.append(c[sample])
        y = np.concatenate(y_parts)
        b = np.concatenate(b_parts)
        c = np.concatenate(c_parts)
        deltas[replicate] = np.median(np.abs(c-y)/y*100.0) - np.median(np.abs(b-y)/y*100.0)
    interval = {"lower95": float(np.quantile(deltas, 0.025)),
                "upper95": float(np.quantile(deltas, 0.975)),
                "replicates": BOOTSTRAP_REPLICATES, "seed": seed}
    gates = _json(PROTOCOL_PATH)["gates"]
    if stage == "screen":
        checks = {"relative_mdape_gain": gain >= gates["screen_relative_mdape_improvement_percent"],
                  "both_months_improved": all(value > 0 for value in month_gains),
                  "mape_guard": mape_change <= gates["screen_mape_regression_max_percent"],
                  "bootstrap_upper_below_zero": interval["upper95"] < 0.0}
    else:
        checks = {"relative_mdape_gain": gain >= gates["development_relative_mdape_improvement_percent"],
                  "months_improved": sum(value > 0 for value in month_gains) >= gates["development_months_improved_min"],
                  "worst_month_regression_guard": max(0.0, max(-value for value in month_gains))
                  <= gates["development_worst_month_regression_max_percent"],
                  "mape_guard": mape_change <= gates["development_mape_regression_max_percent"],
                  "bootstrap_upper_below_zero": interval["upper95"] < 0.0}
    return {"n": len(frame), "benchmark": parent_metrics, "candidate": candidate_metrics,
            "relative_mdape_improvement_percent": float(gain),
            "mape_relative_change_percent": float(mape_change),
            "months_improved": int(sum(value > 0 for value in month_gains)),
            "worst_month_regression_percent": float(max(0.0, max(-value for value in month_gains))),
            "monthly": monthly, "paired_bootstrap_mdape_difference_pp": interval,
            "gate_checks": checks, "qualifies": all(checks.values()),
            "strict_target_below_4_met": candidate_metrics["mdape_percent"] < 4.0}


def _close(left, right, tolerance: float = 1e-10) -> bool:
    return abs(float(left) - float(right)) <= tolerance


def _score_equal(saved: dict, rebuilt: dict) -> bool:
    for key in ("n", "months_improved", "qualifies", "strict_target_below_4_met"):
        if saved.get(key) != rebuilt.get(key):
            return False
    for key in ("relative_mdape_improvement_percent", "mape_relative_change_percent",
                "worst_month_regression_percent"):
        if not _close(saved[key], rebuilt[key]):
            return False
    for part in ("benchmark", "candidate"):
        for key in ("n", "mdape_percent", "mape_percent"):
            if saved[part].get(key) != rebuilt[part].get(key) if key == "n" else not _close(saved[part][key], rebuilt[part][key]):
                return False
    for key in ("lower95", "upper95", "seed", "replicates"):
        if key in ("seed", "replicates"):
            if saved["paired_bootstrap_mdape_difference_pp"][key] != rebuilt["paired_bootstrap_mdape_difference_pp"][key]:
                return False
        elif not _close(saved["paired_bootstrap_mdape_difference_pp"][key],
                        rebuilt["paired_bootstrap_mdape_difference_pp"][key]):
            return False
    if saved["gate_checks"] != rebuilt["gate_checks"]:
        return False
    if set(saved["monthly"]) != set(rebuilt["monthly"]):
        return False
    for month in saved["monthly"]:
        old, new = saved["monthly"][month], rebuilt["monthly"][month]
        if not _close(old["relative_mdape_improvement_percent"], new["relative_mdape_improvement_percent"]):
            return False
        for part in ("benchmark", "candidate"):
            for key in ("n", "mdape_percent", "mape_percent"):
                if old[part][key] != new[part][key] if key == "n" else not _close(old[part][key], new[part][key]):
                    return False
    return True


def audit() -> dict:
    protocol, progress = _json(PROTOCOL_PATH), _json(PROGRESS_PATH)
    card = protocol["cards"][CARD_KEY]
    run_id = progress["r166_run_id"]
    result_path = ROOT / progress["r166_result"]
    run_dir = result_path.parent
    pair_path, manifest_path = run_dir / "paired_predictions.csv", run_dir / "run_manifest.json"
    result, manifest = _json(result_path), _json(manifest_path)
    stage = result["stage"]
    months = SCREEN_MONTHS if stage == "screen" else DEVELOPMENT_MONTHS
    if stage not in {"screen", "development"}:
        raise RuntimeError("R166 result has an invalid stage")
    runner_path = ROOT / "experiments/bill_target74/r166_listaware_r97_shrink035.py"
    registered_auditor_path = ROOT / "experiments/bill_target74/r166_independent_audit.py"
    auditor_path = Path(__file__)
    source_hashes = card["source_sha256"]
    if (card.get("status") not in {"screen_passed", "screen_failed", "development_complete"}
            or result.get("run_id") != run_id or result.get("card") != CARD_KEY
            or result.get("campaign_id") != protocol["campaign_id"]
            or result.get("months") != list(months)
            or result.get("fit_starts") != 0
            or card.get("source_sha256") != manifest.get("source_sha256")
            or source_hashes.get("experiments/bill_target74/r166_listaware_r97_shrink035.py") != _sha(runner_path)
            or source_hashes.get("experiments/bill_target74/r166_independent_audit.py") != _sha(registered_auditor_path)):
        raise RuntimeError("R166 run identity or registered source hashes failed")
    if (result.get("reserved_months_labels_opened") is not False
            or result.get("hidden_labels_opened") is not False
            or manifest.get("reserved_months_labels_opened") is not False
            or manifest.get("hidden_labels_opened") is not False
            or card.get("reserved_months_labels_opened") is not False
            or card.get("hidden_labels_opened") is not False):
        raise RuntimeError("R166 reports reserve or hidden-label access")
    if (manifest.get("paired_predictions_sha256") != _sha(pair_path)
            or manifest.get("result_sha256") != _sha(result_path)
            or manifest.get("protocol_sha256") != _sha(run_dir / manifest["protocol_snapshot_path"])):
        raise RuntimeError("R166 result, predictions, or protocol snapshot hash mismatch")
    for rel, digest in source_hashes.items():
        current = ROOT / rel
        snapshot = run_dir / manifest["source_snapshot_paths"][rel]
        if not current.is_file() or not snapshot.is_file() or _sha(snapshot) != digest:
            raise RuntimeError(f"R166 source snapshot failed: {rel}")
    for rel, artifact in card["input_artifacts"].items():
        path = ROOT / artifact["path"]
        if (not path.is_file() or _sha(path) != artifact["sha256"]
                or manifest.get("input_artifacts", {}).get(rel) != artifact):
            raise RuntimeError(f"R166 frozen input hash failed: {rel}")
    if (card.get("candidate_shrinkage") != CANDIDATE_SHRINKAGE
            or card.get("original_shrinkage") != ORIGINAL_SHRINKAGE
            or result.get("candidate_shrinkage") != CANDIDATE_SHRINKAGE):
        raise RuntimeError("R166 correction shrinkage differs from registration")

    paths = {name: ROOT / artifact["path"] for name, artifact in card["input_artifacts"].items()}
    r97 = pd.read_csv(paths["r97_pairs"], dtype={"ml_num": str})
    r35 = pd.read_csv(paths["r35_pairs"], dtype={"ml_num": str})
    r94 = pd.read_csv(paths["r94_pairs"], dtype={"ml_num": str})
    r97 = r97.loc[r97.track.eq(TRACK) & r97.month.astype(str).isin(months),
                  ["month", "ml_num", "sold_price", "base_prediction", "candidate_prediction", "correction_log"]].copy()
    r35 = r35.loc[r35.track.eq(TRACK) & r35.month.astype(str).isin(months),
                  ["month", "ml_num", "sold_price", "candidate_prediction"]].copy()
    r94 = r94.loc[r94.track.eq(TRACK) & r94.month.astype(str).isin(months),
                  ["month", "ml_num", "sold_price", "base_prediction"]].copy()
    for frame in (r97, r35, r94):
        frame["month"] = frame.month.astype(str)
        frame["ml_num"] = frame.ml_num.astype(str)
        if frame.duplicated(["month", "ml_num"]).any():
            raise RuntimeError("R166 source contains duplicate month/listing keys")
    keys = ["month", "ml_num"]
    source = r97.merge(r35, on=keys, how="outer", suffixes=("_r97", "_r35"),
                       indicator=True, validate="one_to_one")
    if len(source) != len(r97) or not source._merge.eq("both").all():
        raise RuntimeError("R166 R97 and R35 keys do not align")
    source = source.drop(columns="_merge").merge(r94, on=keys, how="outer", suffixes=("", "_r94"),
                                                 indicator=True, validate="one_to_one")
    if not source._merge.eq("both").all():
        raise RuntimeError("R166 R94 and OOF keys do not align")
    for label_col in ("sold_price_r35", "sold_price"):
        if not np.array_equal(source.sold_price_r97.to_numpy(float), source[label_col].to_numpy(float)):
            raise RuntimeError("R166 truth values differ across frozen artifacts")
    if (not np.allclose(source.base_prediction.to_numpy(float),
                        source.candidate_prediction_r35.to_numpy(float), rtol=0.0, atol=0.005)
            or not np.allclose(source.base_prediction.to_numpy(float),
                               source.base_prediction_r94.to_numpy(float), rtol=0.0, atol=0.005)):
        raise RuntimeError("R166 R35 base predictions do not align across artifacts")
    predicted_original = np.rint(source.base_prediction.to_numpy(float)
                                 * np.exp(source.correction_log.to_numpy(float)))
    if not np.array_equal(predicted_original, source.candidate_prediction_r97.to_numpy(float)):
        raise RuntimeError("R166 cannot reconstruct saved R97 predictions from corrections")
    candidate = np.rint(source.base_prediction.to_numpy(float)
                        * np.exp((CANDIDATE_SHRINKAGE / ORIGINAL_SHRINKAGE)
                                 * source.correction_log.to_numpy(float)))
    rebuilt = pd.DataFrame({"month": source.month.astype(str), "track": TRACK,
                            "ml_num": source.ml_num.astype(str),
                            "sold_price": source.sold_price_r97.to_numpy(float),
                            "benchmark_prediction": source.candidate_prediction_r35.to_numpy(float),
                            "current_best_prediction": source.candidate_prediction_r97.to_numpy(float),
                            "candidate_prediction": candidate})
    saved_pairs = pd.read_csv(pair_path, dtype={"ml_num": str})
    saved_pairs["month"] = saved_pairs.month.astype(str)
    saved_pairs["ml_num"] = saved_pairs.ml_num.astype(str)
    rebuilt = rebuilt.sort_values(keys).reset_index(drop=True)
    saved_pairs = saved_pairs.sort_values(keys).reset_index(drop=True)
    fields = ["month", "track", "ml_num", "sold_price", "benchmark_prediction",
              "current_best_prediction", "candidate_prediction"]
    if (len(saved_pairs) != len(rebuilt)
            or saved_pairs[fields].to_dict("records") != rebuilt[fields].to_dict("records")):
        raise RuntimeError("R166 stored predictions differ from independent reconstruction")
    if set(saved_pairs.month) != set(months) or set(saved_pairs.month).intersection(RESERVED_MONTHS):
        raise RuntimeError("R166 result has unexpected or reserved months")
    if stage == "screen" and len(saved_pairs) != 4662:
        raise RuntimeError("R166 screen row count differs from fixed 4,662-row design")
    if any(saved_pairs.groupby("month").size().get(month, 0) == 0 for month in months):
        raise RuntimeError("R166 is missing a registered month")
    if not np.isfinite(saved_pairs[["sold_price", "benchmark_prediction", "current_best_prediction",
                                   "candidate_prediction"]].to_numpy(float)).all():
        raise RuntimeError("R166 contains non-finite prices")
    rebuilt_scores = {"benchmark_prediction": _score(rebuilt, "benchmark_prediction", months, stage),
                      "current_best_prediction": _score(rebuilt, "current_best_prediction", months, stage)}
    if set(result.get("scores_vs_comparators", {})) != set(rebuilt_scores):
        raise RuntimeError("R166 comparator set differs from registration")
    if any(not _score_equal(result["scores_vs_comparators"][key], score)
           for key, score in rebuilt_scores.items()):
        raise RuntimeError("R166 independently recomputed metrics/gates differ from saved result")
    qualifies = all(score["qualifies"] for score in rebuilt_scores.values())
    if result.get("screen_passed") is not (qualifies if stage == "screen" else None):
        raise RuntimeError("R166 screen decision differs from recomputed gates")

    events = []
    for line in LEDGER_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            if item.get("card") == CARD_KEY and item.get("run_id") == run_id:
                events.append(item)
    starts = [item for item in events if item.get("event") == "fit_start"]
    completes = [item for item in events if item.get("event") == "fit_complete"]
    if starts or completes or len([e for e in events if e.get("event") == "screen_fold_complete"]) != 2:
        raise RuntimeError("R166 zero-fit/two-fold ledger does not reconcile")
    if any(item.get("reserved_months_labels_opened") is not False
           or item.get("hidden_labels_opened") is not False for item in events):
        raise RuntimeError("R166 ledger access flags are not false")
    limits = protocol.get("limits", {})
    if any(value is not None for key, value in limits.items() if key != "learner_threads"):
        raise RuntimeError("R166 campaign unexpectedly has a fit or time cap")

    audit_result = {
        "run_id": run_id, "card": CARD_KEY, "stage": stage,
        "rows": int(len(saved_pairs)), "unique_month_listing_keys": True,
        "safe_months_only": list(months), "zero_learner_fits": True,
        "strict_r35_and_r97_row_alignment": True,
        "r97_predictions_reconstructed": True,
        "candidate_prediction_reconstructed": True,
        "independent_score_reconstruction_passed": True,
        "registered_runner_auditor_hashes_match": True,
        "registered_auditor_preflight_issue": (
            "The registered auditor resolved run-relative provenance snapshot paths from the repository root. "
            "This post-run wrapper resolves the same registered paths from the R166 batch directory; it does "
            "not alter the registered runner or auditor and does not refit the model."
        ),
        "post_run_auditor_path": str(auditor_path.relative_to(ROOT)).replace("\\", "/"),
        "post_run_auditor_sha256": _sha(auditor_path),
        "result_prediction_protocol_hashes_match": True,
        "campaign_limits_unbounded": True,
        "r166_reserved_or_hidden_labels_opened": False,
        "campaign_reserved_labels_previously_opened": True,
        "reserve_validation_eligible": False,
        "fit_ledger_events": len(events), "fit_starts": 0, "fit_completes": 0,
        "score_reconstruction": rebuilt_scores,
        "screen_passed_all_comparators": qualifies if stage == "screen" else None,
        "strict_target_below_4_met_by_all_candidates": {
            key: value["strict_target_below_4_met"] for key, value in rebuilt_scores.items()},
    }
    audit_path = run_dir / "independent_audit.json"
    audit_path.write_text(json.dumps(audit_result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    relative_audit = str(audit_path.relative_to(ROOT)).replace("\\", "/")
    if stage == "screen":
        status = "r166_screen_passed_audited" if qualifies else "r166_screen_failed_audited"
        next_action = ("run R166 eight-month development on frozen R97 OOF" if qualifies
                       else "retain R35/R97; register a distinct R166 follow-up hypothesis")
        campaign_promotion = qualifies
    else:
        status = "r166_development_audited"
        next_action = "review R166 development gates and fixed-origin eligibility"
        campaign_promotion = qualifies
    from experiments.bill_target74 import runner as campaign
    campaign._write_progress(status, next_action=next_action,
                             r166_audit_path=relative_audit, r166_audit_passed=True,
                             r166_registered_auditor_preflight_issue=True,
                             r166_postrun_auditor="experiments/bill_target74/r166_independent_audit_postrun.py",
                             r166_promotion_gate_passed=campaign_promotion,
                             latest_run_id=run_id, latest_card=CARD_KEY,
                             latest_audit_passed=True,
                             latest_promotion_gate_passed=campaign_promotion,
                             latest_fit_count=0)
    return {"audit_path": str(audit_path), **audit_result}


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, ensure_ascii=False))
