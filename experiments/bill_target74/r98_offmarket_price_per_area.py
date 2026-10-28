from __future__ import annotations

import hashlib
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.bill_target74 import r22_remarks_tfidf as r22
from experiments.bill_target74 import r04_ablation as provenance
from experiments.bill_target74 import runner as campaign
from experiments.bill_greedy import runner as greedy


CARD = "R98_offmarket_log_price_per_area"
VARIANT = "R98_offmarket_log_price_per_area"
SCREEN_MONTHS = ("2023-03", "2024-06")
DEVELOPMENT_MONTHS = ("2023-03", "2023-07", "2023-11", "2024-03",
                      "2024-06", "2024-09", "2025-01", "2025-03")
R94_DIR = campaign.CAMPAIGN_ROOT / "batches" / "r94g_20261025T202220Z"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _register() -> dict:
    source = provenance._source_identity()
    protocol = campaign.PROTOCOL
    if source["data_sha256"] != protocol["data_sha256"]:
        raise RuntimeError("active data hash differs from campaign protocol")
    if not R94_DIR.is_dir():
        raise FileNotFoundError(f"R94 benchmark directory not found: {R94_DIR}")
    result_path = R94_DIR / "development_results.json"
    paired_path = R94_DIR / "paired_predictions.csv"
    manifest_path = R94_DIR / "run_manifest.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (result.get("run_id") != "r94g_20261025T202220Z"
            or result.get("data_sha256") != source["data_sha256"]
            or manifest.get("paired_predictions_sha256") != _sha256(paired_path)
            or manifest.get("result_sha256") != _sha256(result_path)
            or manifest.get("reserved_months_labels_opened") is not False
            or manifest.get("hidden_labels_opened") is not False):
        raise RuntimeError("R94 benchmark identity, hashes, or label-access flags failed validation")

    card = protocol["cards"].get(CARD)
    own_hash = _sha256(Path(__file__))
    if card is None:
        card = {
            "status": "registered",
            "tracks": ["off_market"],
            "screen_months": list(SCREEN_MONTHS),
            "development_months": list(DEVELOPMENT_MONTHS),
            "expected_screen_fit_starts": 4,
            "expected_development_fit_starts": 16,
            "benchmark": "R94 current best off-market experimental OOF candidate",
            "benchmark_prediction_sha256": manifest["paired_predictions_sha256"],
            "target": "log(sold_price / square_feet), invert with prediction-time square_feet",
            "method": ("Keep the R22 fold-local word TF-IDF, G06 structural/history features, LightGBM "
                       "configuration, and 75% Detached expert / 25% base route fixed. Train the same "
                       "two components on log(sold_price / square_feet), then multiply exp(prediction) "
                       "by the prediction-time square_feet. Training rows without a valid positive area "
                       "are excluded from this target-specific learner; evaluation rows without area "
                       "fall back to the R94 benchmark prediction."),
            "difference_from_prior_work": ("M12 in model_optimization_v2 was registered but not run. "
                                            "R91 tested list_price/area as a feature, not an off-market "
                                            "per-area target. This isolates the target transformation."),
            "parameters": {"target_method": "M12", "screen_gain_min_percent": 0.25,
                           "development_gain_min_percent": 0.5, "min_months_improved": 6,
                           "max_mape_regression_percent": 1.0,
                           "screen_months_improved_required": 2,
                           "bootstrap_replicates": 2000, "bootstrap_seed": 20261098},
            "development_gate": dict(protocol["gates"]),
            "r94_benchmark": str(paired_path.relative_to(campaign.ROOT)).replace("\\", "/"),
            "data_sha256": source["data_sha256"],
            "parent_commit": source["commit"],
            "source_sha256": own_hash,
            "reserved_months_labels_opened": False,
            "hidden_labels_opened": False,
            "registration_amendments": [],
        }
        protocol["cards"][CARD] = card
        campaign.PROTOCOL_PATH.write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + "\n",
                                          encoding="utf-8")
        campaign._append({"event": "card_registered", "card": CARD,
                          "data_sha256": source["data_sha256"], "source_sha256": own_hash,
                          "benchmark_prediction_sha256": manifest["paired_predictions_sha256"],
                          "reserved_months_labels_opened": False, "hidden_labels_opened": False})
    elif card.get("source_sha256") != own_hash:
        raise RuntimeError("registered R98 source hash differs from the current implementation")
    if card.get("status") not in {"registered", "screen_passed", "development_complete"}:
        raise RuntimeError(f"R98 cannot resume from card status {card.get('status')!r}")
    return card


def _read_benchmark(months: tuple[str, ...]) -> pd.DataFrame:
    path = R94_DIR / "paired_predictions.csv"
    frame = pd.read_csv(path, dtype={"ml_num": str})
    frame = frame.loc[frame.track.eq("off_market") & frame.month.astype(str).isin(months),
                      ["month", "ml_num", "sold_price", "candidate_prediction"]].copy()
    frame.rename(columns={"candidate_prediction": "benchmark_prediction"}, inplace=True)
    frame["month"] = frame.month.astype(str)
    if frame.empty or frame.duplicated(["month", "ml_num"]).any():
        raise RuntimeError("R94 benchmark has empty or duplicate off-market development rows")
    return frame


def _metric(actual: np.ndarray, prediction: np.ndarray) -> dict:
    ape = np.abs(prediction - actual) / actual * 100.0
    return {"n": int(len(actual)), "mdape_percent": float(np.median(ape)),
            "mape_percent": float(np.mean(ape))}


def _score(paired: pd.DataFrame, months: tuple[str, ...], *, stage: str) -> dict:
    actual = paired.sold_price.to_numpy(float)
    benchmark = paired.benchmark_prediction.to_numpy(float)
    candidate = paired.candidate_prediction.to_numpy(float)
    if not np.isfinite(np.column_stack([actual, benchmark, candidate])).all():
        raise RuntimeError("non-finite R98 scoring values")
    if (np.column_stack([actual, benchmark, candidate]) <= 0).any():
        raise RuntimeError("R98 requires strictly positive prices")
    base_metrics, candidate_metrics = _metric(actual, benchmark), _metric(actual, candidate)
    relative_gain = 100.0 * (base_metrics["mdape_percent"] - candidate_metrics["mdape_percent"]) / base_metrics["mdape_percent"]
    mape_change = 100.0 * (candidate_metrics["mape_percent"] - base_metrics["mape_percent"]) / base_metrics["mape_percent"]
    monthly = {}
    by_month = []
    for month in months:
        rows = paired.loc[paired.month.astype(str).eq(month)]
        b, c = _metric(rows.sold_price.to_numpy(float), rows.benchmark_prediction.to_numpy(float)), _metric(
            rows.sold_price.to_numpy(float), rows.candidate_prediction.to_numpy(float))
        gain = 100.0 * (b["mdape_percent"] - c["mdape_percent"]) / b["mdape_percent"]
        by_month.append((rows, gain))
        monthly[month] = {"benchmark": b, "candidate": c,
                          "relative_mdape_improvement_percent": gain}
    improved = sum(gain > 0 for _, gain in by_month)
    worst_regression = max(0.0, max(-gain for _, gain in by_month))
    result = {"n": len(paired), "benchmark": base_metrics, "candidate": candidate_metrics,
              "relative_mdape_improvement_percent": relative_gain,
              "mape_relative_change_percent": mape_change, "months_improved": improved,
              "worst_month_regression_percent": worst_regression, "monthly": monthly}
    if stage == "development":
        rng = np.random.default_rng(20261098)
        differences = np.empty(2000, dtype=float)
        month_arrays = [(rows.sold_price.to_numpy(float), rows.benchmark_prediction.to_numpy(float),
                         rows.candidate_prediction.to_numpy(float)) for rows, _ in by_month]
        for i in range(len(differences)):
            sampled = [rng.integers(0, len(a), len(a)) for a, _, _ in month_arrays]
            b_ape = np.concatenate([np.abs(b[idx] - a[idx]) / a[idx] * 100 for idx, (a, b, _) in zip(sampled, month_arrays)])
            c_ape = np.concatenate([np.abs(c[idx] - a[idx]) / a[idx] * 100 for idx, (a, _, c) in zip(sampled, month_arrays)])
            differences[i] = np.median(c_ape) - np.median(b_ape)
        result["paired_bootstrap_mdape_difference_pp"] = {
            "lower95": float(np.quantile(differences, .025)),
            "upper95": float(np.quantile(differences, .975)),
            "replicates": len(differences), "seed": 20261098}
        gates = campaign.PROTOCOL["gates"]
        result["qualifies_development"] = bool(
            relative_gain >= gates["development_relative_mdape_improvement_percent"]
            and improved >= gates["development_months_improved_min"]
            and worst_regression <= gates["development_worst_month_regression_max_percent"]
            and mape_change <= gates["development_mape_regression_max_percent"]
            and result["paired_bootstrap_mdape_difference_pp"]["upper95"] < 0.0)
    else:
        gates = campaign.PROTOCOL["gates"]
        result["qualifies_screen"] = bool(
            relative_gain >= gates["screen_relative_mdape_improvement_percent"]
            and improved == len(months)
            and mape_change <= gates["screen_mape_regression_max_percent"])
    return result


def run(stage: str = "screen") -> dict:
    card = _register()
    if stage not in {"screen", "development"}:
        raise ValueError("stage must be screen or development")
    if stage == "screen" and card["status"] != "registered":
        raise RuntimeError("R98 screen has already been run")
    if stage == "development" and card["status"] != "screen_passed":
        raise RuntimeError("R98 development requires a saved passing screen")
    source = provenance._source_identity()
    months = SCREEN_MONTHS if stage == "screen" else DEVELOPMENT_MONTHS
    benchmark = _read_benchmark(months)
    run_id = f"r98m_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    output = campaign.CAMPAIGN_ROOT / "batches" / run_id
    output.mkdir(parents=True, exist_ok=False)
    jobs = 2 * len(months)
    campaign._append({"event": "card_start", "card": CARD, "stage": stage, "run_id": run_id,
                      "months": list(months), "expected_fit_starts": jobs,
                      "reserved_months_labels_opened": False, "hidden_labels_opened": False})
    from data import hide_answers
    from experiments.model_optimization_v2.learners import _area

    old_fit_record = greedy._fit_record
    old_variant = greedy.VARIANTS.get(VARIANT)
    had_variant = VARIANT in greedy.VARIANTS
    greedy.VARIANTS[VARIANT] = {"target_method": "M12"}
    fit_details, fold_rows, component_artifacts = [], [], []
    fit_context = {}

    def record_fit(_original_id, fit_fn):
        track, month, component, ids = (fit_context[k] for k in ("track", "month", "component", "ids"))
        fit_id = f"{campaign.PROTOCOL['campaign_id']}:{CARD}:{track}:{month}:{component}:{run_id}"
        campaign._append({"event": "fit_start", "card": CARD, "stage": stage,
                          "run_id": run_id, "fit_id": fit_id, "track": track,
                          "month": month, "component": component,
                          "data_sha256": source["data_sha256"],
                          "reserved_months_labels_opened": False, "hidden_labels_opened": False})
        started = time.perf_counter()
        try:
            fitted = fit_fn()
        except Exception as exc:
            campaign._append({"event": "fit_complete", "card": CARD, "stage": stage,
                              "run_id": run_id, "fit_id": fit_id, "track": track,
                              "month": month, "component": component, "status": "failed",
                              "seconds": time.perf_counter() - started,
                              "error": f"{type(exc).__name__}: {exc}"})
            raise
        prediction = np.asarray(fitted["predictions"], dtype=float)
        path = output / "component_predictions" / month / f"{component}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        saved = pd.DataFrame({"ml_num": pd.Series(ids, dtype=str), "predicted_price": prediction})
        saved.to_csv(path, index=False)
        digest = _sha256(path)
        detail = {"fit_id": fit_id, "month": month, "component": component,
                  "seconds": time.perf_counter() - started,
                  "training_rows": fitted["training_rows"],
                  "excluded_invalid_area_rows": fit_context["excluded_invalid_area_rows"],
                  "evaluation_rows": fitted["evaluation_rows"], "target": fitted["target"],
                  "model_family": fitted["model_family"], "model_config": fitted["model_config"],
                  "feature_columns": fitted["feature_columns"],
                  "text_feature_count": sum(c.startswith("remarks_tfidf_") for c in fitted["feature_columns"]),
                  "prediction_file": str(path.relative_to(output)).replace("\\", "/"),
                  "prediction_file_sha256": digest,
                  "prediction_sha256": fitted["prediction_sha256"]}
        fit_details.append(detail)
        component_artifacts.append(detail)
        campaign._append({"event": "fit_complete", "card": CARD, "stage": stage,
                          "run_id": run_id, "fit_id": fit_id, "track": track,
                          "month": month, "component": component, "status": "complete",
                          "seconds": detail["seconds"], "prediction_file_sha256": digest,
                          "target": fitted["target"], "training_rows": fitted["training_rows"]})
        return fitted

    greedy._fit_record = record_fit
    try:
        for month in months:
            listings, train, _, truth, heldout = greedy._load_fold(month)
            cutoff = pd.Period(month, freq="M").start_time
            sold_dates = pd.to_datetime(train.date_sold, errors="raise")
            if (sold_dates >= cutoff).any():
                raise RuntimeError(f"training labels are not strictly before {month}")
            if sold_dates.dt.to_period("M").astype(str).isin(campaign.PROTOCOL["reserved_months"]).any():
                raise RuntimeError("a reserved month entered R98 training data")
            safe_benchmark = benchmark.loc[benchmark.month.eq(month)].copy()
            if len(safe_benchmark) != len(heldout) or set(safe_benchmark.ml_num) != set(heldout.ml_num.astype(str)):
                raise RuntimeError(f"R94 benchmark rows do not align with R98 heldout rows for {month}")
            joined_labels = safe_benchmark.set_index("ml_num").loc[heldout.ml_num.astype(str), "sold_price"].to_numpy(float)
            if not np.array_equal(joined_labels, truth.sold_price.to_numpy(float)):
                raise RuntimeError(f"R94 benchmark labels do not match heldout labels for {month}")
            homes = hide_answers(heldout, "off_market")
            area_eval = _area(homes)
            valid_eval = np.isfinite(area_eval) & (area_eval > 0)
            pred_by_id = {}
            components = {}
            for component, source_rows, eval_mask in (
                ("base", train, np.ones(len(homes), dtype=bool)),
                ("detached_expert", train.loc[greedy._specialist_mask(train, "Detached")].copy(),
                 greedy._specialist_mask(homes, "Detached")),
            ):
                area_train = _area(source_rows)
                valid_train = np.isfinite(area_train) & (area_train > 0)
                fit_rows = source_rows.loc[valid_train].copy()
                excluded = int((~valid_train).sum())
                eval_rows = homes.loc[eval_mask & valid_eval].copy()
                transformer, restore = r22._install_text_features(fit_rows, eval_rows, "off_market")
                fit_context.update(track="off_market", month=month, component=component,
                                   ids=eval_rows.ml_num.astype(str).tolist(), excluded_invalid_area_rows=excluded)
                try:
                    fitted = greedy._fit_history_candidate(
                        month, "off_market", train, eval_rows, variant=VARIANT,
                        base_variant="G06_lot_garage", fit_tag=f"R98-{component}-{month}",
                        fit_rows=fit_rows)
                finally:
                    restore()
                components[component] = fitted
                for listing_id, price in zip(eval_rows.ml_num.astype(str), fitted["predictions"]):
                    pred_by_id[listing_id] = float(price)
            base_prediction = np.round(components["base"]["predictions"])
            expert_prediction = np.round(components["detached_expert"]["predictions"])
            base_ids = homes.loc[valid_eval, "ml_num"].astype(str).to_numpy()
            detached_ids = homes.loc[greedy._specialist_mask(homes, "Detached") & valid_eval, "ml_num"].astype(str).to_numpy()
            base_map = dict(zip(base_ids, base_prediction))
            expert_map = dict(zip(detached_ids, expert_prediction))
            candidate_prediction = np.array([
                (0.75 * expert_map[listing_id] + 0.25 * base_map[listing_id])
                if listing_id in expert_map else base_map[listing_id]
                for listing_id in heldout.ml_num.astype(str)
            ], dtype=float)
            benchmark_prediction = safe_benchmark.set_index("ml_num").loc[
                heldout.ml_num.astype(str), "benchmark_prediction"].to_numpy(float)
            candidate_prediction[~valid_eval] = benchmark_prediction[~valid_eval]
            fold_rows.append(pd.DataFrame({"month": month, "ml_num": heldout.ml_num.astype(str).to_numpy(),
                                           "sold_price": truth.sold_price.to_numpy(float),
                                           "benchmark_prediction": benchmark_prediction,
                                           "candidate_prediction": candidate_prediction,
                                           "area_valid": valid_eval,
                                           "house_type_name": heldout.house_type_name.astype(str).to_numpy()}))
            campaign._append({"event": "screen_fold_complete" if stage == "screen" else "development_fold_complete",
                              "card": CARD, "run_id": run_id, "stage": stage, "month": month,
                              "rows": len(heldout), "area_valid_rows": int(valid_eval.sum()),
                              "training_rows": len(train), "reserved_months_labels_opened": False,
                              "hidden_labels_opened": False})
    except Exception as exc:
        failure = {"run_id": run_id, "card": CARD, "stage": stage,
                   "error": f"{type(exc).__name__}: {exc}", "fit_details": fit_details,
                   "reserved_months_labels_opened": False, "hidden_labels_opened": False}
        (output / "failure_checkpoint.json").write_text(json.dumps(failure, indent=2, ensure_ascii=False) + "\n",
                                                         encoding="utf-8")
        campaign._append({"event": "card_failed", "card": CARD, "stage": stage, "run_id": run_id,
                          "fit_starts": len(fit_details), "error": failure["error"]})
        campaign._write_progress("r98_failed", next_action="inspect R98 failure checkpoint and ledger before retry",
                                 r98_failure=str((output / "failure_checkpoint.json").relative_to(campaign.ROOT)).replace("\\", "/"))
        raise
    finally:
        greedy._fit_record = old_fit_record
        if had_variant:
            greedy.VARIANTS[VARIANT] = old_variant
        else:
            greedy.VARIANTS.pop(VARIANT, None)

    paired = pd.concat(fold_rows, ignore_index=True)
    paired_path = output / "paired_predictions.csv"
    paired.to_csv(paired_path, index=False)
    score = _score(paired, months, stage=stage)
    result = {"campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
              "card": CARD, "stage": stage, "months": list(months), "fit_starts": len(fit_details),
              "benchmark": "R94 off-market candidate", "score": score,
              "fit_details": fit_details, "reserved_months_labels_opened": False,
              "hidden_labels_opened": False, "data_sha256": source["data_sha256"]}
    result_path = output / "results.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    source_files = [Path(__file__), campaign.PROTOCOL_PATH, campaign.PACKAGE / "r22_remarks_tfidf.py",
                    campaign.PACKAGE / "r04_ablation.py", campaign.PACKAGE / "runner.py",
                    campaign.ROOT / "experiments/bill_greedy/runner.py",
                    campaign.ROOT / "experiments/bill_greedy/history.py",
                    campaign.ROOT / "experiments/model_optimization_v2/learners.py",
                    campaign.ROOT / "model_candidate_v2/features.py", R94_DIR / "development_results.json",
                    R94_DIR / "paired_predictions.csv", R94_DIR / "run_manifest.json"]
    provenance_dir = output / "provenance"
    provenance_dir.mkdir(exist_ok=True)
    source_hashes = {}
    for path in source_files:
        key = str(path.relative_to(campaign.ROOT)).replace("\\", "/") if path.is_relative_to(campaign.ROOT) else "external/" + path.name
        source_hashes[key] = _sha256(path)
        if path.is_file():
            shutil.copy2(path, provenance_dir / (path.name + ".snapshot"))
    manifest = {"campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
                "source_identity": source, "data_sha256": source["data_sha256"],
                "source_sha256": source_hashes, "fit_starts": len(fit_details),
                "fit_details": fit_details, "paired_predictions_sha256": _sha256(paired_path),
                "result_sha256": _sha256(result_path),
                "benchmark_prediction_sha256": _sha256(R94_DIR / "paired_predictions.csv"),
                "reserved_months_labels_opened": False, "hidden_labels_opened": False}
    manifest_path = output / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    title = "screen" if stage == "screen" else "development"
    report = [f"# R98 off-market log price-per-area {title}", "",
              "Benchmark: R94 off-market development predictions. Target areas are parsed at prediction time.",
              "R22 TF-IDF and G06 history/structural features are fit only on training rows.",
              "Training labels are strictly before each fold; target-month labels are used only to score.", "",
              f"- Rows: {score['n']}",
              f"- Benchmark MdAPE: {score['benchmark']['mdape_percent']:.6f}%",
              f"- Candidate MdAPE: {score['candidate']['mdape_percent']:.6f}%",
              f"- Relative MdAPE gain: {score['relative_mdape_improvement_percent']:+.4f}%",
              f"- MAPE relative change: {score['mape_relative_change_percent']:+.4f}%",
              f"- Months improved: {score['months_improved']}/{len(months)}",
              f"- Gate: {score.get('qualifies_screen', score.get('qualifies_development'))}", "",
              "No reserve or hidden labels were opened.", f"Fit starts: {len(fit_details)}", ""]
    for month, detail in score["monthly"].items():
        report.append(f"- {month}: benchmark {detail['benchmark']['mdape_percent']:.5f}%, "
                      f"candidate {detail['candidate']['mdape_percent']:.5f}%, "
                      f"gain {detail['relative_mdape_improvement_percent']:+.3f}%")
    (output / "research_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    card["last_run_id"] = run_id
    card["last_result_path"] = str(result_path.relative_to(campaign.ROOT)).replace("\\", "/")
    card["last_status"] = "screen_passed" if score.get("qualifies_screen") else (
        "development_complete" if score.get("qualifies_development") else "development_failed")
    card["status"] = "screen_passed" if score.get("qualifies_screen") else (
        "screen_failed" if stage == "screen" else "development_complete")
    campaign.PROTOCOL_PATH.write_text(json.dumps(campaign.PROTOCOL, indent=2, ensure_ascii=False) + "\n",
                                      encoding="utf-8")
    campaign._append({"event": "card_complete", "card": CARD, "stage": stage, "run_id": run_id,
                      "fit_starts": len(fit_details), "result_path": str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
                      "qualifies": score.get("qualifies_screen", score.get("qualifies_development")),
                      "reserved_months_labels_opened": False, "hidden_labels_opened": False})
    next_action = ("run R98 eight-month development confirmation" if score.get("qualifies_screen")
                   else "retain R94 and register another distinct off-market idea" if stage == "screen"
                   else "retain current parents; R98 failed development gates and register a distinct idea")
    campaign._write_progress("r98_screen_complete" if stage == "screen" else "r98_development_complete",
                             next_action=next_action, r98_run_id=run_id,
                             r98_result=str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
                             latest_fit_count=len(fit_details))
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--development", action="store_true")
    args = parser.parse_args()
    result = run("development" if args.development else "screen")
    score = result["score"]
    print(json.dumps({"run_id": result["run_id"], "fit_starts": result["fit_starts"],
                      "benchmark_mdape": score["benchmark"]["mdape_percent"],
                      "candidate_mdape": score["candidate"]["mdape_percent"],
                      "relative_gain_percent": score["relative_mdape_improvement_percent"],
                      "mape_change_percent": score["mape_relative_change_percent"],
                      "months_improved": score["months_improved"],
                      "gate": score.get("qualifies_screen", score.get("qualifies_development"))},
                     ensure_ascii=False), flush=True)
