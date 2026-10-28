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
from experiments.bill_target74 import r03_catboost as r03
from experiments.bill_target74 import r04_ablation as provenance
from experiments.bill_target74 import runner as campaign
from experiments.bill_greedy import runner as greedy
from experiments.model_optimization_v2.features import asof_market_features
from experiments.model_optimization_v2.pipeline import blend_prices


CARD = "R99_asof_community_type_price_area"
VARIANT = "R99_asof_community_type_price_area"
SCREEN_MONTHS = ("2023-03", "2024-06")
DEVELOPMENT_MONTHS = ("2023-03", "2023-07", "2023-11", "2024-03",
                      "2024-06", "2024-09", "2025-01", "2025-03")
M08_COLUMNS = ["community_type_log_ppsf_mean", "community_type_count",
               "municipality_type_log_ppsf_mean", "municipality_type_count",
               "community_type_log_ppsf_shrunk_50"]
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
    result_path, paired_path, manifest_path = (R94_DIR / name for name in
                                               ("development_results.json", "paired_predictions.csv", "run_manifest.json"))
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
            "status": "registered", "tracks": ["off_market", "list_aware"],
            "screen_months": list(SCREEN_MONTHS), "development_months": list(DEVELOPMENT_MONTHS),
            "expected_screen_fit_starts": 6,
            "expected_development_fit_starts_per_track": {"off_market": 16, "list_aware": 8},
            "benchmark_by_track": {"off_market": "R94 best experimental candidate",
                                    "list_aware": "validated R35 parent"},
            "benchmark_prediction_sha256": manifest["paired_predictions_sha256"],
            "method": ("Retain the R22 fold-local word TF-IDF, G06 features and Detached 75/25 route "
                       "for off-market; retain the R35 text core and frozen 75/25 HGBR blend for "
                       "list-aware. Add only five M08 as-of price-per-area statistics: community/type "
                       "and municipality/type means and counts, plus the community mean shrunk toward "
                       "the municipality mean with pseudocount 50. For training rows, each query uses "
                       "only peer sales strictly before that query's listing date; for evaluation rows, "
                       "history is also strictly before the fold cutoff. No current or future sale labels "
                       "enter the added features."),
            "difference_from_prior_work": ("G14 used a lagged property-type market index; R93/R94 used "
                                            "past parent residuals. M08 adds a price-per-area prior at "
                                            "community and municipality x property-type resolution, "
                                            "strictly as of each listing."),
            "parameters": {"method_id": "M08", "community_group": ["community", "house_type_name"],
                           "municipality_group": ["municipality", "house_type_name"],
                           "shrinkage_pseudocount": 50, "history_cutoff": "strictly before date_listed",
                           "screen_gain_min_percent": 0.25, "development_gain_min_percent": 0.5,
                           "development_months_improved_min": 6, "max_mape_regression_percent": 1.0,
                           "bootstrap_replicates": 2000, "bootstrap_seed": 20261099},
            "development_gate": dict(protocol["gates"]),
            "benchmark_path": str(paired_path.relative_to(campaign.ROOT)).replace("\\", "/"),
            "data_sha256": source["data_sha256"], "parent_commit": source["commit"],
            "source_sha256": own_hash, "reserved_months_labels_opened": False,
            "hidden_labels_opened": False, "registration_amendments": [],
        }
        protocol["cards"][CARD] = card
        campaign.PROTOCOL_PATH.write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + "\n",
                                          encoding="utf-8")
        campaign._append({"event": "card_registered", "card": CARD, "data_sha256": source["data_sha256"],
                          "source_sha256": own_hash,
                          "benchmark_prediction_sha256": manifest["paired_predictions_sha256"],
                          "reserved_months_labels_opened": False, "hidden_labels_opened": False})
    elif card.get("source_sha256") != own_hash:
        starts = [json.loads(line) for line in campaign.LEDGER.read_text(encoding="utf-8").splitlines()
                  if line.strip() and json.loads(line).get("card") == CARD
                  and json.loads(line).get("event") == "fit_start"]
        if starts:
            raise RuntimeError("registered R99 source hash differs after a learner fit started")
        old_hash = card.get("source_sha256")
        amendment = {"old_source_sha256": old_hash, "new_source_sha256": own_hash,
                     "reason": "remove an unused import caught by the zero-fit preflight; no model behavior changed",
                     "fit_starts_before_amendment": 0}
        card.setdefault("registration_amendments", []).append(amendment)
        card["source_sha256"] = own_hash
        campaign.PROTOCOL_PATH.write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + "\n",
                                          encoding="utf-8")
        campaign._append({"event": "card_source_amended", "card": CARD, **amendment,
                          "reserved_months_labels_opened": False, "hidden_labels_opened": False})
    if card.get("status") not in {"registered", "screen_passed", "development_complete"}:
        raise RuntimeError(f"R99 cannot resume from card status {card.get('status')!r}")
    return card


def _read_benchmark(months: tuple[str, ...]) -> pd.DataFrame:
    frame = pd.read_csv(R94_DIR / "paired_predictions.csv", dtype={"ml_num": str})
    selected = []
    for track in ("off_market", "list_aware"):
        rows = frame.loc[frame.track.eq(track) & frame.month.astype(str).isin(months),
                         ["month", "ml_num", "sold_price", "base_prediction", "candidate_prediction"]].copy()
        column = "candidate_prediction" if track == "off_market" else "base_prediction"
        rows["benchmark_prediction"] = pd.to_numeric(rows[column], errors="raise")
        rows["track"] = track
        selected.append(rows[["month", "track", "ml_num", "sold_price", "benchmark_prediction"]])
    result = pd.concat(selected, ignore_index=True)
    if result.empty or result.duplicated(["month", "track", "ml_num"]).any():
        raise RuntimeError("benchmark has empty or duplicate track/month/listing rows")
    result["month"] = result.month.astype(str)
    return result


def _metric(actual: np.ndarray, prediction: np.ndarray) -> dict:
    ape = np.abs(prediction - actual) / actual * 100.0
    return {"n": int(len(actual)), "mdape_percent": float(np.median(ape)),
            "mape_percent": float(np.mean(ape))}


def _score(paired: pd.DataFrame, track: str, months: tuple[str, ...], stage: str) -> dict:
    rows = paired.loc[paired.track.eq(track)]
    actual = rows.sold_price.to_numpy(float)
    benchmark = rows.benchmark_prediction.to_numpy(float)
    candidate = rows.candidate_prediction.to_numpy(float)
    if not np.isfinite(np.column_stack([actual, benchmark, candidate])).all() or (
            np.column_stack([actual, benchmark, candidate]) <= 0).any():
        raise RuntimeError(f"non-finite/non-positive scoring values for {track}")
    bm, cm = _metric(actual, benchmark), _metric(actual, candidate)
    gain = 100.0 * (bm["mdape_percent"] - cm["mdape_percent"]) / bm["mdape_percent"]
    mape_change = 100.0 * (cm["mape_percent"] - bm["mape_percent"]) / bm["mape_percent"]
    monthly, grouped = {}, []
    for month in months:
        part = rows.loc[rows.month.astype(str).eq(month)]
        b, c = _metric(part.sold_price.to_numpy(float), part.benchmark_prediction.to_numpy(float)), _metric(
            part.sold_price.to_numpy(float), part.candidate_prediction.to_numpy(float))
        month_gain = 100.0 * (b["mdape_percent"] - c["mdape_percent"]) / b["mdape_percent"]
        monthly[month] = {"benchmark": b, "candidate": c,
                          "relative_mdape_improvement_percent": month_gain}
        grouped.append((part, month_gain))
    improved = sum(v > 0 for _, v in grouped)
    worst_regression = max(0.0, max(-v for _, v in grouped))
    score = {"n": len(rows), "benchmark": bm, "candidate": cm,
             "relative_mdape_improvement_percent": gain, "mape_relative_change_percent": mape_change,
             "months_improved": improved, "worst_month_regression_percent": worst_regression,
             "monthly": monthly}
    gates = campaign.PROTOCOL["gates"]
    if stage == "screen":
        score["qualifies_screen"] = bool(
            gain >= gates["screen_relative_mdape_improvement_percent"]
            and improved == len(months)
            and mape_change <= gates["screen_mape_regression_max_percent"])
    else:
        rng = np.random.default_rng(20261099)
        diffs = np.empty(2000, dtype=float)
        arrays = [(p.sold_price.to_numpy(float), p.benchmark_prediction.to_numpy(float),
                   p.candidate_prediction.to_numpy(float)) for p, _ in grouped]
        for i in range(len(diffs)):
            samples = [rng.integers(0, len(a), len(a)) for a, _, _ in arrays]
            b_ape = np.concatenate([np.abs(b[ix] - a[ix]) / a[ix] * 100 for ix, (a, b, _) in zip(samples, arrays)])
            c_ape = np.concatenate([np.abs(c[ix] - a[ix]) / a[ix] * 100 for ix, (a, _, c) in zip(samples, arrays)])
            diffs[i] = np.median(c_ape) - np.median(b_ape)
        score["paired_bootstrap_mdape_difference_pp"] = {
            "lower95": float(np.quantile(diffs, .025)), "upper95": float(np.quantile(diffs, .975)),
            "replicates": len(diffs), "seed": 20261099}
        score["qualifies_development"] = bool(
            gain >= gates["development_relative_mdape_improvement_percent"]
            and improved >= gates["development_months_improved_min"]
            and worst_regression <= gates["development_worst_month_regression_max_percent"]
            and mape_change <= gates["development_mape_regression_max_percent"]
            and score["paired_bootstrap_mdape_difference_pp"]["upper95"] < 0)
    return score


def _market_lookup(train: pd.DataFrame, heldout: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    if train.ml_num.astype(str).duplicated().any() or heldout.ml_num.astype(str).duplicated().any():
        raise RuntimeError("M08 listing keys must be unique in train and evaluation rows")
    queries = pd.concat([train, heldout], ignore_index=True)
    extra = asof_market_features(queries, train, cutoff)[M08_COLUMNS]
    extra.index = queries.ml_num.astype(str).to_numpy()
    if extra.index.duplicated().any() or not np.isfinite(extra.to_numpy(float)).all():
        raise RuntimeError("M08 feature lookup contains duplicate IDs or non-finite values")
    return extra


def _fit_with_m08(month, track, train, fit_rows, eval_rows, market, component,
                  run_id, output, source, fit_details, stage):
    from model_candidate_v2 import features as frozen_features
    attr = "off_market_features" if track == "off_market" else "list_aware_features"
    old_base = getattr(frozen_features, attr)
    transformer, restore_tfidf = r22._install_text_features(fit_rows, eval_rows, track)
    text_wrapped = getattr(frozen_features, attr)

    def with_m08(homes):
        base = text_wrapped(homes)
        ids = homes.ml_num.astype(str).to_numpy()
        values = market.loc[ids, M08_COLUMNS].copy()
        values.index = base.index
        return pd.concat([base, values], axis=1)

    setattr(frozen_features, attr, with_m08)
    fit_id = f"{campaign.PROTOCOL['campaign_id']}:{CARD}:{track}:{month}:{component}:{run_id}"

    def record_fit(_original_id, fit_fn):
        campaign._append({"event": "fit_start", "card": CARD, "stage": stage,
                          "run_id": run_id, "fit_id": fit_id, "track": track, "month": month,
                          "component": component, "data_sha256": source["data_sha256"],
                          "reserved_months_labels_opened": False, "hidden_labels_opened": False})
        start = time.perf_counter()
        try:
            fitted = fit_fn()
        except Exception as exc:
            campaign._append({"event": "fit_complete", "card": CARD, "run_id": run_id,
                              "fit_id": fit_id, "track": track, "month": month,
                              "component": component, "status": "failed",
                              "seconds": time.perf_counter() - start,
                              "error": f"{type(exc).__name__}: {exc}"})
            raise
        preds = np.asarray(fitted["predictions"], dtype=float)
        dest = output / "component_predictions" / track / month / f"{component}.csv"
        dest.parent.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame({"ml_num": eval_rows.ml_num.astype(str).to_numpy(),
                              "predicted_price": preds})
        frame.to_csv(dest, index=False)
        detail = {"fit_id": fit_id, "track": track, "month": month, "component": component,
                  "seconds": time.perf_counter() - start, "training_rows": fitted["training_rows"],
                  "evaluation_rows": fitted["evaluation_rows"], "target": fitted["target"],
                  "model_family": fitted["model_family"], "model_config": fitted["model_config"],
                  "feature_columns": fitted["feature_columns"],
                  "m08_features": M08_COLUMNS,
                  "prediction_file": str(dest.relative_to(output)).replace("\\", "/"),
                  "prediction_file_sha256": _sha256(dest)}
        fit_details.append(detail)
        campaign._append({"event": "fit_complete", "card": CARD, "stage": stage, "run_id": run_id,
                          "fit_id": fit_id, "track": track, "month": month, "component": component,
                          "status": "complete", "seconds": detail["seconds"],
                          "prediction_file_sha256": detail["prediction_file_sha256"],
                          "target": detail["target"], "training_rows": detail["training_rows"]})
        return fitted

    old_fit_record = greedy._fit_record
    greedy._fit_record = record_fit
    try:
        return greedy._fit_history_candidate(
            month, track, train, eval_rows, variant=VARIANT,
            base_variant="G06_lot_garage" if track == "off_market" else None,
            fit_tag=f"R99-{component}-{month}", fit_rows=fit_rows)
    finally:
        greedy._fit_record = old_fit_record
        setattr(frozen_features, attr, text_wrapped)
        restore_tfidf()


def run(stage: str = "screen") -> dict:
    card = _register()
    if stage == "screen" and card["status"] != "registered":
        raise RuntimeError("R99 screen has already run")
    if stage == "development" and card["status"] != "screen_passed":
        raise RuntimeError("R99 development requires at least one passing screen track")
    source = provenance._source_identity()
    months = SCREEN_MONTHS if stage == "screen" else DEVELOPMENT_MONTHS
    tracks = ["off_market", "list_aware"] if stage == "screen" else card["screen_passed_tracks"]
    benchmark = _read_benchmark(months)
    run_id = f"r99m_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    output = campaign.CAMPAIGN_ROOT / "batches" / run_id
    output.mkdir(parents=True, exist_ok=False)
    expected = len(months) * (2 if "off_market" in tracks else 0) + len(months) * (1 if "list_aware" in tracks else 0)
    campaign._append({"event": "card_start", "card": CARD, "stage": stage, "run_id": run_id,
                      "months": list(months), "tracks": tracks, "expected_fit_starts": expected,
                      "reserved_months_labels_opened": False, "hidden_labels_opened": False})
    from data import hide_answers
    from experiments.bill_greedy.runner import _blend_specialist_predictions

    old_frozen = greedy._frozen_hgbr_predictions
    greedy._frozen_hgbr_predictions = lambda month, track, train, homes: r03._verified_hgbr_cache(
        greedy, month, track, train, homes)
    fit_details, fold_frames = [], []
    try:
        for month in months:
            listings, train, _, truth, heldout = greedy._load_fold(month)
            cutoff = pd.Period(month, freq="M").start_time
            dates = pd.to_datetime(train.date_sold, errors="raise")
            if (dates >= cutoff).any():
                raise RuntimeError(f"training labels are not strictly before {month}")
            if dates.dt.to_period("M").astype(str).isin(campaign.PROTOCOL["reserved_months"]).any():
                raise RuntimeError("reserved-month labels entered training data")
            market = _market_lookup(train, heldout, cutoff)
            for track in tracks:
                homes = hide_answers(heldout, track)
                base_rows = benchmark.loc[benchmark.month.eq(month) & benchmark.track.eq(track)].copy()
                if len(base_rows) != len(heldout) or set(base_rows.ml_num) != set(heldout.ml_num.astype(str)):
                    raise RuntimeError(f"benchmark row identities do not align for {track}/{month}")
                labels = base_rows.set_index("ml_num").loc[heldout.ml_num.astype(str), "sold_price"].to_numpy(float)
                if not np.array_equal(labels, truth.sold_price.to_numpy(float)):
                    raise RuntimeError(f"benchmark labels do not align for {track}/{month}")
                if track == "off_market":
                    specialist = greedy._specialist_mask(train, "Detached")
                    eval_specialist = greedy._specialist_mask(homes, "Detached")
                    base_fit = _fit_with_m08(month, track, train, train, homes, market, "base",
                                             run_id, output, source, fit_details, stage)
                    expert_train = train.loc[specialist].copy()
                    expert_homes = homes.loc[eval_specialist].copy()
                    expert_fit = _fit_with_m08(month, track, train, expert_train, expert_homes, market,
                                               "detached_expert", run_id, output, source, fit_details, stage)
                    parent_price = np.round(base_fit["predictions"])
                    expert_price = np.round(expert_fit["predictions"])
                    candidate = _blend_specialist_predictions(parent_price, expert_price,
                                                               homes, "Detached", 0.75)
                else:
                    core = _fit_with_m08(month, track, train, train, homes, market, "core",
                                         run_id, output, source, fit_details, stage)
                    hgbr = greedy._frozen_hgbr_predictions(month, track, train, homes)
                    candidate = blend_prices(core["predictions"], hgbr, challenger_weight=0.75)
                benchmark_prices = base_rows.set_index("ml_num").loc[
                    heldout.ml_num.astype(str), "benchmark_prediction"].to_numpy(float)
                fold_frames.append(pd.DataFrame({"month": month, "track": track,
                                                 "ml_num": heldout.ml_num.astype(str).to_numpy(),
                                                 "sold_price": truth.sold_price.to_numpy(float),
                                                 "benchmark_prediction": benchmark_prices,
                                                 "candidate_prediction": np.asarray(candidate, dtype=float)}))
                campaign._append({"event": "screen_fold_complete" if stage == "screen" else "development_fold_complete",
                                  "card": CARD, "run_id": run_id, "stage": stage,
                                  "month": month, "track": track, "rows": len(heldout),
                                  "reserved_months_labels_opened": False, "hidden_labels_opened": False})
    except Exception as exc:
        failure = {"run_id": run_id, "card": CARD, "stage": stage,
                   "error": f"{type(exc).__name__}: {exc}", "fit_details": fit_details,
                   "reserved_months_labels_opened": False, "hidden_labels_opened": False}
        (output / "failure_checkpoint.json").write_text(json.dumps(failure, indent=2, ensure_ascii=False) + "\n",
                                                         encoding="utf-8")
        campaign._append({"event": "card_failed", "card": CARD, "stage": stage, "run_id": run_id,
                          "fit_starts": len(fit_details), "error": failure["error"]})
        campaign._write_progress("r99_failed", next_action="inspect R99 checkpoint and ledger before retry",
                                 r99_failure=str((output / "failure_checkpoint.json").relative_to(campaign.ROOT)).replace("\\", "/"))
        raise
    finally:
        greedy._frozen_hgbr_predictions = old_frozen

    paired = pd.concat(fold_frames, ignore_index=True)
    paired_path = output / "paired_predictions.csv"
    paired.to_csv(paired_path, index=False)
    scores = {track: _score(paired, track, months, stage) for track in tracks}
    passed = [track for track, value in scores.items() if value.get("qualifies_screen" if stage == "screen" else "qualifies_development")]
    result = {"campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
              "card": CARD, "stage": stage, "months": list(months), "tracks": tracks,
              "fit_starts": len(fit_details), "scores": scores,
              "screen_passed_tracks": passed, "fit_details": fit_details,
              "reserved_months_labels_opened": False, "hidden_labels_opened": False,
              "data_sha256": source["data_sha256"]}
    result_path = output / "results.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    source_files = [Path(__file__), campaign.PROTOCOL_PATH, campaign.PACKAGE / "r22_remarks_tfidf.py",
                    campaign.PACKAGE / "r03_catboost.py", campaign.PACKAGE / "r04_ablation.py",
                    campaign.PACKAGE / "runner.py", campaign.ROOT / "experiments/bill_greedy/runner.py",
                    campaign.ROOT / "experiments/bill_greedy/history.py",
                    campaign.ROOT / "experiments/model_optimization_v2/features.py",
                    campaign.ROOT / "experiments/model_optimization_v2/learners.py",
                    campaign.ROOT / "model_candidate_v2/features.py",
                    R94_DIR / "development_results.json", R94_DIR / "paired_predictions.csv",
                    R94_DIR / "run_manifest.json"]
    provenance_dir = output / "provenance"
    provenance_dir.mkdir(exist_ok=True)
    source_hashes = {}
    for path in source_files:
        key = str(path.relative_to(campaign.ROOT)).replace("\\", "/") if path.is_relative_to(campaign.ROOT) else "external/" + path.name
        source_hashes[key] = _sha256(path)
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
    report = ["# R99 strict as-of community/type price-per-area feature", "",
              "M08 statistics use only sales strictly before each listing date; the outer fold also excludes all sales on or after its first day.",
              "Benchmarks: R94 candidate for off-market and R35 parent for list-aware.", ""]
    for track, score in scores.items():
        report.extend([f"## {track}", "",
                       f"- Benchmark MdAPE: {score['benchmark']['mdape_percent']:.6f}%",
                       f"- Candidate MdAPE: {score['candidate']['mdape_percent']:.6f}%",
                       f"- Relative MdAPE gain: {score['relative_mdape_improvement_percent']:+.4f}%",
                       f"- MAPE relative change: {score['mape_relative_change_percent']:+.4f}%",
                       f"- Months improved: {score['months_improved']}/{len(months)}",
                       f"- Gate: {score.get('qualifies_screen', score.get('qualifies_development'))}", ""])
        for month, item in score["monthly"].items():
            report.append(f"- {month}: benchmark {item['benchmark']['mdape_percent']:.5f}%, "
                          f"candidate {item['candidate']['mdape_percent']:.5f}%, "
                          f"gain {item['relative_mdape_improvement_percent']:+.3f}%")
        report.append("")
    report.append("No reserve or hidden labels were opened.")
    (output / "research_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    card["last_run_id"] = run_id
    card["last_result_path"] = str(result_path.relative_to(campaign.ROOT)).replace("\\", "/")
    card["screen_passed_tracks"] = passed if stage == "screen" else card.get("screen_passed_tracks", [])
    if stage == "screen":
        card["status"] = "screen_passed" if passed else "screen_failed"
    else:
        card["status"] = "development_complete"
    campaign.PROTOCOL_PATH.write_text(json.dumps(campaign.PROTOCOL, indent=2, ensure_ascii=False) + "\n",
                                      encoding="utf-8")
    campaign._append({"event": "card_complete", "card": CARD, "stage": stage, "run_id": run_id,
                      "fit_starts": len(fit_details), "result_path": str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
                      "qualifies_by_track": {track: value.get("qualifies_screen" if stage == "screen" else "qualifies_development")
                                              for track, value in scores.items()},
                      "reserved_months_labels_opened": False, "hidden_labels_opened": False})
    next_action = ("run R99 eight-month development for: " + ", ".join(passed)
                   if stage == "screen" and passed else
                   "retain current parents; R99 failed screen; register a distinct method" if stage == "screen" else
                   "review R99 development gates and fixed-origin eligibility")
    campaign._write_progress("r99_screen_complete" if stage == "screen" else "r99_development_complete",
                             next_action=next_action, r99_run_id=run_id,
                             r99_result=str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
                             latest_fit_count=len(fit_details))
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--development", action="store_true")
    args = parser.parse_args()
    result = run("development" if args.development else "screen")
    print(json.dumps({"run_id": result["run_id"], "fit_starts": result["fit_starts"],
                      "screen_passed_tracks": result["screen_passed_tracks"],
                      "scores": {track: {"benchmark_mdape": score["benchmark"]["mdape_percent"],
                                         "candidate_mdape": score["candidate"]["mdape_percent"],
                                         "relative_gain_percent": score["relative_mdape_improvement_percent"],
                                         "mape_change_percent": score["mape_relative_change_percent"],
                                         "months_improved": score["months_improved"],
                                         "gate": score.get("qualifies_screen", score.get("qualifies_development"))}
                                for track, score in result["scores"].items()}},
                     ensure_ascii=False), flush=True)
