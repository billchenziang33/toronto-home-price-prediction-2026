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
from experiments.model_optimization_v2.pipeline import blend_prices


CARD = "R103_catboost_ratio_listaware"
VARIANT = "R103_catboost_ratio_listaware"
CATBOOST_CONFIG = {
    "iterations": 600, "depth": 6, "learning_rate": 0.05,
    "l2_leaf_reg": 3, "loss_function": "RMSE", "has_time": True,
    "random_seed": 0, "thread_count": 4, "verbose": False,
    "allow_writing_files": False,
}
SCREEN_MONTHS = ("2023-03", "2024-06")
DEVELOPMENT_MONTHS = ("2023-03", "2023-07", "2023-11", "2024-03",
                      "2024-06", "2024-09", "2025-01", "2025-03")
R94_DIR = campaign.CAMPAIGN_ROOT / "batches" / "r94g_20261025T202220Z"


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _register():
    source = provenance._source_identity()
    result_path, paired_path, manifest_path = (R94_DIR / n for n in
                                               ("development_results.json", "paired_predictions.csv", "run_manifest.json"))
    result, manifest = json.loads(result_path.read_text(encoding="utf-8")), json.loads(
        manifest_path.read_text(encoding="utf-8"))
    if (result.get("run_id") != "r94g_20261025T202220Z"
            or result.get("data_sha256") != source["data_sha256"]
            or manifest.get("paired_predictions_sha256") != _sha(paired_path)
            or manifest.get("result_sha256") != _sha(result_path)
            or manifest.get("reserved_months_labels_opened") is not False
            or manifest.get("hidden_labels_opened") is not False):
        raise RuntimeError("R94 benchmark identity or integrity verification failed")
    protocol = campaign.PROTOCOL
    own_hash = _sha(Path(__file__))
    card = protocol["cards"].get(CARD)
    if card is None:
        card = {
            "status": "registered", "tracks": ["list_aware"],
            "screen_months": list(SCREEN_MONTHS), "development_months": list(DEVELOPMENT_MONTHS),
            "expected_screen_fit_starts": 2,
            "expected_development_fit_starts_per_track": {"list_aware": 8},
            "benchmark_by_track": {"list_aware": "R35 validated parent"},
            "benchmark_prediction_sha256": manifest["paired_predictions_sha256"],
            "method_id": "M19",
            "features": ["R22 fold-local word TF-IDF", "G01 structured/history features"],
            "method": ("For list-aware only, retain R22 fold-local word TF-IDF and current "
                       "structured/history features. Replace the LightGBM core with the fixed M19 "
                       "CatBoost configuration (600 iterations, depth 6, learning rate 0.05, L2 3, "
                       "RMSE, has_time=True), keep the existing 75/25 core/HGBR blend, and train on "
                       "the existing log(sold_price/list_price) target."),
            "difference_from_prior_work": ("The older v2 M19 screen used this CatBoost family, "
                                            "configuration and ratio target against C0 on different folds; "
                                            "it gained 0.626% MdAPE but did not pass its promotion gate. "
                                            "Current R24 used a different CatBoost configuration with "
                                            "R22 TF-IDF and regressed against R35. R103 tests the exact "
                                            "M19 configuration on current parents and safe folds."),
            "parameters": {"family": "catboost", "model_config": CATBOOST_CONFIG,
                           "target": "log_sold_over_list", "screen_gain_min_percent": 0.25,
                           "screen_months_improved_required": 2,
                           "development_gain_min_percent": 0.5, "development_months_improved_min": 6,
                           "max_mape_regression_percent": 1.0,
                           "bootstrap_replicates": 2000, "bootstrap_seed": 20261100},
            "development_gate": dict(protocol["gates"]),
            "benchmark_path": str(paired_path.relative_to(campaign.ROOT)).replace("\\", "/"),
            "data_sha256": source["data_sha256"], "parent_commit": source["commit"],
            "source_sha256": own_hash, "reserved_months_labels_opened": False,
            "hidden_labels_opened": False, "registration_amendments": [],
        }
        protocol["cards"][CARD] = card
        campaign.PROTOCOL_PATH.write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + "\n",
                                          encoding="utf-8")
        campaign._append({"event": "card_registered", "card": CARD,
                          "data_sha256": source["data_sha256"], "source_sha256": own_hash,
                          "benchmark_prediction_sha256": manifest["paired_predictions_sha256"],
                          "reserved_months_labels_opened": False, "hidden_labels_opened": False})
    elif card.get("source_sha256") != own_hash:
        events = [json.loads(x) for x in campaign.LEDGER.read_text(encoding="utf-8").splitlines() if x.strip()]
        if any(e.get("event") == "fit_start" and e.get("card") == CARD for e in events):
            raise RuntimeError("R103 implementation changed after a learner fit started")
        amend = {"old_source_sha256": card.get("source_sha256"), "new_source_sha256": own_hash,
                 "reason": "zero-fit source correction", "fit_starts_before_amendment": 0}
        card.setdefault("registration_amendments", []).append(amend)
        card["source_sha256"] = own_hash
        campaign.PROTOCOL_PATH.write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + "\n",
                                          encoding="utf-8")
        campaign._append({"event": "card_source_amended", "card": CARD, **amend,
                          "reserved_months_labels_opened": False, "hidden_labels_opened": False})
    if card.get("status") not in {"registered", "screen_passed", "development_complete"}:
        raise RuntimeError(f"R103 cannot run from status {card.get('status')!r}")
    return card


def _benchmark(months):
    f = pd.read_csv(R94_DIR / "paired_predictions.csv", dtype={"ml_num": str})
    pieces = []
    for track in ("off_market", "list_aware"):
        x = f.loc[f.track.eq(track) & f.month.astype(str).isin(months),
                  ["month", "ml_num", "sold_price", "base_prediction", "candidate_prediction"]].copy()
        source_col = "candidate_prediction" if track == "off_market" else "base_prediction"
        x["benchmark_prediction"] = pd.to_numeric(x[source_col], errors="raise")
        x["track"] = track
        pieces.append(x[["month", "track", "ml_num", "sold_price", "benchmark_prediction"]])
    out = pd.concat(pieces, ignore_index=True)
    if out.empty or out.duplicated(["month", "track", "ml_num"]).any():
        raise RuntimeError("benchmark rows are empty or duplicated")
    out["month"] = out.month.astype(str)
    return out


def _feature_wrapper(fit_rows, eval_rows, track):
    from experiments.bill_target74.r24_catboost_word_tfidf import _install_catboost_word_features
    from model_candidate_v2 import features as frozen_features
    attr = "off_market_features" if track == "off_market" else "list_aware_features"
    transformer, restore_text = _install_catboost_word_features(fit_rows, eval_rows, track)
    with_text = getattr(frozen_features, attr)

    return transformer, restore_text, frozen_features, attr, with_text


def _fit(month, track, train, fit_rows, homes, component, run_id, stage, output, source, details):
    transformer, restore_text, module, attr, with_text = _feature_wrapper(fit_rows, homes, track)
    fit_id = f"{campaign.PROTOCOL['campaign_id']}:{CARD}:{track}:{month}:{component}:{run_id}"
    ids = homes.ml_num.astype(str).to_numpy()

    def record(_old_id, fit_fn):
        campaign._append({"event": "fit_start", "card": CARD, "stage": stage,
                          "run_id": run_id, "fit_id": fit_id, "track": track,
                          "month": month, "component": component,
                          "data_sha256": source["data_sha256"],
                          "reserved_months_labels_opened": False, "hidden_labels_opened": False})
        start = time.perf_counter()
        try:
            result = fit_fn()
        except Exception as exc:
            campaign._append({"event": "fit_complete", "card": CARD, "stage": stage,
                              "run_id": run_id, "fit_id": fit_id, "track": track,
                              "month": month, "component": component, "status": "failed",
                              "seconds": time.perf_counter() - start,
                              "error": f"{type(exc).__name__}: {exc}"})
            raise
        dest = output / "component_predictions" / track / month / f"{component}.csv"
        dest.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"ml_num": ids, "predicted_price": np.asarray(result["predictions"], dtype=float)}).to_csv(dest, index=False)
        detail = {"fit_id": fit_id, "track": track, "month": month, "component": component,
                  "seconds": time.perf_counter() - start, "training_rows": result["training_rows"],
                  "evaluation_rows": result["evaluation_rows"], "target": result["target"],
                  "model_family": result["model_family"], "model_config": result["model_config"],
                  "feature_columns": result["feature_columns"],
                  "text_feature_count": sum(c.startswith("remarks_tfidf_") for c in result["feature_columns"]),
                  "actual_objective": result.get("actual_objective"),
                  "prediction_file": str(dest.relative_to(output)).replace("\\", "/"),
                  "prediction_file_sha256": _sha(dest)}
        details.append(detail)
        campaign._append({"event": "fit_complete", "card": CARD, "stage": stage,
                          "run_id": run_id, "fit_id": fit_id, "track": track,
                          "month": month, "component": component, "status": "complete",
                          "seconds": detail["seconds"], "training_rows": detail["training_rows"],
                          "prediction_file_sha256": detail["prediction_file_sha256"], "target": detail["target"]})
        return result

    old_record = greedy._fit_record
    greedy._fit_record = record
    try:
        return greedy._fit_history_candidate(
            month, track, train, homes, variant=VARIANT,
            base_variant="G06_lot_garage" if track == "off_market" else None,
            fit_tag=f"R103-{component}-{month}", fit_rows=fit_rows)
    finally:
        greedy._fit_record = old_record
        setattr(module, attr, with_text)
        restore_text()


def _metrics(actual, pred):
    ape = np.abs(pred - actual) / actual * 100
    return {"n": int(len(actual)), "mdape_percent": float(np.median(ape)), "mape_percent": float(np.mean(ape))}


def _score(paired, track, months, stage):
    rows = paired.loc[paired.track.eq(track)]
    actual, benchmark, candidate = (rows[c].to_numpy(float) for c in
                                    ("sold_price", "benchmark_prediction", "candidate_prediction"))
    if not np.isfinite(np.column_stack([actual, benchmark, candidate])).all() or (
            np.column_stack([actual, benchmark, candidate]) <= 0).any():
        raise RuntimeError(f"invalid prices in {track}")
    bm, cm = _metrics(actual, benchmark), _metrics(actual, candidate)
    gain = 100 * (bm["mdape_percent"] - cm["mdape_percent"]) / bm["mdape_percent"]
    mape = 100 * (cm["mape_percent"] - bm["mape_percent"]) / bm["mape_percent"]
    monthly = {}; groups = []
    for month in months:
        part = rows.loc[rows.month.astype(str).eq(month)]
        b = _metrics(part.sold_price.to_numpy(float), part.benchmark_prediction.to_numpy(float))
        c = _metrics(part.sold_price.to_numpy(float), part.candidate_prediction.to_numpy(float))
        g = 100 * (b["mdape_percent"] - c["mdape_percent"]) / b["mdape_percent"]
        monthly[month] = {"benchmark": b, "candidate": c, "relative_mdape_improvement_percent": g}
        groups.append((part, g))
    improved = sum(g > 0 for _, g in groups)
    worst = max(0.0, max(-g for _, g in groups))
    result = {"n": len(rows), "benchmark": bm, "candidate": cm,
              "relative_mdape_improvement_percent": gain, "mape_relative_change_percent": mape,
              "months_improved": improved, "worst_month_regression_percent": worst, "monthly": monthly}
    gates = campaign.PROTOCOL["gates"]
    if stage == "screen":
        result["qualifies_screen"] = bool(gain >= gates["screen_relative_mdape_improvement_percent"]
                                          and improved == len(months)
                                          and mape <= gates["screen_mape_regression_max_percent"])
    else:
        rng = np.random.default_rng(20261100)
        diff = np.empty(2000)
        arrays = [(p.sold_price.to_numpy(float), p.benchmark_prediction.to_numpy(float),
                   p.candidate_prediction.to_numpy(float)) for p, _ in groups]
        for i in range(len(diff)):
            ix = [rng.integers(0, len(a), len(a)) for a, _, _ in arrays]
            ba = np.concatenate([np.abs(b[j]-a[j])/a[j]*100 for j,(a,b,_) in zip(ix,arrays)])
            ca = np.concatenate([np.abs(c[j]-a[j])/a[j]*100 for j,(a,_,c) in zip(ix,arrays)])
            diff[i] = np.median(ca) - np.median(ba)
        result["paired_bootstrap_mdape_difference_pp"] = {
            "lower95": float(np.quantile(diff,.025)), "upper95": float(np.quantile(diff,.975)),
            "replicates": len(diff), "seed": 20261100}
        result["qualifies_development"] = bool(
            gain >= gates["development_relative_mdape_improvement_percent"]
            and improved >= gates["development_months_improved_min"]
            and worst <= gates["development_worst_month_regression_max_percent"]
            and mape <= gates["development_mape_regression_max_percent"]
            and result["paired_bootstrap_mdape_difference_pp"]["upper95"] < 0)
    return result


def run(stage="screen"):
    card = _register()
    if stage == "screen" and card["status"] != "registered":
        raise RuntimeError("R103 screen already ran")
    if stage == "development" and card["status"] != "screen_passed":
        raise RuntimeError("R103 development requires at least one passed screen track")
    source = provenance._source_identity()
    months = SCREEN_MONTHS if stage == "screen" else DEVELOPMENT_MONTHS
    tracks = ["list_aware"] if stage == "screen" else card["screen_passed_tracks"]
    benchmark = _benchmark(months)
    run_id = f"r103m_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    output = campaign.CAMPAIGN_ROOT / "batches" / run_id
    output.mkdir(parents=True, exist_ok=False)
    expected = len(months) * (2 * ("off_market" in tracks) + ("list_aware" in tracks))
    campaign._append({"event": "card_start", "card": CARD, "stage": stage, "run_id": run_id,
                      "months": list(months), "tracks": tracks, "expected_fit_starts": expected,
                      "reserved_months_labels_opened": False, "hidden_labels_opened": False})
    from data import hide_answers
    old_hgbr = greedy._frozen_hgbr_predictions
    had_variant, old_variant = VARIANT in greedy.VARIANTS, greedy.VARIANTS.get(VARIANT)
    greedy.VARIANTS[VARIANT] = {"model_family": "catboost", "model_config": dict(CATBOOST_CONFIG)}
    greedy._frozen_hgbr_predictions = lambda month, track, train, homes: r03._verified_hgbr_cache(
        greedy, month, track, train, homes)
    details, folds = [], []
    try:
        for month in months:
            _, train, _, truth, heldout = greedy._load_fold(month)
            cutoff = pd.Period(month, freq="M").start_time
            dates = pd.to_datetime(train.date_sold, errors="raise")
            if (dates >= cutoff).any() or dates.dt.to_period("M").astype(str).isin(
                    campaign.PROTOCOL["reserved_months"]).any():
                raise RuntimeError("training data violate strict fold or reserve-month policy")
            for track in tracks:
                homes = hide_answers(heldout, track)
                b = benchmark.loc[benchmark.month.eq(month) & benchmark.track.eq(track)].copy()
                if len(b) != len(heldout) or set(b.ml_num) != set(heldout.ml_num.astype(str)):
                    raise RuntimeError(f"benchmark IDs mismatch for {track}/{month}")
                labels = b.set_index("ml_num").loc[heldout.ml_num.astype(str), "sold_price"].to_numpy(float)
                if not np.array_equal(labels, truth.sold_price.to_numpy(float)):
                    raise RuntimeError(f"benchmark labels mismatch for {track}/{month}")
                if track == "off_market":
                    base = _fit(month, track, train, train, homes, "base", run_id, stage, output, source, details)
                    route = greedy._specialist_mask(train, "Detached")
                    route_eval = greedy._specialist_mask(homes, "Detached")
                    expert = _fit(month, track, train, train.loc[route].copy(), homes.loc[route_eval].copy(),
                                  "detached_expert", run_id, stage, output, source, details)
                    candidate = greedy._blend_specialist_predictions(np.round(base["predictions"]),
                                                                       np.round(expert["predictions"]),
                                                                       homes, "Detached", .75)
                else:
                    core = _fit(month, track, train, train, homes, "core", run_id, stage, output, source, details)
                    frozen = greedy._frozen_hgbr_predictions(month, track, train, homes)
                    candidate = blend_prices(core["predictions"], frozen, challenger_weight=.75)
                b = b.set_index("ml_num").loc[heldout.ml_num.astype(str)]
                folds.append(pd.DataFrame({"month": month, "track": track,
                                           "ml_num": heldout.ml_num.astype(str).to_numpy(),
                                           "sold_price": truth.sold_price.to_numpy(float),
                                           "benchmark_prediction": b.benchmark_prediction.to_numpy(float),
                                           "candidate_prediction": np.asarray(candidate, dtype=float)}))
                campaign._append({"event": "screen_fold_complete" if stage == "screen" else "development_fold_complete",
                                  "card": CARD, "stage": stage, "run_id": run_id, "month": month,
                                  "track": track, "rows": len(heldout), "reserved_months_labels_opened": False,
                                  "hidden_labels_opened": False})
    except Exception as exc:
        failure = {"run_id": run_id, "card": CARD, "stage": stage,
                   "error": f"{type(exc).__name__}: {exc}", "fit_details": details,
                   "reserved_months_labels_opened": False, "hidden_labels_opened": False}
        (output / "failure_checkpoint.json").write_text(json.dumps(failure, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
        campaign._append({"event": "card_failed", "card": CARD, "stage": stage, "run_id": run_id,
                          "fit_starts": len(details), "error": failure["error"]})
        campaign._write_progress("r103_failed", next_action="inspect R103 failure checkpoint before retry",
                                 r103_failure=str((output/"failure_checkpoint.json").relative_to(campaign.ROOT)).replace("\\","/"))
        raise
    finally:
        greedy._frozen_hgbr_predictions = old_hgbr
        if had_variant:
            greedy.VARIANTS[VARIANT] = old_variant
        else:
            greedy.VARIANTS.pop(VARIANT, None)
    paired = pd.concat(folds, ignore_index=True)
    paired_path = output / "paired_predictions.csv"
    paired.to_csv(paired_path, index=False)
    scores = {track: _score(paired, track, months, stage) for track in tracks}
    passed = [t for t,s in scores.items() if s.get("qualifies_screen" if stage=="screen" else "qualifies_development")]
    result = {"campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id, "card": CARD,
              "stage": stage, "months": list(months), "tracks": tracks, "fit_starts": len(details),
              "scores": scores, "screen_passed_tracks": passed, "fit_details": details,
              "reserved_months_labels_opened": False, "hidden_labels_opened": False,
              "data_sha256": source["data_sha256"]}
    result_path = output / "results.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    paths = [Path(__file__), campaign.PROTOCOL_PATH, campaign.PACKAGE/"r22_remarks_tfidf.py",
             campaign.PACKAGE/"r03_catboost.py", campaign.PACKAGE/"r04_ablation.py", campaign.PACKAGE/"runner.py",
             campaign.ROOT/"experiments/bill_greedy/runner.py", campaign.ROOT/"experiments/bill_greedy/history.py",
             campaign.ROOT/"experiments/model_optimization_v2/learners.py",
             campaign.PACKAGE/"r24_catboost_word_tfidf.py",
             campaign.ROOT/"model_candidate_v2/features.py", R94_DIR/"development_results.json",
             R94_DIR/"paired_predictions.csv", R94_DIR/"run_manifest.json"]
    provdir=output/"provenance"; provdir.mkdir(exist_ok=True); hashes={}
    for p in paths:
        key=str(p.relative_to(campaign.ROOT)).replace("\\","/") if p.is_relative_to(campaign.ROOT) else "external/"+p.name
        hashes[key]=_sha(p); shutil.copy2(p,provdir/(p.name+".snapshot"))
    manifest={"campaign_id":campaign.PROTOCOL["campaign_id"],"run_id":run_id,"source_identity":source,
              "data_sha256":source["data_sha256"],"source_sha256":hashes,"fit_starts":len(details),
              "fit_details":details,"paired_predictions_sha256":_sha(paired_path),"result_sha256":_sha(result_path),
              "benchmark_prediction_sha256":_sha(R94_DIR/"paired_predictions.csv"),
              "reserved_months_labels_opened":False,"hidden_labels_opened":False}
    (output/"run_manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    report=["# R103 list-aware CatBoost ratio screen","","M19 replaces the R35 LightGBM core with CatBoost and preserves the 75/25 HGBR blend.",
            "Benchmarks: R94 off-market candidate; R35 list-aware parent.",""]
    for t,s in scores.items():
        report.extend([f"## {t}","",f"Benchmark MdAPE: {s['benchmark']['mdape_percent']:.6f}%",
                       f"Candidate MdAPE: {s['candidate']['mdape_percent']:.6f}%",
                       f"Relative MdAPE gain: {s['relative_mdape_improvement_percent']:+.4f}%",
                       f"MAPE relative change: {s['mape_relative_change_percent']:+.4f}%",
                       f"Months improved: {s['months_improved']}/{len(months)}",
                       f"Gate: {s.get('qualifies_screen',s.get('qualifies_development'))}",""])
        for m,x in s["monthly"].items(): report.append(f"- {m}: {x['benchmark']['mdape_percent']:.5f}% → {x['candidate']['mdape_percent']:.5f}%")
        report.append("")
    report.append("No reserve or hidden labels were opened.")
    (output/"research_report.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    card["last_run_id"]=run_id; card["last_result_path"]=str(result_path.relative_to(campaign.ROOT)).replace("\\","/")
    if stage=="screen": card["screen_passed_tracks"]=passed; card["status"]="screen_passed" if passed else "screen_failed"
    else: card["status"]="development_complete"
    campaign.PROTOCOL_PATH.write_text(json.dumps(campaign.PROTOCOL,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    campaign._append({"event":"card_complete","card":CARD,"stage":stage,"run_id":run_id,
                      "fit_starts":len(details),"result_path":str(result_path.relative_to(campaign.ROOT)).replace("\\","/"),
                      "qualifies_by_track":{t:s.get("qualifies_screen" if stage=="screen" else "qualifies_development") for t,s in scores.items()},
                      "reserved_months_labels_opened":False,"hidden_labels_opened":False})
    next_action=("run R103 full development for "+", ".join(passed) if stage=="screen" and passed else
                 "retain current parents; R103 failed screen and register a distinct method" if stage=="screen" else
                 "review R103 development gates")
    campaign._write_progress("r103_screen_complete" if stage=="screen" else "r103_development_complete",
                             next_action=next_action,r103_run_id=run_id,
                             r103_result=str(result_path.relative_to(campaign.ROOT)).replace("\\","/"),
                             latest_fit_count=len(details))
    return result


if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument("--development",action="store_true"); args=parser.parse_args()
    result=run("development" if args.development else "screen")
    print(json.dumps({"run_id":result["run_id"],"fit_starts":result["fit_starts"],
                      "screen_passed_tracks":result["screen_passed_tracks"],
                      "scores":{t:{"benchmark_mdape":s["benchmark"]["mdape_percent"],
                                  "candidate_mdape":s["candidate"]["mdape_percent"],
                                  "relative_gain_percent":s["relative_mdape_improvement_percent"],
                                  "mape_change_percent":s["mape_relative_change_percent"],
                                  "months_improved":s["months_improved"],
                                  "gate":s.get("qualifies_screen",s.get("qualifies_development"))}
                                for t,s in result["scores"].items()}},ensure_ascii=False),flush=True)
