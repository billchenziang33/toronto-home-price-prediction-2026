"""R29: test a log-sale-price target for list-aware R22 while retaining list price as input."""
from __future__ import annotations

import json
import shutil
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from experiments.bill_target74.r19_expert_weight import _sha256, _write_json


CARD = "R29_list_aware_log_sold_target"
VARIANT = CARD
TRACK = "list_aware"
SCREEN_MONTHS = ("2023-03", "2024-06")
VARIANT_SPEC = {"target_override": "log_sold_price"}


def _source_files(campaign) -> list:
    return [
        campaign.PROTOCOL_PATH,
        campaign.PACKAGE / "r29_list_log_price.py",
        campaign.PACKAGE / "r22_remarks_tfidf.py",
        campaign.PACKAGE / "r23_word_char_tfidf.py",
        campaign.PACKAGE / "r03_catboost.py",
        campaign.PACKAGE / "r04_ablation.py",
        campaign.PACKAGE / "runner.py",
        campaign.ROOT / "model_candidate_v2/features.py",
        campaign.ROOT / "experiments/bill_greedy/runner.py",
        campaign.ROOT / "experiments/bill_greedy/history.py",
        campaign.ROOT / "experiments/model_optimization_v2/learners.py",
    ]


def run_r29_screen() -> dict:
    from data import hide_answers
    from experiments.bill_target74 import r03_catboost as helpers
    from experiments.bill_target74 import r04_ablation as budget
    from experiments.bill_target74 import runner as campaign
    from experiments.bill_target74 import r22_remarks_tfidf as r22
    from experiments.bill_target74 import r23_word_char_tfidf as shared
    from experiments.bill_target74.r03_catboost import _verified_hgbr_cache
    from experiments.bill_greedy import runner as greedy
    from experiments.model_optimization_v2.pipeline import blend_prices

    if campaign.PROTOCOL["cards"].get(CARD, {}).get("status") != "registered":
        raise RuntimeError("R29 must be preregistered before fitting")
    parent_result, parent_path, parent_manifest = r22._verified_screen_result(campaign)
    if not parent_result["tracks"][TRACK]["qualifies_screen"]:
        raise RuntimeError("R29 requires the verified R22 list-aware screen to pass")
    parent_rows = pd.read_csv(parent_path.parent / "paired_predictions.csv", dtype={"ml_num": str})
    source = budget._source_identity()
    run_id = "bill_target74_r29_list_log_price_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = campaign.CAMPAIGN_ROOT / "batches" / run_id
    output.mkdir(parents=True, exist_ok=False)
    campaign._append({"event": "card_start", "card": CARD, "run_id": run_id,
                      "track": TRACK, "screen_months": list(SCREEN_MONTHS),
                      "parent_candidate": parent_result["run_id"],
                      "model_family": "lightgbm", "target_override": VARIANT_SPEC["target_override"],
                      "expected_fit_starts": len(SCREEN_MONTHS), "data_sha256": source["data_sha256"],
                      "parent_result_sha256": _sha256(parent_path),
                      "parent_predictions_sha256": parent_manifest["paired_predictions_sha256"],
                      "reserved_months_labels_opened": False, "hidden_labels_opened": False})
    fit_details, component_manifest, paired_rows = [], [], []
    old_fit_record, old_hgbr = greedy._fit_record, greedy._frozen_hgbr_predictions
    old_variant = greedy.VARIANTS.get(VARIANT)
    greedy.VARIANTS[VARIANT] = VARIANT_SPEC
    fit_context: dict[str, str] = {}
    fit_start_count = 0

    def record_fit(_fit_id_original: str, fit_fn):
        nonlocal fit_start_count
        if not fit_context:
            raise RuntimeError("R29 fit context is not initialized")
        month = fit_context["month"]
        fit_id = f"bill_target74:{CARD}:screen:{TRACK}:{month}:core:{run_id}"
        fit_start_count += 1
        campaign._append({"event": "fit_start", "card": CARD, "stage": "screen",
                          "run_id": run_id, "fit_id": fit_id, "track": TRACK,
                          "month": month, "component": "core",
                          "target_override": VARIANT_SPEC["target_override"],
                          "data_sha256": source["data_sha256"]})
        started = time.perf_counter()
        try:
            fitted = fit_fn()
        except Exception as exc:
            campaign._append({"event": "fit_complete", "card": CARD, "stage": "screen",
                              "run_id": run_id, "fit_id": fit_id, "track": TRACK,
                              "month": month, "component": "core", "status": "failed",
                              "seconds": time.perf_counter() - started,
                              "error": f"{type(exc).__name__}: {exc}"})
            raise
        detail = {"fit_id": fit_id, "track": TRACK, "month": month,
                  "component": "core", "seconds": time.perf_counter() - started,
                  "model_family": fitted["model_family"], "model_config": fitted["model_config"],
                  "target": fitted["target"], "actual_objective": fitted.get("actual_objective"),
                  "training_rows": fitted["training_rows"], "evaluation_rows": fitted["evaluation_rows"],
                  "tfidf_feature_count": sum(name.startswith("remarks_tfidf_")
                                              for name in fitted["feature_columns"]),
                  "prediction_sha256": fitted["prediction_sha256"]}
        fit_details.append(detail)
        campaign._append({"event": "fit_complete", "card": CARD, "stage": "screen",
                          "run_id": run_id, **detail, "status": "complete"})
        return fitted

    greedy._fit_record = record_fit
    greedy._frozen_hgbr_predictions = lambda month, track, train, homes: _verified_hgbr_cache(
        greedy, month, track, train, homes)
    try:
        for month in SCREEN_MONTHS:
            _, train, _, truth, heldout = greedy._load_fold(month)
            cutoff = pd.Period(month, freq="M").start_time
            if (pd.to_datetime(train.date_sold, errors="raise") >= cutoff).any():
                raise RuntimeError(f"R29 training rows include sales at or after cutoff {month}")
            homes = hide_answers(heldout, TRACK)
            reference = shared.r22_reference_frame(parent_rows, truth, month, TRACK)
            transformer, restore = r22._install_text_features(train, homes, TRACK)
            fit_context.update(month=month)
            try:
                core = greedy._fit_history_candidate(
                    month, TRACK, train, homes, variant=VARIANT,
                    fit_tag=f"R29-log-sold-{month}", fit_rows=train)
            finally:
                restore()
            if core["target"] != "log_sold_price":
                raise RuntimeError(f"R29 target override was not applied; found {core['target']!r}")
            frozen = greedy._frozen_hgbr_predictions(month, TRACK, train, homes)
            candidate_prediction = blend_prices(core["predictions"], frozen, challenger_weight=0.75)
            candidate = pd.DataFrame({"ml_num": homes.ml_num.astype(str).to_numpy(),
                                      "sold_price": truth.sold_price.to_numpy(float),
                                      "predicted_price": np.asarray(candidate_prediction, dtype=float)})
            parent = reference[["ml_num", "sold_price", "predicted_price"]].copy()
            paired_rows.append(helpers._candidate_frame(month, TRACK, truth, parent, candidate))
            component_manifest.append({
                "track": TRACK, "month": month, "component": "core",
                "model_family": core["model_family"], "model_config": core["model_config"],
                "target": core["target"], "actual_objective": core.get("actual_objective"),
                "training_rows": core["training_rows"], "evaluation_rows": core["evaluation_rows"],
                "tfidf_feature_count": len(transformer.columns_ or []),
                "vocabulary_terms": transformer.vectorizer.get_feature_names_out().tolist(),
                "prediction_sha256": core["prediction_sha256"],
            })
            campaign._append({"event": "screen_fold_complete", "card": CARD,
                              "run_id": run_id, "track": TRACK, "month": month,
                              "target": core["target"], "date_cutoff_exclusive": cutoff.isoformat(),
                              "reserved_months_labels_opened": False,
                              "hidden_labels_opened": False})
    except Exception as exc:
        failure = {"run_id": run_id, "card": CARD,
                   "error": f"{type(exc).__name__}: {exc}",
                   "fit_starts": fit_start_count, "fit_details": fit_details,
                   "completed_components": [f"{r['month']}/{r['component']}" for r in component_manifest],
                   "reserved_months_labels_opened": False, "hidden_labels_opened": False}
        _write_json(output / "failure_checkpoint.json", failure)
        campaign._append({"event": "card_failed", "card": CARD, "run_id": run_id,
                          "fit_starts": fit_start_count, "error": failure["error"]})
        campaign._write_progress("r29_screen_failed",
                                 next_action="audit R29 failure checkpoint and ledger before any retry",
                                 r29_failure=str((output / "failure_checkpoint.json").relative_to(
                                     campaign.ROOT)).replace("\\", "/"))
        raise
    finally:
        greedy._fit_record, greedy._frozen_hgbr_predictions = old_fit_record, old_hgbr
        if old_variant is None:
            greedy.VARIANTS.pop(VARIANT, None)
        else:
            greedy.VARIANTS[VARIANT] = old_variant

    if fit_start_count != len(SCREEN_MONTHS):
        raise RuntimeError(f"R29 expected 2 fits; observed {fit_start_count}")
    paired = pd.concat(paired_rows, ignore_index=True)
    paired_path = output / "paired_predictions.csv"
    paired.to_csv(paired_path, index=False)
    score = shared._track_result(paired)
    result = {"campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
              "card": CARD, "parent_candidate": parent_result["run_id"],
              "screen_months": list(SCREEN_MONTHS), "fit_starts": fit_start_count,
              "tracks": {TRACK: score}, "target_override": VARIANT_SPEC["target_override"],
              "fit_details": fit_details, "component_manifest": component_manifest,
              "reserved_months_labels_opened": False, "hidden_labels_opened": False,
              "data_sha256": source["data_sha256"]}
    result_path = output / "r29_screen_results.json"
    _write_json(result_path, result)
    provenance = output / "provenance"
    provenance.mkdir(exist_ok=True)
    shutil.copy2(campaign.PACKAGE / "r29_list_log_price.py", provenance / "r29_list_log_price_source.py")
    shutil.copy2(campaign.PROTOCOL_PATH, provenance / "r29_preregistered_campaign_protocol.json")
    manifest = {"campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
                "source_identity": source,
                "code_sha256": {str(path.relative_to(campaign.ROOT)).replace("\\", "/"): _sha256(path)
                                for path in _source_files(campaign)},
                "data_sha256": source["data_sha256"], "fit_starts": fit_start_count,
                "fit_details": fit_details, "component_manifest": component_manifest,
                "paired_predictions_sha256": _sha256(paired_path),
                "result_sha256": _sha256(result_path),
                "parent_run_id": parent_result["run_id"],
                "parent_result_sha256": _sha256(parent_path),
                "parent_predictions_sha256": parent_manifest["paired_predictions_sha256"],
                "reserved_months_labels_opened": False, "hidden_labels_opened": False}
    _write_json(output / "run_manifest.json", manifest)
    lines = ["# R29：list-aware log-sale-price 目标筛选", "",
             f"运行：`{run_id}`；父候选 R22：`{parent_result['run_id']}`。",
             "仅改变 LightGBM core target 为 log(sold_price)，保留 list_price 特征、R22 TF-IDF 和 75/25 HGBR 混合。", "",
             "| R22 MdAPE | R29 MdAPE | 相对改善 | MAPE变化 | 两月均改善 | 通过 |",
             "|---:|---:|---:|---:|---|---|"]
    both = all(score["monthly"][month]["relative_mdape_improvement_percent"] > 0.0
               for month in SCREEN_MONTHS)
    lines.append(f"| {score['parent']['mdape_percent']:.5f}% | "
                 f"{score['candidate']['mdape_percent']:.5f}% | "
                 f"{score['relative_mdape_improvement_percent']:+.3f}% | "
                 f"{score['mape_relative_change_percent']:+.3f}% | "
                 f"{'是' if both else '否'} | {'是' if score['qualifies_screen'] else '否'} |")
    (output / "research_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    campaign._append({"event": "card_complete", "card": CARD, "run_id": run_id,
                      "fit_starts": fit_start_count,
                      "result_path": str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
                      "qualifies_by_track": {TRACK: score["qualifies_screen"]},
                      "reserved_months_labels_opened": False, "hidden_labels_opened": False})
    campaign._write_progress("r29_screen_complete",
                             next_action=("register R29 eight-month development confirmation"
                                          if score["qualifies_screen"] else
                                          "R29 log-sale-price target failed screen; keep R22 and register a distinct method"),
                             r29_result=str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
                             latest_fit_count=fit_start_count)
    return result


def main() -> None:
    result = run_r29_screen()
    score = result["tracks"][TRACK]
    print(f"list_aware: R22={score['parent']['mdape_percent']:.5f}% "
          f"R29={score['candidate']['mdape_percent']:.5f}% "
          f"gain={score['relative_mdape_improvement_percent']:+.3f}% "
          f"screen_pass={score['qualifies_screen']}", flush=True)
    print(f"run_id={result['run_id']}", flush=True)


if __name__ == "__main__":
    main()
