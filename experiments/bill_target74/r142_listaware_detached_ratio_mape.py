"""R142: screen a Detached sale-to-list-ratio MAPE expert against validated R35."""
from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.bill_target74.r19_expert_weight import _sha256, _write_json


CARD = "R142_listaware_detached_ratio_mape"
VARIANT = CARD
PARENT_CARD = "R35_listaware_detached_r22_text_expert"
TRACK = "list_aware"
HOUSE_TYPE = "Detached"
SCREEN_MONTHS = ("2023-03", "2024-06")
SPECIALIST_WEIGHT = 0.75
TARGET_OVERRIDE = "sold_over_list_ratio"
MODEL_OVERRIDES = {TRACK: {"objective": "mape"}}


def _source_files(campaign) -> list[Path]:
    return [
        campaign.PROTOCOL_PATH,
        Path(__file__),
        campaign.PACKAGE / "r15_mape_objective.py",
        campaign.PACKAGE / "r29_list_log_price.py",
        campaign.PACKAGE / "r51_listaware_l1_detached.py",
        campaign.PACKAGE / "r118_listaware_detached_mape.py",
        campaign.PACKAGE / "r34_house_type_text_experts.py",
        campaign.PACKAGE / "r22_remarks_tfidf.py",
        campaign.PACKAGE / "r23_word_char_tfidf.py",
        campaign.PACKAGE / "r03_catboost.py",
        campaign.PACKAGE / "r04_ablation.py",
        campaign.PACKAGE / "runner.py",
        campaign.ROOT / "experiments/bill_greedy/runner.py",
        campaign.ROOT / "experiments/bill_greedy/history.py",
        campaign.ROOT / "model_candidate_v2/features.py",
        campaign.ROOT / "experiments/model_optimization_v2/learners.py",
        campaign.ROOT / "experiments/model_optimization_v2/pipeline.py",
        campaign.ROOT / "experiments/model_optimization_v2/folds.py",
        campaign.ROOT / "model_candidate_v2/winner_config.json",
        campaign.ROOT / "model_candidate_v2/models/frozen_list_aware.json",
    ]


def _parent_artifacts(campaign) -> tuple[dict, Path, Path, dict, pd.DataFrame]:
    parent_card = campaign.PROTOCOL["cards"].get(PARENT_CARD)
    if not parent_card or parent_card.get("status") != "fixed_origin_confirmation_passed":
        raise RuntimeError("R142 requires the validated R35 list-aware parent")
    result_path = campaign.ROOT / parent_card["result_path"]
    paired_path = result_path.parent / "paired_predictions.csv"
    manifest_path = result_path.parent / "run_manifest.json"
    if not all(path.is_file() for path in (result_path, paired_path, manifest_path)):
        raise FileNotFoundError("R35 screen result, predictions, or manifest are missing")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (result.get("card") != PARENT_CARD
            or result.get("campaign_id") != campaign.PROTOCOL["campaign_id"]
            or result.get("run_id") != manifest.get("run_id")
            or manifest.get("run_id") != parent_card.get("run_id")):
        raise RuntimeError("R35 screen artifacts do not identify the registered parent")
    if (manifest.get("result_sha256") != _sha256(result_path)
            or manifest.get("paired_predictions_sha256") != _sha256(paired_path)
            or manifest.get("source_identity", {}).get("data_sha256")
            != campaign.PROTOCOL["data_sha256"]):
        raise RuntimeError("R35 screen artifacts fail hash or data-snapshot verification")
    if result.get("reserved_months_labels_opened") or result.get("hidden_labels_opened"):
        raise RuntimeError("R35 parent records restricted-label access")
    if result.get("tracks", {}).get(TRACK, {}).get("qualifies_screen") is not True:
        raise RuntimeError("R35 did not pass its registered list-aware screen")
    rows = pd.read_csv(paired_path, dtype={"ml_num": str})
    required = {"month", "track", "ml_num", "sold_price", "candidate_prediction"}
    if missing := required - set(rows.columns):
        raise ValueError(f"R35 paired predictions lack columns: {sorted(missing)}")
    rows = rows.loc[rows.track.eq(TRACK) & rows.month.isin(SCREEN_MONTHS)].copy()
    if rows.duplicated(["month", "ml_num"]).any():
        raise RuntimeError("R35 screen rows contain duplicate IDs")
    if set(rows.month.unique()) != set(SCREEN_MONTHS):
        raise RuntimeError("R35 screen predictions do not cover both registered months")
    return result, result_path, paired_path, manifest, rows


def register_card() -> dict:
    from experiments.bill_target74 import r04_ablation as budget
    from experiments.bill_target74 import runner as campaign

    existing = campaign.PROTOCOL["cards"].get(CARD)
    if existing:
        if existing.get("status") == "registered":
            return existing
        raise RuntimeError(f"R142 is already {existing.get('status')}; inspect before rerunning")
    parent = campaign.PROTOCOL["cards"].get(PARENT_CARD)
    if not parent or parent.get("status") != "fixed_origin_confirmation_passed":
        raise RuntimeError("R142 requires the fixed-origin-validated R35 parent")
    source = budget._source_identity()
    _parent_artifacts(campaign)
    registration = {
        "status": "registered",
        "track": TRACK,
        "parent": "hash-verified R35 list-aware predictions on identical rows",
        "parent_run_id": parent["run_id"],
        "house_type": HOUSE_TYPE,
        "screen_months": list(SCREEN_MONTHS),
        "expected_fit_starts": len(SCREEN_MONTHS),
        "model_family": "lightgbm",
        "model_config_overrides": MODEL_OVERRIDES,
        "target_override": TARGET_OVERRIDE,
        "target": "sold_over_list_ratio; reconstruct price as list_price times the predicted positive ratio",
        "method": (
            "Keep the R35 Detached-only route, fold-local word TF-IDF, structured/history "
            "features, 75/25 LightGBM/frozen-HGBR blend, and every non-Detached R35 "
            "prediction unchanged. On the Detached LightGBM expert only, use the existing "
            "positive sold_price/list_price ratio target and change the objective to LightGBM MAPE. "
            "Apply the existing positive-price prediction validation and price-space blend."
        ),
        "difference_from_prior_work": (
            "R118 tested LightGBM MAPE on direct dollar targets within the same R35 Detached route "
            "and failed its screen (MdAPE 4.7274305% to 4.7746276%, MAPE +1.515%). R30 tested "
            "LightGBM Q50 on a log sale/list target against the older R22 route; R131 tested "
            "CatBoost Q50 on the current R35 route. R142 keeps the R118 route and mape objective, "
            "but trains on the positive sold_price/list_price ratio and converts that ratio back to "
            "dollars at inference. This is a distinct output parameterization, not an objective or "
            "parameter grid. No post-screen retuning is allowed."
        ),
        "incremental_screen_gate": campaign.PROTOCOL["gates"],
        "policy": (
            "Use only strict pre-month rows from the two registered safe development folds; "
            "fit TF-IDF on Detached training remarks only; verify positive sale/list ratio target, "
            "mape objective, positive ratio-derived dollar predictions, and frozen HGBR row alignment; retain R35 unless every "
            "screen gate passes. Never read "
            "2025-05/08 or hidden labels."
        ),
        "source_sha256": _sha256(Path(__file__)),
        "data_sha256": source["data_sha256"],
        "reserved_months_labels_opened": False,
        "reserve_validation_eligible": False,
        "hidden_labels_opened": False,
    }
    campaign.PROTOCOL["cards"][CARD] = registration
    campaign.PROTOCOL_PATH.write_text(
        json.dumps(campaign.PROTOCOL, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    campaign._append({
        "event": "card_registered", "card": CARD, "track": TRACK,
        "parent": PARENT_CARD, "parent_run_id": parent["run_id"],
        "screen_months": list(SCREEN_MONTHS), "expected_fit_starts": len(SCREEN_MONTHS),
        "model_config_overrides": MODEL_OVERRIDES,
        "data_sha256": source["data_sha256"],
        "source_sha256": registration["source_sha256"],
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    campaign._write_progress(
        "r142_registered",
        next_action="run the preregistered two-fold R142 Detached sale-to-list-ratio MAPE screen; keep R35 unless every gate passes",
        r142_card=CARD,
    )
    return registration


def run_screen() -> dict:
    from data import hide_answers
    from experiments.bill_target74 import r03_catboost as helpers
    from experiments.bill_target74 import r04_ablation as budget
    from experiments.bill_target74 import r22_remarks_tfidf as r22
    from experiments.bill_target74 import r23_word_char_tfidf as scoring
    from experiments.bill_target74 import runner as campaign
    from experiments.bill_greedy import runner as greedy
    from experiments.model_optimization_v2.pipeline import blend_prices

    card = campaign.PROTOCOL["cards"].get(CARD)
    if not card or card.get("status") != "registered":
        raise RuntimeError("R142 must be preregistered before fitting")
    source = budget._source_identity()
    if source["data_sha256"] != card["data_sha256"]:
        raise RuntimeError("R142 data snapshot changed after preregistration")
    parent_result, parent_path, parent_rows_path, parent_manifest, parent_rows = _parent_artifacts(campaign)
    if parent_result["run_id"] != card["parent_run_id"]:
        raise RuntimeError("R142 parent run differs from preregistration")

    run_id = "r142p_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = campaign.CAMPAIGN_ROOT / "batches" / run_id
    output.mkdir(parents=True, exist_ok=False)
    provenance = output / "provenance"
    provenance.mkdir()
    shutil.copy2(parent_path, provenance / "r35_screen_results.json")
    shutil.copy2(parent_rows_path, provenance / "r35_screen_paired_predictions.csv")
    shutil.copy2(parent_path.parent / "run_manifest.json", provenance / "r35_screen_run_manifest.json")

    plan = [(month, "detached_core_expert") for month in SCREEN_MONTHS]
    campaign._append({
        "event": "card_start", "card": CARD, "run_id": run_id, "track": TRACK,
        "parent_candidate": parent_result["run_id"], "screen_months": list(SCREEN_MONTHS),
        "model_family": "lightgbm", "model_config_overrides": MODEL_OVERRIDES,
        "expected_fit_starts": len(plan), "data_sha256": source["data_sha256"],
        "parent_result_sha256": _sha256(parent_path),
        "parent_predictions_sha256": _sha256(parent_rows_path),
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })

    old_fit_record = greedy._fit_record
    old_variant = greedy.VARIANTS.get(VARIANT)
    greedy.VARIANTS[VARIANT] = {
        "target_override": TARGET_OVERRIDE,
        "model_config_overrides": MODEL_OVERRIDES,
    }
    fit_details: list[dict] = []
    paired_parts: list[pd.DataFrame] = []
    route_details: dict[str, dict] = {}
    active: dict[str, object] = {}
    fit_start_count = 0

    def record_fit(_parent_fit_id: str, fit_fn):
        nonlocal fit_start_count
        month = str(active["month"])
        fit_id = f"{campaign.PROTOCOL['campaign_id']}:{CARD}:screen:{TRACK}:{month}:Detached:{run_id}"
        fit_start_count += 1
        campaign._append({
            "event": "fit_start", "card": CARD, "stage": "screen", "run_id": run_id,
            "fit_id": fit_id, "track": TRACK, "month": month,
            "component": "detached_core_expert", "training_rows": active["training_rows"],
            "evaluation_rows": active["evaluation_rows"], "data_sha256": source["data_sha256"],
            "target_override": TARGET_OVERRIDE, "target": "sold_over_list_ratio",
            "expected_objective": "mape",
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        started = time.perf_counter()
        try:
            fitted = fit_fn()
            if fitted.get("actual_objective") != "mape":
                raise RuntimeError(f"R142 expected mape; got {fitted.get('actual_objective')!r}")
            if fitted.get("target") != "sold_over_list_ratio":
                raise RuntimeError(f"R142 expected sold_over_list_ratio; got {fitted.get('target')!r}")
            if not np.isfinite(fitted["predictions"]).all() or (fitted["predictions"] <= 0).any():
                raise RuntimeError("R142 MAPE learner produced nonpositive or nonfinite prices")
        except Exception as exc:
            campaign._append({
                "event": "fit_complete", "card": CARD, "stage": "screen", "run_id": run_id,
                "fit_id": fit_id, "track": TRACK, "month": month, "status": "failed",
                "seconds": time.perf_counter() - started, "error": f"{type(exc).__name__}: {exc}",
            })
            raise
        detail = {
            "fit_id": fit_id, "track": TRACK, "month": month,
            "component": "detached_core_expert", "seconds": time.perf_counter() - started,
            "model_family": fitted["model_family"], "model_config": fitted["model_config"],
            "actual_objective": fitted["actual_objective"], "target": fitted["target"],
            "training_rows": fitted["training_rows"], "evaluation_rows": fitted["evaluation_rows"],
            "feature_columns": fitted["feature_columns"],
            "prediction_sha256": fitted["prediction_sha256"],
        }
        fit_details.append(detail)
        campaign._append({
            "event": "fit_complete", "card": CARD, "stage": "screen", "run_id": run_id,
            **detail, "status": "complete", "reserved_months_labels_opened": False,
            "hidden_labels_opened": False,
        })
        return fitted

    greedy._fit_record = record_fit
    try:
        for month in SCREEN_MONTHS:
            _, train, _, truth, heldout = greedy._load_fold(month)
            cutoff = pd.Period(month, freq="M").start_time
            if (pd.to_datetime(train.date_sold, errors="raise") >= cutoff).any():
                raise RuntimeError(f"R142 training rows are not strictly before {month}")
            if pd.to_datetime(train.date_sold, errors="raise").dt.to_period("M").astype(str).isin(
                    ["2025-05", "2025-08"]).any():
                raise RuntimeError("R142 training fold contains a restricted reserve month")
            homes = hide_answers(heldout, TRACK)
            route_mask = greedy._specialist_mask(homes, HOUSE_TYPE)
            fit_mask = greedy._specialist_mask(train, HOUSE_TYPE)
            fit_rows = train.loc[fit_mask].copy()
            eval_rows = homes.loc[route_mask].copy()
            if len(fit_rows) < 500 or int(route_mask.sum()) < 100:
                raise RuntimeError(f"R142 Detached route support is too small in {month}")

            saved = parent_rows.loc[parent_rows.month.eq(month)].copy()
            saved = saved.set_index(saved.ml_num.astype(str)).reindex(homes.ml_num.astype(str))
            if (saved.index.has_duplicates
                    or not np.array_equal(saved.sold_price.to_numpy(float), truth.sold_price.to_numpy(float))
                    or not np.isfinite(saved.candidate_prediction.to_numpy(float)).all()):
                raise RuntimeError(f"R35 parent predictions do not align for {month}")
            parent_prediction = saved.candidate_prediction.to_numpy(float)

            transformer, restore = r22._install_text_features(fit_rows, eval_rows, TRACK)
            active.update(month=month, training_rows=len(fit_rows), evaluation_rows=len(eval_rows))
            try:
                fitted = greedy._fit_history_candidate(
                    month, TRACK, train, eval_rows, fit_rows=fit_rows, variant=VARIANT,
                    base_variant=None, fit_tag=f"R142-{month}-Detached-MAPE",
                )
            finally:
                restore()
            frozen_hgbr = helpers._verified_hgbr_cache(greedy, month, TRACK, train, homes)
            specialist_prediction = blend_prices(
                fitted["predictions"], frozen_hgbr[route_mask], challenger_weight=SPECIALIST_WEIGHT,
            )
            prediction = greedy._overlay_specialist_predictions(
                parent_prediction, homes, specialist_prediction, HOUSE_TYPE,
            )
            if not np.array_equal(prediction[~route_mask], parent_prediction[~route_mask]):
                raise RuntimeError(f"R142 changed non-Detached parent rows in {month}")
            parent_frame = pd.DataFrame({
                "ml_num": homes.ml_num.astype(str).to_numpy(),
                "sold_price": truth.sold_price.to_numpy(float),
                "predicted_price": parent_prediction,
            })
            candidate_frame = pd.DataFrame({
                "ml_num": homes.ml_num.astype(str).to_numpy(),
                "sold_price": truth.sold_price.to_numpy(float),
                "predicted_price": prediction,
            })
            paired_parts.append(helpers._candidate_frame(month, TRACK, truth, parent_frame, candidate_frame))
            route_details[month] = {
                "training_rows": int(len(fit_rows)), "evaluation_rows": int(route_mask.sum()),
                "non_detached_predictions_unchanged": True,
                "parent_detached": campaign._metrics(truth.sold_price.to_numpy(float)[route_mask],
                                                       parent_prediction[route_mask]),
                "candidate_detached": campaign._metrics(truth.sold_price.to_numpy(float)[route_mask],
                                                          prediction[route_mask]),
                "tfidf_terms": transformer.vectorizer.get_feature_names_out().tolist(),
                "actual_objective": fitted["actual_objective"], "target": fitted["target"],
                "model_config": fitted["model_config"],
            }
            campaign._append({
                "event": "screen_fold_complete", "card": CARD, "run_id": run_id,
                "track": TRACK, "month": month, "training_cutoff_exclusive": cutoff.isoformat(),
                "routed_rows": int(route_mask.sum()), "routed_training_rows": int(len(fit_rows)),
                "reserved_months_labels_opened": False, "hidden_labels_opened": False,
            })
    except Exception as exc:
        failure_path = output / "failure_checkpoint.json"
        _write_json(failure_path, {
            "run_id": run_id, "card": CARD, "error": f"{type(exc).__name__}: {exc}",
            "fit_starts": fit_start_count,
            "fit_details": fit_details, "completed_months": list(route_details),
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        campaign._append({"event": "card_failed", "card": CARD, "run_id": run_id,
                          "fit_starts": len(fit_details), "error": f"{type(exc).__name__}: {exc}"})
        campaign._write_progress("r142_screen_failed",
                                 next_action="inspect R142 failure checkpoint and ledger before any retry",
                                 r142_failure=str(failure_path.relative_to(campaign.ROOT)).replace("\\", "/"))
        raise
    finally:
        greedy._fit_record = old_fit_record
        if old_variant is None:
            greedy.VARIANTS.pop(VARIANT, None)
        else:
            greedy.VARIANTS[VARIANT] = old_variant

    if fit_start_count != len(plan) or len(fit_details) != len(plan):
        raise RuntimeError(
            f"R142 expected {len(plan)} completed fits; starts={fit_start_count}, completed={len(fit_details)}"
        )
    paired = pd.concat(paired_parts, ignore_index=True)
    paired_path = output / "paired_predictions.csv"
    paired.to_csv(paired_path, index=False)
    score = scoring._track_result(paired)
    result = {
        "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id, "card": CARD,
        "parent_candidate": parent_result["run_id"], "track": TRACK,
        "screen_months": list(SCREEN_MONTHS), "fit_starts": len(fit_details),
        "tracks": {TRACK: score}, "fit_details": fit_details,
        "route_diagnostics": route_details, "model_config_overrides": MODEL_OVERRIDES,
        "target_override": TARGET_OVERRIDE, "target": "sold_over_list_ratio",
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        "reserve_validation_eligible": False, "data_sha256": source["data_sha256"],
    }
    result_path = output / "screen_results.json"
    _write_json(result_path, result)
    source_files = _source_files(campaign)
    code_hashes = {
        str(path.relative_to(campaign.ROOT)).replace("\\", "/"): _sha256(path)
        for path in source_files
    }
    for index, path in enumerate(source_files):
        shutil.copy2(path, provenance / f"src_{index:02d}_{path.name}.source")
    manifest = {
        "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
        "source_identity": source, "code_sha256": code_hashes,
        "data_sha256": source["data_sha256"], "fit_starts": len(fit_details),
        "fit_details": fit_details, "target_override": TARGET_OVERRIDE,
        "target": "sold_over_list_ratio", "model_config_overrides": MODEL_OVERRIDES,
        "result_sha256": _sha256(result_path),
        "paired_predictions_sha256": _sha256(paired_path),
        "parent_run_id": parent_result["run_id"],
        "parent_result_sha256": _sha256(parent_path),
        "parent_predictions_sha256": _sha256(parent_rows_path),
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    }
    _write_json(output / "run_manifest.json", manifest)
    report = [
        "# R142：list-aware Detached 售价挂牌比率 MAPE 专家", "",
        f"Run：`{run_id}`；父模型：已验证 R35 `{parent_result['run_id']}`。",
        "仅替换 R35 Detached 专家：以正值 `sold_price / list_price` 为目标，并使用 LightGBM `mape` objective；预测价格按预测比率乘挂牌价还原。保留 Detached 行特征、训练折内 TF-IDF、LightGBM 其余配置及 75/25 frozen-HGBR blend。所有非 Detached 房源逐行沿用 R35。", "",
        f"- R35 pooled MdAPE：{score['parent']['mdape_percent']:.5f}%",
        f"- R142 pooled MdAPE：{score['candidate']['mdape_percent']:.5f}%",
        f"- 相对改善：{score['relative_mdape_improvement_percent']:+.3f}%",
        f"- MAPE 相对变化：{score['mape_relative_change_percent']:+.3f}%",
        f"- 筛选通过：{'是' if score['qualifies_screen'] else '否'}", "",
        "## 分月结果", "", "| 月份 | R35 MdAPE | R142 MdAPE | 相对改善 |", "|---|---:|---:|---:|",
    ]
    for month, values in score["monthly"].items():
        report.append(f"| {month} | {values['parent']['mdape_percent']:.5f}% | "
                      f"{values['candidate']['mdape_percent']:.5f}% | "
                      f"{values['relative_mdape_improvement_percent']:+.3f}% |")
    report.extend(["", "该双月筛选仅决定是否进入一次八月开发确认；不使用预留月份或隐藏标签。", ""])
    (output / "research_report.md").write_text("\n".join(report), encoding="utf-8")

    card.update({
        "status": "screen_passed" if score["qualifies_screen"] else "screen_failed",
        "run_id": run_id, "fit_starts": len(fit_details),
        "result_path": str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        "paired_predictions_path": str(paired_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        "manifest_path": str((output / "run_manifest.json").relative_to(campaign.ROOT)).replace("\\", "/"),
        "screen_metrics": score, "reserved_months_labels_opened": False,
        "hidden_labels_opened": False,
    })
    campaign.PROTOCOL_PATH.write_text(
        json.dumps(campaign.PROTOCOL, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    campaign._append({
        "event": "card_complete", "card": CARD, "run_id": run_id,
        "fit_starts": len(fit_details), "result_path": card["result_path"],
        "qualifies_screen": score["qualifies_screen"], "screen_metrics": score,
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    next_action = (
        "R142 passed its fixed screen; run one preregistered eight-month development confirmation against R35"
        if score["qualifies_screen"] else
        "R142 failed its fixed screen; keep R35 and register a distinct hypothesis"
    )
    campaign._write_progress(
        "r142_screen_passed" if score["qualifies_screen"] else "r142_screen_failed",
        next_action=next_action,
        r142_run_id=run_id, r142_result=card["result_path"],
        r142_fit_starts=len(fit_details),
    )
    return result


def run() -> dict:
    register_card()
    return run_screen()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--register-only", action="store_true",
                        help="register the experiment card without starting learner fits")
    parser.add_argument("--screen", action="store_true",
                        help="run the preregistered fixed two-fold screen")
    args = parser.parse_args()
    if args.register_only and args.screen:
        parser.error("--register-only and --screen cannot be combined")
    if args.register_only:
        result = register_card()
    elif args.screen:
        register_card()
        result = run_screen()
    else:
        result = run()
    print(json.dumps(result, indent=2, ensure_ascii=False))
