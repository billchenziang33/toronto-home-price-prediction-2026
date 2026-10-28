"""Eight-month development confirmation for R65's off-market cold-start route."""
from __future__ import annotations

import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.bill_target74 import r03_catboost as helpers
from experiments.bill_target74 import r04_ablation as budget
from experiments.bill_target74 import r22_remarks_tfidf as r22
from experiments.bill_target74 import r58_past_log_residual_slope_calibration as r58
from experiments.bill_target74 import r65_cold_start_current_parents as r65
from experiments.bill_target74 import runner as campaign


ROOT = campaign.ROOT
CARD = "R66_R65_offmarket_eight_month_development_confirmation"
TRACK = "off_market"
PARENT = r65.PARENTS[TRACK]
MONTHS = tuple(r22.DEVELOPMENT_MONTHS)
EXPECTED_FIT_STARTS = len(MONTHS) * 2
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20261024


def _write_json(path: Path, value: dict | list) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
                    encoding="utf-8")


def _save_protocol() -> None:
    _write_json(campaign.PROTOCOL_PATH, campaign.PROTOCOL)


def _fit_id(run_id: str, month: str, component: str) -> str:
    if month not in MONTHS or component not in {"base", "expert"}:
        raise ValueError(f"invalid R66 fit coordinates: {month}/{component}")
    return (f"{campaign.PROTOCOL['campaign_id']}:{CARD}:development_confirmation:"
            f"{TRACK}:{month}:{component}:{run_id}")


def _registered_screen() -> tuple[dict, Path, Path, Path]:
    screen_card = campaign.PROTOCOL["cards"].get(r65.CARD)
    if not screen_card or screen_card.get("status") != "screen_passed":
        raise RuntimeError("R65 must pass its registered screen before R66 development confirmation")
    if screen_card.get("qualifies_by_track", {}).get(TRACK) is not True:
        raise RuntimeError("R65 off-market screen is not eligible for R66")
    result_path = ROOT / screen_card["result_path"]
    paired_path = ROOT / screen_card["paired_predictions_path"]
    manifest_path = ROOT / screen_card["manifest_path"]
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if result.get("run_id") != screen_card.get("run_id") or manifest.get("run_id") != result.get("run_id"):
        raise RuntimeError("R65 screen result, manifest, and protocol run IDs do not match")
    if (manifest.get("results_sha256") != helpers._sha256(result_path)
            or manifest.get("paired_predictions_sha256") != helpers._sha256(paired_path)):
        raise RuntimeError("R65 screen artifacts fail manifest hash verification")
    return result, result_path, paired_path, manifest_path


def _verified_parent_inputs() -> tuple[dict, dict]:
    source = r65._source_identity()
    inputs = r65._verified_inputs()
    item = inputs[TRACK]
    available = set(item[4].month.astype(str))
    missing = set(MONTHS) - available
    if missing:
        raise RuntimeError(f"verified R52 parent is missing development months: {sorted(missing)}")
    return source, inputs


def register_card() -> dict:
    existing = campaign.PROTOCOL["cards"].get(CARD)
    if existing:
        current_hash = helpers._sha256(Path(__file__))
        if existing.get("status") == "registered":
            if existing.get("source_sha256") == current_hash:
                return existing
            ledger_path = campaign.CAMPAIGN_ROOT / "fits.jsonl"
            events = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()
                      if line.strip()]
            started = any(event.get("card") == CARD and event.get("event") in {
                "card_start", "fit_start", "development_preflight", "card_complete"
            } for event in events)
            if started:
                raise RuntimeError("R66 source cannot be amended after development execution has started")
            old_hash = existing.get("source_sha256")
            existing["source_sha256"] = current_hash
            existing["source_amendment"] = {
                "previous_source_sha256": old_hash,
                "reason": "add missing shutil import after a zero-fit setup error; experiment method and gates unchanged",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
            _save_protocol()
            campaign._append({
                "event": "card_amended", "card": CARD, "stage": "development_confirmation",
                "previous_source_sha256": old_hash, "source_sha256": current_hash,
                "reason": existing["source_amendment"]["reason"],
                "fit_starts": 0, "reserved_months_labels_opened": False,
                "hidden_labels_opened": False,
            })
            return existing
        raise RuntimeError(f"R66 is already {existing.get('status')}; inspect before re-registering")

    screen, screen_path, screen_paired, screen_manifest = _registered_screen()
    source, inputs = _verified_parent_inputs()
    item = inputs[TRACK]
    parent_hashes = {
        "run_id": item[0]["run_id"],
        "result_sha256": helpers._sha256(item[1]),
        "paired_predictions_sha256": helpers._sha256(item[2]),
        "manifest_sha256": helpers._sha256(item[1].parent / "run_manifest.json"),
    }
    target = float(campaign.PROTOCOL["initial_development_metrics"][TRACK]
                   ["target_mdape_strictly_below"])
    gates = campaign.PROTOCOL["gates"]
    registration = {
        "status": "registered",
        "stage": "development_confirmation",
        "tracks": [TRACK],
        "months": list(MONTHS),
        "expected_fit_starts": EXPECTED_FIT_STARTS,
        "screen_run_id": screen["run_id"],
        "screen_result_sha256": helpers._sha256(screen_path),
        "screen_paired_predictions_sha256": helpers._sha256(screen_paired),
        "screen_manifest_sha256": helpers._sha256(screen_manifest),
        "parent": PARENT,
        "parent_artifacts": parent_hashes,
        "target_mdape_strictly_below": target,
        "development_gates": {
            "minimum_relative_mdape_improvement_percent": gates["development_relative_mdape_improvement_percent"],
            "minimum_months_improved": gates["development_months_improved_min"],
            "maximum_worst_month_regression_percent": gates["development_worst_month_regression_max_percent"],
            "maximum_mape_regression_percent": gates["development_mape_regression_max_percent"],
            "bootstrap_upper95_below_zero": gates["development_bootstrap_upper_bound_below_zero"],
        },
        "method": (
            "For each registered development fold, train R65's R22-text G06 base and Detached expert only on "
            "valid-coordinate rows with zero strictly-as-of building-coordinate history at each training row's "
            "own sale-month cutoff. Route only evaluation homes with zero prior building-coordinate sales into "
            "those models; blend Detached expert predictions at 75% and keep R52 predictions for every home "
            "outside the cold-start route."
        ),
        "secondary_comparator": "R22 fixed-origin-validated off-market parent predictions, descriptive only",
        "routing_rule": (
            "valid latitude/longitude and building_history_features b_n == 0 at each training-row sale-month "
            "start and at the evaluation-month cutoff"
        ),
        "source_sha256": helpers._sha256(Path(__file__)),
        "data_sha256": source["data_sha256"],
        "reserved_months_labels_opened": False,
        "hidden_labels_opened": False,
        "reserve_validation_eligible": False,
    }
    campaign.PROTOCOL["cards"][CARD] = registration
    _save_protocol()
    campaign._append({
        "event": "card_registered", "card": CARD, "stage": "development_confirmation",
        "screen_run_id": screen["run_id"], "tracks": [TRACK], "months": list(MONTHS),
        "expected_fit_starts": EXPECTED_FIT_STARTS, "parent": PARENT,
        "parent_artifacts": parent_hashes, "source_sha256": registration["source_sha256"],
        "data_sha256": source["data_sha256"], "reserved_months_labels_opened": False,
        "hidden_labels_opened": False,
    })
    campaign._write_progress(
        "r66_development_registered",
        next_action="preflight all eight registered development folds, then run R66's 16 off-market fits",
        r66_card=CARD,
    )
    return registration


def run_confirmation() -> dict:
    from data import hide_answers
    from experiments.bill_greedy import runner as greedy
    from experiments.model_optimization_v2.folds import build_month_fold

    card = campaign.PROTOCOL["cards"].get(CARD)
    if not card or card.get("status") != "registered":
        raise RuntimeError("R66 must be registered before fitting")
    if card.get("source_sha256") != helpers._sha256(Path(__file__)):
        raise RuntimeError("R66 source changed after registration")
    screen, screen_path, screen_paired, screen_manifest = _registered_screen()
    source, inputs = _verified_parent_inputs()
    item = inputs[TRACK]
    if (helpers._sha256(item[1]) != card["parent_artifacts"]["result_sha256"]
            or helpers._sha256(item[2]) != card["parent_artifacts"]["paired_predictions_sha256"]
            or helpers._sha256(item[1].parent / "run_manifest.json")
            != card["parent_artifacts"]["manifest_sha256"]):
        raise RuntimeError("R66 R52 parent artifacts changed after registration")
    if (helpers._sha256(screen_path) != card["screen_result_sha256"]
            or helpers._sha256(screen_paired) != card["screen_paired_predictions_sha256"]
            or helpers._sha256(screen_manifest) != card["screen_manifest_sha256"]):
        raise RuntimeError("R65 screen artifacts changed after R66 registration")

    started = datetime.now(timezone.utc)
    run_id = "bill_target74_r66_r65_offmarket_development_" + started.strftime("%Y%m%dT%H%M%SZ")
    output = campaign.CAMPAIGN_ROOT / "batches" / run_id
    output.mkdir(parents=True, exist_ok=False)
    (output / "provenance").mkdir()
    shutil.copy2(screen_path, output / "provenance" / "r65_screen_result.json")
    shutil.copy2(screen_paired, output / "provenance" / "r65_screen_paired_predictions.csv")
    shutil.copy2(screen_manifest, output / "provenance" / "r65_screen_manifest.json")
    shutil.copy2(item[1], output / "provenance" / "r52_parent_result.json")
    shutil.copy2(item[2], output / "provenance" / "r52_parent_paired_predictions.csv")

    code_paths = [
        campaign.PROTOCOL_PATH, Path(__file__), Path(r65.__file__), Path(r22.__file__),
        Path(r58.__file__), Path(budget.__file__), Path(helpers.__file__), Path(campaign.__file__),
        ROOT / "experiments/bill_greedy/runner.py", ROOT / "experiments/bill_greedy/history.py",
        ROOT / "experiments/model_optimization_v2/folds.py",
        ROOT / "experiments/model_optimization_v2/learners.py",
        ROOT / "model_candidate_v2/features.py", ROOT / "model_candidate_v2/winner_config.json",
    ]
    code_paths = list(dict.fromkeys(code_paths))
    code_hashes = {}
    for path in code_paths:
        if not path.is_file():
            raise FileNotFoundError(f"R66 provenance source is missing: {path}")
        code_hashes[str(path.relative_to(ROOT)).replace("\\", "/")] = helpers._sha256(path)

    campaign._append({
        "event": "card_start", "card": CARD, "stage": "development_confirmation",
        "run_id": run_id, "screen_run_id": screen["run_id"], "tracks": [TRACK],
        "months": list(MONTHS), "expected_fit_starts": EXPECTED_FIT_STARTS,
        "parent": PARENT, "data_sha256": source["data_sha256"],
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })

    fit_details: list[dict] = []
    component_manifest: list[dict] = []
    paired_frames: list[pd.DataFrame] = []
    fit_context: dict[str, object] = {}
    old_fit_record = greedy._fit_record

    def record_fit(_original_fit_id: str, fit_fn):
        if not fit_context:
            raise RuntimeError("R66 development fit context was not initialized")
        month = str(fit_context["month"])
        component = str(fit_context["component"])
        fit_id = _fit_id(run_id, month, component)
        fit_context_copy = {key: value for key, value in fit_context.items() if key != "evaluation_ids"}
        evaluation_ids = [str(value) for value in fit_context["evaluation_ids"]]
        campaign._append({
            "event": "fit_start", "card": CARD, "stage": "development_confirmation",
            "run_id": run_id, "fit_id": fit_id, "track": TRACK, "month": month,
            "component": component, "data_sha256": source["data_sha256"], **fit_context_copy,
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        fit_start = time.perf_counter()
        try:
            fitted = fit_fn()
        except Exception as exc:
            campaign._append({
                "event": "fit_complete", "card": CARD, "stage": "development_confirmation",
                "run_id": run_id, "fit_id": fit_id, "track": TRACK, "month": month,
                "component": component, "status": "failed", "seconds": time.perf_counter() - fit_start,
                "error": f"{type(exc).__name__}: {exc}",
                "reserved_months_labels_opened": False, "hidden_labels_opened": False,
            })
            raise
        predictions = np.asarray(fitted["predictions"], dtype=float)
        if predictions.shape != (len(evaluation_ids),) or not np.isfinite(predictions).all() \
                or (predictions <= 0).any():
            raise RuntimeError(f"R66 predictions are invalid or misaligned for {fit_id}")
        frame = pd.DataFrame({"ml_num": evaluation_ids, "predicted_price": predictions})
        if frame.ml_num.duplicated().any():
            raise RuntimeError(f"R66 predictions contain duplicate IDs for {fit_id}")
        path = output / f"component_{len(fit_details):02d}.csv"
        frame.to_csv(path, index=False)
        detail = {
            "fit_id": fit_id, "track": TRACK, **fit_context_copy,
            "seconds": time.perf_counter() - fit_start, "status": "complete",
            "model_family": fitted["model_family"], "model_config": fitted["model_config"],
            "target": fitted["target"], "training_rows": fitted["training_rows"],
            "evaluation_rows": fitted["evaluation_rows"],
            "feature_columns": fitted["feature_columns"],
            "prediction_sha256": fitted["prediction_sha256"],
            "saved_predictions": {"path": path.name, "rows": len(frame),
                                  "sha256": helpers._sha256(path)},
        }
        fit_details.append(detail)
        campaign._append({
            "event": "fit_complete", "card": CARD, "stage": "development_confirmation",
            "run_id": run_id, **detail, "reserved_months_labels_opened": False,
            "hidden_labels_opened": False,
        })
        return fitted

    greedy._fit_record = record_fit
    route_coverage: dict[str, dict] = {}
    try:
        # Preflight every registered month before starting any learner fit.
        for month in MONTHS:
            listings, _, _, _, _ = greedy._load_fold(month)
            cutoff = pd.Period(month, freq="M").start_time
            train, homes, truth = r65._prepare_track_fold(
                listings, month, TRACK, build_month_fold, hide_answers
            )
            if (pd.to_datetime(train.date_sold, errors="raise") >= cutoff).any():
                raise RuntimeError(f"R66 training data include sales at or after cutoff: {month}")
            ids = homes.ml_num.astype(str).tolist()
            parent_rows = r65._aligned_parent_rows(inputs, TRACK, month, ids, truth)
            train_mask = r65.training_cold_start_mask(train, TRACK)
            eval_mask = r65.evaluation_cold_start_mask(train, homes, track=TRACK, cutoff=cutoff)
            cold_train = train.loc[train_mask]
            cold_homes = homes.loc[eval_mask]
            cold_detached = cold_train.loc[greedy._specialist_mask(cold_train, "Detached")]
            cold_detached_homes = cold_homes.loc[greedy._specialist_mask(cold_homes, "Detached")]
            if cold_train.empty or cold_homes.empty or cold_detached.empty or cold_detached_homes.empty:
                raise RuntimeError(f"R66 cold-start or Detached route is empty for {month}")
            route_coverage[month] = {
                "training_rows": int(len(train)), "cold_training_rows": int(len(cold_train)),
                "evaluation_rows": int(len(homes)), "cold_rows": int(len(cold_homes)),
                "coverage_percent": 100.0 * len(cold_homes) / len(homes),
                "cold_detached_training_rows": int(len(cold_detached)),
                "cold_detached_evaluation_rows": int(len(cold_detached_homes)),
                "parent_rows_verified": int(len(parent_rows)),
                "date_cutoff_exclusive": cutoff.isoformat(),
            }
        campaign._append({
            "event": "development_preflight", "card": CARD, "stage": "development_confirmation",
            "run_id": run_id, "route_coverage": route_coverage, "fit_starts": 0,
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })

        for month in MONTHS:
            listings, _, _, _, _ = greedy._load_fold(month)
            cutoff = pd.Period(month, freq="M").start_time
            train, homes, truth = r65._prepare_track_fold(
                listings, month, TRACK, build_month_fold, hide_answers
            )
            ids = homes.ml_num.astype(str).tolist()
            parent_rows = r65._aligned_parent_rows(inputs, TRACK, month, ids, truth)
            parent = pd.DataFrame({
                "ml_num": ids, "sold_price": truth.sold_price.to_numpy(float),
                "predicted_price": parent_rows.candidate_prediction.to_numpy(float),
            })
            train_mask = r65.training_cold_start_mask(train, TRACK)
            eval_mask = r65.evaluation_cold_start_mask(train, homes, track=TRACK, cutoff=cutoff)
            cold_train = train.loc[train_mask].copy()
            cold_homes = homes.loc[eval_mask].copy()
            context_base = {
                "track": TRACK, "month": month, "training_rows": int(len(cold_train)),
                "date_cutoff_exclusive": cutoff.isoformat(),
            }

            fit_context.clear()
            fit_context.update({**context_base, "component": "base",
                                "evaluation_rows": int(len(cold_homes)),
                                "evaluation_ids": cold_homes.ml_num.astype(str).tolist()})
            transformer, restore = r22._install_text_features(cold_train, cold_homes, TRACK)
            try:
                base = greedy._fit_history_candidate(
                    month, TRACK, train, cold_homes, base_variant="G06_lot_garage",
                    variant=r22.VARIANT, fit_tag=f"R66-cold-base-{month}", fit_rows=cold_train,
                )
            finally:
                restore()
            component_manifest.append({
                "track": TRACK, "month": month, "component": "base",
                "model_family": base["model_family"], "model_config": base["model_config"],
                "target": base["target"], "training_rows": base["training_rows"],
                "evaluation_rows": base["evaluation_rows"],
                "text_feature_count": len(transformer.columns_ or []),
                "feature_columns": base["feature_columns"],
                "prediction_sha256": base["prediction_sha256"],
            })

            detached_train_mask = greedy._specialist_mask(cold_train, "Detached")
            detached_eval_mask = greedy._specialist_mask(cold_homes, "Detached")
            cold_detached = cold_train.loc[detached_train_mask].copy()
            detached_homes = cold_homes.loc[detached_eval_mask].copy()
            fit_context.clear()
            fit_context.update({**context_base, "component": "expert",
                                "training_rows": int(len(cold_detached)),
                                "evaluation_rows": int(len(detached_homes)),
                                "evaluation_ids": detached_homes.ml_num.astype(str).tolist()})
            transformer, restore = r22._install_text_features(cold_detached, detached_homes, TRACK)
            try:
                expert = greedy._fit_history_candidate(
                    month, TRACK, train, detached_homes, base_variant="G06_lot_garage",
                    variant=r22.VARIANT, fit_tag=f"R66-cold-expert-{month}", fit_rows=cold_detached,
                )
            finally:
                restore()
            component_manifest.append({
                "track": TRACK, "month": month, "component": "expert",
                "model_family": expert["model_family"], "model_config": expert["model_config"],
                "target": expert["target"], "training_rows": expert["training_rows"],
                "evaluation_rows": expert["evaluation_rows"],
                "text_feature_count": len(transformer.columns_ or []),
                "feature_columns": expert["feature_columns"],
                "prediction_sha256": expert["prediction_sha256"],
            })
            route_prediction = r65.off_market_cold_predictions(
                cold_homes, base_predictions=base["predictions"],
                detached_expert_predictions=expert["predictions"],
            )
            candidate_prediction = r65.replace_routed_predictions(
                parent.predicted_price.to_numpy(float), route_prediction, eval_mask
            )
            candidate = pd.DataFrame({
                "ml_num": ids, "sold_price": truth.sold_price.to_numpy(float),
                "predicted_price": candidate_prediction,
            })
            paired = helpers._candidate_frame(month, TRACK, truth, parent, candidate)
            paired["validated_R22_parent_prediction"] = parent_rows.parent_prediction.to_numpy(float)
            paired["cold_start_route"] = eval_mask
            paired_frames.append(paired)
            paired_partial = pd.concat(paired_frames, ignore_index=True)
            paired_partial.to_csv(output / "partial_paired_predictions.csv", index=False)
            campaign._append({
                "event": "development_fold_complete", "card": CARD,
                "stage": "development_confirmation", "run_id": run_id,
                "track": TRACK, "month": month, "parent_id": PARENT,
                "parent_result_sha256": card["parent_artifacts"]["result_sha256"],
                "date_cutoff_exclusive": cutoff.isoformat(),
                "cold_training_rows": int(len(cold_train)),
                "cold_evaluation_rows": int(len(cold_homes)),
                "reserved_months_labels_opened": False, "hidden_labels_opened": False,
            })
        if len(fit_details) != EXPECTED_FIT_STARTS:
            raise RuntimeError(f"R66 expected {EXPECTED_FIT_STARTS} fits, observed {len(fit_details)}")
    except Exception as exc:
        failure = {
            "run_id": run_id, "card": CARD, "stage": "development_confirmation",
            "error": f"{type(exc).__name__}: {exc}", "fit_starts": len(fit_details),
            "fit_details": fit_details, "completed_components": component_manifest,
            "route_coverage": route_coverage, "reserved_months_labels_opened": False,
            "hidden_labels_opened": False,
        }
        _write_json(output / "failure_checkpoint.json", failure)
        campaign._append({
            "event": "card_failed", "card": CARD, "stage": "development_confirmation",
            "run_id": run_id, "fit_starts": len(fit_details), "error": failure["error"],
            "failure_path": str((output / "failure_checkpoint.json").relative_to(ROOT)).replace("\\", "/"),
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        card.update({"status": "development_failed_runtime", "run_id": run_id,
                     "failure_reason": failure["error"],
                     "failure_path": str((output / "failure_checkpoint.json").relative_to(ROOT)).replace("\\", "/")})
        _save_protocol()
        campaign._write_progress(
            "r66_development_failed",
            next_action="inspect R66 failure checkpoint and fit ledger; preserve the unbounded budget",
            r66_failure=str((output / "failure_checkpoint.json").relative_to(ROOT)).replace("\\", "/"),
        )
        raise
    finally:
        greedy._fit_record = old_fit_record

    paired_all = pd.concat(paired_frames, ignore_index=True)
    paired_path = output / "paired_predictions.csv"
    paired_all.to_csv(paired_path, index=False)
    target = float(campaign.PROTOCOL["initial_development_metrics"][TRACK]
                   ["target_mdape_strictly_below"])
    gates = campaign.PROTOCOL["gates"]
    score = r22.score_development_confirmation(
        paired_all, months=MONTHS, target_mdape_strictly_below=target,
        bootstrap_replicates=BOOTSTRAP_REPLICATES, bootstrap_seed=BOOTSTRAP_SEED,
        minimum_relative_gain_percent=gates["development_relative_mdape_improvement_percent"],
        minimum_months_improved=gates["development_months_improved_min"],
        maximum_worst_month_regression_percent=gates["development_worst_month_regression_max_percent"],
        maximum_mape_regression_percent=gates["development_mape_regression_max_percent"],
    )
    secondary = r22.score_development_confirmation(
        paired_all.rename(columns={"parent_prediction": "R52_parent_prediction"})
        .assign(parent_prediction=paired_all.validated_R22_parent_prediction),
        months=MONTHS, target_mdape_strictly_below=target,
        bootstrap_replicates=BOOTSTRAP_REPLICATES, bootstrap_seed=BOOTSTRAP_SEED,
        minimum_relative_gain_percent=gates["development_relative_mdape_improvement_percent"],
        minimum_months_improved=gates["development_months_improved_min"],
        maximum_worst_month_regression_percent=gates["development_worst_month_regression_max_percent"],
        maximum_mape_regression_percent=gates["development_mape_regression_max_percent"],
    )
    result = {
        "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
        "card": CARD, "stage": "development_confirmation", "screen_run_id": screen["run_id"],
        "track": TRACK, "parent": PARENT, "development_months": list(MONTHS),
        "fit_starts": len(fit_details), "tracks": {TRACK: score},
        "secondary_comparison_vs_R22": secondary, "fit_details": fit_details,
        "component_manifest": component_manifest, "route_coverage": route_coverage,
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        "data_sha256": source["data_sha256"],
    }
    results_path = output / "r66_development_results.json"
    _write_json(results_path, result)
    _write_json(output / "component_manifest.json", component_manifest)
    manifest = {
        "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
        "stage": "development_confirmation", "source_identity": source,
        "screen_run_id": screen["run_id"],
        "screen_result_sha256": card["screen_result_sha256"],
        "screen_paired_predictions_sha256": card["screen_paired_predictions_sha256"],
        "parent": PARENT, "parent_artifacts": card["parent_artifacts"],
        "code_sha256": code_hashes, "data_sha256": source["data_sha256"],
        "fit_starts": len(fit_details), "fit_details": fit_details,
        "paired_predictions_sha256": helpers._sha256(paired_path),
        "result_sha256": helpers._sha256(results_path),
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    }
    _write_json(output / "run_manifest.json", manifest)
    report = [
        "# R66：R65 Off-market 冷启动路由八个月开发确认", "",
        f"Run：`{run_id}`；R65 screen：`{screen['run_id']}`；父版：`{PARENT}`。",
        f"仅使用 {len(MONTHS)} 个登记开发月，完成 {len(fit_details)} 个 learner fits。",
        "训练筛选为每行自身成交月开始时没有历史 building-coordinate 成交且坐标有效的房源。",
        "评估时只替换该冷启动人群的父版预测；其他房源保留 R52。未读取保留月或隐藏标签。", "",
        "## 八个月 pooled 结果", "",
        "| 比较父版 | 父 MdAPE | 候选 MdAPE | 相对改善 | MAPE 变化 | 改善月数 | Bootstrap upper95 | 门槛 | 目标 <7% |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for label, item_score, gate in (("R52 primary", score, score["qualifies_development"]),
                                    ("R22 validated reference", secondary, secondary["qualifies_development"])):
        report.append(
            f"| {label} | {item_score['parent']['mdape_percent']:.5f}% | "
            f"{item_score['candidate']['mdape_percent']:.5f}% | "
            f"{item_score['relative_mdape_improvement_percent']:+.3f}% | "
            f"{item_score['mape_relative_change_percent']:+.3f}% | "
            f"{item_score['months_improved']}/{len(MONTHS)} | "
            f"{item_score['bootstrap_mdape_difference_percent']['upper95']:+.5f} pp | "
            f"{'pass' if gate else 'fail'} | {'met' if item_score['target_met'] else 'not met'} |"
        )
    report.extend(["", "## Monthly results versus R52", "",
                   "| Month | Parent MdAPE | Candidate MdAPE | Relative improvement | Cold-route coverage |",
                   "|---|---:|---:|---:|---:|"])
    for month, item_score in score["monthly"].items():
        coverage = route_coverage[month]
        report.append(
            f"| {month} | {item_score['parent']['mdape_percent']:.5f}% | "
            f"{item_score['candidate']['mdape_percent']:.5f}% | "
            f"{item_score['relative_mdape_improvement_percent']:+.3f}% | "
            f"{coverage['cold_rows']}/{coverage['evaluation_rows']} ({coverage['coverage_percent']:.2f}%) |"
        )
    report.extend(["", "The R52 comparison controls promotion. The R22 row is descriptive. A development pass does not establish fixed-origin validation.",
                   "Artifacts: `paired_predictions.csv`, `component_manifest.json`, `run_manifest.json`, `r66_development_results.json`.", ""])
    (output / "research_report.md").write_text("\n".join(report), encoding="utf-8")

    status = "development_passed" if score["qualifies_development"] else "development_failed"
    card.update({
        "status": status, "run_id": run_id, "fit_starts": len(fit_details),
        "result_path": str(results_path.relative_to(ROOT)).replace("\\", "/"),
        "paired_predictions_path": str(paired_path.relative_to(ROOT)).replace("\\", "/"),
        "manifest_path": str((output / "run_manifest.json").relative_to(ROOT)).replace("\\", "/"),
        "qualifies_development": score["qualifies_development"],
        "target_met": score["target_met"],
        "pooled_mdape_percent": score["candidate"]["mdape_percent"],
    })
    _save_protocol()
    campaign._append({
        "event": "card_complete", "card": CARD, "stage": "development_confirmation",
        "run_id": run_id, "fit_starts": len(fit_details),
        "result_path": str(results_path.relative_to(ROOT)).replace("\\", "/"),
        "qualifies_by_track": {TRACK: score["qualifies_development"]},
        "targets_met_by_track": {TRACK: score["target_met"]},
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    next_action = (
        "register fixed-origin confirmation for R66 off-market; reserve months remain ineligible"
        if score["target_met"] and score["qualifies_development"]
        else "R66 did not meet both the development gates and <7% target; retain validated R22 and register a distinct hypothesis"
    )
    campaign._write_progress(
        "r66_development_complete", next_action=next_action,
        r66_development_result=str(results_path.relative_to(ROOT)).replace("\\", "/"),
        r66_fit_starts=len(fit_details),
    )
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Register or run the R65 off-market eight-month confirmation")
    parser.add_argument("--register", action="store_true")
    args = parser.parse_args()
    result = register_card() if args.register else run_confirmation()
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str), flush=True)


if __name__ == "__main__":
    main()
