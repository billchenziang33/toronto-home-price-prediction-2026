"""Screen LightGBM Extra-Trees split randomization on the R52 Freehold/Semi-Detached route."""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


CARD = "R74_offmarket_freehold_attached_extratrees"
VARIANT = CARD
MODEL_OVERRIDES = {"off_market": {"extra_trees": True}}
TRACK = "off_market"
HOUSE_TYPES = ("Semi-Detached", "Freehold Townhouse")
COMPONENT = "freehold_attached_base_expert"
SCREEN_MONTHS = ("2023-03", "2024-06")
SPECIALIST_WEIGHT = 0.75
MIN_TRAINING_ROWS = 500
MIN_EVALUATION_ROWS = 100
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20261027


def _route_mask(frame: pd.DataFrame) -> np.ndarray:
    if "house_type_name" not in frame:
        raise ValueError("freehold-attached specialist routing requires house_type_name")
    return (frame["house_type_name"].astype("string")
            .isin(HOUSE_TYPES).fillna(False).to_numpy(dtype=bool))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
                    encoding="utf-8")


def register_card() -> dict:
    from experiments.bill_target74 import r04_ablation as budget
    from experiments.bill_target74 import runner as campaign

    existing = campaign.PROTOCOL["cards"].get(CARD)
    if existing:
        if existing.get("status") == "registered" and existing.get("source_sha256") == _sha256(Path(__file__)):
            return existing
        failure = None
        failure_path = None
        if existing.get("status") == "failed" and existing.get("failure_path"):
            failure_path = campaign.ROOT / existing["failure_path"]
            if failure_path.is_file():
                failure = json.loads(failure_path.read_text(encoding="utf-8"))
        recoverable = bool(
            failure
            and "FileNotFoundError" in failure.get("error", "")
            and "component_predictions" in failure.get("error", "")
        )
        if not recoverable:
            raise RuntimeError(
                f"R74 already exists with status {existing.get('status')}; inspect before rerunning"
            )
    parent = _verified_r52_parent(campaign)
    source = budget._source_identity()
    route_support = {
        "2023-03": {"training_rows": 9190, "evaluation_rows": 282},
        "2024-06": {"training_rows": 13248, "evaluation_rows": 289},
    }
    registration = {
        "status": "registered",
        "track": TRACK,
        "tracks": [TRACK],
        "screen_months": list(SCREEN_MONTHS),
        "expected_fit_starts": len(SCREEN_MONTHS),
        "parent": "R52_r48_r50_strict_past_meta_blend",
        "parent_run_id": parent["result"]["run_id"],
        "house_types": list(HOUSE_TYPES),
        "route_support": route_support,
        "specialist_weight": SPECIALIST_WEIGHT,
        "model_family": "lightgbm",
        "model_config_overrides": MODEL_OVERRIDES,
        "minimum_training_rows": MIN_TRAINING_ROWS,
        "minimum_evaluation_rows_per_month": MIN_EVALUATION_ROWS,
        "method": (
            "Keep R52 off-market predictions outside Semi-Detached and Freehold Townhouse unchanged. On the "
            "combined two-category route, fit the same G06 LightGBM expert with R22 fold-local word TF-IDF, "
            "unchanged structured/history features, target, training cutoff, and 75/25 expert/R52 blend. "
            "Change exactly one learner setting: enable LightGBM extra_trees=true for randomized split thresholds."
        ),
        "difference_from_prior_work": (
            "R71 screened this exact expert route with standard GBDT and narrowly missed the fixed gate. "
            "R56 tested LightGBM RF mode and R57 tested GOSS, but neither used GBDT with extra_trees=true. "
            "The older ExtraTrees model-family experiment is a different learner. This card tests only "
            "LightGBM's documented randomized-threshold option, with no other model or route changes."
        ),
        "incremental_screen_gate": {
            "relative_mdape_gain_min_percent": 0.25,
            "both_screen_months_improve": True,
            "mape_regression_max_percent": 1.0,
            "paired_home_bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "paired_home_bootstrap_seed": BOOTSTRAP_SEED,
            "paired_home_bootstrap_upper95_below_zero": True,
        },
        "policy": (
            "Run only 2023-03 and 2024-06; fit TF-IDF on routed training remarks only; train strictly before "
            "each cutoff; verify exact route support, R52 hashes, unchanged non-routed predictions, "
            "regression objective, and effective extra_trees=true. Pass only if pooled MdAPE improves by "
            "at least 0.25%, both months improve, relative MAPE regression is at most 1%, and paired-home "
            "bootstrap upper 95% is below zero. Otherwise retain current parents; do not run confirmation or "
            "retune on these screen labels. Never read 2025-05/08 or hidden labels."
        ),
        "source_sha256": _sha256(Path(__file__)),
        "data_sha256": source["data_sha256"],
        "reserved_months_labels_opened": False,
        "reserve_validation_eligible": False,
        "hidden_labels_opened": False,
    }
    if existing:
        registration["recovery_history"] = list(existing.get("recovery_history", [])) + [{
            "prior_run_id": failure.get("run_id"),
            "prior_failure_path": str(failure_path.relative_to(campaign.ROOT)).replace("\\", "/"),
            "prior_fit_starts": int(failure.get("fit_starts", 0)),
            "prior_fit_attempts_in_ledger": 1,
            "reason": "The model fit returned, but prediction CSV persistence failed because the old component filename made the Windows path 268 characters. The run produced no scored candidate; shortening the component filename keeps the model hypothesis, folds, weight, and gate unchanged.",
            "new_component_name": COMPONENT,
        }]
    campaign.PROTOCOL["cards"][CARD] = registration
    _write_json(campaign.PROTOCOL_PATH, campaign.PROTOCOL)
    if existing:
        campaign._append({
            "event": "fit_attempt_reconciled", "card": CARD,
            "run_id": failure.get("run_id"), "fit_starts": 1,
            "result": "estimator returned; prediction artifact persistence failed before scoring",
            "error": failure.get("error"), "reserved_months_labels_opened": False,
            "hidden_labels_opened": False,
        })
    campaign._append({
        "event": "card_recovery_registered" if existing else "card_registered", "card": CARD, "track": TRACK,
        "parent_run_id": parent["result"]["run_id"], "house_types": list(HOUSE_TYPES),
        "screen_months": list(SCREEN_MONTHS), "route_support": route_support,
        "expected_fit_starts": len(SCREEN_MONTHS),
        "source_sha256": registration["source_sha256"], "data_sha256": source["data_sha256"],
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    campaign._write_progress(
        "r74_registered",
        next_action="run the preregistered two-month Freehold/Semi-Detached expert screen against R52",
        r74_card=CARD,
    )
    return registration


def _verified_r52_parent(campaign):
    state = json.loads(campaign.STATE.read_text(encoding="utf-8"))
    if state.get("campaign_id") != campaign.PROTOCOL["campaign_id"]:
        raise RuntimeError("active progress file belongs to a different campaign")
    run_id = state.get("r52_run_id")
    result_rel = state.get("r52_result")
    if not run_id or not result_rel:
        raise RuntimeError("active progress file does not identify the saved R52 result")
    result_path = (campaign.ROOT / result_rel).resolve()
    if campaign.ROOT.resolve() not in result_path.parents:
        raise RuntimeError("R52 result path escapes the project root")
    rows_path = result_path.parent / "paired_predictions.csv"
    manifest_path = result_path.parent / "run_manifest.json"
    if not all(path.is_file() for path in (result_path, rows_path, manifest_path)):
        raise FileNotFoundError("one or more saved R52 parent artifacts are missing")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("run_id") != run_id
            or manifest.get("result_sha256") != _sha256(result_path)
            or manifest.get("paired_predictions_sha256") != _sha256(rows_path)):
        raise RuntimeError("R52 result or paired-prediction hashes do not match its manifest")
    if (result.get("run_id") != run_id
            or result.get("card") != "R52_r48_r50_strict_past_meta_blend"
            or result.get("track") != TRACK
            or result.get("tracks", {}).get(TRACK, {}).get("target_met") is not False
            or result.get("data_sha256") != campaign.PROTOCOL["data_sha256"]
            or manifest.get("data_sha256") != campaign.PROTOCOL["data_sha256"]
            or manifest.get("source_identity", {}).get("data_sha256") != campaign.PROTOCOL["data_sha256"]):
        raise RuntimeError("R52 is not the expected best experimental off-market parent")
    if set(result.get("development_months", [])) != set(campaign.PROTOCOL["development_months"]):
        raise RuntimeError("R52 result does not cover the registered eight-month development window")
    if (result.get("reserved_months_labels_opened") is not False
            or result.get("hidden_labels_opened") is not False
            or result.get("reserve_validation_eligible") is not False
            or manifest.get("reserved_months_labels_opened") is not False
            or manifest.get("hidden_labels_opened") is not False):
        raise RuntimeError("R52 parent artifacts report reserve or hidden-label access")
    rows = pd.read_csv(rows_path, dtype={"ml_num": str})
    required = {"month", "track", "ml_num", "sold_price", "candidate_prediction"}
    if not required.issubset(rows.columns):
        raise RuntimeError("R52 paired predictions are missing required columns")
    rows = rows.loc[rows.track.eq(TRACK) & rows.month.astype(str).isin(SCREEN_MONTHS)].copy()
    if rows.ml_num.astype(str).duplicated().any():
        raise RuntimeError("R52 contains duplicate home IDs in the registered screen months")
    expected_counts = {
        month: int(result["tracks"][TRACK]["monthly"][month]["n"])
        for month in SCREEN_MONTHS
    }
    actual_counts = rows.groupby(rows.month.astype(str)).size().to_dict()
    if actual_counts != expected_counts:
        raise RuntimeError(f"R52 screen-month rows differ from its result summary: {actual_counts}")
    values = rows[["sold_price", "candidate_prediction"]].to_numpy(float)
    if not np.isfinite(values).all() or (values <= 0.0).any():
        raise RuntimeError("R52 screen-month paired predictions contain invalid values")
    return {
        "result_path": result_path, "rows_path": rows_path,
        "manifest_path": manifest_path, "result": result,
        "rows": rows, "manifest": manifest,
    }


def _source_files(campaign) -> list[Path]:
    return [
        campaign.PROTOCOL_PATH,
        Path(__file__),
        campaign.PACKAGE / "r71_offmarket_freehold_attached_text_expert.py",
        campaign.PACKAGE / "r34_house_type_text_experts.py",
        campaign.PACKAGE / "r52_r48_r50_adaptive_meta_blend.py",
        campaign.PACKAGE / "r48_groupwise_past_oof_r22_r32_blend.py",
        campaign.PACKAGE / "r50_type_price_interaction_past_oof_blend.py",
        campaign.PACKAGE / "r22_remarks_tfidf.py",
        campaign.PACKAGE / "r23_word_char_tfidf.py",
        campaign.PACKAGE / "r04_ablation.py",
        campaign.PACKAGE / "runner.py",
        campaign.ROOT / "experiments/bill_greedy/runner.py",
        campaign.ROOT / "experiments/bill_greedy/history.py",
        campaign.ROOT / "model_candidate_v2/features.py",
        campaign.ROOT / "experiments/model_optimization_v2/learners.py",
        campaign.ROOT / "experiments/model_optimization_v2/pipeline.py",
        campaign.ROOT / "experiments/model_optimization_v2/folds.py",
    ]


def run() -> dict:
    from data import hide_answers
    from experiments.bill_target74 import r04_ablation as budget
    from experiments.bill_target74 import r22_remarks_tfidf as r22
    from experiments.bill_target74 import r23_word_char_tfidf as scoring
    from experiments.bill_target74 import runner as campaign
    from experiments.bill_target74.r19_expert_weight import _write_json as write_json
    from experiments.bill_greedy import runner as greedy
    from experiments.model_optimization_v2.pipeline import blend_prices

    card = campaign.PROTOCOL["cards"].get(CARD)
    if not card or card.get("status") != "registered":
        raise RuntimeError("R74 must be preregistered before fitting")
    if card.get("source_sha256") != _sha256(Path(__file__)):
        raise RuntimeError("R74 source changed after preregistration")
    parent = _verified_r52_parent(campaign)
    if card.get("parent_run_id") != parent["result"]["run_id"]:
        raise RuntimeError("R74 parent run differs from the preregistered R52 development run")
    source = budget._source_identity()
    if source.get("data_sha256") != campaign.PROTOCOL["data_sha256"]:
        raise RuntimeError("active data hash differs from the preregistered campaign")

    # A compact run ID keeps component CSV paths well below Windows MAX_PATH.
    run_id = "r74e_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = campaign.CAMPAIGN_ROOT / "batches" / run_id
    output.mkdir(parents=True, exist_ok=False)
    provenance = output / "provenance"
    provenance.mkdir()
    for source_path, name in [
        (parent["result_path"], "r52_development_results.json"),
        (parent["rows_path"], "r52_paired_predictions.csv"),
        (parent["manifest_path"], "r52_run_manifest.json"),
    ]:
        shutil.copy2(source_path, provenance / name)

    code_hashes = {}
    snapshots = {}
    for index, path in enumerate(_source_files(campaign)):
        key = str(path.relative_to(campaign.ROOT)).replace("\\", "/")
        code_hashes[key] = _sha256(path)
        name = f"src_{index:02d}{path.suffix}.source"
        shutil.copy2(path, provenance / name)
        snapshots[key] = name

    campaign._append({
        "event": "card_start", "card": CARD, "stage": "screen",
        "run_id": run_id, "track": TRACK,
        "parent_run_id": parent["result"]["run_id"],
        "house_types": list(HOUSE_TYPES), "screen_months": list(SCREEN_MONTHS),
        "expected_fit_starts": len(SCREEN_MONTHS),
        "specialist_weight": SPECIALIST_WEIGHT,
        "model_config_overrides": MODEL_OVERRIDES,
        "reserved_months_labels_opened": False,
        "reserve_validation_eligible": False, "hidden_labels_opened": False,
    })

    fit_details: list[dict] = []
    fit_attempts = 0
    paired_parts: list[pd.DataFrame] = []
    route_details: dict[str, dict] = {}
    fit_context: dict[str, object] = {}
    old_fit_record = greedy._fit_record
    old_variant = greedy.VARIANTS.get(VARIANT)
    greedy.VARIANTS[VARIANT] = {"model_config_overrides": MODEL_OVERRIDES}

    def record_fit(_original_id: str, fit_fn):
        nonlocal fit_attempts
        fit_attempts += 1
        month = str(fit_context["month"])
        fit_id = (f"{campaign.PROTOCOL['campaign_id']}:{CARD}:{TRACK}:{month}:"
                  f"{COMPONENT}:{run_id}")
        campaign._append({
            "event": "fit_start", "card": CARD, "run_id": run_id,
            "fit_id": fit_id, "track": TRACK, "month": month,
            "component": COMPONENT, "house_types": list(HOUSE_TYPES),
            "training_rows": fit_context["training_rows"],
            "evaluation_rows": fit_context["evaluation_rows"],
            "model_config_overrides": MODEL_OVERRIDES,
            "data_sha256": source["data_sha256"],
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        started = time.perf_counter()
        try:
            fitted = fit_fn()
            estimator = fitted.get("estimator")
            if estimator is None:
                raise RuntimeError("R74 fit result did not expose its fitted estimator")
            params = estimator.get_params(deep=False)
            booster_params = getattr(getattr(estimator, "booster_", None), "params", {})
            effective_extra_trees = params.get("extra_trees")
            if effective_extra_trees is None:
                effective_extra_trees = booster_params.get("extra_trees")
            if effective_extra_trees is None:
                effective_extra_trees = fitted.get("model_config", {}).get("extra_trees")
            if effective_extra_trees not in (True, 1, "true", "True"):
                raise RuntimeError(f"R74 expected effective extra_trees=true; got {effective_extra_trees!r}")
            if fitted.get("model_family") != "lightgbm" or fitted.get("actual_objective") != "regression":
                raise RuntimeError("R74 expected the unchanged LightGBM regression learner")
        except Exception as exc:
            campaign._append({
                "event": "fit_complete", "card": CARD, "run_id": run_id,
                "fit_id": fit_id, "track": TRACK, "month": month,
                "component": COMPONENT, "status": "failed",
                "seconds": time.perf_counter() - started,
                "error": f"{type(exc).__name__}: {exc}",
            })
            raise
        try:
            saved = r22._save_component_predictions(
                output, TRACK, month, COMPONENT,
                fit_context["evaluation_ids"], fitted["predictions"],
            )
        except Exception as exc:
            campaign._append({
                "event": "fit_complete", "card": CARD, "run_id": run_id,
                "fit_id": fit_id, "track": TRACK, "month": month,
                "component": COMPONENT, "status": "failed",
                "stage": "prediction_artifact_persistence",
                "seconds": time.perf_counter() - started,
                "error": f"{type(exc).__name__}: {exc}",
            })
            raise
        detail = {
            "fit_id": fit_id, "track": TRACK, "month": month,
            "component": COMPONENT, "seconds": time.perf_counter() - started,
            "training_rows": fitted["training_rows"],
            "evaluation_rows": fitted["evaluation_rows"],
            "model_family": fitted["model_family"], "model_config": fitted["model_config"],
            "actual_objective": fitted["actual_objective"],
            "effective_extra_trees": effective_extra_trees,
            "target": fitted["target"], "feature_columns": fitted["feature_columns"],
            "text_feature_count": sum(name.startswith("remarks_tfidf_")
                                       for name in fitted["feature_columns"]),
            "prediction_sha256": fitted["prediction_sha256"],
            "saved_predictions": saved,
        }
        fit_details.append(detail)
        campaign._append({
            "event": "fit_complete", "card": CARD, "run_id": run_id,
            **detail, "status": "complete",
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        return fitted

    greedy._fit_record = record_fit
    try:
        for month in SCREEN_MONTHS:
            _, train, _, truth, heldout = greedy._load_fold(month)
            cutoff = pd.Period(month, freq="M").start_time
            if (pd.to_datetime(train.date_sold, errors="raise") >= cutoff).any():
                raise RuntimeError(f"R74 training rows reach the {month} cutoff")
            homes = hide_answers(heldout, TRACK).reset_index(drop=True)
            training_mask = _route_mask(train)
            route_mask = _route_mask(homes)
            fit_rows = train.loc[training_mask].copy()
            eval_rows = homes.loc[route_mask].copy()
            expected_support = campaign.PROTOCOL["cards"][CARD].get("route_support", {}).get(month)
            actual_support = {
                "training_rows": int(len(fit_rows)),
                "evaluation_rows": int(route_mask.sum()),
            }
            if expected_support is not None and actual_support != expected_support:
                raise RuntimeError(
                    f"R74 preregistered route support changed for {month}: "
                    f"expected {expected_support}, found {actual_support}"
                )
            if len(fit_rows) < MIN_TRAINING_ROWS or int(route_mask.sum()) < MIN_EVALUATION_ROWS:
                raise RuntimeError(f"R74 {month} route is too small")

            reference = scoring.r22_reference_frame(parent["rows"], truth, month, TRACK)
            transformer, restore = r22._install_text_features(fit_rows, eval_rows, TRACK)
            fit_context.update(month=month, training_rows=len(fit_rows),
                               evaluation_rows=len(eval_rows),
                               evaluation_ids=eval_rows.ml_num.astype(str).tolist())
            try:
                fitted = greedy._fit_history_candidate(
                    month, TRACK, train, eval_rows, fit_rows=fit_rows,
                    variant=VARIANT,
                    base_variant="G06_lot_garage",
                    fit_tag=f"R74-{month}-FreeholdAttached-ExtraTrees",
                )
            finally:
                restore()

            parent_values = reference.predicted_price.to_numpy(float)
            prediction = parent_values.copy()
            specialist = blend_prices(
                fitted["predictions"], parent_values[route_mask],
                challenger_weight=SPECIALIST_WEIGHT,
            )
            prediction[route_mask] = specialist
            if not np.array_equal(prediction[~route_mask], parent_values[~route_mask]):
                raise RuntimeError(f"R74 changed homes outside the registered route in {month}")
            candidate = pd.DataFrame({
                "ml_num": homes.ml_num.astype(str).to_numpy(),
                "sold_price": truth.sold_price.to_numpy(float),
                "predicted_price": prediction,
            })
            paired = pd.DataFrame({
                "month": month, "track": TRACK,
                "ml_num": candidate.ml_num,
                "sold_price": candidate.sold_price,
                "parent_prediction": parent_values,
                "candidate_prediction": candidate.predicted_price,
            })
            paired_parts.append(paired)
            route_truth = truth.sold_price.to_numpy(float)[route_mask]
            route_details[month] = {
                "training_rows": int(len(fit_rows)), "evaluation_rows": int(route_mask.sum()),
                "parent": campaign._metrics(route_truth, parent_values[route_mask]),
                "candidate": campaign._metrics(route_truth, prediction[route_mask]),
                "non_routed_predictions_unchanged": True,
                "text_feature_count": len(transformer.columns_ or []),
                "training_cutoff_exclusive": cutoff.isoformat(),
            }
            campaign._append({
                "event": "screen_fold_complete", "card": CARD,
                "run_id": run_id, "track": TRACK, "month": month,
                "routed_rows": int(route_mask.sum()),
                "routed_training_rows": len(fit_rows),
                "training_cutoff_exclusive": cutoff.isoformat(),
                "reserved_months_labels_opened": False, "hidden_labels_opened": False,
            })
            pd.concat(paired_parts, ignore_index=True).to_csv(
                output / "partial_paired_predictions.csv", index=False)
    except Exception as exc:
        greedy._fit_record = old_fit_record
        failure = {
            "run_id": run_id, "card": CARD, "stage": "screen",
            "error": f"{type(exc).__name__}: {exc}", "fit_starts": fit_attempts,
            "fit_attempts": fit_attempts,
            "fit_details": fit_details, "route_details": route_details,
            "reserved_months_labels_opened": False,
            "reserve_validation_eligible": False, "hidden_labels_opened": False,
        }
        write_json(output / "failure_checkpoint.json", failure)
        campaign._append({
            "event": "card_failed", "card": CARD, "run_id": run_id,
            "fit_starts": fit_attempts, "error": failure["error"],
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        campaign.PROTOCOL["cards"][CARD].update({
            "status": "failed", "run_id": run_id,
            "failure_path": str((output / "failure_checkpoint.json").relative_to(campaign.ROOT)).replace("\\", "/"),
            "fit_starts": fit_attempts,
        })
        _write_json(campaign.PROTOCOL_PATH, campaign.PROTOCOL)
        campaign._write_progress(
            "r74_screen_failed", next_action="inspect R74 failure checkpoint before any retry",
            latest_fit_count=fit_attempts,
        )
        raise
    finally:
        greedy._fit_record = old_fit_record
        if old_variant is None:
            greedy.VARIANTS.pop(VARIANT, None)
        else:
            greedy.VARIANTS[VARIANT] = old_variant

    if len(fit_details) != len(SCREEN_MONTHS):
        raise RuntimeError(f"R74 expected {len(SCREEN_MONTHS)} fits, recorded {len(fit_details)}")
    paired = pd.concat(paired_parts, ignore_index=True)
    paired_path = output / "paired_predictions.csv"
    paired.to_csv(paired_path, index=False)
    rows = r22.score_development_confirmation(
        paired, months=SCREEN_MONTHS, bootstrap_replicates=BOOTSTRAP_REPLICATES,
        bootstrap_seed=BOOTSTRAP_SEED, minimum_relative_gain_percent=0.25,
        minimum_months_improved=2, maximum_worst_month_regression_percent=0.0,
        maximum_mape_regression_percent=1.0,
    )
    rows["qualifies_screen"] = rows.pop("qualifies_development")
    passed = bool(rows["qualifies_screen"])
    result = {
        "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
        "card": CARD, "track": TRACK,
        "parent_run_id": parent["result"]["run_id"],
        "parent_candidate": "R52_r48_r50_strict_past_meta_blend",
        "screen_months": list(SCREEN_MONTHS), "fit_starts": len(fit_details),
        "specialist": {"house_types": list(HOUSE_TYPES), "weight": SPECIALIST_WEIGHT,
                       "training_rows_are_routed_house_type_only": True,
                       "tfidf_is_fit_on_routed_remarks_only": True},
        "tracks": {TRACK: rows}, "routed_diagnostics": route_details,
        "fit_details": fit_details, "model_config_overrides": MODEL_OVERRIDES,
        "data_sha256": source["data_sha256"],
        "reserved_months_labels_opened": False,
        "reserve_validation_eligible": False, "hidden_labels_opened": False,
    }
    result_path = output / "screen_results.json"
    _write_json(result_path, result)
    manifest = {
        "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
        "source_identity": source, "code_sha256": code_hashes,
        "source_snapshots": snapshots,
        "parent_run_id": parent["result"]["run_id"],
        "parent_development_result_sha256": _sha256(parent["result_path"]),
        "parent_development_predictions_sha256": _sha256(parent["rows_path"]),
        "parent_development_manifest_sha256": _sha256(parent["manifest_path"]),
        "fit_starts": len(fit_details), "fit_details": fit_details,
        "model_config_overrides": MODEL_OVERRIDES,
        "paired_predictions_sha256": _sha256(paired_path),
        "result_sha256": _sha256(result_path),
        "reserved_months_labels_opened": False,
        "reserve_validation_eligible": False, "hidden_labels_opened": False,
    }
    _write_json(output / "run_manifest.json", manifest)
    report_lines = [
        "# R74：off-market LightGBM Extra-Trees split R52 专家筛选", "",
        f"Run：`{run_id}`；比较对象是当前最佳实验版 R52 `{parent['result']['run_id']}`。", "",
        "候选只在 Semi-Detached 与 Freehold Townhouse 评估行替换 R52 预测。专家训练与 TF-IDF 词表只使用这两个房型的训练行，"
        "并固定采用 75% 专家 + 25% R52 预测。唯一模型变化是 LightGBM extra_trees=true。", "",
        f"R52 pooled MdAPE：{rows['parent']['mdape_percent']:.5f}%；",
        f"R74 pooled MdAPE：{rows['candidate']['mdape_percent']:.5f}%；",
        f"相对改善：{rows['relative_mdape_improvement_percent']:+.3f}%；",
        f"MAPE 相对变化：{rows['mape_relative_change_percent']:+.3f}%。", "",
        f"两个筛选月均改善：{all(item['relative_mdape_improvement_percent'] > 0 for item in rows['monthly'].values())}；",
        f"配对 bootstrap 95% MdAPE 差值上界：{rows['bootstrap_mdape_difference_percent']['upper95']:+.5f} 个百分点。",
        f"筛选门槛通过：{passed}。", "",
        "逐 fit 参数核验了 extra_trees=true 且目标仍是 regression。只读取 2023-03 与 2024-06 开发折；未读取 2025-05/08 或隐藏标签。"
        "保留集因先前完整 parquet 载入事件而不具备独立验证资格。", "",
        f"完整预测、账本快照和哈希位于 `{output.relative_to(campaign.ROOT)}`。", "",
    ]
    (output / "research_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    campaign._append({
        "event": "card_complete", "card": CARD, "run_id": run_id,
        "fit_starts": len(fit_details),
        "result_path": str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        "qualifies_by_track": {TRACK: passed},
        "reserved_months_labels_opened": False,
        "reserve_validation_eligible": False, "hidden_labels_opened": False,
    })
    campaign.PROTOCOL["cards"][CARD].update({
        "status": "screen_passed" if passed else "screen_failed",
        "run_id": run_id, "fit_starts": len(fit_details),
        "result_path": str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        "qualifies_screen": passed,
        "screen_metrics": {
            "parent_mdape_percent": rows["parent"]["mdape_percent"],
            "candidate_mdape_percent": rows["candidate"]["mdape_percent"],
            "relative_mdape_improvement_percent": rows["relative_mdape_improvement_percent"],
            "mape_relative_change_percent": rows["mape_relative_change_percent"],
            "monthly_relative_mdape_improvement_percent": {
                month: value["relative_mdape_improvement_percent"]
                for month, value in rows["monthly"].items()
            },
        },
    })
    _write_json(campaign.PROTOCOL_PATH, campaign.PROTOCOL)
    campaign._write_progress(
        "r74_screen_passed" if passed else "r74_screen_failed",
        next_action=("R74 passed screen; preregister eight-month off-market development confirmation against R52"
                     if passed else "R74 failed screen; retain R22/R35 validated parents and R52 best off-market experiment"),
        r74_result=str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        latest_fit_count=len(fit_details),
    )
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Register or run the R74 LightGBM Extra-Trees split screen")
    parser.add_argument("--register-only", action="store_true",
                        help="preregister R74 without fitting either screen fold")
    args = parser.parse_args()
    registration = register_card()
    if args.register_only:
        print(json.dumps(registration, indent=2, ensure_ascii=False))
    else:
        outcome = run()
        score = outcome["tracks"][TRACK]
        print(
            f"R74 screen: parent_mdape={score['parent']['mdape_percent']:.5f}% "
            f"candidate_mdape={score['candidate']['mdape_percent']:.5f}% "
            f"relative_gain={score['relative_mdape_improvement_percent']:+.3f}% "
            f"pass={score['qualifies_screen']} run={outcome['run_id']}",
            flush=True,
        )
