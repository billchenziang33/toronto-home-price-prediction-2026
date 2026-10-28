"""Test the R35 Detached LightGBM expert without its frozen-HGBR blend."""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.bill_target74 import runner as campaign


CARD = "R134_listaware_detached_core_only"
TRACK = "list_aware"
HOUSE_TYPE = "Detached"
COMPONENT = "detached_core_only"
SCREEN_MONTHS = ("2023-03", "2024-06")
DEVELOPMENT_MONTHS = tuple(campaign.PROTOCOL["development_months"])
RESERVED_MONTHS = set(campaign.PROTOCOL["reserved_months"])
TARGET_MDAPE = 4.0
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20261134
MIN_TRAINING_ROWS = 500
MIN_EVALUATION_ROWS = 100
R35_CARD = "R35_listaware_detached_r22_text_expert"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return str(path.relative_to(campaign.ROOT)).replace("\\", "/")


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
                    encoding="utf-8")


def _source_files() -> list[Path]:
    return [
        Path(__file__),
        campaign.PACKAGE / "r22_remarks_tfidf.py",
        campaign.PACKAGE / "r35_fixed_origin.py",
        campaign.PACKAGE / "r03_catboost.py",
        campaign.PACKAGE / "r04_ablation.py",
        campaign.PACKAGE / "r23_word_char_tfidf.py",
        campaign.PACKAGE / "runner.py",
        campaign.ROOT / "experiments/bill_greedy/runner.py",
        campaign.ROOT / "experiments/bill_greedy/history.py",
        campaign.ROOT / "model_candidate_v2/features.py",
        campaign.ROOT / "experiments/model_optimization_v2/learners.py",
        campaign.ROOT / "experiments/model_optimization_v2/pipeline.py",
        campaign.ROOT / "experiments/model_optimization_v2/folds.py",
    ]


def _code_hashes() -> dict[str, str]:
    return {_relative(path): _sha256(path) for path in _source_files()}


def _verified_r35_parent() -> dict:
    card = campaign.PROTOCOL["cards"].get(R35_CARD)
    if not card or card.get("status") != "fixed_origin_confirmation_passed":
        raise RuntimeError("R134 requires the validated R35 list-aware route")

    screen_path = campaign.ROOT / card["result_path"]
    screen_rows_path = screen_path.parent / "paired_predictions.csv"
    screen_manifest_path = screen_path.parent / "run_manifest.json"
    development_path = campaign.ROOT / card["development_result_path"]
    development_rows_path = development_path.parent / "paired_predictions.csv"
    development_manifest_path = development_path.parent / "run_manifest.json"
    fixed_info = card.get("fixed_origin_confirmation", {})
    fixed_path = campaign.ROOT / fixed_info.get("result_path", "")
    fixed_rows_path = fixed_path.parent / "paired_predictions.csv"
    fixed_manifest_path = campaign.ROOT / fixed_info.get("manifest_path", "")
    paths = [screen_path, screen_rows_path, screen_manifest_path, development_path,
             development_rows_path, development_manifest_path, fixed_path,
             fixed_rows_path, fixed_manifest_path]
    if not all(path.is_file() for path in paths):
        raise FileNotFoundError("one or more verified R35 parent artifacts are missing")

    screen = json.loads(screen_path.read_text(encoding="utf-8"))
    development = json.loads(development_path.read_text(encoding="utf-8"))
    fixed = json.loads(fixed_path.read_text(encoding="utf-8"))
    screen_manifest = json.loads(screen_manifest_path.read_text(encoding="utf-8"))
    development_manifest = json.loads(development_manifest_path.read_text(encoding="utf-8"))
    fixed_manifest = json.loads(fixed_manifest_path.read_text(encoding="utf-8"))
    manifest_checks = [
        (screen_manifest.get("result_sha256") == _sha256(screen_path)
         and screen_manifest.get("paired_predictions_sha256") == _sha256(screen_rows_path)),
        (development_manifest.get("result_sha256") == _sha256(development_path)
         and development_manifest.get("paired_predictions_sha256") == _sha256(development_rows_path)),
        (fixed_manifest.get("result_sha256") == _sha256(fixed_path)
         and fixed_manifest.get("paired_predictions_sha256") == _sha256(fixed_rows_path)),
    ]
    if not all(manifest_checks):
        raise RuntimeError("R35 result or paired-prediction hashes do not match their manifests")
    if (screen.get("run_id") != card.get("run_id")
            or screen.get("track") != TRACK
            or screen.get("tracks", {}).get(TRACK, {}).get("qualifies_screen") is not True
            or fixed.get("passes_fixed_origin") is not True
            or development.get("tracks", {}).get(TRACK, {}).get("target_met") is not False):
        raise RuntimeError("R35 is not the expected validated parent above the strict target")
    for result in (screen, development, fixed):
        if result.get("reserved_months_labels_opened") is True or result.get("hidden_labels_opened") is True:
            raise RuntimeError("R35 parent artifacts report reserve or hidden-label access")

    screen_rows = pd.read_csv(screen_rows_path, dtype={"ml_num": str})
    development_rows = pd.read_csv(development_rows_path, dtype={"ml_num": str})
    screen_rows = screen_rows.loc[screen_rows.track.eq(TRACK)].copy()
    development_rows = development_rows.loc[development_rows.track.eq(TRACK)].copy()
    for name, frame, expected_months in (
        ("screen", screen_rows, set(SCREEN_MONTHS)),
        ("development", development_rows, set(DEVELOPMENT_MONTHS)),
    ):
        frame["month"] = frame.month.astype(str)
        if set(frame.month) != expected_months:
            raise RuntimeError(f"R35 {name} predictions do not cover the required months")
        if frame.duplicated(["month", "ml_num"]).any():
            raise RuntimeError(f"R35 {name} predictions have duplicate month/listing keys")
        values = frame[["sold_price", "candidate_prediction"]].to_numpy(float)
        if not np.isfinite(values).all() or (values <= 0).any():
            raise RuntimeError(f"R35 {name} predictions or labels are non-positive/non-finite")

    return {
        "card": card,
        "screen": screen,
        "screen_path": screen_path,
        "screen_rows_path": screen_rows_path,
        "screen_manifest_path": screen_manifest_path,
        "screen_rows": screen_rows,
        "development": development,
        "development_path": development_path,
        "development_rows_path": development_rows_path,
        "development_manifest_path": development_manifest_path,
        "development_rows": development_rows,
        "fixed": fixed,
        "fixed_path": fixed_path,
        "fixed_rows_path": fixed_rows_path,
        "fixed_manifest_path": fixed_manifest_path,
    }


def _parent_artifact_hashes(parent: dict) -> dict[str, str]:
    return {
        "screen_result_sha256": _sha256(parent["screen_path"]),
        "screen_predictions_sha256": _sha256(parent["screen_rows_path"]),
        "screen_manifest_sha256": _sha256(parent["screen_manifest_path"]),
        "development_result_sha256": _sha256(parent["development_path"]),
        "development_predictions_sha256": _sha256(parent["development_rows_path"]),
        "development_manifest_sha256": _sha256(parent["development_manifest_path"]),
        "fixed_origin_result_sha256": _sha256(parent["fixed_path"]),
        "fixed_origin_predictions_sha256": _sha256(parent["fixed_rows_path"]),
        "fixed_origin_manifest_sha256": _sha256(parent["fixed_manifest_path"]),
    }


def register_card() -> dict:
    existing = campaign.PROTOCOL["cards"].get(CARD)
    code_hashes = _code_hashes()
    if existing:
        if existing.get("status") != "registered":
            raise RuntimeError(f"{CARD} is already {existing.get('status')}; inspect its saved run")
        if existing.get("code_sha256") != code_hashes:
            raise RuntimeError("R134 source changed after preregistration")
        parent = _verified_r35_parent()
        if existing.get("parent_artifact_hashes") != _parent_artifact_hashes(parent):
            raise RuntimeError("R35 parent artifacts changed after R134 preregistration")
        return existing

    parent = _verified_r35_parent()
    from experiments.bill_target74 import r04_ablation as budget

    source = budget._source_identity()
    if source.get("data_sha256") != campaign.PROTOCOL["data_sha256"]:
        raise RuntimeError("active data hash differs from the registered campaign")
    card = {
        "status": "registered",
        "stage": "fixed_two_month_screen_then_eight_month_development",
        "track": TRACK,
        "tracks": [TRACK],
        "house_type": HOUSE_TYPE,
        "screen_months": list(SCREEN_MONTHS),
        "development_months": list(DEVELOPMENT_MONTHS),
        "expected_fit_starts": len(SCREEN_MONTHS),
        "development_fit_starts_if_screen_passes": len(DEVELOPMENT_MONTHS),
        "learner_training_fit_starts": len(SCREEN_MONTHS),
        "comparison_parent": R35_CARD,
        "parent_run_id": parent["screen"]["run_id"],
        "parent_artifact_hashes": _parent_artifact_hashes(parent),
        "method": (
            "Reproduce the validated R35 list-aware Detached-only LightGBM expert with the same "
            "fold-local R22 word-TF-IDF features, log(sold_price/list_price) target, training "
            "rows, date cutoff, and model configuration. Change only the inference blend: use "
            "100% of the rounded LightGBM expert prediction and 0% of the frozen HGBR prediction "
            "on Detached homes. Copy all non-Detached R35 predictions exactly."
        ),
        "single_change": "Detached specialist blend weight: 0.75 -> 1.00",
        "why_distinct": (
            "R35's validated route uses a 75/25 LightGBM/HGBR blend and remains above the strict "
            "4% target. On the fixed eight-month safe development window, the Detached subgroup "
            "contains 4,574 rows (29.24% of list-aware rows) and has MdAPE 5.8481%. R133's as-of AVM feature "
            "candidate worsened the Detached screen subgroup, so R134 tests the existing expert "
            "blend's remaining untested endpoint without carrying the AVM feature change."
        ),
        "prior_development_diagnostics": {
            "source": "safe 2023-03 through 2025-03 R35 OOF rows joined to exact safe-month house types",
            "detached_rows": 4574,
            "detached_row_share_percent": 29.24,
            "detached_mdape_percent": 5.8481,
            "current_specialist_weight": 0.75,
            "candidate_specialist_weight": 1.0,
        },
        "parameters": {
            "specialist_weight": 1.0,
            "frozen_hgbr_weight": 0.0,
            "model_family": "LightGBM (unchanged R35 G01 list-aware core)",
            "feature_route": "R35 Detached-only fit with fold-local R22 word TF-IDF",
        },
        "screen_gates": {
            "relative_mdape_gain_min_percent": campaign.PROTOCOL["gates"]["screen_relative_mdape_improvement_percent"],
            "both_screen_months_improve": True,
            "mape_regression_max_percent": campaign.PROTOCOL["gates"]["screen_mape_regression_max_percent"],
            "paired_bootstrap_upper95_below_zero": True,
        },
        "development_gates": dict(campaign.PROTOCOL["gates"]),
        "target_mdape_strictly_below_percent": TARGET_MDAPE,
        "source_data_filter_policy": (
            "Use only the exact safe development-month date_sold filters in greedy._load_fold. "
            "Fit strictly before each target month. Never read 2025-05/08 or hidden labels."
        ),
        "source_identity": source,
        "code_sha256": code_hashes,
        "data_sha256": campaign.PROTOCOL["data_sha256"],
        "reserved_months_labels_opened": False,
        "hidden_labels_opened": False,
    }
    campaign.PROTOCOL["cards"][CARD] = card
    _write_json(campaign.PROTOCOL_PATH, campaign.PROTOCOL)
    campaign._append({
        "event": "card_registered", "card": CARD,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "expected_fit_starts": card["expected_fit_starts"],
        "development_fit_starts_if_screen_passes": card["development_fit_starts_if_screen_passes"],
        "source_sha256": code_hashes[_relative(Path(__file__))],
        "parent_artifact_hashes": card["parent_artifact_hashes"],
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    campaign._write_progress(
        "r134_registered",
        next_action="run the preregistered R134 fixed two-month Detached blend-weight screen",
        r134_card=CARD,
    )
    return card


def _score(paired: pd.DataFrame, months: tuple[str, ...], stage: str) -> tuple[dict, dict]:
    from experiments.bill_target74 import r22_remarks_tfidf as r22

    frame = paired[["month", "ml_num", "sold_price", "parent_prediction", "candidate_prediction"]].copy()
    common = {
        "months": months,
        "target_mdape_strictly_below": TARGET_MDAPE,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }
    if stage == "screen":
        score = r22.score_development_confirmation(
            frame,
            **common,
            minimum_relative_gain_percent=campaign.PROTOCOL["gates"]["screen_relative_mdape_improvement_percent"],
            minimum_months_improved=len(months),
            maximum_worst_month_regression_percent=0.0,
            maximum_mape_regression_percent=campaign.PROTOCOL["gates"]["screen_mape_regression_max_percent"],
        )
        return score, {
            "relative_mdape_gain": score["relative_mdape_improvement_percent"] >= campaign.PROTOCOL["gates"]["screen_relative_mdape_improvement_percent"],
            "both_screen_months_improved": score["months_improved"] == len(months),
            "mape_guard": score["mape_relative_change_percent"] <= campaign.PROTOCOL["gates"]["screen_mape_regression_max_percent"],
            "paired_bootstrap_upper95_below_zero": score["bootstrap_mdape_difference_percent"]["upper95"] < 0.0,
        }
    score = r22.score_development_confirmation(frame, **common)
    return score, {
        "development_gates": bool(score["qualifies_development"]),
        "strict_mdape_target": bool(score["target_met"]),
    }


def run(stage: str) -> dict:
    from data import hide_answers
    from experiments.bill_target74 import r03_catboost as r03
    from experiments.bill_target74 import r04_ablation as budget
    from experiments.bill_target74 import r22_remarks_tfidf as r22
    from experiments.bill_target74 import r23_word_char_tfidf as scoring
    from experiments.bill_greedy import runner as greedy

    card = campaign.PROTOCOL["cards"].get(CARD)
    required_status = "registered" if stage == "screen" else "screen_passed"
    if not card or card.get("status") != required_status:
        raise RuntimeError(f"R134 {stage} requires card status {required_status}")
    if card.get("code_sha256") != _code_hashes():
        raise RuntimeError("R134 source or dependency hashes changed after preregistration")
    parent = _verified_r35_parent()
    if card.get("parent_artifact_hashes") != _parent_artifact_hashes(parent):
        raise RuntimeError("R35 parent artifacts changed after R134 preregistration")
    if card.get("source_identity", {}).get("data_sha256") != campaign.PROTOCOL["data_sha256"]:
        raise RuntimeError("R134 registered data identity does not match the active campaign")
    source = budget._source_identity()
    if source.get("data_sha256") != campaign.PROTOCOL["data_sha256"]:
        raise RuntimeError("active data hash differs from the preregistered R134 data")

    months = SCREEN_MONTHS if stage == "screen" else DEVELOPMENT_MONTHS
    parent_rows = parent["screen_rows"] if stage == "screen" else parent["development_rows"]
    run_id = ("r134s_" if stage == "screen" else "r134d_") + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = campaign.CAMPAIGN_ROOT / "batches" / run_id
    output.mkdir(parents=True, exist_ok=False)
    provenance = output / "provenance"
    provenance.mkdir()
    for source_path, name in [
        (parent["screen_path"], "r35_screen_results.json"),
        (parent["screen_rows_path"], "r35_screen_paired_predictions.csv"),
        (parent["screen_manifest_path"], "r35_screen_run_manifest.json"),
        (parent["development_path"], "r35_development_results.json"),
        (parent["development_rows_path"], "r35_development_paired_predictions.csv"),
        (parent["development_manifest_path"], "r35_development_run_manifest.json"),
        (parent["fixed_path"], "r35_fixed_origin_results.json"),
        (parent["fixed_rows_path"], "r35_fixed_origin_paired_predictions.csv"),
        (parent["fixed_manifest_path"], "r35_fixed_origin_run_manifest.json"),
    ]:
        shutil.copy2(source_path, provenance / name)
    code_hashes = _code_hashes()
    snapshots = {}
    for index, path in enumerate(_source_files()):
        key = _relative(path)
        name = f"src_{index:02d}{path.suffix}.source"
        shutil.copy2(path, provenance / name)
        snapshots[key] = name

    expected_fit_starts = len(months)
    campaign._append({
        "event": "card_start", "card": CARD, "stage": stage, "run_id": run_id,
        "track": TRACK, "parent_run_id": parent["screen"]["run_id"],
        "house_type": HOUSE_TYPE, "months": list(months),
        "expected_fit_starts": expected_fit_starts, "specialist_weight": 1.0,
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })

    fit_details: list[dict] = []
    paired_parts: list[pd.DataFrame] = []
    route_details: dict[str, dict] = {}
    fit_context: dict[str, object] = {}
    old_fit_record = greedy._fit_record

    def record_fit(_original_id: str, fit_fn):
        month = str(fit_context["month"])
        fit_id = f"{campaign.PROTOCOL['campaign_id']}:{CARD}:{stage}:{month}:{COMPONENT}:{run_id}"
        campaign._append({
            "event": "fit_start", "card": CARD, "stage": stage, "run_id": run_id,
            "fit_id": fit_id, "track": TRACK, "month": month,
            "component": COMPONENT, "house_type": HOUSE_TYPE,
            "training_rows": fit_context["training_rows"],
            "evaluation_rows": fit_context["evaluation_rows"],
            "data_sha256": source["data_sha256"],
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        started = time.perf_counter()
        try:
            fitted = fit_fn()
        except Exception as exc:
            campaign._append({
                "event": "fit_complete", "card": CARD, "stage": stage, "run_id": run_id,
                "fit_id": fit_id, "track": TRACK, "month": month,
                "component": COMPONENT, "status": "failed",
                "seconds": time.perf_counter() - started,
                "error": f"{type(exc).__name__}: {exc}",
            })
            raise
        saved = r22._save_component_predictions(
            output, TRACK, month, COMPONENT,
            fit_context["evaluation_ids"], fitted["predictions"],
        )
        detail = {
            "fit_id": fit_id, "track": TRACK, "month": month,
            "component": COMPONENT, "seconds": time.perf_counter() - started,
            "training_rows": fitted["training_rows"],
            "evaluation_rows": fitted["evaluation_rows"],
            "model_family": fitted["model_family"], "model_config": fitted["model_config"],
            "target": fitted["target"], "feature_columns": fitted["feature_columns"],
            "text_feature_count": sum(name.startswith("remarks_tfidf_")
                                       for name in fitted["feature_columns"]),
            "prediction_sha256": fitted["prediction_sha256"],
            "saved_predictions": saved,
        }
        fit_details.append(detail)
        campaign._append({
            "event": "fit_complete", "card": CARD, "stage": stage, "run_id": run_id,
            **detail, "status": "complete",
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        return fitted

    greedy._fit_record = record_fit
    try:
        for month in months:
            _, train, _, truth, heldout = greedy._load_fold(month)
            cutoff = pd.Period(month, freq="M").start_time
            train_dates = pd.to_datetime(train.date_sold, errors="raise")
            if (train_dates >= cutoff).any():
                raise RuntimeError(f"R134 training rows reach the {month} cutoff")
            if train_dates.dt.to_period("M").astype(str).isin(RESERVED_MONTHS).any():
                raise RuntimeError(f"R134 training rows include a reserved month for {month}")
            homes = hide_answers(heldout, TRACK).reset_index(drop=True)
            training_mask = greedy._specialist_mask(train, HOUSE_TYPE)
            route_mask = greedy._specialist_mask(homes, HOUSE_TYPE)
            fit_rows = train.loc[training_mask].copy()
            eval_rows = homes.loc[route_mask].copy()
            if len(fit_rows) < MIN_TRAINING_ROWS or int(route_mask.sum()) < MIN_EVALUATION_ROWS:
                raise RuntimeError(f"R134 {month} Detached route lacks registered minimum support")

            reference = scoring.r22_reference_frame(parent_rows, truth, month, TRACK)
            if reference.ml_num.astype(str).duplicated().any():
                raise RuntimeError(f"R35 parent rows are duplicated for {month}")
            if not np.array_equal(reference.ml_num.astype(str).to_numpy(), homes.ml_num.astype(str).to_numpy()):
                raise RuntimeError(f"R35 parent predictions are not aligned to the {month} fold")
            if not np.array_equal(reference.sold_price.to_numpy(float), truth.sold_price.to_numpy(float)):
                raise RuntimeError(f"R35 labels do not align to the {month} safe fold")

            transformer, restore = r22._install_text_features(fit_rows, eval_rows, TRACK)
            fit_context.update(
                month=month, training_rows=int(len(fit_rows)),
                evaluation_rows=int(route_mask.sum()),
                evaluation_ids=eval_rows.ml_num.astype(str).tolist(),
            )
            try:
                fitted = greedy._fit_history_candidate(
                    month, TRACK, train, eval_rows, fit_rows=fit_rows,
                    variant=r22.VARIANT, fit_tag=f"R134-{month}-Detached-core-only",
                )
            finally:
                restore()

            # blend_prices(..., challenger_weight=1.0) rounds the R35 core output once.
            specialist = np.round(np.asarray(fitted["predictions"], dtype=float))
            parent_values = reference.predicted_price.to_numpy(float)
            prediction = greedy._overlay_specialist_predictions(
                parent_values, homes, specialist, HOUSE_TYPE,
            )
            if not np.array_equal(prediction[~route_mask], parent_values[~route_mask]):
                raise RuntimeError(f"R134 changed non-Detached predictions in {month}")

            paired_parts.append(pd.DataFrame({
                "month": month, "track": TRACK,
                "ml_num": homes.ml_num.astype(str).to_numpy(),
                "sold_price": truth.sold_price.to_numpy(float),
                "parent_prediction": parent_values,
                "candidate_prediction": prediction,
            }))
            route_truth = truth.sold_price.to_numpy(float)[route_mask]
            route_details[month] = {
                "training_rows": int(len(fit_rows)),
                "evaluation_rows": int(route_mask.sum()),
                "parent": campaign._metrics(route_truth, parent_values[route_mask]),
                "candidate": campaign._metrics(route_truth, prediction[route_mask]),
                "non_detached_predictions_unchanged": True,
                "text_feature_count": len(transformer.columns_ or []),
                "training_cutoff_exclusive": cutoff.isoformat(),
            }
            campaign._append({
                "event": "screen_fold_complete" if stage == "screen" else "development_fold_complete",
                "card": CARD, "stage": stage, "run_id": run_id, "track": TRACK,
                "month": month, "routed_rows": int(route_mask.sum()),
                "routed_training_rows": int(len(fit_rows)),
                "training_cutoff_exclusive": cutoff.isoformat(),
                "reserved_months_labels_opened": False, "hidden_labels_opened": False,
            })
            pd.concat(paired_parts, ignore_index=True).to_csv(
                output / "partial_paired_predictions.csv", index=False,
            )
    except Exception as exc:
        greedy._fit_record = old_fit_record
        failure = {
            "run_id": run_id, "card": CARD, "stage": stage,
            "error": f"{type(exc).__name__}: {exc}",
            "fit_starts": len(fit_details), "fit_details": fit_details,
            "route_details": route_details,
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        }
        _write_json(output / "failure_checkpoint.json", failure)
        campaign._append({
            "event": "card_failed", "card": CARD, "stage": stage, "run_id": run_id,
            "fit_starts": len(fit_details), "error": failure["error"],
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        campaign.PROTOCOL["cards"][CARD].update({
            "status": f"{stage}_failed_runtime", "run_id": run_id,
            "failure_path": _relative(output / "failure_checkpoint.json"),
            "fit_starts": len(fit_details),
        })
        _write_json(campaign.PROTOCOL_PATH, campaign.PROTOCOL)
        campaign._write_progress(
            f"r134_{stage}_failed_runtime",
            next_action=f"audit R134 {stage} failure checkpoint before any retry",
            r134_run_id=run_id,
        )
        raise
    finally:
        greedy._fit_record = old_fit_record

    if len(fit_details) != expected_fit_starts:
        raise RuntimeError(f"R134 expected {expected_fit_starts} fits, recorded {len(fit_details)}")
    paired = pd.concat(paired_parts, ignore_index=True)
    if paired.duplicated(["month", "track", "ml_num"]).any():
        raise RuntimeError("R134 produced duplicate paired prediction keys")
    paired_path = output / "paired_predictions.csv"
    paired.to_csv(paired_path, index=False)
    score, gate_checks = _score(paired, tuple(months), stage)
    if stage == "screen":
        passed = all(gate_checks.values())
    else:
        passed = bool(score["qualifies_development"] and score["target_met"])

    result = {
        "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
        "card": CARD, "track": TRACK, "parent_card": R35_CARD,
        "parent_run_id": parent["screen"]["run_id"],
        "stage": stage, "months": list(months), "fit_starts": len(fit_details),
        "specialist": {
            "house_type": HOUSE_TYPE, "weight": 1.0,
            "frozen_hgbr_weight": 0.0,
            "training_rows_are_house_type_only": True,
            "tfidf_is_fit_on_house_type_rows_only": True,
            "non_detached_predictions_unchanged": True,
        },
        "tracks": {TRACK: score}, "gate_checks": gate_checks,
        "qualifies_by_track": {TRACK: passed},
        "routed_diagnostics": route_details, "fit_details": fit_details,
        "data_sha256": source["data_sha256"],
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    }
    result_path = output / ("screen_results.json" if stage == "screen" else "development_results.json")
    _write_json(result_path, result)
    manifest = {
        "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
        "source_identity": source, "code_sha256": code_hashes,
        "source_snapshots": snapshots, "parent_card": R35_CARD,
        "parent_artifact_hashes": card["parent_artifact_hashes"],
        "fit_starts": len(fit_details), "fit_details": fit_details,
        "paired_predictions_sha256": _sha256(paired_path),
        "result_sha256": _sha256(result_path),
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    }
    _write_json(output / "run_manifest.json", manifest)

    report_lines = [
        "# R134 List-aware Detached core-only blend", "",
        f"Run: `{run_id}`; parent: validated R35 `{parent['screen']['run_id']}`.", "",
        "The only change is the Detached expert weight: 0.75 to 1.00. "
        "The model, fold-local features, target, training cutoffs, and non-Detached predictions are preserved.", "",
        f"Stage: {stage}; months: {', '.join(months)}; fit starts: {len(fit_details)}.",
        f"Parent MdAPE: {score['parent']['mdape_percent']:.6f}%.",
        f"Candidate MdAPE: {score['candidate']['mdape_percent']:.6f}%.",
        f"Relative MdAPE gain: {score['relative_mdape_improvement_percent']:+.4f}%.",
        f"MAPE relative change: {score['mape_relative_change_percent']:+.4f}%.",
        f"Gate checks: `{json.dumps(gate_checks, ensure_ascii=False)}`.",
        "Reserve-month and hidden labels were not opened.",
        f"Artifacts: `{_relative(output)}`.", "",
    ]
    (output / "research_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    campaign._append({
        "event": "card_complete", "card": CARD, "stage": stage, "run_id": run_id,
        "fit_starts": len(fit_details),
        "result_path": _relative(result_path),
        "qualifies_by_track": {TRACK: passed},
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })

    updated = campaign.PROTOCOL["cards"][CARD]
    updated.update({
        "status": ("screen_passed" if passed else "screen_failed") if stage == "screen" else "development_complete",
        "run_id": run_id,
        f"{stage}_fit_starts": len(fit_details),
        f"{stage}_result_path": _relative(result_path),
        f"{stage}_result_sha256": _sha256(result_path),
        f"{stage}_paired_predictions_path": _relative(paired_path),
        f"{stage}_paired_predictions_sha256": _sha256(paired_path),
        f"{stage}_manifest_path": _relative(output / "run_manifest.json"),
        f"{stage}_manifest_sha256": _sha256(output / "run_manifest.json"),
        f"{stage}_metrics": score,
        f"{stage}_gate_checks": gate_checks,
        "qualifies_screen": passed if stage == "screen" else updated.get("qualifies_screen", False),
        "qualifies_development": passed if stage == "development" else updated.get("qualifies_development", False),
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    _write_json(campaign.PROTOCOL_PATH, campaign.PROTOCOL)
    if stage == "screen":
        next_action = (
            "R134 passed every fixed screen gate; run eight-month development"
            if passed else "R134 screen failed; retain R35 and select another distinct candidate"
        )
        progress_status = "r134_screen_passed" if passed else "r134_screen_failed"
    else:
        next_action = (
            "R134 met the strict development target; perform fixed-origin confirmation"
            if passed else "R134 did not meet the strict target and development gates; retain R35"
        )
        progress_status = "r134_development_passed" if passed else "r134_development_failed"
    campaign._write_progress(
        progress_status, next_action=next_action,
        r134_run_id=run_id, r134_result=_relative(result_path),
        r134_learner_training_fit_starts=len(fit_details),
    )
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="R134 Detached-only list-aware blend endpoint")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--register-only", action="store_true")
    group.add_argument("--screen", action="store_true")
    group.add_argument("--development", action="store_true")
    args = parser.parse_args()
    if args.register_only:
        print(json.dumps(register_card(), indent=2, ensure_ascii=False))
        return
    outcome = run("screen" if args.screen else "development")
    score = outcome["tracks"][TRACK]
    print(
        f"R134 {outcome['stage']}: parent={score['parent']['mdape_percent']:.6f}% "
        f"candidate={score['candidate']['mdape_percent']:.6f}% "
        f"relative_gain={score['relative_mdape_improvement_percent']:+.4f}% "
        f"passed={outcome['qualifies_by_track'][TRACK]} run={outcome['run_id']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
