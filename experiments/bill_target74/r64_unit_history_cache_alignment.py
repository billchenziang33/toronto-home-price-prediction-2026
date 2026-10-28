"""R64: continue the R63 unit-history ablation with verified cache alignment."""
from __future__ import annotations

import json
import hashlib
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from data import hide_answers
from experiments.bill_target74 import r03_catboost as helpers
from experiments.bill_target74 import r04_ablation as budget
from experiments.bill_target74 import r19_expert_weight as common
from experiments.bill_target74 import r22_remarks_tfidf as r22
from experiments.bill_target74 import r58_past_log_residual_slope_calibration as r58
from experiments.bill_target74 import r62_current_parent_unit_history_ablation as prior_r62
from experiments.bill_target74 import r63_current_parent_unit_history_ablation_cachefix as prior_r63
from experiments.bill_target74 import runner as campaign
from experiments.bill_greedy import runner as greedy
from experiments.model_optimization_v2 import worker
from experiments.model_optimization_v2.pipeline import blend_prices


CARD = "R64_current_parent_unit_history_ablation_cache_alignment"
R63_CARD = "R63_current_parent_unit_history_ablation"
R63_RUN = "bill_target74_r63_current_parent_unit_history_ablation_20261025T091212Z"
SCREEN_MONTHS = ("2023-03", "2024-06")
REUSED_MONTH = "2023-03"
TRACKS = ("off_market", "list_aware")
EXPECTED_FIT_STARTS = 3
MIN_RELATIVE_MDAPE_GAIN = 0.25
MAX_MAPE_REGRESSION = 1.0


def _save_protocol() -> None:
    campaign.PROTOCOL_PATH.write_text(
        json.dumps(campaign.PROTOCOL, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape or actual.ndim != 1 or not len(actual):
        raise ValueError("screen metrics require nonempty aligned vectors")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all() or (actual <= 0).any() or (predicted <= 0).any():
        raise ValueError("screen metrics require finite positive prices")
    ape = 100.0 * np.abs(predicted - actual) / actual
    return {"n": int(len(actual)), "mdape_percent": float(np.median(ape)), "mape_percent": float(np.mean(ape))}


def _relative_gain(parent: float, candidate: float) -> float:
    return 100.0 * (parent - candidate) / parent


def _verified_inputs() -> dict[str, tuple[dict, Path, Path, dict, pd.DataFrame]]:
    return {
        "off_market": r58._verified_input(campaign, r58.R52_RESULT, r58.R52_RUN, "off_market"),
        "list_aware": r58._verified_input(campaign, r58.R35_DEV_RESULT, r58.R35_DEV_RUN, "list_aware"),
    }


def _verified_r63_reuse() -> dict:
    """Verify R63's three complete 2023-03 fits before reusing their predictions."""
    run_dir = campaign.CAMPAIGN_ROOT / "batches" / R63_RUN
    checkpoint_path = run_dir / "failure_checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if (checkpoint.get("run_id") != R63_RUN or checkpoint.get("card") != R63_CARD
            or checkpoint.get("fit_starts") != 3
            or checkpoint.get("error") != "RuntimeError: frozen HGBR IDs do not align in 2023-03"
            or checkpoint.get("reserved_months_labels_opened") is not False
            or checkpoint.get("hidden_labels_opened") is not False):
        raise RuntimeError("R63 checkpoint does not match the expected cache-alignment failure")
    expected = {
        ("off_market", "base"), ("off_market", "expert"), ("list_aware", "detached_core"),
    }
    details = checkpoint.get("fit_details", [])
    if {(item.get("track"), item.get("component")) for item in details} != expected:
        raise RuntimeError("R63 completed component set differs from the registered reusable folds")
    if any(item.get("month") != REUSED_MONTH or item.get("status") != "complete" for item in details):
        raise RuntimeError("R63 reuse contains an incomplete or unexpected month")
    for item in details:
        if (any(str(name).startswith("u_") for name in item.get("feature_columns", []))
                or not item.get("removed_unit_feature_columns")):
            raise RuntimeError(f"R63 reuse did not verify the unit-history ablation: {item.get('fit_id')}")

    events = [json.loads(line) for line in campaign.LEDGER.read_text(encoding="utf-8").splitlines() if line.strip()]
    failed = [event for event in events if event.get("event") == "card_failed"
              and event.get("card") == R63_CARD and event.get("run_id") == R63_RUN]
    if len(failed) != 1 or failed[0].get("fit_starts") != 3:
        raise RuntimeError("R63 append-only ledger does not confirm the expected failed run")
    card_starts = [event for event in events if event.get("event") == "card_start"
                   and event.get("card") == R63_CARD and event.get("run_id") == R63_RUN]
    if (len(card_starts) != 1 or card_starts[0].get("reserved_months_labels_opened") is not False
            or card_starts[0].get("hidden_labels_opened") is not False
            or card_starts[0].get("data_sha256") != campaign.PROTOCOL["data_sha256"]):
        raise RuntimeError("R63 card-start provenance does not verify")

    components = {}
    for item in details:
        fit_events = [event for event in events if event.get("fit_id") == item.get("fit_id")
                      and event.get("run_id") == R63_RUN]
        start_events = [event for event in fit_events if event.get("event") == "fit_start"]
        if (len(start_events) != 1 or start_events[0].get("data_sha256") != campaign.PROTOCOL["data_sha256"]
                or start_events[0].get("reserved_months_labels_opened") is not False
                or start_events[0].get("hidden_labels_opened") is not False):
            raise RuntimeError(f"R63 fit start is missing for {item.get('fit_id')}")
        if not any(event.get("event") == "fit_complete" and event.get("status") == "complete"
                   for event in fit_events):
            raise RuntimeError(f"R63 completed fit is missing from the ledger: {item.get('fit_id')}")
        saved = item.get("saved_predictions", {})
        path = run_dir / saved.get("path", "")
        if not path.is_file() or common._sha256(path) != saved.get("sha256"):
            raise RuntimeError(f"R63 saved component hash does not verify: {path}")
        frame = pd.read_csv(path, dtype={"ml_num": str})
        if (not {"ml_num", "predicted_price"}.issubset(frame.columns)
                or len(frame) != saved.get("rows") or frame.ml_num.duplicated().any()
                or not np.isfinite(frame.predicted_price.to_numpy(float)).all()
                or (frame.predicted_price.to_numpy(float) <= 0).any()):
            raise RuntimeError(f"R63 saved component predictions are invalid: {path}")
        components[(item["track"], item["component"])] = frame

    r63_source = Path(prior_r63.__file__)
    registered_r63 = campaign.PROTOCOL["cards"].get(R63_CARD, {})
    if common._sha256(r63_source) != registered_r63.get("source_sha256"):
        raise RuntimeError("R63 source file no longer matches its preregistered hash")
    return {
        "run_id": R63_RUN, "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": common._sha256(checkpoint_path),
        "details": details, "components": components,
        "source_sha256": registered_r63["source_sha256"],
    }


def _align_saved_component(frame: pd.DataFrame, ids: list[str], label: str) -> np.ndarray:
    source = frame.copy()
    source["ml_num"] = source.ml_num.astype(str)
    expected = pd.Index([str(value) for value in ids])
    if source.ml_num.duplicated().any() or set(source.ml_num) != set(expected):
        raise RuntimeError(f"reused R63 component IDs do not align for {label}")
    aligned = source.set_index("ml_num").loc[expected]
    values = aligned.predicted_price.to_numpy(float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise RuntimeError(f"reused R63 component predictions are invalid for {label}")
    return values


def _reused_component_manifest(reuse: dict, track: str, component: str) -> dict:
    matches = [item for item in reuse["details"]
               if item.get("track") == track and item.get("component") == component]
    if len(matches) != 1:
        raise RuntimeError(f"expected one reusable R63 component for {track}/{component}")
    item = matches[0]
    return {
        "fit_id": item["fit_id"], "track": track, "month": REUSED_MONTH,
        "component": component, "reused": True, "source_run_id": reuse["run_id"],
        "model_family": item["model_family"], "model_config": item["model_config"],
        "target": item["target"], "training_rows": item["training_rows"],
        "evaluation_rows": item["evaluation_rows"],
        "feature_columns": item["feature_columns"],
        "removed_unit_feature_columns": item["removed_unit_feature_columns"],
        "prediction_sha256": item["prediction_sha256"],
        "saved_predictions_sha256": item["saved_predictions"]["sha256"],
    }


def register_card() -> dict:
    existing = campaign.PROTOCOL["cards"].get(CARD)
    if existing:
        if existing.get("status") == "registered" and existing.get("source_sha256") == common._sha256(Path(__file__)):
            return existing
        raise RuntimeError(f"R64 is already {existing.get('status')}; inspect before rerunning")

    prior_card = campaign.PROTOCOL["cards"].get("R62_current_parent_unit_history_ablation")
    if not prior_card or prior_card.get("status") != "preflight_failed" or prior_card.get("fit_starts") != 0:
        raise RuntimeError("R64 requires R62's recorded zero-fit preflight failure")
    reused = _verified_r63_reuse()

    r52_card = campaign.PROTOCOL["cards"].get("R52_r48_r50_strict_past_meta_blend")
    r35_card = campaign.PROTOCOL["cards"].get("R35_listaware_detached_r22_text_expert")
    if not r52_card or r52_card.get("run_id") != r58.R52_RUN:
        raise RuntimeError("R64 requires the recorded R52 best off-market candidate")
    if not r35_card or r35_card.get("status") != "fixed_origin_confirmation_passed":
        raise RuntimeError("R64 requires the fixed-origin-validated R35 list-aware candidate")

    source = budget._source_identity()
    inputs = _verified_inputs()
    hgbr_cache_hashes = {}
    for month in SCREEN_MONTHS:
        cache = greedy.OUTPUT / "frozen_hgbr" / "list_aware" / f"{month}.csv"
        if not cache.is_file():
            raise FileNotFoundError(f"R64 list-aware frozen HGBR cache is missing for {month}")
        hgbr_cache_hashes[month] = common._sha256(cache)
    hashes = {
        track: {
            "run_id": item[0]["run_id"],
            "result_sha256": common._sha256(item[1]),
            "paired_predictions_sha256": common._sha256(item[2]),
            "manifest_sha256": common._sha256(item[1].parent / "run_manifest.json"),
        }
        for track, item in inputs.items()
    }
    # Verify that the intended screen rows exist in the frozen inputs before registration.
    for track, item in inputs.items():
        rows = item[4]
        if not set(SCREEN_MONTHS).issubset(set(rows.month.astype(str))):
            raise RuntimeError(f"{track} parent artifacts do not cover both registered screen months")

    registration = {
        "status": "registered", "tracks": list(TRACKS), "screen_months": list(SCREEN_MONTHS),
        "expected_fit_starts": EXPECTED_FIT_STARTS,
        "reused_fit_ids": [item["fit_id"] for item in reused["details"]],
        "reused_source_run_id": reused["run_id"],
        "reused_checkpoint_sha256": reused["checkpoint_sha256"],
        "reused_component_hashes": {
            f"{item['track']}:{item['component']}": item["saved_predictions"]["sha256"]
            for item in reused["details"]
        },
        "frozen_hgbr_cache_hashes": hgbr_cache_hashes,
        "parents": {
            "off_market": "R52 best experimental candidate; R22 is retained as a secondary reference",
            "list_aware": "R35 validated Detached-text candidate",
        },
        "parent_run_ids": {track: value["run_id"] for track, value in hashes.items()},
        "parent_artifact_hashes": hashes,
        "method": (
            "Keep the R22/R35 text features, LightGBM configurations, targets, routes and price blends fixed. "
            "Remove only normalized `u_*` unit-history columns, which are keyed by rounded coordinates, before "
            "fitting. Off-market refits the R22 G06 base and Detached expert (75/25 route). List-aware refits "
            "only the R35 Detached G01 text core and keeps its verified 25% HGBR blend; all non-Detached list-aware "
            "predictions remain the R35 parent. Reuse the hash-verified 2023-03 R63 fits and train only the "
            "2024-06 fold. Validate HGBR cache IDs against the complete list-aware heldout set, then select the "
            "Detached rows by route mask. Screen on the two predeclared safe months."
        ),
        "difference_from_prior_work": (
            "R04 removed `u_*` from the older G19/G01 baseline and R06 found its OOF errors highly correlated "
            "with those baselines, so that blend route was rejected. R63 retested the ablation directly inside the "
            "current R22 text architecture and R35 Detached specialist, but stopped because the complete-month "
            "HGBR cache validator was passed the Detached subset. R64 corrects this alignment, reuses the three "
            "verified R63 fits for 2023-03, and compares off-market against R52 and list-aware against R35; it does "
            "not blend the old R04 candidate."
        ),
        "removed_feature_rule": "drop every normalized feature whose name starts with `u_`; retain `b_*` and all other columns",
        "screen_gate": {
            "minimum_relative_pooled_mdape_improvement_percent": MIN_RELATIVE_MDAPE_GAIN,
            "both_screen_months_must_improve": True,
            "maximum_mape_regression_percent": MAX_MAPE_REGRESSION,
            "primary_parent": {"off_market": "R52 candidate_prediction", "list_aware": "R35 candidate_prediction"},
            "secondary_reference": "R22 parent_prediction on the same rows",
        },
        "policy": (
            "Each fold's parquet load is row-filtered to date_sold < the following month; fit rows are asserted "
            "strictly earlier than the evaluation month. The only label-bearing evaluations are 2023-03 and "
            "2024-06. Never read 2025-05/08 or hidden labels. No parameters are selected after seeing screen labels."
        ),
        "source_sha256": common._sha256(Path(__file__)), "data_sha256": source["data_sha256"],
        "reused_r63": {
            "run_id": reused["run_id"],
            "checkpoint_sha256": reused["checkpoint_sha256"],
            "source_sha256": reused["source_sha256"],
            "fit_ids": [item["fit_id"] for item in reused["details"]],
            "component_hashes": {
                f"{item['track']}:{item['component']}": item["saved_predictions"]["sha256"]
                for item in reused["details"]
            },
        },
        "reserve_validation_eligible": False,
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    }
    campaign.PROTOCOL["cards"][CARD] = registration
    _save_protocol()
    campaign._append({
        "event": "card_registered", "card": CARD,
        "tracks": list(TRACKS), "screen_months": list(SCREEN_MONTHS),
        "expected_fit_starts": EXPECTED_FIT_STARTS,
        "reused_fit_ids": registration["reused_r63"]["fit_ids"],
        "reused_run_id": reused["run_id"],
        "source_sha256": registration["source_sha256"], "data_sha256": source["data_sha256"],
        "parent_artifact_hashes": hashes,
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    campaign._write_progress(
        "r64_registered",
        next_action="reuse R63's verified 2023-03 components and screen the corrected R22/R35 unit-history ablation",
        r64_card=CARD,
    )
    return registration


def _aligned_parent_rows(inputs: dict, track: str, month: str, ids: list[str], truth: pd.DataFrame) -> pd.DataFrame:
    source_rows = inputs[track][4]
    subset = source_rows.loc[source_rows.month.astype(str).eq(month)].copy()
    if subset.ml_num.astype(str).duplicated().any():
        raise RuntimeError(f"duplicate frozen parent IDs for {track}/{month}")
    subset["ml_num"] = subset.ml_num.astype(str)
    if set(subset.ml_num) != set(ids):
        raise RuntimeError(f"R64 parent predictions do not match heldout IDs for {track}/{month}")
    aligned = subset.set_index("ml_num").loc[ids].reset_index()
    if not np.array_equal(aligned.sold_price.to_numpy(float), truth.sold_price.to_numpy(float)):
        raise RuntimeError(f"R64 parent labels do not match heldout labels for {track}/{month}")
    return aligned


def _save_fit_prediction(output: Path, index: int, predictions: np.ndarray, ids) -> dict:
    frame = pd.DataFrame({"ml_num": pd.Series(ids, dtype=str).to_numpy(),
                          "predicted_price": np.asarray(predictions, dtype=float)})
    if frame.ml_num.duplicated().any() or not np.isfinite(frame.predicted_price).all() or (frame.predicted_price <= 0).any():
        raise ValueError("R64 fit produced invalid component predictions")
    path = output / f"component_{index:02d}.csv"
    frame.to_csv(path, index=False)
    return {"path": path.name, "sha256": common._sha256(path), "rows": len(frame)}


def run() -> dict:
    card = campaign.PROTOCOL["cards"].get(CARD)
    if not card or card.get("status") != "registered":
        raise RuntimeError("R64 must be preregistered before fitting")
    if card.get("source_sha256") != common._sha256(Path(__file__)):
        raise RuntimeError("R64 source changed after registration")
    source = budget._source_identity()
    if source["data_sha256"] != card["data_sha256"]:
        raise RuntimeError("R64 data hash differs from its preregistration")
    reused = _verified_r63_reuse()
    if (card.get("reused_r63", {}).get("checkpoint_sha256") != reused["checkpoint_sha256"]
            or card.get("reused_r63", {}).get("fit_ids") != [item["fit_id"] for item in reused["details"]]):
        raise RuntimeError("R63 reusable artifacts differ from the R64 preregistration")
    inputs = _verified_inputs()
    for track, item in inputs.items():
        expected = card["parent_artifact_hashes"][track]
        if (common._sha256(item[1]) != expected["result_sha256"]
                or common._sha256(item[2]) != expected["paired_predictions_sha256"]
                or common._sha256(item[1].parent / "run_manifest.json") != expected["manifest_sha256"]):
            raise RuntimeError(f"R64 {track} input artifact hashes changed after registration")

    code_sources = [
        campaign.PROTOCOL_PATH, Path(__file__), Path(r58.__file__), Path(prior_r62.__file__),
        Path(prior_r63.__file__), Path(r22.__file__),
        Path(helpers.__file__), Path(budget.__file__), Path(common.__file__), Path(campaign.__file__),
        Path(greedy.__file__), Path(worker.__file__),
        campaign.ROOT / "experiments/bill_greedy/history.py",
        campaign.ROOT / "experiments/model_optimization_v2/learners.py",
        campaign.ROOT / "experiments/model_optimization_v2/pipeline.py",
        campaign.ROOT / "model_candidate_v2/features.py",
        campaign.ROOT / "model_candidate_v2/winner_config.json",
    ]
    code_sources = list(dict.fromkeys(code_sources))
    for path in code_sources:
        if not path.is_file():
            raise FileNotFoundError(f"R64 source file is missing before fit start: {path}")
    hgbr_sources = {}
    for month in SCREEN_MONTHS:
        cache = greedy.OUTPUT / "frozen_hgbr" / "list_aware" / f"{month}.csv"
        if not cache.is_file():
            raise FileNotFoundError(f"R64 list-aware frozen HGBR cache is missing for {month}")
        if common._sha256(cache) != card.get("frozen_hgbr_cache_hashes", {}).get(month):
            raise RuntimeError(f"R64 frozen HGBR cache changed after preregistration: {month}")
        hgbr_sources[month] = {
            "path": str(cache.relative_to(campaign.ROOT)).replace("\\", "/"),
            "csv_sha256": common._sha256(cache), "cache": cache,
        }

    run_id = "bill_target74_r64_unit_history_cache_alignment_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = campaign.CAMPAIGN_ROOT / "batches" / run_id
    output.mkdir(parents=True, exist_ok=False)
    provenance = output / "provenance"
    provenance.mkdir()
    for track, item in inputs.items():
        prefix = "r52_off_market" if track == "off_market" else "r35_list_aware"
        shutil.copy2(item[1], provenance / f"{prefix}_result.json")
        shutil.copy2(item[2], provenance / f"{prefix}_paired.csv")
        shutil.copy2(item[1].parent / "run_manifest.json", provenance / f"{prefix}_manifest.json")
    code_map = []
    for i, path in enumerate(code_sources):
        short = provenance / f"code{i:02d}.source"
        shutil.copy2(path, short)
        code_map.append({"source_path": str(path.relative_to(campaign.ROOT)).replace("\\", "/"),
                         "snapshot_path": str(short.relative_to(output)).replace("\\", "/"),
                         "sha256": common._sha256(path)})
    common._write_json(provenance / "source_map.json", code_map)
    shutil.copy2(reused["checkpoint_path"], provenance / "r63_failure_checkpoint.json")
    reuse_manifest = []
    for index, item in enumerate(reused["details"]):
        source_path = reused["checkpoint_path"].parent / item["saved_predictions"]["path"]
        short_path = provenance / f"r63_component_{index:02d}.csv"
        shutil.copy2(source_path, short_path)
        reuse_manifest.append({
            "fit_id": item["fit_id"], "track": item["track"], "month": item["month"],
            "component": item["component"], "source_sha256": item["saved_predictions"]["sha256"],
            "snapshot_path": str(short_path.relative_to(output)).replace("\\", "/"),
        })
    common._write_json(provenance / "reused_r63_components.json", {
        "run_id": reused["run_id"], "checkpoint_sha256": reused["checkpoint_sha256"],
        "source_sha256": reused["source_sha256"], "components": reuse_manifest,
    })
    for index, (month, item) in enumerate(hgbr_sources.items()):
        shutil.copy2(item["cache"], provenance / f"hgbr{index:02d}.csv")

    campaign._append({
        "event": "card_start", "card": CARD, "run_id": run_id,
        "tracks": list(TRACKS), "screen_months": list(SCREEN_MONTHS),
        "expected_fit_starts": EXPECTED_FIT_STARTS,
        "reused_fit_ids": card["reused_r63"]["fit_ids"],
        "reused_source_run_id": reused["run_id"],
        "reused_checkpoint_sha256": reused["checkpoint_sha256"],
        "parent_run_ids": card["parent_run_ids"], "parent_artifact_hashes": card["parent_artifact_hashes"],
        "data_sha256": source["data_sha256"],
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })

    old_normalize = worker._normalize_categories
    old_fit_record = greedy._fit_record
    old_hgbr = greedy._frozen_hgbr_predictions
    active: dict[str, object] = {}
    fit_details, component_manifest, paired_parts = [], [], []
    reused_fit_details = list(reused["details"])
    removed_features: set[str] = set()
    cache_preflight = {}
    for item in reused_fit_details:
        removed_features.update(item.get("removed_unit_feature_columns", []))

    def normalize_without_unit(train_x, eval_x, family):
        train_norm, eval_norm, categories = old_normalize(train_x, eval_x, family)
        train_out, eval_out, categories_out, removed = budget.remove_unit_history_features(
            train_norm, eval_norm, categories
        )
        removed_features.update(removed)
        active["removed_features"] = removed
        if any(str(name).startswith("u_") for name in train_out.columns):
            raise RuntimeError("R64 feature ablation left a u_* column in the training matrix")
        return train_out, eval_out, categories_out

    def record_fit(parent_fit_id: str, fit_fn):
        if not active:
            raise RuntimeError("R64 fit context was not set")
        track, month, component = active["track"], active["month"], active["component"]
        fit_id = f"{campaign.PROTOCOL['campaign_id']}:{CARD}:{track}:{month}:{component}:{run_id}"
        campaign._append({
            "event": "fit_start", "card": CARD, "run_id": run_id, "fit_id": fit_id,
            "parent_fit_id": parent_fit_id, "track": track, "month": month, "component": component,
            "removed_feature_prefix": "u_", "data_sha256": source["data_sha256"],
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        started = time.perf_counter()
        try:
            fitted = fit_fn()
            if any(str(name).startswith("u_") for name in fitted["feature_columns"]):
                raise RuntimeError("R64 fitted model reports a retained u_* feature")
            component_index = len(fit_details)
            saved = _save_fit_prediction(output, component_index, fitted["predictions"], active["evaluation_ids"])
            detail = {
                "fit_id": fit_id, "track": track, "month": month, "component": component,
                "seconds": time.perf_counter() - started,
                "model_family": fitted["model_family"], "model_config": fitted["model_config"],
                "target": fitted["target"], "training_rows": fitted["training_rows"],
                "evaluation_rows": fitted["evaluation_rows"],
                "feature_columns": fitted["feature_columns"],
                "removed_unit_feature_columns": sorted(active.get("removed_features", [])),
                "prediction_sha256": fitted["prediction_sha256"], "saved_predictions": saved,
                "status": "complete",
            }
            fit_details.append(detail)
            campaign._append({"event": "fit_complete", "card": CARD, "run_id": run_id, **detail})
            return fitted
        except Exception as exc:
            campaign._append({
                "event": "fit_complete", "card": CARD, "run_id": run_id,
                "fit_id": fit_id, "track": track, "month": month, "component": component,
                "status": "failed", "seconds": time.perf_counter() - started,
                "error": f"{type(exc).__name__}: {exc}",
            })
            raise

    worker._normalize_categories = normalize_without_unit
    greedy._fit_record = record_fit
    greedy._frozen_hgbr_predictions = lambda month, track, train, homes: helpers._verified_hgbr_cache(
        greedy, month, track, train, homes
    )

    try:
        # Validate both registered frozen-cache ID sequences before starting any new learner fit.
        for month in SCREEN_MONTHS:
            _, train, _, _, heldout = greedy._load_fold(month)
            cutoff = pd.Period(month, freq="M").start_time
            dates = pd.to_datetime(train.date_sold, errors="raise")
            if dates.max() >= cutoff:
                raise RuntimeError(f"R64 preflight found training rows at or after the fold cutoff: {month}")
            homes = hide_answers(heldout, "list_aware")
            ids = homes.ml_num.astype(str).tolist()
            predictions = greedy._frozen_hgbr_predictions(month, "list_aware", train, homes)
            if len(predictions) != len(ids):
                raise RuntimeError(f"R64 HGBR cache row count differs after ID verification: {month}")
            cache_preflight[month] = {
                "rows": len(ids),
                "ids_sha256": hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest(),
                "cache_sha256": hgbr_sources[month]["csv_sha256"],
            }
        campaign._append({
            "event": "cache_alignment_preflight", "card": CARD, "run_id": run_id,
            "months": cache_preflight, "new_fit_starts": 0,
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })

        for month in SCREEN_MONTHS:
            # _load_fold pushes the parquet date filter below month_end before reading rows.
            listings, train, _, truth, heldout = greedy._load_fold(month)
            cutoff = pd.Period(month, freq="M").start_time
            dates = pd.to_datetime(train.date_sold, errors="raise")
            if dates.max() >= cutoff:
                raise RuntimeError(f"R64 training data include rows at or after the fold cutoff: {month}")

            for track in TRACKS:
                homes = hide_answers(heldout, track)
                ids = homes.ml_num.astype(str).tolist()
                reference_rows = _aligned_parent_rows(inputs, track, month, ids, truth)
                if month == REUSED_MONTH:
                    primary_parent = reference_rows.candidate_prediction.to_numpy(float)
                    validated_parent = reference_rows.parent_prediction.to_numpy(float)
                    specialist_eval_mask = greedy._specialist_mask(homes, "Detached")
                    specialist_ids = homes.loc[specialist_eval_mask, "ml_num"].astype(str).tolist()
                    if track == "off_market":
                        base_prediction = _align_saved_component(
                            reused["components"][(track, "base")], ids, f"{track}/{month}/base"
                        )
                        expert_prediction = _align_saved_component(
                            reused["components"][(track, "expert")], specialist_ids,
                            f"{track}/{month}/expert",
                        )
                        candidate_prediction = greedy._blend_specialist_predictions(
                            np.round(base_prediction), np.round(expert_prediction), homes, "Detached", 0.75
                        )
                        reused_components = ("base", "expert")
                        parent_name = "R52 best experimental candidate"
                    else:
                        candidate_prediction = primary_parent.copy()
                        core_prediction = _align_saved_component(
                            reused["components"][(track, "detached_core")], specialist_ids,
                            f"{track}/{month}/detached_core",
                        )
                        # The frozen CSV covers all list-aware homes; validate it before selecting Detached rows.
                        frozen_all = greedy._frozen_hgbr_predictions(month, track, train, homes)
                        detached_prediction = blend_prices(
                            core_prediction, frozen_all[specialist_eval_mask], challenger_weight=0.75
                        )
                        candidate_prediction[specialist_eval_mask] = np.asarray(detached_prediction, dtype=float)
                        if not np.array_equal(candidate_prediction[~specialist_eval_mask], primary_parent[~specialist_eval_mask]):
                            raise RuntimeError("R64 changed non-Detached list-aware parent predictions")
                        reused_components = ("detached_core",)
                        parent_name = "R35 validated candidate"

                    for component in reused_components:
                        item = _reused_component_manifest(reused, track, component)
                        component_manifest.append(item)
                        campaign._append({
                            "event": "fit_reused", "card": CARD, "run_id": run_id,
                            "source_run_id": reused["run_id"], "fit_id": item["fit_id"],
                            "track": track, "month": month, "component": component,
                            "prediction_sha256": item["saved_predictions_sha256"],
                            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
                        })
                    pair = pd.DataFrame({
                        "month": month, "track": track, "ml_num": ids,
                        "sold_price": truth.sold_price.to_numpy(float),
                        "parent_prediction": primary_parent,
                        "validated_R22_parent_prediction": validated_parent,
                        "candidate_prediction": np.asarray(candidate_prediction, dtype=float),
                        "primary_parent": parent_name,
                    })
                    if not np.isfinite(pair[["sold_price", "parent_prediction", "candidate_prediction"]].to_numpy(float)).all():
                        raise RuntimeError(f"R64 produced non-finite reused predictions for {track}/{month}")
                    paired_parts.append(pair)
                    campaign._append({
                        "event": "screen_fold_complete", "card": CARD, "run_id": run_id,
                        "track": track, "month": month, "parent_name": parent_name,
                        "rows": len(pair), "date_cutoff_exclusive": cutoff.isoformat(),
                        "reused_source_run_id": reused["run_id"],
                        "removed_unit_feature_columns": sorted(removed_features),
                        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
                    })
                    continue
                if track == "off_market":
                    primary_parent = reference_rows.candidate_prediction.to_numpy(float)
                    validated_parent = reference_rows.parent_prediction.to_numpy(float)
                    base_context = {"track": track, "month": month, "component": "base"}
                    transformer, restore_text = r22._install_text_features(train, homes, track)
                    active.clear(); active.update(base_context, evaluation_ids=ids)
                    try:
                        base_fit = greedy._fit_history_candidate(
                            month, track, train, homes, base_variant="G06_lot_garage",
                            variant=r22.VARIANT, fit_tag=f"R64-base-{month}", fit_rows=train,
                        )
                    finally:
                        restore_text()
                    component_manifest.append({
                        "track": track, "month": month, "component": "base",
                        "model_family": base_fit["model_family"], "model_config": base_fit["model_config"],
                        "target": base_fit["target"], "training_rows": base_fit["training_rows"],
                        "evaluation_rows": base_fit["evaluation_rows"],
                        "feature_columns": base_fit["feature_columns"],
                        "removed_unit_feature_columns": sorted(active.get("removed_features", [])),
                        "text_feature_count": len(transformer.columns_ or []),
                    })
                    base_prediction = np.round(base_fit["predictions"])

                    specialist_train_mask = greedy._specialist_mask(train, "Detached")
                    specialist_eval_mask = greedy._specialist_mask(homes, "Detached")
                    specialist_train = train.loc[specialist_train_mask].copy()
                    specialist_homes = homes.loc[specialist_eval_mask].copy()
                    if specialist_train.empty or not specialist_eval_mask.any():
                        raise RuntimeError(f"R64 Detached specialist has no rows in {month}")
                    specialist_ids = specialist_homes.ml_num.astype(str).tolist()
                    active.clear(); active.update(track=track, month=month, component="expert", evaluation_ids=specialist_ids)
                    transformer, restore_text = r22._install_text_features(specialist_train, specialist_homes, track)
                    try:
                        expert_fit = greedy._fit_history_candidate(
                            month, track, train, specialist_homes, base_variant="G06_lot_garage",
                            variant=r22.VARIANT, fit_tag=f"R64-expert-{month}", fit_rows=specialist_train,
                        )
                    finally:
                        restore_text()
                    component_manifest.append({
                        "track": track, "month": month, "component": "expert",
                        "model_family": expert_fit["model_family"], "model_config": expert_fit["model_config"],
                        "target": expert_fit["target"], "training_rows": expert_fit["training_rows"],
                        "evaluation_rows": expert_fit["evaluation_rows"],
                        "feature_columns": expert_fit["feature_columns"],
                        "removed_unit_feature_columns": sorted(active.get("removed_features", [])),
                        "text_feature_count": len(transformer.columns_ or []),
                    })
                    candidate_prediction = greedy._blend_specialist_predictions(
                        base_prediction, np.round(expert_fit["predictions"]), homes, "Detached", 0.75
                    )
                    parent_name = "R52 best experimental candidate"
                else:
                    # Keep non-Detached rows exactly at R35; only replace its Detached text core.
                    candidate_prediction = primary_parent.copy()
                    specialist_train_mask = greedy._specialist_mask(train, "Detached")
                    specialist_eval_mask = greedy._specialist_mask(homes, "Detached")
                    specialist_train = train.loc[specialist_train_mask].copy()
                    specialist_homes = homes.loc[specialist_eval_mask].copy()
                    if specialist_train.empty or not specialist_eval_mask.any():
                        raise RuntimeError(f"R64 list-aware Detached specialist has no rows in {month}")
                    specialist_ids = specialist_homes.ml_num.astype(str).tolist()
                    active.clear(); active.update(track=track, month=month, component="detached_core", evaluation_ids=specialist_ids)
                    transformer, restore_text = r22._install_text_features(specialist_train, specialist_homes, track)
                    try:
                        core_fit = greedy._fit_history_candidate(
                            month, track, train, specialist_homes, variant=r22.VARIANT,
                            fit_tag=f"R64-list-detached-{month}", fit_rows=specialist_train,
                        )
                    finally:
                        restore_text()
                    frozen_hgbr = greedy._frozen_hgbr_predictions(month, track, train, homes)[specialist_eval_mask]
                    detached_prediction = blend_prices(
                        core_fit["predictions"], frozen_hgbr, challenger_weight=0.75
                    )
                    candidate_prediction[specialist_eval_mask] = np.asarray(detached_prediction, dtype=float)
                    if not np.array_equal(candidate_prediction[~specialist_eval_mask], primary_parent[~specialist_eval_mask]):
                        raise RuntimeError("R64 changed non-Detached list-aware parent predictions")
                    component_manifest.append({
                        "track": track, "month": month, "component": "detached_core",
                        "model_family": core_fit["model_family"], "model_config": core_fit["model_config"],
                        "target": core_fit["target"], "training_rows": core_fit["training_rows"],
                        "evaluation_rows": core_fit["evaluation_rows"],
                        "feature_columns": core_fit["feature_columns"],
                        "removed_unit_feature_columns": sorted(active.get("removed_features", [])),
                        "text_feature_count": len(transformer.columns_ or []),
                    })
                    parent_name = "R35 validated candidate"

                pair = pd.DataFrame({
                    "month": month, "track": track, "ml_num": ids,
                    "sold_price": truth.sold_price.to_numpy(float),
                    "parent_prediction": primary_parent,
                    "validated_R22_parent_prediction": validated_parent,
                    "candidate_prediction": np.asarray(candidate_prediction, dtype=float),
                    "primary_parent": parent_name,
                })
                if not np.isfinite(pair[["sold_price", "parent_prediction", "candidate_prediction"]].to_numpy(float)).all():
                    raise RuntimeError(f"R64 produced non-finite paired predictions for {track}/{month}")
                paired_parts.append(pair)
                campaign._append({
                    "event": "screen_fold_complete", "card": CARD, "run_id": run_id,
                    "track": track, "month": month, "parent_name": parent_name,
                    "rows": len(pair), "date_cutoff_exclusive": cutoff.isoformat(),
                    "removed_unit_feature_columns": sorted(removed_features),
                    "reserved_months_labels_opened": False, "hidden_labels_opened": False,
                })
    except Exception as exc:
        failure = {
            "run_id": run_id, "card": CARD, "error": f"{type(exc).__name__}: {exc}",
            "fit_starts": len(fit_details), "fit_details": fit_details,
            "reused_fit_details": reused_fit_details,
            "cache_alignment_preflight": cache_preflight,
            "completed_components": component_manifest,
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        }
        common._write_json(output / "failure_checkpoint.json", failure)
        campaign._append({
            "event": "card_failed", "card": CARD, "run_id": run_id,
            "fit_starts": len(fit_details), "error": failure["error"],
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        card.update({"status": "screen_failed_runtime", "run_id": run_id,
                     "fit_starts": len(fit_details), "failure_path": str((output / "failure_checkpoint.json").relative_to(campaign.ROOT)).replace("\\", "/")})
        _save_protocol()
        campaign._write_progress(
            "r64_runtime_failed", next_action="inspect R64 checkpoint and ledger before any new fit",
            r64_failure=str((output / "failure_checkpoint.json").relative_to(campaign.ROOT)).replace("\\", "/"),
        )
        raise
    finally:
        worker._normalize_categories = old_normalize
        greedy._fit_record = old_fit_record
        greedy._frozen_hgbr_predictions = old_hgbr

    if len(fit_details) != EXPECTED_FIT_STARTS:
        raise RuntimeError(f"R64 expected {EXPECTED_FIT_STARTS} new fits, observed {len(fit_details)}")
    paired = pd.concat(paired_parts, ignore_index=True)
    paired_path = output / "paired_predictions.csv"
    paired.to_csv(paired_path, index=False)

    track_results = {}
    for track in TRACKS:
        group = paired.loc[paired.track.eq(track)].copy()
        primary = _metrics(group.sold_price.to_numpy(float), group.parent_prediction.to_numpy(float))
        candidate = _metrics(group.sold_price.to_numpy(float), group.candidate_prediction.to_numpy(float))
        validated = _metrics(group.sold_price.to_numpy(float), group.validated_R22_parent_prediction.to_numpy(float))
        mape_change = 100.0 * (candidate["mape_percent"] - primary["mape_percent"]) / primary["mape_percent"]
        monthly = {}
        for month in SCREEN_MONTHS:
            current = group.loc[group.month.eq(month)]
            pm = _metrics(current.sold_price.to_numpy(float), current.parent_prediction.to_numpy(float))
            cm = _metrics(current.sold_price.to_numpy(float), current.candidate_prediction.to_numpy(float))
            monthly[month] = {
                "parent": pm, "candidate": cm,
                "relative_mdape_improvement_percent": _relative_gain(pm["mdape_percent"], cm["mdape_percent"]),
            }
        gain = _relative_gain(primary["mdape_percent"], candidate["mdape_percent"])
        both_improved = all(monthly[m]["relative_mdape_improvement_percent"] > 0.0 for m in SCREEN_MONTHS)
        track_results[track] = {
            "primary_parent_name": "R52 best experimental candidate" if track == "off_market" else "R35 validated candidate",
            "primary_parent": primary, "candidate": candidate, "validated_R22_reference": validated,
            "relative_mdape_improvement_percent": gain,
            "mape_relative_change_percent": mape_change,
            "monthly": monthly, "both_screen_months_improved": both_improved,
            "qualifies_screen": (gain >= MIN_RELATIVE_MDAPE_GAIN and both_improved and mape_change <= MAX_MAPE_REGRESSION),
            "target_met_on_screen": candidate["mdape_percent"] < (7.0 if track == "off_market" else 4.0),
        }
    if len(removed_features) == 0:
        raise RuntimeError("R64 did not remove any u_* features")

    result = {
        "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id, "card": CARD,
        "screen_months": list(SCREEN_MONTHS), "tracks": track_results,
        "fit_starts": len(fit_details), "fit_details": fit_details,
        "reused_fit_starts": len(reused_fit_details), "reused_fit_details": reused_fit_details,
        "component_manifest": component_manifest,
        "cache_alignment_preflight": cache_preflight,
        "removed_unit_feature_columns": sorted(removed_features),
        "data_sha256": source["data_sha256"],
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    }
    result_path = output / "screen_results.json"
    common._write_json(result_path, result)
    common._write_json(output / "component_manifest.json", component_manifest)

    hgbr_artifacts = {
        month: {"path": item["path"], "csv_sha256": item["csv_sha256"]}
        for month, item in hgbr_sources.items()
    }

    manifest = {
        "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
        "source_identity": source, "code_snapshots": code_map,
        "data_sha256": source["data_sha256"], "fit_starts": len(fit_details),
        "fit_details": fit_details,
        "reused_fit_starts": len(reused_fit_details), "reused_fit_details": reused_fit_details,
        "reused_source_run_id": reused["run_id"],
        "reused_checkpoint_sha256": reused["checkpoint_sha256"],
        "component_manifest": component_manifest,
        "cache_alignment_preflight": cache_preflight,
        "parent_artifacts": card["parent_artifact_hashes"],
        "frozen_hgbr_artifacts": hgbr_artifacts,
        "removed_unit_feature_columns": sorted(removed_features),
        "paired_predictions_sha256": common._sha256(paired_path),
        "result_sha256": common._sha256(result_path),
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    }
    manifest_path = output / "run_manifest.json"
    common._write_json(manifest_path, manifest)
    report = [
        "# R64 current-parent coordinate unit-history ablation cache-alignment continuation", "",
        f"Run: `{run_id}`; new fits: {len(fit_details)}; reused R63 fits: {len(reused_fit_details)}; removed `u_*` features: {len(removed_features)}.",
        "Only coordinate-keyed unit-history columns were removed. `b_*`, remarks TF-IDF, model configurations, routes, targets and blends were kept fixed.",
        "Off-market is screened against R52; list-aware is screened against R35. The R22 predictions are also reported as a secondary reference.",
        "Only 2023-03 and 2024-06 were evaluated. The loader applies a date filter before reading; no reserve or hidden labels were accessed.", "",
        "| Track | Primary parent MdAPE | Candidate MdAPE | Relative gain | MAPE change | Both months improved | Gate | Target on screen |",
        "|---|---:|---:|---:|---:|---|---|---|",
    ]
    for track, score in track_results.items():
        report.append(
            f"| {track} | {score['primary_parent']['mdape_percent']:.5f}% | {score['candidate']['mdape_percent']:.5f}% | "
            f"{score['relative_mdape_improvement_percent']:+.3f}% | {score['mape_relative_change_percent']:+.3f}% | "
            f"{'yes' if score['both_screen_months_improved'] else 'no'} | "
            f"{'pass' if score['qualifies_screen'] else 'fail'} | {'yes' if score['target_met_on_screen'] else 'no'} |"
        )
    report.extend(["", "## Monthly results", "", "| Track | Month | Parent MdAPE | Candidate MdAPE | Relative gain |", "|---|---|---:|---:|---:|"])
    for track, score in track_results.items():
        for month, item in score["monthly"].items():
            report.append(
                f"| {track} | {month} | {item['parent']['mdape_percent']:.5f}% | "
                f"{item['candidate']['mdape_percent']:.5f}% | {item['relative_mdape_improvement_percent']:+.3f}% |"
            )
    (output / "research_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    qualifies = {track: score["qualifies_screen"] for track, score in track_results.items()}
    card.update({
        "status": "screen_passed" if any(qualifies.values()) else "screen_failed",
        "run_id": run_id, "fit_starts": len(fit_details),
        "reused_fit_starts": len(reused_fit_details),
        "screen_result_path": str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        "paired_predictions_path": str(paired_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        "manifest_path": str(manifest_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        "report_path": str((output / "research_report.md").relative_to(campaign.ROOT)).replace("\\", "/"),
        "qualifies_by_track": qualifies, "screen_metrics": track_results,
    })
    _save_protocol()
    campaign._append({
        "event": "card_complete", "card": CARD, "run_id": run_id,
        "fit_starts": len(fit_details), "qualifies_by_track": qualifies,
        "result_path": str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        "paired_predictions_sha256": manifest["paired_predictions_sha256"],
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    passing = [track for track, ok in qualifies.items() if ok]
    next_action = (
        f"run eight-month R64 development confirmation for {passing}; keep validated parents until confirmation passes"
        if passing else "R64 failed current-parent unit-history ablation screen; retain R22/R35 and register a different coordinate/comparable mechanism"
    )
    campaign._write_progress(
        "r64_screen_passed" if passing else "r64_screen_failed",
        next_action=next_action, r64_run_id=run_id,
        r64_result=str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        r64_fit_starts=len(fit_details), r64_reused_fit_starts=len(reused_fit_details),
    )
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Register or run R64 current-parent unit-history ablation continuation")
    parser.add_argument("--register", action="store_true")
    args = parser.parse_args()
    print(json.dumps(register_card() if args.register else run(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
