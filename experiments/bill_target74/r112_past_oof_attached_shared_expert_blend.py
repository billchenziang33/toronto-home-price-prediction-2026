"""R112: choose the R110 shared-expert share using strictly earlier OOF rows."""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.bill_target74 import r02_structure_matched_current_parent as parent_verify
from experiments.bill_target74 import r04_ablation as identity
from experiments.bill_target74 import r22_remarks_tfidf as r22
from experiments.bill_target74 import r23_word_char_tfidf as scoring
from experiments.bill_target74 import r36_listaware_semidetached_text as r36
from experiments.bill_target74 import r44_past_oof_r35_shrinkage as r44
from experiments.bill_target74 import r97_past_oof_catboost_residual_meta as r97
from experiments.bill_target74 import r109_seeded_past_oof_catboost_residual_meta as r109
from experiments.bill_target74 import runner as campaign
from experiments.bill_greedy import runner as greedy
from experiments.model_optimization_v2.pipeline import blend_prices


CARD = "R112_past_oof_attached_shared_expert_blend"
TRACK = "list_aware"
HOUSE_TYPES = ("Semi-Detached", "Freehold Townhouse", "Condo Townhouse")
SCREEN_MONTHS = ("2023-03", "2024-06")
HISTORY_FIT_MONTHS = ("2022-06", "2022-11", "2023-07", "2023-11", "2024-03")
ALL_PAST_MONTHS = ("2022-06", "2022-11", "2023-03", "2023-07", "2023-11", "2024-03")
R110_RUN_ID = "r110m_20261025T235931Z"
R44_RUN_ID = "bill_target74_r44_past_oof_r35_blend_20261025T055003Z"
R35_RUN_ID = "bill_target74_r35_house_type_text_20261025T035638Z"
COMPONENT = "attached_shared_core_expert_oof"
MINIMUM_PAST_ROWS = 1000
MAX_MAPE_REGRESSION_PERCENT = 1.0
WEIGHTS = tuple(np.round(np.arange(0.0, 1.0001, 0.05), 2))
EXPECTED_FIT_STARTS = len(HISTORY_FIT_MONTHS)


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


def _mask(frame: pd.DataFrame) -> np.ndarray:
    return frame.house_type_name.astype("string").isin(HOUSE_TYPES).to_numpy(bool)


def _hgbr_cache_source(month: str) -> dict:
    shared = greedy.OUTPUT / "frozen_hgbr" / TRACK / f"{month}.csv"
    if shared.is_file():
        path, source_run_id = shared, "shared_campaign_cache"
    else:
        r44_root = campaign.CAMPAIGN_ROOT / "batches" / R44_RUN_ID
        manifest_path = r44_root / "run_manifest.json"
        path = r44_root / "greedy_cache" / "frozen_hgbr" / TRACK / f"{month}.csv"
        if not manifest_path.is_file() or not path.is_file():
            raise FileNotFoundError(f"R112 has no frozen HGBR cache for {month}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest.get("run_id") != R44_RUN_ID
                or manifest.get("source_identity", {}).get("data_sha256") != campaign.PROTOCOL["data_sha256"]
                or manifest.get("reserved_months_labels_opened") is not False
                or manifest.get("hidden_labels_opened") is not False
                or month not in {"2022-06", "2022-11"}):
            raise RuntimeError(f"R112 R44 cache provenance failed for {month}")
        source_run_id = R44_RUN_ID
    return {
        "path": _relative(path), "sha256": _sha256(path),
        "source_run_id": source_run_id,
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    }


def _read_verified_hgbr_cache(month: str, cache_info: dict, homes: pd.DataFrame) -> np.ndarray:
    path = campaign.ROOT / cache_info["path"]
    if not path.is_file() or _sha256(path) != cache_info["sha256"]:
        raise RuntimeError(f"R112 frozen HGBR cache changed after preflight: {month}")
    cached = pd.read_csv(path, dtype={"ml_num": str})
    expected_ids = homes.ml_num.astype(str).to_numpy()
    if not np.array_equal(cached.ml_num.astype(str).to_numpy(), expected_ids):
        raise RuntimeError(f"R112 frozen HGBR IDs do not align in {month}")
    prediction = cached.raw_prediction.to_numpy(dtype=float)
    if not np.isfinite(prediction).all() or (prediction <= 0.0).any():
        raise RuntimeError(f"R112 frozen HGBR predictions are invalid in {month}")
    return prediction


def _source_rows() -> tuple[pd.DataFrame, dict, pd.DataFrame, dict, dict, dict]:
    source = identity._source_identity()
    parent = r36._verified_r35_parent(campaign)
    if parent["screen"]["run_id"] != R35_RUN_ID:
        raise RuntimeError("R112 did not resolve the expected validated R35 parent")
    seeds, seed_info = r109._verify_r44_seed()
    if "track" not in seeds.columns:
        seeds["track"] = TRACK
    r93_info = parent_verify._verify_r93()
    r93_path = campaign.ROOT / r93_info["predictions_path"]
    r93_rows = pd.read_csv(r93_path, dtype={"ml_num": str})
    r93_rows = r93_rows.loc[r93_rows.track.eq(TRACK)].copy()
    r93_rows["month"] = r93_rows.month.astype(str)
    if set(r93_rows.month.unique()) != set(r97.MONTHS) or r93_rows.duplicated(["month", "ml_num"]).any():
        raise RuntimeError("R112 R93 parent OOF coverage or keys are invalid")

    r110_card = campaign.PROTOCOL["cards"].get("R110_listaware_attached_shared_text_expert", {})
    r110_result_path = campaign.ROOT / r110_card.get("result_path", "")
    r110_pair_path = r110_result_path.parent / "paired_predictions.csv"
    r110_manifest_path = r110_result_path.parent / "run_manifest.json"
    if not all(path.is_file() for path in (r110_result_path, r110_pair_path, r110_manifest_path)):
        raise FileNotFoundError("R112 requires complete R110 screen artifacts")
    r110_result = json.loads(r110_result_path.read_text(encoding="utf-8"))
    r110_manifest = json.loads(r110_manifest_path.read_text(encoding="utf-8"))
    if (r110_result.get("run_id") != R110_RUN_ID
            or r110_result.get("track") != TRACK
            or r110_result.get("reserved_months_labels_opened") is not False
            or r110_result.get("hidden_labels_opened") is not False
            or r110_manifest.get("paired_predictions_sha256") != _sha256(r110_pair_path)
            or r110_manifest.get("result_sha256") != _sha256(r110_result_path)
            or r110_manifest.get("reserved_months_labels_opened") is not False
            or r110_manifest.get("hidden_labels_opened") is not False
            or r110_manifest.get("parent_run_id") != R35_RUN_ID):
        raise RuntimeError("R110 paired OOF source, hashes, or access flags failed verification")
    r110_rows = pd.read_csv(r110_pair_path, dtype={"ml_num": str})
    r110_rows = r110_rows.loc[r110_rows.track.eq(TRACK)].copy()
    r110_rows["month"] = r110_rows.month.astype(str)
    if (set(r110_rows.month.unique()) != set(SCREEN_MONTHS)
            or r110_rows.duplicated(["month", "ml_num"]).any()):
        raise RuntimeError("R110 screen predictions have incomplete months or duplicate keys")
    if set(ALL_PAST_MONTHS) & set(campaign.PROTOCOL["reserved_months"]):
        raise RuntimeError("R112 history month list includes an ineligible reserve month")
    r110_info = {
        "run_id": R110_RUN_ID,
        "result_path": _relative(r110_result_path), "result_sha256": _sha256(r110_result_path),
        "paired_predictions_path": _relative(r110_pair_path), "paired_predictions_sha256": _sha256(r110_pair_path),
        "manifest_path": _relative(r110_manifest_path), "manifest_sha256": _sha256(r110_manifest_path),
    }
    r93_info["parent_run_id"] = r93_info["run_id"]
    r93_info["result_path"] = r93_info.get("result_path")
    r93_info["paired_predictions_path"] = r93_info["predictions_path"]
    return seeds, seed_info, r93_rows, r93_info, r110_rows, {"source": source,
        "parent": parent, "r110_info": r110_info, "r110_result_path": r110_result_path,
        "r110_pair_path": r110_pair_path, "r110_manifest_path": r110_manifest_path,
        "r93_path": r93_path, "r93_manifest_path": campaign.ROOT / r93_info["manifest_path"],
        "r93_result_path": campaign.ROOT / r93_info["result_path"]}


def _preflight() -> dict:
    seeds, seed_info, r93_rows, r93_info, r110_rows, paths = _source_rows()
    expected_seed_rows = {"2022-06": 2352, "2022-11": 1610}
    observed_seed_rows = seeds.groupby("month").size().to_dict()
    if observed_seed_rows != expected_seed_rows:
        raise RuntimeError(f"R112 seed OOF counts changed: {observed_seed_rows}")
    expected_dev_rows = {"2023-03": 2466, "2023-07": 2010, "2023-11": 1625,
                         "2024-03": 2250, "2024-06": 2196, "2024-09": 1825,
                         "2025-01": 1339, "2025-03": 1931}
    observed_dev_rows = r93_rows.groupby("month").size().to_dict()
    if observed_dev_rows != expected_dev_rows:
        raise RuntimeError(f"R112 development OOF row counts changed: {observed_dev_rows}")
    screen_rows = r110_rows.groupby("month").size().to_dict()
    if screen_rows != {"2023-03": 2466, "2024-06": 2196}:
        raise RuntimeError(f"R112 R110 screen row counts changed: {screen_rows}")
    from data import hide_answers
    cache_sources = {month: _hgbr_cache_source(month) for month in HISTORY_FIT_MONTHS}
    for month, cache_info in cache_sources.items():
        _, _, _, _, heldout = greedy._load_fold(month)
        homes = hide_answers(heldout, TRACK).reset_index(drop=True)
        _read_verified_hgbr_cache(month, cache_info, homes)
    return {
        "seeds": seeds, "seed_info": seed_info,
        "r93_rows": r93_rows, "r93_info": r93_info,
        "r110_rows": r110_rows, "paths": paths,
        "history_rows_by_month": {**expected_seed_rows,
                                   **{m: expected_dev_rows[m] for m in ALL_PAST_MONTHS if m in expected_dev_rows}},
        "hgbr_cache_sources": cache_sources,
    }


def _registration_spec(preflight: dict) -> dict:
    return {
        "status": "registered",
        "method": (
            "Generate strict walk-forward OOF predictions for the fixed R110 shared attached-home expert "
            "on the 2022-06/2022-11 seed folds and 2023-07/2023-11/2024-03 historical folds; reuse "
            "the already saved R110 2023-03 and 2024-06 screen predictions. For each screen cutoff, "
            "select the shared-expert share from only strictly earlier paired R35/R110 OOF rows on a "
            "fixed 0.05 grid, minimizing MdAPE subject to at most 1% MAPE regression."
        ),
        "difference_from_prior_work": (
            "R110 used a fixed 75% shared-expert share and failed its screen. R111 stopped after one "
            "historical fit because its 2022-11 HGBR cache was stored in R44's run-specific artifact. "
            "R112 repeats the same past-only weight-selection method after adding explicit cache-source "
            "verification during preflight; it does not tune on either current screen month's labels."
        ),
        "track": TRACK, "parent": "validated R35 list-aware candidate",
        "parent_run_id": R35_RUN_ID,
        "candidate_expert": "R110_listaware_attached_shared_text_expert",
        "r110_source": preflight["paths"]["r110_info"],
        "seed_oof_source": preflight["seed_info"],
        "parent_oof_source": preflight["r93_info"],
        "hgbr_cache_sources": preflight["hgbr_cache_sources"],
        "screen_months": list(SCREEN_MONTHS), "history_oof_months": list(ALL_PAST_MONTHS),
        "weight_selection": {
            "cutoff": "month < current screen month; OOF sale rows strictly before month start",
            "minimum_past_oof_rows": MINIMUM_PAST_ROWS,
            "grid": list(WEIGHTS), "objective": "minimize pooled past-only MdAPE",
            "mape_regression_max_percent": MAX_MAPE_REGRESSION_PERCENT,
            "ties": ["lower MAPE", "smaller expert share"],
            "fallback": 0.0,
        },
        "fit_plan": {"expected_new_fit_starts": EXPECTED_FIT_STARTS,
                     "historical_model_fits": list(HISTORY_FIT_MONTHS),
                     "reused_saved_screen_fits": list(SCREEN_MONTHS)},
        "screen_gate": {
            "pooled_relative_mdape_improvement_percent_min": campaign.PROTOCOL["gates"]["screen_relative_mdape_improvement_percent"],
            "both_months_must_improve": campaign.PROTOCOL["gates"]["screen_both_months_improve"],
            "mape_regression_max_percent": campaign.PROTOCOL["gates"]["screen_mape_regression_max_percent"],
        },
        "data_sha256": preflight["paths"]["source"]["data_sha256"],
        "parent_commit": preflight["paths"]["source"]["commit"],
        "source_sha256": _sha256(Path(__file__)),
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    }


def _register(preflight: dict) -> dict:
    card = _registration_spec(preflight)
    existing = campaign.PROTOCOL["cards"].get(CARD)
    if existing:
        if existing.get("source_sha256") == card["source_sha256"]:
            if existing.get("status") != "registered":
                raise RuntimeError(f"R112 already has status {existing.get('status')!r}")
            return existing
        if existing.get("status") != "screen_failed_runtime":
            raise RuntimeError("R112 source changed after a non-recoverable registered state")
        ledger = [json.loads(line) for line in campaign.LEDGER.read_text(encoding="utf-8").splitlines()
                  if line.strip()]
        prior = [row for row in ledger if row.get("card") == CARD]
        starts = [row for row in prior if row.get("event") == "fit_start"]
        failures = [row for row in prior if row.get("event") == "card_failed"]
        if starts or not failures or any(row.get("fit_starts", 0) != 0 for row in failures):
            raise RuntimeError("R112 cannot amend its source after any fit started")
        card["registration_amendments"] = list(existing.get("registration_amendments", [])) + [{
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "reason": "Zero-fit setup recovery: R44 seed verification returns a normalized frame without track; restore the list-aware track before OOF assembly. Method, weights, models, cutoffs, and gates are unchanged.",
            "previous_source_sha256": existing.get("source_sha256"),
            "current_source_sha256": card["source_sha256"],
            "fit_starts_before_amendment": 0,
            "failed_run_ids": [row.get("run_id") for row in failures],
            "error": failures[-1].get("error"),
        }]
        campaign.PROTOCOL["cards"][CARD] = card
        _write_json(campaign.PROTOCOL_PATH, campaign.PROTOCOL)
        campaign._append({
            "event": "card_registration_amended", "card": CARD, "fit_starts": 0,
            "old_source_sha256": existing.get("source_sha256"),
            "new_source_sha256": card["source_sha256"],
            "reason": "restore list-aware track after zero-fit OOF assembly failure",
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        campaign._write_progress(
            "r112_registered_recovered",
            next_action="Run the registered R112 past-only blend after zero-fit recovery",
            r112_card=CARD,
        )
        return card
    campaign.PROTOCOL["cards"][CARD] = card
    _write_json(campaign.PROTOCOL_PATH, campaign.PROTOCOL)
    campaign._append({
        "event": "card_registered", "card": CARD, "fit_starts": 0,
        "source_sha256": card["source_sha256"], "data_sha256": card["data_sha256"],
        "parent_run_id": R35_RUN_ID, "r110_run_id": R110_RUN_ID,
        "expected_fit_starts": EXPECTED_FIT_STARTS,
        "screen_months": list(SCREEN_MONTHS), "history_oof_months": list(ALL_PAST_MONTHS),
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    campaign._write_progress(
        "r112_registered", next_action="Generate historical OOF expert predictions and score the registered past-only R112 blend",
        r112_card=CARD,
    )
    return card


def _fit_history_fold(month: str, history_parent: pd.DataFrame, output: Path,
                      source: dict, fit_details: list[dict], cache_info: dict) -> pd.DataFrame:
    from data import hide_answers

    _, train, _, truth, heldout = greedy._load_fold(month)
    cutoff = pd.Period(month, freq="M").start_time
    if (pd.to_datetime(train.date_sold, errors="raise") >= cutoff).any():
        raise RuntimeError(f"R112 training rows reach the {month} cutoff")
    homes = hide_answers(heldout, TRACK).reset_index(drop=True)
    route_train, route_eval = _mask(train), _mask(homes)
    fit_rows, eval_rows = train.loc[route_train].copy(), homes.loc[route_eval].copy()
    if len(fit_rows) < 500 or int(route_eval.sum()) < 100:
        raise RuntimeError(f"R112 shared route has insufficient training or evaluation support for {month}")

    base_rows = history_parent.loc[history_parent.month.eq(month)].copy()
    if base_rows.empty or base_rows.duplicated("ml_num").any():
        raise RuntimeError(f"R112 R35 base OOF rows are missing or duplicated for {month}")
    base_by_id = base_rows.set_index("ml_num")
    ids = homes.ml_num.astype(str).to_numpy()
    if not pd.Index(ids).equals(base_by_id.index.sort_values()):
        # The parent file may have another stable order; align explicitly by the unique MLS key.
        if set(ids) != set(base_by_id.index.astype(str)):
            raise RuntimeError(f"R112 R35 baseline IDs do not match the fold rows for {month}")
    base_prediction = base_by_id.loc[ids, "parent_prediction"].to_numpy(float)
    actual = truth.sold_price.to_numpy(float)
    if (not np.isfinite(base_prediction).all() or (base_prediction <= 0).any()
            or not np.isfinite(actual).all() or (actual <= 0).any()
            or not np.array_equal(base_rows.set_index("ml_num").loc[ids, "sold_price"].to_numpy(float), actual)):
        raise RuntimeError(f"R112 parent prediction or truth alignment failed for {month}")

    all_hgbr = _read_verified_hgbr_cache(month, cache_info, homes)
    transformer, restore = r22._install_text_features(fit_rows, eval_rows, TRACK)
    fit_context = {"month": month, "training_rows": len(fit_rows),
                   "evaluation_rows": len(eval_rows), "evaluation_ids": eval_rows.ml_num.astype(str).tolist()}
    old_fit_record = greedy._fit_record
    fit_id = f"{campaign.PROTOCOL['campaign_id']}:{CARD}:{TRACK}:{month}:{COMPONENT}"

    def record_fit(_original_id, fit_fn):
        campaign._append({
            "event": "fit_start", "card": CARD, "run_id": output.name, "fit_id": fit_id,
            "track": TRACK, "month": month, "component": COMPONENT,
            "route_house_types": list(HOUSE_TYPES), "training_rows": len(fit_rows),
            "evaluation_rows": len(eval_rows), "history_cutoff_exclusive": cutoff.isoformat(),
            "data_sha256": source["data_sha256"],
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        started = time.perf_counter()
        try:
            fitted = fit_fn()
        except BaseException as exc:
            campaign._append({
                "event": "fit_complete", "card": CARD, "run_id": output.name,
                "fit_id": fit_id, "track": TRACK, "month": month,
                "status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                "seconds": time.perf_counter() - started,
                "error": f"{type(exc).__name__}: {exc}",
                "reserved_months_labels_opened": False, "hidden_labels_opened": False,
            })
            raise
        fit_path = output / "fit_predictions" / f"{month}_shared_expert.csv"
        pd.DataFrame({"ml_num": fit_context["evaluation_ids"],
                      "prediction": np.asarray(fitted["predictions"], dtype=float)}).to_csv(fit_path, index=False)
        detail = {
            "fit_id": fit_id, "track": TRACK, "month": month, "component": COMPONENT,
            "seconds": time.perf_counter() - started,
            "training_rows": fitted["training_rows"], "evaluation_rows": fitted["evaluation_rows"],
            "model_family": fitted["model_family"], "model_config": fitted["model_config"],
            "target": fitted["target"], "feature_columns": fitted["feature_columns"],
            "text_feature_count": sum(str(x).startswith("remarks_tfidf_") for x in fitted["feature_columns"]),
            "prediction_sha256": _sha256(fit_path),
        }
        fit_details.append(detail)
        campaign._append({
            "event": "fit_complete", "card": CARD, "run_id": output.name,
            **detail, "status": "complete",
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        return fitted

    greedy._fit_record = record_fit
    try:
        try:
            fitted = greedy._fit_history_candidate(
                month, TRACK, train, eval_rows, fit_rows=fit_rows,
                variant=r22.VARIANT, fit_tag=f"R112-{month}-shared-attached",
            )
        finally:
            restore()
    finally:
        greedy._fit_record = old_fit_record

    specialist = blend_prices(fitted["predictions"], all_hgbr[route_eval], challenger_weight=0.75)
    candidate = base_prediction.copy()
    candidate[route_eval] = specialist
    dates = heldout.assign(ml_num=heldout.ml_num.astype(str)).set_index("ml_num").loc[ids, "date_sold"]
    frame = pd.DataFrame({
        "month": month, "track": TRACK, "ml_num": ids,
        "date_sold": pd.to_datetime(dates.to_numpy(), errors="raise"),
        "sold_price": actual, "parent_prediction": base_prediction,
        "variant_prediction": candidate,
    })
    if frame.duplicated(["month", "ml_num"]).any() or not np.isfinite(
            frame[["sold_price", "parent_prediction", "variant_prediction"]].to_numpy(float)).all():
        raise RuntimeError(f"R112 generated invalid OOF predictions for {month}")
    route_info = {
        "month": month, "training_rows": int(len(fit_rows)),
        "evaluation_rows": int(len(eval_rows)),
        "training_rows_by_type": {kind: int(train.house_type_name.astype("string").eq(kind).sum())
                                  for kind in HOUSE_TYPES},
        "cutoff_exclusive": cutoff.isoformat(),
        "tfidf_feature_count": len(transformer.columns_ or []),
        "non_routed_predictions_unchanged": bool(np.array_equal(candidate[~route_eval], base_prediction[~route_eval])),
    }
    frame["history_route_training_rows"] = len(fit_rows)
    frame.to_csv(output / "oof_history" / f"{month}.csv", index=False)
    return frame


def _select_weight(past: pd.DataFrame, cutoff_month: str) -> dict:
    past = past.loc[past.month.astype(str) < cutoff_month].copy()
    if past.empty or past.duplicated(["month", "ml_num"]).any():
        raise RuntimeError(f"R112 past OOF rows are empty or duplicated before {cutoff_month}")
    if (pd.to_datetime(past.date_sold, errors="raise") >= pd.Timestamp(cutoff_month + "-01")).any():
        raise RuntimeError(f"R112 past OOF date crossed the {cutoff_month} cutoff")
    actual = past.sold_price.to_numpy(float)
    parent = past.parent_prediction.to_numpy(float)
    variant = past.variant_prediction.to_numpy(float)
    base_metrics = r40_score(actual, parent)
    if len(past) < MINIMUM_PAST_ROWS:
        return {"weight": 0.0, "past_rows": int(len(past)),
                "past_months": sorted(past.month.astype(str).unique().tolist()),
                "reason": "insufficient_past_oof_rows", "selected": base_metrics}
    eligible = []
    for weight in WEIGHTS:
        pred = np.round((1.0 - weight) * parent + weight * variant)
        metrics = r40_score(actual, pred)
        mape_change = 100.0 * (metrics["mape_percent"] - base_metrics["mape_percent"]) / base_metrics["mape_percent"]
        if mape_change <= MAX_MAPE_REGRESSION_PERCENT:
            eligible.append({"weight": float(weight), "metrics": metrics,
                             "mape_relative_change_percent": float(mape_change)})
    if not eligible:
        return {"weight": 0.0, "past_rows": int(len(past)),
                "past_months": sorted(past.month.astype(str).unique().tolist()),
                "reason": "no_weight_passed_mape_guard", "selected": base_metrics}
    best = min(eligible, key=lambda x: (x["metrics"]["mdape_percent"],
                                        x["metrics"]["mape_percent"], x["weight"]))
    return {"weight": best["weight"], "past_rows": int(len(past)),
            "past_months": sorted(past.month.astype(str).unique().tolist()),
            "reason": "selected_from_strictly_past_oof",
            "selected": best["metrics"],
            "selected_mape_relative_change_percent": best["mape_relative_change_percent"],
            "eligible_weights": [row["weight"] for row in eligible]}


def r40_score(actual: np.ndarray, prediction: np.ndarray) -> dict:
    ape = 100.0 * np.abs(np.asarray(prediction, float) - np.asarray(actual, float)) / np.asarray(actual, float)
    return {"mdape_percent": float(np.median(ape)), "mape_percent": float(np.mean(ape))}


def _run(preflight: dict, card: dict) -> dict:
    run_id = "r112m_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = campaign.CAMPAIGN_ROOT / "batches" / run_id
    output.mkdir(parents=True, exist_ok=False)
    (output / "fit_predictions").mkdir()
    (output / "oof_history").mkdir()
    provenance = output / "provenance"
    provenance.mkdir()
    paths = preflight["paths"]
    snapshots = [
        (campaign.PROTOCOL_PATH, "registered_campaign_protocol.json"),
        (paths["r110_result_path"], "r110_screen_results.json"),
        (paths["r110_pair_path"], "r110_screen_paired_predictions.csv"),
        (paths["r110_manifest_path"], "r110_screen_run_manifest.json"),
        (paths["r93_result_path"], "r93_development_results.json"),
        (paths["r93_path"], "r93_paired_predictions.csv"),
        (paths["r93_manifest_path"], "r93_run_manifest.json"),
    ]
    seed_path = campaign.CAMPAIGN_ROOT / "batches" / r44.CARD / "past_oof_predictions.csv"
    # R44 has a fixed run path; the verified seed helper records that same artifact in seed_info.
    seed_path = campaign.ROOT / preflight["seed_info"]["seed_predictions_path"]
    seed_manifest_path = campaign.ROOT / preflight["seed_info"]["manifest_path"]
    seed_result_path = campaign.ROOT / preflight["seed_info"]["result_path"]
    snapshots.extend([(seed_path, "r44_seed_oof_predictions.csv"),
                      (seed_manifest_path, "r44_seed_run_manifest.json"),
                      (seed_result_path, "r44_seed_screen_results.json")])
    snapshots.extend((campaign.ROOT / info["path"], f"hgbr_cache_{month}.csv")
                     for month, info in preflight["hgbr_cache_sources"].items())
    for src, name in snapshots:
        shutil.copy2(src, provenance / name)
    source_files = [campaign.PROTOCOL_PATH, Path(__file__),
                    campaign.PACKAGE / "r112_past_oof_attached_shared_expert_blend.py",
                    campaign.PACKAGE / "r110_listaware_attached_shared_text_expert.py",
                    campaign.PACKAGE / "r36_listaware_semidetached_text.py",
                    campaign.PACKAGE / "r35_fixed_origin.py",
                    campaign.PACKAGE / "r22_remarks_tfidf.py",
                    campaign.PACKAGE / "r23_word_char_tfidf.py",
                    campaign.PACKAGE / "r03_catboost.py",
                    campaign.PACKAGE / "r04_ablation.py",
                    campaign.PACKAGE / "runner.py",
                    campaign.ROOT / "experiments/bill_greedy/runner.py",
                    campaign.ROOT / "experiments/model_optimization_v2/pipeline.py"]
    code_hashes, code_snapshots = {}, {}
    for i, src in enumerate(source_files):
        key = _relative(src)
        name = f"src_{i:02d}{src.suffix}.source"
        code_hashes[key] = _sha256(src)
        shutil.copy2(src, provenance / name)
        code_snapshots[key] = name

    campaign._append({
        "event": "card_start", "card": CARD, "run_id": run_id,
        "stage": "past_oof_weighted_screen", "screen_months": list(SCREEN_MONTHS),
        "expected_fit_starts": EXPECTED_FIT_STARTS, "history_fit_months": list(HISTORY_FIT_MONTHS),
        "reused_r110_screen_run_id": R110_RUN_ID,
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    source = paths["source"]
    fit_details: list[dict] = []
    try:
        history_parts = []
        r110_rows = preflight["r110_rows"].copy()
        r110_rows = r110_rows.rename(columns={"candidate_prediction": "variant_prediction"})
        r110_rows["date_sold"] = pd.to_datetime(r110_rows.month.astype(str) + "-01", errors="raise")
        for month in ALL_PAST_MONTHS:
            if month in SCREEN_MONTHS:
                fold = r110_rows.loc[r110_rows.month.eq(month)].copy()
                if month == "2023-03":
                    # R110's March rows are strictly OOF and serve as prior evidence for June.
                    history_parts.append(fold[["month", "track", "ml_num", "date_sold", "sold_price",
                                               "parent_prediction", "variant_prediction"]])
                continue
            parent_rows = preflight["seeds"] if month in r44.SEED_MONTHS else preflight["r93_rows"]
            if month in r44.SEED_MONTHS:
                base = parent_rows.loc[parent_rows.month.eq(month), ["month", "track", "ml_num",
                    "date_sold", "sold_price", "base_prediction"]].copy()
                base = base.rename(columns={"base_prediction": "parent_prediction"})
            else:
                base = parent_rows.loc[parent_rows.month.eq(month), ["month", "track", "ml_num",
                    "sold_price", "base_prediction"]].copy()
                base = base.rename(columns={"base_prediction": "parent_prediction"})
                base["date_sold"] = pd.to_datetime(base.month + "-01", errors="raise")
            candidate = _fit_history_fold(
                month, base, output, source, fit_details,
                preflight["hgbr_cache_sources"][month],
            )
            history_parts.append(candidate[["month", "track", "ml_num", "date_sold", "sold_price",
                                            "parent_prediction", "variant_prediction"]])

        history = pd.concat(history_parts, ignore_index=True)
        history["month"] = history.month.astype(str)
        history["ml_num"] = history.ml_num.astype(str)
        history["date_sold"] = pd.to_datetime(history.date_sold, errors="raise")
        if (history.duplicated(["month", "ml_num"]).any()
                or set(history.month.unique()) != set(ALL_PAST_MONTHS)
                or len(history) != 12313):
            raise RuntimeError("R112 historical OOF coverage is incomplete or duplicated")
    except BaseException as exc:
        failure = {
            "run_id": run_id, "card": CARD, "error": f"{type(exc).__name__}: {exc}",
            "fit_starts": len(fit_details), "fit_details": fit_details,
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        }
        _write_json(output / "failure_checkpoint.json", failure)
        campaign._append({
            "event": "card_failed", "card": CARD, "run_id": run_id,
            "fit_starts": len(fit_details), "error": failure["error"],
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        campaign.PROTOCOL["cards"][CARD].update({"status": "screen_failed_runtime", "run_id": run_id})
        _write_json(campaign.PROTOCOL_PATH, campaign.PROTOCOL)
        campaign._write_progress("r112_screen_failed_runtime",
            next_action="Inspect R112 runtime failure; keep all OOF fit records",
            r112_run_id=run_id, r112_error=failure["error"])
        raise

    if len(fit_details) != EXPECTED_FIT_STARTS:
        raise RuntimeError(f"R112 expected {EXPECTED_FIT_STARTS} historical fits; found {len(fit_details)}")
    screen_parts, selected = [], {}
    for month in SCREEN_MONTHS:
        current = r110_rows.loc[r110_rows.month.eq(month)].copy()
        if current.empty:
            raise RuntimeError(f"R112 has no saved R110 predictions for {month}")
        past = history.loc[history.month.astype(str) < month].copy()
        if len(past) < MINIMUM_PAST_ROWS:
            raise RuntimeError(f"R112 has insufficient strictly past history for {month}")
        weight = _select_weight(past, month)
        selected[month] = weight
        blended = np.round(
            (1.0 - weight["weight"]) * current.parent_prediction.to_numpy(float)
            + weight["weight"] * current.variant_prediction.to_numpy(float)
        )
        screen_parts.append(pd.DataFrame({
            "month": month, "track": TRACK,
            "ml_num": current.ml_num.astype(str).to_numpy(),
            "sold_price": current.sold_price.to_numpy(float),
            "parent_prediction": current.parent_prediction.to_numpy(float),
            "candidate_prediction": blended,
            "selected_expert_weight": weight["weight"],
            "past_oof_rows": weight["past_rows"],
            "past_oof_months": ",".join(weight["past_months"]),
        }))
    paired = pd.concat(screen_parts, ignore_index=True)
    if len(paired) != 4662 or paired.duplicated(["month", "track", "ml_num"]).any():
        raise RuntimeError("R112 paired screen rows are incomplete or duplicated")
    metrics = scoring._track_result(paired)
    metrics["selected_weights_by_screen_month"] = {
        m: {"weight": item["weight"], "past_rows": item["past_rows"],
            "past_months": item["past_months"], "reason": item["reason"],
            "selected_past_metrics": item["selected"],
            "eligible_weights": item.get("eligible_weights", [])}
        for m, item in selected.items()
    }
    result = {
        "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
        "card": CARD, "track": TRACK, "parent_run_id": R35_RUN_ID,
        "candidate_expert_run_id": R110_RUN_ID,
        "stage": "strict_past_oof_weighted_screen",
        "screen_months": list(SCREEN_MONTHS), "history_oof_months": list(ALL_PAST_MONTHS),
        "hgbr_cache_sources": preflight["hgbr_cache_sources"],
        "fit_starts": len(fit_details), "fit_details": fit_details,
        "history_oof_rows": int(len(history)), "selected_weights": selected,
        "tracks": {TRACK: metrics},
        "reserved_months_labels_opened": False,
        "reserve_validation_eligible": False, "hidden_labels_opened": False,
    }
    result_path = output / "screen_results.json"
    paired_path = output / "paired_predictions.csv"
    history_path = output / "oof_history_predictions.csv"
    paired.to_csv(paired_path, index=False)
    history.to_csv(history_path, index=False)
    _write_json(result_path, result)
    manifest = {
        "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
        "source_identity": source, "code_sha256": code_hashes,
        "source_snapshots": code_snapshots,
        "input_sha256": {"R44_seed_oof": preflight["seed_info"]["seed_predictions_sha256"],
                         "R93_parent_oof": preflight["r93_info"]["predictions_sha256"],
                         "R110_result": preflight["paths"]["r110_info"]["result_sha256"],
                         "R110_screen_predictions": preflight["paths"]["r110_info"]["paired_predictions_sha256"],
                         "HGBR_caches": {month: info["sha256"] for month, info in
                                         preflight["hgbr_cache_sources"].items()},
                         "data": source["data_sha256"]},
        "parent_run_id": R35_RUN_ID, "candidate_expert_run_id": R110_RUN_ID,
        "screen_months": list(SCREEN_MONTHS), "history_oof_months": list(ALL_PAST_MONTHS),
        "history_fit_months": list(HISTORY_FIT_MONTHS),
        "minimum_past_rows": MINIMUM_PAST_ROWS, "weight_grid": list(WEIGHTS),
        "fit_starts": len(fit_details), "fit_details": fit_details,
        "history_oof_sha256": _sha256(history_path),
        "paired_predictions_sha256": _sha256(paired_path),
        "result_sha256": _sha256(result_path),
        "reserved_months_labels_opened": False,
        "reserve_validation_eligible": False, "hidden_labels_opened": False,
    }
    _write_json(output / "run_manifest.json", manifest)
    status = "screen_passed" if metrics["qualifies_screen"] else "screen_failed"
    campaign.PROTOCOL["cards"][CARD].update({
        "status": status, "run_id": run_id, "fit_starts": len(fit_details),
        "result_path": _relative(result_path), "paired_predictions_path": _relative(paired_path),
        "history_oof_predictions_path": _relative(history_path),
        "qualifies_screen": metrics["qualifies_screen"],
        "screen_metrics": {
            "parent_mdape_percent": metrics["parent"]["mdape_percent"],
            "candidate_mdape_percent": metrics["candidate"]["mdape_percent"],
            "relative_mdape_improvement_percent": metrics["relative_mdape_improvement_percent"],
            "mape_relative_change_percent": metrics["mape_relative_change_percent"],
            "selected_weights_by_screen_month": {
                m: info["weight"] for m, info in selected.items()},
        },
    })
    _write_json(campaign.PROTOCOL_PATH, campaign.PROTOCOL)
    campaign._append({
        "event": "card_complete", "card": CARD, "run_id": run_id,
        "status": status, "fit_starts": len(fit_details),
        "qualifies_screen": metrics["qualifies_screen"],
        "result_path": _relative(result_path),
        "result_sha256": _sha256(result_path),
        "paired_predictions_sha256": _sha256(paired_path),
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    campaign._write_progress(
        "r112_screen_complete",
        next_action=("R112 passed screen; register eight-month walk-forward development confirmation"
                     if status == "screen_passed" else "R112 failed screen; keep R35 and register a distinct hypothesis"),
        r112_run_id=run_id, r112_result=_relative(result_path), r112_fit_starts=len(fit_details),
    )
    return result


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in {"--preflight", "--run"}:
        raise SystemExit("usage: python -m experiments.bill_target74.r112_past_oof_attached_shared_expert_blend --preflight|--run")
    preflight = _preflight()
    preview = {"card": CARD, "R110_run_id": R110_RUN_ID,
               "history_fit_months": list(HISTORY_FIT_MONTHS),
               "history_rows_by_month": preflight["history_rows_by_month"],
               "hgbr_cache_sources": preflight["hgbr_cache_sources"],
               "screen_months": list(SCREEN_MONTHS),
               "reserved_or_hidden_opened": False}
    if sys.argv[1] == "--preflight":
        print(json.dumps(preview, indent=2, ensure_ascii=False))
        return
    card = _register(preflight)
    result = _run(preflight, card)
    score = result["tracks"][TRACK]
    print(json.dumps({"run_id": result["run_id"], "fit_starts": result["fit_starts"],
                      "weights": {m: v["weight"] for m, v in result["selected_weights"].items()},
                      "parent_mdape_percent": score["parent"]["mdape_percent"],
                      "candidate_mdape_percent": score["candidate"]["mdape_percent"],
                      "relative_mdape_improvement_percent": score["relative_mdape_improvement_percent"],
                      "mape_relative_change_percent": score["mape_relative_change_percent"],
                      "qualifies_screen": score["qualifies_screen"],
                      "reserved_months_labels_opened": False, "hidden_labels_opened": False},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
