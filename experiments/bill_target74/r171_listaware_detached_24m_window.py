"""R171: screen a 24-month rolling training window in the R35 Detached expert route."""
from __future__ import annotations

import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.bill_target74.r19_expert_weight import _sha256, _write_json


CARD = "R171_listaware_detached_24m_window"
VARIANT = CARD
PARENT_CARD = "R35_listaware_detached_r22_text_expert"
TRACK = "list_aware"
HOUSE_TYPE = "Detached"
SCREEN_MONTHS = ("2023-03", "2024-06")
CORE_WEIGHT = 0.75
MODEL_OVERRIDES = {}
WINDOW_MONTHS = 24


def _source_files(campaign) -> list[Path]:
    return [
        campaign.PROTOCOL_PATH,
        Path(__file__),
        campaign.PACKAGE / "r171_independent_audit.py",
        campaign.PACKAGE / "r34_house_type_text_experts.py",
        campaign.PACKAGE / "r22_remarks_tfidf.py",
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
        campaign.ROOT / "model_candidate_v2/winner_config.json",
        campaign.ROOT / "model_candidate_v2/models/frozen_list_aware.json",
    ]


def _safe_window_preflight(campaign) -> dict:
    """Count only metadata rows in the exact past-only training windows."""
    reserved = set(campaign.PROTOCOL["reserved_months"])
    hidden = set(campaign.PROTOCOL.get("hidden_months", []))
    windows = []
    for month in SCREEN_MONTHS:
        cutoff = pd.Timestamp(month + "-01")
        start = cutoff - pd.DateOffset(months=WINDOW_MONTHS)
        covered = pd.period_range(start=start, end=cutoff - pd.offsets.MonthBegin(1), freq="M").astype(str)
        if set(covered) & (reserved | hidden):
            raise RuntimeError("R171 rolling window intersects restricted labels")
        frame = pd.read_parquet(
            campaign.DATA,
            columns=["date_sold", "house_type_name"],
            filters=[[ ("date_sold", ">=", start), ("date_sold", "<", cutoff) ]],
        )
        dates = pd.to_datetime(frame.date_sold, errors="raise")
        if frame.empty or (dates < start).any() or (dates >= cutoff).any():
            raise RuntimeError(f"R171 metadata preflight failed exact window for {month}")
        detached = frame.house_type_name.astype("string").eq(HOUSE_TYPE)
        if int(detached.sum()) < 500:
            raise RuntimeError(f"R171 has too few Detached training rows in {month}'s 24-month window")
        windows.append({
            "target_month": month, "window_start_inclusive": str(start.date()),
            "cutoff_exclusive": str(cutoff.date()), "training_rows": int(len(frame)),
            "detached_training_rows": int(detached.sum()),
        })
    return {
        "columns_read": ["date_sold", "house_type_name"],
        "window_months": WINDOW_MONTHS, "screen_windows": windows,
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    }

def _parent_artifacts(campaign) -> tuple[dict, Path, Path, dict, pd.DataFrame]:
    parent_card = campaign.PROTOCOL["cards"].get(PARENT_CARD)
    if not parent_card or parent_card.get("status") != "fixed_origin_confirmation_passed":
        raise RuntimeError("R171 requires the validated R35 list-aware parent")
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
        raise RuntimeError(f"R171 is already {existing.get('status')}; inspect its artifacts before retrying")
    if tuple(campaign.PROTOCOL["screen_months"]) != SCREEN_MONTHS:
        raise RuntimeError("R171 screen months differ from the locked campaign screen")
    limits = campaign.PROTOCOL["limits"]
    if any(value is not None for key, value in limits.items() if key != "learner_threads"):
        raise RuntimeError("R171 must use the user's active no-fit/time-cap campaign protocol")
    parent = campaign.PROTOCOL["cards"].get(PARENT_CARD)
    if not parent or parent.get("status") != "fixed_origin_confirmation_passed":
        raise RuntimeError("R171 requires the validated R35 list-aware parent")
    source = budget._source_identity()
    preflight = _safe_window_preflight(campaign)
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
        "target": "log(sold_price/list_price), unchanged from R35",
        "training_window_months": WINDOW_MONTHS,
        "preflight": preflight,
        "method": (
            "Keep the validated R35 list-aware parent unchanged outside Detached homes. Fit only a Detached "
            "LightGBM text expert on Detached rows sold within the 24 calendar months before each screen cutoff; "
            "blend the expert 75/25 with the same frozen HGBR prediction on Detached rows and overlay it on R35. "
            "Keep the target, features, learner configuration, rounding, and every non-Detached R35 prediction fixed."
        ),
        "difference_from_prior_work": (
            "R77 tested recency sample weights and R162 used an 18-month window for a CatBoost residual model. "
            "R171 instead tests one fixed 24-month cutoff for R35's primary Detached LightGBM expert, with no "
            "recency weights, target change, or feature change."
        ),
        "incremental_screen_gate": campaign.PROTOCOL["gates"],
        "policy": (
            "Use only strict pre-month rows from the two registered development folds; read sale-price labels "
            "only for those safe screen months; fit TF-IDF on list-aware training remarks in that fold; verify "
            "the exact time window and frozen HGBR row alignment; preserve every R35 non-Detached "
            "prediction and retain R35 unless every screen gate passes. Never read reserved or hidden labels."
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
        "preflight": preflight,
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    campaign._write_progress(
        "r171_registered",
        next_action="run preregistered R171 two-month Detached 24-month-window screen; retain R35 unless every gate passes",
        r171_card=CARD,
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
        raise RuntimeError("R171 must be preregistered before fitting")
    source = budget._source_identity()
    if source["data_sha256"] != card["data_sha256"]:
        raise RuntimeError("R171 data snapshot changed after preregistration")
    parent_result, parent_path, parent_rows_path, parent_manifest, parent_rows = _parent_artifacts(campaign)
    if parent_result["run_id"] != card["parent_run_id"]:
        raise RuntimeError("R171 parent run differs from preregistration")

    run_id = "r171s_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = campaign.CAMPAIGN_ROOT / "batches" / run_id
    output.mkdir(parents=True, exist_ok=False)
    provenance = output / "provenance"
    provenance.mkdir()
    shutil.copy2(parent_path, provenance / "r35_screen_results.json")
    shutil.copy2(parent_rows_path, provenance / "r35_screen_paired_predictions.csv")
    shutil.copy2(parent_path.parent / "run_manifest.json", provenance / "r35_screen_run_manifest.json")

    plan = [(month, "detached_expert") for month in SCREEN_MONTHS]
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
    greedy.VARIANTS[VARIANT] = {"model_config_overrides": MODEL_OVERRIDES}
    fit_details: list[dict] = []
    paired_parts: list[pd.DataFrame] = []
    route_details: dict[str, dict] = {}
    active: dict[str, object] = {}
    fit_start_count = 0

    def record_fit(_parent_fit_id: str, fit_fn):
        nonlocal fit_start_count
        month = str(active["month"])
        fit_id = f"{campaign.PROTOCOL['campaign_id']}:{CARD}:screen:{TRACK}:{month}:detached_expert:{run_id}"
        fit_start_count += 1
        campaign._append({
            "event": "fit_start", "card": CARD, "stage": "screen", "run_id": run_id,
            "fit_id": fit_id, "track": TRACK, "month": month,
            "component": "detached_expert", "training_rows": active["training_rows"],
            "evaluation_rows": active["evaluation_rows"], "data_sha256": source["data_sha256"],
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        started = time.perf_counter()
        try:
            fitted = fit_fn()
            if fitted.get("target") != "log_sold_over_list":
                raise RuntimeError(f"R171 expected log_sold_over_list; got {fitted.get('target')!r}")
            if int(fitted.get("training_rows", -1)) != int(active["training_rows"]):
                raise RuntimeError("R171 learner training rows differ from the registered 24-month window")
        except Exception as exc:
            campaign._append({
                "event": "fit_complete", "card": CARD, "stage": "screen", "run_id": run_id,
                "fit_id": fit_id, "track": TRACK, "month": month, "status": "failed",
                "seconds": time.perf_counter() - started, "error": f"{type(exc).__name__}: {exc}",
            })
            raise
        detail = {
            "fit_id": fit_id, "track": TRACK, "month": month,
            "component": "detached_expert", "seconds": time.perf_counter() - started,
            "model_family": fitted["model_family"], "model_config": fitted["model_config"],
            "actual_objective": fitted["actual_objective"], "target": fitted["target"],
            "training_rows": fitted["training_rows"], "evaluation_rows": fitted["evaluation_rows"],
            "training_window_months": WINDOW_MONTHS,
            "training_window_start": active["training_window_start"],
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
                raise RuntimeError(f"R171 training rows are not strictly before {month}")
            homes = hide_answers(heldout, TRACK)
            detached_mask = greedy._specialist_mask(homes, HOUSE_TYPE)
            fit_mask = greedy._specialist_mask(train, HOUSE_TYPE)
            fit_rows = train.loc[fit_mask].copy()
            window_start = cutoff - pd.DateOffset(months=WINDOW_MONTHS)
            train_dates = pd.to_datetime(fit_rows.date_sold, errors="raise")
            fit_rows = fit_rows.loc[train_dates.ge(window_start)].copy()
            if ((pd.to_datetime(fit_rows.date_sold, errors="raise") < window_start).any()
                    or (pd.to_datetime(fit_rows.date_sold, errors="raise") >= cutoff).any()):
                raise RuntimeError(f"R171 Detached training rows violate the 24-month window for {month}")
            eval_rows = homes.loc[detached_mask].copy()
            if len(fit_rows) < 500 or len(eval_rows) < 100 or not (~detached_mask).any():
                raise RuntimeError(f"R171 Detached route support is too small in {month}")

            saved = parent_rows.loc[parent_rows.month.eq(month)].copy()
            saved = saved.set_index(saved.ml_num.astype(str)).reindex(homes.ml_num.astype(str))
            if (saved.index.has_duplicates
                    or not np.array_equal(saved.sold_price.to_numpy(float), truth.sold_price.to_numpy(float))
                    or not np.isfinite(saved.candidate_prediction.to_numpy(float)).all()):
                raise RuntimeError(f"R35 parent predictions do not align for {month}")
            parent_prediction = saved.candidate_prediction.to_numpy(float)

            transformer, restore = r22._install_text_features(fit_rows, eval_rows, TRACK)
            active.update(month=month, training_rows=len(fit_rows), evaluation_rows=len(eval_rows),
                          training_window_start=str(window_start.date()))
            try:
                fitted = greedy._fit_history_candidate(
                    month, TRACK, train, eval_rows, fit_rows=fit_rows, variant=VARIANT,
                    base_variant=None, fit_tag=f"R171-{month}-Detached-24m-window",
                )
            finally:
                restore()
            frozen_hgbr = helpers._verified_hgbr_cache(greedy, month, TRACK, train, homes)
            specialist_prediction = blend_prices(
                fitted["predictions"], frozen_hgbr[detached_mask], challenger_weight=CORE_WEIGHT,
            )
            prediction = parent_prediction.copy()
            prediction[detached_mask] = specialist_prediction
            if not np.array_equal(prediction[~detached_mask], parent_prediction[~detached_mask]):
                raise RuntimeError(f"R171 changed non-Detached R35 predictions in {month}")
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
            actual = truth.sold_price.to_numpy(float)
            route_details[month] = {
                "training_rows": int(len(fit_rows)), "evaluation_rows": int(len(homes)),
                "detached_training_rows": int(len(fit_rows)),
                "detached_evaluation_rows": int(detached_mask.sum()),
                "non_detached_predictions_unchanged": True,
                "training_window_months": WINDOW_MONTHS,
                "training_window_start": str(window_start.date()),
                "training_cutoff_exclusive": cutoff.isoformat(),
                "parent_all": campaign._metrics(actual, parent_prediction),
                "candidate_all": campaign._metrics(actual, prediction),
                "parent_detached": campaign._metrics(actual[detached_mask], parent_prediction[detached_mask]),
                "candidate_detached": campaign._metrics(actual[detached_mask], prediction[detached_mask]),
                "feature_columns": fitted["feature_columns"],
                "tfidf_terms": transformer.vectorizer.get_feature_names_out().tolist(),
                "actual_objective": fitted["actual_objective"], "target": fitted["target"],
            }
            campaign._append({
                "event": "screen_fold_complete", "card": CARD, "run_id": run_id,
                "track": TRACK, "month": month, "training_cutoff_exclusive": cutoff.isoformat(),
                "training_rows": int(len(fit_rows)), "evaluation_rows": int(detached_mask.sum()),
                "training_window_months": WINDOW_MONTHS,
                "training_window_start": str(window_start.date()),
                "non_detached_predictions_unchanged": True,
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
        campaign._write_progress("r171_screen_failed",
                                 next_action="inspect R171 failure checkpoint and ledger before any retry",
                                 r171_failure=str(failure_path.relative_to(campaign.ROOT)).replace("\\", "/"))
        raise
    finally:
        greedy._fit_record = old_fit_record
        if old_variant is None:
            greedy.VARIANTS.pop(VARIANT, None)
        else:
            greedy.VARIANTS[VARIANT] = old_variant

    if fit_start_count != len(plan) or len(fit_details) != len(plan):
        raise RuntimeError(
            f"R171 expected {len(plan)} completed fits; starts={fit_start_count}, completed={len(fit_details)}"
        )
    paired = pd.concat(paired_parts, ignore_index=True)
    paired_path = output / "paired_predictions.csv"
    paired.to_csv(paired_path, index=False)
    score = scoring._track_result(paired)
    result = {
        "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id, "card": CARD,
        "stage": "screen",
        "parent_candidate": parent_result["run_id"], "track": TRACK,
        "screen_months": list(SCREEN_MONTHS), "fit_starts": len(fit_details),
        "tracks": {TRACK: score}, "fit_details": fit_details,
        "route_diagnostics": route_details, "model_config_overrides": MODEL_OVERRIDES,
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
    source_snapshot_paths = {}
    for index, path in enumerate(source_files):
        relative = str(path.relative_to(campaign.ROOT)).replace("\\", "/")
        snapshot = provenance / f"src_{index:02d}_{path.name}.source"
        shutil.copy2(path, snapshot)
        source_snapshot_paths[relative] = str(snapshot.relative_to(campaign.ROOT)).replace("\\", "/")
    manifest = {
        "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
        "card": CARD, "stage": "screen", "source_identity": source,
        "source_sha256": code_hashes, "source_snapshot_paths": source_snapshot_paths,
        "code_sha256": code_hashes,
        "data_sha256": source["data_sha256"], "fit_starts": len(fit_details),
        "fit_details": fit_details, "result_sha256": _sha256(result_path),
        "paired_predictions_sha256": _sha256(paired_path),
        "parent_run_id": parent_result["run_id"],
        "parent_result_sha256": _sha256(parent_path),
        "parent_predictions_sha256": _sha256(parent_rows_path),
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    }
    _write_json(output / "run_manifest.json", manifest)
    report = [
        "# R171: Detached-only 24-month training-window screen", "",
        f"Run: `{run_id}`; parent: validated R35 `{parent_result['run_id']}`.",
        "Only the Detached LightGBM expert's training rows are restricted to the 24 calendar months before each "
        "target cutoff. Features, target, fold-local word TF-IDF, learner configuration, and 75/25 "
        "LightGBM/frozen-HGBR blend are unchanged. Every non-Detached R35 prediction is copied exactly.", "",
        f"- R35 screen MdAPE: {score['parent']['mdape_percent']:.5f}%",
        f"- R171 screen MdAPE: {score['candidate']['mdape_percent']:.5f}%",
        f"- Relative MdAPE improvement: {score['relative_mdape_improvement_percent']:+.3f}%",
        f"- Relative MAPE change: {score['mape_relative_change_percent']:+.3f}%",
        f"- Screen passed: {score['qualifies_screen']}", "",
        "## Monthly results", "", "| Month | R35 MdAPE | R171 MdAPE | Relative improvement |",
        "|---|---:|---:|---:|",
    ]
    for month, values in score["monthly"].items():
        report.append(f"| {month} | {values['parent']['mdape_percent']:.5f}% | "
                      f"{values['candidate']['mdape_percent']:.5f}% | "
                      f"{values['relative_mdape_improvement_percent']:+.3f}% |")
    report.extend(["", "Restricted reserve and hidden labels were not read. A screen failure ends this hypothesis; "
                   "do not tune the feature or model on these months.", ""])
    (output / "research_report.md").write_text("\n".join(report), encoding="utf-8")

    card.update({
        "status": "screen_pending_independent_audit",
        "run_id": run_id, "last_run_id": run_id, "fit_starts": len(fit_details),
        "result_path": str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        "last_result_path": str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        "paired_predictions_path": str(paired_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        "manifest_path": str((output / "run_manifest.json").relative_to(campaign.ROOT)).replace("\\", "/"),
        "screen_metrics": score, "reserved_months_labels_opened": False,
        "hidden_labels_opened": False,
    })
    campaign.PROTOCOL_PATH.write_text(
        json.dumps(campaign.PROTOCOL, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    campaign._append({
        "event": "screen_complete", "card": CARD, "run_id": run_id,
        "fit_starts": len(fit_details), "result_path": card["result_path"],
        "qualifies_screen": score["qualifies_screen"], "audit_pending": True,
        "screen_metrics": score,
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    campaign._write_progress(
        "r171_screen_pending_audit",
        next_action="independently refit and audit R171's two safe screen folds before deciding whether to proceed",
        r171_run_id=run_id, r171_result=card["result_path"],
        r171_fit_starts=len(fit_details),
    )
    return result


def run() -> dict:
    register_card()
    return run_screen()


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, ensure_ascii=False))
