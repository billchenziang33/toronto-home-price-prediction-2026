"""Reserve-safe fixed-origin confirmation for the frozen off-market R33 blend."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.bill_target74.r19_expert_weight import _sha256, _write_json


CARD = "R33_R22_R32_fixed_half_blend"
R22_FIXED_ORIGIN_RUN = "bill_target74_r22_fixedorigin_20261024T201711Z"
R32_SCREEN_RUN = "bill_target74_r32_comparable_r22_20261025T024940Z"
R33_SCREEN_RUN = "bill_target74_r33_r22_r32_blend_20261025T025855Z"
R33_DEVELOPMENT_RUN = "bill_target74_r33_offmarket_development_20261025T030109Z"
TRACK = "off_market"
VARIANT = "R32_r22_comparable_features"
EXPECTED_NEW_FITS = 4
SEALED_MONTHS = {"2025-05", "2025-08"}


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _verify_artifact(path: Path, expected_hash: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    if not expected_hash or _sha256(path) != expected_hash:
        raise RuntimeError(f"{label} hash mismatch: {path}")


def _load_inputs(campaign, budget, r22, r32) -> dict:
    protocol = campaign.PROTOCOL
    card = protocol["cards"].get(CARD)
    if not card or card.get("status") != "fixed_origin_confirmation_registered":
        raise RuntimeError("R33 fixed-origin confirmation must be registered before running")
    fixed = card.get("fixed_origin_confirmation", {})
    if fixed.get("status") != "registered" or fixed.get("expected_new_fit_starts") != EXPECTED_NEW_FITS:
        raise RuntimeError("R33 fixed-origin protocol is missing or its fit plan changed")
    if tuple(fixed.get("origins", ())) != tuple(protocol["fixed_origins"]):
        raise RuntimeError("R33 fixed-origin dates differ from the campaign protocol")
    if tuple(fixed.get("horizons", ())) != (1, 2, 3, 4):
        raise RuntimeError("R33 fixed-origin horizons differ from the frozen plan")
    if fixed.get("reserved_months_labels_opened") or fixed.get("hidden_labels_opened"):
        raise RuntimeError("R33 fixed-origin protocol must keep sealed labels closed")
    if any(protocol["limits"].get(key) is not None for key in (
        "campaign_fit_starts", "campaign_minutes", "exploration_fit_starts",
        "exploration_minutes", "final_reserve_fit_starts", "final_reserve_minutes",
        "batch_fit_starts", "batch_minutes", "single_fit_seconds",
    )):
        raise RuntimeError("active campaign unexpectedly has a finite training budget")

    source = budget._source_identity()
    r22_dir = campaign.CAMPAIGN_ROOT / "batches" / R22_FIXED_ORIGIN_RUN
    r22_manifest_path = r22_dir / "run_manifest.json"
    r22_result_path = r22_dir / "r22_fixed_origin_results.json"
    r22_pairs_path = r22_dir / "paired_predictions.csv"
    r22_manifest = _read_json(r22_manifest_path)
    r22_result = _read_json(r22_result_path)
    _verify_artifact(r22_result_path, r22_manifest["result_sha256"], "R22 fixed-origin result")
    _verify_artifact(r22_pairs_path, r22_manifest["paired_predictions_sha256"], "R22 fixed-origin predictions")
    if r22_manifest.get("run_id") != R22_FIXED_ORIGIN_RUN:
        raise RuntimeError("R22 fixed-origin manifest has the wrong run ID")
    if r22_manifest.get("source_identity", {}).get("data_sha256") != source["data_sha256"]:
        raise RuntimeError("R22 fixed-origin run uses a different dataset")
    if r22_manifest.get("code_sha256", {}).get("experiments/bill_target74/r22_remarks_tfidf.py") != _sha256(
        campaign.PACKAGE / "r22_remarks_tfidf.py"
    ):
        raise RuntimeError("R22 fixed-origin predictions were produced by different R22 code")
    if r22_result.get("origins") != list(protocol["fixed_origins"]):
        raise RuntimeError("R22 reference result uses different fixed origins")
    if r22_result.get("tracks", {}).get(TRACK, {}).get("passes_fixed_origin") is not True:
        raise RuntimeError("R22 is not a fixed-origin-qualified reference for off-market")

    def verified_run(run_id: str, result_name: str) -> tuple[Path, Path, dict, dict]:
        folder = campaign.CAMPAIGN_ROOT / "batches" / run_id
        manifest_path = folder / "run_manifest.json"
        result_path = folder / result_name
        manifest = _read_json(manifest_path)
        result = _read_json(result_path)
        _verify_artifact(result_path, manifest["result_sha256"], f"{run_id} result")
        if manifest.get("source_identity", {}).get("data_sha256") != source["data_sha256"]:
            raise RuntimeError(f"{run_id} uses a different dataset")
        if result.get("reserved_months_labels_opened") or result.get("hidden_labels_opened"):
            raise RuntimeError(f"{run_id} touched a sealed label set")
        return folder, result_path, manifest, result

    r32_dir, r32_result_path, r32_manifest, r32_result = verified_run(
        R32_SCREEN_RUN, "r32_screen_results.json"
    )
    _verify_artifact(
        r32_dir / "provenance" / "r32_comparable_r22_source.py",
        _sha256(campaign.PACKAGE / "r32_comparable_r22.py"),
        "R32 source snapshot",
    )
    if r32_result.get("fit_starts") != 6 or r32_result.get("card") != "R32_r22_comparable_features":
        raise RuntimeError("R32 screen result does not match the registered comparable model")

    r33_screen_dir, r33_screen_result_path, r33_screen_manifest, r33_screen = verified_run(
        R33_SCREEN_RUN, "r33_screen_results.json"
    )
    _verify_artifact(
        r33_screen_dir / "paired_predictions.csv",
        r33_screen_manifest["paired_predictions_sha256"],
        "R33 screen predictions",
    )
    if r33_screen.get("tracks", {}).get(TRACK, {}).get("qualifies_screen") is not True:
        raise RuntimeError("off-market did not pass the frozen R33 screen")

    r33_dev_dir, r33_dev_result_path, r33_dev_manifest, r33_dev = verified_run(
        R33_DEVELOPMENT_RUN, "r33_off_market_development_results.json"
    )
    _verify_artifact(
        r33_dev_dir / "paired_predictions.csv",
        r33_dev_manifest["paired_predictions_sha256"],
        "R33 development predictions",
    )
    dev_track = r33_dev.get("tracks", {}).get(TRACK, {})
    if (r33_dev.get("run_id") != R33_DEVELOPMENT_RUN
            or dev_track.get("qualifies_development") is not True
            or dev_track.get("target_met") is not False):
        raise RuntimeError("R33 off-market development gate or target state differs from the protocol")
    if r33_dev.get("reserved_months_labels_opened") or r33_dev.get("hidden_labels_opened"):
        raise RuntimeError("R33 development run touched a sealed label set")

    r22_rows = pd.read_csv(r22_pairs_path, dtype={"ml_num": str})
    r22_rows = r22_rows.loc[r22_rows.track.eq(TRACK)].copy()
    if r22_rows.empty or r22_rows.duplicated(["origin", "month", "ml_num"]).any():
        raise RuntimeError("R22 fixed-origin reference has empty or duplicate off-market rows")
    origins = tuple(protocol["fixed_origins"])
    horizons = (1, 2, 3, 4)
    if set(r22_rows.origin.astype(str)) != set(origins) or set(r22_rows.horizon.astype(int)) != set(horizons):
        raise RuntimeError("R22 fixed-origin reference does not cover every registered origin and horizon")
    if set(r22_rows.month.astype(str)) & SEALED_MONTHS:
        raise RuntimeError("R22 fixed-origin reference contains a sealed month")
    for origin in origins:
        cells = r22_rows.loc[r22_rows.origin.astype(str).eq(origin)]
        if set(cells.horizon.astype(int)) != set(horizons):
            raise RuntimeError(f"R22 fixed-origin reference has incomplete horizons at {origin}")

    return {
        "source": source,
        "fixed": fixed,
        "r22_dir": r22_dir,
        "r22_manifest": r22_manifest,
        "r22_result": r22_result,
        "r22_result_path": r22_result_path,
        "r22_pairs_path": r22_pairs_path,
        "r22_rows": r22_rows,
        "r32_result_path": r32_result_path,
        "r32_manifest": r32_manifest,
        "r33_screen_result_path": r33_screen_result_path,
        "r33_screen_manifest": r33_screen_manifest,
        "r33_dev_result_path": r33_dev_result_path,
        "r33_dev_manifest": r33_dev_manifest,
        "origins": origins,
        "horizons": horizons,
    }


def preflight() -> dict:
    from experiments.bill_target74 import r04_ablation as budget
    from experiments.bill_target74 import r22_remarks_tfidf as r22
    from experiments.bill_target74 import runner as campaign
    from experiments.bill_target74 import r32_comparable_r22 as r32

    inputs = _load_inputs(campaign, budget, r22, r32)
    return {
        "campaign_id": campaign.PROTOCOL["campaign_id"],
        "data_sha256": inputs["source"]["data_sha256"],
        "reference_run_id": R22_FIXED_ORIGIN_RUN,
        "reference_rows": int(len(inputs["r22_rows"])),
        "origins": list(inputs["origins"]),
        "horizons": list(inputs["horizons"]),
        "expected_new_fit_starts": EXPECTED_NEW_FITS,
        "training_caps_unbounded": True,
        "reserved_months_labels_opened": False,
        "hidden_labels_opened": False,
    }


def _row_id_hash(frame: pd.DataFrame) -> str:
    ids = sorted(frame.ml_num.astype(str).tolist())
    return _sha256_text("\n".join(ids))


def _fit_r32_component(greedy, r22, run_id: str, source: dict, origin: str,
                       component: str, fit_rows: pd.DataFrame, eval_rows: pd.DataFrame,
                       all_train: pd.DataFrame, run_output: Path,
                       fit_details: list[dict], component_manifest: list[dict],
                       fit_ids_started: list[str]) -> dict:
    old_fit_record = greedy._fit_record

    def record_fit(_original_fit_id, fit_fn):
        fit_id = (f"bill_target74:{CARD}:fixed_origin:{origin}:{TRACK}:"
                  f"R32_{component}:{run_id}")
        fit_start = {
            "event": "fit_start", "card": CARD,
            "stage": "fixed_origin_confirmation", "run_id": run_id,
            "fit_id": fit_id, "track": TRACK, "origin": origin,
            "component": f"R32_{component}",
            "data_sha256": source["data_sha256"],
            "training_rows": int(len(fit_rows)),
            "evaluation_rows": int(len(eval_rows)),
            "training_row_ids_sha256": _row_id_hash(fit_rows),
            "evaluation_row_ids_sha256": _row_id_hash(eval_rows),
            "reserved_months_labels_opened": False,
            "hidden_labels_opened": False,
        }
        from experiments.bill_target74 import runner as campaign

        campaign._append(fit_start)
        fit_ids_started.append(fit_id)
        started = time.perf_counter()
        try:
            fitted = fit_fn()
            prediction = np.asarray(fitted["predictions"], dtype=float)
            if prediction.shape != (len(eval_rows),) or not np.isfinite(prediction).all() or (prediction <= 0).any():
                raise RuntimeError(f"invalid R32 predictions for {origin}/{component}")
            prediction_path = run_output / "component_predictions" / f"{origin[:7]}_{component}.csv"
            prediction_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"ml_num": eval_rows.ml_num.astype(str).to_numpy(),
                          "prediction": prediction}).to_csv(prediction_path, index=False)
        except Exception as exc:
            seconds = time.perf_counter() - started
            campaign._append({"event": "fit_complete", "card": CARD,
                              "stage": "fixed_origin_confirmation", "run_id": run_id,
                              "fit_id": fit_id, "track": TRACK, "origin": origin,
                              "component": f"R32_{component}", "status": "failed",
                              "seconds": seconds, "error": f"{type(exc).__name__}: {exc}"})
            raise
        seconds = time.perf_counter() - started
        detail = {
            **{key: value for key, value in fit_start.items() if key != "event"},
            "seconds": seconds,
            "model_family": fitted.get("model_family"),
            "model_config": fitted.get("model_config"),
            "target": fitted.get("target"),
            "actual_objective": fitted.get("actual_objective"),
            "feature_columns": fitted.get("feature_columns", []),
            "tfidf_feature_count": sum(name.startswith("remarks_tfidf_")
                                        for name in fitted.get("feature_columns", [])),
            "neighbor_comparable_feature_count": sum(name.startswith("nn_")
                                                       for name in fitted.get("feature_columns", [])),
            "prediction_sha256": fitted["prediction_sha256"],
            "saved_predictions": str(prediction_path.relative_to(campaign.ROOT)).replace("\\", "/"),
            "saved_predictions_sha256": _sha256(prediction_path),
        }
        fit_details.append(detail)
        component_manifest.append({
            "origin": origin,
            "component": f"R32_{component}",
            "training_rows": int(len(fit_rows)),
            "training_max_date_sold": str(pd.to_datetime(fit_rows.date_sold).max().date()),
            "training_row_ids_sha256": _row_id_hash(fit_rows),
            "evaluation_rows": int(len(eval_rows)),
            "evaluation_row_ids_sha256": _row_id_hash(eval_rows),
            "tfidf_vocabulary": [],
            "prediction_sha256": fitted["prediction_sha256"],
            "saved_predictions_sha256": _sha256(prediction_path),
        })
        campaign._append({"card": CARD,
                          "stage": "fixed_origin_confirmation", "run_id": run_id,
                          **detail, "status": "complete", "event": "fit_complete"})
        return fitted

    greedy._fit_record = record_fit
    try:
        transformer, restore = r22._install_text_features(fit_rows, eval_rows, TRACK)
        try:
            result = greedy._fit_history_candidate(
                pd.Timestamp(origin).strftime("%Y-%m"), TRACK, all_train, eval_rows,
                cutoff=pd.Timestamp(origin),
                fit_tag=f"R33-fixed-origin-R32-{component}-{origin}",
                variant=VARIANT, base_variant="G06_lot_garage", fit_rows=fit_rows,
            )
        finally:
            restore()
        component_manifest[-1]["tfidf_vocabulary"] = transformer.vectorizer.get_feature_names_out().tolist()
        component_manifest[-1]["tfidf_feature_count"] = len(transformer.columns_ or [])
        return result
    finally:
        greedy._fit_record = old_fit_record


def _update_protocol(campaign, card_status: str, **updates) -> None:
    protocol_path = campaign.PROTOCOL_PATH
    protocol = _read_json(protocol_path)
    card = protocol["cards"][CARD]
    card["status"] = card_status
    card.setdefault("fixed_origin_confirmation", {}).update(updates)
    _write_json(protocol_path, protocol)
    campaign.PROTOCOL.clear()
    campaign.PROTOCOL.update(protocol)


def run_confirmation() -> dict:
    from data import hide_answers
    from experiments.bill_target74 import r04_ablation as budget
    from experiments.bill_target74 import r22_remarks_tfidf as r22
    from experiments.bill_target74 import runner as campaign
    from experiments.bill_target74 import r32_comparable_r22 as r32
    from experiments.bill_greedy import runner as greedy

    inputs = _load_inputs(campaign, budget, r22, r32)
    source = inputs["source"]
    run_id = "bill_target74_r33_fixedorigin_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_output = campaign.CAMPAIGN_ROOT / "batches" / run_id
    run_output.mkdir(parents=True, exist_ok=False)
    (run_output / "component_predictions").mkdir()
    protocol_snapshot = run_output / "provenance" / "campaign_protocol_preregistered.json"
    protocol_snapshot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(campaign.PROTOCOL_PATH, protocol_snapshot)
    protocol_hash = _sha256(protocol_snapshot)
    _update_protocol(
        campaign, "fixed_origin_confirmation_running",
        status="running", run_id=run_id,
    )

    fit_details: list[dict] = []
    component_manifest: list[dict] = []
    fit_ids_started: list[str] = []
    paired_parts: list[pd.DataFrame] = []
    old_variant = greedy.VARIANTS.get(VARIANT)
    greedy.VARIANTS[VARIANT] = dict(r32.VARIANT_SPEC)

    campaign._append({
        "event": "card_start", "card": CARD,
        "stage": "fixed_origin_confirmation", "run_id": run_id,
        "track": TRACK, "origins": list(inputs["origins"]),
        "horizons": list(inputs["horizons"]),
        "expected_fit_starts": EXPECTED_NEW_FITS,
        "reference_run_id": R22_FIXED_ORIGIN_RUN,
        "reference_result_sha256": _sha256(inputs["r22_result_path"]),
        "reference_predictions_sha256": _sha256(inputs["r22_pairs_path"]),
        "development_run_id": R33_DEVELOPMENT_RUN,
        "development_result_sha256": _sha256(inputs["r33_dev_result_path"]),
        "screen_run_id": R33_SCREEN_RUN,
        "variant_run_id": R32_SCREEN_RUN,
        "blend_weight": {"R22": 0.5, "R32": 0.5},
        "data_sha256": source["data_sha256"],
        "protocol_snapshot_sha256": protocol_hash,
        "reserved_months_labels_opened": False,
        "hidden_labels_opened": False,
    })
    campaign._write_progress(
        "r33_fixed_origin_confirmation_running",
        next_action="finish the registered R33 off-market fixed-origin confirmation; keep reserve and hidden labels sealed",
        r33_fixed_origin_run_id=run_id,
        latest_fit_count=0,
    )

    try:
        for origin_text in inputs["origins"]:
            origin = pd.Timestamp(origin_text)
            end = origin + pd.DateOffset(months=4)
            train = r22.load_fixed_origin_training(campaign.DATA, origin_text)
            if set(train.date_sold.dt.to_period("M").astype(str)) & SEALED_MONTHS:
                raise RuntimeError(f"sealed training labels entered the R33 fixed-origin fit at {origin_text}")
            if (train.date_sold >= origin).any():
                raise RuntimeError(f"training data is not strictly earlier than origin {origin_text}")
            heldout = pd.read_parquet(
                campaign.DATA,
                filters=[("date_sold", ">=", origin), ("date_sold", "<", end)],
            )
            heldout["date_sold"] = pd.to_datetime(heldout.date_sold, errors="raise")
            heldout["ml_num"] = heldout.ml_num.astype(str)
            if heldout.empty or heldout.ml_num.duplicated().any():
                raise RuntimeError(f"invalid fixed-origin evaluation rows at {origin_text}")
            months = set(heldout.date_sold.dt.to_period("M").astype(str))
            expected_months = {(origin.to_period("M") + offset).strftime("%Y-%m")
                               for offset in range(4)}
            if months != expected_months:
                raise RuntimeError(f"incomplete fixed-origin horizon at {origin_text}: {sorted(months)}")
            if months & SEALED_MONTHS:
                raise RuntimeError(f"sealed evaluation labels entered the R33 fixed-origin run at {origin_text}")

            homes = hide_answers(heldout, TRACK).reset_index(drop=True)
            homes["ml_num"] = homes.ml_num.astype(str)
            ids = homes.ml_num.to_numpy()
            reference = inputs["r22_rows"].loc[
                inputs["r22_rows"].origin.astype(str).eq(origin_text)
            ].copy()
            if reference.ml_num.astype(str).duplicated().any():
                raise RuntimeError(f"duplicate R22 reference IDs at {origin_text}")
            ref_indexed = reference.set_index(reference.ml_num.astype(str), drop=False)
            aligned_reference = ref_indexed.reindex(ids)
            if aligned_reference.ml_num.isna().any():
                raise RuntimeError(f"R22 reference does not cover evaluation IDs at {origin_text}")
            sold = heldout.sold_price.to_numpy(float)
            expected_ref_sold = aligned_reference.sold_price.to_numpy(float)
            expected_ref_month = aligned_reference.month.astype(str).to_numpy()
            expected_month = heldout.date_sold.dt.to_period("M").astype(str).to_numpy()
            expected_horizon = greedy._month_horizons(heldout.date_sold, origin)
            if not np.array_equal(sold, expected_ref_sold):
                raise RuntimeError(f"R22 reference labels do not match source data at {origin_text}")
            if not np.array_equal(expected_month, expected_ref_month):
                raise RuntimeError(f"R22 reference sale months do not match source data at {origin_text}")
            if not np.array_equal(expected_horizon, aligned_reference.horizon.to_numpy(int)):
                raise RuntimeError(f"R22 reference horizons do not match source data at {origin_text}")
            r22_prediction = aligned_reference.candidate_prediction.to_numpy(float)
            if not np.isfinite(r22_prediction).all() or (r22_prediction <= 0).any():
                raise RuntimeError(f"R22 candidate predictions are invalid at {origin_text}")

            detached_train = greedy._specialist_mask(train, "Detached")
            detached_homes = greedy._specialist_mask(homes, "Detached")
            if not detached_train.any() or not detached_homes.any():
                raise RuntimeError(f"Detached specialist lacks rows at {origin_text}")

            base = _fit_r32_component(
                greedy, r22, run_id, source, origin_text, "base", train, homes,
                train, run_output, fit_details, component_manifest, fit_ids_started,
            )
            expert_rows = train.loc[detached_train].copy()
            expert_homes = homes.loc[detached_homes].copy()
            expert = _fit_r32_component(
                greedy, r22, run_id, source, origin_text, "expert", expert_rows,
                expert_homes, train, run_output, fit_details, component_manifest, fit_ids_started,
            )
            r32_prediction = greedy._blend_specialist_predictions(
                np.round(base["predictions"]), np.round(expert["predictions"]),
                homes, "Detached", 0.75,
            )
            r33_prediction = np.round(0.5 * r22_prediction + 0.5 * r32_prediction)
            if not np.isfinite(r33_prediction).all() or (r33_prediction <= 0).any():
                raise RuntimeError(f"invalid R33 blend at {origin_text}")

            paired = pd.DataFrame({
                "month": expected_month,
                "track": TRACK,
                "ml_num": ids,
                "sold_price": sold,
                "parent_prediction": r22_prediction,
                "candidate_prediction": r33_prediction,
                "origin": origin_text,
                "horizon": expected_horizon,
            })
            paired_parts.append(paired)
            pd.concat(paired_parts, ignore_index=True).to_csv(
                run_output / "partial_off_market.csv", index=False
            )
            campaign._append({
                "event": "fixed_origin_complete", "card": CARD,
                "stage": "fixed_origin_confirmation", "run_id": run_id,
                "track": TRACK, "origin": origin_text,
                "training_rows": int(len(train)),
                "training_max_date_sold": str(train.date_sold.max().date()),
                "training_intervals": [list(interval) for interval in
                                       r22.fixed_origin_training_intervals(origin_text)],
                "evaluation_rows": int(len(heldout)),
                "evaluation_months": sorted(months),
                "R22_reference_ids_and_labels_equal": True,
                "reserved_months_labels_opened": False,
                "hidden_labels_opened": False,
            })
    except Exception as exc:
        failure = {
            "run_id": run_id, "card": CARD,
            "stage": "fixed_origin_confirmation",
            "error": f"{type(exc).__name__}: {exc}",
            "fit_starts": len(fit_ids_started),
            "fit_ids_started": fit_ids_started,
            "fit_details": fit_details,
            "completed_components": [f"{row['origin']}/{row['component']}" for row in component_manifest],
            "reserved_months_labels_opened": False,
            "hidden_labels_opened": False,
        }
        _write_json(run_output / "failure_checkpoint.json", failure)
        campaign._append({"event": "card_failed", "card": CARD,
                          "stage": "fixed_origin_confirmation", "run_id": run_id,
                          "fit_starts": len(fit_ids_started), "error": failure["error"]})
        campaign._write_progress(
            "r33_fixed_origin_confirmation_failed",
            next_action="audit R33 fixed-origin failure checkpoint and fit ledger before any retry",
            r33_fixed_origin_failure=str((run_output / "failure_checkpoint.json").relative_to(
                campaign.ROOT)).replace("\\", "/"),
        )
        _update_protocol(campaign, "fixed_origin_confirmation_failed",
                         status="failed", run_id=run_id,
                         failure_path=str((run_output / "failure_checkpoint.json").relative_to(
                             campaign.ROOT)).replace("\\", "/"),
                         fit_starts=len(fit_ids_started))
        raise
    finally:
        if old_variant is None:
            greedy.VARIANTS.pop(VARIANT, None)
        else:
            greedy.VARIANTS[VARIANT] = old_variant

    if len(fit_details) != EXPECTED_NEW_FITS:
        raise RuntimeError(f"R33 fixed-origin expected {EXPECTED_NEW_FITS} fits; observed {len(fit_details)}")
    paired = pd.concat(paired_parts, ignore_index=True)
    if paired.duplicated(["origin", "month", "ml_num"]).any():
        raise RuntimeError("R33 fixed-origin output contains duplicate rows")
    track_score = r22.score_fixed_origin_confirmation(
        paired,
        origins=inputs["origins"], horizons=inputs["horizons"],
        maximum_pooled_mdape_regression_percent=campaign.PROTOCOL["gates"][
            "fixed_origin_pooled_mdape_regression_max_percent"],
        maximum_horizon_mdape_regression_percent=campaign.PROTOCOL["gates"][
            "fixed_origin_horizon_regression_max_percent"],
        maximum_mape_regression_percent=campaign.PROTOCOL["gates"][
            "fixed_origin_mape_regression_max_percent"],
    )
    paired_path = run_output / "paired_predictions.csv"
    paired.to_csv(paired_path, index=False)
    result = {
        "campaign_id": campaign.PROTOCOL["campaign_id"],
        "run_id": run_id,
        "card": CARD,
        "stage": "fixed_origin_confirmation",
        "track": TRACK,
        "reference_candidate": "R22_train_fold_tfidf_remarks",
        "reference_run_id": R22_FIXED_ORIGIN_RUN,
        "development_run_id": R33_DEVELOPMENT_RUN,
        "screen_run_id": R33_SCREEN_RUN,
        "variant_run_id": R32_SCREEN_RUN,
        "origins": list(inputs["origins"]),
        "horizons": list(inputs["horizons"]),
        "fit_starts": len(fit_details),
        "blend_weight": {"R22": 0.5, "R32": 0.5},
        "training_policy": inputs["fixed"]["training_policy"],
        "tracks": {TRACK: track_score},
        "fit_details": fit_details,
        "component_manifest": component_manifest,
        "data_sha256": source["data_sha256"],
        "reserved_months_labels_opened": False,
        "hidden_labels_opened": False,
    }
    result_path = run_output / "r33_fixed_origin_results.json"
    _write_json(result_path, result)

    source_files = [
        campaign.PROTOCOL_PATH,
        campaign.PACKAGE / "r33_fixed_origin.py",
        campaign.PACKAGE / "r32_comparable_r22.py",
        campaign.PACKAGE / "r22_remarks_tfidf.py",
        campaign.PACKAGE / "r04_ablation.py",
        campaign.PACKAGE / "r03_catboost.py",
        campaign.PACKAGE / "runner.py",
        campaign.ROOT / "model_candidate_v2/features.py",
        campaign.ROOT / "experiments/bill_greedy/runner.py",
        campaign.ROOT / "experiments/bill_greedy/history.py",
        campaign.ROOT / "experiments/bill_greedy/comparables.py",
        campaign.ROOT / "experiments/model_optimization_v2/learners.py",
    ]
    input_hashes = {
        "r22_fixed_origin_manifest_sha256": _sha256(inputs["r22_dir"] / "run_manifest.json"),
        "r22_fixed_origin_result_sha256": _sha256(inputs["r22_result_path"]),
        "r22_fixed_origin_predictions_sha256": _sha256(inputs["r22_pairs_path"]),
        "r32_screen_result_sha256": _sha256(inputs["r32_result_path"]),
        "r33_screen_result_sha256": _sha256(inputs["r33_screen_result_path"]),
        "r33_development_result_sha256": _sha256(inputs["r33_dev_result_path"]),
        "preregistered_protocol_sha256": protocol_hash,
    }
    manifest = {
        "campaign_id": campaign.PROTOCOL["campaign_id"],
        "run_id": run_id,
        "stage": "fixed_origin_confirmation",
        "source_identity": source,
        "input_hashes": input_hashes,
        "code_sha256": {str(path.relative_to(campaign.ROOT)).replace("\\", "/"): _sha256(path)
                         for path in source_files},
        "fit_starts": len(fit_details),
        "fit_details": fit_details,
        "component_manifest": component_manifest,
        "paired_predictions_sha256": _sha256(paired_path),
        "result_sha256": _sha256(result_path),
        "reserved_months_labels_opened": False,
        "hidden_labels_opened": False,
    }
    manifest_path = run_output / "run_manifest.json"
    _write_json(manifest_path, manifest)

    lines = [
        "# R33 off-market 固定起点确认", "",
        f"运行：`{run_id}`；R22 固定起点参考：`{R22_FIXED_ORIGIN_RUN}`。",
        f"Learner fit starts：{len(fit_details)}；固定 R33 权重为 R22/R32 各 50%。",
        "R22 参考预测来自已通过固定起点门槛并经哈希验证的候选预测。R32 只在两个预登记起点重训 base 与 Detached expert。",
        "训练严格早于起点；读取训练行前排除了 2025-05 和 2025-08。没有读取隐藏标签。", "",
        "| 赛道 | R22 MdAPE | R33 MdAPE | 相对改善 | MAPE 变化 | 最差 horizon 退化 | 通过 |",
        "|---|---:|---:|---:|---:|---:|---|",
        f"| {TRACK} | {track_score['parent']['mdape_percent']:.5f}% | "
        f"{track_score['candidate']['mdape_percent']:.5f}% | "
        f"{track_score['relative_mdape_improvement_percent']:+.3f}% | "
        f"{track_score['mape_relative_change_percent']:+.3f}% | "
        f"{track_score['worst_horizon_mdape_regression_percent']:.3f}% | "
        f"{'通过' if track_score['passes_fixed_origin'] else '未通过'} |",
        "", "## 按起点", "",
        "| 起点 | n | R22 MdAPE | R33 MdAPE | R22 MAPE | R33 MAPE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for origin, values in track_score["by_origin"].items():
        lines.append(
            f"| {origin} | {values['candidate']['n']} | "
            f"{values['parent']['mdape_percent']:.5f}% | {values['candidate']['mdape_percent']:.5f}% | "
            f"{values['parent']['mape_percent']:.5f}% | {values['candidate']['mape_percent']:.5f}% |"
        )
    lines.extend(["", "## 按 horizon", "", "| 月数 | n | R22 MdAPE | R33 MdAPE | 相对 MdAPE 变化 |",
                  "|---:|---:|---:|---:|---:|"])
    for horizon, values in track_score["by_horizon"].items():
        lines.append(
            f"| {horizon} | {values['candidate']['n']} | "
            f"{values['parent']['mdape_percent']:.5f}% | {values['candidate']['mdape_percent']:.5f}% | "
            f"{values['relative_mdape_change_percent']:+.3f}% |"
        )
    lines.extend([
        "", "固定起点门槛：pooled MdAPE 不退化、最差 horizon MdAPE 退化不超过 5%、MAPE 退化不超过 1%。",
        "开发阶段的 off-market MdAPE 目标仍是严格低于 7%；本确认不代表目标已达到。预留月份与隐藏标签保持封存。", "",
    ])
    report_path = run_output / "research_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    campaign._append({
        "event": "card_complete", "card": CARD,
        "stage": "fixed_origin_confirmation", "run_id": run_id,
        "fit_starts": len(fit_details),
        "result_path": str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        "passes_by_track": {TRACK: track_score["passes_fixed_origin"]},
        "reserved_months_labels_opened": False,
        "hidden_labels_opened": False,
    })
    _update_protocol(
        campaign, "fixed_origin_confirmation_complete",
        status="complete", run_id=run_id,
        result_path=str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        result_sha256=_sha256(result_path),
        paired_predictions_sha256=_sha256(paired_path),
        fit_starts=len(fit_details),
        passes_fixed_origin=track_score["passes_fixed_origin"],
        metrics=track_score,
        reserved_months_labels_opened=False,
        hidden_labels_opened=False,
    )
    next_action = (
        "R33 passed fixed-origin but off-market development MdAPE remains at or above 7%; register a distinct improvement and keep reserve labels sealed"
        if track_score["passes_fixed_origin"] else
        "R33 failed fixed-origin; retain R22 as the validated off-market candidate and register a distinct improvement; keep reserve labels sealed"
    )
    campaign._write_progress(
        "r33_fixed_origin_confirmation_complete",
        next_action=next_action,
        r33_fixed_origin_run_id=run_id,
        r33_fixed_origin_result=str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        latest_fit_count=len(fit_details),
    )
    print(f"run_id={run_id}", flush=True)
    print(f"off_market R22={track_score['parent']['mdape_percent']:.5f}% "
          f"R33={track_score['candidate']['mdape_percent']:.5f}% "
          f"fixed_origin_pass={track_score['passes_fixed_origin']}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight", action="store_true", help="verify frozen inputs without fitting")
    modes.add_argument("--run", action="store_true", help="run the preregistered fixed-origin confirmation")
    args = parser.parse_args()
    result = preflight() if args.preflight else run_confirmation()
    print(json.dumps(result, indent=2, sort_keys=True, default=str), flush=True)


if __name__ == "__main__":
    main()
