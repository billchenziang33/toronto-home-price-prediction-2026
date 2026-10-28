"""Reserve-safe fixed-origin confirmation for the R35 list-aware expert."""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


CARD = "R35_listaware_detached_r22_text_expert"
TRACK = "list_aware"
HOUSE_TYPE = "Detached"
COMPONENT = "candidate_detached_core"
MIN_TRAINING_ROWS = 500
MIN_EVALUATION_ROWS = 100


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
                    encoding="utf-8")


def _verified_reference(campaign, card: dict):
    progress = json.loads(campaign.STATE.read_text(encoding="utf-8"))
    relative = progress.get("r22_fixed_origin_result")
    if not relative:
        raise RuntimeError("campaign progress does not identify the R22 fixed-origin result")
    result_path = campaign.ROOT / relative
    paired_path = result_path.parent / "paired_predictions.csv"
    manifest_path = result_path.parent / "run_manifest.json"
    if not all(path.is_file() for path in (result_path, paired_path, manifest_path)):
        raise FileNotFoundError("R22 fixed-origin result, predictions, or manifest is missing")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (result.get("campaign_id") != campaign.PROTOCOL["campaign_id"]
            or result.get("stage") != "fixed_origin_confirmation"
            or result.get("card") != "R22_train_fold_tfidf_remarks"
            or result.get("run_id") != manifest.get("run_id")):
        raise RuntimeError("R22 fixed-origin artifacts do not belong to the active campaign")
    if (manifest.get("result_sha256") != _sha256(result_path)
            or manifest.get("paired_predictions_sha256") != _sha256(paired_path)
            or manifest.get("fit_starts") != result.get("fit_starts")):
        raise RuntimeError("R22 fixed-origin artifacts fail manifest hash or fit-count verification")
    if result.get("reserved_months_labels_opened") or result.get("hidden_labels_opened"):
        raise RuntimeError("R22 fixed-origin reference reports reserve or hidden-label access")
    if manifest.get("source_identity", {}).get("data_sha256") != campaign.PROTOCOL["data_sha256"]:
        raise RuntimeError("R22 fixed-origin reference uses a different dataset snapshot")
    origins = tuple(campaign.PROTOCOL["fixed_origins"])
    horizons = (1, 2, 3, 4)
    if result.get("origins") != list(origins) or result.get("horizons") != list(horizons):
        raise RuntimeError("R22 fixed-origin reference does not cover the registered windows")
    if result.get("tracks", {}).get(TRACK, {}).get("passes_fixed_origin") is not True:
        raise RuntimeError("R22 list-aware fixed-origin parent did not pass its registered gate")

    development_relative = card.get("development_result_path")
    development_run_id = card.get("development_run_id")
    if not development_relative or not development_run_id:
        raise RuntimeError("R35 development confirmation is not fully registered")
    development_path = campaign.ROOT / development_relative
    development_paired = development_path.parent / "paired_predictions.csv"
    development_manifest_path = development_path.parent / "run_manifest.json"
    if not all(path.is_file() for path in (development_path, development_paired,
                                            development_manifest_path)):
        raise FileNotFoundError("R35 development confirmation artifacts are incomplete")
    development = json.loads(development_path.read_text(encoding="utf-8"))
    development_manifest = json.loads(development_manifest_path.read_text(encoding="utf-8"))
    development_score = development.get("tracks", {}).get(TRACK, {})
    if (development.get("run_id") != development_run_id
            or development.get("campaign_id") != campaign.PROTOCOL["campaign_id"]
            or development.get("stage") != "development_confirmation"
            or development_score.get("qualifies_development") is not True
            or development.get("reserved_months_labels_opened")
            or development.get("hidden_labels_opened")):
        raise RuntimeError("R35 development result does not qualify for fixed-origin confirmation")
    if (development_manifest.get("result_sha256") != _sha256(development_path)
            or development_manifest.get("paired_predictions_sha256") != _sha256(development_paired)
            or development_manifest.get("fit_starts") != development.get("fit_starts")):
        raise RuntimeError("R35 development artifacts fail manifest verification")

    paired = pd.read_csv(paired_path, dtype={"ml_num": str})
    paired = paired.loc[paired.track.eq(TRACK)].copy()
    required = {"origin", "horizon", "month", "ml_num", "sold_price",
                "parent_prediction", "candidate_prediction"}
    if missing := required - set(paired.columns):
        raise ValueError(f"R22 fixed-origin predictions lack columns: {sorted(missing)}")
    if (paired.duplicated(["origin", "month", "ml_num"]).any()
            or set(paired.origin.astype(str)) != set(origins)
            or set(paired.horizon.astype(int)) != set(horizons)):
        raise RuntimeError("R22 list-aware fixed-origin rows have duplicate or incomplete coverage")
    return (result_path, paired_path, manifest_path, result, manifest, paired,
            development_path, development_paired, development_manifest_path,
            development, development_manifest)


def _register(campaign, card: dict, origins: tuple[str, ...], gates: dict) -> dict:
    existing = card.get("fixed_origin_confirmation")
    if existing:
        status = existing.get("status")
        if status == "complete":
            result_path = campaign.ROOT / existing["result_path"]
            return {"already_complete": True, "result_path": result_path}
        if status != "registered":
            raise RuntimeError(f"R35 fixed-origin card is already {status}; inspect before retrying")
        if not existing.get("run_id"):
            raise RuntimeError("registered R35 fixed-origin card is missing its run ID")
        return existing

    run_id = "bill_target74_r35_fixedorigin_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    registration = {
        "status": "registered",
        "parent": "hash-verified R22 list-aware candidate predictions on identical rows",
        "origins": list(origins),
        "horizons": [1, 2, 3, 4],
        "expected_fit_starts": len(origins),
        "method": "For Detached homes only, refit the G01 list-aware core with R22 fold-local word TF-IDF using only Detached pre-origin training rows; blend 75% specialist with the hash-verified fixed-origin shared HGBR predictions; leave all other homes at R22.",
        "training_policy": "Use r22.load_fixed_origin_training, which excludes 2025-05 and 2025-08 before reading; read evaluation labels only for the registered 2024-06 and 2025-12 origins and four-month horizons; fit TF-IDF on Detached training rows only.",
        "gates": {
            "pooled_mdape_regression_max_percent": gates["fixed_origin_pooled_mdape_regression_max_percent"],
            "worst_horizon_mdape_regression_max_percent": gates["fixed_origin_horizon_regression_max_percent"],
            "mape_regression_max_percent": gates["fixed_origin_mape_regression_max_percent"],
        },
        "run_id": run_id,
        "result_path": f"outputs/bill_target74/{campaign.PROTOCOL['campaign_id']}/batches/{run_id}/r35_fixed_origin_results.json",
    }
    card["status"] = "fixed_origin_confirmation_registered"
    card["fixed_origin_confirmation"] = registration
    _write_json(campaign.PROTOCOL_PATH, campaign.PROTOCOL)
    campaign._append({
        "event": "card_registered", "card": CARD,
        "stage": "fixed_origin_confirmation", "run_id": run_id,
        "track": TRACK, "origins": list(origins), "horizons": [1, 2, 3, 4],
        "expected_fit_starts": len(origins), "parent": registration["parent"],
        "gates": registration["gates"],
        "reserved_months_labels_opened": False,
        "reserve_validation_eligible": False, "hidden_labels_opened": False,
    })
    campaign._write_progress(
        "r35_fixed_origin_registered",
        next_action="run the preregistered R35 fixed-origin confirmation; never read 2025-05/08 or hidden labels",
    )
    return registration


def run() -> dict:
    from data import hide_answers
    from experiments.bill_target74 import r04_ablation as budget
    from experiments.bill_target74 import r22_remarks_tfidf as r22
    from experiments.bill_target74 import runner as campaign
    from experiments.bill_target74.r19_expert_weight import _write_json as write_json
    from experiments.bill_greedy import runner as greedy
    from experiments.model_optimization_v2.pipeline import blend_prices

    card = campaign.PROTOCOL["cards"].get(CARD)
    if not card or card.get("status") not in {
        "development_confirmation_passed", "fixed_origin_confirmation_registered",
        "fixed_origin_confirmation_passed", "fixed_origin_confirmation_failed",
    }:
        raise RuntimeError("R35 must first pass its registered eight-month development gate")
    origins = tuple(campaign.PROTOCOL["fixed_origins"])
    (reference_path, reference_paired_path, reference_manifest_path,
     reference_result, reference_manifest, reference_rows,
     development_path, development_paired_path, development_manifest_path,
     development, development_manifest) = _verified_reference(campaign, card)
    if card.get("fixed_origin_confirmation", {}).get("status") == "complete":
        result_path = campaign.ROOT / card["fixed_origin_confirmation"]["result_path"]
        if not result_path.is_file():
            raise FileNotFoundError("completed R35 fixed-origin result is missing")
        return json.loads(result_path.read_text(encoding="utf-8"))

    registration = _register(campaign, card, origins, campaign.PROTOCOL["gates"])
    if registration.get("already_complete"):
        return json.loads(registration["result_path"].read_text(encoding="utf-8"))
    run_id = registration["run_id"]
    output = campaign.ROOT / registration["result_path"]
    output = output.parent
    if output.exists():
        raise FileExistsError(f"R35 fixed-origin run directory already exists: {output}")
    output.mkdir(parents=True)
    provenance = output / "provenance"
    provenance.mkdir()

    source = budget._source_identity()
    if source.get("data_sha256") != campaign.PROTOCOL["data_sha256"]:
        raise RuntimeError("active training data hash differs from the preregistered campaign snapshot")
    if source.get("data_sha256") != reference_manifest["source_identity"]["data_sha256"]:
        raise RuntimeError("R22 fixed-origin reference data hash differs from R35 source data")

    upstream = [
        (reference_path, "r22_fixed_origin_results.json"),
        (reference_paired_path, "r22_fixed_origin_paired_predictions.csv"),
        (reference_manifest_path, "r22_fixed_origin_run_manifest.json"),
        (development_path, "r35_development_results.json"),
        (development_paired_path, "r35_development_paired_predictions.csv"),
        (development_manifest_path, "r35_development_run_manifest.json"),
    ]
    for source_path, name in upstream:
        shutil.copy2(source_path, provenance / name)

    source_files = [
        campaign.PROTOCOL_PATH,
        Path(__file__),
        campaign.PACKAGE / "r22_remarks_tfidf.py",
        campaign.PACKAGE / "r03_catboost.py",
        campaign.PACKAGE / "r04_ablation.py",
        campaign.PACKAGE / "runner.py",
        campaign.ROOT / "experiments/bill_greedy/runner.py",
        campaign.ROOT / "experiments/bill_greedy/history.py",
        campaign.ROOT / "model_candidate_v2/features.py",
        campaign.ROOT / "experiments/model_optimization_v2/learners.py",
        campaign.ROOT / "experiments/model_optimization_v2/pipeline.py",
        campaign.ROOT / "experiments/model_optimization_v2/folds.py",
    ]
    code_hashes = {
        str(path.relative_to(campaign.ROOT)).replace("\\", "/"): _sha256(path)
        for path in source_files
    }
    source_snapshots = {}
    for index, path in enumerate(source_files):
        key = str(path.relative_to(campaign.ROOT)).replace("\\", "/")
        name = f"src_{index:02d}{path.suffix}.source"
        shutil.copy2(path, provenance / name)
        source_snapshots[key] = name

    gates = registration["gates"]
    campaign._append({
        "event": "card_start", "card": CARD,
        "stage": "fixed_origin_confirmation", "run_id": run_id,
        "track": TRACK, "origins": list(origins), "horizons": [1, 2, 3, 4],
        "expected_fit_starts": len(origins),
        "training_policy": registration["training_policy"],
        "reserved_months_labels_opened": False,
        "reserve_validation_eligible": False, "hidden_labels_opened": False,
    })

    fit_details: list[dict] = []
    paired_parts: list[pd.DataFrame] = []
    origin_details: dict[str, dict] = {}
    fit_context: dict[str, object] = {}
    old_fit_record = greedy._fit_record

    def record_fit(_original_id: str, fit_fn):
        if not fit_context:
            raise RuntimeError("R35 fixed-origin fit context was not initialized")
        origin_text = str(fit_context["origin"])
        fit_id = (f"{campaign.PROTOCOL['campaign_id']}:{CARD}:fixed_origin:"
                  f"{origin_text}:{TRACK}:{COMPONENT}:{run_id}")
        campaign._append({
            "event": "fit_start", "card": CARD,
            "stage": "fixed_origin_confirmation", "run_id": run_id,
            "fit_id": fit_id, "track": TRACK, "origin": origin_text,
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
                "event": "fit_complete", "card": CARD,
                "stage": "fixed_origin_confirmation", "run_id": run_id,
                "fit_id": fit_id, "track": TRACK, "origin": origin_text,
                "component": COMPONENT, "status": "failed",
                "seconds": time.perf_counter() - started,
                "error": f"{type(exc).__name__}: {exc}",
            })
            raise
        seconds = time.perf_counter() - started
        saved = r22._save_component_predictions(
            output, TRACK, origin_text[:7], COMPONENT,
            fit_context["evaluation_ids"], fitted["predictions"],
        )
        detail = {
            "fit_id": fit_id, "track": TRACK, "origin": origin_text,
            "component": COMPONENT, "seconds": seconds,
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
            "event": "fit_complete", "card": CARD,
            "stage": "fixed_origin_confirmation", "run_id": run_id,
            **detail, "status": "complete",
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        return fitted

    greedy._fit_record = record_fit
    try:
        for origin_text in origins:
            origin = pd.Timestamp(origin_text)
            end = origin + pd.DateOffset(months=4)
            train = r22.load_fixed_origin_training(campaign.DATA, origin_text)
            training_months = train.date_sold.dt.to_period("M").astype(str)
            if training_months.isin(campaign.PROTOCOL["reserved_months"]).any():
                raise RuntimeError("reserved sale-month rows entered the R35 training frame")
            if (train.date_sold >= origin).any():
                raise RuntimeError(f"R35 training data are not strictly before {origin_text}")

            heldout = pd.read_parquet(
                campaign.DATA,
                filters=[("date_sold", ">=", origin), ("date_sold", "<", end)],
            )
            heldout["date_sold"] = pd.to_datetime(heldout.date_sold, errors="raise")
            heldout["ml_num"] = heldout.ml_num.astype(str)
            if heldout.empty or not heldout.ml_num.is_unique:
                raise RuntimeError(f"R35 heldout IDs are empty or duplicated at {origin_text}")
            evaluation_months = set(heldout.date_sold.dt.to_period("M").astype(str))
            expected_months = {(origin.to_period("M") + offset).strftime("%Y-%m")
                               for offset in range(4)}
            if evaluation_months != expected_months:
                raise RuntimeError(f"R35 heldout horizon is incomplete at {origin_text}")
            if evaluation_months.intersection(campaign.PROTOCOL["reserved_months"]):
                raise RuntimeError("registered fixed-origin evaluation overlaps compromised reserve months")

            reference = reference_rows.loc[
                reference_rows.origin.astype(str).eq(origin_text)
            ].copy()
            if (reference.empty or reference.ml_num.astype(str).duplicated().any()
                    or set(reference.ml_num.astype(str)) != set(heldout.ml_num.astype(str))):
                raise RuntimeError(f"R22 list-aware row IDs do not match the {origin_text} evaluation")
            reference = reference.set_index(reference.ml_num.astype(str)).loc[heldout.ml_num].reset_index(drop=True)
            truth = heldout.set_index("ml_num").loc[reference.ml_num.astype(str), "sold_price"].to_numpy(float)
            if not np.array_equal(truth, reference.sold_price.to_numpy(float)):
                raise RuntimeError(f"R22 saved labels do not match the permitted {origin_text} rows")
            horizons = greedy._month_horizons(heldout.date_sold, origin)
            if (not np.array_equal(horizons, reference.horizon.to_numpy(int))
                    or not np.array_equal(heldout.date_sold.dt.to_period("M").astype(str).to_numpy(),
                                          reference.month.astype(str).to_numpy())):
                raise RuntimeError(f"R22 origin/month/horizon alignment differs at {origin_text}")

            homes = hide_answers(heldout, TRACK).reset_index(drop=True)
            route_mask = greedy._specialist_mask(homes, HOUSE_TYPE)
            fit_rows = train.loc[greedy._specialist_mask(train, HOUSE_TYPE)].copy()
            eval_rows = homes.loc[route_mask].copy()
            if len(fit_rows) < MIN_TRAINING_ROWS or int(route_mask.sum()) < MIN_EVALUATION_ROWS:
                raise RuntimeError(f"R35 Detached route is too small at {origin_text}")

            shared_detail = [row for row in reference_manifest["fit_details"]
                             if row.get("track") == TRACK and row.get("origin") == origin_text
                             and row.get("component") == "shared_hgbr"]
            if len(shared_detail) != 1:
                raise RuntimeError(f"R22 must have exactly one shared HGBR artifact at {origin_text}")
            shared_saved = shared_detail[0].get("saved_predictions", {})
            shared_path = reference_path.parent / shared_saved.get("path", "")
            if (not shared_path.is_file() or _sha256(shared_path) != shared_saved.get("sha256")):
                raise RuntimeError(f"R22 shared HGBR artifact hash fails at {origin_text}")
            shared = pd.read_csv(shared_path, dtype={"ml_num": str})
            if shared.ml_num.astype(str).duplicated().any():
                raise RuntimeError(f"R22 shared HGBR IDs are duplicated at {origin_text}")
            shared = shared.set_index(shared.ml_num.astype(str)).reindex(homes.ml_num.astype(str))
            if shared.predicted_price.isna().any():
                raise RuntimeError(f"R22 shared HGBR IDs do not cover R35 homes at {origin_text}")
            shared_values = shared.predicted_price.to_numpy(float)

            transformer, restore = r22._install_text_features(fit_rows, eval_rows, TRACK)
            fit_context.update(
                origin=origin_text, training_rows=len(fit_rows),
                evaluation_rows=len(eval_rows), evaluation_ids=eval_rows.ml_num.astype(str).tolist(),
            )
            try:
                fitted = greedy._fit_history_candidate(
                    origin.strftime("%Y-%m"), TRACK, train, eval_rows, cutoff=origin,
                    fit_rows=fit_rows, variant=r22.VARIANT,
                    fit_tag=f"R35-fixed-{origin_text}-Detached",
                )
            finally:
                restore()

            detached_prediction = blend_prices(
                fitted["predictions"], shared_values[route_mask], challenger_weight=0.75,
            )
            candidate = greedy._overlay_specialist_predictions(
                reference.candidate_prediction.to_numpy(float), homes,
                detached_prediction, HOUSE_TYPE,
            )
            if not np.array_equal(candidate[~route_mask],
                                  reference.candidate_prediction.to_numpy(float)[~route_mask]):
                raise RuntimeError(f"R35 changed non-Detached homes at {origin_text}")

            paired = pd.DataFrame({
                "origin": origin_text,
                "horizon": horizons,
                "month": heldout.date_sold.dt.to_period("M").astype(str).to_numpy(),
                "track": TRACK,
                "ml_num": homes.ml_num.astype(str).to_numpy(),
                "sold_price": truth,
                "parent_prediction": reference.candidate_prediction.to_numpy(float),
                "candidate_prediction": candidate,
            })
            paired_parts.append(paired)
            origin_details[origin_text] = {
                "training_rows": len(train), "detached_training_rows": len(fit_rows),
                "evaluation_rows": len(heldout), "detached_evaluation_rows": int(route_mask.sum()),
                "training_intervals": [list(item) for item in r22.fixed_origin_training_intervals(origin_text)],
                "evaluation_months": sorted(evaluation_months),
                "reserved_months_labels_opened": False,
                "shared_hgbr_sha256": shared_saved["sha256"],
                "shared_hgbr_path": shared_saved["path"],
                "text_feature_count": len(transformer.columns_ or []),
                "non_detached_predictions_unchanged": True,
            }
            pd.concat(paired_parts, ignore_index=True).to_csv(output / "partial_paired_predictions.csv", index=False)
            campaign._append({
                "event": "fixed_origin_complete", "card": CARD,
                "stage": "fixed_origin_confirmation", "run_id": run_id,
                "track": TRACK, "origin": origin_text,
                "training_rows": len(train), "detached_training_rows": len(fit_rows),
                "evaluation_rows": len(heldout), "detached_evaluation_rows": int(route_mask.sum()),
                "training_intervals": origin_details[origin_text]["training_intervals"],
                "reserved_months_labels_opened": False, "hidden_labels_opened": False,
            })
    except Exception as exc:
        greedy._fit_record = old_fit_record
        failure = {
            "run_id": run_id, "card": CARD, "stage": "fixed_origin_confirmation",
            "error": f"{type(exc).__name__}: {exc}", "fit_starts": len(fit_details),
            "fit_details": fit_details, "completed_origins": origin_details,
            "reserved_months_labels_opened": False,
            "reserve_validation_eligible": False, "hidden_labels_opened": False,
        }
        write_json(output / "failure_checkpoint.json", failure)
        campaign._append({
            "event": "card_failed", "card": CARD,
            "stage": "fixed_origin_confirmation", "run_id": run_id,
            "fit_starts": len(fit_details), "error": failure["error"],
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        card["status"] = "fixed_origin_confirmation_failed"
        card["fixed_origin_confirmation"].update({
            "status": "failed", "fit_starts": len(fit_details),
            "failure_path": str((output / "failure_checkpoint.json").relative_to(campaign.ROOT)).replace("\\", "/"),
        })
        _write_json(campaign.PROTOCOL_PATH, campaign.PROTOCOL)
        campaign._write_progress(
            "r35_fixed_origin_confirmation_failed",
            next_action="inspect R35 fixed-origin failure checkpoint and ledger before any retry",
            latest_fit_count=len(fit_details),
        )
        raise
    finally:
        greedy._fit_record = old_fit_record

    if len(fit_details) != len(origins):
        raise RuntimeError(f"R35 expected {len(origins)} new fits, recorded {len(fit_details)}")
    paired = pd.concat(paired_parts, ignore_index=True)
    paired_path = output / "paired_predictions.csv"
    paired.to_csv(paired_path, index=False)
    score = r22.score_fixed_origin_confirmation(
        paired, origins=origins, horizons=(1, 2, 3, 4),
        maximum_pooled_mdape_regression_percent=gates["pooled_mdape_regression_max_percent"],
        maximum_horizon_mdape_regression_percent=gates["worst_horizon_mdape_regression_max_percent"],
        maximum_mape_regression_percent=gates["mape_regression_max_percent"],
    )
    result = {
        "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
        "card": CARD, "stage": "fixed_origin_confirmation",
        "development_run_id": development["run_id"],
        "parent_fixed_origin_run_id": reference_result["run_id"],
        "track": TRACK, "origins": list(origins), "horizons": [1, 2, 3, 4],
        "fit_starts": len(fit_details), "training_policy": registration["training_policy"],
        "tracks": {TRACK: score}, "fit_details": fit_details,
        "origin_details": origin_details,
        "target_mdape_strictly_below": 4.0,
        "target_met": bool(score["candidate"]["mdape_percent"] < 4.0),
        "passes_fixed_origin": bool(score["passes_fixed_origin"]),
        "reserved_months_labels_opened": False,
        "reserve_validation_eligible": False, "hidden_labels_opened": False,
    }
    result_path = output / "r35_fixed_origin_results.json"
    _write_json(result_path, result)
    manifest = {
        "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
        "stage": "fixed_origin_confirmation", "source_identity": source,
        "code_sha256": code_hashes, "source_snapshots": source_snapshots,
        "parent_fixed_origin_run_id": reference_result["run_id"],
        "parent_fixed_origin_result_sha256": _sha256(reference_path),
        "parent_fixed_origin_predictions_sha256": _sha256(reference_paired_path),
        "parent_fixed_origin_manifest_sha256": _sha256(reference_manifest_path),
        "development_result_sha256": _sha256(development_path),
        "development_predictions_sha256": _sha256(development_paired_path),
        "development_manifest_sha256": _sha256(development_manifest_path),
        "fit_starts": len(fit_details), "fit_details": fit_details,
        "origin_details": origin_details,
        "paired_predictions_sha256": _sha256(paired_path),
        "result_sha256": _sha256(result_path),
        "reserved_months_labels_opened": False,
        "reserve_validation_eligible": False, "hidden_labels_opened": False,
    }
    _write_json(output / "run_manifest.json", manifest)
    report = [
        "# R35：固定起点确认", "",
        f"Run：`{run_id}`。对照为哈希核验的 R22 list-aware 候选 `R22_train_fold_tfidf_remarks`。",
        f"训练 fit starts：{len(fit_details)}；评估房源：{score['n']}。", "",
        f"R22 pooled MdAPE：{score['parent']['mdape_percent']:.5f}%；",
        f"R35 pooled MdAPE：{score['candidate']['mdape_percent']:.5f}%；",
        f"相对改善：{score['relative_mdape_improvement_percent']:+.3f}%；",
        f"MAPE 相对变化：{score['mape_relative_change_percent']:+.3f}%。", "",
        f"固定起点门槛：{'通过' if score['passes_fixed_origin'] else '未通过'}；",
        f"list-aware 开发目标 `<4%`：{'达到' if result['target_met'] else '未达到'}。", "",
        "候选只替换 Detached 行；其他房型逐行沿用 R22。R35 专家使用严格早于起点的 Detached 训练行、训练折内 TF-IDF 和哈希核验的 R22 shared HGBR 预测。",
        "训练读取在 2025-05/08 之外分段执行；评估只读登记的 2024-06 至 2024-09、2025-12 至 2026-03。未读取隐藏标签。",
        "2025-05/08 标签曾被意外载入，不能再用于一次性独立保留集验证；本结果只是固定起点检查，不能替代该保留集。", "",
        f"结果、逐房预测、模型预测、源码快照与哈希位于 `{output.relative_to(campaign.ROOT)}`。", "",
    ]
    (output / "research_report.md").write_text("\n".join(report), encoding="utf-8")

    passed = bool(score["passes_fixed_origin"])
    card["fixed_origin_confirmation"].update({
        "status": "complete", "fit_starts": len(fit_details),
        "result_path": str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        "manifest_path": str((output / "run_manifest.json").relative_to(campaign.ROOT)).replace("\\", "/"),
        "passes_fixed_origin": passed, "target_met": result["target_met"],
        "metrics": {
            "parent_mdape_percent": score["parent"]["mdape_percent"],
            "candidate_mdape_percent": score["candidate"]["mdape_percent"],
            "relative_mdape_improvement_percent": score["relative_mdape_improvement_percent"],
            "parent_mape_percent": score["parent"]["mape_percent"],
            "candidate_mape_percent": score["candidate"]["mape_percent"],
            "mape_relative_change_percent": score["mape_relative_change_percent"],
            "worst_horizon_mdape_regression_percent": score["worst_horizon_mdape_regression_percent"],
        },
    })
    card["status"] = "fixed_origin_confirmation_passed" if passed else "fixed_origin_confirmation_failed"
    _write_json(campaign.PROTOCOL_PATH, campaign.PROTOCOL)
    campaign._append({
        "event": "card_complete", "card": CARD,
        "stage": "fixed_origin_confirmation", "run_id": run_id,
        "fit_starts": len(fit_details),
        "result_path": str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        "passes_by_track": {TRACK: passed},
        "targets_met_by_track": {TRACK: result["target_met"]},
        "reserved_months_labels_opened": False,
        "reserve_validation_eligible": False, "hidden_labels_opened": False,
    })

    progress = json.loads(campaign.STATE.read_text(encoding="utf-8"))
    best = progress.get("best_validated_candidates_by_track", {})
    if passed and score["candidate"]["mdape_percent"] < best.get(TRACK, {}).get("fixed_origin_mdape_percent", float("inf")):
        best[TRACK] = {
            "candidate": "R35_listaware_detached_r22_text_expert",
            "development_mdape_percent": development["tracks"][TRACK]["candidate"]["mdape_percent"],
            "fixed_origin_mdape_percent": score["candidate"]["mdape_percent"],
            "development_run_id": development["run_id"], "fixed_origin_run_id": run_id,
        }
    campaign._write_progress(
        "r35_fixed_origin_confirmation_complete",
        next_action=("R35 passed fixed-origin checks but remains above the <4% development target; register a distinct list-aware hypothesis"
                     if passed and not result["target_met"] else
                     "R35 failed fixed-origin checks; keep R22 as the best validated list-aware candidate and register a distinct hypothesis"
                     if not passed else
                     "R35 met the list-aware development threshold and passed fixed-origin checks; freeze list-aware and continue off-market research"),
        r35_fixed_origin_result=str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        best_validated_candidates_by_track=best,
        latest_fit_count=len(fit_details),
    )
    return result


if __name__ == "__main__":
    outcome = run()
    result = outcome["tracks"][TRACK]
    print(
        f"R35 fixed-origin: parent_mdape={result['parent']['mdape_percent']:.5f}% "
        f"candidate_mdape={result['candidate']['mdape_percent']:.5f}% "
        f"relative_gain={result['relative_mdape_improvement_percent']:+.3f}% "
        f"pass={result['passes_fixed_origin']} run={outcome['run_id']}",
        flush=True,
    )
