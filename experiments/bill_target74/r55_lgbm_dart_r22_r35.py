"""R55: screen LightGBM DART on the validated R22 / R35 text routes."""
from __future__ import annotations

import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.bill_target74.r19_expert_weight import _sha256, _write_json


CARD = "R55_lightgbm_dart_r22_r35"
VARIANT = CARD
R22_CARD = "R22_train_fold_tfidf_remarks"
R35_CARD = "R35_listaware_detached_r22_text_expert"
TRACKS = ("off_market", "list_aware")
SCREEN_MONTHS = ("2023-03", "2024-06")
HOUSE_TYPE = "Detached"
SPECIALIST_WEIGHT = 0.75
MODEL_OVERRIDES = {track: {"boosting_type": "dart"} for track in TRACKS}


def _source_files(campaign) -> list[Path]:
    return [
        campaign.PROTOCOL_PATH,
        Path(__file__),
        campaign.PACKAGE / "r22_remarks_tfidf.py",
        campaign.PACKAGE / "r23_word_char_tfidf.py",
        campaign.PACKAGE / "r34_house_type_text_experts.py",
        campaign.PACKAGE / "r03_catboost.py",
        campaign.PACKAGE / "r04_ablation.py",
        campaign.PACKAGE / "runner.py",
        campaign.ROOT / "experiments/bill_greedy/runner.py",
        campaign.ROOT / "experiments/bill_greedy/history.py",
        campaign.ROOT / "model_candidate_v2/features.py",
        campaign.ROOT / "experiments/model_optimization_v2/learners.py",
        campaign.ROOT / "experiments/model_optimization_v2/pipeline.py",
        campaign.ROOT / "model_candidate_v2/winner_config.json",
        campaign.ROOT / "model_candidate_v2/models/frozen_list_aware.json",
    ]


def register_card() -> dict:
    from experiments.bill_target74 import r04_ablation as budget
    from experiments.bill_target74 import runner as campaign

    existing = campaign.PROTOCOL["cards"].get(CARD)
    if existing:
        if existing.get("status") == "registered":
            current_sha256 = _sha256(Path(__file__))
            if existing.get("source_sha256") == current_sha256:
                return existing
            ledger_rows = [json.loads(line) for line in campaign.LEDGER.read_text(encoding="utf-8").splitlines()
                           if line.strip()] if campaign.LEDGER.is_file() else []
            if any(row.get("event") == "fit_start" and row.get("card") == CARD for row in ledger_rows):
                raise RuntimeError("R55 source cannot change after its first fit_start")
            previous_sha256 = existing.get("source_sha256")
            existing["source_sha256"] = current_sha256
            campaign.PROTOCOL_PATH.write_text(
                json.dumps(campaign.PROTOCOL, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            campaign._append({
                "event": "preregistered_source_hash_updated_before_fit", "card": CARD,
                "previous_source_sha256": previous_sha256, "source_sha256": current_sha256,
                "reason": "fixed parent-manifest data-hash lookup during zero-fit preflight",
                "fit_starts": 0, "reserved_months_labels_opened": False,
                "hidden_labels_opened": False,
            })
            return existing
        raise RuntimeError(f"R55 is already {existing.get('status')}; inspect before rerunning")
    if any("dart" in json.dumps(card, sort_keys=True).lower()
           for name, card in campaign.PROTOCOL["cards"].items() if name != CARD):
        raise RuntimeError("a DART experiment is already registered; inspect it before adding R55")

    r22 = campaign.PROTOCOL["cards"].get(R22_CARD)
    r35 = campaign.PROTOCOL["cards"].get(R35_CARD)
    if not r22 or not r35 or r35.get("status") != "fixed_origin_confirmation_passed":
        raise RuntimeError("R55 requires the recorded R22 screen and validated R35 parent")
    source = budget._source_identity()
    registration = {
        "status": "registered",
        "tracks": list(TRACKS),
        "screen_months": list(SCREEN_MONTHS),
        "expected_fit_starts": 6,
        "parents": {
            "off_market": "hash-verified R22 candidate predictions on identical rows",
            "list_aware": "hash-verified R35 candidate predictions on identical rows",
        },
        "parent_run_ids": {
            "off_market": r35["parent_run_id"],
            "list_aware": r35["run_id"],
        },
        "model_family": "lightgbm",
        "model_config_overrides": MODEL_OVERRIDES,
        "method": (
            "Keep R22 train-fold word TF-IDF, structured/history features, targets, routes, "
            "training cutoffs, and 75/25 Detached specialist blend unchanged. For off-market, "
            "switch only the R22 G06 base and Detached expert to LightGBM boosting_type=dart. "
            "For list-aware, switch only the R35 Detached R22-text expert to boosting_type=dart; "
            "keep its frozen 25% HGBR blend and every non-Detached R35 prediction unchanged."
        ),
        "difference_from_prior_work": (
            "The target74 registry and fit ledger contain no LightGBM DART candidate. R24 changed "
            "the learner family to CatBoost; R28, R30, and R51 changed the objective. R55 changes "
            "only the LightGBM boosting mechanism and tests it on the currently selected R22/R35 routes."
        ),
        "incremental_screen_gate": campaign.PROTOCOL["gates"],
        "policy": (
            "Run only the registered 2023-03 and 2024-06 development folds; train strictly before "
            "each fold cutoff; fit TF-IDF using component training remarks only; require the fitted "
            "LightGBM booster to report DART; verify frozen HGBR and parent row alignment. Keep R22/R35 "
            "unless a track passes every fixed screen gate. Never read 2025-05/08 or hidden labels."
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
        "event": "card_registered", "card": CARD,
        "tracks": list(TRACKS), "parent_run_ids": registration["parent_run_ids"],
        "screen_months": list(SCREEN_MONTHS), "expected_fit_starts": registration["expected_fit_starts"],
        "model_config_overrides": MODEL_OVERRIDES,
        "data_sha256": source["data_sha256"], "source_sha256": registration["source_sha256"],
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    campaign._write_progress(
        "r55_registered",
        next_action="run the preregistered two-month R55 DART screen; preserve R22/R35 unless each track passes",
        r55_card=CARD,
    )
    return registration


def _verified_parent(campaign, track: str) -> tuple[dict, Path, Path, dict, pd.DataFrame]:
    r35_card = campaign.PROTOCOL["cards"][R35_CARD]
    if track == "off_market":
        run_id = r35_card["parent_run_id"]
        result_path = campaign.CAMPAIGN_ROOT / "batches" / run_id / "r22_screen_results.json"
        expected_card = R22_CARD
    else:
        run_id = r35_card["run_id"]
        result_path = campaign.ROOT / r35_card["result_path"]
        expected_card = R35_CARD
    paired_path = result_path.parent / "paired_predictions.csv"
    manifest_path = result_path.parent / "run_manifest.json"
    if not all(path.is_file() for path in (result_path, paired_path, manifest_path)):
        raise FileNotFoundError(f"{track} parent result, predictions, or manifest is missing")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (result.get("card") != expected_card
            or result.get("campaign_id") != campaign.PROTOCOL["campaign_id"]
            or result.get("run_id") != run_id
            or manifest.get("run_id") != run_id):
        raise RuntimeError(f"{track} parent artifacts do not identify the registered candidate")
    if (manifest.get("result_sha256") != _sha256(result_path)
            or manifest.get("paired_predictions_sha256") != _sha256(paired_path)
            or (manifest.get("data_sha256")
                or manifest.get("source_identity", {}).get("data_sha256"))
            != campaign.PROTOCOL["data_sha256"]):
        raise RuntimeError(f"{track} parent artifacts fail result, prediction, or data hash verification")
    if result.get("reserved_months_labels_opened") or result.get("hidden_labels_opened"):
        raise RuntimeError(f"{track} parent records restricted-label access")
    if track not in result.get("tracks", {}):
        raise RuntimeError(f"{track} parent result does not contain its track")
    if result["tracks"][track].get("qualifies_screen") is not True:
        raise RuntimeError(f"{track} parent did not pass its registered development screen")
    rows = pd.read_csv(paired_path, dtype={"ml_num": str})
    required = {"month", "track", "ml_num", "sold_price", "candidate_prediction"}
    if missing := required - set(rows.columns):
        raise ValueError(f"{track} parent predictions lack columns: {sorted(missing)}")
    rows = rows.loc[rows.track.eq(track) & rows.month.isin(SCREEN_MONTHS)].copy()
    if rows.duplicated(["month", "ml_num"]).any():
        raise RuntimeError(f"{track} parent rows contain duplicate IDs")
    if set(rows.month.astype(str).unique()) != set(SCREEN_MONTHS):
        raise RuntimeError(f"{track} parent predictions do not cover both screen months")
    if (not np.isfinite(rows[["sold_price", "candidate_prediction"]].to_numpy(float)).all()
            or (rows[["sold_price", "candidate_prediction"]].to_numpy(float) <= 0).any()):
        raise RuntimeError(f"{track} parent rows contain invalid prices")
    return result, result_path, paired_path, manifest, rows


def _align_parent(rows: pd.DataFrame, month: str, homes: pd.DataFrame,
                  truth: pd.DataFrame) -> np.ndarray:
    saved = rows.loc[rows.month.astype(str).eq(month)].copy()
    ids = homes.ml_num.astype(str).to_numpy()
    if saved.ml_num.astype(str).duplicated().any():
        raise RuntimeError(f"R55 parent IDs are duplicated for {month}")
    indexed = saved.assign(ml_num=saved.ml_num.astype(str)).set_index("ml_num")
    aligned = indexed.reindex(ids)
    if (len(saved) != len(truth) or aligned.candidate_prediction.isna().any()
            or not np.array_equal(aligned.sold_price.to_numpy(float), truth.sold_price.to_numpy(float))):
        raise RuntimeError(f"R55 parent predictions and labels do not align for {month}")
    prediction = aligned.candidate_prediction.to_numpy(float)
    if not np.isfinite(prediction).all() or (prediction <= 0).any():
        raise RuntimeError(f"R55 parent predictions are invalid for {month}")
    return prediction


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
        raise RuntimeError("R55 must be preregistered before fitting")
    if card.get("source_sha256") != _sha256(Path(__file__)):
        raise RuntimeError("R55 source changed after preregistration")
    source = budget._source_identity()
    if source["data_sha256"] != card["data_sha256"]:
        raise RuntimeError("R55 data snapshot changed after preregistration")
    parents = {track: _verified_parent(campaign, track) for track in TRACKS}

    run_id = "bill_target74_r55_lgbm_dart_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = campaign.CAMPAIGN_ROOT / "batches" / run_id
    output.mkdir(parents=True, exist_ok=False)
    provenance = output / "provenance"
    provenance.mkdir()
    for track, (result, result_path, paired_path, manifest, _) in parents.items():
        prefix = "r22" if track == "off_market" else "r35"
        shutil.copy2(result_path, provenance / f"{prefix}_screen_results.json")
        shutil.copy2(paired_path, provenance / f"{prefix}_screen_paired_predictions.csv")
        shutil.copy2(result_path.parent / "run_manifest.json", provenance / f"{prefix}_screen_run_manifest.json")

    plan = [(track, month, component) for month in SCREEN_MONTHS for track in TRACKS
            for component in (("base", "expert") if track == "off_market" else ("detached_expert",))]
    if len(plan) != card["expected_fit_starts"]:
        raise RuntimeError("R55 preregistered fit plan has an unexpected size")
    campaign._append({
        "event": "card_start", "card": CARD, "run_id": run_id, "tracks": list(TRACKS),
        "parent_run_ids": card["parent_run_ids"], "screen_months": list(SCREEN_MONTHS),
        "model_family": "lightgbm", "model_config_overrides": MODEL_OVERRIDES,
        "expected_fit_starts": len(plan), "data_sha256": source["data_sha256"],
        "parent_result_sha256": {track: _sha256(item[1]) for track, item in parents.items()},
        "parent_predictions_sha256": {track: _sha256(item[2]) for track, item in parents.items()},
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })

    old_fit_record = greedy._fit_record
    old_variant = greedy.VARIANTS.get(VARIANT)
    greedy.VARIANTS[VARIANT] = {"model_config_overrides": MODEL_OVERRIDES}
    fit_details: list[dict] = []
    paired_parts: dict[str, list[pd.DataFrame]] = {track: [] for track in TRACKS}
    route_details: dict[str, dict] = {track: {} for track in TRACKS}
    active: dict[str, object] = {}

    def record_fit(_parent_fit_id: str, fit_fn):
        month, track, component = (str(active[key]) for key in ("month", "track", "component"))
        fit_id = f"{campaign.PROTOCOL['campaign_id']}:{CARD}:screen:{track}:{month}:{component}:{run_id}"
        campaign._append({
            "event": "fit_start", "card": CARD, "stage": "screen", "run_id": run_id,
            "fit_id": fit_id, "track": track, "month": month, "component": component,
            "training_rows": active["training_rows"], "evaluation_rows": active["evaluation_rows"],
            "data_sha256": source["data_sha256"], "reserved_months_labels_opened": False,
            "hidden_labels_opened": False,
        })
        started = time.perf_counter()
        try:
            fitted = fit_fn()
            estimator = fitted["estimator"]
            params = estimator.get_params(deep=False)
            booster_params = getattr(getattr(estimator, "booster_", None), "params", {})
            actual_boosting_type = (booster_params.get("boosting")
                                    or booster_params.get("boosting_type")
                                    or params.get("boosting_type"))
            if actual_boosting_type != "dart":
                raise RuntimeError(f"R55 expected trained DART booster; got {actual_boosting_type!r}")
            if fitted.get("actual_objective") != "regression":
                raise RuntimeError(f"R55 expected regression objective; got {fitted.get('actual_objective')!r}")
        except Exception as exc:
            campaign._append({
                "event": "fit_complete", "card": CARD, "stage": "screen", "run_id": run_id,
                "fit_id": fit_id, "track": track, "month": month, "component": component,
                "status": "failed", "seconds": time.perf_counter() - started,
                "error": f"{type(exc).__name__}: {exc}",
            })
            raise
        detail = {
            "fit_id": fit_id, "track": track, "month": month, "component": component,
            "seconds": time.perf_counter() - started, "model_family": fitted["model_family"],
            "model_config": fitted["model_config"], "actual_objective": fitted["actual_objective"],
            "actual_boosting_type": actual_boosting_type, "target": fitted["target"],
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
    completed_months: dict[str, list[str]] = {track: [] for track in TRACKS}
    try:
        for month in SCREEN_MONTHS:
            _, train, _, truth, heldout = greedy._load_fold(month)
            cutoff = pd.Period(month, freq="M").start_time
            if (pd.to_datetime(train.date_sold, errors="raise") >= cutoff).any():
                raise RuntimeError(f"R55 training rows are not strictly before {month}")
            if train.date_sold.dt.to_period("M").astype(str).isin(("2025-05", "2025-08")).any():
                raise RuntimeError("R55 training rows unexpectedly contain a sealed month")

            for track in TRACKS:
                homes = hide_answers(heldout, track)
                parent_rows = parents[track][4]
                parent_prediction = _align_parent(parent_rows, month, homes, truth)

                if track == "off_market":
                    route_mask = greedy._specialist_mask(homes, HOUSE_TYPE)
                    fit_mask = greedy._specialist_mask(train, HOUSE_TYPE)
                    fit_jobs = [
                        ("base", train.copy(), homes.copy()),
                        ("expert", train.loc[fit_mask].copy(), homes.loc[route_mask].copy()),
                    ]
                    fitted_components = {}
                    for component, fit_rows, eval_rows in fit_jobs:
                        transformer, restore = r22._install_text_features(fit_rows, eval_rows, track)
                        active.update(month=month, track=track, component=component,
                                      training_rows=len(fit_rows), evaluation_rows=len(eval_rows))
                        try:
                            fitted = greedy._fit_history_candidate(
                                month, track, train, eval_rows, base_variant="G06_lot_garage",
                                variant=VARIANT, fit_tag=f"R55-{component}-{month}", fit_rows=fit_rows,
                            )
                        finally:
                            restore()
                        fitted_components[component] = fitted
                    prediction = greedy._blend_specialist_predictions(
                        np.round(fitted_components["base"]["predictions"]),
                        np.round(fitted_components["expert"]["predictions"]),
                        homes, HOUSE_TYPE, SPECIALIST_WEIGHT,
                    )
                    route_details[track][month] = {
                        "training_rows": {name: int(item["training_rows"])
                                          for name, item in fitted_components.items()},
                        "detached_evaluation_rows": int(route_mask.sum()),
                        "non_detached_predictions_unchanged_from_r22": True,
                        "text_feature_counts": {
                            name: sum(column.startswith("remarks_tfidf_") for column in item["feature_columns"])
                            for name, item in fitted_components.items()
                        },
                    }
                else:
                    route_mask = greedy._specialist_mask(homes, HOUSE_TYPE)
                    fit_mask = greedy._specialist_mask(train, HOUSE_TYPE)
                    fit_rows = train.loc[fit_mask].copy()
                    eval_rows = homes.loc[route_mask].copy()
                    if len(fit_rows) < 500 or int(route_mask.sum()) < 100:
                        raise RuntimeError(f"R55 list-aware Detached route support is too small in {month}")
                    transformer, restore = r22._install_text_features(fit_rows, eval_rows, track)
                    active.update(month=month, track=track, component="detached_expert",
                                  training_rows=len(fit_rows), evaluation_rows=len(eval_rows))
                    try:
                        fitted = greedy._fit_history_candidate(
                            month, track, train, eval_rows, variant=VARIANT,
                            fit_tag=f"R55-{month}-Detached", fit_rows=fit_rows,
                        )
                    finally:
                        restore()
                    frozen_hgbr = helpers._verified_hgbr_cache(greedy, month, track, train, homes)
                    specialist_prediction = blend_prices(
                        fitted["predictions"], frozen_hgbr[route_mask],
                        challenger_weight=SPECIALIST_WEIGHT,
                    )
                    prediction = greedy._overlay_specialist_predictions(
                        parent_prediction, homes, specialist_prediction, HOUSE_TYPE,
                    )
                    if not np.array_equal(prediction[~route_mask], parent_prediction[~route_mask]):
                        raise RuntimeError(f"R55 changed non-Detached R35 predictions in {month}")
                    route_details[track][month] = {
                        "training_rows": int(len(fit_rows)),
                        "detached_evaluation_rows": int(route_mask.sum()),
                        "non_detached_predictions_unchanged_from_r35": True,
                        "text_feature_count": len(transformer.columns_ or []),
                        "frozen_hgbr_sha256": _sha256(
                            campaign.CAMPAIGN_ROOT / "frozen_hgbr" / track / f"{month}.csv"
                        ) if (campaign.CAMPAIGN_ROOT / "frozen_hgbr" / track / f"{month}.csv").is_file() else None,
                    }

                parent_frame = pd.DataFrame({
                    "ml_num": homes.ml_num.astype(str).to_numpy(),
                    "sold_price": truth.sold_price.to_numpy(float),
                    "predicted_price": parent_prediction,
                })
                candidate_frame = pd.DataFrame({
                    "ml_num": homes.ml_num.astype(str).to_numpy(),
                    "sold_price": truth.sold_price.to_numpy(float),
                    "predicted_price": np.asarray(prediction, dtype=float),
                })
                paired_parts[track].append(helpers._candidate_frame(
                    month, track, truth, parent_frame, candidate_frame,
                ))
                completed_months[track].append(month)
                campaign._append({
                    "event": "screen_fold_complete", "card": CARD, "run_id": run_id,
                    "track": track, "month": month, "training_cutoff_exclusive": cutoff.isoformat(),
                    "routed_rows": int(route_mask.sum()),
                    "reserved_months_labels_opened": False, "hidden_labels_opened": False,
                })
    except Exception as exc:
        failure_path = output / "failure_checkpoint.json"
        _write_json(failure_path, {
            "run_id": run_id, "card": CARD, "error": f"{type(exc).__name__}: {exc}",
            "fit_starts": len(fit_details), "fit_details": fit_details,
            "completed_months": completed_months,
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        campaign._append({"event": "card_failed", "card": CARD, "run_id": run_id,
                          "fit_starts": len(fit_details), "error": f"{type(exc).__name__}: {exc}"})
        campaign._write_progress(
            "r55_screen_failed", next_action="inspect R55 failure checkpoint and fit ledger before any retry",
            r55_failure=str(failure_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        )
        raise
    finally:
        greedy._fit_record = old_fit_record
        if old_variant is None:
            greedy.VARIANTS.pop(VARIANT, None)
        else:
            greedy.VARIANTS[VARIANT] = old_variant

    if len(fit_details) != len(plan) or sum(map(len, completed_months.values())) != 4:
        raise RuntimeError(f"R55 expected 6 complete fits and 4 track-months; got {len(fit_details)}")
    paired = pd.concat([frame for track in TRACKS for frame in paired_parts[track]], ignore_index=True)
    paired_path = output / "paired_predictions.csv"
    paired.to_csv(paired_path, index=False)
    tracks = {track: scoring._track_result(pd.concat(paired_parts[track], ignore_index=True))
              for track in TRACKS}
    result = {
        "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
        "card": CARD, "parent_run_ids": card["parent_run_ids"],
        "screen_months": list(SCREEN_MONTHS), "fit_starts": len(fit_details),
        "tracks": tracks, "fit_details": fit_details, "route_diagnostics": route_details,
        "model_config_overrides": MODEL_OVERRIDES,
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
    parent_details = {
        track: {"run_id": item[0]["run_id"], "result_sha256": _sha256(item[1]),
                "paired_predictions_sha256": _sha256(item[2])}
        for track, item in parents.items()
    }
    manifest = {
        "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
        "source_identity": source, "code_sha256": code_hashes,
        "data_sha256": source["data_sha256"], "fit_starts": len(fit_details),
        "fit_details": fit_details, "parent_artifacts": parent_details,
        "result_sha256": _sha256(result_path), "paired_predictions_sha256": _sha256(paired_path),
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    }
    manifest_path = output / "run_manifest.json"
    _write_json(manifest_path, manifest)

    lines = [
        "# R55：LightGBM DART on R22 / R35 text routes", "",
        f"Campaign: `{result['campaign_id']}`; run: `{run_id}`.",
        "Only LightGBM `boosting_type` changed to `dart`; each track is scored incrementally against its verified parent on identical rows.",
        "Off-market changes both R22 G06 base and Detached expert while preserving the 75/25 route. List-aware changes only the R35 Detached text expert and keeps the frozen HGBR blend and non-Detached predictions.",
        "Only 2023-03 and 2024-06 were read for this screen. No reserve or hidden labels were accessed.", "",
        "| Track | Parent MdAPE | DART MdAPE | Relative gain | MAPE change | Months improved | Pass |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for track, score in tracks.items():
        improved = sum(value["relative_mdape_improvement_percent"] > 0
                       for value in score["monthly"].values())
        lines.append(
            f"| {track} | {score['parent']['mdape_percent']:.5f}% | "
            f"{score['candidate']['mdape_percent']:.5f}% | "
            f"{score['relative_mdape_improvement_percent']:+.3f}% | "
            f"{score['mape_relative_change_percent']:+.3f}% | {improved}/2 | "
            f"{'yes' if score['qualifies_screen'] else 'no'} |"
        )
    (output / "research_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    qualifies = {track: score["qualifies_screen"] for track, score in tracks.items()}
    card.update({
        "status": "screen_passed" if any(qualifies.values()) else "screen_failed",
        "run_id": run_id, "fit_starts": len(fit_details),
        "result_path": str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        "paired_predictions_path": str(paired_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        "manifest_path": str(manifest_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        "qualifies_by_track": qualifies, "screen_metrics": tracks,
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    campaign.PROTOCOL_PATH.write_text(
        json.dumps(campaign.PROTOCOL, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    campaign._append({
        "event": "card_complete", "card": CARD, "run_id": run_id,
        "fit_starts": len(fit_details), "result_path": card["result_path"],
        "qualifies_by_track": qualifies, "screen_metrics": tracks,
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    passed = [track for track, qualifies_track in qualifies.items() if qualifies_track]
    next_action = (f"run one preregistered eight-month confirmation for R55 tracks {passed}"
                   if passed else "R55 failed both fixed screens; keep R22/R35 and register a distinct mechanism")
    campaign._write_progress(
        "r55_screen_passed" if passed else "r55_screen_failed",
        next_action=next_action, r55_run_id=run_id, r55_result=card["result_path"],
        r55_fit_starts=len(fit_details),
    )
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Register or run the R55 LightGBM DART screen")
    parser.add_argument("--register-only", action="store_true",
                        help="preregister R55 without reading screen rows or starting fits")
    args = parser.parse_args()
    registration = register_card()
    if args.register_only:
        print(json.dumps(registration, indent=2, ensure_ascii=False))
        return
    print(json.dumps(run_screen(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
