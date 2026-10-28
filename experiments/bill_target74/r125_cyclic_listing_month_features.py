"""R125: add cyclic month-of-year features to the R22/R35 routes."""
from __future__ import annotations

import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.bill_target74 import r03_catboost as helpers
from experiments.bill_target74 import r04_ablation as identity
from experiments.bill_target74 import r22_remarks_tfidf as r22
from experiments.bill_target74 import r121_covariate_shift_weights_allowlist as parent_loader
from experiments.bill_target74 import runner as campaign


CARD = "R125_two_track_cyclic_listing_month_features"
TRACKS = ("off_market", "list_aware")
SCREEN_MONTHS = ("2023-03", "2024-06")
PARENT_CARDS = {
    "off_market": "R22_train_fold_tfidf_remarks",
    "list_aware": "R35_listaware_detached_r22_text_expert",
}
PARENT_IDS = {"off_market": "R22", "list_aware": "R35"}
HOUSE_TYPE = "Detached"
SPECIALIST_WEIGHT = 0.75
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20261054
ADDED_COLUMNS = ("listing_month_sin", "listing_month_cos")
VARIANT = CARD


def _source_files() -> list[Path]:
    return [
        campaign.PROTOCOL_PATH, Path(__file__),
        campaign.PACKAGE / "r121_covariate_shift_weights_allowlist.py",
        campaign.PACKAGE / "r22_remarks_tfidf.py", campaign.PACKAGE / "r03_catboost.py",
        campaign.PACKAGE / "r04_ablation.py", campaign.PACKAGE / "runner.py",
        campaign.ROOT / "experiments/bill_greedy/runner.py",
        campaign.ROOT / "experiments/bill_greedy/history.py",
        campaign.ROOT / "experiments/bill_greedy/comparables.py",
        campaign.ROOT / "experiments/model_optimization_v2/worker.py",
        campaign.ROOT / "experiments/model_optimization_v2/learners.py",
        campaign.ROOT / "model_candidate_v2/features.py",
        campaign.ROOT / "model_candidate_v2/winner_config.json",
        campaign.ROOT / "model_candidate_v2/models/frozen_list_aware.json",
    ]


def register_card() -> dict:
    existing = campaign.PROTOCOL["cards"].get(CARD)
    if existing:
        if existing.get("status") != "registered":
            raise RuntimeError(f"{CARD} is already {existing.get('status')}; inspect its run before continuing")
        ledger_rows = ([json.loads(line) for line in campaign.LEDGER.read_text(encoding="utf-8").splitlines()
                        if line.strip()] if campaign.LEDGER.is_file() else [])
        if any(row.get("event") == "fit_start" and row.get("card") == CARD for row in ledger_rows):
            raise RuntimeError("R125 source cannot change after its first fit_start")
        current_sha = helpers._sha256(Path(__file__))
        if existing.get("source_sha256") != current_sha:
            previous_sha = existing.get("source_sha256")
            existing["source_sha256"] = current_sha
            campaign.PROTOCOL_PATH.write_text(
                json.dumps(campaign.PROTOCOL, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            campaign._append({"event": "preregistered_source_hash_updated_before_fit", "card": CARD,
                              "previous_source_sha256": previous_sha, "source_sha256": current_sha,
                              "reason": "update the preregistered R125 method before any fit starts",
                              "fit_starts": 0, "reserved_months_labels_opened": False,
                              "hidden_labels_opened": False})
        return existing

    r35_card = campaign.PROTOCOL["cards"].get(PARENT_CARDS["list_aware"])
    if not r35_card or r35_card.get("status") != "fixed_origin_confirmation_passed":
        raise RuntimeError("R125 requires the verified R22/R35 parent cards")

    # Verify parent artifacts before preregistration; this reads only the two safe screen folds.
    parent_provenance = {}
    for track in TRACKS:
        result, result_path, paired_path, manifest, rows = parent_loader._verified_parent(campaign, track)
        parent_provenance[track] = {
            "run_id": result["run_id"],
            "result_sha256": helpers._sha256(result_path),
            "paired_predictions_sha256": helpers._sha256(paired_path),
            "manifest_sha256": helpers._sha256(result_path.parent / "run_manifest.json"),
            "rows": int(len(rows)),
        }

    source_identity = identity._source_identity()
    source_sha = helpers._sha256(Path(__file__))
    registration = {
        "status": "registered",
        "tracks": list(TRACKS), "screen_months": list(SCREEN_MONTHS),
        "expected_fit_starts": 6,
        "comparison": "hash-verified current R22 off-market / R35 list-aware parent predictions on identical rows",
        "parent_ids": PARENT_IDS, "parent_artifacts": parent_provenance,
            "single_change": "add sine and cosine of listing month-of-year to the existing date feature; preserve all existing features, learner settings, and routes",
        "retained_feature_groups": [
            "fold-local R22 remarks TF-IDF", "house area, rooms, parking, property type, linear month and cyclic month",
            "building/unit history", "off-market lot/garage extras", "list-aware list price",
            "existing Detached specialist and frozen HGBR blend",
        ],
        "model_family": "LightGBM",
        "model_config_overrides": {track: {} for track in TRACKS},
        "prediction_routes": {
            "off_market": "refit the R22 base and Detached expert with two cyclic listing-month features; keep the existing 75/25 Detached route",
            "list_aware": "refit only the R35 Detached expert with two cyclic listing-month features; keep frozen HGBR at 25%, expert at 75%, and all non-Detached R35 predictions unchanged",
        },
        "gate": {
            "relative_mdape_gain_min_percent": 0.25,
            "both_screen_months_improve": True,
            "maximum_mape_regression_percent": 1.0,
            "paired_home_bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "paired_home_bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_upper95_below_zero": True,
        },
        "policy": "Use only 2023-03 and 2024-06; every model fit trains strictly before the target month. Derive listing_month_sin/cos from each row's date_listed without fitting on held-out data. Fit remarks vocabulary on that component's training rows only. Do not tune after inspecting these screen labels. Never read 2025-05/08 or hidden labels.",
        "why_distinct": "R122 removed raw coordinates, while R104 tested spatial grid/type interactions. R125 instead preserves all location features and adds a smooth cyclic encoding of listing month, testing calendar seasonality that the current linear months feature cannot represent.",
        "data_sha256": source_identity["data_sha256"],
        "source_sha256": source_sha,
        "parent_commit": source_identity["commit"],
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    }
    campaign.PROTOCOL["cards"][CARD] = registration
    campaign.PROTOCOL_PATH.write_text(
        json.dumps(campaign.PROTOCOL, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    campaign._append({"event": "card_registered", "card": CARD,
                      "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                      "expected_fit_starts": 6, "source_sha256": source_sha,
                      "data_sha256": source_identity["data_sha256"],
                      "parent_artifacts": parent_provenance,
                      "reserved_months_labels_opened": False, "hidden_labels_opened": False})
    campaign._write_progress("r125_registered", next_action="run registered two-track cyclic listing-month feature screen",
                             r125_card=CARD)
    return registration


def run_screen() -> dict:
    from data import hide_answers
    from experiments.bill_greedy import runner as greedy
    from experiments.model_optimization_v2 import worker

    card = campaign.PROTOCOL["cards"].get(CARD)
    if not card or card.get("status") != "registered":
        raise RuntimeError("R125 must be preregistered before any fit starts")
    if helpers._sha256(Path(__file__)) != card["source_sha256"]:
        raise RuntimeError("R125 source changed after preregistration")
    source = identity._source_identity()
    parents = {track: parent_loader._verified_parent(campaign, track) for track in TRACKS}

    run_id = "r125a_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = campaign.CAMPAIGN_ROOT / "batches" / run_id
    output.mkdir(parents=True, exist_ok=False)
    provenance = output / "provenance"
    provenance.mkdir()
    for track, (_, result_path, paired_path, _, _) in parents.items():
        prefix = "r22" if track == "off_market" else "r35"
        shutil.copy2(result_path, provenance / f"{prefix}_screen_results.json")
        shutil.copy2(paired_path, provenance / f"{prefix}_screen_paired_predictions.csv")
        shutil.copy2(result_path.parent / "run_manifest.json", provenance / f"{prefix}_screen_run_manifest.json")
    code_hashes = {str(path.relative_to(campaign.ROOT)).replace("\\", "/"): helpers._sha256(path)
                   for path in _source_files()}

    campaign._append({"event": "card_start", "card": CARD, "run_id": run_id,
                      "screen_months": list(SCREEN_MONTHS), "expected_fit_starts": 6,
                      "parent_ids": PARENT_IDS, "added_columns": list(ADDED_COLUMNS),
                      "data_sha256": source["data_sha256"],
                      "reserved_months_labels_opened": False, "hidden_labels_opened": False})

    old_normalize = worker._normalize_categories
    old_fit_record = greedy._fit_record
    old_variant = greedy.VARIANTS.get(VARIANT)
    greedy.VARIANTS[VARIANT] = {"model_config_overrides": {track: {} for track in TRACKS}}
    active: dict[str, str] = {}
    fit_details: list[dict] = []
    fold_rows: dict[str, list[pd.DataFrame]] = {track: [] for track in TRACKS}
    route_diagnostics: dict[str, dict] = {track: {} for track in TRACKS}

    def normalize_with_cyclic_month_features(train_x, eval_x, family):
        train_out, eval_out, categories = old_normalize(train_x, eval_x, family)
        missing_train = [name for name in ADDED_COLUMNS if name not in train_out]
        missing_eval = [name for name in ADDED_COLUMNS if name not in eval_out]
        if missing_train or missing_eval:
            raise RuntimeError(f"R125 cyclic features missing; train missing={missing_train}, eval missing={missing_eval}")
        for name in ADDED_COLUMNS:
            if not np.isfinite(pd.to_numeric(train_out[name], errors="coerce")).all():
                raise RuntimeError(f"R125 train feature {name} contains non-finite values")
            if not np.isfinite(pd.to_numeric(eval_out[name], errors="coerce")).all():
                raise RuntimeError(f"R125 evaluation feature {name} contains non-finite values")
        if list(train_out.columns) != list(eval_out.columns):
            raise RuntimeError("R125 cyclic features caused train/evaluation schema mismatch")
        return train_out, eval_out, categories

    def install_cyclic_text_features(fit_rows, eval_rows, track):
        from model_candidate_v2 import features as frozen_features

        original = frozen_features.off_market_features

        def with_cyclic_listing_month(homes):
            features = original(homes).copy()
            listed = pd.to_datetime(homes["date_listed"], errors="raise")
            if listed.isna().any():
                raise RuntimeError("R125 date_listed contains missing values")
            angle = 2.0 * np.pi * (listed.dt.month.astype(float) - 1.0) / 12.0
            features["listing_month_sin"] = np.sin(angle.to_numpy())
            features["listing_month_cos"] = np.cos(angle.to_numpy())
            return features

        frozen_features.off_market_features = with_cyclic_listing_month
        try:
            transformer, restore_text = r22._install_text_features(fit_rows, eval_rows, track)
        except Exception:
            frozen_features.off_market_features = original
            raise

        def restore():
            try:
                restore_text()
            finally:
                frozen_features.off_market_features = original

        return transformer, restore

    def record_fit(parent_fit_id: str, fit_fn):
        context = dict(active)
        if set(context) != {"track", "month", "component"}:
            raise RuntimeError("R125 fit context is incomplete")
        fit_id = (f"{campaign.PROTOCOL['campaign_id']}:{CARD}:screen:{context['track']}:{context['month']}"
                  f":{context['component']}:{run_id}")
        campaign._append({"event": "fit_start", "card": CARD, "stage": "screen", "run_id": run_id,
                          "fit_id": fit_id, "parent_fit_id": parent_fit_id, **context,
                          "data_sha256": source["data_sha256"],
                          "reserved_months_labels_opened": False, "hidden_labels_opened": False})
        started = time.perf_counter()
        try:
            fitted = fit_fn()
            columns = fitted["feature_columns"]
            missing = [name for name in ADDED_COLUMNS if name not in columns]
            if missing:
                raise RuntimeError(f"R125 fitted model omitted cyclic features: {missing}")
        except Exception as exc:
            campaign._append({"event": "fit_complete", "card": CARD, "stage": "screen", "run_id": run_id,
                              "fit_id": fit_id, **context, "status": "failed",
                              "seconds": time.perf_counter() - started,
                              "error": f"{type(exc).__name__}: {exc}"})
            raise
        detail = {"fit_id": fit_id, "parent_fit_id": parent_fit_id, **context,
                  "status": "complete", "seconds": time.perf_counter() - started,
                  "model_family": fitted["model_family"], "model_config": fitted["model_config"],
                  "target": fitted["target"], "actual_objective": fitted["actual_objective"],
                  "training_rows": fitted["training_rows"], "evaluation_rows": fitted["evaluation_rows"],
                  "added_features": list(ADDED_COLUMNS), "feature_columns": fitted["feature_columns"],
                  "prediction_sha256": fitted["prediction_sha256"]}
        fit_details.append(detail)
        campaign._append({"event": "fit_complete", "card": CARD, "stage": "screen", "run_id": run_id,
                          **detail, "reserved_months_labels_opened": False, "hidden_labels_opened": False})
        return fitted

    greedy._fit_record = record_fit
    worker._normalize_categories = normalize_with_cyclic_month_features
    try:
        for month in SCREEN_MONTHS:
            _, train, _, truth, heldout = greedy._load_fold(month)
            cutoff = pd.Period(month, freq="M").start_time
            train_dates = pd.to_datetime(train.date_sold, errors="raise")
            if train_dates.empty or (train_dates >= cutoff).any():
                raise RuntimeError(f"R125 training rows are not strictly before {month}")
            if train_dates.dt.to_period("M").astype(str).isin(("2025-05", "2025-08")).any():
                raise RuntimeError("R125 training rows include a sealed month")
            for track in TRACKS:
                homes = hide_answers(heldout, track)
                parent_prediction = parent_loader._align_parent(parents[track][4], month, homes, truth)
                detached_eval = greedy._specialist_mask(homes, HOUSE_TYPE)
                detached_train = greedy._specialist_mask(train, HOUSE_TYPE)
                if detached_eval.sum() < 100 or detached_train.sum() < 500:
                    raise RuntimeError(f"R125 Detached support too small in {track}/{month}")

                if track == "off_market":
                    fit_jobs = [
                        ("base", train.copy(), homes.copy(), "G06_lot_garage"),
                        ("detached_expert", train.loc[detached_train].copy(),
                         homes.loc[detached_eval].copy(), "G06_lot_garage"),
                    ]
                    fitted = {}
                    for component, fit_rows, eval_rows, base_variant in fit_jobs:
                        _, restore = install_cyclic_text_features(fit_rows, eval_rows, track)
                        active.clear(); active.update(track=track, month=month, component=component)
                        try:
                            model = greedy._fit_history_candidate(
                                month, track, train, eval_rows, base_variant=base_variant,
                                variant=VARIANT, fit_tag=f"R125-{component}-{month}", fit_rows=fit_rows)
                        finally:
                            restore()
                        fitted[component] = model
                    candidate_prediction = greedy._blend_specialist_predictions(
                        np.round(fitted["base"]["predictions"]),
                        np.round(fitted["detached_expert"]["predictions"]), homes,
                        HOUSE_TYPE, SPECIALIST_WEIGHT)
                    route_diagnostics[track][month] = {
                        "detached_rows": int(detached_eval.sum()),
                        "non_detached_route": "refit R22 base with cyclic month features; preserve the existing route",
                        "specialist_blend_weight": SPECIALIST_WEIGHT,
                    }
                else:
                    fit_rows = train.loc[detached_train].copy()
                    eval_rows = homes.loc[detached_eval].copy()
                    _, restore = install_cyclic_text_features(fit_rows, eval_rows, track)
                    active.clear(); active.update(track=track, month=month, component="detached_expert")
                    try:
                        model = greedy._fit_history_candidate(
                            month, track, train, eval_rows, variant=VARIANT,
                            fit_tag=f"R125-Detached-{month}", fit_rows=fit_rows)
                    finally:
                        restore()
                    frozen_hgbr = helpers._verified_hgbr_cache(greedy, month, track, train, homes)
                    specialist_prediction = (SPECIALIST_WEIGHT * np.asarray(model["predictions"], dtype=float)
                                             + (1.0 - SPECIALIST_WEIGHT) * frozen_hgbr[detached_eval])
                    candidate_prediction = greedy._overlay_specialist_predictions(
                        parent_prediction, homes, specialist_prediction, HOUSE_TYPE)
                    if not np.array_equal(candidate_prediction[~detached_eval], parent_prediction[~detached_eval]):
                        raise RuntimeError("R125 changed non-Detached R35 predictions")
                    route_diagnostics[track][month] = {
                        "detached_rows": int(detached_eval.sum()),
                        "non_detached_predictions_unchanged_from_R35": True,
                        "specialist_blend_weight": SPECIALIST_WEIGHT,
                        "frozen_hgbr_sha256": helpers._sha256(
                            greedy.OUTPUT / "frozen_hgbr" / track / f"{month}.csv"),
                    }

                parent_frame = pd.DataFrame({"ml_num": homes.ml_num.astype(str).to_numpy(),
                                             "sold_price": truth.sold_price.to_numpy(float),
                                             "predicted_price": parent_prediction})
                candidate_frame = pd.DataFrame({"ml_num": homes.ml_num.astype(str).to_numpy(),
                                                "sold_price": truth.sold_price.to_numpy(float),
                                                "predicted_price": candidate_prediction})
                paired = helpers._candidate_frame(month, track, truth, parent_frame, candidate_frame)
                if paired.ml_num.astype(str).duplicated().any() or len(paired) != len(truth):
                    raise RuntimeError(f"R125 paired scoring rows are not one-to-one for {track}/{month}")
                fold_rows[track].append(paired)
                campaign._append({"event": "screen_fold_complete", "card": CARD, "run_id": run_id,
                                  "track": track, "month": month, "rows": int(len(paired)),
                                  "training_cutoff_exclusive": cutoff.isoformat(),
                                  "added_features": list(ADDED_COLUMNS),
                                  "reserved_months_labels_opened": False, "hidden_labels_opened": False})
    except Exception as exc:
        campaign._append({"event": "card_failed", "card": CARD, "run_id": run_id,
                          "fit_starts": len(fit_details), "error": f"{type(exc).__name__}: {exc}",
                          "reserved_months_labels_opened": False, "hidden_labels_opened": False})
        campaign._write_progress("r125_screen_failed", next_action="audit R125 fit ledger and failure before any retry",
                                 r125_run_id=run_id)
        (output / "failure_checkpoint.json").write_text(
            json.dumps({"card": CARD, "run_id": run_id, "fit_details": fit_details,
                        "error": f"{type(exc).__name__}: {exc}",
                        "reserved_months_labels_opened": False, "hidden_labels_opened": False},
                       indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        raise
    finally:
        worker._normalize_categories = old_normalize
        greedy._fit_record = old_fit_record
        if old_variant is None:
            greedy.VARIANTS.pop(VARIANT, None)
        else:
            greedy.VARIANTS[VARIANT] = old_variant

    if len(fit_details) != card["expected_fit_starts"]:
        raise RuntimeError(f"R125 expected 6 completed fits, found {len(fit_details)}")
    paired_by_track = {track: pd.concat(fold_rows[track], ignore_index=True) for track in TRACKS}
    paired = pd.concat([paired_by_track[track] for track in TRACKS], ignore_index=True)
    paired_path = output / "paired_predictions.csv"
    paired.to_csv(paired_path, index=False)
    tracks = {}
    for track in TRACKS:
        score = r22.score_development_confirmation(
            paired_by_track[track], months=SCREEN_MONTHS,
            bootstrap_replicates=BOOTSTRAP_REPLICATES, bootstrap_seed=BOOTSTRAP_SEED,
            minimum_relative_gain_percent=0.25, minimum_months_improved=2,
            maximum_worst_month_regression_percent=0.0, maximum_mape_regression_percent=1.0)
        score["qualifies_screen"] = score.pop("qualifies_development")
        tracks[track] = score

    result = {"campaign_id": campaign.PROTOCOL["campaign_id"], "card": CARD, "run_id": run_id,
              "parent_ids": PARENT_IDS, "screen_months": list(SCREEN_MONTHS),
              "added_features": list(ADDED_COLUMNS), "fit_starts": len(fit_details),
              "fit_details": fit_details, "route_diagnostics": route_diagnostics,
              "tracks": tracks, "reserved_months_labels_opened": False,
              "hidden_labels_opened": False, "reserve_validation_eligible": False,
              "data_sha256": source["data_sha256"]}
    result_path = output / "screen_results.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    manifest = {
        "campaign_id": campaign.PROTOCOL["campaign_id"], "card": CARD, "run_id": run_id,
        "source_identity": source, "code_sha256": code_hashes,
        "parent_artifacts": card["parent_artifacts"],
        "result_sha256": helpers._sha256(result_path),
        "paired_predictions_sha256": helpers._sha256(paired_path),
        "fit_starts": len(fit_details), "fit_details": fit_details,
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                                               encoding="utf-8")
    report = [
        "# R125 挂牌月份周期特征", "",
        f"Campaign: `{campaign.PROTOCOL['campaign_id']}`; run: `{run_id}`.",
        "唯一变化：在原有线性挂牌月份特征上增加 `sin(2π(month-1)/12)` 与 `cos(2π(month-1)/12)`。原始经纬度、社区/市镇、建筑与 unit 历史、折内 TF-IDF、房型路线和冻结 HGBR 均保留。",
        "筛选仅使用 2023-03 与 2024-06；训练严格早于目标月。未读取预留月份或隐藏标签。", "",
        "| Track | Parent MdAPE | Candidate MdAPE | Relative gain | MAPE change | Months improved | Bootstrap upper95 (pp) | Pass |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for track, score in tracks.items():
        improved = sum(m["relative_mdape_improvement_percent"] > 0 for m in score["monthly"].values())
        report.append(f"| {track} | {score['parent']['mdape_percent']:.5f}% | {score['candidate']['mdape_percent']:.5f}% | "
                      f"{score['relative_mdape_improvement_percent']:+.3f}% | {score['mape_relative_change_percent']:+.3f}% | "
                      f"{improved}/2 | {score['bootstrap_mdape_difference_percent']['upper95']:+.5f} | "
                      f"{'yes' if score['qualifies_screen'] else 'no'} |")
    report.extend(["", "通过固定筛选的赛道才进入一次性八个月开发确认；本结果不能直接更换已验证父模型。", ""])
    (output / "research_report.md").write_text("\n".join(report), encoding="utf-8")

    qualifies = {track: score["qualifies_screen"] for track, score in tracks.items()}
    card.update({"status": "screen_passed" if any(qualifies.values()) else "screen_failed",
                 "run_id": run_id, "fit_starts": len(fit_details), "qualifies_by_track": qualifies,
                 "result_path": str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
                 "paired_predictions_path": str(paired_path.relative_to(campaign.ROOT)).replace("\\", "/"),
                 "manifest_path": str((output / "run_manifest.json").relative_to(campaign.ROOT)).replace("\\", "/"),
                 "screen_metrics": tracks, "reserved_months_labels_opened": False,
                 "hidden_labels_opened": False})
    campaign.PROTOCOL_PATH.write_text(json.dumps(campaign.PROTOCOL, indent=2, ensure_ascii=False) + "\n",
                                     encoding="utf-8")
    campaign._append({"event": "card_complete", "card": CARD, "run_id": run_id,
                      "fit_starts": len(fit_details), "qualifies_by_track": qualifies,
                      "result_path": str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
                      "paired_predictions_sha256": manifest["paired_predictions_sha256"],
                      "reserved_months_labels_opened": False, "hidden_labels_opened": False})
    next_action = ("run eight-month development confirmation for R125 tracks " +
                   ", ".join(track for track, passed in qualifies.items() if passed)
                   if any(qualifies.values()) else
                   "R125 failed both screens; keep R22/R35 and select a new preregistered mechanism")
    campaign._write_progress("r125_screen_passed" if any(qualifies.values()) else "r125_screen_failed",
                             next_action=next_action, r125_run_id=run_id,
                             r125_result=str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
                             r125_fit_starts=len(fit_details))
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Preregister or run R125 cyclic listing-month screen")
    parser.add_argument("--register-only", action="store_true")
    args = parser.parse_args()
    registration = register_card()
    if args.register_only:
        print(json.dumps(registration, indent=2, ensure_ascii=False))
        return
    print(json.dumps(run_screen(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

