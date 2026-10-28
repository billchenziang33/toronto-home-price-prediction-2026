"""Eight-month development confirmation for R125's screen-passing off-market route."""
from __future__ import annotations

import csv
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
from experiments.bill_target74 import runner as campaign


CARD = "R126_R125_offmarket_cyclic_month_development_confirmation"
R125_CARD = "R125_two_track_cyclic_listing_month_features"
TRACK = "off_market"
MONTHS = tuple(campaign.PROTOCOL["development_months"])
HOUSE_TYPE = "Detached"
SPECIALIST_WEIGHT = 0.75
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20261055
ADDED_COLUMNS = ("listing_month_sin", "listing_month_cos")
VARIANT = "R125_cyclic_listing_month_confirmation"


def _source_files() -> list[Path]:
    return [
        campaign.PROTOCOL_PATH, Path(__file__),
        campaign.PACKAGE / "r125_cyclic_listing_month_features.py",
        campaign.PACKAGE / "r22_remarks_tfidf.py",
        campaign.PACKAGE / "r03_catboost.py", campaign.PACKAGE / "r04_ablation.py",
        campaign.PACKAGE / "runner.py",
        campaign.ROOT / "experiments/bill_greedy/runner.py",
        campaign.ROOT / "experiments/bill_greedy/history.py",
        campaign.ROOT / "experiments/model_optimization_v2/worker.py",
        campaign.ROOT / "experiments/model_optimization_v2/learners.py",
        campaign.ROOT / "model_candidate_v2/features.py",
    ]


def _verified_r125_screen() -> tuple[dict, Path, Path, dict]:
    screen_card = campaign.PROTOCOL["cards"].get(R125_CARD)
    if not screen_card or screen_card.get("status") != "screen_passed":
        raise RuntimeError("R125 must have a completed screen before development confirmation")
    if screen_card.get("qualifies_by_track", {}).get(TRACK) is not True:
        raise RuntimeError("R125 off-market screen did not pass")
    result_path = campaign.ROOT / screen_card["result_path"]
    paired_path = campaign.ROOT / screen_card["paired_predictions_path"]
    manifest_path = campaign.ROOT / screen_card["manifest_path"]
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (result.get("card") != R125_CARD or result.get("run_id") != screen_card.get("run_id")
            or manifest.get("run_id") != result.get("run_id")
            or manifest.get("result_sha256") != helpers._sha256(result_path)
            or manifest.get("paired_predictions_sha256") != helpers._sha256(paired_path)
            or result.get("reserved_months_labels_opened") is not False
            or result.get("hidden_labels_opened") is not False):
        raise RuntimeError("R125 screen artifacts failed identity, hash, or data-access checks")
    if not result["tracks"][TRACK]["qualifies_screen"]:
        raise RuntimeError("R125 result does not confirm the off-market screen gate")
    return result, result_path, paired_path, manifest


def _verified_r22_development() -> tuple[dict, Path, Path, dict]:
    result, result_path, manifest = r22._verified_development_result(campaign)
    paired_path = result_path.parent / "paired_predictions.csv"
    if (result.get("data_sha256") != identity._source_identity()["data_sha256"]
            or result.get("reserved_months_labels_opened") is not False
            or result.get("hidden_labels_opened") is not False
            or tuple(result.get("development_months", ())) != MONTHS):
        raise RuntimeError("R22 development parent does not match the active safe data/month window")
    return result, result_path, paired_path, manifest


def register_card() -> dict:
    existing = campaign.PROTOCOL["cards"].get(CARD)
    if existing:
        if existing.get("status") != "registered":
            raise RuntimeError(f"{CARD} is already {existing.get('status')}; inspect its run before continuing")
        rows = ([json.loads(line) for line in campaign.LEDGER.read_text(encoding="utf-8").splitlines()
                 if line.strip()] if campaign.LEDGER.is_file() else [])
        if any(row.get("event") == "fit_start" and row.get("card") == CARD for row in rows):
            raise RuntimeError("R126 source cannot change after its first fit_start")
        current_sha = helpers._sha256(Path(__file__))
        if existing.get("source_sha256") != current_sha:
            raise RuntimeError("R126 source changed after registration; do not silently revise its method")
        return existing

    screen, screen_path, screen_paired, screen_manifest = _verified_r125_screen()
    parent, parent_path, parent_paired, parent_manifest = _verified_r22_development()
    source = identity._source_identity()
    if parent["data_sha256"] != source["data_sha256"]:
        raise RuntimeError("R22 parent predictions and active source data hashes differ")
    screen_track = screen["tracks"][TRACK]
    parent_track = parent["tracks"][TRACK]
    registration = {
        "status": "registered",
        "stage": "development_confirmation",
        "tracks": [TRACK], "development_months": list(MONTHS),
        "expected_fit_starts": len(MONTHS) * 2,
        "comparison": "R125 cyclic-feature candidate versus hash-verified R22 eight-month OOF predictions on identical sale rows",
        "method": "Repeat the screen-passing R125 route unchanged: add listing_month_sin/cos derived from date_listed; refit R22 off-market base and Detached expert with G06_lot_garage; round component predictions and blend Detached at 75/25.",
        "parent_id": "R22",
        "screen_run_id": screen["run_id"],
        "screen_result_sha256": helpers._sha256(screen_path),
        "screen_paired_predictions_sha256": helpers._sha256(screen_paired),
        "screen_gain_percent": screen_track["relative_mdape_improvement_percent"],
        "r22_development_run_id": parent["run_id"],
        "r22_development_result_sha256": helpers._sha256(parent_path),
        "r22_development_paired_predictions_sha256": helpers._sha256(parent_paired),
        "r22_development_parent_mdape_percent": parent_track["candidate"]["mdape_percent"],
        "target_mdape_strictly_below": campaign.PROTOCOL["initial_development_metrics"][TRACK]["target_mdape_strictly_below"],
        "model_family": "LightGBM",
        "added_features": list(ADDED_COLUMNS),
        "policy": "Use the fixed eight development months only. Each month trains strictly before its first day; fit each text vocabulary on that component's training remarks only. Do not tune this candidate on the development results. Never read 2025-05/08 or hidden labels.",
        "promotion_gates": campaign.PROTOCOL["gates"],
        "source_sha256": helpers._sha256(Path(__file__)),
        "data_sha256": source["data_sha256"],
        "parent_commit": source["commit"],
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    }
    campaign.PROTOCOL["cards"][CARD] = registration
    campaign.PROTOCOL_PATH.write_text(
        json.dumps(campaign.PROTOCOL, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    campaign._append({"event": "card_registered", "card": CARD,
                      "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                      "expected_fit_starts": registration["expected_fit_starts"],
                      "source_sha256": registration["source_sha256"],
                      "data_sha256": registration["data_sha256"],
                      "screen_run_id": screen["run_id"], "r22_development_run_id": parent["run_id"],
                      "reserved_months_labels_opened": False, "hidden_labels_opened": False})
    campaign._write_progress(
        "r126_registered",
        next_action="run the fixed eight-month R125 off-market cyclic-feature development confirmation",
        r126_card=CARD,
    )
    return registration


def _install_cyclic_text_features(fit_rows: pd.DataFrame, eval_rows: pd.DataFrame):
    from model_candidate_v2 import features as frozen_features

    original = frozen_features.off_market_features

    def with_cyclic_listing_month(homes):
        features = original(homes).copy()
        listed = pd.to_datetime(homes["date_listed"], errors="raise")
        if listed.isna().any():
            raise RuntimeError("R126 date_listed contains missing values")
        angle = 2.0 * np.pi * (listed.dt.month.astype(float) - 1.0) / 12.0
        features["listing_month_sin"] = np.sin(angle.to_numpy())
        features["listing_month_cos"] = np.cos(angle.to_numpy())
        return features

    frozen_features.off_market_features = with_cyclic_listing_month
    try:
        transformer, restore_text = r22._install_text_features(fit_rows, eval_rows, TRACK)
    except Exception:
        frozen_features.off_market_features = original
        raise

    def restore():
        try:
            restore_text()
        finally:
            frozen_features.off_market_features = original

    return transformer, restore


def run_confirmation() -> dict:
    from data import hide_answers
    from experiments.bill_greedy import runner as greedy
    from experiments.model_optimization_v2 import worker

    card = campaign.PROTOCOL["cards"].get(CARD)
    if not card or card.get("status") != "registered":
        raise RuntimeError("R126 must be preregistered before any fit starts")
    if helpers._sha256(Path(__file__)) != card["source_sha256"]:
        raise RuntimeError("R126 source changed after preregistration")
    screen, screen_path, screen_paired, screen_manifest = _verified_r125_screen()
    parent, parent_path, parent_paired_path, parent_manifest = _verified_r22_development()
    source = identity._source_identity()
    if source["data_sha256"] != card["data_sha256"]:
        raise RuntimeError("R126 data source changed after preregistration")
    parent_rows = pd.read_csv(parent_paired_path, dtype={"ml_num": str})
    parent_rows = parent_rows.loc[parent_rows.track.eq(TRACK)].copy()
    if parent_rows.duplicated(["month", "ml_num"]).any():
        raise RuntimeError("R22 development parent contains duplicate month/home keys")

    run_id = "r126d_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = campaign.CAMPAIGN_ROOT / "batches" / run_id
    output.mkdir(parents=True, exist_ok=False)
    provenance = output / "provenance"
    provenance.mkdir()
    for source_path, name in [
        (screen_path, "r125_screen_results.json"),
        (screen_paired, "r125_screen_paired_predictions.csv"),
        (screen_path.parent / "run_manifest.json", "r125_screen_run_manifest.json"),
        (parent_path, "r22_development_results.json"),
        (parent_paired_path, "r22_development_paired_predictions.csv"),
        (parent_path.parent / "run_manifest.json", "r22_development_run_manifest.json"),
    ]:
        shutil.copy2(source_path, provenance / name)
    code_hashes = {str(path.relative_to(campaign.ROOT)).replace("\\", "/"): helpers._sha256(path)
                   for path in _source_files()}
    campaign._append({"event": "card_start", "card": CARD,
                      "stage": "development_confirmation", "run_id": run_id,
                      "screen_run_id": screen["run_id"], "development_months": list(MONTHS),
                      "expected_fit_starts": card["expected_fit_starts"],
                      "added_features": list(ADDED_COLUMNS), "data_sha256": source["data_sha256"],
                      "reserved_months_labels_opened": False, "hidden_labels_opened": False})

    old_fit_record = greedy._fit_record
    old_normalize = worker._normalize_categories
    old_variant = greedy.VARIANTS.get(VARIANT)
    greedy.VARIANTS[VARIANT] = {"model_config_overrides": {TRACK: {}}}
    fit_details: list[dict] = []
    paired_by_month: list[pd.DataFrame] = []
    active: dict[str, str] = {}

    def normalize_with_cyclic_features(train_x, eval_x, family):
        train_out, eval_out, categories = old_normalize(train_x, eval_x, family)
        for name in ADDED_COLUMNS:
            if name not in train_out or name not in eval_out:
                raise RuntimeError(f"R126 model matrix is missing {name}")
            if (not np.isfinite(pd.to_numeric(train_out[name], errors="coerce")).all()
                    or not np.isfinite(pd.to_numeric(eval_out[name], errors="coerce")).all()):
                raise RuntimeError(f"R126 model matrix contains non-finite {name}")
        if list(train_out.columns) != list(eval_out.columns):
            raise RuntimeError("R126 training and evaluation feature schemas differ")
        return train_out, eval_out, categories

    def record_fit(parent_fit_id: str, fit_fn):
        if set(active) != {"month", "component"}:
            raise RuntimeError("R126 fit context is incomplete")
        month, component = active["month"], active["component"]
        fit_id = f"{campaign.PROTOCOL['campaign_id']}:{CARD}:development:{month}:{component}:{run_id}"
        campaign._append({"event": "fit_start", "card": CARD, "stage": "development_confirmation",
                          "run_id": run_id, "fit_id": fit_id, "parent_fit_id": parent_fit_id,
                          "track": TRACK, "month": month, "component": component,
                          "data_sha256": source["data_sha256"],
                          "reserved_months_labels_opened": False, "hidden_labels_opened": False})
        started = time.perf_counter()
        try:
            fitted = fit_fn()
            columns = fitted["feature_columns"]
            if any(name not in columns for name in ADDED_COLUMNS):
                raise RuntimeError("R126 fitted model did not use both cyclic features")
        except Exception as exc:
            campaign._append({"event": "fit_complete", "card": CARD,
                              "stage": "development_confirmation", "run_id": run_id,
                              "fit_id": fit_id, "track": TRACK, "month": month,
                              "component": component, "status": "failed",
                              "seconds": time.perf_counter() - started,
                              "error": f"{type(exc).__name__}: {exc}"})
            raise
        detail = {"fit_id": fit_id, "parent_fit_id": parent_fit_id,
                  "track": TRACK, "month": month, "component": component,
                  "status": "complete", "seconds": time.perf_counter() - started,
                  "model_family": fitted["model_family"], "model_config": fitted["model_config"],
                  "target": fitted["target"], "training_rows": fitted["training_rows"],
                  "evaluation_rows": fitted["evaluation_rows"],
                  "feature_columns": fitted["feature_columns"],
                  "prediction_sha256": fitted["prediction_sha256"]}
        fit_details.append(detail)
        campaign._append({"event": "fit_complete", "card": CARD,
                          "stage": "development_confirmation", "run_id": run_id, **detail,
                          "reserved_months_labels_opened": False, "hidden_labels_opened": False})
        return fitted

    greedy._fit_record = record_fit
    worker._normalize_categories = normalize_with_cyclic_features
    try:
        for month in MONTHS:
            _, train, _, truth, heldout = greedy._load_fold(month)
            cutoff = pd.Period(month, freq="M").start_time
            train_dates = pd.to_datetime(train.date_sold, errors="raise")
            if train_dates.empty or (train_dates >= cutoff).any():
                raise RuntimeError(f"R126 training rows are not strictly before {month}")
            if train_dates.dt.to_period("M").astype(str).isin(("2025-05", "2025-08")).any():
                raise RuntimeError("R126 training rows include a sealed month")
            homes = hide_answers(heldout, TRACK)
            baseline = parent_rows.loc[parent_rows.month.eq(month)].copy()
            if baseline["ml_num"].duplicated().any() or len(baseline) != len(homes):
                raise RuntimeError(f"R22 baseline rows are not one-to-one with {month} heldout homes")
            baseline = baseline.set_index(baseline.ml_num.astype(str))
            ids = homes.ml_num.astype(str).tolist()
            if set(ids) != set(baseline.index.astype(str)):
                raise RuntimeError(f"R22 baseline ID set differs from {month} heldout homes")
            baseline = baseline.loc[ids]
            if not np.array_equal(baseline.sold_price.to_numpy(float), truth.sold_price.to_numpy(float)):
                raise RuntimeError(f"R22 baseline truth differs from the current {month} fold")
            parent_prediction = baseline.candidate_prediction.to_numpy(float)

            detached_train = greedy._specialist_mask(train, HOUSE_TYPE)
            detached_eval = greedy._specialist_mask(homes, HOUSE_TYPE)
            if detached_eval.sum() < 100 or detached_train.sum() < 500:
                raise RuntimeError(f"R126 Detached support too small in {month}")
            components = {}
            jobs = [
                ("base", train.copy(), homes.copy()),
                ("detached_expert", train.loc[detached_train].copy(), homes.loc[detached_eval].copy()),
            ]
            for component, fit_rows, eval_rows in jobs:
                transformer, restore = _install_cyclic_text_features(fit_rows, eval_rows)
                active.clear(); active.update(month=month, component=component)
                try:
                    fitted = greedy._fit_history_candidate(
                        month, TRACK, train, eval_rows, base_variant="G06_lot_garage",
                        variant=VARIANT, fit_tag=f"R126-{component}-{month}", fit_rows=fit_rows)
                finally:
                    restore()
                components[component] = fitted
                if not transformer.columns_:
                    raise RuntimeError(f"R126 fold-local text transformer was not fitted in {month}/{component}")

            candidate_prediction = greedy._blend_specialist_predictions(
                np.round(components["base"]["predictions"]),
                np.round(components["detached_expert"]["predictions"]),
                homes, HOUSE_TYPE, SPECIALIST_WEIGHT)
            parent_frame = pd.DataFrame({"ml_num": ids, "sold_price": truth.sold_price.to_numpy(float),
                                         "predicted_price": parent_prediction})
            candidate_frame = pd.DataFrame({"ml_num": ids, "sold_price": truth.sold_price.to_numpy(float),
                                            "predicted_price": np.asarray(candidate_prediction, dtype=float)})
            paired = helpers._candidate_frame(month, TRACK, truth, parent_frame, candidate_frame)
            if paired.ml_num.astype(str).duplicated().any() or len(paired) != len(truth):
                raise RuntimeError(f"R126 paired rows are not one-to-one for {month}")
            paired_by_month.append(paired)
            pd.concat(paired_by_month, ignore_index=True).to_csv(output / "partial_off_market.csv", index=False)
            campaign._append({"event": "development_fold_complete", "card": CARD,
                              "stage": "development_confirmation", "run_id": run_id,
                              "track": TRACK, "month": month, "rows": len(paired),
                              "cutoff_exclusive": cutoff.isoformat(),
                              "reserved_months_labels_opened": False, "hidden_labels_opened": False})
    except Exception as exc:
        checkpoint = {"card": CARD, "run_id": run_id,
                      "fit_starts": len(fit_details), "fit_details": fit_details,
                      "completed_months": [frame.month.iloc[0] for frame in paired_by_month],
                      "error": f"{type(exc).__name__}: {exc}",
                      "reserved_months_labels_opened": False, "hidden_labels_opened": False}
        (output / "failure_checkpoint.json").write_text(
            json.dumps(checkpoint, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        campaign._append({"event": "card_failed", "card": CARD,
                          "stage": "development_confirmation", "run_id": run_id,
                          "fit_starts": len(fit_details), "error": checkpoint["error"],
                          "reserved_months_labels_opened": False, "hidden_labels_opened": False})
        campaign._write_progress("r126_development_failed",
                                 next_action="audit R126 checkpoint and fit ledger before any retry",
                                 r126_run_id=run_id)
        raise
    finally:
        greedy._fit_record = old_fit_record
        worker._normalize_categories = old_normalize
        if old_variant is None:
            greedy.VARIANTS.pop(VARIANT, None)
        else:
            greedy.VARIANTS[VARIANT] = old_variant

    if len(fit_details) != card["expected_fit_starts"]:
        raise RuntimeError(f"R126 expected {card['expected_fit_starts']} fits, found {len(fit_details)}")
    paired = pd.concat(paired_by_month, ignore_index=True)
    paired_path = output / "paired_predictions.csv"
    paired.to_csv(paired_path, index=False)
    score = r22.score_development_confirmation(
        paired, months=MONTHS,
        target_mdape_strictly_below=card["target_mdape_strictly_below"],
        bootstrap_replicates=BOOTSTRAP_REPLICATES, bootstrap_seed=BOOTSTRAP_SEED,
        minimum_relative_gain_percent=campaign.PROTOCOL["gates"]["development_relative_mdape_improvement_percent"],
        minimum_months_improved=campaign.PROTOCOL["gates"]["development_months_improved_min"],
        maximum_worst_month_regression_percent=campaign.PROTOCOL["gates"]["development_worst_month_regression_max_percent"],
        maximum_mape_regression_percent=campaign.PROTOCOL["gates"]["development_mape_regression_max_percent"],
    )
    result = {"campaign_id": campaign.PROTOCOL["campaign_id"], "card": CARD,
              "run_id": run_id, "stage": "development_confirmation", "screen_run_id": screen["run_id"],
              "parent_run_id": parent["run_id"], "track": TRACK, "development_months": list(MONTHS),
              "fit_starts": len(fit_details), "fit_details": fit_details,
              "added_features": list(ADDED_COLUMNS), "tracks": {TRACK: score},
              "reserved_months_labels_opened": False, "hidden_labels_opened": False,
              "reserve_validation_eligible": False, "data_sha256": source["data_sha256"]}
    result_path = output / "development_results.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n",
                           encoding="utf-8")
    manifest = {"campaign_id": campaign.PROTOCOL["campaign_id"], "card": CARD, "run_id": run_id,
                "source_identity": source, "code_sha256": code_hashes,
                "screen_result_sha256": helpers._sha256(screen_path),
                "screen_paired_predictions_sha256": screen_manifest["paired_predictions_sha256"],
                "r22_development_result_sha256": helpers._sha256(parent_path),
                "r22_development_paired_predictions_sha256": helpers._sha256(parent_paired_path),
                "result_sha256": helpers._sha256(result_path),
                "paired_predictions_sha256": helpers._sha256(paired_path),
                "fit_starts": len(fit_details), "fit_details": fit_details,
                "reserved_months_labels_opened": False, "hidden_labels_opened": False}
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                                               encoding="utf-8")
    report = [
        "# R126 — R125 off-market 八个月开发确认", "",
        f"Campaign: `{campaign.PROTOCOL['campaign_id']}`; run: `{run_id}`.",
        "固定复用 R125 筛选通过的单一方法：只新增挂牌月份 sine/cosine 特征，其他模型、特征、训练窗口、Detached 路由及 75/25 混合保持一致。",
        "父版为哈希验证的 R22 八个月 walk-forward OOF 预测。没有读取 2025-05/08 或隐藏标签。", "",
        f"Pooled MdAPE: `{score['parent']['mdape_percent']:.6f}%` → `{score['candidate']['mdape_percent']:.6f}%`; "
        f"relative gain `{score['relative_mdape_improvement_percent']:+.4f}%`; "
        f"MAPE change `{score['mape_relative_change_percent']:+.4f}%`; "
        f"months improved `{score['months_improved']}/8`; "
        f"target `<{card['target_mdape_strictly_below']:.2f}%`: `{score['candidate']['mdape_percent'] < card['target_mdape_strictly_below']}`; "
        f"development gate: `{score['qualifies_development']}`.", "",
        "| Month | Parent MdAPE | Candidate MdAPE | Relative gain |",
        "|---|---:|---:|---:|",
    ]
    for month, monthly in score["monthly"].items():
        report.append(f"| {month} | {monthly['parent']['mdape_percent']:.5f}% | "
                      f"{monthly['candidate']['mdape_percent']:.5f}% | "
                      f"{monthly['relative_mdape_improvement_percent']:+.3f}% |")
    report.extend(["", "保留集验证因既往完整 parquet 访问事件不具备资格；本次仅作固定开发窗口确认。", ""])
    (output / "research_report.md").write_text("\n".join(report), encoding="utf-8")

    card.update({"status": "development_complete",
                 "run_id": run_id, "fit_starts": len(fit_details),
                 "development_metrics": score,
                 "qualifies_by_track": {TRACK: score["qualifies_development"]},
                 "result_path": str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
                 "paired_predictions_path": str(paired_path.relative_to(campaign.ROOT)).replace("\\", "/"),
                 "manifest_path": str((output / "run_manifest.json").relative_to(campaign.ROOT)).replace("\\", "/"),
                 "result_sha256": manifest["result_sha256"],
                 "paired_predictions_sha256": manifest["paired_predictions_sha256"],
                 "reserved_months_labels_opened": False, "hidden_labels_opened": False})
    campaign.PROTOCOL_PATH.write_text(json.dumps(campaign.PROTOCOL, indent=2, ensure_ascii=False) + "\n",
                                     encoding="utf-8")
    campaign._append({"event": "card_complete", "card": CARD,
                      "stage": "development_confirmation", "run_id": run_id,
                      "fit_starts": len(fit_details),
                      "qualifies_by_track": {TRACK: score["qualifies_development"]},
                      "result_path": str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
                      "result_sha256": manifest["result_sha256"],
                      "paired_predictions_sha256": manifest["paired_predictions_sha256"],
                      "reserved_months_labels_opened": False, "hidden_labels_opened": False})
    next_action = ("run R125 off-market fixed-origin confirmation; reserve months remain ineligible"
                   if score["qualifies_development"] else
                   "retain R22; continue with a distinct preregistered off-market experiment")
    campaign._write_progress("r126_development_complete", next_action=next_action,
                             r126_run_id=run_id,
                             r126_result=str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
                             r126_fit_starts=len(fit_details))
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Preregister or run R125 off-market development confirmation")
    parser.add_argument("--register-only", action="store_true")
    args = parser.parse_args()
    registration = register_card()
    if args.register_only:
        print(json.dumps(registration, indent=2, ensure_ascii=False))
        return
    print(json.dumps(run_confirmation(), indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
