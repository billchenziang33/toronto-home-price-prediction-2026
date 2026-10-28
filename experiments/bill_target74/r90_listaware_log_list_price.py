"""R90: screen raw log(list_price) as a list-aware feature against R35."""
from __future__ import annotations

import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.bill_target74.r19_expert_weight import _sha256, _write_json


CARD = "R90_listaware_log_list_price"
VARIANT = CARD
PARENT_CARD = "R35_listaware_detached_r22_text_expert"
TRACK = "list_aware"
HOUSE_TYPE = "Detached"
SCREEN_MONTHS = ("2023-03", "2024-06")
CORE_WEIGHT = 0.75
MODEL_OVERRIDES = {}


def _source_files(campaign) -> list[Path]:
    return [
        campaign.PROTOCOL_PATH,
        Path(__file__),
        campaign.PACKAGE / "r34_house_type_text_experts.py",
        campaign.PACKAGE / "r73_listaware_log_list_ppsf.py",
        campaign.ROOT / "outputs/bill_target74/bill_target74_unbounded_20261024/diagnostics/r35_listaware_residual_segments_20261025.json",
        campaign.PACKAGE / "r22_remarks_tfidf.py",
        campaign.PACKAGE / "r23_word_char_tfidf.py",
        campaign.PACKAGE / "r03_catboost.py",
        campaign.PACKAGE / "r04_ablation.py",
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



def _install_list_price_text_features(r22, fit_rows: pd.DataFrame, eval_rows: pd.DataFrame):
    """Add raw log(list_price) and fold-local TF-IDF; restore global builders afterward."""
    from model_candidate_v2 import features as frozen_features

    original = frozen_features.list_aware_features
    fit_identity, eval_identity = id(fit_rows), id(eval_rows)
    support: dict[str, dict[str, int]] = {}

    def with_list_price(homes: pd.DataFrame) -> pd.DataFrame:
        result = original(homes)
        if "log_list_price" in result:
            raise RuntimeError("R90 feature already exists in the frozen builder")
        listed = pd.to_numeric(result["list_price"], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(listed) & (listed > 0.0)
        values = np.full(len(result), np.nan, dtype=float)
        values[valid] = np.log(listed[valid])
        result["log_list_price"] = values
        key = "training" if id(homes) == fit_identity else "evaluation" if id(homes) == eval_identity else None
        if key is None:
            raise RuntimeError("R90 feature builder received an unregistered frame")
        support[key] = {"rows": int(len(result)), "valid": int(valid.sum()), "missing": int((~valid).sum())}
        return result

    frozen_features.list_aware_features = with_list_price
    try:
        transformer, restore_text = r22._install_text_features(fit_rows, eval_rows, TRACK)
    except Exception:
        frozen_features.list_aware_features = original
        raise

    def restore() -> None:
        restore_text()
        frozen_features.list_aware_features = original

    return transformer, restore, support

def _parent_artifacts(campaign) -> tuple[dict, Path, Path, dict, pd.DataFrame]:
    parent_card = campaign.PROTOCOL["cards"].get(PARENT_CARD)
    if not parent_card or parent_card.get("status") != "fixed_origin_confirmation_passed":
        raise RuntimeError("R90 requires the validated R35 list-aware parent")
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
        raise RuntimeError(f"R90 is already {existing.get('status')}; inspect before rerunning")
    parent = campaign.PROTOCOL["cards"][PARENT_CARD]
    source = budget._source_identity()
    diagnostic_path = campaign.ROOT / "outputs/bill_target74/bill_target74_unbounded_20261024/diagnostics/r35_listaware_residual_segments_20261025.json"
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    quartiles = {item["segment"]["list_price_quartile"]: item["pooled"]["mdape_percent"]
                 for item in diagnostic.get("by_list_price_quartile", [])}
    if (diagnostic.get("track") != TRACK
            or diagnostic.get("data_sha256_expected") != source["data_sha256"]
            or diagnostic.get("development_months") != [
                "2023-03", "2023-07", "2023-11", "2024-03",
                "2024-06", "2024-09", "2025-01", "2025-03",
            ]
            or diagnostic.get("reserved_months_labels_opened") is not False
            or diagnostic.get("hidden_labels_opened") is not False
            or not (quartiles.get("Q4", 0.0) > quartiles.get("Q1", 0.0))):
        raise RuntimeError("R90 motivation diagnostic does not match the safe R35 development audit")
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
        "new_feature": "log_list_price = log(list_price), missing when list_price is nonpositive or unavailable",
        "motivation_diagnostic": {
            "path": "outputs/bill_target74/bill_target74_unbounded_20261024/diagnostics/r35_listaware_residual_segments_20261025.json",
            "sha256": _sha256(campaign.ROOT / "outputs/bill_target74/bill_target74_unbounded_20261024/diagnostics/r35_listaware_residual_segments_20261025.json"),
        },
        "method": (
            "Fit the unchanged full-population R22 list-aware LightGBM core with its fold-local word TF-IDF, "
            "history features, log(sold_price/list_price) target, and frozen 75/25 core/HGBR blend. Add only "
            "the listing-time feature log(list_price). Retain every R35 Detached prediction "
            "unchanged; use the feature-augmented core only on non-Detached rows."
        ),
        "difference_from_prior_work": (
            "R73 added log(list_price / square_feet) to the same non-Detached core route and improved pooled screen MdAPE 0.463%, "
            "but June regressed 0.887%, so it failed the fixed screen. The hash-checked R35 eight-month diagnostic shows "
            "MdAPE rising from 4.092% in the lowest list-price quartile to 5.403% in the highest. R90 tests the distinct raw "
            "log(list_price) level feature, without dividing by area, against the same R35 route. It preserves every Detached "
            "prediction, target, learner, blend, and other feature. The diagnostic is descriptive and does not tune on the two screen labels."
        ),
        "incremental_screen_gate": campaign.PROTOCOL["gates"],
        "policy": (
            "Use only strict pre-month rows from the two registered development folds; "
            "fit TF-IDF on all list-aware training remarks in that fold; verify the new feature reaches "
            "the learner and frozen HGBR row alignment; preserve all R35 Detached predictions and retain "
            "R35 unless every screen gate passes. Never read "
            "2025-05/08 or hidden labels."
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
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    campaign._write_progress(
        "r90_registered",
        next_action="run the preregistered two-fold R90 list-aware log(list_price) feature screen; keep R35 unless every gate passes",
        r90_card=CARD,
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
        raise RuntimeError("R90 must be preregistered before fitting")
    source = budget._source_identity()
    if source["data_sha256"] != card["data_sha256"]:
        raise RuntimeError("R90 data snapshot changed after preregistration")
    parent_result, parent_path, parent_rows_path, parent_manifest, parent_rows = _parent_artifacts(campaign)
    if parent_result["run_id"] != card["parent_run_id"]:
        raise RuntimeError("R90 parent run differs from preregistration")

    run_id = "r90p_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = campaign.CAMPAIGN_ROOT / "batches" / run_id
    output.mkdir(parents=True, exist_ok=False)
    provenance = output / "provenance"
    provenance.mkdir()
    shutil.copy2(parent_path, provenance / "r35_screen_results.json")
    shutil.copy2(parent_rows_path, provenance / "r35_screen_paired_predictions.csv")
    shutil.copy2(parent_path.parent / "run_manifest.json", provenance / "r35_screen_run_manifest.json")

    plan = [(month, "full_listaware_core") for month in SCREEN_MONTHS]
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
        fit_id = f"{campaign.PROTOCOL['campaign_id']}:{CARD}:screen:{TRACK}:{month}:full_listaware_core:{run_id}"
        fit_start_count += 1
        campaign._append({
            "event": "fit_start", "card": CARD, "stage": "screen", "run_id": run_id,
            "fit_id": fit_id, "track": TRACK, "month": month,
            "component": "full_listaware_core", "training_rows": active["training_rows"],
            "evaluation_rows": active["evaluation_rows"], "data_sha256": source["data_sha256"],
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        started = time.perf_counter()
        try:
            fitted = fit_fn()
            if fitted.get("target") != "log_sold_over_list":
                raise RuntimeError(f"R90 expected log_sold_over_list; got {fitted.get('target')!r}")
            if "log_list_price" not in fitted.get("feature_columns", []):
                raise RuntimeError("R90 raw log_list_price feature did not reach the learner")
        except Exception as exc:
            campaign._append({
                "event": "fit_complete", "card": CARD, "stage": "screen", "run_id": run_id,
                "fit_id": fit_id, "track": TRACK, "month": month, "status": "failed",
                "seconds": time.perf_counter() - started, "error": f"{type(exc).__name__}: {exc}",
            })
            raise
        detail = {
            "fit_id": fit_id, "track": TRACK, "month": month,
            "component": "full_listaware_core", "seconds": time.perf_counter() - started,
            "model_family": fitted["model_family"], "model_config": fitted["model_config"],
            "actual_objective": fitted["actual_objective"], "target": fitted["target"],
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
    try:
        for month in SCREEN_MONTHS:
            _, train, _, truth, heldout = greedy._load_fold(month)
            cutoff = pd.Period(month, freq="M").start_time
            if (pd.to_datetime(train.date_sold, errors="raise") >= cutoff).any():
                raise RuntimeError(f"R90 training rows are not strictly before {month}")
            homes = hide_answers(heldout, TRACK)
            fit_rows = train.copy()
            eval_rows = homes.copy()
            if len(fit_rows) < 1000 or len(eval_rows) < 100:
                raise RuntimeError(f"R90 full-track support is too small in {month}")

            saved = parent_rows.loc[parent_rows.month.eq(month)].copy()
            saved = saved.set_index(saved.ml_num.astype(str)).reindex(homes.ml_num.astype(str))
            if (saved.index.has_duplicates
                    or not np.array_equal(saved.sold_price.to_numpy(float), truth.sold_price.to_numpy(float))
                    or not np.isfinite(saved.candidate_prediction.to_numpy(float)).all()):
                raise RuntimeError(f"R35 parent predictions do not align for {month}")
            parent_prediction = saved.candidate_prediction.to_numpy(float)
            detached_mask = greedy._specialist_mask(homes, HOUSE_TYPE)
            if not detached_mask.any() or detached_mask.all():
                raise RuntimeError(f"R90 expected both Detached and non-Detached rows in {month}")

            transformer, restore, feature_support = _install_list_price_text_features(
                r22, fit_rows, eval_rows,
            )
            active.update(month=month, training_rows=len(fit_rows), evaluation_rows=len(eval_rows))
            try:
                fitted = greedy._fit_history_candidate(
                    month, TRACK, train, eval_rows, fit_rows=fit_rows, variant=VARIANT,
                    base_variant=None, fit_tag=f"R90-{month}-full-listaware",
                )
            finally:
                restore()
            if set(feature_support) != {"training", "evaluation"}:
                raise RuntimeError(f"R90 feature coverage was not recorded for {month}")
            frozen_hgbr = helpers._verified_hgbr_cache(greedy, month, TRACK, train, homes)
            full_candidate = blend_prices(
                fitted["predictions"], frozen_hgbr, challenger_weight=CORE_WEIGHT,
            )
            prediction = full_candidate.copy()
            prediction[detached_mask] = parent_prediction[detached_mask]
            if not np.array_equal(prediction[detached_mask], parent_prediction[detached_mask]):
                raise RuntimeError(f"R90 changed validated R35 Detached predictions in {month}")
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
                "detached_rows": int(detached_mask.sum()),
                "non_detached_rows_changed_by_candidate": int((~detached_mask).sum()),
                "detached_predictions_unchanged": True,
                "feature_support": feature_support,
                "parent_all": campaign._metrics(actual, parent_prediction),
                "candidate_all": campaign._metrics(actual, prediction),
                "parent_non_detached": campaign._metrics(actual[~detached_mask], parent_prediction[~detached_mask]),
                "candidate_non_detached": campaign._metrics(actual[~detached_mask], prediction[~detached_mask]),
                "feature_columns": fitted["feature_columns"],
                "tfidf_terms": transformer.vectorizer.get_feature_names_out().tolist(),
                "actual_objective": fitted["actual_objective"], "target": fitted["target"],
            }
            campaign._append({
                "event": "screen_fold_complete", "card": CARD, "run_id": run_id,
                "track": TRACK, "month": month, "training_cutoff_exclusive": cutoff.isoformat(),
                "training_rows": int(len(fit_rows)), "evaluation_rows": int(len(homes)),
                "detached_rows_preserved": int(detached_mask.sum()),
                "new_feature": "log_list_price",
                "feature_support": feature_support,
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
        campaign._write_progress("r90_screen_failed",
                                 next_action="inspect R90 failure checkpoint and ledger before any retry",
                                 r90_failure=str(failure_path.relative_to(campaign.ROOT)).replace("\\", "/"))
        raise
    finally:
        greedy._fit_record = old_fit_record
        if old_variant is None:
            greedy.VARIANTS.pop(VARIANT, None)
        else:
            greedy.VARIANTS[VARIANT] = old_variant

    if fit_start_count != len(plan) or len(fit_details) != len(plan):
        raise RuntimeError(
            f"R90 expected {len(plan)} completed fits; starts={fit_start_count}, completed={len(fit_details)}"
        )
    paired = pd.concat(paired_parts, ignore_index=True)
    paired_path = output / "paired_predictions.csv"
    paired.to_csv(paired_path, index=False)
    score = scoring._track_result(paired)
    result = {
        "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id, "card": CARD,
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
    for index, path in enumerate(source_files):
        shutil.copy2(path, provenance / f"src_{index:02d}_{path.name}.source")
    manifest = {
        "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
        "source_identity": source, "code_sha256": code_hashes,
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
        "# R90: list-aware raw log(list_price) feature screen", "",
        f"Run: `{run_id}`; parent: validated R35 `{parent_result['run_id']}`.",
        "The only new feature is log(list_price), computed from listing-time inputs. "
        "The model uses the unchanged log(sold/list) target, fold-local word TF-IDF, strict time cutoff, "
        "and 75/25 LightGBM/frozen-HGBR blend. R35 Detached predictions are copied exactly; the new "
        "feature-augmented predictions apply only to non-Detached rows.", "",
        f"- R35 screen MdAPE: {score['parent']['mdape_percent']:.5f}%",
        f"- R90 screen MdAPE: {score['candidate']['mdape_percent']:.5f}%",
        f"- Relative MdAPE improvement: {score['relative_mdape_improvement_percent']:+.3f}%",
        f"- Relative MAPE change: {score['mape_relative_change_percent']:+.3f}%",
        f"- Screen passed: {score['qualifies_screen']}", "",
        "## Monthly results", "", "| Month | R35 MdAPE | R90 MdAPE | Relative improvement |",
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
        "status": "screen_passed" if score["qualifies_screen"] else "screen_failed",
        "run_id": run_id, "fit_starts": len(fit_details),
        "result_path": str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        "paired_predictions_path": str(paired_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        "manifest_path": str((output / "run_manifest.json").relative_to(campaign.ROOT)).replace("\\", "/"),
        "screen_metrics": score, "reserved_months_labels_opened": False,
        "hidden_labels_opened": False,
    })
    campaign.PROTOCOL_PATH.write_text(
        json.dumps(campaign.PROTOCOL, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    campaign._append({
        "event": "card_complete", "card": CARD, "run_id": run_id,
        "fit_starts": len(fit_details), "result_path": card["result_path"],
        "qualifies_screen": score["qualifies_screen"], "screen_metrics": score,
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    next_action = (
        "R90 passed its fixed screen; run an eight-month confirmation against R35 while preserving Detached predictions"
        if score["qualifies_screen"] else
        "R90 failed its fixed screen; keep R35 and register a distinct list-aware hypothesis"
    )
    campaign._write_progress(
        "r90_screen_passed" if score["qualifies_screen"] else "r90_screen_failed",
        next_action=next_action,
        r90_run_id=run_id, r90_result=card["result_path"],
        r90_fit_starts=len(fit_details),
    )
    return result


def run() -> dict:
    register_card()
    return run_screen()


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, ensure_ascii=False))
