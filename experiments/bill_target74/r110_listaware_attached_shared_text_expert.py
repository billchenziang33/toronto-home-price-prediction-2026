"""R110: shared R22-text list-aware expert for attached, non-apartment home types."""
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

from experiments.bill_target74 import r03_catboost as r03
from experiments.bill_target74 import r04_ablation as identity
from experiments.bill_target74 import r22_remarks_tfidf as r22
from experiments.bill_target74 import r23_word_char_tfidf as scoring
from experiments.bill_target74 import r36_listaware_semidetached_text as r36
from experiments.bill_target74 import runner as campaign
from experiments.bill_greedy import runner as greedy
from experiments.model_optimization_v2.pipeline import blend_prices


CARD = "R110_listaware_attached_shared_text_expert"
TRACK = "list_aware"
HOUSE_TYPES = ("Semi-Detached", "Freehold Townhouse", "Condo Townhouse")
SCREEN_MONTHS = ("2023-03", "2024-06")
SPECIALIST_WEIGHT = 0.75
COMPONENT = "attached_shared_core_expert"
MIN_TRAINING_ROWS = 500
MIN_EVALUATION_ROWS = 100
R35_RUN_ID = "bill_target74_r35_house_type_text_20261025T035638Z"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
                    encoding="utf-8")


def _mask(frame: pd.DataFrame) -> np.ndarray:
    if "house_type_name" not in frame.columns:
        raise RuntimeError("R110 route requires house_type_name")
    return frame.house_type_name.astype("string").isin(HOUSE_TYPES).to_numpy(bool)


def _preflight() -> tuple[dict, dict, dict]:
    parent = r36._verified_r35_parent(campaign)
    if parent["screen"]["run_id"] != R35_RUN_ID:
        raise RuntimeError("R110 did not resolve the registered R35 parent")
    source = identity._source_identity()
    if source.get("data_sha256") != campaign.PROTOCOL["data_sha256"]:
        raise RuntimeError("R110 data identity differs from the active campaign")
    from data import hide_answers

    route_plan = {}
    for month in SCREEN_MONTHS:
        _, train, _, truth, heldout = greedy._load_fold(month)
        cutoff = pd.Period(month, freq="M").start_time
        if (pd.to_datetime(train.date_sold, errors="raise") >= cutoff).any():
            raise RuntimeError(f"R110 training rows reach the {month} cutoff")
        homes = hide_answers(heldout, TRACK).reset_index(drop=True)
        train_mask, eval_mask = _mask(train), _mask(homes)
        if int(train_mask.sum()) < MIN_TRAINING_ROWS or int(eval_mask.sum()) < MIN_EVALUATION_ROWS:
            raise RuntimeError(f"R110 combined attached route is too small for {month}")
        parent_pairs = scoring.r22_reference_frame(parent["screen_rows"], truth, month, TRACK)
        if (not np.array_equal(parent_pairs.ml_num.astype(str).to_numpy(), homes.ml_num.astype(str).to_numpy())
                or parent_pairs.ml_num.astype(str).duplicated().any()
                or truth.ml_num.astype(str).duplicated().any()):
            raise RuntimeError(f"R110 R35 reference and route rows are misaligned for {month}")
        route_plan[month] = {
            "training_rows": int(train_mask.sum()),
            "evaluation_rows": int(eval_mask.sum()),
            "evaluation_rows_by_type": {
                kind: int(homes.house_type_name.astype("string").eq(kind).sum())
                for kind in HOUSE_TYPES
            },
            "cutoff_exclusive": cutoff.isoformat(),
        }
    return parent, source, route_plan


def _registration_spec(source: dict, route_plan: dict) -> dict:
    return {
        "status": "registered",
        "method": (
            "Train one G01 list-aware LightGBM core using only the combined Semi-Detached, "
            "Freehold Townhouse, and Condo Townhouse training rows, with R22 fold-local remarks "
            "TF-IDF fit on that same route. Blend 75% shared expert with 25% verified frozen HGBR "
            "on those three evaluation types; retain R35 predictions everywhere else."
        ),
        "difference_from_prior_work": (
            "R35 routes only Detached homes to its expert; R36 trained a Semi-Detached-only "
            "expert and R43 a Condo Apt-only expert, both of which failed. R110 tests shared "
            "learning across the three attached townhouse/semi-detached types with a single fixed model."
        ),
        "track": TRACK,
        "parent": "validated R35 list-aware predictions",
        "parent_run_id": R35_RUN_ID,
        "screen_months": list(SCREEN_MONTHS),
        "route_house_types": list(HOUSE_TYPES),
        "route_preflight": route_plan,
        "expected_fit_starts": len(SCREEN_MONTHS),
        "training_policy": "Each fit uses source rows with date_sold strictly before the screen month; all route members share one training population.",
        "features": "R35/G01 feature and target pipeline; R22 fold-local word TF-IDF fit only on routed training rows.",
        "blend": {"expert_weight": SPECIALIST_WEIGHT, "frozen_hgbr_weight": 1.0 - SPECIALIST_WEIGHT,
                  "rounding": "same blend_prices price-space rounding as R35"},
        "screen_gate": {
            "pooled_relative_mdape_improvement_percent_min": campaign.PROTOCOL["gates"]["screen_relative_mdape_improvement_percent"],
            "both_months_must_improve": campaign.PROTOCOL["gates"]["screen_both_months_improve"],
            "mape_regression_max_percent": campaign.PROTOCOL["gates"]["screen_mape_regression_max_percent"],
        },
        "data_sha256": source["data_sha256"], "parent_commit": source["commit"],
        "source_sha256": _sha256(Path(__file__)),
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    }


def _register(source: dict, route_plan: dict) -> dict:
    card = _registration_spec(source, route_plan)
    existing = campaign.PROTOCOL["cards"].get(CARD)
    if existing:
        if existing.get("source_sha256") != card["source_sha256"]:
            raise RuntimeError("R110 already exists with a different source snapshot")
        if existing.get("status") != "registered":
            raise RuntimeError(f"R110 already has status {existing.get('status')!r}")
        return existing
    campaign.PROTOCOL["cards"][CARD] = card
    _write_json(campaign.PROTOCOL_PATH, campaign.PROTOCOL)
    campaign._append({
        "event": "card_registered", "card": CARD, "fit_starts": 0,
        "source_sha256": card["source_sha256"], "data_sha256": source["data_sha256"],
        "parent_run_id": R35_RUN_ID, "route_house_types": list(HOUSE_TYPES),
        "screen_months": list(SCREEN_MONTHS), "expected_fit_starts": len(SCREEN_MONTHS),
        "route_preflight": route_plan,
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    campaign._write_progress(
        "r110_registered", next_action="Run the registered R110 shared attached-home expert screen",
        r110_card=CARD,
    )
    return card


def _run_screen(parent: dict, source: dict, route_plan: dict, card: dict) -> dict:
    from data import hide_answers

    run_id = "r110m_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = campaign.CAMPAIGN_ROOT / "batches" / run_id
    output.mkdir(parents=True, exist_ok=False)
    (output / "fold_predictions").mkdir()
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
        (parent["fixed_manifest_path"], "r35_fixed_origin_run_manifest.json"),
    ]:
        shutil.copy2(source_path, provenance / name)
    source_paths = [
        campaign.PROTOCOL_PATH, Path(__file__),
        campaign.PACKAGE / "r36_listaware_semidetached_text.py",
        campaign.PACKAGE / "r35_fixed_origin.py",
        campaign.PACKAGE / "r22_remarks_tfidf.py",
        campaign.PACKAGE / "r23_word_char_tfidf.py",
        campaign.PACKAGE / "r03_catboost.py",
        campaign.PACKAGE / "r04_ablation.py",
        campaign.PACKAGE / "runner.py",
        campaign.ROOT / "experiments/bill_greedy/runner.py",
        campaign.ROOT / "experiments/bill_greedy/history.py",
        campaign.ROOT / "experiments/model_optimization_v2/pipeline.py",
    ]
    source_hashes, source_snapshots = {}, {}
    for index, path in enumerate(source_paths):
        key = str(path.relative_to(campaign.ROOT)).replace("\\", "/")
        source_hashes[key] = _sha256(path)
        snap_name = f"src_{index:02d}{path.suffix}.source"
        shutil.copy2(path, provenance / snap_name)
        source_snapshots[key] = snap_name

    campaign._append({
        "event": "card_start", "card": CARD, "run_id": run_id,
        "stage": "screen", "track": TRACK, "parent_run_id": R35_RUN_ID,
        "route_house_types": list(HOUSE_TYPES), "screen_months": list(SCREEN_MONTHS),
        "expected_fit_starts": len(SCREEN_MONTHS), "expert_weight": SPECIALIST_WEIGHT,
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    old_fit_record = greedy._fit_record
    fit_context, fit_details, paired_parts, route_details = {}, [], [], {}

    def record_fit(_original_id, fit_fn):
        month = str(fit_context["month"])
        fit_id = f"{campaign.PROTOCOL['campaign_id']}:{CARD}:{TRACK}:{month}:{COMPONENT}:{run_id}"
        campaign._append({
            "event": "fit_start", "card": CARD, "run_id": run_id,
            "fit_id": fit_id, "track": TRACK, "month": month,
            "component": COMPONENT, "route_house_types": list(HOUSE_TYPES),
            "training_rows": fit_context["training_rows"],
            "evaluation_rows": fit_context["evaluation_rows"],
            "data_sha256": source["data_sha256"],
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        started = time.perf_counter()
        try:
            fitted = fit_fn()
        except BaseException as exc:
            campaign._append({
                "event": "fit_complete", "card": CARD, "run_id": run_id,
                "fit_id": fit_id, "track": TRACK, "month": month,
                "component": COMPONENT,
                "status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                "seconds": time.perf_counter() - started,
                "error": f"{type(exc).__name__}: {exc}",
                "reserved_months_labels_opened": False, "hidden_labels_opened": False,
            })
            raise
        fit_path = output / "fold_predictions" / f"{month}_{COMPONENT}.csv"
        pd.DataFrame({"ml_num": fit_context["evaluation_ids"],
                      "prediction": np.asarray(fitted["predictions"], dtype=float)}).to_csv(fit_path, index=False)
        detail = {
            "fit_id": fit_id, "track": TRACK, "month": month, "component": COMPONENT,
            "seconds": time.perf_counter() - started,
            "training_rows": fitted["training_rows"], "evaluation_rows": fitted["evaluation_rows"],
            "model_family": fitted["model_family"], "model_config": fitted["model_config"],
            "target": fitted["target"], "feature_columns": fitted["feature_columns"],
            "text_feature_count": sum(str(name).startswith("remarks_tfidf_") for name in fitted["feature_columns"]),
            "prediction_sha256": _sha256(fit_path),
        }
        fit_details.append(detail)
        campaign._append({
            "event": "fit_complete", "card": CARD, "run_id": run_id,
            **detail, "status": "complete",
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        return fitted

    greedy._fit_record = record_fit
    try:
        for month in SCREEN_MONTHS:
            _, train, _, truth, heldout = greedy._load_fold(month)
            cutoff = pd.Period(month, freq="M").start_time
            if (pd.to_datetime(train.date_sold, errors="raise") >= cutoff).any():
                raise RuntimeError(f"R110 training rows reach the {month} cutoff")
            homes = hide_answers(heldout, TRACK).reset_index(drop=True)
            train_mask, route_mask = _mask(train), _mask(homes)
            fit_rows, eval_rows = train.loc[train_mask].copy(), homes.loc[route_mask].copy()
            if len(fit_rows) < MIN_TRAINING_ROWS or int(route_mask.sum()) < MIN_EVALUATION_ROWS:
                raise RuntimeError(f"R110 route is too small for {month}")
            reference = scoring.r22_reference_frame(parent["screen_rows"], truth, month, TRACK)
            parent_values = reference.predicted_price.to_numpy(float)
            if (not np.array_equal(reference.ml_num.astype(str).to_numpy(), homes.ml_num.astype(str).to_numpy())
                    or not np.isfinite(parent_values).all() or (parent_values <= 0).any()):
                raise RuntimeError(f"R110 parent rows are misaligned or invalid for {month}")

            transformer, restore = r22._install_text_features(fit_rows, eval_rows, TRACK)
            fit_context.update(month=month, training_rows=len(fit_rows),
                               evaluation_rows=len(eval_rows),
                               evaluation_ids=eval_rows.ml_num.astype(str).tolist())
            try:
                fitted = greedy._fit_history_candidate(
                    month, TRACK, train, eval_rows, fit_rows=fit_rows,
                    variant=r22.VARIANT, fit_tag=f"R110-{month}-shared-attached",
                )
            finally:
                restore()
            all_hgbr = r03._verified_hgbr_cache(greedy, month, TRACK, train, homes)
            specialist = blend_prices(
                fitted["predictions"], all_hgbr[route_mask], challenger_weight=SPECIALIST_WEIGHT,
            )
            candidate_values = parent_values.copy()
            candidate_values[route_mask] = specialist
            if not np.array_equal(candidate_values[~route_mask], parent_values[~route_mask]):
                raise RuntimeError(f"R110 changed predictions outside the shared expert route in {month}")
            paired_parts.append(pd.DataFrame({
                "month": month, "track": TRACK,
                "ml_num": homes.ml_num.astype(str).to_numpy(),
                "house_type_name": homes.house_type_name.astype("string").fillna("__MISSING__").to_numpy(),
                "sold_price": truth.sold_price.to_numpy(float),
                "parent_prediction": parent_values,
                "candidate_prediction": candidate_values,
            }))
            route_truth = truth.sold_price.to_numpy(float)[route_mask]
            route_details[month] = {
                "training_rows": int(len(fit_rows)), "evaluation_rows": int(route_mask.sum()),
                "training_rows_by_type": {
                    kind: int(train.house_type_name.astype("string").eq(kind).sum())
                    for kind in HOUSE_TYPES
                },
                "parent": campaign._metrics(route_truth, parent_values[route_mask]),
                "candidate": campaign._metrics(route_truth, candidate_values[route_mask]),
                "non_routed_predictions_unchanged": True,
                "tfidf_feature_count": len(transformer.columns_ or []),
                "training_cutoff_exclusive": cutoff.isoformat(),
            }
            campaign._append({
                "event": "screen_fold_complete", "card": CARD, "run_id": run_id,
                "track": TRACK, "month": month, "routed_rows": int(route_mask.sum()),
                "routed_training_rows": len(fit_rows),
                "training_cutoff_exclusive": cutoff.isoformat(),
                "reserved_months_labels_opened": False, "hidden_labels_opened": False,
            })
            pd.concat(paired_parts, ignore_index=True).to_csv(
                output / "partial_paired_predictions.csv", index=False)
    except BaseException as exc:
        greedy._fit_record = old_fit_record
        failure = {
            "run_id": run_id, "card": CARD, "stage": "screen",
            "error": f"{type(exc).__name__}: {exc}", "fit_starts": len(fit_details),
            "fit_details": fit_details, "route_details": route_details,
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        }
        _write_json(output / "failure_checkpoint.json", failure)
        campaign._append({
            "event": "card_failed", "card": CARD, "run_id": run_id,
            "fit_starts": len(fit_details), "error": failure["error"],
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        campaign.PROTOCOL["cards"][CARD].update({
            "status": "screen_failed_runtime", "run_id": run_id,
            "failure_path": str((output / "failure_checkpoint.json").relative_to(campaign.ROOT)).replace("\\", "/"),
            "fit_starts": len(fit_details),
        })
        _write_json(campaign.PROTOCOL_PATH, campaign.PROTOCOL)
        campaign._write_progress(
            "r110_screen_failed_runtime", next_action="Inspect R110 failure checkpoint; retain all fit records",
            r110_run_id=run_id, r110_fit_starts=len(fit_details),
        )
        raise
    finally:
        greedy._fit_record = old_fit_record

    if len(fit_details) != len(SCREEN_MONTHS):
        raise RuntimeError(f"R110 expected {len(SCREEN_MONTHS)} fits; found {len(fit_details)}")
    paired = pd.concat(paired_parts, ignore_index=True)
    paired_path = output / "paired_predictions.csv"
    paired.to_csv(paired_path, index=False)
    metrics = scoring._track_result(paired)
    passed = bool(metrics["qualifies_screen"])
    result = {
        "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
        "card": CARD, "track": TRACK, "parent_run_id": R35_RUN_ID,
        "screen_months": list(SCREEN_MONTHS), "fit_starts": len(fit_details),
        "specialist": {"house_types": list(HOUSE_TYPES), "weight": SPECIALIST_WEIGHT,
                       "shared_training_population": True,
                       "tfidf_fit_only_on_routed_training_rows": True},
        "tracks": {TRACK: metrics}, "routed_diagnostics": route_details,
        "fit_details": fit_details, "data_sha256": source["data_sha256"],
        "reserved_months_labels_opened": False,
        "reserve_validation_eligible": False, "hidden_labels_opened": False,
    }
    result_path = output / "screen_results.json"
    _write_json(result_path, result)
    manifest = {
        "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
        "source_identity": source, "code_sha256": source_hashes,
        "source_snapshots": source_snapshots,
        "parent_run_id": R35_RUN_ID,
        "parent_screen_result_sha256": _sha256(parent["screen_path"]),
        "parent_screen_predictions_sha256": _sha256(parent["screen_rows_path"]),
        "parent_screen_manifest_sha256": _sha256(parent["screen_manifest_path"]),
        "parent_development_result_sha256": _sha256(parent["development_path"]),
        "parent_fixed_origin_result_sha256": _sha256(parent["fixed_path"]),
        "route_house_types": list(HOUSE_TYPES), "screen_months": list(SCREEN_MONTHS),
        "fit_starts": len(fit_details), "fit_details": fit_details,
        "paired_predictions_sha256": _sha256(paired_path),
        "result_sha256": _sha256(result_path),
        "reserved_months_labels_opened": False,
        "reserve_validation_eligible": False, "hidden_labels_opened": False,
    }
    _write_json(output / "run_manifest.json", manifest)
    report = [
        "# R110：list-aware 共享 attached-home R22-text 专家筛选", "",
        f"Run `{run_id}`；与已验证 R35 `{R35_RUN_ID}` 在相同筛选房源上配对比较。", "",
        f"共享专家路由：{', '.join(HOUSE_TYPES)}；模型固定为 G01 LightGBM + R22 fold-local word TF-IDF，"
        "并按 75% 专家 / 25% 冻结 HGBR 融合。", "",
        f"R35 pooled MdAPE `{metrics['parent']['mdape_percent']:.6f}%`；候选 `{metrics['candidate']['mdape_percent']:.6f}%`；"
        f"相对改善 `{metrics['relative_mdape_improvement_percent']:+.3f}%`；MAPE 相对变化 `{metrics['mape_relative_change_percent']:+.3f}%`。", "",
        f"筛选门槛通过：`{passed}`。", "",
        "只使用安全开发月份并按月初严格截止训练；未读取 2025-05/08 保留月份或隐藏标签。", "",
        f"配对结果、源代码快照和 manifest 位于 `{output.relative_to(campaign.ROOT)}`。", "",
    ]
    (output / "research_report.md").write_text("\n".join(report), encoding="utf-8")
    status = "screen_passed" if passed else "screen_failed"
    campaign._append({
        "event": "card_complete", "card": CARD, "run_id": run_id,
        "fit_starts": len(fit_details), "status": status,
        "result_path": str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        "qualifies_by_track": {TRACK: passed},
        "reserved_months_labels_opened": False,
        "reserve_validation_eligible": False, "hidden_labels_opened": False,
    })
    campaign.PROTOCOL["cards"][CARD].update({
        "status": status, "run_id": run_id, "fit_starts": len(fit_details),
        "result_path": str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        "paired_predictions_path": str(paired_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        "qualifies_screen": passed,
        "screen_metrics": {
            "parent_mdape_percent": metrics["parent"]["mdape_percent"],
            "candidate_mdape_percent": metrics["candidate"]["mdape_percent"],
            "relative_mdape_improvement_percent": metrics["relative_mdape_improvement_percent"],
            "mape_relative_change_percent": metrics["mape_relative_change_percent"],
            "monthly_relative_mdape_improvement_percent": {
                m: item["relative_mdape_improvement_percent"]
                for m, item in metrics["monthly"].items()
            },
        },
    })
    _write_json(campaign.PROTOCOL_PATH, campaign.PROTOCOL)
    campaign._write_progress(
        "r110_screen_complete",
        next_action=("R110 passed screen; register eight-month list-aware development confirmation against R35"
                     if passed else "R110 failed screen; retain R35 and register a distinct list-aware hypothesis"),
        r110_run_id=run_id, r110_result=str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        r110_fit_starts=len(fit_details),
    )
    return result


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in {"--preflight", "--run"}:
        raise SystemExit("usage: python -m experiments.bill_target74.r110_listaware_attached_shared_text_expert --preflight|--run")
    parent, source, route_plan = _preflight()
    if sys.argv[1] == "--preflight":
        print(json.dumps({"card": CARD, "parent_run_id": parent["screen"]["run_id"],
                          "route_house_types": list(HOUSE_TYPES), "screen_months": list(SCREEN_MONTHS),
                          "route_plan": route_plan, "reserved_or_hidden_opened": False},
                         indent=2, ensure_ascii=False))
        return
    card = _register(source, route_plan)
    result = _run_screen(parent, source, route_plan, card)
    metrics = result["tracks"][TRACK]
    print(json.dumps({"run_id": result["run_id"], "fit_starts": result["fit_starts"],
                      "parent_mdape_percent": metrics["parent"]["mdape_percent"],
                      "candidate_mdape_percent": metrics["candidate"]["mdape_percent"],
                      "relative_mdape_improvement_percent": metrics["relative_mdape_improvement_percent"],
                      "mape_relative_change_percent": metrics["mape_relative_change_percent"],
                      "qualifies_screen": metrics["qualifies_screen"],
                      "reserved_months_labels_opened": False,
                      "hidden_labels_opened": False}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
