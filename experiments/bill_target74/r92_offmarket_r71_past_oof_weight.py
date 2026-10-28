"""R92: choose the R71 off-market expert share from strictly past OOF rows."""
from __future__ import annotations

import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.bill_target74.r19_expert_weight import _sha256, _write_json


CARD = "R92_offmarket_r71_past_oof_expert_weight"
TRACK = "off_market"
HOUSE_TYPES = ("Semi-Detached", "Freehold Townhouse")
COMPONENT = "freehold_attached_base_expert"
SCREEN_MONTHS = ("2023-03", "2024-06")
OOF_MONTHS = ("2023-03", "2023-07", "2023-11", "2024-03")
FIT_MONTHS = (*OOF_MONTHS, "2024-06")
WEIGHT_GRID = tuple(round(i * 0.05, 2) for i in range(21))
PARENT_EXPERT_WEIGHT = 0.75
MINIMUM_ROUTED_OOF_ROWS = 1000
MAX_MAPE_REGRESSION_PERCENT = 1.0
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20261026


def _route_mask(frame: pd.DataFrame) -> np.ndarray:
    if "house_type_name" not in frame:
        raise ValueError("R92 routing requires house_type_name")
    return frame.house_type_name.astype("string").isin(HOUSE_TYPES).fillna(False).to_numpy(bool)


def _source_files(campaign, r71, parent) -> list[Path]:
    r71_card = campaign.PROTOCOL["cards"].get(r71.CARD)
    if not r71_card or not r71_card.get("result_path"):
        raise RuntimeError("R92 cannot snapshot the registered R71 screen artifacts")
    r71_result_path = campaign.ROOT / r71_card["result_path"]
    r71_rows_path = r71_result_path.parent / "paired_predictions.csv"
    r71_manifest_path = r71_result_path.parent / "run_manifest.json"
    return [
        campaign.PROTOCOL_PATH,
        Path(__file__),
        campaign.PACKAGE / "r71_offmarket_freehold_attached_text_expert.py",
        campaign.PACKAGE / "r52_r48_r50_adaptive_meta_blend.py",
        campaign.PACKAGE / "r48_groupwise_past_oof_r22_r32_blend.py",
        campaign.PACKAGE / "r50_type_price_interaction_past_oof_blend.py",
        campaign.PACKAGE / "r22_remarks_tfidf.py",
        campaign.PACKAGE / "r04_ablation.py",
        campaign.PACKAGE / "runner.py",
        campaign.ROOT / "experiments/bill_greedy/runner.py",
        campaign.ROOT / "experiments/bill_greedy/history.py",
        campaign.ROOT / "model_candidate_v2/features.py",
        campaign.ROOT / "experiments/model_optimization_v2/learners.py",
        campaign.ROOT / "experiments/model_optimization_v2/pipeline.py",
        campaign.ROOT / "experiments/model_optimization_v2/folds.py",
        parent["result_path"], parent["rows_path"], parent["manifest_path"],
        r71_result_path, r71_rows_path, r71_manifest_path,
    ]


def _verified_parent(campaign, r71_module) -> dict:
    parent = r71_module._verified_r52_parent(campaign)
    result = json.loads(parent["result_path"].read_text(encoding="utf-8"))
    rows = pd.read_csv(parent["rows_path"], dtype={"ml_num": str})
    dev_months = set(campaign.PROTOCOL["development_months"])
    rows = rows.loc[rows.track.eq(TRACK) & rows.month.astype(str).isin(dev_months)].copy()
    if rows.duplicated(["month", "ml_num"]).any():
        raise RuntimeError("R52 has duplicate IDs within a development month")
    expected = result["tracks"][TRACK]["monthly"]
    counts = rows.groupby(rows.month.astype(str)).size().to_dict()
    if counts != {m: int(expected[m]["n"]) for m in dev_months}:
        raise RuntimeError("R52 development predictions do not cover every registered month")
    values = rows[["sold_price", "candidate_prediction"]].to_numpy(float)
    if not np.isfinite(values).all() or (values <= 0.0).any():
        raise RuntimeError("R52 development predictions contain invalid values")
    parent["development_rows"] = rows
    return parent


def _fold_support(greedy, month: str) -> dict[str, int]:
    _, train, _, _, heldout = greedy._load_fold(month)
    cutoff = pd.Period(month, freq="M").start_time
    dates = pd.to_datetime(train.date_sold, errors="raise")
    if dates.empty or (dates >= cutoff).any():
        raise RuntimeError(f"R92 training fold is not strictly before {month}")
    return {"training_rows": int(_route_mask(train).sum()),
            "evaluation_rows": int(_route_mask(heldout).sum())}


def register_card() -> dict:
    from experiments.bill_target74 import r04_ablation as budget
    from experiments.bill_target74 import r71_offmarket_freehold_attached_text_expert as r71
    from experiments.bill_target74 import runner as campaign

    existing = campaign.PROTOCOL["cards"].get(CARD)
    if existing:
        if existing.get("status") == "registered" and existing.get("source_sha256") == _sha256(Path(__file__)):
            return existing
        raise RuntimeError(f"R92 already exists with status {existing.get('status')}; inspect before retry")
    r71_card = campaign.PROTOCOL["cards"].get(r71.CARD)
    if not r71_card or r71_card.get("status") != "screen_failed":
        raise RuntimeError("R92 requires the completed, failed R71 screen")
    r71_result_path = campaign.ROOT / r71_card["result_path"]
    r71_rows_path = r71_result_path.parent / "paired_predictions.csv"
    r71_manifest_path = r71_result_path.parent / "run_manifest.json"
    if not all(p.is_file() for p in (r71_result_path, r71_rows_path, r71_manifest_path)):
        raise FileNotFoundError("R71 result, paired rows, or manifest is missing")
    r71_result = json.loads(r71_result_path.read_text(encoding="utf-8"))
    r71_manifest = json.loads(r71_manifest_path.read_text(encoding="utf-8"))
    if (r71_result.get("run_id") != r71_card.get("run_id")
            or r71_manifest.get("result_sha256") != _sha256(r71_result_path)
            or r71_manifest.get("paired_predictions_sha256") != _sha256(r71_rows_path)
            or r71_result.get("reserved_months_labels_opened") is not False
            or r71_result.get("hidden_labels_opened") is not False):
        raise RuntimeError("R71 artifacts do not pass identity, hash, or label-boundary checks")
    parent = _verified_parent(campaign, r71)
    source = budget._source_identity()
    from experiments.bill_greedy import runner as greedy

    route_support = {month: _fold_support(greedy, month) for month in FIT_MONTHS}
    expected_support = {
        "2023-03": {"training_rows": 9190, "evaluation_rows": 282},
        "2023-07": {"training_rows": 10588, "evaluation_rows": 246},
        "2023-11": {"training_rows": 11510, "evaluation_rows": 218},
        "2024-03": {"training_rows": 12278, "evaluation_rows": 265},
        "2024-06": {"training_rows": 13248, "evaluation_rows": 289},
    }
    if route_support != expected_support:
        raise RuntimeError(f"R92 route support changed: {route_support}")
    past_route_rows = sum(route_support[m]["evaluation_rows"] for m in OOF_MONTHS)
    if past_route_rows < MINIMUM_ROUTED_OOF_ROWS:
        raise RuntimeError(f"R92 past route OOF has only {past_route_rows} rows")

    registration = {
        "status": "registered", "track": TRACK, "parent": "R52_r48_r50_strict_past_meta_blend",
        "parent_run_id": parent["result"]["run_id"], "source_r71_run_id": r71_result["run_id"],
        "source_r71_result_path": str(r71_result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        "screen_months": list(SCREEN_MONTHS), "past_oof_months": list(OOF_MONTHS),
        "fit_months": list(FIT_MONTHS), "expected_fit_starts": len(FIT_MONTHS),
        "route_house_types": list(HOUSE_TYPES), "route_support": route_support,
        "past_oof_routed_rows_for_2024_06": past_route_rows,
        "minimum_routed_oof_rows": MINIMUM_ROUTED_OOF_ROWS,
        "weight_grid": list(WEIGHT_GRID), "fallback_expert_weight": PARENT_EXPERT_WEIGHT,
        "model_family": "lightgbm", "model_config_overrides": {},
        "target": "unchanged R71 off-market target and feature pipeline",
        "method": (
            "Keep the R52 off-market predictions outside the combined Semi-Detached/Freehold Townhouse route. "
            "Re-fit the exact R71 G06 LightGBM text expert on five strict time folds. At each screen cutoff, "
            "select the expert share on the 0.00..1.00 grid by 0.05 using only earlier registered OOF folds; "
            "minimize whole-track pooled MdAPE subject to no more than 1% whole-track MAPE regression. "
            "If fewer than 1,000 routed OOF homes exist, fall back to the R71 0.75 share. Ties prefer lower "
            "MAPE and then the share closest to 0.75. Apply the selected share only to the routed houses."
        ),
        "difference_from_prior_work": (
            "R71 used a fixed 0.75 specialist share, improved both fixed screen months but missed the pooled "
            "0.25% gain and bootstrap gates. R92 changes only the share, selecting it from earlier OOF months. "
            "For 2023-03 no earlier R71 expert OOF month is registered, so the predeclared fallback is 0.75. "
            "For 2024-06 the four earlier safe months contain 1,011 routed OOF homes."
        ),
        "incremental_screen_gate": r71_card["incremental_screen_gate"],
        "policy": (
            "Fit TF-IDF only on routed training remarks and train strictly before each fold. The 2024-06 "
            "weight may use only 2023-03, 2023-07, 2023-11, and 2024-03 OOF predictions and labels. "
            "The 2023-03 weight uses the registered 0.75 fallback. Preserve all R52 predictions outside the "
            "route; retain R52 unless all original R71 screen gates pass. Never read 2025-05/08 or hidden labels."
        ),
        "parent_result_sha256": _sha256(parent["result_path"]),
        "parent_predictions_sha256": _sha256(parent["rows_path"]),
        "source_r71_result_sha256": _sha256(r71_result_path),
        "source_r71_predictions_sha256": _sha256(r71_rows_path),
        "data_sha256": source["data_sha256"], "source_sha256": _sha256(Path(__file__)),
        "reserved_months_labels_opened": False, "reserve_validation_eligible": False,
        "hidden_labels_opened": False,
    }
    campaign.PROTOCOL["cards"][CARD] = registration
    campaign.PROTOCOL_PATH.write_text(
        json.dumps(campaign.PROTOCOL, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    campaign._append({
        "event": "card_registered", "card": CARD, "track": TRACK,
        "parent_run_id": parent["result"]["run_id"], "source_r71_run_id": r71_result["run_id"],
        "screen_months": list(SCREEN_MONTHS), "past_oof_months": list(OOF_MONTHS),
        "expected_fit_starts": len(FIT_MONTHS), "route_support": route_support,
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    campaign._write_progress(
        "r92_registered",
        next_action="run the five-fold R92 strict-past OOF-selected R71 expert-weight screen against R52",
        r92_card=CARD,
    )
    return registration


def _parent_month_rows(parent_rows: pd.DataFrame, month: str, truth: pd.DataFrame) -> pd.DataFrame:
    rows = parent_rows.loc[parent_rows.month.astype(str).eq(month)].copy()
    rows["ml_num"] = rows.ml_num.astype(str)
    if rows.ml_num.duplicated().any():
        raise RuntimeError(f"R52 has duplicate home IDs in {month}")
    rows = rows.set_index("ml_num").reindex(truth.ml_num.astype(str)).reset_index()
    if (not np.array_equal(rows.sold_price.to_numpy(float), truth.sold_price.to_numpy(float))
            or not np.isfinite(rows.candidate_prediction.to_numpy(float)).all()
            or (rows.candidate_prediction.to_numpy(float) <= 0).any()):
        raise RuntimeError(f"R52 parent rows do not align with safe fold {month}")
    return rows


def _select_weight(folds: list[dict], cutoff: str) -> dict:
    earlier = [row for row in folds if row["month"] in OOF_MONTHS
               and pd.Period(row["month"], freq="M") < pd.Period(cutoff, freq="M")]
    routed_rows = sum(int(row["route_mask"].sum()) for row in earlier)
    if routed_rows < MINIMUM_ROUTED_OOF_ROWS:
        return {
            "cutoff_month": cutoff, "past_oof_months": [row["month"] for row in earlier],
            "routed_oof_rows": routed_rows, "fallback": True,
            "fallback_reason": "fewer than 1,000 strictly past routed OOF rows",
            "selected_weight": PARENT_EXPERT_WEIGHT, "grid_results": [],
        }
    actual = np.concatenate([row["actual"] for row in earlier])
    parent = np.concatenate([row["parent"] for row in earlier])
    route = np.concatenate([row["route_mask"] for row in earlier])
    expert = np.concatenate([row["expert"] for row in earlier])
    base_metrics = None
    grid_results = []
    for weight in WEIGHT_GRID:
        candidate = parent.copy()
        candidate[route] = _blend_prices(expert, parent[route], weight)
        metrics = _metrics(actual, candidate)
        if base_metrics is None:
            base_metrics = _metrics(actual, parent)
        mape_change = 100.0 * (metrics["mape_percent"] - base_metrics["mape_percent"]) / base_metrics["mape_percent"]
        grid_results.append({
            "weight": weight, "mdape_percent": metrics["mdape_percent"],
            "mape_percent": metrics["mape_percent"],
            "mape_relative_change_percent": mape_change,
            "eligible": bool(mape_change <= MAX_MAPE_REGRESSION_PERCENT),
        })
    eligible = [row for row in grid_results if row["eligible"]]
    if not eligible:
        selected = PARENT_EXPERT_WEIGHT
        fallback = True
        fallback_reason = "no grid weight met the past-only MAPE guard"
    else:
        chosen = min(eligible, key=lambda row: (
            round(row["mdape_percent"], 12), round(row["mape_percent"], 12),
            abs(row["weight"] - PARENT_EXPERT_WEIGHT), row["weight"],
        ))
        selected = float(chosen["weight"])
        fallback = False
        fallback_reason = None
    return {
        "cutoff_month": cutoff, "past_oof_months": [row["month"] for row in earlier],
        "routed_oof_rows": routed_rows, "fallback": fallback,
        "fallback_reason": fallback_reason, "minimum_routed_oof_rows": MINIMUM_ROUTED_OOF_ROWS,
        "past_parent_mdape_percent": base_metrics["mdape_percent"],
        "past_parent_mape_percent": base_metrics["mape_percent"],
        "selected_weight": selected, "grid_results": grid_results,
    }


def _blend_prices(expert: np.ndarray, parent: np.ndarray, weight: float) -> np.ndarray:
    from experiments.model_optimization_v2.pipeline import blend_prices
    return blend_prices(expert, parent, challenger_weight=weight)


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    from experiments.bill_target74 import runner as campaign
    return campaign._metrics(actual, predicted)


def run_screen() -> dict:
    from data import hide_answers
    from experiments.bill_target74 import r04_ablation as budget
    from experiments.bill_target74 import r22_remarks_tfidf as r22
    from experiments.bill_target74 import runner as campaign
    from experiments.bill_target74 import r71_offmarket_freehold_attached_text_expert as r71
    from experiments.bill_greedy import runner as greedy

    card = campaign.PROTOCOL["cards"].get(CARD)
    if not card or card.get("status") != "registered":
        raise RuntimeError("R92 must be preregistered before fitting")
    if card.get("source_sha256") != _sha256(Path(__file__)):
        raise RuntimeError("R92 source changed after preregistration")
    source = budget._source_identity()
    if source["data_sha256"] != card["data_sha256"]:
        raise RuntimeError("R92 data snapshot changed after preregistration")
    parent = _verified_parent(campaign, r71)
    if parent["result"]["run_id"] != card["parent_run_id"]:
        raise RuntimeError("R92 R52 parent differs from registration")
    r71_result_path = campaign.ROOT / card["source_r71_result_path"]
    if _sha256(r71_result_path) != card["source_r71_result_sha256"]:
        raise RuntimeError("R92 R71 source screen changed after registration")

    run_id = "r92o_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = campaign.CAMPAIGN_ROOT / "batches" / run_id
    output.mkdir(parents=True, exist_ok=False)
    provenance = output / "provenance"
    provenance.mkdir()
    for source_path, name in [
        (parent["result_path"], "r52_development_results.json"),
        (parent["rows_path"], "r52_paired_predictions.csv"),
        (parent["manifest_path"], "r52_run_manifest.json"),
        (r71_result_path, "r71_screen_results.json"),
        (r71_result_path.parent / "paired_predictions.csv", "r71_screen_paired_predictions.csv"),
        (r71_result_path.parent / "run_manifest.json", "r71_screen_run_manifest.json"),
    ]:
        shutil.copy2(source_path, provenance / name)
    files = _source_files(campaign, r71, parent)
    code_hashes, snapshots = {}, {}
    for index, path in enumerate(files):
        rel = str(path.relative_to(campaign.ROOT)).replace("\\", "/")
        snap = f"src_{index:02d}{path.suffix}.source"
        code_hashes[rel] = _sha256(path)
        snapshots[rel] = snap
        shutil.copy2(path, provenance / snap)

    campaign._append({
        "event": "card_start", "card": CARD, "run_id": run_id, "track": TRACK,
        "parent_run_id": parent["result"]["run_id"], "screen_months": list(SCREEN_MONTHS),
        "past_oof_months": list(OOF_MONTHS), "expected_fit_starts": len(FIT_MONTHS),
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })

    old_fit_record = greedy._fit_record
    fit_context: dict[str, object] = {}
    fit_details: list[dict] = []
    fold_rows: list[dict] = []
    fit_attempts = 0

    def record_fit(_original_id: str, fit_fn):
        nonlocal fit_attempts
        fit_attempts += 1
        month = str(fit_context["month"])
        fit_id = f"{campaign.PROTOCOL['campaign_id']}:{CARD}:{TRACK}:{month}:{COMPONENT}:{run_id}"
        campaign._append({
            "event": "fit_start", "card": CARD, "run_id": run_id,
            "fit_id": fit_id, "track": TRACK, "month": month,
            "component": COMPONENT, "training_rows": fit_context["training_rows"],
            "evaluation_rows": fit_context["evaluation_rows"], "data_sha256": source["data_sha256"],
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        started = time.perf_counter()
        try:
            fitted = fit_fn()
            if fitted.get("target") != "log_sold_price":
                raise RuntimeError(f"R92 expected the off-market log_sold_price target; got {fitted.get('target')!r}")
        except Exception as exc:
            campaign._append({
                "event": "fit_complete", "card": CARD, "run_id": run_id,
                "fit_id": fit_id, "track": TRACK, "month": month, "component": COMPONENT,
                "status": "failed", "seconds": time.perf_counter() - started,
                "error": f"{type(exc).__name__}: {exc}",
                "reserved_months_labels_opened": False, "hidden_labels_opened": False,
            })
            raise
        component_path = output / "components" / f"expert_{month}.csv"
        component_path.parent.mkdir(exist_ok=True)
        pd.DataFrame({"ml_num": fit_context["evaluation_ids"],
                      "expert_prediction": fitted["predictions"]}).to_csv(component_path, index=False)
        detail = {
            "fit_id": fit_id, "track": TRACK, "month": month, "component": COMPONENT,
            "seconds": time.perf_counter() - started, "training_rows": fitted["training_rows"],
            "evaluation_rows": fitted["evaluation_rows"], "model_family": fitted["model_family"],
            "model_config": fitted["model_config"], "actual_objective": fitted["actual_objective"],
            "target": fitted["target"], "feature_columns": fitted["feature_columns"],
            "prediction_sha256": fitted["prediction_sha256"],
            "component_predictions_path": str(component_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        }
        fit_details.append(detail)
        campaign._append({
            "event": "fit_complete", "card": CARD, "run_id": run_id, **detail,
            "status": "complete", "reserved_months_labels_opened": False,
            "hidden_labels_opened": False,
        })
        return fitted

    greedy._fit_record = record_fit
    try:
        for month in FIT_MONTHS:
            _, train, _, truth, heldout = greedy._load_fold(month)
            cutoff = pd.Period(month, freq="M").start_time
            if (pd.to_datetime(train.date_sold, errors="raise") >= cutoff).any():
                raise RuntimeError(f"R92 training rows reach the {month} cutoff")
            homes = hide_answers(heldout, TRACK).reset_index(drop=True)
            route_train = _route_mask(train)
            route_eval = _route_mask(homes)
            fit_rows = train.loc[route_train].copy()
            eval_rows = homes.loc[route_eval].copy()
            support = {"training_rows": int(len(fit_rows)), "evaluation_rows": int(len(eval_rows))}
            if support != card["route_support"][month]:
                raise RuntimeError(f"R92 route support changed in {month}: {support}")
            parent_rows = _parent_month_rows(parent["development_rows"], month, truth)
            parent_prediction = parent_rows.candidate_prediction.to_numpy(float)
            transformer, restore = r22._install_text_features(fit_rows, eval_rows, TRACK)
            fit_context.update(
                month=month, training_rows=len(fit_rows), evaluation_rows=len(eval_rows),
                evaluation_ids=eval_rows.ml_num.astype(str).tolist(),
            )
            try:
                fitted = greedy._fit_history_candidate(
                    month, TRACK, train, eval_rows, fit_rows=fit_rows,
                    variant=r22.VARIANT, base_variant="G06_lot_garage",
                    fit_tag=f"R92-{month}-past-oof-weight",
                )
            finally:
                restore()
            expert = np.asarray(fitted["predictions"], dtype=float)
            if len(expert) != int(route_eval.sum()) or not np.isfinite(expert).all() or (expert <= 0).any():
                raise RuntimeError(f"R92 expert predictions fail route alignment in {month}")
            route_full = np.zeros(len(homes), dtype=bool)
            route_full[:] = route_eval
            fold_rows.append({
                "month": month, "ml_num": homes.ml_num.astype(str).to_numpy(),
                "actual": truth.sold_price.to_numpy(float), "parent": parent_prediction,
                "expert": expert, "route_mask": route_full,
                "training_rows": len(fit_rows), "routed_rows": int(route_eval.sum()),
                "feature_columns": fitted["feature_columns"],
                "text_feature_count": len(transformer.columns_ or []),
            })

        if len(fit_details) != len(FIT_MONTHS):
            raise RuntimeError(f"R92 expected {len(FIT_MONTHS)} fit completions, found {len(fit_details)}")

        selections = {month: _select_weight(fold_rows, month) for month in SCREEN_MONTHS}
        paired_parts, route_details = [], {}
        by_month = {row["month"]: row for row in fold_rows}
        for month in SCREEN_MONTHS:
            row = by_month[month]
            weight = selections[month]["selected_weight"]
            prediction = row["parent"].copy()
            prediction[row["route_mask"]] = _blend_prices(
                row["expert"], row["parent"][row["route_mask"]], weight,
            )
            if not np.array_equal(prediction[~row["route_mask"]], row["parent"][~row["route_mask"]]):
                raise RuntimeError(f"R92 changed R52 predictions outside the route in {month}")
            paired_parts.append(pd.DataFrame({
                "month": month, "track": TRACK, "ml_num": row["ml_num"],
                "sold_price": row["actual"], "parent_prediction": row["parent"],
                "candidate_prediction": prediction,
            }))
            route_details[month] = {
                "training_rows": row["training_rows"], "routed_rows": row["routed_rows"],
                "selected_expert_weight": weight, "weight_selection": selections[month],
                "parent_route": campaign._metrics(row["actual"][row["route_mask"]],
                                                   row["parent"][row["route_mask"]]),
                "candidate_route": campaign._metrics(row["actual"][row["route_mask"]],
                                                      prediction[row["route_mask"]]),
                "non_routed_predictions_unchanged": True,
            }
            campaign._append({
                "event": "screen_fold_complete", "card": CARD, "run_id": run_id,
                "track": TRACK, "month": month, "selected_expert_weight": weight,
                "past_oof_months": selections[month]["past_oof_months"],
                "routed_oof_rows": selections[month]["routed_oof_rows"],
                "non_routed_predictions_unchanged": True,
                "reserved_months_labels_opened": False, "hidden_labels_opened": False,
            })
    except Exception as exc:
        failure_path = output / "failure_checkpoint.json"
        _write_json(failure_path, {
            "run_id": run_id, "card": CARD, "error": f"{type(exc).__name__}: {exc}",
            "fit_starts": fit_attempts, "fit_details": fit_details,
            "completed_fit_months": [x["month"] for x in fold_rows],
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        campaign._append({
            "event": "card_failed", "card": CARD, "run_id": run_id,
            "fit_starts": fit_attempts, "error": f"{type(exc).__name__}: {exc}",
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        campaign.PROTOCOL["cards"][CARD].update({
            "status": "failed", "run_id": run_id, "fit_starts": fit_attempts,
            "failure_path": str(failure_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        })
        campaign.PROTOCOL_PATH.write_text(
            json.dumps(campaign.PROTOCOL, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        campaign._write_progress("r92_screen_failed_runtime", next_action="inspect R92 failure checkpoint before any retry",
                                 r92_failure=str(failure_path.relative_to(campaign.ROOT)).replace("\\", "/"))
        raise
    finally:
        greedy._fit_record = old_fit_record

    paired = pd.concat(paired_parts, ignore_index=True)
    paired_path = output / "paired_predictions.csv"
    paired.to_csv(paired_path, index=False)
    score = r22.score_development_confirmation(
        paired, months=SCREEN_MONTHS, bootstrap_replicates=BOOTSTRAP_REPLICATES,
        bootstrap_seed=BOOTSTRAP_SEED, minimum_relative_gain_percent=0.25,
        minimum_months_improved=2, maximum_worst_month_regression_percent=0.0,
        maximum_mape_regression_percent=1.0,
    )
    score["qualifies_screen"] = score.pop("qualifies_development")
    result = {
        "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id, "card": CARD,
        "track": TRACK, "parent_candidate": parent["result"]["run_id"],
        "screen_months": list(SCREEN_MONTHS), "past_oof_months": list(OOF_MONTHS),
        "fit_months": list(FIT_MONTHS), "fit_starts": len(fit_details),
        "tracks": {TRACK: score}, "route_diagnostics": route_details,
        "fit_details": fit_details, "data_sha256": source["data_sha256"],
        "reserved_months_labels_opened": False, "reserve_validation_eligible": False,
        "hidden_labels_opened": False,
    }
    result_path = output / "screen_results.json"
    _write_json(result_path, result)
    manifest = {
        "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
        "source_identity": source, "code_sha256": code_hashes, "source_snapshots": snapshots,
        "data_sha256": source["data_sha256"], "fit_starts": len(fit_details),
        "fit_details": fit_details, "result_sha256": _sha256(result_path),
        "paired_predictions_sha256": _sha256(paired_path),
        "parent_run_id": parent["result"]["run_id"],
        "parent_result_sha256": _sha256(parent["result_path"]),
        "parent_predictions_sha256": _sha256(parent["rows_path"]),
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    }
    _write_json(output / "run_manifest.json", manifest)
    score_track = score
    report = [
        "# R92：Off-market R71 专家严格过去 OOF 权重筛选", "",
        f"Run：`{run_id}`；父候选：R52 `{parent['result']['run_id']}`。",
        f"仅改变 R71 专家分享；房型路由、LightGBM、R22 TF-IDF、训练 cutoff 和特征保持不变。权重按 0.05 网格从严格过去 OOF 选择，MAPE 退化上限 1%；不足 1,000 个路由 OOF 样本时回退 0.75。", "",
        f"Pooled MdAPE：{score_track['parent']['mdape_percent']:.5f}%→{score_track['candidate']['mdape_percent']:.5f}%（相对改善 {score_track['relative_mdape_improvement_percent']:+.3f}%）。",
        f"MAPE 相对变化：{score_track['mape_relative_change_percent']:+.3f}%；筛选通过：{score_track['qualifies_screen']}。", "",
        "## 每折权重选择", "", "| 目标月 | OOF 月份 | 路由 OOF 行 | 回退 | 专家权重 |", "|---|---|---:|---|---:|",
    ]
    for month, details in selections.items():
        report.append(f"| {month} | {', '.join(details['past_oof_months']) or '无'} | "
                      f"{details['routed_oof_rows']} | {details['fallback']} | {details['selected_weight']:.2f} |")
    report.extend(["", "## 逐月结果", "", "| 月份 | R52 MdAPE | R92 MdAPE | 相对改善 |",
                   "|---|---:|---:|---:|"])
    for month, details in score_track["monthly"].items():
        report.append(f"| {month} | {details['parent']['mdape_percent']:.5f}% | "
                      f"{details['candidate']['mdape_percent']:.5f}% | "
                      f"{details['relative_mdape_improvement_percent']:+.3f}% |")
    report.extend(["", "预留月份与隐藏标签未读取；失败筛选不得在相同两个月上继续调权重。", ""])
    (output / "research_report.md").write_text("\n".join(report), encoding="utf-8")

    campaign.PROTOCOL["cards"][CARD].update({
        "status": "screen_passed" if score_track["qualifies_screen"] else "screen_failed",
        "run_id": run_id, "fit_starts": len(fit_details),
        "result_path": str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        "paired_predictions_path": str(paired_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        "manifest_path": str((output / "run_manifest.json").relative_to(campaign.ROOT)).replace("\\", "/"),
        "screen_metrics": score_track, "selected_weights": {
            month: details["selected_weight"] for month, details in selections.items()},
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    campaign.PROTOCOL_PATH.write_text(
        json.dumps(campaign.PROTOCOL, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    campaign._append({
        "event": "card_complete", "card": CARD, "run_id": run_id,
        "fit_starts": len(fit_details), "result_path": str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        "qualifies_screen": score_track["qualifies_screen"],
        "selected_weights": {month: details["selected_weight"] for month, details in selections.items()},
        "screen_metrics": score_track, "reserved_months_labels_opened": False,
        "hidden_labels_opened": False,
    })
    next_action = (
        "R92 passed; run the registered eight-month off-market development confirmation against R52"
        if score_track["qualifies_screen"] else
        "R92 failed; keep R22 as validated parent and R52 as best experiment; register a distinct hypothesis"
    )
    campaign._write_progress(
        "r92_screen_passed" if score_track["qualifies_screen"] else "r92_screen_failed",
        next_action=next_action, r92_run_id=run_id,
        r92_result=str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        r92_fit_starts=len(fit_details),
    )
    return result


def run() -> dict:
    register_card()
    return run_screen()


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, ensure_ascii=False))
