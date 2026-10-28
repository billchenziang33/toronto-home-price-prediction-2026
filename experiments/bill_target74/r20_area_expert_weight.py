from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.bill_target74.r19_expert_weight import (
    EXPERT, PARENT, TRACK, _load_fold_components, _sha256, _write_json,
)


CARD = "R20_area_banded_past_oof_G18_weights"
SCREEN_MONTHS = ("2023-03", "2024-06")
OOF_MONTHS = ("2022-06", "2022-11", "2023-03", "2023-07", "2023-11", "2024-03")
PROCESS_MONTHS = (*OOF_MONTHS, "2024-06")
AREA_SPLIT_SQFT = 1600.0
PARENT_EXPERT_WEIGHT = 0.75
MINIMUM_OOF_ROWS = 1000
MINIMUM_BAND_ROWS = 500
MAX_MAPE_REGRESSION_PERCENT = 1.0
AREA_BANDS = ("detached_le_1600", "detached_gt_1600")


def area_band_keys(detached, house_area) -> np.ndarray:
    route = np.asarray(detached, dtype=bool)
    area = pd.to_numeric(pd.Series(house_area), errors="coerce").to_numpy(dtype=float)
    if route.ndim != 1 or area.ndim != 1 or route.shape != area.shape:
        raise ValueError("Detached route and house area must be aligned one-dimensional vectors")
    keys = np.full(len(route), "not_detached", dtype=object)
    unknown = route & (~np.isfinite(area) | (area <= 0.0))
    keys[unknown] = "detached_area_unknown"
    known = route & ~unknown
    keys[known & (area <= AREA_SPLIT_SQFT)] = AREA_BANDS[0]
    keys[known & (area > AREA_SPLIT_SQFT)] = AREA_BANDS[1]
    return keys


def blend_detached_by_area(base_prediction, expert_prediction, detached_mask, house_area,
                           weights: dict[str, float], *, fallback_weight: float = PARENT_EXPERT_WEIGHT) -> np.ndarray:
    base = np.asarray(base_prediction, dtype=float)
    expert = np.asarray(expert_prediction, dtype=float)
    route = np.asarray(detached_mask, dtype=bool)
    area = pd.to_numeric(pd.Series(house_area), errors="coerce").to_numpy(dtype=float)
    if any(value.ndim != 1 for value in (base, expert, route, area)) or not (
            base.shape == expert.shape == route.shape == area.shape):
        raise ValueError("base, expert, route, and area inputs must be aligned one-dimensional vectors")
    if (not np.isfinite(base).all() or (base <= 0.0).any()
            or not np.isfinite(expert[route]).all() or (expert[route] <= 0.0).any()):
        raise ValueError("base and expert prices must be finite positive values")
    if not np.isfinite(fallback_weight) or not 0.0 <= float(fallback_weight) <= 1.0:
        raise ValueError("fallback expert weight must be finite and between 0 and 1")
    keys = area_band_keys(route, area)
    result = np.round(base)
    for key in (*AREA_BANDS, "detached_area_unknown"):
        mask = keys == key
        if not mask.any():
            continue
        weight = float(weights.get(key, fallback_weight))
        if not np.isfinite(weight) or not 0.0 <= weight <= 1.0:
            raise ValueError(f"expert weight for {key} must be finite and between 0 and 1")
        result[mask] = np.round((1.0 - weight) * base[mask] + weight * expert[mask])
    return result


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    ape = 100.0 * np.abs(predicted - actual) / actual
    return {"mdape_percent": float(np.median(ape)), "mape_percent": float(np.mean(ape))}


def _mape_relative_change(parent_mape: float, candidate_mape: float) -> float:
    if parent_mape == 0.0:
        return 0.0 if candidate_mape == 0.0 else float("inf")
    return 100.0 * (candidate_mape - parent_mape) / parent_mape


def select_area_weights(
    oof: pd.DataFrame,
    cutoff,
    *,
    minimum_rows: int = MINIMUM_OOF_ROWS,
    minimum_band_rows: int = MINIMUM_BAND_ROWS,
    max_mape_regression_percent: float = MAX_MAPE_REGRESSION_PERCENT,
) -> dict:
    required = {"month", "ml_num", "date_sold", "house_area", "sold_price", "parent_prediction",
                "g06_prediction", "g18_prediction", "detached"}
    missing = sorted(required.difference(oof.columns))
    if missing:
        raise ValueError(f"OOF frame is missing columns: {', '.join(missing)}")
    if minimum_rows < 1 or minimum_band_rows < 1:
        raise ValueError("OOF minimum row counts must be positive")
    if not np.isfinite(max_mape_regression_percent) or max_mape_regression_percent < 0.0:
        raise ValueError("max_mape_regression_percent must be finite and nonnegative")

    cutoff_ts = pd.Timestamp(cutoff)
    dates = pd.to_datetime(oof.date_sold, errors="raise")
    past = oof.loc[dates < cutoff_ts].copy()
    past["date_sold"] = dates.loc[past.index]
    if not past.empty:
        if not past.date_sold.dt.to_period("M").astype(str).eq(past.month.astype(str)).all():
            raise ValueError("OOF month values do not match their sale dates")
        if past.duplicated(["month", "ml_num"]).any():
            raise ValueError("OOF rows must be unique by month and ml_num")

    fallback = {band: PARENT_EXPERT_WEIGHT for band in AREA_BANDS}
    actual = past.sold_price.to_numpy(dtype=float)
    parent = past.parent_prediction.to_numpy(dtype=float)
    base = past.g06_prediction.to_numpy(dtype=float)
    expert = past.g18_prediction.to_numpy(dtype=float)
    route = past.detached.to_numpy(dtype=bool)
    area = pd.to_numeric(past.house_area, errors="coerce").to_numpy(dtype=float)
    if len(past):
        if not np.isfinite(np.column_stack((actual, parent, base))).all() or (
                np.column_stack((actual, parent, base)) <= 0.0).any():
            raise ValueError("OOF actual, parent, and G06 prices must be finite positive values")
        if not np.isfinite(expert[route]).all() or (expert[route] <= 0.0).any():
            raise ValueError("Detached G18 OOF prices must be finite positive values")
    keys = area_band_keys(route, area)
    band_counts = {band: int(np.sum(keys == band)) for band in AREA_BANDS}
    supported = [band for band in AREA_BANDS if band_counts[band] >= minimum_band_rows]
    parent_metrics = _metrics(actual, parent) if len(past) else {
        "mdape_percent": float("nan"), "mape_percent": float("nan")
    }
    fallback_prediction = blend_detached_by_area(base, np.where(route, expert, base), route, area,
                                                   fallback, fallback_weight=PARENT_EXPERT_WEIGHT)
    fallback_metrics = _metrics(actual, fallback_prediction) if len(past) else parent_metrics
    result = {
        "weights": dict(fallback), "oof_rows": int(len(past)),
        "detached_oof_rows": int(route.sum()), "band_rows": band_counts,
        "supported_bands": supported,
        "oof_months": sorted(past.month.astype(str).unique().tolist()) if len(past) else [],
        "parent": parent_metrics, "selected": fallback_metrics,
        "selected_mape_relative_change_percent": (
            _mape_relative_change(parent_metrics["mape_percent"], fallback_metrics["mape_percent"])
            if len(past) else 0.0),
        "eligible_pairs": [], "reason": "insufficient_past_oof_rows",
    }
    if len(past) < minimum_rows or not supported:
        return result

    grid = [float(round(value, 10)) for value in np.linspace(0.0, 1.0, 21)]
    candidates = []
    import itertools
    for values in itertools.product(grid, repeat=len(supported)):
        weights = dict(fallback)
        weights.update(dict(zip(supported, values)))
        prediction = blend_detached_by_area(
            base, np.where(route, expert, base), route, area, weights,
            fallback_weight=PARENT_EXPERT_WEIGHT,
        )
        metrics = _metrics(actual, prediction)
        mape_change = _mape_relative_change(parent_metrics["mape_percent"], metrics["mape_percent"])
        if mape_change <= max_mape_regression_percent:
            distance = sum((weights[band] - PARENT_EXPERT_WEIGHT) ** 2 for band in AREA_BANDS)
            candidates.append({"weights": weights, "metrics": metrics,
                               "mape_relative_change_percent": float(mape_change),
                               "distance_from_parent_weights": float(distance)})
    if not candidates:
        return {**result, "reason": "no_weight_pair_passed_mape_guard"}

    selected = min(candidates, key=lambda row: (
        row["metrics"]["mdape_percent"], row["metrics"]["mape_percent"],
        row["distance_from_parent_weights"],
    ))
    return {
        **result, "weights": selected["weights"], "selected": selected["metrics"],
        "selected_mape_relative_change_percent": selected["mape_relative_change_percent"],
        "eligible_pairs": [{"weights": row["weights"],
                            "mdape_percent": row["metrics"]["mdape_percent"],
                            "mape_percent": row["metrics"]["mape_percent"]}
                           for row in candidates],
        "reason": "selected_from_strictly_past_oof",
    }


def _report(result: dict) -> str:
    score = result["tracks"][TRACK]
    lines = [
        "# R20：Detached 按面积段选择 G18 专家权重", "",
        f"Campaign：`{result['campaign_id']}`；run：`{result['run_id']}`。",
        "只对 Detached 房源按 `house_area <= 1,600 sq ft` 与 `> 1,600 sq ft` 分段；非 Detached 保持 G06。两个权重按每个外层 cutoff 的过去 OOF pooled MdAPE 联合选择，并用 pooled MAPE 回退护栏约束。面积缺失或非正时使用父模型 0.75 专家权重。", "",
        "## 过去 OOF 联合权重", "", "| 外层月份 | <=1600权重 | >1600权重 | OOF行数 | 两段 Detached 行数 | OOF月份 | 父→选中 MdAPE | MAPE变化 |",
        "|---|---:|---:|---:|---:|---|---:|---:|"]
    for row in result["weight_selections"]:
        lines.append(
            f"| {row['screen_month']} | {row['weights']['detached_le_1600']:.2f} | "
            f"{row['weights']['detached_gt_1600']:.2f} | {row['oof_rows']} | "
            f"{row['band_rows']['detached_le_1600']} / {row['band_rows']['detached_gt_1600']} | "
            f"{', '.join(row['oof_months'])} | {row['parent']['mdape_percent']:.5f}% → "
            f"{row['selected']['mdape_percent']:.5f}% | "
            f"{row['selected_mape_relative_change_percent']:+.3f}% |"
        )
    lines.extend(["", "## 双月筛选", "",
                  "| 赛道 | 父→候选 MdAPE | 相对改善 | MAPE相对变化 | 父→候选 P90 APE | 通过筛选 |",
                  "|---|---:|---:|---:|---:|---|"])
    lines.append(
        f"| {TRACK} | {score['parent']['mdape_percent']:.5f}% → {score['candidate']['mdape_percent']:.5f}% | "
        f"{score['relative_mdape_improvement_percent']:+.3f}% | "
        f"{score['mape_relative_change_percent']:+.3f}% | "
        f"{score['parent']['p90_ape_percent']:.3f}% → {score['candidate']['p90_ape_percent']:.3f}% | "
        f"{'是' if score['qualifies_screen'] else '否'} |"
    )
    lines.extend(["", "## 逐月结果", "", "| 月份 | 父 MdAPE | 候选 MdAPE | 相对改善 | 父 MAPE | 候选 MAPE |",
                  "|---|---:|---:|---:|---:|---:|"])
    for month, monthly in score["monthly"].items():
        lines.append(
            f"| {month} | {monthly['parent']['mdape_percent']:.5f}% | "
            f"{monthly['candidate']['mdape_percent']:.5f}% | "
            f"{monthly['relative_mdape_improvement_percent']:+.3f}% | "
            f"{monthly['parent']['mape_percent']:.5f}% | {monthly['candidate']['mape_percent']:.5f}% |"
        )
    lines.extend(["", "## 运行边界", "",
                  "真实 learner fits：0。权重仅使用严格早于筛选 cutoff 的 OOF 标签和 G06/G18 预测；G06/G18 在七个 OOF/筛选月份均已逐行重建 G19。",
                  "筛选门槛仍为 pooled MdAPE 改善至少 0.25%、两个筛选月分别改善、MAPE 恶化不超过 1%。未通过时不做八个月确认或更改父模型。保留月份和隐藏标签未读取。", "",
                  "- 逐房配对预测：`paired_predictions.csv`",
                  "- OOF 来源与组权重：`r20_screen_results.json`",
                  "- 输入哈希和筛选结果哈希：`run_manifest.json`",
                  "- Campaign 事件账本：`../../fits.jsonl`", ""])
    return "\n".join(lines)


def run_r20_screen() -> dict:
    from experiments.bill_target74 import r03_catboost as helpers
    from experiments.bill_target74 import r04_ablation as budget
    from experiments.bill_target74 import runner as campaign
    from experiments.bill_greedy import runner as greedy
    from experiments.model_optimization_v2.folds import build_month_fold

    ROOT = campaign.ROOT
    source = budget._source_identity()
    if campaign.PROTOCOL["cards"].get(CARD, {}).get("status") != "registered":
        raise RuntimeError("R20 must be registered in campaign_protocol.json before scoring")
    run_id = "bill_target74_r20_area_expert_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = campaign.CAMPAIGN_ROOT / "batches" / run_id
    output.mkdir(parents=True, exist_ok=False)
    campaign._append({"event": "card_start", "card": CARD, "run_id": run_id,
                      "screen_months": list(SCREEN_MONTHS), "oof_months": list(OOF_MONTHS),
                      "fit_start_count": 0, "parent": PARENT,
                      "area_split_sqft": AREA_SPLIT_SQFT, "parent_weight_by_band": PARENT_EXPERT_WEIGHT,
                      "minimum_oof_rows": MINIMUM_OOF_ROWS, "minimum_band_rows": MINIMUM_BAND_ROWS,
                      "reserved_months_labels_opened": False, "hidden_labels_opened": False})
    input_hashes: dict[str, str] = {}
    fold_rows: dict[str, pd.DataFrame] = {}
    weight_selections = []
    screen_frames = []
    try:
        for month in PROCESS_MONTHS:
            frame, hashes = _load_fold_components(month, greedy, budget, campaign)
            listings, _, homes, truth, _ = greedy._load_fold(month)
            # _load_fold_components guarantees the row order matches its fold; attach area by ID.
            from model_candidate_v2.features import square_feet
            area = pd.to_numeric(square_feet(homes.house_area), errors="coerce").to_numpy(dtype=float)
            if (not np.array_equal(frame.ml_num.astype(str).to_numpy(), homes.ml_num.astype(str).to_numpy())
                    or not np.array_equal(frame.sold_price.to_numpy(float), truth.sold_price.to_numpy(float))):
                raise RuntimeError(f"R19 component fold rows no longer align for {month}")
            frame["house_area"] = area
            fold_rows[month] = frame
            input_hashes.update(hashes)
            if month in SCREEN_MONTHS:
                parent_path = campaign._row_prediction_path(month, TRACK)
                sidecar = parent_path.with_suffix(".json")
                input_hashes[str(sidecar.relative_to(ROOT)).replace("\\", "/")] = _sha256(sidecar)

        oof = pd.concat([fold_rows[month] for month in OOF_MONTHS], ignore_index=True)
        if oof.duplicated(["month", "ml_num"]).any():
            raise RuntimeError("R20 OOF rows contain duplicate fold/home pairs")
        for month in SCREEN_MONTHS:
            cutoff = pd.Period(month, freq="M").start_time
            selection = select_area_weights(
                oof, cutoff, minimum_rows=MINIMUM_OOF_ROWS,
                minimum_band_rows=MINIMUM_BAND_ROWS,
                max_mape_regression_percent=MAX_MAPE_REGRESSION_PERCENT,
            )
            selection["screen_month"] = month
            weight_selections.append(selection)
            frame = fold_rows[month]
            candidate_prediction = blend_detached_by_area(
                frame.g06_prediction.to_numpy(float), frame.g18_prediction.fillna(
                    frame.g06_prediction).to_numpy(float), frame.detached.to_numpy(bool),
                frame.house_area.to_numpy(float), selection["weights"],
                fallback_weight=PARENT_EXPERT_WEIGHT,
            )
            parent = pd.DataFrame({"ml_num": frame.ml_num.astype(str).to_numpy(),
                                   "sold_price": frame.sold_price.to_numpy(float),
                                   "predicted_price": frame.parent_prediction.to_numpy(float)})
            candidate = pd.DataFrame({"ml_num": frame.ml_num.astype(str).to_numpy(),
                                      "sold_price": frame.sold_price.to_numpy(float),
                                      "predicted_price": candidate_prediction})
            truth = pd.DataFrame({"ml_num": frame.ml_num.astype(str).to_numpy(),
                                  "sold_price": frame.sold_price.to_numpy(float)})
            screen_frames.append(helpers._candidate_frame(month, TRACK, truth, parent, candidate))
            campaign._append({"event": "screen_fold_complete", "card": CARD, "run_id": run_id,
                              "track": TRACK, "month": month, "parent_id": PARENT,
                              "expert_weight_by_area": selection["weights"],
                              "oof_rows": selection["oof_rows"], "oof_band_rows": selection["band_rows"],
                              "oof_months": selection["oof_months"],
                              "date_cutoff_exclusive": cutoff.isoformat()})

        paired = pd.concat(screen_frames, ignore_index=True)
        paired_path = output / "paired_predictions.csv"
        paired.to_csv(paired_path, index=False)
        tracks = {TRACK: helpers._track_result(paired)}
        result = {"campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
                  "card": CARD, "parents": {TRACK: PARENT}, "screen_months": list(SCREEN_MONTHS),
                  "weight_selections": weight_selections, "tracks": tracks,
                  "fit_starts": 0, "component_reconstruction_months": list(PROCESS_MONTHS),
                  "reserved_months_labels_opened": False, "hidden_labels_opened": False,
                  "data_sha256": source["data_sha256"]}
        result_path = output / "r20_screen_results.json"
        _write_json(result_path, result)
        source_files = [campaign.PROTOCOL_PATH, campaign.PACKAGE / "r20_area_expert_weight.py",
                        campaign.PACKAGE / "r19_expert_weight.py", campaign.PACKAGE / "r03_catboost.py",
                        campaign.PACKAGE / "r04_ablation.py", ROOT / "experiments/bill_greedy/runner.py",
                        ROOT / "experiments/model_optimization_v2/folds.py",
                        ROOT / "model_candidate_v2/features.py"]
        manifest = {"campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
                    "source_identity": source,
                    "code_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): _sha256(path)
                                    for path in source_files},
                    "input_sha256": input_hashes, "weight_selections": weight_selections,
                    "fit_starts": 0, "paired_predictions_sha256": _sha256(paired_path),
                    "result_sha256": _sha256(result_path), "reserved_months_labels_opened": False,
                    "hidden_labels_opened": False}
        _write_json(output / "run_manifest.json", manifest)
        (output / "research_report.md").write_text(_report(result), encoding="utf-8")
        passed = tracks[TRACK]["qualifies_screen"]
        next_action = ("R20 off-market passed screen; run strict sequential eight-month confirmation" if passed
                      else "R20 did not qualify; preserve G19 and register a distinct next candidate")
        campaign._append({"event": "card_complete", "card": CARD, "run_id": run_id,
                          "fit_starts": 0, "result_path": str(result_path.relative_to(ROOT)),
                          "qualifies_by_track": {TRACK: passed}, "reserved_months_labels_opened": False,
                          "hidden_labels_opened": False})
        campaign._write_progress("r20_area_expert_screen_complete", next_action=next_action,
                                 r20_result=str(result_path.relative_to(ROOT)).replace("\\", "/"),
                                 latest_fit_count=0)
        return result
    except Exception as exc:
        failure = {"run_id": run_id, "card": CARD, "error": f"{type(exc).__name__}: {exc}",
                   "completed_component_months": list(fold_rows), "fit_starts": 0,
                   "reserved_months_labels_opened": False, "hidden_labels_opened": False}
        _write_json(output / "failure_checkpoint.json", failure)
        campaign._append({"event": "card_failed", "card": CARD, "run_id": run_id,
                          "error": failure["error"], "fit_starts": 0})
        campaign._write_progress("r20_area_expert_failed",
                                 next_action="inspect R20 failure checkpoint before choosing another card",
                                 r20_failure=str((output / "failure_checkpoint.json").relative_to(ROOT)).replace("\\", "/"))
        raise
