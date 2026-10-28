from __future__ import annotations

import hashlib
import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

from experiments.bill_target74.r16_q50_oof_blend import blend_parent_q50, select_blend_weight


def _group_values(values) -> np.ndarray:
    return pd.Series(values, dtype="string").fillna("__missing__").astype(str).to_numpy()


def select_group_weights(
    oof: pd.DataFrame,
    cutoff,
    *,
    group_col: str,
    fallback_weight: float,
    minimum_group_rows: int = 500,
    max_mape_regression_percent: float = 1.0,
) -> dict:
    """Choose Q50 shrinkage by group using only strictly earlier OOF rows."""
    if group_col not in oof.columns:
        raise ValueError(f"OOF frame is missing group column {group_col!r}")
    if (not np.isfinite(fallback_weight) or not 0.0 <= float(fallback_weight) <= 1.0):
        raise ValueError("fallback_weight must be finite and between 0 and 1")
    if minimum_group_rows < 1:
        raise ValueError("minimum_group_rows must be positive")

    cutoff_ts = pd.Timestamp(cutoff)
    dates = pd.to_datetime(oof.date_sold, errors="raise")
    past = oof.loc[dates < cutoff_ts].copy()
    past["date_sold"] = dates.loc[past.index]
    if not past.empty:
        past["_group_key"] = _group_values(past[group_col])
    else:
        past["_group_key"] = pd.Series(dtype=str)

    weights: dict[str, float] = {}
    details: dict[str, dict] = {}
    for group in sorted(past["_group_key"].unique().tolist()):
        local = past.loc[past._group_key.eq(group)].drop(columns="_group_key")
        selection = select_blend_weight(
            local, cutoff_ts, minimum_rows=minimum_group_rows,
            max_mape_regression_percent=max_mape_regression_percent,
        )
        used_fallback = selection["reason"] != "selected_from_strictly_past_oof"
        weight = float(fallback_weight if used_fallback else selection["weight"])
        weights[group] = weight
        details[group] = {**selection, "weight": weight, "used_fallback": used_fallback}

    return {
        "group_col": group_col,
        "cutoff_exclusive": cutoff_ts.isoformat(),
        "oof_rows": int(len(past)),
        "oof_months": sorted(past.month.astype(str).unique().tolist()) if len(past) else [],
        "fallback_weight": float(fallback_weight),
        "minimum_group_rows": int(minimum_group_rows),
        "weights": weights,
        "details": details,
    }


def blend_by_group(parent_prediction, q50_prediction, groups, weights: dict[str, float],
                   *, fallback_weight: float) -> np.ndarray:
    """Apply a frozen group-weight map; unseen groups use the supplied fallback."""
    parent = np.asarray(parent_prediction, dtype=float)
    q50 = np.asarray(q50_prediction, dtype=float)
    group_values = _group_values(groups)
    if parent.ndim != 1 or q50.ndim != 1 or parent.shape != q50.shape or len(group_values) != len(parent):
        raise ValueError("parent, Q50, and group inputs must be aligned one-dimensional vectors")
    if not np.isfinite(fallback_weight) or not 0.0 <= float(fallback_weight) <= 1.0:
        raise ValueError("fallback_weight must be finite and between 0 and 1")
    result = np.empty_like(parent)
    for group in np.unique(group_values):
        weight = float(weights.get(group, fallback_weight))
        mask = group_values == group
        result[mask] = blend_parent_q50(parent[mask], q50[mask], weight)
    return result


TRACK = "list_aware"
PARENT = "G01"
CARD = "R17_house_type_q50_oof_shrinkage"
SCREEN_MONTHS = ("2023-03", "2024-06")
GROUP_COLUMN = "house_type_name"
MINIMUM_GROUP_ROWS = 500
MAX_MAPE_REGRESSION_PERCENT = 1.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def _report(result: dict) -> str:
    score = result["tracks"][TRACK]
    lines = [
        "# R17：按房屋类型使用过去 OOF 选择 Q50 混合权重", "",
        f"Campaign：`{result['campaign_id']}`；run：`{result['run_id']}`。",
        "本轮不重训模型，复用哈希验证通过的 R16 Q50 walk-forward 预测。每个筛选 cutoff 和房屋类型分别在严格更早的 OOF 行上选权重；该类型历史行少于 500 时，回退到同一 cutoff 下 R16 已选出的全局权重。",
        "每个类型的权重在 0.00 至 1.00、步长 0.05 的冻结网格上最小化 MdAPE，并要求组内 MAPE 相对父模型恶化不超过 1.00%。", "",
        "## 按外层月份选择的组权重", "",
        "| 外层月份 | 类型 | 权重 | OOF行数 | OOF月份 | 父→混合 MdAPE | MAPE相对变化 | 回退全局 |",
        "|---|---|---:|---:|---|---:|---:|---|"]
    for screen in result["weight_selections"]:
        for group, detail in screen["details"].items():
            parent_mdape = detail["parent"].get("mdape_percent", float("nan"))
            selected_mdape = detail["selected"].get("mdape_percent", float("nan"))
            lines.append(
                f"| {screen['screen_month']} | {group} | {detail['weight']:.2f} | {detail['oof_rows']} | "
                f"{', '.join(detail['oof_months']) or '无'} | {parent_mdape:.5f}% → "
                f"{selected_mdape:.5f}% | {detail['selected_mape_relative_change_percent']:+.3f}% | "
                f"{'是' if detail['used_fallback'] else '否'} |"
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
                  "真实 learner fits：0；这是预注册的预测变换筛选。screen 权重只依赖 cutoff 之前的 OOF 标签与预测。",
                  "筛选门槛：pooled MdAPE 至少改善 0.25%、两个月分别改善、MAPE 相对恶化不超过 1%。失败则不做八个月确认、不改 G01 父模型。",
                  "2025-05/08 保留月份与隐藏标签未读取。", "",
                  "- 逐房筛选预测：`paired_predictions.csv`",
                  "- 复用的 OOF 预测与来源：`run_manifest.json`",
                  "- 分组权重与指标：`r17_screen_results.json`",
                  "- Campaign 事件账本：`../../fits.jsonl`", ""])
    return "\n".join(lines)


def run_r17_screen() -> dict:
    from experiments.bill_target74 import r03_catboost as helpers
    from experiments.bill_target74 import r04_ablation as budget
    from experiments.bill_target74 import runner as campaign
    from experiments.bill_greedy import runner as greedy
    from experiments.model_optimization_v2.folds import build_month_fold

    source = budget._source_identity()
    if campaign.PROTOCOL["cards"].get(CARD, {}).get("status") != "registered":
        raise RuntimeError("R17 must be registered in campaign_protocol.json before scoring")
    progress = json.loads(campaign.STATE.read_text(encoding="utf-8"))
    r16_result_path = campaign.ROOT / progress["r16_result"]
    r16_root = r16_result_path.parent
    r16_result = json.loads(r16_result_path.read_text(encoding="utf-8"))
    r16_manifest_path = r16_root / "run_manifest.json"
    r16_manifest = json.loads(r16_manifest_path.read_text(encoding="utf-8"))
    oof_path = r16_root / "oof_predictions.csv"
    if (_sha256(r16_result_path) != r16_manifest.get("result_sha256")
            or _sha256(oof_path) != r16_manifest.get("oof_prediction_sha256")
            or r16_manifest.get("campaign_id") != campaign.PROTOCOL["campaign_id"]
            or r16_result.get("card") != "R16_past_oof_q50_shrinkage"
            or r16_result.get("reserved_months_labels_opened") is not False
            or r16_result.get("hidden_labels_opened") is not False):
        raise RuntimeError("R16 source predictions failed closed-run hash or identity verification")
    ledger = [json.loads(line) for line in campaign.LEDGER.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not any(row.get("event") == "card_complete" and row.get("card") == r16_result["card"]
               and row.get("run_id") == r16_result["run_id"] for row in ledger):
        raise RuntimeError("R16 source run is not complete in the campaign ledger")

    oof_raw = pd.read_csv(oof_path, dtype={"ml_num": str})
    if oof_raw.duplicated(["month", "ml_num"]).any():
        raise RuntimeError("R16 OOF predictions contain duplicate fold/home rows")
    oof_parts = []
    provenance_inputs = {
        str(r16_result_path.relative_to(campaign.ROOT)).replace("\\", "/"): _sha256(r16_result_path),
        str(r16_manifest_path.relative_to(campaign.ROOT)).replace("\\", "/"): _sha256(r16_manifest_path),
        str(oof_path.relative_to(campaign.ROOT)).replace("\\", "/"): _sha256(oof_path),
    }
    for month in sorted(oof_raw.month.astype(str).unique()):
        fold_rows = oof_raw.loc[oof_raw.month.astype(str).eq(month)].copy()
        listings, _, _, _, _ = greedy._load_fold(month)
        _, homes, truth = build_month_fold(listings, month, TRACK)
        expected_ids = homes.ml_num.astype(str).to_numpy()
        if (not np.array_equal(fold_rows.ml_num.astype(str).to_numpy(), expected_ids)
                or not np.array_equal(fold_rows.sold_price.to_numpy(float), truth.sold_price.to_numpy(float))):
            raise RuntimeError(f"R16 OOF rows do not align with current source fold {month}")
        fold_rows[GROUP_COLUMN] = homes[GROUP_COLUMN].astype("string").fillna("__missing__").astype(str).to_numpy()
        fold_rows["date_sold"] = pd.to_datetime(fold_rows.date_sold, errors="raise")
        oof_parts.append(fold_rows)
    oof = pd.concat(oof_parts, ignore_index=True)
    global_weights = {row["screen_month"]: float(row["weight"])
                      for row in r16_result["weight_selections"]}

    run_id = "bill_target74_r17_q50_groupblend_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = campaign.CAMPAIGN_ROOT / "batches" / run_id
    output.mkdir(parents=True, exist_ok=False)
    campaign._append({"event": "card_start", "card": CARD, "run_id": run_id,
                      "screen_months": list(SCREEN_MONTHS), "fit_start_count": 0,
                      "source_run_id": r16_result["run_id"], "group_col": GROUP_COLUMN,
                      "minimum_group_oof_rows": MINIMUM_GROUP_ROWS,
                      "fallback_weights": global_weights,
                      "strict_past_only_weight_selection": True,
                      "reserved_months_labels_opened": False, "hidden_labels_opened": False})

    weight_selections = []
    screen_rows = []
    try:
        for month in SCREEN_MONTHS:
            listings, _, _, _, _ = greedy._load_fold(month)
            _, homes, truth = build_month_fold(listings, month, TRACK)
            cutoff = pd.Period(month, freq="M").start_time
            global_weight = global_weights.get(month)
            if global_weight is None:
                raise RuntimeError(f"R16 global fallback is missing for screen month {month}")
            selection = select_group_weights(
                oof, cutoff, group_col=GROUP_COLUMN, fallback_weight=global_weight,
                minimum_group_rows=MINIMUM_GROUP_ROWS,
                max_mape_regression_percent=MAX_MAPE_REGRESSION_PERCENT,
            )
            selection["screen_month"] = month
            weight_selections.append(selection)
            candidate_source = oof.loc[oof.month.astype(str).eq(month)].copy()
            expected_ids = homes.ml_num.astype(str).to_numpy()
            if (not np.array_equal(candidate_source.ml_num.astype(str).to_numpy(), expected_ids)
                    or not np.array_equal(candidate_source.sold_price.to_numpy(float), truth.sold_price.to_numpy(float))):
                raise RuntimeError(f"R16 candidate predictions do not align with screen fold {month}")
            predictions = blend_by_group(
                candidate_source.parent_prediction.to_numpy(float),
                candidate_source.q50_prediction.to_numpy(float),
                homes[GROUP_COLUMN].astype("string").fillna("__missing__").astype(str).to_numpy(),
                selection["weights"], fallback_weight=global_weight,
            )
            parent, parent_path = budget._check_parent_artifact(greedy, month, TRACK, homes, truth, PARENT)
            if not np.array_equal(candidate_source.parent_prediction.to_numpy(float),
                                  parent.predicted_price.to_numpy(float)):
                raise RuntimeError(f"R16 parent predictions differ from the frozen G01 comparator in {month}")
            candidate = pd.DataFrame({"ml_num": expected_ids, "sold_price": truth.sold_price.to_numpy(float),
                                      "predicted_price": predictions})
            screen_rows.append(helpers._candidate_frame(month, TRACK, truth, parent, candidate))
            campaign._append({"event": "screen_fold_complete", "card": CARD, "run_id": run_id,
                              "track": TRACK, "month": month, "parent_id": PARENT,
                              "source_run_id": r16_result["run_id"],
                              "source_parent_csv_sha256": _sha256(parent_path),
                              "global_fallback_weight": global_weight,
                              "oof_rows": selection["oof_rows"], "oof_months": selection["oof_months"],
                              "group_weights": selection["weights"],
                              "date_cutoff_exclusive": cutoff.isoformat()})
        paired = pd.concat(screen_rows, ignore_index=True)
        paired_path = output / "paired_predictions.csv"
        paired.to_csv(paired_path, index=False)
        tracks = {TRACK: helpers._track_result(paired)}
        result = {"campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
                  "card": CARD, "parents": {TRACK: PARENT}, "screen_months": list(SCREEN_MONTHS),
                  "source_run_id": r16_result["run_id"], "tracks": tracks,
                  "weight_selections": weight_selections, "fit_starts": 0,
                  "prediction_transform_count": len(SCREEN_MONTHS),
                  "minimum_group_oof_rows": MINIMUM_GROUP_ROWS,
                  "reserved_months_labels_opened": False, "hidden_labels_opened": False,
                  "data_sha256": source["data_sha256"]}
        result_path = output / "r17_screen_results.json"
        _write_json(result_path, result)
        manifest = {
            "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
            "source_identity": source, "source_r16_run_id": r16_result["run_id"],
            "source_r16_manifest_sha256": _sha256(r16_manifest_path),
            "source_r16_oof_sha256": _sha256(oof_path),
            "source_file_sha256": provenance_inputs,
            "source_input_sha256": r16_manifest.get("input_sha256", {}),
            "candidate_rows_sha256": _sha256(paired_path),
            "result_sha256": _sha256(result_path),
            "group_col": GROUP_COLUMN, "minimum_group_oof_rows": MINIMUM_GROUP_ROWS,
            "fallback_global_weights": global_weights,
            "weight_selection": "past-only MdAPE grid with <=1% within-group MAPE regression",
            "fit_starts": 0, "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        }
        _write_json(output / "run_manifest.json", manifest)
        (output / "research_report.md").write_text(_report(result), encoding="utf-8")
        passed = tracks[TRACK]["qualifies_screen"]
        next_action = ("R17 list-aware passed screen; run strict sequential eight-month confirmation" if passed
                      else "R17 did not qualify; freeze G01 and register a distinct next candidate")
        campaign._append({"event": "card_complete", "card": CARD, "run_id": run_id,
                          "fit_starts": 0, "result_path": str(result_path.relative_to(campaign.ROOT)),
                          "qualifies_by_track": {TRACK: passed}, "reserved_months_labels_opened": False,
                          "hidden_labels_opened": False})
        campaign._write_progress("r17_q50_group_blend_screen_complete", next_action=next_action,
                                 r17_result=str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
                                 latest_fit_count=0)
        return result
    except Exception as exc:
        failure = {"run_id": run_id, "card": CARD, "error": f"{type(exc).__name__}: {exc}",
                   "source_run_id": r16_result["run_id"], "reserved_months_labels_opened": False,
                   "hidden_labels_opened": False}
        _write_json(output / "failure_checkpoint.json", failure)
        campaign._append({"event": "card_failed", "card": CARD, "run_id": run_id,
                          "error": failure["error"], "fit_starts": 0})
        campaign._write_progress("r17_q50_group_blend_failed",
                                 next_action="inspect R17 failure checkpoint before selecting another card",
                                 r17_failure=str((output / "failure_checkpoint.json").relative_to(campaign.ROOT)).replace("\\", "/"))
        raise
