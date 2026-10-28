from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.bill_target74.r16_q50_oof_blend import blend_parent_q50, select_blend_weight
from experiments.bill_target74.r17_q50_group_blend import blend_by_group


TRACK = "list_aware"
PARENT = "G01"
CARD = "R18_global_oof_shrinkage_of_R17"
SCREEN_MONTHS = ("2023-03", "2024-06")
GROUP_COLUMN = "house_type_name"
MINIMUM_OOF_ROWS = 1000
MAX_MAPE_REGRESSION_PERCENT = 1.0


def select_global_shrinkage(history: pd.DataFrame, cutoff, grouped_candidate_prediction, *,
                            minimum_rows: int = MINIMUM_OOF_ROWS,
                            max_mape_regression_percent: float = MAX_MAPE_REGRESSION_PERCENT) -> dict:
    candidate = np.asarray(grouped_candidate_prediction, dtype=float)
    if candidate.ndim != 1 or len(candidate) != len(history):
        raise ValueError("grouped candidate predictions must align with one-dimensional OOF rows")
    if not np.isfinite(candidate).all() or (candidate <= 0.0).any():
        raise ValueError("grouped candidate predictions must be finite positive prices")
    rows = history.copy()
    rows["q50_prediction"] = candidate
    return select_blend_weight(
        rows, cutoff, minimum_rows=minimum_rows,
        max_mape_regression_percent=max_mape_regression_percent,
    )


def apply_global_shrinkage(parent_prediction, grouped_candidate_prediction, weight: float) -> np.ndarray:
    return blend_parent_q50(parent_prediction, grouped_candidate_prediction, weight)


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
        "# R18：在分类型 Q50 候选上使用过去 OOF 进行全局收缩", "",
        f"Campaign：`{result['campaign_id']}`；run：`{result['run_id']}`。",
        "候选沿用 R17 的按房屋类型权重和 R16 的 Q50 walk-forward 预测，再在完整分类型候选与 G01 之间做一次全局价格空间混合。每个外层 cutoff 的全局权重只由严格更早的 OOF 标签与预测选择。",
        "权重在 0.00 至 1.00、步长 0.05 的冻结网格上最小化过去 OOF MdAPE，并限制 pooled MAPE 相对 G01 恶化不超过 1.00%。", "",
        "## 全局收缩权重", "",
        "| 外层月份 | 权重 | 严格过去 OOF 月份 | 行数 | 父→选中 MdAPE | MAPE相对变化 |",
        "|---|---:|---|---:|---:|---:|"]
    for row in result["weight_selections"]:
        lines.append(
            f"| {row['screen_month']} | {row['weight']:.2f} | {', '.join(row['oof_months']) or '无'} | "
            f"{row['oof_rows']} | {row['parent'].get('mdape_percent', float('nan')):.5f}% → "
            f"{row['selected'].get('mdape_percent', float('nan')):.5f}% | "
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
                  "真实 learner fits：0；R16/R17 来源文件与行顺序逐一核验。当前筛选标签没有进入权重选择。",
                  "筛选门槛：pooled MdAPE 至少改善 0.25%、两个筛选月都改善、MAPE 相对恶化不超过 1%。失败则不做八个月确认，也不更改 G01。",
                  "2025-05/08 保留月份与隐藏标签未读取。", "",
                  "- 逐房筛选预测：`paired_predictions.csv`",
                  "- 全局权重及逐月结果：`r18_screen_results.json`",
                  "- 来源哈希与复现信息：`run_manifest.json`",
                  "- Campaign 事件账本：`../../fits.jsonl`", ""])
    return "\n".join(lines)


def run_r18_screen() -> dict:
    from experiments.bill_target74 import r03_catboost as helpers
    from experiments.bill_target74 import r04_ablation as budget
    from experiments.bill_target74 import runner as campaign
    from experiments.bill_greedy import runner as greedy
    from experiments.model_optimization_v2.folds import build_month_fold

    source = budget._source_identity()
    if campaign.PROTOCOL["cards"].get(CARD, {}).get("status") != "registered":
        raise RuntimeError("R18 must be registered in campaign_protocol.json before scoring")
    progress = json.loads(campaign.STATE.read_text(encoding="utf-8"))
    r17_result_path = campaign.ROOT / progress["r17_result"]
    r17_root = r17_result_path.parent
    r17_result = json.loads(r17_result_path.read_text(encoding="utf-8"))
    r17_manifest_path = r17_root / "run_manifest.json"
    r17_manifest = json.loads(r17_manifest_path.read_text(encoding="utf-8"))
    if (_sha256(r17_result_path) != r17_manifest.get("result_sha256")
            or r17_result.get("card") != "R17_house_type_q50_oof_shrinkage"
            or r17_result.get("reserved_months_labels_opened") is not False
            or r17_result.get("hidden_labels_opened") is not False):
        raise RuntimeError("R17 source result failed run hash or identity verification")
    ledger = [json.loads(line) for line in campaign.LEDGER.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not any(row.get("event") == "card_complete" and row.get("card") == r17_result["card"]
               and row.get("run_id") == r17_result["run_id"] for row in ledger):
        raise RuntimeError("R17 source run is not complete in the campaign ledger")

    r16_result_path = campaign.ROOT / progress["r16_result"]
    r16_root = r16_result_path.parent
    r16_result = json.loads(r16_result_path.read_text(encoding="utf-8"))
    r16_manifest_path = r16_root / "run_manifest.json"
    r16_manifest = json.loads(r16_manifest_path.read_text(encoding="utf-8"))
    oof_path = r16_root / "oof_predictions.csv"
    if (_sha256(r16_result_path) != r16_manifest.get("result_sha256")
            or _sha256(oof_path) != r16_manifest.get("oof_prediction_sha256")
            or r17_result.get("source_run_id") != r16_result.get("run_id")):
        raise RuntimeError("R16 OOF predictions failed closed-run hash verification")
    oof_raw = pd.read_csv(oof_path, dtype={"ml_num": str})
    if oof_raw.duplicated(["month", "ml_num"]).any():
        raise RuntimeError("R16 OOF source contains duplicate fold/home rows")
    oof_parts = []
    for month in sorted(oof_raw.month.astype(str).unique()):
        fold_rows = oof_raw.loc[oof_raw.month.astype(str).eq(month)].copy()
        listings, _, _, _, _ = greedy._load_fold(month)
        _, homes, truth = build_month_fold(listings, month, TRACK)
        if (not np.array_equal(fold_rows.ml_num.astype(str).to_numpy(), homes.ml_num.astype(str).to_numpy())
                or not np.array_equal(fold_rows.sold_price.to_numpy(float), truth.sold_price.to_numpy(float))):
            raise RuntimeError(f"R16 OOF source rows do not align for {month}")
        fold_rows[GROUP_COLUMN] = homes[GROUP_COLUMN].astype("string").fillna("__missing__").astype(str).to_numpy()
        fold_rows["date_sold"] = pd.to_datetime(fold_rows.date_sold, errors="raise")
        oof_parts.append(fold_rows)
    oof = pd.concat(oof_parts, ignore_index=True)
    r17_selections = {row["screen_month"]: row for row in r17_result["weight_selections"]}
    r16_global_weights = {row["screen_month"]: float(row["weight"])
                          for row in r16_result["weight_selections"]}
    source_hashes = {
        str(path.relative_to(campaign.ROOT)).replace("\\", "/"): _sha256(path)
        for path in (r16_result_path, r16_manifest_path, oof_path,
                     r17_result_path, r17_manifest_path)
    }

    run_id = "bill_target74_r18_q50_meta_shrink_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = campaign.CAMPAIGN_ROOT / "batches" / run_id
    output.mkdir(parents=True, exist_ok=False)
    campaign._append({"event": "card_start", "card": CARD, "run_id": run_id,
                      "screen_months": list(SCREEN_MONTHS), "fit_start_count": 0,
                      "source_r16_run_id": r16_result["run_id"], "source_r17_run_id": r17_result["run_id"],
                      "minimum_oof_rows": MINIMUM_OOF_ROWS,
                      "max_mape_regression_percent": MAX_MAPE_REGRESSION_PERCENT,
                      "reserved_months_labels_opened": False, "hidden_labels_opened": False})
    weight_selections = []
    screen_rows = []
    try:
        for month in SCREEN_MONTHS:
            cutoff = pd.Period(month, freq="M").start_time
            r17_selection = r17_selections.get(month)
            global_fallback = r16_global_weights.get(month)
            if r17_selection is None or global_fallback is None:
                raise RuntimeError(f"R16/R17 selections are missing for {month}")
            history = oof.loc[pd.to_datetime(oof.date_sold) < cutoff].copy()
            grouped_history = blend_by_group(
                history.parent_prediction.to_numpy(float), history.q50_prediction.to_numpy(float),
                history[GROUP_COLUMN].astype(str).to_numpy(), r17_selection["weights"],
                fallback_weight=float(r17_selection["fallback_weight"]),
            )
            selection = select_global_shrinkage(
                history, cutoff, grouped_history, minimum_rows=MINIMUM_OOF_ROWS,
                max_mape_regression_percent=MAX_MAPE_REGRESSION_PERCENT,
            )
            selection["screen_month"] = month
            selection["group_weights"] = r17_selection["weights"]
            weight_selections.append(selection)

            listings, _, _, _, _ = greedy._load_fold(month)
            _, homes, truth = build_month_fold(listings, month, TRACK)
            candidate_source = oof.loc[oof.month.astype(str).eq(month)].copy()
            expected_ids = homes.ml_num.astype(str).to_numpy()
            if (not np.array_equal(candidate_source.ml_num.astype(str).to_numpy(), expected_ids)
                    or not np.array_equal(candidate_source.sold_price.to_numpy(float), truth.sold_price.to_numpy(float))):
                raise RuntimeError(f"R16 Q50 prediction rows do not align with {month}")
            parent, parent_path = budget._check_parent_artifact(greedy, month, TRACK, homes, truth, PARENT)
            if not np.array_equal(candidate_source.parent_prediction.to_numpy(float),
                                  parent.predicted_price.to_numpy(float)):
                raise RuntimeError(f"R16 parent predictions do not match the G01 comparator in {month}")
            grouped_screen = blend_by_group(
                candidate_source.parent_prediction.to_numpy(float),
                candidate_source.q50_prediction.to_numpy(float),
                homes[GROUP_COLUMN].astype("string").fillna("__missing__").astype(str).to_numpy(),
                r17_selection["weights"], fallback_weight=float(r17_selection["fallback_weight"]),
            )
            predictions = apply_global_shrinkage(
                parent.predicted_price.to_numpy(float), grouped_screen, selection["weight"])
            candidate = pd.DataFrame({"ml_num": expected_ids, "sold_price": truth.sold_price.to_numpy(float),
                                      "predicted_price": predictions})
            screen_rows.append(helpers._candidate_frame(month, TRACK, truth, parent, candidate))
            source_hashes[str(parent_path.relative_to(campaign.ROOT)).replace("\\", "/")] = _sha256(parent_path)
            source_hashes[str(parent_path.with_suffix(".json").relative_to(campaign.ROOT)).replace("\\", "/")] = _sha256(
                parent_path.with_suffix(".json"))
            campaign._append({"event": "screen_fold_complete", "card": CARD, "run_id": run_id,
                              "track": TRACK, "month": month, "parent_id": PARENT,
                              "source_r16_run_id": r16_result["run_id"],
                              "source_r17_run_id": r17_result["run_id"],
                              "global_shrinkage_weight": selection["weight"],
                              "oof_rows": selection["oof_rows"], "oof_months": selection["oof_months"],
                              "date_cutoff_exclusive": cutoff.isoformat()})

        paired = pd.concat(screen_rows, ignore_index=True)
        paired_path = output / "paired_predictions.csv"
        paired.to_csv(paired_path, index=False)
        tracks = {TRACK: helpers._track_result(paired)}
        result = {"campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
                  "card": CARD, "parents": {TRACK: PARENT}, "screen_months": list(SCREEN_MONTHS),
                  "source_r16_run_id": r16_result["run_id"], "source_r17_run_id": r17_result["run_id"],
                  "tracks": tracks, "weight_selections": weight_selections,
                  "fit_starts": 0, "reserved_months_labels_opened": False,
                  "hidden_labels_opened": False, "data_sha256": source["data_sha256"]}
        result_path = output / "r18_screen_results.json"
        _write_json(result_path, result)
        manifest = {
            "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
            "source_identity": source, "source_file_sha256": source_hashes,
            "candidate_predictions_sha256": _sha256(paired_path),
            "result_sha256": _sha256(result_path), "weight_selections": weight_selections,
            "fit_starts": 0, "reserved_months_labels_opened": False,
            "hidden_labels_opened": False,
        }
        _write_json(output / "run_manifest.json", manifest)
        (output / "research_report.md").write_text(_report(result), encoding="utf-8")
        passed = tracks[TRACK]["qualifies_screen"]
        next_action = ("R18 list-aware passed screen; run sequential eight-month confirmation" if passed
                      else "R18 did not qualify; keep G01 and register a distinct next candidate")
        campaign._append({"event": "card_complete", "card": CARD, "run_id": run_id,
                          "fit_starts": 0, "result_path": str(result_path.relative_to(campaign.ROOT)),
                          "qualifies_by_track": {TRACK: passed}, "reserved_months_labels_opened": False,
                          "hidden_labels_opened": False})
        campaign._write_progress("r18_q50_meta_shrink_screen_complete", next_action=next_action,
                                 r18_result=str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
                                 latest_fit_count=0)
        return result
    except Exception as exc:
        failure = {"run_id": run_id, "card": CARD, "error": f"{type(exc).__name__}: {exc}",
                   "source_r16_run_id": r16_result["run_id"], "source_r17_run_id": r17_result["run_id"],
                   "fit_starts": 0, "reserved_months_labels_opened": False, "hidden_labels_opened": False}
        _write_json(output / "failure_checkpoint.json", failure)
        campaign._append({"event": "card_failed", "card": CARD, "run_id": run_id,
                          "error": failure["error"], "fit_starts": 0})
        campaign._write_progress("r18_q50_meta_shrink_failed",
                                 next_action="inspect R18 failure checkpoint before selecting another card",
                                 r18_failure=str((output / "failure_checkpoint.json").relative_to(campaign.ROOT)).replace("\\", "/"))
        raise
