from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from experiments.bill_target74 import r03_catboost as helpers
from experiments.bill_target74 import runner as campaign

ROOT = campaign.ROOT
SCREEN_MONTHS = tuple(campaign.PROTOCOL["screen_months"])
TRACKS = tuple(campaign.PROTOCOL["parent_by_track"])


def _feature_frames(month: str, track: str, train: pd.DataFrame, heldout: pd.DataFrame):
    from data import hide_answers
    from experiments.bill_greedy import runner as greedy
    from experiments.bill_greedy.history import building_history_features
    from model_candidate_v2 import features

    homes = hide_answers(heldout, track)
    cutoff = pd.Period(month, freq="M").start_time
    if pd.to_datetime(train.date_sold, errors="raise").max() >= cutoff:
        raise RuntimeError(f"training rows cross prediction cutoff in {month}")
    base = "G06_lot_garage" if track == "off_market" else None
    options, include_comparables = greedy._history_feature_options(base_variant=base, variant=None)
    if include_comparables:
        raise RuntimeError("baseline parent unexpectedly requests neighbor-comparable features")
    builder = features.off_market_features if track == "off_market" else features.list_aware_features
    train_x = pd.concat([builder(train), building_history_features(
        train, train, track=track, training_rows=True, **options)], axis=1)
    eval_x = pd.concat([builder(homes), building_history_features(
        train, homes, track=track, cutoff=cutoff, **options)], axis=1)
    if list(train_x.columns) != list(eval_x.columns):
        raise RuntimeError(f"training and evaluation feature schemas differ in {month}/{track}")
    return train_x, eval_x, homes


def _profile_features(train_x: pd.DataFrame, eval_x: pd.DataFrame) -> dict:
    changes = []
    for column in train_x.columns:
        train = train_x[column]
        evaluate = eval_x[column]
        train_missing = float(train.isna().mean())
        eval_missing = float(evaluate.isna().mean())
        if pd.api.types.is_numeric_dtype(train.dtype):
            train_values = pd.to_numeric(train, errors="coerce").to_numpy(dtype=float)
            eval_values = pd.to_numeric(evaluate, errors="coerce").to_numpy(dtype=float)
            train_nonfinite = int((~np.isfinite(train_values) & ~np.isnan(train_values)).sum())
            eval_nonfinite = int((~np.isfinite(eval_values) & ~np.isnan(eval_values)).sum())
            item = {"feature": str(column), "kind": "numeric", "train_missing_rate": train_missing,
                    "eval_missing_rate": eval_missing, "missing_rate_delta": eval_missing - train_missing,
                    "train_nonfinite": train_nonfinite, "eval_nonfinite": eval_nonfinite}
        else:
            train_levels = set(train.dropna().astype(str).unique())
            eval_nonnull = evaluate.dropna().astype(str)
            unseen = ~eval_nonnull.isin(train_levels)
            item = {"feature": str(column), "kind": "categorical", "train_missing_rate": train_missing,
                    "eval_missing_rate": eval_missing, "missing_rate_delta": eval_missing - train_missing,
                    "eval_unseen_level_rate": float(unseen.mean()) if len(eval_nonnull) else 0.0,
                    "eval_unseen_levels": sorted(set(eval_nonnull.loc[unseen].unique()))[:15]}
        if abs(item["missing_rate_delta"]) >= 0.05 or item.get("eval_unseen_level_rate", 0) >= 0.01:
            changes.append(item)
    changes.sort(key=lambda item: max(abs(item["missing_rate_delta"]),
                                      item.get("eval_unseen_level_rate", 0)), reverse=True)
    history = {}
    for column in ("b_n", "b_res_n", "b_res_mean", "b_age",
                   "u_n", "u_res_n", "u_res_mean", "u_last_res", "u_age"):
        if column not in eval_x:
            continue
        values = pd.to_numeric(eval_x[column], errors="coerce")
        history[column] = {"missing_rate": float(values.isna().mean()),
                           "zero_rate": float(values.eq(0).mean()),
                           "median": float(values.median()) if values.notna().any() else None}
    return {"feature_count": int(len(train_x.columns)), "features_with_material_missing_or_category_shift": changes[:20],
            "history_coverage": history}


def run_audit() -> dict:
    from experiments.bill_greedy import runner as greedy

    source_hash = helpers._sha256(campaign.DATA)
    if source_hash != campaign.PROTOCOL["data_sha256"]:
        raise RuntimeError("campaign data hash differs from the frozen protocol")
    audit_id = "bill_target74_r10_feature_quality_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = campaign.CAMPAIGN_ROOT / "batches" / audit_id
    out.mkdir(parents=True, exist_ok=False)
    campaign._append({"event": "card_start", "card": "R10_feature_quality_and_time_drift_audit",
                      "run_id": audit_id, "screen_months": list(SCREEN_MONTHS), "fit_start_count": 0,
                      "data_sha256": source_hash, "read_only": True})
    results = {"campaign_id": campaign.PROTOCOL["campaign_id"], "audit_id": audit_id,
               "data_sha256": source_hash, "reserved_months_labels_opened": False,
               "hidden_labels_opened": False, "tracks": {}}
    for track in TRACKS:
        results["tracks"][track] = {}
        for month in SCREEN_MONTHS:
            listings, train, _, _, heldout = greedy._load_fold(month)
            train_x, eval_x, _ = _feature_frames(month, track, train, heldout)
            if listings.ml_num.astype(str).duplicated().any():
                duplicate_ids = int(listings.ml_num.astype(str).duplicated(keep=False).sum())
            else:
                duplicate_ids = 0
            train_price = pd.to_numeric(train.sold_price, errors="coerce")
            train_area = pd.to_numeric(train_x.get("square_feet"), errors="coerce") if "square_feet" in train_x else None
            results["tracks"][track][month] = {
                "train_rows": int(len(train)), "evaluation_rows": int(len(eval_x)),
                "duplicate_ids_in_loaded_fold": duplicate_ids,
                "train_invalid_or_nonpositive_sale_price": int((~np.isfinite(train_price) | (train_price <= 0)).sum()),
                "feature_profile": _profile_features(train_x, eval_x),
                "train_date_min": str(pd.to_datetime(train.date_sold).min()),
                "train_date_max": str(pd.to_datetime(train.date_sold).max()),
                "prediction_cutoff": pd.Period(month, freq="M").start_time.isoformat(),
            }
            if train_area is not None:
                results["tracks"][track][month]["train_nonpositive_area_features"] = int(
                    (train_area.notna() & (train_area <= 0)).sum())
    results_path = out / "r10_feature_quality_audit.json"
    results_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    lines = ["# R10: 特征质量与时间漂移只读审计", "",
             f"Campaign：`{results['campaign_id']}`；审计：`{audit_id}`。",
             f"数据 SHA-256：`{source_hash}`。只检查固定筛选月及其训练窗口；不启动 learner fits，不读取预留月或隐藏标签。",
             "训练/验证特征由当前 parent 的特征函数生成；unit/building history 对训练行按各自月份起点截断，对筛选月按该月起点截断。", ""]
    for track, months in results["tracks"].items():
        lines.extend([f"## {track}", ""])
        for month, detail in months.items():
            lines.extend([f"### {month}", "",
                          f"训练行 {detail['train_rows']}，筛选行 {detail['evaluation_rows']}，加载数据中的重复 ID 行 {detail['duplicate_ids_in_loaded_fold']}；训练售价无效/非正行 {detail['train_invalid_or_nonpositive_sale_price']}。",
                          f"特征列 {detail['feature_profile']['feature_count']}；预测 cutoff `{detail['prediction_cutoff']}`。"])
            history = detail["feature_profile"]["history_coverage"]
            if history:
                lines.extend(["", "历史特征覆盖率（筛选月）：", "", "| 特征 | 缺失率 | 零值率 | 中位数 |", "|---|---:|---:|---:|"])
                for column, value in history.items():
                    median = "—" if value["median"] is None else f"{value['median']:.4g}"
                    lines.append(f"| `{column}` | {value['missing_rate']:.2%} | {value['zero_rate']:.2%} | {median} |")
            shifts = detail["feature_profile"]["features_with_material_missing_or_category_shift"]
            if shifts:
                lines.extend(["", "缺失率变化 ≥5 个百分点或验证类别新值占比 ≥1% 的特征：", "",
                              "| 特征 | 类型 | 训练缺失 | 验证缺失 | 验证-训练 | 新类别占比 |", "|---|---|---:|---:|---:|---:|"])
                for item in shifts:
                    unseen = item.get("eval_unseen_level_rate", 0.0)
                    lines.append(f"| `{item['feature']}` | {item['kind']} | {item['train_missing_rate']:.2%} | "
                                 f"{item['eval_missing_rate']:.2%} | {item['missing_rate_delta']:+.2%} | {unseen:.2%} |")
            else:
                lines.extend(["", "没有发现达到预设阈值的缺失率或类别新值变化。"])
            lines.append("")
    lines.extend(["## 解读边界", "",
                  "这些是输入覆盖率与时间漂移诊断，不证明任何字段造成了模型误差，也不是新的模型分数。重复 ID 行按折加载数据统计，须结合数据粒度解释；历史父预测与当前源代码存在已记录的 sidecar hash 差异。",
                  "", f"完整机器可读结果：`{results_path.relative_to(ROOT).as_posix()}`。", ""])
    report_path = out / "research_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    campaign._append({"event": "card_complete", "card": "R10_feature_quality_and_time_drift_audit",
                      "run_id": audit_id, "fit_starts": 0, "read_only": True,
                      "result_path": str(report_path.relative_to(ROOT)).replace("\\", "/")})
    campaign._write_progress("r10_feature_quality_audit_complete",
                             next_action="use verified feature-quality findings to register one distinct, leakage-safe model experiment",
                             r10_audit=str(report_path.relative_to(ROOT)).replace("\\", "/"))
    return {"audit_id": audit_id, "report_path": report_path, "results_path": results_path, "results": results}


def main() -> None:
    result = run_audit()
    print(f"report={result['report_path']}", flush=True)


if __name__ == "__main__":
    main()
