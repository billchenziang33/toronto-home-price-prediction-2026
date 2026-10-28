from __future__ import annotations

import numpy as np
import pandas as pd
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path


REQUIRED_OOF_COLUMNS = {
    "month", "ml_num", "date_sold", "sold_price", "parent_prediction", "q50_prediction",
}


def blend_parent_q50(parent_prediction, q50_prediction, weight: float) -> np.ndarray:
    """Blend whole-dollar parent and Q50 prices, rounding the delivered price."""
    parent = np.asarray(parent_prediction, dtype=float)
    q50 = np.asarray(q50_prediction, dtype=float)
    if parent.ndim != 1 or q50.ndim != 1 or parent.shape != q50.shape:
        raise ValueError("parent and Q50 predictions must be aligned one-dimensional vectors")
    if (not np.isfinite(parent).all() or not np.isfinite(q50).all()
            or (parent <= 0.0).any() or (q50 <= 0.0).any()):
        raise ValueError("parent and Q50 predictions must be finite positive prices")
    if not np.isfinite(weight) or not 0.0 <= float(weight) <= 1.0:
        raise ValueError("Q50 blend weight must be finite and between 0 and 1")
    return np.round(parent * (1.0 - float(weight)) + q50 * float(weight))


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    ape = 100.0 * np.abs(predicted - actual) / actual
    return {
        "mdape_percent": float(np.median(ape)),
        "mape_percent": float(np.mean(ape)),
    }


def _mape_relative_change(parent_mape: float, candidate_mape: float) -> float:
    if parent_mape == 0.0:
        return 0.0 if candidate_mape == 0.0 else float("inf")
    return 100.0 * (candidate_mape - parent_mape) / parent_mape


def select_blend_weight(
    oof: pd.DataFrame,
    cutoff,
    *,
    minimum_rows: int = 1000,
    max_mape_regression_percent: float = 1.0,
) -> dict:
    """Choose a parent/Q50 mix using only strictly earlier OOF rows.

    MdAPE is minimized over a frozen 0.05 grid. Candidate weights violating
    the MAPE regression guard are excluded; ties favor lower MAPE and then the
    smaller Q50 contribution.
    """
    missing = sorted(REQUIRED_OOF_COLUMNS.difference(oof.columns))
    if missing:
        raise ValueError(f"OOF frame is missing columns: {', '.join(missing)}")
    if minimum_rows < 1:
        raise ValueError("minimum_rows must be positive")
    if not np.isfinite(max_mape_regression_percent) or max_mape_regression_percent < 0:
        raise ValueError("max_mape_regression_percent must be finite and nonnegative")

    cutoff_ts = pd.Timestamp(cutoff)
    dates = pd.to_datetime(oof["date_sold"], errors="raise")
    past = oof.loc[dates < cutoff_ts].copy()
    past["date_sold"] = dates.loc[past.index]
    if not past.empty:
        expected_months = past.date_sold.dt.to_period("M").astype(str)
        if not expected_months.eq(past.month.astype(str)).all():
            raise ValueError("OOF month values do not match their sale dates")
        if past.duplicated(["month", "ml_num"]).any():
            raise ValueError("OOF rows must be unique by month and ml_num")
        values = past[["sold_price", "parent_prediction", "q50_prediction"]].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values <= 0.0).any():
            raise ValueError("OOF prices must be finite positive values")

    actual = past.sold_price.to_numpy(dtype=float)
    parent = past.parent_prediction.to_numpy(dtype=float)
    q50 = past.q50_prediction.to_numpy(dtype=float)
    parent_metrics = _metrics(actual, parent) if len(past) else {
        "mdape_percent": float("nan"), "mape_percent": float("nan")
    }
    result = {
        "weight": 0.0,
        "oof_rows": int(len(past)),
        "oof_months": sorted(past.month.astype(str).unique().tolist()),
        "parent": parent_metrics,
        "selected": parent_metrics,
        "selected_mape_relative_change_percent": 0.0,
        "eligible_weights": [0.0],
        "reason": "insufficient_past_oof_rows",
    }
    if len(past) < minimum_rows:
        return result

    eligible = []
    for weight in np.linspace(0.0, 1.0, 21):
        prediction = blend_parent_q50(parent, q50, float(weight))
        metrics = _metrics(actual, prediction)
        mape_change = _mape_relative_change(parent_metrics["mape_percent"], metrics["mape_percent"])
        if mape_change <= max_mape_regression_percent:
            eligible.append({
                "weight": float(round(weight, 10)),
                "metrics": metrics,
                "mape_relative_change_percent": float(mape_change),
            })
    if not eligible:
        return {**result, "reason": "no_weight_passed_mape_guard"}

    selected = min(eligible, key=lambda row: (
        row["metrics"]["mdape_percent"],
        row["metrics"]["mape_percent"],
        row["weight"],
    ))
    return {
        **result,
        "weight": selected["weight"],
        "selected": selected["metrics"],
        "selected_mape_relative_change_percent": selected["mape_relative_change_percent"],
        "eligible_weights": [row["weight"] for row in eligible],
        "reason": "selected_from_strictly_past_oof",
    }


TRACK = "list_aware"
PARENT = "G01"
CARD = "R16_past_oof_q50_shrinkage"
VARIANT = "BILL_R16_quantile_median"
SEED_MONTHS = ("2022-06", "2022-11")
OOF_MONTHS = ("2022-06", "2022-11", "2023-03", "2023-07", "2023-11", "2024-03")
SCREEN_MONTHS = ("2023-03", "2024-06")
PROCESS_MONTHS = ("2022-06", "2022-11", "2023-03", "2023-07", "2023-11", "2024-03", "2024-06")
MINIMUM_OOF_ROWS = 1000
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


def _verified_n03_seed(month: str) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Load the closed N03 G01/HGBR OOF artifacts after per-file verification."""
    from experiments.bill_target74 import runner as campaign
    from experiments.bill_greedy.aligned_inference import apply_prediction_pipeline

    if month not in SEED_MONTHS:
        raise ValueError(f"month is not a registered N03 seed fold: {month!r}")
    run_root = campaign.NEXT_STAGE_ROOT
    manifest_path = run_root / "run_manifest.json"
    audit_path = run_root / "n03_seed_oof_audit.json"
    state_path = run_root / "run_state.json"
    ledger_path = run_root / "fit_ledger.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if (manifest.get("run_id") != run_root.name
            or manifest.get("commit") != campaign.PROTOCOL["parent_commit"]
            or manifest.get("data_sha256") != campaign.PROTOCOL["data_sha256"]
            or manifest.get("reserved_months_labels_opened") is not False
            or state.get("active_fit") is not None
            or state.get("active_stage") != "screen_search_complete_no_qualifier"):
        raise RuntimeError("N03 seed artifacts are not a closed run on the expected commit/data")

    audit_row = audit["tracks"][TRACK][month]
    oof_path = run_root / "predictions/N03_seed_oof" / TRACK / f"{month}.csv"
    if _sha256(oof_path) != audit_row["feature_file_sha256"]:
        raise RuntimeError(f"N03 seed OOF hash mismatch: {oof_path}")
    if audit_row.get("strict_cutoff") != f"{month}-01":
        raise RuntimeError(f"N03 seed cutoff metadata differs for {month}")
    oof = pd.read_csv(oof_path, dtype={"ml_num": str})
    if (oof.ml_num.astype(str).duplicated().any()
            or not oof.month.astype(str).eq(month).all()
            or not np.isfinite(oof.parent_prediction.to_numpy(float)).all()
            or (oof.parent_prediction.to_numpy(float) <= 0.0).any()):
        raise RuntimeError(f"N03 seed OOF rows are invalid for {month}")
    vector_hash = hashlib.sha256(oof.parent_prediction.to_numpy(dtype="float64").tobytes()).hexdigest()
    if vector_hash != audit_row["prediction_vector_sha256"]:
        raise RuntimeError(f"N03 parent prediction vector hash mismatch for {month}")

    ledger_rows = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    components: dict[str, pd.DataFrame] = {}
    inputs = {str(path.relative_to(campaign.ROOT)).replace("\\", "/"): _sha256(path)
              for path in (manifest_path, audit_path, state_path, ledger_path, oof_path)}
    for role in ("g01", "hgbr"):
        fit_id = f"N03-seed:{role}:{TRACK}:{month}"
        fit = next((row for row in audit["component_fits"] if row.get("fit_id") == fit_id), None)
        csv_path = run_root / "predictions/N03_components" / TRACK / f"{month}_{role}.csv"
        meta_path = csv_path.with_suffix(".json")
        ledger_fit = next((row for row in ledger_rows
                           if row.get("event") == "fit_complete" and row.get("fit_id") == fit_id
                           and row.get("status") == "complete"), None)
        ledger_start = next((row for row in ledger_rows
                             if row.get("event") == "fit_start" and row.get("fit_id") == fit_id), None)
        if (fit is None or ledger_fit is None or _sha256(csv_path) != fit.get("prediction_sha256")
                or _sha256(meta_path) != fit.get("metadata_sha256")
                or _sha256(csv_path) != ledger_fit.get("prediction_file_sha256")
                or ledger_start is None
                or ledger_start.get("data_sha256") != campaign.PROTOCOL["data_sha256"]):
            raise RuntimeError(f"N03 component file/metadata/ledger hash mismatch: {fit_id}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if (meta.get("fit_id") != fit_id or meta.get("track") != TRACK
                or meta.get("month") != month):
            raise RuntimeError(f"N03 component metadata identity mismatch: {fit_id}")
        components[role] = pd.read_csv(csv_path, dtype={"ml_num": str})
        inputs[str(csv_path.relative_to(campaign.ROOT)).replace("\\", "/")] = _sha256(csv_path)
        inputs[str(meta_path.relative_to(campaign.ROOT)).replace("\\", "/")] = _sha256(meta_path)

    ids = oof.ml_num.astype(str).to_numpy()
    for role, frame in components.items():
        if not np.array_equal(frame.ml_num.astype(str).to_numpy(), ids):
            raise RuntimeError(f"N03 {role} component IDs do not align for {month}")
    reconstructed = apply_prediction_pipeline(
        TRACK, components["g01"].raw_prediction.to_numpy(float),
        components["hgbr"].raw_prediction.to_numpy(float), 0.75,
    )
    if not np.array_equal(reconstructed, oof.parent_prediction.to_numpy(float)):
        raise RuntimeError(f"N03 saved parent OOF does not reconstruct from G01/HGBR for {month}")
    observed_max_date = pd.to_datetime(oof.date_sold, errors="raise").max().date().isoformat()
    if observed_max_date != audit_row["max_date_sold"]:
        raise RuntimeError(f"N03 seed maximum date differs from its audit for {month}")
    oof["date_sold"] = pd.to_datetime(oof.date_sold, errors="raise")
    oof["parent_prediction"] = reconstructed
    frozen = components["hgbr"].raw_prediction.to_numpy(dtype=float)
    return oof, frozen, {"inputs": inputs, "parent_prediction_vector_sha256": vector_hash}


def _parent_and_frozen(greedy, helpers, budget, month: str, train: pd.DataFrame,
                       homes: pd.DataFrame, truth: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, dict]:
    from experiments.bill_target74 import runner as campaign

    if month in SEED_MONTHS:
        seed, frozen, provenance = _verified_n03_seed(month)
        expected_ids = homes.ml_num.astype(str).to_numpy()
        if (not np.array_equal(seed.ml_num.astype(str).to_numpy(), expected_ids)
                or not np.array_equal(seed.sold_price.to_numpy(float), truth.sold_price.to_numpy(float))):
            raise RuntimeError(f"N03 parent OOF labels/IDs do not align with current fold {month}")
        parent = pd.DataFrame({"ml_num": expected_ids,
                               "sold_price": seed.sold_price.to_numpy(float),
                               "predicted_price": seed.parent_prediction.to_numpy(float)})
        return parent, frozen, provenance

    parent, parent_path = budget._check_parent_artifact(greedy, month, TRACK, homes, truth, PARENT)
    frozen = helpers._verified_hgbr_cache(greedy, month, TRACK, train, homes)
    frozen_path = greedy.OUTPUT / "frozen_hgbr" / TRACK / f"{month}.csv"
    sidecar_path = parent_path.with_suffix(".json")
    provenance = {
        "inputs": {
            str(parent_path.relative_to(campaign.ROOT)).replace("\\", "/"): helpers._sha256(parent_path),
            str(sidecar_path.relative_to(campaign.ROOT)).replace("\\", "/"): helpers._sha256(sidecar_path),
            str(frozen_path.relative_to(campaign.ROOT)).replace("\\", "/"): helpers._sha256(frozen_path),
        },
        "historical_parent_is_frozen_walk_forward_comparator": True,
    }
    return parent, frozen, provenance


def r16_fit_id(run_id: str, month: str) -> str:
    if not run_id or any(char in run_id for char in "\\/"):
        raise ValueError("run_id must be path-safe")
    if month not in PROCESS_MONTHS:
        raise ValueError(f"month is not registered for R16: {month!r}")
    return f"{CARD}:{TRACK}:{month}:core:{run_id}"


def _report(result: dict) -> str:
    score = result["tracks"][TRACK]
    lines = [
        "# R16：严格过去 OOF 选择 Q50 与 G01 的收缩混合", "",
        f"Campaign：`{result['campaign_id']}`；run：`{result['run_id']}`。",
        "候选把当前 G01 父预测与 R08 相同的 Q50 LightGBM core + 冻结 HGBR 25% 完整预测按价格空间混合。每个筛选折单独选权重；只使用销售日期严格早于该折 cutoff 的 OOF 数据。",
        "权重在 0.00 至 1.00 的 0.05 网格上使 OOF MdAPE 最小，同时要求 OOF MAPE 相对父模型恶化不超过 1.00%；并列时优先 MAPE 较低、再优先更小权重。历史筛选折不足 1,000 行时权重固定为 0。", "",
        "## 按筛选 cutoff 选择的权重", "",
        "| 外层月份 | 权重 | 严格过去 OOF 月份 | OOF 行数 | 父→选中 MdAPE | MAPE相对变化 | 原因 |",
        "|---|---:|---|---:|---:|---:|---|",
    ]
    for row in result["weight_selections"]:
        lines.append(
            f"| {row['screen_month']} | {row['weight']:.2f} | {', '.join(row['oof_months']) or '无'} | "
            f"{row['oof_rows']} | {row['parent'].get('mdape_percent', float('nan')):.5f}% → "
            f"{row['selected'].get('mdape_percent', float('nan')):.5f}% | "
            f"{row['selected_mape_relative_change_percent']:+.3f}% | {row['reason']} |"
        )
    lines.extend(["", "## 双月筛选结果", "",
                  "| 赛道 | 父→候选 MdAPE | 相对改善 | MAPE相对变化 | 父→候选 P90 APE | APE>20%变化 | 通过筛选 |",
                  "|---|---:|---:|---:|---:|---:|---|"])
    lines.append(
        f"| {TRACK} | {score['parent']['mdape_percent']:.5f}% → {score['candidate']['mdape_percent']:.5f}% | "
        f"{score['relative_mdape_improvement_percent']:+.3f}% | "
        f"{score['mape_relative_change_percent']:+.3f}% | "
        f"{score['parent']['p90_ape_percent']:.3f}% → {score['candidate']['p90_ape_percent']:.3f}% | "
        f"{score['parent']['ape_over_20_percent']:.3f}% → {score['candidate']['ape_over_20_percent']:.3f}% | "
        f"{'是' if score['qualifies_screen'] else '否'} |"
    )
    lines.extend(["", "## 逐月分数", "", "| 月份 | 父 MdAPE | 候选 MdAPE | 相对改善 | 父 MAPE | 候选 MAPE |",
                  "|---|---:|---:|---:|---:|---:|"])
    for month, monthly in score["monthly"].items():
        lines.append(
            f"| {month} | {monthly['parent']['mdape_percent']:.5f}% | "
            f"{monthly['candidate']['mdape_percent']:.5f}% | "
            f"{monthly['relative_mdape_improvement_percent']:+.3f}% | "
            f"{monthly['parent']['mape_percent']:.5f}% | {monthly['candidate']['mape_percent']:.5f}% |"
        )
    lines.extend(["", "## 运行边界", "",
                  f"实际 learner fit starts：{result['fit_starts']}。Q50 core 的严格时间 OOF 月份：{', '.join(OOF_MONTHS)}。",
                  "筛选门槛：pooled MdAPE 至少改善 0.25%、两个筛选月分别改善、MAPE 相对恶化不超过 1%。未通过则不做八个月确认、不更新父模型。",
                  "2022-06/11 的父与 HGBR 预测逐文件核验了 N03 关闭运行中的 OOF 审计、组件元数据与 fit ledger。后续父预测沿用已保存的 walk-forward 比较器；其旧 sidecar 代码哈希差异按历史冻结来源记录。",
                  "筛选仅使用 2023-03 与 2024-06。2025-05/08 保留月份和隐藏标签未读取。", "",
                  "- OOF 配对预测与 Q50 输出：`oof_predictions.csv`",
                  "- 筛选配对预测：`paired_predictions.csv`",
                  "- 权重与逐月结果：`r16_screen_results.json`",
                  "- 来源、组件参数、文件哈希与 fit 账本摘要：`run_manifest.json` / campaign `fits.jsonl`", ""])
    return "\n".join(lines)


def run_r16_screen() -> dict:
    from data import hide_answers
    from experiments.bill_target74 import r03_catboost as helpers
    from experiments.bill_target74 import r04_ablation as budget
    from experiments.bill_target74 import runner as campaign
    from experiments.bill_target74 import r08_quantile_median as r08
    from experiments.bill_greedy import runner as greedy
    from experiments.model_optimization_v2.folds import build_month_fold
    from experiments.model_optimization_v2.pipeline import blend_prices

    ROOT = campaign.ROOT
    source = budget._source_identity()
    if campaign.PROTOCOL["cards"].get(CARD, {}).get("status") != "registered":
        raise RuntimeError("R16 must be registered in campaign_protocol.json before fitting")
    run_started = datetime.now(timezone.utc)
    run_id = "bill_target74_r16_q50_oof_blend_" + run_started.strftime("%Y%m%dT%H%M%SZ")
    output = campaign.CAMPAIGN_ROOT / "batches" / run_id
    output.mkdir(parents=True, exist_ok=False)
    source_files = [campaign.PROTOCOL_PATH, campaign.PACKAGE / "r16_q50_oof_blend.py",
                    campaign.PACKAGE / "r03_catboost.py", campaign.PACKAGE / "r04_ablation.py",
                    campaign.PACKAGE / "r08_quantile_median.py", ROOT / "experiments/bill_greedy/runner.py",
                    ROOT / "experiments/bill_greedy/history.py", ROOT / "experiments/bill_greedy/aligned_inference.py",
                    ROOT / "experiments/model_optimization_v2/learners.py",
                    ROOT / "experiments/model_optimization_v2/pipeline.py",
                    ROOT / "model_candidate_v2/features.py", ROOT / "model_candidate_v2/winner_config.json"]
    campaign._append({"event": "card_start", "card": CARD, "run_id": run_id,
                      "screen_months": list(SCREEN_MONTHS), "oof_months": list(OOF_MONTHS),
                      "expected_fit_starts": len(PROCESS_MONTHS), "parent": PARENT,
                      "objective": "quantile", "alpha": 0.5,
                      "weight_grid": [round(value, 2) for value in np.linspace(0, 1, 21)],
                      "strict_past_only_weight_selection": True,
                      "reserved_months_labels_opened": False, "hidden_labels_opened": False})

    old_fit_record = greedy._fit_record
    old_hgbr = greedy._frozen_hgbr_predictions
    old_variant = greedy.VARIANTS.get(VARIANT)
    greedy.VARIANTS[VARIANT] = r08.quantile_variant_spec()
    fit_context: dict[str, str] = {}
    fit_start_ids: list[str] = []
    fit_details: list[dict] = []
    used_input_hashes: dict[str, str] = {}
    oof_rows: list[pd.DataFrame] = []
    screen_rows: list[pd.DataFrame] = []
    weight_selections: list[dict] = []

    def record_fit(parent_fit_id: str, fit_fn):
        month = fit_context["month"]
        fit_id = r16_fit_id(run_id, month)
        fit_start_ids.append(fit_id)
        campaign._append({"event": "fit_start", "card": CARD, "run_id": run_id,
                          "fit_id": fit_id, "parent_fit_id": parent_fit_id,
                          "track": TRACK, "month": month, "component": "core",
                          "objective": "quantile", "alpha": 0.5,
                          "data_sha256": source["data_sha256"]})
        started = time.perf_counter()
        try:
            fitted = fit_fn()
        except Exception as exc:
            campaign._append({"event": "fit_complete", "card": CARD, "run_id": run_id,
                              "fit_id": fit_id, "status": "failed", "seconds": time.perf_counter() - started,
                              "error": f"{type(exc).__name__}: {exc}"})
            raise
        detail = {"fit_id": fit_id, "parent_fit_id": parent_fit_id, "track": TRACK,
                  "month": month, "component": "core", "seconds": time.perf_counter() - started,
                  "status": "complete", "objective": fitted["actual_objective"],
                  "model_family": fitted["model_family"], "model_config": fitted["model_config"],
                  "target": fitted["target"], "training_rows": fitted["training_rows"],
                  "evaluation_rows": fitted["evaluation_rows"],
                  "feature_columns": fitted["feature_columns"],
                  "prediction_sha256": fitted["prediction_sha256"]}
        fit_details.append(detail)
        campaign._append({"event": "fit_complete", "card": CARD, "run_id": run_id, **detail})
        if fitted["actual_objective"] != "quantile":
            raise RuntimeError(f"R16 expected quantile objective but received {fitted['actual_objective']!r}")
        return fitted

    greedy._fit_record = record_fit
    greedy._frozen_hgbr_predictions = lambda month, track, train, homes: helpers._verified_hgbr_cache(
        greedy, month, track, train, homes)
    try:
        for month in PROCESS_MONTHS:
            listings, _, _, _, _ = greedy._load_fold(month)
            train, homes, truth = build_month_fold(listings, month, TRACK)
            cutoff = pd.Period(month, freq="M").start_time
            if (pd.to_datetime(listings.date_sold, errors="raise").max() >=
                    (pd.Period(month, freq="M") + 1).start_time
                    or (pd.to_datetime(train.date_sold, errors="raise") >= cutoff).any()):
                raise RuntimeError(f"R16 fold is not strictly as-of for {month}")
            heldout = listings.loc[pd.to_datetime(listings.date_sold).dt.to_period("M").astype(str).eq(month)].copy()
            heldout["ml_num"] = heldout.ml_num.astype(str)
            heldout = heldout.set_index("ml_num").loc[homes.ml_num.astype(str)].reset_index()
            if (not np.array_equal(heldout.ml_num.astype(str).to_numpy(), truth.ml_num.astype(str).to_numpy())
                    or not np.array_equal(heldout.sold_price.to_numpy(float), truth.sold_price.to_numpy(float))):
                raise RuntimeError(f"R16 heldout rows and truth do not align for {month}")

            parent, frozen_hgbr, provenance = _parent_and_frozen(
                greedy, helpers, budget, month, train, homes, truth)
            used_input_hashes.update(provenance["inputs"])
            if month in SCREEN_MONTHS:
                historical = pd.concat(oof_rows, ignore_index=True) if oof_rows else pd.DataFrame(
                    columns=sorted(REQUIRED_OOF_COLUMNS))
                selection = select_blend_weight(
                    historical, cutoff, minimum_rows=MINIMUM_OOF_ROWS,
                    max_mape_regression_percent=MAX_MAPE_REGRESSION_PERCENT)
                weight_selections.append({"screen_month": month, **selection})

            fit_context.clear()
            fit_context.update({"month": month})
            q50 = greedy._fit_history_candidate(
                month, TRACK, train, homes, variant=VARIANT, fit_tag=f"R16-q50-{month}")
            q50_prediction = r08.list_aware_q50_prediction(q50["predictions"], frozen_hgbr)
            q50_frame = pd.DataFrame({
                "month": month, "ml_num": homes.ml_num.astype(str).to_numpy(),
                "date_sold": pd.to_datetime(heldout.date_sold, errors="raise").to_numpy(),
                "sold_price": truth.sold_price.to_numpy(float),
                "parent_prediction": parent.predicted_price.to_numpy(float),
                "q50_prediction": q50_prediction,
            })
            if not np.isfinite(q50_frame[["sold_price", "parent_prediction", "q50_prediction"]].to_numpy()).all():
                raise RuntimeError(f"R16 produced non-finite OOF/screen predictions for {month}")
            oof_rows.append(q50_frame)
            oof_checkpoint = pd.concat(oof_rows, ignore_index=True)
            oof_checkpoint.to_csv(output / "oof_predictions.partial.csv", index=False)
            if month in SCREEN_MONTHS:
                selected_weight = weight_selections[-1]["weight"]
                candidate_prediction = blend_parent_q50(
                    parent.predicted_price.to_numpy(float), q50_prediction, selected_weight)
                candidate = pd.DataFrame({"ml_num": homes.ml_num.astype(str).to_numpy(),
                                          "sold_price": truth.sold_price.to_numpy(float),
                                          "predicted_price": candidate_prediction})
                screen_rows.append(helpers._candidate_frame(month, TRACK, truth, parent, candidate))
                paired_checkpoint = pd.concat(screen_rows, ignore_index=True)
                paired_checkpoint.to_csv(output / "paired_predictions.partial.csv", index=False)
                _write_json(output / "weights.partial.json", {"weight_selections": weight_selections})
                campaign._append({"event": "screen_fold_complete", "card": CARD, "run_id": run_id,
                                  "track": TRACK, "month": month, "parent_id": PARENT,
                                  "q50_fit_id": r16_fit_id(run_id, month), "q50_objective": q50["actual_objective"],
                                  "weight": selected_weight,
                                  "oof_rows": selection["oof_rows"], "oof_months": selection["oof_months"],
                                  "parent_csv_sha256": used_input_hashes.get(
                                      str(campaign._row_prediction_path(month, TRACK).relative_to(ROOT)).replace("\\", "/")),
                                  "date_cutoff_exclusive": cutoff.isoformat()})

        paired_path = output / "paired_predictions.csv"
        paired = pd.concat(screen_rows, ignore_index=True)
        paired.to_csv(paired_path, index=False)
        oof_path = output / "oof_predictions.csv"
        pd.concat(oof_rows, ignore_index=True).to_csv(oof_path, index=False)
        tracks = {TRACK: helpers._track_result(paired)}
        result = {"campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
                  "card": CARD, "parents": {TRACK: PARENT}, "screen_months": list(SCREEN_MONTHS),
                  "tracks": tracks, "weight_selections": weight_selections,
                  "fit_starts": len(fit_start_ids), "fit_details": fit_details,
                  "oof_months": list(OOF_MONTHS), "minimum_oof_rows": MINIMUM_OOF_ROWS,
                  "max_mape_regression_percent": MAX_MAPE_REGRESSION_PERCENT,
                  "reserved_months_labels_opened": False, "hidden_labels_opened": False,
                  "data_sha256": source["data_sha256"]}
        result_path = output / "r16_screen_results.json"
        _write_json(result_path, result)
        manifest = {
            "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
            "source_identity": source,
            "code_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): _sha256(path)
                            for path in source_files},
            "n03_seed_provenance": {month: _verified_n03_seed(month)[2]
                                    for month in SEED_MONTHS},
            "input_sha256": used_input_hashes,
            "model_config": q50["model_config"], "target": q50["target"],
            "fit_details": fit_details,
            "oof_prediction_sha256": _sha256(oof_path),
            "paired_prediction_sha256": _sha256(paired_path),
            "result_sha256": _sha256(result_path),
            "runtime": {"python": platform.python_version(), "lightgbm": metadata.version("lightgbm"),
                        "numpy": np.__version__, "pandas": pd.__version__},
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        }
        _write_json(output / "run_manifest.json", manifest)
        (output / "research_report.md").write_text(_report(result), encoding="utf-8")
        passed = tracks[TRACK]["qualifies_screen"]
        next_action = ("R16 list-aware passed screen; run strict sequential eight-month confirmation" if passed
                      else "R16 did not qualify; freeze current G01 and register a distinct next candidate")
        campaign._append({"event": "card_complete", "card": CARD, "run_id": run_id,
                          "fit_starts": len(fit_start_ids), "result_path": str(result_path.relative_to(ROOT)),
                          "qualifies_by_track": {TRACK: passed}, "reserved_months_labels_opened": False,
                          "hidden_labels_opened": False})
        campaign._write_progress("r16_q50_oof_blend_screen_complete", next_action=next_action,
                                 r16_result=str(result_path.relative_to(ROOT)).replace("\\", "/"),
                                 latest_fit_count=len(fit_details))
        for partial in (output / "oof_predictions.partial.csv", output / "paired_predictions.partial.csv",
                        output / "weights.partial.json"):
            if partial.exists():
                partial.unlink()
        return result
    except Exception as exc:
        failure = {"run_id": run_id, "card": CARD, "error": f"{type(exc).__name__}: {exc}",
                   "fit_starts": len(fit_start_ids), "fit_start_ids": fit_start_ids,
                   "fit_details": fit_details,
                   "reserved_months_labels_opened": False, "hidden_labels_opened": False}
        _write_json(output / "failure_checkpoint.json", failure)
        campaign._append({"event": "card_failed", "card": CARD, "run_id": run_id,
                          "error": failure["error"], "fit_starts": len(fit_start_ids)})
        campaign._write_progress("r16_q50_oof_blend_failed",
                                 next_action="inspect R16 failure checkpoint and fit ledger before deciding whether to resume",
                                 r16_failure=str((output / "failure_checkpoint.json").relative_to(ROOT)).replace("\\", "/"))
        raise
    finally:
        greedy._fit_record = old_fit_record
        greedy._frozen_hgbr_predictions = old_hgbr
        if old_variant is None:
            greedy.VARIANTS.pop(VARIANT, None)
        else:
            greedy.VARIANTS[VARIANT] = old_variant
