"""R52: strict-past adaptive blend of the R48 and R50 off-market candidates."""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.bill_target74.r19_expert_weight import _sha256, _write_json


CARD = "R52_r48_r50_strict_past_meta_blend"
TRACK = "off_market"
PARENT_RUN = "bill_target74_r48_20261025T071216Z"
VARIANT_RUN = "bill_target74_r50_20261025T072426Z"
MONTHS = ("2023-03", "2023-07", "2023-11", "2024-03",
          "2024-06", "2024-09", "2025-01", "2025-03")
MIN_PAST_ROWS = 1000
MAX_MAPE_REGRESSION_PERCENT = 1.0
SHARES = tuple(round(value, 2) for value in np.arange(0.0, 1.0001, 0.05))


def register_card() -> dict:
    from experiments.bill_target74 import r04_ablation as budget
    from experiments.bill_target74 import runner as campaign

    existing = campaign.PROTOCOL["cards"].get(CARD)
    if existing:
        if existing.get("status") == "registered":
            return existing
        raise RuntimeError(f"R52 is already {existing.get('status')}; inspect before rerunning")
    source = budget._source_identity()
    progress = json.loads(campaign.STATE.read_text(encoding="utf-8"))
    if not all(progress.get(key) for key in ("r48_result", "r50_result")):
        raise RuntimeError("R52 requires the registered R48 and R50 development results")
    registration = {
        "status": "registered",
        "track": TRACK,
        "parent": "R48 house-type past-OOF blend",
        "parent_run_id": PARENT_RUN,
        "variant_candidate": "R50 type-by-parent-price-tercile past-OOF blend",
        "variant_run_id": VARIANT_RUN,
        "method": (
            "Reuse the hash-verified R48 and R50 eight-month off-market predictions. For each "
            "target month, select the R50 price-space share on the fixed 0.00..1.00 grid by "
            "0.05 using only strictly earlier development-month rows; require at least 1,000 "
            "past rows and no more than 1% past-only MAPE regression versus R48. If no weight "
            "is eligible, use R48. Ties favor lower MdAPE, lower MAPE, then the smaller R50 share."
        ),
        "difference_from_prior_work": (
            "R48 chose R32 shares by house type and R50 by house type × predicted-price tercile. "
            "R52 tests a separately registered outer past-only blend between their saved "
            "same-row predictions; it is neither a refit nor a retuning of either card."
        ),
        "development_months": list(MONTHS),
        "expected_fit_starts": 0,
        "minimum_past_rows": MIN_PAST_ROWS,
        "r50_share_grid": list(SHARES),
        "past_mape_regression_max_percent": MAX_MAPE_REGRESSION_PERCENT,
        "development_gate": {
            "minimum_relative_mdape_improvement_percent": 0.5,
            "minimum_months_improved": 6,
            "maximum_worst_month_regression_percent": 5.0,
            "maximum_mape_regression_percent": 1.0,
            "paired_bootstrap_upper95_below_zero": True,
            "strict_target_mdape_percent": 7.0,
        },
        "policy": (
            "Register the fixed method before opening either candidate's row-level paired results. "
            "For each month use only predictions and labels from earlier development months to "
            "choose the share; never use the current month's labels in its coefficient. Verify "
            "source result, paired-prediction and manifest hashes and row/label alignment. Do not "
            "read 2025-05/08 or hidden labels. This adaptive development score is not an independent test."
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
        "parent_run_id": PARENT_RUN, "variant_run_id": VARIANT_RUN,
        "development_months": list(MONTHS), "expected_fit_starts": 0,
        "minimum_past_rows": MIN_PAST_ROWS, "r50_share_grid": list(SHARES),
        "data_sha256": source["data_sha256"], "source_sha256": registration["source_sha256"],
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    campaign._write_progress(
        "r52_registered",
        next_action="verify R48/R50 artifacts and run the preregistered strict-past adaptive blend",
        r52_card=CARD,
    )
    return registration


def _verified_result(campaign, relative: str, expected_run: str, label: str):
    result_path = campaign.ROOT / relative
    paired_path = result_path.parent / "paired_predictions.csv"
    manifest_path = result_path.parent / "run_manifest.json"
    if not all(path.is_file() for path in (result_path, paired_path, manifest_path)):
        raise FileNotFoundError(f"{label} result, paired predictions, or manifest is missing")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (result.get("run_id") != expected_run or manifest.get("run_id") != expected_run
            or result.get("campaign_id") != campaign.PROTOCOL["campaign_id"]
            or manifest.get("campaign_id") != campaign.PROTOCOL["campaign_id"]):
        raise RuntimeError(f"{label} artifacts do not identify the preregistered run")
    manifest_data_sha = manifest.get("data_sha256") or manifest.get("source_identity", {}).get("data_sha256")
    if (manifest.get("result_sha256") != _sha256(result_path)
            or manifest.get("paired_predictions_sha256") != _sha256(paired_path)
            or manifest_data_sha != campaign.PROTOCOL["data_sha256"]):
        raise RuntimeError(f"{label} artifacts fail result, paired-row, or data hash verification")
    if result.get("reserved_months_labels_opened") or result.get("hidden_labels_opened"):
        raise RuntimeError(f"{label} run records restricted-label access")
    return result, result_path, paired_path, manifest


def _select_share(prior: pd.DataFrame) -> tuple[float, dict]:
    parent = prior.r48_prediction.to_numpy(dtype=float)
    variant = prior.r50_prediction.to_numpy(dtype=float)
    actual = prior.sold_price.to_numpy(dtype=float)
    parent_ape = 100.0 * np.abs(parent - actual) / actual
    parent_mdape = float(np.median(parent_ape))
    parent_mape = float(np.mean(parent_ape))
    eligible = []
    for share in SHARES:
        prediction = np.round((1.0 - share) * parent + share * variant)
        ape = 100.0 * np.abs(prediction - actual) / actual
        mape = float(np.mean(ape))
        mape_change = 100.0 * (mape - parent_mape) / parent_mape
        if mape_change <= MAX_MAPE_REGRESSION_PERCENT:
            eligible.append((float(np.median(ape)), mape, share))
    if not eligible:
        return 0.0, {"fallback": "no share passed past-only MAPE guard"}
    mdape, mape, share = min(eligible)
    return share, {
        "past_rows": int(len(prior)),
        "past_parent_mdape_percent": parent_mdape,
        "selected_past_mdape_percent": mdape,
        "past_parent_mape_percent": parent_mape,
        "selected_past_mape_percent": mape,
        "past_mape_relative_change_percent": 100.0 * (mape - parent_mape) / parent_mape,
        "eligible_shares": len(eligible),
        "selection": "minimize MdAPE, then MAPE, then R50 share",
    }


def run() -> dict:
    from experiments.bill_target74 import r04_ablation as budget
    from experiments.bill_target74 import r19_expert_weight as common
    from experiments.bill_target74 import r22_remarks_tfidf as r22
    from experiments.bill_target74 import runner as campaign

    card = campaign.PROTOCOL["cards"].get(CARD)
    if not card or card.get("status") != "registered":
        raise RuntimeError("R52 must be registered before loading candidate predictions")
    source = budget._source_identity()
    if source["data_sha256"] != card["data_sha256"]:
        raise RuntimeError("R52 data snapshot changed after preregistration")
    progress = json.loads(campaign.STATE.read_text(encoding="utf-8"))
    result48, path48, paired48_path, manifest48 = _verified_result(
        campaign, progress["r48_result"], PARENT_RUN, "R48"
    )
    result50, path50, paired50_path, manifest50 = _verified_result(
        campaign, progress["r50_result"], VARIANT_RUN, "R50"
    )
    rows48 = pd.read_csv(paired48_path, dtype={"ml_num": str})
    rows50 = pd.read_csv(paired50_path, dtype={"ml_num": str})
    required = {"month", "track", "ml_num", "sold_price", "parent_prediction", "candidate_prediction"}
    for label, frame in (("R48", rows48), ("R50", rows50)):
        if missing := required - set(frame.columns):
            raise ValueError(f"{label} paired predictions lack columns: {sorted(missing)}")
        if ("reserved_months_labels_opened" in frame
                and frame["reserved_months_labels_opened"].fillna(False).astype(bool).any()):
            raise RuntimeError(f"{label} paired rows record restricted-label access")
        if ("hidden_labels_opened" in frame
                and frame["hidden_labels_opened"].fillna(False).astype(bool).any()):
            raise RuntimeError(f"{label} paired rows record hidden-label access")

    rows48 = rows48.loc[rows48.track.eq(TRACK) & rows48.month.isin(MONTHS)].copy()
    rows50 = rows50.loc[rows50.track.eq(TRACK) & rows50.month.isin(MONTHS)].copy()
    keys = ["month", "track", "ml_num"]
    if rows48.duplicated(keys).any() or rows50.duplicated(keys).any():
        raise RuntimeError("R48/R50 paired rows are not unique by month and listing ID")
    merged = rows48[keys + ["sold_price", "parent_prediction", "candidate_prediction"]].merge(
        rows50[keys + ["sold_price", "parent_prediction", "candidate_prediction"]],
        on=keys, how="outer", suffixes=("_r48", "_r50"), validate="one_to_one", indicator=True,
    )
    if not merged["_merge"].eq("both").all() or set(merged.month.unique()) != set(MONTHS):
        raise RuntimeError("R48/R50 development rows do not have exact eight-month coverage")
    if (not np.array_equal(merged.sold_price_r48.to_numpy(float), merged.sold_price_r50.to_numpy(float))
            or not np.array_equal(merged.parent_prediction_r48.to_numpy(float),
                                  merged.parent_prediction_r50.to_numpy(float))):
        raise RuntimeError("R48/R50 actuals or initial R22 parent predictions differ")
    merged = merged.rename(columns={
        "sold_price_r48": "sold_price", "parent_prediction_r48": "initial_parent_prediction",
        "candidate_prediction_r48": "r48_prediction", "candidate_prediction_r50": "r50_prediction",
    }).drop(columns=["sold_price_r50", "parent_prediction_r50", "_merge"])
    for column in ("sold_price", "initial_parent_prediction", "r48_prediction", "r50_prediction"):
        values = merged[column].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError(f"R52 input column {column} contains invalid prices")

    merged["month_period"] = pd.PeriodIndex(merged.month.astype(str), freq="M")
    candidate_parts = []
    selection_details = {}
    for month in MONTHS:
        target_period = pd.Period(month, freq="M")
        current = merged.loc[merged.month.eq(month)].copy()
        prior = merged.loc[merged.month_period < target_period].copy()
        if len(prior) >= MIN_PAST_ROWS:
            share, selection = _select_share(prior)
        else:
            share, selection = 0.0, {
                "past_rows": int(len(prior)), "fallback": "fewer than minimum strict-past rows",
            }
        current["parent_prediction"] = current.r48_prediction.to_numpy(float)
        current["candidate_prediction"] = np.round(
            (1.0 - share) * current.r48_prediction.to_numpy(float)
            + share * current.r50_prediction.to_numpy(float)
        )
        current["r50_share"] = share
        current["past_max_month"] = (
            str(prior.month.max()) if len(prior) else None
        )
        candidate_parts.append(current)
        selection_details[month] = {
            "selected_r50_share": share,
            "past_max_month": current.past_max_month.iloc[0],
            **selection,
        }
    paired = pd.concat(candidate_parts, ignore_index=True)
    if (paired.past_max_month.notna()
            & (paired.past_max_month.astype("string") >= paired.month.astype("string"))).any():
        raise RuntimeError("R52 share selection used current-month or future-month labels")

    score = r22.score_development_confirmation(
        paired, months=MONTHS, target_mdape_strictly_below=7.0,
        bootstrap_replicates=2000, bootstrap_seed=20261024,
        minimum_relative_gain_percent=0.5, minimum_months_improved=6,
        maximum_worst_month_regression_percent=5.0,
        maximum_mape_regression_percent=1.0,
    )
    initial = r22._metrics(
        paired.sold_price.to_numpy(float), paired.initial_parent_prediction.to_numpy(float)
    )
    candidate = r22._metrics(
        paired.sold_price.to_numpy(float), paired.candidate_prediction.to_numpy(float)
    )
    score["initial_R22_parent"] = initial
    score["relative_mdape_improvement_vs_initial_R22_percent"] = (
        100.0 * (initial["mdape_percent"] - candidate["mdape_percent"]) / initial["mdape_percent"]
    )

    run_id = "bill_target74_r52_r48_r50_adaptive_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = campaign.CAMPAIGN_ROOT / "batches" / run_id
    output.mkdir(parents=True, exist_ok=False)
    provenance = output / "provenance"
    provenance.mkdir()
    shutil.copy2(path48, provenance / "r48_development_results.json")
    shutil.copy2(paired48_path, provenance / "r48_paired_predictions.csv")
    shutil.copy2(path48.parent / "run_manifest.json", provenance / "r48_run_manifest.json")
    shutil.copy2(path50, provenance / "r50_development_results.json")
    shutil.copy2(paired50_path, provenance / "r50_paired_predictions.csv")
    shutil.copy2(path50.parent / "run_manifest.json", provenance / "r50_run_manifest.json")
    paired_path = output / "paired_predictions.csv"
    paired.drop(columns="month_period").to_csv(paired_path, index=False)
    result = {
        "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id, "card": CARD,
        "parent_run_id": PARENT_RUN, "variant_run_id": VARIANT_RUN,
        "track": TRACK, "development_months": list(MONTHS), "fit_starts": 0,
        "selection_by_month": selection_details, "tracks": {TRACK: score},
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        "reserve_validation_eligible": False, "data_sha256": source["data_sha256"],
    }
    result_path = output / "development_results.json"
    _write_json(result_path, result)
    source_files = [campaign.PROTOCOL_PATH, Path(__file__), campaign.PACKAGE / "r19_expert_weight.py",
                    campaign.PACKAGE / "r22_remarks_tfidf.py", campaign.PACKAGE / "r04_ablation.py",
                    campaign.PACKAGE / "runner.py"]
    code_hashes = {
        str(path.relative_to(campaign.ROOT)).replace("\\", "/"): _sha256(path)
        for path in source_files
    }
    for index, path in enumerate(source_files):
        shutil.copy2(path, provenance / f"src_{index:02d}_{path.name}.source")
    manifest = {
        "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
        "source_identity": source, "code_sha256": code_hashes,
        "data_sha256": source["data_sha256"], "fit_starts": 0,
        "result_sha256": _sha256(result_path), "paired_predictions_sha256": _sha256(paired_path),
        "r48_result_sha256": _sha256(path48), "r48_predictions_sha256": _sha256(paired48_path),
        "r48_manifest_sha256": _sha256(path48.parent / "run_manifest.json"),
        "r50_result_sha256": _sha256(path50), "r50_predictions_sha256": _sha256(paired50_path),
        "r50_manifest_sha256": _sha256(path50.parent / "run_manifest.json"),
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    }
    _write_json(output / "run_manifest.json", manifest)
    report = [
        "# R52：R48 / R50 严格过去自适应融合", "",
        f"Run：`{run_id}`；父候选 R48：`{PARENT_RUN}`；候选源 R50：`{VARIANT_RUN}`。",
        "每月 R50 权重只由更早开发月的已保存 OOF 预测选出；不重训模型。样本不足 1,000 行或无合格权重时沿用 R48。", "",
        f"- R48 MdAPE：{score['parent']['mdape_percent']:.5f}%",
        f"- R52 MdAPE：{score['candidate']['mdape_percent']:.5f}%",
        f"- R52 vs R48 相对改善：{score['relative_mdape_improvement_percent']:+.3f}%",
        f"- R52 vs 初始 R22 MdAPE：{score['relative_mdape_improvement_vs_initial_R22_percent']:+.3f}%",
        f"- MAPE 相对变化：{score['mape_relative_change_percent']:+.3f}%",
        f"- 改善月份：{score['months_improved']}/8；最坏月退化：{score['worst_month_regression_percent']:.3f}%",
        f"- paired-bootstrap 95% 区间：[{score['bootstrap_mdape_difference_percent']['lower95']:.5f}, "
        f"{score['bootstrap_mdape_difference_percent']['upper95']:.5f}] pp",
        f"- 严格 `<7%` 目标：{'达到' if score['target_met'] else '未达到'}；晋级：{'是' if score['qualifies_development'] else '否'}", "",
        "## 每月选择", "", "| 月份 | 仅用到的最晚月份 | Past 行数 | R50 share |", "|---|---|---:|---:|",
    ]
    for month, selection in selection_details.items():
        report.append(f"| {month} | {selection['past_max_month'] or '无（回退 R48）'} | "
                      f"{selection.get('past_rows', 0)} | {selection['selected_r50_share']:.2f} |")
    report.extend(["", "本结果是预登记的自适应开发评估，不是独立固定起点或保留集验证。", ""])
    (output / "research_report.md").write_text("\n".join(report), encoding="utf-8")

    card.update({
        "status": "development_qualified" if score["qualifies_development"] and score["target_met"]
        else "development_failed",
        "run_id": run_id, "fit_starts": 0,
        "development_result_path": str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        "paired_predictions_path": str(paired_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        "manifest_path": str((output / "run_manifest.json").relative_to(campaign.ROOT)).replace("\\", "/"),
        "metrics": score, "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    campaign.PROTOCOL_PATH.write_text(
        json.dumps(campaign.PROTOCOL, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    campaign._append({
        "event": "card_complete", "card": CARD, "run_id": run_id,
        "fit_starts": 0, "development_result_path": card["development_result_path"],
        "qualifies_development": score["qualifies_development"],
        "target_met": score["target_met"], "metrics": score,
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    next_action = (
        "R52 met <7% and development gates; preregister fixed-origin confirmation using only registered origins"
        if score["qualifies_development"] and score["target_met"] else
        "R52 did not qualify; preserve R22/R35 and register a distinct off-market or list-aware hypothesis"
    )
    campaign._write_progress(
        "r52_development_passed" if score["qualifies_development"] and score["target_met"]
        else "r52_development_failed",
        next_action=next_action, r52_run_id=run_id,
        r52_result=card["development_result_path"], r52_fit_starts=0,
    )
    return result


if __name__ == "__main__":
    register_card()
    print(json.dumps(run(), indent=2, ensure_ascii=False))
