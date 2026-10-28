"""R68: strictly past-OOF calibration of the R66 cold-start route share."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.bill_target74 import r03_catboost as helpers
from experiments.bill_target74 import r04_ablation as budget
from experiments.bill_target74 import r22_remarks_tfidf as r22
from experiments.bill_target74 import r52_r48_r50_adaptive_meta_blend as r52
from experiments.bill_target74 import r58_past_log_residual_slope_calibration as r58
from experiments.bill_target74 import runner as campaign


CARD = "R68_R66_strict_past_cold_start_share"
R66_CARD = "R66_R65_offmarket_eight_month_development_confirmation"
R66_RUN = "bill_target74_r66_r65_offmarket_development_20261025T094703Z"
R66_BASE = campaign.CAMPAIGN_ROOT / "batches" / R66_RUN
R66_RESULT = R66_BASE / "r66_development_results.json"
R66_PAIRED = R66_BASE / "paired_predictions.csv"
R66_MANIFEST = R66_BASE / "run_manifest.json"
TRACK = "off_market"
PARENT_RUN = r58.R52_RUN
MONTHS = tuple(r22.DEVELOPMENT_MONTHS)
MIN_PAST_ROWS = r52.MIN_PAST_ROWS
MAX_MAPE_REGRESSION_PERCENT = r52.MAX_MAPE_REGRESSION_PERCENT
SHARES = r52.SHARES
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20261025


def _save_protocol() -> None:
    campaign.PROTOCOL_PATH.write_text(
        json.dumps(campaign.PROTOCOL, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _verified_inputs() -> dict:
    source = budget._source_identity()
    card = campaign.PROTOCOL["cards"].get(R66_CARD)
    if not card or card.get("run_id") != R66_RUN or card.get("fit_starts") != 16:
        raise RuntimeError("R68 requires the completed, registered 16-fit R66 development run")
    result = json.loads(R66_RESULT.read_text(encoding="utf-8"))
    manifest = json.loads(R66_MANIFEST.read_text(encoding="utf-8"))
    if (result.get("run_id") != R66_RUN or manifest.get("run_id") != R66_RUN
            or result.get("fit_starts") != 16 or manifest.get("fit_starts") != 16
            or result.get("reserved_months_labels_opened") is not False
            or result.get("hidden_labels_opened") is not False
            or manifest.get("reserved_months_labels_opened") is not False
            or manifest.get("hidden_labels_opened") is not False):
        raise RuntimeError("R68 R66 provenance or label-access flags do not verify")
    if (result.get("data_sha256") != source["data_sha256"]
            or manifest.get("data_sha256") != source["data_sha256"]):
        raise RuntimeError("R68 R66 data hash differs from the active campaign")
    if (helpers._sha256(R66_RESULT) != manifest.get("result_sha256")
            or helpers._sha256(R66_PAIRED) != manifest.get("paired_predictions_sha256")):
        raise RuntimeError("R68 R66 result or paired prediction hash does not match its manifest")

    parent = r58._verified_input(campaign, r58.R52_RESULT, PARENT_RUN, TRACK)
    parent_rows = parent[4].loc[parent[4].month.astype(str).isin(MONTHS)].copy()
    candidate_rows = pd.read_csv(R66_PAIRED, dtype={"ml_num": str})
    required = {"month", "track", "ml_num", "sold_price", "parent_prediction", "candidate_prediction"}
    if required - set(candidate_rows.columns):
        raise RuntimeError(f"R68 R66 paired predictions lack fields: {sorted(required - set(candidate_rows.columns))}")
    candidate_rows = candidate_rows.loc[
        candidate_rows.track.eq(TRACK) & candidate_rows.month.astype(str).isin(MONTHS)
    ].copy()
    if (candidate_rows.duplicated(["month", "ml_num"]).any()
            or parent_rows.duplicated(["month", "ml_num"]).any()):
        raise RuntimeError("R68 source predictions are not unique by month/listing")
    merged = parent_rows[["month", "ml_num", "sold_price", "candidate_prediction"]].merge(
        candidate_rows[["month", "ml_num", "sold_price", "parent_prediction", "candidate_prediction"]],
        on=["month", "ml_num"], how="outer", suffixes=("_r52", "_r66"),
        validate="one_to_one", indicator=True,
    )
    if (not merged["_merge"].eq("both").all()
            or set(merged.month.astype(str).unique()) != set(MONTHS)
            or not np.array_equal(merged.sold_price_r52.to_numpy(float), merged.sold_price_r66.to_numpy(float))
            or not np.array_equal(
                merged.candidate_prediction_r52.to_numpy(float),
                merged.parent_prediction.to_numpy(float),
            )):
        raise RuntimeError("R68 R66 rows, labels, or R52 parent predictions do not align exactly")
    if len(candidate_rows) != int(result["tracks"][TRACK]["n"]):
        raise RuntimeError("R68 R66 paired row count differs from the registered result")
    return {
        "source": source, "r66_result": result, "r66_manifest": manifest,
        "r66_rows": candidate_rows, "r52_input": parent,
        "hashes": {
            "r66_result_sha256": helpers._sha256(R66_RESULT),
            "r66_paired_predictions_sha256": helpers._sha256(R66_PAIRED),
            "r66_manifest_sha256": helpers._sha256(R66_MANIFEST),
            "r52_result_sha256": helpers._sha256(parent[1]),
            "r52_paired_predictions_sha256": helpers._sha256(parent[2]),
            "r52_manifest_sha256": helpers._sha256(parent[1].parent / "run_manifest.json"),
        },
    }


def _select_share(prior: pd.DataFrame) -> tuple[float, dict]:
    if len(prior) < MIN_PAST_ROWS:
        return 0.0, {"past_rows": int(len(prior)), "fallback": "insufficient strictly earlier OOF rows"}
    mapped = prior.rename(columns={
        "parent_prediction": "r48_prediction",
        "candidate_prediction": "r50_prediction",
    })
    share, details = r52._select_share(mapped)
    return share, {"minimum_past_rows": MIN_PAST_ROWS, **details}


def register_card() -> dict:
    existing = campaign.PROTOCOL["cards"].get(CARD)
    current_hash = helpers._sha256(Path(__file__))
    if existing:
        if existing.get("status") == "registered" and existing.get("source_sha256") == current_hash:
            return existing
        raise RuntimeError(f"R68 is already {existing.get('status')}; inspect it before rerunning")
    inputs = _verified_inputs()
    gates = campaign.PROTOCOL["gates"]
    registration = {
        "status": "registered", "track": TRACK, "months": list(MONTHS),
        "expected_fit_starts": 0, "reused_source_fit_starts": 16,
        "parent_run_id": PARENT_RUN, "source_run_id": R66_RUN,
        "source_artifact_hashes": inputs["hashes"],
        "method": (
            "Use R66's strict-as-of, eight-month walk-forward cold-start candidate. For each outer month, choose "
            "one R66 challenger share on only strictly earlier OOF rows, using the registered R52 grid 0.00..1.00 "
            "by 0.05, minimum 1,000 prior rows, and the existing <=1% MAPE guard; minimize past pooled MdAPE, "
            "then MAPE, then share. If no prior rows or no eligible share exists, use share 0.00. Blend rounded "
            "R52 and R66 prices with dollar rounding. Do not use the current month's labels to choose its share."
        ),
        "selection": {
            "alpha_grid": list(SHARES), "minimum_strictly_past_rows": MIN_PAST_ROWS,
            "maximum_past_mape_regression_percent": MAX_MAPE_REGRESSION_PERCENT,
            "objective": "minimize past pooled MdAPE, tie-break by MAPE then smaller R66 share",
            "first_fold_fallback": "R52 parent (R66 share 0.00)",
        },
        "development_gate": {
            "target_mdape_strictly_below": float(campaign.PROTOCOL["initial_development_metrics"][TRACK]["target_mdape_strictly_below"]),
            "minimum_relative_gain_percent": gates["development_relative_mdape_improvement_percent"],
            "minimum_months_improved": gates["development_months_improved_min"],
            "maximum_worst_month_regression_percent": gates["development_worst_month_regression_max_percent"],
            "maximum_mape_regression_percent": gates["development_mape_regression_max_percent"],
            "paired_bootstrap_upper95_below_zero": True,
        },
        "source_sha256": current_hash, "data_sha256": inputs["source"]["data_sha256"],
        "reserve_validation_eligible": False,
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    }
    campaign.PROTOCOL["cards"][CARD] = registration
    _save_protocol()
    campaign._append({
        "event": "card_registered", "card": CARD, "track": TRACK,
        "months": list(MONTHS), "expected_fit_starts": 0,
        "source_run_id": R66_RUN, "parent_run_id": PARENT_RUN,
        "source_artifact_hashes": inputs["hashes"],
        "source_sha256": current_hash, "data_sha256": inputs["source"]["data_sha256"],
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    campaign._write_progress(
        "r68_registered", next_action="apply the preregistered strictly past-OOF share rule to R66 predictions",
        r68_card=CARD,
    )
    return registration


def run() -> dict:
    card = campaign.PROTOCOL["cards"].get(CARD)
    if not card or card.get("status") != "registered":
        raise RuntimeError("R68 must be preregistered before scoring")
    if card.get("source_sha256") != helpers._sha256(Path(__file__)):
        raise RuntimeError("R68 source changed after registration")
    inputs = _verified_inputs()
    if inputs["hashes"] != card["source_artifact_hashes"]:
        raise RuntimeError("R68 source artifact hashes changed after registration")

    rows = inputs["r66_rows"].sort_values(["month", "ml_num"], kind="stable").reset_index(drop=True)
    selected_rows = []
    selection_details = []
    for month in MONTHS:
        month_rows = rows.loc[rows.month.astype(str).eq(month)].copy()
        prior_months = [
            past for past in MONTHS
            if pd.Period(past, freq="M") < pd.Period(month, freq="M")
        ]
        prior = rows.loc[rows.month.astype(str).isin(prior_months)].copy()
        if month_rows.empty or (
            not prior.empty
            and prior.month.astype(str).map(
                lambda value: pd.Period(value, freq="M") >= pd.Period(month, freq="M")
            ).any()
        ):
            raise RuntimeError(f"R68 past-only fold selection is invalid for {month}")
        share, detail = _select_share(prior)
        parent = month_rows.parent_prediction.to_numpy(float)
        challenger = month_rows.candidate_prediction.to_numpy(float)
        actual = month_rows.sold_price.to_numpy(float)
        if (not np.isfinite(parent).all() or not np.isfinite(challenger).all()
                or not np.isfinite(actual).all() or (parent <= 0).any()
                or (challenger <= 0).any() or (actual <= 0).any()):
            raise RuntimeError(f"R68 contains invalid prices in {month}")
        prediction = np.round((1.0 - share) * parent + share * challenger)
        truth = pd.DataFrame({
            "ml_num": month_rows.ml_num.astype(str).to_numpy(), "sold_price": actual,
        })
        parent_frame = truth.assign(predicted_price=parent)
        candidate_frame = truth.assign(predicted_price=prediction)
        paired = helpers._candidate_frame(month, TRACK, truth, parent_frame, candidate_frame)
        paired["R66_prediction"] = challenger
        paired["R66_share"] = float(share)
        selected_rows.append(paired)
        detail.update({
            "month": month, "selected_R66_share": float(share),
            "outer_rows": int(len(month_rows)),
            "outer_parent_mdape_percent": campaign._metrics(actual, parent)["mdape_percent"],
            "outer_R66_mdape_percent": campaign._metrics(actual, challenger)["mdape_percent"],
            "outer_blend_mdape_percent": campaign._metrics(actual, prediction)["mdape_percent"],
        })
        selection_details.append(detail)

    paired_all = pd.concat(selected_rows, ignore_index=True)
    target = float(campaign.PROTOCOL["initial_development_metrics"][TRACK]["target_mdape_strictly_below"])
    gates = campaign.PROTOCOL["gates"]
    score = r22.score_development_confirmation(
        paired_all, months=MONTHS, target_mdape_strictly_below=target,
        bootstrap_replicates=BOOTSTRAP_REPLICATES, bootstrap_seed=BOOTSTRAP_SEED,
        minimum_relative_gain_percent=gates["development_relative_mdape_improvement_percent"],
        minimum_months_improved=gates["development_months_improved_min"],
        maximum_worst_month_regression_percent=gates["development_worst_month_regression_max_percent"],
        maximum_mape_regression_percent=gates["development_mape_regression_max_percent"],
    )
    secondary_score = None
    if "validated_R22_parent_prediction" in rows:
        secondary_rows = paired_all.copy()
        validated_parent = rows.sort_values(["month", "ml_num"], kind="stable")
        check = paired_all[["month", "ml_num"]].merge(
            validated_parent[["month", "ml_num", "validated_R22_parent_prediction"]],
            on=["month", "ml_num"], how="left", validate="one_to_one",
        )
        if check.validated_R22_parent_prediction.isna().any():
            raise RuntimeError("R68 secondary R22 parent predictions do not align")
        secondary_rows["parent_prediction"] = check.validated_R22_parent_prediction.to_numpy(float)
        secondary_score = r22.score_development_confirmation(
            secondary_rows, months=MONTHS, target_mdape_strictly_below=target,
            bootstrap_replicates=BOOTSTRAP_REPLICATES, bootstrap_seed=BOOTSTRAP_SEED,
            minimum_relative_gain_percent=gates["development_relative_mdape_improvement_percent"],
            minimum_months_improved=gates["development_months_improved_min"],
            maximum_worst_month_regression_percent=gates["development_worst_month_regression_max_percent"],
            maximum_mape_regression_percent=gates["development_mape_regression_max_percent"],
        )

    started = datetime.now(timezone.utc)
    run_id = "bill_target74_r68_r66_past_share_" + started.strftime("%Y%m%dT%H%M%SZ")
    output = campaign.CAMPAIGN_ROOT / "batches" / run_id
    output.mkdir(parents=True, exist_ok=False)
    paired_path = output / "paired_predictions.csv"
    selection_path = output / "alpha_selection.json"
    result_path = output / "r68_development_results.json"
    manifest_path = output / "run_manifest.json"
    report_path = output / "research_report.md"
    paired_all.to_csv(paired_path, index=False)
    selection_path.write_text(json.dumps(selection_details, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    result = {
        "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id, "card": CARD,
        "track": TRACK, "parent_run_id": PARENT_RUN, "source_run_id": R66_RUN,
        "development_months": list(MONTHS), "fit_starts": 0,
        "reused_source_fit_starts": 16, "alpha_selection": selection_details,
        "tracks": {TRACK: score}, "secondary_comparison_vs_R22": secondary_score,
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        "data_sha256": inputs["source"]["data_sha256"],
    }
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    code_paths = [
        campaign.PROTOCOL_PATH, Path(__file__), Path(r22.__file__), Path(r52.__file__),
        Path(r58.__file__), Path(helpers.__file__), Path(campaign.__file__),
    ]
    manifest = {
        "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
        "source_identity": inputs["source"], "source_artifact_hashes": inputs["hashes"],
        "method": card["method"], "selection": card["selection"],
        "code_sha256": {str(path.relative_to(campaign.ROOT)).replace("\\", "/"): helpers._sha256(path)
                        for path in code_paths},
        "fit_starts": 0, "reused_source_fit_starts": 16,
        "paired_predictions_sha256": helpers._sha256(paired_path),
        "alpha_selection_sha256": helpers._sha256(selection_path),
        "result_sha256": helpers._sha256(result_path),
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# R68：R66 冷启动专家的严格过去 OOF 份额选择", "",
        f"Run：{run_id}；父：R52；OOF challenger：R66。真实新 learner fit 为 0，复用 R66 的 16 个已完成 fits。",
        "每个外层月份的 R66 权重只用更早月份的 walk-forward 预测和标签选择；本月标签不参与本月权重。",
        "", "## 开发窗口结果", "",
        "| 父模型 | 候选 pooled MdAPE | 相对改善 | MAPE变化 | 月份改善 | bootstrap upper95 | 低于 7% | 完整开发门槛 |",
        "|---:|---:|---:|---:|---:|---:|---|---|",
        f"| {score['parent']['mdape_percent']:.5f}% | {score['candidate']['mdape_percent']:.5f}% | "
        f"{score['relative_mdape_improvement_percent']:+.3f}% | {score['mape_relative_change_percent']:+.3f}% | "
        f"{score['months_improved']}/{len(MONTHS)} | {score['bootstrap_mdape_difference_percent']['upper95']:+.5f} pp | "
        f"{'是' if score['target_met'] else '否'} | {'通过' if score['qualifies_development'] else '未通过'} |",
        "", "## 每月权重与 MdAPE", "", "| 月份 | 仅早期 OOF 行数 | R66 份额 | R52 MdAPE | 混合 MdAPE |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in selection_details:
        lines.append(f"| {item['month']} | {item['past_rows']} | {item['selected_R66_share']:.2f} | "
                     f"{item['outer_parent_mdape_percent']:.5f}% | {item['outer_blend_mdape_percent']:.5f}% |")
    lines.extend([
        "", "只使用固定八个月开发窗口；2025-05/08 与隐藏标签均未读取。保留集仍因已记录访问事件而不具备独立确认资格。",
        "详表：paired_predictions.csv、alpha_selection.json；重现来源见 run_manifest.json。", "",
    ])
    report_path.write_text("\n".join(lines), encoding="utf-8")

    status = "development_passed" if score["qualifies_development"] and score["target_met"] else "development_failed"
    card.update({
        "status": status, "run_id": run_id, "fit_starts": 0, "reused_source_fit_starts": 16,
        "result_path": str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        "paired_predictions_path": str(paired_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        "manifest_path": str(manifest_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        "qualifies_development": bool(score["qualifies_development"]),
        "target_met": bool(score["target_met"]),
    })
    _save_protocol()
    campaign._append({
        "event": "card_complete", "card": CARD, "run_id": run_id,
        "fit_starts": 0, "reused_source_fit_starts": 16,
        "qualifies_development": bool(score["qualifies_development"]),
        "target_met": bool(score["target_met"]),
        "result_path": str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    next_action = (
        "freeze the off-market R68 candidate and run fixed-origin confirmation"
        if score["qualifies_development"] and score["target_met"]
        else "R68 did not satisfy the off-market goal; register another distinct past-only hypothesis"
    )
    campaign._write_progress(
        "r68_development_complete", next_action=next_action,
        r68_result=str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        r68_fit_starts=0, r68_source_fit_starts=16,
    )
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="R68 strict-past OOF share calibration for R66")
    parser.add_argument("--register", action="store_true", help="verify and preregister without scoring")
    args = parser.parse_args()
    result = register_card() if args.register else run()
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str), flush=True)


if __name__ == "__main__":
    main()
