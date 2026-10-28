"""R40 eight-month confirmation using fixed, past-only OOF blend selection."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.bill_target74 import r19_expert_weight as ledger
from experiments.bill_target74 import r04_ablation as data_helpers
from experiments.bill_target74 import r22_remarks_tfidf as r22
from experiments.bill_target74 import r26_word_tfidf_512 as r26
from experiments.bill_target74 import r40_past_oof_r26_blend as r40
from experiments.bill_target74 import runner as campaign
from experiments.bill_greedy import runner as greedy


CARD = r40.CARD
STAGE = "development_confirmation"
TRACK = r40.TRACK


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_directory() -> Path:
    result_path = campaign.PROTOCOL["cards"][CARD].get("result_path")
    if not result_path:
        raise RuntimeError("R40 screen result is not registered")
    return campaign.ROOT / Path(result_path).parent


def preregister() -> dict:
    card = campaign.PROTOCOL["cards"].get(CARD)
    if not card or card.get("status") != "screen_passed":
        raise RuntimeError("R40 development confirmation requires a passed R40 screen")
    registered = card.get(STAGE)
    if registered:
        if registered.get("status") == "registered":
            return registered
        raise RuntimeError(f"R40 development stage already has status {registered.get('status')!r}")

    gates = campaign.PROTOCOL["gates"]
    screen_path = campaign.ROOT / card["result_path"]
    screen = json.loads(screen_path.read_text(encoding="utf-8"))
    if not screen["tracks"][TRACK]["qualifies_screen"]:
        raise RuntimeError("R40 screen result does not satisfy the registered screen gate")
    configuration = {
        "status": "registered",
        "method": (
            "For each of the eight registered development cutoffs, choose an R26 share on the existing 0.05 grid "
            "by minimizing MdAPE over only OOF rows whose date_sold is strictly before that cutoff; require at "
            "least 1,000 prior rows and no more than 1% MAPE regression versus R22. Use the frozen R22 and R26 "
            "walk-forward predictions for the held-out fold and round the price-space blend exactly once."
        ),
        "parent_run_id": screen["parent_run_id"],
        "variant_run_id": screen["variant_screen_run_id"],
        "screen_run_id": screen["run_id"],
        "development_months": list(r40.DEVELOPMENT_MONTHS),
        "seed_oof_months": list(r40.SEED_MONTHS),
        "expected_new_fit_starts": 0,
        "minimum_past_oof_rows": r40.MINIMUM_OOF_ROWS,
        "weight_grid": list(r40.WEIGHTS),
        "mape_regression_guard_percent": 1.0,
        "gates": {
            "relative_mdape_improvement_percent_min": gates["development_relative_mdape_improvement_percent"],
            "months_improved_min": gates["development_months_improved_min"],
            "worst_month_regression_percent_max": gates["development_worst_month_regression_max_percent"],
            "mape_regression_percent_max": gates["development_mape_regression_max_percent"],
            "paired_bootstrap_upper95_below_zero": gates["development_bootstrap_upper_bound_below_zero"],
            "target_mdape_strictly_below": campaign.PROTOCOL["initial_development_metrics"][TRACK][
                "target_mdape_strictly_below"
            ],
            "bootstrap_replicates": campaign.PROTOCOL["cards"][CARD].get("bootstrap_replicates", 2000),
            "bootstrap_seed": campaign.PROTOCOL["cards"][CARD].get("bootstrap_seed", 20261024),
        },
        "integrity_policy": (
            "Verify R22/R26 development result and paired-prediction hashes, row/label alignment, R40 screen/OOF "
            "hashes, and safe month-fold provenance before scoring. Use no reserve or hidden labels. The eight-month "
            "development window is adaptive and is not an independent final holdout."
        ),
    }
    card[STAGE] = configuration
    ledger._write_json(campaign.PROTOCOL_PATH, campaign.PROTOCOL)
    campaign._append({
        "event": "stage_registered",
        "card": CARD,
        "stage": STAGE,
        "expected_new_fit_starts": 0,
        "screen_run_id": screen["run_id"],
        "development_months": list(r40.DEVELOPMENT_MONTHS),
        "strictly_past_only_selection": True,
        "reserved_months_labels_opened": False,
        "hidden_labels_opened": False,
    })
    campaign._write_progress(
        "r40_development_registered",
        next_action="run R40 eight-month development confirmation with cutoff-specific past-only weights",
        r40_development_expected_new_fit_starts=0,
    )
    return configuration


def run_confirmation() -> dict:
    card = campaign.PROTOCOL["cards"].get(CARD)
    stage = card.get(STAGE) if card else None
    if not stage or stage.get("status") != "registered":
        raise RuntimeError("R40 development confirmation must be preregistered before scoring")
    if card.get("status") != "screen_passed":
        raise RuntimeError("R40 screen is no longer marked as passed")

    progress = json.loads(campaign.STATE.read_text(encoding="utf-8"))
    r22_relative = progress.get("r22_development_result")
    r26_relative = progress.get("r26_development_result")
    if not r22_relative or not r26_relative:
        raise RuntimeError("verified R22 and R26 development OOF results are required")
    r22_path = campaign.ROOT / r22_relative
    r26_path = campaign.ROOT / r26_relative
    r22_result = json.loads(r22_path.read_text(encoding="utf-8"))
    r26_result = json.loads(r26_path.read_text(encoding="utf-8"))
    r22_manifest = json.loads((r22_path.parent / "run_manifest.json").read_text(encoding="utf-8"))
    r26_manifest = json.loads((r26_path.parent / "run_manifest.json").read_text(encoding="utf-8"))
    source_sha = campaign.PROTOCOL["data_sha256"]
    if r22_result.get("stage") != STAGE or r26_result.get("stage") != STAGE:
        raise RuntimeError("R22/R26 inputs are not development-confirmation artifacts")
    r22_rows = r40._read_checked_result(r22_path, r22_result, r22_manifest, r22.CARD, source_sha)
    r26_rows = r40._read_checked_result(r26_path, r26_result, r26_manifest, r26.CARD, source_sha)

    screen_path = campaign.ROOT / card["result_path"]
    screen_result = json.loads(screen_path.read_text(encoding="utf-8"))
    screen_manifest = json.loads((screen_path.parent / "run_manifest.json").read_text(encoding="utf-8"))
    if (screen_result.get("run_id") != stage["screen_run_id"]
            or screen_result["tracks"][TRACK].get("qualifies_screen") is not True
            or screen_result.get("data_sha256") != source_sha
            or screen_result.get("reserved_months_labels_opened")
            or screen_result.get("hidden_labels_opened")
            or screen_manifest.get("run_id") != screen_result.get("run_id")
            or screen_manifest.get("result_sha256") != _sha256(screen_path)
            or screen_manifest.get("paired_predictions_sha256")
            != _sha256(screen_path.parent / "paired_predictions.csv")):
        raise RuntimeError("R40 screen result or manifest failed verification")
    r40_oof_path = screen_path.parent / "past_oof_predictions.csv"
    if screen_manifest.get("past_oof_predictions_sha256") != _sha256(r40_oof_path):
        raise RuntimeError("R40 past-OOF predictions failed hash verification")
    seed_oof = pd.read_csv(r40_oof_path, dtype={"ml_num": str})
    seed_oof = seed_oof.loc[seed_oof.month.astype(str).isin(r40.SEED_MONTHS)].copy()
    if set(seed_oof.month.astype(str).unique()) != set(r40.SEED_MONTHS):
        raise RuntimeError("R40 seed OOF file does not contain exactly the registered safe seed months")

    development_oof = r40._development_oof(r40.DEVELOPMENT_MONTHS, r22_rows, r26_rows, greedy)
    oof = pd.concat([seed_oof, development_oof], ignore_index=True)
    if oof.duplicated(["month", "ml_num"]).any():
        raise RuntimeError("R40 seed and development OOF rows overlap")

    paired_parts = []
    selections = []
    for month in r40.DEVELOPMENT_MONTHS:
        cutoff = pd.Period(month, freq="M").start_time
        selection = r40.select_weight(oof, cutoff)
        if (selection["reason"] != "selected_from_strictly_past_oof"
                or selection["oof_rows"] < r40.MINIMUM_OOF_ROWS):
            raise RuntimeError(f"R40 lacks sufficient strictly past OOF support for {month}")
        current = development_oof.loc[development_oof.month.eq(month)].copy()
        if current.empty:
            raise RuntimeError(f"R40 has no development predictions for {month}")
        current["candidate_prediction"] = r40.blend(
            current.parent_prediction.to_numpy(float),
            current.variant_prediction.to_numpy(float),
            selection["weight"],
        )
        current["r26_blend_weight"] = selection["weight"]
        paired_parts.append(current)
        selections.append({
            "month": month,
            "cutoff_exclusive": cutoff.isoformat(),
            **selection,
        })

    screen_weights = {
        row["screen_month"]: row["weight"]
        for row in screen_result["weight_selections"]
    }
    for month in r40.SCREEN_MONTHS:
        development_weight = next(row["weight"] for row in selections if row["month"] == month)
        if development_weight != screen_weights[month]:
            raise RuntimeError(f"R40 development and screen past-only weights differ for {month}")

    paired = pd.concat(paired_parts, ignore_index=True)
    gates = campaign.PROTOCOL["gates"]
    target = campaign.PROTOCOL["initial_development_metrics"][TRACK]["target_mdape_strictly_below"]
    score = r22.score_development_confirmation(
        paired,
        months=r40.DEVELOPMENT_MONTHS,
        target_mdape_strictly_below=target,
        bootstrap_replicates=stage["gates"]["bootstrap_replicates"],
        bootstrap_seed=stage["gates"]["bootstrap_seed"],
        minimum_relative_gain_percent=gates["development_relative_mdape_improvement_percent"],
        minimum_months_improved=gates["development_months_improved_min"],
        maximum_worst_month_regression_percent=gates["development_worst_month_regression_max_percent"],
        maximum_mape_regression_percent=gates["development_mape_regression_max_percent"],
    )

    run_id = "bill_target74_r40_development_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = campaign.CAMPAIGN_ROOT / "batches" / run_id
    output.mkdir(parents=True, exist_ok=False)
    paired_path = output / "paired_predictions.csv"
    paired.to_csv(paired_path, index=False)
    _write_json = ledger._write_json
    result = {
        "campaign_id": campaign.PROTOCOL["campaign_id"],
        "run_id": run_id,
        "card": CARD,
        "stage": STAGE,
        "screen_run_id": screen_result["run_id"],
        "parent_run_id": r22_result["run_id"],
        "variant_run_id": r26_result["run_id"],
        "development_months": list(r40.DEVELOPMENT_MONTHS),
        "fit_starts": 0,
        "new_fit_starts": 0,
        "tracks": {TRACK: score},
        "weight_selections": selections,
        "strictly_past_only_selection": True,
        "data_sha256": source_sha,
        "reserved_months_labels_opened": False,
        "reserve_validation_eligible": False,
        "hidden_labels_opened": False,
    }
    result_path = output / "development_results.json"
    _write_json(result_path, result)

    source_identity = data_helpers._source_identity()
    if source_identity["data_sha256"] != source_sha:
        raise RuntimeError("R40 development source data hash differs from the active campaign")
    source_paths = [
        campaign.PROTOCOL_PATH,
        campaign.PACKAGE / "r40_development_confirmation.py",
        campaign.PACKAGE / "r40_past_oof_r26_blend.py",
        campaign.PACKAGE / "r22_remarks_tfidf.py",
        campaign.PACKAGE / "r26_word_tfidf_512.py",
        campaign.PACKAGE / "r19_expert_weight.py",
        campaign.PACKAGE / "r04_ablation.py",
        campaign.PACKAGE / "runner.py",
        campaign.ROOT / "experiments/bill_greedy/runner.py",
        campaign.ROOT / "experiments/bill_greedy/history.py",
    ]
    manifest = {
        "campaign_id": campaign.PROTOCOL["campaign_id"],
        "run_id": run_id,
        "source_identity": source_identity,
        "screen_result_sha256": _sha256(screen_path),
        "screen_paired_predictions_sha256": screen_manifest["paired_predictions_sha256"],
        "r22_development_result_sha256": _sha256(r22_path),
        "r26_development_result_sha256": _sha256(r26_path),
        "r22_development_paired_predictions_sha256": r22_manifest["paired_predictions_sha256"],
        "r26_development_paired_predictions_sha256": r26_manifest["paired_predictions_sha256"],
        "r40_past_oof_predictions_sha256": _sha256(r40_oof_path),
        "code_sha256": {
            str(path.relative_to(campaign.ROOT)).replace("\\", "/"): _sha256(path)
            for path in source_paths
        },
        "paired_predictions_sha256": _sha256(paired_path),
        "result_sha256": _sha256(result_path),
        "fit_starts": 0,
        "new_fit_starts": 0,
        "reserved_months_labels_opened": False,
        "reserve_validation_eligible": False,
        "hidden_labels_opened": False,
    }
    _write_json(output / "run_manifest.json", manifest)

    lines = [
        "# R40 八个月开发确认：按截止日选择严格过去 OOF 权重",
        "",
        f"运行：`{run_id}`；筛选运行：`{screen_result['run_id']}`。本阶段复用已保存 OOF 预测，新增 learner fits 为 0。",
        "",
        "每个开发折都在固定 `0.00..1.00`、步长 `0.05` 的网格中，使用 `date_sold` 严格早于该折截止日的 OOF 行选择 R26 权重。每折至少需要 1,000 行，过去 OOF MAPE 相对 R22 的恶化不得超过 1%。",
        "",
        "| 开发月 | 过去 OOF 月份 | OOF 行 | R26 权重 | 过去 OOF MdAPE 父→混合 | 过去 OOF MAPE 变化 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in selections:
        lines.append(
            f"| {row['month']} | {', '.join(row['oof_months'])} | {row['oof_rows']} | {row['weight']:.2f} | "
            f"{row['parent']['mdape_percent']:.5f}%→{row['selected']['mdape_percent']:.5f}% | "
            f"{row['selected_mape_relative_change_percent']:+.3f}% |"
        )
    lines.extend([
        "",
        "## 八个月总体结果",
        "",
        f"- R22 MdAPE：{score['parent']['mdape_percent']:.5f}%；R40 MdAPE：{score['candidate']['mdape_percent']:.5f}%。",
        f"- MdAPE 相对改善：{score['relative_mdape_improvement_percent']:+.3f}%；MAPE 相对变化：{score['mape_relative_change_percent']:+.3f}%。",
        f"- 改善月份：{score['months_improved']}/8；最差月份回归：{score['worst_month_regression_percent']:.3f}%。",
        f"- 配对 bootstrap 95% MdAPE 差值区间：[{score['bootstrap_mdape_difference_percent']['lower95']:+.5f}, {score['bootstrap_mdape_difference_percent']['upper95']:+.5f}] 个百分点。",
        f"- 开发门槛通过：{score['qualifies_development']}；严格低于 {target:.2f}% 目标：{score['target_met']}。",
        "",
        "此八个月开发窗用于自适应开发确认，不是独立最终留出集。2025-05/08 预留标签因先前完整 parquet 载入事件而不可用于一次性验证；本阶段未读取这些月份或隐藏标签。",
        "",
        f"配对预测、R22/R26 输入来源、past-OOF 文件和 SHA-256 清单位于 `{output.relative_to(campaign.ROOT)}`。",
        "",
    ])
    (output / "research_report.md").write_text("\n".join(lines), encoding="utf-8")

    card[STAGE].update({
        "status": "passed" if score["qualifies_development"] else "failed",
        "run_id": run_id,
        "result_path": str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        "new_fit_starts": 0,
        "qualifies_development": score["qualifies_development"],
        "target_met": score["target_met"],
        "metrics": {
            "parent_mdape_percent": score["parent"]["mdape_percent"],
            "candidate_mdape_percent": score["candidate"]["mdape_percent"],
            "relative_mdape_improvement_percent": score["relative_mdape_improvement_percent"],
            "mape_relative_change_percent": score["mape_relative_change_percent"],
            "months_improved": score["months_improved"],
            "worst_month_regression_percent": score["worst_month_regression_percent"],
            "bootstrap_mdape_difference_percent": score["bootstrap_mdape_difference_percent"],
        },
    })
    card["status"] = "development_passed" if score["qualifies_development"] else "development_failed"
    ledger._write_json(campaign.PROTOCOL_PATH, campaign.PROTOCOL)
    campaign._append({
        "event": "stage_complete",
        "card": CARD,
        "stage": STAGE,
        "run_id": run_id,
        "fit_starts": 0,
        "new_fit_starts": 0,
        "result_path": str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        "qualifies_development": score["qualifies_development"],
        "target_met": score["target_met"],
        "reserved_months_labels_opened": False,
        "hidden_labels_opened": False,
    })
    if score["qualifies_development"]:
        next_action = (
            "run locked fixed-origin confirmation for R40; reserve validation is ineligible"
            if score["target_met"]
            else "run locked fixed-origin confirmation for R40, then continue off-market search toward MdAPE below 7%"
        )
        status = "r40_development_complete"
    else:
        next_action = "preserve R22 as off-market parent and register a distinct off-market hypothesis"
        status = "r40_development_failed"
    campaign._write_progress(
        status,
        next_action=next_action,
        r40_development_result=str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        latest_fit_count=0,
    )
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--register", action="store_true", help="preregister the development stage only")
    args = parser.parse_args()
    if args.register:
        print(json.dumps(preregister(), indent=2))
    else:
        result = run_confirmation()
        score = result["tracks"][TRACK]
        print(
            f"R40 development: parent_mdape={score['parent']['mdape_percent']:.5f}% "
            f"candidate_mdape={score['candidate']['mdape_percent']:.5f}% "
            f"relative_gain={score['relative_mdape_improvement_percent']:+.3f}% "
            f"mape_change={score['mape_relative_change_percent']:+.3f}% "
            f"qualifies={score['qualifies_development']} target_met={score['target_met']} "
            f"run={result['run_id']}",
            flush=True,
        )
