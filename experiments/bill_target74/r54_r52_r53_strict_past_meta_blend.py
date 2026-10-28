"""R54: strict-past month-adaptive blend of R52 and R53 off-market predictions."""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.bill_target74.r19_expert_weight import _sha256, _write_json


CARD = "R54_r52_r53_strict_past_meta_blend"
TRACK = "off_market"
R52_RUN = "bill_target74_r52_r48_r50_adaptive_20261025T074051Z"
R53_RUN = "bill_target74_r53_disagreement_tercile_20261025T074713Z"
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
            current_source_hash = _sha256(Path(__file__))
            if existing.get("source_sha256") != current_source_hash:
                previous_source_hash = existing.get("source_sha256")
                reason = (
                    "The first R54 preflight required date_sold in R52 paired rows, but that "
                    "artifact records its safe month key and past_max_month instead. The schema "
                    "check failed before share selection or scoring. R54 now verifies strict-past "
                    "cutoffs using the registered month key."
                )
                existing.setdefault("implementation_corrections", []).append({
                    "reason": reason,
                    "previous_source_sha256": previous_source_hash,
                    "corrected_source_sha256": current_source_hash,
                    "fit_starts": 0,
                    "row_level_pair_files_loaded_before_schema_failure": True,
                    "weight_selection_or_scoring_started": False,
                    "reserved_months_labels_opened": False,
                    "hidden_labels_opened": False,
                })
                existing["source_sha256"] = current_source_hash
                campaign.PROTOCOL_PATH.write_text(
                    json.dumps(campaign.PROTOCOL, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                campaign._append({
                    "event": "card_preflight_failed", "card": CARD,
                    "error": "R52 paired-row schema does not contain date_sold",
                    "fit_starts": 0, "row_level_pair_files_loaded": True,
                    "weight_selection_or_scoring_started": False,
                    "reserved_months_labels_opened": False, "hidden_labels_opened": False,
                })
                campaign._append({
                    "event": "card_implementation_corrected", "card": CARD,
                    "previous_source_sha256": previous_source_hash,
                    "source_sha256": current_source_hash,
                    "reason": "use the verified safe development month key for strict-past cutoff checks",
                    "method_changed": False, "fit_starts": 0,
                    "reserved_months_labels_opened": False, "hidden_labels_opened": False,
                })
            return existing
        raise RuntimeError(f"R54 is already {existing.get('status')}; inspect before rerunning")

    source = budget._source_identity()
    progress = json.loads(campaign.STATE.read_text(encoding="utf-8"))
    if not all(progress.get(key) for key in ("r52_result", "r53_result")):
        raise RuntimeError("R54 requires the registered R52 and R53 development results")

    registration = {
        "status": "registered",
        "track": TRACK,
        "parent": "R52 strict-past R48/R50 adaptive blend",
        "parent_run_id": R52_RUN,
        "variant_candidate": "R53 strict-past R22/R32 disagreement-tercile blend",
        "variant_run_id": R53_RUN,
        "method": (
            "Reuse hash-verified R52 and R53 same-row predictions. For each target month, "
            "choose the R53 share on the fixed 0.00..1.00 grid by 0.05 using only earlier "
            "development-month rows; require at least 1,000 past rows and no more than 1% "
            "past-only MAPE regression versus R52. If support is smaller, use R52. Ties favor "
            "lower MdAPE, lower MAPE, then the smaller R53 share. Blend rounded price-space "
            "candidate predictions and round the final prediction once. No models are refit."
        ),
        "difference_from_prior_work": (
            "R52 blended R48/R50 using strict-past monthly shares. R53 independently grouped "
            "R22/R32 shares by within-month normalized prediction disagreement. R54 tests an "
            "outer strict-past temporal blend between those two different candidate predictions."
        ),
        "development_months": list(MONTHS),
        "expected_fit_starts": 0,
        "minimum_past_rows": MIN_PAST_ROWS,
        "r53_share_grid": list(SHARES),
        "past_mape_regression_max_percent": MAX_MAPE_REGRESSION_PERCENT,
        "development_gate": {
            "comparison_parent": "R22 validated off-market parent",
            "strict_target_mdape_percent": 7.0,
            "minimum_relative_mdape_improvement_percent": 0.5,
            "minimum_months_improved": 6,
            "maximum_worst_month_regression_percent": 5.0,
            "maximum_mape_regression_percent": 1.0,
            "paired_bootstrap_upper95_below_zero": True,
            "bootstrap_replicates": 2000,
            "bootstrap_seed": 20261032,
        },
        "policy": (
            "Register before loading R52 or R53 row-level paired results. Verify source result, "
            "paired-prediction and manifest hashes, campaign/data/run identity, exact safe-month "
            "coverage, unique month/listing keys, actuals and R22 parent alignment. Select each "
            "month's outer share only from strictly earlier development months. Do not read or "
            "use 2025-05/08 or hidden labels; this adaptive development score is not an independent test."
        ),
        "source_sha256": _sha256(Path(__file__)),
        "data_sha256": source["data_sha256"],
        "reserve_validation_eligible": False,
        "reserved_months_labels_opened": False,
        "hidden_labels_opened": False,
    }
    campaign.PROTOCOL["cards"][CARD] = registration
    campaign.PROTOCOL_PATH.write_text(
        json.dumps(campaign.PROTOCOL, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    campaign._append({
        "event": "card_registered", "card": CARD, "track": TRACK,
        "parent_run_id": R52_RUN, "variant_run_id": R53_RUN,
        "development_months": list(MONTHS), "expected_fit_starts": 0,
        "minimum_past_rows": MIN_PAST_ROWS, "r53_share_grid": list(SHARES),
        "data_sha256": source["data_sha256"], "source_sha256": registration["source_sha256"],
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    campaign._write_progress(
        "r54_registered",
        next_action="verify R52/R53 artifacts and run preregistered strict-past adaptive blend",
        r54_card=CARD,
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
    if (result.get("reserved_months_labels_opened") or result.get("hidden_labels_opened")
            or manifest.get("reserved_months_labels_opened") or manifest.get("hidden_labels_opened")):
        raise RuntimeError(f"{label} run records restricted-label access")
    return result, result_path, paired_path, manifest


def _select_share(prior: pd.DataFrame) -> tuple[float, dict]:
    parent = prior.r52_prediction.to_numpy(dtype=float)
    variant = prior.r53_prediction.to_numpy(dtype=float)
    actual = prior.sold_price.to_numpy(dtype=float)
    parent_ape = 100.0 * np.abs(parent - actual) / actual
    parent_mdape = float(np.median(parent_ape))
    parent_mape = float(np.mean(parent_ape))
    eligible = []
    for share in SHARES:
        prediction = np.round((1.0 - share) * parent + share * variant)
        ape = 100.0 * np.abs(prediction - actual) / actual
        mape = float(np.mean(ape))
        mape_change = 100.0 * (mape - parent_mape) / parent_mape if parent_mape else 0.0
        if mape_change <= MAX_MAPE_REGRESSION_PERCENT:
            eligible.append((float(np.median(ape)), mape, share))
    if not eligible:
        return 0.0, {"reason": "no share passed the past-only MAPE guard; use R52"}
    mdape, mape, share = min(eligible)
    return share, {
        "past_rows": int(len(prior)),
        "r52_mdape_percent": parent_mdape,
        "selected_mdape_percent": mdape,
        "r52_mape_percent": parent_mape,
        "selected_mape_percent": mape,
        "mape_relative_change_percent": 100.0 * (mape - parent_mape) / parent_mape if parent_mape else 0.0,
        "eligible_shares": len(eligible),
    }


def run() -> dict:
    from experiments.bill_target74 import r04_ablation as budget
    from experiments.bill_target74 import r22_remarks_tfidf as scoring
    from experiments.bill_target74 import runner as campaign

    card = campaign.PROTOCOL["cards"].get(CARD)
    if not card or card.get("status") != "registered":
        raise RuntimeError("R54 must be preregistered before loading candidate predictions")
    if card.get("source_sha256") != _sha256(Path(__file__)):
        raise RuntimeError("R54 source code changed after preregistration")
    source = budget._source_identity()
    if source["data_sha256"] != card["data_sha256"]:
        raise RuntimeError("R54 data snapshot changed after preregistration")

    progress = json.loads(campaign.STATE.read_text(encoding="utf-8"))
    result52, path52, paired52_path, manifest52 = _verified_result(
        campaign, progress["r52_result"], R52_RUN, "R52"
    )
    result53, path53, paired53_path, manifest53 = _verified_result(
        campaign, progress["r53_result"], R53_RUN, "R53"
    )
    rows52 = pd.read_csv(paired52_path, dtype={"ml_num": str})
    rows53 = pd.read_csv(paired53_path, dtype={"ml_num": str})
    required52 = {"month", "track", "ml_num", "sold_price",
                  "initial_parent_prediction", "candidate_prediction"}
    required53 = {"month", "track", "ml_num", "sold_price",
                  "parent_prediction", "candidate_prediction"}
    for label, frame, required in (("R52", rows52, required52), ("R53", rows53, required53)):
        if missing := required - set(frame.columns):
            raise ValueError(f"{label} paired predictions lack columns: {sorted(missing)}")
        if frame.duplicated(["month", "track", "ml_num"]).any():
            raise RuntimeError(f"{label} paired rows are not unique by month and listing ID")
        if set(frame.month.astype(str)) != set(MONTHS):
            raise RuntimeError(f"{label} paired rows do not exactly cover the eight safe months")
        if set(frame.track.astype(str)) != {TRACK}:
            raise RuntimeError(f"{label} paired rows contain an unexpected track")

    keys = ["month", "track", "ml_num"]
    left = rows52[keys + ["sold_price", "initial_parent_prediction", "candidate_prediction"]]
    right = rows53[keys + ["sold_price", "parent_prediction", "candidate_prediction"]].rename(
        columns={"parent_prediction": "r53_parent_prediction"}
    )
    merged = left.merge(right, on=keys, how="outer", suffixes=("_r52", "_r53"),
                        validate="one_to_one", indicator=True)
    if not merged["_merge"].eq("both").all() or set(merged.month.astype(str)) != set(MONTHS):
        raise RuntimeError("R52/R53 development rows do not have exact eight-month coverage")
    for left_column, right_column, label in (
        ("sold_price_r52", "sold_price_r53", "sale prices"),
        ("initial_parent_prediction", "r53_parent_prediction", "R22 parent predictions"),
    ):
        if not np.array_equal(merged[left_column].to_numpy(float), merged[right_column].to_numpy(float)):
            raise RuntimeError(f"R52/R53 {label} do not match row-for-row")
    merged = merged.rename(columns={
        "sold_price_r52": "sold_price",
        "initial_parent_prediction": "parent_prediction",
        "candidate_prediction_r52": "r52_prediction",
        "candidate_prediction_r53": "r53_prediction",
    }).drop(columns=["sold_price_r53", "r53_parent_prediction", "_merge"])
    for column in ("sold_price", "parent_prediction", "r52_prediction", "r53_prediction"):
        values = merged[column].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError(f"R54 input column {column} contains invalid prices")
    merged["month_period"] = pd.PeriodIndex(merged.month.astype(str), freq="M")

    paired_parts = []
    selection_details = {}
    for month in MONTHS:
        cutoff = pd.Period(month, freq="M").start_time
        current = merged.loc[merged.month.astype(str).eq(month)].copy()
        prior = merged.loc[merged.month_period < pd.Period(month, freq="M")].copy()
        if len(prior) >= MIN_PAST_ROWS:
            if (prior.month_period >= pd.Period(month, freq="M")).any():
                raise RuntimeError(f"R54 prior selection includes labels at/after {month} cutoff")
            share, selection = _select_share(prior)
        else:
            share, selection = 0.0, {
                "past_rows": int(len(prior)), "fallback": "fewer than minimum strict-past rows; use R52",
            }
        current["candidate_prediction"] = np.round(
            (1.0 - share) * current.r52_prediction.to_numpy(float)
            + share * current.r53_prediction.to_numpy(float)
        )
        current["selected_r53_share"] = float(share)
        current["past_max_month"] = str(prior.month.max()) if len(prior) else None
        if len(prior) and prior.month_period.max() >= pd.Period(month, freq="M"):
            raise RuntimeError(f"R54 used labels at/after the {month} cutoff")
        paired_parts.append(current)
        selection_details[month] = {
            "selected_r53_share": float(share),
            "past_max_month": current.past_max_month.iloc[0],
            **selection,
        }
    paired = pd.concat(paired_parts, ignore_index=True)
    score = scoring.score_development_confirmation(
        paired, months=MONTHS, target_mdape_strictly_below=7.0,
        bootstrap_replicates=2000, bootstrap_seed=20261032,
        minimum_relative_gain_percent=0.5, minimum_months_improved=6,
        maximum_worst_month_regression_percent=5.0,
        maximum_mape_regression_percent=1.0,
    )
    r52_metrics = scoring._metrics(
        paired.sold_price.to_numpy(float), paired.r52_prediction.to_numpy(float)
    )
    score["R52_parent"] = r52_metrics
    score["relative_mdape_improvement_vs_R52_percent"] = (
        100.0 * (r52_metrics["mdape_percent"] - score["candidate"]["mdape_percent"])
        / r52_metrics["mdape_percent"]
    )

    run_id = "bill_target74_r54_r52_r53_adaptive_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = campaign.CAMPAIGN_ROOT / "batches" / run_id
    output.mkdir(parents=True, exist_ok=False)
    provenance = output / "provenance"
    provenance.mkdir()
    for label, result_path, paired_path, manifest_path in (
        ("r52", path52, paired52_path, path52.parent / "run_manifest.json"),
        ("r53", path53, paired53_path, path53.parent / "run_manifest.json"),
    ):
        shutil.copy2(result_path, provenance / f"{label}_development_results.json")
        shutil.copy2(paired_path, provenance / f"{label}_paired_predictions.csv")
        shutil.copy2(manifest_path, provenance / f"{label}_run_manifest.json")

    paired_path = output / "paired_predictions.csv"
    paired.to_csv(paired_path, index=False)
    result = {
        "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
        "card": CARD, "parent_run_id": R52_RUN, "variant_run_id": R53_RUN,
        "track": TRACK, "development_months": list(MONTHS), "fit_starts": 0,
        "selection_by_month": selection_details, "tracks": {TRACK: score},
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        "reserve_validation_eligible": False, "data_sha256": source["data_sha256"],
    }
    result_path = output / "development_results.json"
    _write_json(result_path, result)
    manifest_out = {
        "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
        "source_identity": source, "data_sha256": source["data_sha256"],
        "fit_starts": 0, "r52_result_sha256": _sha256(path52),
        "r52_paired_predictions_sha256": _sha256(paired52_path),
        "r52_manifest_sha256": _sha256(path52.parent / "run_manifest.json"),
        "r53_result_sha256": _sha256(path53),
        "r53_paired_predictions_sha256": _sha256(paired53_path),
        "r53_manifest_sha256": _sha256(path53.parent / "run_manifest.json"),
        "result_sha256": _sha256(result_path), "paired_predictions_sha256": _sha256(paired_path),
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    }
    _write_json(output / "run_manifest.json", manifest_out)
    report = [
        "# R54：R52/R53 严格过去月度融合", "",
        f"Run：`{run_id}`；比较父版 R22；R52 候选：`{R52_RUN}`；R53 候选：`{R53_RUN}`。", "",
        f"- R22 MdAPE：{score['parent']['mdape_percent']:.5f}%",
        f"- R54 MdAPE：{score['candidate']['mdape_percent']:.5f}%",
        f"- R52 MdAPE：{r52_metrics['mdape_percent']:.5f}%",
        f"- 相对 R22 改善：{score['relative_mdape_improvement_percent']:+.3f}%",
        f"- 相对 R52 增益：{score['relative_mdape_improvement_vs_R52_percent']:+.3f}%",
        f"- MAPE 相对 R22 变化：{score['mape_relative_change_percent']:+.3f}%",
        f"- 改善月份：{score['months_improved']}/8；最坏月退化：{score['worst_month_regression_percent']:.3f}%",
        f"- paired-bootstrap 95% 区间：[{score['bootstrap_mdape_difference_percent']['lower95']:.5f}, "
        f"{score['bootstrap_mdape_difference_percent']['upper95']:.5f}] pp",
        f"- 严格 `<7%`：{'达到' if score['target_met'] else '未达到'}；晋级：{'是' if score['qualifies_development'] else '否'}", "",
        "## 每月选择的 R53 share", "", "| 月份 | R53 share | 最晚 past 日期 |", "|---|---:|---|",
    ]
    for month, detail in selection_details.items():
        report.append(f"| {month} | {detail['selected_r53_share']:.2f} | {detail['past_max_month'] or '无（沿用 R52）'} |")
    report.extend(["", "自适应开发分数不是独立固定起点或保留集验证。", ""])
    (output / "research_report.md").write_text("\n".join(report), encoding="utf-8")

    qualified = bool(score["qualifies_development"] and score["target_met"])
    card.update({
        "status": "development_qualified" if qualified else "development_failed",
        "run_id": run_id, "fit_starts": 0,
        "development_result_path": str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        "paired_predictions_path": str(paired_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        "manifest_path": str((output / "run_manifest.json").relative_to(campaign.ROOT)).replace("\\", "/"),
        "metrics": score, "reserved_months_labels_opened": False,
        "hidden_labels_opened": False,
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
    campaign._write_progress(
        "r54_development_passed" if qualified else "r54_development_failed",
        next_action=("R54 met <7%; preregister fixed-origin confirmation" if qualified else
                     "R54 did not qualify; keep validated R22/R35 and register a distinct method"),
        r54_run_id=run_id, r54_result=card["development_result_path"], r54_fit_starts=0,
    )
    return result


if __name__ == "__main__":
    register_card()
    print(json.dumps(run(), indent=2, ensure_ascii=False))
