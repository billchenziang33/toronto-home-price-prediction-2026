"""R53: strict-past R22/R32 blending by within-month prediction disagreement."""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.bill_target74.r19_expert_weight import _sha256, _write_json


CARD = "R53_disagreement_tercile_past_oof_blend"
SOURCE_CARD = "R47_past_oof_r22_r32_adaptive_blend"
TRACK = "off_market"
SEED_MONTHS = ("2022-06", "2022-11")
MONTHS = ("2023-03", "2023-07", "2023-11", "2024-03",
          "2024-06", "2024-09", "2025-01", "2025-03")
SAFE_MONTHS = (*SEED_MONTHS, *MONTHS)
SHARES = tuple(round(value, 2) for value in np.arange(0.0, 1.0001, 0.05))
MIN_GLOBAL_ROWS = 1000
MIN_GROUP_ROWS = 500
MAX_MAPE_REGRESSION_PERCENT = 1.0
GROUPS = ("low", "middle", "high")


def _disagreement(parent: np.ndarray, variant: np.ndarray) -> np.ndarray:
    parent = np.asarray(parent, dtype=float)
    variant = np.asarray(variant, dtype=float)
    if (parent.ndim != 1 or variant.shape != parent.shape
            or not np.isfinite(parent).all() or not np.isfinite(variant).all()
            or (parent <= 0).any() or (variant <= 0).any()):
        raise ValueError("R22/R32 predictions must be aligned finite positive price vectors")
    return np.abs(variant - parent) / parent


def _groups(values: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 1 or not np.isfinite(values).all():
        raise ValueError("disagreement groups require a nonempty finite vector")
    q33, q67 = (float(value) for value in np.quantile(values, [1.0 / 3.0, 2.0 / 3.0], method="linear"))
    labels = np.where(values < q33, "low", np.where(values < q67, "middle", "high"))
    return labels.astype(object), (q33, q67)


def _select_share(rows: pd.DataFrame) -> tuple[float | None, dict]:
    actual = rows.sold_price.to_numpy(dtype=float)
    parent = rows.parent_prediction.to_numpy(dtype=float)
    variant = rows.variant_prediction.to_numpy(dtype=float)
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
        return None, {"reason": "no share passed the past-only MAPE guard", "rows": int(len(rows))}
    mdape, mape, share = min(eligible)
    return share, {
        "rows": int(len(rows)), "parent_mdape_percent": parent_mdape,
        "selected_mdape_percent": mdape, "parent_mape_percent": parent_mape,
        "selected_mape_percent": mape,
        "mape_relative_change_percent": 100.0 * (mape - parent_mape) / parent_mape,
        "eligible_shares": int(len(eligible)),
    }


def register_card() -> dict:
    from experiments.bill_target74 import r04_ablation as budget
    from experiments.bill_target74 import runner as campaign

    cards = campaign.PROTOCOL["cards"]
    existing = cards.get(CARD)
    if existing:
        if existing.get("status") == "registered":
            current_source_hash = _sha256(Path(__file__))
            if existing.get("source_sha256") != current_source_hash:
                previous_source_hash = existing.get("source_sha256")
                correction = {
                    "reason": (
                        "The first preflight expected a nonexistent R47 manifest field "
                        "past_oof_predictions_sha256; R47 records this file under "
                        "seed_oof_sha256. Corrected the field mapping before parsing or scoring OOF rows."
                    ),
                    "previous_source_sha256": previous_source_hash,
                    "corrected_source_sha256": current_source_hash,
                    "fit_starts": 0,
                    "row_level_oof_parsed_or_scored": False,
                    "reserved_months_labels_opened": False,
                    "hidden_labels_opened": False,
                }
                existing.setdefault("implementation_corrections", []).append(correction)
                existing["source_sha256"] = current_source_hash
                campaign.PROTOCOL_PATH.write_text(
                    json.dumps(campaign.PROTOCOL, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                campaign._append({
                    "event": "card_preflight_failed", "card": CARD,
                    "error": "R47 OOF hash field name mismatch",
                    "expected_field": "past_oof_predictions_sha256",
                    "actual_field": "seed_oof_sha256", "fit_starts": 0,
                    "row_level_oof_parsed_or_scored": False,
                    "reserved_months_labels_opened": False, "hidden_labels_opened": False,
                })
                campaign._append({
                    "event": "card_implementation_corrected", "card": CARD,
                    "previous_source_sha256": previous_source_hash,
                    "source_sha256": current_source_hash,
                    "reason": "align the R47 OOF hash lookup with its verified manifest schema",
                    "method_changed": False, "fit_starts": 0,
                    "reserved_months_labels_opened": False, "hidden_labels_opened": False,
                })
            return existing
        raise RuntimeError(f"R53 is already {existing.get('status')}; inspect before rerunning")
    source = budget._source_identity()
    registration = {
        "status": "registered", "track": TRACK,
        "parent": "R22 off-market parent predictions",
        "variant": "R32 comparable-feature predictions from hash-verified R47 safe-fold OOF",
        "source_card": SOURCE_CARD,
        "source_run_id": "bill_target74_r47_20261025T070352Z",
        "development_months": list(MONTHS), "seed_oof_months": list(SEED_MONTHS),
        "expected_fit_starts": 0,
        "method": (
            "For each safe OOF month, rank rows into low/middle/high groups by that month's "
            "R22/R32 normalized absolute prediction disagreement, |R32-R22|/R22, using the "
            "month's prediction-only 33rd/67th percentiles. At every development cutoff choose "
            "a R32 share on the fixed 0.00..1.00 grid by 0.05 from only earlier-date OOF rows. "
            "Choose a global share with at least 1,000 past rows; choose group shares only with "
            "at least 500 past rows in that group; use the same 1% past-only MAPE guard and "
            "tie-break by MdAPE, MAPE, then smaller share. Unsupported groups fall back to the "
            "cutoff-specific global share. Blend in price space and round once. Refit no models."
        ),
        "difference_from_prior_work": (
            "R48 grouped the share by house type; R49 by R22 price tercile; R50 by their "
            "interaction. R53 uses a distinct prediction-only uncertainty proxy: normalized "
            "R22/R32 disagreement tercile, with the same strictly-past weight-selection discipline."
        ),
        "weight_selection": {
            "grid": list(SHARES), "minimum_global_past_oof_rows": MIN_GLOBAL_ROWS,
            "minimum_group_past_oof_rows": MIN_GROUP_ROWS,
            "past_mape_regression_max_percent": MAX_MAPE_REGRESSION_PERCENT,
            "feature": "abs(R32_prediction - R22_prediction) / R22_prediction",
            "groups": list(GROUPS), "cutpoints": "within-month prediction-only linear 33rd/67th percentiles",
            "cutoff": "date_sold strictly earlier than target month start",
            "fallback": "unsupported disagreement group uses cutoff-specific global share",
            "tie_break": ["lower MdAPE", "lower MAPE", "smaller R32 share"],
        },
        "development_gate": {
            "target_mdape_strictly_below": 7.0,
            "minimum_relative_mdape_improvement_percent": 0.5,
            "minimum_months_improved": 6,
            "maximum_worst_month_regression_percent": 5.0,
            "maximum_mape_regression_percent": 1.0,
            "paired_bootstrap_upper95_below_zero": True,
            "bootstrap_replicates": 2000, "bootstrap_seed": 20261031,
        },
        "policy": (
            "Register before reading R47's row-level OOF file. Verify the result, OOF, paired "
            "prediction, manifest, data and parent hashes; require exactly the two safe seed "
            "folds plus eight development months. Do not read or use 2025-05/08 or hidden labels. "
            "This adaptive development evaluation is not an independent test."
        ),
        "source_sha256": _sha256(Path(__file__)), "data_sha256": source["data_sha256"],
        "reserve_validation_eligible": False,
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    }
    cards[CARD] = registration
    campaign.PROTOCOL_PATH.write_text(
        json.dumps(campaign.PROTOCOL, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    campaign._append({
        "event": "card_registered", "card": CARD, "track": TRACK,
        "source_card": SOURCE_CARD, "source_run_id": registration["source_run_id"],
        "development_months": list(MONTHS), "seed_oof_months": list(SEED_MONTHS),
        "expected_fit_starts": 0, "data_sha256": source["data_sha256"],
        "source_sha256": registration["source_sha256"],
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    campaign._write_progress(
        "r53_registered",
        next_action="verify the hash-locked R47 OOF source and run the preregistered disagreement-grouped adaptive blend",
        r53_card=CARD,
    )
    return registration


def _verified_source(campaign, card: dict) -> tuple[dict, Path, Path, Path, dict, pd.DataFrame]:
    source_result_path = campaign.ROOT / card["r47_result_path"]
    folder = source_result_path.parent
    oof_path = folder / "past_oof_predictions.csv"
    paired_path = folder / "paired_predictions.csv"
    manifest_path = folder / "run_manifest.json"
    if not all(path.is_file() for path in (source_result_path, oof_path, paired_path, manifest_path)):
        raise FileNotFoundError("R47 result, OOF rows, paired rows, or manifest is missing")
    result = json.loads(source_result_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (result.get("run_id") != card["source_run_id"]
            or manifest.get("run_id") != card["source_run_id"]
            or result.get("campaign_id") != campaign.PROTOCOL["campaign_id"]
            or manifest.get("campaign_id") != campaign.PROTOCOL["campaign_id"]):
        raise RuntimeError("R47 artifacts do not identify the preregistered source run")
    if (manifest.get("result_sha256") != _sha256(source_result_path)
            or manifest.get("seed_oof_sha256") != _sha256(oof_path)
            or manifest.get("paired_predictions_sha256") != _sha256(paired_path)
            or (manifest.get("data_sha256") or manifest.get("source_identity", {}).get("data_sha256"))
            != campaign.PROTOCOL["data_sha256"]):
        raise RuntimeError("R47 result, OOF, paired-row, or data hash verification failed")
    if (result.get("reserved_months_labels_opened") or result.get("hidden_labels_opened")
            or manifest.get("reserved_months_labels_opened") or manifest.get("hidden_labels_opened")):
        raise RuntimeError("R47 source records restricted-label access")
    oof = pd.read_csv(oof_path, dtype={"ml_num": str})
    required = {"month", "track", "ml_num", "date_sold", "sold_price",
                "parent_prediction", "variant_prediction"}
    if missing := required - set(oof.columns):
        raise ValueError(f"R47 OOF rows lack columns: {sorted(missing)}")
    oof = oof.loc[oof.track.eq(TRACK)].copy()
    if (oof.duplicated(["month", "ml_num"]).any()
            or set(oof.month.astype(str)) != set(SAFE_MONTHS)
            or set(oof.month.astype(str)).intersection({"2025-05", "2025-08"})):
        raise RuntimeError("R47 OOF rows do not exactly cover the registered safe folds")
    oof["month"] = oof.month.astype(str)
    oof["date_sold"] = pd.to_datetime(oof.date_sold, errors="raise")
    if (oof.date_sold.dt.to_period("M").astype(str) != oof.month).any():
        raise RuntimeError("R47 OOF sale dates do not match their registered month")
    if (oof["sold_price"].to_numpy(float) <= 0).any():
        raise ValueError("R47 OOF labels must be positive")
    oof["disagreement"] = _disagreement(
        oof.parent_prediction.to_numpy(float), oof.variant_prediction.to_numpy(float)
    )
    oof["group"] = None
    thresholds = {}
    for month in SAFE_MONTHS:
        mask = oof.month.eq(month)
        groups, cutpoints = _groups(oof.loc[mask, "disagreement"].to_numpy(float))
        oof.loc[mask, "group"] = groups
        thresholds[month] = {"q33": cutpoints[0], "q67": cutpoints[1],
                             "rows": int(mask.sum())}
    return result, source_result_path, oof_path, paired_path, manifest, (oof, thresholds)


def run() -> dict:
    from experiments.bill_target74 import r04_ablation as budget
    from experiments.bill_target74 import r22_remarks_tfidf as scoring
    from experiments.bill_target74 import runner as campaign

    card = campaign.PROTOCOL["cards"].get(CARD)
    if not card or card.get("status") != "registered":
        raise RuntimeError("R53 must be preregistered before reading OOF rows")
    if card.get("source_sha256") != _sha256(Path(__file__)):
        raise RuntimeError("R53 source code changed after preregistration")
    source = budget._source_identity()
    if source["data_sha256"] != card["data_sha256"]:
        raise RuntimeError("R53 data snapshot changed after preregistration")
    progress = json.loads(campaign.STATE.read_text(encoding="utf-8"))
    card["r47_result_path"] = progress["r47_result"]
    result47, source_result_path, oof_path, source_paired_path, manifest, loaded = _verified_source(campaign, card)
    oof, thresholds = loaded

    selection_details = {}
    paired_parts = []
    for month in MONTHS:
        cutoff = pd.Period(month, freq="M").start_time
        current = oof.loc[oof.month.eq(month)].copy()
        past = oof.loc[pd.to_datetime(oof.date_sold, errors="raise") < cutoff].copy()
        if len(past) < MIN_GLOBAL_ROWS:
            raise RuntimeError(f"R53 has insufficient global past OOF rows for {month}: {len(past)}")
        global_share, global_detail = _select_share(past)
        if global_share is None:
            global_share = 0.0
        group_details = {}
        shares = {}
        for group in GROUPS:
            group_rows = past.loc[past.group.eq(group)]
            if len(group_rows) >= MIN_GROUP_ROWS:
                share, details = _select_share(group_rows)
            else:
                share, details = None, {"rows": int(len(group_rows)), "reason": "below group support"}
            if share is None:
                share = global_share
                details["fallback_global_share"] = float(global_share)
            shares[group] = float(share)
            group_details[group] = details

        current_share = current.group.map(shares).to_numpy(dtype=float)
        parent_prediction = current.parent_prediction.to_numpy(float)
        variant_prediction = current.variant_prediction.to_numpy(float)
        prediction = np.round((1.0 - current_share) * parent_prediction + current_share * variant_prediction)
        paired_parts.append(pd.DataFrame({
            "month": month, "track": TRACK, "ml_num": current.ml_num.astype(str).to_numpy(),
            "date_sold": current.date_sold.astype(str).to_numpy(),
            "sold_price": current.sold_price.to_numpy(float),
            "parent_prediction": parent_prediction, "candidate_prediction": prediction,
            "variant_prediction": variant_prediction, "disagreement": current.disagreement.to_numpy(float),
            "disagreement_group": current.group.to_numpy(), "selected_r32_share": current_share,
            "past_max_date_sold": str(past.date_sold.max().date()),
        }))
        if past.date_sold.max() >= cutoff:
            raise RuntimeError(f"R53 used labels at/after the {month} cutoff")
        selection_details[month] = {
            "cutoff_exclusive": cutoff.isoformat(),
            "past_max_date_sold": str(past.date_sold.max().date()),
            "past_rows": int(len(past)), "global_share": float(global_share),
            "global_selection": global_detail,
            "group_shares": shares, "group_selection": group_details,
            "current_disagreement_thresholds": thresholds[month],
        }
    paired = pd.concat(paired_parts, ignore_index=True)
    score = scoring.score_development_confirmation(
        paired, months=MONTHS, target_mdape_strictly_below=7.0,
        bootstrap_replicates=2000, bootstrap_seed=20261031,
        minimum_relative_gain_percent=0.5, minimum_months_improved=6,
        maximum_worst_month_regression_percent=5.0,
        maximum_mape_regression_percent=1.0,
    )
    run_id = "bill_target74_r53_disagreement_tercile_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = campaign.CAMPAIGN_ROOT / "batches" / run_id
    output.mkdir(parents=True, exist_ok=False)
    provenance = output / "provenance"
    provenance.mkdir()
    shutil.copy2(source_result_path, provenance / "r47_development_results.json")
    shutil.copy2(oof_path, provenance / "r47_past_oof_predictions.csv")
    shutil.copy2(source_paired_path, provenance / "r47_paired_predictions.csv")
    shutil.copy2(source_result_path.parent / "run_manifest.json", provenance / "r47_run_manifest.json")
    paired_path = output / "paired_predictions.csv"
    paired.to_csv(paired_path, index=False)
    result = {
        "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id, "card": CARD,
        "source_run_id": card["source_run_id"], "track": TRACK,
        "development_months": list(MONTHS), "seed_oof_months": list(SEED_MONTHS),
        "fit_starts": 0, "selection_by_month": selection_details,
        "tracks": {TRACK: score}, "reserved_months_labels_opened": False,
        "hidden_labels_opened": False, "reserve_validation_eligible": False,
        "data_sha256": source["data_sha256"],
    }
    result_path = output / "development_results.json"
    _write_json(result_path, result)
    source_files = [campaign.PROTOCOL_PATH, Path(__file__), campaign.PACKAGE / "r47_past_oof_r22_r32_adaptive_blend.py",
                    campaign.PACKAGE / "r48_groupwise_past_oof_r22_r32_blend.py",
                    campaign.PACKAGE / "r40_past_oof_r26_blend.py",
                    campaign.PACKAGE / "r22_remarks_tfidf.py", campaign.PACKAGE / "r04_ablation.py",
                    campaign.PACKAGE / "runner.py"]
    code_hashes = {
        str(path.relative_to(campaign.ROOT)).replace("\\", "/"): _sha256(path)
        for path in source_files
    }
    for index, path in enumerate(source_files):
        shutil.copy2(path, provenance / f"src_{index:02d}_{path.name}.source")
    manifest_out = {
        "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
        "source_identity": source, "code_sha256": code_hashes,
        "data_sha256": source["data_sha256"], "fit_starts": 0,
        "source_run_id": card["source_run_id"], "source_result_sha256": _sha256(source_result_path),
        "source_oof_sha256": _sha256(oof_path), "source_paired_predictions_sha256": _sha256(provenance / "r47_paired_predictions.csv"),
        "source_manifest_sha256": _sha256(provenance / "r47_run_manifest.json"),
        "result_sha256": _sha256(result_path), "paired_predictions_sha256": _sha256(paired_path),
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    }
    _write_json(output / "run_manifest.json", manifest_out)
    report = [
        "# R53：按 R22/R32 预测分歧分组的严格过去融合", "",
        f"Run：`{run_id}`；源 OOF：`{card['source_run_id']}`。",
        "组别按每个安全 OOF 月自身的预测分歧 tercile 划分；每个目标月的权重只用严格早于该月的 OOF 成交记录选择。", "",
        f"- R22 MdAPE：{score['parent']['mdape_percent']:.5f}%",
        f"- R53 MdAPE：{score['candidate']['mdape_percent']:.5f}%",
        f"- 相对改善：{score['relative_mdape_improvement_percent']:+.3f}%",
        f"- MAPE 相对变化：{score['mape_relative_change_percent']:+.3f}%",
        f"- 改善月份：{score['months_improved']}/8；最坏月退化：{score['worst_month_regression_percent']:.3f}%",
        f"- paired-bootstrap 95% 区间：[{score['bootstrap_mdape_difference_percent']['lower95']:.5f}, "
        f"{score['bootstrap_mdape_difference_percent']['upper95']:.5f}] pp",
        f"- 严格 `<7%`：{'达到' if score['target_met'] else '未达到'}；晋级：{'是' if score['qualifies_development'] else '否'}", "",
        "## 每月选择的 R32 share", "", "| 月份 | R22/R32 分歧 tercile share（低/中/高） | 最晚 past 日期 |", "|---|---:|---|",
    ]
    for month, detail in selection_details.items():
        group_shares = detail["group_shares"]
        report.append(f"| {month} | {group_shares['low']:.2f} / {group_shares['middle']:.2f} / "
                      f"{group_shares['high']:.2f} | {detail['past_max_date_sold']} |")
    report.extend(["", "本自适应开发评估不是独立固定起点或保留集验证。", ""])
    (output / "research_report.md").write_text("\n".join(report), encoding="utf-8")
    qualified = bool(score["qualifies_development"] and score["target_met"])
    card.update({
        "status": "development_qualified" if qualified else "development_failed",
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
        "qualifies_development": score["qualifies_development"], "target_met": score["target_met"],
        "metrics": score, "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    campaign._write_progress(
        "r53_development_passed" if qualified else "r53_development_failed",
        next_action=("R53 met <7%; preregister fixed-origin confirmation" if qualified else
                     "R53 did not qualify; keep validated R22/R35 and register a distinct method"),
        r53_run_id=run_id, r53_result=card["development_result_path"], r53_fit_starts=0,
    )
    return result


if __name__ == "__main__":
    register_card()
    print(json.dumps(run(), indent=2, ensure_ascii=False))
