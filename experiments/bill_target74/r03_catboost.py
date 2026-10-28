from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.bill_target74 import runner as campaign

ROOT = campaign.ROOT
SCREEN_MONTHS = tuple(campaign.PROTOCOL["screen_months"])
PARENTS = dict(campaign.PROTOCOL["parent_by_track"])
CATBOOST_CONFIG = {
    "iterations": 1000,
    "depth": 6,
    "learning_rate": 0.05,
    "l2_leaf_reg": 10,
    "loss_function": "RMSE",
    "has_time": True,
    "task_type": "CPU",
    "random_seed": 0,
    "thread_count": 4,
    "verbose": False,
    "allow_writing_files": False,
}


def r03_fit_id(campaign_id: str, track: str, month: str) -> str:
    if track not in {"off_market", "list_aware"}:
        raise ValueError(f"unknown track {track!r}")
    if month not in SCREEN_MONTHS:
        raise ValueError(f"month {month!r} is not a registered screen fold")
    return f"{campaign_id}:R03_full_feature_CatBoost:{track}:{month}:1000i_l2_10"


def qualifies_screen(relative_mdape_improvement_percent: float,
                     monthly_improvements_percent: list[float],
                     mape_relative_change_percent: float) -> bool:
    gates = campaign.PROTOCOL["gates"]
    return bool(
        relative_mdape_improvement_percent >= gates["screen_relative_mdape_improvement_percent"]
        and len(monthly_improvements_percent) == len(SCREEN_MONTHS)
        and all(value > 0.0 for value in monthly_improvements_percent)
        and mape_relative_change_percent <= gates["screen_mape_regression_max_percent"]
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fit_count() -> int:
    if not campaign.LEDGER.exists():
        return 0
    return sum(
        json.loads(line).get("event") == "fit_start"
        for line in campaign.LEDGER.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _source_identity() -> dict:
    branch = subprocess.run(["git", "branch", "--show-current"], cwd=ROOT,
                            capture_output=True, text=True, check=True).stdout.strip()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True, check=True).stdout.strip()
    if branch != "bill" or commit != campaign.PROTOCOL["parent_commit"]:
        raise RuntimeError(f"expected bill at {campaign.PROTOCOL['parent_commit']}, found {branch} at {commit}")
    if _sha256(campaign.DATA) != campaign.PROTOCOL["data_sha256"]:
        raise RuntimeError("cached listings parquet hash differs from the campaign data hash")
    return {"branch": branch, "commit": commit, "data_sha256": _sha256(campaign.DATA)}


def _verified_hgbr_cache(greedy, month: str, track: str, train: pd.DataFrame,
                         homes: pd.DataFrame) -> np.ndarray:
    del train
    if track != "list_aware":
        raise ValueError("the frozen HGBR component is only used in the list-aware track")
    path = greedy.OUTPUT / "frozen_hgbr" / track / f"{month}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"verified frozen HGBR cache is missing: {path}")
    cached = pd.read_csv(path, dtype={"ml_num": str})
    expected_ids = homes.ml_num.astype(str).to_numpy()
    if not np.array_equal(cached.ml_num.astype(str).to_numpy(), expected_ids):
        raise RuntimeError(f"frozen HGBR IDs do not align in {month}")
    prediction = cached.raw_prediction.to_numpy(dtype=float)
    if not np.isfinite(prediction).all() or (prediction <= 0.0).any():
        raise RuntimeError(f"frozen HGBR predictions are invalid in {month}")
    return prediction


def _candidate_frame(month: str, track: str, truth: pd.DataFrame,
                     parent: pd.DataFrame, candidate: pd.DataFrame) -> pd.DataFrame:
    expected_ids = truth.ml_num.astype(str).to_numpy()
    for label, frame in (("parent", parent), ("candidate", candidate)):
        if (not np.array_equal(frame.ml_num.astype(str).to_numpy(), expected_ids)
                or not np.array_equal(frame.sold_price.to_numpy(float), truth.sold_price.to_numpy(float))):
            raise RuntimeError(f"{label} rows or labels do not align for {month}/{track}")
    result = truth[["ml_num", "sold_price"]].copy()
    result.insert(0, "track", track)
    result.insert(0, "month", month)
    result["parent_prediction"] = parent.predicted_price.to_numpy(float)
    result["candidate_prediction"] = candidate.predicted_price.to_numpy(float)
    if ((result[["parent_prediction", "candidate_prediction"]] <= 0.0).any().any()
            or not np.isfinite(result[["parent_prediction", "candidate_prediction"]].to_numpy()).all()):
        raise RuntimeError(f"non-positive or non-finite predictions for {month}/{track}")
    result["parent_ape_percent"] = 100.0 * np.abs(
        result.sold_price.to_numpy(float) - result.parent_prediction.to_numpy(float)
    ) / result.sold_price.to_numpy(float)
    result["candidate_ape_percent"] = 100.0 * np.abs(
        result.sold_price.to_numpy(float) - result.candidate_prediction.to_numpy(float)
    ) / result.sold_price.to_numpy(float)
    return result


def _track_result(rows: pd.DataFrame) -> dict:
    parent = campaign._metrics(rows.sold_price.to_numpy(float), rows.parent_prediction.to_numpy(float))
    candidate = campaign._metrics(rows.sold_price.to_numpy(float), rows.candidate_prediction.to_numpy(float))
    relative = campaign._relative_gain(parent["mdape_percent"], candidate["mdape_percent"])
    mape_change = 100.0 * (candidate["mape_percent"] - parent["mape_percent"]) / parent["mape_percent"]
    monthly = {}
    for month in SCREEN_MONTHS:
        part = rows.loc[rows.month.eq(month)]
        parent_metric = campaign._metrics(part.sold_price.to_numpy(float), part.parent_prediction.to_numpy(float))
        candidate_metric = campaign._metrics(part.sold_price.to_numpy(float), part.candidate_prediction.to_numpy(float))
        monthly[month] = {
            "parent": parent_metric,
            "candidate": candidate_metric,
            "relative_mdape_improvement_percent": campaign._relative_gain(
                parent_metric["mdape_percent"], candidate_metric["mdape_percent"]),
        }
    monthly_improvements = [monthly[month]["relative_mdape_improvement_percent"] for month in SCREEN_MONTHS]
    return {
        "parent": parent,
        "candidate": candidate,
        "relative_mdape_improvement_percent": relative,
        "mape_relative_change_percent": mape_change,
        "monthly": monthly,
        "qualifies_screen": qualifies_screen(relative, monthly_improvements, mape_change),
    }


def _render_report(result: dict) -> str:
    lines = [
        "# BILL <7% / <4% 持续研究：R03 CatBoost 1000/L2=10 双月筛选", "",
        f"Campaign：`{result['campaign_id']}`；run：`{result['run_id']}`。",
        f"CatBoost 配置：`{json.dumps(result['catboost_config'], ensure_ascii=False, sort_keys=True)}`。",
        "历史 G15 的 600/L2=3 配置与当前 1000/L2=10 配置不同；本次按当前 campaign 注册值运行。",
        "历史父预测文件的 sidecar 代码哈希与当前工作区不同，作为冻结的 walk-forward 配对比较器，不称为当前代码的重新复现。",
        "", "## 双月合并结果", "",
        "| 赛道 | 父 MdAPE | 候选 MdAPE | 相对改善 | MAPE变化 | 筛选通过 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for track, score in result["tracks"].items():
        lines.append(
            f"| {track} | {score['parent']['mdape_percent']:.5f}% | "
            f"{score['candidate']['mdape_percent']:.5f}% | "
            f"{score['relative_mdape_improvement_percent']:+.3f}% | "
            f"{score['mape_relative_change_percent']:+.3f}% | "
            f"{'是' if score['qualifies_screen'] else '否'} |"
        )
    lines.extend(["", "## 各筛选月份", "", "| 赛道 | 月份 | 父 MdAPE | 候选 MdAPE | 相对改善 |", "|---|---|---:|---:|---:|"])
    for track, score in result["tracks"].items():
        for month, monthly in score["monthly"].items():
            lines.append(
                f"| {track} | {month} | {monthly['parent']['mdape_percent']:.5f}% | "
                f"{monthly['candidate']['mdape_percent']:.5f}% | "
                f"{monthly['relative_mdape_improvement_percent']:+.3f}% |"
            )
    lines.extend([
        "", "## 运行边界", "",
        f"真实 learner fit starts：{result['fit_starts']}；预留折/隐藏标签已保持封存。",
        "双月筛选要求 pooled MdAPE 相对改善至少 0.25%、两月均改善、MAPE 相对退化不超过 1%。",
        "只有过筛的赛道才进入八个月开发确认；本报告自身不改变父模型。",
        "",
        "- 逐房配对结果：`paired_predictions.csv`",
        "- 参数/来源/代码哈希：`run_manifest.json`",
        "- Campaign 事件账本：`../../fits.jsonl`",
        "",
    ])
    return "\n".join(lines)


def run_r03_screen() -> dict:
    from data import hide_answers
    from experiments.bill_greedy import runner as greedy

    source = _source_identity()
    if metadata.version("catboost") != "1.2.10":
        raise RuntimeError(f"expected CatBoost 1.2.10, found {metadata.version('catboost')}")
    run_started_at = datetime.now(timezone.utc)
    run_id = "bill_target74_r03_catboost_1000_l2_10_" + run_started_at.strftime("%Y%m%dT%H%M%SZ")
    batch_root = campaign.CAMPAIGN_ROOT / "batches" / run_id
    batch_root.mkdir(parents=True, exist_ok=False)
    card = "R03_full_feature_CatBoost_1000_l2_10"
    campaign._append({"event": "card_start", "card": card, "run_id": run_id,
                      "screen_months": list(SCREEN_MONTHS), "fit_start_count": 0,
                      "catboost_config": CATBOOST_CONFIG})

    original_variant = dict(greedy.VARIANTS["G15_catboost"])
    original_fit_record = greedy._fit_record
    original_candidate_path = greedy._candidate_path
    original_retry_tag = greedy._screen_fit_retry_tag
    original_hgbr = greedy._frozen_hgbr_predictions
    greedy.VARIANTS["G15_catboost"] = {**original_variant, "model_config": dict(CATBOOST_CONFIG)}
    greedy._candidate_path = lambda variant, track, month: batch_root / "candidate_cache" / track / f"{month}.csv"
    greedy._screen_fit_retry_tag = lambda ledger, variant, track, month: f"{month}-r03-{run_id}"
    greedy._frozen_hgbr_predictions = lambda month, track, train, homes: _verified_hgbr_cache(
        greedy, month, track, train, homes)

    def isolated_fit_record(fit_id: str, fit_fn):
        limits = campaign.PROTOCOL["limits"]
        if campaign.limit_reached(_fit_count(), limits["batch_fit_starts"]):
            raise RuntimeError("R03 batch fit-start cap reached")
        if campaign.limit_exceeded((datetime.now(timezone.utc) - run_started_at).total_seconds(),
                                   None if limits["batch_minutes"] is None
                                   else limits["batch_minutes"] * 60):
            raise RuntimeError("R03 batch wall-clock cap reached before fit start")
        started_at = datetime.now(timezone.utc).isoformat()
        timer = time.perf_counter()
        fit_parts = fit_id.split(":")
        track = fit_parts[1]
        month = fit_parts[-1].split("-r03-", 1)[0]
        campaign_fit_id = r03_fit_id(campaign.PROTOCOL["campaign_id"], track, month)
        campaign._append({"event": "fit_start", "card": card, "run_id": run_id,
                          "fit_id": campaign_fit_id,
                          "parent_fit_id": fit_id, "track": fit_id.split(":")[1],
                          "month": month,
                          "timestamp_utc": started_at, "config": CATBOOST_CONFIG,
                          "data_sha256": source["data_sha256"]})
        try:
            fitted = fit_fn()
        except Exception as exc:
            elapsed = time.perf_counter() - timer
            campaign._append({"event": "fit_complete", "card": card, "run_id": run_id,
                              "fit_id": campaign_fit_id,
                              "status": "failed", "seconds": elapsed,
                              "error": f"{type(exc).__name__}: {exc}"})
            raise
        elapsed = time.perf_counter() - timer
        campaign._append({"event": "fit_complete", "card": card, "run_id": run_id,
                          "fit_id": campaign_fit_id,
                          "status": "complete" if not campaign.limit_exceeded(
                              elapsed, limits["single_fit_seconds"]) else "over_limit",
                          "seconds": elapsed, "prediction_sha256": fitted.get("prediction_sha256")})
        fitted["fit_seconds"] = elapsed
        if campaign.limit_exceeded(elapsed, limits["single_fit_seconds"]):
            raise TimeoutError(f"fit finished after the registered {campaign.PROTOCOL['limits']['single_fit_seconds']}s ceiling")
        return fitted

    greedy._fit_record = isolated_fit_record
    rows_by_track: dict[str, list[pd.DataFrame]] = {track: [] for track in PARENTS}
    try:
        for month in SCREEN_MONTHS:
            listings, train, _, truth, heldout = greedy._load_fold(month)
            if not listings.date_sold.max() < (pd.Period(month, freq="M") + 1).start_time:
                raise RuntimeError(f"fold input contains future sale rows for {month}")
            for track, parent_id in PARENTS.items():
                homes = hide_answers(heldout, track)
                parent_path = campaign._row_prediction_path(month, track)
                sidecar = parent_path.with_suffix(".json")
                meta = json.loads(sidecar.read_text(encoding="utf-8"))
                if meta.get("data_sha256") != campaign.PROTOCOL["data_sha256"]:
                    raise RuntimeError(f"parent data hash mismatch for {month}/{track}")
                parent = greedy._load_parent_prediction(month, track, homes, truth, parent_id)
                candidate, fit = greedy._screen_variant(
                    month, track, train, homes, truth, "G15_catboost", parent_id=parent_id)
                if fit is None:
                    raise RuntimeError("unexpected candidate cache hit in a fresh R03 run")
                paired = _candidate_frame(month, track, truth, parent, candidate)
                rows_by_track[track].append(paired)
                campaign._append({"event": "screen_fold_complete", "card": card, "run_id": run_id,
                                  "track": track, "month": month,
                                  "parent_prediction_sha256": _sha256(parent_path),
                                  "candidate_prediction_sha256": hashlib.sha256(
                                      candidate.predicted_price.to_numpy(dtype="float64").tobytes()).hexdigest(),
                                  "parent_id": parent_id})
    finally:
        greedy.VARIANTS["G15_catboost"] = original_variant
        greedy._fit_record = original_fit_record
        greedy._candidate_path = original_candidate_path
        greedy._screen_fit_retry_tag = original_retry_tag
        greedy._frozen_hgbr_predictions = original_hgbr

    paired_all = pd.concat([frame for frames in rows_by_track.values() for frame in frames], ignore_index=True)
    paired_path = batch_root / "paired_predictions.csv"
    paired_all.to_csv(paired_path, index=False)
    track_results = {track: _track_result(pd.concat(frames, ignore_index=True))
                     for track, frames in rows_by_track.items()}
    result = {
        "campaign_id": campaign.PROTOCOL["campaign_id"],
        "run_id": run_id,
        "card": card,
        "parents": PARENTS,
        "screen_months": list(SCREEN_MONTHS),
        "catboost_config": CATBOOST_CONFIG,
        "tracks": track_results,
        "fit_starts": _fit_count(),
        "reserved_months_labels_opened": False,
        "hidden_labels_opened": False,
        "parent_provenance_note": "Historical development parent CSV sidecar code hashes differ from the current worktree; they are retained as paired walk-forward comparators.",
        "data_sha256": source["data_sha256"],
    }
    results_path = batch_root / "r03_screen_results.json"
    results_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    source_files = [
        campaign.PROTOCOL_PATH,
        campaign.PACKAGE / "r03_catboost.py",
        campaign.PACKAGE / "runner.py",
        ROOT / "experiments/bill_greedy/runner.py",
        ROOT / "experiments/bill_greedy/history.py",
        ROOT / "experiments/model_optimization_v2/learners.py",
        ROOT / "experiments/model_optimization_v2/worker.py",
        ROOT / "model_candidate_v2/features.py",
        ROOT / "model_candidate_v2/winner_config.json",
    ]
    manifest = {
        "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
        "source_identity": source,
        "code_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): _sha256(path) for path in source_files},
        "config_sha256": hashlib.sha256(json.dumps(CATBOOST_CONFIG, sort_keys=True).encode()).hexdigest(),
        "runtime": {"python": platform.python_version(), "catboost": metadata.version("catboost"),
                    "numpy": np.__version__, "pandas": pd.__version__,
                    "lightgbm": metadata.version("lightgbm"),
                    "scikit_learn": metadata.version("scikit-learn")},
        "parent_artifacts": {
            f"{track}/{month}": {
                "path": str(campaign._row_prediction_path(month, track).relative_to(ROOT)),
                "csv_sha256": _sha256(campaign._row_prediction_path(month, track)),
                "sidecar_sha256": _sha256(campaign._row_prediction_path(month, track).with_suffix(".json")),
            }
            for track in PARENTS for month in SCREEN_MONTHS
        },
        "frozen_hgbr_artifacts": {
            month: {
                "path": str((greedy.OUTPUT / "frozen_hgbr/list_aware" / f"{month}.csv").relative_to(ROOT)),
                "csv_sha256": _sha256(greedy.OUTPUT / "frozen_hgbr/list_aware" / f"{month}.csv"),
            }
            for month in SCREEN_MONTHS
        },
        "paired_prediction_csv_sha256": _sha256(paired_path),
        "results_sha256": _sha256(results_path),
        "fit_starts": result["fit_starts"],
    }
    (batch_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (batch_root / "research_report.md").write_text(_render_report(result), encoding="utf-8")
    passed = {track: score["qualifies_screen"] for track, score in track_results.items()}
    next_action = (
        "run eight-month R03 development confirmation for: " + ", ".join(track for track, ok in passed.items() if ok)
        if any(passed.values()) else
        "R03 did not qualify; proceed to registered R04 feature-group ablations, starting with the error-supported track/feature groups"
    )
    campaign._append({"event": "card_complete", "card": card, "run_id": run_id,
                      "fit_starts": result["fit_starts"], "qualifies_by_track": passed,
                      "result_path": str(results_path.relative_to(ROOT)),
                      "paired_prediction_csv_sha256": manifest["paired_prediction_csv_sha256"]})
    campaign._write_progress("r03_screen_complete", next_action=next_action,
                             r03_result=str(results_path.relative_to(ROOT)))
    return result


def record_r01_dependency_blocked() -> None:
    ledger = []
    if campaign.LEDGER.exists():
        ledger = [json.loads(line) for line in campaign.LEDGER.read_text(encoding="utf-8").splitlines() if line.strip()]
    if any(row.get("event") == "card_dependency_blocked" and row.get("card") == "R01_XGBoost" for row in ledger):
        return
    campaign._append({
        "event": "card_dependency_blocked",
        "card": "R01_XGBoost",
        "fit_start_count": 0,
        "diagnostics": [
            "No xgboost module in .venv-opt, project .venv, parent .venv, or system Python.",
            "No xgboost package in the local pip cache.",
            "Configured package index returned no matching xgboost version.",
            "Direct official PyPI query also returned no matching xgboost version.",
        ],
        "decision": "dependency_blocked; do not spend more time on package installation in this campaign",
    })


def main() -> None:
    record_r01_dependency_blocked()
    try:
        result = run_r03_screen()
    except Exception as exc:
        campaign._append({"event": "card_failed", "card": "R03_full_feature_CatBoost_1000_l2_10",
                          "error": f"{type(exc).__name__}: {exc}"})
        campaign._write_progress("r03_failed", next_action="inspect the R03 failure record before any retry",
                                 latest_error=f"{type(exc).__name__}: {exc}")
        raise
    for track, score in result["tracks"].items():
        print(f"{track}: parent MdAPE={score['parent']['mdape_percent']:.5f}% "
              f"candidate={score['candidate']['mdape_percent']:.5f}% "
              f"relative_gain={score['relative_mdape_improvement_percent']:+.3f}% "
              f"MAPE_change={score['mape_relative_change_percent']:+.3f}% "
              f"screen_pass={score['qualifies_screen']}", flush=True)
    print(f"report={campaign.CAMPAIGN_ROOT / 'batches' / result['run_id'] / 'research_report.md'}", flush=True)


if __name__ == "__main__":
    main()
