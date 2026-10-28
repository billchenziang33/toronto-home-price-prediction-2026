"""R40: choose a walk-forward R22/R26 blend from strictly earlier OOF rows."""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.bill_target74.r19_expert_weight import _write_json


CARD = "R40_past_oof_r26_blend"
TRACK = "off_market"
SEED_MONTHS = ("2022-06", "2022-11")
SCREEN_MONTHS = ("2023-03", "2024-06")
DEVELOPMENT_MONTHS = ("2023-03", "2023-07", "2023-11", "2024-03",
                      "2024-06", "2024-09", "2025-01", "2025-03")
MINIMUM_OOF_ROWS = 1000
WEIGHTS = tuple(round(float(value), 2) for value in np.linspace(0.0, 1.0, 21))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_checked_result(path: Path, result: dict, manifest: dict,
                         expected_card: str, expected_data_sha: str) -> pd.DataFrame:
    paired_path = path.parent / "paired_predictions.csv"
    if (result.get("card") != expected_card
            or manifest.get("run_id") != result.get("run_id")
            or manifest.get("result_sha256") != _sha256(path)
            or manifest.get("paired_predictions_sha256") != _sha256(paired_path)):
        raise RuntimeError(f"{expected_card} result or paired prediction hash failed verification")
    source_identity = manifest.get("source_identity", {})
    actual_data_sha = result.get("data_sha256", source_identity.get("data_sha256"))
    if actual_data_sha != expected_data_sha:
        raise RuntimeError(f"{expected_card} result uses a different data snapshot")
    if result.get("reserved_months_labels_opened") or result.get("hidden_labels_opened"):
        raise RuntimeError(f"{expected_card} run touched a sealed label set")
    return pd.read_csv(paired_path, dtype={"ml_num": str})


def _score(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    ape = 100.0 * np.abs(predicted - actual) / actual
    return {"mdape_percent": float(np.median(ape)), "mape_percent": float(np.mean(ape))}


def blend(parent: np.ndarray, variant: np.ndarray, weight: float) -> np.ndarray:
    parent = np.asarray(parent, dtype=float)
    variant = np.asarray(variant, dtype=float)
    if parent.ndim != 1 or variant.shape != parent.shape:
        raise ValueError("R22 and R26 predictions must be aligned one-dimensional vectors")
    if (not np.isfinite(parent).all() or not np.isfinite(variant).all()
            or (parent <= 0.0).any() or (variant <= 0.0).any()):
        raise ValueError("R22 and R26 predictions must be finite positive prices")
    if not np.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError("R26 blend weight must be between 0 and 1")
    return np.round((1.0 - weight) * parent + weight * variant)


def select_weight(oof: pd.DataFrame, cutoff, *, minimum_rows: int = MINIMUM_OOF_ROWS,
                  max_mape_regression_percent: float = 1.0) -> dict:
    required = {"month", "ml_num", "date_sold", "sold_price",
                "parent_prediction", "variant_prediction"}
    if missing := required - set(oof.columns):
        raise ValueError(f"OOF rows are missing columns: {sorted(missing)}")
    cutoff = pd.Timestamp(cutoff)
    dates = pd.to_datetime(oof.date_sold, errors="raise")
    if not dates.dt.to_period("M").astype(str).eq(oof.month.astype(str)).all():
        raise ValueError("OOF month labels do not match sale dates")
    if oof.duplicated(["month", "ml_num"]).any():
        raise ValueError("OOF rows must be unique by month and listing ID")
    past = oof.loc[dates < cutoff].copy()
    if len(past):
        values = past[["sold_price", "parent_prediction", "variant_prediction"]].to_numpy(float)
        if (not np.isfinite(values).all() or (values <= 0.0).any()):
            raise ValueError("OOF prices must be finite and positive")
    base = _score(past.sold_price.to_numpy(float), past.parent_prediction.to_numpy(float)) if len(past) else None
    empty = {
        "weight": 0.0,
        "oof_rows": int(len(past)),
        "oof_months": sorted(past.month.astype(str).unique().tolist()),
        "parent": base,
        "selected": base,
        "selected_mape_relative_change_percent": 0.0,
        "eligible_weights": [0.0],
        "reason": "insufficient_past_oof_rows",
    }
    if len(past) < minimum_rows:
        return empty
    actual = past.sold_price.to_numpy(float)
    parent = past.parent_prediction.to_numpy(float)
    candidate = past.variant_prediction.to_numpy(float)
    eligible = []
    for weight in WEIGHTS:
        prediction = blend(parent, candidate, weight)
        metrics = _score(actual, prediction)
        mape_change = 100.0 * (metrics["mape_percent"] - base["mape_percent"]) / base["mape_percent"]
        if mape_change <= max_mape_regression_percent:
            eligible.append({"weight": weight, "metrics": metrics,
                             "mape_relative_change_percent": float(mape_change)})
    if not eligible:
        return {**empty, "reason": "no_weight_passed_mape_guard"}
    selected = min(eligible, key=lambda row: (
        row["metrics"]["mdape_percent"], row["metrics"]["mape_percent"], row["weight"],
    ))
    return {
        **empty,
        "weight": selected["weight"],
        "selected": selected["metrics"],
        "selected_mape_relative_change_percent": selected["mape_relative_change_percent"],
        "eligible_weights": [row["weight"] for row in eligible],
        "reason": "selected_from_strictly_past_oof",
    }


def _seed_oof(month: str, run_id: str, output: Path, source: dict,
              campaign, greedy, r22, r26, hide_answers,
              fit_details: list[dict]) -> pd.DataFrame:
    _, train, _, truth, heldout = greedy._load_fold(month)
    cutoff = pd.Period(month, freq="M").start_time
    train_dates = pd.to_datetime(train.date_sold, errors="raise")
    if train_dates.ge(cutoff).any():
        raise RuntimeError(f"seed OOF training rows reach the {month} cutoff")
    homes = hide_answers(heldout, TRACK).reset_index(drop=True)
    ids = homes.ml_num.astype(str).to_numpy()
    if not np.array_equal(ids, truth.ml_num.astype(str).to_numpy()):
        raise RuntimeError(f"seed OOF homes and labels do not align for {month}")
    detached_train = greedy._specialist_mask(train, "Detached")
    detached_eval = greedy._specialist_mask(homes, "Detached")
    expected_support = campaign.PROTOCOL["cards"][CARD]["seed_oof_route_support"][month]
    actual_support = {
        "training_rows": int(len(train)),
        "detached_training_rows": int(detached_train.sum()),
        "evaluation_rows": int(len(homes)),
        "detached_evaluation_rows": int(detached_eval.sum()),
    }
    if actual_support != expected_support:
        raise RuntimeError(
            f"R40 preregistered OOF support changed for {month}: "
            f"expected {expected_support}, found {actual_support}"
        )
    models = {"R22": (r22.VARIANT, r22._install_text_features),
              "R26": (r26.VARIANT, r26._install_wide_word_features)}
    predictions: dict[str, np.ndarray] = {}
    fit_context: dict[str, object] = {}
    old_fit_record = greedy._fit_record

    def record_fit(_parent_id: str, fit_fn):
        context = fit_context.copy()
        fit_id = (f"{campaign.PROTOCOL['campaign_id']}:{CARD}:past_oof_seed:"
                  f"{context['month']}:{context['model']}:{context['component']}:{run_id}")
        campaign._append({
            "event": "fit_start", "card": CARD, "stage": "past_oof_seed",
            "run_id": run_id, "fit_id": fit_id, "month": context["month"],
            "model": context["model"], "component": context["component"],
            "training_rows": context["training_rows"],
            "evaluation_rows": context["evaluation_rows"],
            "data_sha256": source["data_sha256"],
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        started = time.perf_counter()
        try:
            fitted = fit_fn()
        except Exception as exc:
            campaign._append({
                "event": "fit_complete", "card": CARD, "stage": "past_oof_seed",
                "run_id": run_id, "fit_id": fit_id, "month": context["month"],
                "model": context["model"], "component": context["component"],
                "status": "failed", "seconds": time.perf_counter() - started,
                "error": f"{type(exc).__name__}: {exc}",
                "reserved_months_labels_opened": False, "hidden_labels_opened": False,
            })
            raise
        detail = {
            "fit_id": fit_id, "month": context["month"],
            "model": context["model"], "component": context["component"],
            "seconds": time.perf_counter() - started,
            "training_rows": fitted["training_rows"],
            "evaluation_rows": fitted["evaluation_rows"],
            "feature_columns": fitted["feature_columns"],
            "model_family": fitted["model_family"],
            "model_config": fitted["model_config"],
            "prediction_sha256": fitted["prediction_sha256"],
        }
        fit_details.append(detail)
        campaign._append({
            "event": "fit_complete", "card": CARD, "stage": "past_oof_seed",
            "run_id": run_id, **detail, "status": "complete",
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        return fitted

    greedy._fit_record = record_fit
    try:
        for model_name, (variant, install_text) in models.items():
            components: dict[str, dict] = {}
            fit_jobs = (
                ("base", train, homes, "base"),
                ("expert", train.loc[detached_train].copy(),
                 homes.loc[detached_eval].copy(), "expert"),
            )
            for component, fit_rows, eval_rows, tag in fit_jobs:
                transformer, restore = install_text(fit_rows, eval_rows, TRACK)
                fit_context.update(month=month, model=model_name, component=component,
                                   training_rows=len(fit_rows), evaluation_rows=len(eval_rows))
                try:
                    fitted = greedy._fit_history_candidate(
                        month, TRACK, train, eval_rows, fit_rows=fit_rows,
                        base_variant="G06_lot_garage", variant=variant,
                        fit_tag=f"R40-{model_name}-{month}-{tag}",
                    )
                finally:
                    restore()
                components[component] = fitted
            predictions[model_name] = greedy._blend_specialist_predictions(
                np.round(components["base"]["predictions"]),
                np.round(components["expert"]["predictions"]),
                homes, "Detached", 0.75,
            )
    finally:
        greedy._fit_record = old_fit_record

    date_by_id = heldout.assign(ml_num=heldout.ml_num.astype(str)).set_index("ml_num").loc[ids, "date_sold"]
    frame = pd.DataFrame({
        "month": month,
        "ml_num": ids,
        "date_sold": pd.to_datetime(date_by_id.to_numpy(), errors="raise"),
        "sold_price": truth.sold_price.to_numpy(float),
        "parent_prediction": predictions["R22"],
        "variant_prediction": predictions["R26"],
    })
    if frame.duplicated(["month", "ml_num"]).any():
        raise RuntimeError(f"duplicate seed OOF IDs for {month}")
    frame.to_csv(output / "oof_seed" / f"{month}.csv", index=False)
    return frame


def _development_oof(months: tuple[str, ...], r22_rows: pd.DataFrame,
                     r26_rows: pd.DataFrame, greedy) -> pd.DataFrame:
    frames = []
    for month in months:
        left = r22_rows.loc[r22_rows.month.eq(month) & r22_rows.track.eq(TRACK)].copy()
        right = r26_rows.loc[r26_rows.month.eq(month) & r26_rows.track.eq(TRACK)].copy()
        if not len(left) or left.ml_num.astype(str).tolist() != right.ml_num.astype(str).tolist():
            raise RuntimeError(f"R22/R26 development OOF rows do not align for {month}")
        if (not np.array_equal(left.sold_price.to_numpy(float), right.sold_price.to_numpy(float))
                or not np.array_equal(left.candidate_prediction.to_numpy(float),
                                      right.parent_prediction.to_numpy(float))):
            raise RuntimeError(f"R26 is not aligned to the frozen R22 parent for {month}")
        _, _, _, truth, heldout = greedy._load_fold(month)
        if (left.ml_num.astype(str).tolist() != truth.ml_num.astype(str).tolist()
                or not np.array_equal(left.sold_price.to_numpy(float), truth.sold_price.to_numpy(float))):
            raise RuntimeError(f"saved development OOF rows do not match data fold {month}")
        ids = left.ml_num.astype(str).to_numpy()
        date_by_id = heldout.assign(ml_num=heldout.ml_num.astype(str)).set_index("ml_num").loc[ids, "date_sold"]
        frames.append(pd.DataFrame({
            "month": month, "ml_num": ids,
            "date_sold": pd.to_datetime(date_by_id.to_numpy(), errors="raise"),
            "sold_price": left.sold_price.to_numpy(float),
            "parent_prediction": left.candidate_prediction.to_numpy(float),
            "variant_prediction": right.candidate_prediction.to_numpy(float),
        }))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["month", "ml_num", "date_sold", "sold_price", "parent_prediction", "variant_prediction"])


def run_r40_screen() -> dict:
    from data import hide_answers
    from experiments.bill_target74 import r03_catboost as helpers
    from experiments.bill_target74 import r04_ablation as budget
    from experiments.bill_target74 import r22_remarks_tfidf as r22
    from experiments.bill_target74 import r23_word_char_tfidf as shared
    from experiments.bill_target74 import r26_word_tfidf_512 as r26
    from experiments.bill_target74 import runner as campaign
    from experiments.bill_greedy import runner as greedy

    card = campaign.PROTOCOL["cards"].get(CARD)
    if not card or card.get("status") != "registered":
        raise RuntimeError("R40 must be preregistered before fitting")
    source = budget._source_identity()
    if source["data_sha256"] != campaign.PROTOCOL["data_sha256"]:
        raise RuntimeError("R40 data hash differs from the active campaign")

    r22_screen, r22_screen_path, r22_screen_manifest = r22._verified_screen_result(campaign)
    r26_screen, r26_screen_path, r26_screen_manifest = r26._verified_r26_screen_result(campaign)
    r22_dev, r22_dev_path, r22_dev_manifest = r22._verified_development_result(campaign)
    progress = json.loads(campaign.STATE.read_text(encoding="utf-8"))
    r26_dev_relative = progress.get("r26_development_result")
    if not r26_dev_relative:
        raise RuntimeError("R40 requires the completed R26 eight-month development OOF run")
    r26_dev_path = campaign.ROOT / r26_dev_relative
    r26_dev = json.loads(r26_dev_path.read_text(encoding="utf-8"))
    r26_dev_manifest = json.loads((r26_dev_path.parent / "run_manifest.json").read_text(encoding="utf-8"))
    if r26_dev.get("card") != r26.CARD or r26_dev.get("stage") != "development_confirmation":
        raise RuntimeError("R26 development result has an unexpected identity or stage")

    r22_screen_rows = _read_checked_result(r22_screen_path, r22_screen, r22_screen_manifest,
                                           r22.CARD, source["data_sha256"])
    r26_screen_rows = _read_checked_result(r26_screen_path, r26_screen, r26_screen_manifest,
                                           r26.CARD, source["data_sha256"])
    r22_dev_rows = _read_checked_result(r22_dev_path, r22_dev, r22_dev_manifest,
                                        r22.CARD, source["data_sha256"])
    r26_dev_rows = _read_checked_result(r26_dev_path, r26_dev, r26_dev_manifest,
                                        r26.CARD, source["data_sha256"])

    run_id = "bill_target74_r40_past_oof_r26_blend_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = campaign.CAMPAIGN_ROOT / "batches" / run_id
    output.mkdir(parents=True, exist_ok=False)
    (output / "oof_seed").mkdir()
    provenance = output / "provenance"
    provenance.mkdir()
    for name, path in (
        ("r22_screen_results.json", r22_screen_path),
        ("r22_development_results.json", r22_dev_path),
        ("r26_screen_results.json", r26_screen_path),
        ("r26_development_results.json", r26_dev_path),
    ):
        shutil.copy2(path, provenance / name)
    for name, manifest in (("r22_screen_manifest.json", r22_screen_manifest),
                           ("r22_development_manifest.json", r22_dev_manifest),
                           ("r26_screen_manifest.json", r26_screen_manifest),
                           ("r26_development_manifest.json", r26_dev_manifest)):
        (provenance / name).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    source_files = [
        campaign.PROTOCOL_PATH,
        campaign.PACKAGE / "r40_past_oof_r26_blend.py",
        campaign.PACKAGE / "r22_remarks_tfidf.py",
        campaign.PACKAGE / "r23_word_char_tfidf.py",
        campaign.PACKAGE / "r26_word_tfidf_512.py",
        campaign.PACKAGE / "r03_catboost.py",
        campaign.PACKAGE / "r04_ablation.py",
        campaign.PACKAGE / "r19_expert_weight.py",
        campaign.PACKAGE / "runner.py",
        campaign.ROOT / "experiments/bill_greedy/runner.py",
        campaign.ROOT / "experiments/bill_greedy/history.py",
        campaign.ROOT / "model_candidate_v2/features.py",
        campaign.ROOT / "experiments/model_optimization_v2/learners.py",
        campaign.ROOT / "experiments/model_optimization_v2/pipeline.py",
        campaign.ROOT / "experiments/model_optimization_v2/folds.py",
    ]
    code_hashes = {
        str(path.relative_to(campaign.ROOT)).replace("\\", "/"): _sha256(path)
        for path in source_files
    }
    for path in source_files:
        relative = str(path.relative_to(campaign.ROOT)).replace("\\", "__").replace("/", "__")
        shutil.copy2(path, provenance / f"{relative}.source")

    campaign._append({
        "event": "card_start", "card": CARD, "run_id": run_id, "track": TRACK,
        "parent_run_id": r22_screen["run_id"], "variant_run_id": r26_screen["run_id"],
        "screen_months": list(SCREEN_MONTHS), "seed_oof_months": list(SEED_MONTHS),
        "expected_fit_starts": 8,
        "weight_grid": list(WEIGHTS), "minimum_oof_rows": MINIMUM_OOF_ROWS,
        "mape_regression_guard_percent": 1.0,
        "strictly_past_only_selection": True,
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })

    seed_frames = []
    fit_details = []
    try:
        for month in SEED_MONTHS:
            frame = _seed_oof(month, run_id, output, source, campaign, greedy,
                              r22, r26, hide_answers, fit_details)
            seed_frames.append(frame)
    except Exception as exc:
        failure_path = output / "failure_checkpoint.json"
        _write_json(failure_path, {
            "run_id": run_id, "card": CARD,
            "stage": "past_oof_seed", "error": f"{type(exc).__name__}: {exc}",
            "fit_starts": len(fit_details), "fit_details": fit_details,
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        campaign._append({
            "event": "card_failed", "card": CARD, "run_id": run_id,
            "fit_starts": len(fit_details), "error": f"{type(exc).__name__}: {exc}",
            "failure_path": str(failure_path.relative_to(campaign.ROOT)).replace("\\", "/"),
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        campaign._write_progress("r40_seed_oof_failed",
                                 next_action="inspect R40 failure checkpoint before any retry",
                                 latest_fit_count=len(fit_details))
        raise

    if len(fit_details) != 8:
        raise RuntimeError(f"R40 expected 8 seed OOF fits, found {len(fit_details)}")

    development_months = tuple(month for month in DEVELOPMENT_MONTHS
                                if pd.Period(month, freq="M").start_time
                                < pd.Period(SCREEN_MONTHS[-1], freq="M").start_time)
    development_history = _development_oof(
        development_months, r22_dev_rows, r26_dev_rows, greedy)
    oof = pd.concat([*seed_frames, development_history], ignore_index=True)
    oof_path = output / "past_oof_predictions.csv"
    oof.to_csv(oof_path, index=False)

    selections = []
    paired_parts = []
    for month in SCREEN_MONTHS:
        cutoff = pd.Period(month, freq="M").start_time
        selection = select_weight(oof, cutoff)
        selections.append({"screen_month": month, "cutoff_exclusive": cutoff.isoformat(), **selection})
        _, _, _, truth, _ = greedy._load_fold(month)
        parent = shared.r22_reference_frame(r22_screen_rows, truth, month, TRACK)
        alternate = r26_screen_rows.loc[
            r26_screen_rows.month.eq(month) & r26_screen_rows.track.eq(TRACK)
        ].copy()
        if (alternate.ml_num.astype(str).tolist() != parent.ml_num.astype(str).tolist()
                or not np.array_equal(alternate.sold_price.to_numpy(float), truth.sold_price.to_numpy(float))
                or not np.array_equal(alternate.parent_prediction.to_numpy(float),
                                      parent.predicted_price.to_numpy(float))):
            raise RuntimeError(f"R26 screen predictions do not align to R22 in {month}")
        prediction = blend(parent.predicted_price.to_numpy(float),
                           alternate.candidate_prediction.to_numpy(float), selection["weight"])
        candidate = pd.DataFrame({
            "ml_num": parent.ml_num.astype(str).to_numpy(),
            "sold_price": truth.sold_price.to_numpy(float),
            "predicted_price": prediction,
        })
        paired = helpers._candidate_frame(month, TRACK, truth, parent, candidate)
        paired["r26_blend_weight"] = selection["weight"]
        paired_parts.append(paired)

    paired = pd.concat(paired_parts, ignore_index=True)
    paired_path = output / "paired_predictions.csv"
    paired.to_csv(paired_path, index=False)
    rows = shared._track_result(paired)
    result = {
        "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
        "card": CARD, "track": TRACK, "parent_run_id": r22_screen["run_id"],
        "variant_screen_run_id": r26_screen["run_id"],
        "screen_months": list(SCREEN_MONTHS), "seed_oof_months": list(SEED_MONTHS),
        "fit_starts": len(fit_details), "tracks": {TRACK: rows},
        "weight_selections": selections,
        "oof_months_for_screen": sorted(oof.month.astype(str).unique().tolist()),
        "minimum_oof_rows": MINIMUM_OOF_ROWS,
        "weight_grid": list(WEIGHTS), "mape_regression_guard_percent": 1.0,
        "strictly_past_only_selection": True,
        "fit_details": fit_details,
        "data_sha256": source["data_sha256"],
        "reserved_months_labels_opened": False,
        "reserve_validation_eligible": False, "hidden_labels_opened": False,
    }
    result_path = output / "screen_results.json"
    _write_json(result_path, result)
    manifest = {
        "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
        "source_identity": source, "code_sha256": code_hashes,
        "parent_run_id": r22_screen["run_id"], "variant_run_id": r26_screen["run_id"],
        "input_result_sha256": {
            "r22_screen": _sha256(r22_screen_path),
            "r22_development": _sha256(r22_dev_path),
            "r26_screen": _sha256(r26_screen_path),
            "r26_development": _sha256(r26_dev_path),
        },
        "past_oof_predictions_sha256": _sha256(oof_path),
        "paired_predictions_sha256": _sha256(paired_path),
        "result_sha256": _sha256(result_path),
        "fit_starts": len(fit_details), "fit_details": fit_details,
        "reserved_months_labels_opened": False,
        "reserve_validation_eligible": False, "hidden_labels_opened": False,
    }
    _write_json(output / "run_manifest.json", manifest)
    report = [
        "# R40：只用严格过去 OOF 行选择 R22/R26 混合权重", "",
        f"Run：`{run_id}`；父模型是经过核验的 R22，候选模型是经过核验的 R26 512 词项模型。", "",
        "R40 不重新拟合筛选月模型。它在 `0.00..1.00` 的 0.05 网格上，按每个筛选截止日之前的 OOF MdAPE 选择 R26 权重，并要求 OOF MAPE 恶化不超过 1%。并列时优先较低 MAPE，再优先较小 R26 权重。", "",
        "筛选折之外的 R26 预测保持为 R22；screen 月份不参与自身权重选择。", "",
        "| 月份 | 仅早于截止日的 OOF 月份 | OOF 行数 | 选中 R26 权重 | OOF MdAPE 父→混合 | OOF MAPE变化 |", "|---|---|---:|---:|---:|---:|",
    ]
    for row in selections:
        report.append(
            f"| {row['screen_month']} | {', '.join(row['oof_months'])} | {row['oof_rows']} | "
            f"{row['weight']:.2f} | {row['parent']['mdape_percent']:.5f}%→{row['selected']['mdape_percent']:.5f}% | "
            f"{row['selected_mape_relative_change_percent']:+.3f}% |"
        )
    report.extend(["", "## 双月筛选结果", "",
                   f"R22 pooled MdAPE：{rows['parent']['mdape_percent']:.5f}%；",
                   f"R40 pooled MdAPE：{rows['candidate']['mdape_percent']:.5f}%；",
                   f"相对改善：{rows['relative_mdape_improvement_percent']:+.3f}%；",
                   f"MAPE 相对变化：{rows['mape_relative_change_percent']:+.3f}%；",
                   f"筛选门槛通过：{rows['qualifies_screen']}。", "",
                   "筛选月为 `2023-03`、`2024-06`。本轮未读取 2025-05/08 或隐藏标签。",
                   "预留标签因先前完整 parquet 载入事件而不具备独立确认资格。", "",
                   f"配对预测、过去 OOF 行、输入快照、模型 fit 账本与哈希位于 `{output.relative_to(campaign.ROOT)}`。", ""])
    (output / "research_report.md").write_text("\n".join(report), encoding="utf-8")
    campaign._append({
        "event": "card_complete", "card": CARD, "run_id": run_id,
        "fit_starts": len(fit_details),
        "result_path": str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        "qualifies_by_track": {TRACK: rows["qualifies_screen"]},
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    campaign.PROTOCOL["cards"][CARD].update({
        "status": "screen_passed" if rows["qualifies_screen"] else "screen_failed",
        "run_id": run_id, "fit_starts": len(fit_details),
        "result_path": str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        "qualifies_screen": rows["qualifies_screen"],
        "screen_metrics": {
            "parent_mdape_percent": rows["parent"]["mdape_percent"],
            "candidate_mdape_percent": rows["candidate"]["mdape_percent"],
            "relative_mdape_improvement_percent": rows["relative_mdape_improvement_percent"],
            "mape_relative_change_percent": rows["mape_relative_change_percent"],
            "monthly_relative_mdape_improvement_percent": {
                month: values["relative_mdape_improvement_percent"]
                for month, values in rows["monthly"].items()
            },
            "weight_by_screen_month": {
                row["screen_month"]: row["weight"] for row in selections
            },
        },
    })
    _write_json(campaign.PROTOCOL_PATH, campaign.PROTOCOL)
    campaign._write_progress(
        "r40_screen_complete",
        next_action=("run R40 eight-month walk-forward development confirmation with past-only weights"
                     if rows["qualifies_screen"]
                     else "R40 failed screen; preserve R22 and register a distinct off-market hypothesis"),
        r40_result=str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        latest_fit_count=len(fit_details),
    )
    return result


if __name__ == "__main__":
    outcome = run_r40_screen()
    score = outcome["tracks"][TRACK]
    print(
        f"R40: parent_mdape={score['parent']['mdape_percent']:.5f}% "
        f"candidate_mdape={score['candidate']['mdape_percent']:.5f}% "
        f"relative_gain={score['relative_mdape_improvement_percent']:+.3f}% "
        f"pass={score['qualifies_screen']} run={outcome['run_id']}",
        flush=True,
    )
