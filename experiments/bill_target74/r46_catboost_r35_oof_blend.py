"""R46: select a past-only CatBoost/R35 blend for the Detached list-aware route."""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.bill_target74 import r03_catboost as r03
from experiments.bill_target74 import r04_ablation as budget
from experiments.bill_target74 import r24_catboost_word_tfidf as r24
from experiments.bill_target74 import r40_past_oof_r26_blend as weight_selection
from experiments.bill_target74 import r43_listaware_condo_apt_text as parent_verifier
from experiments.bill_target74 import r44_past_oof_r35_shrinkage as r44
from experiments.bill_target74 import r45_listaware_detached_catboost as r45
from experiments.bill_target74 import r23_word_char_tfidf as scoring
from experiments.bill_target74 import runner as campaign
from experiments.bill_greedy import runner as greedy


CARD = "R46_listaware_catboost_r35_past_oof_blend"
TRACK = "list_aware"
HOUSE_TYPE = "Detached"
COMPONENT = "detached_catboost_expert_oof"
SCREEN_MONTHS = ("2023-03", "2024-06")
SEED_MONTHS = ("2022-06", "2022-11")
FIT_OOF_MONTHS = ("2022-06", "2022-11", "2023-07", "2023-11", "2024-03")
PAST_SOURCE_MONTHS = ("2022-06", "2022-11", "2023-03", "2023-07",
                      "2023-11", "2024-03", "2024-06")
MINIMUM_PAST_OOF_ROWS = 1000
CATBOOST_SPECIALIST_WEIGHT = 0.75


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
                    encoding="utf-8")


def _source_files() -> list[Path]:
    return [
        campaign.PROTOCOL_PATH,
        Path(__file__),
        campaign.PACKAGE / "r03_catboost.py",
        campaign.PACKAGE / "r04_ablation.py",
        campaign.PACKAGE / "r22_remarks_tfidf.py",
        campaign.PACKAGE / "r23_word_char_tfidf.py",
        campaign.PACKAGE / "r24_catboost_word_tfidf.py",
        campaign.PACKAGE / "r40_past_oof_r26_blend.py",
        campaign.PACKAGE / "r43_listaware_condo_apt_text.py",
        campaign.PACKAGE / "r44_past_oof_r35_shrinkage.py",
        campaign.PACKAGE / "r45_listaware_detached_catboost.py",
        campaign.PACKAGE / "runner.py",
        campaign.ROOT / "experiments/bill_greedy/runner.py",
        campaign.ROOT / "experiments/bill_greedy/history.py",
        campaign.ROOT / "model_candidate_v2/features.py",
        campaign.ROOT / "model_candidate_v2/winner_config.json",
        campaign.ROOT / "experiments/model_optimization_v2/learners.py",
        campaign.ROOT / "experiments/model_optimization_v2/pipeline.py",
        campaign.ROOT / "experiments/model_optimization_v2/folds.py",
        campaign.ROOT / "experiments/model_optimization_v2/worker.py",
    ]


def _verify_limits_and_card() -> dict:
    card = campaign.PROTOCOL["cards"].get(CARD)
    if not card or card.get("status") != "registered":
        raise RuntimeError("R46 must be registered before fitting")
    limit_keys = (
        "campaign_fit_starts", "campaign_minutes", "exploration_fit_starts",
        "exploration_minutes", "final_reserve_fit_starts", "final_reserve_minutes",
        "batch_fit_starts", "batch_minutes", "single_fit_seconds",
    )
    if any(campaign.PROTOCOL["limits"].get(key) is not None for key in limit_keys):
        raise RuntimeError("R46 requires the active campaign's unbounded limits")
    if int(card.get("expected_fit_starts", -1)) != len(FIT_OOF_MONTHS):
        raise RuntimeError("R46 preregistration has the wrong past-OOF fit count")
    if card.get("past_oof_months_to_fit") != list(FIT_OOF_MONTHS):
        raise RuntimeError("R46 fit months differ from its preregistered plan")
    return card


def _verify_r45_screen(parent: dict) -> tuple[dict, dict, pd.DataFrame, Path, Path, Path]:
    card = campaign.PROTOCOL["cards"].get(r45.CARD)
    if not card or card.get("status") != "screen_failed":
        raise RuntimeError("R46 requires the completed, failed R45 CatBoost screen")
    result_path = campaign.ROOT / card["result_path"]
    rows_path = result_path.parent / "paired_predictions.csv"
    manifest_path = result_path.parent / "run_manifest.json"
    if not all(path.is_file() for path in (result_path, rows_path, manifest_path)):
        raise FileNotFoundError("R45 result, paired rows, or manifest is missing")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (result.get("run_id") != card.get("run_id")
            or result.get("fit_starts") != len(SCREEN_MONTHS)
            or result.get("data_sha256") != campaign.PROTOCOL["data_sha256"]
            or result.get("reserved_months_labels_opened")
            or result.get("hidden_labels_opened")):
        raise RuntimeError("R45 result provenance or safety flags do not match the registered run")
    if (manifest.get("result_sha256") != _sha256(result_path)
            or manifest.get("paired_predictions_sha256") != _sha256(rows_path)
            or manifest.get("source_identity", {}).get("data_sha256") != campaign.PROTOCOL["data_sha256"]
            or manifest.get("parent_run_id") != parent["screen"]["run_id"]):
        raise RuntimeError("R45 result or parent hashes failed verification")
    for source, snapshot in manifest.get("source_snapshots", {}).items():
        snapshot_path = result_path.parent / "provenance" / snapshot
        if (not snapshot_path.is_file()
                or manifest.get("code_sha256", {}).get(source) != _sha256(snapshot_path)):
            raise RuntimeError(f"R45 source snapshot failed verification: {source}")
    if (manifest.get("parent_screen_result_sha256") != _sha256(parent["screen_path"])
            or manifest.get("parent_screen_rows_sha256") != _sha256(parent["screen_rows_path"])):
        raise RuntimeError("R45 did not use the currently verified R35 screen artifacts")
    for fit in result.get("fit_details", []):
        saved = fit.get("saved_predictions", {})
        path = result_path.parent / saved.get("path", "")
        if not path.is_file() or _sha256(path) != saved.get("sha256"):
            raise RuntimeError(f"R45 component prediction artifact failed verification: {path}")
    rows = pd.read_csv(rows_path, dtype={"ml_num": str})
    if (rows.duplicated(["month", "ml_num"]).any()
            or set(rows.month.astype(str)) != set(SCREEN_MONTHS)):
        raise RuntimeError("R45 paired screen rows are incomplete or duplicated")
    for month in SCREEN_MONTHS:
        expected = parent["screen_rows"].loc[
            parent["screen_rows"].month.astype(str).eq(month)
        ].reset_index(drop=True)
        observed = rows.loc[rows.month.astype(str).eq(month)].reset_index(drop=True)
        if (observed.ml_num.astype(str).tolist() != expected.ml_num.astype(str).tolist()
                or not np.array_equal(observed.sold_price.to_numpy(float),
                                      expected.sold_price.to_numpy(float))
                or not np.array_equal(observed.parent_prediction.to_numpy(float),
                                      expected.candidate_prediction.to_numpy(float))):
            raise RuntimeError(f"R45 rows do not match R35 on {month}")
    return result, manifest, rows, result_path, rows_path, manifest_path


def _verify_r44_seeds(parent: dict) -> tuple[dict, dict, Path]:
    card = campaign.PROTOCOL["cards"].get(r44.CARD)
    if not card or card.get("status") != "screen_failed":
        raise RuntimeError("R46 requires the completed R44 seed OOF artifacts")
    result_path = campaign.ROOT / card["result_path"]
    rows_path = result_path.parent / "paired_predictions.csv"
    manifest_path = result_path.parent / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if (manifest.get("result_sha256") != _sha256(result_path)
            or manifest.get("paired_predictions_sha256") != _sha256(rows_path)
            or manifest.get("seed_oof_sha256") != _sha256(result_path.parent / "past_oof_predictions.csv")
            or result.get("data_sha256") != campaign.PROTOCOL["data_sha256"]
            or result.get("reserved_months_labels_opened")
            or result.get("hidden_labels_opened")):
        raise RuntimeError("R44 OOF provenance failed verification")
    if (manifest.get("parent_screen_result_sha256") != _sha256(parent["screen_path"])
            or manifest.get("parent_screen_rows_sha256") != _sha256(parent["screen_rows_path"])
            or manifest.get("parent_development_rows_sha256") != _sha256(parent["development_rows_path"])):
        raise RuntimeError("R44 OOF rows were not built from the currently verified R35 artifacts")
    details = {row["month"]: row for row in result.get("seed_oof_details", [])}
    frames = {}
    for month in SEED_MONTHS:
        detail = details.get(month)
        path = result_path.parent / "oof_seed" / f"{month}.csv"
        if not detail or not path.is_file() or detail.get("data_sha256") != campaign.PROTOCOL["data_sha256"]:
            raise RuntimeError(f"R44 {month} OOF artifact is missing or has wrong data identity")
        frame = pd.read_csv(path, dtype={"ml_num": str})
        parent_hash = hashlib.sha256(frame.parent_prediction.to_numpy(dtype="float64").tobytes()).hexdigest()
        variant_hash = hashlib.sha256(frame.variant_prediction.to_numpy(dtype="float64").tobytes()).hexdigest()
        cache_path = campaign.ROOT / detail["HGBR_cache_path"]
        if (parent_hash != detail["parent_prediction_sha256"]
                or variant_hash != detail["variant_prediction_sha256"]
                or not cache_path.is_file()
                or _sha256(cache_path) != detail["HGBR_cache_sha256"]):
            raise RuntimeError(f"R44 {month} predictions or HGBR cache failed verification")
        if (frame.month.astype(str).ne(month).any()
                or frame.duplicated(["month", "ml_num"]).any()
                or not pd.to_datetime(frame.date_sold, errors="raise").dt.to_period("M").astype(str).eq(month).all()):
            raise RuntimeError(f"R44 {month} OOF rows have invalid IDs or dates")
        frames[month] = frame
    return result, frames, result_path


def _asof_reference_frames(parent: dict, r45_rows: pd.DataFrame,
                           seed_frames: dict, data_path: Path) -> dict[str, pd.DataFrame]:
    dates = r44._safe_date_metadata(PAST_SOURCE_MONTHS, data_path)
    development_rows = pd.read_csv(parent["development_rows_path"], dtype={"ml_num": str})
    development = development_rows.loc[development_rows.track.eq(TRACK)].copy()
    development["month"] = development.month.astype(str)
    development = development.loc[development.month.isin(PAST_SOURCE_MONTHS)].copy()
    development = development.rename(columns={"candidate_prediction": "r35_prediction"})
    development = r44._attach_oof_dates(
        development[["month", "ml_num", "sold_price", "r35_prediction"]], dates,
    )
    by_month: dict[str, pd.DataFrame] = {}
    for month in ("2023-07", "2023-11", "2024-03"):
        frame = development.loc[development.month.eq(month)].copy()
        frame["catboost_prediction"] = np.nan
        by_month[month] = frame
    screen = r45_rows.copy()
    screen["month"] = screen.month.astype(str)
    screen = screen.rename(columns={
        "parent_prediction": "r35_prediction",
        "candidate_prediction": "catboost_prediction",
    })
    screen = screen[["month", "ml_num", "sold_price", "r35_prediction", "catboost_prediction"]]
    screen = r44._attach_oof_dates(screen, dates)
    for month in SCREEN_MONTHS:
        by_month[month] = screen.loc[screen.month.eq(month)].copy()
    for month, seed in seed_frames.items():
        frame = seed[["month", "ml_num", "date_sold", "sold_price", "variant_prediction"]].copy()
        frame = frame.rename(columns={"variant_prediction": "r35_prediction"})
        frame["catboost_prediction"] = np.nan
        by_month[month] = frame
    expected = set(PAST_SOURCE_MONTHS)
    if set(by_month) != expected:
        raise RuntimeError(f"R46 OOF source months differ from preregistration: {sorted(by_month)}")
    for month, frame in by_month.items():
        if (frame.duplicated(["month", "ml_num"]).any()
                or not pd.to_datetime(frame.date_sold, errors="raise").dt.to_period("M").astype(str).eq(month).all()
                or not np.isfinite(frame.sold_price.to_numpy(float)).all()
                or (frame.sold_price.to_numpy(float) <= 0.0).any()):
            raise RuntimeError(f"R46 {month} reference OOF rows are not valid")
    return by_month


def _fit_catboost_oof(month: str, reference: pd.DataFrame, card: dict, run_id: str,
                      output: Path, source: dict, fit_details: list[dict],
                      fit_start_ids: list[str], r44_root: Path,
                      fit_stage: str = "past_oof") -> pd.DataFrame:
    from data import hide_answers
    from experiments.model_optimization_v2.pipeline import blend_prices

    _, train, _, truth, heldout = greedy._load_fold(month)
    cutoff = pd.Period(month, freq="M").start_time
    train_dates = pd.to_datetime(train.date_sold, errors="raise")
    if train_dates.ge(cutoff).any():
        raise RuntimeError(f"R46 OOF train rows reach the {month} cutoff")
    homes = hide_answers(heldout, TRACK).reset_index(drop=True)
    ids = homes.ml_num.astype(str).to_numpy()
    if (not np.array_equal(ids, truth.ml_num.astype(str).to_numpy())
            or not pd.to_datetime(heldout.date_sold, errors="raise").dt.to_period("M").astype(str).eq(month).all()):
        raise RuntimeError(f"R46 {month} heldout rows do not match the registered OOF month")
    expected = card["past_oof_support"][month]
    train_route = greedy._specialist_mask(train, HOUSE_TYPE)
    eval_route = greedy._specialist_mask(homes, HOUSE_TYPE)
    support = {
        "training_rows": int(train_route.sum()),
        "evaluation_rows": int(eval_route.sum()),
        "all_evaluation_rows": int(len(homes)),
    }
    if support != expected:
        raise RuntimeError(f"R46 {month} route support differs from preregistration: {support}")
    ref = reference.copy()
    ref["ml_num"] = ref.ml_num.astype(str)
    if (ref.ml_num.duplicated().any()
            or set(ref.ml_num) != set(ids)
            or not np.array_equal(ref.set_index("ml_num").loc[ids, "sold_price"].to_numpy(float),
                                  truth.sold_price.to_numpy(float))):
        raise RuntimeError(f"R46 {month} R35 OOF reference rows do not align to the fold")
    ref = ref.set_index("ml_num").loc[ids]

    cache_path = campaign.ROOT / card["hgbr_cache_paths"][month]
    if not cache_path.is_file() or _sha256(cache_path) != card["hgbr_cache_sha256"][month]:
        raise RuntimeError(f"R46 {month} HGBR cache hash differs from preregistration")
    old_output = greedy.OUTPUT
    greedy.OUTPUT = cache_path.parents[2]
    try:
        frozen_hgbr = r03._verified_hgbr_cache(greedy, month, TRACK, train, homes)
    finally:
        greedy.OUTPUT = old_output
    if frozen_hgbr.shape != (len(homes),):
        raise RuntimeError(f"R46 {month} HGBR output is not row-aligned")

    fit_rows = train.loc[train_route].copy()
    eval_rows = homes.loc[eval_route].copy()
    fit_context = {
        "month": month, "training_rows": len(fit_rows),
        "evaluation_rows": len(eval_rows),
        "evaluation_ids": eval_rows.ml_num.astype(str).tolist(),
    }
    old_fit_record = greedy._fit_record

    def record_fit(parent_fit_id: str, fit_fn):
        fit_id = (f"{campaign.PROTOCOL['campaign_id']}:{CARD}:{TRACK}:{fit_stage}:"
                  f"{month}:{COMPONENT}:{run_id}")
        campaign._append({
            "event": "fit_start", "card": CARD, "stage": fit_stage,
            "run_id": run_id, "fit_id": fit_id, "parent_fit_id": parent_fit_id,
            "track": TRACK, "month": month, "component": COMPONENT,
            "house_type": HOUSE_TYPE, "training_rows": fit_context["training_rows"],
            "evaluation_rows": fit_context["evaluation_rows"],
            "data_sha256": source["data_sha256"],
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        fit_start_ids.append(fit_id)
        started = time.perf_counter()
        try:
            fitted = fit_fn()
        except Exception as exc:
            campaign._append({
                "event": "fit_complete", "card": CARD, "stage": fit_stage,
                "run_id": run_id, "fit_id": fit_id, "month": month,
                "component": COMPONENT, "status": "failed",
                "seconds": time.perf_counter() - started,
                "error": f"{type(exc).__name__}: {exc}",
                "reserved_months_labels_opened": False, "hidden_labels_opened": False,
            })
            raise
        saved = r45._save_component_prediction(
            output, month, fit_context["evaluation_ids"], fitted["predictions"],
        )
        detail = {
            "fit_id": fit_id, "month": month, "component": COMPONENT,
            "seconds": time.perf_counter() - started,
            "training_rows": fitted["training_rows"],
            "evaluation_rows": fitted["evaluation_rows"],
            "feature_columns": fitted["feature_columns"],
            "model_family": fitted["model_family"],
            "model_config": fitted["model_config"],
            "target": fitted["target"],
            "text_feature_count": sum(str(name).startswith("remarks_tfidf_")
                                       for name in fitted["feature_columns"]),
            "prediction_sha256": fitted["prediction_sha256"],
            "saved_predictions": saved,
        }
        fit_details.append(detail)
        campaign._append({
            "event": "fit_complete", "card": CARD, "stage": fit_stage,
            "run_id": run_id, **detail, "status": "complete",
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        return fitted

    old_variant = greedy.VARIANTS.get(r45.VARIANT)
    had_variant = r45.VARIANT in greedy.VARIANTS
    greedy.VARIANTS[r45.VARIANT] = {
        "model_family": "catboost", "model_config": dict(r03.CATBOOST_CONFIG),
    }
    transformer, restore = r24.r24_installer(fit_rows, eval_rows, TRACK)
    greedy._fit_record = record_fit
    try:
        fitted = greedy._fit_history_candidate(
            month, TRACK, train, eval_rows, fit_rows=fit_rows,
            variant=r45.VARIANT, fit_tag=f"R46-CatBoost-{month}-Detached",
        )
    finally:
        greedy._fit_record = old_fit_record
        restore()
        if had_variant:
            greedy.VARIANTS[r45.VARIANT] = old_variant
        else:
            greedy.VARIANTS.pop(r45.VARIANT, None)

    specialist_prediction = blend_prices(
        fitted["predictions"], frozen_hgbr[eval_route],
        challenger_weight=CATBOOST_SPECIALIST_WEIGHT,
    )
    candidate = ref.r35_prediction.to_numpy(float).copy()
    candidate[eval_route] = specialist_prediction
    if not np.array_equal(candidate[~eval_route], ref.r35_prediction.to_numpy(float)[~eval_route]):
        raise RuntimeError(f"R46 changed non-Detached R35 OOF rows in {month}")
    frame = pd.DataFrame({
        "month": month, "track": TRACK, "ml_num": ids,
        "date_sold": pd.to_datetime(heldout.date_sold.to_numpy(), errors="raise"),
        "sold_price": truth.sold_price.to_numpy(float),
        "r35_prediction": ref.r35_prediction.to_numpy(float),
        "catboost_prediction": candidate,
    })
    if (not np.isfinite(frame[["sold_price", "r35_prediction", "catboost_prediction"]].to_numpy(float)).all()
            or (frame[["sold_price", "r35_prediction", "catboost_prediction"]] <= 0.0).any().any()):
        raise RuntimeError(f"R46 {month} OOF predictions are invalid")
    return frame


def _select_and_screen(oof: pd.DataFrame, screen_rows: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    selection = {}
    paired_parts = []
    for month in SCREEN_MONTHS:
        cutoff = pd.Period(month, freq="M").start_time
        source = oof.rename(columns={
            "r35_prediction": "parent_prediction",
            "catboost_prediction": "variant_prediction",
        })
        picked = weight_selection.select_weight(
            source, cutoff, minimum_rows=MINIMUM_PAST_OOF_ROWS,
            max_mape_regression_percent=1.0,
        )
        if picked["reason"] != "selected_from_strictly_past_oof":
            raise RuntimeError(f"R46 could not select a weight for {month}: {picked['reason']}")
        selection[month] = picked
        fold = screen_rows.loc[screen_rows.month.astype(str).eq(month)].copy()
        weight = float(picked["weight"])
        fold["r35_prediction"] = fold.parent_prediction.to_numpy(float)
        fold["catboost_prediction"] = fold.candidate_prediction.to_numpy(float)
        fold["parent_prediction"] = fold.r35_prediction
        fold["candidate_prediction"] = weight_selection.blend(
            fold.r35_prediction.to_numpy(float),
            fold.catboost_prediction.to_numpy(float), weight,
        )
        fold["catboost_share_selected"] = weight
        paired_parts.append(fold)
    paired = pd.concat(paired_parts, ignore_index=True)
    metrics = scoring._track_result(paired)
    return paired, selection, metrics


def _copy_provenance(parent: dict, r44_root: Path, r45_root: Path,
                     output: Path) -> dict[str, str]:
    provenance = output / "provenance"
    provenance.mkdir()
    paths = {
        "r35_screen_results.json": parent["screen_path"],
        "r35_screen_paired_predictions.csv": parent["screen_rows_path"],
        "r35_screen_run_manifest.json": parent["screen_manifest_path"],
        "r35_development_results.json": parent["development_path"],
        "r35_development_paired_predictions.csv": parent["development_rows_path"],
        "r35_development_run_manifest.json": parent["development_manifest_path"],
        "r35_fixed_origin_results.json": parent["fixed_path"],
        "r35_fixed_origin_paired_predictions.csv": parent["fixed_path"].parent / "paired_predictions.csv",
        "r35_fixed_origin_run_manifest.json": parent["fixed_manifest_path"],
        "r44_screen_results.json": r44_root / "screen_results.json",
        "r44_screen_paired_predictions.csv": r44_root / "paired_predictions.csv",
        "r44_run_manifest.json": r44_root / "run_manifest.json",
        "r44_past_oof_predictions.csv": r44_root / "past_oof_predictions.csv",
        "r45_screen_results.json": r45_root / "screen_results.json",
        "r45_screen_paired_predictions.csv": r45_root / "paired_predictions.csv",
        "r45_run_manifest.json": r45_root / "run_manifest.json",
    }
    for month in SEED_MONTHS:
        paths[f"r44_seed_oof_{month}.csv"] = r44_root / "oof_seed" / f"{month}.csv"
        paths[f"r44_seed_hgbr_{month}.csv"] = r44_root / "greedy_cache" / "frozen_hgbr" / TRACK / f"{month}.csv"
    r45_result = json.loads((r45_root / "screen_results.json").read_text(encoding="utf-8"))
    for fit in r45_result.get("fit_details", []):
        saved = fit.get("saved_predictions", {})
        paths[f"r45_component_{fit['month']}.csv"] = r45_root / saved["path"]
    hashes = {}
    for name, source in paths.items():
        if not source.is_file():
            raise FileNotFoundError(f"R46 provenance artifact is missing: {source}")
        shutil.copy2(source, provenance / name)
        hashes[name] = _sha256(source)
    return hashes


def run() -> dict:
    from data import hide_answers

    card = _verify_limits_and_card()
    parent = parent_verifier._verified_r35_parent(campaign)
    if (card.get("parent_run_id") != parent["screen"]["run_id"]
            or card.get("variant_run_id") != parent["screen"]["run_id"]):
        raise RuntimeError("R46 parent is not the verified R35 list-aware candidate")
    source = budget._source_identity()
    if source.get("data_sha256") != campaign.PROTOCOL["data_sha256"]:
        raise RuntimeError("R46 data identity differs from the active campaign")
    if metadata.version("catboost") != card.get("catboost_version"):
        raise RuntimeError("R46 CatBoost version differs from preregistration")
    r45_result, r45_manifest, r45_rows, r45_result_path, r45_rows_path, r45_manifest_path = (
        _verify_r45_screen(parent)
    )
    r44_result, seed_frames, r44_result_path = _verify_r44_seeds(parent)
    r44_root = r44_result_path.parent
    r45_root = r45_result_path.parent

    run_id = "bill_target74_r46_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = campaign.CAMPAIGN_ROOT / "batches" / run_id
    output.mkdir(parents=True, exist_ok=False)
    provenance_hashes = _copy_provenance(parent, r44_root, r45_root, output)
    code_hashes = {}
    source_snapshots = {}
    provenance = output / "provenance"
    for index, path in enumerate(_source_files()):
        key = str(path.relative_to(campaign.ROOT)).replace("\\", "/")
        code_hashes[key] = _sha256(path)
        snapshot_name = f"src_{index:02d}{path.suffix}.source"
        shutil.copy2(path, provenance / snapshot_name)
        source_snapshots[key] = snapshot_name

    fit_details: list[dict] = []
    fit_start_ids: list[str] = []
    fit_context: dict[str, object] = {}
    old_variant = greedy.VARIANTS.get(r45.VARIANT)
    had_variant = r45.VARIANT in greedy.VARIANTS
    greedy.VARIANTS[r45.VARIANT] = {
        "model_family": "catboost", "model_config": dict(r03.CATBOOST_CONFIG),
    }
    old_greedy_output = greedy.OUTPUT
    oof_frames = []
    campaign._append({
        "event": "card_start", "card": CARD, "stage": "past_oof_and_screen",
        "run_id": run_id, "track": TRACK,
        "parent_run_id": parent["screen"]["run_id"],
        "screen_months": list(SCREEN_MONTHS),
        "past_oof_months_to_fit": list(FIT_OOF_MONTHS),
        "expected_fit_starts": len(FIT_OOF_MONTHS),
        "reused_screen_run_id": r45_result["run_id"],
        "reserved_months_labels_opened": False,
        "hidden_labels_opened": False,
    })
    try:
        references = _asof_reference_frames(parent, r45_rows, seed_frames, campaign.DATA)
        for month in FIT_OOF_MONTHS:
            oof_frames.append(_fit_catboost_oof(
                month, references[month], card, run_id, output, source,
                fit_details, fit_start_ids, r44_root,
            ))
        oof_frames.extend([
            references["2023-03"],
            references["2024-06"],
        ])
        oof = pd.concat(oof_frames, ignore_index=True)
        if oof.duplicated(["month", "ml_num"]).any():
            raise RuntimeError("R46 assembled OOF rows contain duplicate month/listing IDs")
        oof = oof.sort_values(["month", "ml_num"], kind="mergesort").reset_index(drop=True)
        if set(oof.month.astype(str)) != set(PAST_SOURCE_MONTHS):
            raise RuntimeError("R46 OOF artifact does not cover the preregistered safe months")
        oof.to_csv(output / "past_oof_predictions.csv", index=False)

        paired, selection, metrics = _select_and_screen(oof, r45_rows)
        paired_path = output / "paired_predictions.csv"
        paired.to_csv(paired_path, index=False)
        result = {
            "campaign_id": campaign.PROTOCOL["campaign_id"],
            "run_id": run_id, "card": CARD, "track": TRACK,
            "parent_run_id": parent["screen"]["run_id"],
            "initial_parent_run_id": card["initial_parent_run_id"],
            "reused_screen_run_id": r45_result["run_id"],
            "screen_months": list(SCREEN_MONTHS),
            "past_oof_months": list(PAST_SOURCE_MONTHS),
            "new_fit_months": list(FIT_OOF_MONTHS),
            "fit_starts": len(fit_details), "fit_details": fit_details,
            "selected_catboost_share_by_screen_month": selection,
            "tracks": {TRACK: metrics},
            "data_sha256": source["data_sha256"],
            "reserved_months_labels_opened": False,
            "reserve_validation_eligible": False,
            "hidden_labels_opened": False,
            "reserve_incident_id": campaign.PROTOCOL.get("reserve_label_access_incident", {}).get("incident_id"),
        }
        result_path = output / "screen_results.json"
        _write_json(result_path, result)
        parent_hashes = {
            "r35_screen_result_sha256": _sha256(parent["screen_path"]),
            "r35_screen_rows_sha256": _sha256(parent["screen_rows_path"]),
            "r35_development_result_sha256": _sha256(parent["development_path"]),
            "r35_development_rows_sha256": _sha256(parent["development_rows_path"]),
            "r35_fixed_result_sha256": _sha256(parent["fixed_path"]),
            "r35_fixed_rows_sha256": _sha256(parent["fixed_path"].parent / "paired_predictions.csv"),
            "r44_result_sha256": _sha256(r44_result_path),
            "r44_paired_sha256": _sha256(r44_root / "paired_predictions.csv"),
            "r45_result_sha256": _sha256(r45_result_path),
            "r45_paired_sha256": _sha256(r45_rows_path),
        }
        manifest = {
            "campaign_id": campaign.PROTOCOL["campaign_id"],
            "run_id": run_id, "source_identity": source,
            "code_sha256": code_hashes, "source_snapshots": source_snapshots,
            "parent_artifact_sha256": parent_hashes,
            "copied_provenance_sha256": provenance_hashes,
            "catboost_version": metadata.version("catboost"),
            "catboost_model_config": dict(r03.CATBOOST_CONFIG),
            "r45_model_component_outputs_verified": True,
            "past_oof_sha256": _sha256(output / "past_oof_predictions.csv"),
            "paired_predictions_sha256": _sha256(paired_path),
            "result_sha256": _sha256(result_path),
            "fit_starts": len(fit_details), "fit_details": fit_details,
            "reserved_months_labels_opened": False,
            "reserve_validation_eligible": False,
            "hidden_labels_opened": False,
        }
        _write_json(output / "run_manifest.json", manifest)
        screen = metrics
        qualifies = bool(screen["qualifies_screen"])
        report = [
            "# R46：CatBoost 与 R35 LightGBM Detached 专家的严格过去 OOF 融合",
            "",
            f"Run：`{run_id}`；筛选父模型：已验证的 R35 `{parent['screen']['run_id']}`。",
            "只在 Detached 路由内，将 R35 LightGBM 专家与 R45 CatBoost 专家输出按 past-only OOF 选择的比例融合；其他房型继续使用 R35 原预测。CatBoost 单模型替换已在 R45 失败，本轮唯一新问题是它能否作为互补专家提升 R35。",
            "屏幕预测直接复用已哈希验证的 R45 2023-03 与 2024-06 折结果，因此 R46 新增 5 个 CatBoost OOF fits，不重训筛选折。每个外层月份只用严格更早的安全 OOF 月份选择 0.05 网格上的 CatBoost share，并要求过去 OOF MAPE 相对 R35 不恶化超过 1%。",
            "",
            f"Screen pooled MdAPE：{screen['parent']['mdape_percent']:.5f}% → {screen['candidate']['mdape_percent']:.5f}%；",
            f"相对改善：{screen['relative_mdape_improvement_percent']:+.3f}%；MAPE 相对变化：{screen['mape_relative_change_percent']:+.3f}%；通过：{qualifies}。",
            "",
            "## 各筛选月",
            "",
            "| 月份 | 过去 OOF 月份/行数 | CatBoost share | R35 MdAPE | 候选 MdAPE | 相对改善 |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for month in SCREEN_MONTHS:
            picked = selection[month]
            row = screen["monthly"][month]
            report.append(
                f"| {month} | {','.join(picked['oof_months'])} / {picked['oof_rows']} | "
                f"{picked['weight']:.2f} | {row['parent']['mdape_percent']:.5f}% | "
                f"{row['candidate']['mdape_percent']:.5f}% | "
                f"{row['relative_mdape_improvement_percent']:+.3f}% |"
            )
        report.extend([
            "",
            "只有双月筛选全部通过，才进入已预登记的八个月开发确认；不在筛选标签上调权重。保留月份 2025-05/08 已因既往数据访问事件失去独立验证资格，本轮及后续不得读取；隐藏标签未读取。",
            "",
        ])
        (output / "research_report.md").write_text("\n".join(report), encoding="utf-8")
        campaign.PROTOCOL["cards"][CARD].update({
            "status": "screen_passed" if qualifies else "screen_failed",
            "run_id": run_id,
            "fit_starts": len(fit_details),
            "result_path": str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
            "qualifies_screen": qualifies,
            "screen_metrics": {
                "parent_mdape_percent": screen["parent"]["mdape_percent"],
                "candidate_mdape_percent": screen["candidate"]["mdape_percent"],
                "relative_mdape_improvement_percent": screen["relative_mdape_improvement_percent"],
                "mape_relative_change_percent": screen["mape_relative_change_percent"],
                "monthly_relative_mdape_improvement_percent": {
                    month: values["relative_mdape_improvement_percent"]
                    for month, values in screen["monthly"].items()
                },
            },
            "selected_catboost_share_by_screen_month": {
                month: selection[month]["weight"] for month in SCREEN_MONTHS
            },
        })
        campaign.PROTOCOL_PATH.write_text(
            json.dumps(campaign.PROTOCOL, indent=2) + "\n", encoding="utf-8",
        )
        campaign._append({
            "event": "card_complete", "card": CARD, "run_id": run_id,
            "fit_starts": len(fit_details),
            "result_path": str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
            "qualifies_by_track": {TRACK: qualifies},
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        next_action = (
            "R46 passed screen; run the pre-registered eight-month OOF blend confirmation"
            if qualifies else
            "R46 failed screen; preserve R35 and register a distinct list-aware or off-market hypothesis"
        )
        campaign._write_progress(
            "r46_screen_complete", next_action=next_action,
            latest_fit_count=len(fit_details),
            r46_result=str(result_path.relative_to(campaign.ROOT)).replace("\\", "/"),
        )
        print(
            f"R46: parent_mdape={screen['parent']['mdape_percent']:.5f}% "
            f"candidate_mdape={screen['candidate']['mdape_percent']:.5f}% "
            f"relative_gain={screen['relative_mdape_improvement_percent']:+.3f}% "
            f"pass={qualifies} run={run_id}", flush=True,
        )
        return result
    except Exception as exc:
        failure = {
            "run_id": run_id, "card": CARD, "stage": "past_oof_and_screen",
            "error": f"{type(exc).__name__}: {exc}",
            "fit_starts": len(fit_start_ids), "fit_completions": len(fit_details),
            "fit_ids": fit_start_ids, "fit_details": fit_details,
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        }
        _write_json(output / "failure_checkpoint.json", failure)
        campaign._append({
            "event": "card_failed", "card": CARD, "run_id": run_id,
            "fit_starts": failure["fit_starts"],
            "fit_completions": failure["fit_completions"],
            "error": failure["error"],
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        campaign._write_progress(
            "r46_screen_failed",
            next_action="Inspect the R46 failure checkpoint and fit ledger before retrying",
            r46_failure=str((output / "failure_checkpoint.json").relative_to(campaign.ROOT)).replace("\\", "/"),
        )
        raise
    finally:
        greedy.OUTPUT = old_greedy_output
        if had_variant:
            greedy.VARIANTS[r45.VARIANT] = old_variant
        else:
            greedy.VARIANTS.pop(r45.VARIANT, None)


if __name__ == "__main__":
    run()
