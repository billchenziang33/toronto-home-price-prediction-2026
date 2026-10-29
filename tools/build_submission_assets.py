"""Package verified, safe development OOF frames for the submission routes.

This command reads only the registered eight development months from the training
parquet and the corresponding saved OOF predictions. It never reads hidden inputs
or reserve-month labels. It refuses to overwrite an existing model_assets folder.
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.bill_target74 import r151_offmarket_r93_radius15km_clip035_vs_r22 as r151
from experiments.bill_target74 import r97_past_oof_catboost_residual_meta as r97
from experiments.bill_target74 import r48_groupwise_past_oof_r22_r32_blend as r48
from experiments.bill_target74 import r50_type_price_interaction_past_oof_blend as r50
from experiments.bill_target74 import r52_r48_r50_adaptive_meta_blend as r52
from experiments.bill_target74 import runner as campaign


OUTPUT = ROOT / "model_assets"
R97_CARD = "R97_past_oof_catboost_residual_meta"
R48_CARD = "R48_groupwise_past_oof_r22_r32_blend"
R50_CARD = "R50_type_price_interaction_past_oof_blend"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_output_dir(output_dir: str | Path) -> Path:
    path = Path(output_dir)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"asset output must remain inside the repository: {path}")
    return path


def validate_oof_months(months, allowed_months, reserved_months) -> set[str]:
    """Reject reserved or unregistered dates before any OOF labels are packaged."""
    observed = {str(month) for month in months}
    if not observed:
        raise ValueError("OOF month set is empty")
    reserved = {str(month) for month in reserved_months}
    unexpected = observed - {str(month) for month in allowed_months}
    if observed & reserved:
        raise ValueError(f"reserved OOF months are forbidden: {sorted(observed & reserved)}")
    if unexpected:
        raise ValueError(f"unexpected OOF months are not registered: {sorted(unexpected)}")
    return observed


def _safe_r97_artifacts() -> tuple[pd.DataFrame, dict]:
    progress = json.loads(campaign.STATE.read_text(encoding="utf-8"))
    result_rel = progress.get("r97_result")
    run_id = progress.get("r97_run_id")
    if not result_rel or not run_id:
        raise RuntimeError("campaign progress does not identify a completed R97 run")
    result_path = (ROOT / result_rel).resolve()
    campaign_root = campaign.CAMPAIGN_ROOT.resolve()
    if not result_path.is_relative_to(campaign_root):
        raise RuntimeError("R97 result path escapes its registered campaign")
    predictions_path = result_path.parent / "paired_predictions.csv"
    manifest_path = result_path.parent / "run_manifest.json"
    if not all(path.is_file() for path in (result_path, predictions_path, manifest_path)):
        raise FileNotFoundError("R97 result, paired predictions, and run manifest must all exist")

    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_months = set(campaign.PROTOCOL["development_months"])
    reserved_months = set(campaign.PROTOCOL["reserved_months"])
    if result.get("card") != R97_CARD or result.get("run_id") != run_id:
        raise RuntimeError("R97 result identity does not match campaign progress")
    if result.get("data_sha256") != campaign.PROTOCOL["data_sha256"]:
        raise RuntimeError("R97 result uses a different training-data snapshot")
    if result.get("reserved_months_labels_opened") is not False:
        raise RuntimeError("R97 run did not attest that reserved labels stayed closed")
    if result.get("hidden_labels_opened") is not False:
        raise RuntimeError("R97 run did not attest that hidden labels stayed closed")
    if (manifest.get("run_id") != run_id
            or manifest.get("development_results_sha256") != sha256(result_path)):
        raise RuntimeError("R97 manifest does not verify the result file")
    if manifest.get("paired_predictions_sha256") != sha256(predictions_path):
        raise RuntimeError("R97 manifest does not verify the paired predictions")
    if manifest.get("reserved_months_labels_opened") is not False:
        raise RuntimeError("R97 manifest does not attest reserved-label isolation")
    if manifest.get("hidden_labels_opened") is not False:
        raise RuntimeError("R97 manifest does not attest hidden-label isolation")

    paired = pd.read_csv(predictions_path, dtype={"ml_num": str})
    required = {
        "month", "track", "ml_num", "sold_price", "base_prediction",
        "candidate_prediction", "correction_log",
    }
    if missing := required - set(paired.columns):
        raise RuntimeError(f"R97 predictions lack required columns: {sorted(missing)}")
    paired = paired.loc[paired.track.eq("list_aware")].copy()
    paired["month"] = paired.month.astype(str)
    paired["ml_num"] = paired.ml_num.astype(str)
    months = validate_oof_months(paired.month.tolist(), expected_months, reserved_months)
    if months != expected_months:
        raise RuntimeError("R97 OOF does not cover every registered development month")
    if paired.duplicated(["month", "ml_num"]).any():
        raise RuntimeError("R97 list-aware OOF keys are duplicated")

    metadata = r97._load_safe_metadata()
    metadata_months = validate_oof_months(metadata.month.tolist(), expected_months, reserved_months)
    if metadata_months != expected_months:
        raise RuntimeError("R97 source metadata does not cover exactly the safe development months")
    merged = paired.merge(
        metadata, on=["month", "ml_num"], how="outer", validate="one_to_one", indicator=True,
    )
    if not merged._merge.eq("both").all():
        raise RuntimeError("R97 OOF and safe source metadata do not have identical listing keys")
    merged.drop(columns="_merge", inplace=True)
    if not np.array_equal(merged.sold_price.to_numpy(float), merged.source_sold_price.to_numpy(float)):
        raise RuntimeError("R97 OOF labels differ from exact-month-filtered training labels")
    for column in ("sold_price", "base_prediction", "candidate_prediction"):
        values = merged[column].to_numpy(float)
        if not np.isfinite(values).all() or (values <= 0).any():
            raise RuntimeError(f"R97 {column} values must be finite and positive")
    merged = merged.sort_values(["month", "ml_num"], kind="mergesort").reset_index(drop=True)
    provenance = {
        "run_id": run_id,
        "card": R97_CARD,
        "result_path": result_path.relative_to(ROOT).as_posix(),
        "result_sha256": sha256(result_path),
        "paired_predictions_path": predictions_path.relative_to(ROOT).as_posix(),
        "paired_predictions_sha256": sha256(predictions_path),
        "run_manifest_path": manifest_path.relative_to(ROOT).as_posix(),
        "run_manifest_sha256": sha256(manifest_path),
        "development_metrics": result.get("tracks", {}).get("list_aware"),
        "reserved_months_labels_opened": False,
        "hidden_labels_opened": False,
    }
    return merged, provenance


def _safe_r151_route_oof() -> tuple[pd.DataFrame, dict]:
    progress = json.loads(campaign.STATE.read_text(encoding="utf-8"))
    result_rel = progress.get("r151_result")
    run_id = progress.get("r151_run_id")
    if not result_rel or not run_id:
        raise RuntimeError("campaign progress does not identify a completed R151 run")
    result_path = (ROOT / result_rel).resolve()
    if not result_path.is_relative_to(campaign.CAMPAIGN_ROOT.resolve()):
        raise RuntimeError("R151 result path escapes its registered campaign")
    predictions_path = result_path.parent / "paired_predictions.csv"
    manifest_path = result_path.parent / "run_manifest.json"
    if not all(path.is_file() for path in (result_path, predictions_path, manifest_path)):
        raise FileNotFoundError("R151 result, paired predictions, and manifest must all exist")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_card = "R151_offmarket_r93_radius15km_shrink20_clip035_vs_r22"
    if result.get("card") != expected_card or result.get("run_id") != run_id:
        raise RuntimeError("R151 result identity does not match campaign progress")
    if (manifest.get("data_sha256") != campaign.PROTOCOL["data_sha256"]
            or manifest.get("result_sha256") != sha256(result_path)
            or manifest.get("paired_predictions_sha256") != sha256(predictions_path)):
        raise RuntimeError("R151 result, predictions, or data identity fail manifest verification")
    for field in ("reserved_months_labels_opened", "hidden_labels_opened"):
        if result.get(field) is not False or manifest.get(field) is not False:
            raise RuntimeError(f"R151 did not attest {field}=false")

    paired = pd.read_csv(predictions_path, dtype={"ml_num": str})
    required = {"month", "track", "ml_num", "sold_price", "parent_prediction",
                "candidate_prediction", "correction_log"}
    if missing := required - set(paired.columns):
        raise RuntimeError(f"R151 paired predictions lack columns: {sorted(missing)}")
    paired = paired.loc[paired.track.eq("off_market")].copy()
    paired["month"] = paired.month.astype(str)
    paired["ml_num"] = paired.ml_num.astype(str)
    months = validate_oof_months(paired.month.tolist(), campaign.PROTOCOL["development_months"],
                                 campaign.PROTOCOL["reserved_months"])
    if months != set(campaign.PROTOCOL["development_months"]):
        raise RuntimeError("R151 candidate output does not cover the exact eight development months")
    if paired.duplicated(["month", "ml_num"]).any():
        raise RuntimeError("R151 candidate output contains duplicate listing keys")

    base = r151._read_safe_frame(r151._inputs(), "development")
    merged = base.merge(
        paired, on=["month", "ml_num"], how="outer", suffixes=("_base", "_r151"),
        validate="one_to_one", indicator=True,
    )
    if not merged._merge.eq("both").all():
        raise RuntimeError("R151 final predictions and safe base OOF have different listing keys")
    if (not np.array_equal(merged.sold_price_base.to_numpy(float), merged.sold_price_r151.to_numpy(float))
            or not np.array_equal(merged.prediction_r22.to_numpy(float),
                                  merged.parent_prediction.to_numpy(float))):
        raise RuntimeError("R151 labels or declared R22 parent do not match its safe base OOF")
    route = merged[[
        "month", "ml_num", "sold_price_base", "prediction_r22", "prediction_r52",
        "candidate_prediction", "correction_log",
    ]].rename(columns={"sold_price_base": "sold_price"})
    route = route.sort_values(["month", "ml_num"], kind="mergesort").reset_index(drop=True)
    for column in ("sold_price", "prediction_r22", "prediction_r52", "candidate_prediction"):
        values = route[column].to_numpy(float)
        if not np.isfinite(values).all() or (values <= 0.0).any():
            raise RuntimeError(f"R151 route OOF column {column} is invalid")
    provenance = {
        "run_id": run_id, "card": expected_card,
        "result_sha256": sha256(result_path),
        "paired_predictions_sha256": sha256(predictions_path),
        "run_manifest_sha256": sha256(manifest_path),
        "data_sha256": manifest.get("data_sha256"),
        "development_status": campaign.PROTOCOL["cards"].get(expected_card, {}).get("status"),
        "scores": result.get("scores"),
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    }
    return route, provenance


def _asset_record(path: Path, frame: pd.DataFrame) -> dict:
    return {
        "path": path.name,
        "sha256": sha256(path),
        "rows": int(len(frame)),
        "months": sorted(frame.month.astype(str).unique().tolist()),
        "columns": frame.columns.tolist(),
    }


def _offmarket_meta_sources() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load and verify safe R47 OOF plus the paired R48/R50 outputs used by R52."""
    source_card = campaign.PROTOCOL["cards"].get(r48.SOURCE_CARD, {})
    result_path = (ROOT / source_card.get("development_result_path", "")).resolve()
    if not result_path.is_relative_to(campaign.CAMPAIGN_ROOT.resolve()):
        raise RuntimeError("R47 result path escapes the registered campaign")
    oof_path = result_path.parent / "past_oof_predictions.csv"
    paired_path = result_path.parent / "paired_predictions.csv"
    manifest_path = result_path.parent / "run_manifest.json"
    if not all(path.is_file() for path in (result_path, oof_path, paired_path, manifest_path)):
        raise FileNotFoundError("R47 result, OOF, paired predictions, and manifest must all exist")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_run = source_card.get("run_id")
    if (result.get("card") != r48.SOURCE_CARD or result.get("run_id") != expected_run
            or result_manifest.get("run_id") != expected_run):
        raise RuntimeError("R47 result identity does not match its registered campaign card")
    if (result.get("data_sha256") != campaign.PROTOCOL["data_sha256"]
            or result_manifest.get("source_identity", {}).get("data_sha256")
            != campaign.PROTOCOL["data_sha256"]
            or result_manifest.get("result_sha256") != sha256(result_path)
            or result_manifest.get("seed_oof_sha256") != sha256(oof_path)
            or result_manifest.get("paired_predictions_sha256") != sha256(paired_path)):
        raise RuntimeError("R47 data or saved result/OOF hashes do not match its manifest")
    for field in ("reserved_months_labels_opened", "hidden_labels_opened"):
        if result.get(field) is not False or result_manifest.get(field) is not False:
            raise RuntimeError(f"R47 did not attest {field}=false")
    for source_key, snapshot in result_manifest.get("source_snapshots", {}).items():
        snapshot_path = result_path.parent / "provenance" / snapshot
        if (not snapshot_path.is_file()
                or result_manifest.get("code_sha256", {}).get(source_key) != sha256(snapshot_path)):
            raise RuntimeError(f"R47 immutable source snapshot failed verification: {source_key}")

    r47 = pd.read_csv(oof_path, dtype={"ml_num": str})
    required_r47 = {"month", "track", "ml_num", "date_sold", "sold_price",
                    "parent_prediction", "variant_prediction"}
    if missing := required_r47 - set(r47.columns):
        raise RuntimeError(f"R47 OOF artifact lacks columns: {sorted(missing)}")
    r47["month"] = r47.month.astype(str)
    r47["ml_num"] = r47.ml_num.astype(str)
    safe_months = set((*r48.SEED_MONTHS, *r48.DEVELOPMENT_MONTHS))
    months = validate_oof_months(r47.month.astype(str).tolist(), safe_months,
                                 campaign.PROTOCOL["reserved_months"])
    if months != safe_months or r47.duplicated(["month", "track", "ml_num"]).any():
        raise RuntimeError("R47 weight-history rows do not match the registered safe folds")
    r47 = r47.loc[r47.track.eq("off_market")].copy()
    if r47.duplicated(["month", "ml_num"]).any():
        raise RuntimeError("off-market R47 weight-history keys are duplicated")
    data_path = campaign.DATA
    if sha256(data_path) != campaign.PROTOCOL["data_sha256"]:
        raise RuntimeError("current training parquet does not match the R47 registered snapshot")
    filters = []
    for month in sorted(safe_months):
        start = pd.Timestamp(month + "-01")
        filters.append([("date_sold", ">=", start),
                        ("date_sold", "<", start + pd.offsets.MonthBegin(1))])
    metadata = pd.read_parquet(
        data_path, columns=["ml_num", "date_sold", "house_type_name", "sold_price"], filters=filters,
    )
    metadata["date_sold"] = pd.to_datetime(metadata.date_sold, errors="raise")
    metadata["month"] = metadata.date_sold.dt.to_period("M").astype(str)
    metadata["ml_num"] = metadata.ml_num.astype(str)
    validate_oof_months(metadata.month.tolist(), safe_months, campaign.PROTOCOL["reserved_months"])
    if metadata.duplicated(["month", "ml_num"]).any():
        raise RuntimeError("exact-month R47 source metadata has duplicate listing keys")
    checked = r47.merge(
        metadata, on=["month", "ml_num"], how="outer", suffixes=("_oof", "_source"),
        validate="one_to_one", indicator=True,
    )
    if not checked._merge.eq("both").all():
        raise RuntimeError("R47 OOF and exact safe-month training metadata have different IDs")
    if (not np.array_equal(pd.to_datetime(checked.date_sold_oof).to_numpy(),
                           pd.to_datetime(checked.date_sold_source).to_numpy())
            or not np.array_equal(checked.sold_price_oof.to_numpy(float),
                                  checked.sold_price_source.to_numpy(float))):
        raise RuntimeError("R47 OOF labels/date do not match the exact safe training rows")
    r47 = r47.merge(metadata[["month", "ml_num", "house_type_name"]],
                    on=["month", "ml_num"], how="left", validate="one_to_one")
    r47["house_type_name"] = r47.house_type_name.astype("string").fillna("__MISSING__").astype(str)
    r47 = r47[[
        "month", "ml_num", "date_sold", "house_type_name", "sold_price",
        "parent_prediction", "variant_prediction",
    ]].sort_values(["month", "ml_num"], kind="mergesort").reset_index(drop=True)
    for column in ("sold_price", "parent_prediction", "variant_prediction"):
        values = r47[column].to_numpy(float)
        if not np.isfinite(values).all() or (values <= 0.0).any():
            raise RuntimeError(f"R47 {column} values are not finite positive prices")

    progress = json.loads(campaign.STATE.read_text(encoding="utf-8"))
    result48, path48, paired48_path, manifest48 = r52._verified_result(
        campaign, progress["r48_result"], r52.PARENT_RUN, "R48"
    )
    result50, path50, paired50_path, manifest50 = r52._verified_result(
        campaign, progress["r50_result"], r52.VARIANT_RUN, "R50"
    )
    dev_months = set(campaign.PROTOCOL["development_months"])
    pair_frames = []
    for label, path in (("R48", paired48_path), ("R50", paired50_path)):
        frame = pd.read_csv(path, dtype={"ml_num": str})
        required = {"month", "track", "ml_num", "sold_price", "parent_prediction", "candidate_prediction"}
        if missing := required - set(frame.columns):
            raise RuntimeError(f"{label} paired rows lack columns: {sorted(missing)}")
        frame = frame.loc[frame.track.eq("off_market")].copy()
        frame["month"] = frame.month.astype(str)
        frame["ml_num"] = frame.ml_num.astype(str)
        observed = validate_oof_months(frame.month.tolist(), dev_months,
                                       campaign.PROTOCOL["reserved_months"])
        if observed != dev_months or frame.duplicated(["month", "ml_num"]).any():
            raise RuntimeError(f"{label} paired rows do not exactly cover safe development keys")
        frame = frame[["month", "ml_num", "sold_price", "parent_prediction", "candidate_prediction"]]
        pair_frames.append(frame)

    merged = pair_frames[0].merge(
        pair_frames[1], on=["month", "ml_num"], how="outer", suffixes=("_r48", "_r50"),
        validate="one_to_one", indicator=True,
    )
    if not merged._merge.eq("both").all():
        raise RuntimeError("R48/R50 OOF outputs do not have identical listing keys")
    for column in ("sold_price", "parent_prediction"):
        if not np.array_equal(merged[f"{column}_r48"].to_numpy(float),
                              merged[f"{column}_r50"].to_numpy(float)):
            raise RuntimeError(f"R48/R50 OOF {column} values differ")
    merged.rename(columns={
        "sold_price_r48": "sold_price",
        "parent_prediction_r48": "r22_prediction",
        "candidate_prediction_r48": "r48_prediction",
        "candidate_prediction_r50": "r50_prediction",
    }, inplace=True)
    history = merged[[
        "month", "ml_num", "sold_price", "r22_prediction", "r48_prediction", "r50_prediction",
    ]].sort_values(["month", "ml_num"], kind="mergesort").reset_index(drop=True)
    for column in ("sold_price", "r22_prediction", "r48_prediction", "r50_prediction"):
        values = history[column].to_numpy(float)
        if not np.isfinite(values).all() or (values <= 0.0).any():
            raise RuntimeError(f"R48/R50 weight-history column {column} is invalid")

    provenance = {
        "r47_source": {
            "card": r48.SOURCE_CARD, "status": source_card.get("status"),
            "run_id": expected_run,
            "result_sha256": sha256(result_path), "oof_sha256": sha256(oof_path),
            "paired_predictions_sha256": sha256(paired_path),
            "manifest_sha256": sha256(manifest_path),
            "data_sha256": campaign.PROTOCOL["data_sha256"],
            "verified_historical_source_snapshots": len(result_manifest.get("source_snapshots", {})),
        },
        "r48": {
            "run_id": result48["run_id"], "result_sha256": r52._sha256(path48),
            "paired_predictions_sha256": r52._sha256(paired48_path),
            "manifest_sha256": r52._sha256(path48.parent / "run_manifest.json"),
        },
        "r50": {
            "run_id": result50["run_id"], "result_sha256": r52._sha256(path50),
            "paired_predictions_sha256": r52._sha256(paired50_path),
            "manifest_sha256": r52._sha256(path50.parent / "run_manifest.json"),
        },
        "reserved_months_labels_opened": False,
        "hidden_labels_opened": False,
    }
    return r47, history, provenance


def extend_offmarket_meta_assets(output_dir: str | Path = OUTPUT) -> dict:
    """Add exact R47/R48/R50 past-only history inputs required by an R52 refit."""
    output = _resolve_output_dir(output_dir)
    manifest_path = output / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("build the base safe OOF assets before extending them")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("training_data_sha256") != campaign.PROTOCOL["data_sha256"]:
        raise RuntimeError("existing model_assets manifest has a different training snapshot")
    # Refuse to alter an asset package whose existing contents have drifted.
    for route in manifest.get("routes", {}).values():
        record = route.get("asset")
        if record:
            path = output / record["path"]
            if not path.is_file() or sha256(path) != record["sha256"]:
                raise RuntimeError(f"existing asset failed its recorded hash: {path}")

    r47, r48_r50, provenance = _offmarket_meta_sources()
    r151_route, r151_provenance = _safe_r151_route_oof()
    additions = {
        "off_market_r47_weight_history.csv": r47,
        "off_market_r48_r50_weight_history.csv": r48_r50,
        "off_market_r151_candidate_oof.csv": r151_route,
    }
    pending = []
    with tempfile.TemporaryDirectory(prefix=".meta_asset_build_", dir=output) as temp_name:
        stage = Path(temp_name)
        for filename, frame in additions.items():
            validate_oof_months(frame.month.astype(str).tolist(),
                                set((*r48.SEED_MONTHS, *r48.DEVELOPMENT_MONTHS)),
                                campaign.PROTOCOL["reserved_months"])
            staged_path = stage / filename
            frame.to_csv(staged_path, index=False, lineterminator="\n")
            destination = output / filename
            if destination.exists() and sha256(destination) != sha256(staged_path):
                raise FileExistsError(f"refusing to replace non-matching existing asset: {destination}")
            pending.append((staged_path, destination))
        for staged_path, destination in pending:
            if not destination.exists():
                staged_path.replace(destination)

    source_files = [
        Path(r48.__file__), Path(r50.__file__), Path(r52.__file__),
        Path(r48.weight_selection.__file__), Path(r50.price_source.__file__),
        Path(r48.greedy.__file__), Path(r48.campaign.__file__),
    ]
    source_hashes = manifest.setdefault("source_sha256", {})
    source_hashes.update({
        path.relative_to(ROOT).as_posix(): sha256(path)
        for path in source_files if path.is_file()
    })
    builder_path = Path(__file__).resolve()
    manifest["builder_sha256_by_stage"] = {
        "initial_oof_package": source_hashes.get(builder_path.relative_to(ROOT).as_posix()),
        "r52_history_extension": sha256(builder_path),
    }
    meta_assets = {
        filename: _asset_record(output / filename, frame)
        for filename, frame in additions.items() if "weight_history" in filename
    }
    manifest["routes"]["off_market"]["meta_assets"] = meta_assets
    manifest["routes"]["off_market"]["meta_provenance"] = provenance
    manifest["routes"]["off_market"]["route_oof_assets"] = {
        "r151_candidate": _asset_record(output / "off_market_r151_candidate_oof.csv", r151_route),
    }
    manifest["routes"]["off_market"]["route_oof_provenance"] = {
        "r151_candidate": r151_provenance,
    }
    temp_manifest = manifest_path.with_suffix(".json.tmp")
    temp_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp_manifest.replace(manifest_path)
    return manifest


def build_assets(output_dir: str | Path = OUTPUT) -> dict:
    output = _resolve_output_dir(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing asset directory: {output}")
    expected_months = set(campaign.PROTOCOL["development_months"])
    reserved_months = set(campaign.PROTOCOL["reserved_months"])

    r151_inputs = r151._inputs()
    off_market = r151._read_safe_frame(r151_inputs, "development")
    off_months = validate_oof_months(off_market.month.tolist(), expected_months, reserved_months)
    if off_months != expected_months:
        raise RuntimeError("R151 OOF does not cover every registered development month")

    list_aware, r97_provenance = _safe_r97_artifacts()
    if set(list_aware.month.unique()) != expected_months:
        raise RuntimeError("R97 packaged frame does not cover every registered development month")

    source_files = [
        Path(__file__), Path(r151.__file__), Path(r151.r93.__file__), Path(r151.r22.__file__),
        Path(r151.campaign.__file__), Path(r97.__file__), Path(r97.r02.__file__),
    ]
    source_hashes = {
        path.relative_to(ROOT).as_posix(): sha256(path)
        for path in source_files if path.is_file()
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".model_assets_build_", dir=output.parent) as temp_name:
        stage = Path(temp_name)
        off_path = stage / "off_market_r151_safe_oof.csv"
        list_path = stage / "list_aware_r97_safe_oof.csv"
        off_market.to_csv(off_path, index=False, lineterminator="\n")
        list_aware.to_csv(list_path, index=False, lineterminator="\n")
        manifest = {
            "format_version": 1,
            "campaign_id": campaign.PROTOCOL["campaign_id"],
            "training_data_sha256": campaign.PROTOCOL["data_sha256"],
            "development_months": sorted(expected_months),
            "reserved_months": sorted(reserved_months),
            "access_scope": {
                "training_rows_read": "registered eight-month development OOF rows only",
                "reserved_month_labels_opened": False,
                "hidden_test_labels_opened": False,
                "reserve_validation_eligible": False,
                "note": "The campaign's reserve validation remains ineligible; these exports contain no reserve-month rows.",
            },
            "routes": {
                "off_market": {
                    "candidate": "R151",
                    "parents": r151._input_hashes(r151_inputs),
                    "asset": _asset_record(off_path, off_market),
                },
                "list_aware": {
                    "candidate": "R97",
                    "provenance": r97_provenance,
                    "asset": _asset_record(list_path, list_aware),
                },
            },
            "source_sha256": source_hashes,
        }
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        # The destination was checked before building; atomic rename prevents a partial package.
        stage.rename(output)
    return manifest


def verify_assets(output_dir: str | Path = OUTPUT) -> dict:
    """Verify a packaged asset directory without opening the source parquet."""
    output = _resolve_output_dir(output_dir)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("training_data_sha256") != campaign.PROTOCOL["data_sha256"]:
        raise RuntimeError("asset manifest training-data identity differs from campaign protocol")
    access = manifest.get("access_scope", {})
    if (access.get("reserved_month_labels_opened") is not False
            or access.get("hidden_test_labels_opened") is not False
            or access.get("reserve_validation_eligible") is not False):
        raise RuntimeError("asset manifest does not attest the required label-access scope")

    expected_dev = set(manifest["development_months"])
    reserved = set(manifest["reserved_months"])
    expected_seed = set((*r48.SEED_MONTHS, *r48.DEVELOPMENT_MONTHS))
    checked = []
    for route_name, route in manifest["routes"].items():
        records = []
        if route.get("asset"):
            records.append(route["asset"])
        records.extend(route.get("meta_assets", {}).values())
        records.extend(route.get("route_oof_assets", {}).values())
        for record in records:
            path = (output / record["path"]).resolve()
            if not path.is_relative_to(output):
                raise RuntimeError(f"asset path escapes package directory: {record['path']}")
            if not path.is_file() or sha256(path) != record["sha256"]:
                raise RuntimeError(f"asset file hash mismatch: {record['path']}")
            frame = pd.read_csv(path, dtype={"ml_num": str})
            if len(frame) != record["rows"] or frame.columns.tolist() != record["columns"]:
                raise RuntimeError(f"asset row count or schema mismatch: {record['path']}")
            months = set(frame.month.astype(str).unique())
            allowed = expected_seed if "r47_weight_history" in record["path"] else expected_dev
            validate_oof_months(frame.month.astype(str).tolist(), allowed, reserved)
            if months != set(record["months"]):
                raise RuntimeError(f"asset month list mismatch: {record['path']}")
            key_columns = ["month", "ml_num"]
            if "track" in frame.columns:
                key_columns.insert(1, "track")
            if frame.duplicated(key_columns).any():
                raise RuntimeError(f"asset listing keys are duplicated: {record['path']}")
            prediction_columns = [
                column for column in frame.columns
                if column in {"prediction_r52", "prediction_r22", "parent_prediction",
                              "variant_prediction", "r22_prediction", "r48_prediction",
                              "r50_prediction", "base_prediction", "candidate_prediction"}
            ]
            for column in prediction_columns:
                values = pd.to_numeric(frame[column], errors="raise").to_numpy(float)
                if not np.isfinite(values).all() or (values <= 0.0).any():
                    raise RuntimeError(f"asset has invalid positive-price predictions: {record['path']}:{column}")
            if "sold_price_r52" in frame.columns and not np.array_equal(
                    frame.sold_price_r52.to_numpy(float), frame.sold_price.to_numpy(float)):
                raise RuntimeError("R151 OOF label mismatch in packaged asset")
            if "source_sold_price" in frame.columns and not np.array_equal(
                    frame.source_sold_price.to_numpy(float), frame.sold_price.to_numpy(float)):
                raise RuntimeError("R97 source-label mismatch in packaged asset")
            checked.append({"path": record["path"], "rows": len(frame), "sha256": record["sha256"]})
    return {"status": "verified", "asset_count": len(checked), "assets": checked}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extend-offmarket-meta", action="store_true",
                        help="add the verified R47/R48/R50 weight history required to refit R52")
    parser.add_argument("--verify", action="store_true",
                        help="verify packaged file hashes/schemas without opening the training parquet")
    parser.add_argument("--output-dir", default="model_assets",
                        help="repository-relative destination (used for build, extension, and verification)")
    args = parser.parse_args()
    if args.verify and args.extend_offmarket_meta:
        parser.error("--verify cannot be combined with --extend-offmarket-meta")
    result = (verify_assets(args.output_dir) if args.verify else
              extend_offmarket_meta_assets(args.output_dir) if args.extend_offmarket_meta else
              build_assets(args.output_dir))
    print(json.dumps(result, indent=2, ensure_ascii=False))
