from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.model_optimization_v2.evaluation import evaluate_increment
from experiments.model_optimization_v2.pipeline import blend_prices
from experiments.model_research_fresh.candidates import candidate_spec
from experiments.model_research_next.audit_months import build_audit
from experiments.model_research_next.methods import (
    apply_oof_calibration,
    apply_oof_prediction_band_calibration,
    fit_oof_blend_weight,
    fit_oof_calibration,
    fit_oof_prediction_band_calibration,
    validate_fold_plan,
)


PACKAGE = ROOT / "experiments/model_research_next"
OUTPUT = ROOT / "outputs/model_research_next"
PROTOCOL = json.loads((PACKAGE / "protocol.json").read_text(encoding="utf-8"))
REGISTRY = json.loads((PACKAGE / "method_registry.json").read_text(encoding="utf-8"))
CARDS = REGISTRY["cards"]
LEDGER = OUTPUT / "fit_ledger.jsonl"
STATE = OUTPUT / "run_state.json"
BASE_COMPONENTS = {
    "off_market": ("C0_LGBM", "Q50_LGBM", "RECENCY_LGBM"),
    "list_aware": ("C0_LGBM", "C0_HGBR", "Q50_LGBM", "CB_MAE", "RECENCY_LGBM"),
}
BASE_PIPELINES = ("C0", "Q50", "CB_MAE", "RECENCY")
FEATURE_COLUMNS = ("house_type_name", "house_style", "municipality", "community")
FIT_START_STATUSES = {"fit_start"}
_CODE_HASH_CACHE: dict[str, str] | None = None


class BudgetStop(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    os.replace(temporary, path)


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, separators=(",", ":"), default=str) + "\n")


def _code_hashes() -> dict[str, str]:
    global _CODE_HASH_CACHE
    if _CODE_HASH_CACHE is not None:
        return dict(_CODE_HASH_CACHE)
    relative = (
        "experiments/model_research_next/protocol.json",
        "experiments/model_research_next/method_registry.json",
        "experiments/model_research_next/methods.py",
        "experiments/model_research_next/diagnostics.py",
        "experiments/model_research_next/audit_months.py",
        "experiments/model_research_next/worker.py",
        "experiments/model_research_next/runner.py",
        "experiments/model_optimization_v2/folds.py",
        "experiments/model_optimization_v2/features.py",
        "experiments/model_optimization_v2/learners.py",
        "experiments/model_optimization_v2/worker.py",
        "experiments/model_optimization_v2/pipeline.py",
        "model_candidate_v2/winner_config.json",
        "model_candidate_v2/features.py",
        "model_candidate_v2/models/frozen_list_aware.json",
        PROTOCOL["source_data"],
    )
    _CODE_HASH_CACHE = {item: _sha256_file(ROOT / item) for item in relative}
    return dict(_CODE_HASH_CACHE)


def _runtime_versions() -> dict:
    names = ("numpy", "pandas", "pyarrow", "scikit-learn", "lightgbm", "catboost")
    versions = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": platform.python_version(), "packages": versions}


def _fit_starts() -> int:
    if not LEDGER.exists():
        return 0
    count = 0
    for line in LEDGER.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        count += record.get("event") == "fit_start"
    return count


def _elapsed_seconds() -> float:
    if not STATE.exists():
        return 0.0
    state = json.loads(STATE.read_text(encoding="utf-8"))
    start = datetime.fromisoformat(state["started_at_utc"])
    return max(0.0, (datetime.now(timezone.utc) - start).total_seconds())


def _ensure_state() -> dict:
    code_hashes = _code_hashes()
    if STATE.exists():
        state = json.loads(STATE.read_text(encoding="utf-8"))
        if state.get("run_id") != PROTOCOL["run_id"] or state.get("code_sha256") != code_hashes:
            raise ValueError("existing run state uses another protocol or code fingerprint; refusing to mix runs")
        return state
    state = {
        "run_id": PROTOCOL["run_id"],
        "started_at_utc": _utc_now(),
        "data_sha256": PROTOCOL["data_sha256"],
        "code_sha256": code_hashes,
        "runtime_versions": _runtime_versions(),
        "fit_cap": PROTOCOL["budget"]["fit_cap"],
        "development_minutes": PROTOCOL["budget"]["development_minutes"],
        "final_reserve_minutes": PROTOCOL["budget"]["final_reserve_minutes"],
    }
    _write_json(STATE, state)
    return state


def _within_budget(stage: str) -> bool:
    budget = PROTOCOL["budget"]
    wall_limit = budget["development_minutes"] if stage == "development" else budget["wallclock_minutes"]
    return _fit_starts() < budget["fit_cap"] and _elapsed_seconds() < wall_limit * 60


def _assert_budget(stage: str) -> None:
    if not _within_budget(stage):
        raise BudgetStop(f"stop before next fit: {_fit_starts()}/{PROTOCOL['budget']['fit_cap']} fits, "
                         f"{_elapsed_seconds()/60:.1f} minutes elapsed")


def _model_configs() -> tuple[dict, dict]:
    winner = json.loads((ROOT / "model_candidate_v2/winner_config.json").read_text(encoding="utf-8"))
    frozen_hgbr = json.loads((ROOT / "model_candidate_v2/models/frozen_list_aware.json").read_text(encoding="utf-8"))
    configs = {track: winner["locked"][track] for track in ("off_market", "list_aware")}
    return configs, frozen_hgbr


def _component_spec(track: str, component_id: str) -> dict:
    configs, frozen_hgbr = _model_configs()
    c0 = configs[track]
    if component_id == "C0_LGBM":
        return {"model_family": c0["model_family"], "model_config": dict(c0["model_config"], n_jobs=4),
                "method_id": "C0", "target_override": None}
    if component_id == "C0_HGBR" and track == "list_aware":
        return {"model_family": frozen_hgbr["model_family"], "model_config": frozen_hgbr["model_config"],
                "method_id": "C0", "target_override": "log_sold_price"}
    if component_id == "Q50_LGBM":
        config = dict(c0["model_config"], n_jobs=4)
        config.update(objective="quantile", alpha=0.5)
        return {"model_family": c0["model_family"], "model_config": config,
                "method_id": "C0", "target_override": None}
    if component_id == "CB_MAE" and track == "list_aware":
        spec = candidate_spec("CB_MAE", track, c0["model_config"])
        return {key: spec[key] for key in ("model_family", "model_config", "method_id", "target_override")}
    if component_id == "RECENCY_LGBM":
        return {"model_family": c0["model_family"], "model_config": dict(c0["model_config"], n_jobs=4),
                "method_id": "M01", "target_override": None}
    raise ValueError(f"unsupported component {component_id} for {track}")


def _component_paths(role: str, track: str, component_id: str, month: str) -> tuple[Path, Path]:
    root = OUTPUT / "predictions" / role / "components" / track
    stem = f"{component_id}__{month}"
    return root / f"{stem}.csv", root / f"{stem}.json"


def _cache_component(csv_path: Path, meta_path: Path, fingerprint: str) -> tuple[dict | None, pd.DataFrame | None]:
    if not csv_path.exists() or not meta_path.exists():
        return None, None
    try:
        metadata_record = json.loads(meta_path.read_text(encoding="utf-8"))
        if metadata_record.get("job_fingerprint") != fingerprint:
            return None, None
        if metadata_record.get("prediction_sha256") != _sha256_file(csv_path):
            return None, None
        frame = pd.read_csv(csv_path, dtype={"ml_num": str}, parse_dates=["date_sold"])
    except (OSError, ValueError, json.JSONDecodeError, pd.errors.ParserError):
        return None, None
    return metadata_record, frame


def _run_component(role: str, month: str, track: str, component_id: str) -> tuple[pd.DataFrame, dict]:
    spec = _component_spec(track, component_id)
    code_hashes = _code_hashes()
    fingerprint_data = {"run_id": PROTOCOL["run_id"], "role": role, "month": month,
                        "track": track, "component_id": component_id, "spec": spec,
                        "data_sha256": PROTOCOL["data_sha256"], "code_sha256": code_hashes}
    fingerprint = _sha256_text(_canonical(fingerprint_data))
    csv_path, meta_path = _component_paths(role, track, component_id, month)
    cached_meta, cached_frame = _cache_component(csv_path, meta_path, fingerprint)
    if cached_frame is not None:
        _append_jsonl(LEDGER, {"event": "cache_hit", "stage": role, "month": month, "track": track,
                               "component_id": component_id, "job_fingerprint": fingerprint,
                               "at_utc": _utc_now()})
        return cached_frame, cached_meta

    _assert_budget("development" if role in {"seed", "screen", "confirmation"} else "final")
    component_rel = csv_path.relative_to(ROOT)
    allowed_months = PROTOCOL["development_months"] if role != "final" else PROTOCOL["final_months"]
    job = {
        "stage": role,
        "month": month,
        "track": track,
        "allowed_months": allowed_months,
        "data_path": PROTOCOL["source_data"],
        "prediction_path": str(component_rel),
        "component_id": component_id,
        "spec": spec,
        "job_fingerprint": fingerprint,
        "data_sha256": PROTOCOL["data_sha256"],
        "code_sha256": code_hashes,
    }
    job_path = OUTPUT / "jobs" / f"{role}_{track}_{component_id}_{month}.json"
    _write_json(job_path, job)
    fit_id = str(uuid.uuid4())
    _append_jsonl(LEDGER, {"event": "fit_start", "fit_id": fit_id, "kind": "learner",
                           "stage": role, "month": month, "track": track, "component_id": component_id,
                           "spec": spec, "job_fingerprint": fingerprint, "data_sha256": PROTOCOL["data_sha256"],
                           "code_sha256": code_hashes, "started_at_utc": _utc_now()})
    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4",
                "NUMEXPR_NUM_THREADS": "4"})
    started = time.perf_counter()
    try:
        result = subprocess.run(
            [sys.executable, "-m", "experiments.model_research_next.worker", str(job_path)],
            cwd=ROOT, env=env, capture_output=True, text=True,
            timeout=PROTOCOL["budget"]["fit_timeout_seconds"], check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"worker exited {result.returncode}: {result.stderr[-2500:]}")
        metadata_record = json.loads(result.stdout.strip().splitlines()[-1])
        frame = pd.read_csv(csv_path, dtype={"ml_num": str}, parse_dates=["date_sold"])
        if metadata_record.get("prediction_sha256") != _sha256_file(csv_path):
            raise ValueError("worker prediction hash failed its post-run check")
        metadata_record["job_fingerprint"] = fingerprint
        _write_json(meta_path, metadata_record)
        _append_jsonl(LEDGER, {"event": "fit_finish", "fit_id": fit_id, "kind": "learner",
                               "status": "completed", "stage": role, "month": month, "track": track,
                               "component_id": component_id, "fit_seconds_wall": time.perf_counter() - started,
                               "metadata": metadata_record, "finished_at_utc": _utc_now()})
        return frame, metadata_record
    except subprocess.TimeoutExpired as exc:
        _append_jsonl(LEDGER, {"event": "fit_finish", "fit_id": fit_id, "kind": "learner",
                               "status": "timed_out", "stage": role, "month": month, "track": track,
                               "component_id": component_id, "fit_seconds_wall": time.perf_counter() - started,
                               "finished_at_utc": _utc_now(), "error": str(exc)})
        raise
    except Exception as exc:
        _append_jsonl(LEDGER, {"event": "fit_finish", "fit_id": fit_id, "kind": "learner",
                               "status": "failed", "stage": role, "month": month, "track": track,
                               "component_id": component_id, "fit_seconds_wall": time.perf_counter() - started,
                               "finished_at_utc": _utc_now(), "error": f"{type(exc).__name__}: {exc}"})
        raise


def _pipeline_paths(role: str, track: str, pipeline_id: str, month: str) -> tuple[Path, Path]:
    root = OUTPUT / "predictions" / role / "pipelines" / track
    return root / f"{pipeline_id}__{month}.csv", root / f"{pipeline_id}__{month}.json"


def _make_pipeline(role: str, track: str, month: str, pipeline_id: str,
                   components: dict[str, pd.DataFrame]) -> pd.DataFrame:
    source_component = {"C0": "C0_LGBM", "Q50": "Q50_LGBM", "CB_MAE": "CB_MAE",
                        "RECENCY": "RECENCY_LGBM"}[pipeline_id]
    source = components[source_component].copy()
    prediction = source.raw_prediction.to_numpy(dtype=float)
    if track == "list_aware":
        frozen = components["C0_HGBR"]
        keys = ["ml_num", "sold_price", "date_sold"]
        left, right = source.set_index("ml_num"), frozen.set_index("ml_num")
        if not left.index.equals(right.index) or not np.array_equal(left.sold_price, right.sold_price):
            raise ValueError(f"component rows do not align for {track}/{pipeline_id}/{month}")
        if not np.array_equal(pd.to_datetime(left.date_sold).to_numpy(), pd.to_datetime(right.date_sold).to_numpy()):
            raise ValueError("component dates do not align")
        prediction = blend_prices(prediction, frozen.raw_prediction.to_numpy(dtype=float), 0.75)
    else:
        prediction = np.round(prediction)
    if not np.isfinite(prediction).all() or (prediction <= 0).any():
        raise ValueError("base pipeline predictions must be finite positive dollar prices")
    source["predicted_price"] = prediction
    source["pipeline_id"] = pipeline_id
    path, meta_path = _pipeline_paths(role, track, pipeline_id, month)
    path.parent.mkdir(parents=True, exist_ok=True)
    source.to_csv(path, index=False)
    _write_json(meta_path, {"track": track, "month": month, "pipeline_id": pipeline_id,
                            "prediction_sha256": _sha256_file(path), "data_sha256": PROTOCOL["data_sha256"],
                            "code_sha256": _code_hashes(), "created_at_utc": _utc_now()})
    return source


def _pipeline_frame(track: str, pipeline_id: str, month: str, role: str) -> pd.DataFrame:
    path, _ = _pipeline_paths(role, track, pipeline_id, month)
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, dtype={"ml_num": str}, parse_dates=["date_sold"])


def _features(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in FEATURE_COLUMNS if column in frame.columns]
    return frame[["ml_num", *columns]].copy()


def _past_oof(history: dict, track: str, pipeline_id: str, cutoff: pd.Timestamp) -> pd.DataFrame:
    frames = history.get((track, pipeline_id), [])
    if not frames:
        raise ValueError(f"no earlier OOF predictions for {track}/{pipeline_id}")
    result = pd.concat(frames, ignore_index=True)
    result["date_sold"] = pd.to_datetime(result.date_sold)
    result = result.loc[result.date_sold < cutoff].copy()
    if result.empty:
        raise ValueError("no strictly earlier OOF predictions before outer cutoff")
    return result


def _oof_blend_frame(history: dict, track: str, parent_id: str, challenger_id: str,
                     cutoff: pd.Timestamp) -> pd.DataFrame:
    left = _past_oof(history, track, parent_id, cutoff)
    right = _past_oof(history, track, challenger_id, cutoff)
    keys = ["month", "ml_num"]
    columns = keys + ["sold_price", "predicted_price"]
    left = left[columns].rename(columns={"predicted_price": "parent_prediction"})
    right = right[columns].rename(columns={"predicted_price": "challenger_prediction"})
    joined = left.merge(right, on=keys, suffixes=("_parent", "_challenger"), validate="one_to_one")
    if not np.array_equal(joined.sold_price_parent.to_numpy(), joined.sold_price_challenger.to_numpy()):
        raise ValueError("OOF blend training targets do not align")
    return joined.rename(columns={"sold_price_parent": "sold_price"})[
        keys + ["sold_price", "parent_prediction", "challenger_prediction"]]


def _fit_event(kind: str, stage: str, month: str, track: str, card_id: str,
               parameters: dict, data_sha: str, fit_function):
    _assert_budget("development" if stage in {"screen", "confirmation"} else "final")
    fit_id = str(uuid.uuid4())
    history_sha = _sha256_text(_canonical(parameters))
    _append_jsonl(LEDGER, {"event": "fit_start", "fit_id": fit_id, "kind": kind, "stage": stage,
                           "month": month, "track": track, "card_id": card_id, "parameters": parameters,
                           "oof_signature": history_sha, "data_sha256": data_sha,
                           "code_sha256": _code_hashes(), "started_at_utc": _utc_now()})
    started = time.perf_counter()
    try:
        result = fit_function()
        elapsed = time.perf_counter() - started
        _append_jsonl(LEDGER, {"event": "fit_finish", "fit_id": fit_id, "kind": kind, "status": "completed",
                               "stage": stage, "month": month, "track": track, "card_id": card_id,
                               "fit_seconds_wall": elapsed, "finished_at_utc": _utc_now()})
        return result, elapsed
    except Exception as exc:
        _append_jsonl(LEDGER, {"event": "fit_finish", "fit_id": fit_id, "kind": kind, "status": "failed",
                               "stage": stage, "month": month, "track": track, "card_id": card_id,
                               "fit_seconds_wall": time.perf_counter() - started,
                               "finished_at_utc": _utc_now(), "error": f"{type(exc).__name__}: {exc}"})
        raise


def _candidate_path(stage: str, track: str, card_id: str, month: str) -> tuple[Path, Path]:
    root = OUTPUT / "predictions" / stage / "candidates" / track
    return root / f"{card_id}__{month}.csv", root / f"{card_id}__{month}.json"


def _history_signature(history: dict, track: str, pipeline_ids: tuple[str, ...], cutoff: pd.Timestamp) -> str:
    pieces = []
    for pipeline_id in pipeline_ids:
        frame = _past_oof(history, track, pipeline_id, cutoff)
        columns = [column for column in ("month", "ml_num", "date_sold", "sold_price", "predicted_price") if column in frame]
        frame = frame[columns].sort_values(["month", "ml_num"], kind="stable")
        pieces.append(pd.util.hash_pandas_object(frame, index=False).to_numpy(dtype=np.uint64).tobytes())
    return hashlib.sha256(b"".join(pieces)).hexdigest()


def _candidate_fingerprint(card: dict, stage: str, month: str, oof_signature: str | None) -> str:
    return _sha256_text(_canonical({"card": card, "stage": stage, "month": month,
                                    "oof_signature": oof_signature, "data_sha256": PROTOCOL["data_sha256"],
                                    "code_sha256": _code_hashes()}))


def _candidate_cache(stage: str, track: str, card_id: str, month: str,
                     fingerprint: str) -> tuple[pd.DataFrame | None, dict | None]:
    path, meta_path = _candidate_path(stage, track, card_id, month)
    if not path.exists() or not meta_path.exists():
        return None, None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("job_fingerprint") != fingerprint or meta.get("prediction_sha256") != _sha256_file(path):
            return None, None
        frame = pd.read_csv(path, dtype={"ml_num": str}, parse_dates=["date_sold"])
    except (OSError, ValueError, json.JSONDecodeError, pd.errors.ParserError):
        return None, None
    _append_jsonl(LEDGER, {"event": "cache_hit", "stage": stage, "month": month, "track": track,
                           "card_id": card_id, "job_fingerprint": fingerprint, "at_utc": _utc_now()})
    return frame, meta


def _make_candidate(card: dict, stage: str, month: str, current: dict[str, pd.DataFrame],
                    history: dict) -> tuple[pd.DataFrame, dict]:
    track = card["track"]
    operation = card["operation"]
    cutoff = pd.Period(month, freq="M").start_time
    base_id = card["base"]
    base = current[base_id].copy()
    fit_record = {"operation": operation, "base": base_id, "card_id": card["card_id"]}
    signature = None

    if operation == "raw_pipeline":
        fingerprint = _candidate_fingerprint(card, stage, month, signature)
        cached, cached_meta = _candidate_cache(stage, track, card["card_id"], month, fingerprint)
        if cached is not None:
            return cached, cached_meta
        result = base.copy()
        fit_record["fit_count"] = 0
    elif operation in {"oof_global_calibration", "oof_group_calibration", "oof_predicted_band_calibration"}:
        oof = _past_oof(history, track, base_id, cutoff)
        params = dict(card.get("calibration", {}))
        signature = _history_signature(history, track, (base_id,), cutoff)
        fingerprint = _candidate_fingerprint(card, stage, month, signature)
        cached, cached_meta = _candidate_cache(stage, track, card["card_id"], month, fingerprint)
        if cached is not None:
            return cached, cached_meta
        if operation == "oof_predicted_band_calibration":
            groups = tuple(params.get("groups", ()))
            fit_args = dict(cutoff=cutoff, group_columns=groups, n_bands=params["n_bands"],
                            pseudocount=params["pseudocount"], min_group=params["min_group"],
                            clip=tuple(params["clip"]), max_months=params.get("max_months"),
                            min_oof_months=params.get("min_oof_months", 1))
            fitted, seconds = _fit_event("calibrator", stage, month, track, card["card_id"],
                                         {**fit_args, "oof_signature": signature}, PROTOCOL["data_sha256"],
                                         lambda: fit_oof_prediction_band_calibration(oof, **fit_args))
            prediction = apply_oof_prediction_band_calibration(
                base.predicted_price.to_numpy(dtype=float), _features(base), fitted)
        else:
            groups = tuple(params.get("groups", ()))
            fit_args = dict(cutoff=cutoff, group_columns=groups,
                            pseudocount=params.get("pseudocount", 200), min_group=params.get("min_group", 100),
                            clip=tuple(params.get("clip", (0.95, 1.05))),
                            max_months=params.get("max_months"), min_oof_months=params.get("min_oof_months", 1))
            fitted, seconds = _fit_event("calibrator", stage, month, track, card["card_id"],
                                         {**fit_args, "oof_signature": signature}, PROTOCOL["data_sha256"],
                                         lambda: fit_oof_calibration(oof, **fit_args))
            prediction = apply_oof_calibration(base.predicted_price.to_numpy(dtype=float), _features(base), fitted)
        result = base.copy()
        result["predicted_price"] = np.round(prediction)
        fit_record.update({"fit_count": 1, "fit_seconds": seconds, "calibration": fitted,
                           "oof_signature": signature})
    elif operation == "oof_blend":
        challenger_id = card["challenger"]
        challenger = current[challenger_id].copy()
        oof = _oof_blend_frame(history, track, base_id, challenger_id, cutoff)
        signature = _history_signature(history, track, (base_id, challenger_id), cutoff)
        fingerprint = _candidate_fingerprint(card, stage, month, signature)
        cached, cached_meta = _candidate_cache(stage, track, card["card_id"], month, fingerprint)
        if cached is not None:
            return cached, cached_meta
        grid = tuple(card["blend"]["weight_grid"])
        weight, seconds = _fit_event("blend_weight", stage, month, track, card["card_id"],
                                     {"grid": grid, "oof_signature": signature}, PROTOCOL["data_sha256"],
                                     lambda: fit_oof_blend_weight(oof, weights=grid))
        right = challenger.set_index("ml_num").loc[base.ml_num.astype(str)].predicted_price.to_numpy(dtype=float)
        if not np.array_equal(base.sold_price.to_numpy(dtype=float),
                              challenger.set_index("ml_num").loc[base.ml_num.astype(str)].sold_price.to_numpy(dtype=float)):
            raise ValueError("same-fold blend candidate labels do not align")
        result = base.copy()
        result["predicted_price"] = np.round((1.0 - weight) * base.predicted_price.to_numpy(dtype=float) + weight * right)
        fit_record.update({"fit_count": 1, "fit_seconds": seconds, "challenger": challenger_id,
                           "challenger_weight": weight, "oof_signature": signature})
    else:
        raise ValueError(f"unsupported candidate operation: {operation}")

    result["card_id"] = card["card_id"]
    output_path, meta_path = _candidate_path(stage, track, card["card_id"], month)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    metadata_record = {"card_id": card["card_id"], "track": track, "month": month, "stage": stage,
                       "fit": fit_record, "prediction_sha256": _sha256_file(output_path),
                       "data_sha256": PROTOCOL["data_sha256"], "code_sha256": _code_hashes(),
                       "job_fingerprint": fingerprint, "oof_signature": signature,
                       "created_at_utc": _utc_now()}
    _write_json(meta_path, metadata_record)
    return result, metadata_record


def _read_candidate(stage: str, track: str, card_id: str, month: str) -> pd.DataFrame:
    path, meta_path = _candidate_path(stage, track, card_id, month)
    metadata_record = json.loads(meta_path.read_text(encoding="utf-8"))
    if metadata_record.get("prediction_sha256") != _sha256_file(path):
        raise ValueError(f"candidate prediction hash mismatch: {path}")
    return pd.read_csv(path, dtype={"ml_num": str}, parse_dates=["date_sold"])


def _frame_score(frame: pd.DataFrame) -> dict:
    actual = frame.sold_price.to_numpy(dtype=float)
    prediction = frame.predicted_price.to_numpy(dtype=float)
    ape = 100.0 * np.abs(prediction - actual) / actual
    return {"n": int(len(frame)), "mdape_percent": float(np.median(ape)), "mape_percent": float(np.mean(ape))}


def _screen_rank() -> dict:
    results = {}
    shortlist = {}
    months = PROTOCOL["screen_months"]
    for track in ("off_market", "list_aware"):
        parent = pd.concat([_pipeline_frame(track, "C0", month, "screen") for month in months], ignore_index=True)
        ranked = []
        track_cards = [card for card in CARDS if card["track"] == track]
        for card in track_cards:
            candidate = pd.concat([_read_candidate("screen", track, card["card_id"], month)
                                   for month in months], ignore_index=True)
            score = _frame_score(candidate)
            by_month = {month: _frame_score(candidate.loc[candidate.month.astype(str).eq(month)]) for month in months}
            parent_score = _frame_score(parent)
            ranked.append({"card_id": card["card_id"], "n": score["n"],
                           "parent_mdape_percent": parent_score["mdape_percent"],
                           "candidate_mdape_percent": score["mdape_percent"],
                           "relative_mdape_improvement": (parent_score["mdape_percent"] - score["mdape_percent"]) / parent_score["mdape_percent"],
                           "parent_mape_percent": parent_score["mape_percent"],
                           "candidate_mape_percent": score["mape_percent"],
                           "relative_mape_change": (score["mape_percent"] - parent_score["mape_percent"]) / parent_score["mape_percent"],
                           "monthly": by_month})
        ranked.sort(key=lambda row: (row["candidate_mdape_percent"], row["candidate_mape_percent"], row["card_id"]))
        results[track] = ranked
        shortlist[track] = [row["card_id"] for row in ranked[:PROTOCOL["screening"]["shortlist_per_track"]]]
    record = {"screen_months": months, "data_sha256": PROTOCOL["data_sha256"],
              "screen_results_are_not_confirmation": True, "rankings": results,
              "shortlist": shortlist, "created_at_utc": _utc_now()}
    _write_json(OUTPUT / "screen_results.json", record)
    _write_json(OUTPUT / "candidate_shortlist.json", record)
    return record


def _gate(report: dict, min_months: int) -> dict:
    ci = report.get("paired_home_bootstrap", {}).get("ci95_percentage_points", [float("nan"), float("nan")])
    criteria = {
        "mdape_improvement_at_least_1pct": report.get("relative_mdape_improvement", float("-inf")) >= 0.01,
        "months_improved_at_least_fraction": report.get("months_improved", 0) >= min_months,
        "worst_month_regression_at_most_5pct": report.get("worst_month_relative_regression", float("inf")) <= 0.05,
        "mape_relative_regression_at_most_1pct": report.get("mape_relative_change", float("inf")) <= 0.01,
        "paired_home_bootstrap_upper_below_zero": len(ci) == 2 and ci[1] < 0.0,
    }
    return {"passed": all(criteria.values()), "criteria": criteria}


def _confirmation_results(shortlist: dict) -> dict:
    months = PROTOCOL["confirmation_months"]
    results = {"confirmation_months": months, "data_sha256": PROTOCOL["data_sha256"], "tracks": {}}
    selected = {}
    for track in ("off_market", "list_aware"):
        parent = pd.concat([_pipeline_frame(track, "C0", month, "confirmation") for month in months], ignore_index=True)
        track_results = {}
        for index, card_id in enumerate(shortlist[track]):
            candidate = pd.concat([_read_candidate("confirmation", track, card_id, month)
                                   for month in months], ignore_index=True)
            report = evaluate_increment(parent, candidate,
                                        repeats=PROTOCOL["budget"]["bootstrap_repeats_confirmation"],
                                        seed=1000 + index + (0 if track == "off_market" else 100))
            report["gate"] = _gate(report, PROTOCOL["development_gate"]["min_months_improved"])
            track_results[card_id] = report
        passers = [(card_id, report) for card_id, report in track_results.items() if report["gate"]["passed"]]
        passers.sort(key=lambda item: (item[1]["candidate"]["mdape_percent"],
                                       item[1]["candidate"]["mape_percent"], item[0]))
        selected[track] = ({"card_id": passers[0][0], "development_report": passers[0][1]}
                           if passers else None)
        results["tracks"][track] = track_results
    results["selected"] = selected
    results["both_tracks_qualified"] = all(selected[track] is not None for track in selected)
    results["created_at_utc"] = _utc_now()
    _write_json(OUTPUT / "confirmation_results.json", results)
    return results


def _write_method_status(shortlist: dict | None, confirmation: dict | None) -> None:
    status = {}
    for card in CARDS:
        card_id, track = card["card_id"], card["track"]
        item = {"card_id": card_id, "track": track, "status": "not_started"}
        if shortlist is not None:
            item["screened"] = True
            item["selected_for_confirmation"] = card_id in shortlist[track]
            item["screen_status"] = "completed" if item["selected_for_confirmation"] else "budget_skipped"
        if confirmation is not None and card_id in confirmation["tracks"][track]:
            item["confirmation_report"] = confirmation["tracks"][track][card_id]
            item["status"] = "completed" if confirmation["tracks"][track][card_id]["gate"]["passed"] else "no_gain"
        elif shortlist is not None:
            item["status"] = "completed" if card_id in shortlist[track] else "budget_skipped"
        _write_json(OUTPUT / "method_status.json", status | {card_id: item})
        status[card_id] = item


def _preflight() -> dict:
    audit = build_audit()
    if audit["data_sha256"] != PROTOCOL["data_sha256"]:
        raise ValueError("data source hash differs from the frozen protocol")
    previous = set(audit["prior_month_evidence"]["used_months_union"])
    screen_with_seed = tuple(PROTOCOL["oof_seed_months"] + PROTOCOL["screen_months"])
    validate_fold_plan(screen_with_seed, PROTOCOL["confirmation_months"], PROTOCOL["final_months"], previous)
    by_month = {row["month"]: row for row in audit["unused_candidate_folds"]}
    selected_months = PROTOCOL["development_months"]
    for month in selected_months:
        if month not in by_month:
            raise ValueError(f"fold month is not present in the audit's unused month inventory: {month}")
        actual = by_month[month]
        expected = PROTOCOL["fold_counts"][month]
        if actual["train_rows"] != expected["train_rows"] or actual["evaluation_rows"] != expected["evaluation_rows"]:
            raise ValueError(f"fold counts differ from frozen protocol for {month}: {actual}")
    hashes = _code_hashes()
    record = {"created_at_utc": _utc_now(), "data_sha256": audit["data_sha256"],
              "code_sha256": hashes, "runtime_versions": _runtime_versions(),
              "audit_path": PROTOCOL["month_audit"], "residual_diagnostic": PROTOCOL["residual_diagnostic"],
              "development_months": selected_months, "final_months": PROTOCOL["final_months"],
              "fold_counts": PROTOCOL["fold_counts"], "previously_evaluated_months": sorted(previous),
              "final_labels_opened": False}
    _write_json(OUTPUT / "preflight_manifest.json", record)
    return record


def _load_method_status() -> dict:
    path = OUTPUT / "method_status.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _process_role(role: str, month: str, history: dict, cards: list[dict]) -> dict[str, dict[str, pd.DataFrame]]:
    current: dict[str, dict[str, pd.DataFrame]] = {}
    for track in ("off_market", "list_aware"):
        components = {}
        for component_id in BASE_COMPONENTS[track]:
            components[component_id], _ = _run_component(role, month, track, component_id)
        pipelines = {}
        for pipeline_id in BASE_PIPELINES:
            if pipeline_id == "CB_MAE" and track == "off_market":
                continue
            pipelines[pipeline_id] = _make_pipeline(role, track, month, pipeline_id, components)
        current[track] = pipelines

    if role != "seed":
        for card in cards:
            _make_candidate(card, role, month, current[card["track"]], history)
    for track in ("off_market", "list_aware"):
        for pipeline_id, frame in current[track].items():
            history.setdefault((track, pipeline_id), []).append(frame.copy())
    return current


def run_development() -> dict:
    if (OUTPUT / "freeze_manifest.json").exists():
        existing = json.loads((OUTPUT / "freeze_manifest.json").read_text(encoding="utf-8"))
        if existing.get("development_complete"):
            return existing
    _ensure_state()
    _preflight()
    if STATE.exists():
        state = json.loads(STATE.read_text(encoding="utf-8"))
        state["active_stage"] = "development"
        _write_json(STATE, state)

    shortlist_path = OUTPUT / "candidate_shortlist.json"
    shortlist_record = json.loads(shortlist_path.read_text(encoding="utf-8")) if shortlist_path.exists() else None
    all_months = PROTOCOL["development_months"]
    roles = {month: "seed" if month in PROTOCOL["oof_seed_months"] else
             "screen" if month in PROTOCOL["screen_months"] else "confirmation" for month in all_months}
    history: dict[tuple[str, str], list[pd.DataFrame]] = {}
    for month in all_months:
        role = roles[month]
        if role == "confirmation" and shortlist_record is None:
            shortlist_record = _screen_rank()
            _write_method_status(shortlist_record["shortlist"], None)
        if role in {"seed", "screen"}:
            cards = [card for card in CARDS if role == "screen" and card["card_id"] in
                     {item["card_id"] for item in CARDS}]
        else:
            selected_ids = set(shortlist_record["shortlist"]["off_market"] + shortlist_record["shortlist"]["list_aware"])
            cards = [card for card in CARDS if card["card_id"] in selected_ids]
        _process_role(role, month, history, cards)
        if role == "screen" and month == PROTOCOL["screen_months"][-1] and shortlist_record is None:
            shortlist_record = _screen_rank()
            _write_method_status(shortlist_record["shortlist"], None)
        state = json.loads(STATE.read_text(encoding="utf-8"))
        state["last_completed_month"] = month
        state["fit_attempts"] = _fit_starts()
        state["elapsed_minutes"] = round(_elapsed_seconds() / 60.0, 3)
        _write_json(STATE, state)

    if shortlist_record is None:
        shortlist_record = _screen_rank()
    confirmation = _confirmation_results(shortlist_record["shortlist"])
    _write_method_status(shortlist_record["shortlist"], confirmation)
    selected = confirmation["selected"]
    freeze = {
        "run_id": PROTOCOL["run_id"],
        "created_at_utc": _utc_now(),
        "data_sha256": PROTOCOL["data_sha256"],
        "code_sha256": _code_hashes(),
        "screen_result_sha256": _sha256_file(OUTPUT / "screen_results.json"),
        "confirmation_result_sha256": _sha256_file(OUTPUT / "confirmation_results.json"),
        "selected": {track: (None if selected[track] is None else {"card_id": selected[track]["card_id"]})
                     for track in selected},
        "both_tracks_qualified": confirmation["both_tracks_qualified"],
        "development_complete": True,
        "final_months_unlocked": confirmation["both_tracks_qualified"],
        "final_labels_opened": False,
        "fit_attempts": _fit_starts(),
        "elapsed_minutes": round(_elapsed_seconds() / 60.0, 3),
        "reasons": {track: ("selected best passing screen-shortlisted candidate" if selected[track]
                            else "no confirmation candidate passed all preregistered gates")
                    for track in selected},
    }
    _write_json(OUTPUT / "freeze_manifest.json", freeze)
    state = json.loads(STATE.read_text(encoding="utf-8"))
    state.update({"active_stage": "development_complete", "development_complete": True,
                  "both_tracks_qualified": freeze["both_tracks_qualified"], "fit_attempts": _fit_starts(),
                  "elapsed_minutes": round(_elapsed_seconds() / 60.0, 3)})
    _write_json(STATE, state)
    return freeze


def run_final() -> dict:
    freeze_path = OUTPUT / "freeze_manifest.json"
    if not freeze_path.exists():
        raise ValueError("final month labels remain locked until a development freeze exists")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if not freeze.get("final_months_unlocked") or not freeze.get("both_tracks_qualified"):
        raise ValueError("final month labels remain locked because both tracks did not pass development")
    if (OUTPUT / "final_results.json").exists():
        raise ValueError("the final months have already been evaluated once")
    _assert_budget("final")
    # Final evaluation is intentionally dispatched only after the development
    # freeze. The selected estimator/postprocessor pair is fixed at this point.
    selected = {track: freeze["selected"][track]["card_id"] for track in ("off_market", "list_aware")}
    selected_cards = {card_id: next(card for card in CARDS if card["card_id"] == card_id)
                      for card_id in selected.values()}
    history: dict[tuple[str, str], list[pd.DataFrame]] = {}
    for month in PROTOCOL["development_months"]:
        role = "seed" if month in PROTOCOL["oof_seed_months"] else \
            "screen" if month in PROTOCOL["screen_months"] else "confirmation"
        for track in ("off_market", "list_aware"):
            for pipeline_id in BASE_PIPELINES:
                if pipeline_id == "CB_MAE" and track == "off_market":
                    continue
                history.setdefault((track, pipeline_id), []).append(_pipeline_frame(track, pipeline_id, month, role))
    reports = {"final_months": PROTOCOL["final_months"], "selected": selected,
               "final_labels_opened": True, "tracks": {}}
    for month in PROTOCOL["final_months"]:
        current = _process_role("final", month, history, list(selected_cards.values()))
        # _process_role writes every selected card; ignore the other track's card
        # output when its card is being scored.
        for track in ("off_market", "list_aware"):
            parent = _pipeline_frame(track, "C0", month, "final")
            card_id = selected[track]
            candidate = _read_candidate("final", track, card_id, month)
            reports["tracks"].setdefault(track, {"parent_rows": [], "candidate_rows": []})
            reports["tracks"][track]["parent_rows"].append(parent)
            reports["tracks"][track]["candidate_rows"].append(candidate)
    for track in ("off_market", "list_aware"):
        parent = pd.concat(reports["tracks"][track].pop("parent_rows"), ignore_index=True)
        candidate = pd.concat(reports["tracks"][track].pop("candidate_rows"), ignore_index=True)
        report = evaluate_increment(parent, candidate,
                                    repeats=PROTOCOL["budget"]["bootstrap_repeats_confirmation"],
                                    seed=9000 + (0 if track == "off_market" else 1))
        report["gate"] = _gate(report, PROTOCOL["final_gate"]["months_must_improve"])
        reports["tracks"][track] = report
    reports["both_tracks_pass"] = all(reports["tracks"][track]["gate"]["passed"] for track in reports["tracks"])
    reports["created_at_utc"] = _utc_now()
    _write_json(OUTPUT / "final_results.json", reports)
    freeze["final_months_unlocked"] = False
    freeze["final_labels_opened"] = True
    freeze["final_pass"] = reports["both_tracks_pass"]
    freeze["final_results_sha256"] = _sha256_file(OUTPUT / "final_results.json")
    _write_json(freeze_path, freeze)
    return reports


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("preflight", "development", "final", "all"), default="development")
    args = parser.parse_args()
    result = {}
    if args.stage == "preflight":
        result["preflight"] = _preflight()
    if args.stage in {"development", "all"}:
        result["development"] = run_development()
    if args.stage == "final":
        result["final"] = run_final()
    elif args.stage == "all" and result["development"].get("final_months_unlocked"):
        result["final"] = run_final()
    print(json.dumps({"completed_at_utc": _utc_now(), "fit_attempts": _fit_starts(),
                      "elapsed_minutes": round(_elapsed_seconds() / 60.0, 3), "result": result}, indent=2))


if __name__ == "__main__":
    main()
