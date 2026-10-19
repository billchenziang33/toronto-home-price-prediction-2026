from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import pipeline_fingerprint
from .calibration import (apply_local_residual, apply_multiplier_calibration,
                          apply_type_multiplier, fit_multiplier, fit_type_multiplier)
from .evaluation import _score, evaluate_increment
from .learners import make_learner
from .pipeline import blend_prices

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "model_optimization_v2"
PROTOCOL_PATH = PACKAGE_DIR / "protocol.json"
REGISTRY_PATH = PACKAGE_DIR / "method_registry.json"
GOAL_START_FALLBACK = "2026-10-19T00:27:43+00:00"
TRACKS = ("off_market", "list_aware")
BASELINE_MONTHS = ("2025-02", "2025-04", "2025-06", "2025-07", "2025-10", "2025-11")
SCREEN_MONTHS = ("2025-06", "2025-11")
CODE_FILES = ("features.py", "learners.py", "worker.py", "pipeline.py", "calibration.py", "runner.py")


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n",
                     encoding="utf-8")
    os.replace(temp, path)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _runtime_versions() -> dict:
    packages = ("numpy", "pandas", "scikit-learn", "lightgbm", "catboost", "xgboost", "pyarrow")
    versions = {"python": sys.version.split()[0]}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _source_hashes() -> dict:
    files = {name: _sha256(PACKAGE_DIR / name) for name in CODE_FILES}
    files["model_candidate_v2/features.py"] = _sha256(PROJECT_ROOT / "model_candidate_v2" / "features.py")
    files["model_candidate_v2/winner_config.json"] = _sha256(PROJECT_ROOT / "model_candidate_v2" / "winner_config.json")
    return files


def _paths():
    return {"state": OUTPUT_ROOT / "run_state.json", "ledger": OUTPUT_ROOT / "trial_ledger.jsonl",
            "registry": OUTPUT_ROOT / "method_status.json", "audit": OUTPUT_ROOT / "preflight.json"}


def _load_state(protocol: dict) -> dict:
    path = _paths()["state"]
    if path.exists():
        state = _json(path)
        events = _ledger_events()
        terminal = {"completed", "invalid", "timed_out", "interrupted", "aborted"}
        completed_attempts = {event.get("attempt") for event in events
                              if event.get("attempt") is not None and event.get("status") in terminal}
        for event in events:
            attempt = event.get("attempt")
            if attempt is not None and event.get("status") == "running" and attempt not in completed_attempts:
                _ledger({"attempt": attempt, "stage": event.get("stage"), "key": event.get("key"),
                         "track": event.get("track"), "month": event.get("month"),
                         "method_id": event.get("method_id"), "status": "interrupted",
                         "counted_fit": True, "reason": "runner was interrupted during fit"})
                completed_attempts.add(attempt)
        state["active_stage"] = None
        if state.get("status") != "final_benchmark_completed":
            state["status"] = "running"
        _flush_state(state)
        return state
    state = {"run_id": protocol["run_id"], "started_at_utc": protocol.get("goal_started_at_utc", GOAL_START_FALLBACK),
             "status": "running", "fit_attempts": 0, "stage_fits": {}, "last_updated_utc": _now().isoformat(),
             "active_stage": None, "notes": ["timer includes implementation from the active goal start"]}
    _atomic_json(path, state)
    return state


def _elapsed(state: dict) -> float:
    start = datetime.fromisoformat(state["started_at_utc"])
    return max(0.0, (_now() - start).total_seconds())


def _flush_state(state: dict) -> None:
    state["last_updated_utc"] = _now().isoformat()
    _atomic_json(_paths()["state"], state)


def _ledger(event: dict) -> None:
    path = _paths()["ledger"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps({"timestamp_utc": _now().isoformat(), **event}, sort_keys=True, default=str) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _ledger_events() -> list[dict]:
    path = _paths()["ledger"]
    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(json.loads(line))
    return events


def can_reuse_after_scoped_code_change(method_id: str, previous: dict, current: dict) -> bool:
    """Reuse unaffected artifacts after the as-of-area fix; M08/M09 must refit."""
    changed = {name for name in set(previous) | set(current) if previous.get(name) != current.get(name)}
    if changed - {"runner.py", "features.py"}:
        return False
    if method_id in {"M08", "M09"} and "features.py" in changed:
        return False
    return True


def _completed_event(key: str) -> dict | None:
    matches = [event for event in _ledger_events() if event.get("key") == key and event.get("status") == "completed"]
    return matches[-1] if matches else None


def _has_terminal_event(method_id: str, track: str, month: str, status: str) -> bool:
    return any(event.get("method_id") == method_id and event.get("track") == track and
               event.get("month") == month and event.get("status") == status for event in _ledger_events())


def _semantic_cached_prediction(path: Path, fingerprint: str, key: str, method_id: str,
                                expected_config: dict, code_hashes: dict, versions: dict,
                                data_hash: str) -> dict | None:
    meta_path = _model_meta_path(path)
    if not path.exists() or not meta_path.exists():
        return None
    meta = _json(meta_path)
    if meta.get("prediction_sha256") != _sha256(path):
        return None
    if meta.get("fingerprint") == fingerprint:
        return meta
    event = _completed_event(key)
    if event is None or event.get("data_sha256") != data_hash or event.get("runtime_versions") != versions:
        return None
    previous_config = event.get("config")
    if previous_config is None:
        previous_config = {"key": key, "track": event.get("track"), "month": event.get("month"),
                           "method_id": event.get("method_id"), "model_family": event.get("family"),
                           "model_config": event.get("model_config"),
                           "target_override": event.get("target_override"),
                           "threads": event.get("threads"), "fold_policy": "date_sold < month start"}
    config_matches = previous_config == expected_config or (event.get("config") == expected_config)
    if not config_matches or not can_reuse_after_scoped_code_change(method_id, event.get("code_sha256", {}), code_hashes):
        return None
    previous_fingerprint = meta.get("fingerprint")
    meta["prior_fingerprint"] = previous_fingerprint
    meta["fingerprint"] = fingerprint
    meta["compatibility_reuse"] = {"method_id": method_id,
        "reason": "code change is limited to runner orchestration and M08/M09-only house_area parsing",
        "source_completed_event_timestamp_utc": event.get("timestamp_utc"),
        "source_completed_event_attempt": event.get("attempt")}
    _atomic_json(meta_path, meta)
    _ledger({"stage": event.get("stage"), "key": key, "method_id": method_id,
             "status": "semantic_cache_reuse", "counted_fit": False,
             "prior_fingerprint": previous_fingerprint, "fingerprint": fingerprint,
             "reason": meta["compatibility_reuse"]["reason"],
             "prediction_sha256": meta["prediction_sha256"]})
    return meta


def _model_meta_path(prediction_path: Path) -> Path:
    return prediction_path.with_suffix(prediction_path.suffix + ".meta.json")


def _fingerprint(config: dict, protocol: dict, code_hashes: dict, versions: dict) -> str:
    return pipeline_fingerprint(config, protocol["data_sha256"], code_hashes, versions)


def _cached_prediction(path: Path, fingerprint: str):
    meta_path = _model_meta_path(path)
    if not path.exists() or not meta_path.exists():
        return None
    meta = _json(meta_path)
    if meta.get("fingerprint") != fingerprint or meta.get("prediction_sha256") != _sha256(path):
        return None
    return meta


def _write_prediction_artifact(frame: pd.DataFrame, path: Path, fingerprint: str, metadata: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False)
    os.replace(temp, path)
    digest = _sha256(path)
    _atomic_json(_model_meta_path(path), {"fingerprint": fingerprint, "prediction_sha256": digest, **metadata})
    return digest


def _fit_budget_ok(state: dict, protocol: dict, stage: str, timeout: int) -> tuple[bool, str | None]:
    budget = protocol["budget"]
    if state["fit_attempts"] >= budget["fit_cap"]:
        return False, "total_fit_cap"
    elapsed = _elapsed(state)
    if elapsed >= budget["wallclock_minutes"] * 60:
        return False, "wallclock_cap"
    stage_cap = budget["stage_fit_caps"].get(stage)
    if stage_cap is not None and state["stage_fits"].get(stage, 0) >= stage_cap:
        return False, f"{stage}_fit_cap"
    if not budget_allows_fit(elapsed, stage, timeout):
        return False, "acceptance_reserve"
    return True, None


def _mark_attempt(state: dict, stage: str) -> int:
    state["fit_attempts"] += 1
    state["stage_fits"][stage] = state["stage_fits"].get(stage, 0) + 1
    return int(state["fit_attempts"])


def _component_path(key: str) -> Path:
    return OUTPUT_ROOT / "predictions" / "components" / f"{key}.csv"


def _run_worker(state: dict, protocol: dict, code_hashes: dict, versions: dict, stage: str,
                key: str, track: str, month: str, method_id: str, family: str,
                config: dict, target_override: str | None = None) -> Path | None:
    output_path = _component_path(key)
    job_config = {"key": key, "track": track, "month": month, "method_id": method_id,
                  "model_family": family, "model_config": config, "target_override": target_override,
                  "threads": protocol["budget"]["threads"], "fold_policy": "date_sold < month start"}
    fingerprint = _fingerprint(job_config, protocol, code_hashes, versions)
    cache = _cached_prediction(output_path, fingerprint)
    if not cache:
        cache = _semantic_cached_prediction(output_path, fingerprint, key, method_id, job_config,
                                            code_hashes, versions, protocol["data_sha256"])
    if cache:
        _ledger({"stage": stage, "key": key, "track": track, "month": month, "method_id": method_id,
                 "status": "cache_hit", "counted_fit": False, "prediction_sha256": cache["prediction_sha256"],
                 "fingerprint": fingerprint})
        print(f"CACHE {key}", flush=True)
        return output_path
    final_stage = stage in {"A4", "A5", "benchmark", "package"}
    timeout = (protocol["budget"]["final_fit_timeout_seconds"] if final_stage
               else protocol["budget"]["development_fit_timeout_seconds"])
    allowed, reason = _fit_budget_ok(state, protocol, stage, timeout)
    if not allowed:
        _ledger({"stage": stage, "key": key, "track": track, "month": month, "method_id": method_id,
                 "status": "budget_skipped", "counted_fit": False, "reason": reason})
        print(f"SKIP {key}: {reason}", flush=True)
        return None
    attempt = _mark_attempt(state, stage)
    state["active_stage"] = stage
    _flush_state(state)
    job_path = OUTPUT_ROOT / "jobs" / f"{key}.json"
    job = {"data_path": protocol["source_data"], "track": track, "month": month,
           "method_id": method_id, "model_family": family, "model_config": config,
           "target_override": target_override, "prediction_path": str(output_path.resolve())}
    _atomic_json(job_path, job)
    base_event = {"attempt": attempt, "stage": stage, "key": key, "track": track, "month": month,
                  "method_id": method_id, "parent_id": "C0", "family": family,
                  "model_config": config, "target_override": target_override,
                  "data_sha256": protocol["data_sha256"], "code_sha256": code_hashes,
                  "runtime_versions": versions, "threads": protocol["budget"]["threads"],
                  "fingerprint": fingerprint, "counted_fit": True, "status": "running"}
    _ledger(base_event)
    print(f"FIT {attempt:03d} {key} ({family})", flush=True)
    started = time.perf_counter()
    try:
        proc = subprocess.run([sys.executable, "-m", "experiments.model_optimization_v2.worker", "--job", str(job_path)],
                              cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=timeout, check=False)
        elapsed = time.perf_counter() - started
        log_path = OUTPUT_ROOT / "logs" / f"{key}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("STDOUT\n" + proc.stdout + "\nSTDERR\n" + proc.stderr, encoding="utf-8")
        if proc.returncode != 0 or not output_path.exists():
            _ledger({**base_event, "status": "invalid", "fit_seconds": elapsed,
                     "return_code": proc.returncode, "log_path": str(log_path),
                     "error_tail": (proc.stderr or proc.stdout)[-4000:]})
            print(f"FAIL {key}; see {log_path}", flush=True)
            return None
        try:
            worker_result = json.loads(proc.stdout.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError):
            worker_result = {"worker_stdout": proc.stdout[-4000:]}
        digest = _sha256(output_path)
        _atomic_json(_model_meta_path(output_path), {"fingerprint": fingerprint, "prediction_sha256": digest,
                     "worker_result": worker_result, "job_config": job_config,
                     "fit_attempt": attempt, "fit_seconds": elapsed})
        _ledger({**base_event, "status": "completed", "fit_seconds": elapsed,
                 "actual_objective": worker_result.get("actual_objective"),
                 "target": worker_result.get("target"), "train_rows": worker_result.get("train_rows"),
                 "train_ids_sha256": worker_result.get("train_ids_sha256"),
                 "evaluation_rows": worker_result.get("evaluation_rows"),
                 "evaluation_ids_sha256": worker_result.get("evaluation_ids_sha256"),
                 "prediction_sha256": digest, "prediction_path": str(output_path)})
        print(f"DONE {key} {elapsed:.1f}s", flush=True)
        return output_path
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - started
        _ledger({**base_event, "status": "timed_out", "fit_seconds": elapsed,
                 "error": f"worker exceeded {timeout}s", "partial_output": str(exc.stdout or "")[-1000:]})
        print(f"TIMEOUT {key} after {elapsed:.1f}s", flush=True)
        return None
    finally:
        state["active_stage"] = None
        _flush_state(state)


def _c0_config(protocol: dict, track: str) -> dict:
    return _json(PROJECT_ROOT / "model_candidate_v2" / "winner_config.json")["locked"][track]["model_config"]


def _frozen_hgbr_config() -> dict:
    import pickle
    bundle = _json(PROJECT_ROOT / "model_candidate_v2" / "models" / "frozen_list_aware.json")
    with (PROJECT_ROOT / "model_candidate_v2" / bundle["model_file"]).open("rb") as stream:
        estimator = pickle.load(stream)
    return estimator.get_params(deep=False)


def _save_final_c0(raw_lgbm: Path, raw_hgbr: Path | None, track: str, month: str,
                   code_hashes: dict, protocol: dict, versions: dict) -> Path:
    model = pd.read_csv(raw_lgbm)
    if track == "list_aware":
        if raw_hgbr is None:
            raise ValueError("list-aware C0 requires its frozen HGBR component")
        frozen = pd.read_csv(raw_hgbr)
        if not np.array_equal(model.ml_num.astype(str).to_numpy(), frozen.ml_num.astype(str).to_numpy()):
            raise ValueError("list-aware baseline component IDs do not align")
        prediction = blend_prices(model.predicted_price.to_numpy(float), frozen.predicted_price.to_numpy(float), 0.75)
    else:
        prediction = np.round(model.predicted_price.to_numpy(float))
    result = model[["ml_num", "track", "month", "sold_price", "house_type_name", "date_sold",
                    "date_listed", "latitude", "longitude"]].copy()
    result["predicted_price"] = prediction
    config = {"parent": "C0", "track": track, "month": month,
              "lightgbm_component_sha256": _sha256(raw_lgbm),
              "frozen_hgbr_component_sha256": _sha256(raw_hgbr) if raw_hgbr else None,
              "list_blend_weight": 0.75 if track == "list_aware" else 1.0,
              "postprocessing": "round HGBR first, price-space blend, round final" if track == "list_aware" else "round final"}
    fingerprint = _fingerprint(config, protocol, code_hashes, versions)
    path = OUTPUT_ROOT / "predictions" / "C0" / f"{track}__{month}.csv"
    digest = _write_prediction_artifact(result, path, fingerprint, {"pipeline_config": config})
    return path


def _run_c0(state: dict, protocol: dict, code_hashes: dict, versions: dict,
            months: tuple[str, ...] = BASELINE_MONTHS) -> dict[tuple[str, str], Path]:
    result = {}
    hgbr_config = _frozen_hgbr_config()
    for month in months:
        for track in TRACKS:
            key = f"C0_LGBM__{track}__{month}"
            component = _run_worker(state, protocol, code_hashes, versions, "A0", key,
                                    track, month, "C0", "lightgbm", _c0_config(protocol, track))
            if component is None:
                continue
            hgbr_path = None
            if track == "list_aware":
                hkey = f"C0_HGBR__list_aware__{month}"
                hgbr_path = _run_worker(state, protocol, code_hashes, versions, "A0", hkey,
                                        track, month, "C0", "hgbr", hgbr_config,
                                        target_override="log_sold_price")
                if hgbr_path is None:
                    continue
            result[(track, month)] = _save_final_c0(component, hgbr_path, track, month,
                                                   code_hashes, protocol, versions)
    return result


def _method_config(method_id: str, track: str, c0_config: dict) -> tuple[str, dict, str | None]:
    methods = set(method_id.split("+"))
    config = dict(c0_config)
    target_override = None
    family = "lightgbm"
    if "M11" in methods:
        config.update(objective="huber", alpha=0.9)
    if "M12" in methods:
        if track != "off_market":
            raise ValueError("M12 is only eligible for off-market")
    if "M13" in methods:
        config.update(objective="mape")
    if method_id == "M16":
        family, config = "sklearn_ridge", {}
    elif method_id == "M17":
        family, config = "extra_trees", {}
    elif method_id == "M18":
        family = "xgboost"
        config = {"n_estimators": 600, "max_depth": 6, "learning_rate": 0.05,
                  "min_child_weight": 5, "reg_lambda": 3, "objective": "reg:squarederror",
                  "enable_categorical": True, "tree_method": "hist"}
    elif method_id == "M19":
        if track != "list_aware":
            raise ValueError("M19 is only eligible for list-aware")
        family = "catboost"
        config = {"iterations": 600, "depth": 6, "learning_rate": 0.05,
                  "l2_leaf_reg": 3, "loss_function": "RMSE", "has_time": True,
                  "random_seed": 0, "thread_count": 4, "verbose": False,
                  "allow_writing_files": False}
    return family, config, target_override


def _candidate_path(method_id: str, track: str, month: str) -> Path:
    return OUTPUT_ROOT / "predictions" / "methods" / method_id / f"{track}__{month}.csv"


def _build_model_candidate(state: dict, protocol: dict, code_hashes: dict, versions: dict,
                           method_id: str, track: str, month: str, stage: str = "A1") -> Path | None:
    family, config, target_override = _method_config(method_id, track, _c0_config(protocol, track))
    key = f"{method_id}__{track}__{month}"
    component = _run_worker(state, protocol, code_hashes, versions, stage, key, track, month,
                            method_id, family, config, target_override)
    if component is None:
        return None
    raw = pd.read_csv(component)
    if track == "list_aware":
        frozen_path = _component_path(f"C0_HGBR__list_aware__{month}")
        if not frozen_path.exists():
            raise FileNotFoundError(f"missing same-fold frozen HGBR component for {month}")
        frozen = pd.read_csv(frozen_path)
        prediction = blend_prices(raw.predicted_price.to_numpy(float), frozen.predicted_price.to_numpy(float), 0.75)
    else:
        prediction = np.round(raw.predicted_price.to_numpy(float))
    result = raw[["ml_num", "track", "month", "sold_price", "house_type_name", "date_sold",
                  "date_listed", "latitude", "longitude"]].copy()
    result["predicted_price"] = prediction
    pipeline = {"method_id": method_id, "parent": "C0", "track": track, "month": month,
                "model_family": family, "model_config": config,
                "target_override": target_override, "candidate_component_sha256": _sha256(component),
                "frozen_component_sha256": _sha256(_component_path(f"C0_HGBR__list_aware__{month}")) if track == "list_aware" else None,
                "list_blend_weight": 0.75 if track == "list_aware" else 1.0,
                "round_rule": "frozen first; blend in price space; final round" if track == "list_aware" else "round final"}
    fingerprint = _fingerprint(pipeline, protocol, code_hashes, versions)
    path = _candidate_path(method_id, track, month)
    _write_prediction_artifact(result, path, fingerprint, {"pipeline_config": pipeline})
    return path


def _oof_frame(track: str) -> pd.DataFrame:
    files = [OUTPUT_ROOT / "predictions" / "C0" / f"{track}__{month}.csv" for month in ("2025-02", "2025-04")]
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError("OOF calibration seeds missing: " + ", ".join(missing))
    return pd.concat([pd.read_csv(path) for path in files], ignore_index=True)


def _run_calibration(state: dict, protocol: dict, code_hashes: dict, versions: dict,
                     method_id: str, track: str, month: str, stage: str = "A1") -> Path | None:
    parent_path = OUTPUT_ROOT / "predictions" / "C0" / f"{track}__{month}.csv"
    if not parent_path.exists():
        return None
    parent = pd.read_csv(parent_path)
    homes = pd.read_parquet(PROJECT_ROOT / protocol["source_data"])
    month_mask = pd.to_datetime(homes.date_sold).dt.to_period("M").astype(str).eq(month)
    homes = homes.loc[month_mask].copy()
    homes = homes.set_index(homes.ml_num.astype(str)).loc[parent.ml_num.astype(str)].reset_index(drop=True)
    oof = _oof_frame(track)
    cutoff = pd.Period(month, freq="M").start_time
    config = {"method_id": method_id, "track": track, "month": month,
              "oof_prediction_sha256": [_sha256(OUTPUT_ROOT / "predictions" / "C0" / f"{track}__{m}.csv")
                                        for m in ("2025-02", "2025-04")],
              "cutoff": cutoff.isoformat(), "method_registry_sha256": _sha256(REGISTRY_PATH)}
    fingerprint = _fingerprint(config, protocol, code_hashes, versions)
    output_path = _candidate_path(method_id, track, month)
    cache = _cached_prediction(output_path, fingerprint)
    if not cache:
        cache = _semantic_cached_prediction(output_path, fingerprint,
                                            f"{method_id}__{track}__{month}", method_id, config,
                                            code_hashes, versions, protocol["data_sha256"])
    if cache:
        _ledger({"stage": "A1", "key": f"{method_id}__{track}__{month}", "status": "cache_hit",
                 "counted_fit": False, "fingerprint": fingerprint, "prediction_sha256": cache["prediction_sha256"]})
        return output_path
    allowed, reason = _fit_budget_ok(state, protocol, stage, 60)
    if not allowed:
        _ledger({"stage": "A1", "key": f"{method_id}__{track}__{month}", "status": "budget_skipped",
                 "counted_fit": False, "reason": reason})
        return None
    attempt = _mark_attempt(state, stage)
    state["active_stage"] = stage
    _flush_state(state)
    event = {"attempt": attempt, "stage": stage, "key": f"{method_id}__{track}__{month}",
             "method_id": method_id, "track": track, "month": month, "family": "calibrator",
             "config": config, "data_sha256": protocol["data_sha256"], "code_sha256": code_hashes,
             "runtime_versions": versions, "threads": 1, "fingerprint": fingerprint,
             "counted_fit": True, "status": "running"}
    _ledger(event)
    started = time.perf_counter()
    try:
        if method_id == "M10":
            predictions, params = apply_local_residual(parent.predicted_price.to_numpy(float), homes, oof, cutoff)
        elif method_id == "M14":
            params = fit_multiplier(oof, cutoff)
            predictions = apply_multiplier_calibration(parent.predicted_price.to_numpy(float), params)
        elif method_id == "M15":
            params = fit_type_multiplier(oof, cutoff)
            predictions = apply_type_multiplier(parent.predicted_price.to_numpy(float), homes, params)
        else:
            raise ValueError(f"unknown calibration method {method_id}")
        predictions = np.round(predictions)
        output = parent[["ml_num", "track", "month", "sold_price", "house_type_name", "date_sold",
                         "date_listed", "latitude", "longitude"]].copy()
        output["predicted_price"] = predictions
        digest = _write_prediction_artifact(output, output_path, fingerprint,
                                           {"pipeline_config": config, "calibration_params": params})
        _ledger({**event, "status": "completed", "fit_seconds": time.perf_counter() - started,
                 "calibration_params": params, "max_oof_date": params.get("max_oof_date"),
                 "prediction_sha256": digest, "prediction_path": str(output_path)})
        print(f"DONE {method_id}__{track}__{month} calibration", flush=True)
        return output_path
    except Exception as exc:
        _ledger({**event, "status": "invalid", "fit_seconds": time.perf_counter() - started,
                 "error": f"{type(exc).__name__}: {exc}"})
        print(f"FAIL {method_id}__{track}__{month}: {exc}", flush=True)
        return None
    finally:
        state["active_stage"] = None
        _flush_state(state)


def _score_screen_method(method_id: str, track: str, months: tuple[str, ...], stage: str = "screen_only",
                         repeats: int = 2000, seed: int = 101, parent_method_id: str = "C0") -> dict:
    parent_frames, candidate_frames = [], []
    for month in months:
        parent_path = (OUTPUT_ROOT / "predictions" / "C0" / f"{track}__{month}.csv"
                       if parent_method_id == "C0" else _candidate_path(parent_method_id, track, month))
        if not parent_path.exists():
            return {"status": "incomplete", "parent_id": parent_method_id}
        parent = pd.read_csv(parent_path)
        candidate_path = _candidate_path(method_id, track, month)
        if not candidate_path.exists():
            continue
        candidate = pd.read_csv(candidate_path)
        parent_frames.append(parent[["ml_num", "month", "sold_price", "predicted_price"]])
        candidate_frames.append(candidate[["ml_num", "month", "sold_price", "predicted_price"]])
    if not parent_frames or len(parent_frames) != len(months):
        return {"status": "incomplete", "months_available": len(candidate_frames)}
    parent = pd.concat(parent_frames, ignore_index=True)
    candidate = pd.concat(candidate_frames, ignore_index=True)
    report = evaluate_increment(parent, candidate, repeats=repeats, seed=seed)
    report["method_id"] = method_id
    report["track"] = track
    report["stage"] = stage
    report["parent_id"] = parent_method_id
    return report


def _run_blend_method(state: dict, protocol: dict, registry_state: dict,
                     code_hashes: dict, versions: dict) -> list[dict]:
    reports = []
    for track in TRACKS:
        allowed = ["M16", "M17", "M18", "M19"]
        options = []
        for method_id in allowed:
            files = [_candidate_path(method_id, track, month) for month in SCREEN_MONTHS]
            if not all(path.exists() for path in files):
                continue
            frames = [pd.read_csv(path) for path in files]
            score = np.median(np.concatenate([
                100.0 * np.abs(frame.predicted_price.to_numpy(float) - frame.sold_price.to_numpy(float)) /
                frame.sold_price.to_numpy(float) for frame in frames]))
            options.append((float(score), method_id, frames))
        if not options:
            registry_state["M20"]["track_status"][track] = "budget_skipped"
            registry_state["M20"]["notes"].append(f"no completed different-family challenger available for {track}")
            continue
        _, challenger_id, _ = min(options, key=lambda item: item[0])
        challenger_by_month = {}
        for month in SCREEN_MONTHS:
            parent = pd.read_csv(OUTPUT_ROOT / "predictions" / "C0" / f"{track}__{month}.csv")
            challenger = pd.read_csv(_candidate_path(challenger_id, track, month))
            if not np.array_equal(parent.ml_num.astype(str).to_numpy(), challenger.ml_num.astype(str).to_numpy()):
                raise ValueError("M20 challenger homes do not align with C0")
            prediction = blend_prices(challenger.predicted_price.to_numpy(float),
                                      parent.predicted_price.to_numpy(float), 0.5)
            result = parent.copy()
            result["predicted_price"] = prediction
            parent_error = (parent.predicted_price.to_numpy(float) - parent.sold_price.to_numpy(float)) / parent.sold_price.to_numpy(float)
            challenger_error = (challenger.predicted_price.to_numpy(float) - challenger.sold_price.to_numpy(float)) / challenger.sold_price.to_numpy(float)
            correlation = float(np.corrcoef(parent_error, challenger_error)[0, 1]) if len(parent) > 1 else float("nan")
            config = {"method_id": "M20", "parent": "C0", "track": track, "month": month,
                      "challenger_method_id": challenger_id, "challenger_family":
                      _method_config(challenger_id, track, _c0_config(protocol, track))[0],
                      "fixed_challenger_weight": 0.5, "residual_correlation": correlation,
                      "round_rule": "round C0 then fixed price-space average and final round"}
            fingerprint = _fingerprint(config, protocol, code_hashes, versions)
            path = _candidate_path("M20", track, month)
            digest = _write_prediction_artifact(result, path, fingerprint, {"pipeline_config": config})
            _ledger({"stage": "A1", "key": f"M20__{track}__{month}", "method_id": "M20",
                     "track": track, "month": month, "status": "completed", "counted_fit": False,
                     "challenger_method_id": challenger_id, "residual_correlation": correlation,
                     "prediction_sha256": digest, "prediction_path": str(path)})
            challenger_by_month[month] = (correlation, digest)
        if len(challenger_by_month) == len(SCREEN_MONTHS):
            report = _score_screen_method("M20", track, SCREEN_MONTHS)
            report["challenger_method_id"] = challenger_id
            report["residual_correlations"] = {month: challenger_by_month[month][0] for month in SCREEN_MONTHS}
            reports.append(report)
            registry_state["M20"]["track_status"][track] = (
                "no_gain" if report["relative_mdape_improvement"] <= 0 and report["months_improved"] == 0
                else "completed")
            registry_state["M20"].setdefault("screen_results", {})[track] = report
            registry_state["M20"]["notes"].append(f"{track}: fixed 0.5 blend selected {challenger_id}")
        _atomic_json(_paths()["registry"], registry_state)
    track_status = registry_state["M20"]["track_status"].values()
    if track_status and all(value == "no_gain" for value in track_status):
        registry_state["M20"]["status"] = "no_gain"
    elif track_status and all(value == "budget_skipped" for value in track_status):
        registry_state["M20"]["status"] = "budget_skipped"
    else:
        registry_state["M20"]["status"] = "completed"
    return reports


def _build_blend_month(protocol: dict, code_hashes: dict, versions: dict, *, track: str, month: str,
                       parent_path: Path, challenger_path: Path, challenger_id: str,
                       output_method_id: str, parent_id: str, challenger_weight: float = 0.5) -> Path:
    parent = pd.read_csv(parent_path)
    challenger = pd.read_csv(challenger_path)
    if not np.array_equal(parent.ml_num.astype(str).to_numpy(), challenger.ml_num.astype(str).to_numpy()):
        raise ValueError("blend components do not contain identical homes in identical order")
    if not np.array_equal(parent.sold_price.to_numpy(float), challenger.sold_price.to_numpy(float)):
        raise ValueError("blend components do not have identical held-out targets")
    prediction = blend_prices(challenger.predicted_price.to_numpy(float),
                              parent.predicted_price.to_numpy(float), challenger_weight)
    result = parent.copy()
    result["predicted_price"] = prediction
    parent_error = (parent.predicted_price.to_numpy(float) - parent.sold_price.to_numpy(float)) / parent.sold_price.to_numpy(float)
    challenger_error = (challenger.predicted_price.to_numpy(float) - challenger.sold_price.to_numpy(float)) / challenger.sold_price.to_numpy(float)
    correlation = float(np.corrcoef(parent_error, challenger_error)[0, 1]) if len(parent) > 1 else float("nan")
    config = {"method_id": output_method_id, "parent_id": parent_id, "track": track, "month": month,
              "challenger_method_id": challenger_id,
              "challenger_family": _method_config(challenger_id, track, _c0_config(protocol, track))[0],
              "fixed_challenger_weight": challenger_weight, "residual_correlation": correlation,
              "round_rule": "round parent first; fixed price-space average; round final"}
    fingerprint = _fingerprint(config, protocol, code_hashes, versions)
    path = _candidate_path(output_method_id, track, month)
    digest = _write_prediction_artifact(result, path, fingerprint, {"pipeline_config": config})
    _ledger({"stage": "A2" if output_method_id == "M20" else "A3",
             "key": f"{output_method_id}__{track}__{month}", "method_id": output_method_id,
             "track": track, "month": month, "status": "completed", "counted_fit": False,
             "parent_id": parent_id, "challenger_method_id": challenger_id,
             "residual_correlation": correlation, "prediction_sha256": digest,
             "prediction_path": str(path)})
    return path


def _run_confirmation(state: dict, protocol: dict, code_hashes: dict, versions: dict) -> dict:
    plan = {"stage": "A2", "months_to_add": ["2025-07", "2025-10"],
            "candidates": {"off_market": ["M04", "M05"],
                           "list_aware": ["M04", "M13", "M14", "M19"],
                           "derived_blends": {"list_aware": ["M20"]}},
            "selection_basis": "best two-month screen candidates by track, including all list-aware candidates with positive screen delta or materially different learner family; no off-market candidate screened positive",
            "fit_cap": protocol["budget"]["stage_fit_caps"]["A2"],
            "created_at_utc": _now().isoformat()}
    _atomic_json(OUTPUT_ROOT / "confirmation_plan.json", plan)
    completed = []
    for track, methods in plan["candidates"].items():
        if track == "derived_blends":
            continue
        for method_id in methods:
            for month in plan["months_to_add"]:
                if method_id == "M14":
                    path = _run_calibration(state, protocol, code_hashes, versions,
                                            method_id, track, month, stage="A2")
                else:
                    path = _build_model_candidate(state, protocol, code_hashes, versions,
                                                  method_id, track, month, stage="A2")
                if path is not None:
                    completed.append({"track": track, "method_id": method_id, "month": month,
                                      "prediction_path": str(path)})
                if state["fit_attempts"] >= protocol["budget"]["fit_cap"]:
                    break
    # M20 is derived from the already fixed list-aware CatBoost challenger; it adds no learner fit.
    m19_screens = [_candidate_path("M19", "list_aware", month) for month in SCREEN_MONTHS]
    m19_confirms = [_candidate_path("M19", "list_aware", month) for month in plan["months_to_add"]]
    if all(path.exists() for path in m19_screens + m19_confirms):
        for month in (*SCREEN_MONTHS, *plan["months_to_add"]):
            _build_blend_month(protocol, code_hashes, versions, track="list_aware", month=month,
                parent_path=OUTPUT_ROOT / "predictions" / "C0" / f"list_aware__{month}.csv",
                challenger_path=_candidate_path("M19", "list_aware", month), challenger_id="M19",
                output_method_id="M20", parent_id="C0", challenger_weight=0.5)

    reports = []
    development_months = tuple(protocol["development_months"])
    candidates_by_track = {"off_market": ["M04", "M05"],
                           "list_aware": ["M04", "M13", "M14", "M19", "M20"]}
    for track, methods in candidates_by_track.items():
        for method_id in methods:
            paths = [_candidate_path(method_id, track, month) for month in development_months]
            if not all(path.exists() for path in paths):
                continue
            report = _score_screen_method(method_id, track, development_months,
                                          stage="four_month_confirmation", repeats=10000,
                                          seed=20261019)
            reports.append(report)
            print(f"CONFIRM {method_id} {track}: relative MdAPE "
                  f"{100*report['relative_mdape_improvement']:.2f}%, pass={report['passes_metrics']}", flush=True)
    selected = {}
    for track in TRACKS:
        passing = [report for report in reports if report["track"] == track and report["passes_metrics"]]
        selected[track] = (max(passing, key=lambda report: report["relative_mdape_improvement"]) if passing else None)
    checkpoint = {"checkpoint": "C1", "parent_id": "C0", "generated_at_utc": _now().isoformat(),
        "selection_gate": protocol["development_promotion"],
        "selected_by_track": {track: ({"method_id": report["method_id"],
              "relative_mdape_improvement": report["relative_mdape_improvement"],
              "metrics": report} if report else None) for track, report in selected.items()},
        "dual_track_upgrade": all(selected.values()),
        "reports": reports,
        "interpretation": "A C1 promotion requires both tracks to independently pass the prespecified development gate; single-track passes are recorded but do not count as a dual-track upgrade."}
    _atomic_json(OUTPUT_ROOT / "checkpoint_C1.json", checkpoint)
    summary = {"fit_attempts": state["fit_attempts"], "stage_fits": state["stage_fits"],
               "elapsed_minutes": round(_elapsed(state) / 60.0, 2), "confirmation_plan": plan,
               "completed_predictions": completed, "checkpoint_C1": checkpoint}
    _atomic_json(OUTPUT_ROOT / "confirmation_results.json", summary)
    state["active_stage"] = None
    state["status"] = "confirmation_completed"
    _flush_state(state)
    return summary


def _run_compositions(state: dict, protocol: dict, code_hashes: dict, versions: dict) -> dict:
    dev_months = tuple(protocol["development_months"])
    bundles = ["M13+M04", "M13+M05", "M13+M03"]
    plan = {"stage": "A3", "parent_candidates": {"off_market": "C0", "list_aware": "M13"},
            "feature_bundles": bundles, "months": list(dev_months),
            "fit_cap": protocol["budget"]["stage_fit_caps"]["A3"],
            "rationale": "test fixed listing-time feature bundles; compare list-aware children to their passing direct-M13 C1 parent and off-market bundles to C0 because M13 off-market did not pass C1",
            "created_at_utc": _now().isoformat()}
    _atomic_json(OUTPUT_ROOT / "composition_plan.json", plan)
    reports = []
    for track in TRACKS:
        for method_id in bundles:
            for month in dev_months:
                _build_model_candidate(state, protocol, code_hashes, versions,
                                       method_id, track, month, stage="A3")
                if state["fit_attempts"] >= protocol["budget"]["fit_cap"]:
                    break
            paths = [_candidate_path(method_id, track, month) for month in dev_months]
            if not all(path.exists() for path in paths):
                continue
            parent_id = "M13" if track == "list_aware" else "C0"
            report = _score_screen_method(method_id, track, dev_months,
                                          stage="four_month_composition", repeats=10000,
                                          seed=20261020, parent_method_id=parent_id)
            report["comparison_role"] = ("incremental_bundle_vs_direct_M13_parent" if track == "list_aware"
                                          else "candidate_vs_C0_for_first_track_promotion")
            reports.append(report)
            print(f"COMPOSE {method_id} {track}: vs {parent_id} relative MdAPE "
                  f"{100*report['relative_mdape_improvement']:.2f}%, pass={report['passes_metrics']}", flush=True)

    # Fixed 50/50 fusion of the passing list-aware C1 parent and the already fitted, different-family M19 model.
    blend_method_id = "M13_M19_blend"
    for month in dev_months:
        m13 = _candidate_path("M13", "list_aware", month)
        m19 = _candidate_path("M19", "list_aware", month)
        if not m13.exists() or not m19.exists():
            break
        _build_blend_month(protocol, code_hashes, versions, track="list_aware", month=month,
            parent_path=m13, challenger_path=m19, challenger_id="M19",
            output_method_id=blend_method_id, parent_id="M13", challenger_weight=0.5)
    blend_paths = [_candidate_path(blend_method_id, "list_aware", month) for month in dev_months]
    if all(path.exists() for path in blend_paths):
        blend_report = _score_screen_method(blend_method_id, "list_aware", dev_months,
                                           stage="four_month_C2_blend", repeats=10000,
                                           seed=20261022, parent_method_id="M13")
        blend_report["comparison_role"] = "fixed_half_blend_vs_C1_M13"
        reports.append(blend_report)
        print(f"COMPOSE {blend_method_id} list_aware: relative MdAPE "
              f"{100*blend_report['relative_mdape_improvement']:.2f}%, pass={blend_report['passes_metrics']}", flush=True)

    c1 = _json(OUTPUT_ROOT / "checkpoint_C1.json")
    selected_c1 = {track: c1["selected_by_track"].get(track) for track in TRACKS}
    c2_selected = {}
    for track in TRACKS:
        if track == "list_aware":
            pool = [report for report in reports if report.get("track") == track and
                    report.get("comparison_role") == "incremental_bundle_vs_direct_M13_parent" and
                    report.get("passes_metrics")]
            pool.extend(report for report in reports if report.get("track") == track and
                        report.get("comparison_role") == "fixed_half_blend_vs_C1_M13" and
                        report.get("passes_metrics"))
            c2_selected[track] = max(pool, key=lambda report: report["relative_mdape_improvement"]) if pool else None
        else:
            pool = [report for report in reports if report.get("track") == track and
                    report.get("comparison_role") == "candidate_vs_C0_for_first_track_promotion" and
                    report.get("passes_metrics")]
            c2_selected[track] = max(pool, key=lambda report: report["relative_mdape_improvement"]) if pool else None
    checkpoint_c2 = {"checkpoint": "C2", "generated_at_utc": _now().isoformat(),
        "candidate_parent": {"off_market": "C0", "list_aware": "M13"},
        "selected_by_track": {track: ({"method_id": report["method_id"], "metrics": report}
                                      if report else None) for track, report in c2_selected.items()},
        "dual_track_cumulative_upgrade": all(c1["selected_by_track"].get(track) and c2_selected.get(track)
                                             for track in TRACKS),
        "reports": reports,
        "notes": ["C2 list-aware promotion is measured against the passing C1 M13 parent",
                  "off-market candidates are measured against C0 because C1 did not pass for that track"]}
    _atomic_json(OUTPUT_ROOT / "checkpoint_C2.json", checkpoint_c2)

    c3_selected = {track: None for track in TRACKS}
    parent = c2_selected["list_aware"]
    if parent and parent["method_id"] in bundles:
        parent_id = parent["method_id"]
        for month in dev_months:
            _build_blend_month(protocol, code_hashes, versions, track="list_aware", month=month,
                parent_path=_candidate_path(parent_id, "list_aware", month),
                challenger_path=_candidate_path("M19", "list_aware", month), challenger_id="M19",
                output_method_id="C3", parent_id=parent_id, challenger_weight=0.5)
        c3_report = _score_screen_method("C3", "list_aware", dev_months,
                                         stage="four_month_C3_blend", repeats=10000,
                                         seed=20261023, parent_method_id=parent_id)
        c3_report["comparison_role"] = "fixed_half_blend_vs_C2_bundle"
        reports.append(c3_report)
        if c3_report["passes_metrics"]:
            c3_selected["list_aware"] = c3_report
        print(f"COMPOSE C3 list_aware: relative MdAPE "
              f"{100*c3_report['relative_mdape_improvement']:.2f}%, pass={c3_report['passes_metrics']}", flush=True)
    checkpoint_c3 = {"checkpoint": "C3", "generated_at_utc": _now().isoformat(),
        "parent_by_track": {"off_market": "C0", "list_aware":
                            (c2_selected["list_aware"]["method_id"] if c2_selected["list_aware"] else "M13")},
        "selected_by_track": {track: ({"method_id": "C3", "metrics": report} if report else None)
                              for track, report in c3_selected.items()},
        "dual_track_cumulative_upgrade": all(c1["selected_by_track"].get(track) and c2_selected.get(track)
                                             and c3_selected.get(track) for track in TRACKS),
        "notes": ["C3 is attempted only after a C2 feature bundle passes its direct-parent gate",
                  "the C3 blend uses fixed 0.5 weight and the already fitted CatBoost challenger"]}
    _atomic_json(OUTPUT_ROOT / "checkpoint_C3.json", checkpoint_c3)
    checkpoints = {"C1": c1, "C2": checkpoint_c2, "C3": checkpoint_c3}
    final_by_track = {}
    for track in TRACKS:
        if c3_selected[track]:
            final_by_track[track] = "C3"
        elif c2_selected[track]:
            final_by_track[track] = c2_selected[track]["method_id"]
        elif selected_c1.get(track):
            final_by_track[track] = selected_c1[track]["method_id"]
        else:
            final_by_track[track] = "C0"
    pipeline_specs = {}
    for track in TRACKS:
        candidate_id = final_by_track[track]
        if candidate_id == "C0":
            spec = {"candidate_id": "C0", "model_family": "lightgbm",
                    "model_config": _c0_config(protocol, track),
                    "target": "log_sold_price" if track == "off_market" else "log_sold_over_list",
                    "feature_cards": [], "blend_weight": 1.0 if track == "off_market" else 0.75,
                    "postprocessing": "round final" if track == "off_market" else
                                     "round frozen HGBR first; blend in price space; round final"}
            if track == "list_aware":
                spec["frozen_hgbr_config"] = _frozen_hgbr_config()
        elif candidate_id == "M13":
            model_family, model_config, _ = _method_config("M13", track, _c0_config(protocol, track))
            spec = {"candidate_id": "M13", "model_family": model_family, "model_config": model_config,
                    "target": "dollar_price", "feature_cards": ["M13"], "blend_weight": 0.75,
                    "frozen_hgbr_config": _frozen_hgbr_config(),
                    "postprocessing": "round frozen HGBR first; blend in price space; round final"}
        else:
            spec = {"candidate_id": candidate_id,
                    "parent_pipeline": candidate_id, "notes": "configuration is recorded by the selected checkpoint and prediction sidecars"}
        pipeline_specs[track] = spec
    freeze = {"freeze_id": "v2_research_final", "frozen_at_utc": _now().isoformat(),
              "data_sha256": protocol["data_sha256"], "code_sha256": code_hashes,
              "runtime_versions": versions, "final_by_track": final_by_track,
              "benchmark_months": protocol["retrospective_benchmark_months"],
              "benchmark_is_retrospective": True,
              "frozen_pipeline_configs": pipeline_specs,
              "pipeline_fingerprints": {track: _fingerprint(pipeline_specs[track], protocol, code_hashes, versions)
                                         for track in TRACKS},
              "notes": ["Final pipeline selection uses only predeclared development folds.",
                        "Do not change selection after examining final benchmark results."]}
    _atomic_json(OUTPUT_ROOT / "freeze_manifest.json", freeze)
    summary = {"fit_attempts": state["fit_attempts"], "stage_fits": state["stage_fits"],
               "elapsed_minutes": round(_elapsed(state) / 60.0, 2), "plan": plan,
               "checkpoint_C1": c1, "checkpoint_C2": checkpoint_c2,
               "checkpoint_C3": checkpoint_c3, "final_by_track": final_by_track,
               "freeze_manifest": freeze}
    _atomic_json(OUTPUT_ROOT / "composition_results.json", summary)
    state["active_stage"] = None
    state["status"] = "composition_completed_and_frozen"
    _flush_state(state)
    return summary


def _passes_final_gate(report: dict, protocol: dict) -> dict:
    gate = protocol["final_promotion"]
    monthly = report["monthly_relative_improvement"]
    worst_regression = max((max(0.0, -value) for value in monthly.values()), default=0.0)
    result = {"relative_mdape_improvement": report["relative_mdape_improvement"],
              "months_improved": report["months_improved"],
              "worst_month_relative_regression": worst_regression,
              "mape_relative_change": report["mape_relative_change"],
              "paired_home_bootstrap_ci95_percentage_points":
                  report["paired_home_bootstrap"]["ci95_percentage_points"],
              "criteria": {"relative_mdape_at_least_2_percent":
                           report["relative_mdape_improvement"] >= gate["min_relative_mdape_improvement_each_track"],
                           "at_least_four_of_six_months_improve": report["months_improved"] >= gate["min_months_improved"],
                           "worst_month_regression_at_most_10_percent":
                           worst_regression <= gate["max_month_relative_regression"],
                           "mape_regression_at_most_1_percent":
                           report["mape_relative_change"] <= gate["max_mape_relative_regression"],
                           "paired_home_bootstrap_upper_below_zero":
                           report["paired_home_bootstrap"]["ci95_percentage_points"][1] <
                           gate["paired_home_bootstrap_upper_percentage_points_lt"]}}
    result["passed"] = all(result["criteria"].values())
    result["aspirational_3_percent_reached"] = (
        report["relative_mdape_improvement"] >= gate["aspirational_relative_mdape_improvement_each_track"])
    return result


def _protocol_c0_metrics(protocol: dict, track: str) -> dict:
    expected = protocol["c0"]
    return {"mdape_percent": float(expected[f"{track}_mdape_percent"]),
            "mape_percent": float(expected[f"{track}_mape_percent"])}


def _run_final_benchmark(state: dict, protocol: dict, code_hashes: dict, versions: dict) -> dict:
    saved_result = OUTPUT_ROOT / "final_benchmark_results.json"
    if saved_result.exists() and state.get("status") == "final_benchmark_completed":
        return _json(saved_result)
    freeze_path = OUTPUT_ROOT / "freeze_manifest.json"
    if not freeze_path.exists():
        raise RuntimeError("final benchmark requires a locked freeze_manifest.json")
    freeze = _json(freeze_path)
    if freeze.get("data_sha256") != protocol["data_sha256"]:
        raise ValueError("frozen data hash differs from the research protocol")
    freeze["execution_code_sha256"] = code_hashes
    _atomic_json(freeze_path, freeze)
    months = tuple(protocol["retrospective_benchmark_months"])
    references = {}
    for track in TRACKS:
        reference_path = PROJECT_ROOT / "outputs" / "model_optimization_v1" / f"benchmark_predictions_{track}.csv"
        reference = pd.read_csv(reference_path)
        ref_metrics = _score(reference)
        expected = _protocol_c0_metrics(protocol, track)
        if abs(ref_metrics["mdape_percent"] - expected["mdape_percent"]) > 1e-8 or \
                abs(ref_metrics["mape_percent"] - expected["mape_percent"]) > 1e-8:
            raise ValueError(f"saved retrospective C0 reference does not reproduce protocol metrics for {track}")
        ref_months = set(reference.month.astype(str).unique())
        if ref_months != set(months) or reference.duplicated(["month", "ml_num"]).any():
            raise ValueError(f"saved C0 benchmark rows do not match the frozen six-month split for {track}")
        references[track] = reference

    final_by_track = freeze["final_by_track"]
    benchmark_predictions = {}
    for track in TRACKS:
        selected = final_by_track[track]
        if selected == "C0":
            source = PROJECT_ROOT / "outputs" / "model_optimization_v1" / f"benchmark_predictions_{track}.csv"
            benchmark_predictions[track] = pd.read_csv(source)
            continue
        if track != "list_aware" or selected != "M13":
            raise RuntimeError(f"unsupported frozen candidate for final benchmark: {track}={selected}")
        for month in months:
            hkey = f"C0_HGBR__list_aware__{month}"
            _run_worker(state, protocol, code_hashes, versions, "A4", hkey,
                        "list_aware", month, "C0", "hgbr", _frozen_hgbr_config(),
                        target_override="log_sold_price")
            _build_model_candidate(state, protocol, code_hashes, versions,
                                   "M13", "list_aware", month, stage="A4")
        frames = [pd.read_csv(_candidate_path("M13", "list_aware", month)) for month in months]
        benchmark_predictions[track] = pd.concat(frames, ignore_index=True)

    reports = {}
    for track in TRACKS:
        parent = references[track]
        candidate = benchmark_predictions[track]
        report = evaluate_increment(parent[["ml_num", "month", "sold_price", "predicted_price"]],
                                    candidate[["ml_num", "month", "sold_price", "predicted_price"]],
                                    repeats=10000, seed=20261024 + TRACKS.index(track))
        report["track"] = track
        report["frozen_candidate"] = final_by_track[track]
        report["relative_to_v1"] = {"v1_mdape_percent": protocol["historical_v1"][track]["mdape_percent"],
                                    "v1_mape_percent": protocol["historical_v1"][track]["mape_percent"],
                                    "candidate_mdape_percent": report["candidate"]["mdape_percent"],
                                    "candidate_mape_percent": report["candidate"]["mape_percent"]}
        report["final_gate"] = _passes_final_gate(report, protocol)
        reports[track] = report
        path = OUTPUT_ROOT / "predictions" / "final_benchmark" / f"{track}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        benchmark_predictions[track].to_csv(path, index=False)
        report["prediction_path"] = str(path)
        report["prediction_sha256"] = _sha256(path)
        print(f"FINAL {track}: MdAPE {report['candidate']['mdape_percent']:.4f}%, "
              f"relative improvement {100*report['relative_mdape_improvement']:.2f}%, "
              f"pass={report['final_gate']['passed']}", flush=True)
    dual_pass = all(report["final_gate"]["passed"] for report in reports.values())
    decision = {"decision": "create_model_candidate_v3" if dual_pass else "retain_model_candidate_v2",
                "dual_track_final_gate_passed": dual_pass,
                "v3_exported": False, "submission_files_read": False,
                "reason": ("both tracks passed the final gate" if dual_pass else
                           "the final two-track gate was not met; preserve the frozen v2 submission candidate"),
                "selected_development_pipelines": final_by_track}
    result = {"generated_at_utc": _now().isoformat(), "freeze_id": freeze["freeze_id"],
              "benchmark_months": list(months), "benchmark_is_retrospective": True,
              "fit_attempts": state["fit_attempts"], "stage_fits": state["stage_fits"],
              "elapsed_minutes": round(_elapsed(state) / 60.0, 2),
              "reports": reports, "decision": decision,
              "notes": ["These six months were examined in earlier project work and are reported as a retrospective benchmark.",
                        "No candidate was selected or replaced after these results were read."]}
    _atomic_json(OUTPUT_ROOT / "final_benchmark_results.json", result)
    freeze["execution_code_sha256"] = code_hashes
    freeze["benchmark_completed_at_utc"] = result["generated_at_utc"]
    freeze["final_decision"] = decision["decision"]
    _atomic_json(freeze_path, freeze)
    state["active_stage"] = None
    state["status"] = "final_benchmark_completed"
    _flush_state(state)
    return result


def _preflight(protocol: dict, registry: list[dict]) -> dict:
    errors = []
    data_path = PROJECT_ROOT / protocol["source_data"]
    if not data_path.exists():
        errors.append(f"missing source dataset: {data_path}")
        data_hash = None
    else:
        data_hash = _sha256(data_path)
        if data_hash != protocol["data_sha256"]:
            errors.append(f"dataset SHA-256 mismatch: expected {protocol['data_sha256']}, got {data_hash}")
    package_versions = _runtime_versions()
    required_packages = ("numpy", "pandas", "scikit-learn", "lightgbm", "catboost", "pyarrow")
    missing = [package for package in required_packages if not package_versions.get(package)]
    if missing:
        errors.append("missing required packages: " + ", ".join(missing))
    if data_path.exists() and data_hash == protocol["data_sha256"]:
        listings = pd.read_parquet(data_path)
        needed = {"ml_num", "date_sold", "date_listed", "sold_price", "list_price", "house_area",
                  "house_type_name", "community", "municipality", "latitude", "longitude"}
        if needed - set(listings.columns):
            errors.append("source data missing required fields: " + ", ".join(sorted(needed - set(listings.columns))))
        dates = pd.to_datetime(listings.date_sold)
        counts = {month: int(dates.dt.to_period("M").astype(str).eq(month).sum())
                  for month in BASELINE_MONTHS}
        if len(listings) != 150_000:
            errors.append(f"unexpected source row count: {len(listings)}")
    else:
        counts = {}
    old_feature_hash = _sha256(PROJECT_ROOT / "release_candidate" / "features.py")
    new_feature_hash = _sha256(PROJECT_ROOT / "model_candidate_v2" / "features.py")
    feature_match = old_feature_hash == new_feature_hash
    if not feature_match:
        errors.append("v1 and v2 frozen feature modules differ; cached predictions cannot be reused")
    expected_target = {"off_market": "log_sold_price", "list_aware": "log_sold_over_list"}
    configs = {}
    for track in TRACKS:
        configs[track] = _c0_config(protocol, track)
        if configs[track].get("objective") != "regression":
            errors.append(f"C0 objective mismatch before fit for {track}")
    preflight = {"checked_at_utc": _now().isoformat(), "source_data": str(data_path),
                 "data_sha256": data_hash, "expected_data_sha256": protocol["data_sha256"],
                 "rows": 150_000 if data_path.exists() and data_hash == protocol["data_sha256"] else None,
                 "development_and_seed_month_counts": counts, "runtime_versions": package_versions,
                 "xgboost_available": bool(package_versions.get("xgboost")),
                 "v1_v2_frozen_feature_sha256_match": feature_match,
                 "c0_configs": configs, "baseline_targets": expected_target,
                 "registry_cards": len(registry), "hidden_target_files_read": False,
                 "fit_attempts_at_preflight": _load_state(protocol)["fit_attempts"],
                 "errors": errors, "status": "passed" if not errors else "failed"}
    _atomic_json(_paths()["audit"], preflight)
    return preflight


def _run_screen(state: dict, protocol: dict, registry: list[dict], code_hashes: dict, versions: dict) -> dict:
    state["status"] = "running"
    _flush_state(state)
    c0 = _run_c0(state, protocol, code_hashes, versions)
    # Cached complete C0 folds are accepted only when both prediction and metadata hashes validate.
    for month in BASELINE_MONTHS:
        for track in TRACKS:
            path = OUTPUT_ROOT / "predictions" / "C0" / f"{track}__{month}.csv"
            if not path.exists():
                print(f"C0 unavailable: {track} {month}", flush=True)
    registry_state = {row["id"]: {"status": "not_started", "track_status": {}, "notes": []} for row in registry}
    _atomic_json(_paths()["registry"], registry_state)
    results = []
    for row in registry:
        method_id = row["id"]
        previous_events = _ledger_events()
        if method_id == "M17" and any(event.get("method_id") == "M17" and event.get("status") == "timed_out"
                                      for event in previous_events):
            registry_state[method_id]["status"] = "timed_out"
            for track in row["tracks"]:
                registry_state[method_id]["track_status"][track] = "timed_out"
            registry_state[method_id]["notes"].append(
                "stopped remaining folds after the first 180-second fit timeout")
            _atomic_json(_paths()["registry"], registry_state)
            continue
        if method_id == "M09" and any(event.get("method_id") == "M09" and event.get("status") == "timed_out"
                                      for event in previous_events):
            registry_state[method_id]["status"] = "timed_out"
            for track in row["tracks"]:
                registry_state[method_id]["track_status"][track] = "timed_out"
            registry_state[method_id]["notes"].append(
                "stopped the remaining folds after the first 180-second as-of feature timeout")
            _atomic_json(_paths()["registry"], registry_state)
            continue
        if method_id in {"M10", "M14", "M15"}:
            for track in row["tracks"]:
                status = "completed"
                for month in SCREEN_MONTHS:
                    path = _run_calibration(state, protocol, code_hashes, versions, method_id, track, month)
                    if path is None:
                        status = "budget_skipped" if state["fit_attempts"] >= protocol["budget"]["fit_cap"] else "invalid"
                registry_state[method_id]["track_status"][track] = status
            registry_state[method_id]["status"] = "completed"
        elif method_id == "M20":
            blend_reports = _run_blend_method(state, protocol, registry_state, code_hashes, versions)
            results.extend(blend_reports)
            _atomic_json(_paths()["registry"], registry_state)
            continue
        else:
            for track in row["tracks"]:
                if method_id == "M18" and not versions.get("xgboost"):
                    registry_state[method_id]["track_status"][track] = "dependency_blocked"
                    continue
                status = "completed"
                for month in SCREEN_MONTHS:
                    if _has_terminal_event(method_id, track, month, "timed_out"):
                        status = "timed_out"
                        break
                    try:
                        path = _build_model_candidate(state, protocol, code_hashes, versions,
                                                      method_id, track, month)
                    except Exception as exc:
                        _ledger({"stage": "A1", "method_id": method_id, "track": track,
                                 "month": month, "status": "invalid", "counted_fit": False,
                                 "error": f"{type(exc).__name__}: {exc}"})
                        print(f"INVALID {method_id} {track} {month}: {exc}", flush=True)
                        path = None
                    if path is None:
                        if _has_terminal_event(method_id, track, month, "timed_out"):
                            status = "timed_out"
                            break
                        status = "budget_skipped" if state["fit_attempts"] >= protocol["budget"]["fit_cap"] else "invalid"
                registry_state[method_id]["track_status"][track] = status
        _atomic_json(_paths()["registry"], registry_state)
        for track in row["tracks"]:
            if all(_candidate_path(method_id, track, month).exists() for month in SCREEN_MONTHS):
                report = _score_screen_method(method_id, track, SCREEN_MONTHS)
                results.append(report)
                registry_state[method_id].setdefault("screen_results", {})[track] = report
                score = report.get("candidate", {}).get("mdape_percent")
                print(f"SCORE {method_id} {track}: screen MdAPE {score}", flush=True)
                if report.get("relative_mdape_improvement", 0.0) <= 0.0 and report.get("months_improved", 0) == 0:
                    registry_state[method_id]["track_status"][track] = "no_gain"
                elif report.get("status") != "incomplete":
                    registry_state[method_id]["track_status"][track] = "completed"
        if method_id == "M18" and all(value == "dependency_blocked" for value in registry_state[method_id]["track_status"].values()):
            registry_state[method_id]["status"] = "dependency_blocked"
        elif registry_state[method_id]["track_status"] and all(
                value == "no_gain" for value in registry_state[method_id]["track_status"].values()):
            registry_state[method_id]["status"] = "no_gain"
        elif any(value == "timed_out" for value in registry_state[method_id]["track_status"].values()):
            registry_state[method_id]["status"] = "timed_out"
        elif any(value in {"completed", "no_gain"} for value in registry_state[method_id]["track_status"].values()):
            registry_state[method_id]["status"] = "completed"
        _atomic_json(_paths()["registry"], registry_state)
        if state["fit_attempts"] >= protocol["budget"]["fit_cap"] or _elapsed(state) >= protocol["budget"]["wallclock_minutes"] * 60:
            break
    summary = {"run_id": protocol["run_id"], "generated_at_utc": _now().isoformat(),
               "fit_attempts": state["fit_attempts"], "stage_fits": state["stage_fits"],
               "elapsed_minutes": round(_elapsed(state) / 60.0, 2), "screen_months": list(SCREEN_MONTHS),
               "method_results": results, "registry_state": registry_state,
               "notes": ["screening results use June and November only; promotion requires four development months",
                         "the six-month retrospective benchmark has not been read by this phase"]}
    _atomic_json(OUTPUT_ROOT / "screen_results.json", summary)
    state["active_stage"] = None
    state["status"] = "screen_completed"
    _flush_state(state)
    return summary


def budget_allows_fit(elapsed_seconds: float, stage: str, timeout_seconds: int) -> bool:
    total = 240 * 60
    research = 180 * 60
    final_stage = stage in {"A4", "A5", "benchmark", "package"}
    end = float(elapsed_seconds) + int(timeout_seconds)
    if end > total:
        return False
    if not final_stage and end > research:
        return False
    return True


def validate_registry(registry: list[dict]) -> None:
    ids = [item.get("id") for item in registry]
    expected = [f"M{i:02d}" for i in range(1, 21)]
    if len(ids) != 20 or len(set(ids)) != 20:
        raise ValueError("method registry ids must be unique and contain exactly twenty cards")
    if set(ids) != set(expected):
        raise ValueError("method registry must contain M01 through M20")
    valid_statuses = {"not_started", "running", "completed", "no_gain", "invalid", "timed_out",
                      "dependency_blocked", "budget_skipped"}
    for item in registry:
        if item.get("status") not in valid_statuses:
            raise ValueError(f"invalid method status for {item.get('id')}")
        if not item.get("tracks"):
            raise ValueError(f"method has no eligible track: {item.get('id')}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Time-safe v2 model research runner")
    parser.add_argument("command", choices=["preflight", "screen", "confirm", "compose", "benchmark"])
    parser.add_argument("--resume", action="store_true", help="reuse only hash-verified research artifacts")
    args = parser.parse_args(argv)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    protocol = _json(PROTOCOL_PATH)
    registry = _json(REGISTRY_PATH)
    validate_registry(registry)
    state = _load_state(protocol)
    code_hashes, versions = _source_hashes(), _runtime_versions()
    if args.command == "preflight":
        audit = _preflight(protocol, registry)
        print(json.dumps(audit, indent=2, ensure_ascii=False))
        return 0 if audit["status"] == "passed" else 1
    audit_path = _paths()["audit"]
    audit = _json(audit_path) if audit_path.exists() else _preflight(protocol, registry)
    if audit["status"] != "passed":
        print("Preflight failed; no model fits were started.", file=sys.stderr)
        return 2
    if args.command == "screen":
        summary = _run_screen(state, protocol, registry, code_hashes, versions)
        result_count = len(summary["method_results"])
        result_status = "screen_complete"
    elif args.command == "confirm":
        summary = _run_confirmation(state, protocol, code_hashes, versions)
        result_count = sum(len(checkpoint.get("reports", [])) for checkpoint in [summary["checkpoint_C1"]])
        result_status = "confirmation_complete"
    elif args.command == "compose":
        summary = _run_compositions(state, protocol, code_hashes, versions)
        result_count = len(summary["checkpoint_C2"]["reports"])
        result_status = "composition_complete_and_frozen"
    else:
        summary = _run_final_benchmark(state, protocol, code_hashes, versions)
        result_count = len(summary["reports"])
        result_status = "final_benchmark_complete"
    print(json.dumps({"status": result_status, "fit_attempts": summary["fit_attempts"],
                      "elapsed_minutes": summary["elapsed_minutes"], "results": result_count}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
