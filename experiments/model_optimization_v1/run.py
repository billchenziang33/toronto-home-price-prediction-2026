"""Auditable, resumable model-comparison runner.

The initial ``audit`` stage is intentionally training-free. Later stages append fit
events and save outputs under ``outputs/model_optimization_v1``.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import pickle
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import importlib.metadata as metadata
import numpy as np
import pandas as pd

from . import __version__
from .adapters import FORBIDDEN_COMMON, FrozenFeatureTransformer, PriceModel, PROJECT_ROOT
from .metrics import monthly_scores, score, stable_fingerprint

CONFIG_PATH = PROJECT_ROOT / "experiments" / "model_optimization_v1" / "configs.json"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "model_optimization_v1"
RUN_STATE_PATH = OUTPUT_ROOT / "run_state.json"
BASELINE_PATH = PROJECT_ROOT / "release_candidate" / "evidence" / "validation_predictions_frozen.csv"
BASELINE_METRICS_PATH = PROJECT_ROOT / "release_candidate" / "evidence" / "validation_metrics.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _append_event(event: dict) -> None:
    path = OUTPUT_ROOT / "fit_ledger.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(event, sort_keys=True, default=str) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _load_state() -> dict:
    if not RUN_STATE_PATH.exists():
        audit()
    return json.loads(RUN_STATE_PATH.read_text(encoding="utf-8"))


def _save_state(state: dict) -> None:
    state["last_updated_at_utc"] = _utc_now()
    _dump(RUN_STATE_PATH, state)


def _runtime_versions() -> dict[str, str | None]:
    return {name: _package_version(name) for name in ["numpy", "pandas", "scikit-learn", "pyarrow", "catboost", "lightgbm", "xgboost"]}


def _trial_code_hashes() -> dict[str, str]:
    frozen_features = PROJECT_ROOT / "release_candidate" / "features.py"
    files = ["configs.json", "adapters.py", "fit_worker.py", "metrics.py"]
    result = {name: sha256_file(PROJECT_ROOT / "experiments" / "model_optimization_v1" / name) for name in files}
    result["release_candidate/features.py"] = sha256_file(frozen_features)
    return result


def _trial_fingerprint(track: str, month: str, model_family: str, target: str, model_config: dict,
                       variant: str | None = None) -> str:
    config = {"track": track, "month": month, "model_family": model_family,
              "target": target, "model_config": model_config, "versions": _runtime_versions()}
    if variant is not None:
        config["variant"] = variant
    return stable_fingerprint(config, load_config()["data_sha256"], _trial_code_hashes())


def _ledger_events() -> list[dict]:
    path = OUTPUT_ROOT / "fit_ledger.jsonl"
    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(json.loads(line))
    return events


def _cached_result(signature: str) -> dict | None:
    events = _ledger_events()
    for event in reversed(events):
        if event.get("signature") != signature or event.get("status") != "completed":
            continue
        prediction_path = PROJECT_ROOT / event["prediction_path"]
        result_path = PROJECT_ROOT / event["result_path"]
        if not prediction_path.is_file() or not result_path.is_file():
            return None
        if sha256_file(prediction_path) != event.get("prediction_sha256"):
            return None
        return json.loads(result_path.read_text(encoding="utf-8"))
    return None


def _baseline_prediction_path(track: str, month: str) -> str | None:
    config = load_config()
    signature = _trial_fingerprint(track, month, "hgbr", "log_sold_price", config["initial_models"]["frozen_hgbr"])
    cached = _cached_result(signature)
    if cached is not None:
        return cached["prediction_path"]
    if month in config["benchmark_months"]:
        # The authoritative file contains both tracks and all six months. Materialize
        # the one-track/one-month slice expected by fit_worker before its ID reindex.
        frame = pd.read_csv(BASELINE_PATH, dtype={"ml_num": str})
        frame = frame.loc[(frame.track == track) & (frame.month.astype(str) == month), ["ml_num", "predicted_price"]]
        path = OUTPUT_ROOT / "baseline" / f"{track}_{month}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        return str(path.relative_to(PROJECT_ROOT))
    return None


def run_fit_trial(stage: str, track: str, month: str, model_family: str,
                  target: str = "log_sold_price", model_config: dict | None = None,
                  timeout_seconds: int | None = None, trial_tag: str | None = None) -> dict:
    config = load_config()
    config_name = "frozen_hgbr" if model_family == "hgbr" else model_family
    model_config = dict(model_config or config["initial_models"][config_name])
    signature = _trial_fingerprint(track, month, model_family, target, model_config, trial_tag)
    cached = _cached_result(signature)
    if cached is not None:
        print(f"CACHE {track} {month} {model_family} {target}", flush=True)
        return {"status": "cached", **cached}

    state = _load_state()
    started_at = datetime.now(timezone.utc)
    elapsed_minutes = (started_at - datetime.fromisoformat(state["started_at_utc"])).total_seconds() / 60
    if elapsed_minutes >= config["wallclock_cap_minutes"]:
        raise RuntimeError("global wall-clock budget is exhausted")
    fit_count = int(state.get("real_data_fits", 0))
    stage_counts = state.setdefault("fit_count_by_stage", {})
    stage_cap = config["stage_fit_caps"].get(stage)
    if stage == "methods":
        # Three screen slots were unused (14 of 18 fits). Reassign them to correct
        # the initially mislabeled LightGBM L1 trials; the global 72-fit cap remains.
        unused_screen_slots = max(0, config["stage_fit_caps"]["screen"] - int(stage_counts.get("screen", 0)))
        stage_cap = int(stage_cap or 0) + min(3, unused_screen_slots)
    if fit_count >= config["fit_cap"]:
        raise RuntimeError("global real-data fit cap is exhausted")
    if stage_cap is not None and int(stage_counts.get(stage, 0)) >= stage_cap:
        raise RuntimeError(f"{stage} stage fit cap is exhausted ({stage_cap})")

    trial_id = signature[:16]
    trial_dir = OUTPUT_ROOT / "trials"
    prediction_dir = OUTPUT_ROOT / "predictions"
    job_dir = OUTPUT_ROOT / "jobs"
    log_dir = OUTPUT_ROOT / "logs"
    for folder in [trial_dir, prediction_dir, job_dir, log_dir]:
        folder.mkdir(parents=True, exist_ok=True)
    prediction_path = prediction_dir / f"{trial_id}.csv"
    result_path = trial_dir / f"{trial_id}.json"
    job_path = job_dir / f"{trial_id}.json"
    baseline_path = _baseline_prediction_path(track, month)
    audit_record = json.loads((OUTPUT_ROOT / "audit.json").read_text(encoding="utf-8"))
    fold_record = audit_record["data"]["cutoff_folds"][month]
    worker_job = {
        "trial_id": trial_id, "track": track, "month": month, "model_family": model_family,
        "target": target, "model_config": model_config, "data_path": config["input_data"]["listings"],
        "prediction_path": str(prediction_path.relative_to(PROJECT_ROOT)),
        "result_path": str(result_path.relative_to(PROJECT_ROOT)),
        "baseline_prediction_path": baseline_path,
    }
    _dump(job_path, worker_job)
    timeout_seconds = int(timeout_seconds or config["fit_timeout_seconds"]["development"])
    attempt = fit_count + 1
    state["real_data_fits"] = attempt
    stage_counts[stage] = int(stage_counts.get(stage, 0)) + 1
    state["last_stage"] = stage
    _save_state(state)
    base_event = {
        "trial_id": trial_id, "signature": signature, "stage": stage, "track": track, "month": month,
        "model_family": model_family, "target": target, "model_config": model_config,
        "data_sha256": config["data_sha256"], "code_sha256": _trial_code_hashes(),
        "cutoff_exclusive": fold_record["cutoff_exclusive"],
        "train_rows": fold_record["train_rows"], "evaluation_rows": fold_record["evaluation_rows"],
        "seed": config["seed"], "threads": config["threads"], "attempt_number": attempt,
        "started_at_utc": started_at.isoformat(), "timeout_seconds": timeout_seconds,
        "prediction_path": str(prediction_path.relative_to(PROJECT_ROOT)),
        "result_path": str(result_path.relative_to(PROJECT_ROOT)),
    }
    _append_event({**base_event, "status": "running"})
    print(f"FIT {attempt}/{config['fit_cap']} {track} {month} {model_family} {target}; timeout={timeout_seconds}s", flush=True)

    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS": str(config["threads"]), "MKL_NUM_THREADS": str(config["threads"]),
                "OPENBLAS_NUM_THREADS": str(config["threads"]), "NUMEXPR_NUM_THREADS": str(config["threads"])})
    temp_dir = PROJECT_ROOT / "tmp"
    env["TEMP"] = str(temp_dir)
    env["TMP"] = str(temp_dir)
    command = [sys.executable, "-m", "experiments.model_optimization_v1.fit_worker", "--job", str(job_path)]
    start = time.perf_counter()
    status, error = "failed", None
    stdout = stderr = ""
    try:
        process = subprocess.run(command, cwd=PROJECT_ROOT, env=env, capture_output=True, text=True,
                                 timeout=timeout_seconds, check=False)
        stdout, stderr = process.stdout, process.stderr
        status = "completed" if process.returncode == 0 else "failed"
        if process.returncode != 0:
            error = f"worker exited {process.returncode}"
    except subprocess.TimeoutExpired as exception:
        status = "timed_out"
        error = f"fit exceeded {timeout_seconds} seconds"
        stdout = exception.stdout.decode(errors="replace") if isinstance(exception.stdout, bytes) else (exception.stdout or "")
        stderr = exception.stderr.decode(errors="replace") if isinstance(exception.stderr, bytes) else (exception.stderr or "")
    elapsed_seconds = time.perf_counter() - start
    (log_dir / f"{trial_id}.stdout.txt").write_text(stdout, encoding="utf-8")
    (log_dir / f"{trial_id}.stderr.txt").write_text(stderr, encoding="utf-8")
    event = {**base_event, "status": status, "ended_at_utc": _utc_now(), "elapsed_seconds": elapsed_seconds,
             "error": error, "stdout_path": str((log_dir / f"{trial_id}.stdout.txt").relative_to(PROJECT_ROOT)),
             "stderr_path": str((log_dir / f"{trial_id}.stderr.txt").relative_to(PROJECT_ROOT))}
    if status == "completed" and result_path.is_file() and prediction_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        event.update({"prediction_sha256": sha256_file(prediction_path), "metrics": result["metrics"],
                      "fit_seconds": result["fit_seconds"]})
    else:
        status = "timed_out" if status == "timed_out" else "failed"
        event["status"] = status
        event["error"] = error or "worker did not create complete result and predictions"
    _append_event(event)
    if event["status"] == "completed":
        print(f"DONE {track} {month} {model_family}: {event['metrics']} ({elapsed_seconds:.1f}s)", flush=True)
        return {"status": "completed", **json.loads(result_path.read_text(encoding="utf-8"))}
    print(f"{event['status'].upper()} {track} {month} {model_family}: {event['error']}", flush=True)
    return event


def pilot() -> dict:
    config = load_config()
    audit_record = audit()
    if audit_record["data"]["sha256"] != config["data_sha256"]:
        raise RuntimeError("data hash changed after audit")
    trials = [
        ("off_market", "2025-06", "hgbr", "log_sold_price"),
        ("off_market", "2025-06", "catboost", "log_sold_price"),
        ("off_market", "2025-06", "lightgbm", "log_sold_price"),
        ("list_aware", "2025-06", "hgbr", "log_sold_price"),
    ]
    versions = _runtime_versions()
    environment = {"python": sys.version, "platform": platform.platform(), "logical_cpus": os.cpu_count(),
                   "versions": versions, "executable": sys.executable, "threads": config["threads"],
                   "install_method": "binary wheels from official PyPI into .venv-opt; base dependencies read from .venv via .pth",
                   "install_outcome": {"catboost": "1.2.10", "lightgbm": "4.7.0", "status": "installed successfully"},
                   "requirements_lock_path": "outputs/model_optimization_v1/environment-lock.txt"}
    _dump(OUTPUT_ROOT / "environment.json", environment)
    results = []
    for track, month, family, target in trials:
        results.append(run_fit_trial("pilot", track, month, family, target,
                                     timeout_seconds=config["fit_timeout_seconds"]["pilot"]))
    state = _load_state()
    state["last_stage"] = "pilot"
    state["pilot_completed_at_utc"] = _utc_now()
    _save_state(state)
    _dump(OUTPUT_ROOT / "pilot_summary.json", {"completed_at_utc": _utc_now(), "results": results,
                                                "fits_used": state["fit_count_by_stage"].get("pilot", 0),
                                                "total_fits_used": state["real_data_fits"],
                                                "versions": versions})
    print(json.dumps({"pilot_fit_attempts": state["fit_count_by_stage"].get("pilot", 0),
                      "total_fit_attempts": state["real_data_fits"], "versions": versions}, indent=2))
    return {"results": results, "state": state, "versions": versions}


def _trial_result(track: str, month: str, family: str, target: str = "log_sold_price",
                  model_config: dict | None = None) -> dict | None:
    config = load_config()
    config_name = "frozen_hgbr" if family == "hgbr" else family
    params = dict(model_config or config["initial_models"][config_name])
    signature = _trial_fingerprint(track, month, family, target, params)
    return _cached_result(signature)


def _preset_config(family: str, preset: list) -> dict:
    """Translate the frozen, ordered parameter presets into library arguments."""
    if family == "lightgbm":
        rounds, rate, leaves, child, regularization = preset
        return {**load_config()["initial_models"]["lightgbm"], "n_estimators": rounds,
                "learning_rate": rate, "num_leaves": leaves, "min_child_samples": child,
                "reg_lambda": regularization}
    if family == "catboost":
        rounds, rate, depth, regularization = preset
        return {**load_config()["initial_models"]["catboost"], "iterations": rounds,
                "learning_rate": rate, "depth": depth, "l2_leaf_reg": regularization}
    if family == "hgbr":
        rounds, rate, leaves, child, regularization = preset
        return {"max_iter": rounds, "learning_rate": rate, "max_leaf_nodes": leaves,
                "min_samples_leaf": child, "l2_regularization": regularization,
                "early_stopping": False, "random_state": 0, "categorical_features": "from_dtype"}
    raise ValueError(f"no tuning presets for {family}")


def _method_model_config(family: str, target: str, config: dict) -> dict:
    """Override an initial RMSE objective when the declared method is L1."""
    params = dict(config)
    if target == "log_sold_price_mae":
        if family == "lightgbm":
            params["objective"] = "regression_l1"
        elif family == "catboost":
            params["loss_function"] = "MAE"
        elif family == "hgbr":
            params["loss"] = "absolute_error"
    return params


def _trial_frame(track: str, month: str, family: str, target: str, params: dict) -> pd.DataFrame | None:
    result = _trial_result(track, month, family, target, params)
    if result is None:
        return None
    frame = pd.read_csv(PROJECT_ROOT / result["prediction_path"], dtype={"ml_num": str})
    return frame


def _pooled_score(frames: list[pd.DataFrame], prediction_column: str = "predicted_price") -> dict:
    if not frames:
        raise ValueError("cannot score an empty prediction set")
    all_rows = pd.concat(frames, ignore_index=True)
    metrics = score(all_rows[prediction_column], all_rows.sold_price)
    return {**metrics, "monthly": monthly_scores(all_rows.assign(predicted_price=all_rows[prediction_column]))}


def tune() -> dict:
    """Run each screen winner's fixed six-config search on the two earlier months."""
    config = load_config()
    screen = json.loads((OUTPUT_ROOT / "screen_leaderboard.json").read_text(encoding="utf-8"))
    winners = {}
    for track in ["off_market", "list_aware"]:
        candidates = [row for row in screen["results"] if row["track"] == track and len(row["months"]) == 3]
        if not candidates:
            raise RuntimeError(f"screen contains no complete family for {track}")
        winner = min(candidates, key=lambda row: (row["mdape_percent"], row["mape_percent"], row["model_family"]))
        winners[track] = winner["model_family"]

    attempted = []
    results_by_trial = {}
    for track, family in winners.items():
        for index, preset in enumerate(config["tuning_presets"][family], start=1):
            params = _preset_config(family, preset)
            for month in config["screening_months"]:
                result = run_fit_trial("tune", track, month, family, "log_sold_price", params,
                                       timeout_seconds=config["fit_timeout_seconds"]["development"])
                attempted.append({"track": track, "family": family, "preset_index": index,
                                  "month": month, "status": result["status"], "trial_id": result["trial_id"]})
                if result["status"] in {"completed", "cached"}:
                    results_by_trial[(track, index, month)] = params

    leaderboard, locked = [], {}
    for track, family in winners.items():
        rows = []
        for index, preset in enumerate(config["tuning_presets"][family], start=1):
            params = _preset_config(family, preset)
            frames = [_trial_frame(track, month, family, "log_sold_price", params)
                      for month in config["screening_months"]]
            frames = [frame for frame in frames if frame is not None]
            if len(frames) != len(config["screening_months"]):
                continue
            metrics = _pooled_score(frames)
            rows.append({"track": track, "model_family": family, "preset_index": index,
                         "preset": preset, "model_config": params, "mdape_percent": metrics["mdape_percent"],
                         "mape_percent": metrics["mape_percent"], "monthly": metrics["monthly"]})
        if not rows:
            raise RuntimeError(f"no complete tuning configuration for {track}")
        best = min(rows, key=lambda row: (row["mdape_percent"], row["mape_percent"], row["preset_index"]))
        locked[track] = {"model_family": family, "target": "log_sold_price",
                         "model_config": best["model_config"], "preset_index": best["preset_index"],
                         "tuning_mdape_percent": best["mdape_percent"],
                         "tuning_mape_percent": best["mape_percent"]}
        leaderboard.extend(rows)
    state = _load_state()
    payload = {"completed_at_utc": _utc_now(), "winners": winners, "results": leaderboard,
               "attempted": attempted, "fit_attempts": state.get("fit_count_by_stage", {}).get("tune", 0),
               "total_fit_attempts": state["real_data_fits"], "locked": locked}
    _dump(OUTPUT_ROOT / "tune_leaderboard.json", payload)
    lines = ["# Fixed-configuration tuning results", "",
             "Only the screen-selected family for each track was tuned. Scores pool the fixed June and November development folds.", "",
             "| Track | Family | Preset | MdAPE | MAPE | June MdAPE | November MdAPE |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for row in sorted(leaderboard, key=lambda item: (item["track"], item["mdape_percent"])):
        monthly = row["monthly"]
        lines.append(f"| {row['track']} | {row['model_family']} | {row['preset_index']} | {row['mdape_percent']:.4f}% | {row['mape_percent']:.4f}% | {monthly['2025-06']['mdape_percent']:.4f}% | {monthly['2025-11']['mdape_percent']:.4f}% |")
    lines += ["", "The selected configuration is a development choice, not an independent test result.", ""]
    (OUTPUT_ROOT / "tune_leaderboard.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return payload


def methods() -> dict:
    """Compare the tuned price model with its one predeclared target alternative and blend."""
    config = load_config()
    tuning = json.loads((OUTPUT_ROOT / "tune_leaderboard.json").read_text(encoding="utf-8"))
    tuned = tuning["locked"]
    attempted = []
    # Complete the third development fold for each tuned raw-price model, then run
    # one target alternative over all three folds. Exact screen/tune matches reuse cache.
    for track, entry in tuned.items():
        family, params = entry["model_family"], entry["model_config"]
        for month in config["development_months"]:
            result = run_fit_trial("methods", track, month, family, "log_sold_price", params,
                                   timeout_seconds=config["fit_timeout_seconds"]["development"])
            attempted.append({"track": track, "month": month, "target": "log_sold_price", "status": result["status"]})
        alternative = "log_sold_price_mae" if track == "off_market" else "log_sold_over_list"
        alternative_params = _method_model_config(family, alternative, params)
        for month in config["development_months"]:
            result = run_fit_trial("methods", track, month, family, alternative, alternative_params,
                                   timeout_seconds=config["fit_timeout_seconds"]["development"])
            attempted.append({"track": track, "month": month, "target": alternative, "status": result["status"]})

    locked = {}
    diagnostics = {}
    for track, entry in tuned.items():
        family, params = entry["model_family"], entry["model_config"]
        alternative = "log_sold_price_mae" if track == "off_market" else "log_sold_over_list"
        alternative_params = _method_model_config(family, alternative, params)
        raw_frames = [_trial_frame(track, month, family, "log_sold_price", params) for month in config["development_months"]]
        alt_frames = [_trial_frame(track, month, family, alternative, alternative_params) for month in config["development_months"]]
        if any(frame is None for frame in raw_frames + alt_frames):
            raise RuntimeError(f"incomplete development predictions for {track} method comparison")
        raw = pd.concat(raw_frames, ignore_index=True)
        alt = pd.concat(alt_frames, ignore_index=True)
        # The family screen's HGBR scores are the predeclared same-fold frozen-model reference.
        base = raw.baseline_price.to_numpy(dtype=float)
        actual = raw.sold_price.to_numpy(dtype=float)
        raw_score = score(raw.predicted_price, actual)
        alt_score = score(alt.predicted_price, actual)
        preferred = alternative if alt_score["mdape_percent"] < raw_score["mdape_percent"] else "log_sold_price"
        challenger = alt if preferred == alternative else raw
        weights = []
        for weight in [0.0, 0.25, 0.5, 0.75, 1.0]:
            prediction = np.round((1.0 - weight) * base + weight * challenger.predicted_price.to_numpy(dtype=float))
            measure = score(prediction, actual)
            weights.append({"weight": weight, "mdape_percent": measure["mdape_percent"],
                            "mape_percent": measure["mape_percent"]})
        best_weight = min(weights, key=lambda row: (row["mdape_percent"], row["mape_percent"], -row["weight"]))
        chosen_params = alternative_params if preferred == alternative else params
        locked[track] = {**entry, "target": preferred, "model_config": chosen_params,
                         "blend_weight": best_weight["weight"],
                         "development_mdape_percent": best_weight["mdape_percent"],
                         "development_mape_percent": best_weight["mape_percent"],
                         "alternative_target": alternative, "raw_target_mdape_percent": raw_score["mdape_percent"],
                         "alternative_target_mdape_percent": alt_score["mdape_percent"]}
        diagnostics[track] = {"raw_target": raw_score, "alternative_target": alt_score,
                              "selected_target": preferred, "blend_weights": weights,
                              "best_weight": best_weight}
    invalid_legacy_l1 = [event["trial_id"] for event in _ledger_events()
                         if event.get("track") == "off_market" and event.get("target") == "log_sold_price_mae"
                         and event.get("model_family") == "lightgbm"
                         and event.get("model_config", {}).get("objective") != "regression_l1"]
    payload = {"locked_at_utc": _utc_now(), "locked": locked, "diagnostics": diagnostics,
               "attempted": attempted, "fit_attempts": _load_state().get("fit_count_by_stage", {}).get("methods", 0),
               "total_fit_attempts": _load_state()["real_data_fits"],
               "invalid_legacy_l1_trials_excluded": sorted(set(invalid_legacy_l1)),
               "reallocated_fit_slots": {"from_stage": "screen", "to_stage": "methods", "count": 3}}
    payload["config_sha256"] = hashlib.sha256(json.dumps(locked, sort_keys=True).encode()).hexdigest()
    _dump(OUTPUT_ROOT / "winner_config.json", payload)
    lines = ["# Target and blend comparison", "",
             "Scores use the three fixed development folds. A weight of 0 is the same-fold HGBR reference; 1 is the selected challenger.", "",
             "| Track | Raw log-price MdAPE | Alternative target MdAPE | Chosen target | Chosen blend weight | Development blend MdAPE | MAPE |",
             "|---|---:|---:|---|---:|---:|---:|"]
    for track, item in diagnostics.items():
        chosen = locked[track]
        lines.append(f"| {track} | {item['raw_target']['mdape_percent']:.4f}% | {item['alternative_target']['mdape_percent']:.4f}% | {chosen['target']} | {chosen['blend_weight']:.2f} | {chosen['development_mdape_percent']:.4f}% | {chosen['development_mape_percent']:.4f}% |")
    lines += ["", "The blend weight and target are now locked; later benchmark results will not change them.", ""]
    (OUTPUT_ROOT / "methods_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return payload


def _cluster_bootstrap_mdape(baseline: pd.DataFrame, candidate: pd.DataFrame,
                             repeats: int = 10000, seed: int = 0) -> dict:
    """Resample whole months to show sensitivity to shared market shocks."""
    from .metrics import ape_percent
    months = sorted(set(baseline.month.astype(str)))
    base_by_month, new_by_month = [], []
    for month in months:
        b = baseline.loc[baseline.month.astype(str) == month]
        n = candidate.loc[candidate.month.astype(str) == month]
        if set(b.ml_num.astype(str)) != set(n.ml_num.astype(str)):
            raise ValueError(f"month-cluster pairing mismatch for {month}")
        base_by_month.append(ape_percent(b.predicted_price, b.sold_price))
        new_by_month.append(ape_percent(n.predicted_price, n.sold_price))
    rng = np.random.default_rng(seed)
    deltas = np.empty(repeats, dtype=float)
    for index in range(repeats):
        picked = rng.integers(0, len(months), size=len(months))
        b = np.concatenate([base_by_month[i] for i in picked])
        n = np.concatenate([new_by_month[i] for i in picked])
        deltas[index] = np.median(n) - np.median(b)
    return {"months": months, "clusters": len(months), "repeats": repeats, "seed": seed,
            "delta_percentage_points": float(np.median(np.concatenate(new_by_month)) - np.median(np.concatenate(base_by_month))),
            "ci95_percentage_points": [float(value) for value in np.percentile(deltas, [2.5, 97.5])]}


def benchmark() -> dict:
    """Run each locked candidate once over the six-month retrospective benchmark."""
    from .metrics import paired_mdape_bootstrap
    config = load_config()
    locked = json.loads((OUTPUT_ROOT / "winner_config.json").read_text(encoding="utf-8"))["locked"]
    attempts = []
    for track, item in locked.items():
        if item["blend_weight"] <= 0:
            continue
        for month in config["benchmark_months"]:
            result = run_fit_trial("benchmark", track, month, item["model_family"], item["target"],
                                   item["model_config"], timeout_seconds=config["fit_timeout_seconds"]["benchmark"])
            attempts.append({"track": track, "month": month, "status": result["status"],
                             "trial_id": result["trial_id"]})

    frozen = pd.read_csv(BASELINE_PATH, dtype={"ml_num": str})
    results, promotions, diagnostics = {}, {}, {}
    for track, item in locked.items():
        base = frozen.loc[frozen.track == track].copy()
        if item["blend_weight"] <= 0:
            final = base.copy()
            final["predicted_price"] = np.round(final.predicted_price)
        else:
            frames = []
            for month in config["benchmark_months"]:
                frame = _trial_frame(track, month, item["model_family"], item["target"], item["model_config"])
                if frame is None:
                    continue
                frames.append(frame)
            if len(frames) != len(config["benchmark_months"]):
                raise RuntimeError(f"locked benchmark is incomplete for {track}")
            raw = pd.concat(frames, ignore_index=True)
            base_lookup = base.set_index(["ml_num", "track", "month"]).predicted_price
            keys = pd.MultiIndex.from_frame(raw[["ml_num", "track", "month"]])
            exact_base = base_lookup.reindex(keys).to_numpy(dtype=float)
            if not np.isfinite(exact_base).all():
                raise RuntimeError(f"saved frozen benchmark predictions do not pair for {track}")
            raw["predicted_price"] = np.round((1.0 - item["blend_weight"]) * exact_base +
                                               item["blend_weight"] * raw.predicted_price.to_numpy(dtype=float))
            final = raw
        final = final.sort_values(["month", "ml_num"], kind="mergesort").reset_index(drop=True)
        base = base.sort_values(["month", "ml_num"], kind="mergesort").reset_index(drop=True)
        if not np.array_equal(final.ml_num.astype(str).to_numpy(), base.ml_num.astype(str).to_numpy()) or not np.array_equal(final.month.astype(str), base.month.astype(str)):
            raise RuntimeError(f"frozen and candidate benchmark rows do not align for {track}")
        score_columns = ["ml_num", "track", "month", "sold_price", "predicted_price"]
        if "house_type_name" in final and "house_type_name" in base:
            score_columns.append("house_type_name")
        candidate_view = final[score_columns].copy()
        base_view = base[score_columns].copy()
        candidate_view["predicted_price"] = np.round(candidate_view.predicted_price)
        base_score = score(base_view.predicted_price, base_view.sold_price)
        candidate_score = score(candidate_view.predicted_price, candidate_view.sold_price)
        base_month = monthly_scores(base_view)
        candidate_month = monthly_scores(candidate_view)
        home_bootstrap = paired_mdape_bootstrap(base_view, candidate_view, repeats=10000, seed=0)
        month_bootstrap = _cluster_bootstrap_mdape(base_view, candidate_view)
        group_diagnostics = {}
        for column in ["house_type_name"]:
            if column in candidate_view:
                group_diagnostics[column] = {}
                for group, rows in candidate_view.groupby(column, dropna=False, sort=True):
                    ref_rows = base_view.loc[base_view[column].fillna("<missing>") == ("<missing>" if pd.isna(group) else group)]
                    group_rows = rows.assign(_group=rows[column].fillna("<missing>"))
                    group_ref = ref_rows.assign(_group=ref_rows[column].fillna("<missing>"))
                    group_diagnostics[column][str(group)] = {"baseline": score(group_ref.predicted_price, group_ref.sold_price),
                                                             "candidate": score(group_rows.predicted_price, group_rows.sold_price)}
        # Actual price deciles are descriptive post-evaluation diagnostics only.
        price_decile = pd.qcut(candidate_view.sold_price, 10, labels=False, duplicates="drop")
        candidate_by_decile = candidate_view.assign(price_decile=price_decile)
        baseline_by_decile = base_view.assign(price_decile=price_decile)
        group_diagnostics["actual_sold_price_decile"] = {
            str(int(group) + 1): {"baseline": score(group_rows.predicted_price, group_rows.sold_price),
                                  "candidate": score(candidate_by_decile.loc[candidate_by_decile.price_decile == group, "predicted_price"],
                                                     candidate_by_decile.loc[candidate_by_decile.price_decile == group, "sold_price"])}
            for group, group_rows in baseline_by_decile.groupby("price_decile", sort=True)
        }
        monthly_relative = {month: (base_month[month]["mdape_percent"] - candidate_month[month]["mdape_percent"]) /
                                      base_month[month]["mdape_percent"] for month in config["benchmark_months"]}
        relative_improvement = (base_score["mdape_percent"] - candidate_score["mdape_percent"]) / base_score["mdape_percent"]
        promotion = {"relative_mdape_improvement": relative_improvement,
                     "months_improved": sum(value > 0 for value in monthly_relative.values()),
                     "monthly_relative_improvement": monthly_relative,
                     "worst_month_relative_regression": max(-value for value in monthly_relative.values()),
                     "mape_relative_change": (candidate_score["mape_percent"] - base_score["mape_percent"]) / base_score["mape_percent"],
                     "paired_home_bootstrap_upper_pp": home_bootstrap["ci95_percentage_points"][1],
                     "passes": bool(relative_improvement >= config["promotion"]["min_relative_mdape_improvement"] and
                                    sum(value > 0 for value in monthly_relative.values()) >= config["promotion"]["min_months_improved"] and
                                    max(-value for value in monthly_relative.values()) <= config["promotion"]["max_month_relative_regression"] and
                                    (candidate_score["mape_percent"] - base_score["mape_percent"]) / base_score["mape_percent"] <= config["promotion"]["max_mape_relative_regression"] and
                                    home_bootstrap["ci95_percentage_points"][1] < config["promotion"]["paired_home_bootstrap_upper_percent_points_lt"])}
        diagnostics[track] = {"baseline": base_score, "candidate": candidate_score, "baseline_monthly": base_month,
                              "candidate_monthly": candidate_month, "paired_home_bootstrap": home_bootstrap,
                              "month_cluster_bootstrap_sensitivity": month_bootstrap,
                              "post_evaluation_groups": group_diagnostics,
                              "promotion": promotion}
        promotions[track] = promotion
        candidate_view.to_csv(OUTPUT_ROOT / f"benchmark_predictions_{track}.csv", index=False)
        base_view.to_csv(OUTPUT_ROOT / f"benchmark_baseline_{track}.csv", index=False)
        results[track] = {"n": candidate_score["n"], "mdape_percent": candidate_score["mdape_percent"],
                          "mape_percent": candidate_score["mape_percent"], "baseline_mdape_percent": base_score["mdape_percent"],
                          "baseline_mape_percent": base_score["mape_percent"]}
    payload = {"completed_at_utc": _utc_now(), "locked_config_sha256": json.loads((OUTPUT_ROOT / "winner_config.json").read_text(encoding="utf-8"))["config_sha256"],
               "fit_attempts": _load_state().get("fit_count_by_stage", {}).get("benchmark", 0),
               "total_fit_attempts": _load_state()["real_data_fits"], "attempts": attempts,
               "results": results, "diagnostics": diagnostics,
               "dual_track_upgrade_eligible": all(promotions.get(track, {}).get("passes", False) for track in ["off_market", "list_aware"])}
    _dump(OUTPUT_ROOT / "benchmark.json", payload)
    lines = ["# Six-month locked benchmark", "",
             "The December 2025–May 2026 period has been examined before and is a retrospective benchmark, not a fresh independent confirmation set.", "",
             "| Track | Frozen MdAPE | Candidate MdAPE | Relative improvement | Frozen MAPE | Candidate MAPE | Home-paired 95% CI (pp) | Months improved | Promotion |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for track, item in diagnostics.items():
        p, h = item["promotion"], item["paired_home_bootstrap"]
        lines.append(f"| {track} | {item['baseline']['mdape_percent']:.4f}% | {item['candidate']['mdape_percent']:.4f}% | {p['relative_mdape_improvement']:.2%} | {item['baseline']['mape_percent']:.4f}% | {item['candidate']['mape_percent']:.4f}% | [{h['ci95_percentage_points'][0]:+.4f}, {h['ci95_percentage_points'][1]:+.4f}] | {p['months_improved']}/6 | {'PASS' if p['passes'] else 'FAIL'} |")
    lines += ["", "## Monthly MdAPE", "", "| Track | Month | Frozen | Candidate | Relative improvement |", "|---|---|---:|---:|---:|"]
    for track, item in diagnostics.items():
        for month in config["benchmark_months"]:
            base_value = item["baseline_monthly"][month]["mdape_percent"]
            candidate_value = item["candidate_monthly"][month]["mdape_percent"]
            lines.append(f"| {track} | {month} | {base_value:.4f}% | {candidate_value:.4f}% | {(base_value-candidate_value)/base_value:.2%} |")
    lines += ["", "## Post-evaluation error diagnostics", "", "Actual sold-price deciles and house types are reported only to locate residual error patterns. They did not affect model inputs or selection.", ""]
    for track, item in diagnostics.items():
        lines += [f"### {track} by actual sold-price decile", "", "| Decile | Frozen MdAPE | Candidate MdAPE | Frozen MAPE | Candidate MAPE |", "|---:|---:|---:|---:|---:|"]
        for decile, scores in item["post_evaluation_groups"]["actual_sold_price_decile"].items():
            lines.append(f"| {decile} | {scores['baseline']['mdape_percent']:.3f}% | {scores['candidate']['mdape_percent']:.3f}% | {scores['baseline']['mape_percent']:.3f}% | {scores['candidate']['mape_percent']:.3f}% |")
        lines += ["", f"### {track} by property type", "", "| Property type | Frozen MdAPE | Candidate MdAPE | Frozen MAPE | Candidate MAPE |", "|---|---:|---:|---:|---:|"]
        for property_type, scores in item["post_evaluation_groups"]["house_type_name"].items():
            lines.append(f"| {property_type} | {scores['baseline']['mdape_percent']:.3f}% | {scores['candidate']['mdape_percent']:.3f}% | {scores['baseline']['mape_percent']:.3f}% | {scores['candidate']['mape_percent']:.3f}% |")
        lines.append("")
    lines += ["The month-cluster interval resamples six months and is highly uncertain; it is shown as a shock-sensitivity diagnostic, not a reliable population interval.", ""]
    (OUTPUT_ROOT / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return payload


def _counted_full_fit(label: str, track: str, family: str, target: str, params: dict) -> tuple[PriceModel, dict]:
    """Fit on all labeled rows and persist an estimator with a separately saved schema."""
    config = load_config()
    identity = {"kind": "full_data_export", "label": label, "track": track, "family": family,
                "target": target, "params": params, "versions": _runtime_versions()}
    signature = stable_fingerprint(identity, config["data_sha256"], _trial_code_hashes())
    trial_id = signature[:16]
    output_dir = OUTPUT_ROOT / "export_models"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{label}_{trial_id}.pkl"
    metadata_path = output_dir / f"{label}_{trial_id}.json"
    if metadata_path.is_file() and model_path.is_file():
        saved = json.loads(metadata_path.read_text(encoding="utf-8"))
        if saved.get("signature") == signature and sha256_file(model_path) == saved.get("model_sha256"):
            model = PriceModel(track, family, params, target)
            model.estimator = pickle.loads(model_path.read_bytes())
            model.transformer.columns = saved["feature_columns"]
            model.transformer.category_values = saved["category_values"]
            return model, saved

    state = _load_state()
    started_at = datetime.now(timezone.utc)
    elapsed_minutes = (started_at - datetime.fromisoformat(state["started_at_utc"])).total_seconds() / 60
    if elapsed_minutes >= config["wallclock_cap_minutes"] or int(state.get("real_data_fits", 0)) >= config["fit_cap"]:
        raise RuntimeError("global optimization budget is exhausted before full-data export")
    stage_counts = state.setdefault("fit_count_by_stage", {})
    stage_cap = int(config["stage_fit_caps"]["benchmark"])
    if int(stage_counts.get("benchmark", 0)) >= stage_cap:
        raise RuntimeError("benchmark stage fit cap is exhausted before full-data export")
    attempt = int(state.get("real_data_fits", 0)) + 1
    stage_counts["benchmark"] = int(stage_counts.get("benchmark", 0)) + 1
    state["real_data_fits"] = attempt
    state["last_stage"] = "package"
    _save_state(state)
    event = {"trial_id": trial_id, "signature": signature, "stage": "benchmark", "kind": "full_data_export",
             "label": label, "track": track, "month": "full_data", "model_family": family,
             "target": target, "model_config": params, "data_sha256": config["data_sha256"],
             "code_sha256": _trial_code_hashes(), "seed": config["seed"], "threads": config["threads"],
             "attempt_number": attempt, "started_at_utc": started_at.isoformat(),
             "model_path": str(model_path.relative_to(PROJECT_ROOT)), "status": "running"}
    _append_event(event)
    try:
        listings = pd.read_parquet(PROJECT_ROOT / config["input_data"]["listings"])
        started = time.perf_counter()
        model = PriceModel(track, family, params, target).fit(listings)
        fit_seconds = time.perf_counter() - started
        payload = pickle.dumps(model.estimator, protocol=pickle.HIGHEST_PROTOCOL)
        temp_path = model_path.with_suffix(model_path.suffix + ".tmp")
        temp_path.write_bytes(payload)
        os.replace(temp_path, model_path)
        saved = {"signature": signature, "label": label, "track": track, "model_family": family,
                 "target": target, "model_config": params, "feature_columns": model.transformer.columns,
                 "category_values": {key: [str(value) for value in values]
                                     for key, values in model.transformer.category_values.items()},
                 "train_rows": len(listings), "fit_seconds": fit_seconds,
                 "model_path": str(model_path.relative_to(PROJECT_ROOT)), "model_sha256": sha256_file(model_path),
                 "versions": _runtime_versions()}
        _dump(metadata_path, saved)
        _append_event({**event, "status": "completed", "ended_at_utc": _utc_now(),
                       "fit_seconds": fit_seconds, "model_path": saved["model_path"],
                       "model_sha256": saved["model_sha256"], "metadata_path": str(metadata_path.relative_to(PROJECT_ROOT))})
        return model, saved
    except Exception as error:
        _append_event({**event, "status": "failed", "ended_at_utc": _utc_now(),
                       "error": f"{type(error).__name__}: {error}"})
        raise


def package() -> dict:
    """Export the qualifying two-track candidate and verify its prediction path."""
    benchmark_path = OUTPUT_ROOT / "benchmark.json"
    if not benchmark_path.is_file():
        raise RuntimeError("run the locked six-month benchmark before packaging")
    benchmark_result = json.loads(benchmark_path.read_text(encoding="utf-8"))
    if not benchmark_result["dual_track_upgrade_eligible"]:
        return {"status": "no_dual_track_candidate", "benchmark": benchmark_result}

    from .metrics import monthly_scores
    locked = json.loads((OUTPUT_ROOT / "winner_config.json").read_text(encoding="utf-8"))["locked"]
    export_models: dict[str, tuple[PriceModel, dict]] = {}
    for track, item in locked.items():
        if item["blend_weight"] > 0:
            export_models[f"candidate_{track}"] = _counted_full_fit(
                f"candidate_{track}", track, item["model_family"], item["target"], item["model_config"])
        if item["blend_weight"] < 1:
            export_models[f"frozen_{track}"] = _counted_full_fit(
                f"frozen_{track}", track, "hgbr", "log_sold_price", load_config()["initial_models"]["frozen_hgbr"])

    # Refit one fixed development month for each track and demand identical predictions.
    repeat_checks = {}
    for track, item in locked.items():
        month = "2025-11"
        repeated = run_fit_trial("benchmark", track, month, item["model_family"], item["target"],
                                 item["model_config"], timeout_seconds=load_config()["fit_timeout_seconds"]["benchmark"],
                                 trial_tag="reproducibility-repeat")
        if repeated["status"] not in {"completed", "cached"}:
            raise RuntimeError(f"reproducibility repeat failed for {track}: {repeated.get('error')}")
        original = _trial_frame(track, month, item["model_family"], item["target"], item["model_config"])
        second = pd.read_csv(PROJECT_ROOT / repeated["prediction_path"], dtype={"ml_num": str})
        original = original.sort_values("ml_num", kind="mergesort").reset_index(drop=True)
        second = second.sort_values("ml_num", kind="mergesort").reset_index(drop=True)
        same = (original.ml_num.astype(str).equals(second.ml_num.astype(str)) and
                np.array_equal(original.predicted_price.to_numpy(dtype=float), second.predicted_price.to_numpy(dtype=float)))
        if not same:
            raise RuntimeError(f"fixed-seed prediction repeat differs for {track}")
        repeat_checks[track] = {"month": month, "identical": True, "rows": len(second),
                                "repeat_trial_id": repeated["trial_id"]}

    candidate_root = PROJECT_ROOT / "model_candidate_v2"
    models_dir = candidate_root / "models"
    submissions_dir = candidate_root / "submissions"
    models_dir.mkdir(parents=True, exist_ok=True)
    submissions_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(PROJECT_ROOT / "release_candidate" / "features.py", candidate_root / "features.py")
    saved_bundles = {}
    for label, (model, saved) in export_models.items():
        filename = label + ".pkl"
        target_path = models_dir / filename
        shutil.copyfile(PROJECT_ROOT / saved["model_path"], target_path)
        bundle = {**saved, "model_file": f"models/{filename}",
                  "blend_weight": locked[saved["track"]]["blend_weight"]}
        _dump(models_dir / f"{label}.json", bundle)
        saved_bundles[label] = bundle

        # Confirm that the disk estimator plus stored category schema reproduces the
        # in-memory full-data model on the actual competition test rows.
        test_key = "test_off_market" if saved["track"] == "off_market" else "test_list_aware"
        homes = pd.read_parquet(PROJECT_ROOT / load_config()["input_data"][test_key])
        restored = PriceModel(saved["track"], saved["model_family"], saved["model_config"], saved["target"])
        restored.estimator = pickle.loads(target_path.read_bytes())
        restored.transformer.columns = saved["feature_columns"]
        restored.transformer.category_values = saved["category_values"]
        if not np.array_equal(restored.predict(homes), model.predict(homes)):
            raise RuntimeError(f"serialized model changes predictions for {label}")

    predictor_source = '''"""Predict competition homes with model_candidate_v2 and validate both CSVs."""
from __future__ import annotations
import importlib.util, json, os, pickle, sys
from pathlib import Path
import numpy as np
import pandas as pd

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / "data" / ".hf_cache"))
import data

feature_spec = importlib.util.spec_from_file_location("candidate_features", PACKAGE / "features.py")
features = importlib.util.module_from_spec(feature_spec)
feature_spec.loader.exec_module(features)

def predict(bundle_name: str, homes: pd.DataFrame) -> np.ndarray:
    bundle_path = PACKAGE / "models" / f"{bundle_name}.json"
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    X = (features.off_market_features(homes) if bundle["track"] == "off_market"
         else features.list_aware_features(homes)).copy()
    if list(X.columns) != bundle["feature_columns"]:
        raise ValueError(f"feature schema differs for {bundle_name}")
    for column, categories in bundle["category_values"].items():
        X[column] = pd.Categorical(X[column], categories=categories)
    with (PACKAGE / bundle["model_file"]).open("rb") as stream:
        estimator = pickle.load(stream)
    transformed = np.asarray(estimator.predict(X), dtype=float)
    if bundle["target"] == "log_sold_over_list":
        price = pd.to_numeric(homes["list_price"], errors="raise").to_numpy(dtype=float) * np.exp(transformed)
    else:
        price = np.exp(transformed)
    if not np.isfinite(price).all() or (price <= 0).any():
        raise ValueError(f"non-positive or non-finite output from {bundle_name}")
    return price

def main() -> None:
    check_spec = importlib.util.spec_from_file_location("candidate_submission_checker", ROOT / "release_candidate" / "check_submission.py")
    checker = importlib.util.module_from_spec(check_spec)
    check_spec.loader.exec_module(checker)
    results = {}
    for track in ("off_market", "list_aware"):
        homes = data.load(track)
        bundle = json.loads((PACKAGE / "models" / f"candidate_{track}.json").read_text(encoding="utf-8"))
        predicted = predict(f"candidate_{track}", homes)
        weight = float(bundle["blend_weight"])
        if weight < 1.0:
            baseline = predict(f"frozen_{track}", homes)
            predicted = (1.0 - weight) * baseline + weight * predicted
        output = pd.DataFrame({"ml_num": homes["ml_num"].astype(str), "predicted_price": np.round(predicted)})
        path = PACKAGE / "submissions" / f"submission_{track}.csv"
        output.to_csv(path, index=False)
        problems = checker.check_file(path, homes["ml_num"].astype(str))
        if problems:
            raise RuntimeError(f"{track} submission check failed: {problems}")
        results[track] = {"rows": len(output), "sha256": __import__("hashlib").sha256(path.read_bytes()).hexdigest(), "check": "passed"}
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
'''
    (candidate_root / "predict_candidate.py").write_text(predictor_source, encoding="utf-8")
    shutil.copyfile(OUTPUT_ROOT / "comparison.md", candidate_root / "benchmark_comparison.md")
    summary_metrics = benchmark_result["results"]
    readme = ["# Toronto home price model candidate v2", "",
              "This package contains the two LightGBM models selected on the predeclared June, September, and November 2025 development folds, plus the frozen HGBR list-aware model needed for the locked 0.75 challenger blend.", "",
              "The retrospective December 2025–May 2026 benchmark improved off-market MdAPE from "
              f"{summary_metrics['off_market']['baseline_mdape_percent']:.4f}% to {summary_metrics['off_market']['mdape_percent']:.4f}% and list-aware from "
              f"{summary_metrics['list_aware']['baseline_mdape_percent']:.4f}% to {summary_metrics['list_aware']['mdape_percent']:.4f}%. These months have been inspected before and are not a new independent test.", "",
              "Run from the repository root with the pinned dependencies in `environment-lock.txt`:", "",
              "```powershell", "python model_candidate_v2/predict_candidate.py", "```", "",
              "The script reads the competition test files through the repository's `data.py`, writes both CSVs under `submissions/`, and applies the official ID/schema/finite-positive-price checks. `features.py` is copied from the frozen candidate and its hash is recorded in the package manifest.", "",
              "The package uses the competition's synthetic training data only. Model pickle files must be loaded with the Python/scikit-learn/LightGBM versions recorded in the metadata. The six-month benchmark is retrospective; hidden-set performance is unknown.", ""]
    (candidate_root / "README.md").write_text("\n".join(readme), encoding="utf-8")
    shutil.copyfile(OUTPUT_ROOT / "environment-lock.txt", candidate_root / "environment-lock.txt")
    _dump(candidate_root / "winner_config.json", {"config_sha256": json.loads((OUTPUT_ROOT / "winner_config.json").read_text(encoding="utf-8"))["config_sha256"],
                                                  "locked": locked, "source_candidate_commit": load_config()["frozen_commit"],
                                                  "data_sha256": load_config()["data_sha256"]})

    env = os.environ.copy()
    env.setdefault("HF_HOME", str(PROJECT_ROOT / "data" / ".hf_cache"))
    temp_dir = PROJECT_ROOT / "tmp"
    env["TEMP"] = str(temp_dir)
    env["TMP"] = str(temp_dir)
    completed = subprocess.run([sys.executable, str(candidate_root / "predict_candidate.py")], cwd=PROJECT_ROOT,
                               env=env, capture_output=True, text=True, timeout=300, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"portable candidate prediction failed: {completed.stderr or completed.stdout}")
    prediction_check = json.loads(completed.stdout)

    # Verify every generated CSV against the exact cached test IDs in the official checker.
    checker_spec = importlib.util.spec_from_file_location("candidate_submission_checker", PROJECT_ROOT / "release_candidate" / "check_submission.py")
    checker = importlib.util.module_from_spec(checker_spec)
    checker_spec.loader.exec_module(checker)
    for track in ["off_market", "list_aware"]:
        test_key = "test_off_market" if track == "off_market" else "test_list_aware"
        homes = pd.read_parquet(PROJECT_ROOT / load_config()["input_data"][test_key])
        problems = checker.check_file(submissions_dir / f"submission_{track}.csv", homes.ml_num.astype(str))
        if problems:
            raise RuntimeError(f"official submission check failed for {track}: {problems}")

    files = []
    for path in sorted(candidate_root.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS.json" and "__pycache__" not in path.parts and path.suffix != ".pyc":
            files.append({"path": str(path.relative_to(candidate_root)).replace("\\", "/"),
                          "sha256": sha256_file(path), "bytes": path.stat().st_size})
    manifest = {"created_at_utc": _utc_now(), "source_candidate_commit": load_config()["frozen_commit"],
                "data_sha256": load_config()["data_sha256"], "benchmark_config_sha256": benchmark_result["locked_config_sha256"],
                "files": files, "prediction_check": prediction_check,
                "reproducibility": repeat_checks,
                "full_data_fits": {label: saved for label, (_, saved) in export_models.items()},
                "benchmark_upgrade_eligible": benchmark_result["dual_track_upgrade_eligible"]}
    _dump(candidate_root / "SHA256SUMS.json", manifest)
    _dump(OUTPUT_ROOT / "reproducibility.json", {"checks": repeat_checks,
                                                  "full_fit_exports": {label: saved for label, (_, saved) in export_models.items()},
                                                  "prediction_check": prediction_check,
                                                  "fit_attempts": _load_state().get("fit_count_by_stage", {}).get("benchmark", 0),
                                                  "total_fit_attempts": _load_state()["real_data_fits"]})
    state = _load_state()
    state["last_stage"] = "package"
    state["package_completed_at_utc"] = _utc_now()
    _save_state(state)
    print(json.dumps({"status": "candidate_created", "path": str(candidate_root.relative_to(PROJECT_ROOT)),
                      "manifest_files": len(files), "fit_attempts": state["fit_count_by_stage"]["benchmark"],
                      "total_fit_attempts": state["real_data_fits"], "prediction_check": prediction_check,
                      "reproducibility": repeat_checks}, indent=2))
    return manifest


def _render_leaderboard(rows: list[dict]) -> str:
    lines = ["# Development model-family screen", "",
             "Fixed development months: 2025-06, 2025-09, and 2025-11. Scores are selection diagnostics, not independent test results.", "",
             "| Track | Family | Months complete | MdAPE | MAPE | vs fold HGBR MdAPE | Worst month MdAPE | Fit seconds |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in sorted(rows, key=lambda x: (x["track"], x["mdape_percent"])):
        complete = f"{len(row['months'])}/3"
        values = [row["track"], row["model_family"], complete, f"{row['mdape_percent']:.4f}%",
                  f"{row['mape_percent']:.4f}%", f"{row['delta_vs_hgbr_percentage_points']:+.4f} pp",
                  f"{row['worst_month_mdape_percent']:.4f}%", f"{row['fit_seconds']:.1f}"]
        lines.append("| " + " | ".join(values) + " |")
    lines += ["", "These are development-fold results used to pick the next small search. The six-month benchmark is separate and remains locked until the final review.", ""]
    return "\n".join(lines)


def screen() -> dict:
    config = load_config()
    audit_record = audit()
    if audit_record["data"]["sha256"] != config["data_sha256"]:
        raise RuntimeError("data hash changed after audit")
    families = ["hgbr", "catboost", "lightgbm"]
    results = []
    # Fit HGBR first for every held-out month; challenger outputs then pair against it.
    jobs = [(track, month, "hgbr") for track in ["off_market", "list_aware"] for month in config["development_months"]]
    jobs += [(track, month, family) for family in ["catboost", "lightgbm"]
             for track in ["off_market", "list_aware"] for month in config["development_months"]]
    for track, month, family in jobs:
        results.append(run_fit_trial("screen", track, month, family, "log_sold_price",
                                     timeout_seconds=config["fit_timeout_seconds"]["development"]))

    leaderboard = []
    for track in ["off_market", "list_aware"]:
        for family in families:
            frames, result_rows = [], []
            for month in config["development_months"]:
                record = _trial_result(track, month, family)
                if record is None:
                    continue
                frame = pd.read_csv(PROJECT_ROOT / record["prediction_path"], dtype={"ml_num": str})
                frames.append(frame)
                result_rows.append(record)
            if not frames:
                continue
            combined = pd.concat(frames, ignore_index=True)
            actual_score = score(combined.predicted_price, combined.sold_price)
            base_score = score(combined.baseline_price, combined.sold_price)
            by_month = monthly_scores(combined)
            leaderboard.append({"track": track, "model_family": family, "months": sorted(combined.month.unique().tolist()),
                                "mdape_percent": actual_score["mdape_percent"], "mape_percent": actual_score["mape_percent"],
                                "baseline_mdape_percent": base_score["mdape_percent"],
                                "delta_vs_hgbr_percentage_points": actual_score["mdape_percent"] - base_score["mdape_percent"],
                                "relative_mdape_improvement": (base_score["mdape_percent"] - actual_score["mdape_percent"]) / base_score["mdape_percent"],
                                "worst_month_mdape_percent": max(row["mdape_percent"] for row in by_month.values()),
                                "monthly": by_month, "fit_seconds": float(sum(row["fit_seconds"] for row in result_rows)),
                                "result_paths": [str((OUTPUT_ROOT / "trials" / f"{row['trial_id']}.json").relative_to(PROJECT_ROOT)) for row in result_rows]})
    _dump(OUTPUT_ROOT / "screen_leaderboard.json", {"months": config["development_months"], "results": leaderboard,
                                                       "fit_attempts": state_fit_count("screen"),
                                                       "total_fit_attempts": _load_state()["real_data_fits"]})
    (OUTPUT_ROOT / "screen_leaderboard.md").write_text(_render_leaderboard(leaderboard), encoding="utf-8")
    state = _load_state()
    state["last_stage"] = "screen"
    state["screen_completed_at_utc"] = _utc_now()
    _save_state(state)
    print((OUTPUT_ROOT / "screen_leaderboard.md").read_text(encoding="utf-8"))
    return {"results": results, "leaderboard": leaderboard}


def state_fit_count(stage: str) -> int:
    return int(_load_state().get("fit_count_by_stage", {}).get(stage, 0))


def audit() -> dict:
    config = load_config()
    data_path = PROJECT_ROOT / config["input_data"]["listings"]
    expected_data_hash = config["data_sha256"]
    actual_data_hash = sha256_file(data_path)
    if actual_data_hash != expected_data_hash:
        raise RuntimeError(f"training data hash differs: expected {expected_data_hash}, got {actual_data_hash}")

    # Read only training labels and feature metadata; hidden test labels are not read.
    listings = pd.read_parquet(data_path)
    if len(listings) != 150_000 or not (listings.sold_price > 0).all():
        raise RuntimeError("training row count or positive sale-price check failed")
    dates = pd.to_datetime(listings["date_sold"])
    month_counts = {month: int((dates.dt.to_period("M").astype(str) == month).sum())
                    for month in config["development_months"] + config["benchmark_months"]}
    cutoff_audit = {}
    for month in config["development_months"] + config["benchmark_months"]:
        start = pd.Period(month, freq="M").start_time
        train_ids = listings.loc[dates < start, "ml_num"].astype(str)
        eval_ids = listings.loc[dates.dt.to_period("M").astype(str) == month, "ml_num"].astype(str)
        if train_ids.empty or eval_ids.empty:
            raise RuntimeError(f"empty training or evaluation fold for {month}")
        cutoff_audit[month] = {"cutoff_exclusive": start.isoformat(), "train_rows": len(train_ids),
                               "evaluation_rows": len(eval_ids),
                               "train_ids_sha256": hashlib.sha256("\n".join(sorted(train_ids)).encode()).hexdigest(),
                               "evaluation_ids_sha256": hashlib.sha256("\n".join(sorted(eval_ids)).encode()).hexdigest()}

    test_schemas = {}
    for track in ("off_market", "list_aware"):
        test_path = PROJECT_ROOT / config["input_data"][f"test_{track}"]
        test_frame = pd.read_parquet(test_path)
        if "sold_price" in test_frame or "date_sold" in test_frame:
            raise RuntimeError(f"hidden test file unexpectedly contains target or sale date: {track}")
        if track == "off_market" and "list_price" in test_frame:
            raise RuntimeError("off-market input unexpectedly contains list_price")
        if track == "list_aware":
            listed = pd.to_numeric(test_frame["list_price"], errors="coerce").to_numpy(dtype=float)
            if not np.isfinite(listed).all() or (listed <= 0).any():
                raise RuntimeError("list-aware test list_price must be finite and positive")
        test_schemas[track] = {"rows": len(test_frame), "columns": list(test_frame.columns),
                               "id_unique": bool(test_frame.ml_num.is_unique)}

    # Independently recompute published scores from the exact frozen prediction file.
    baseline = pd.read_csv(BASELINE_PATH)
    published = json.loads(BASELINE_METRICS_PATH.read_text(encoding="utf-8"))
    baseline_scores = {}
    for track, rows in baseline.groupby("track", sort=True):
        fresh = score(rows.predicted_price.to_numpy(), rows.sold_price.to_numpy())
        baseline_scores[track] = {**fresh, "monthly": monthly_scores(rows),
                                  "source_rounded": published[track],
                                  "unrounded_mdape_target": config["tracks"][track]["frozen_mdape_percent"],
                                  "unrounded_mape_target": config["tracks"][track]["frozen_mape_percent"]}

    frozen_root = PROJECT_ROOT / "release_candidate"
    source_files = ["features.py", "models.py", "run.py", "data.py", "requirements.txt",
                    "submissions/submission_off_market.csv", "submissions/submission_list_aware.csv",
                    "evidence/validation_predictions_frozen.csv", "evidence/validation_metrics.json"]
    frozen_hashes = {relative: sha256_file(frozen_root / relative) for relative in source_files}
    code_hashes = {relative: sha256_file(PROJECT_ROOT / "experiments" / "model_optimization_v1" / relative)
                   for relative in ["configs.json", "adapters.py", "metrics.py", "run.py"]}

    audit_record = {
        "run_id": config["run_id"], "runner_version": __version__, "audited_at_utc": _utc_now(),
        "git_head": subprocess.check_output(["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "python": sys.version.split()[0], "platform": platform.platform(), "logical_cpus": os.cpu_count(),
        "library_versions": {name: _package_version(name) for name in ["numpy", "pandas", "scikit-learn", "pyarrow", "catboost", "lightgbm", "xgboost"]},
        "data": {"path": str(data_path.relative_to(PROJECT_ROOT)), "sha256": actual_data_hash,
                 "rows": len(listings), "columns": list(listings.columns),
                 "date_sold_min": str(dates.min()), "date_sold_max": str(dates.max()),
                 "month_counts": month_counts, "cutoff_folds": cutoff_audit,
                 "critical_null_counts": listings[["ml_num", "date_sold", "date_listed", "sold_price", "house_type_name"]].isna().sum().to_dict(),
                 "frozen_feature_source_null_counts": listings[["house_area", "bedrooms", "washrooms", "parking_total", "latitude", "longitude", "house_style", "municipality", "community", "list_price"]].isna().sum().to_dict(),
                 "raw_category_cardinalities": {column: int(listings[column].nunique(dropna=True))
                                                for column in ["house_type_name", "house_style", "municipality", "community"]},
                 "sold_price_min": float(listings.sold_price.min()),
                 "sold_price_max": float(listings.sold_price.max()),
                 "sold_price_all_finite_positive": bool(np.isfinite(listings.sold_price.to_numpy(dtype=float)).all() and (listings.sold_price > 0).all())},
        "hidden_inputs": test_schemas,
        "baseline": baseline_scores,
        "frozen_candidate_file_sha256": frozen_hashes,
        "optimization_code_sha256": code_hashes,
        "fit_count": 0, "wallclock_budget_minutes": config["wallclock_cap_minutes"],
        "status": "audited; no model fit run"
    }
    _dump(OUTPUT_ROOT / "audit.json", audit_record)
    _dump(OUTPUT_ROOT / "baseline.json", baseline_scores)
    (OUTPUT_ROOT / "fit_ledger.jsonl").touch(exist_ok=True)
    if not RUN_STATE_PATH.exists():
        _dump(RUN_STATE_PATH, {"status": "active", "started_at_utc": "2026-10-18T20:47:29+00:00",
                               "wallclock_cap_minutes": config["wallclock_cap_minutes"], "real_data_fits": 0,
                               "last_stage": "audit", "last_updated_at_utc": _utc_now()})
    return audit_record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["audit", "pilot", "screen", "tune", "methods", "benchmark", "package"], help="experiment stage")
    args = parser.parse_args()
    if args.stage == "audit":
        record = audit()
        print(json.dumps({"status": record["status"], "data_sha256": record["data"]["sha256"],
                          "baseline": {track: {"mdape": score["mdape_percent"], "mape": score["mape_percent"]}
                                       for track, score in record["baseline"].items()},
                          "output": str(OUTPUT_ROOT)}, indent=2))
    elif args.stage == "pilot":
        pilot()
    elif args.stage == "screen":
        screen()
    elif args.stage == "tune":
        tune()
    elif args.stage == "methods":
        methods()
    elif args.stage == "benchmark":
        benchmark()
    elif args.stage == "package":
        package()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
