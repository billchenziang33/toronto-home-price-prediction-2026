from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

from .audit_months import build_audit, sha256_file
from .candidates import (
    CANDIDATE_ORDER,
    DEVELOPMENT_MONTHS,
    FINAL_MONTHS,
    TRACKS,
    candidate_spec,
    development_gate,
    final_gate,
)
from experiments.model_optimization_v2.evaluation import evaluate_increment
from experiments.model_optimization_v2.pipeline import blend_prices


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parent
PROTOCOL = json.loads((PACKAGE / "protocol.json").read_text(encoding="utf-8"))
OUTPUT = ROOT / "outputs/model_research_fresh"
LEDGER = OUTPUT / "trial_ledger.jsonl"
STATE = OUTPUT / "run_state.json"
DATA_PATH = ROOT / PROTOCOL["source_data"]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    os.replace(temporary, path)


def _append_ledger(record: dict) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, sort_keys=True, default=str) + "\n")


def _code_hashes() -> dict[str, str]:
    paths = [
        PACKAGE / "protocol.json",
        PACKAGE / "candidates.py",
        PACKAGE / "audit_months.py",
        PACKAGE / "worker.py",
        PACKAGE / "runner.py",
        ROOT / "model_candidate_v2/winner_config.json",
        ROOT / "model_candidate_v2/features.py",
        ROOT / "model_candidate_v2/models/frozen_list_aware.json",
        ROOT / "data.py",
        ROOT / "experiments/model_optimization_v2/folds.py",
        ROOT / "experiments/model_optimization_v2/features.py",
        ROOT / "experiments/model_optimization_v2/learners.py",
        ROOT / "experiments/model_optimization_v2/worker.py",
        ROOT / "experiments/model_optimization_v2/pipeline.py",
        ROOT / "experiments/model_optimization_v2/evaluation.py",
    ]
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in paths}


def _runtime_versions() -> dict:
    packages = ("numpy", "pandas", "pyarrow", "scikit-learn", "lightgbm", "catboost")
    result = {name: metadata.version(name) if _has_package(name) else None for name in packages}
    result["python"] = platform.python_version()
    return result


def _has_package(name: str) -> bool:
    try:
        metadata.version(name)
        return True
    except metadata.PackageNotFoundError:
        return False


def _model_configs() -> tuple[dict, dict]:
    winner = json.loads((ROOT / "model_candidate_v2/winner_config.json").read_text(encoding="utf-8"))
    list_hgbr = json.loads((ROOT / "model_candidate_v2/models/frozen_list_aware.json").read_text(encoding="utf-8"))
    configs = {track: winner["locked"][track] for track in TRACKS}
    return configs, list_hgbr


def _baseline_specs(track: str, c0_config: dict, list_hgbr: dict) -> list[tuple[str, dict]]:
    specs = [("C0_LGBM", {
        "model_family": c0_config["model_family"],
        "model_config": c0_config["model_config"],
        "method_id": "C0",
        "target_override": None,
    })]
    if track == "list_aware":
        specs.append(("C0_HGBR", {
            "model_family": list_hgbr["model_family"],
            "model_config": list_hgbr["model_config"],
            "method_id": "C0",
            "target_override": "log_sold_price",
        }))
    return specs


def _component_paths(stage: str, track: str, component_id: str, month: str) -> tuple[Path, Path]:
    stem = f"{component_id}__{month}"
    root = OUTPUT / "predictions" / stage / "components" / track
    return root / f"{stem}.csv", root / f"{stem}.json"


def _cache_component(csv_path: Path, metadata_path: Path, fingerprint: str, data_sha: str,
                     code_hashes: dict) -> tuple[pd.DataFrame | None, dict | None]:
    if not csv_path.exists() or not metadata_path.exists():
        return None, None
    try:
        record = json.loads(metadata_path.read_text(encoding="utf-8"))
        if record.get("job_fingerprint") != fingerprint or record.get("data_sha256") != data_sha:
            return None, None
        if record.get("code_sha256") != code_hashes or record.get("prediction_sha256") != sha256_file(csv_path):
            return None, None
        frame = pd.read_csv(csv_path, dtype={"ml_num": str})
        if frame.duplicated(["ml_num", "month", "track"]).any() or len(frame) != record["evaluation_rows"]:
            return None, None
        return frame, record
    except (OSError, ValueError, KeyError, json.JSONDecodeError, pd.errors.ParserError):
        return None, None


def _counted_fits() -> int:
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
        if record.get("status") == "started" and record.get("counted_fit") is True:
            count += 1
    return count


def _elapsed_seconds() -> float:
    if not STATE.exists():
        return 0.0
    state = json.loads(STATE.read_text(encoding="utf-8"))
    started = datetime.fromisoformat(state["started_at_utc"])
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    return max(0.0, (datetime.now(timezone.utc) - started).total_seconds())


def _ensure_state() -> dict:
    code_hashes = _code_hashes()
    if STATE.exists():
        state = json.loads(STATE.read_text(encoding="utf-8"))
        if state.get("code_sha256") != code_hashes:
            raise ValueError("research code changed after run start; preserve the old ledger and start a new run ID")
        if state.get("data_sha256") != PROTOCOL["data_sha256"]:
            raise ValueError("research data hash differs from the frozen run state")
    else:
        state = {
            "run_id": PROTOCOL["run_id"],
            "started_at_utc": _utc_now(),
            "data_sha256": PROTOCOL["data_sha256"],
            "code_sha256": code_hashes,
            "runtime_versions": _runtime_versions(),
            "fit_cap": PROTOCOL["budget"]["fit_cap"],
            "wallclock_minutes": PROTOCOL["budget"]["wallclock_minutes"],
            "development_minutes": PROTOCOL["budget"]["development_minutes"],
        }
    _write_json(STATE, state)
    return state


def _within_budget(stage: str) -> bool:
    budget = PROTOCOL["budget"]
    elapsed_minutes = _elapsed_seconds() / 60.0
    limit = budget["development_minutes"] if stage == "development" else budget["wallclock_minutes"]
    return _counted_fits() < budget["fit_cap"] and elapsed_minutes < limit


def _run_component(stage: str, month: str, track: str, component_id: str, spec: dict,
                   timeout_seconds: int) -> tuple[pd.DataFrame | None, dict | None]:
    state = _ensure_state()
    code_hashes = state["code_sha256"]
    data_sha = PROTOCOL["data_sha256"]
    relative_csv = Path("outputs/model_research_fresh") / "predictions" / stage / "components" / track / f"{component_id}__{month}.csv"
    csv_path, metadata_path = _component_paths(stage, track, component_id, month)
    fingerprint_payload = {
        "stage": stage, "month": month, "track": track, "component_id": component_id,
        "spec": spec, "data_sha256": data_sha, "code_sha256": code_hashes,
        "runtime_versions": state["runtime_versions"], "threads": 4,
    }
    fingerprint = _sha256_text(_canonical(fingerprint_payload))
    cached_frame, cached_meta = _cache_component(csv_path, metadata_path, fingerprint, data_sha, code_hashes)
    if cached_frame is not None:
        cached_meta["cache_hit"] = True
        _append_ledger({
            "timestamp_utc": _utc_now(), "stage": stage, "month": month, "track": track,
            "component_id": component_id, "status": "cached", "counted_fit": False,
            "job_fingerprint": fingerprint, "prediction_path": str(csv_path.relative_to(ROOT)),
            "prediction_sha256": cached_meta["prediction_sha256"],
        })
        return cached_frame, cached_meta

    if not _within_budget(stage):
        _append_ledger({"timestamp_utc": _utc_now(), "stage": stage, "month": month,
                        "track": track, "component_id": component_id, "status": "not_started_budget_cap",
                        "counted_fit": False, "elapsed_seconds": _elapsed_seconds(),
                        "fit_attempts": _counted_fits()})
        return None, None

    attempt = _counted_fits() + 1
    started_record = {
        "timestamp_utc": _utc_now(), "stage": stage, "month": month, "track": track,
        "component_id": component_id, "attempt": attempt, "status": "started",
        "counted_fit": True, "job_fingerprint": fingerprint, "data_sha256": data_sha,
        "code_sha256": code_hashes, "runtime_versions": state["runtime_versions"],
        "model_family": spec["model_family"], "model_config": spec["model_config"],
        "method_id": spec["method_id"], "target_override": spec.get("target_override"),
        "threads": 4, "timeout_seconds": timeout_seconds,
        "prediction_path": str(csv_path.relative_to(ROOT)),
    }
    _append_ledger(started_record)

    job = {
        "stage": stage, "month": month, "track": track, "component_id": component_id,
        "spec": spec, "data_path": PROTOCOL["source_data"],
        "prediction_path": str(relative_csv), "job_fingerprint": fingerprint,
        "data_sha256": data_sha, "code_sha256": code_hashes,
    }
    job_path = OUTPUT / "jobs" / f"{fingerprint}.json"
    _write_json(job_path, job)
    env = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = "4"
    command = [sys.executable, "-m", "experiments.model_research_fresh.worker", str(job_path)]
    started = time.perf_counter()
    try:
        process = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True,
                                 timeout=timeout_seconds, check=False)
    except subprocess.TimeoutExpired as exc:
        _append_ledger({**started_record, "timestamp_utc": _utc_now(), "status": "timed_out",
                        "elapsed_seconds": time.perf_counter() - started,
                        "error": f"fit exceeded {timeout_seconds}s", "counted_fit": True})
        return None, None
    elapsed = time.perf_counter() - started
    log_dir = OUTPUT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"{fingerprint}.stdout.txt").write_text(process.stdout, encoding="utf-8")
    (log_dir / f"{fingerprint}.stderr.txt").write_text(process.stderr, encoding="utf-8")
    if process.returncode != 0 or not metadata_path.exists() or not csv_path.exists():
        _append_ledger({**started_record, "timestamp_utc": _utc_now(), "status": "failed",
                        "elapsed_seconds": elapsed, "returncode": process.returncode,
                        "error": (process.stderr or process.stdout)[-6000:], "counted_fit": True})
        return None, None

    record = json.loads(metadata_path.read_text(encoding="utf-8"))
    if record.get("job_fingerprint") != fingerprint or record.get("prediction_sha256") != sha256_file(csv_path):
        _append_ledger({**started_record, "timestamp_utc": _utc_now(), "status": "failed",
                        "elapsed_seconds": elapsed, "error": "worker artifact fingerprint/hash mismatch",
                        "counted_fit": True})
        return None, None
    _append_ledger({**started_record, **record, "timestamp_utc": _utc_now(), "status": "completed",
                    "elapsed_seconds": elapsed, "counted_fit": True,
                    "prediction_path": str(csv_path.relative_to(ROOT))})
    return pd.read_csv(csv_path, dtype={"ml_num": str}), record


def _pipeline_frame(stage: str, track: str, month: str, pipeline_id: str,
                    candidate_component: str, blend_component: str | None = None) -> pd.DataFrame:
    challenger_path, _ = _component_paths(stage, track, candidate_component, month)
    challenger = pd.read_csv(challenger_path, dtype={"ml_num": str})
    if blend_component is None:
        prediction = np.round(challenger.raw_prediction.to_numpy(dtype=float))
    else:
        baseline_path, _ = _component_paths(stage, track, blend_component, month)
        frozen = pd.read_csv(baseline_path, dtype={"ml_num": str})
        if not np.array_equal(challenger.ml_num.to_numpy(), frozen.ml_num.to_numpy()):
            raise ValueError("challenger and same-fold HGBR IDs are not aligned")
        if not np.array_equal(challenger.sold_price.to_numpy(), frozen.sold_price.to_numpy()):
            raise ValueError("challenger and same-fold HGBR targets are not aligned")
        prediction = blend_prices(challenger.raw_prediction, frozen.raw_prediction, 0.75)
    if not np.isfinite(prediction).all() or (prediction <= 0).any():
        raise ValueError("pipeline returned invalid rounded predictions")
    frame = challenger[["ml_num", "track", "month", "sold_price"]].copy()
    frame["predicted_price"] = prediction
    frame = frame[["ml_num", "track", "month", "sold_price", "predicted_price"]]
    path = OUTPUT / "predictions" / stage / "pipelines" / track / f"{pipeline_id}__{month}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def _fold_jobs(stage: str, months: tuple[str, ...], selected: dict | None = None) -> dict:
    model_configs, list_hgbr = _model_configs()
    frames: dict[tuple[str, str, str], pd.DataFrame] = {}
    statuses: dict[str, list[str]] = {"completed": [], "cached": [], "failed": [], "skipped": []}
    timeout = 180 if stage == "development" else 300

    for month in months:
        for track in TRACKS:
            baseline_specs = _baseline_specs(track, model_configs[track], list_hgbr)
            component_results = {}
            for component_id, spec in baseline_specs:
                frame, meta = _run_component(stage, month, track, component_id, spec, timeout)
                component_results[component_id] = frame
                if frame is None:
                    statuses["failed"].append(f"{component_id}/{track}/{month}")
                elif meta:
                    statuses["cached" if meta.get("cache_hit") else "completed"].append(f"{component_id}/{track}/{month}")
            if component_results.get("C0_LGBM") is None:
                statuses["skipped"].append(f"C0/{track}/{month}: baseline unavailable")
                continue
            if track == "list_aware" and component_results.get("C0_HGBR") is None:
                statuses["skipped"].append(f"C0/{track}/{month}: HGBR blend unavailable")
                continue
            c0_frame = _pipeline_frame(stage, track, month, "C0", "C0_LGBM",
                                       "C0_HGBR" if track == "list_aware" else None)
            frames[(track, "C0", month)] = c0_frame
            if selected is None:
                candidate_ids = CANDIDATE_ORDER
            else:
                candidate_ids = (selected[track]["candidate_id"],)
            for candidate_id in candidate_ids:
                spec = candidate_spec(candidate_id, track, model_configs[track]["model_config"])
                component_id = f"candidate_{candidate_id}"
                frame, meta = _run_component(stage, month, track, component_id, spec, timeout)
                if frame is None:
                    statuses["failed"].append(f"{candidate_id}/{track}/{month}")
                    continue
                statuses["cached" if meta and meta.get("cache_hit") else "completed"].append(f"{candidate_id}/{track}/{month}")
                pipeline_id = candidate_id
                candidate_frame = _pipeline_frame(stage, track, month, pipeline_id, component_id,
                                                  "C0_HGBR" if track == "list_aware" else None)
                frames[(track, candidate_id, month)] = candidate_frame
    return {"frames": frames, "statuses": statuses}


def _evaluate(stage: str, months: tuple[str, ...], frames: dict,
              candidate_ids_by_track: dict[str, tuple[str, ...]]) -> dict:
    result: dict[str, dict] = {}
    repeats = PROTOCOL["budget"]["bootstrap_repeats"]
    for track in TRACKS:
        parent_parts = [frames[(track, "C0", month)] for month in months if (track, "C0", month) in frames]
        if len(parent_parts) != len(months):
            result[track] = {"baseline_available": False, "candidates": {}}
            continue
        parent = pd.concat(parent_parts, ignore_index=True)
        candidate_results = {}
        for candidate_index, candidate_id in enumerate(candidate_ids_by_track[track]):
            parts = [frames[(track, candidate_id, month)] for month in months
                     if (track, candidate_id, month) in frames]
            if len(parts) != len(months):
                candidate_results[candidate_id] = {"available": False, "months_available": len(parts)}
                continue
            candidate = pd.concat(parts, ignore_index=True)
            report = evaluate_increment(parent, candidate, repeats=repeats,
                                        seed=4000 + candidate_index + (100 if track == "list_aware" else 0))
            gate = development_gate(report) if stage == "development" else final_gate(report)
            report["gate"] = gate
            report["available"] = True
            report["monthly_scores"] = {}
            for month in months:
                p_month = parent.loc[parent.month.astype(str).eq(month)]
                c_month = candidate.loc[candidate.month.astype(str).eq(month)]
                if p_month.empty or c_month.empty:
                    continue
                actual = p_month.sold_price.to_numpy(dtype=float)
                p_ape = 100.0 * np.abs(p_month.predicted_price.to_numpy(dtype=float) - actual) / actual
                c_ape = 100.0 * np.abs(c_month.predicted_price.to_numpy(dtype=float) - actual) / actual
                report["monthly_scores"][month] = {
                    "n": len(actual),
                    "parent_mdape_percent": float(np.median(p_ape)),
                    "candidate_mdape_percent": float(np.median(c_ape)),
                    "parent_mape_percent": float(np.mean(p_ape)),
                    "candidate_mape_percent": float(np.mean(c_ape)),
                    "relative_mdape_improvement": report["monthly_relative_improvement"][month],
                }
            candidate_results[candidate_id] = report
        result[track] = {
            "baseline_available": True,
            "baseline": {
                "n": len(parent),
                "mdape_percent": float(np.median(100.0 * np.abs(parent.predicted_price - parent.sold_price) / parent.sold_price)),
                "mape_percent": float(np.mean(100.0 * np.abs(parent.predicted_price - parent.sold_price) / parent.sold_price)),
            },
            "candidates": candidate_results,
        }
    return result


def _freeze_development(results: dict) -> dict:
    model_configs, _ = _model_configs()
    selected = {}
    reasons = {}
    for track in TRACKS:
        passing = []
        for candidate_id, report in results.get(track, {}).get("candidates", {}).items():
            if report.get("available") and report.get("gate", {}).get("passed"):
                candidate_mdape = report["candidate"]["mdape_percent"]
                candidate_mape = report["candidate"]["mape_percent"]
                passing.append((candidate_mdape, candidate_mape, CANDIDATE_ORDER.index(candidate_id), candidate_id, report))
        if passing:
            _, _, _, candidate_id, report = min(passing)
            selected[track] = {
                "candidate_id": candidate_id,
                "spec": candidate_spec(candidate_id, track, model_configs[track]["model_config"]),
                "development_report_sha256": _sha256_text(_canonical(report)),
            }
            reasons[track] = f"selected {candidate_id} among development gate passers by pooled MdAPE, then MAPE, then frozen order"
        else:
            selected[track] = None
            reasons[track] = "no candidate passed every preregistered development gate"
    unlocked = all(selected[track] is not None for track in TRACKS)
    freeze = {
        "created_at_utc": _utc_now(),
        "data_sha256": PROTOCOL["data_sha256"],
        "code_sha256": _code_hashes(),
        "runtime_versions": _runtime_versions(),
        "development_months": list(DEVELOPMENT_MONTHS),
        "selected": selected,
        "track_reasons": reasons,
        "both_tracks_qualified": unlocked,
        "final_months_unlocked": unlocked,
        "selection_changed_after_final": False,
    }
    _write_json(OUTPUT / "freeze_manifest.json", freeze)
    return freeze


def run_development() -> dict:
    _ensure_state()
    audit = build_audit(final_stage=False)
    _write_json(OUTPUT / "month_audit.json", audit)
    expected = {"2024-10": 2451, "2024-12": 1345, "2025-01": 1339, "2025-03": 1931}
    actual = {month: audit["fold_counts"][month]["evaluation_rows"] for month in DEVELOPMENT_MONTHS}
    if actual != expected:
        raise ValueError(f"development fold counts differ from frozen preflight: {actual}")
    jobs = _fold_jobs("development", DEVELOPMENT_MONTHS)
    results = _evaluate("development", DEVELOPMENT_MONTHS, jobs["frames"],
                        {track: CANDIDATE_ORDER for track in TRACKS})
    development_record = {
        "created_at_utc": _utc_now(), "data_sha256": PROTOCOL["data_sha256"],
        "code_sha256": _code_hashes(), "runtime_versions": _runtime_versions(),
        "development_months": list(DEVELOPMENT_MONTHS),
        "candidate_order": list(CANDIDATE_ORDER), "results": results,
        "fit_statuses": jobs["statuses"], "counted_fit_attempts": _counted_fits(),
    }
    _write_json(OUTPUT / "development_results.json", development_record)
    freeze = _freeze_development(results)
    state = _ensure_state()
    state["development_finished_at_utc"] = _utc_now()
    state["both_tracks_qualified"] = freeze["both_tracks_qualified"]
    state["fit_attempts_after_development"] = _counted_fits()
    _write_json(STATE, state)
    return {"development": development_record, "freeze": freeze}


def run_final() -> dict:
    freeze_path = OUTPUT / "freeze_manifest.json"
    if not freeze_path.exists():
        raise ValueError("final stage is locked until the development freeze manifest exists")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if not freeze.get("final_months_unlocked") or not freeze.get("both_tracks_qualified"):
        raise ValueError("final labels remain locked because both tracks did not qualify on development")
    if (OUTPUT / "final_results.json").exists():
        raise ValueError("final stage is complete; the frozen candidate cannot be evaluated a second time")
    if _elapsed_seconds() >= PROTOCOL["budget"]["development_minutes"] * 60:
        raise ValueError("the reserved 60-minute final window has expired; final labels remain unopened")
    if not freeze.get("final_started_at_utc"):
        freeze["final_started_at_utc"] = _utc_now()
        _write_json(freeze_path, freeze)

    audit_path = OUTPUT / "month_audit_final.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if not audit.get("final_stage") or audit.get("data_sha256") != PROTOCOL["data_sha256"]:
            raise ValueError("cached final month audit does not match this frozen run")
    else:
        audit = build_audit(final_stage=True)
        _write_json(audit_path, audit)
    selected = freeze["selected"]
    jobs = _fold_jobs("final", FINAL_MONTHS, selected=selected)
    candidate_by_track = {track: (selected[track]["candidate_id"],) for track in TRACKS}
    results = _evaluate("final", FINAL_MONTHS, jobs["frames"], candidate_by_track)
    track_gates = {
        track: next(iter(results.get(track, {}).get("candidates", {}).values()), {}).get("gate", {"passed": False})
        for track in TRACKS
    }
    accepted = all(track_gates[track].get("passed") for track in TRACKS)
    final_record = {
        "created_at_utc": _utc_now(), "data_sha256": PROTOCOL["data_sha256"],
        "code_sha256": _code_hashes(), "runtime_versions": _runtime_versions(),
        "final_months": list(FINAL_MONTHS), "selected": selected,
        "results": results, "track_gates": track_gates,
        "both_tracks_passed": accepted, "v3_export_authorized": accepted,
        "fit_statuses": jobs["statuses"], "counted_fit_attempts": _counted_fits(),
        "selection_changed_after_final": False,
    }
    _write_json(OUTPUT / "final_results.json", final_record)
    freeze["final_completed_at_utc"] = _utc_now()
    _write_json(freeze_path, freeze)
    state = _ensure_state()
    state["final_finished_at_utc"] = _utc_now()
    state["both_tracks_passed_final"] = accepted
    state["fit_attempts_after_final"] = _counted_fits()
    _write_json(STATE, state)
    return final_record


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run the pre-registered fresh-fold model research.")
    parser.add_argument("--stage", choices=("development", "final", "all"), default="all")
    args = parser.parse_args()
    result = {}
    if args.stage in {"development", "all"}:
        result["development"] = run_development()
        if args.stage == "all" and result["development"]["freeze"]["final_months_unlocked"]:
            result["final"] = run_final()
    elif args.stage == "final":
        result["final"] = run_final()
    print(json.dumps({
        "completed_at_utc": _utc_now(), "fit_attempts": _counted_fits(),
        "elapsed_seconds": _elapsed_seconds(), "result": result,
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
