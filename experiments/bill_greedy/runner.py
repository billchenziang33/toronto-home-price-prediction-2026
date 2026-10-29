from __future__ import annotations

import argparse
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
PACKAGE = ROOT / "experiments/bill_greedy"
OUTPUT = ROOT / "outputs/bill_greedy"
PARENT_OUTPUT = OUTPUT
PROTOCOL_PATH = PACKAGE / "protocol.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
DATA = ROOT / PROTOCOL["data_path"]
LEDGER = OUTPUT / "fit_ledger.jsonl"
STATE = OUTPUT / "run_state.json"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _code_hashes() -> dict[str, str]:
    paths = [PROTOCOL_PATH, PACKAGE / "history.py", PACKAGE / "comparables.py", PACKAGE / "runner.py",
             PACKAGE / "round20.py", PACKAGE / "round21.py", PACKAGE / "round22.py", PACKAGE / "round23.py",
             PACKAGE / "round24.py", PACKAGE / "round25.py",
             ROOT / "model_candidate_v2/features.py", ROOT / "model_candidate_v2/winner_config.json",
             ROOT / "model_candidate_v2/models/frozen_list_aware.json",
             ROOT / "experiments/model_optimization_v2/worker.py",
             ROOT / "experiments/model_optimization_v2/learners.py",
             ROOT / "experiments/model_optimization_v2/pipeline.py",
             PACKAGE / "aligned_inference.py"]
    return {str(path.relative_to(ROOT)).replace("\\", "/"): _sha(path) for path in paths}


def _append(record: dict) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, sort_keys=True, default=str) + "\n")


def _state() -> dict:
    records = [json.loads(line) for line in LEDGER.read_text(encoding="utf-8").splitlines()] if LEDGER.exists() else []
    started = sum(row.get("event") == "fit_start" for row in records)
    old = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}
    started_at = old.get("started_at_utc", datetime.now(timezone.utc).isoformat())
    elapsed = max(0.0, (datetime.now(timezone.utc) - datetime.fromisoformat(started_at)).total_seconds())
    computed = {"run_id": PROTOCOL["run_id"], "started_at_utc": started_at,
            "data_sha256": PROTOCOL["data_sha256"], "code_sha256": _code_hashes(),
            "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                        "pandas": pd.__version__, "lightgbm": metadata.version("lightgbm"),
                        "catboost": metadata.version("catboost"),
                        "scipy": metadata.version("scipy"),
                        "scikit_learn": metadata.version("scikit-learn")},
            "fit_attempts": started, "elapsed_minutes": elapsed / 60,
            "fit_cap": PROTOCOL["limits"]["fits"], "wallclock_cap_minutes": PROTOCOL["limits"]["minutes"]}
    state = {**old, **computed}
    state.setdefault("active_stage", "g01_development")
    return state


def _write_state(updates: dict | None = None) -> None:
    state = _state()
    if updates:
        state.update(updates)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(tmp, STATE)


def _round21_protocol_for_run(source_protocol: dict, run_id: str) -> dict:
    """Copy the frozen R21 protocol into the separately authorized 32-fit run."""
    import copy

    if not run_id or any(char in run_id for char in "\\/:"):
        raise ValueError("run_id must be a non-empty path-safe identifier")
    protocol = copy.deepcopy(source_protocol)
    rounds = protocol.get("candidate_search", {}).get("rounds", [])
    if not any(item.get("round") == 21 for item in rounds):
        raise RuntimeError("Round 21 must be preregistered before creating its run")
    protocol["run_id"] = run_id
    protocol["limits"].update({"minutes": 180, "development_minutes": 180,
                               "final_reserve_minutes": 0, "fits": 32,
                               "final_reserve_fits": 0, "batch_minutes": 90,
                               "batch_fits": 32, "threads": 4})
    protocol["round21_run_budget"] = {"authorized_total_fits": 32,
                                      "expected_primary_fits_max": 24,
                                      "retry_allowance_fits": 8,
                                      "authorized_minutes": 180,
                                      "reserved_months_remain_sealed": True}
    return protocol


def initialize_round21_run(run_id: str) -> dict:
    """Create a fresh R21 run folder while leaving the expired study untouched."""
    source_state = _state()
    if source_state.get("active_stage") != "round_20_complete" or source_state.get("active_batch") is not None:
        raise RuntimeError("a separate Round 21 run requires the closed Round 20 checkpoint")
    expected_parents = {"off_market": "G19_detached_shrink_075", "list_aware": "G01"}
    if source_state.get("current_parents") != expected_parents:
        raise RuntimeError(f"Round 21 requires frozen parents {expected_parents}")
    source_audit = _source_commit_and_hash()
    source_protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    run_protocol = _round21_protocol_for_run(source_protocol, run_id)
    run_protocol["source_run_id"] = source_state.get("run_id")
    run_root = OUTPUT / "runs" / run_id
    run_root.parent.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(exist_ok=False)
    protocol_path = run_root / "protocol.json"
    protocol_temp = protocol_path.with_suffix(".json.tmp")
    protocol_temp.write_text(json.dumps(run_protocol, indent=2), encoding="utf-8")
    os.replace(protocol_temp, protocol_path)
    run_output = run_root / "outputs"
    run_output.mkdir()
    created_at = datetime.now(timezone.utc).isoformat()
    manifest = {"run_id": run_id, "created_at_utc": created_at,
                "source_run_id": source_state.get("run_id"),
                "source_stage": source_state.get("active_stage"),
                "source_fit_attempts": source_state.get("fit_attempts"),
                "source_elapsed_minutes": source_state.get("elapsed_minutes"),
                "source_output": str(OUTPUT), "parent_commit": source_audit["commit"],
                "data_sha256": source_audit["data_sha256"],
                "source_code_sha256": source_audit["code_sha256"],
                "limits": run_protocol["limits"],
                "round21_run_budget": run_protocol["round21_run_budget"]}
    manifest_path = run_root / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"run_id": run_id, "run_root": run_root, "protocol_path": protocol_path,
            "run_output": run_output, "parent_output": OUTPUT, "protocol": run_protocol,
            "manifest_path": manifest_path}


def activate_round21_run(context: dict, *, initialize: bool = True,
                         allow_closed_checkpoint: bool = False) -> None:
    """Switch this process to the isolated paths and initialize its own ledger/state."""
    global PROTOCOL_PATH, PROTOCOL, DATA, OUTPUT, PARENT_OUTPUT, LEDGER, STATE
    PROTOCOL_PATH = Path(context["protocol_path"])
    PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    DATA = ROOT / PROTOCOL["data_path"]
    PARENT_OUTPUT = Path(context["parent_output"])
    OUTPUT = Path(context["run_output"])
    LEDGER = OUTPUT / "fit_ledger.jsonl"
    STATE = OUTPUT / "run_state.json"
    if initialize:
        _write_state({"run_id": PROTOCOL["run_id"], "active_stage": "round_20_complete",
                      "batch_id": 0, "active_batch": None,
                      "current_parents": {"off_market": "G19_detached_shrink_075", "list_aware": "G01"},
                      "source_run_id": context["protocol"].get("source_run_id")})
        _append({"event": "run_start", "run_id": PROTOCOL["run_id"],
                 "at_utc": datetime.now(timezone.utc).isoformat(),
                 "parent_commit": PROTOCOL["parent_commit"], "data_sha256": PROTOCOL["data_sha256"],
                 "code_sha256": _code_hashes(), "limits": PROTOCOL["limits"]})
    else:
        if not STATE.is_file() or not LEDGER.is_file():
            raise RuntimeError("cannot resume an isolated Round 21 run without its state and ledger")
        state = _state()
        valid_stage = state.get("active_stage") in {"batch_started", "round_21_running"}
        active_batch_required = state.get("active_batch") is not None or allow_closed_checkpoint
        if state.get("run_id") != PROTOCOL["run_id"] or not active_batch_required or not valid_stage:
            raise RuntimeError("isolated Round 21 run is not at a resumable batch checkpoint")


def _load_round21_run_context(run_id: str) -> dict:
    if not run_id or any(char in run_id for char in "\\/:"):
        raise ValueError("run_id must be a non-empty path-safe identifier")
    run_root = OUTPUT / "runs" / run_id
    protocol_path = run_root / "protocol.json"
    manifest_path = run_root / "run_manifest.json"
    if not protocol_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"isolated Round 21 run not found: {run_root}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if protocol.get("run_id") != run_id or manifest.get("run_id") != run_id:
        raise RuntimeError("isolated Round 21 run ID does not match its manifest")
    return {"run_id": run_id, "run_root": run_root, "protocol_path": protocol_path,
            "run_output": run_root / "outputs", "parent_output": Path(manifest["source_output"]),
            "protocol": protocol, "manifest_path": manifest_path, "manifest": manifest}


def run_new_round21() -> dict:
    """Start the authorized isolated R21 run and execute its preregistered gates."""
    run_id = f"bill_greedy_round21_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    context = initialize_round21_run(run_id)
    activate_round21_run(context)
    start_batch()
    result = __import__("experiments.bill_greedy.round21", fromlist=["run_round_21"]).run_round_21()
    result["run_id"] = run_id
    result["run_root"] = str(context["run_root"])
    manifest = json.loads(context["manifest_path"].read_text(encoding="utf-8"))
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["result_summary"] = {"accepted": result["accepted"], "fit_attempts": result["fit_attempts"]}
    context["manifest_path"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return result


def resume_round21(run_id: str) -> dict:
    """Resume the one isolated R21 run without resetting its clock or fit ledger."""
    context = _load_round21_run_context(run_id)
    activate_round21_run(context, initialize=False)
    audit = _source_commit_and_hash()
    manifest = context["manifest"]
    if (audit["commit"] != manifest["parent_commit"]
            or audit["data_sha256"] != manifest["data_sha256"]):
        raise RuntimeError("resume stopped: source commit or data hash differs from the isolated run")
    rows = _ledger_rows()
    fit_starts = [row for row in rows if row.get("event") == "fit_start"]
    current_hashes = _code_hashes()
    if fit_starts:
        recorded_hashes = _round21_resume_code_hashes(rows)
        if recorded_hashes != current_hashes:
            if not _round21_unscored_code_update_safe(rows):
                raise RuntimeError("resume stopped: code hash changed after a scored or persisted candidate")
            _append({"event": "code_update_after_unscored_fit", "run_id": run_id,
                     "at_utc": datetime.now(timezone.utc).isoformat(),
                     "previous_code_sha256": recorded_hashes,
                     "current_code_sha256": current_hashes,
                     "fit_attempts": len(fit_starts),
                     "reason": "Round 21 stopped after a completed fit but before candidate persistence or scoring; "
                               "the fit remains counted and the retry uses a unique fit ID"})
    else:
        last_code_hashes = next((row.get("code_sha256") for row in reversed(rows)
                                 if row.get("code_sha256")), {})
        if last_code_hashes != current_hashes:
            _append({"event": "code_update_before_first_fit", "run_id": run_id,
                     "at_utc": datetime.now(timezone.utc).isoformat(),
                     "previous_code_sha256": last_code_hashes,
                     "current_code_sha256": current_hashes,
                     "fit_attempts": 0,
                     "reason": "CLI runner alias fix after setup exited before any model fit"})
    result = __import__("experiments.bill_greedy.round21", fromlist=["run_round_21"]).run_round_21()
    result["run_id"] = run_id
    result["run_root"] = str(context["run_root"])
    manifest["last_resumed_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["result_summary"] = {"accepted": result["accepted"], "fit_attempts": result["fit_attempts"]}
    context["manifest_path"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return result


def start_round21_batch(run_id: str) -> dict:
    """Activate a closed, still-budgeted R21 checkpoint and open its next bounded batch."""
    context = _load_round21_run_context(run_id)
    activate_round21_run(context, initialize=False, allow_closed_checkpoint=True)
    state = _state()
    if state.get("active_stage") != "round_21_running" or state.get("active_batch") is not None:
        raise RuntimeError("a new Round 21 batch requires a closed round_21_running checkpoint")
    return start_batch()


def checkpoint_round21_batch(run_id: str) -> dict:
    """Close the active R21 batch while keeping its isolated state resumable."""
    context = _load_round21_run_context(run_id)
    activate_round21_run(context, initialize=False, allow_closed_checkpoint=True)
    state = _state()
    if state.get("active_stage") not in {"batch_started", "round_21_running"} or state.get("active_batch") is None:
        raise RuntimeError("Round 21 has no open batch to checkpoint")
    return checkpoint_batch()


def _round22_protocol_for_run(source_protocol: dict, run_id: str) -> dict:
    """Copy the preregistered R22 candidates into the newly authorized run budget."""
    import copy

    if not run_id or any(char in run_id for char in "\\/:"):
        raise ValueError("run_id must be a non-empty path-safe identifier")
    if not any(item.get("round") == 22 for item in source_protocol.get("candidate_search", {}).get("rounds", [])):
        raise RuntimeError("Round 22 candidates must be preregistered before creating the run")
    protocol = copy.deepcopy(source_protocol)
    protocol["run_id"] = run_id
    protocol["limits"].update({"minutes": 177, "development_minutes": 177,
                               "final_reserve_minutes": 0, "fits": 28,
                               "final_reserve_fits": 0, "batch_minutes": 90,
                               "batch_fits": 28, "threads": 4})
    protocol["round22_run_budget"] = {
        "authorized_total_fits": 28,
        "expected_primary_fits_max": 24,
        "retry_allowance_fits": 4,
        "authorized_minutes": 177,
        "batch_minutes": 90,
        "reserved_months_remain_sealed": True,
    }
    return protocol


def initialize_round22_run(run_id: str, source_run_id: str) -> dict:
    """Create R22 state only from the completed R21 checkpoint, leaving it untouched."""
    source_context = _load_round21_run_context(source_run_id)
    source_state = json.loads((source_context["run_output"] / "run_state.json").read_text(encoding="utf-8"))
    source_result_path = source_context["run_output"] / "round_21_results.json"
    if not source_result_path.is_file():
        raise RuntimeError("Round 22 requires a completed Round 21 result artifact")
    source_result = json.loads(source_result_path.read_text(encoding="utf-8"))
    expected_parents = {"off_market": "G19_detached_shrink_075", "list_aware": "G01"}
    if (source_state.get("active_stage") != "round_21_complete"
            or source_state.get("active_batch") is not None
            or source_state.get("current_parents") != expected_parents
            or source_result.get("accepted") != {}
            or source_result.get("current_parents") != expected_parents):
        raise RuntimeError("Round 22 requires the completed R21 checkpoint with unchanged G19/G01 parents")
    source_audit = _source_commit_and_hash()
    source_protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    run_protocol = _round22_protocol_for_run(source_protocol, run_id)
    run_protocol["source_run_id"] = source_run_id
    run_root = OUTPUT / "runs" / run_id
    run_root.parent.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(exist_ok=False)
    protocol_path = run_root / "protocol.json"
    temp = protocol_path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(run_protocol, indent=2), encoding="utf-8")
    os.replace(temp, protocol_path)
    run_output = run_root / "outputs"
    run_output.mkdir()
    created_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "run_id": run_id, "created_at_utc": created_at,
        "source_run_id": source_run_id, "source_stage": source_state.get("active_stage"),
        "source_fit_attempts": source_state.get("fit_attempts"),
        "source_elapsed_minutes": source_state.get("elapsed_minutes"),
        "source_output": str(OUTPUT), "parent_commit": source_audit["commit"],
        "data_sha256": source_audit["data_sha256"], "source_code_sha256": source_audit["code_sha256"],
        "limits": run_protocol["limits"], "round22_run_budget": run_protocol["round22_run_budget"],
    }
    manifest_path = run_root / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"run_id": run_id, "run_root": run_root, "protocol_path": protocol_path,
            "run_output": run_output, "parent_output": OUTPUT, "protocol": run_protocol,
            "manifest_path": manifest_path}


def activate_round22_run(context: dict, *, initialize: bool = True,
                         allow_closed_checkpoint: bool = False) -> None:
    """Switch the runner to a fresh isolated Round 22 state and ledger."""
    global PROTOCOL_PATH, PROTOCOL, DATA, OUTPUT, PARENT_OUTPUT, LEDGER, STATE
    PROTOCOL_PATH = Path(context["protocol_path"])
    PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    DATA = ROOT / PROTOCOL["data_path"]
    PARENT_OUTPUT = Path(context["parent_output"])
    OUTPUT = Path(context["run_output"])
    LEDGER = OUTPUT / "fit_ledger.jsonl"
    STATE = OUTPUT / "run_state.json"
    if initialize:
        parents = {"off_market": "G19_detached_shrink_075", "list_aware": "G01"}
        _write_state({"run_id": PROTOCOL["run_id"], "active_stage": "round_21_complete",
                      "batch_id": 0, "active_batch": None, "current_parents": parents,
                      "source_run_id": PROTOCOL.get("source_run_id")})
        _append({"event": "run_start", "run_id": PROTOCOL["run_id"],
                 "at_utc": datetime.now(timezone.utc).isoformat(),
                 "parent_commit": PROTOCOL["parent_commit"], "data_sha256": PROTOCOL["data_sha256"],
                 "code_sha256": _code_hashes(), "limits": PROTOCOL["limits"]})
        return
    if not STATE.is_file() or not LEDGER.is_file():
        raise RuntimeError("cannot resume an isolated Round 22 run without its state and ledger")
    state = _state()
    valid_stage = state.get("active_stage") in {"batch_started", "round_22_running"}
    active_batch_required = state.get("active_batch") is not None or allow_closed_checkpoint
    if state.get("run_id") != PROTOCOL["run_id"] or not active_batch_required or not valid_stage:
        raise RuntimeError("isolated Round 22 run is not at a resumable batch checkpoint")


def _load_round22_run_context(run_id: str) -> dict:
    if not run_id or any(char in run_id for char in "\\/:"):
        raise ValueError("run_id must be a non-empty path-safe identifier")
    run_root = OUTPUT / "runs" / run_id
    protocol_path = run_root / "protocol.json"
    manifest_path = run_root / "run_manifest.json"
    if not protocol_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"isolated Round 22 run not found: {run_root}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if protocol.get("run_id") != run_id or manifest.get("run_id") != run_id:
        raise RuntimeError("isolated Round 22 run ID does not match its manifest")
    return {"run_id": run_id, "run_root": run_root, "protocol_path": protocol_path,
            "run_output": run_root / "outputs", "parent_output": Path(manifest["source_output"]),
            "protocol": protocol, "manifest_path": manifest_path, "manifest": manifest}


def _run_round22_with_batch_resume() -> dict:
    """Continue across the existing 90-minute batch checkpoint without resetting R22."""
    from experiments.bill_greedy.round22 import run_round_22

    while True:
        try:
            return run_round_22()
        except RuntimeError as exc:
            if "batch stopped at its 60-fit or 90-minute checkpoint" not in str(exc):
                raise
            checkpoint_batch()
            start_batch()


def run_new_round22(source_run_id: str) -> dict:
    """Run the authorized R22 candidates in a separate 28-fit / 177-minute envelope."""
    run_id = f"bill_greedy_round22_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    context = initialize_round22_run(run_id, source_run_id)
    activate_round22_run(context)
    start_batch()
    result = _run_round22_with_batch_resume()
    result["run_id"] = run_id
    result["run_root"] = str(context["run_root"])
    manifest = json.loads(context["manifest_path"].read_text(encoding="utf-8"))
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["result_summary"] = {"accepted": result["accepted"],
                                  "fit_attempts": result["fit_attempts"]}
    context["manifest_path"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return result


def resume_round22_run(run_id: str) -> dict:
    """Resume an existing R22 run only when code, data, and locked commit still match."""
    context = _load_round22_run_context(run_id)
    activate_round22_run(context, initialize=False)
    audit = _source_commit_and_hash()
    manifest = context["manifest"]
    if audit["commit"] != manifest["parent_commit"] or audit["data_sha256"] != manifest["data_sha256"]:
        raise RuntimeError("resume stopped: source commit or data hash differs from the isolated R22 run")
    rows = _ledger_rows()
    first_fit = next((row for row in rows if row.get("event") == "fit_start"), None)
    if first_fit and first_fit.get("code_sha256") != _code_hashes():
        raise RuntimeError("resume stopped: source code hash differs from the isolated R22 fit provenance")
    return _run_round22_with_batch_resume()


def _round21_resume_code_hashes(rows: list[dict]) -> dict:
    """Return the latest code fingerprint explicitly approved for this unscored run."""
    for row in reversed(rows):
        if row.get("event") == "code_update_after_unscored_fit":
            return row.get("current_code_sha256", {})
    first_fit = next((row for row in rows if row.get("event") == "fit_start"), None)
    return {} if first_fit is None else first_fit.get("code_sha256", {})


def _round21_unscored_code_update_safe(rows: list[dict]) -> bool:
    """Allow a repair only if all prior fits are resolved and no candidate was scored or saved."""
    fit_starts = [row for row in rows if row.get("event") == "fit_start"]
    if not fit_starts:
        return False
    completed = {row.get("fit_id") for row in rows if row.get("event") == "fit_complete"}
    if any(row.get("fit_id") not in completed for row in fit_starts):
        return False
    if any(str(row.get("event", "")).startswith("round21_screen_complete")
           or str(row.get("event", "")).startswith("round21_development_complete")
           or str(row.get("event", "")).startswith("round21_fixed_origin_complete")
           for row in rows):
        return False
    if (OUTPUT / "round_21_results.json").exists():
        return False
    for directory in (OUTPUT / "neighbors", OUTPUT / "predictions"):
        if directory.exists() and any(path.is_file() for path in directory.rglob("*")):
            return False
    return True


def _fit_budget_check() -> None:
    state = _state()
    final_validation = state.get("budget_phase") == "final_validation"
    fit_limit = (PROTOCOL["limits"]["fits"] if final_validation else
                 PROTOCOL["limits"]["fits"] - PROTOCOL["limits"].get("final_reserve_fits", 0))
    time_limit = (PROTOCOL["limits"]["minutes"] if final_validation else
                  PROTOCOL["limits"]["development_minutes"])
    if state["fit_attempts"] >= fit_limit:
        phase = "total" if final_validation else "development"
        raise RuntimeError(f"study stopped at the {phase} fit cap")
    if state["elapsed_minutes"] >= time_limit:
        phase = "total" if final_validation else "development"
        raise RuntimeError(f"study stopped at the {phase} time cap")
    rows = _ledger_rows()
    latest_batch = next((row for row in reversed(rows) if row.get("event") in {"batch_start", "batch_end"}), None)
    if latest_batch is None or latest_batch.get("event") != "batch_start":
        raise RuntimeError("start a new bounded batch from a completed checkpoint before fitting")
    starts_after_batch = sum(row.get("event") == "fit_start" for row in rows[rows.index(latest_batch) + 1:])
    batch_minutes = (datetime.now(timezone.utc) - datetime.fromisoformat(latest_batch["at_utc"])).total_seconds() / 60
    if _batch_budget_exceeded(starts_after_batch, batch_minutes):
        raise RuntimeError("batch stopped at its 60-fit or 90-minute checkpoint; close and checkpoint before resuming")


def _ledger_rows() -> list[dict]:
    return [json.loads(line) for line in LEDGER.read_text(encoding="utf-8").splitlines() if line.strip()] if LEDGER.exists() else []


def _batch_budget_exceeded(fit_attempts: int, elapsed_minutes: float) -> bool:
    limits = PROTOCOL["limits"]
    return fit_attempts >= limits["batch_fits"] or elapsed_minutes >= limits["batch_minutes"]


def _development_budget_exceeded(fit_attempts: int, elapsed_minutes: float) -> bool:
    limits = PROTOCOL["limits"]
    development_fit_cap = limits["fits"] - limits.get("final_reserve_fits", 0)
    return fit_attempts >= development_fit_cap or elapsed_minutes >= limits["development_minutes"]


def start_batch() -> dict:
    rows = _ledger_rows()
    last_batch = next((row for row in reversed(rows) if row.get("event") in {"batch_start", "batch_end"}), None)
    if last_batch and last_batch.get("event") == "batch_start":
        raise RuntimeError("the previous batch is still open; resume it or close it at a checkpoint first")
    if rows:
        starts = {row["fit_id"] for row in rows if row.get("event") == "fit_start"}
        finished = {row["fit_id"] for row in rows if row.get("event") == "fit_complete"}
        if starts - finished:
            raise RuntimeError("cannot begin a batch while a fit is unresolved in the ledger")
    prior_state = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}
    current_state = _state()
    if current_state.get("budget_phase") == "final_validation":
        if (current_state["fit_attempts"] >= PROTOCOL["limits"]["fits"]
                or current_state["elapsed_minutes"] >= PROTOCOL["limits"]["minutes"]):
            raise RuntimeError("cannot start final validation: the total fit or time cap is exhausted")
    elif _development_budget_exceeded(current_state["fit_attempts"], current_state["elapsed_minutes"]):
        raise RuntimeError("cannot start another development batch: the fit or development-time cap is exhausted")
    allowed = {"round_2_complete", "round_3_complete", "round_3_running", "round_4_complete",
               "round_4_running", "round_5_complete", "round_5_running", "round_6_complete",
               "round_6_running", "round_7_complete", "round_7_running",
               "round_8_complete", "round_8_running", "round_9_complete", "round_9_running",
               "round_10_complete", "round_10_running",
               "round_11_complete", "round_11_running",
               "round_12_complete", "round_12_running",
               "round_13_complete", "round_13_running",
               "round_14_complete", "round_14_running",
               "round_15_complete", "round_15_running",
               "round_16_complete", "round_16_running",
               "round_17_complete", "round_17_running",
               "round_18_complete", "round_18_running",
               "round_19_complete", "round_19_running", "round_20_complete", "round_20_running",
               "round_21_complete", "round_21_running", "round_22_running", "round_22_complete",
               "round_23_complete", "round_23_development_complete", "round_23_final_validation",
                "round_24_complete", "round_24_development_complete", "round_24_running", "round_24_final_validation",
                "round_25_complete", "round_25_development_complete", "round_25_running", "round_25_final_validation",
               "g01_development_complete", "fixed_origin_complete"}
    if prior_state.get("active_stage") not in allowed:
        raise RuntimeError(f"batch can start only from a completed checkpoint, found {prior_state.get('active_stage')}")
    batch_id = int(prior_state.get("batch_id", 1)) + 1
    record = {"event": "batch_start", "batch_id": batch_id, "fit_attempts_before": _state()["fit_attempts"],
              "at_utc": datetime.now(timezone.utc).isoformat(), "stage": prior_state["active_stage"]}
    _append(record)
    _write_state({"batch_id": batch_id, "active_batch": batch_id, "active_stage": "batch_started"})
    return record


def end_batch(stage: str) -> None:
    rows = _ledger_rows()
    latest_batch = next((row for row in reversed(rows) if row.get("event") in {"batch_start", "batch_end"}), None)
    if latest_batch is None or latest_batch.get("event") != "batch_start":
        raise RuntimeError("no open batch to checkpoint")
    starts = {row["fit_id"] for row in rows if row.get("event") == "fit_start"}
    finished = {row["fit_id"] for row in rows if row.get("event") == "fit_complete"}
    if starts - finished:
        raise RuntimeError("cannot checkpoint while a fit is unresolved in the ledger")
    _append({"event": "batch_end", "batch_id": latest_batch["batch_id"],
             "fit_attempts": _state()["fit_attempts"], "stage": stage,
             "at_utc": datetime.now(timezone.utc).isoformat()})
    _write_state({"active_batch": None, "active_stage": stage})


def checkpoint_batch() -> dict:
    prior_state = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}
    end_batch(str(prior_state.get("active_stage", "checkpointed")))
    return json.loads(STATE.read_text(encoding="utf-8"))


def _fit_record(fit_id: str, fit_fn):
    _fit_budget_check()
    start = datetime.now(timezone.utc).isoformat()
    _append({"event": "fit_start", "fit_id": fit_id, "at_utc": start,
             "data_sha256": PROTOCOL["data_sha256"], "code_sha256": _code_hashes()})
    _write_state()
    timer = time.perf_counter()
    try:
        result = fit_fn()
    except Exception as exc:
        _append({"event": "fit_complete", "fit_id": fit_id, "status": "failed",
                 "seconds": time.perf_counter() - timer, "error": f"{type(exc).__name__}: {exc}"})
        _write_state()
        raise
    result["fit_seconds"] = time.perf_counter() - timer
    _append({"event": "fit_complete", "fit_id": fit_id, "status": "complete",
             "seconds": result["fit_seconds"], "prediction_sha256": result.get("prediction_sha256")})
    _write_state()
    return result


def _source_commit_and_hash() -> dict:
    branch = subprocess.run(["git", "branch", "--show-current"], cwd=ROOT, capture_output=True,
                            text=True, check=True).stdout.strip()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                            text=True, check=True).stdout.strip()
    if branch != "bill" or commit != PROTOCOL["parent_commit"]:
        raise RuntimeError(f"expected bill at {PROTOCOL['parent_commit']}, found {branch} at {commit}")
    if _sha(DATA) != PROTOCOL["data_sha256"]:
        raise RuntimeError("cached listings parquet hash differs from the frozen v2 data hash")
    return {"branch": branch, "commit": commit, "data_sha256": _sha(DATA), "code_sha256": _code_hashes()}


def _load_fold(month: str):
    from experiments.model_optimization_v2.folds import build_month_fold
    month_end = (pd.Period(month, freq="M") + 1).start_time
    listings = pd.read_parquet(DATA, filters=[("date_sold", "<", month_end)])
    listings["date_sold"] = pd.to_datetime(listings.date_sold, errors="coerce")
    train, homes, truth = build_month_fold(listings, month, "off_market")
    heldout = listings.loc[listings.date_sold.dt.to_period("M").astype(str).eq(month)].copy()
    heldout["ml_num"] = heldout.ml_num.astype(str)
    heldout = heldout.set_index("ml_num").loc[homes.ml_num.astype(str)].reset_index()
    if not np.array_equal(homes.ml_num.astype(str).to_numpy(), truth.ml_num.astype(str).to_numpy()):
        raise RuntimeError(f"heldout rows and labels differ in {month}")
    return listings, train, homes, truth, heldout


def _old_c0(month: str, track: str, homes: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame | None:
    base = OUTPUT.parent / "model_research_next" / "predictions" / "confirmation"
    csv_path = base / "pipelines" / track / f"C0__{month}.csv"
    meta_path = csv_path.with_suffix(".json")
    if not csv_path.exists() or not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("pipeline_id") != "C0" or meta.get("month") != month or meta.get("track") != track:
        raise RuntimeError(f"unexpected cached baseline metadata: {meta_path}")
    if meta.get("data_sha256") != PROTOCOL["data_sha256"]:
        raise RuntimeError(f"cached baseline has wrong data hash: {csv_path}")
    for relative, expected in meta.get("code_sha256", {}).items():
        path = ROOT / relative
        if not path.is_file() or _sha(path) != expected:
            raise RuntimeError(f"cached C0 code hash mismatch for {relative}")
    frame = pd.read_csv(csv_path, dtype={"ml_num": str})
    expected_ids = homes.ml_num.astype(str).to_numpy()
    if not np.array_equal(frame.ml_num.astype(str).to_numpy(), expected_ids):
        raise RuntimeError(f"cached C0 IDs do not match current {month} fold")
    if not np.array_equal(frame.sold_price.to_numpy(float), truth.sold_price.to_numpy(float)):
        raise RuntimeError(f"cached C0 targets do not match current {month} fold")
    if not np.isfinite(frame.predicted_price.to_numpy(float)).all():
        raise RuntimeError(f"cached C0 predictions are not finite: {csv_path}")
    return frame


def _component_spec(track: str, component: str):
    from experiments.model_optimization_v2.worker import fit_component
    winner = json.loads((ROOT / "model_candidate_v2/winner_config.json").read_text(encoding="utf-8"))
    frozen = json.loads((ROOT / "model_candidate_v2/models/frozen_list_aware.json").read_text(encoding="utf-8"))
    if component == "LGBM":
        item = winner["locked"][track]
        return item["model_family"], item["model_config"], None
    if component == "HGBR":
        return frozen["model_family"], frozen["model_config"], "log_sold_price"
    raise ValueError(component)


def _fit_baseline_component(month, track, train, homes, component, *, fit_tag=None):
    from experiments.model_optimization_v2.worker import fit_component
    family, config, target_override = _component_spec(track, component)
    def perform():
        result = fit_component(train, homes, track, "C0", month, family, config,
                               target_override=target_override)
        result["prediction_sha256"] = hashlib.sha256(np.asarray(result["predictions"], dtype="float64").tobytes()).hexdigest()
        return result
    return _fit_record(f"baseline:{track}:{component}:{fit_tag or month}", perform)


VARIANTS = {
    "G02_shrink30": {"residual_shrink": 30.0},
    "G03_decay12": {"decay_half_life_months": 12.0},
    "G05_residual_std": {"include_residual_std": True},
    "G06_lot_garage": {"include_property_extras": True},
    "G07_area_range": {"include_area_range": True},
    "G08_num_leaves": {"model_config_overrides": {
        "off_market": {"num_leaves": 63}, "list_aware": {"num_leaves": 31}}},
    "G09_min_child50": {"model_config_overrides": {
        "off_market": {"min_child_samples": 50}, "list_aware": {"min_child_samples": 50}}},
    "G10_l2_10": {"model_config_overrides": {
        "off_market": {"reg_lambda": 10}, "list_aware": {"reg_lambda": 10}}},
    "G11_learning_rate_002": {"model_config_overrides": {
        "off_market": {"learning_rate": 0.02}, "list_aware": {"learning_rate": 0.02}}},
    "G12_mape_objective": {"target_method": "M13", "model_config_overrides": {
        "off_market": {"objective": "mape"}, "list_aware": {"objective": "mape"}}},
    "G13_residual_std_on_G06": {"include_residual_std": True},
    "G14_lagged_type_market_index": {"include_market_index": True},
    "G15_catboost": {
        "model_family": "catboost",
        "model_config": {
            "iterations": 600, "depth": 6, "learning_rate": 0.05,
            "l2_leaf_reg": 3, "loss_function": "RMSE", "has_time": True,
            "random_seed": 0, "thread_count": 4, "verbose": False,
            "allow_writing_files": False,
        },
    },
    "G16_l1_log_objective": {"model_config_overrides": {
        "off_market": {"objective": "regression_l1"},
        "list_aware": {"objective": "regression_l1"}}},
    "G17_nearest_comparables": {"include_neighbor_comparables": True},
    "G18_detached_specialist": {"specialist_house_type": "Detached"},
    "G19_detached_shrink_025": {"specialist_house_type": "Detached",
                                "specialist_blend_source": "G18_detached_specialist",
                                "specialist_blend_weight": 0.25,
                                "feature_parent": "G06_lot_garage"},
    "G19_detached_shrink_050": {"specialist_house_type": "Detached",
                                "specialist_blend_source": "G18_detached_specialist",
                                "specialist_blend_weight": 0.50,
                                "feature_parent": "G06_lot_garage"},
    "G19_detached_shrink_075": {"specialist_house_type": "Detached",
                                "specialist_blend_source": "G18_detached_specialist",
                                "specialist_blend_weight": 0.75,
                                "feature_parent": "G06_lot_garage"},
    "G20_list_anchor_050": {"list_anchor_alpha": 0.50},
    "G20_list_anchor_075": {"list_anchor_alpha": 0.75},
    "G20_list_anchor_090": {"list_anchor_alpha": 0.90},
    "G21_list_aware_semi_detached_specialist": {"specialist_house_type": "Semi-Detached"},
    "G22_off_market_semi_detached_specialist": {"specialist_house_type": "Semi-Detached"},
    "G23_list_aware_l1_mix_010": {"prediction_blend_source": "G16_l1_log_objective",
                                   "prediction_blend_weight": 0.10},
    "G24_off_market_detached_leaves63": {"specialist_house_type": "Detached",
                                         "specialist_blend_weight": 0.75,
                                         "prediction_base": "G06_lot_garage",
                                         "model_config_overrides": {"off_market": {"num_leaves": 63}}},
    "G25_list_aware_huber_objective": {"model_config_overrides": {
        "list_aware": {"objective": "huber", "alpha": 0.9}}},
    "G26_off_market_detached_l1_specialist": {"specialist_house_type": "Detached",
                                              "specialist_blend_weight": 0.75,
                                              "prediction_base": "G06_lot_garage",
                                              "model_config_overrides": {"off_market": {"objective": "regression_l1"}}},
    "G27_off_market_detached_minchild200": {"specialist_house_type": "Detached",
                                             "specialist_blend_weight": 0.75,
                                             "prediction_base": "G06_lot_garage",
                                             "model_config_overrides": {"off_market": {"min_child_samples": 200}}},
    "G28_off_market_hierarchical_market_index": {"include_hierarchical_market_index": True,
                                                  "specialist_house_type": "Detached",
                                                  "specialist_blend_weight": 0.75,
                                                  "prediction_base": "G06_lot_garage"},
    "G33_list_aware_q90_specialist": {},
    "G34_off_market_large_detached_catboost": {
        "model_family": "catboost",
        "model_config": {
            "iterations": 600, "depth": 6, "learning_rate": 0.05,
            "l2_leaf_reg": 3, "loss_function": "RMSE", "has_time": True,
            "random_seed": 0, "thread_count": 4, "verbose": False,
            "allow_writing_files": False,
        },
    },
    "G35_off_market_cold_start_specialist": {},
    "G36_list_aware_q90_specialist_050": {},
    "G37_off_market_cold_start_catboost": {
        "model_family": "catboost",
        "model_config": {
            "iterations": 600, "depth": 6, "learning_rate": 0.05,
            "l2_leaf_reg": 3, "loss_function": "RMSE", "has_time": True,
            "random_seed": 0, "thread_count": 4, "verbose": False,
            "allow_writing_files": False,
        },
    },
    "G38_list_aware_q90_catboost_050": {
        "model_family": "catboost",
        "model_config": {
            "iterations": 600, "depth": 6, "learning_rate": 0.05,
            "l2_leaf_reg": 3, "loss_function": "RMSE", "has_time": True,
            "random_seed": 0, "thread_count": 4, "verbose": False,
            "allow_writing_files": False,
        },
    },
    "G39_list_aware_q90_extratrees_050": {
        "model_family": "extra_trees",
        "model_method": "M17",
        "model_config": {
            "n_estimators": 200, "min_samples_leaf": 20, "max_features": 0.8,
            "random_state": 0, "n_jobs": 4,
        },
    },
}


def _feature_parent(parent_id: str) -> str:
    """Return the feature set used to fit a parent, including derived specialist blends."""
    return str(VARIANTS.get(parent_id, {}).get("feature_parent", parent_id))


def _catboost_training_rows(train: pd.DataFrame) -> pd.DataFrame:
    """Use stable chronological order for CatBoost's time-aware ordered statistics."""
    dates = pd.to_datetime(train["date_sold"], errors="raise").astype("int64").to_numpy()
    ids = train["ml_num"].astype(str).to_numpy()
    order = np.lexsort((ids, dates))
    return train.iloc[order].copy()


def _variant_model_spec(track: str, variant: str | None, base_model: dict) -> tuple[str, dict]:
    variant_spec = VARIANTS.get(variant, {}) if variant else {}
    family = variant_spec.get("model_family", base_model["model_family"])
    if family in {"catboost", "extra_trees"}:
        config = dict(variant_spec["model_config"])
    else:
        config = dict(base_model["model_config"], n_jobs=4)
        config.update(variant_spec.get("model_config_overrides", {}).get(track, {}))
    return family, config


def _make_history_estimator(model_family: str, config: dict, train_x: pd.DataFrame):
    if model_family == "extra_trees":
        from experiments.model_optimization_v2.worker import _sklearn_pipeline

        return _sklearn_pipeline("M17", train_x)
    from experiments.model_optimization_v2.learners import make_learner

    return make_learner(model_family, config)


def _history_feature_options(*, base_variant: str | None, variant: str | None) -> tuple[dict, bool]:
    excluded = {"model_config_overrides", "target_method", "target_override", "model_family", "model_config", "model_method",
                "include_neighbor_comparables", "specialist_house_type", "specialist_blend_source",
                "specialist_blend_weight", "feature_parent", "prediction_base",
                "prediction_blend_source", "prediction_blend_weight"}
    options = {}
    include_comparables = False
    for inherited in (base_variant, variant):
        if inherited:
            spec = VARIANTS.get(inherited, {})
            options.update({key: value for key, value in spec.items() if key not in excluded})
            include_comparables |= spec.get("include_neighbor_comparables", False)
    return options, include_comparables


def _fit_history_candidate(month, track, train, homes, *, cutoff=None, fit_tag=None, variant=None,
                           base_variant=None, fit_rows=None):
    from experiments.bill_greedy.history import building_history_features
    from experiments.model_optimization_v2.worker import _normalize_categories
    from experiments.model_optimization_v2.learners import (inverse_price, training_target,
                                                            validate_actual_objective)
    from data import AFTER_SALE_COLUMNS

    hidden = set(AFTER_SALE_COLUMNS)
    if track == "off_market":
        hidden.add("list_price")
    leaked = sorted(hidden.intersection(homes.columns))
    if leaked:
        raise ValueError(f"prediction inputs contain hidden columns: {', '.join(leaked)}")

    winner = json.loads((ROOT / "model_candidate_v2/winner_config.json").read_text(encoding="utf-8"))
    model = winner["locked"][track]
    fit_rows = train if fit_rows is None else fit_rows
    variant_spec = VARIANTS.get(variant, {}) if variant else {}
    model_family, config = _variant_model_spec(track, variant, model)
    if model_family == "catboost":
        fit_rows = _catboost_training_rows(fit_rows)
    def perform():
        from model_candidate_v2 import features as frozen_features
        train_x = frozen_features.off_market_features(fit_rows) if track == "off_market" else frozen_features.list_aware_features(fit_rows)
        eval_x = frozen_features.off_market_features(homes) if track == "off_market" else frozen_features.list_aware_features(homes)
        options, include_comparables = _history_feature_options(base_variant=base_variant, variant=variant)
        train_hist = building_history_features(train, fit_rows, track=track, training_rows=True, **options)
        eval_cutoff = pd.Period(month, freq="M").start_time if cutoff is None else pd.Timestamp(cutoff)
        eval_hist = building_history_features(train, homes, track=track, cutoff=eval_cutoff, **options)
        train_x = pd.concat([train_x, train_hist], axis=1)
        eval_x = pd.concat([eval_x, eval_hist], axis=1)
        if include_comparables:
            from experiments.bill_greedy.comparables import nearest_comparable_features
            train_comparables = nearest_comparable_features(train, fit_rows, training_rows=True)
            eval_comparables = nearest_comparable_features(train, homes, cutoff=eval_cutoff)
            train_x = pd.concat([train_x, train_comparables], axis=1)
            eval_x = pd.concat([eval_x, eval_comparables], axis=1)
        if list(train_x.columns) != list(eval_x.columns):
            raise RuntimeError("G01 train and heldout feature columns differ")
        train_x, eval_x, categories = _normalize_categories(train_x.copy(), eval_x.copy(), model_family)
        target_method = VARIANTS.get(variant, {}).get("target_method", "C0") if variant else "C0"
        target_override = variant_spec.get("target_override")
        target, target_name = training_target(fit_rows, track, target_method,
                                              target_override=target_override)
        estimator = _make_history_estimator(model_family, config, train_x)
        fit_kwargs = {"cat_features": list(categories)} if model_family == "catboost" else {}
        estimator.fit(train_x, target, **fit_kwargs)
        expected_objective = str(config.get("objective", config.get("loss_function", "regression")))
        actual_objective = validate_actual_objective(estimator, model_family, expected_objective)
        raw = np.asarray(estimator.predict(eval_x), dtype=float)
        predictions = inverse_price(raw, homes, track, target_method,
                                    target_override=target_override)
        digest = hashlib.sha256(np.asarray(predictions, dtype="float64").tobytes()).hexdigest()
        return {"estimator": estimator, "predictions": predictions, "prediction_sha256": digest,
                "feature_columns": list(train_x.columns), "target": target_name,
                "model_family": model_family, "model_config": config,
                "actual_objective": actual_objective, "training_rows": int(len(fit_rows)),
                "evaluation_rows": int(len(homes))}
    family_tag = {"lightgbm": "LGBM", "catboost": "CAT"}.get(model_family, model_family.upper())
    fit_id = f"{variant or 'G01'}:{track}:{family_tag}:{fit_tag or month}"
    return _fit_record(fit_id, perform)


def _score(frame: pd.DataFrame) -> dict:
    actual = frame.sold_price.to_numpy(dtype=float)
    predicted = frame.predicted_price.to_numpy(dtype=float)
    ape = 100.0 * np.abs(predicted - actual) / actual
    return {"n": int(len(frame)), "mdape_percent": float(np.median(ape)), "mape_percent": float(np.mean(ape))}


def _month_horizons(dates: pd.Series, origin: str | pd.Timestamp) -> np.ndarray:
    periods = pd.to_datetime(dates, errors="raise").dt.to_period("M")
    origin_period = pd.Period(pd.Timestamp(origin), freq="M")
    return ((periods.dt.year - origin_period.year) * 12 + periods.dt.month - origin_period.month + 1).to_numpy(dtype=np.int64)


def _load_hgbr_cache(month: str) -> pd.DataFrame | None:
    path = OUTPUT.parent / "model_research_next" / "predictions" / "confirmation" / "components" / "list_aware" / f"C0_HGBR__{month}.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path, dtype={"ml_num": str})
    return frame


def _frozen_hgbr_predictions(month: str, track: str, train: pd.DataFrame, homes: pd.DataFrame) -> np.ndarray:
    if track != "list_aware":
        raise ValueError("frozen HGBR blend exists only for list-aware")
    path = OUTPUT / "frozen_hgbr" / track / f"{month}.csv"
    if path.exists():
        cached = pd.read_csv(path, dtype={"ml_num": str})
        if not np.array_equal(cached.ml_num.to_numpy(), homes.ml_num.astype(str).to_numpy()):
            raise RuntimeError(f"saved frozen HGBR IDs do not align in {month}")
        return cached.raw_prediction.to_numpy(float)
    old = _load_hgbr_cache(month)
    if old is not None:
        if not np.array_equal(old.ml_num.astype(str).to_numpy(), homes.ml_num.astype(str).to_numpy()):
            raise RuntimeError(f"old frozen HGBR IDs do not align in {month}")
        prediction = old.raw_prediction.to_numpy(float)
    else:
        fit = _fit_baseline_component(month, track, train, homes, "HGBR")
        prediction = np.asarray(fit["predictions"], dtype=float)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ml_num": homes.ml_num.astype(str).to_numpy(), "raw_prediction": prediction}).to_csv(path, index=False)
    return prediction


def _load_parent_prediction(month: str, track: str, homes: pd.DataFrame, truth: pd.DataFrame,
                            parent_id: str = "G01") -> pd.DataFrame:
    if parent_id == "G01":
        path = PARENT_OUTPUT / "predictions" / parent_id / track / f"{month}.csv"
    else:
        path = PARENT_OUTPUT / "neighbors" / parent_id / track / f"{month}.csv"
    if not path.is_file():
        raise RuntimeError(f"{parent_id} parent predictions are missing: {path}")
    frame = pd.read_csv(path, dtype={"ml_num": str})
    if (not np.array_equal(frame.ml_num.to_numpy(), homes.ml_num.astype(str).to_numpy())
            or not np.array_equal(frame.sold_price.to_numpy(float), truth.sold_price.to_numpy(float))):
        raise RuntimeError(f"{parent_id} parent IDs/labels do not align in {month}/{track}")
    return frame


def _fit_timed_out(ledger_rows: list[dict], variant: str, track: str) -> bool:
    prefix = f"{variant}:{track}:"
    return any(row.get("event") == "fit_complete" and row.get("status") == "timeout"
               and str(row.get("fit_id", "")).startswith(prefix) for row in ledger_rows)


def _screen_fit_retry_tag(ledger_rows: list[dict], variant: str, track: str, month: str) -> str | None:
    family = VARIANTS.get(variant, {}).get("model_family", "lightgbm")
    family_tag = {"lightgbm": "LGBM", "catboost": "CAT"}.get(family, family.upper())
    base_id = f"{variant}:{track}:{family_tag}:{month}"
    attempts = [row for row in ledger_rows if row.get("event") == "fit_start"
                and (row.get("fit_id") == base_id or str(row.get("fit_id", "")).startswith(base_id + "-retry"))]
    return f"{month}-retry{len(attempts)}" if attempts else None


def _fit_variant(month: str, track: str, train: pd.DataFrame, homes: pd.DataFrame, variant: str) -> dict:
    result = _fit_history_candidate(month, track, train, homes, variant=variant)
    return result


def _screen_variant(month: str, track: str, train: pd.DataFrame, homes: pd.DataFrame,
                    truth: pd.DataFrame, variant: str, *, parent_id: str = "G01") -> tuple[pd.DataFrame, dict]:
    path = _candidate_path(variant, track, month)
    meta_path = path.with_suffix(".json")
    if path.is_file() and meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            prior_hashes = meta.get("code_sha256", {})
            current_hashes = _code_hashes()
            stable_inputs = (meta.get("variant") == variant and meta.get("track") == track
                             and meta.get("parent_id", "G01") == parent_id
                             and meta.get("month") == month
                             and meta.get("data_sha256") == PROTOCOL["data_sha256"]
                             and _cache_code_matches(prior_hashes, current_hashes))
            if stable_inputs:
                cached = pd.read_csv(path, dtype={"ml_num": str})
                if (np.array_equal(cached.ml_num.to_numpy(), homes.ml_num.astype(str).to_numpy())
                        and np.array_equal(cached.sold_price.to_numpy(float), truth.sold_price.to_numpy(float))):
                    return cached, None
        except (OSError, ValueError, json.JSONDecodeError, pd.errors.ParserError):
            pass
    fit_tag = _screen_fit_retry_tag(_ledger_rows(), variant, track, month)
    specialist_type = VARIANTS.get(variant, {}).get("specialist_house_type")
    blend_source = VARIANTS.get(variant, {}).get("specialist_blend_source")
    prediction_blend_source = VARIANTS.get(variant, {}).get("prediction_blend_source")
    if prediction_blend_source:
        if track != "list_aware":
            raise ValueError("full-track prediction blending is registered only for list-aware")
        source, source_fit = _screen_variant(month, track, train, homes, truth,
                                             prediction_blend_source, parent_id=parent_id)
        parent = _load_parent_prediction(month, track, homes, truth, parent_id)
        weight = VARIANTS[variant]["prediction_blend_weight"]
        candidate = _blend_prediction_frames(parent, source, weight)
        path.parent.mkdir(parents=True, exist_ok=True)
        candidate.to_csv(path, index=False)
        derived_sha = hashlib.sha256(candidate.predicted_price.to_numpy(dtype="float64").tobytes()).hexdigest()
        source_meta = json.loads(_candidate_path(prediction_blend_source, track, month)
                                 .with_suffix(".json").read_text(encoding="utf-8"))
        path.with_suffix(".json").write_text(json.dumps({
            "month": month, "track": track, "variant": variant, "parent_id": parent_id,
            "data_sha256": PROTOCOL["data_sha256"], "code_sha256": _code_hashes(),
            "prediction_sha256": derived_sha, "derived_from": prediction_blend_source,
            "source_prediction_sha256": source_meta.get("prediction_sha256"),
            "prediction_blend_weight": weight, "fit_attempts_added": 0,
        }, indent=2), encoding="utf-8")
        _append({"event": "prediction_transform_complete", "round": 14, "track": track,
                 "variant": variant, "source_variant": prediction_blend_source,
                 "month": month, "prediction_blend_weight": weight,
                 "fit_attempts_added": 0, "prediction_sha256": derived_sha})
        return candidate, source_fit
    if blend_source:
        source_path = _candidate_path(blend_source, track, month)
        source_meta_path = source_path.with_suffix(".json")
        source_meta = json.loads(source_meta_path.read_text(encoding="utf-8"))
        run9_path = OUTPUT / "round_9_results.json"
        run9 = json.loads(run9_path.read_text(encoding="utf-8"))
        expected_source_hashes = run9["preflight"]["code_sha256"]
        if (source_meta.get("variant") != blend_source or source_meta.get("track") != track
                or source_meta.get("parent_id") != parent_id or source_meta.get("month") != month
                or source_meta.get("data_sha256") != PROTOCOL["data_sha256"]
                or source_meta.get("code_sha256") != expected_source_hashes):
            raise RuntimeError(f"specialist source provenance mismatch for {source_path}")
        source = pd.read_csv(source_path, dtype={"ml_num": str})
        parent = _load_parent_prediction(month, track, homes, truth, parent_id)
        expected_ids = homes.ml_num.astype(str).to_numpy()
        if (not np.array_equal(source.ml_num.astype(str).to_numpy(), expected_ids)
                or not np.array_equal(source.sold_price.to_numpy(float), truth.sold_price.to_numpy(float))):
            raise RuntimeError(f"specialist source rows or labels do not align for {month}/{track}")
        prediction = _blend_specialist_predictions(
            parent.predicted_price.to_numpy(float), source.predicted_price.to_numpy(float), homes,
            specialist_type, VARIANTS[variant]["specialist_blend_weight"])
        candidate = pd.DataFrame({"ml_num": expected_ids, "sold_price": truth.sold_price.to_numpy(float),
                                  "predicted_price": prediction})
        path.parent.mkdir(parents=True, exist_ok=True)
        candidate.to_csv(path, index=False)
        derived_sha = hashlib.sha256(np.asarray(prediction, dtype="float64").tobytes()).hexdigest()
        meta = {"month": month, "track": track, "variant": variant, "parent_id": parent_id,
                "data_sha256": PROTOCOL["data_sha256"], "code_sha256": _code_hashes(),
                "prediction_sha256": derived_sha, "derived_from": blend_source,
                "source_prediction_sha256": source_meta.get("prediction_sha256"),
                "specialist_house_type": specialist_type,
                "specialist_blend_weight": VARIANTS[variant]["specialist_blend_weight"],
                "fit_attempts_added": 0}
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        _append({"event": "prediction_transform_complete", "round": 10, "track": track,
                 "variant": variant, "source_variant": blend_source, "month": month,
                 "specialist_blend_weight": VARIANTS[variant]["specialist_blend_weight"],
                 "fit_attempts_added": 0, "prediction_sha256": derived_sha})
        return candidate, None
    anchor_alpha = VARIANTS.get(variant, {}).get("list_anchor_alpha")
    if anchor_alpha is not None:
        if track != "list_aware":
            raise ValueError("list-price anchoring is allowed only for the list-aware track")
        parent = _load_parent_prediction(month, track, homes, truth, parent_id)
        prediction = _shrink_to_list_anchor(parent.predicted_price.to_numpy(float),
                                             homes.list_price.to_numpy(float), anchor_alpha)
        candidate = pd.DataFrame({"ml_num": homes.ml_num.astype(str).to_numpy(),
                                  "sold_price": truth.sold_price.to_numpy(float),
                                  "predicted_price": prediction})
        path.parent.mkdir(parents=True, exist_ok=True)
        candidate.to_csv(path, index=False)
        derived_sha = hashlib.sha256(np.asarray(prediction, dtype="float64").tobytes()).hexdigest()
        meta = {"month": month, "track": track, "variant": variant, "parent_id": parent_id,
                "data_sha256": PROTOCOL["data_sha256"], "code_sha256": _code_hashes(),
                "prediction_sha256": derived_sha, "derived_from": parent_id,
                "list_anchor_alpha": anchor_alpha, "fit_attempts_added": 0}
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        _append({"event": "prediction_transform_complete", "round": 11, "track": track,
                 "variant": variant, "source_variant": parent_id, "month": month,
                 "list_anchor_alpha": anchor_alpha, "fit_attempts_added": 0,
                 "prediction_sha256": derived_sha})
        return candidate, None
    if specialist_type:
        prediction_base_id = VARIANTS.get(variant, {}).get("prediction_base", parent_id)
        parent = _load_parent_prediction(month, track, homes, truth, prediction_base_id)
        specialist_mask = _specialist_mask(homes, specialist_type)
        specialist_train = _specialist_training_rows(train, specialist_type)
        if not specialist_mask.any() or specialist_train.empty:
            raise RuntimeError(f"{specialist_type} specialist has no training or evaluation rows in {month}/{track}")
        specialist_homes = homes.loc[specialist_mask].copy()
        fit = _fit_history_candidate(month, track, train, specialist_homes, variant=variant,
                                     base_variant=_feature_parent(parent_id), fit_tag=fit_tag,
                                     fit_rows=specialist_train)
        specialist_prediction = np.asarray(fit["predictions"], dtype=float)
        if track == "list_aware":
            frozen = _frozen_hgbr_predictions(month, track, train, homes)[specialist_mask]
            specialist_prediction = np.round(0.25 * np.round(frozen) + 0.75 * specialist_prediction)
        else:
            specialist_prediction = np.round(specialist_prediction)
        specialist_weight = VARIANTS.get(variant, {}).get("specialist_blend_weight")
        if specialist_weight is None:
            prediction = _overlay_specialist_predictions(parent.predicted_price.to_numpy(float), homes,
                                                           specialist_prediction, specialist_type)
        else:
            prediction = _blend_specialist_predictions(parent.predicted_price.to_numpy(float),
                                                        specialist_prediction, homes, specialist_type,
                                                        specialist_weight)
    else:
        fit = _fit_history_candidate(month, track, train, homes, variant=variant,
                                     base_variant=_feature_parent(parent_id), fit_tag=fit_tag)
        prediction = np.asarray(fit["predictions"], dtype=float)
        if track == "list_aware":
            frozen = _frozen_hgbr_predictions(month, track, train, homes)
            prediction = np.round(0.25 * np.round(frozen) + 0.75 * prediction)
        else:
            prediction = np.round(prediction)
    candidate = pd.DataFrame({"ml_num": homes.ml_num.astype(str).to_numpy(),
                              "sold_price": truth.sold_price.to_numpy(float),
                              "predicted_price": prediction})
    path.parent.mkdir(parents=True, exist_ok=True)
    candidate.to_csv(path, index=False)
    prediction_sha = hashlib.sha256(candidate.predicted_price.to_numpy(dtype="float64").tobytes()).hexdigest()
    meta = {"month": month, "track": track, "variant": variant, "parent_id": parent_id,
            "prediction_base_id": VARIANTS.get(variant, {}).get("prediction_base", parent_id),
            "data_sha256": PROTOCOL["data_sha256"],
            "code_sha256": _code_hashes(), "feature_columns": fit["feature_columns"],
            "prediction_sha256": prediction_sha, "model_prediction_sha256": fit["prediction_sha256"],
            "model_family": fit.get("model_family", "lightgbm"),
            "model_config": fit.get("model_config", {}),
            "actual_objective": fit.get("actual_objective"),
            "training_rows": fit.get("training_rows", len(train)),
            "evaluation_rows": fit.get("evaluation_rows", len(homes)),
            "specialist_house_type": specialist_type}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return candidate, fit


def _candidate_path(variant: str, track: str, month: str) -> Path:
    return OUTPUT / "neighbors" / variant / track / f"{month}.csv"


def _specialist_mask(frame: pd.DataFrame, house_type: str) -> np.ndarray:
    if "house_type_name" not in frame:
        raise ValueError("specialist routing requires house_type_name")
    return frame["house_type_name"].astype("string").eq(house_type).fillna(False).to_numpy(dtype=bool)


def _specialist_training_rows(train: pd.DataFrame, house_type: str) -> pd.DataFrame:
    return train.loc[_specialist_mask(train, house_type)].copy()


def _overlay_specialist_predictions(parent_prediction, homes: pd.DataFrame, specialist_prediction,
                                    house_type: str) -> np.ndarray:
    parent = np.asarray(parent_prediction, dtype=float)
    specialist = np.asarray(specialist_prediction, dtype=float)
    mask = _specialist_mask(homes, house_type)
    if parent.shape != (len(homes),):
        raise ValueError("parent predictions must align one-to-one with homes")
    if specialist.shape != (int(mask.sum()),):
        raise ValueError("specialist predictions must align one-to-one with specialist homes")
    if not np.isfinite(specialist).all():
        raise ValueError("specialist predictions must be finite")
    result = parent.copy()
    result[mask] = specialist
    return result


def _blend_specialist_predictions(parent_prediction, specialist_prediction, homes: pd.DataFrame,
                                  house_type: str, specialist_weight: float) -> np.ndarray:
    if not 0.0 <= specialist_weight <= 1.0:
        raise ValueError("specialist blend weight must be between 0 and 1")
    parent = np.asarray(parent_prediction, dtype=float)
    specialist = np.asarray(specialist_prediction, dtype=float)
    mask = _specialist_mask(homes, house_type)
    if parent.shape != (len(homes),):
        raise ValueError("parent predictions must align one-to-one with homes")
    if specialist.shape == parent.shape:
        specialist_values = specialist[mask]
    elif specialist.shape == (int(mask.sum()),):
        specialist_values = specialist
    else:
        raise ValueError("specialist predictions must align with all homes or only specialist homes")
    if not np.isfinite(specialist_values).all():
        raise ValueError("specialist predictions must be finite")
    blended = np.round((1.0 - specialist_weight) * parent[mask] + specialist_weight * specialist_values)
    return _overlay_specialist_predictions(parent, homes, blended, house_type)


def _blend_prediction_frames(parent: pd.DataFrame, source: pd.DataFrame, source_weight: float) -> pd.DataFrame:
    """Blend aligned, already postprocessed prediction arms and round final dollar values once."""
    if not 0.0 <= source_weight <= 1.0:
        raise ValueError("prediction blend weight must be between 0 and 1")
    if not np.array_equal(parent.ml_num.astype(str).to_numpy(), source.ml_num.astype(str).to_numpy()):
        raise ValueError("parent and source IDs must align in the same row order")
    if not np.array_equal(parent.sold_price.to_numpy(float), source.sold_price.to_numpy(float)):
        raise ValueError("parent and source labels must match")
    parent_prediction = parent.predicted_price.to_numpy(float)
    source_prediction = source.predicted_price.to_numpy(float)
    if (not np.isfinite(parent_prediction).all() or not np.isfinite(source_prediction).all()
            or (parent_prediction <= 0).any() or (source_prediction <= 0).any()):
        raise ValueError("parent and source predictions must be finite and positive")
    prediction = np.round((1.0 - source_weight) * parent_prediction + source_weight * source_prediction)
    return pd.DataFrame({"ml_num": parent.ml_num.astype(str).to_numpy(),
                         "sold_price": parent.sold_price.to_numpy(float),
                         "predicted_price": prediction})


def _shrink_to_list_anchor(parent_prediction, list_price, alpha: float) -> np.ndarray:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("list-price anchor alpha must be between 0 and 1")
    parent = np.asarray(parent_prediction, dtype=float)
    anchor = np.asarray(list_price, dtype=float)
    if parent.ndim != 1 or parent.shape != anchor.shape:
        raise ValueError("parent predictions and list prices must align one-to-one")
    if not np.isfinite(parent).all() or not np.isfinite(anchor).all() or (parent <= 0).any() or (anchor <= 0).any():
        raise ValueError("parent predictions and list prices must be finite and positive")
    log_ratio = np.log(parent) - np.log(anchor)
    return np.round(anchor * np.exp(alpha * log_ratio))


def _cache_code_matches(recorded_hashes: dict[str, str], current_hashes: dict[str, str]) -> bool:
    return bool(recorded_hashes) and recorded_hashes == current_hashes


def _screen_report(parent: pd.DataFrame, candidate: pd.DataFrame, months: list[str]) -> dict:
    monthly = {}
    for month in months:
        p = parent.loc[parent.month.astype(str).eq(month)]
        c = candidate.loc[candidate.month.astype(str).eq(month)]
        p_score, c_score = _score(p), _score(c)
        monthly[month] = {"parent": p_score, "candidate": c_score,
                          "relative_mdape_improvement": (p_score["mdape_percent"] - c_score["mdape_percent"]) / p_score["mdape_percent"]}
    p_score, c_score = _score(parent), _score(candidate)
    relative = (p_score["mdape_percent"] - c_score["mdape_percent"]) / p_score["mdape_percent"]
    mape_change = (c_score["mape_percent"] - p_score["mape_percent"]) / p_score["mape_percent"]
    qualifies = relative >= 0.0025 and all(item["relative_mdape_improvement"] > 0 for item in monthly.values()) and mape_change <= 0.01
    return {"parent": p_score, "candidate": c_score, "relative_mdape_improvement": relative,
            "relative_mape_change": mape_change, "monthly": monthly, "qualifies_screen": bool(qualifies)}


def run_neighbors() -> dict:
    """Screen four fixed neighbors per track and confirm one at most per track."""
    audit = _source_commit_and_hash()
    g01 = json.loads((OUTPUT / "g01_results.json").read_text(encoding="utf-8"))
    origin_result = json.loads((OUTPUT / "fixed_origin_results.json").read_text(encoding="utf-8"))
    eligible_tracks = {track for track in ("off_market", "list_aware")
                       if g01["results"][track]["passes_development_gate"]
                       and origin_result["combined"][track]["passes_origin_guardrail"]}
    if eligible_tracks != {"off_market", "list_aware"}:
        raise RuntimeError("G01 must pass development and fixed-origin gates before neighbor search")
    variants_by_track = {
        "off_market": ["G02_shrink30", "G03_decay12", "G05_residual_std", "G06_lot_garage"],
        "list_aware": ["G02_shrink30", "G03_decay12", "G05_residual_std", "G07_area_range"],
    }
    reports = {track: {} for track in eligible_tracks}
    candidate_frames = {track: {} for track in eligible_tracks}
    for track in ("off_market", "list_aware"):
        for variant in variants_by_track[track]:
            ledger_rows = [json.loads(line) for line in LEDGER.read_text(encoding="utf-8").splitlines() if line.strip()] if LEDGER.exists() else []
            timed_out = _fit_timed_out(ledger_rows, variant, track)
            if timed_out:
                reports[track][variant] = {"status": "timeout", "qualifies_screen": False,
                                           "reason": "A neighbor fit exceeded the per-fit time limit; it is not rerun."}
                continue
            parent_parts, candidate_parts = [], []
            for month in PROTOCOL["screen_months"]:
                _, train, base_homes, truth, heldout = _load_fold(month)
                homes = __import__("data").hide_answers(heldout, track)
                parent = _load_parent_prediction(month, track, homes, truth).copy()
                parent["month"] = month
                candidate, _ = _screen_variant(month, track, train, homes, truth, variant)
                candidate["month"] = month
                parent_parts.append(parent)
                candidate_parts.append(candidate)
            parent = pd.concat(parent_parts, ignore_index=True)
            candidate = pd.concat(candidate_parts, ignore_index=True)
            reports[track][variant] = _screen_report(parent, candidate, PROTOCOL["screen_months"])
            _append({"event": "neighbor_screen_complete", "track": track, "variant": variant,
                     **reports[track][variant]})
            if reports[track][variant]["qualifies_screen"]:
                candidate_frames[track][variant] = candidate
            print(f"screened {variant} {track}: {reports[track][variant]['parent']['mdape_percent']:.3f}% -> "
                  f"{reports[track][variant]['candidate']['mdape_percent']:.3f}% "
                  f"qualified={reports[track][variant]['qualifies_screen']}", flush=True)

    selected = {}
    confirmation = {}
    fixed_origin_confirmation = {}
    for track, reports_for_track in reports.items():
        qualifying = [variant for variant, report in reports_for_track.items() if report.get("qualifies_screen", False)]
        qualifying.sort(key=lambda variant: (reports_for_track[variant]["candidate"]["mdape_percent"],
                                             reports_for_track[variant]["candidate"]["mape_percent"], variant))
        if not qualifying:
            selected[track] = None
            continue
        winner = qualifying[0]
        selected[track] = winner
        baseline_parts, candidate_parts = [], []
        for month in PROTOCOL["development_months"]:
            _, train, _, truth, heldout = _load_fold(month)
            homes = __import__("data").hide_answers(heldout, track)
            parent = _load_parent_prediction(month, track, homes, truth).copy()
            parent["month"] = month
            candidate_path = _candidate_path(winner, track, month)
            if candidate_path.exists():
                candidate = pd.read_csv(candidate_path, dtype={"ml_num": str})
                if not np.array_equal(candidate.ml_num.to_numpy(), homes.ml_num.astype(str).to_numpy()):
                    raise RuntimeError(f"cached candidate IDs do not align for {winner}/{month}/{track}")
            else:
                candidate, fit = _screen_variant(month, track, train, homes, truth, winner)
                if fit is not None:
                    _save_neighbor_fold(month, track, winner, candidate, fit)
            candidate = candidate.copy()
            candidate["month"] = month
            baseline_parts.append(parent)
            candidate_parts.append(candidate)
        confirmation[track] = _development_report(pd.concat(baseline_parts, ignore_index=True),
                                                   pd.concat(candidate_parts, ignore_index=True),
                                                   PROTOCOL["development_months"])
        print(f"confirmed {winner} {track}: relative={confirmation[track]['relative_mdape_improvement']:.3%}, "
              f"months={confirmation[track]['months_improved']}/8, gate={confirmation[track]['passes_development_gate']}", flush=True)
        if confirmation[track]["passes_development_gate"]:
            fixed_origin_confirmation[track] = _fixed_origin_candidate(winner, track)
            print(f"fixed-origin {winner} {track}: improvement={fixed_origin_confirmation[track]['relative_mdape_improvement']:.3%}, "
                  f"gate={fixed_origin_confirmation[track]['passes_origin_guardrail']}", flush=True)

    result = {"round": 2, "parent": "G01", "screen_months": PROTOCOL["screen_months"],
              "variants_by_track": variants_by_track, "screens": reports, "selected": selected,
              "confirmation": confirmation, "fixed_origin_confirmation": fixed_origin_confirmation,
              "fit_attempts": _state()["fit_attempts"], "preflight": audit}
    path = OUTPUT / "round_2_results.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    accepted = {track: selected.get(track) for track in selected
                if selected.get(track) and confirmation[track]["passes_development_gate"]
                and fixed_origin_confirmation.get(track, {}).get("passes_origin_guardrail", False)}
    _write_state({"active_stage": "round_2_complete", "round_2_accepted": accepted,
                  "round_2_rejected": [track for track in selected if track not in accepted]})
    return result


def run_round_3() -> dict:
    """Search four one-parameter LightGBM neighbors around each track's current parent."""
    audit = _source_commit_and_hash()
    state = json.loads(STATE.read_text(encoding="utf-8"))
    if state.get("active_stage") not in {"batch_started", "round_3_running"}:
        raise RuntimeError("Round 3 requires an explicitly started bounded batch")
    _write_state({"active_stage": "round_3_running"})
    round2_accepted = state.get("round_2_accepted", {})
    parent_by_track = {"off_market": round2_accepted.get("off_market", "G01"),
                       "list_aware": round2_accepted.get("list_aware", "G01")}
    variants_by_track = {"off_market": ["G08_num_leaves", "G09_min_child50", "G10_l2_10", "G11_learning_rate_002"],
                         "list_aware": ["G08_num_leaves", "G09_min_child50", "G10_l2_10", "G11_learning_rate_002"]}
    reports = {track: {} for track in parent_by_track}
    for track, parent_id in parent_by_track.items():
        for variant in variants_by_track[track]:
            ledger_rows = [json.loads(line) for line in LEDGER.read_text(encoding="utf-8").splitlines() if line.strip()] if LEDGER.exists() else []
            if _fit_timed_out(ledger_rows, variant, track):
                reports[track][variant] = {"status": "timeout", "qualifies_screen": False,
                                           "reason": "A neighbor fit exceeded the per-fit time limit; it is not rerun."}
                continue
            parent_parts, candidate_parts = [], []
            for month in PROTOCOL["screen_months"]:
                _, train, _, truth, heldout = _load_fold(month)
                from data import hide_answers
                homes = hide_answers(heldout, track)
                parent = _load_parent_prediction(month, track, homes, truth, parent_id).copy()
                parent["month"] = month
                candidate, _ = _screen_variant(month, track, train, homes, truth, variant, parent_id=parent_id)
                candidate["month"] = month
                parent_parts.append(parent)
                candidate_parts.append(candidate)
            report = _screen_report(pd.concat(parent_parts, ignore_index=True),
                                    pd.concat(candidate_parts, ignore_index=True), PROTOCOL["screen_months"])
            report["status"] = "complete"
            report["parent_id"] = parent_id
            reports[track][variant] = report
            _append({"event": "neighbor_screen_complete", "round": 3, "track": track,
                     "variant": variant, "parent_id": parent_id, **report})
            print(f"round3 screened {variant} {track} ({parent_id}): "
                  f"{report['parent']['mdape_percent']:.3f}% -> {report['candidate']['mdape_percent']:.3f}% "
                  f"qualified={report['qualifies_screen']}", flush=True)

    selected, confirmation, fixed_confirmation = {}, {}, {}
    round3_accepted = {}
    for track, parent_id in parent_by_track.items():
        track_reports = reports[track]
        qualifying = [variant for variant, report in track_reports.items()
                      if report.get("qualifies_screen", False)]
        qualifying.sort(key=lambda variant: (track_reports[variant]["candidate"]["mdape_percent"],
                                             track_reports[variant]["candidate"]["mape_percent"], variant))
        if not qualifying:
            selected[track] = None
            continue
        winner = qualifying[0]
        selected[track] = winner
        baseline_parts, candidate_parts = [], []
        from data import hide_answers
        for month in PROTOCOL["development_months"]:
            _, train, _, truth, heldout = _load_fold(month)
            homes = hide_answers(heldout, track)
            parent = _load_parent_prediction(month, track, homes, truth, parent_id).copy()
            parent["month"] = month
            candidate, fit = _screen_variant(month, track, train, homes, truth, winner, parent_id=parent_id)
            if fit is not None:
                _save_neighbor_fold(month, track, winner, candidate, fit, parent_id=parent_id)
            candidate["month"] = month
            baseline_parts.append(parent)
            candidate_parts.append(candidate)
        confirmation[track] = _development_report(pd.concat(baseline_parts, ignore_index=True),
                                                   pd.concat(candidate_parts, ignore_index=True),
                                                   PROTOCOL["development_months"])
        confirmation[track]["parent_id"] = parent_id
        print(f"round3 confirmed {winner} {track} ({parent_id}): "
              f"relative={confirmation[track]['relative_mdape_improvement']:.3%}, "
              f"months={confirmation[track]['months_improved']}/8, "
              f"gate={confirmation[track]['passes_development_gate']}", flush=True)
        if confirmation[track]["passes_development_gate"]:
            fixed_confirmation[track] = _fixed_origin_candidate(winner, track, parent_id=parent_id)
            if fixed_confirmation[track]["passes_origin_guardrail"]:
                round3_accepted[track] = winner

    current_parents = dict(parent_by_track)
    current_parents.update(round3_accepted)
    result = {"round": 3, "parents": parent_by_track, "screen_months": PROTOCOL["screen_months"],
              "variants_by_track": variants_by_track, "screens": reports, "selected": selected,
              "confirmation": confirmation, "fixed_origin_confirmation": fixed_confirmation,
              "accepted": round3_accepted, "current_parents": current_parents,
              "fit_attempts": _state()["fit_attempts"], "preflight": audit}
    target = OUTPUT / "round_3_results.json"
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2), encoding="utf-8")
    os.replace(tmp, target)
    _write_state({"active_stage": "round_3_complete", "round_3_accepted": round3_accepted,
                  "current_parents": current_parents})
    end_batch("round_3_complete")
    return result


def run_round_4() -> dict:
    """Test a new target objective and the residual-dispersion feature on current parents."""
    audit = _source_commit_and_hash()
    state = json.loads(STATE.read_text(encoding="utf-8"))
    if state.get("active_stage") not in {"batch_started", "round_4_running"}:
        raise RuntimeError("Round 4 requires an explicitly started bounded batch")
    _write_state({"active_stage": "round_4_running"})
    parent_by_track = {"off_market": state.get("current_parents", {}).get("off_market", "G06_lot_garage"),
                       "list_aware": state.get("current_parents", {}).get("list_aware", "G01")}
    variants_by_track = {"off_market": ["G12_mape_objective", "G13_residual_std_on_G06"],
                         "list_aware": ["G12_mape_objective"]}
    reports = {track: {} for track in parent_by_track}
    from data import hide_answers
    for track, parent_id in parent_by_track.items():
        for variant in variants_by_track[track]:
            ledger_rows = _ledger_rows()
            if _fit_timed_out(ledger_rows, variant, track):
                reports[track][variant] = {"status": "timeout", "qualifies_screen": False,
                                           "reason": "A neighbor fit exceeded the per-fit time limit; it is not rerun."}
                continue
            parent_parts, candidate_parts = [], []
            for month in PROTOCOL["screen_months"]:
                _, train, _, truth, heldout = _load_fold(month)
                homes = hide_answers(heldout, track)
                parent = _load_parent_prediction(month, track, homes, truth, parent_id).copy()
                parent["month"] = month
                candidate, _ = _screen_variant(month, track, train, homes, truth, variant, parent_id=parent_id)
                candidate["month"] = month
                parent_parts.append(parent)
                candidate_parts.append(candidate)
            report = _screen_report(pd.concat(parent_parts, ignore_index=True),
                                    pd.concat(candidate_parts, ignore_index=True), PROTOCOL["screen_months"])
            report.update({"status": "complete", "parent_id": parent_id})
            reports[track][variant] = report
            _append({"event": "neighbor_screen_complete", "round": 4, "track": track,
                     "variant": variant, "parent_id": parent_id, **report})
            print(f"round4 screened {variant} {track} ({parent_id}): "
                  f"{report['parent']['mdape_percent']:.3f}% -> {report['candidate']['mdape_percent']:.3f}% "
                  f"qualified={report['qualifies_screen']}", flush=True)

    selected, confirmation, fixed_confirmation, accepted = {}, {}, {}, {}
    for track, parent_id in parent_by_track.items():
        qualifying = [variant for variant, report in reports[track].items()
                      if report.get("qualifies_screen", False)]
        qualifying.sort(key=lambda variant: (reports[track][variant]["candidate"]["mdape_percent"],
                                             reports[track][variant]["candidate"]["mape_percent"], variant))
        if not qualifying:
            selected[track] = None
            continue
        winner = qualifying[0]
        selected[track] = winner
        parent_parts, candidate_parts = [], []
        for month in PROTOCOL["development_months"]:
            _, train, _, truth, heldout = _load_fold(month)
            homes = hide_answers(heldout, track)
            parent = _load_parent_prediction(month, track, homes, truth, parent_id).copy()
            parent["month"] = month
            candidate, fit = _screen_variant(month, track, train, homes, truth, winner, parent_id=parent_id)
            if fit is not None:
                _save_neighbor_fold(month, track, winner, candidate, fit, parent_id=parent_id)
            candidate["month"] = month
            parent_parts.append(parent)
            candidate_parts.append(candidate)
        confirmation[track] = _development_report(pd.concat(parent_parts, ignore_index=True),
                                                   pd.concat(candidate_parts, ignore_index=True),
                                                   PROTOCOL["development_months"])
        confirmation[track]["parent_id"] = parent_id
        print(f"round4 confirmed {winner} {track} ({parent_id}): "
              f"relative={confirmation[track]['relative_mdape_improvement']:.3%}, "
              f"months={confirmation[track]['months_improved']}/8, "
              f"gate={confirmation[track]['passes_development_gate']}", flush=True)
        if confirmation[track]["passes_development_gate"]:
            fixed_confirmation[track] = _fixed_origin_candidate(winner, track, parent_id=parent_id)
            if fixed_confirmation[track]["passes_origin_guardrail"]:
                accepted[track] = winner

    current_parents = dict(parent_by_track)
    current_parents.update(accepted)
    result = {"round": 4, "parents": parent_by_track, "screen_months": PROTOCOL["screen_months"],
              "variants_by_track": variants_by_track, "screens": reports, "selected": selected,
              "confirmation": confirmation, "fixed_origin_confirmation": fixed_confirmation,
              "accepted": accepted, "current_parents": current_parents,
              "fit_attempts": _state()["fit_attempts"], "preflight": audit}
    target = OUTPUT / "round_4_results.json"
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2), encoding="utf-8")
    os.replace(tmp, target)
    _write_state({"active_stage": "round_4_complete", "round_4_accepted": accepted,
                  "current_parents": current_parents})
    end_batch("round_4_complete")
    return result


def run_round_5() -> dict:
    """Screen a strict one-calendar-month-lagged type-level price-per-area feature."""
    audit = _source_commit_and_hash()
    state = json.loads(STATE.read_text(encoding="utf-8"))
    if state.get("active_stage") not in {"batch_started", "round_5_running"}:
        raise RuntimeError("Round 5 requires an explicitly started bounded batch")
    _write_state({"active_stage": "round_5_running"})
    parent_by_track = {"off_market": state.get("current_parents", {}).get("off_market", "G06_lot_garage"),
                       "list_aware": state.get("current_parents", {}).get("list_aware", "G01")}
    variant = "G14_lagged_type_market_index"
    reports, confirmation, fixed_confirmation, accepted = {}, {}, {}, {}
    from data import hide_answers
    for track, parent_id in parent_by_track.items():
        parent_parts, candidate_parts = [], []
        for month in PROTOCOL["screen_months"]:
            _, train, _, truth, heldout = _load_fold(month)
            homes = hide_answers(heldout, track)
            parent = _load_parent_prediction(month, track, homes, truth, parent_id).copy()
            parent["month"] = month
            candidate, _ = _screen_variant(month, track, train, homes, truth, variant, parent_id=parent_id)
            candidate["month"] = month
            parent_parts.append(parent)
            candidate_parts.append(candidate)
        report = _screen_report(pd.concat(parent_parts, ignore_index=True),
                                pd.concat(candidate_parts, ignore_index=True), PROTOCOL["screen_months"])
        report.update({"status": "complete", "parent_id": parent_id})
        reports[track] = report
        _append({"event": "neighbor_screen_complete", "round": 5, "track": track,
                 "variant": variant, "parent_id": parent_id, **report})
        print(f"round5 screened {variant} {track} ({parent_id}): "
              f"{report['parent']['mdape_percent']:.3f}% -> {report['candidate']['mdape_percent']:.3f}% "
              f"qualified={report['qualifies_screen']}", flush=True)
        if not report["qualifies_screen"]:
            continue

        parent_parts, candidate_parts = [], []
        for month in PROTOCOL["development_months"]:
            _, train, _, truth, heldout = _load_fold(month)
            homes = hide_answers(heldout, track)
            parent = _load_parent_prediction(month, track, homes, truth, parent_id).copy()
            parent["month"] = month
            candidate, fit = _screen_variant(month, track, train, homes, truth, variant, parent_id=parent_id)
            if fit is not None:
                _save_neighbor_fold(month, track, variant, candidate, fit, parent_id=parent_id)
            candidate["month"] = month
            parent_parts.append(parent)
            candidate_parts.append(candidate)
        confirmation[track] = _development_report(pd.concat(parent_parts, ignore_index=True),
                                                   pd.concat(candidate_parts, ignore_index=True),
                                                   PROTOCOL["development_months"])
        confirmation[track]["parent_id"] = parent_id
        print(f"round5 confirmed {variant} {track} ({parent_id}): "
              f"relative={confirmation[track]['relative_mdape_improvement']:.3%}, "
              f"months={confirmation[track]['months_improved']}/8, "
              f"gate={confirmation[track]['passes_development_gate']}", flush=True)
        if confirmation[track]["passes_development_gate"]:
            fixed_confirmation[track] = _fixed_origin_candidate(variant, track, parent_id=parent_id)
            if fixed_confirmation[track]["passes_origin_guardrail"]:
                accepted[track] = variant

    current_parents = dict(parent_by_track)
    current_parents.update(accepted)
    result = {"round": 5, "variant": variant, "parents": parent_by_track,
              "screen_months": PROTOCOL["screen_months"], "screens": reports,
              "confirmation": confirmation, "fixed_origin_confirmation": fixed_confirmation,
              "accepted": accepted, "current_parents": current_parents,
              "fit_attempts": _state()["fit_attempts"], "preflight": audit}
    target = OUTPUT / "round_5_results.json"
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2), encoding="utf-8")
    os.replace(tmp, target)
    _write_state({"active_stage": "round_5_complete", "round_5_accepted": accepted,
                  "current_parents": current_parents})
    end_batch("round_5_complete")
    return result


def run_round_6() -> dict:
    """Screen the current track parents with one fixed CatBoost family challenger."""
    audit = _source_commit_and_hash()
    state = json.loads(STATE.read_text(encoding="utf-8"))
    if state.get("active_stage") not in {"batch_started", "round_6_running"}:
        raise RuntimeError("Round 6 requires an explicitly started bounded batch")
    _write_state({"active_stage": "round_6_running"})
    parent_by_track = dict(state.get("current_parents", {
        "off_market": "G06_lot_garage", "list_aware": "G01"}))
    variant = "G15_catboost"
    reports, confirmation, fixed_confirmation, accepted = {}, {}, {}, {}
    from data import hide_answers

    ledger_rows = _ledger_rows()
    for track, parent_id in parent_by_track.items():
        if _fit_timed_out(ledger_rows, variant, track):
            reports[track] = {"status": "timeout", "parent_id": parent_id,
                              "qualifies_screen": False,
                              "reason": "This exact CatBoost track already timed out and is not rerun."}
            continue
        parent_parts, candidate_parts = [], []
        for month in PROTOCOL["screen_months"]:
            _, train, _, truth, heldout = _load_fold(month)
            homes = hide_answers(heldout, track)
            parent = _load_parent_prediction(month, track, homes, truth, parent_id).copy()
            parent["month"] = month
            candidate, _ = _screen_variant(month, track, train, homes, truth, variant,
                                           parent_id=parent_id)
            candidate["month"] = month
            parent_parts.append(parent)
            candidate_parts.append(candidate)
        report = _screen_report(pd.concat(parent_parts, ignore_index=True),
                                pd.concat(candidate_parts, ignore_index=True), PROTOCOL["screen_months"])
        report.update({"status": "complete", "parent_id": parent_id,
                       "model_family": VARIANTS[variant]["model_family"]})
        reports[track] = report
        _append({"event": "neighbor_screen_complete", "round": 6, "track": track,
                 "variant": variant, "parent_id": parent_id, **report})
        print(f"round6 screened {variant} {track} ({parent_id}): "
              f"{report['parent']['mdape_percent']:.3f}% -> {report['candidate']['mdape_percent']:.3f}% "
              f"qualified={report['qualifies_screen']}", flush=True)
        if not report["qualifies_screen"]:
            continue

        parent_parts, candidate_parts = [], []
        for month in PROTOCOL["development_months"]:
            _, train, _, truth, heldout = _load_fold(month)
            homes = hide_answers(heldout, track)
            parent = _load_parent_prediction(month, track, homes, truth, parent_id).copy()
            parent["month"] = month
            candidate, fit = _screen_variant(month, track, train, homes, truth, variant,
                                             parent_id=parent_id)
            if fit is not None:
                _save_neighbor_fold(month, track, variant, candidate, fit, parent_id=parent_id)
            candidate["month"] = month
            parent_parts.append(parent)
            candidate_parts.append(candidate)
        confirmation[track] = _development_report(pd.concat(parent_parts, ignore_index=True),
                                                   pd.concat(candidate_parts, ignore_index=True),
                                                   PROTOCOL["development_months"])
        confirmation[track]["parent_id"] = parent_id
        print(f"round6 confirmed {variant} {track}: "
              f"relative={confirmation[track]['relative_mdape_improvement']:.3%}, "
              f"months={confirmation[track]['months_improved']}/8, "
              f"gate={confirmation[track]['passes_development_gate']}", flush=True)
        if confirmation[track]["passes_development_gate"]:
            fixed_confirmation[track] = _fixed_origin_candidate(variant, track, parent_id=parent_id)
            if fixed_confirmation[track]["passes_origin_guardrail"]:
                accepted[track] = variant

    current_parents = dict(parent_by_track)
    current_parents.update(accepted)
    result = {"round": 6, "variant": variant, "parents": parent_by_track,
              "screen_months": PROTOCOL["screen_months"], "screens": reports,
              "confirmation": confirmation, "fixed_origin_confirmation": fixed_confirmation,
              "accepted": accepted, "current_parents": current_parents,
              "fit_attempts": _state()["fit_attempts"], "preflight": audit}
    target = OUTPUT / "round_6_results.json"
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2), encoding="utf-8")
    os.replace(tmp, target)
    _write_state({"active_stage": "round_6_complete", "round_6_accepted": accepted,
                  "current_parents": current_parents})
    end_batch("round_6_complete")
    return result


def run_round_7() -> dict:
    """Screen LightGBM absolute error on the unchanged transformed price targets."""
    audit = _source_commit_and_hash()
    state = json.loads(STATE.read_text(encoding="utf-8"))
    if state.get("active_stage") not in {"batch_started", "round_7_running"}:
        raise RuntimeError("Round 7 requires an explicitly started bounded batch")
    _write_state({"active_stage": "round_7_running"})
    parent_by_track = dict(state.get("current_parents", {
        "off_market": "G06_lot_garage", "list_aware": "G01"}))
    variant = "G16_l1_log_objective"
    reports, confirmation, fixed_confirmation, accepted = {}, {}, {}, {}
    from data import hide_answers

    ledger_rows = _ledger_rows()
    for track, parent_id in parent_by_track.items():
        if _fit_timed_out(ledger_rows, variant, track):
            reports[track] = {"status": "timeout", "parent_id": parent_id,
                              "qualifies_screen": False,
                              "reason": "This exact objective/track already timed out and is not rerun."}
            continue
        parent_parts, candidate_parts = [], []
        for month in PROTOCOL["screen_months"]:
            _, train, _, truth, heldout = _load_fold(month)
            homes = hide_answers(heldout, track)
            parent = _load_parent_prediction(month, track, homes, truth, parent_id).copy()
            parent["month"] = month
            candidate, _ = _screen_variant(month, track, train, homes, truth, variant,
                                           parent_id=parent_id)
            candidate["month"] = month
            parent_parts.append(parent)
            candidate_parts.append(candidate)
        report = _screen_report(pd.concat(parent_parts, ignore_index=True),
                                pd.concat(candidate_parts, ignore_index=True), PROTOCOL["screen_months"])
        report.update({"status": "complete", "parent_id": parent_id,
                       "model_family": "lightgbm", "objective": "regression_l1"})
        reports[track] = report
        _append({"event": "neighbor_screen_complete", "round": 7, "track": track,
                 "variant": variant, "parent_id": parent_id, **report})
        print(f"round7 screened {variant} {track} ({parent_id}): "
              f"{report['parent']['mdape_percent']:.3f}% -> {report['candidate']['mdape_percent']:.3f}% "
              f"qualified={report['qualifies_screen']}", flush=True)
        if not report["qualifies_screen"]:
            continue

        parent_parts, candidate_parts = [], []
        for month in PROTOCOL["development_months"]:
            _, train, _, truth, heldout = _load_fold(month)
            homes = hide_answers(heldout, track)
            parent = _load_parent_prediction(month, track, homes, truth, parent_id).copy()
            parent["month"] = month
            candidate, fit = _screen_variant(month, track, train, homes, truth, variant,
                                             parent_id=parent_id)
            if fit is not None:
                _save_neighbor_fold(month, track, variant, candidate, fit, parent_id=parent_id)
            candidate["month"] = month
            parent_parts.append(parent)
            candidate_parts.append(candidate)
        confirmation[track] = _development_report(pd.concat(parent_parts, ignore_index=True),
                                                   pd.concat(candidate_parts, ignore_index=True),
                                                   PROTOCOL["development_months"])
        confirmation[track]["parent_id"] = parent_id
        print(f"round7 confirmed {variant} {track}: "
              f"relative={confirmation[track]['relative_mdape_improvement']:.3%}, "
              f"months={confirmation[track]['months_improved']}/8, "
              f"gate={confirmation[track]['passes_development_gate']}", flush=True)
        if confirmation[track]["passes_development_gate"]:
            fixed_confirmation[track] = _fixed_origin_candidate(variant, track, parent_id=parent_id)
            if fixed_confirmation[track]["passes_origin_guardrail"]:
                accepted[track] = variant

    current_parents = dict(parent_by_track)
    current_parents.update(accepted)
    result = {"round": 7, "variant": variant, "parents": parent_by_track,
              "screen_months": PROTOCOL["screen_months"], "screens": reports,
              "confirmation": confirmation, "fixed_origin_confirmation": fixed_confirmation,
              "accepted": accepted, "current_parents": current_parents,
              "fit_attempts": _state()["fit_attempts"], "preflight": audit}
    target = OUTPUT / "round_7_results.json"
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2), encoding="utf-8")
    os.replace(tmp, target)
    _write_state({"active_stage": "round_7_complete", "round_7_accepted": accepted,
                  "current_parents": current_parents})
    end_batch("round_7_complete")
    return result


def run_round_8() -> dict:
    """Screen strict-as-of same-type neighborhood comparable features."""
    audit = _source_commit_and_hash()
    state = json.loads(STATE.read_text(encoding="utf-8"))
    if state.get("active_stage") not in {"batch_started", "round_8_running"}:
        raise RuntimeError("Round 8 requires an explicitly started bounded batch")
    _write_state({"active_stage": "round_8_running"})
    parent_by_track = dict(state.get("current_parents", {
        "off_market": "G06_lot_garage", "list_aware": "G01"}))
    variant = "G17_nearest_comparables"
    reports, confirmation, fixed_confirmation, accepted = {}, {}, {}, {}
    from data import hide_answers

    ledger_rows = _ledger_rows()
    for track, parent_id in parent_by_track.items():
        if _fit_timed_out(ledger_rows, variant, track):
            reports[track] = {"status": "timeout", "parent_id": parent_id,
                              "qualifies_screen": False,
                              "reason": "This exact comparable-feature track already timed out and is not rerun."}
            continue
        parent_parts, candidate_parts = [], []
        for month in PROTOCOL["screen_months"]:
            _, train, _, truth, heldout = _load_fold(month)
            homes = hide_answers(heldout, track)
            parent = _load_parent_prediction(month, track, homes, truth, parent_id).copy()
            parent["month"] = month
            candidate, _ = _screen_variant(month, track, train, homes, truth, variant,
                                           parent_id=parent_id)
            candidate["month"] = month
            parent_parts.append(parent)
            candidate_parts.append(candidate)
        report = _screen_report(pd.concat(parent_parts, ignore_index=True),
                                pd.concat(candidate_parts, ignore_index=True), PROTOCOL["screen_months"])
        report.update({"status": "complete", "parent_id": parent_id,
                       "mechanism": "strict-as-of same-type nearest-sale residual features"})
        reports[track] = report
        _append({"event": "neighbor_screen_complete", "round": 8, "track": track,
                 "variant": variant, "parent_id": parent_id, **report})
        print(f"round8 screened {variant} {track} ({parent_id}): "
              f"{report['parent']['mdape_percent']:.3f}% -> {report['candidate']['mdape_percent']:.3f}% "
              f"qualified={report['qualifies_screen']}", flush=True)
        if not report["qualifies_screen"]:
            continue

        parent_parts, candidate_parts = [], []
        for month in PROTOCOL["development_months"]:
            _, train, _, truth, heldout = _load_fold(month)
            homes = hide_answers(heldout, track)
            parent = _load_parent_prediction(month, track, homes, truth, parent_id).copy()
            parent["month"] = month
            candidate, fit = _screen_variant(month, track, train, homes, truth, variant,
                                             parent_id=parent_id)
            if fit is not None:
                _save_neighbor_fold(month, track, variant, candidate, fit, parent_id=parent_id)
            candidate["month"] = month
            parent_parts.append(parent)
            candidate_parts.append(candidate)
        confirmation[track] = _development_report(pd.concat(parent_parts, ignore_index=True),
                                                   pd.concat(candidate_parts, ignore_index=True),
                                                   PROTOCOL["development_months"])
        confirmation[track]["parent_id"] = parent_id
        print(f"round8 confirmed {variant} {track}: "
              f"relative={confirmation[track]['relative_mdape_improvement']:.3%}, "
              f"months={confirmation[track]['months_improved']}/8, "
              f"gate={confirmation[track]['passes_development_gate']}", flush=True)
        if confirmation[track]["passes_development_gate"]:
            fixed_confirmation[track] = _fixed_origin_candidate(variant, track, parent_id=parent_id)
            if fixed_confirmation[track]["passes_origin_guardrail"]:
                accepted[track] = variant

    current_parents = dict(parent_by_track)
    current_parents.update(accepted)
    result = {"round": 8, "variant": variant, "parents": parent_by_track,
              "screen_months": PROTOCOL["screen_months"], "screens": reports,
              "confirmation": confirmation, "fixed_origin_confirmation": fixed_confirmation,
              "accepted": accepted, "current_parents": current_parents,
              "fit_attempts": _state()["fit_attempts"], "preflight": audit}
    target = OUTPUT / "round_8_results.json"
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2), encoding="utf-8")
    os.replace(tmp, target)
    _write_state({"active_stage": "round_8_complete", "round_8_accepted": accepted,
                  "current_parents": current_parents})
    end_batch("round_8_complete")
    return result


def run_round_9() -> dict:
    """Screen Detached-only specialists while retaining parent predictions for other types."""
    audit = _source_commit_and_hash()
    state = json.loads(STATE.read_text(encoding="utf-8"))
    if state.get("active_stage") not in {"batch_started", "round_9_running"}:
        raise RuntimeError("Round 9 requires an explicitly started bounded batch")
    _write_state({"active_stage": "round_9_running"})
    parent_by_track = dict(state.get("current_parents", {
        "off_market": "G06_lot_garage", "list_aware": "G01"}))
    variant = "G18_detached_specialist"
    reports, confirmation, fixed_confirmation, accepted = {}, {}, {}, {}
    from data import hide_answers

    ledger_rows = _ledger_rows()
    for track, parent_id in parent_by_track.items():
        if _fit_timed_out(ledger_rows, variant, track):
            reports[track] = {"status": "timeout", "parent_id": parent_id,
                              "qualifies_screen": False,
                              "reason": "This exact Detached specialist track already timed out and is not rerun."}
            continue
        parent_parts, candidate_parts = [], []
        group_counts = {}
        for month in PROTOCOL["screen_months"]:
            _, train, _, truth, heldout = _load_fold(month)
            homes = hide_answers(heldout, track)
            parent = _load_parent_prediction(month, track, homes, truth, parent_id).copy()
            parent["month"] = month
            group_counts[month] = {
                "training_detached": int(_specialist_mask(train, "Detached").sum()),
                "evaluation_detached": int(_specialist_mask(homes, "Detached").sum()),
            }
            candidate, _ = _screen_variant(month, track, train, homes, truth, variant,
                                           parent_id=parent_id)
            candidate["month"] = month
            parent_parts.append(parent)
            candidate_parts.append(candidate)
        report = _screen_report(pd.concat(parent_parts, ignore_index=True),
                                pd.concat(candidate_parts, ignore_index=True), PROTOCOL["screen_months"])
        report.update({"status": "complete", "parent_id": parent_id,
                       "mechanism": "Detached-only LightGBM specialist with parent fallback",
                       "group_counts": group_counts})
        reports[track] = report
        _append({"event": "neighbor_screen_complete", "round": 9, "track": track,
                 "variant": variant, "parent_id": parent_id, **report})
        print(f"round9 screened {variant} {track} ({parent_id}): "
              f"{report['parent']['mdape_percent']:.3f}% -> {report['candidate']['mdape_percent']:.3f}% "
              f"qualified={report['qualifies_screen']}", flush=True)
        if not report["qualifies_screen"]:
            continue

        parent_parts, candidate_parts = [], []
        for month in PROTOCOL["development_months"]:
            _, train, _, truth, heldout = _load_fold(month)
            homes = hide_answers(heldout, track)
            parent = _load_parent_prediction(month, track, homes, truth, parent_id).copy()
            parent["month"] = month
            candidate, fit = _screen_variant(month, track, train, homes, truth, variant,
                                             parent_id=parent_id)
            if fit is not None:
                _save_neighbor_fold(month, track, variant, candidate, fit, parent_id=parent_id)
            candidate["month"] = month
            parent_parts.append(parent)
            candidate_parts.append(candidate)
        confirmation[track] = _development_report(pd.concat(parent_parts, ignore_index=True),
                                                   pd.concat(candidate_parts, ignore_index=True),
                                                   PROTOCOL["development_months"])
        confirmation[track]["parent_id"] = parent_id
        print(f"round9 confirmed {variant} {track}: "
              f"relative={confirmation[track]['relative_mdape_improvement']:.3%}, "
              f"months={confirmation[track]['months_improved']}/8, "
              f"gate={confirmation[track]['passes_development_gate']}", flush=True)
        if confirmation[track]["passes_development_gate"]:
            fixed_confirmation[track] = _fixed_origin_candidate(variant, track, parent_id=parent_id)
            if fixed_confirmation[track]["passes_origin_guardrail"]:
                accepted[track] = variant

    current_parents = dict(parent_by_track)
    current_parents.update(accepted)
    result = {"round": 9, "variant": variant, "parents": parent_by_track,
              "screen_months": PROTOCOL["screen_months"], "screens": reports,
              "confirmation": confirmation, "fixed_origin_confirmation": fixed_confirmation,
              "accepted": accepted, "current_parents": current_parents,
              "fit_attempts": _state()["fit_attempts"], "preflight": audit}
    target = OUTPUT / "round_9_results.json"
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2), encoding="utf-8")
    os.replace(tmp, target)
    _write_state({"active_stage": "round_9_complete", "round_9_accepted": accepted,
                  "current_parents": current_parents})
    end_batch("round_9_complete")
    return result


def run_round_10() -> dict:
    """Compare fixed shrinkage weights for the existing Detached specialist predictions."""
    audit = _source_commit_and_hash()
    state = json.loads(STATE.read_text(encoding="utf-8"))
    if state.get("active_stage") not in {"batch_started", "round_10_running"}:
        raise RuntimeError("Round 10 requires an explicitly started bounded batch")
    _write_state({"active_stage": "round_10_running"})
    parent_by_track = dict(state.get("current_parents", {
        "off_market": "G06_lot_garage", "list_aware": "G01"}))
    round_spec = next(item for item in PROTOCOL["candidate_search"]["rounds"] if item["round"] == 10)
    variants = round_spec["neighbors"]
    reports, confirmation, fixed_confirmation, accepted, selected = {}, {}, {}, {}, {}
    from data import hide_answers

    ledger_rows = _ledger_rows()
    for track, parent_id in parent_by_track.items():
        track_reports = {}
        for variant in variants:
            if _fit_timed_out(ledger_rows, variant, track):
                track_reports[variant] = {"status": "timeout", "parent_id": parent_id,
                                          "qualifies_screen": False,
                                          "reason": "This exact post-processing candidate already timed out and is not rerun."}
                continue
            parent_parts, candidate_parts = [], []
            for month in PROTOCOL["screen_months"]:
                _, train, _, truth, heldout = _load_fold(month)
                homes = hide_answers(heldout, track)
                parent = _load_parent_prediction(month, track, homes, truth, parent_id).copy()
                parent["month"] = month
                candidate, _ = _screen_variant(month, track, train, homes, truth, variant,
                                               parent_id=parent_id)
                candidate["month"] = month
                parent_parts.append(parent)
                candidate_parts.append(candidate)
            report = _screen_report(pd.concat(parent_parts, ignore_index=True),
                                    pd.concat(candidate_parts, ignore_index=True), PROTOCOL["screen_months"])
            report.update({"status": "complete", "parent_id": parent_id,
                           "mechanism": "fixed parent/specialist shrinkage on Detached rows",
                           "specialist_blend_weight": VARIANTS[variant]["specialist_blend_weight"],
                           "fit_attempts_added": 0})
            track_reports[variant] = report
            _append({"event": "neighbor_screen_complete", "round": 10, "track": track,
                     "variant": variant, "parent_id": parent_id, **report})
            print(f"round10 screened {variant} {track}: "
                  f"{report['parent']['mdape_percent']:.3f}% -> {report['candidate']['mdape_percent']:.3f}% "
                  f"qualified={report['qualifies_screen']}", flush=True)
        reports[track] = track_reports
        qualifying = [(variant, report) for variant, report in track_reports.items()
                      if report.get("status") == "complete" and report.get("qualifies_screen")]
        if not qualifying:
            continue
        chosen, chosen_screen = min(qualifying,
                                    key=lambda item: (item[1]["candidate"]["mdape_percent"],
                                                      item[1]["candidate"]["mape_percent"], item[0]))
        selected[track] = chosen
        parent_parts, candidate_parts = [], []
        for month in PROTOCOL["development_months"]:
            _, train, _, truth, heldout = _load_fold(month)
            homes = hide_answers(heldout, track)
            parent = _load_parent_prediction(month, track, homes, truth, parent_id).copy()
            parent["month"] = month
            candidate, fit = _screen_variant(month, track, train, homes, truth, chosen,
                                             parent_id=parent_id)
            if fit is not None:
                _save_neighbor_fold(month, track, chosen, candidate, fit, parent_id=parent_id)
            candidate["month"] = month
            parent_parts.append(parent)
            candidate_parts.append(candidate)
        confirmation[track] = _development_report(pd.concat(parent_parts, ignore_index=True),
                                                   pd.concat(candidate_parts, ignore_index=True),
                                                   PROTOCOL["development_months"])
        confirmation[track].update({"parent_id": parent_id, "selected_variant": chosen,
                                    "screen_relative_mdape_improvement": chosen_screen["relative_mdape_improvement"],
                                    "fit_attempts_added": 0})
        print(f"round10 confirmed {chosen} {track}: "
              f"relative={confirmation[track]['relative_mdape_improvement']:.3%}, "
              f"months={confirmation[track]['months_improved']}/8, "
              f"gate={confirmation[track]['passes_development_gate']}", flush=True)
        if confirmation[track]["passes_development_gate"]:
            fixed_confirmation[track] = _fixed_origin_candidate(chosen, track, parent_id=parent_id)
            if fixed_confirmation[track]["passes_origin_guardrail"]:
                accepted[track] = chosen

    current_parents = dict(parent_by_track)
    current_parents.update(accepted)
    result = {"round": 10, "variants": variants, "parents": parent_by_track,
              "screen_months": PROTOCOL["screen_months"], "screens": reports,
              "selected_by_track": selected, "confirmation": confirmation,
              "fixed_origin_confirmation": fixed_confirmation, "accepted": accepted,
              "current_parents": current_parents, "fit_attempts": _state()["fit_attempts"],
              "preflight": audit}
    target = OUTPUT / "round_10_results.json"
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2), encoding="utf-8")
    os.replace(tmp, target)
    _write_state({"active_stage": "round_10_complete", "round_10_accepted": accepted,
                  "current_parents": current_parents})
    end_batch("round_10_complete")
    return result


def run_round_11() -> dict:
    """Test fixed list-price ratio shrinkage on the list-aware track only."""
    audit = _source_commit_and_hash()
    state = json.loads(STATE.read_text(encoding="utf-8"))
    if state.get("active_stage") not in {"batch_started", "round_11_running"}:
        raise RuntimeError("Round 11 requires an explicitly started bounded batch")
    _write_state({"active_stage": "round_11_running"})
    parent_by_track = dict(state.get("current_parents", {
        "off_market": "G06_lot_garage", "list_aware": "G01"}))
    round_spec = next(item for item in PROTOCOL["candidate_search"]["rounds"] if item["round"] == 11)
    variants = round_spec["neighbors"]
    reports, confirmation, fixed_confirmation, accepted, selected = {}, {}, {}, {}, {}
    from data import hide_answers

    for track, parent_id in parent_by_track.items():
        if track != "list_aware":
            reports[track] = {"status": "not_applicable", "reason": "list_price is prohibited for off-market"}
            continue
        track_reports = {}
        for variant in variants:
            parent_parts, candidate_parts = [], []
            for month in PROTOCOL["screen_months"]:
                _, train, _, truth, heldout = _load_fold(month)
                homes = hide_answers(heldout, track)
                if "list_price" not in homes or not np.isfinite(homes.list_price.to_numpy(float)).all():
                    raise RuntimeError(f"list-aware list_price input is missing or invalid in {month}")
                parent = _load_parent_prediction(month, track, homes, truth, parent_id).copy()
                parent["month"] = month
                candidate, _ = _screen_variant(month, track, train, homes, truth, variant,
                                               parent_id=parent_id)
                candidate["month"] = month
                parent_parts.append(parent)
                candidate_parts.append(candidate)
            report = _screen_report(pd.concat(parent_parts, ignore_index=True),
                                    pd.concat(candidate_parts, ignore_index=True), PROTOCOL["screen_months"])
            report.update({"status": "complete", "parent_id": parent_id,
                           "mechanism": "log sold/list ratio shrinkage toward list price",
                           "list_anchor_alpha": VARIANTS[variant]["list_anchor_alpha"],
                           "fit_attempts_added": 0})
            track_reports[variant] = report
            _append({"event": "neighbor_screen_complete", "round": 11, "track": track,
                     "variant": variant, "parent_id": parent_id, **report})
            print(f"round11 screened {variant}: {report['parent']['mdape_percent']:.3f}% -> "
                  f"{report['candidate']['mdape_percent']:.3f}% "
                  f"qualified={report['qualifies_screen']}", flush=True)
        reports[track] = track_reports
        qualifying = [(variant, report) for variant, report in track_reports.items()
                      if report.get("qualifies_screen")]
        if not qualifying:
            continue
        chosen, chosen_screen = min(qualifying,
                                    key=lambda item: (item[1]["candidate"]["mdape_percent"],
                                                      item[1]["candidate"]["mape_percent"], item[0]))
        selected[track] = chosen
        parent_parts, candidate_parts = [], []
        for month in PROTOCOL["development_months"]:
            _, train, _, truth, heldout = _load_fold(month)
            homes = hide_answers(heldout, track)
            parent = _load_parent_prediction(month, track, homes, truth, parent_id).copy()
            parent["month"] = month
            candidate, fit = _screen_variant(month, track, train, homes, truth, chosen,
                                             parent_id=parent_id)
            if fit is not None:
                _save_neighbor_fold(month, track, chosen, candidate, fit, parent_id=parent_id)
            candidate["month"] = month
            parent_parts.append(parent)
            candidate_parts.append(candidate)
        confirmation[track] = _development_report(pd.concat(parent_parts, ignore_index=True),
                                                   pd.concat(candidate_parts, ignore_index=True),
                                                   PROTOCOL["development_months"])
        confirmation[track].update({"parent_id": parent_id, "selected_variant": chosen,
                                    "screen_relative_mdape_improvement": chosen_screen["relative_mdape_improvement"],
                                    "fit_attempts_added": 0})
        print(f"round11 confirmed {chosen}: relative="
              f"{confirmation[track]['relative_mdape_improvement']:.3%}, "
              f"months={confirmation[track]['months_improved']}/8, "
              f"gate={confirmation[track]['passes_development_gate']}", flush=True)
        if confirmation[track]["passes_development_gate"]:
            fixed_confirmation[track] = _fixed_origin_candidate(chosen, track, parent_id=parent_id)
            if fixed_confirmation[track]["passes_origin_guardrail"]:
                accepted[track] = chosen

    current_parents = dict(parent_by_track)
    current_parents.update(accepted)
    result = {"round": 11, "variants": variants, "parents": parent_by_track,
              "screen_months": PROTOCOL["screen_months"], "screens": reports,
              "selected_by_track": selected, "confirmation": confirmation,
              "fixed_origin_confirmation": fixed_confirmation, "accepted": accepted,
              "current_parents": current_parents, "fit_attempts": _state()["fit_attempts"],
              "preflight": audit}
    target = OUTPUT / "round_11_results.json"
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2), encoding="utf-8")
    os.replace(tmp, target)
    _write_state({"active_stage": "round_11_complete", "round_11_accepted": accepted,
                  "current_parents": current_parents})
    end_batch("round_11_complete")
    return result


def run_round_12() -> dict:
    """Screen a list-aware Semi-Detached specialist with parent fallback."""
    audit = _source_commit_and_hash()
    state = json.loads(STATE.read_text(encoding="utf-8"))
    if state.get("active_stage") not in {"batch_started", "round_12_running"}:
        raise RuntimeError("Round 12 requires an explicitly started bounded batch")
    _write_state({"active_stage": "round_12_running"})
    parent_by_track = dict(state.get("current_parents", {
        "off_market": "G06_lot_garage", "list_aware": "G01"}))
    variant = "G21_list_aware_semi_detached_specialist"
    track = "list_aware"
    parent_id = parent_by_track[track]
    reports, confirmation, fixed_confirmation, accepted = {}, {}, {}, {}
    from data import hide_answers

    parent_parts, candidate_parts = [], []
    group_counts = {}
    for month in PROTOCOL["screen_months"]:
        _, train, _, truth, heldout = _load_fold(month)
        homes = hide_answers(heldout, track)
        parent = _load_parent_prediction(month, track, homes, truth, parent_id).copy()
        parent["month"] = month
        group_counts[month] = {
            "training_semi_detached": int(_specialist_mask(train, "Semi-Detached").sum()),
            "evaluation_semi_detached": int(_specialist_mask(homes, "Semi-Detached").sum()),
        }
        candidate, _ = _screen_variant(month, track, train, homes, truth, variant,
                                       parent_id=parent_id)
        candidate["month"] = month
        parent_parts.append(parent)
        candidate_parts.append(candidate)
    report = _screen_report(pd.concat(parent_parts, ignore_index=True),
                            pd.concat(candidate_parts, ignore_index=True), PROTOCOL["screen_months"])
    report.update({"status": "complete", "parent_id": parent_id,
                   "mechanism": "list-aware Semi-Detached-only LightGBM specialist with parent fallback",
                   "group_counts": group_counts})
    _append({"event": "neighbor_screen_complete", "round": 12, "track": track,
             "variant": variant, "parent_id": parent_id, **report})
    print(f"round12 screened {variant}: {report['parent']['mdape_percent']:.3f}% -> "
          f"{report['candidate']['mdape_percent']:.3f}% qualified={report['qualifies_screen']}", flush=True)
    reports[track] = report
    if report["qualifies_screen"]:
        parent_parts, candidate_parts = [], []
        for month in PROTOCOL["development_months"]:
            _, train, _, truth, heldout = _load_fold(month)
            homes = hide_answers(heldout, track)
            parent = _load_parent_prediction(month, track, homes, truth, parent_id).copy()
            parent["month"] = month
            candidate, fit = _screen_variant(month, track, train, homes, truth, variant,
                                             parent_id=parent_id)
            if fit is not None:
                _save_neighbor_fold(month, track, variant, candidate, fit, parent_id=parent_id)
            candidate["month"] = month
            parent_parts.append(parent)
            candidate_parts.append(candidate)
        confirmation[track] = _development_report(pd.concat(parent_parts, ignore_index=True),
                                                   pd.concat(candidate_parts, ignore_index=True),
                                                   PROTOCOL["development_months"])
        confirmation[track]["parent_id"] = parent_id
        print(f"round12 confirmed {variant}: relative="
              f"{confirmation[track]['relative_mdape_improvement']:.3%}, "
              f"months={confirmation[track]['months_improved']}/8, "
              f"gate={confirmation[track]['passes_development_gate']}", flush=True)
        if confirmation[track]["passes_development_gate"]:
            fixed_confirmation[track] = _fixed_origin_candidate(variant, track, parent_id=parent_id)
            if fixed_confirmation[track]["passes_origin_guardrail"]:
                accepted[track] = variant

    current_parents = dict(parent_by_track)
    current_parents.update(accepted)
    result = {"round": 12, "variant": variant, "parents": parent_by_track,
              "screen_months": PROTOCOL["screen_months"], "screens": reports,
              "confirmation": confirmation, "fixed_origin_confirmation": fixed_confirmation,
              "accepted": accepted, "current_parents": current_parents,
              "fit_attempts": _state()["fit_attempts"], "preflight": audit}
    target = OUTPUT / "round_12_results.json"
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2), encoding="utf-8")
    os.replace(tmp, target)
    _write_state({"active_stage": "round_12_complete", "round_12_accepted": accepted,
                  "current_parents": current_parents})
    end_batch("round_12_complete")
    return result


def run_round_13() -> dict:
    """Screen an off-market Semi-Detached specialist on the accepted feature lineage."""
    audit = _source_commit_and_hash()
    state = json.loads(STATE.read_text(encoding="utf-8"))
    if state.get("active_stage") not in {"batch_started", "round_13_running"}:
        raise RuntimeError("Round 13 requires an explicitly started bounded batch")
    _write_state({"active_stage": "round_13_running"})
    parent_by_track = dict(state.get("current_parents", {
        "off_market": "G19_detached_shrink_075", "list_aware": "G01"}))
    variant = "G22_off_market_semi_detached_specialist"
    track = "off_market"
    parent_id = parent_by_track[track]
    reports, confirmation, fixed_confirmation, accepted = {}, {}, {}, {}
    from data import hide_answers

    parent_parts, candidate_parts = [], []
    group_counts = {}
    for month in PROTOCOL["screen_months"]:
        _, train, _, truth, heldout = _load_fold(month)
        homes = hide_answers(heldout, track)
        parent = _load_parent_prediction(month, track, homes, truth, parent_id).copy()
        parent["month"] = month
        group_counts[month] = {
            "training_semi_detached": int(_specialist_mask(train, "Semi-Detached").sum()),
            "evaluation_semi_detached": int(_specialist_mask(homes, "Semi-Detached").sum()),
        }
        candidate, _ = _screen_variant(month, track, train, homes, truth, variant,
                                       parent_id=parent_id)
        candidate["month"] = month
        parent_parts.append(parent)
        candidate_parts.append(candidate)
    report = _screen_report(pd.concat(parent_parts, ignore_index=True),
                            pd.concat(candidate_parts, ignore_index=True), PROTOCOL["screen_months"])
    report.update({"status": "complete", "parent_id": parent_id,
                   "feature_parent": _feature_parent(parent_id),
                   "mechanism": "off-market Semi-Detached-only LightGBM specialist with parent fallback",
                   "group_counts": group_counts})
    _append({"event": "neighbor_screen_complete", "round": 13, "track": track,
             "variant": variant, "parent_id": parent_id, **report})
    print(f"round13 screened {variant}: {report['parent']['mdape_percent']:.3f}% -> "
          f"{report['candidate']['mdape_percent']:.3f}% qualified={report['qualifies_screen']}", flush=True)
    reports[track] = report
    if report["qualifies_screen"]:
        parent_parts, candidate_parts = [], []
        for month in PROTOCOL["development_months"]:
            _, train, _, truth, heldout = _load_fold(month)
            homes = hide_answers(heldout, track)
            parent = _load_parent_prediction(month, track, homes, truth, parent_id).copy()
            parent["month"] = month
            candidate, fit = _screen_variant(month, track, train, homes, truth, variant,
                                             parent_id=parent_id)
            if fit is not None:
                _save_neighbor_fold(month, track, variant, candidate, fit, parent_id=parent_id)
            candidate["month"] = month
            parent_parts.append(parent)
            candidate_parts.append(candidate)
        confirmation[track] = _development_report(pd.concat(parent_parts, ignore_index=True),
                                                   pd.concat(candidate_parts, ignore_index=True),
                                                   PROTOCOL["development_months"])
        confirmation[track].update({"parent_id": parent_id,
                                    "feature_parent": _feature_parent(parent_id)})
        print(f"round13 confirmed {variant}: relative="
              f"{confirmation[track]['relative_mdape_improvement']:.3%}, "
              f"months={confirmation[track]['months_improved']}/8, "
              f"gate={confirmation[track]['passes_development_gate']}", flush=True)
        if confirmation[track]["passes_development_gate"]:
            fixed_confirmation[track] = _fixed_origin_candidate(variant, track, parent_id=parent_id)
            if fixed_confirmation[track]["passes_origin_guardrail"]:
                accepted[track] = variant

    current_parents = dict(parent_by_track)
    current_parents.update(accepted)
    result = {"round": 13, "variant": variant, "parents": parent_by_track,
              "screen_months": PROTOCOL["screen_months"], "screens": reports,
              "confirmation": confirmation, "fixed_origin_confirmation": fixed_confirmation,
              "accepted": accepted, "current_parents": current_parents,
              "fit_attempts": _state()["fit_attempts"], "preflight": audit}
    target = OUTPUT / "round_13_results.json"
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2), encoding="utf-8")
    os.replace(tmp, target)
    _write_state({"active_stage": "round_13_complete", "round_13_accepted": accepted,
                  "current_parents": current_parents})
    end_batch("round_13_complete")
    return result


def run_round_14() -> dict:
    """Test a small list-aware blend with the existing log-ratio L1 model."""
    audit = _source_commit_and_hash()
    state = json.loads(STATE.read_text(encoding="utf-8"))
    if state.get("active_stage") not in {"batch_started", "round_14_running"}:
        raise RuntimeError("Round 14 requires an explicitly started bounded batch")
    _write_state({"active_stage": "round_14_running"})
    parent_by_track = dict(state.get("current_parents", {
        "off_market": "G19_detached_shrink_075", "list_aware": "G01"}))
    variant = "G23_list_aware_l1_mix_010"
    track = "list_aware"
    parent_id = parent_by_track[track]
    source_id = VARIANTS[variant]["prediction_blend_source"]
    weight = VARIANTS[variant]["prediction_blend_weight"]
    reports, confirmation, fixed_confirmation, accepted = {}, {}, {}, {}
    from data import hide_answers

    parent_parts, candidate_parts = [], []
    for month in PROTOCOL["screen_months"]:
        _, train, _, truth, heldout = _load_fold(month)
        homes = hide_answers(heldout, track)
        parent = _load_parent_prediction(month, track, homes, truth, parent_id).copy()
        parent["month"] = month
        candidate, _ = _screen_variant(month, track, train, homes, truth, variant,
                                       parent_id=parent_id)
        candidate["month"] = month
        parent_parts.append(parent)
        candidate_parts.append(candidate)
    report = _screen_report(pd.concat(parent_parts, ignore_index=True),
                            pd.concat(candidate_parts, ignore_index=True), PROTOCOL["screen_months"])
    report.update({"status": "complete", "parent_id": parent_id,
                   "source_variant": source_id, "prediction_blend_weight": weight,
                   "screen_parameter_source": "exploratory weight scan on these same folds; optimistic"})
    _append({"event": "neighbor_screen_complete", "round": 14, "track": track,
             "variant": variant, "parent_id": parent_id, **report})
    print(f"round14 screened {variant}: {report['parent']['mdape_percent']:.3f}% -> "
          f"{report['candidate']['mdape_percent']:.3f}% qualified={report['qualifies_screen']}", flush=True)
    reports[track] = report
    if report["qualifies_screen"]:
        parent_parts, candidate_parts = [], []
        for month in PROTOCOL["development_months"]:
            _, train, _, truth, heldout = _load_fold(month)
            homes = hide_answers(heldout, track)
            parent = _load_parent_prediction(month, track, homes, truth, parent_id).copy()
            parent["month"] = month
            candidate, _ = _screen_variant(month, track, train, homes, truth, variant,
                                           parent_id=parent_id)
            candidate["month"] = month
            parent_parts.append(parent)
            candidate_parts.append(candidate)
        confirmation[track] = _development_report(pd.concat(parent_parts, ignore_index=True),
                                                   pd.concat(candidate_parts, ignore_index=True),
                                                   PROTOCOL["development_months"])
        confirmation[track].update({"parent_id": parent_id, "source_variant": source_id,
                                    "prediction_blend_weight": weight})
        print(f"round14 confirmed {variant}: relative="
              f"{confirmation[track]['relative_mdape_improvement']:.3%}, "
              f"months={confirmation[track]['months_improved']}/8, "
              f"gate={confirmation[track]['passes_development_gate']}", flush=True)
        if confirmation[track]["passes_development_gate"]:
            fixed_confirmation[track] = _fixed_origin_candidate(variant, track, parent_id=parent_id)
            if fixed_confirmation[track]["passes_origin_guardrail"]:
                accepted[track] = variant

    current_parents = dict(parent_by_track)
    current_parents.update(accepted)
    result = {"round": 14, "variant": variant, "parents": parent_by_track,
              "screen_months": PROTOCOL["screen_months"], "screens": reports,
              "confirmation": confirmation, "fixed_origin_confirmation": fixed_confirmation,
              "accepted": accepted, "current_parents": current_parents,
              "fit_attempts": _state()["fit_attempts"], "preflight": audit}
    target = OUTPUT / "round_14_results.json"
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2), encoding="utf-8")
    os.replace(tmp, target)
    _write_state({"active_stage": "round_14_complete", "round_14_accepted": accepted,
                  "current_parents": current_parents})
    end_batch("round_14_complete")
    return result


def run_round_15() -> dict:
    """Tune the off-market Detached specialist's capacity on the current G19 router."""
    audit = _source_commit_and_hash()
    state = json.loads(STATE.read_text(encoding="utf-8"))
    if state.get("active_stage") not in {"batch_started", "round_15_running"}:
        raise RuntimeError("Round 15 requires an explicitly started bounded batch")
    _write_state({"active_stage": "round_15_running"})
    parent_by_track = dict(state.get("current_parents", {
        "off_market": "G19_detached_shrink_075", "list_aware": "G01"}))
    variant = "G24_off_market_detached_leaves63"
    track = "off_market"
    parent_id = parent_by_track[track]
    reports, confirmation, fixed_confirmation, accepted = {}, {}, {}, {}
    from data import hide_answers

    parent_parts, candidate_parts = [], []
    group_counts = {}
    for month in PROTOCOL["screen_months"]:
        _, train, _, truth, heldout = _load_fold(month)
        homes = hide_answers(heldout, track)
        parent = _load_parent_prediction(month, track, homes, truth, parent_id).copy()
        parent["month"] = month
        group_counts[month] = {
            "training_detached": int(_specialist_mask(train, "Detached").sum()),
            "evaluation_detached": int(_specialist_mask(homes, "Detached").sum()),
        }
        candidate, _ = _screen_variant(month, track, train, homes, truth, variant,
                                       parent_id=parent_id)
        candidate["month"] = month
        parent_parts.append(parent)
        candidate_parts.append(candidate)
    report = _screen_report(pd.concat(parent_parts, ignore_index=True),
                            pd.concat(candidate_parts, ignore_index=True), PROTOCOL["screen_months"])
    report.update({"status": "complete", "parent_id": parent_id,
                   "prediction_base_id": VARIANTS[variant]["prediction_base"],
                   "feature_parent": _feature_parent(parent_id),
                   "mechanism": "Detached-only LightGBM specialist, 63 leaves, 75% blend; scored against current G19",
                   "group_counts": group_counts})
    _append({"event": "neighbor_screen_complete", "round": 15, "track": track,
             "variant": variant, "parent_id": parent_id, **report})
    print(f"round15 screened {variant}: {report['parent']['mdape_percent']:.3f}% -> "
          f"{report['candidate']['mdape_percent']:.3f}% qualified={report['qualifies_screen']}", flush=True)
    reports[track] = report
    if report["qualifies_screen"]:
        parent_parts, candidate_parts = [], []
        for month in PROTOCOL["development_months"]:
            _, train, _, truth, heldout = _load_fold(month)
            homes = hide_answers(heldout, track)
            parent = _load_parent_prediction(month, track, homes, truth, parent_id).copy()
            parent["month"] = month
            candidate, _ = _screen_variant(month, track, train, homes, truth, variant,
                                           parent_id=parent_id)
            candidate["month"] = month
            parent_parts.append(parent)
            candidate_parts.append(candidate)
        confirmation[track] = _development_report(pd.concat(parent_parts, ignore_index=True),
                                                   pd.concat(candidate_parts, ignore_index=True),
                                                   PROTOCOL["development_months"])
        confirmation[track].update({"parent_id": parent_id,
                                    "prediction_base_id": VARIANTS[variant]["prediction_base"],
                                    "feature_parent": _feature_parent(parent_id)})
        print(f"round15 confirmed {variant}: relative="
              f"{confirmation[track]['relative_mdape_improvement']:.3%}, "
              f"months={confirmation[track]['months_improved']}/8, "
              f"gate={confirmation[track]['passes_development_gate']}", flush=True)
        if confirmation[track]["passes_development_gate"]:
            fixed_confirmation[track] = _fixed_origin_candidate(variant, track, parent_id=parent_id)
            if fixed_confirmation[track]["passes_origin_guardrail"]:
                accepted[track] = variant

    current_parents = dict(parent_by_track)
    current_parents.update(accepted)
    result = {"round": 15, "variant": variant, "parents": parent_by_track,
              "screen_months": PROTOCOL["screen_months"], "screens": reports,
              "confirmation": confirmation, "fixed_origin_confirmation": fixed_confirmation,
              "accepted": accepted, "current_parents": current_parents,
              "fit_attempts": _state()["fit_attempts"], "preflight": audit}
    target = OUTPUT / "round_15_results.json"
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2), encoding="utf-8")
    os.replace(tmp, target)
    _write_state({"active_stage": "round_15_complete", "round_15_accepted": accepted,
                  "current_parents": current_parents})
    end_batch("round_15_complete")
    return result


def run_round_16() -> dict:
    """Test one fixed Huber-loss neighbor on the list-aware track."""
    audit = _source_commit_and_hash()
    state = json.loads(STATE.read_text(encoding="utf-8"))
    if state.get("active_stage") not in {"batch_started", "round_16_running"}:
        raise RuntimeError("Round 16 requires an explicitly started bounded batch")
    _write_state({"active_stage": "round_16_running"})
    parent_by_track = dict(state.get("current_parents", {
        "off_market": "G19_detached_shrink_075", "list_aware": "G01"}))
    variant = "G25_list_aware_huber_objective"
    track = "list_aware"
    parent_id = parent_by_track[track]
    reports, confirmation, fixed_confirmation, accepted = {}, {}, {}, {}
    from data import hide_answers

    parent_parts, candidate_parts = [], []
    for month in PROTOCOL["screen_months"]:
        _, train, _, truth, heldout = _load_fold(month)
        homes = hide_answers(heldout, track)
        parent = _load_parent_prediction(month, track, homes, truth, parent_id).copy()
        parent["month"] = month
        candidate, _ = _screen_variant(month, track, train, homes, truth, variant,
                                       parent_id=parent_id)
        candidate["month"] = month
        parent_parts.append(parent)
        candidate_parts.append(candidate)
    report = _screen_report(pd.concat(parent_parts, ignore_index=True),
                            pd.concat(candidate_parts, ignore_index=True), PROTOCOL["screen_months"])
    report.update({"status": "complete", "parent_id": parent_id,
                   "objective": "huber", "alpha": 0.9})
    _append({"event": "neighbor_screen_complete", "round": 16, "track": track,
             "variant": variant, "parent_id": parent_id, **report})
    print(f"round16 screened {variant}: {report['parent']['mdape_percent']:.3f}% -> "
          f"{report['candidate']['mdape_percent']:.3f}% qualified={report['qualifies_screen']}", flush=True)
    reports[track] = report
    if report["qualifies_screen"]:
        parent_parts, candidate_parts = [], []
        for month in PROTOCOL["development_months"]:
            _, train, _, truth, heldout = _load_fold(month)
            homes = hide_answers(heldout, track)
            parent = _load_parent_prediction(month, track, homes, truth, parent_id).copy()
            parent["month"] = month
            candidate, _ = _screen_variant(month, track, train, homes, truth, variant,
                                           parent_id=parent_id)
            candidate["month"] = month
            parent_parts.append(parent)
            candidate_parts.append(candidate)
        confirmation[track] = _development_report(pd.concat(parent_parts, ignore_index=True),
                                                   pd.concat(candidate_parts, ignore_index=True),
                                                   PROTOCOL["development_months"])
        confirmation[track].update({"parent_id": parent_id, "objective": "huber", "alpha": 0.9})
        print(f"round16 confirmed {variant}: relative="
              f"{confirmation[track]['relative_mdape_improvement']:.3%}, "
              f"months={confirmation[track]['months_improved']}/8, "
              f"gate={confirmation[track]['passes_development_gate']}", flush=True)
        if confirmation[track]["passes_development_gate"]:
            fixed_confirmation[track] = _fixed_origin_candidate(variant, track, parent_id=parent_id)
            if fixed_confirmation[track]["passes_origin_guardrail"]:
                accepted[track] = variant

    current_parents = dict(parent_by_track)
    current_parents.update(accepted)
    result = {"round": 16, "variant": variant, "parents": parent_by_track,
              "screen_months": PROTOCOL["screen_months"], "screens": reports,
              "confirmation": confirmation, "fixed_origin_confirmation": fixed_confirmation,
              "accepted": accepted, "current_parents": current_parents,
              "fit_attempts": _state()["fit_attempts"], "preflight": audit}
    target = OUTPUT / "round_16_results.json"
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2), encoding="utf-8")
    os.replace(tmp, target)
    _write_state({"active_stage": "round_16_complete", "round_16_accepted": accepted,
                  "current_parents": current_parents})
    end_batch("round_16_complete")
    return result


def run_round_17() -> dict:
    """Test a median-oriented loss only within the off-market Detached specialist."""
    audit = _source_commit_and_hash()
    state = json.loads(STATE.read_text(encoding="utf-8"))
    if state.get("active_stage") not in {"batch_started", "round_17_running"}:
        raise RuntimeError("Round 17 requires an explicitly started bounded batch")
    _write_state({"active_stage": "round_17_running"})
    parent_by_track = dict(state.get("current_parents", {
        "off_market": "G19_detached_shrink_075", "list_aware": "G01"}))
    variant = "G26_off_market_detached_l1_specialist"
    track = "off_market"
    parent_id = parent_by_track[track]
    reports, confirmation, fixed_confirmation, accepted = {}, {}, {}, {}
    from data import hide_answers

    parent_parts, candidate_parts = [], []
    group_counts = {}
    for month in PROTOCOL["screen_months"]:
        _, train, _, truth, heldout = _load_fold(month)
        homes = hide_answers(heldout, track)
        parent = _load_parent_prediction(month, track, homes, truth, parent_id).copy()
        parent["month"] = month
        group_counts[month] = {
            "training_detached": int(_specialist_mask(train, "Detached").sum()),
            "evaluation_detached": int(_specialist_mask(homes, "Detached").sum()),
        }
        candidate, _ = _screen_variant(month, track, train, homes, truth, variant,
                                       parent_id=parent_id)
        candidate["month"] = month
        parent_parts.append(parent)
        candidate_parts.append(candidate)
    report = _screen_report(pd.concat(parent_parts, ignore_index=True),
                            pd.concat(candidate_parts, ignore_index=True), PROTOCOL["screen_months"])
    report.update({"status": "complete", "parent_id": parent_id,
                   "prediction_base_id": VARIANTS[variant]["prediction_base"],
                   "feature_parent": _feature_parent(parent_id),
                   "objective": "regression_l1", "specialist_blend_weight": 0.75,
                   "group_counts": group_counts})
    _append({"event": "neighbor_screen_complete", "round": 17, "track": track,
             "variant": variant, "parent_id": parent_id, **report})
    print(f"round17 screened {variant}: {report['parent']['mdape_percent']:.3f}% -> "
          f"{report['candidate']['mdape_percent']:.3f}% qualified={report['qualifies_screen']}", flush=True)
    reports[track] = report
    if report["qualifies_screen"]:
        parent_parts, candidate_parts = [], []
        for month in PROTOCOL["development_months"]:
            _, train, _, truth, heldout = _load_fold(month)
            homes = hide_answers(heldout, track)
            parent = _load_parent_prediction(month, track, homes, truth, parent_id).copy()
            parent["month"] = month
            candidate, _ = _screen_variant(month, track, train, homes, truth, variant,
                                           parent_id=parent_id)
            candidate["month"] = month
            parent_parts.append(parent)
            candidate_parts.append(candidate)
        confirmation[track] = _development_report(pd.concat(parent_parts, ignore_index=True),
                                                   pd.concat(candidate_parts, ignore_index=True),
                                                   PROTOCOL["development_months"])
        confirmation[track].update({"parent_id": parent_id,
                                    "prediction_base_id": VARIANTS[variant]["prediction_base"],
                                    "feature_parent": _feature_parent(parent_id),
                                    "objective": "regression_l1", "specialist_blend_weight": 0.75})
        print(f"round17 confirmed {variant}: relative="
              f"{confirmation[track]['relative_mdape_improvement']:.3%}, "
              f"months={confirmation[track]['months_improved']}/8, "
              f"gate={confirmation[track]['passes_development_gate']}", flush=True)
        if confirmation[track]["passes_development_gate"]:
            fixed_confirmation[track] = _fixed_origin_candidate(variant, track, parent_id=parent_id)
            if fixed_confirmation[track]["passes_origin_guardrail"]:
                accepted[track] = variant

    current_parents = dict(parent_by_track)
    current_parents.update(accepted)
    result = {"round": 17, "variant": variant, "parents": parent_by_track,
              "screen_months": PROTOCOL["screen_months"], "screens": reports,
              "confirmation": confirmation, "fixed_origin_confirmation": fixed_confirmation,
              "accepted": accepted, "current_parents": current_parents,
              "fit_attempts": _state()["fit_attempts"], "preflight": audit}
    target = OUTPUT / "round_17_results.json"
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2), encoding="utf-8")
    os.replace(tmp, target)
    _write_state({"active_stage": "round_17_complete", "round_17_accepted": accepted,
                  "current_parents": current_parents})
    end_batch("round_17_complete")
    return result


def run_round_18() -> dict:
    """Test stronger leaf-size regularization inside the off-market Detached specialist."""
    audit = _source_commit_and_hash()
    state = json.loads(STATE.read_text(encoding="utf-8"))
    if state.get("active_stage") not in {"batch_started", "round_18_running"}:
        raise RuntimeError("Round 18 requires an explicitly started bounded batch")
    _write_state({"active_stage": "round_18_running"})
    parent_by_track = dict(state.get("current_parents", {
        "off_market": "G19_detached_shrink_075", "list_aware": "G01"}))
    variant = "G27_off_market_detached_minchild200"
    track = "off_market"
    parent_id = parent_by_track[track]
    reports, confirmation, fixed_confirmation, accepted = {}, {}, {}, {}
    from data import hide_answers

    parent_parts, candidate_parts = [], []
    group_counts = {}
    for month in PROTOCOL["screen_months"]:
        _, train, _, truth, heldout = _load_fold(month)
        homes = hide_answers(heldout, track)
        parent = _load_parent_prediction(month, track, homes, truth, parent_id).copy()
        parent["month"] = month
        group_counts[month] = {
            "training_detached": int(_specialist_mask(train, "Detached").sum()),
            "evaluation_detached": int(_specialist_mask(homes, "Detached").sum()),
        }
        candidate, _ = _screen_variant(month, track, train, homes, truth, variant,
                                       parent_id=parent_id)
        candidate["month"] = month
        parent_parts.append(parent)
        candidate_parts.append(candidate)
    report = _screen_report(pd.concat(parent_parts, ignore_index=True),
                            pd.concat(candidate_parts, ignore_index=True), PROTOCOL["screen_months"])
    report.update({"status": "complete", "parent_id": parent_id,
                   "prediction_base_id": VARIANTS[variant]["prediction_base"],
                   "feature_parent": _feature_parent(parent_id),
                   "model_config_override": {"min_child_samples": 200},
                   "specialist_blend_weight": 0.75, "group_counts": group_counts})
    _append({"event": "neighbor_screen_complete", "round": 18, "track": track,
             "variant": variant, "parent_id": parent_id, **report})
    print(f"round18 screened {variant}: {report['parent']['mdape_percent']:.3f}% -> "
          f"{report['candidate']['mdape_percent']:.3f}% qualified={report['qualifies_screen']}", flush=True)
    reports[track] = report
    if report["qualifies_screen"]:
        parent_parts, candidate_parts = [], []
        for month in PROTOCOL["development_months"]:
            _, train, _, truth, heldout = _load_fold(month)
            homes = hide_answers(heldout, track)
            parent = _load_parent_prediction(month, track, homes, truth, parent_id).copy()
            parent["month"] = month
            candidate, _ = _screen_variant(month, track, train, homes, truth, variant,
                                           parent_id=parent_id)
            candidate["month"] = month
            parent_parts.append(parent)
            candidate_parts.append(candidate)
        confirmation[track] = _development_report(pd.concat(parent_parts, ignore_index=True),
                                                   pd.concat(candidate_parts, ignore_index=True),
                                                   PROTOCOL["development_months"])
        confirmation[track].update({"parent_id": parent_id,
                                    "prediction_base_id": VARIANTS[variant]["prediction_base"],
                                    "feature_parent": _feature_parent(parent_id),
                                    "min_child_samples": 200, "specialist_blend_weight": 0.75})
        print(f"round18 confirmed {variant}: relative="
              f"{confirmation[track]['relative_mdape_improvement']:.3%}, "
              f"months={confirmation[track]['months_improved']}/8, "
              f"gate={confirmation[track]['passes_development_gate']}", flush=True)
        if confirmation[track]["passes_development_gate"]:
            fixed_confirmation[track] = _fixed_origin_candidate(variant, track, parent_id=parent_id)
            if fixed_confirmation[track]["passes_origin_guardrail"]:
                accepted[track] = variant

    current_parents = dict(parent_by_track)
    current_parents.update(accepted)
    result = {"round": 18, "variant": variant, "parents": parent_by_track,
              "screen_months": PROTOCOL["screen_months"], "screens": reports,
              "confirmation": confirmation, "fixed_origin_confirmation": fixed_confirmation,
              "accepted": accepted, "current_parents": current_parents,
              "fit_attempts": _state()["fit_attempts"], "preflight": audit}
    target = OUTPUT / "round_18_results.json"
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2), encoding="utf-8")
    os.replace(tmp, target)
    _write_state({"active_stage": "round_18_complete", "round_18_accepted": accepted,
                  "current_parents": current_parents})
    end_batch("round_18_complete")
    return result


def run_round_19() -> dict:
    """Test a strict-as-of, geographically hierarchical market index in the Detached expert."""
    audit = _source_commit_and_hash()
    state = json.loads(STATE.read_text(encoding="utf-8"))
    if state.get("active_stage") not in {"batch_started", "round_19_running"}:
        raise RuntimeError("Round 19 requires an explicitly started bounded batch")
    _write_state({"active_stage": "round_19_running"})
    parent_by_track = dict(state.get("current_parents", {
        "off_market": "G19_detached_shrink_075", "list_aware": "G01"}))
    track = "off_market"
    parent_id = parent_by_track[track]
    if parent_id != "G19_detached_shrink_075":
        raise RuntimeError(f"Round 19 was preregistered for G19, found parent {parent_id}")
    variant = "G28_off_market_hierarchical_market_index"
    reports, confirmation, fixed_confirmation, accepted = {}, {}, {}, {}
    from data import hide_answers

    parent_parts, candidate_parts = [], []
    group_counts = {}
    for month in PROTOCOL["screen_months"]:
        _, train, _, truth, heldout = _load_fold(month)
        homes = hide_answers(heldout, track)
        parent = _load_parent_prediction(month, track, homes, truth, parent_id).copy()
        parent["month"] = month
        group_counts[month] = {
            "training_detached": int(_specialist_mask(train, "Detached").sum()),
            "evaluation_detached": int(_specialist_mask(homes, "Detached").sum()),
        }
        candidate, fit = _screen_variant(month, track, train, homes, truth, variant,
                                         parent_id=parent_id)
        if fit is not None:
            _save_neighbor_fold(month, track, variant, candidate, fit, parent_id=parent_id)
        candidate["month"] = month
        parent_parts.append(parent)
        candidate_parts.append(candidate)
    report = _screen_report(pd.concat(parent_parts, ignore_index=True),
                            pd.concat(candidate_parts, ignore_index=True), PROTOCOL["screen_months"])
    report.update({"status": "complete", "parent_id": parent_id,
                   "feature_parent": _feature_parent(parent_id),
                   "local_index_window_months": 12,
                   "local_index_shrinkage_count": 20,
                   "specialist_house_type": "Detached", "specialist_blend_weight": 0.75,
                   "group_counts": group_counts})
    reports[track] = report
    _append({"event": "neighbor_screen_complete", "round": 19, "track": track,
             "variant": variant, "parent_id": parent_id, **report})
    print(f"round19 screened {variant}: {report['parent']['mdape_percent']:.3f}% -> "
          f"{report['candidate']['mdape_percent']:.3f}% qualified={report['qualifies_screen']}", flush=True)

    if report["qualifies_screen"]:
        parent_parts, candidate_parts = [], []
        for month in PROTOCOL["development_months"]:
            _, train, _, truth, heldout = _load_fold(month)
            homes = hide_answers(heldout, track)
            parent = _load_parent_prediction(month, track, homes, truth, parent_id).copy()
            parent["month"] = month
            candidate, fit = _screen_variant(month, track, train, homes, truth, variant,
                                             parent_id=parent_id)
            if fit is not None:
                _save_neighbor_fold(month, track, variant, candidate, fit, parent_id=parent_id)
            candidate["month"] = month
            parent_parts.append(parent)
            candidate_parts.append(candidate)
        confirmation[track] = _development_report(pd.concat(parent_parts, ignore_index=True),
                                                   pd.concat(candidate_parts, ignore_index=True),
                                                   PROTOCOL["development_months"])
        confirmation[track].update({"parent_id": parent_id,
                                    "feature_parent": _feature_parent(parent_id),
                                    "local_index_window_months": 12,
                                    "local_index_shrinkage_count": 20,
                                    "specialist_house_type": "Detached",
                                    "specialist_blend_weight": 0.75})
        print(f"round19 confirmed {variant}: relative="
              f"{confirmation[track]['relative_mdape_improvement']:.3%}, "
              f"months={confirmation[track]['months_improved']}/8, "
              f"gate={confirmation[track]['passes_development_gate']}", flush=True)
        if confirmation[track]["passes_development_gate"]:
            fixed_confirmation[track] = _fixed_origin_candidate(variant, track, parent_id=parent_id)
            if fixed_confirmation[track]["passes_origin_guardrail"]:
                accepted[track] = variant

    current_parents = dict(parent_by_track)
    current_parents.update(accepted)
    result = {"round": 19, "variant": variant, "parents": parent_by_track,
              "screen_months": PROTOCOL["screen_months"], "screens": reports,
              "confirmation": confirmation, "fixed_origin_confirmation": fixed_confirmation,
              "accepted": accepted, "current_parents": current_parents,
              "fit_attempts": _state()["fit_attempts"], "preflight": audit}
    target = OUTPUT / "round_19_results.json"
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2), encoding="utf-8")
    os.replace(tmp, target)
    _write_state({"active_stage": "round_19_complete", "round_19_accepted": accepted,
                  "current_parents": current_parents})
    end_batch("round_19_complete")
    return result


def _development_report(parent: pd.DataFrame, candidate: pd.DataFrame, months: list[str]) -> dict:
    p_score, c_score = _score(parent), _score(candidate)
    monthly = {}
    for month in months:
        p = _score(parent.loc[parent.month.astype(str).eq(month)])
        c = _score(candidate.loc[candidate.month.astype(str).eq(month)])
        monthly[month] = {"parent": p, "candidate": c,
                          "relative_mdape_improvement": (p["mdape_percent"] - c["mdape_percent"]) / p["mdape_percent"]}
    relative = (p_score["mdape_percent"] - c_score["mdape_percent"]) / p_score["mdape_percent"]
    mape_change = (c_score["mape_percent"] - p_score["mape_percent"]) / p_score["mape_percent"]
    improved = sum(item["relative_mdape_improvement"] > 0 for item in monthly.values())
    gate = PROTOCOL["development_gate"]
    passes = (relative >= gate["minimum_relative_mdape_improvement"] and improved >= gate["minimum_months_improved"]
              and min(item["relative_mdape_improvement"] for item in monthly.values()) >= -gate["max_month_regression"]
              and mape_change <= gate["max_mape_regression"])
    return {"parent": p_score, "candidate": c_score, "relative_mdape_improvement": relative,
            "relative_mape_change": mape_change, "months_improved": improved, "monthly": monthly,
            "passes_development_gate": bool(passes)}


def _fixed_origin_candidate(variant: str, track: str, *, parent_id: str = "G01") -> dict:
    """Compare one frozen candidate with its selected parent at both origins."""
    from data import hide_answers
    from experiments.model_optimization_v2.pipeline import blend_prices

    origin_rows = {origin: {"parent": [], "candidate": []} for origin in PROTOCOL["fixed_origins"]}
    pooled_parent, pooled_candidate = [], []
    for origin_text in PROTOCOL["fixed_origins"]:
        origin = pd.Timestamp(origin_text)
        end = origin + pd.DateOffset(months=PROTOCOL["fixed_origin_horizons"])
        listings = pd.read_parquet(DATA, filters=[("date_sold", "<", end)])
        listings["date_sold"] = pd.to_datetime(listings.date_sold, errors="coerce")
        train = listings.loc[listings.date_sold < origin].copy()
        heldout = listings.loc[(listings.date_sold >= origin) & (listings.date_sold < end)].copy()
        heldout["ml_num"] = heldout.ml_num.astype(str)
        if train.empty or heldout.empty or not heldout.ml_num.is_unique:
            raise RuntimeError(f"invalid fixed-origin rows for {origin_text}/{track}")
        homes = hide_answers(heldout, track)
        parent_parts = []
        for month, month_rows in heldout.groupby(heldout.date_sold.dt.to_period("M").astype(str), sort=True):
            parent_path = OUTPUT / "predictions" / "fixed_origin" / parent_id / track / f"{origin_text[:7]}__{month}.csv"
            parent_parts.append(pd.read_csv(parent_path, dtype={"ml_num": str}))
        parent_all = pd.concat(parent_parts, ignore_index=True)
        if parent_all.ml_num.astype(str).duplicated().any():
            raise RuntimeError(f"fixed-origin parent has duplicate IDs at {origin_text}/{track}")
        parent_all = parent_all.set_index(parent_all.ml_num.astype(str)).loc[homes.ml_num.astype(str)].reset_index(drop=True)
        parent_all["ml_num"] = homes.ml_num.astype(str).to_numpy()
        if (not np.array_equal(parent_all.ml_num.to_numpy(), homes.ml_num.astype(str).to_numpy())
                or not np.array_equal(parent_all.sold_price.to_numpy(float), heldout.sold_price.to_numpy(float))):
            raise RuntimeError(f"fixed-origin IDs/labels do not align for {origin_text}/{track}")
        variant_spec = VARIANTS.get(variant, {})
        specialist_type = variant_spec.get("specialist_house_type")
        blend_source = variant_spec.get("specialist_blend_source")
        prediction_blend_source = variant_spec.get("prediction_blend_source")
        prediction_base_id = variant_spec.get("prediction_base", parent_id)
        prediction_base_all = parent_all
        if specialist_type and prediction_base_id != parent_id:
            base_parts = []
            for month, month_rows in heldout.groupby(heldout.date_sold.dt.to_period("M").astype(str), sort=True):
                base_path = OUTPUT / "predictions" / "fixed_origin" / prediction_base_id / track / f"{origin_text[:7]}__{month}.csv"
                base_parts.append(pd.read_csv(base_path, dtype={"ml_num": str}))
            prediction_base_all = pd.concat(base_parts, ignore_index=True)
            if prediction_base_all.ml_num.astype(str).duplicated().any():
                raise RuntimeError(f"fixed-origin prediction base has duplicate IDs at {origin_text}/{track}")
            prediction_base_all = prediction_base_all.set_index(prediction_base_all.ml_num.astype(str)).loc[
                homes.ml_num.astype(str)].reset_index(drop=True)
            prediction_base_all["ml_num"] = homes.ml_num.astype(str).to_numpy()
            if (not np.array_equal(prediction_base_all.ml_num.to_numpy(), homes.ml_num.astype(str).to_numpy())
                    or not np.array_equal(prediction_base_all.sold_price.to_numpy(float), heldout.sold_price.to_numpy(float))):
                raise RuntimeError(f"fixed-origin prediction base IDs/labels do not align for {origin_text}/{track}")
        anchor_alpha = variant_spec.get("list_anchor_alpha")
        if anchor_alpha is not None:
            if track != "list_aware":
                raise ValueError("list-price anchoring is allowed only for the list-aware track")
            prediction = _shrink_to_list_anchor(parent_all.predicted_price.to_numpy(float),
                                                homes.list_price.to_numpy(float), anchor_alpha)
        elif prediction_blend_source:
            if track != "list_aware":
                raise ValueError("full-track prediction blending is registered only for list-aware")
            base_fit_tag = f"candidate-origin-{origin_text}"
            source_family = VARIANTS.get(prediction_blend_source, {}).get("model_family", "lightgbm")
            source_family_tag = {"lightgbm": "LGBM", "catboost": "CAT"}.get(source_family, source_family.upper())
            base_fit_id = f"{prediction_blend_source}:{track}:{source_family_tag}:{base_fit_tag}"
            previous_attempts = sum(row.get("event") == "fit_start"
                                    and (row.get("fit_id") == base_fit_id
                                         or str(row.get("fit_id", "")).startswith(base_fit_id + "-retry"))
                                    for row in _ledger_rows())
            fit_tag = f"{base_fit_tag}-retry{previous_attempts}" if previous_attempts else base_fit_tag
            fitted = _fit_history_candidate(month=origin.strftime("%Y-%m"), track=track, train=train,
                                            homes=homes, cutoff=origin, fit_tag=fit_tag,
                                            variant=prediction_blend_source,
                                            base_variant=_feature_parent(parent_id))
            source_prediction = np.asarray(fitted["predictions"], dtype=float)
            frozen = _fit_baseline_component(origin.strftime("%Y-%m"), track, train, homes, "HGBR",
                                             fit_tag=f"fixed-origin-{origin_text}")
            source_prediction = blend_prices(source_prediction, frozen["predictions"], 0.75)
            prediction = _blend_prediction_frames(
                parent_all, pd.DataFrame({"ml_num": homes.ml_num.astype(str).to_numpy(),
                                          "sold_price": heldout.sold_price.to_numpy(float),
                                          "predicted_price": source_prediction}),
                variant_spec["prediction_blend_weight"]).predicted_price.to_numpy(float)
        elif specialist_type:
            specialist_mask = _specialist_mask(homes, specialist_type)
            specialist_train = _specialist_training_rows(train, specialist_type)
            if not specialist_mask.any() or specialist_train.empty:
                raise RuntimeError(f"{specialist_type} specialist has no training or evaluation rows at {origin_text}/{track}")
            fit_variant = blend_source or variant
            base_fit_tag = f"candidate-origin-{origin_text}"
            family = VARIANTS.get(fit_variant, {}).get("model_family", "lightgbm")
            family_tag = {"lightgbm": "LGBM", "catboost": "CAT"}.get(family, family.upper())
            base_fit_id = f"{fit_variant}:{track}:{family_tag}:{base_fit_tag}"
            previous_attempts = sum(row.get("event") == "fit_start"
                                    and (row.get("fit_id") == base_fit_id
                                         or str(row.get("fit_id", "")).startswith(base_fit_id + "-retry"))
                                    for row in _ledger_rows())
            fit_tag = f"{base_fit_tag}-retry{previous_attempts}" if previous_attempts else base_fit_tag
            fitted = _fit_history_candidate(month=origin.strftime("%Y-%m"), track=track, train=train,
                                            homes=homes.loc[specialist_mask].copy(), fit_rows=specialist_train,
                                            cutoff=origin, fit_tag=fit_tag,
                                            variant=fit_variant, base_variant=_feature_parent(parent_id))
            specialist_prediction = np.asarray(fitted["predictions"], dtype=float)
            if track == "list_aware":
                frozen = _fit_baseline_component(origin.strftime("%Y-%m"), track, train, homes, "HGBR",
                                                 fit_tag=f"fixed-origin-{origin_text}")
                specialist_prediction = blend_prices(specialist_prediction,
                                                      frozen["predictions"][specialist_mask], 0.75)
            else:
                specialist_prediction = np.round(specialist_prediction)
            specialist_weight = variant_spec.get("specialist_blend_weight")
            if blend_source:
                prediction = _blend_specialist_predictions(parent_all.predicted_price.to_numpy(float),
                                                            specialist_prediction, homes, specialist_type,
                                                            variant_spec["specialist_blend_weight"])
            elif specialist_weight is not None:
                prediction = _blend_specialist_predictions(prediction_base_all.predicted_price.to_numpy(float),
                                                            specialist_prediction, homes, specialist_type,
                                                            specialist_weight)
            else:
                prediction = _overlay_specialist_predictions(prediction_base_all.predicted_price.to_numpy(float), homes,
                                                               specialist_prediction, specialist_type)
        else:
            fitted = _fit_history_candidate(month=origin.strftime("%Y-%m"), track=track, train=train, homes=homes,
                                            cutoff=origin, fit_tag=f"candidate-origin-{origin_text}", variant=variant,
                                            base_variant=_feature_parent(parent_id))
            prediction = np.asarray(fitted["predictions"], dtype=float)
            if track == "list_aware":
                frozen = _fit_baseline_component(origin.strftime("%Y-%m"), track, train, homes, "HGBR",
                                                 fit_tag=f"fixed-origin-{origin_text}")
                prediction = blend_prices(prediction, frozen["predictions"], 0.75)
            else:
                prediction = np.round(prediction)
        candidate_all = pd.DataFrame({"ml_num": homes.ml_num.astype(str).to_numpy(),
                                      "sold_price": heldout.sold_price.to_numpy(float),
                                      "predicted_price": prediction,
                                      "month": heldout.date_sold.dt.to_period("M").astype(str).to_numpy(),
                                      "origin": origin_text,
                                      "horizon": _month_horizons(heldout.date_sold, origin)})
        candidate_dir = OUTPUT / "predictions" / "fixed_origin" / variant / track
        candidate_dir.mkdir(parents=True, exist_ok=True)
        for month in sorted(candidate_all.month.unique()):
            candidate_all.loc[candidate_all.month.eq(month)].to_csv(candidate_dir / f"{origin_text[:7]}__{month}.csv", index=False)
        origin_rows[origin_text]["parent"].append(parent_all)
        origin_rows[origin_text]["candidate"].append(candidate_all)
        pooled_parent.append(parent_all)
        pooled_candidate.append(candidate_all)

    by_origin = {}
    for origin, arms in origin_rows.items():
        parent = pd.concat(arms["parent"], ignore_index=True)
        candidate = pd.concat(arms["candidate"], ignore_index=True)
        by_origin[origin] = {"parent": _score(parent), "candidate": _score(candidate)}
    parent = pd.concat(pooled_parent, ignore_index=True)
    candidate = pd.concat(pooled_candidate, ignore_index=True)
    p_score, c_score = _score(parent), _score(candidate)
    by_horizon = {}
    for horizon in sorted(candidate.horizon.unique()):
        p = _score(parent.loc[parent.horizon.eq(horizon)])
        c = _score(candidate.loc[candidate.horizon.eq(horizon)])
        by_horizon[str(int(horizon))] = {"parent": p, "candidate": c,
                                         "relative_mdape_change": (c["mdape_percent"] - p["mdape_percent"]) / p["mdape_percent"]}
    mape_change = (c_score["mape_percent"] - p_score["mape_percent"]) / p_score["mape_percent"]
    passes = (c_score["mdape_percent"] <= p_score["mdape_percent"]
              and max(item["relative_mdape_change"] for item in by_horizon.values()) <= 0.05
              and mape_change <= 0.01)
    return {"variant": variant, "parent_id": parent_id, "track": track, "by_origin": by_origin,
            "parent": p_score, "candidate": c_score,
            "relative_mdape_improvement": (p_score["mdape_percent"] - c_score["mdape_percent"]) / p_score["mdape_percent"],
            "relative_mape_change": mape_change, "by_horizon": by_horizon,
            "passes_origin_guardrail": bool(passes)}


def _save_neighbor_fold(month: str, track: str, variant: str, frame: pd.DataFrame, fit: dict,
                        *, parent_id: str = "G01") -> None:
    path = _candidate_path(variant, track, month)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    path.with_suffix(".json").write_text(json.dumps({"month": month, "track": track, "variant": variant,
        "parent_id": parent_id,
        "data_sha256": PROTOCOL["data_sha256"], "code_sha256": _code_hashes(),
        "prediction_sha256": fit["prediction_sha256"], "feature_columns": fit["feature_columns"],
        "model_family": fit.get("model_family", "lightgbm"),
        "model_config": fit.get("model_config", {}),
        "actual_objective": fit.get("actual_objective")}, indent=2), encoding="utf-8")


def run_g01() -> dict:
    audit = _source_commit_and_hash()
    audit["previous_month_audit"] = "outputs/model_research_next/month_audit.json"
    audit["old_preflight_final_labels_opened"] = False
    (OUTPUT / "preflight.json").parent.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "preflight.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    rows = {track: {"baseline": [], "candidate": []} for track in ("off_market", "list_aware")}
    fitted_hgbr = {}
    for month in PROTOCOL["development_months"]:
        _, train, _, truth, heldout = _load_fold(month)
        for track in ("off_market", "list_aware"):
            from data import hide_answers
            homes = hide_answers(heldout, track)
            cached = _old_c0(month, track, homes, truth)
            if cached is not None:
                baseline = cached[["ml_num", "sold_price", "predicted_price"]].copy()
                _append({"event": "baseline_cache_hit", "month": month, "track": track,
                         "source": str((OUTPUT.parent / "model_research_next" / "predictions" / "confirmation" / "pipelines" / track / f"C0__{month}.csv").relative_to(ROOT)),
                         "data_sha256": PROTOCOL["data_sha256"]})
            else:
                lgb_result = _fit_baseline_component(month, track, train, homes, "LGBM")
                pred = lgb_result["predictions"]
                if track == "list_aware":
                    hgbr = _fit_baseline_component(month, track, train, homes, "HGBR")
                    fitted_hgbr[month] = hgbr["predictions"]
                    pred = np.round(0.25 * np.round(hgbr["predictions"]) + 0.75 * pred)
                else:
                    pred = np.round(pred)
                baseline = pd.DataFrame({"ml_num": homes.ml_num.astype(str).to_numpy(),
                                         "sold_price": truth.sold_price.to_numpy(float),
                                         "predicted_price": pred})
            candidate_result = _fit_history_candidate(month, track, train, homes)
            candidate_price = candidate_result["predictions"]
            if track == "list_aware":
                hgbr = _load_hgbr_cache(month)
                if month in fitted_hgbr:
                    frozen_prediction = fitted_hgbr[month]
                elif hgbr is not None:
                    if not np.array_equal(hgbr.ml_num.astype(str).to_numpy(), homes.ml_num.astype(str).to_numpy()):
                        raise RuntimeError(f"cached HGBR does not align in {month}")
                    frozen_prediction = hgbr.raw_prediction.to_numpy(float)
                else:
                    hgb_result = _fit_baseline_component(month, track, train, homes, "HGBR")
                    frozen_prediction = hgb_result["predictions"]
                candidate_price = np.round(0.25 * np.round(frozen_prediction) + 0.75 * candidate_price)
            else:
                candidate_price = np.round(candidate_price)
            candidate = pd.DataFrame({"ml_num": homes.ml_num.astype(str).to_numpy(),
                                      "sold_price": truth.sold_price.to_numpy(float),
                                      "predicted_price": candidate_price})
            if not np.array_equal(baseline.ml_num.astype(str).to_numpy(), candidate.ml_num.astype(str).to_numpy()):
                raise RuntimeError("G01 prediction pairing failed")
            for arm, frame in (("baseline", baseline), ("candidate", candidate)):
                frame = frame.copy()
                frame["month"] = month
                rows[track][arm].append(frame)
            _save_fold(month, track, baseline, candidate, candidate_result)
            print(f"completed {month} {track}: C0={_score(baseline)['mdape_percent']:.3f}% G01={_score(candidate)['mdape_percent']:.3f}%", flush=True)

    results = {}
    for track in ("off_market", "list_aware"):
        baseline = pd.concat(rows[track]["baseline"], ignore_index=True)
        candidate = pd.concat(rows[track]["candidate"], ignore_index=True)
        b = _score(baseline)
        c = _score(candidate)
        monthly = {}
        for month in PROTOCOL["development_months"]:
            month_b = _score(baseline.loc[baseline.month.eq(month)])
            month_c = _score(candidate.loc[candidate.month.eq(month)])
            monthly[month] = {"baseline": month_b["mdape_percent"], "candidate": month_c["mdape_percent"],
                              "relative_improvement": (month_b["mdape_percent"] - month_c["mdape_percent"]) / month_b["mdape_percent"]}
        relative = (b["mdape_percent"] - c["mdape_percent"]) / b["mdape_percent"]
        mape_change = (c["mape_percent"] - b["mape_percent"]) / b["mape_percent"]
        improved = sum(item["relative_improvement"] > 0 for item in monthly.values())
        passes = (relative >= PROTOCOL["development_gate"]["minimum_relative_mdape_improvement"]
                  and improved >= PROTOCOL["development_gate"]["minimum_months_improved"]
                  and min(item["relative_improvement"] for item in monthly.values()) >= -PROTOCOL["development_gate"]["max_month_regression"]
                  and mape_change <= PROTOCOL["development_gate"]["max_mape_regression"])
        results[track] = {"baseline": b, "G01": c, "relative_mdape_improvement": relative,
                          "relative_mape_change": mape_change, "months_improved": improved,
                          "monthly": monthly, "passes_development_gate": bool(passes)}

    result = {"run_id": PROTOCOL["run_id"], "card_id": "G01", "evaluation_months": PROTOCOL["development_months"],
              "results": results, "fit_attempts": _state()["fit_attempts"], "elapsed_minutes": _state()["elapsed_minutes"]}
    target = OUTPUT / "g01_results.json"
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2), encoding="utf-8")
    os.replace(tmp, target)
    state = _state()
    state["active_stage"] = "g01_development_complete"
    state["g01_qualified_tracks"] = [track for track, item in results.items() if item["passes_development_gate"]]
    _write_state({"active_stage": "g01_development_complete",
                  "g01_qualified_tracks": [track for track, item in results.items() if item["passes_development_gate"]]})
    return result


def run_fixed_origins() -> dict:
    """Fit once at each origin, then score four future months without updates."""
    from data import hide_answers
    from experiments.model_optimization_v2.pipeline import blend_prices

    _source_commit_and_hash()
    results = {track: {"baseline": [], "candidate": []} for track in ("off_market", "list_aware")}
    origin_results = {}
    for origin_text in PROTOCOL["fixed_origins"]:
        origin = pd.Timestamp(origin_text)
        end = origin + pd.DateOffset(months=PROTOCOL["fixed_origin_horizons"])
        listings = pd.read_parquet(DATA, filters=[("date_sold", "<", end)])
        listings["date_sold"] = pd.to_datetime(listings.date_sold, errors="coerce")
        train = listings.loc[listings.date_sold < origin].copy()
        heldout = listings.loc[(listings.date_sold >= origin) & (listings.date_sold < end)].copy()
        if train.empty or heldout.empty or not heldout.ml_num.astype(str).is_unique:
            raise RuntimeError(f"invalid fixed-origin rows for {origin_text}")
        heldout["ml_num"] = heldout.ml_num.astype(str)
        for track in ("off_market", "list_aware"):
            homes = hide_answers(heldout, track)
            lgb = _fit_baseline_component(origin.strftime("%Y-%m"), track, train, homes, "LGBM")
            if track == "list_aware":
                hgbr = _fit_baseline_component(origin.strftime("%Y-%m"), track, train, homes, "HGBR")
                c0_prediction = blend_prices(lgb["predictions"], hgbr["predictions"], 0.75)
            else:
                c0_prediction = np.round(lgb["predictions"])
            candidate_fit = _fit_history_candidate(origin.strftime("%Y-%m"), track, train, homes,
                                                    cutoff=origin, fit_tag=f"origin-{origin_text}")
            if track == "list_aware":
                candidate_prediction = blend_prices(candidate_fit["predictions"], hgbr["predictions"], 0.75)
            else:
                candidate_prediction = np.round(candidate_fit["predictions"])
            for month, rows in heldout.groupby(heldout.date_sold.dt.to_period("M").astype(str), sort=True):
                indices = rows.index
                positions = heldout.index.get_indexer(indices)
                baseline = pd.DataFrame({"ml_num": rows.ml_num.astype(str).to_numpy(),
                                         "sold_price": rows.sold_price.to_numpy(float),
                                         "predicted_price": c0_prediction[positions],
                                         "month": month, "origin": origin_text,
                                         "horizon": (pd.Period(month, freq="M") - pd.Period(origin_text[:7], freq="M")).n + 1})
                candidate = baseline.drop(columns="predicted_price").copy()
                candidate["predicted_price"] = candidate_prediction[positions]
                results[track]["baseline"].append(baseline)
                results[track]["candidate"].append(candidate)
                for arm, frame in (("C0", baseline), ("G01", candidate)):
                    path = OUTPUT / "predictions" / "fixed_origin" / arm / track / f"{origin_text[:7]}__{month}.csv"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    frame.to_csv(path, index=False)
        origin_results[origin_text] = {track: _fixed_origin_track_result(results[track], origin_text)
                                       for track in ("off_market", "list_aware")}
        print(f"completed fixed origin {origin_text} through {end:%Y-%m-%d}", flush=True)

    combined = {}
    for track in ("off_market", "list_aware"):
        baseline = pd.concat(results[track]["baseline"], ignore_index=True)
        candidate = pd.concat(results[track]["candidate"], ignore_index=True)
        base_score, candidate_score = _score(baseline), _score(candidate)
        by_horizon = {}
        for horizon in sorted(candidate.horizon.unique()):
            by_horizon[str(int(horizon))] = {"baseline": _score(baseline.loc[baseline.horizon.eq(horizon)]),
                                               "G01": _score(candidate.loc[candidate.horizon.eq(horizon)])}
        combined[track] = {"baseline": base_score, "G01": candidate_score,
                           "relative_mdape_change": (candidate_score["mdape_percent"] - base_score["mdape_percent"]) / base_score["mdape_percent"],
                           "relative_mape_change": (candidate_score["mape_percent"] - base_score["mape_percent"]) / base_score["mape_percent"],
                           "by_horizon": by_horizon,
                           "passes_origin_guardrail": candidate_score["mdape_percent"] <= base_score["mdape_percent"] and all(
                               (item["G01"]["mdape_percent"] - item["baseline"]["mdape_percent"]) / item["baseline"]["mdape_percent"] <= 0.05
                               for item in by_horizon.values())}
    result = {"run_id": PROTOCOL["run_id"], "origins": PROTOCOL["fixed_origins"],
              "horizons": PROTOCOL["fixed_origin_horizons"], "origin_results": origin_results,
              "combined": combined, "fit_attempts": _state()["fit_attempts"]}
    target = OUTPUT / "fixed_origin_results.json"
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    _write_state({"active_stage": "fixed_origin_complete",
                  "fixed_origin_qualified_tracks": [track for track, item in combined.items() if item["passes_origin_guardrail"]]})
    return result


def _fixed_origin_track_result(rows: dict, origin: str) -> dict:
    baseline = pd.concat([frame for frame in rows["baseline"] if frame.origin.iloc[0] == origin], ignore_index=True)
    candidate = pd.concat([frame for frame in rows["candidate"] if frame.origin.iloc[0] == origin], ignore_index=True)
    return {"baseline": _score(baseline), "G01": _score(candidate),
            "by_horizon": {str(int(horizon)): {"baseline": _score(baseline.loc[baseline.horizon.eq(horizon)]),
                                                "G01": _score(candidate.loc[candidate.horizon.eq(horizon)])}
                           for horizon in sorted(candidate.horizon.unique())}}


def _save_fold(month: str, track: str, baseline: pd.DataFrame, candidate: pd.DataFrame, fit: dict) -> None:
    for arm, frame in (("C0", baseline), ("G01", candidate)):
        path = OUTPUT / "predictions" / arm / track / f"{month}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
    metadata = {"month": month, "track": track, "candidate": "G01", "fit_seconds": fit["fit_seconds"],
                "prediction_sha256": fit["prediction_sha256"], "data_sha256": PROTOCOL["data_sha256"],
                "code_sha256": _code_hashes(), "feature_columns": fit["feature_columns"]}
    (OUTPUT / "predictions" / "G01" / track / f"{month}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-g01", action="store_true")
    parser.add_argument("--fixed-origins", action="store_true")
    parser.add_argument("--neighbors", action="store_true")
    parser.add_argument("--round-3", action="store_true")
    parser.add_argument("--round-4", action="store_true")
    parser.add_argument("--round-5", action="store_true")
    parser.add_argument("--round-6", action="store_true")
    parser.add_argument("--round-7", action="store_true")
    parser.add_argument("--round-8", action="store_true")
    parser.add_argument("--round-9", action="store_true")
    parser.add_argument("--round-10", action="store_true")
    parser.add_argument("--round-11", action="store_true")
    parser.add_argument("--round-12", action="store_true")
    parser.add_argument("--round-13", action="store_true")
    parser.add_argument("--round-14", action="store_true")
    parser.add_argument("--round-15", action="store_true")
    parser.add_argument("--round-16", action="store_true")
    parser.add_argument("--round-17", action="store_true")
    parser.add_argument("--round-18", action="store_true")
    parser.add_argument("--round-19", action="store_true")
    parser.add_argument("--round-20", action="store_true")
    parser.add_argument("--round-21", action="store_true")
    parser.add_argument("--new-round-21-run", action="store_true")
    parser.add_argument("--new-round-22-run", metavar="SOURCE_R21_RUN_ID")
    parser.add_argument("--resume-round-22-run", metavar="RUN_ID")
    parser.add_argument("--resume-round-21-run")
    parser.add_argument("--start-round-21-batch")
    parser.add_argument("--checkpoint-round-21-batch")
    parser.add_argument("--start-batch", action="store_true")
    parser.add_argument("--checkpoint-batch", action="store_true")
    args = parser.parse_args()
    choices = [args.run_g01, args.fixed_origins, args.neighbors, args.round_3, args.round_4, args.round_5,
               args.round_6, args.round_7, args.round_8, args.round_9, args.round_10, args.round_11,
               args.round_12, args.round_13, args.round_14, args.round_15, args.round_16, args.round_17,
               args.round_18, args.round_19, args.round_20, args.round_21, args.new_round_21_run,
               args.new_round_22_run is not None, args.resume_round_22_run is not None,
               args.resume_round_21_run is not None,
               args.start_round_21_batch is not None, args.checkpoint_round_21_batch is not None,
               args.start_batch, args.checkpoint_batch]
    if sum(choices) != 1:
        parser.error("choose exactly one stage")
    result = (start_batch() if args.start_batch else checkpoint_batch() if args.checkpoint_batch
              else start_round21_batch(args.start_round_21_batch) if args.start_round_21_batch is not None
              else checkpoint_round21_batch(args.checkpoint_round_21_batch)
              if args.checkpoint_round_21_batch is not None
              else run_g01() if args.run_g01 else run_fixed_origins()
              if args.fixed_origins else run_neighbors() if args.neighbors
              else run_round_3() if args.round_3 else run_round_4() if args.round_4
              else run_round_5() if args.round_5 else run_round_6() if args.round_6
              else run_round_7() if args.round_7 else run_round_8() if args.round_8
              else run_round_9() if args.round_9 else run_round_10() if args.round_10
              else run_round_11() if args.round_11 else run_round_12() if args.round_12
              else run_round_13() if args.round_13 else run_round_14() if args.round_14
              else run_round_15() if args.round_15 else run_round_16() if args.round_16
              else run_round_17() if args.round_17 else run_round_18() if args.round_18
              else run_round_19() if args.round_19
              else __import__("experiments.bill_greedy.round20", fromlist=["run_round_20"]).run_round_20()
              if args.round_20
              else run_new_round21() if args.new_round_21_run
              else run_new_round22(args.new_round_22_run) if args.new_round_22_run is not None
              else resume_round22_run(args.resume_round_22_run) if args.resume_round_22_run is not None
              else resume_round21(args.resume_round_21_run) if args.resume_round_21_run is not None
              else __import__("experiments.bill_greedy.round21", fromlist=["run_round_21"]).run_round_21())
    print(json.dumps(result, indent=2))


def _register_package_alias() -> None:
    """Keep script execution and package imports on the same runner module state."""
    if __name__ == "__main__":
        sys.modules["experiments.bill_greedy.runner"] = sys.modules[__name__]


if __name__ == "__main__":
    _register_package_alias()
    main()
