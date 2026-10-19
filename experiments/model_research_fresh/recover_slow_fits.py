from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from . import runner
from .candidates import CANDIDATE_ORDER, DEVELOPMENT_MONTHS, TRACKS, candidate_spec


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/model_research_fresh"
RECOVERY_PATH = Path(__file__)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def _preserve_first_stage(path: Path, backup: Path) -> None:
    if path.exists() and not backup.exists():
        backup.write_bytes(path.read_bytes())


def recover() -> dict:
    state = runner._ensure_state()
    development_path = OUTPUT / "development_results.json"
    freeze_path = OUTPUT / "freeze_manifest.json"
    if not development_path.exists() or not freeze_path.exists():
        raise ValueError("the first development run and freeze record must exist before recovery")
    prior_freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if prior_freeze.get("final_started_at_utc"):
        raise ValueError("cannot recover development fits after final labels have been opened")

    _preserve_first_stage(development_path, OUTPUT / "development_results_pre_recovery.json")
    _preserve_first_stage(freeze_path, OUTPUT / "freeze_manifest_pre_recovery.json")

    model_configs, list_hgbr = runner._model_configs()
    frames: dict[tuple[str, str, str], object] = {}
    statuses = {"completed": [], "cached": [], "failed": [], "skipped": []}
    recovery_timeout_seconds = 300

    for month in DEVELOPMENT_MONTHS:
        for track in TRACKS:
            baseline_specs = runner._baseline_specs(track, model_configs[track], list_hgbr)
            baseline_ready = True
            for component_id, spec in baseline_specs:
                frame, metadata = runner._run_component(
                    "development", month, track, component_id, spec,
                    timeout_seconds=recovery_timeout_seconds,
                )
                if frame is None:
                    baseline_ready = False
                    statuses["failed"].append(f"{component_id}/{track}/{month}")
                elif metadata.get("cache_hit"):
                    statuses["cached"].append(f"{component_id}/{track}/{month}")
                else:
                    statuses["completed"].append(f"{component_id}/{track}/{month}")
            if not baseline_ready:
                statuses["skipped"].append(f"all candidates/{track}/{month}: fold baseline unavailable")
                continue

            frames[(track, "C0", month)] = runner._pipeline_frame(
                "development", track, month, "C0", "C0_LGBM",
                "C0_HGBR" if track == "list_aware" else None,
            )
            for candidate_id in CANDIDATE_ORDER:
                spec = candidate_spec(candidate_id, track, model_configs[track]["model_config"])
                component_id = f"candidate_{candidate_id}"
                frame, metadata = runner._run_component(
                    "development", month, track, component_id, spec,
                    timeout_seconds=recovery_timeout_seconds,
                )
                if frame is None:
                    statuses["failed"].append(f"{candidate_id}/{track}/{month}")
                    continue
                if metadata.get("cache_hit"):
                    statuses["cached"].append(f"{candidate_id}/{track}/{month}")
                else:
                    statuses["completed"].append(f"{candidate_id}/{track}/{month}")
                frames[(track, candidate_id, month)] = runner._pipeline_frame(
                    "development", track, month, candidate_id, component_id,
                    "C0_HGBR" if track == "list_aware" else None,
                )

    results = runner._evaluate(
        "development", DEVELOPMENT_MONTHS, frames,
        {track: CANDIDATE_ORDER for track in TRACKS},
    )
    recovery_record = {
        "started_at_utc": state.get("development_finished_at_utc", state["started_at_utc"]),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "reason": "Repeat only missing or timed-out fold components with the identical frozen learner config and a 300-second per-fit timeout; model, feature, target, blend, data, split, and gates remain unchanged.",
        "previous_development_results_sha256": _sha(OUTPUT / "development_results_pre_recovery.json"),
        "previous_freeze_manifest_sha256": _sha(OUTPUT / "freeze_manifest_pre_recovery.json"),
        "recovery_code_sha256": _sha(RECOVERY_PATH),
        "recovery_timeout_seconds": recovery_timeout_seconds,
        "data_sha256": runner.PROTOCOL["data_sha256"],
        "code_sha256": runner._code_hashes(),
        "development_months": list(DEVELOPMENT_MONTHS),
        "fit_statuses": statuses,
        "counted_fit_attempts": runner._counted_fits(),
        "results": results,
    }
    _save_json(OUTPUT / "development_recovery_results.json", recovery_record)

    development = {
        "created_at_utc": recovery_record["completed_at_utc"],
        "data_sha256": runner.PROTOCOL["data_sha256"],
        "code_sha256": runner._code_hashes(),
        "runtime_versions": runner._runtime_versions(),
        "development_months": list(DEVELOPMENT_MONTHS),
        "candidate_order": list(CANDIDATE_ORDER),
        "results": results,
        "fit_statuses": statuses,
        "counted_fit_attempts": runner._counted_fits(),
        "recovery_record": "development_recovery_results.json",
    }
    _save_json(development_path, development)
    freeze = runner._freeze_development(results)
    freeze["recovery_record"] = "development_recovery_results.json"
    freeze["initial_freeze_manifest"] = "freeze_manifest_pre_recovery.json"
    _save_json(freeze_path, freeze)

    state["development_finished_at_utc"] = recovery_record["completed_at_utc"]
    state["both_tracks_qualified"] = freeze["both_tracks_qualified"]
    state["fit_attempts_after_development"] = runner._counted_fits()
    _save_json(runner.STATE, state)

    final_result = None
    if freeze["final_months_unlocked"]:
        final_result = runner.run_final()
    return {
        "recovery": recovery_record,
        "freeze": freeze,
        "final": final_result,
        "total_elapsed_seconds": runner._elapsed_seconds(),
    }


def main() -> None:
    print(json.dumps(recover(), indent=2, default=str))


if __name__ == "__main__":
    main()
