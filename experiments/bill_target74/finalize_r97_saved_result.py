"""Zero-fit recovery for R97 after all fold predictions were checkpointed."""
from __future__ import annotations

import hashlib
import json
import platform
from importlib import metadata as package_metadata
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.bill_target74 import r02_structure_matched_current_parent as r02
from experiments.bill_target74 import r93_spatial_oof_calibration as common
from experiments.bill_target74 import r97_past_oof_catboost_residual_meta as r97
from experiments.bill_target74 import runner as campaign


CARD = r97.CARD
TRACKS = r97.TRACKS
MONTHS = r97.MONTHS


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return str(path.relative_to(campaign.ROOT)).replace("\\", "/")


def _independent_metrics(actual: np.ndarray, prediction: np.ndarray) -> dict:
    ape = np.abs(prediction - actual) / actual * 100.0
    return {
        "n": int(ape.size), "mdape_percent": float(np.median(ape)),
        "mape_percent": float(np.mean(ape)), "p90_ape_percent": float(np.quantile(ape, 0.90)),
        "ape_over_20_percent": float(np.mean(ape > 20.0) * 100.0),
    }


def _independent_score(frame: pd.DataFrame) -> dict:
    actual = frame.sold_price.to_numpy(float)
    base = frame.base_prediction.to_numpy(float)
    candidate = frame.candidate_prediction.to_numpy(float)
    base_score = _independent_metrics(actual, base)
    candidate_score = _independent_metrics(actual, candidate)
    monthly = {}
    gains = []
    for month, group in frame.groupby("month", sort=True):
        before = _independent_metrics(group.sold_price.to_numpy(float), group.base_prediction.to_numpy(float))
        after = _independent_metrics(group.sold_price.to_numpy(float), group.candidate_prediction.to_numpy(float))
        gain = 100.0 * (before["mdape_percent"] - after["mdape_percent"]) / before["mdape_percent"]
        gains.append(gain)
        monthly[str(month)] = {"n": int(len(group)), "base": before, "candidate": after,
                               "relative_mdape_improvement_percent": float(gain)}
    gain = 100.0 * (base_score["mdape_percent"] - candidate_score["mdape_percent"]) / base_score["mdape_percent"]
    mape_change = 100.0 * (candidate_score["mape_percent"] - base_score["mape_percent"]) / base_score["mape_percent"]
    rng = np.random.default_rng(r97.BOOTSTRAP_SEED)
    base_ape = np.abs(base - actual) / actual * 100.0
    candidate_ape = np.abs(candidate - actual) / actual * 100.0
    differences = np.empty(r97.BOOTSTRAP_REPLICATES)
    for i in range(r97.BOOTSTRAP_REPLICATES):
        idx = rng.integers(0, len(actual), size=len(actual))
        differences[i] = np.median(candidate_ape[idx]) - np.median(base_ape[idx])
    bootstrap = {
        "lower95": float(np.quantile(differences, .025)),
        "upper95": float(np.quantile(differences, .975)),
        "replicates": r97.BOOTSTRAP_REPLICATES, "seed": r97.BOOTSTRAP_SEED,
    }
    gates = campaign.PROTOCOL["gates"]
    target = 7.0 if frame.track.iloc[0] == "off_market" else 4.0
    checks = {
        "relative_mdape_improvement": gain >= gates["development_relative_mdape_improvement_percent"],
        "months_improved": sum(value > 0.0 for value in gains) >= gates["development_months_improved_min"],
        "worst_month_regression": max([0.0] + [-value for value in gains])
        <= gates["development_worst_month_regression_max_percent"],
        "mape_regression": mape_change <= gates["development_mape_regression_max_percent"],
        "bootstrap_upper_below_zero": bootstrap["upper95"] < 0.0,
        "strict_target": candidate_score["mdape_percent"] < target,
    }
    return {
        "base": base_score, "candidate": candidate_score,
        "relative_mdape_improvement_percent": float(gain),
        "mape_relative_change_percent": float(mape_change),
        "months_improved": int(sum(value > 0.0 for value in gains)),
        "worst_month_regression_percent": float(max([0.0] + [-value for value in gains])),
        "paired_bootstrap_mdape_difference_pp": bootstrap,
        "monthly": monthly, "gate_checks": checks,
        "qualifies_development": all(checks.values()),
        "target_mdape_strictly_below": target,
    }


def _assert_score_matches(stored: dict, actual: dict, track: str) -> None:
    for key in ("relative_mdape_improvement_percent", "mape_relative_change_percent",
                "months_improved", "worst_month_regression_percent", "qualifies_development"):
        left, right = stored[key], actual[key]
        if isinstance(left, float):
            if not np.isclose(left, right, rtol=0, atol=1e-10):
                raise RuntimeError(f"independent R97 {track} score mismatch for {key}: {left} vs {right}")
        elif left != right:
            raise RuntimeError(f"independent R97 {track} score mismatch for {key}: {left} vs {right}")
    for section in ("base", "candidate"):
        for key in ("n", "mdape_percent", "mape_percent", "p90_ape_percent", "ape_over_20_percent"):
            if not np.isclose(stored[section][key], actual[section][key], rtol=0, atol=1e-10):
                raise RuntimeError(f"independent R97 {track} {section}.{key} mismatch")
    for key in ("lower95", "upper95"):
        if not np.isclose(
            stored["paired_bootstrap_mdape_difference_pp"][key],
            actual["paired_bootstrap_mdape_difference_pp"][key], rtol=0, atol=1e-10,
        ):
            raise RuntimeError(f"independent R97 {track} bootstrap {key} mismatch")


def finalize() -> dict:
    card = campaign.PROTOCOL["cards"].get(CARD)
    if not card or card.get("status") != "development_complete":
        raise RuntimeError("R97 saved results are not in the expected development-complete state")
    run_id = card["run_id"]
    output = campaign.CAMPAIGN_ROOT / "batches" / run_id
    result_path = output / "development_results.json"
    paired_path = output / "paired_predictions.csv"
    registered_protocol = output / "provenance" / "registered_campaign_protocol.json"
    if not all(path.is_file() for path in (result_path, paired_path, registered_protocol)):
        raise FileNotFoundError("R97 result, paired rows, or registered protocol snapshot is missing")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    snapshot = json.loads(registered_protocol.read_text(encoding="utf-8"))
    if snapshot["cards"][CARD].get("source_sha256") != card.get("source_sha256"):
        raise RuntimeError("R97 saved registration snapshot does not match model source hash")
    model_source = campaign.PACKAGE / "r97_past_oof_catboost_residual_meta.py"
    if _sha256(model_source) != card["source_sha256"]:
        raise RuntimeError("R97 model source no longer matches the source used for its 12 fits")
    if result.get("run_id") != run_id or result.get("fit_starts") != 12:
        raise RuntimeError("R97 result identity or model-fit count is invalid")
    if result.get("reserved_months_labels_opened") is not False or result.get("hidden_labels_opened") is not False:
        raise RuntimeError("R97 result reports a forbidden label access")

    ledger = [
        json.loads(line) for line in campaign.LEDGER.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    run_events = [row for row in ledger if row.get("card") == CARD and row.get("run_id") == run_id]
    starts = [row for row in run_events if row.get("event") == "fit_start"]
    completions = [row for row in run_events if row.get("event") == "fit_complete"]
    if len(starts) != 12 or len(completions) != 12 or any(row.get("status") != "complete" for row in completions):
        raise RuntimeError("R97 learner fit ledger does not show 12 successful fits")
    if {row.get("fit_id") for row in starts} != {row.get("fit_id") for row in completions}:
        raise RuntimeError("R97 fit-start and fit-complete IDs do not match")

    folds = result.get("fold_prediction_hashes", {})
    if set(folds) != {f"{track}/{month}" for track in TRACKS for month in MONTHS}:
        raise RuntimeError("R97 result does not list all 16 fold checkpoints")
    checkpoint_frames = []
    for key, item in folds.items():
        path = campaign.ROOT / item["path"]
        if not path.is_file() or _sha256(path) != item["sha256"]:
            raise RuntimeError(f"R97 fold checkpoint hash mismatch: {key}")
        frame = pd.read_csv(path, dtype={"month": str, "track": str, "ml_num": str})
        if len(frame) != item["rows"] or set(frame.month.unique()) - set(MONTHS):
            raise RuntimeError(f"R97 fold checkpoint coverage mismatch: {key}")
        checkpoint_frames.append(frame)
    recomposed = pd.concat(checkpoint_frames, ignore_index=True)
    paired = pd.read_csv(paired_path, dtype={"month": str, "track": str, "ml_num": str})
    columns = ["month", "track", "ml_num", "sold_price", "base_prediction",
               "candidate_prediction", "correction_log", "history_oof_rows"]
    if len(paired) != 31284 or paired.duplicated(["month", "track", "ml_num"]).any():
        raise RuntimeError("R97 paired output does not have 31,284 unique safe OOF rows")
    if set(paired.month.unique()) != set(MONTHS) or set(paired.track.unique()) != set(TRACKS):
        raise RuntimeError("R97 paired output includes unexpected months or tracks")
    left = recomposed.loc[:, columns].sort_values(["month", "track", "ml_num"]).reset_index(drop=True)
    right = paired.loc[:, columns].sort_values(["month", "track", "ml_num"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right, check_dtype=False, rtol=0, atol=1e-10)

    r93_info = r02._verify_r93()
    r93_rows = pd.read_csv(campaign.ROOT / r93_info["predictions_path"], dtype={"ml_num": str})
    r93_rows = r93_rows[["month", "track", "ml_num", "sold_price", "base_prediction"]].copy()
    r93_rows["month"] = r93_rows.month.astype(str)
    aligned = paired.merge(
        r93_rows, on=["month", "track", "ml_num"], how="left", suffixes=("", "_r93"), validate="one_to_one",
    )
    if aligned.sold_price_r93.isna().any() or aligned.base_prediction_r93.isna().any():
        raise RuntimeError("R97 predictions do not align to the registered R52/R35 OOF parents")
    if not np.array_equal(aligned.sold_price.to_numpy(float), aligned.sold_price_r93.to_numpy(float)):
        raise RuntimeError("R97 labels differ from R93 parent labels")
    if not np.array_equal(aligned.base_prediction.to_numpy(float), aligned.base_prediction_r93.to_numpy(float)):
        raise RuntimeError("R97 parent predictions differ from R93 R52/R35 rows")

    verified_scores = {}
    for track in TRACKS:
        score = _independent_score(paired.loc[paired.track.eq(track)].copy())
        _assert_score_matches(result["tracks"][track], score, track)
        verified_scores[track] = score

    passed = {track: score["qualifies_development"] for track, score in verified_scores.items()}
    card["finalization_recovery"] = {
        "status": "verified_from_saved_checkpoints",
        "reason": "All fits and fold predictions were complete; the original run stopped only while creating metadata.version in the manifest.",
        "fit_starts": 0,
        "finalizer_source_sha256": _sha256(Path(__file__)),
        "independent_score_recomputation": True,
        "fold_checkpoints_verified": len(folds),
    }
    card["manifest_path"] = _relative(output / "run_manifest.json")
    card["paired_predictions_sha256"] = _sha256(paired_path)
    campaign.PROTOCOL_PATH.write_text(
        json.dumps(campaign.PROTOCOL, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    source_files = [
        model_source, Path(__file__), campaign.PACKAGE / "r02_structure_matched_current_parent.py",
        campaign.PACKAGE / "r93_spatial_oof_calibration.py", campaign.PACKAGE / "runner.py",
        registered_protocol,
    ]
    manifest = {
        "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
        "recovered_finalization": True, "new_fit_starts_during_recovery": 0,
        "source_identity": {"branch": "bill", "parent_commit": card["parent_commit"],
                            "data_sha256": card["data_sha256"]},
        "source_sha256": {_relative(path): _sha256(path) for path in source_files if path.is_file()},
        "registered_model_source_sha256": card["source_sha256"],
        "finalizer_source_sha256": _sha256(Path(__file__)),
        "runtime": {"python": platform.python_version(), "catboost": package_metadata.version("catboost"),
                    "numpy": np.__version__, "pandas": pd.__version__},
        "r93_parent_oof_source": r93_info,
        "fit_starts": 12, "fit_details": result["fit_details"],
        "independent_recomputed_metrics": verified_scores,
        "fold_prediction_hashes": folds,
        "paired_predictions_sha256": _sha256(paired_path),
        "development_results_sha256": _sha256(result_path),
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    }
    manifest_path = output / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    report = [
        "# R97: past-OOF CatBoost residual meta-learner", "",
        f"Campaign `{campaign.PROTOCOL['campaign_id']}`; run `{run_id}`.",
        "Each model was trained only on earlier R52/R35 OOF rows; target-month labels were used only for scoring.",
        "`date_sold` was used for safe-month filters and temporal cutoffs, never as a model feature.",
        "", "## Development results", "",
        "| Track | Base MdAPE | Candidate MdAPE | Relative MdAPE gain | MAPE change | Months improved | Bootstrap upper95 pp | Gate |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for track, score in verified_scores.items():
        report.append(
            f"| {track} | {score['base']['mdape_percent']:.5f}% | "
            f"{score['candidate']['mdape_percent']:.5f}% | "
            f"{score['relative_mdape_improvement_percent']:+.3f}% | "
            f"{score['mape_relative_change_percent']:+.3f}% | {score['months_improved']}/8 | "
            f"{score['paired_bootstrap_mdape_difference_pp']['upper95']:+.5f} | "
            f"{'PASS' if score['qualifies_development'] else 'FAIL'} |"
        )
    report.extend([
        "", "## Execution and recovery", "",
        "All 12/12 registered CatBoost fits completed and all 16 per-fold checkpoints passed hash checks.",
        "The first two setup attempts started zero fits and are recorded in the campaign ledger.",
        "The initial finalizer stopped after saving paired predictions and scores because of a metadata namespace collision; this recovery independently rescored the saved rows and started zero new fits.",
        "Reserved and hidden labels were not opened.",
        "", "Artifacts: `paired_predictions.csv`, `development_results.json`, `run_manifest.json`, "
        "and 16 verified files under `fold_predictions/`.", "",
    ])
    (output / "research_report.md").write_text("\n".join(report), encoding="utf-8")

    campaign._append({
        "event": "card_finalization_recovered", "card": CARD, "run_id": run_id,
        "fit_starts": 0, "model_fit_starts_completed": 12,
        "verified_fold_checkpoints": 16,
        "result_path": _relative(result_path), "manifest_path": _relative(manifest_path),
        "paired_predictions_sha256": manifest["paired_predictions_sha256"],
        "qualifies_by_track": passed,
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    campaign._append({
        "event": "card_complete", "card": CARD, "run_id": run_id,
        "fit_starts": 12, "fit_starts_during_finalization": 0,
        "qualifies_by_track": passed, "result_path": _relative(result_path),
        "manifest_path": _relative(manifest_path),
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    next_action = (
        "R97 development-qualified tracks require fixed-origin confirmation: "
        + ", ".join(track for track, ok in passed.items() if ok)
        if any(passed.values()) else
        "R97 failed development gates; retain R22/R35 and register a distinct method"
    )
    campaign._write_progress(
        "r97_development_complete", next_action=next_action,
        r97_run_id=run_id, r97_result=_relative(result_path), r97_fit_starts=12,
        r97_recovered_finalization=True,
    )
    return {"run_id": run_id, "tracks": verified_scores, "qualifies_by_track": passed,
            "manifest_path": _relative(manifest_path)}


if __name__ == "__main__":
    result = finalize()
    for track, score in result["tracks"].items():
        print(
            f"{track}: base={score['base']['mdape_percent']:.5f}% "
            f"candidate={score['candidate']['mdape_percent']:.5f}% "
            f"gain={score['relative_mdape_improvement_percent']:+.3f}% "
            f"MAPE_change={score['mape_relative_change_percent']:+.3f}% "
            f"months_improved={score['months_improved']}/8 "
            f"bootstrap_upper95={score['paired_bootstrap_mdape_difference_pp']['upper95']:+.5f} "
            f"qualifies={score['qualifies_development']}", flush=True,
        )
    print(f"manifest={result['manifest_path']}", flush=True)
