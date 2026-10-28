"""R167: cap extreme R97 corrections on frozen strict-past OOF predictions."""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.bill_target74 import r04_ablation as provenance
from experiments.bill_target74 import runner as campaign


ROOT = Path(__file__).resolve().parents[2]
CARD = "R167_listaware_r97_clip005"
AUDITOR_PATH = Path(__file__).with_name("r167_independent_audit.py")
TRACK = "list_aware"
R35_CARD = "R35_listaware_detached_r22_text_expert"
R97_CARD = "R97_past_oof_catboost_residual_meta"
R94_DIR = campaign.CAMPAIGN_ROOT / "batches" / "r94g_20261025T202220Z"
SCREEN_MONTHS = ("2023-03", "2024-06")
DEVELOPMENT_MONTHS = tuple(campaign.PROTOCOL["development_months"])
CORRECTION_SHRINKAGE = 0.50
ORIGINAL_RESIDUAL_CLIP = 0.10
CANDIDATE_RESIDUAL_CLIP = 0.05
ORIGINAL_CORRECTION_CLIP = CORRECTION_SHRINKAGE * ORIGINAL_RESIDUAL_CLIP
CANDIDATE_CORRECTION_CLIP = CORRECTION_SHRINKAGE * CANDIDATE_RESIDUAL_CLIP
SCREEN_BOOTSTRAP_SEED = 20261167
DEVELOPMENT_BOOTSTRAP_SEED = 20261168
BOOTSTRAP_REPLICATES = 2000


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact_paths() -> dict[str, Path]:
    progress = _read_json(campaign.STATE)
    r97_result = ROOT / progress["r97_result"]
    r35_card = campaign.PROTOCOL["cards"].get(R35_CARD)
    if not r35_card or r35_card.get("status") != "fixed_origin_confirmation_passed":
        raise RuntimeError("R167 requires the validated R35 list-aware parent")
    r35_result = ROOT / r35_card["development_result_path"]
    return {
        "r97_result": r97_result,
        "r97_pairs": r97_result.parent / "paired_predictions.csv",
        "r97_manifest": r97_result.parent / "run_manifest.json",
        "r35_result": r35_result,
        "r35_pairs": r35_result.parent / "paired_predictions.csv",
        "r35_manifest": r35_result.parent / "run_manifest.json",
        "r94_result": R94_DIR / "development_results.json",
        "r94_pairs": R94_DIR / "paired_predictions.csv",
        "r94_manifest": R94_DIR / "run_manifest.json",
    }


def _preflight() -> tuple[dict, dict[str, Path]]:
    progress = _read_json(campaign.STATE)
    limits = campaign.PROTOCOL.get("limits", {})
    if any(value is not None for key, value in limits.items() if key != "learner_threads"):
        raise RuntimeError("R167 requires the active unbounded campaign limits")
    paths = _artifact_paths()
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"R167 source artifacts are missing: {missing}")
    r97_result, r97_manifest = _read_json(paths["r97_result"]), _read_json(paths["r97_manifest"])
    r35_result, r35_manifest = _read_json(paths["r35_result"]), _read_json(paths["r35_manifest"])
    r94_result, r94_manifest = _read_json(paths["r94_result"]), _read_json(paths["r94_manifest"])
    if (r97_result.get("run_id") != progress.get("r97_run_id")
            or r97_result.get("card") != R97_CARD
            or r97_result.get("data_sha256") != campaign.PROTOCOL["data_sha256"]
            or r97_manifest.get("paired_predictions_sha256") != _sha(paths["r97_pairs"])
            or r97_manifest.get("development_results_sha256") != _sha(paths["r97_result"])):
        raise RuntimeError("R167 R97 saved OOF result/manifest identity failed")
    for result, manifest, label in ((r97_result, r97_manifest, "R97"),
                                    (r35_result, r35_manifest, "R35"),
                                    (r94_result, r94_manifest, "R94")):
        if (result.get("data_sha256") != campaign.PROTOCOL["data_sha256"]
                or result.get("reserved_months_labels_opened") is not False
                or result.get("hidden_labels_opened") is not False
                or manifest.get("reserved_months_labels_opened") is not False
                or manifest.get("hidden_labels_opened") is not False):
            raise RuntimeError(f"R167 {label} result or manifest has invalid data/access flags")
    for name, result, manifest, result_key in (
        ("R35", r35_result, r35_manifest, "result_sha256"),
        ("R94", r94_result, r94_manifest, "result_sha256"),
    ):
        if (manifest.get("paired_predictions_sha256") != _sha(paths[name.lower() + "_pairs"])
                or manifest.get(result_key) != _sha(paths[name.lower() + "_result"])):
            raise RuntimeError(f"R167 {name} benchmark hash mismatch")
    if not set(DEVELOPMENT_MONTHS).issubset(set(r97_result.get("months", DEVELOPMENT_MONTHS))):
        raise RuntimeError("R167 R97 result does not contain every registered development month")
    safe = set(DEVELOPMENT_MONTHS)
    if safe.intersection(set(campaign.PROTOCOL["reserved_months"])):
        raise RuntimeError("R167 development window overlaps a reserved month")
    inputs = {
        name: {"path": _relative(path), "sha256": _sha(path)}
        for name, path in paths.items()
    }
    return inputs, paths


def _source_hashes(inputs: dict) -> dict[str, str]:
    source_paths = [Path(__file__), AUDITOR_PATH,
                    campaign.PACKAGE / "r04_ablation.py", campaign.PACKAGE / "runner.py"]
    hashes = {_relative(path): _sha(path) for path in source_paths}
    hashes.update({value["path"]: value["sha256"] for value in inputs.values()})
    return hashes


def _register() -> dict:
    inputs, paths = _preflight()
    source = provenance._source_identity()
    if source.get("data_sha256") != campaign.PROTOCOL["data_sha256"]:
        raise RuntimeError("R167 source data identity differs from the campaign")
    hashes = _source_hashes(inputs)
    card = campaign.PROTOCOL["cards"].get(CARD)
    if card is not None:
        if card.get("source_sha256") != hashes or card.get("input_artifacts") != inputs:
            raise RuntimeError("R167 registered source or frozen OOF inputs changed")
        return card
    card = {
        "status": "registered", "track": TRACK,
        "screen_months": list(SCREEN_MONTHS), "development_months": list(DEVELOPMENT_MONTHS),
        "expected_screen_fit_starts": 0, "expected_development_fit_starts": 0,
        "expected_screen_transform_folds": 2,
        "baseline_by_track": {TRACK: "validated R35 list-aware OOF"},
        "current_best_comparator": "R97 saved development OOF",
        "method_id": "R167_R97_RESIDUAL_PREDICTION_CLIP_005",
        "method": ("Reuse frozen R97 strict-past OOF predictions and keep its 0.50 correction "
                   "shrinkage. Clip only the saved final log correction at +/-0.025, equivalent "
                   "to reducing the raw residual prediction clip from 0.10 to 0.05; no learner refit."),
        "difference_from_prior_work": ("R161-R165 changed history sources, model settings, targets, "
                                       "text or calibration families. R167 leaves small R97 corrections "
                                       "unchanged and caps only extremes at a 0.05 residual clip."),
        "input_artifacts": inputs, "source_sha256": hashes,
        "correction_shrinkage": CORRECTION_SHRINKAGE,
        "original_residual_prediction_clip": ORIGINAL_RESIDUAL_CLIP,
        "candidate_residual_prediction_clip": CANDIDATE_RESIDUAL_CLIP,
        "original_final_log_correction_clip": ORIGINAL_CORRECTION_CLIP,
        "candidate_final_log_correction_clip": CANDIDATE_CORRECTION_CLIP,
        "correction_formula": "round(base_prediction * exp(clip(correction_log, -0.025, 0.025)))",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "screen_bootstrap_seed": SCREEN_BOOTSTRAP_SEED,
        "development_bootstrap_seed": DEVELOPMENT_BOOTSTRAP_SEED,
        "screen_gates": {"minimum_relative_mdape_improvement_percent": 0.25,
                         "both_months_improved": True,
                         "maximum_mape_regression_percent": 1.0,
                         "paired_bootstrap_upper95_below_zero": True,
                         "comparators": ["R35 validated parent", "R97 current best"]},
        "development_gates": dict(campaign.PROTOCOL["gates"]),
        "data_sha256": source["data_sha256"], "parent_commit": source["commit"],
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        "reserve_validation_eligible": False,
    }
    campaign.PROTOCOL["cards"][CARD] = card
    campaign.PROTOCOL_PATH.write_text(json.dumps(campaign.PROTOCOL, indent=2, ensure_ascii=False) + "\n",
                                      encoding="utf-8")
    campaign._append({"event": "card_registered", "card": CARD,
                      "source_sha256": hashes[_relative(Path(__file__))],
                      "input_artifacts": inputs, "expected_fit_starts": 0,
                      "reserved_months_labels_opened": False, "hidden_labels_opened": False})
    campaign._write_progress("r167_registered", next_action="score frozen R97 OOF with residual clip 0.05",
                              r167_card=CARD)
    return card


def _load_pairs(paths: dict[str, Path], months: tuple[str, ...]) -> pd.DataFrame:
    r97 = pd.read_csv(paths["r97_pairs"], dtype={"ml_num": str})
    r35 = pd.read_csv(paths["r35_pairs"], dtype={"ml_num": str})
    r94 = pd.read_csv(paths["r94_pairs"], dtype={"ml_num": str})
    r97 = r97.loc[r97.track.eq(TRACK) & r97.month.astype(str).isin(months),
                  ["month", "ml_num", "sold_price", "base_prediction", "candidate_prediction", "correction_log"]].copy()
    r35 = r35.loc[r35.track.eq(TRACK) & r35.month.astype(str).isin(months),
                  ["month", "ml_num", "sold_price", "candidate_prediction"]].copy()
    r94 = r94.loc[r94.track.eq(TRACK) & r94.month.astype(str).isin(months),
                  ["month", "ml_num", "sold_price", "base_prediction"]].copy()
    for frame in (r97, r35, r94):
        frame["month"] = frame.month.astype(str)
        frame["ml_num"] = frame.ml_num.astype(str)
        if frame.duplicated(["month", "ml_num"]).any():
            raise RuntimeError("R167 input artifact has duplicate month/listing keys")
    keys = ["month", "ml_num"]
    paired = r97.merge(r35, on=keys, how="outer", suffixes=("_r97", "_r35"), indicator=True,
                       validate="one_to_one", sort=False)
    if len(paired) != len(r97) or not paired._merge.eq("both").all():
        raise RuntimeError("R167 R97 rows do not exactly align with R35")
    paired = paired.drop(columns="_merge")
    paired = paired.merge(r94, on=keys, how="outer", suffixes=("", "_r94"), indicator=True,
                          validate="one_to_one", sort=False)
    if len(paired) != len(r97) or not paired._merge.eq("both").all():
        raise RuntimeError("R167 rows do not exactly align with R94 benchmark")
    if (not np.array_equal(paired.sold_price_r97.to_numpy(float), paired.sold_price_r35.to_numpy(float))
            or not np.array_equal(paired.sold_price_r97.to_numpy(float), paired.sold_price.to_numpy(float))):
        raise RuntimeError("R167 truth labels differ across frozen artifacts")
    if not np.allclose(paired.base_prediction.to_numpy(float),
                       paired.candidate_prediction_r35.to_numpy(float), rtol=0.0, atol=0.005):
        raise RuntimeError("R167 R97 base prediction does not match validated R35")
    if not np.allclose(paired.base_prediction.to_numpy(float),
                       paired.base_prediction_r94.to_numpy(float), rtol=0.0, atol=0.005):
        raise RuntimeError("R167 R94 parent prediction does not match R35 base")
    reconstructed_r97 = np.rint(paired.base_prediction.to_numpy(float)
                                 * np.exp(paired.correction_log.to_numpy(float)))
    if not np.array_equal(reconstructed_r97, paired.candidate_prediction_r97.to_numpy(float)):
        raise RuntimeError("R167 cannot reconstruct original R97 rounded predictions")
    paired["benchmark_prediction"] = paired.candidate_prediction_r35.astype(float)
    paired["current_best_prediction"] = paired.candidate_prediction_r97.astype(float)
    clipped_correction = np.clip(paired.correction_log.to_numpy(float),
                                 -CANDIDATE_CORRECTION_CLIP, CANDIDATE_CORRECTION_CLIP)
    paired["candidate_prediction"] = np.rint(
        paired.base_prediction.to_numpy(float)
        * np.exp(clipped_correction))
    paired["track"] = TRACK
    output = paired[["month", "track", "ml_num", "sold_price_r97", "benchmark_prediction",
                     "current_best_prediction", "candidate_prediction"]].rename(
        columns={"sold_price_r97": "sold_price"})
    if not np.isfinite(output[["sold_price", "benchmark_prediction", "current_best_prediction",
                              "candidate_prediction"]].to_numpy(float)).all():
        raise RuntimeError("R167 produced non-finite prices")
    return output.sort_values(keys).reset_index(drop=True)


def _metric(actual: np.ndarray, prediction: np.ndarray) -> dict:
    ape = np.abs(prediction - actual) / actual * 100.0
    return {"n": int(len(actual)), "mdape_percent": float(np.median(ape)),
            "mape_percent": float(np.mean(ape))}


def _score(frame: pd.DataFrame, reference: str, months: tuple[str, ...], stage: str) -> dict:
    actual = frame.sold_price.to_numpy(float)
    before = frame[reference].to_numpy(float)
    after = frame.candidate_prediction.to_numpy(float)
    bm, cm = _metric(actual, before), _metric(actual, after)
    gain = 100.0 * (bm["mdape_percent"] - cm["mdape_percent"]) / bm["mdape_percent"]
    mape_change = 100.0 * (cm["mape_percent"] - bm["mape_percent"]) / bm["mape_percent"]
    monthly, improvements = {}, []
    groups = []
    for month in months:
        part = frame.loc[frame.month.eq(month)]
        y = part.sold_price.to_numpy(float)
        b = part[reference].to_numpy(float)
        c = part.candidate_prediction.to_numpy(float)
        base_metric, candidate_metric = _metric(y, b), _metric(y, c)
        month_gain = 100.0 * (base_metric["mdape_percent"] - candidate_metric["mdape_percent"]) / base_metric["mdape_percent"]
        improvements.append(month_gain)
        monthly[month] = {"benchmark": base_metric, "candidate": candidate_metric,
                          "relative_mdape_improvement_percent": float(month_gain)}
        groups.append((y, b, c))
    seed = SCREEN_BOOTSTRAP_SEED if stage == "screen" else DEVELOPMENT_BOOTSTRAP_SEED
    rng = np.random.default_rng(seed)
    differences = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    for index in range(BOOTSTRAP_REPLICATES):
        parts = [rng.integers(0, len(y), size=len(y)) for y, _, _ in groups]
        y = np.concatenate([actuals[ix] for ix, (actuals, _, _) in zip(parts, groups)])
        b = np.concatenate([base[ix] for ix, (_, base, _) in zip(parts, groups)])
        c = np.concatenate([candidate[ix] for ix, (_, _, candidate) in zip(parts, groups)])
        differences[index] = np.median(np.abs(c-y)/y*100.0) - np.median(np.abs(b-y)/y*100.0)
    interval = {"lower95": float(np.quantile(differences, 0.025)),
                "upper95": float(np.quantile(differences, 0.975)),
                "replicates": BOOTSTRAP_REPLICATES, "seed": seed}
    gates = campaign.PROTOCOL["gates"]
    if stage == "screen":
        checks = {"relative_mdape_gain": gain >= gates["screen_relative_mdape_improvement_percent"],
                  "both_months_improved": all(value > 0 for value in improvements),
                  "mape_guard": mape_change <= gates["screen_mape_regression_max_percent"],
                  "bootstrap_upper_below_zero": interval["upper95"] < 0}
    else:
        checks = {"relative_mdape_gain": gain >= gates["development_relative_mdape_improvement_percent"],
                  "months_improved": sum(value > 0 for value in improvements) >= gates["development_months_improved_min"],
                  "worst_month_regression_guard": max(0.0, max(-value for value in improvements))
                  <= gates["development_worst_month_regression_max_percent"],
                  "mape_guard": mape_change <= gates["development_mape_regression_max_percent"],
                  "bootstrap_upper_below_zero": interval["upper95"] < 0}
    return {"n": len(frame), "benchmark": bm, "candidate": cm,
            "relative_mdape_improvement_percent": float(gain),
            "mape_relative_change_percent": float(mape_change),
            "months_improved": int(sum(value > 0 for value in improvements)),
            "worst_month_regression_percent": float(max(0.0, max(-value for value in improvements))),
            "monthly": monthly, "paired_bootstrap_mdape_difference_pp": interval,
            "gate_checks": checks, "qualifies": all(checks.values()),
            "strict_target_below_4_met": cm["mdape_percent"] < 4.0}


def run(stage: str = "screen") -> dict:
    card = _register()
    if stage == "screen" and card.get("status") != "registered":
        raise RuntimeError("R167 screen already ran")
    progress = _read_json(campaign.STATE)
    if stage == "development" and (card.get("status") != "screen_passed"
            or progress.get("r167_audit_passed") is not True):
        raise RuntimeError("R167 development requires passed screen and independent audit")
    months = SCREEN_MONTHS if stage == "screen" else DEVELOPMENT_MONTHS
    paths = _artifact_paths()
    paired = _load_pairs(paths, months)
    run_id = f"r167c_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    output = campaign.CAMPAIGN_ROOT / "batches" / run_id
    output.mkdir(parents=True, exist_ok=False)
    campaign._append({"event": "card_start", "card": CARD, "stage": stage,
                      "run_id": run_id, "months": list(months), "tracks": [TRACK],
                      "expected_fit_starts": 0, "reserved_months_labels_opened": False,
                      "hidden_labels_opened": False})
    frames = []
    for month in months:
        fold = paired.loc[paired.month.eq(month)].copy()
        if fold.empty:
            raise RuntimeError(f"R167 has no saved OOF rows for {month}")
        frames.append(fold)
        campaign._append({"event": "screen_fold_complete" if stage == "screen" else "development_fold_complete",
                          "card": CARD, "stage": stage, "run_id": run_id,
                          "month": month, "track": TRACK, "rows": len(fold),
                          "reserved_months_labels_opened": False, "hidden_labels_opened": False})
    scored = pd.concat(frames, ignore_index=True)
    scores = {reference: _score(scored, reference, months, stage)
              for reference in ("benchmark_prediction", "current_best_prediction")}
    passed = all(score["qualifies"] for score in scores.values())
    paired_path = output / "paired_predictions.csv"
    scored.to_csv(paired_path, index=False)
    result = {"campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
              "card": CARD, "stage": stage, "months": list(months), "tracks": [TRACK],
              "fit_starts": 0, "scores_vs_comparators": scores,
              "screen_passed": passed if stage == "screen" else None,
              "development_passed": passed if stage == "development" else None,
              "candidate_residual_prediction_clip": CANDIDATE_RESIDUAL_CLIP,
              "candidate_final_log_correction_clip": CANDIDATE_CORRECTION_CLIP,
              "reserved_months_labels_opened": False, "hidden_labels_opened": False,
              "data_sha256": campaign.PROTOCOL["data_sha256"]}
    result_path = output / "results.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    source_hashes = _source_hashes(card["input_artifacts"])
    provenance_dir = output / "provenance"
    provenance_dir.mkdir()
    source_snapshot_paths = {}
    for index, (rel, _hash) in enumerate(source_hashes.items()):
        source_path = ROOT / rel
        if source_path.is_file():
            snapshot_relative = f"provenance/src_{index:02d}_{source_path.name}.snapshot"
            shutil.copy2(source_path, output / snapshot_relative)
            source_snapshot_paths[rel] = snapshot_relative
    protocol_snapshot_path = "provenance/campaign_protocol.json.snapshot"
    shutil.copy2(campaign.PROTOCOL_PATH, output / protocol_snapshot_path)
    manifest = {"campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id,
                "card": CARD, "stage": stage, "source_sha256": source_hashes,
                "source_snapshot_paths": source_snapshot_paths,
                "protocol_sha256": _sha(campaign.PROTOCOL_PATH),
                "protocol_snapshot_path": protocol_snapshot_path,
                "fit_starts": 0, "paired_predictions_sha256": _sha(paired_path),
                "result_sha256": _sha(result_path),
                "reserved_months_labels_opened": False, "hidden_labels_opened": False,
                "input_artifacts": card["input_artifacts"]}
    manifest_path = output / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report = ["# R167 R97 residual-clip screen", "",
              "The 0.50 correction shrinkage is retained; only final log corrections are capped at +/-0.025.",
              "No learner fits were run.", ""]
    for reference, score in scores.items():
        report.extend([f"## Compared with {reference}", "",
                       f"Benchmark MdAPE: {score['benchmark']['mdape_percent']:.6f}%",
                       f"Candidate MdAPE: {score['candidate']['mdape_percent']:.6f}%",
                       f"Relative MdAPE improvement: {score['relative_mdape_improvement_percent']:+.4f}%",
                       f"MAPE relative change: {score['mape_relative_change_percent']:+.4f}%",
                       f"Months improved: {score['months_improved']}/{len(months)}",
                       f"All gates: {score['qualifies']}", ""])
    report.append("No reserved or hidden labels were opened.")
    (output / "research_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    card["status"] = ("screen_passed" if passed else "screen_failed") if stage == "screen" else "development_complete"
    card["last_run_id"] = run_id
    card["last_result_path"] = _relative(result_path)
    campaign.PROTOCOL_PATH.write_text(json.dumps(campaign.PROTOCOL, indent=2, ensure_ascii=False) + "\n",
                                      encoding="utf-8")
    campaign._append({"event": "card_complete", "card": CARD, "stage": stage,
                      "run_id": run_id, "fit_starts": 0,
                      "qualifies_by_comparator": {key: value["qualifies"] for key, value in scores.items()},
                      "reserved_months_labels_opened": False, "hidden_labels_opened": False})
    campaign._write_progress("r167_screen_complete" if stage == "screen" else "r167_development_complete",
                              next_action=("audit R167 before any development evaluation" if stage == "screen" and passed
                                           else "retain R35/R97; R167 failed screen" if stage == "screen"
                                           else "review R167 development gates"),
                              r167_run_id=run_id, r167_result=_relative(result_path),
                              latest_fit_count=0)
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--development", action="store_true")
    parser.add_argument("--register-only", action="store_true")
    args = parser.parse_args()
    if args.register_only and args.development:
        parser.error("--register-only cannot be combined with --development")
    if args.register_only:
        print(json.dumps(_register(), indent=2, ensure_ascii=False), flush=True)
    else:
        print(json.dumps(run("development" if args.development else "screen"),
                         indent=2, ensure_ascii=False), flush=True)
