"""R113: select a past-only R97 residual-correction strength by correction magnitude."""
from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.bill_target74 import runner as campaign


CARD = "R113_r97_correction_magnitude_gate"
R93_RUN = "r93s_20261025T201838Z"
R97_RUN = "r97m_20261025T210147Z"
TRACKS = ("off_market", "list_aware")
MONTHS = tuple(campaign.PROTOCOL["development_months"])
BIN_EDGES = (0.005, 0.020)  # absolute log-price correction: <=0.5%, 0.5-2%, >2%
BIN_NAMES = ("small", "medium", "large")
GAMMA_GRID = tuple(float(x) for x in np.linspace(0.0, 1.0, 5))
MIN_HISTORY_ROWS = 500
MIN_HISTORY_MONTHS = 2
MAPE_REGRESSION_MAX_PERCENT = 1.0
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20261025


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return str(path.relative_to(campaign.ROOT)).replace("\\", "/")


def _source_identity() -> dict:
    branch = subprocess.run(
        ["git", "branch", "--show-current"], cwd=campaign.ROOT,
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=campaign.ROOT,
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    if branch != "bill" or commit != campaign.PROTOCOL["parent_commit"]:
        raise RuntimeError(f"expected bill at {campaign.PROTOCOL['parent_commit']}, found {branch} at {commit}")
    if _sha256(campaign.DATA) != campaign.PROTOCOL["data_sha256"]:
        raise RuntimeError("cached listings parquet hash differs from the active campaign")
    return {"branch": branch, "commit": commit, "data_sha256": campaign.PROTOCOL["data_sha256"]}


def _input_paths() -> dict[str, Path]:
    root = campaign.ROOT / "outputs/bill_target74" / campaign.PROTOCOL["campaign_id"] / "batches"
    return {
        "r93": root / R93_RUN / "paired_predictions.csv",
        "r97": root / R97_RUN / "paired_predictions.csv",
    }


def _input_identity() -> dict:
    paths = _input_paths()
    protocol_cards = campaign.PROTOCOL["cards"]
    r93_source = protocol_cards["R02_structure_matched_past_oof"]["safe_current_execution"]["r93_oof_source"]
    r97_card = protocol_cards["R97_past_oof_catboost_residual_meta"]
    expected = {
        "r93": r93_source["predictions_sha256"],
        "r97": r97_card["paired_predictions_sha256"],
    }
    if r93_source.get("run_id") != R93_RUN or r97_card.get("run_id") != R97_RUN:
        raise RuntimeError("registered R93/R97 run identities differ from the intended calibration inputs")
    actual = {name: _sha256(path) for name, path in paths.items()}
    if actual != expected:
        raise RuntimeError(f"R93/R97 paired-prediction hashes differ from the registered cards: {actual}")
    return {
        name: {"path": _relative(path), "sha256": actual[name]}
        for name, path in paths.items()
    }


def _registration_spec(source: dict, inputs: dict) -> dict:
    return {
        "status": "registered",
        "tracks": list(TRACKS),
        "development_months": list(MONTHS),
        "parents": {"off_market": "R52 experimental OOF parent", "list_aware": "R35 validated OOF parent"},
        "method": (
            "Use the saved strict-past R97 residual correction, but select its multiplier separately in three "
            "fixed bins of absolute predicted correction magnitude (<=0.005, (0.005,0.020], >0.020). "
            "For each target month and bin, choose gamma from {0,0.25,0.50,0.75,1.00} using only earlier "
            "safe OOF rows in that bin, minimizing MdAPE subject to MAPE no more than 1% above gamma=0. "
            "Require at least 500 past rows across at least two earlier months; otherwise use gamma=0. "
            "Apply round(parent_prediction * exp(gamma * R97_correction_log))."
        ),
        "difference_from_prior_work": (
            "R112 selected one global attached-home expert share per cutoff; R94 selected R93 spatial strength "
            "by house type. R113 instead gates the existing nonlinear R97 residual correction by its predicted "
            "magnitude, learning a separate shrinkage only from strictly earlier OOF rows in that magnitude bin. "
            "This is a calibration analysis and starts no learner training fits."
        ),
        "selection": {
            "magnitude_bin_edges_absolute_log_correction": list(BIN_EDGES),
            "bins": list(BIN_NAMES),
            "gamma_grid": list(GAMMA_GRID),
            "minimum_past_rows_per_bin": MIN_HISTORY_ROWS,
            "minimum_past_months_per_bin": MIN_HISTORY_MONTHS,
            "mape_regression_max_percent": MAPE_REGRESSION_MAX_PERCENT,
            "unsupported_bin_fallback_gamma": 0.0,
            "tie_break": ["lower past MdAPE", "lower past MAPE", "smaller gamma"],
        },
        "inputs": inputs,
        "source_sha256": _sha256(Path(__file__)),
        "data_sha256": source["data_sha256"],
        "parent_commit": source["commit"],
        "expected_calibration_fit_starts": (len(MONTHS) - 1) * len(TRACKS),
        "learner_training_fit_starts": 0,
        "reserved_months_labels_opened": False,
        "hidden_labels_opened": False,
    }


def register_card() -> dict:
    source = _source_identity()
    inputs = _input_identity()
    card = _registration_spec(source, inputs)
    previous = campaign.PROTOCOL["cards"].get(CARD)
    if previous:
        raise RuntimeError(f"{CARD} is already present with status {previous.get('status')!r}; do not rerun")
    campaign.PROTOCOL["cards"][CARD] = card
    campaign.PROTOCOL_PATH.write_text(
        json.dumps(campaign.PROTOCOL, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    campaign._append({
        "event": "card_registered", "card": CARD, "fit_starts": 0,
        "learner_training_fit_starts": 0, "source_sha256": card["source_sha256"],
        "inputs": inputs, "data_sha256": source["data_sha256"],
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    campaign._write_progress(
        "r113_registered", next_action="Run R113 strictly past correction-magnitude calibration",
        r113_card=CARD,
    )
    return card


def _magnitude_bin(values: np.ndarray) -> np.ndarray:
    absolute = np.abs(np.asarray(values, dtype=float))
    return np.where(absolute <= BIN_EDGES[0], 0,
                    np.where(absolute <= BIN_EDGES[1], 1, 2)).astype(int)


def _load_inputs() -> pd.DataFrame:
    identity = _input_identity()
    paths = _input_paths()
    r93 = pd.read_csv(paths["r93"], dtype={"ml_num": str})
    r97 = pd.read_csv(paths["r97"], dtype={"ml_num": str})
    keys = ["month", "track", "ml_num"]
    if r93.duplicated(keys).any() or r97.duplicated(keys).any():
        raise RuntimeError("R93/R97 input contains duplicate month-track-listing keys")
    rows = r93.merge(r97, on=keys, how="inner", suffixes=("_r93", "_r97"), validate="one_to_one")
    if len(rows) != 2 * 15642 or sorted(rows.month.unique().tolist()) != sorted(MONTHS):
        raise RuntimeError("R93/R97 exact safe development coverage is incomplete or unexpected")
    if len(rows) != len(r93) or len(rows) != len(r97):
        raise RuntimeError("R93/R97 input rows do not align exactly")
    if set(rows.track.unique()) != set(TRACKS):
        raise RuntimeError("unexpected track appeared in the safe OOF files")
    if rows.sold_price_r93.isna().any() or not np.array_equal(
        rows.sold_price_r93.to_numpy(float), rows.sold_price_r97.to_numpy(float)
    ):
        raise RuntimeError("R93 and R97 actual labels are not exactly aligned")
    if not np.array_equal(rows.base_prediction_r93.to_numpy(float), rows.base_prediction_r97.to_numpy(float)):
        raise RuntimeError("R93 and R97 base prediction vectors differ")
    rows["base_prediction"] = rows["base_prediction_r97"]
    rows["correction_log"] = rows["correction_log_r97"]
    rows["candidate_prediction"] = rows["candidate_prediction_r97"]
    for name in ("sold_price_r93", "sold_price_r97", "base_prediction", "candidate_prediction_r93",
                 "candidate_prediction", "correction_log"):
        values = pd.to_numeric(rows[name], errors="raise").to_numpy(float)
        if not np.isfinite(values).all():
            raise RuntimeError(f"non-finite values in R113 source field {name}")
    if (rows.sold_price_r93.to_numpy(float) <= 0).any() or (rows.base_prediction.to_numpy(float) <= 0).any():
        raise RuntimeError("R113 requires positive actual and base predictions")
    if set(rows.month).intersection(set(campaign.PROTOCOL.get("reserved_months", []))):
        raise RuntimeError("reserved-month rows appeared in the R93/R97 safe OOF files")
    reconstructed = np.rint(rows.base_prediction.to_numpy(float) * np.exp(rows.correction_log.to_numpy(float)))
    if not np.array_equal(reconstructed, rows.candidate_prediction.to_numpy(float)):
        raise RuntimeError("R97 candidate predictions do not reproduce from the registered correction logs")
    rows["correction_bin"] = _magnitude_bin(rows.correction_log.to_numpy(float))
    rows["r113_prediction"] = np.nan
    rows["selected_gamma"] = np.nan
    rows["past_bin_rows"] = 0
    rows["past_bin_months"] = 0
    return rows


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    return campaign._metrics(actual, predicted)


def _select_gamma(history: pd.DataFrame, bin_id: int) -> tuple[float, dict]:
    sample = history.loc[history.correction_bin.eq(bin_id)]
    months = sorted(sample.month.unique().tolist())
    diagnostic = {"bin": BIN_NAMES[bin_id], "past_rows": int(len(sample)), "past_months": months}
    if len(sample) < MIN_HISTORY_ROWS or len(months) < MIN_HISTORY_MONTHS:
        return 0.0, {**diagnostic, "gamma": 0.0, "fallback": "insufficient_past_support"}
    y = sample.sold_price_r93.to_numpy(float)
    base = sample.base_prediction.to_numpy(float)
    delta = sample.correction_log.to_numpy(float)
    baseline = _metrics(y, np.rint(base))
    feasible = []
    for gamma in GAMMA_GRID:
        prediction = np.rint(base * np.exp(gamma * delta))
        score = _metrics(y, prediction)
        change = 100.0 * (score["mape_percent"] - baseline["mape_percent"]) / baseline["mape_percent"]
        if change <= MAPE_REGRESSION_MAX_PERCENT + 1e-12:
            feasible.append((score["mdape_percent"], score["mape_percent"], gamma, change))
    if not feasible:
        return 0.0, {**diagnostic, "gamma": 0.0, "fallback": "no_mape_feasible_gamma"}
    mdape, mape, gamma, mape_change = min(feasible, key=lambda x: (x[0], x[1], x[2]))
    return gamma, {
        **diagnostic, "gamma": gamma, "fallback": None,
        "selected_past_mdape_percent": float(mdape),
        "selected_past_mape_percent": float(mape),
        "selected_past_mape_change_percent": float(mape_change),
    }


def _score_track(frame: pd.DataFrame) -> dict:
    actual = frame.sold_price_r93.to_numpy(float)
    base = np.rint(frame.base_prediction.to_numpy(float))
    fixed = frame.candidate_prediction.to_numpy(float)
    candidate = frame.r113_prediction.to_numpy(float)
    track = str(frame.track.iloc[0])
    best = (frame.candidate_prediction_r93.to_numpy(float) if track == "off_market"
            else frame.candidate_prediction.to_numpy(float))
    scores = {
        "parent": _metrics(actual, base),
        "r97_fixed": _metrics(actual, fixed),
        "r113": _metrics(actual, candidate),
        "previous_best": _metrics(actual, best),
    }
    monthly = {}
    for month, group in frame.groupby("month", sort=True):
        before = _metrics(group.sold_price_r93.to_numpy(float), np.rint(group.base_prediction.to_numpy(float)))
        after = _metrics(group.sold_price_r93.to_numpy(float), group.r113_prediction.to_numpy(float))
        monthly[str(month)] = {
            "n": int(len(group)), "parent": before, "candidate": after,
            "relative_mdape_improvement_percent": float(
                100.0 * (before["mdape_percent"] - after["mdape_percent"]) / before["mdape_percent"]
            ),
        }
    gain = 100.0 * (scores["parent"]["mdape_percent"] - scores["r113"]["mdape_percent"]) / scores["parent"]["mdape_percent"]
    mape_change = 100.0 * (scores["r113"]["mape_percent"] - scores["parent"]["mape_percent"]) / scores["parent"]["mape_percent"]
    monthly_gains = [row["relative_mdape_improvement_percent"] for row in monthly.values()]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    base_ape = 100.0 * np.abs(base - actual) / actual
    candidate_ape = 100.0 * np.abs(candidate - actual) / actual
    differences = np.empty(BOOTSTRAP_REPLICATES)
    for replicate in range(BOOTSTRAP_REPLICATES):
        index = rng.integers(0, len(actual), size=len(actual))
        differences[replicate] = np.median(candidate_ape[index]) - np.median(base_ape[index])
    bootstrap = {
        "lower95": float(np.quantile(differences, 0.025)),
        "upper95": float(np.quantile(differences, 0.975)),
        "replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED,
    }
    gates = campaign.PROTOCOL["gates"]
    target = 7.0 if track == "off_market" else 4.0
    checks = {
        "relative_mdape_improvement": gain >= gates["development_relative_mdape_improvement_percent"],
        "months_improved": sum(value > 0 for value in monthly_gains) >= gates["development_months_improved_min"],
        "worst_month_regression": max([0.0] + [-value for value in monthly_gains]) <= gates["development_worst_month_regression_max_percent"],
        "mape_regression": mape_change <= gates["development_mape_regression_max_percent"],
        "bootstrap_upper_below_zero": bootstrap["upper95"] < 0.0,
        "strict_target": scores["r113"]["mdape_percent"] < target,
        "beats_previous_best": scores["r113"]["mdape_percent"] < scores["previous_best"]["mdape_percent"],
    }
    return {
        **scores, "relative_mdape_improvement_percent": float(gain),
        "mape_relative_change_percent": float(mape_change),
        "months_improved": int(sum(value > 0 for value in monthly_gains)),
        "worst_month_regression_percent": float(max([0.0] + [-value for value in monthly_gains])),
        "paired_bootstrap_mdape_difference_pp": bootstrap,
        "monthly": monthly, "gate_checks": checks,
        "qualifies_development": all(checks.values()),
        "target_mdape_strictly_below": target,
    }


def _report(result: dict) -> str:
    lines = [
        "# R113 — strictly past correction-magnitude gate", "",
        f"Campaign `{result['campaign_id']}`; run `{result['run_id']}`.",
        "The method uses the saved R97 OOF correction and selects a multiplier by fixed correction-magnitude bins, using only strictly earlier rows. No learner was trained.",
        "", "## Pooled scores", "",
        "| Track | Base MdAPE | Fixed R97 MdAPE | R113 MdAPE | Previous best MdAPE | MAPE change | Qualifies |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for track, score in result["tracks"].items():
        lines.append(
            f"| {track} | {score['parent']['mdape_percent']:.6f}% | "
            f"{score['r97_fixed']['mdape_percent']:.6f}% | {score['r113']['mdape_percent']:.6f}% | "
            f"{score['previous_best']['mdape_percent']:.6f}% | "
            f"{score['mape_relative_change_percent']:+.3f}% | {'yes' if score['qualifies_development'] else 'no'} |"
        )
    lines.extend(["", "## Monthly paired scores", "", "| Track | Month | Parent MdAPE | R113 MdAPE | Relative improvement |", "|---|---|---:|---:|---:|"])
    for track, score in result["tracks"].items():
        for month, item in score["monthly"].items():
            lines.append(
                f"| {track} | {month} | {item['parent']['mdape_percent']:.6f}% | "
                f"{item['candidate']['mdape_percent']:.6f}% | {item['relative_mdape_improvement_percent']:+.3f}% |"
            )
    lines.extend([
        "", "## Boundary", "",
        f"Calibration fit starts: {result['calibration_fit_starts']}; learner training fits: 0.",
        "This is a development-only, strictly past OOF analysis. It does not promote or replace R22/R35, and it does not read reserved or hidden labels.",
        "",
    ])
    return "\n".join(lines)


def run() -> dict:
    source = _source_identity()
    inputs = _input_identity()
    current_card = campaign.PROTOCOL["cards"].get(CARD)
    if not current_card or current_card.get("source_sha256") != _sha256(Path(__file__)):
        raise RuntimeError("R113 is not registered against this exact source file")
    if current_card.get("status") != "registered":
        raise RuntimeError(f"R113 is not in runnable state: {current_card.get('status')}")
    rows = _load_inputs()
    started = datetime.now(timezone.utc)
    run_id = "r113c_" + started.strftime("%Y%m%dT%H%M%SZ")
    output = campaign.CAMPAIGN_ROOT / "batches" / run_id
    output.mkdir(parents=True, exist_ok=False)
    provenance = output / "provenance"
    provenance.mkdir()
    shutil.copy2(Path(__file__), provenance / Path(__file__).name)
    (provenance / "registered_campaign_protocol.json").write_text(
        json.dumps(campaign.PROTOCOL, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    campaign._append({
        "event": "card_start", "card": CARD, "run_id": run_id,
        "tracks": list(TRACKS), "development_months": list(MONTHS),
        "expected_calibration_fit_starts": (len(MONTHS) - 1) * len(TRACKS),
        "learner_training_fit_starts": 0, "inputs": inputs,
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    output_frames = []
    fit_details = []
    try:
        for track in TRACKS:
            track_rows = rows.loc[rows.track.eq(track)].copy()
            for month in MONTHS:
                current = track_rows.loc[track_rows.month.eq(month)].copy().reset_index(drop=True)
                history = track_rows.loc[track_rows.month.lt(month)].copy()
                if current.empty:
                    raise RuntimeError(f"no safe OOF rows for {track}/{month}")
                cutoff = pd.Timestamp(month + "-01")
                if not history.empty and pd.to_datetime(history.month + "-01").max() >= cutoff:
                    raise RuntimeError(f"R113 history crossed target cutoff {track}/{month}")
                fit_id = f"{campaign.PROTOCOL['campaign_id']}:{CARD}:{track}:{month}:{run_id}"
                if history.empty:
                    gammas = {0: 0.0, 1: 0.0, 2: 0.0}
                    diagnostics = {
                        name: {"bin": name, "past_rows": 0, "past_months": [], "gamma": 0.0,
                               "fallback": "first_development_fold"}
                        for name in BIN_NAMES
                    }
                else:
                    campaign._append({
                        "event": "fit_start", "card": CARD, "run_id": run_id,
                        "fit_id": fit_id, "stage": "past_oof_calibration", "track": track,
                        "month": month, "training_oof_rows": int(len(history)),
                        "query_rows": int(len(current)), "history_cutoff_exclusive": cutoff.isoformat(),
                        "learner_training_fit": False,
                        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
                    })
                    gammas = {}
                    diagnostics = {}
                    for bin_id, name in enumerate(BIN_NAMES):
                        gamma, detail = _select_gamma(history, bin_id)
                        gammas[bin_id] = gamma
                        diagnostics[name] = detail
                current_bins = current.correction_bin.to_numpy(int)
                delta = current.correction_log.to_numpy(float)
                base = current.base_prediction.to_numpy(float)
                selected = np.array([gammas[int(bin_id)] for bin_id in current_bins], dtype=float)
                candidate = np.rint(base * np.exp(selected * delta))
                if not np.isfinite(candidate).all() or (candidate <= 0).any():
                    raise RuntimeError(f"invalid R113 predictions for {track}/{month}")
                current["r113_prediction"] = candidate
                current["selected_gamma"] = selected
                current["past_bin_rows"] = [diagnostics[BIN_NAMES[int(i)]] ["past_rows"] for i in current_bins]
                current["past_bin_months"] = [len(diagnostics[BIN_NAMES[int(i)]] ["past_months"]) for i in current_bins]
                current["selected_bin"] = [BIN_NAMES[int(i)] for i in current_bins]
                output_frames.append(current)
                if history.empty:
                    continue
                detail = {
                    "fit_id": fit_id, "track": track, "month": month,
                    "status": "complete", "training_oof_rows": int(len(history)),
                    "query_rows": int(len(current)), "history_cutoff_exclusive": cutoff.isoformat(),
                    "gamma_by_bin": gammas, "support_by_bin": diagnostics,
                }
                fit_details.append(detail)
                campaign._append({
                    "event": "fit_complete", "card": CARD, "run_id": run_id,
                    "stage": "past_oof_calibration", **detail,
                    "learner_training_fit": False,
                    "reserved_months_labels_opened": False, "hidden_labels_opened": False,
                })
        paired = pd.concat(output_frames, ignore_index=True)
        scores = {track: _score_track(paired.loc[paired.track.eq(track)].copy()) for track in TRACKS}
        result = {
            "campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id, "card": CARD,
            "stage": "strict_past_walk_forward_development", "development_months": list(MONTHS),
            "tracks": scores, "calibration_fit_starts": len(fit_details),
            "learner_training_fit_starts": 0, "fit_details": fit_details,
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
            "inputs": inputs,
        }
        pair_path = output / "paired_predictions.csv"
        result_path = output / "development_results.json"
        paired.to_csv(pair_path, index=False)
        result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        report_path = output / "research_report.md"
        report_path.write_text(_report(result), encoding="utf-8")
        manifest = {
            "run_id": run_id, "card": CARD, "branch": source["branch"], "commit": source["commit"],
            "source_sha256": _sha256(Path(__file__)), "data_sha256": source["data_sha256"],
            "input_files": inputs,
            "result_path": _relative(result_path), "result_sha256": _sha256(result_path),
            "paired_predictions_path": _relative(pair_path), "paired_predictions_sha256": _sha256(pair_path),
            "report_path": _relative(report_path), "report_sha256": _sha256(report_path),
            "python": sys.version, "platform": platform.platform(),
            "packages": {name: metadata.version(name) for name in ("numpy", "pandas")},
            "calibration_fit_starts": len(fit_details), "learner_training_fit_starts": 0,
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        }
        manifest_path = output / "run_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        card = campaign.PROTOCOL["cards"][CARD]
        card.update({
            "status": "development_complete", "run_id": run_id,
            "fit_starts": len(fit_details), "learner_training_fit_starts": 0,
            "development_metrics": scores,
            "qualifies_by_track": {track: score["qualifies_development"] for track, score in scores.items()},
            "result_path": _relative(result_path), "result_sha256": _sha256(result_path),
            "paired_predictions_path": _relative(pair_path), "paired_predictions_sha256": _sha256(pair_path),
            "manifest_path": _relative(manifest_path), "manifest_sha256": _sha256(manifest_path),
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        campaign.PROTOCOL_PATH.write_text(
            json.dumps(campaign.PROTOCOL, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        campaign._append({
            "event": "card_complete", "card": CARD, "run_id": run_id,
            "fit_starts": len(fit_details), "learner_training_fit_starts": 0,
            "qualifies_by_track": {track: score["qualifies_development"] for track, score in scores.items()},
            "result_path": _relative(result_path), "result_sha256": _sha256(result_path),
            "paired_predictions_sha256": _sha256(pair_path),
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        })
        campaign._write_progress(
            "r113_development_complete",
            next_action="Use R113 results to register a distinct next hypothesis; retain R22/R35 unless all promotion gates pass",
            r113_run_id=run_id, r113_result=_relative(result_path),
            r113_fit_starts=len(fit_details), r113_learner_training_fit_starts=0,
        )
        print(json.dumps(result, ensure_ascii=False))
        return result
    except Exception as exc:
        checkpoint = {
            "run_id": run_id, "card": CARD, "calibration_fit_starts": len(fit_details),
            "learner_training_fit_starts": 0, "completed_calibrations": fit_details,
            "error": f"{type(exc).__name__}: {exc}",
            "reserved_months_labels_opened": False, "hidden_labels_opened": False,
        }
        (output / "failure_checkpoint.json").write_text(
            json.dumps(checkpoint, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        campaign._append({
            "event": "card_failed", "card": CARD, "run_id": run_id,
            "fit_starts": len(fit_details), "learner_training_fit_starts": 0,
            "error": checkpoint["error"], "reserved_months_labels_opened": False,
            "hidden_labels_opened": False,
        })
        campaign._write_progress(
            "r113_failed", next_action="Inspect R113 failure checkpoint before any retry",
            r113_run_id=run_id, r113_failure=_relative(output / "failure_checkpoint.json"),
        )
        raise


def finalize_existing_run(run_id: str) -> dict:
    """Recompute R113 scoring from its saved paired rows without recalibration or training."""
    if not run_id.startswith("r113c_"):
        raise ValueError("only an R113 run id can be finalized")
    output = campaign.CAMPAIGN_ROOT / "batches" / run_id
    paired_path = output / "paired_predictions.csv"
    initial_result_path = output / "development_results.json"
    initial_manifest_path = output / "run_manifest.json"
    if not all(path.is_file() for path in (paired_path, initial_result_path, initial_manifest_path)):
        raise FileNotFoundError(f"incomplete R113 artifact set under {output}")
    initial = json.loads(initial_result_path.read_text(encoding="utf-8"))
    manifest = json.loads(initial_manifest_path.read_text(encoding="utf-8"))
    paired_hash = _sha256(paired_path)
    if manifest.get("paired_predictions_sha256") != paired_hash:
        raise RuntimeError("R113 paired-prediction hash differs from the original manifest")
    if initial.get("run_id") != run_id or initial.get("card") != CARD:
        raise RuntimeError("R113 initial result identity mismatch")
    if initial.get("score_finalization"):
        raise RuntimeError("R113 result already has a finalization record")
    initial_copy = output / "development_results_initial.json"
    if initial_copy.exists():
        raise FileExistsError(f"refusing to overwrite prior initial-score copy: {initial_copy}")
    shutil.copy2(initial_result_path, initial_copy)
    paired = pd.read_csv(paired_path, dtype={"ml_num": str})
    if len(paired) != 2 * 15642 or sorted(paired.month.unique().tolist()) != sorted(MONTHS):
        raise RuntimeError("saved R113 paired rows do not cover the exact safe development set")
    corrected_tracks = {
        track: _score_track(paired.loc[paired.track.eq(track)].copy()) for track in TRACKS
    }
    final = dict(initial)
    final["tracks"] = corrected_tracks
    final["score_finalization"] = {
        "status": "recomputed_from_unchanged_paired_predictions",
        "reason": (
            "The initial scorer referenced R93 as previous_best for both tracks. The correct comparison is "
            "R93 for off_market and fixed R97 for list_aware. Recomputed all metrics and gates from the "
            "same saved paired predictions; no calibration or learner fit was rerun."
        ),
        "initial_result_path": _relative(initial_result_path),
        "initial_result_sha256": _sha256(initial_result_path),
        "initial_score_copy_path": _relative(initial_copy),
        "paired_predictions_sha256": paired_hash,
        "finalizer_source_sha256": _sha256(Path(__file__)),
        "finalized_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    result_path = output / "development_results_revised.json"
    result_path.write_text(json.dumps(final, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report_path = output / "research_report_revised.md"
    report_path.write_text(_report(final), encoding="utf-8")
    revision_manifest = {
        "run_id": run_id, "card": CARD,
        "original_manifest_path": _relative(initial_manifest_path),
        "original_manifest_sha256": _sha256(initial_manifest_path),
        "paired_predictions_path": _relative(paired_path),
        "paired_predictions_sha256": paired_hash,
        "initial_result_path": _relative(initial_result_path),
        "initial_result_sha256": _sha256(initial_result_path),
        "revised_result_path": _relative(result_path),
        "revised_result_sha256": _sha256(result_path),
        "revised_report_path": _relative(report_path),
        "revised_report_sha256": _sha256(report_path),
        "finalizer_source_sha256": _sha256(Path(__file__)),
        "calibration_rerun": False, "learner_training_fit_starts_added": 0,
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    }
    revision_path = output / "run_manifest_revision.json"
    revision_path.write_text(json.dumps(revision_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    card = campaign.PROTOCOL["cards"][CARD]
    if card.get("run_id") != run_id:
        raise RuntimeError("R113 protocol card points to a different run")
    audit = final["score_finalization"]
    card["score_finalization"] = audit
    card["development_metrics"] = corrected_tracks
    card["qualifies_by_track"] = {
        track: score["qualifies_development"] for track, score in corrected_tracks.items()
    }
    card["result_path"] = _relative(result_path)
    card["result_sha256"] = _sha256(result_path)
    card["manifest_revision_path"] = _relative(revision_path)
    card["manifest_revision_sha256"] = _sha256(revision_path)
    campaign.PROTOCOL_PATH.write_text(
        json.dumps(campaign.PROTOCOL, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    campaign._append({
        "event": "result_finalized", "card": CARD, "run_id": run_id,
        "fit_starts_added": 0, "learner_training_fit_starts_added": 0,
        "reason": audit["reason"], "initial_result_sha256": audit["initial_result_sha256"],
        "revised_result_sha256": _sha256(result_path), "paired_predictions_sha256": paired_hash,
        "reserved_months_labels_opened": False, "hidden_labels_opened": False,
    })
    campaign._write_progress(
        "r113_development_complete", next_action="Use corrected R113 scores to choose a distinct next hypothesis; retain R22/R35",
        r113_run_id=run_id, r113_result=_relative(result_path),
        r113_fit_starts=card["fit_starts"], r113_learner_training_fit_starts=0,
    )
    return {
        "run_id": run_id, "result_path": _relative(result_path),
        "tracks": {
            track: {
                "parent_mdape": score["parent"]["mdape_percent"],
                "r97_fixed_mdape": score["r97_fixed"]["mdape_percent"],
                "r113_mdape": score["r113"]["mdape_percent"],
                "previous_best_mdape": score["previous_best"]["mdape_percent"],
                "qualifies": score["qualifies_development"],
            }
            for track, score in corrected_tracks.items()
        },
        "fit_starts_added": 0,
    }


if __name__ == "__main__":
    if "--register" in sys.argv:
        card = register_card()
        print(json.dumps({"card": CARD, "status": card["status"],
                          "expected_calibration_fit_starts": card["expected_calibration_fit_starts"],
                          "learner_training_fit_starts": 0}, ensure_ascii=False))
    elif "--finalize" in sys.argv:
        index = sys.argv.index("--finalize")
        if len(sys.argv) <= index + 1:
            raise SystemExit("pass the R113 run id after --finalize")
        print(json.dumps(finalize_existing_run(sys.argv[index + 1]), ensure_ascii=False))
    else:
        if not campaign.PROTOCOL["cards"].get(CARD):
            raise SystemExit("R113 is not registered; pass --register before running it")
        run()
