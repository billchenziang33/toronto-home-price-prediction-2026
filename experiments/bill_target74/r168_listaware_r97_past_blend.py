from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.bill_target74 import runner as campaign

CARD = "R168_listaware_r97_global_past_blend"
R97_RUN = "r97m_20261025T210147Z"
R35_RUN = "bill_target74_r35_house_type_text_development_20261025T040058Z"
R97_REL = Path("outputs/bill_target74") / campaign.PROTOCOL["campaign_id"] / "batches" / R97_RUN / "paired_predictions.csv"
R35_REL = Path("outputs/bill_target74") / campaign.PROTOCOL["campaign_id"] / "batches" / R35_RUN / "paired_predictions.csv"
R97_SHA256 = "cea1400e859f1da376b51d9e9ccc9a73c297d188b71993050d106a63d2971704"
R35_SHA256 = "2b5373586166ea55297367eec47dda803aee4a9ac84439598eaa4fa3384053cb"
ALPHA_GRID = (0.0, 0.25, 0.50, 0.75, 1.0)
MIN_HISTORY_ROWS = 3000
MIN_HISTORY_MONTHS = 2
MAPE_GUARD_PERCENT = 1.0
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20261028
MONTHS = tuple(campaign.PROTOCOL["development_months"])
TARGET = 4.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return path.relative_to(campaign.ROOT).as_posix()


def source_identity() -> dict:
    branch = subprocess.run(["git", "branch", "--show-current"], cwd=campaign.ROOT,
                            capture_output=True, text=True, check=True).stdout.strip()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=campaign.ROOT,
                            capture_output=True, text=True, check=True).stdout.strip()
    if branch != "bill" or commit != campaign.PROTOCOL["parent_commit"]:
        raise RuntimeError(f"expected bill at {campaign.PROTOCOL['parent_commit']}, found {branch} at {commit}")
    if sha256(campaign.DATA) != campaign.PROTOCOL["data_sha256"]:
        raise RuntimeError("cached listings data hash differs from active campaign")
    return {"branch": branch, "commit": commit, "data_sha256": campaign.PROTOCOL["data_sha256"]}


def input_paths() -> dict[str, Path]:
    return {"r35": campaign.ROOT / R35_REL, "r97": campaign.ROOT / R97_REL}


def input_identity() -> dict:
    paths = input_paths()
    expected = {"r35": R35_SHA256, "r97": R97_SHA256}
    actual = {name: sha256(path) for name, path in paths.items()}
    if actual != expected:
        raise RuntimeError(f"source prediction hash mismatch: {actual}")
    r97_card = campaign.PROTOCOL["cards"]["R97_past_oof_catboost_residual_meta"]
    r35_card = campaign.PROTOCOL["cards"]["R35_listaware_detached_r22_text_expert"]
    if r97_card.get("run_id") != R97_RUN or r97_card.get("paired_predictions_sha256") != R97_SHA256:
        raise RuntimeError("R97 input does not match its registered run")
    if r35_card.get("development_run_id") != R35_RUN:
        raise RuntimeError("R35 input does not match its registered development run")
    return {name: {"path": rel(path), "sha256": actual[name]} for name, path in paths.items()}


def registration(source: dict, inputs: dict) -> dict:
    return {
        "status": "registered",
        "run_id": None,
        "track": "list_aware",
        "development_months": list(MONTHS),
        "parents": {"validated": "R35", "current_best": "R97"},
        "method": (
            "For each safe development month, choose one global alpha from the fixed grid "
            "{0,.25,.50,.75,1} using only strictly earlier safe months. Minimize past pooled MdAPE "
            "subject to past MAPE no more than 1% above R35; tie-break by lower MAPE then lower alpha. "
            "Require 3000 prior rows across two months, otherwise use alpha=0. Predict round((1-alpha)*R35+alpha*R97)."
        ),
        "difference_from_prior_work": (
            "R116 selected correction multipliers by residual sign and R123/R124 by house type/sign. "
            "R168 selects one price-space blend weight globally by cutoff, and compares against both fixed R35 and R97."
        ),
        "selection": {
            "alpha_grid": list(ALPHA_GRID), "minimum_history_rows": MIN_HISTORY_ROWS,
            "minimum_history_months": MIN_HISTORY_MONTHS, "past_mape_regression_max_percent": MAPE_GUARD_PERCENT,
            "tie_break": ["lower past MdAPE", "lower past MAPE", "smaller alpha"],
            "unsupported_history_fallback_alpha": 0.0,
        },
        "inputs": inputs,
        "source_sha256": sha256(Path(__file__)),
        "audit_source_sha256": sha256(Path(__file__).with_name("r168_independent_audit.py")),
        "data_sha256": source["data_sha256"],
        "parent_commit": source["commit"],
        "expected_calibration_fit_starts": max(0, len(MONTHS) - 2),
        "learner_training_fit_starts": 0,
        "bootstrap": {"replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED},
        "gates": {
            "comparators": ["R35", "R97"],
            "minimum_relative_mdape_improvement_percent": 0.25,
            "minimum_months_improved": 6,
            "maximum_mape_regression_percent": 1.0,
            "maximum_worst_month_regression_percent": 5.0,
            "paired_bootstrap_upper95_below_zero": True,
            "strict_target_mdape_percent": TARGET,
        },
        "reserved_months_labels_opened": False,
        "hidden_labels_opened": False,
    }


def register() -> dict:
    source = source_identity()
    inputs = input_identity()
    card = registration(source, inputs)
    if CARD in campaign.PROTOCOL["cards"]:
        raise RuntimeError(f"{CARD} is already registered; do not repeat this analysis")
    campaign.PROTOCOL["cards"][CARD] = card
    campaign.PROTOCOL_PATH.write_text(json.dumps(campaign.PROTOCOL, indent=2, ensure_ascii=False) + "\n",
                                       encoding="utf-8")
    campaign._append({"event": "card_registered", "card": CARD, "fit_starts": 0,
                      "source_sha256": card["source_sha256"], "inputs": inputs,
                      "reserved_months_labels_opened": False, "hidden_labels_opened": False})
    campaign._write_progress("r168_registered", next_action="Run registered R35/R97 strict-past price blend",
                             r168_card=CARD)
    return card


def amend_registration_after_zero_fit_failure() -> dict:
    current = campaign.PROTOCOL["cards"].get(CARD)
    if not current or current.get("status") != "registered":
        raise RuntimeError("only a registered, unstarted R168 card can be amended")
    ledger = [json.loads(line) for line in campaign.LEDGER.read_text(encoding="utf-8").splitlines() if line.strip()]
    card_rows = [row for row in ledger if row.get("card") == CARD]
    if any(row.get("event") == "fit_start" for row in card_rows):
        raise RuntimeError("R168 has a learner or calibration fit start; source amendment is forbidden")
    starts = [row for row in card_rows if row.get("event") == "card_start"]
    terminal_ids = {row.get("run_id") for row in card_rows
                    if row.get("event") in {"card_complete", "card_failed", "card_setup_failed"}}
    abandoned = [row for row in starts if row.get("run_id") not in terminal_ids]
    for row in abandoned:
        run_id = row["run_id"]
        run_root = campaign.CAMPAIGN_ROOT / "batches" / run_id
        if not run_root.exists() or (run_root / "run_manifest.json").exists():
            raise RuntimeError(f"cannot resolve the unterminated zero-fit attempt {run_id}")
        campaign._append({"event": "card_failed", "card": CARD, "run_id": run_id,
                          "status": "failed", "fit_starts": 0,
                          "error": "AttributeError in load_inputs: pandas merge suffixed duplicate candidate_prediction columns; failure occurred before fit_start",
                          "reserved_months_labels_opened": False, "hidden_labels_opened": False})
    if not abandoned:
        raise RuntimeError("no unterminated zero-fit setup attempt was found to amend")

    source = source_identity()
    inputs = input_identity()
    updated = registration(source, inputs)
    updated["registration_amendments"] = list(current.get("registration_amendments", [])) + [{
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "reason": "Correct R35/R97 merged prediction column access to use the pandas-generated _r97 suffix; add setup-failure logging around input loading. Method and thresholds unchanged.",
        "previous_source_sha256": current.get("source_sha256"),
        "current_source_sha256": updated["source_sha256"],
        "fit_starts": 0,
        "failed_setup_run_ids": [row["run_id"] for row in abandoned],
    }]
    campaign.PROTOCOL["cards"][CARD] = updated
    campaign.PROTOCOL_PATH.write_text(json.dumps(campaign.PROTOCOL, indent=2, ensure_ascii=False) + "\n",
                                      encoding="utf-8")
    campaign._append({"event": "card_registration_amended", "card": CARD, "fit_starts": 0,
                      "old_source_sha256": current.get("source_sha256"),
                      "new_source_sha256": updated["source_sha256"],
                      "failed_setup_run_ids": [row["run_id"] for row in abandoned],
                      "reason": "fix merged candidate_prediction column suffix and ensure pre-fit setup failures are ledgered",
                      "reserved_months_labels_opened": False, "hidden_labels_opened": False})
    campaign._write_progress("r168_registered_after_zero_fit_repair",
                             next_action="run amended R168 source after zero-fit setup repair",
                             r168_card=CARD, latest_run_id=abandoned[-1]["run_id"],
                             latest_audit_passed=False, latest_promotion_gate_passed=False)
    return updated


def load_inputs() -> pd.DataFrame:
    paths = input_paths()
    r35 = pd.read_csv(paths["r35"], dtype={"month": str, "track": str, "ml_num": str})
    r97 = pd.read_csv(paths["r97"], dtype={"month": str, "track": str, "ml_num": str})
    required35 = {"month", "track", "ml_num", "sold_price", "candidate_prediction"}
    required97 = {"month", "track", "ml_num", "sold_price", "base_prediction", "candidate_prediction"}
    if not required35.issubset(r35.columns) or not required97.issubset(r97.columns):
        raise RuntimeError("paired prediction inputs are missing required columns")
    r35 = r35.loc[r35.track.eq("list_aware"), ["month", "track", "ml_num", "sold_price", "candidate_prediction"]]
    r97 = r97.loc[r97.track.eq("list_aware"), ["month", "track", "ml_num", "sold_price", "base_prediction", "candidate_prediction"]]
    key = ["month", "track", "ml_num"]
    if r35.duplicated(key).any() or r97.duplicated(key).any():
        raise RuntimeError("duplicate row keys in R35/R97 paired inputs")
    if set(r35.month.unique()) != set(MONTHS) or set(r97.month.unique()) != set(MONTHS):
        raise RuntimeError("inputs do not contain exactly the registered safe development months")
    rows = r35.merge(r97, on=key, how="outer", suffixes=("_r35", "_r97"), validate="one_to_one", indicator=True)
    if not rows._merge.eq("both").all():
        raise RuntimeError("R35 and R97 row keys do not align one-to-one")
    if not np.array_equal(rows.sold_price_r35.to_numpy(float), rows.sold_price_r97.to_numpy(float)):
        raise RuntimeError("R35 and R97 labels differ on aligned rows")
    if not np.array_equal(rows.candidate_prediction_r35.to_numpy(float), rows.base_prediction.to_numpy(float)):
        raise RuntimeError("R97 saved base predictions do not exactly match R35 predictions")
    rows["sold_price"] = rows.sold_price_r35.astype(float)
    rows["r35_prediction"] = rows.candidate_prediction_r35.astype(float)
    rows["r97_prediction"] = rows.candidate_prediction_r97.astype(float)
    if (rows[["sold_price", "r35_prediction", "r97_prediction"]].to_numpy(float) <= 0).any():
        raise RuntimeError("aligned rows contain non-positive prices")
    return rows.drop(columns=["_merge"]).sort_values(["month", "ml_num"]).reset_index(drop=True)


def select_alpha(history: pd.DataFrame) -> tuple[float, dict]:
    past_months = sorted(history.month.unique().tolist())
    detail = {"past_rows": int(len(history)), "past_months": past_months}
    if len(history) < MIN_HISTORY_ROWS or len(past_months) < MIN_HISTORY_MONTHS:
        return 0.0, {**detail, "alpha": 0.0, "fallback": "insufficient_past_support"}
    actual = history.sold_price.to_numpy(float)
    p35 = history.r35_prediction.to_numpy(float)
    p97 = history.r97_prediction.to_numpy(float)
    base = campaign._metrics(actual, p35)
    feasible = []
    for alpha in ALPHA_GRID:
        pred = np.rint((1.0 - alpha) * p35 + alpha * p97)
        score = campaign._metrics(actual, pred)
        mape_change = 100.0 * (score["mape_percent"] - base["mape_percent"]) / base["mape_percent"]
        if mape_change <= MAPE_GUARD_PERCENT + 1e-12:
            feasible.append((score["mdape_percent"], score["mape_percent"], alpha, mape_change))
    if not feasible:
        return 0.0, {**detail, "alpha": 0.0, "fallback": "no_mape_feasible_alpha"}
    mdape, mape, alpha, mape_change = min(feasible, key=lambda row: (row[0], row[1], row[2]))
    return alpha, {**detail, "alpha": alpha, "fallback": None,
                   "selected_past_mdape_percent": float(mdape),
                   "selected_past_mape_percent": float(mape),
                   "selected_past_mape_change_percent": float(mape_change)}


def bootstrap_difference(actual: np.ndarray, candidate: np.ndarray, comparator: np.ndarray) -> dict:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    candidate_ape = 100.0 * np.abs(candidate - actual) / actual
    comparator_ape = 100.0 * np.abs(comparator - actual) / actual
    differences = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    for replicate in range(BOOTSTRAP_REPLICATES):
        indices = rng.integers(0, len(actual), size=len(actual))
        differences[replicate] = np.median(candidate_ape[indices]) - np.median(comparator_ape[indices])
    return {"lower95_pp": float(np.quantile(differences, 0.025)),
            "upper95_pp": float(np.quantile(differences, 0.975)),
            "replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED}


def score(rows: pd.DataFrame) -> dict:
    actual = rows.sold_price.to_numpy(float)
    candidate = rows.r168_prediction.to_numpy(float)
    r35 = rows.r35_prediction.to_numpy(float)
    r97 = rows.r97_prediction.to_numpy(float)
    comparators = {"R35": r35, "R97": r97}
    pooled = {name: campaign._metrics(actual, pred) for name, pred in {"R35": r35, "R97": r97, "R168": candidate}.items()}
    comparisons = {}
    for name, pred in comparators.items():
        gain = 100.0 * (pooled[name]["mdape_percent"] - pooled["R168"]["mdape_percent"]) / pooled[name]["mdape_percent"]
        mape_change = 100.0 * (pooled["R168"]["mape_percent"] - pooled[name]["mape_percent"]) / pooled[name]["mape_percent"]
        monthly = {}
        gains = []
        for month, group in rows.groupby("month", sort=True):
            y = group.sold_price.to_numpy(float)
            after = campaign._metrics(y, group.r168_prediction.to_numpy(float))
            before = campaign._metrics(y, group[f"{name.lower()}_prediction"].to_numpy(float))
            relative = 100.0 * (before["mdape_percent"] - after["mdape_percent"]) / before["mdape_percent"]
            gains.append(relative)
            monthly[str(month)] = {"n": int(len(group)), "comparator": before, "candidate": after,
                                   "relative_mdape_improvement_percent": float(relative)}
        ci = bootstrap_difference(actual, candidate, pred)
        gates = {
            "relative_mdape_improvement": gain >= 0.25,
            "months_improved": sum(value > 0 for value in gains) >= 6,
            "worst_month_regression": max([0.0] + [-value for value in gains]) <= 5.0,
            "mape_regression": mape_change <= 1.0,
            "paired_bootstrap_upper_below_zero": ci["upper95_pp"] < 0.0,
            "strict_target": pooled["R168"]["mdape_percent"] < TARGET,
        }
        comparisons[name] = {"relative_mdape_improvement_percent": float(gain),
                             "mape_relative_change_percent": float(mape_change),
                             "months_improved": int(sum(value > 0 for value in gains)),
                             "worst_month_regression_percent": float(max([0.0] + [-value for value in gains])),
                             "paired_bootstrap_mdape_difference": ci,
                             "monthly": monthly, "gates": gates, "qualifies": all(gates.values())}
    return {"pooled": pooled, "comparisons": comparisons,
            "qualifies_development": all(v["qualifies"] for v in comparisons.values())}


def run() -> dict:
    source = source_identity()
    inputs = input_identity()
    registered = campaign.PROTOCOL["cards"].get(CARD)
    if not registered or registered.get("source_sha256") != sha256(Path(__file__)):
        raise RuntimeError("R168 source is not registered by SHA-256")
    if registered.get("status") != "registered":
        raise RuntimeError(f"R168 is not runnable: {registered.get('status')}")
    started = datetime.now(timezone.utc)
    run_id = "r168c_" + started.strftime("%Y%m%dT%H%M%SZ")
    root = campaign.CAMPAIGN_ROOT / "batches" / run_id
    root.mkdir(parents=True, exist_ok=False)
    provenance = root / "provenance"
    provenance.mkdir()
    for name, path in input_paths().items():
        shutil.copy2(path, provenance / f"{name}_paired_predictions.csv")
    campaign._append({"event": "card_start", "card": CARD, "run_id": run_id,
                      "stage": "strict_past_walk_forward_development", "fit_start_count": 0,
                      "months": list(MONTHS), "inputs": inputs,
                      "reserved_months_labels_opened": False, "hidden_labels_opened": False})
    fit_details = []
    try:
        rows = load_inputs()
        rows["r168_prediction"] = np.nan
        rows["selected_alpha"] = np.nan
        rows["selection_detail"] = ""
        for month in MONTHS:
            current_mask = rows.month.eq(month)
            current = rows.loc[current_mask]
            history = rows.loc[rows.month.lt(month)]
            cutoff = pd.Timestamp(month + "-01")
            if not history.empty and pd.to_datetime(history.month + "-01").max() >= cutoff:
                raise RuntimeError(f"history crosses month cutoff {month}")
            eligible = len(history) >= MIN_HISTORY_ROWS and history.month.nunique() >= MIN_HISTORY_MONTHS
            fit_id = f"{campaign.PROTOCOL['campaign_id']}:{CARD}:list_aware:{month}:{run_id}"
            if eligible:
                campaign._append({"event": "fit_start", "card": CARD, "run_id": run_id,
                                  "fit_id": fit_id, "stage": "past_oof_calibration", "track": "list_aware",
                                  "month": month, "training_oof_rows": int(len(history)),
                                  "training_oof_months": sorted(history.month.unique().tolist()),
                                  "query_rows": int(len(current)), "history_cutoff_exclusive": cutoff.isoformat(),
                                  "learner_training_fit": False,
                                  "reserved_months_labels_opened": False, "hidden_labels_opened": False})
            alpha, detail = select_alpha(history)
            prediction = np.rint((1.0 - alpha) * current.r35_prediction.to_numpy(float)
                                 + alpha * current.r97_prediction.to_numpy(float))
            if not np.isfinite(prediction).all() or (prediction <= 0).any():
                raise RuntimeError(f"invalid R168 prices in {month}")
            rows.loc[current_mask, "r168_prediction"] = prediction
            rows.loc[current_mask, "selected_alpha"] = alpha
            rows.loc[current_mask, "selection_detail"] = json.dumps(detail, sort_keys=True)
            if eligible:
                fit_details.append({"fit_id": fit_id, "track": "list_aware", "month": month,
                                    "status": "complete", "alpha": alpha, **detail,
                                    "training_oof_rows": int(len(history)), "query_rows": int(len(current)),
                                    "history_cutoff_exclusive": cutoff.isoformat()})
                campaign._append({"event": "fit_complete", "card": CARD, "run_id": run_id,
                                  "stage": "past_oof_calibration", **fit_details[-1],
                                  "learner_training_fit": False,
                                  "reserved_months_labels_opened": False, "hidden_labels_opened": False})
        if rows.r168_prediction.isna().any() or rows.selected_alpha.isna().any():
            raise RuntimeError("R168 left missing predictions or weights")
        metrics = score(rows)
        result = {"campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id, "card": CARD,
                  "stage": "strict_past_walk_forward_development", "track": "list_aware",
                  "development_months": list(MONTHS), "row_count": int(len(rows)), "scores": metrics,
                  "calibration_fit_starts": len(fit_details), "learner_training_fit_starts": 0,
                  "fit_details": fit_details, "inputs": inputs,
                  "reserved_months_labels_opened": False, "hidden_labels_opened": False}
        pair_path = root / "paired_predictions.csv"
        result_path = root / "development_results.json"
        rows.to_csv(pair_path, index=False)
        result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        report_path = root / "research_report.md"
        report_path.write_text(render_report(result), encoding="utf-8")
        manifest = {"campaign_id": campaign.PROTOCOL["campaign_id"], "run_id": run_id, "card": CARD,
                    "branch": source["branch"], "commit": source["commit"],
                    "source_sha256": sha256(Path(__file__)),
                    "audit_source_sha256": sha256(Path(__file__).with_name("r168_independent_audit.py")),
                    "data_sha256": source["data_sha256"],
                    "input_files": inputs,
                    "provenance_snapshots": {p.name: sha256(p) for p in provenance.iterdir()},
                    "paired_predictions_path": rel(pair_path), "paired_predictions_sha256": sha256(pair_path),
                    "result_path": rel(result_path), "result_sha256": sha256(result_path),
                    "report_path": rel(report_path), "report_sha256": sha256(report_path),
                    "reserved_months_labels_opened": False, "hidden_labels_opened": False}
        manifest_path = root / "run_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        card = campaign.PROTOCOL["cards"][CARD]
        card.update({"status": "development_complete", "run_id": run_id,
                     "result_sha256": manifest["result_sha256"],
                     "paired_predictions_sha256": manifest["paired_predictions_sha256"],
                     "manifest_sha256": sha256(manifest_path),
                     "calibration_fit_starts": len(fit_details), "learner_training_fit_starts": 0,
                     "qualifies_development": metrics["qualifies_development"],
                     "reserved_months_labels_opened": False, "hidden_labels_opened": False})
        campaign.PROTOCOL_PATH.write_text(json.dumps(campaign.PROTOCOL, indent=2, ensure_ascii=False) + "\n",
                                          encoding="utf-8")
        campaign._append({"event": "card_complete", "card": CARD, "run_id": run_id,
                          "stage": "strict_past_walk_forward_development",
                          "fit_starts": len(fit_details), "qualifies_development": metrics["qualifies_development"],
                          "reserved_months_labels_opened": False, "hidden_labels_opened": False})
        campaign._write_progress("r168_development_complete_pending_audit",
                                 next_action="independently audit R168 row alignment, monthly weights, scores and ledger",
                                 latest_run_id=run_id, latest_card=CARD,
                                 latest_promotion_gate_passed=False)
        return result
    except Exception as exc:
        campaign._append({"event": "card_failed", "card": CARD, "run_id": run_id,
                          "error": f"{type(exc).__name__}: {exc}",
                          "reserved_months_labels_opened": False, "hidden_labels_opened": False})
        campaign._write_progress("r168_failed", next_action="audit R168 failure before another attempt",
                                 latest_run_id=run_id, latest_card=CARD)
        raise


def render_report(result: dict) -> str:
    lines = ["# R168 — list-aware R35/R97 strict-past price blend", "",
             f"Run: `{result['run_id']}`; safe rows: {result['row_count']}; calibration starts: {result['calibration_fit_starts']}; learner fits: 0.",
             "Each month's alpha was selected only from earlier development months. The first unsupported folds used R35 (`alpha=0`).", "",
             "## Pooled metrics", "", "| Model | MdAPE | MAPE |", "|---|---:|---:|"]
    for model, score_row in result["scores"]["pooled"].items():
        lines.append(f"| {model} | {score_row['mdape_percent']:.6f}% | {score_row['mape_percent']:.6f}% |")
    lines.extend(["", "## Paired decision gates", "", "| Comparator | Relative MdAPE gain | MAPE change | Months improved | CI upper (pp) | Pass |", "|---|---:|---:|---:|---:|---|"])
    for name, row in result["scores"]["comparisons"].items():
        lines.append(f"| {name} | {row['relative_mdape_improvement_percent']:+.3f}% | {row['mape_relative_change_percent']:+.3f}% | "
                     f"{row['months_improved']}/8 | {row['paired_bootstrap_mdape_difference']['upper95_pp']:+.5f} | {row['qualifies']} |")
    lines.extend(["", "## Monthly weights", "", "| Month | Alpha | Prior rows | Prior months |", "|---|---:|---:|---|"])
    for detail in result["fit_details"]:
        lines.append(f"| {detail['month']} | {detail['alpha']:.2f} | {detail['past_rows']} | {', '.join(detail['past_months'])} |")
    lines.extend(["", "Reserved and hidden labels were not opened. R168 is eligible for locked confirmation only if every registered gate passes.", ""])
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("register", "amend", "run"))
    args = parser.parse_args()
    output = register() if args.stage == "register" else amend_registration_after_zero_fit_failure() if args.stage == "amend" else run()
    print(json.dumps(output, indent=2, ensure_ascii=False))
