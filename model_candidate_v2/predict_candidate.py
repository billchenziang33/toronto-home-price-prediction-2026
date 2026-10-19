"""Predict competition homes with model_candidate_v2 and validate both CSVs."""
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
