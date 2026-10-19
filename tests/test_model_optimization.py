"""Focused safety checks for the two-track optimization runner."""
from __future__ import annotations

import importlib.util
import pickle

import numpy as np
import pandas as pd
import pytest

from experiments.model_optimization_v1.adapters import (FrozenFeatureTransformer, PriceModel, build_fold,
                                                        inverse_target, make_target)
from experiments.model_optimization_v1.metrics import (paired_mdape_bootstrap, paired_mdape_delta,
                                                       stable_fingerprint, validate_pairing)
from experiments.model_optimization_v1.run import (_cluster_bootstrap_mdape, _method_model_config,
                                                   _preset_config)


def _paired(ape: list[float], prices: list[float] | None = None) -> pd.DataFrame:
    actual = np.asarray(prices or [100.0] * len(ape))
    predicted = actual * (1.0 + np.asarray(ape) / 100.0)
    return pd.DataFrame({"ml_num": [f"id-{i}" for i in range(len(ape))], "track": "off_market",
                         "month": "2025-06", "sold_price": actual, "predicted_price": predicted})


def _homes() -> pd.DataFrame:
    return pd.DataFrame({
        "ml_num": ["a", "b"], "date_listed": pd.to_datetime(["2025-06-01", "2025-06-02"]),
        "house_type_name": ["Detached", "Condo Apt"], "house_style": ["A", "unseen"],
        "bedrooms": [3, 1], "washrooms": [2, 1], "parking_total": [2, 1],
        "house_area": ["1000", "500-699"], "municipality": ["Toronto", "Toronto"],
        "community": ["Known", "New Community"], "latitude": [43.7, 43.6], "longitude": [-79.4, -79.5],
        "list_price": [1_000_000, 500_000]
    })


def test_difference_of_medians_not_median_of_pair_differences() -> None:
    baseline = _paired([0, 100, 100])
    candidate = _paired([1, 2, 200])
    assert paired_mdape_delta(baseline, candidate) == pytest.approx(-98.0)
    assert np.median(np.array([1, 2, 200]) - np.array([0, 100, 100])) == pytest.approx(1.0)
    interval = paired_mdape_bootstrap(baseline, candidate, repeats=100, seed=0)
    assert interval["delta_percentage_points"] == pytest.approx(-98.0)
    assert interval["repeats"] == 100


def test_pairing_rejects_duplicate_and_mismatched_homes() -> None:
    baseline = _paired([1, 2])
    duplicate = pd.concat([baseline, baseline.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        validate_pairing(duplicate, duplicate)
    changed = _paired([1, 2]).assign(ml_num=["other", "id-1"])
    with pytest.raises(ValueError, match="keys differ"):
        validate_pairing(baseline, changed)


def test_paired_bootstrap_is_reproducible() -> None:
    baseline, candidate = _paired([10, 20, 30, 40]), _paired([8, 18, 28, 38])
    first = paired_mdape_bootstrap(baseline, candidate, repeats=250, seed=4)
    second = paired_mdape_bootstrap(baseline, candidate, repeats=250, seed=4)
    assert first == second


def test_off_market_features_exclude_list_and_targets_and_keep_row_order() -> None:
    transformer = FrozenFeatureTransformer("off_market").fit(_homes().iloc[:1])
    transformed = transformer.transform(_homes(), "hgbr")
    assert len(transformed) == 2
    assert "list_price" not in transformed
    assert not {"sold_price", "date_sold", "ml_num"}.intersection(transformed.columns)
    assert transformed.index.tolist() == [0, 1]
    assert pd.isna(transformed.loc[1, "community"])


def test_list_aware_features_keep_list_price_but_not_targets() -> None:
    transformer = FrozenFeatureTransformer("list_aware").fit(_homes())
    transformed = transformer.transform(_homes(), "hgbr")
    assert "list_price" in transformed
    assert "sold_price" not in transformed


def test_ratio_target_and_inverse_require_valid_list_prices() -> None:
    train = _homes().assign(sold_price=[1_050_000, 475_000])
    target = make_target(train, "log_sold_over_list")
    assert np.exp(target) == pytest.approx([1.05, 0.95])
    assert inverse_target(target, train, "log_sold_over_list") == pytest.approx([1_050_000, 475_000])
    with pytest.raises(ValueError, match="positive"):
        make_target(train.assign(list_price=[0, 500_000]), "log_sold_over_list")


def test_log_target_inverse_requires_finite_output() -> None:
    assert inverse_target(np.log([10.0, 100.0]), _homes(), "log_sold_price").tolist() == pytest.approx([10, 100])
    with pytest.raises(ValueError, match="non-finite"):
        inverse_target(np.array([np.nan]), _homes().iloc[:1], "log_sold_price")


def test_walk_forward_fold_excludes_eval_month_and_hides_answers() -> None:
    rows = _homes().assign(sold_price=[900_000, 510_000], date_sold=pd.to_datetime(["2025-05-31", "2025-06-30"]))
    train, homes = build_fold(rows, "2025-06", "off_market")
    assert train.date_sold.max() < pd.Timestamp("2025-06-01")
    assert len(homes) == 1
    assert "sold_price" not in homes and "date_sold" not in homes and "list_price" not in homes


def test_prediction_api_rejects_outcome_columns() -> None:
    model = PriceModel("off_market", "hgbr", {"max_iter": 1})
    with pytest.raises(ValueError, match="unavailable fields"):
        model.predict(_homes().assign(sold_price=[1.0, 1.0]))


def test_cache_fingerprint_changes_for_config_data_or_code() -> None:
    cfg = {"model": "hgbr", "max_iter": 100}
    code = {"adapters.py": "a"}
    original = stable_fingerprint(cfg, "data-a", code)
    assert stable_fingerprint({**cfg, "max_iter": 101}, "data-a", code) != original
    assert stable_fingerprint(cfg, "data-b", code) != original
    assert stable_fingerprint(cfg, "data-a", {"adapters.py": "b"}) != original


def test_lightgbm_preset_preserves_the_predeclared_parameter_order() -> None:
    params = _preset_config("lightgbm", [800, 0.03, 63, 100, 5])
    assert params["n_estimators"] == 800
    assert params["learning_rate"] == 0.03
    assert params["num_leaves"] == 63
    assert params["min_child_samples"] == 100
    assert params["reg_lambda"] == 5
    assert params["deterministic"] is True


def test_l1_method_overrides_the_rmse_objective_in_the_initial_config() -> None:
    base = {"objective": "regression", "n_estimators": 20}
    method = _method_model_config("lightgbm", "log_sold_price_mae", base)
    assert method["objective"] == "regression_l1"
    assert base["objective"] == "regression"
    assert _method_model_config("lightgbm", "log_sold_over_list", base)["objective"] == "regression"


def test_month_cluster_bootstrap_is_deterministic_and_reports_cluster_count() -> None:
    baseline = pd.concat([_paired([10, 20]).assign(month=month) for month in ["2025-12", "2026-01", "2026-02"]], ignore_index=True)
    candidate = baseline.copy()
    candidate["predicted_price"] *= 0.98
    first = _cluster_bootstrap_mdape(baseline, candidate, repeats=100, seed=3)
    second = _cluster_bootstrap_mdape(baseline, candidate, repeats=100, seed=3)
    assert first == second
    assert first["clusters"] == 3
    assert first["months"] == ["2025-12", "2026-01", "2026-02"]


def test_price_model_can_be_serialized_and_loaded_with_identical_predictions() -> None:
    train = _homes().drop(columns="list_price").assign(sold_price=[1_000_000.0, 500_000.0])
    homes = train.drop(columns="sold_price")
    model = PriceModel("off_market", "hgbr", {"max_iter": 5, "early_stopping": False}).fit(train)
    before = model.predict(homes)
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(restored.predict(homes), before)


@pytest.mark.parametrize("family,available", [
    ("catboost", importlib.util.find_spec("catboost") is not None),
    ("lightgbm", importlib.util.find_spec("lightgbm") is not None),
])
def test_optional_adapter_fits_toy_data_when_package_is_installed(family: str, available: bool) -> None:
    if not available:
        pytest.skip(f"{family} is not installed in this environment")
    train = _homes().drop(columns="list_price").assign(sold_price=[1_000_000.0, 500_000.0])
    homes = train.drop(columns="sold_price")
    params = {"iterations": 2, "depth": 2, "learning_rate": 0.2, "l2_leaf_reg": 3,
              "thread_count": 1, "verbose": False} if family == "catboost" else {
              "n_estimators": 2, "num_leaves": 3, "learning_rate": 0.2, "min_child_samples": 1,
              "n_jobs": 1, "verbosity": -1}
    model = PriceModel("off_market", family, params).fit(train)
    assert np.isfinite(model.predict(homes)).all()

