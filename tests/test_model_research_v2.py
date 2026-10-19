from __future__ import annotations

import math

import pandas as pd
import pytest

from experiments.model_optimization_v2.artifacts import pipeline_fingerprint
from experiments.model_optimization_v2.calibration import (apply_local_residual, apply_multiplier_calibration,
                                                            fit_multiplier)
from experiments.model_optimization_v2.features import asof_market_features, build_features
from experiments.model_optimization_v2.folds import build_month_fold
from experiments.model_optimization_v2.evaluation import evaluate_increment
from experiments.model_optimization_v2.learners import (inverse_price, make_learner, recency_weights,
                                                        select_training_rows, training_target,
                                                        validate_actual_objective)
from experiments.model_optimization_v2.pipeline import blend_prices
from experiments.model_optimization_v2.runner import (budget_allows_fit, validate_registry,
                                                       can_reuse_after_scoped_code_change, _method_config,
                                                       _protocol_c0_metrics)
from experiments.model_optimization_v2.worker import fit_component


def test_month_fold_hides_sale_fields_and_respects_exclusive_cutoff():
    rows = pd.DataFrame({
        "ml_num": ["A", "B", "C"],
        "date_sold": pd.to_datetime(["2025-05-31", "2025-06-01", "2025-06-30"]),
        "date_listed": pd.to_datetime(["2025-05-01", "2025-05-20", "2025-06-05"]),
        "sold_price": [100.0, 200.0, 300.0],
        "list_price": [90.0, 180.0, 280.0],
        "house_type_name": ["Detached"] * 3,
    })

    train, homes, truth = build_month_fold(rows, "2025-06", "off_market")

    assert train.ml_num.tolist() == ["A"]
    assert homes.ml_num.tolist() == ["B", "C"]
    assert "sold_price" not in homes and "date_sold" not in homes and "list_price" not in homes
    assert truth.set_index("ml_num").sold_price.to_dict() == {"B": 200.0, "C": 300.0}


def test_asof_features_exclude_same_day_future_and_self_sales():
    history = pd.DataFrame({
        "ml_num": ["OLD", "SAME", "SELF", "FUTURE"],
        "date_sold": pd.to_datetime(["2025-01-01", "2025-02-01", "2025-01-15", "2025-03-01"]),
        "sold_price": [100.0, 1000.0, 5000.0, 9000.0],
        "area": [10.0, 10.0, 10.0, 10.0],
        "community": ["C"] * 4,
        "municipality": ["M"] * 4,
        "house_type_name": ["Detached"] * 4,
    })
    queries = pd.DataFrame({
        "ml_num": ["SELF"],
        "date_listed": pd.to_datetime(["2025-02-01"]),
        "community": ["C"],
        "municipality": ["M"],
        "house_type_name": ["Detached"],
    })

    result = asof_market_features(queries, history, outer_cutoff="2025-02-01")

    assert result.loc[0, "community_type_count"] == 1
    assert result.loc[0, "community_type_log_ppsf_mean"] == pytest.approx(math.log(10.0))
    assert result.loc[0, "type_log_ppsf_mean_90d"] == pytest.approx(math.log(10.0))


def test_asof_market_features_parse_area_from_competition_house_area_field():
    history = pd.DataFrame({"ml_num": ["OLD"], "date_sold": pd.to_datetime(["2025-01-01"]),
        "sold_price": [2000.0], "house_area": ["100-100"], "community": ["C"],
        "municipality": ["M"], "house_type_name": ["Detached"]})
    query = pd.DataFrame({"ml_num": ["Q"], "date_listed": pd.to_datetime(["2025-02-01"]),
        "community": ["C"], "municipality": ["M"], "house_type_name": ["Detached"]})

    result = asof_market_features(query, history, "2025-03-01")

    assert result.loc[0, "community_type_log_ppsf_mean"] == pytest.approx(math.log(20.0))


def test_feature_cards_add_only_listing_time_features_and_respect_grid_origin():
    homes = pd.DataFrame({
        "house_area": ["1000-1200"], "bedrooms": [2], "washrooms": [3], "parking_total": [1],
        "latitude": [43.61], "longitude": [-79.39], "house_style": ["2-Storey"],
        "municipality": ["Toronto"], "community": ["C"], "house_type_name": ["Detached"],
        "date_listed": pd.to_datetime(["2025-02-01"]), "list_price": [500000.0], "ml_num": ["A"],
    })

    seasonal = build_features(homes, "list_aware", "M03")
    area = build_features(homes, "list_aware", "M04")
    interaction = build_features(homes, "list_aware", "M05")
    native_type = build_features(homes, "list_aware", "M06")
    grid = build_features(homes, "list_aware", "M07", grid_origin=(43.60, -79.40))
    combined = build_features(homes, "list_aware", "M04+M05")

    assert seasonal.loc[0, "listed_month_sin"] == pytest.approx(0.5)
    assert area.loc[0, "log_area"] == pytest.approx(math.log(1100.0))
    assert area.loc[0, "area_range_width"] == pytest.approx(200.0)
    assert interaction.loc[0, "area_per_bedroom"] == pytest.approx(550.0)
    assert "house_type_name" in native_type and not any(c.startswith("is_") for c in native_type)
    assert grid.loc[0, "grid_cell"] == "0:0"
    assert "log_area" in combined and "area_per_bedroom" in combined
    assert not {"sold_price", "date_sold"}.intersection(seasonal.columns)


def test_m08_and_m09_are_separate_methods_and_combine_only_when_requested():
    history = pd.DataFrame({"ml_num": ["OLD"], "date_sold": pd.to_datetime(["2025-01-01"]),
        "sold_price": [200000.0], "house_area": ["1000-1000"], "bedrooms": [2], "washrooms": [1],
        "parking_total": [1], "latitude": [43.6], "longitude": [-79.4], "house_style": ["2-Storey"],
        "municipality": ["Toronto"], "community": ["C"], "house_type_name": ["Detached"],
        "date_listed": pd.to_datetime(["2024-12-01"])})
    query = history.drop(columns=["date_sold", "sold_price"]).assign(
        ml_num="Q", date_listed=pd.to_datetime(["2025-02-01"]))

    m08 = build_features(query, "off_market", "M08", history=history, cutoff="2025-03-01")
    m09 = build_features(query, "off_market", "M09", history=history, cutoff="2025-03-01")
    combined = build_features(query, "off_market", "M08+M09", history=history, cutoff="2025-03-01")

    assert "community_type_log_ppsf_mean" in m08
    assert "type_log_ppsf_mean_90d" not in m08
    assert "type_log_ppsf_mean_90d" in m09
    assert "community_type_log_ppsf_mean" not in m09
    assert {"community_type_log_ppsf_mean", "type_log_ppsf_mean_90d"}.issubset(combined.columns)


def test_huber_and_mape_use_explicit_actual_objectives():
    huber = make_learner("lightgbm", {"objective": "huber", "alpha": 0.9})
    mape = make_learner("lightgbm", {"objective": "mape"})

    assert validate_actual_objective(huber, "lightgbm", "huber") == "huber"
    assert validate_actual_objective(mape, "lightgbm", "mape") == "mape"
    with pytest.raises(ValueError, match="objective"):
        validate_actual_objective(huber, "lightgbm", "regression")


def test_target_transforms_and_area_inverse_are_explicit():
    homes = pd.DataFrame({"sold_price": [400.0], "list_price": [200.0], "square_feet": [20.0]})

    area_target, area_name = training_target(homes, "off_market", "M12")
    dollar_target, dollar_name = training_target(homes, "off_market", "M13")
    ratio_target, ratio_name = training_target(homes, "list_aware", "C0")

    assert area_name == "log_price_per_area" and area_target[0] == pytest.approx(math.log(20.0))
    assert inverse_price([math.log(20.0)], homes, "off_market", "M12")[0] == pytest.approx(400.0)
    assert dollar_name == "dollar_price" and dollar_target.tolist() == [400.0]
    assert ratio_name == "log_sold_over_list" and ratio_target[0] == pytest.approx(math.log(2.0))


def test_incremental_method_bundle_preserves_parent_mechanism():
    train = pd.DataFrame({"date_sold": pd.to_datetime(["2025-02-28"]), "sold_price": [400.0],
                          "house_area": ["20"], "list_price": [200.0]})

    target, name = training_target(train, "off_market", "M01+M12")
    weights = recency_weights(train, "2025-03", "M01+M03")

    assert name == "log_price_per_area"
    assert target[0] == pytest.approx(math.log(20.0))
    assert weights.tolist() == pytest.approx([1.0])


def test_calibration_uses_only_strictly_past_oof_residuals():
    oof = pd.DataFrame({"ml_num": ["OLD", "FUTURE"],
        "date_sold": pd.to_datetime(["2025-02-01", "2025-06-01"]),
        "sold_price": [110.0, 1000.0], "predicted_price": [100.0, 100.0],
        "house_type_name": ["Detached", "Detached"], "latitude": [43.6, 43.6],
        "longitude": [-79.4, -79.4]})
    params = fit_multiplier(oof, "2025-06-01")

    assert params["n"] == 1
    assert params["max_oof_date"] == "2025-02-01"
    assert apply_multiplier_calibration([100.0], params)[0] == pytest.approx(105.0)

    homes = pd.DataFrame({"ml_num": ["Q"], "date_listed": pd.to_datetime(["2025-06-10"]),
        "house_type_name": ["Detached"], "latitude": [43.6], "longitude": [-79.4]})
    adjusted, audit = apply_local_residual([100.0], homes, oof, "2025-06-01")
    assert audit["coverage"] == 1
    assert adjusted[0] > 100.0


def test_recency_weights_use_exclusive_cutoff_and_24_month_window():
    train = pd.DataFrame({"date_sold": pd.to_datetime(["2024-03-01", "2025-03-01", "2025-02-28"])})

    weights = recency_weights(train, "2025-03", "M01")
    rolling = select_training_rows(train, "2025-03", "M02")

    assert weights[1] == pytest.approx(weights[0] * 2.0)
    assert weights.mean() == pytest.approx(1.0)
    assert rolling.date_sold.min() >= pd.Timestamp("2023-03-01")
    assert rolling.date_sold.max() < pd.Timestamp("2025-03-01")


def test_component_fit_reports_the_fitted_objective_and_returns_positive_prices():
    count = 24
    train = pd.DataFrame({
        "ml_num": [f"T{i}" for i in range(count)],
        "date_sold": pd.date_range("2024-01-01", periods=count, freq="20D"),
        "date_listed": pd.date_range("2023-12-15", periods=count, freq="20D"),
        "sold_price": [400000.0 + 1000.0 * i for i in range(count)],
        "house_area": ["1000-1100"] * count, "bedrooms": [3] * count,
        "washrooms": [2] * count, "parking_total": [1] * count,
        "latitude": [43.6] * count, "longitude": [-79.4] * count,
        "house_style": ["2-Storey"] * count, "municipality": ["Toronto"] * count,
        "community": ["C"] * count, "house_type_name": ["Detached"] * count,
    })
    homes = train.iloc[-3:].drop(columns=["date_sold", "sold_price"]).copy()

    result = fit_component(train, homes, "off_market", "C0", "2025-06", "lightgbm",
                           {"n_estimators": 3, "num_leaves": 3, "min_child_samples": 2,
                            "objective": "regression", "n_jobs": 1})

    assert result["actual_objective"] == "regression"
    assert len(result["predictions"]) == 3
    assert all(value > 0 for value in result["predictions"])


def test_increment_evaluation_uses_same_homes_and_reports_month_stability():
    months = ["2025-06", "2025-07", "2025-10", "2025-11"]
    parent = pd.DataFrame({"ml_num": [f"{m}-{i}" for m in months for i in range(30)],
                           "month": [m for m in months for _ in range(30)],
                           "sold_price": [100.0] * 120, "predicted_price": [110.0] * 120})
    candidate = parent.copy()
    candidate["predicted_price"] = 101.0

    result = evaluate_increment(parent, candidate, repeats=500)

    assert result["relative_mdape_improvement"] >= 0.08
    assert result["months_improved"] == 4
    assert result["paired_home_bootstrap"]["ci95_percentage_points"][1] < 0
    assert result["month_cluster_sensitivity"]["ci95_percentage_points"][1] < 0


def test_increment_evaluation_rejects_unpaired_ids():
    parent = pd.DataFrame({"ml_num": ["A"], "month": ["2025-06"],
                           "sold_price": [100.0], "predicted_price": [110.0]})
    candidate = parent.assign(ml_num=["B"])

    with pytest.raises(ValueError, match="align"):
        evaluate_increment(parent, candidate, repeats=20)


def test_list_aware_blend_rounds_frozen_component_before_mixing():
    result = blend_prices(challenger=[100.49], frozen=[100.51], challenger_weight=0.75)

    assert result.tolist() == [101.0]


def test_pipeline_fingerprint_covers_postprocessing_and_dependencies():
    base = {"model": {"name": "lightgbm"}, "blend": 0.75, "round_rule": "frozen_first"}
    a = pipeline_fingerprint(base, "data", {"runner.py": "one"}, {"lightgbm": "4.7.0"})
    b = pipeline_fingerprint({**base, "round_rule": "final_only"}, "data", {"runner.py": "one"}, {"lightgbm": "4.7.0"})
    c = pipeline_fingerprint(base, "data", {"runner.py": "one"}, {"lightgbm": "4.7.1"})

    assert a != b
    assert a != c


def test_cache_reuse_after_scoped_asof_feature_change_is_method_limited():
    previous = {"runner.py": "old-runner", "features.py": "old-features", "worker.py": "same"}
    current = {"runner.py": "new-runner", "features.py": "new-features", "worker.py": "same"}

    assert can_reuse_after_scoped_code_change("M01", previous, current)
    assert not can_reuse_after_scoped_code_change("M08", previous, current)
    assert not can_reuse_after_scoped_code_change("M01", previous, {**current, "worker.py": "different"})


def test_incremental_method_config_keeps_mape_target_and_adds_one_feature_card():
    family, config, target_override = _method_config("M13+M04", "off_market", {"objective": "regression"})

    assert family == "lightgbm"
    assert config["objective"] == "mape"
    assert target_override is None


def test_registry_contains_exactly_twenty_unique_cards_with_known_statuses():
    registry = [{"id": f"M{i:02d}", "status": "not_started", "tracks": ["off_market", "list_aware"]}
                for i in range(1, 21)]

    validate_registry(registry)
    with pytest.raises(ValueError, match="unique"):
        validate_registry(registry + [registry[0]])


def test_fit_gate_preserves_final_hour_until_locked_acceptance_phase():
    assert not budget_allows_fit(elapsed_seconds=10_700, stage="A1", timeout_seconds=180)
    assert budget_allows_fit(elapsed_seconds=10_500, stage="A4", timeout_seconds=300)
    assert not budget_allows_fit(elapsed_seconds=14_200, stage="A4", timeout_seconds=300)


def test_final_benchmark_resolves_track_specific_frozen_c0_metrics():
    protocol = {"c0": {"off_market_mdape_percent": 9.17, "off_market_mape_percent": 11.98,
                       "list_aware_mdape_percent": 4.23, "list_aware_mape_percent": 5.35}}

    assert _protocol_c0_metrics(protocol, "off_market") == {
        "mdape_percent": 9.17, "mape_percent": 11.98}
    assert _protocol_c0_metrics(protocol, "list_aware") == {
        "mdape_percent": 4.23, "mape_percent": 5.35}
