import numpy as np
import pandas as pd

import features
from experiments.remarks_text import score_pair


def test_remarks_topic_features_count_predeclared_phrases():
    remarks = pd.Series([
        "Recently renovated, needs work on the electrical panel; near transit and schools.",
        "Ravine views, luxury finishes, and income potential.",
    ])

    result = features.remarks_features(remarks)

    assert result.loc[0, "remarks_renovation_count"] == 1
    assert result.loc[0, "remarks_condition_count"] == 1
    assert result.loc[0, "remarks_systems_count"] == 1
    assert result.loc[0, "remarks_transit_schools_count"] == 2
    assert result.loc[1, "remarks_amenities_count"] >= 1
    assert result.loc[1, "remarks_views_count"] == 1
    assert result.loc[1, "remarks_investment_count"] == 1
    assert result.loc[1, "remarks_luxury_count"] == 1


def test_missing_remarks_and_append_preserve_rows_and_category_dtype():
    structured = pd.DataFrame({
        "community": pd.Categorical(["North", "South"], categories=["North", "South"]),
        "square_feet": [700.0, 900.0],
    }, index=[7, 3])

    combined = features.append_remarks_features(structured, pd.Series([None, "Bright home"], index=[7, 3]))

    assert combined.index.tolist() == [7, 3]
    assert combined["community"].dtype == structured["community"].dtype
    assert combined["square_feet"].tolist() == [700.0, 900.0]
    assert combined.loc[7, "remarks_word_count"] == 0
    assert np.isfinite(combined.select_dtypes(include=["number"]).to_numpy()).all()


def test_default_prediction_tracks_include_remarks_but_keep_list_price_separate():
    homes = pd.DataFrame({
        "house_area": ["700"], "bedrooms": [2], "washrooms": [1], "parking_total": [1],
        "latitude": [43.7], "longitude": [-79.4], "house_style": ["Apartment"],
        "municipality": ["Toronto"], "community": ["Test"],
        "date_listed": [pd.Timestamp("2026-01-01")], "house_type_name": ["Condo Apt"],
        "remarks": ["Recently renovated"], "list_price": [800_000],
    })

    off_market = features.off_market_features(homes)
    list_aware = features.list_aware_features(homes)

    assert off_market.loc[0, "remarks_renovation_count"] == 1
    assert "list_price" not in off_market
    assert list_aware.loc[0, "remarks_renovation_count"] == 1
    assert list_aware.loc[0, "list_price"] == 800_000


def test_score_pair_reports_remarks_minus_baseline():
    rows = pd.DataFrame({
        "ml_num": ["a", "b", "c"],
        "sold_price": [100.0, 100.0, 100.0],
        "baseline_price": [80.0, 100.0, 120.0],
        "remarks_price": [90.0, 100.0, 110.0],
    })

    result = score_pair(rows, repeats=100)

    assert result["baseline_mdape"] == 20.0
    assert result["remarks_mdape"] == 10.0
    assert result["mdape_delta_pp_remarks_minus_baseline"] == -10.0
    assert result["mape_delta_pp_remarks_minus_baseline"] == -6.667
    assert result["paired_mdape_delta_95ci"][0] <= -10.0 <= result["paired_mdape_delta_95ci"][1]
