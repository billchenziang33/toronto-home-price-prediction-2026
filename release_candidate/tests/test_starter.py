"""Quick checks on tiny made-up data (nothing is downloaded). Run from the repo folder:

    pytest
"""
import numpy as np
import pandas as pd
import pytest

import check_submission
import data
import features
import models
import run


def make_listings(n, seed=0, first_id=0):
    """A tiny made-up listings table with the same columns as the real one."""
    rng = np.random.default_rng(seed)
    sold = pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 516, n), unit="D")
    price = rng.uniform(4e5, 2e6, n).round(-2)
    return pd.DataFrame({
        "ml_num": [f"SYN-{first_id + i:07d}" for i in range(n)],
        "sold_price": price,
        "list_price": (price * rng.uniform(0.9, 1.1, n)).round(-2),
        "date_listed": sold - pd.to_timedelta(rng.integers(1, 60, n), unit="D"),
        "date_sold": sold,
        "days_on_market": 10,
        "house_type_name": rng.choice(["Condo Apt", "Detached"], n),
        "house_style": "Apartment",
        "bedrooms": rng.integers(1, 5, n),
        "washrooms": rng.integers(1, 4, n),
        "parking_total": rng.integers(0, 3, n),
        "garage": 0.0,
        "house_area": rng.choice(["600-699", "1200"], n),
        "lot_front": np.nan,
        "lot_depth": np.nan,
        "municipality": "Toronto",
        "community": "Testville",
        "latitude": 43.7 + rng.normal(0, 0.05, n),
        "longitude": -79.4 + rng.normal(0, 0.05, n),
        "external_avm_estimate": price,
        "external_avm_estimate_date": sold,
        "remarks": "A home.",
    })


# ---- the nearest-sales baseline never looks into the future ----

def test_nearest_sales_ignores_later_sales():
    listings = make_listings(400)
    homes = data.hide_answers(make_listings(50, seed=1, first_id=1000), "off_market")
    homes = homes[homes["date_listed"] > pd.Timestamp("2025-03-01")]   # so earlier sales exist
    with_later_sales = models.NearestSalesBaseline(k=10).fit(listings).predict(homes)
    for i, (_, home) in enumerate(homes.iterrows()):
        earlier_only = listings[listings["date_sold"] < home["date_listed"]]
        alone = models.NearestSalesBaseline(k=10).fit(earlier_only).predict(homes.loc[[home.name]])
        assert alone[0] == with_later_sales[i]


def test_a_later_sale_next_door_is_not_used():
    listings = make_listings(400)
    home = listings.iloc[[0]].assign(ml_num="SYN-8888888", latitude=44.0, date_listed=pd.Timestamp("2026-01-15"))
    next_door = home.assign(ml_num="SYN-9999999", date_sold=pd.Timestamp("2026-02-01"), sold_price=1e9)
    model = models.NearestSalesBaseline(k=1).fit(pd.concat([listings, next_door]))
    assert model.predict(data.hide_answers(home, "off_market"))[0] < 1e9


def test_nearest_sales_uses_the_nearest_homes_of_the_same_type():
    def sales(n, kind, latitude, price, first_id):
        rows = make_listings(n, first_id=first_id)
        return rows.assign(house_type_name=kind, latitude=latitude, longitude=-79.4,
                           sold_price=price, date_sold=pd.Timestamp("2025-01-01"))
    listings = pd.concat([sales(10, "Detached", 43.70, 500_000, 0),     # near, same type
                          sales(10, "Condo Apt", 43.70, 2_000_000, 100),  # near, other type
                          sales(10, "Detached", 43.90, 100_000, 200)])    # far, same type
    home = listings.iloc[[0]].assign(date_listed=pd.Timestamp("2026-01-01"))
    model = models.NearestSalesBaseline(k=10).fit(listings)
    assert model.predict(data.hide_answers(home, "off_market"))[0] == 500_000


# ---- check_submission rejects bad files ----

IDS = ["SYN-0000001", "SYN-0000002", "SYN-0000003"]
GOOD = "ml_num,predicted_price\nSYN-0000001,500000\nSYN-0000002,600000\nSYN-0000003,700000\n"
BAD_FILES = {
    "wrong column name": GOOD.replace("predicted_price", "price"),
    "extra column": GOOD.replace("predicted_price\n", "predicted_price,x\n"),
    "a home missing": "ml_num,predicted_price\nSYN-0000001,500000\nSYN-0000002,600000\n",
    "a home twice": GOOD + "SYN-0000003,700000\n",
    "an id from another file": GOOD + "SYN-0000004,700000\n",
    "the other track's homes": "ml_num,predicted_price\nSYN-0000007,500000\nSYN-0000008,600000\n",
    "empty price": GOOD.replace("700000", ""),
    "price is text": GOOD.replace("700000", "seven hundred"),
    "negative price": GOOD.replace("700000", "-700000"),
    "zero price": GOOD.replace("700000", "0"),
    "infinite price": GOOD.replace("700000", "inf"),
    "empty file": "",
}


def test_check_accepts_a_good_file(tmp_path):
    path = tmp_path / "submission.csv"
    path.write_text(GOOD)
    assert check_submission.check_file(path, IDS) == []


@pytest.mark.parametrize("problem", BAD_FILES)
def test_check_rejects_a_bad_file(tmp_path, problem):
    path = tmp_path / "submission.csv"
    path.write_text(BAD_FILES[problem])
    assert check_submission.check_file(path, IDS) != []


# ---- the report computes the metrics correctly ----

def test_metrics_on_known_errors():
    # Errors of -10%, +10%, +20% and 0%.
    predictions = pd.DataFrame({"sold_price": [100.0] * 4, "predicted_price": [90.0, 110.0, 120.0, 100.0]})
    scores = run.metrics(predictions)
    assert scores["n"] == 4
    assert scores["mape"] == 10.0                  # mean of 10, 10, 20, 0
    assert scores["mdape"] == 10.0                 # median of 0, 10, 10, 20
    assert scores["median_signed_error"] == 5.0    # median of -10, 0, 10, 20
    low, high = scores["mape_95ci"]
    assert low <= 10.0 <= high


def test_price_deciles_use_shared_cut_points_across_tracks():
    prices = pd.Series([100, 200, 300, 400] * 5 + [100, 200, 300, 400] * 5)
    tracks = pd.Series(["off_market"] * 20 + ["list_aware"] * 20)
    deciles = run.price_decile_labels(prices)
    assignments = pd.DataFrame({"price": prices, "track": tracks, "decile": deciles})
    assert assignments.groupby("price")["decile"].nunique().max() == 1


@pytest.mark.parametrize("track, expected_features", [
    ("off_market", features.off_market_features),
    ("list_aware", features.list_aware_features),
])
def test_make_model_selects_the_feature_model_for_each_track(track, expected_features):
    model = models.make_model(track)
    assert isinstance(model, models.FeatureModel)
    assert model.make_features is expected_features


def test_feature_model_encodes_categories_and_ignores_unseen_values():
    train = make_listings(80)
    homes = data.hide_answers(make_listings(3, seed=3, first_id=9000), "off_market")
    homes["community"] = "NeverSeenCommunity"
    homes["municipality"] = "NeverSeenMunicipality"
    homes["house_style"] = "NeverSeenStyle"

    X = features.off_market_features(data.hide_answers(train, "off_market"))
    assert {"community", "municipality", "house_style"}.issubset(X.columns)
    model = models.make_model("off_market").fit(train)
    predictions = model.predict(homes)
    assert predictions.shape == (3,)
    assert np.isfinite(predictions).all()
    assert model.model._bin_mapper.is_categorical_.sum() == 3


# ---- every command runs end to end on tiny files ----

def test_whole_pipeline_on_tiny_data(tmp_path, monkeypatch):
    folder = tmp_path / "data"
    folder.mkdir()
    make_listings(1500).to_parquet(folder / "listings.parquet")
    later = make_listings(40, seed=2, first_id=5000)
    data.hide_answers(later.iloc[:20], "off_market").to_parquet(folder / "test_off_market.parquet")
    data.hide_answers(later.iloc[20:], "list_aware").to_parquet(folder / "test_list_aware.parquet")
    monkeypatch.setenv("DATA_DIR", str(folder))
    monkeypatch.setattr(run, "ROOT", tmp_path)

    run.validate()
    run.report()
    run.predict()
    assert run.check()
    assert (tmp_path / "figures" / "error_by_price_range.png").exists()
