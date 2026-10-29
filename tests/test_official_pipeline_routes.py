import json

import numpy as np
import pandas as pd

import run


class FakeCandidateModel:
    def __init__(self, track):
        self.track = track
        self.cutoffs = []
        self.route_names = ("R22", "R52", "R151") if track == "off_market" else ("R35", "R97")

    def fit(self, train):
        self.train = train.copy()
        return self

    def predict_routes(self, homes, cutoff):
        assert "sold_price" not in homes
        assert "date_sold" not in homes
        if self.track == "off_market":
            assert "list_price" not in homes
        assert (self.train.date_sold < cutoff).all()
        self.cutoffs.append(pd.Timestamp(cutoff))
        return {name: np.full(len(homes), 100_000 + index)
                for index, name in enumerate(self.route_names)}

    def predict(self, homes):
        return self.predict_routes(homes, pd.Timestamp("2026-01-01"))[self.route_names[-1]]


def test_validate_writes_every_frozen_route_with_explicit_month_cutoff(monkeypatch, tmp_path):
    listings = pd.DataFrame({
        "ml_num": ["train-1", "off-1", "list-1"],
        "date_sold": pd.to_datetime(["2025-11-15", "2026-01-10", "2026-01-11"]),
        "date_listed": pd.to_datetime(["2025-10-01", "2025-12-01", "2025-12-02"]),
        "sold_price": [90_000.0, 110_000.0, 120_000.0],
        "list_price": [95_000.0, 115_000.0, 125_000.0],
        "house_type_name": ["Detached", "Detached", "Condo Apt"],
        "community": ["A", "A", "B"],
    })
    models = {track: FakeCandidateModel(track) for track in run.TRACKS}
    training_path = tmp_path / "listings.parquet"
    training_path.write_bytes(b"test training data fingerprint")
    monkeypatch.setattr(run.data, "load", lambda name: listings.copy())
    monkeypatch.setattr(run.data, "find_file", lambda name: training_path)
    monkeypatch.setattr(run.models, "make_model", lambda track: models[track])
    monkeypatch.setattr(run, "model_manifest", lambda: {"routes": {"test": True}}, raising=False)
    monkeypatch.setattr(run, "VALIDATION_MONTHS", pd.period_range("2026-01", periods=1, freq="M"))
    monkeypatch.setattr(run, "ROOT", tmp_path)

    run.validate()

    predictions = pd.read_csv(tmp_path / "outputs" / "validation_predictions.csv")
    assert len(predictions) == 10
    assert set(zip(predictions.track, predictions.model)) == {
        ("off_market", "R22"), ("off_market", "R52"), ("off_market", "R151"),
        ("list_aware", "R35"), ("list_aware", "R97"),
    }
    assert set(predictions.ml_num) == {"off-1", "list-1"}
    assert all(model.cutoffs == [pd.Timestamp("2026-01-01")] for model in models.values())
    manifest = json.loads((tmp_path / "outputs" / "validation_run_manifest.json").read_text())
    assert manifest["run_id"].startswith("validation_")
    assert manifest["validation_scope"] == "retrospective standard validation; not an independent holdout"
    assert manifest["validation_months"] == ["2026-01"]
