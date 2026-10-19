from __future__ import annotations

import pandas as pd


def build_month_fold(listings: pd.DataFrame, month: str, track: str):
    """Return past-only training rows, answer-hidden homes, and keyed labels."""
    if track not in {"off_market", "list_aware"}:
        raise ValueError(f"unknown track: {track}")
    dates = pd.to_datetime(listings["date_sold"])
    start = pd.Period(month, freq="M").start_time
    mask = dates.dt.to_period("M").astype(str).eq(month)
    train = listings.loc[dates < start].copy()
    heldout = listings.loc[mask].copy()
    if train.empty or heldout.empty:
        raise ValueError(f"empty training or evaluation fold: {month}")
    if (pd.to_datetime(train.date_sold) >= start).any():
        raise AssertionError("walk-forward training includes the evaluation month")
    truth = heldout[["ml_num", "sold_price"]].copy()
    from data import hide_answers
    homes = hide_answers(heldout, track)
    if {"sold_price", "date_sold"}.intersection(homes.columns):
        raise AssertionError("sale outcome fields were not hidden")
    if track == "off_market" and "list_price" in homes:
        raise AssertionError("off-market homes must not expose list_price")
    return train, homes, truth
