"""Screen: do lot size and garage help on top of building history?

v2's feature function never reads lot_front, lot_depth, or garage. The candidate
adds them (plus lot area) to the building-history parent; nothing else changes.
"""
import importlib.util, sys
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("bh", HERE.parent / "building_history" / "screen.py")
bh = importlib.util.module_from_spec(spec); spec.loader.exec_module(bh)
L = bh.L


def extras(homes):
    X = pd.DataFrame(index=homes.index)
    X["lot_front"] = homes.lot_front
    X["lot_depth"] = homes.lot_depth
    X["lot_area"] = homes.lot_front * homes.lot_depth
    X["garage"] = homes.garage
    return X


def fit_predict(track, train, test, use_extra):
    feat = bh.f2.off_market_features if track == "off_market" else bh.f2.list_aware_features
    Xtr = bh.add_building(feat(train), train, train, train.date_sold)
    cutoff = pd.Series(train.date_sold.max() + pd.Timedelta(days=1), index=test.index)
    Xte = bh.add_building(feat(test), test, train, cutoff)
    if use_extra:
        Xtr = pd.concat([Xtr, extras(train)], axis=1)
        Xte = pd.concat([Xte, extras(test)], axis=1)
    for c in bh.CATS:
        cats = pd.Categorical(Xtr[c]).categories
        Xtr[c] = pd.Categorical(Xtr[c], categories=cats); Xte[c] = pd.Categorical(Xte[c], categories=cats)
    if track == "off_market":
        y, cfg = train.lp, bh.OM
    else:
        y, cfg = np.log(train.sold_price / train.list_price), bh.LA
    m = lgb.LGBMRegressor(objective="regression", random_state=0, n_jobs=4, deterministic=True,
                          force_col_wise=True, verbose=-1, **cfg).fit(Xtr, y)
    p = m.predict(Xte)
    return np.exp(p) if track == "off_market" else test.list_price.to_numpy() * np.exp(p)


def main(months, out):
    rows = []
    for month in months:
        start = pd.Timestamp(month + "-01"); end = start + pd.offsets.MonthBegin(1)
        train = L[L.date_sold < start]; ev = L[(L.date_sold >= start) & (L.date_sold < end)]
        for track in ["off_market", "list_aware"]:
            hidden = ["sold_price"] if track == "list_aware" else ["sold_price", "list_price"]
            test = ev.drop(columns=hidden)
            if track == "list_aware":
                test = test.assign(list_price=ev.list_price)
            for use_extra in (False, True):
                p = fit_predict(track, train, test, use_extra)
                ape = np.abs(p - ev.sold_price.to_numpy()) / ev.sold_price.to_numpy() * 100
                rows.append(pd.DataFrame({"month": month, "track": track, "arm": "extras" if use_extra else "parent",
                                          "ml_num": ev.ml_num.to_numpy(), "type": ev.house_type_name.to_numpy(), "ape": ape}))
            print(month, track, "done", flush=True)
    r = pd.concat(rows)
    r.to_csv(out, index=False)
    piv = r.pivot_table(index=["track", "month", "ml_num", "type"], columns="arm", values="ape").reset_index()
    rng = np.random.default_rng(0)
    for track, d in piv.groupby("track"):
        a, b = d["parent"].to_numpy(), d["extras"].to_numpy()
        boots = [np.median(b[i]) - np.median(a[i]) for i in (rng.integers(0, len(a), len(a)) for _ in range(1000))]
        print(f"{track}: n={len(d)} MdAPE {np.median(a):.3f} -> {np.median(b):.3f} "
              f"({(1 - np.median(b) / np.median(a)) * 100:+.2f}% gain), MAPE {a.mean():.3f} -> {b.mean():.3f}, "
              f"95% CI pp [{np.percentile(boots, 2.5):.3f}, {np.percentile(boots, 97.5):.3f}]")
        for mo, dm in d.groupby("month"):
            print(f"   {mo}: {np.median(dm.parent):.3f} -> {np.median(dm.extras):.3f}")
        for ty, dt in d.groupby("type"):
            print(f"   {ty:20s} n={len(dt):5d}: {np.median(dt.parent):.3f} -> {np.median(dt.extras):.3f}")


if __name__ == "__main__":
    main(sys.argv[1].split(","), sys.argv[2])
