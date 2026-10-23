"""Screen: does an as-of same-coordinate (building) price history help v2's LightGBM?"""
import sys, importlib.util
import numpy as np, pandas as pd, lightgbm as lgb

ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[2])
sys.path.insert(0, ROOT)
import data
spec = importlib.util.spec_from_file_location("f2", f"{ROOT}/model_candidate_v2/features.py")
f2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(f2)

OM = dict(learning_rate=0.05, n_estimators=1000, num_leaves=31, min_child_samples=100, reg_lambda=5)
LA = dict(learning_rate=0.03, n_estimators=800, num_leaves=63, min_child_samples=100, reg_lambda=5)
CATS = ["house_style", "municipality", "community"]

L = data.load("listings").sort_values("date_sold").reset_index(drop=True)
L["k"] = L.latitude.round(6).astype(str) + "_" + L.longitude.round(6).astype(str)
L["u"] = L.k + "|" + L.house_type_name + "|" + L.bedrooms.astype(str) + "|" + L.washrooms.astype(str) + "|" + L.house_area.astype(str)
L["sqft"] = f2.square_feet(L.house_area)
L["lp"] = np.log(L.sold_price)
L["lppsf"] = L.lp - np.log(L.sqft)
L["mon"] = L.date_sold.dt.to_period("M")
# Lagged (previous complete month) type index, so a sale's residual never uses later sales.
idx = L.groupby(["house_type_name", "mon"]).lppsf.median().rename("ix").reset_index()
idx["mon"] = idx["mon"] + 1
L = L.merge(idx, on=["house_type_name", "mon"], how="left")
L["res"] = L.lppsf - L.ix
L = L.sort_values("date_sold").reset_index(drop=True)


def history(sales, queries, key, prefix):
    """Stats of `sales` at the same key sold strictly before each query's time `t`."""
    s = sales[[key, "date_sold", "lp", "lppsf", "res"]].dropna(subset=["date_sold"]).copy()
    s["date_sold"] = s.date_sold.astype("datetime64[ns]"); s = s.sort_values("date_sold")
    g = s.groupby(key)
    s[prefix + "n"] = g.cumcount() + 1
    s[prefix + "res_mean"] = g.res.cumsum() / g.res.cumcount().add(1)
    s[prefix + "last_lp"] = s.lp
    s[prefix + "last_res"] = s.res
    s["last_t"] = s.date_sold
    q = queries[[key, "t"]].reset_index(); q["t"] = q.t.astype("datetime64[ns]"); q = q.sort_values("t")
    m = pd.merge_asof(q, s.drop(columns=["lp", "lppsf", "res"]).rename(columns={"date_sold": "t_s"}),
                      left_on="t", right_on="t_s", by=key, allow_exact_matches=False, direction="backward")
    m[prefix + "age"] = (m.t - m.last_t).dt.days / 30.44
    m = m.set_index("index").sort_index()
    cols = [prefix + c for c in ["n", "res_mean", "last_lp", "last_res", "age"]]
    out = m[cols]
    out[prefix + "n"] = out[prefix + "n"].fillna(0)
    return out


def add_building(X, homes, sales, t):
    q = homes.assign(t=t)
    q["k"] = q.latitude.round(6).astype(str) + "_" + q.longitude.round(6).astype(str)
    q["u"] = q.k + "|" + q.house_type_name + "|" + q.bedrooms.astype(str) + "|" + q.washrooms.astype(str) + "|" + q.house_area.astype(str)
    b = history(sales, q, "k", "b_")
    u = history(sales, q, "u", "u_")[["u_n", "u_last_lp", "u_age"]]
    return pd.concat([X, b.set_index(X.index), u.set_index(X.index)], axis=1)


def fit_predict(track, train, test, use_b):
    feat = f2.off_market_features if track == "off_market" else f2.list_aware_features
    Xtr, Xte = feat(train), feat(test)
    if use_b:
        Xtr = add_building(Xtr, train, train, train.date_sold)      # each train row sees only earlier sales
        cutoff = pd.Series(train.date_sold.max() + pd.Timedelta(days=1), index=test.index)
        Xte = add_building(Xte, test, train, cutoff)                  # test rows see all training sales
    for c in CATS:
        cats = pd.Categorical(Xtr[c]).categories
        Xtr[c] = pd.Categorical(Xtr[c], categories=cats); Xte[c] = pd.Categorical(Xte[c], categories=cats)
    if track == "off_market":
        y, cfg = train.lp, OM
    else:
        y, cfg = np.log(train.sold_price / train.list_price), LA
    m = lgb.LGBMRegressor(objective="regression", random_state=0, n_jobs=4, deterministic=True,
                          force_col_wise=True, verbose=-1, **cfg).fit(Xtr, y)
    p = m.predict(Xte)
    return np.exp(p) if track == "off_market" else test.list_price.to_numpy() * np.exp(p)


def main(months):
    rows = []
    for month in months:
        start = pd.Timestamp(month + "-01"); end = start + pd.offsets.MonthBegin(1)
        train = L[L.date_sold < start]; ev = L[(L.date_sold >= start) & (L.date_sold < end)]
        for track in ["off_market", "list_aware"]:
            test = data.hide_answers(ev, track) if hasattr(data, "hide_answers") else ev
            test = ev.drop(columns=["sold_price"]) if track == "list_aware" else ev.drop(columns=["sold_price", "list_price"])
            for use_b in (False, True):
                p = fit_predict(track, train, test.assign(list_price=ev.list_price) if track == "list_aware" else test, use_b)
                ape = np.abs(p - ev.sold_price.to_numpy()) / ev.sold_price.to_numpy() * 100
                rows.append(pd.DataFrame({"month": month, "track": track, "arm": "building" if use_b else "C0",
                                          "ml_num": ev.ml_num.to_numpy(), "type": ev.house_type_name.to_numpy(), "ape": ape}))
            print(month, track, "done", flush=True)
    r = pd.concat(rows)
    r.to_csv(sys.argv[2] if len(sys.argv) > 2 else "screen.csv", index=False)
    piv = r.pivot_table(index=["track", "month", "ml_num", "type"], columns="arm", values="ape").reset_index()
    rng = np.random.default_rng(0)
    for track, d in piv.groupby("track"):
        a, b = d.C0.to_numpy(), d.building.to_numpy()
        boots = [np.median(b[i]) - np.median(a[i]) for i in (rng.integers(0, len(a), len(a)) for _ in range(1000))]
        print(f"{track}: n={len(d)} MdAPE {np.median(a):.3f} -> {np.median(b):.3f} "
              f"({(1 - np.median(b) / np.median(a)) * 100:+.2f}% gain), MAPE {a.mean():.3f} -> {b.mean():.3f}, "
              f"95% CI pp [{np.percentile(boots, 2.5):.3f}, {np.percentile(boots, 97.5):.3f}]")
        for mo, dm in d.groupby("month"):
            print(f"   {mo}: {np.median(dm.C0):.3f} -> {np.median(dm.building):.3f}")
        for ty, dt in d.groupby("type"):
            print(f"   {ty:20s} n={len(dt):5d}: {np.median(dt.C0):.3f} -> {np.median(dt.building):.3f}")


if __name__ == "__main__":
    main(sys.argv[1].split(","))
