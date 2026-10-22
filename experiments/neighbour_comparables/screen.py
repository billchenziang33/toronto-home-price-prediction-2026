"""Screen: do as-of nearest-neighbour comparables help on top of building history?

Parent arm = building-history model from experiments/building_history.
Candidate arm = parent + features from the k nearest earlier sales of the same property type.
"""
import importlib.util, sys
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("bh", HERE.parent / "building_history" / "screen.py")
bh = importlib.util.module_from_spec(spec); spec.loader.exec_module(bh)
L = bh.L
KS = (10, 30)
WINDOW_MONTHS = 24


def knn_table(L):
    """Per sale: stats of the nearest same-type sales sold before its own sale month.

    Evaluation homes are predicted at the start of their sale month, so the same
    table serves training rows and evaluation rows without seeing later sales.
    """
    km = np.c_[L.longitude.to_numpy() * 111.32 * np.cos(np.radians(43.7)), L.latitude.to_numpy() * 110.57]
    month = L.date_sold.dt.to_period("M")
    ok = L.res.notna().to_numpy()
    sold_days = L.date_sold.to_numpy().astype("datetime64[D]")
    out = np.full((len(L), 4 + len(KS)), np.nan)
    for m in sorted(month.unique()):
        lo = (m - WINDOW_MONTHS).start_time
        q_all = (month == m).to_numpy()
        for kind in L.house_type_name.unique():
            same = (L.house_type_name == kind).to_numpy()
            pool = np.flatnonzero(same & ok & (L.date_sold < m.start_time).to_numpy() & (L.date_sold >= lo).to_numpy())
            q = np.flatnonzero(q_all & same)
            if len(pool) < max(KS) or len(q) == 0:
                continue
            dist, idx = cKDTree(km[pool]).query(km[q], k=max(KS))
            res = L.res.to_numpy()[pool][idx]
            age = (np.datetime64(m.start_time, "D") - sold_days[pool][idx]) / np.timedelta64(1, "D") / 30.44
            k0 = KS[0]
            out[q, 0] = res[:, :k0].mean(1)
            out[q, 1] = dist[:, :k0].mean(1)
            out[q, 2] = age[:, :k0].mean(1)
            out[q, 3] = np.average(res[:, :k0], axis=1, weights=1 / (dist[:, :k0] + 0.05))
            for j, k in enumerate(KS):
                out[q, 4 + j] = np.median(res[:, :k], axis=1)
    cols = ["nn_res_mean10", "nn_dist10", "nn_age10", "nn_res_idw10"] + [f"nn_res_med{k}" for k in KS]
    return pd.DataFrame(out, columns=cols, index=L.index)


KNN = knn_table(L)


def fit_predict(track, train, test, use_knn):
    feat = bh.f2.off_market_features if track == "off_market" else bh.f2.list_aware_features
    Xtr = bh.add_building(feat(train), train, train, train.date_sold)
    cutoff = pd.Series(train.date_sold.max() + pd.Timedelta(days=1), index=test.index)
    Xte = bh.add_building(feat(test), test, train, cutoff)
    if use_knn:
        Xtr = pd.concat([Xtr, KNN.loc[train.index]], axis=1)
        Xte = pd.concat([Xte, KNN.loc[test.index]], axis=1)
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
            for use_knn in (False, True):
                p = fit_predict(track, train, test, use_knn)
                ape = np.abs(p - ev.sold_price.to_numpy()) / ev.sold_price.to_numpy() * 100
                rows.append(pd.DataFrame({"month": month, "track": track, "arm": "knn" if use_knn else "parent",
                                          "ml_num": ev.ml_num.to_numpy(), "type": ev.house_type_name.to_numpy(), "ape": ape}))
            print(month, track, "done", flush=True)
    r = pd.concat(rows)
    r.to_csv(out, index=False)
    summarize(r, "parent", "knn")


def summarize(r, base, cand):
    piv = r.pivot_table(index=["track", "month", "ml_num", "type"], columns="arm", values="ape").reset_index()
    rng = np.random.default_rng(0)
    for track, d in piv.groupby("track"):
        a, b = d[base].to_numpy(), d[cand].to_numpy()
        boots = [np.median(b[i]) - np.median(a[i]) for i in (rng.integers(0, len(a), len(a)) for _ in range(1000))]
        print(f"{track}: n={len(d)} MdAPE {np.median(a):.3f} -> {np.median(b):.3f} "
              f"({(1 - np.median(b) / np.median(a)) * 100:+.2f}% gain), MAPE {a.mean():.3f} -> {b.mean():.3f}, "
              f"95% CI pp [{np.percentile(boots, 2.5):.3f}, {np.percentile(boots, 97.5):.3f}]")
        for mo, dm in d.groupby("month"):
            print(f"   {mo}: {np.median(dm[base]):.3f} -> {np.median(dm[cand]):.3f}")
        for ty, dt in d.groupby("type"):
            print(f"   {ty:20s} n={len(dt):5d}: {np.median(dt[base]):.3f} -> {np.median(dt[cand]):.3f}")


if __name__ == "__main__":
    main(sys.argv[1].split(","), sys.argv[2])
