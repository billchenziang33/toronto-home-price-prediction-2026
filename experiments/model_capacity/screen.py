"""Screen and confirm LightGBM capacity presets on top of the current best feature sets.

Usage:
    python experiments/model_capacity/screen.py screen  <out.csv>
    python experiments/model_capacity/screen.py confirm <out.csv> <off_market_config> <list_aware_config>
See protocol.json for the fixed configs, months, and selection rule.
"""
import importlib.util, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("bh", HERE.parent / "building_history" / "screen.py")
bh = importlib.util.module_from_spec(spec); spec.loader.exec_module(bh)
spec = importlib.util.spec_from_file_location("px", HERE.parent / "property_extras" / "screen.py")
px = importlib.util.module_from_spec(spec); spec.loader.exec_module(px)
L = bh.L
PROTOCOL = json.loads((HERE / "protocol.json").read_text())
CONFIGS = {"P0": {"off_market": bh.OM, "list_aware": bh.LA}}
for name in ("P1", "P2", "P3", "P4"):
    CONFIGS[name] = {"off_market": PROTOCOL["configs"][name], "list_aware": PROTOCOL["configs"][name]}


def features(track, train, test):
    feat = bh.f2.off_market_features if track == "off_market" else bh.f2.list_aware_features
    Xtr = bh.add_building(feat(train), train, train, train.date_sold)
    cutoff = pd.Series(train.date_sold.max() + pd.Timedelta(days=1), index=test.index)
    Xte = bh.add_building(feat(test), test, train, cutoff)
    if track == "off_market":
        Xtr = pd.concat([Xtr, px.extras(train)], axis=1)
        Xte = pd.concat([Xte, px.extras(test)], axis=1)
    for c in bh.CATS:
        cats = pd.Categorical(Xtr[c]).categories
        Xtr[c] = pd.Categorical(Xtr[c], categories=cats); Xte[c] = pd.Categorical(Xte[c], categories=cats)
    return Xtr, Xte


def run(months, arms):
    """arms: {arm_name: {track: config_name}}"""
    rows, timing = [], []
    for month in months:
        start = pd.Timestamp(month + "-01"); end = start + pd.offsets.MonthBegin(1)
        train = L[L.date_sold < start]; ev = L[(L.date_sold >= start) & (L.date_sold < end)]
        for track in ["off_market", "list_aware"]:
            hidden = ["sold_price"] if track == "list_aware" else ["sold_price", "list_price"]
            test = ev.drop(columns=hidden)
            if track == "list_aware":
                test = test.assign(list_price=ev.list_price)
            Xtr, Xte = features(track, train, test)
            y = train.lp if track == "off_market" else np.log(train.sold_price / train.list_price)
            for arm, per_track in arms.items():
                cfg = CONFIGS[per_track[track]][track]
                t0 = time.time()
                m = lgb.LGBMRegressor(objective="regression", random_state=0, n_jobs=4, deterministic=True,
                                      force_col_wise=True, verbose=-1, **cfg).fit(Xtr, y)
                p = m.predict(Xte)
                p = np.exp(p) if track == "off_market" else ev.list_price.to_numpy() * np.exp(p)
                timing.append((month, track, arm, round(time.time() - t0, 1)))
                ape = np.abs(p - ev.sold_price.to_numpy()) / ev.sold_price.to_numpy() * 100
                rows.append(pd.DataFrame({"month": month, "track": track, "arm": arm,
                                          "ml_num": ev.ml_num.to_numpy(), "type": ev.house_type_name.to_numpy(), "ape": ape}))
                print(month, track, arm, f"{np.median(ape):.3f}", f"{timing[-1][-1]}s", flush=True)
    return pd.concat(rows)


def compare(r, track, base, cand, boot=True):
    d = r[r.track == track].pivot_table(index=["month", "ml_num", "type"], columns="arm", values="ape").reset_index()
    a, b = d[base].to_numpy(), d[cand].to_numpy()
    months = d.groupby("month").apply(lambda g: (np.median(g[base]), np.median(g[cand])))
    out = {"base_mdape": np.median(a), "cand_mdape": np.median(b), "gain_pct": (1 - np.median(b) / np.median(a)) * 100,
           "base_mape": a.mean(), "cand_mape": b.mean(), "mape_change_pct": (b.mean() / a.mean() - 1) * 100,
           "months_improved": int(sum(c < m for m, c in months)), "n_months": len(months),
           "worst_month_regression_pct": max((c / m - 1) * 100 for m, c in months),
           "monthly": {k: [round(m, 3), round(c, 3)] for k, (m, c) in months.items()}}
    if boot:
        rng = np.random.default_rng(0)
        boots = [np.median(b[i]) - np.median(a[i]) for i in (rng.integers(0, len(a), len(a)) for _ in range(1000))]
        out["ci_pp"] = [np.percentile(boots, 2.5), np.percentile(boots, 97.5)]
    return out


def main():
    mode, out = sys.argv[1], sys.argv[2]
    if mode == "screen":
        arms = {name: {"off_market": name, "list_aware": name} for name in CONFIGS}
        r = run(PROTOCOL["screening_months"], arms)
        r.to_csv(out, index=False)
        for track in ["off_market", "list_aware"]:
            print(f"== {track}")
            for cand in ("P1", "P2", "P3", "P4"):
                s = compare(r, track, "P0", cand, boot=False)
                ok = s["gain_pct"] >= 1 and s["months_improved"] >= 2
                print(f"  {cand}: {s['base_mdape']:.3f} -> {s['cand_mdape']:.3f} ({s['gain_pct']:+.2f}%), "
                      f"MAPE {s['mape_change_pct']:+.2f}%, months {s['months_improved']}/3, qualifies={ok}")
    else:
        om, la = sys.argv[3], sys.argv[4]
        arms = {"parent": {"off_market": "P0", "list_aware": "P0"}, "cand": {"off_market": om, "list_aware": la}}
        r = run(PROTOCOL["confirmation_months"], arms)
        r.to_csv(out, index=False)
        for track, name in (("off_market", om), ("list_aware", la)):
            if name == "P0":
                continue
            s = compare(r, track, "parent", "cand")
            print(f"== {track} {name}: {json.dumps(s, default=float, indent=1)}")


if __name__ == "__main__":
    main()
