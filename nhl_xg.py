"""
nhl_xg.py
----------
Fits an expected-goals (xG) model on data/shots.parquet (from nhl_shots.py), scores every shot,
and writes team and goalie xG tables you can feed into the game model.

The model: logistic regression, P(goal | shot), on
    distance  (piecewise-linear spline, knots at 10/20/35/55 ft)
    angle     (linear + squared)
    shot type (wrist = reference; snap, slap, backhand, tip-in, deflected, wrap-around, other)
    strength  (EV = reference; PP, SH)
    rebound, rush, and rebound x angle
Empty-net shots get their own small distance-only model; penalty shots get the league conversion rate.

Steps
  1. Holdout check: fit on all seasons but the last complete one, test on it
     (log-loss and AUC vs a constant and a distance-only model, calibration by decile).
  2. Refit on everything and score every shot -> data/shot_xg.parquet
  3. Per team-game xG          -> data/game_xg.parquet   (xgf/xga all situations, xgf_reg/xga_reg periods 1-3
                                                          only (what nhl_model.py uses), EV, empty-net split out)
  4. Per goalie-game GSAx      -> data/goalie_xg.parquet (xG faced minus goals allowed, unblocked attempts)
  5. Coefficients              -> model/xg_params.json   (enough to score new shots without refitting)
  6. Report: do xG-based numbers predict future results better than goals and shot counts?

Usage
  python nhl_xg.py                   # fit, score, write tables, print the report
  python nhl_xg.py --rink-adjust     # also correct each arena's shot-distance recording bias
  python nhl_xg.py --score-only      # score new shots with the saved model/xg_params.json (fast; for daily refresh)
  python nhl_xg.py --no-report       # skip step 6
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
PARAMS_PATH = ROOT / "model" / "xg_params.json"

DIST_KNOTS = [10, 20, 35, 55]
SHOT_TYPES = ["snap", "slap", "backhand", "tip-in", "deflected", "wrap-around"]   # wrist = reference
RINK_SHRINK = 400           # shots needed before an arena's distance bias counts at half weight
                            # (a full season is ~3,500 shots per arena, so that keeps ~90% of a real bias)
L2 = 1e-4


# ======================================================================================
# Data
# ======================================================================================

def _read(name, required=True):
    for ext, fn in ((".parquet", pd.read_parquet), (".csv", pd.read_csv)):
        p = DATA / f"{name}{ext}"
        if p.exists():
            return fn(p)
    if required:
        raise FileNotFoundError(f"data/{name}.parquet not found - run nhl_shots.py first")
    return None


def _save(df, name):
    df.to_parquet(DATA / f"{name}.parquet", index=False)
    df.to_csv(DATA / f"{name}.csv", index=False)
    print(f"  saved {len(df):,} rows -> data/{name}.parquet (+ .csv)")


def load_shots():
    s = _read("shots")
    s = s[s["game_type"].isin([2, 3])].copy()
    s["season"] = s["season"].astype(int)
    s["game_date"] = pd.to_datetime(s["game_date"])
    for c in ("is_goal", "empty_net", "penalty_shot", "rebound", "rush", "is_home"):
        s[c] = s[c].astype(bool)
    s["dist"] = s["dist"].clip(1, 100)
    s["angle"] = s["angle"].clip(0, 180)
    s["shot_type"] = s["shot_type"].fillna("unknown").str.lower()
    return s.reset_index(drop=True)


# ======================================================================================
# Rink bias (optional): some scorers record shots systematically closer/farther than others
# ======================================================================================

def rink_bias(s):
    """Distance bias per (season, home arena) in feet, shrunk toward 0.
    For each shot: residual = distance - the shooting team's average distance in OTHER teams' arenas.
    An arena's bias = average residual of all shots taken there (both teams)."""
    reg = s[~s["empty_net"] & ~s["penalty_shot"]]
    road = reg[~reg["is_home"]].groupby(["season", "team"])["dist"].mean()
    resid = reg["dist"] - pd.Series(list(zip(reg["season"], reg["team"]))).map(road).values
    g = pd.DataFrame({"season": reg["season"].values, "home_team": reg["home_team"].values, "r": resid.values})
    agg = g.dropna().groupby(["season", "home_team"])["r"].agg(["mean", "size"])
    agg["bias"] = agg["mean"] * agg["size"] / (agg["size"] + RINK_SHRINK)
    return {f"{int(se)}|{t}": round(float(b), 3) for (se, t), b in agg["bias"].items()}


def apply_rink(s, bias):
    if not bias:
        return s["dist"]
    key = s["season"].astype(str) + "|" + s["home_team"].astype(str)
    return (s["dist"] - key.map(bias).fillna(0.0)).clip(1, 100)


# ======================================================================================
# Features + logistic fit
# ======================================================================================

def design(s, dist=None):
    """Design matrix for the main model. Column order is fixed and saved with the coefficients."""
    d = (s["dist"] if dist is None else dist).values.astype(float)
    a = s["angle"].values.astype(float) / 45.0
    st = s["shot_type"].values
    reb = s["rebound"].values.astype(float)
    cols = {"intercept": np.ones(len(s)), "dist": d / 10}
    for k in DIST_KNOTS:
        cols[f"dist_gt_{k}"] = np.maximum(d - k, 0) / 10
    cols["angle"] = a
    cols["angle_sq"] = a * a
    known = set(SHOT_TYPES) | {"wrist"}
    for t in SHOT_TYPES:
        cols[f"type_{t}"] = (st == t).astype(float)
    cols["type_other"] = (~pd.Series(st).isin(known)).values.astype(float)
    cols["pp"] = (s["strength"].values == "PP").astype(float)
    cols["sh"] = (s["strength"].values == "SH").astype(float)
    cols["rebound"] = reb
    cols["rush"] = s["rush"].values.astype(float)
    cols["rebound_x_angle"] = reb * a
    return np.column_stack(list(cols.values())), list(cols)


def fit_logit(X, y, l2=L2):
    n = len(y)
    b0 = np.zeros(X.shape[1])
    b0[0] = np.log(y.mean() / (1 - y.mean()))
    pen = np.ones(X.shape[1]); pen[0] = 0

    def f(b):
        eta = X @ b
        ll = (y * eta - np.logaddexp(0, eta)).sum() / n
        p = 1 / (1 + np.exp(-eta))
        return -ll + 0.5 * l2 * (pen * b * b).sum(), -(X.T @ (y - p)) / n + l2 * pen * b

    return minimize(f, b0, jac=True, method="L-BFGS-B", options={"maxiter": 2000}).x


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def logloss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def auc(p, y):
    r = rankdata(p)
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def fit_all(s, rink=False):
    """Fit the main, empty-net and penalty-shot pieces. Returns a JSON-safe params dict."""
    bias = rink_bias(s) if rink else {}
    main = s[~s["empty_net"] & ~s["penalty_shot"]]
    X, names = design(main, apply_rink(main, bias))
    beta = fit_logit(X, main["is_goal"].values.astype(float))
    en = s[s["empty_net"] & ~s["penalty_shot"]]
    en_beta = [0.0, 0.0]
    if len(en) >= 200 and 0 < en["is_goal"].mean() < 1:
        Xe = np.column_stack([np.ones(len(en)), en["dist"].values / 10])
        en_beta = fit_logit(Xe, en["is_goal"].values.astype(float)).tolist()
    ps = s[s["penalty_shot"]]
    ps_rate = float(ps["is_goal"].mean()) if len(ps) >= 30 else 0.32
    return {"version": 1, "fitted_at": datetime.now().isoformat(timespec="seconds"),
            "seasons": sorted(int(x) for x in s["season"].unique()), "n_shots": int(len(main)),
            "features": names, "beta": [round(float(b), 6) for b in beta],
            "dist_knots": DIST_KNOTS, "shot_types": SHOT_TYPES,
            "empty_net_beta": [round(float(b), 6) for b in en_beta], "penalty_shot_rate": ps_rate,
            "rink_bias": bias}


def score(s, params):
    """xG for every shot in `s` using a params dict (from fit_all or xg_params.json)."""
    X, names = design(s, apply_rink(s, params.get("rink_bias") or {}))
    if names != params["features"]:
        raise ValueError("xg_params.json was fitted with a different feature set - refit with python nhl_xg.py")
    xg = sigmoid(X @ np.array(params["beta"]))
    b0, b1 = params["empty_net_beta"]
    en_xg = sigmoid(b0 + b1 * s["dist"].values / 10) if (b0 or b1) else np.full(len(s), 0.5)
    xg = np.where(s["empty_net"].values, en_xg, xg)
    xg = np.where(s["penalty_shot"].values, params["penalty_shot_rate"], xg)
    return xg


# ======================================================================================
# Holdout check
# ======================================================================================

def holdout_check(s, rink=False):
    n_reg = s[s["game_type"] == 2].groupby("season")["game_id"].nunique()
    full = sorted(n_reg[n_reg >= 1000].index) or sorted(n_reg.index)
    if len(full) < 2:
        print("  (need two seasons of shots for a holdout check - skipped)")
        return
    test_season = full[-1]
    tr, te = s[s["season"] < test_season], s[s["season"] == test_season]
    params = fit_all(tr, rink)
    te_m = te[~te["empty_net"] & ~te["penalty_shot"]]
    tr_m = tr[~tr["empty_net"] & ~tr["penalty_shot"]]
    y = te_m["is_goal"].values.astype(float)
    p_full = score(te_m, params)
    # distance-only baseline (same spline)
    Xd_tr = design(tr_m)[0][:, :1 + 1 + len(DIST_KNOTS)]
    bd = fit_logit(Xd_tr, tr_m["is_goal"].values.astype(float))
    p_dist = sigmoid(design(te_m)[0][:, :1 + 1 + len(DIST_KNOTS)] @ bd)
    p_const = np.full(len(y), tr_m["is_goal"].mean())

    print(f"\n=== Holdout: trained on seasons before {test_season}, tested on {test_season} "
          f"({len(te_m):,} unblocked non-empty-net shots, {int(y.sum()):,} goals) ===")
    print(f"  {'model':<22}{'log-loss':>10}{'AUC':>8}{'total xG / goals':>20}")
    for lab, p in (("constant rate", p_const), ("distance only", p_dist), ("full xG", p_full)):
        print(f"  {lab:<22}{logloss(p, y):>10.4f}{auc(p, y) if p.std() > 0 else 0.5:>8.3f}{p.sum() / y.sum():>20.3f}")

    dec = pd.qcut(p_full, 10, labels=False, duplicates="drop")
    cal = pd.DataFrame({"decile": dec + 1, "p": p_full, "y": y}).groupby("decile").agg(
        predicted=("p", "mean"), actual=("y", "mean"), shots=("y", "size"))
    print("\n  Calibration by predicted-xG decile (predicted vs actual goal rate):")
    print(cal.round(4).to_string().replace("\n", "\n  "))

    by = pd.DataFrame({"strength": te_m["strength"].values, "type": te_m["shot_type"].values, "p": p_full, "y": y})
    print("\n  Goals / xG by situation (1.00 = perfectly calibrated):")
    for col in ("strength", "type"):
        t = by.groupby(col).agg(shots=("y", "size"), goals=("y", "sum"), xg=("p", "sum"))
        t = t[t["shots"] >= 300]
        t["goals/xG"] = t["goals"] / t["xg"]
        print("  " + t.round(2).to_string().replace("\n", "\n  "))


# ======================================================================================
# Team + goalie tables
# ======================================================================================

def team_game_table(s):
    s = s.assign(ev=s["strength"] == "EV", reg_goal=s["is_goal"] & ~s["empty_net"],
                 xg_noen=np.where(s["empty_net"], 0.0, s["xg"]),
                 xg_reg=np.where(~s["empty_net"] & (s["period"] <= 3), s["xg"], 0.0),
                 xg_ev=np.where((s["strength"] == "EV") & ~s["empty_net"], s["xg"], 0.0),
                 xg_en=np.where(s["empty_net"], s["xg"], 0.0),
                 on_goal=s["event"].isin(["goal", "shot-on-goal"]),
                 ev_goal=(s["strength"] == "EV") & s["is_goal"] & ~s["empty_net"])
    keys = ["game_id", "season", "game_type", "game_date", "team", "opp", "is_home"]
    t = s.groupby(keys).agg(xgf=("xg_noen", "sum"), xgf_reg=("xg_reg", "sum"), xgf_ev=("xg_ev", "sum"),
                            xgf_en=("xg_en", "sum"), gf=("reg_goal", "sum"), gf_ev=("ev_goal", "sum"),
                            sf=("on_goal", "sum"), ff=("event", "size")).reset_index()
    opp = t[["game_id", "team", "xgf", "xgf_reg", "xgf_ev", "gf", "gf_ev", "sf", "ff"]].rename(columns={
        "team": "opp", "xgf": "xga", "xgf_reg": "xga_reg", "xgf_ev": "xga_ev", "gf": "ga", "gf_ev": "ga_ev",
        "sf": "sa", "ff": "fa"})
    t = t.merge(opp, on=["game_id", "opp"], how="left")
    for c in ("xga", "xga_reg", "xga_ev", "ga", "ga_ev", "sa", "fa"):
        t[c] = t[c].fillna(0)
    t["game_date"] = t["game_date"].dt.strftime("%Y-%m-%d")
    return t.sort_values(["game_date", "game_id", "is_home"]).reset_index(drop=True)


def goalie_game_table(s):
    g = s[s["goalie_id"].notna() & ~s["empty_net"] & ~s["penalty_shot"]].copy()
    g["on_goal"] = g["event"].isin(["goal", "shot-on-goal"])
    g["xg_var"] = g["xg"] * (1 - g["xg"])            # sampling variance of GSAx (used to set goalie shrinkage)
    t = g.groupby(["game_id", "season", "game_type", "game_date", "opp", "goalie_id"]).agg(
        attempts=("event", "size"), shots_on_goal=("on_goal", "sum"), goals=("is_goal", "sum"),
        xga=("xg", "sum"), xg_var=("xg_var", "sum")).reset_index().rename(columns={"opp": "team"})
    # (opp of the shooter = the goalie's team)
    t["goalie_id"] = t["goalie_id"].astype("int64")
    t["gsax"] = t["xga"] - t["goals"]
    t["game_date"] = t["game_date"].dt.strftime("%Y-%m-%d")
    return t


# ======================================================================================
# Report: is xG more predictive than goals / shots?
# ======================================================================================

def predictive_report(tg, gg):
    reg = tg[tg["game_type"] == 2].copy()
    n_games = reg.groupby("season")["game_id"].nunique()
    reg = reg[reg["season"].isin(n_games[n_games >= 1000].index)]
    if reg.empty:
        print("\n(report skipped - needs at least one complete regular season)")
        return
    reg = reg.sort_values(["season", "team", "game_date"])
    reg["n"] = reg.groupby(["season", "team"]).cumcount()
    cols = ["gf", "ga", "sf", "sa", "ff", "fa", "xgf", "xga", "xgf_ev", "xga_ev"]

    def shares(x):
        t = x[cols].sum()
        return pd.Series({"Goal share": t.gf / (t.gf + t.ga), "Shot share (on goal)": t.sf / (t.sf + t.sa),
                          "Fenwick share": t.ff / (t.ff + t.fa), "xG share": t.xgf / (t.xgf + t.xga),
                          "EV xG share": t.xgf_ev / (t.xgf_ev + t.xga_ev)})

    reg["odd"] = reg["n"] % 2
    A = reg.groupby(["season", "team", "odd"]).apply(shares, include_groups=False).unstack("odd")
    print("\n=== Team stats: how repeatable? (odd vs even games, same team-season) ===")
    for k in A.columns.levels[0]:
        print(f"  {k:<22} r = {np.corrcoef(A[(k, 0)], A[(k, 1)])[0, 1]:.2f}")

    print("\n=== Team stats after N games -> goal share over the rest of the season (r) ===")
    rows = []
    for N in (10, 20, 30, 41):
        f = reg[reg["n"] < N].groupby(["season", "team"]).apply(shares, include_groups=False)
        r = reg[reg["n"] >= N].groupby(["season", "team"]).apply(shares, include_groups=False)["Goal share"]
        row = {"after N games": N}
        for k in f.columns:
            row[k] = np.corrcoef(f[k], r)[0, 1]
        z = lambda v: (v - v.mean()) / v.std()
        X = np.c_[np.ones(len(f)), z(f["Goal share"]), z(f["xG share"])]
        b = np.linalg.lstsq(X, r.values, rcond=None)[0]
        row["Best blend"] = np.corrcoef(X @ b, r)[0, 1]
        row["xG weight in blend"] = b[2] / (b[1] + b[2]) if (b[1] + b[2]) else np.nan
        rows.append(row)
    print(pd.DataFrame(rows).set_index("after N games").round(2).to_string())

    g = gg[gg["game_type"] == 2].sort_values(["game_date", "game_id"]).copy()
    g["odd"] = g.groupby(["season", "goalie_id"]).cumcount() % 2
    h = g.groupby(["season", "goalie_id", "odd"])[["shots_on_goal", "attempts", "goals", "xga"]].sum().unstack("odd")
    h = h[(h[("shots_on_goal", 0)] >= 600) & (h[("shots_on_goal", 1)] >= 600)]
    if len(h) >= 20:
        print(f"\n=== Goalies: how repeatable? ({len(h)} goalie-seasons with 600+ shots on goal per half) ===")
        sv = [1 - h[("goals", i)] / h[("shots_on_goal", i)] for i in (0, 1)]
        gs = [(h[("xga", i)] - h[("goals", i)]) / h[("attempts", i)] for i in (0, 1)]
        print(f"  save %                       r = {np.corrcoef(*sv)[0, 1]:.2f}")
        print(f"  GSAx per unblocked attempt   r = {np.corrcoef(*gs)[0, 1]:.2f}")


# ======================================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rink-adjust", action="store_true", help="correct per-arena shot distance bias")
    ap.add_argument("--score-only", action="store_true", help="score shots with the saved model/xg_params.json")
    ap.add_argument("--no-report", action="store_true", help="skip the predictive report")
    args = ap.parse_args()

    s = load_shots()
    print(f"shots: {len(s):,} unblocked attempts in {s['game_id'].nunique():,} games, "
          f"seasons {', '.join(map(str, sorted(s['season'].unique())))}")

    if args.score_only:
        params = json.loads(PARAMS_PATH.read_text())
        print(f"scoring with {PARAMS_PATH.relative_to(ROOT)} (fitted {params['fitted_at']})")
    else:
        holdout_check(s, args.rink_adjust)
        print("\n=== Final fit on all seasons ===")
        params = fit_all(s, args.rink_adjust)
        PARAMS_PATH.parent.mkdir(exist_ok=True)
        PARAMS_PATH.write_text(json.dumps(params, indent=2))
        print(f"  saved {PARAMS_PATH.relative_to(ROOT)}")
        for n, b in zip(params["features"], params["beta"]):
            print(f"    {n:<18}{b:+.3f}")
        if params["rink_bias"]:
            rb = pd.Series(params["rink_bias"])
            latest = rb[rb.index.str.startswith(str(max(params["seasons"])))]
            print("  largest arena distance biases, latest season (ft; + = recorded farther out than reality):")
            print("    " + latest.reindex(latest.abs().sort_values(ascending=False).index).head(6)
                  .round(1).to_string().replace("\n", "\n    "))

    s["xg"] = score(s, params)
    print("\n=== Writing tables ===")
    _save(s[["game_id", "event_id", "season", "game_date", "period", "team", "opp", "goalie_id", "shooter_id",
             "event", "is_goal", "strength", "empty_net", "dist", "angle", "shot_type", "rebound", "rush", "xg"]]
          .assign(game_date=lambda d: d["game_date"].dt.strftime("%Y-%m-%d")), "shot_xg")
    tg = team_game_table(s)
    gg = goalie_game_table(s)
    _save(tg, "game_xg")
    _save(gg, "goalie_xg")

    if not args.no_report:
        predictive_report(tg, gg)
    print("\nDone.")


if __name__ == "__main__":
    main()
