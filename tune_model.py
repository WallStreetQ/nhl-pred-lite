"""
tune_model.py
--------------
Walk-forward tuning of the six model settings in nhl_model.py:
    DEFAULT_HALF_LIFE, DEFAULT_ALPHA, DEFAULT_CARRYOVER, DEFAULT_SHOT_BLEND, DEFAULT_XG_BLEND,
    DEFAULT_GOALIE_XG_BLEND

How it works
  1. Backtests the CURRENT defaults (the baseline).
  2. Sweeps each setting on its own, holding the others at their current values.
  3. Compares every run to the baseline on the SAME games, with a bootstrap standard error,
     so you can tell a real improvement from noise.
  4. Combines the per-setting winners into one run and checks that the combo still beats the baseline
     (settings can interact).
  5. Prints exactly which DEFAULT_ lines to change, or tells you to keep what you have.

Each backtest run takes ~15-20 s; the default sweep is ~26 runs (about 8 minutes).

  python tune_model.py                          # sweep all six settings, judge on moneyline Brier
  python tune_model.py --only goalie_xg_blend   # the goalie test: GSAx vs save% (0 = sv%, 1 = GSAx)
  python tune_model.py --metric o55_brier       # judge on a different market (see METRICS below)
  python tune_model.py --only alpha half_life   # sweep just some settings
  python tune_model.py --only shot_blend xg_blend   # the xG test: does xG add anything over shots?
  python tune_model.py --seasons 2              # test on the last 2 complete seasons (default 3)

xg_blend needs data/game_xg and goalie_xg_blend needs data/goalie_xg (run nhl_shots.py then nhl_xg.py
first); without them they're skipped.
shot_blend and xg_blend overlap (both measure shot-based play), so judge them together with the combo check.

Note: the backtest only scores seasons with 500+ finished games, so the current season starts
counting around mid-December. Re-tuning before then mostly re-tests the same past seasons.
"""
import argparse
import time
import warnings

import numpy as np
import pandas as pd

import nhl_model as nm
from nhl_model import load_data, walk_forward

warnings.filterwarnings("ignore")

# setting name -> (walk_forward keyword, DEFAULT_ constant, refresh.py flag, values to try)
SETTINGS = {
    "half_life": ("half_life_days", "DEFAULT_HALF_LIFE", "--half-life", [120, 150, 200, 250, 300, 400]),
    "alpha": ("alpha", "DEFAULT_ALPHA", "--alpha", [0.01, 0.02, 0.03, 0.05, 0.1]),
    "carryover": ("carryover", "DEFAULT_CARRYOVER", "--carryover", [0.5, 0.6, 0.7, 0.85, 1.0]),
    "shot_blend": ("shot_blend", "DEFAULT_SHOT_BLEND", "--shot-blend", [0.0, 0.15, 0.25, 0.35, 0.5]),
    "xg_blend": ("xg_blend", "DEFAULT_XG_BLEND", "--xg-blend", [0.0, 0.15, 0.25, 0.35, 0.5]),
    "goalie_xg_blend": ("goalie_xg_blend", "DEFAULT_GOALIE_XG_BLEND", "--goalie-xg-blend", [0.0, 0.25, 0.5, 0.75, 1.0]),
}


def _brier(p, y):
    return (np.asarray(p, float) - np.asarray(y, float)) ** 2


def _reg3_ll(raw):
    p = raw[["p_h", "p_t", "p_a"]].values[np.arange(len(raw)), raw["y_reg"].values]
    return -np.log(np.clip(p, 1e-6, 1))


# metric name -> per-game loss (lower is better) computed from walk_forward's raw rows
METRICS = {
    "win_brier": lambda r: _brier(r["p_win"], r["y_win"]),
    "reg3_logloss": _reg3_ll,
    "o55_brier": lambda r: _brier(r["p_o55"], r["y_o55"]),
    "o65_brier": lambda r: _brier(r["p_o65"], r["y_o65"]),
    "pl_brier": lambda r: _brier(r["p_cover"], r["y_cover"]),
}


def current_defaults():
    return {kw: getattr(nm, const) for kw, const, _, _ in SETTINGS.values()}


def run(D, n_seasons, **kw):
    r = walk_forward(D=D, n_seasons=n_seasons, log=lambda *a: None, **kw)
    return r["raw"].reset_index(drop=True)


def compare(raw, base_raw, metric, rng, reps=1000):
    """Mean loss difference vs baseline on the same games (negative = better) and its bootstrap SE."""
    if len(raw) != len(base_raw):
        raise RuntimeError("runs scored different games - can't compare them pairwise")
    d = METRICS[metric](raw) - METRICS[metric](base_raw)
    idx = rng.integers(0, len(d), (reps, len(d)))
    return float(d.mean()), float(d[idx].mean(axis=1).std())


def scores(raw):
    return {m: float(f(raw).mean()) for m, f in METRICS.items()}


def fmt(x):
    return f"{x:g}"


def _valid(kw):
    return kw.get("shot_blend", 0) + kw.get("xg_blend", 0) <= 1 + 1e-9


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metric", default="win_brier", choices=list(METRICS), help="what to optimise (default win_brier)")
    ap.add_argument("--only", nargs="+", choices=list(SETTINGS), help="sweep only these settings")
    ap.add_argument("--seasons", type=int, default=3, help="how many recent complete seasons to test on")
    ap.add_argument("--z", type=float, default=2.0, help="improvement must exceed z x SE to count (default 2)")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    D = load_data("data", log=lambda *a: None)
    base = current_defaults()
    names = args.only or list(SETTINGS)
    if "xg_blend" in names and D.get("xg_coverage", 0) == 0:
        print("Skipping xg_blend: no data/game_xg found (run nhl_shots.py, then nhl_xg.py).\n")
        names = [n for n in names if n != "xg_blend"]
        if not names:
            return
    elif "xg_blend" in names:
        print(f"xG data covers {D['xg_coverage']:.1%} of team-games.")
    if "goalie_xg_blend" in names and not D.get("gsax_on"):
        print("Skipping goalie_xg_blend: no usable data/goalie_xg (run nhl_shots.py, then nhl_xg.py) "
              "or goalies are off.\n")
        names = [n for n in names if n != "goalie_xg_blend"]
        if not names:
            return
    elif "goalie_xg_blend" in names:
        print(f"GSAx data covers {D['gsax_coverage']:.1%} of goalie starts.")

    t0 = time.time()
    print(f"Baseline (current defaults): {', '.join(f'{k}={fmt(v)}' for k, v in base.items())}")
    base_raw = run(D, args.seasons, **base)
    seasons = sorted(base_raw["season"].astype(str).unique())
    print(f"  {len(base_raw):,} test games in seasons {', '.join(seasons)}  ({time.time() - t0:.0f} s per run)\n")

    rows = [{"setting": "baseline", "value": "", **scores(base_raw), "diff": 0.0, "se": 0.0}]
    winners = {}
    for name in names:
        kw, const, flag, values = SETTINGS[name]
        best = None
        for v in values:
            if np.isclose(v, base[kw]) or not _valid({**base, kw: v}):
                continue
            raw = run(D, args.seasons, **{**base, kw: v})
            diff, se = compare(raw, base_raw, args.metric, rng)
            rows.append({"setting": name, "value": fmt(v), **scores(raw), "diff": diff, "se": se})
            print(f"  {name} = {fmt(v):<6} {args.metric} diff {diff:+.5f} (SE {se:.5f})", flush=True)
            if diff < -args.z * se and (best is None or diff < best[1]):
                best = (v, diff, se)
        if best:
            winners[name] = best[0]

    table = pd.DataFrame(rows).set_index(["setting", "value"])
    pd.set_option("display.width", 220)
    print(f"\nAll runs (lower is better; diff/se are {args.metric} vs baseline on the same games):")
    print(table.round(5).to_string())

    print("\n" + "=" * 70)
    if not winners:
        print(f"RECOMMENDATION: keep the current defaults. Nothing beat them on {args.metric} by more "
              f"than {args.z:g} standard errors.")
        return

    combo = {**base, **{SETTINGS[n][0]: v for n, v in winners.items()}}
    if not _valid(combo):
        print("The shot_blend and xg_blend winners add up to more than 1 - try them together with smaller values,")
        print("e.g. edit the value lists in SETTINGS and re-run with --only shot_blend xg_blend.")
        return
    if len(winners) > 1:
        print("Checking the per-setting winners together...")
        diff, se = compare(run(D, args.seasons, **combo), base_raw, args.metric, rng)
        print(f"  combined: {args.metric} diff {diff:+.5f} (SE {se:.5f})")
        if diff >= -args.z * se:
            print(f"\nRECOMMENDATION: keep the current defaults. Individually some settings helped, but "
                  f"together they don't clearly beat the baseline. Try --only with a single setting.")
            return

    print("RECOMMENDATION: in nhl_model.py change")
    for n, v in winners.items():
        _, const, _, _ = SETTINGS[n]
        print(f"    {const} = {fmt(v)}      (was {fmt(base[SETTINGS[n][0]])})")
    print("then run:   python refresh.py --skip-fetch --backtest --push")
    flags = " ".join(f"{SETTINGS[n][2]} {fmt(v)}" for n, v in winners.items())
    print(f"(or test first without editing:   python refresh.py --skip-fetch --backtest {flags})")
    edge = [n for n, v in winners.items() if v in (SETTINGS[n][3][0], SETTINGS[n][3][-1])]
    if edge:
        print(f"Note: {', '.join(edge)} won at the edge of its range - extend the values in SETTINGS and re-run.")


if __name__ == "__main__":
    main()
