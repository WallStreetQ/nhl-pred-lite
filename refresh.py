"""
refresh.py
-----------
One command to update everything as the season progresses, and (optionally) publish to
GitHub so the Streamlit app redeploys with the new model.

  python refresh.py                  # fetch new games/periods/goalies + upcoming schedule, retrain,
                                     # log pre-game predictions, write model/model_artifact.json
  python refresh.py --push           # ...then git commit + push  (Streamlit Cloud redeploys automatically)
  python refresh.py --skip-fetch     # just retrain from the data you already have
  python refresh.py --backtest       # also run the walk-forward backtest (~20 s) and store its scores
  python refresh.py --tune-k         # walk-forward check of the confidence adjustment (1.0-1.4), then exit
  python refresh.py --days-back 3    # sweep more days for late/postponed games (default 3)

Expected goals: once you've run nhl_shots.py and nhl_xg.py once (which writes model/xg_params.json),
each refresh also pulls shot data for new games and re-scores xG with the saved model (no refit).
Refit the xG model itself occasionally (e.g. monthly) with: python nhl_xg.py --rink-adjust

Sportsbook odds: if an Odds API key is set (see nhl_odds.py), each refresh pulls Hard Rock Bet FL + Pinnacle
prices (~3 credits), puts them in the artifact for the app's Edges board, and updates model/edge_log.csv:
every side of every priced market, with its price and model edge when first logged ("open") and the latest
pre-game price ("close"), then the result and closing-line value once the game is final.
Run once in the morning and once shortly before the first puck drop to capture opening and closing prices:
  python refresh.py --skip-fetch --push   (late run: skips the slow data fetch, still pulls fresh odds)
  python refresh.py --no-odds             # don't call the odds API this run

Flags (edge log): a side is flagged by nhl_model.flag_rule - model at least 3 points above Hard Rock's no-vig
probability AND Pinnacle leaning the same way (or, with no Pinnacle price, the side isn't an underdog).
Rows record which rule flagged them (flag_rule: v1 = old edge-only rule, v2 = current), so the Track Record
never re-grades old picks with a rule chosen after seeing their results.

Closing-line value: Pinnacle's closing no-vig probability minus its opening no-vig probability for the
side (Hard Rock's when Pinnacle has no price at that line). Positive = the market moved toward the pick.
The book's margin is NOT in it (the old version compared against the price's break-even, which put the
~2.5-point margin into every number). price_clv keeps the old figure: beating it means beating the margin too.

What gets published (--push): the model files (model/model_artifact.json, prediction_log.csv, edge_log.csv,
xg_params.json) AND the data the next refresh needs (data/*.parquet, data/upcoming.json), so the GitHub
Action (.github/workflows/daily-refresh.yml) can run the refresh itself. Two shot-level files are too big
to commit twice a day (see LOCAL_ONLY); the Action keeps those in its cache. CSV twins stay local
(.gitignore). If the remote has newer commits, the push rebases onto them and retries by itself.
Run it from the project folder.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
ARTIFACT = ROOT / "model" / "model_artifact.json"
PRED_LOG = ROOT / "model" / "prediction_log.csv"
XG_PARAMS = ROOT / "model" / "xg_params.json"
EDGE_LOG = ROOT / "model" / "edge_log.csv"
UPCOMING = ROOT / "data" / "upcoming.json"
LOCAL_ONLY = {"shots.parquet", "shot_xg.parquet"}   # too big to commit twice a day; never pushed
VOLATILE_META = {"fitted_at", "as_of"}


def _same_model(old, new):
    """True if two artifacts are identical apart from timestamps."""
    strip = lambda a: {**a, "meta": {k: v for k, v in a["meta"].items() if k not in VOLATILE_META}}
    return strip(old) == strip(new)


def run(cmd, label):
    print(f"\n=== {label} ===")
    print("$", " ".join(cmd))
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        sys.exit(f"\n[stopped] '{label}' failed (exit code {result.returncode}). Nothing was published. "
                 f"Fix the error above (or re-run with --skip-fetch to retrain on existing data).")


def git(*args, check=True, capture=False):
    return subprocess.run(["git", *args], cwd=ROOT, check=check, text=True, capture_output=capture)


def published_files():
    """Model files + the data the next refresh needs (minus the big shot-level files)."""
    files = [ARTIFACT, PRED_LOG, EDGE_LOG, XG_PARAMS, UPCOMING]
    files += sorted(p for p in (ROOT / "data").glob("*.parquet") if p.name not in LOCAL_ONLY)
    return [p for p in files if p.exists()]


def publish(message):
    print("\n=== Publish to GitHub ===")
    if git("rev-parse", "--is-inside-work-tree", check=False, capture=True).returncode != 0:
        sys.exit("[stopped] This folder isn't a git repository yet. See DEPLOY.md, 'First-time setup'.")
    for p in published_files():
        git("add", str(p.relative_to(ROOT)))
    if git("diff", "--cached", "--quiet", check=False).returncode == 0:
        print("Model and data files are unchanged - nothing to commit.")
        return
    git("commit", "-m", message)
    for attempt in range(1, 4):
        if git("push", check=False).returncode == 0:
            print("Pushed. Streamlit Cloud will redeploy in about a minute.")
            return
        print(f"  push rejected (try {attempt}/3) - rebasing onto the newer remote commits and retrying")
        if git("pull", "--rebase", check=False).returncode != 0:
            git("rebase", "--abort", check=False)
            sys.exit("\n[push failed] Rebasing hit a conflict. The commit is saved locally; run\n"
                     "  git pull --rebase\nfix any conflicts, then\n  git push")
    sys.exit("\n[push failed] after 3 tries. The commit is saved locally; run  git pull --rebase  then  git push")


def _started(g, now):
    """True once a slate game's scheduled start has passed (unknown start time -> not started)."""
    s = g.get("start_utc")
    if not s:
        return False
    t = pd.to_datetime(s, utc=True, errors="coerce")
    return pd.notna(t) and t <= now


# ------------------------------------------------------------------------------------------
# Prediction log: pre-game predictions for upcoming games, settled once results arrive
# ------------------------------------------------------------------------------------------

def update_prediction_log(art, data_dir):
    from nhl_model import Predictor, load_data
    pred = Predictor(art)
    log = pd.read_csv(PRED_LOG) if PRED_LOG.exists() else pd.DataFrame()
    now = pd.Timestamp.now(tz="UTC")
    primary = pred.odds.get("primary", "hardrockbet_fl")

    # 1) settle finished games
    if not log.empty:
        fin = load_data(data_dir, log=lambda *a: None)["fin"].set_index("game_id")
        for col in ("home_score", "away_score", "home_reg", "away_reg"):
            if col not in log:
                log[col] = np.nan
            log[col] = log[col].fillna(log["game_id"].map(fin[col]))

    # 2) log / overwrite predictions for games that haven't STARTED. A game that has started keeps the
    #    last prediction logged before puck drop (or none, if no run happened in time), so every row in
    #    the log is a genuine pre-game prediction.
    new, skipped = [], 0
    for g in pred.slate:
        if _started(g, now):
            skipped += 1
            continue
        hb, ab = g.get("home_b2b", False), g.get("away_b2b", False)
        book = pred.game_odds(g["game_id"]).get(primary, {}) or {}
        total_line = (book.get("totals") or {}).get("point")
        home_spread = (book.get("spreads") or {}).get("point")
        try:
            hs = pred.goalie_spec(g["home"], None, b2b=hb)
            as_ = pred.goalie_spec(g["away"], None, b2b=ab)
            r = pred.predict_game(g["home"], g["away"], hs, as_, home_b2b=hb, away_b2b=ab)
        except ValueError as e:
            print(f"  [skip] {g}: {e}")
            continue
        row = {"game_id": g["game_id"], "date": g["date"], "home": g["home"], "away": g["away"],
               "home_b2b": hb, "away_b2b": ab,
               "p_home_win": r["p_home_win"], "p_home_reg": r["p_home_reg"], "p_tie_reg": r["p_tie_reg"],
               "p_away_reg": r["p_away_reg"], "exp_home": r["exp_home"], "exp_away": r["exp_away"],
               "logged_at": datetime.now().isoformat(timespec="seconds"),
               "model_through": art["meta"]["trained_through"]}
        # totals at the two common lines
        d = pred.predict_game(g["home"], g["away"], hs, as_, total_line=5.5, home_b2b=hb, away_b2b=ab)
        row["p_over_5_5"] = d["total"]["over"]
        row["p_over_6_5"] = r["total"]["over"]
        row["p_home_m1_5"] = r["spread"]["cover"]
        # the book's own lines, so the Track Record can grade totals and puck lines vs the market
        row["exp_total"] = r["exp_total"]
        row["total_line"] = np.nan if total_line is None else float(total_line)
        row["p_over_line"] = np.nan
        if total_line is not None:
            row["p_over_line"] = pred.predict_game(g["home"], g["away"], hs, as_, total_line=float(total_line),
                                                   home_b2b=hb, away_b2b=ab)["total"]["over"]
        row["home_spread"] = np.nan if home_spread is None else float(home_spread)
        row["p_home_cover_line"] = np.nan
        if home_spread is not None:
            row["p_home_cover_line"] = pred.predict_game(g["home"], g["away"], hs, as_,
                                                         home_spread=float(home_spread), home_b2b=hb,
                                                         away_b2b=ab)["spread"]["cover"]
        row["confidence_k"] = art.get("confidence_k", 1.0)
        new.append(row)
    if skipped:
        print(f"  prediction log: {skipped} slate games had already started - kept their pre-game rows")
    if new:
        new = pd.DataFrame(new)
        if not log.empty:
            settled = log["home_score"].notna()
            keep = log[settled | ~log["game_id"].isin(new["game_id"])]
            new = new[~new["game_id"].isin(log.loc[settled, "game_id"])]
            log = pd.concat([keep, new], ignore_index=True)
        else:
            log = new
    if log.empty:
        print("  prediction log: no upcoming games to log")
        return
    log = log.sort_values(["date", "game_id"]).reset_index(drop=True)
    PRED_LOG.parent.mkdir(exist_ok=True)
    log.to_csv(PRED_LOG, index=False)
    n_set = int(log["home_score"].notna().sum()) if "home_score" in log else 0
    print(f"  prediction log: {len(log)} games ({n_set} settled) -> {PRED_LOG.relative_to(ROOT)}")


# ------------------------------------------------------------------------------------------
# Edge log: model vs Hard Rock prices, open -> close -> result
# ------------------------------------------------------------------------------------------

EDGE_FIELDS = ["label", "line", "price", "model_p", "fair", "edge", "ev", "ref_fair", "logged_at"]


def _grade(market, side, line, home_score, away_score):
    """win / loss / push for one side, on the final score (OT/SO included, as books grade it).
    `line` is the home team's puck line for both puck-line sides, and the total for both total sides."""
    margin, total = home_score - away_score, home_score + away_score
    if market == "Moneyline":
        v = margin if side == "home" else -margin
    elif market == "Puck line":
        v = margin + line if side == "home" else -margin - line
    else:
        v = total - line if side == "over" else line - total
    return "win" if v > 1e-9 else ("push" if abs(v) <= 1e-9 else "loss")


def add_clv(log):
    """Closing-line value per side, with the book's margin taken out.
        clv       = closing no-vig prob - opening no-vig prob (Pinnacle when it priced the side at the same
                    line at open AND close, else Hard Rock). Positive = the market moved toward the pick.
        price_clv = closing no-vig prob - break-even prob of the OPEN price (the old figure; includes the
                    ~2.5-point margin, so it runs negative even when the line doesn't move).
    Only when the line is the same at open and close."""
    from nhl_model import american_to_decimal
    for c in ("close_ref_fair", "open_ref_fair", "close_fair", "open_fair", "open_price", "open_line", "close_line"):
        log[c] = pd.to_numeric(log[c], errors="coerce")
    same_line = (log["open_line"].isna() & log["close_line"].isna()) | ((log["open_line"] - log["close_line"]).abs() < 1e-9)
    use_ref = log["open_ref_fair"].notna() & log["close_ref_fair"].notna()
    move = np.where(use_ref, log["close_ref_fair"] - log["open_ref_fair"], log["close_fair"] - log["open_fair"])
    log["clv"] = np.where(same_line, move, np.nan)
    log["clv_vs"] = np.where(same_line, np.where(use_ref, "pinnacle", "hardrock"), "")
    breakeven = 1 / log["open_price"].map(american_to_decimal)
    log["price_clv"] = np.where(same_line, log["close_ref_fair"].fillna(log["close_fair"]) - breakeven, np.nan)
    return log


def update_edge_log(art, data_dir):
    from nhl_model import EDGE_FLAG, FLAG_RULE, Predictor, american_to_decimal, load_data
    pred = Predictor(art)
    log = pd.read_csv(EDGE_LOG) if EDGE_LOG.exists() else pd.DataFrame()
    key = ["game_id", "market", "side"]
    now = pd.Timestamp.now(tz="UTC")
    stamp = datetime.now().isoformat(timespec="seconds")

    # 1) price every upcoming game that has Hard Rock odds and hasn't started
    cur = []
    for g in pred.slate:
        go = pred.game_odds(g["game_id"])
        if not go or _started(g, now):
            continue
        hb, ab = g.get("home_b2b", False), g.get("away_b2b", False)
        try:
            rows = pred.edge_board(g["home"], g["away"], go, pred.goalie_spec(g["home"], None, b2b=hb),
                                   pred.goalie_spec(g["away"], None, b2b=ab), home_b2b=hb, away_b2b=ab)
        except ValueError as e:
            print(f"  [skip] {g['away']} @ {g['home']}: {e}")
            continue
        for r in rows:
            cur.append({"game_id": int(g["game_id"]), "date": g["date"], "home": g["home"], "away": g["away"],
                        "market": r["market"], "side": r["side"], "label": r["label"], "line": r["line"],
                        "price": r["price"], "model_p": r["model_cond"], "fair": r["fair"], "edge": r["edge"],
                        "ev": r["ev"], "ref_fair": r["ref_fair"], "logged_at": stamp, "_flag": r["flag"]})
    cur = pd.DataFrame(cur)

    # 2) new sides get their "open" snapshot (and their flag, under the current rule); every priced side
    #    gets its "close" overwritten
    if not log.empty and "flag_rule" not in log.columns:
        log["flag_rule"] = "v1"                          # rows flagged before the Pinnacle rule existed
    if not cur.empty:
        C = cur.set_index(key)
        L = log.set_index(key) if not log.empty else pd.DataFrame()
        new = C.index if L.empty else C.index.difference(L.index)
        if len(new):
            add = pd.concat([C.loc[new, ["date", "home", "away"]], C.loc[new, EDGE_FIELDS].add_prefix("open_")], axis=1)
            add["flag"] = C.loc[new, "_flag"].astype(bool)
            add["flag_rule"] = FLAG_RULE
            L = add if L.empty else pd.concat([L, add])
        close = C[EDGE_FIELDS].add_prefix("close_")
        for c in close.columns:
            if c not in L.columns:
                L[c] = np.nan
            L[c] = L[c].astype(object)
            L.loc[close.index, c] = close[c].astype(object)
        log = L.reset_index()
    if log.empty:
        print("  edge log: no Hard Rock prices for upcoming games yet")
        return

    # 3) closing-line value, margin removed
    log = add_clv(log)

    # 4) settle finished games at the OPEN price (1 unit staked)
    for c in ("result", "profit", "final"):
        if c not in log:
            log[c] = np.nan
        log[c] = log[c].astype(object)
    fin = load_data(data_dir, log=lambda *a: None)["fin"].set_index("game_id")
    todo = log["result"].isna() & log["game_id"].isin(fin.index)
    for i in log.index[todo]:
        f = fin.loc[log.at[i, "game_id"]]
        res = _grade(log.at[i, "market"], log.at[i, "side"], log.at[i, "open_line"], f["home_score"], f["away_score"])
        dec = american_to_decimal(log.at[i, "open_price"])
        log.at[i, "result"] = res
        log.at[i, "profit"] = round(dec - 1, 4) if res == "win" else (0.0 if res == "push" else -1.0)
        log.at[i, "final"] = f"{int(f['away_score'])}-{int(f['home_score'])}"

    order = ["game_id", "date", "away", "home", "market", "side", "flag", "flag_rule"] \
        + [f"open_{c}" for c in EDGE_FIELDS] + [f"close_{c}" for c in EDGE_FIELDS] \
        + ["clv", "clv_vs", "price_clv", "result", "profit", "final"]
    log = log[[c for c in order if c in log.columns]].sort_values(["date", "game_id", "market", "side"])
    EDGE_LOG.parent.mkdir(exist_ok=True)
    log.to_csv(EDGE_LOG, index=False)
    flag = log[log["flag"].astype(str).str.lower() == "true"]
    for rule, fr in flag.groupby("flag_rule"):
        settled = fr[fr["result"].notna()]
        rec = settled["result"].value_counts()
        print(f"  edge log [{rule}]: {len(fr)} flagged | settled {rec.get('win', 0)}-{rec.get('loss', 0)}-"
              f"{rec.get('push', 0)}, {pd.to_numeric(settled['profit']).sum():+.2f} units, "
              f"avg CLV {100 * pd.to_numeric(fr['clv']).mean():+.2f} pts")
    print(f"  edge log: {len(log)} sides (flag = edge >= {100 * EDGE_FLAG:.0f} pts + Pinnacle agrees) "
          f"-> {EDGE_LOG.relative_to(ROOT)}")


def backtest_summary(data_dir, **kw):
    from nhl_model import walk_forward
    r = walk_forward(data_dir, log=lambda *a: None, **kw)
    raw = r.pop("raw")
    bins = [0, .35, .40, .45, .50, .55, .60, .65, 1]
    raw["bucket"] = pd.cut(raw["p_win"], bins)
    cal = raw.groupby("bucket", observed=True).agg(predicted=("p_win", "mean"), actual=("y_win", "mean"),
                                                    n=("y_win", "size")).reset_index()
    cal["bucket"] = cal["bucket"].astype(str)
    r["calibration"] = cal.to_dict("records")
    r["settings"] = kw
    return r


def tune_confidence_k(data_dir, settings, ks=(1.0, 1.1, 1.2, 1.3, 1.4)):
    """Walk-forward each confidence_k and print moneyline / 3-way / puck-line scores, per season too."""
    from nhl_model import load_data, walk_forward
    D = load_data(data_dir, log=lambda *a: None)
    base = {k: v for k, v in settings.items() if k != "confidence_k"}
    print(f"{'k':>5} {'ML logloss':>11} {'ML Brier':>9} {'3-way LL':>9} {'PL Brier':>9}   ML logloss by season")
    for k in ks:
        r = walk_forward(data_dir, n_seasons=4, confidence_k=k, D=D, log=lambda *a: None, **base)
        seas = "  ".join(f"{s[2:4]}-{s[6:]} {v['win_logloss']:.4f}" for s, v in r["by_season"].items())
        print(f"{k:5.2f} {r['win_logloss']:11.5f} {r['win_brier']:9.5f} {r['reg3_logloss']:9.5f} "
              f"{r['pl_brier']:9.5f}   {seas}")
    print("Lower is better. Set the winner with --confidence-k, or as DEFAULT_CONFIDENCE_K in nhl_model.py.")


def main():
    from nhl_model import (DEFAULT_ALPHA, DEFAULT_CARRYOVER, DEFAULT_HALF_LIFE, DEFAULT_SHOT_BLEND,
                           DEFAULT_XG_BLEND, DEFAULT_GOALIE_XG_BLEND, DEFAULT_CONFIDENCE_K)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-fetch", action="store_true", help="don't call the NHL API; retrain from existing data/")
    ap.add_argument("--days-back", type=int, default=3, help="days to sweep for newly finished games (default 3)")
    ap.add_argument("--push", action="store_true", help="git commit + push the model and data files when done")
    ap.add_argument("--backtest", action="store_true", help="run the walk-forward backtest and store its scores")
    ap.add_argument("--tune-k", action="store_true", help="walk-forward the confidence adjustment and exit")
    ap.add_argument("--half-life", type=float, default=DEFAULT_HALF_LIFE, help="recency half-life (days)")
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="ridge strength")
    ap.add_argument("--carryover", type=float, default=DEFAULT_CARRYOVER, help="weight per past season (1 = off)")
    ap.add_argument("--shot-blend", type=float, default=DEFAULT_SHOT_BLEND, help="share of ratings from shots model")
    ap.add_argument("--xg-blend", type=float, default=DEFAULT_XG_BLEND, help="share of ratings from xG model")
    ap.add_argument("--goalie-xg-blend", type=float, default=DEFAULT_GOALIE_XG_BLEND,
                    help="share of goalie ratings from GSAx instead of save%% (0-1)")
    ap.add_argument("--confidence-k", type=float, default=DEFAULT_CONFIDENCE_K,
                    help="confidence adjustment on home/away goal rates (1 = off)")
    ap.add_argument("--no-odds", action="store_true", help="don't pull sportsbook odds this run")
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT))
    settings = dict(half_life_days=args.half_life, alpha=args.alpha, carryover=args.carryover,
                    shot_blend=args.shot_blend, xg_blend=args.xg_blend, goalie_xg_blend=args.goalie_xg_blend,
                    confidence_k=args.confidence_k)
    if args.tune_k:
        tune_confidence_k(ROOT / "data", settings)
        return

    py = sys.executable
    if not args.skip_fetch:
        steps = [([py, "nhl_update.py", "--days-back", str(args.days_back)], "New games + standings"),
                 ([py, "nhl_periods.py"], "Period data for new games"),
                 ([py, "nhl_goalies.py"], "Goalie data for new games")]
        if XG_PARAMS.exists():
            steps += [([py, "nhl_shots.py"], "Shot data for new games"),
                      ([py, "nhl_xg.py", "--score-only", "--no-report"], "Expected goals (saved xG model)")]
        steps.append(([py, "nhl_schedule.py"], "Upcoming schedule"))
        for i, (cmd, label) in enumerate(steps, 1):
            run(cmd, f"{i}/{len(steps)} {label}")

    if not args.no_odds:
        print("\n=== Sportsbook odds (Hard Rock Bet FL + Pinnacle) ===")
        try:
            import nhl_odds
            nhl_odds.pull_and_save(log=print)
        except Exception as e:        # odds are optional: never let them stop the model refresh
            print(f"  [warn] odds pull failed: {e}\n  continuing with the last saved odds")

    upcoming = json.loads(UPCOMING.read_text()) if UPCOMING.exists() else []
    print("\n=== Retrain model ===")
    from nhl_model import load_artifact, save_artifact, train_artifact
    art = train_artifact(ROOT / "data", upcoming=upcoming, log=lambda *a: print(" ", *a), **settings)
    try:
        import nhl_odds
        art["odds"] = nhl_odds.latest_block([g["game_id"] for g in art["slate"]])
    except Exception as e:
        print(f"  [warn] couldn't attach odds to the model file: {e}")
        art["odds"] = None
    if art.get("odds"):
        print(f"  odds attached for {len(art['odds']['games'])} upcoming games (pulled {art['odds']['pulled_at']})")
    old = load_artifact(ARTIFACT) if ARTIFACT.exists() else None
    if old and old.get("version", 0) < art["version"]:
        old = None

    if args.backtest:
        print("\n=== Walk-forward backtest ===")
        bt = backtest_summary(ROOT / "data", **settings)
        print(f"  {bt['n_games']:,} games ({', '.join(bt['seasons'])}) | moneyline Brier {bt['win_brier']:.4f} "
              f"(constant {bt['const_brier']:.4f}) | 3-way log-loss {bt['reg3_logloss']:.4f}")
        art["meta"]["backtest"] = json.loads(json.dumps(bt, default=float))
    elif old and old["meta"].get("backtest", {}).get("settings") == settings:
        art["meta"]["backtest"] = old["meta"]["backtest"]

    m = art["meta"]
    if old and _same_model(old, art):
        print(f"\nModel unchanged - keeping existing {ARTIFACT.relative_to(ROOT)}")
    else:
        save_artifact(art, ARTIFACT)
        print(f"\nSaved {ARTIFACT.relative_to(ROOT)}  |  trained through {m['trained_through']}  |  "
              f"goalies {'ON' if m['goalies_on'] else 'OFF'}  |  xG blend {m.get('xg_blend_used', 0):g}  |  "
              f"goalie GSAx blend {m.get('goalie_xg_blend_used', 0):g}  |  confidence k {m.get('confidence_k', 1):g}  |  "
              f"{len(art['slate'])} upcoming games")

    print("\n=== Prediction log ===")
    update_prediction_log(art, ROOT / "data")

    print("\n=== Edge log (model vs Hard Rock) ===")
    try:
        update_edge_log(art, ROOT / "data")
    except Exception as e:
        print(f"  [warn] edge log not updated: {e}")

    if args.push:
        publish(f"Update model through {m['trained_through']}")
    else:
        print("\nTo publish: python refresh.py --skip-fetch --push   (or git add/commit/push the model/ and data/ files)")


if __name__ == "__main__":
    main()
