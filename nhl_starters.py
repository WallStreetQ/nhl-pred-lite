"""
nhl_starters.py
---------------
Projected starting goalies for upcoming games (the 10am run), built from data/game_goalies alone.

Why: confirmed starters usually aren't public until after morning skate, so the morning refresh
needs a best guess. The evening run later overwrites these with confirmations (status "confirmed");
this script only ever writes status "projected".

How a starter is projected (per team, per upcoming game), using ONLY games before that date:
  1. Available goalies = dressed for this team in at least one of its last 3 games (within the
     last 45 days), and this is the last team they dressed for (drops traded / sent-down / injured
     goalies, and last season's departures). On live runs, if the
     NHL roster endpoint answers, they must also be on the team's current roster.
  2. Start share = each available goalie's share of the team's last 15 starts. The top goalie is
     the one with the highest share (ties -> most recent starter).
  3. Back-to-back swap: on the 2nd night of a back-to-back, if the top goalie started last night,
     project the next goalie instead.
  4. Tandem rotation: if the top goalie started the last game but holds under 60% of starts,
     project the other goalie (tandems mostly alternate).
  5. Otherwise project the top goalie.
  6. p_start = how often that same situation (rule used x how dominant the top goalie is) was right
     in the backtest. Confidence label comes from p_start: high >= 75%, medium 60-75%, low < 60%.
     So the probability shown is measured, not guessed.

Usage
  python nhl_starters.py --backtest    # replay past seasons, print hit rates, save calibration
  python nhl_starters.py               # project starters for the next date in data/upcoming.json
  python nhl_starters.py --days 2      # next two dates on the schedule
  python nhl_starters.py --no-roster   # skip the NHL roster check (offline)

Output: model/projected_starters.json (small; publish it with the other model files).
"""

import argparse
import json
import sys
from collections import Counter, defaultdict, deque
from copy import deepcopy
from datetime import date, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
UPCOMING = DATA_DIR / "upcoming.json"
OUT = ROOT / "model" / "projected_starters.json"
CALIB = ROOT / "model" / "starter_calibration.json"

START_WINDOW = 15            # team starts used for the start share
DRESS_WINDOW = 3             # a goalie must have dressed in one of the team's last N games...
DRESS_MAX_DAYS = 45          # ...played within this many days (so April rosters don't carry into October)
TANDEM_SHARE = 0.60          # below this, a top goalie who just started is expected to rotate out
HIGH, MEDIUM = 0.75, 0.60    # p_start cut-offs for the confidence label
SHARE_BANDS = ((0.75, "clear No. 1"), (0.55, "leaning"), (0.0, "split"))
# p_start by situation, used for any cell the backtest hasn't measured (rough values from 2023-26)
DEFAULT_P = {"b2b_swap": 0.85, "rotation": 0.60, "repeat": 0.62, "fresh": 0.75}
TEAM_ALIASES = {"ARI": "UTA"}
ROSTER_URL = "https://api-web.nhle.com/v1/roster/{team}/current"


# ------------------------------------------------------------------------------------------
# Data
# ------------------------------------------------------------------------------------------

def load_goalies(data_dir=DATA_DIR):
    gk = None
    for ext, reader in ((".parquet", pd.read_parquet), (".csv", pd.read_csv)):
        p = Path(data_dir) / f"game_goalies{ext}"
        if not p.exists():
            continue
        try:
            gk = reader(p)
            break
        except ImportError:          # pyarrow missing -> use the CSV twin
            continue
    if gk is None:
        sys.exit(f"No game_goalies.parquet/.csv in {Path(data_dir).resolve()} - run nhl_goalies.py first.")
    gk = gk[gk["game_type"].isin([2, 3])].copy()
    gk["game_date"] = pd.to_datetime(gk["game_date"])
    gk["starter"] = gk["starter"].astype(str).str.lower().isin(["true", "1"])
    gk["team"] = gk["team"].replace(TEAM_ALIASES)
    gk = gk.drop_duplicates(["game_id", "player_id"], keep="last")
    return gk.sort_values(["game_date", "game_id"]).reset_index(drop=True)


class State:
    """What is known before a given date: each team's recent games, and each goalie's latest team."""

    def __init__(self):
        self.recent = defaultdict(lambda: deque(maxlen=START_WINDOW))   # team -> (date, starter, dressed)
        self.player_team = {}
        self.names = {}

    def add_game(self, team, game_date, starter, dressed):
        self.recent[team].append((pd.Timestamp(game_date), int(starter), frozenset(int(p) for p in dressed)))
        for p in dressed:
            self.player_team[int(p)] = team

    def add_day(self, day):
        """Feed one date's goalie rows (all games that day) into the state."""
        self.names.update(zip(day["player_id"].astype(int), day["name"]))
        for (_, team), g in day.groupby(["game_id", "team"]):
            st = g[g["starter"]]
            starter = (st if len(st) else g).sort_values("toi_sec")["player_id"].iloc[-1]
            self.add_game(team, g["game_date"].iloc[0], starter, g["player_id"].tolist())


def build_state(gk):
    s = State()
    for _, day in gk.groupby("game_date", sort=True):
        s.add_day(day)
    return s


# ------------------------------------------------------------------------------------------
# The projection rules
# ------------------------------------------------------------------------------------------

def share_band(share):
    return next(label for cut, label in SHARE_BANDS if share >= cut)


def confidence_label(p):
    return "high" if p >= HIGH else ("medium" if p >= MEDIUM else "low")


def p_start(calib, situation, band):
    return float(calib.get(f"{situation}|{band}", DEFAULT_P[situation]))


def available(state, team, game_date, roster=None):
    """Goalies who can start for `team` on `game_date`, with their share of the team's last starts.
    Returns ({player_id: share}, hist) - shares sum to 1 over the available goalies."""
    hist = list(state.recent.get(team, ()))
    if not hist:
        return {}, hist
    game_date = pd.Timestamp(game_date)
    last_starter = hist[-1][1]
    # only recent games count, so last season's goalies don't carry into October
    window = [h for h in hist[-DRESS_WINDOW:] if (game_date - h[0]).days <= DRESS_MAX_DAYS] or hist[-1:]
    dressed = set().union(*(d for _, _, d in window))
    avail = {p for p in dressed if state.player_team.get(p) == team}
    if roster:
        avail = (avail & set(roster)) or set(roster)     # roster wins; unseen call-ups get share 0
    if not avail:
        avail = {last_starter}
    starts = Counter(s for _, s, _ in hist if s in avail)
    total = sum(starts.values())
    return {p: (starts[p] / total if total else 1 / len(avail)) for p in avail}, hist


def project(state, team, game_date, roster=None, calib=None):
    """Projected starter for `team` on `game_date`, or None if the team has no history.
    roster: optional {player_id: name} of goalies on the current NHL roster."""
    share, hist = available(state, team, game_date, roster)
    if not hist:
        return None
    game_date = pd.Timestamp(game_date)
    last_date, last_starter, _ = hist[-1]
    b2b = (game_date - last_date).days == 1
    names = dict(state.names)
    if roster:
        names.update(roster)
    avail = set(share)
    last_start = {s: i for i, (_, s, _) in enumerate(hist)}
    last_dressed = {p: i for i, (_, _, d) in enumerate(hist) for p in d}
    # ties (e.g. two backups with no starts) -> most recent starter, then most recently dressed
    ranked = sorted(avail, key=lambda p: (-share[p], -last_start.get(p, -1), -last_dressed.get(p, -1), p))

    # 3-5. pick by situation
    top = ranked[0]
    band = share_band(share[top])
    last_name = names.get(last_starter, last_starter)
    can_swap = len(ranked) > 1
    if b2b and top == last_starter and can_swap:
        situation, pick = "b2b_swap", ranked[1]
        reason = f"2nd night of a back-to-back; {last_name} started last night"
    elif top == last_starter and share[top] < TANDEM_SHARE and can_swap:
        situation, pick = "rotation", ranked[1]
        reason = f"tandem ({share[top]:.0%}/{share[ranked[1]]:.0%}); {last_name} started last game"
    elif top == last_starter:
        situation, pick = "repeat", top
        reason = f"No. 1 ({share[top]:.0%} of last {len(hist)} starts), also started last game"
    else:
        situation, pick = "fresh", top
        reason = f"No. 1 ({share[top]:.0%} of last {len(hist)} starts), rested last game"
    p = p_start(calib or {}, situation, band)
    return {"team": team, "player_id": int(pick), "name": names.get(pick, str(pick)),
            "confidence": confidence_label(p), "p_start": round(p, 3), "situation": situation, "band": band,
            "start_share": round(share[pick], 3), "b2b": b2b, "reason": reason,
            "pool": {int(k): round(v, 4) for k, v in share.items()},
            "alternates": [{"player_id": int(p), "name": names.get(p, str(p)), "start_share": round(share[p], 3)}
                           for p in ranked if p != pick][:2]}


def goalie_spec(pr):
    """{player_id: probability} for a projection: the pick at p_start, the rest spread over the other
    AVAILABLE goalies by start share (all on the pick if he's the only one)."""
    if not pr:
        return None
    pick, p = int(pr["player_id"]), float(pr["p_start"])
    others = {int(k): v for k, v in pr["pool"].items() if int(k) != pick}
    tot = sum(others.values())
    if tot <= 0:
        others = {int(a["player_id"]): 1.0 for a in pr.get("alternates", [])}
        tot = sum(others.values())
    if tot <= 0:
        return {pick: 1.0}
    return {pick: p, **{k: (1 - p) * v / tot for k, v in others.items()}}


# ------------------------------------------------------------------------------------------
# Backtest: replay history day by day, projecting each game before learning its result
# ------------------------------------------------------------------------------------------

def backtest(gk, n_seasons=3):
    seasons = sorted(gk["season"].unique())
    complete = [s for s in seasons if (gk["season"] == s).sum() > 2000]
    test = set(complete[-n_seasons:])
    state, rows = State(), []
    for d, day in gk.groupby("game_date", sort=True):
        if day["season"].iloc[0] in test:
            for (gid, team), g in day.groupby(["game_id", "team"]):
                actual = g.loc[g["starter"], "player_id"]
                pr = project(state, team, d)
                if pr is None or actual.empty:
                    continue
                hist = state.recent[team]
                rows.append({"season": int(g["season"].iloc[0]), "date": d, "game_type": int(g["game_type"].iloc[0]),
                             "team": team, "situation": pr["situation"], "band": pr["band"], "b2b": pr["b2b"],
                             "hit": pr["player_id"] == int(actual.iloc[0]),
                             "naive_hit": hist[-1][1] == int(actual.iloc[0])})
        state.add_day(day)
    r = pd.DataFrame(rows)
    r["early"] = (r["date"] - r.groupby("season")["date"].transform("min")).dt.days < 14
    r["cell"] = r["situation"] + "|" + r["band"]
    return r


def measure(rs, min_games=40):
    """Hit rate per situation|band cell (cells with too few games fall back to DEFAULT_P)."""
    c = rs.groupby("cell")["hit"].agg(["mean", "size"])
    return {k: round(float(v), 3) for k, v in c.loc[c["size"] >= min_games, "mean"].items()}


def report(r):
    rs = r[r["game_type"] == 2].copy()
    seasons = sorted(rs["season"].unique())
    print(f"\nRegular-season team-games scored: {len(rs):,}  (seasons {', '.join(map(str, seasons))})")
    print(f"  hit rate              {rs['hit'].mean():.1%}")
    print(f"  naive baseline        {rs['naive_hit'].mean():.1%}   (last game's starter goes again)")

    print("\nBy rule used:")
    for s, lab in (("b2b_swap", "back-to-back swap"), ("rotation", "tandem rotation"),
                   ("fresh", "No. 1, rested last game"), ("repeat", "No. 1, started last game")):
        m = rs["situation"] == s
        if m.any():
            print(f"  {lab:<26} {rs.loc[m, 'hit'].mean():6.1%}   ({m.mean():5.1%} of games)")
    for lab, m in (("first 2 weeks of season", rs["early"]), ("rest of season", ~rs["early"])):
        print(f"  {lab:<26} {rs.loc[m, 'hit'].mean():6.1%}   ({m.sum():,} games)")

    # honest check: measure p_start on earlier seasons, score the confidence labels on the last one
    if len(seasons) > 1:
        cal = measure(rs[rs["season"] < seasons[-1]])
        hold = rs[rs["season"] == seasons[-1]].copy()
        hold["p"] = [p_start(cal, s, b) for s, b in zip(hold["situation"], hold["band"])]
        hold["conf"] = hold["p"].map(confidence_label)
        print(f"\nConfidence check on {seasons[-1]} (p_start measured on earlier seasons only):")
        for c in ("high", "medium", "low"):
            s = hold[hold["conf"] == c]
            if len(s):
                print(f"  {c:<7} predicted {s['p'].mean():5.1%}  actual {s['hit'].mean():5.1%}  "
                      f"({len(s) / len(hold):5.1%} of games)")

    calib = measure(rs)
    CALIB.parent.mkdir(exist_ok=True)
    CALIB.write_text(json.dumps({"p_start": calib, "n_games": int(len(rs)), "seasons": [str(s) for s in seasons],
                                 "measured_at": datetime.now().isoformat(timespec="seconds")}, indent=2))
    print(f"\nSaved p_start for {len(calib)} situations -> {CALIB.relative_to(ROOT)}")


# ------------------------------------------------------------------------------------------
# Live: project the upcoming slate
# ------------------------------------------------------------------------------------------

def fetch_roster(team):
    """{player_id: 'F. Last'} for goalies on the current roster, or None if the API can't be reached."""
    try:
        import requests
        resp = requests.get(ROSTER_URL.format(team=team), timeout=10,
                            headers={"User-Agent": "Mozilla/5.0 (compatible; nhl-data-script/1.0)"})
        if resp.status_code != 200:
            return None
        out = {}
        for gk in resp.json().get("goalies", []):
            first = (gk.get("firstName") or {}).get("default", "")
            last = (gk.get("lastName") or {}).get("default", "")
            out[int(gk["id"])] = f"{first[:1]}. {last}".strip()
        return out or None
    except Exception:
        return None


def load_calibration():
    try:
        return json.loads(CALIB.read_text())["p_start"], True
    except (FileNotFoundError, KeyError, ValueError):
        return {}, False


def project_slate(gk, slate, use_roster=True, log=print):
    """Projected starters for every game in `slate` (list of {game_id, date, home, away, ...}).
    Returns {str(game_id): {team: projection}}. Later dates see earlier dates' projected starters, so
    back-to-backs inside the slate stay consistent."""
    calib, measured = load_calibration()
    if not measured:
        log("  [note] no starter calibration yet - run `python nhl_starters.py --backtest` once")
    state = build_state(gk)
    rosters, roster_ok, out = {}, 0, {}
    for g in sorted(slate, key=lambda g: (g["date"], g.get("start_utc") or "")):
        picks = {}
        for side in ("home", "away"):
            team = TEAM_ALIASES.get(g[side], g[side])
            if use_roster and team not in rosters:
                rosters[team] = fetch_roster(team)
                roster_ok += rosters[team] is not None
            pr = project(state, team, g["date"], rosters.get(team), calib)
            if pr is not None:
                picks[team] = pr
        out[str(g["game_id"])] = picks
        for team, pr in picks.items():
            dressed = {pr["player_id"]} | {a["player_id"] for a in pr["alternates"][:1]}
            state.add_game(team, g["date"], pr["player_id"], dressed)
    if use_roster and rosters:
        log(f"  starters: roster check reached {roster_ok}/{len(rosters)} teams")
    return out


def live(gk, days=1, use_roster=True):
    if not UPCOMING.exists():
        sys.exit(f"No {UPCOMING.relative_to(ROOT)} - run nhl_schedule.py (or refresh.py) first.")
    through = gk["game_date"].max()
    upcoming = [g for g in json.loads(UPCOMING.read_text()) if pd.Timestamp(g["date"]) > through]
    dates = sorted({g["date"] for g in upcoming})[:days]
    slate = [g for g in upcoming if g["date"] in dates]
    proj = project_slate(gk, slate, use_roster)
    out = [{"game_id": gid, "date": next(g["date"] for g in slate if str(g["game_id"]) == gid), **pr,
            "status": "projected"} for gid, picks in proj.items() for pr in picks.values()]
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({"generated_at": datetime.now().isoformat(timespec="seconds"),
                               "data_through": str(through.date()), "dates": dates,
                               "calibrated": load_calibration()[1], "starters": out}, indent=2))
    print(f"\nProjected starters for {', '.join(dates) or 'no upcoming dates'}:")
    for r in out:
        print(f"  {r['date']}  {r['team']:<4} {r['name']:<20} {r['confidence']:<7} "
              f"p={r['p_start']:.2f}  {r['reason']}")
    print(f"\nSaved {len(out)} rows -> {OUT.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backtest", action="store_true", help="replay past seasons and save calibration")
    ap.add_argument("--seasons", type=int, default=3, help="seasons to score in the backtest (default 3)")
    ap.add_argument("--days", type=int, default=1, help="schedule dates to project (default 1)")
    ap.add_argument("--no-roster", action="store_true", help="skip the NHL roster availability check")
    ap.add_argument("--data-dir", default=str(DATA_DIR), help="folder with game_goalies.parquet/.csv")
    args = ap.parse_args()

    gk = load_goalies(args.data_dir)
    if args.backtest:
        report(backtest(gk, args.seasons))
    else:
        live(gk, args.days, use_roster=not args.no_roster)


if __name__ == "__main__":
    main()
