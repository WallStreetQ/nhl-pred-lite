"""
nhl_shots.py
-------------
Backfills every unblocked shot attempt (goals, shots on goal, missed shots) for every game
already in data/games.parquet, from the play-by-play endpoint:

    https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play

Each row is one shot attempt with what an expected-goals (xG) model needs:
  location   x, y (raw rink coords), x_n / y_n (normalised so the target net is at x = +89),
             dist (feet to the net), angle (degrees off the centre line; > 90 = from behind the net)
  type       shot_type (wrist, snap, slap, backhand, tip-in, deflected, wrap-around, ...)
  context    strength (EV / PP / SH from the shooter's view), skaters_for / skaters_against,
             empty_net (defending goalie pulled), rebound (same team shot within 3 s),
             rush (previous event was outside the attacking zone within 4 s),
             time_since_prev, prev_event
  who        team, opp, is_home, shooter_id, goalie_id

Blocked shots are skipped: their recorded location is where the block happened, not the shot.
Shootout attempts are skipped. Penalty shots are kept but flagged (penalty_shot=True).

Output: data/shots.parquet / .csv  (~80-90 rows per game)

Run AFTER nhl_backfill.py (needs game IDs). Resume-friendly; checkpoints every 250 games.
A full 6-season backfill is ~8,500 calls (roughly 40-60 minutes).

Usage:
  python nhl_shots.py                # pull every game missing shot data
  python nhl_shots.py --limit 500    # cap this run
  python nhl_shots.py --file pbp.json  # parse one saved play-by-play file and print a summary (debugging)
"""

import argparse
import json
import time

import numpy as np
import pandas as pd

from nhl_common import BASE, DATA_DIR, GAMES_PATH, get, save_both

SHOTS_PATH = DATA_DIR / "shots.parquet"
NET_X = 89.0                      # goal lines sit 89 ft from centre ice
SHOT_EVENTS = {"goal", "shot-on-goal", "missed-shot"}
REBOUND_SEC = 3
RUSH_SEC = 4
BLUE_LINE_X = 25.0                # blue lines at x = +/-25

COLUMNS = [
    "game_id", "season", "game_type", "game_date", "home_team", "event_id", "sort_order",
    "period", "period_type", "sec", "team", "opp", "is_home", "shooter_id", "goalie_id",
    "event", "is_goal", "x", "y", "x_n", "y_n", "dist", "angle", "shot_type",
    "skaters_for", "skaters_against", "strength", "empty_net", "penalty_shot",
    "rebound", "rush", "prev_event", "time_since_prev",
]


def _clock(s) -> int:
    """'MM:SS' -> seconds. Bad/missing -> 0."""
    try:
        m, sec = str(s).split(":")
        return int(m) * 60 + int(sec)
    except (ValueError, AttributeError):
        return 0


def _situation(code):
    """situationCode is 4 digits: away goalie in net, away skaters, home skaters, home goalie in net.
    '1551' = 5v5 with both goalies in. Returns (away_g, away_sk, home_sk, home_g) or None."""
    s = str(code or "")
    if len(s) != 4 or not s.isdigit():
        return None
    return int(s[0]), int(s[1]), int(s[2]), int(s[3])


def _target_signs(plays, home_id, away_id):
    """Which net (x = +89 or -89) each team attacks in each period.

    Uses homeTeamDefendingSide when present ('left' = home defends the x<0 net, so home shoots at +89).
    Otherwise falls back to a majority vote of each team's offensive-zone event x-coordinates."""
    signs = {}
    votes = {}
    for p in plays:
        per = (p.get("periodDescriptor") or {}).get("number")
        side = p.get("homeTeamDefendingSide")
        if per is not None and side in ("left", "right") and (per, home_id) not in signs:
            h = 1 if side == "left" else -1
            signs[(per, home_id)] = h
            signs[(per, away_id)] = -h
        d = p.get("details") or {}
        x, owner = d.get("xCoord"), d.get("eventOwnerTeamId")
        if per is not None and x not in (None, 0) and owner is not None and d.get("zoneCode") == "O":
            votes.setdefault((per, owner), []).append(np.sign(x))
    for (per, team), v in votes.items():
        if (per, team) not in signs and v:
            s = int(np.sign(np.sum(v)) or 1)
            signs[(per, team)] = s
            other = away_id if team == home_id else home_id
            signs.setdefault((per, other), -s)
    return signs


def parse_shots(pbp: dict) -> list[dict]:
    """One play-by-play payload -> list of shot-attempt rows."""
    if not pbp or not pbp.get("plays"):
        return []
    home, away = pbp.get("homeTeam") or {}, pbp.get("awayTeam") or {}
    home_id, away_id = home.get("id"), away.get("id")
    abbrev = {home_id: home.get("abbrev"), away_id: away.get("abbrev")}
    plays = sorted(pbp["plays"], key=lambda p: (p.get("sortOrder") or 0, p.get("eventId") or 0))
    signs = _target_signs(plays, home_id, away_id)

    rows = []
    prev = None                  # previous play with a clock (any type)
    last_shot = {}               # team id -> (period, sec) of that team's last unblocked attempt
    for p in plays:
        pd_ = p.get("periodDescriptor") or {}
        per, ptype = pd_.get("number"), pd_.get("periodType")
        sec = (per - 1) * 1200 + _clock(p.get("timeInPeriod")) if per else None
        kind = p.get("typeDescKey")
        d = p.get("details") or {}

        if kind in SHOT_EVENTS and ptype != "SO" and sec is not None:
            team_id = d.get("eventOwnerTeamId")
            x, y = d.get("xCoord"), d.get("yCoord")
            sit = _situation(p.get("situationCode"))
            if team_id in abbrev and x is not None and y is not None:
                is_home = team_id == home_id
                sign = signs.get((per, team_id))
                if sign is None:                       # last resort: shoot at the nearer net
                    sign = 1 if x >= 0 else -1
                x_n, y_n = x * sign, y * sign
                dx = NET_X - x_n
                dist = float(np.hypot(dx, y_n))
                angle = float(np.degrees(np.arctan2(abs(y_n), dx)))

                sk_for = sk_against = None
                empty_net = False
                eff_for = eff_against = None
                if sit:
                    a_g, a_sk, h_sk, h_g = sit
                    sk_for, sk_against = (h_sk, a_sk) if is_home else (a_sk, h_sk)
                    g_for, g_against = (h_g, a_g) if is_home else (a_g, h_g)
                    empty_net = g_against == 0
                    # a pulled goalie's extra attacker doesn't make it a power play: 6v5 with the net empty is EV
                    eff_for = sk_for - (1 if g_for == 0 and sk_for >= 4 else 0)
                    eff_against = sk_against - (1 if g_against == 0 and sk_against >= 4 else 0)
                penalty_shot = sk_for == 1 and sk_against == 0
                if eff_for is None or penalty_shot:
                    strength = "PS" if penalty_shot else "EV"
                elif eff_for > eff_against:
                    strength = "PP"
                elif eff_for < eff_against:
                    strength = "SH"
                else:
                    strength = "EV"

                tsp = prev_type = None
                rush = False
                if prev is not None and prev["period"] == per:
                    tsp = sec - prev["sec"]
                    prev_type = prev["kind"]
                    if tsp <= RUSH_SEC and prev["x"] is not None:
                        # previous event's x, seen from THIS shooter's attacking direction
                        rush = prev["x"] * sign < BLUE_LINE_X
                ls = last_shot.get(team_id)
                rebound = bool(ls and ls[0] == per and 0 <= sec - ls[1] <= REBOUND_SEC)

                rows.append({
                    "game_id": pbp.get("id"), "season": pbp.get("season"), "game_type": pbp.get("gameType"),
                    "game_date": pbp.get("gameDate"), "home_team": abbrev[home_id],
                    "event_id": p.get("eventId"), "sort_order": p.get("sortOrder"),
                    "period": per, "period_type": ptype, "sec": sec,
                    "team": abbrev[team_id], "opp": abbrev[away_id if is_home else home_id], "is_home": is_home,
                    "shooter_id": d.get("shootingPlayerId") or d.get("scoringPlayerId"),
                    "goalie_id": d.get("goalieInNetId"),
                    "event": kind, "is_goal": kind == "goal",
                    "x": x, "y": y, "x_n": x_n, "y_n": y_n, "dist": dist, "angle": angle,
                    "shot_type": (d.get("shotType") or "unknown").lower(),
                    "skaters_for": sk_for, "skaters_against": sk_against, "strength": strength,
                    "empty_net": empty_net, "penalty_shot": penalty_shot,
                    "rebound": rebound, "rush": bool(rush),
                    "prev_event": prev_type, "time_since_prev": tsp,
                })
                last_shot[team_id] = (per, sec)

        if sec is not None:
            prev = {"period": per, "sec": sec, "kind": kind, "x": d.get("xCoord")}
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None, help="max games to pull this run")
    ap.add_argument("--file", type=str, default=None, help="parse one saved play-by-play JSON and print a summary")
    args = ap.parse_args()

    if args.file:
        with open(args.file) as f:
            df = pd.DataFrame(parse_shots(json.load(f)), columns=COLUMNS)
        print(df.groupby(["team", "event"]).size().unstack(fill_value=0))
        print(df[["dist", "angle"]].describe().round(1))
        return

    if not GAMES_PATH.exists():
        print("No data/games.parquet found - run nhl_backfill.py first.")
        return
    games = pd.read_parquet(GAMES_PATH)
    games = games[games["game_type"].isin([2, 3]) & games["game_state"].isin(["OFF", "FINAL"])]
    existing = pd.read_parquet(SHOTS_PATH) if SHOTS_PATH.exists() else pd.DataFrame(columns=COLUMNS)
    have = set(existing["game_id"].unique())
    todo = [g for g in games["game_id"].unique() if g not in have]
    if args.limit:
        todo = todo[: args.limit]
    print(f"Pulling shot data for {len(todo)} games ({len(have)} already done)...")

    buffer, frames, empty = [], ([existing] if not existing.empty else []), 0
    for i, gid in enumerate(todo, 1):
        rows = parse_shots(get(f"{BASE}/gamecenter/{gid}/play-by-play"))
        empty += not rows
        buffer.extend(rows)
        if i % 250 == 0 or i == len(todo):
            if buffer:
                frames.append(pd.DataFrame(buffer, columns=COLUMNS))
                buffer = []
                out = pd.concat(frames, ignore_index=True).drop_duplicates(
                    subset=["game_id", "event_id"], keep="last")
                save_both(out, SHOTS_PATH)
                frames = [out]
            print(f"  {i}/{len(todo)} games processed")
        time.sleep(0.15)  # be polite to the API
    if empty:
        print(f"  note: {empty} games returned no shots (missing play-by-play); they'll be retried next run.")
    print("Done.")


if __name__ == "__main__":
    main()
