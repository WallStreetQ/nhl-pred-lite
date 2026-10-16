"""
nhl_periods.py
---------------
Backfills period-by-period goals for every game already in data/games.parquet.

Data source: the `landing` endpoint (NOT boxscore — boxscore has no period
breakdown; landing's `summary.scoring` lists every goal, grouped by period,
with a `teamAbbrev` and `isHome` flag on each). We tally goals per period per
team ourselves from that goal-by-goal list.

Note on shootouts: a shootout-deciding goal isn't recorded as a period-play
scoring event the same way regulation/OT goals are, so for games decided in
a shootout, the period-level goal totals here may not sum to the final score.
Those games are flagged with went_to_shootout=True in the output so you can
filter or handle them separately.

Run this AFTER nhl_backfill.py since it needs game IDs to already exist in
games.parquet.

Usage:
  python nhl_periods.py                 # backfill any games missing period data
  python nhl_periods.py --limit 500     # cap how many games to pull this run (resume-friendly)
"""

import argparse
import time

import pandas as pd

from nhl_common import BASE, GAMES_PATH, PERIODS_PATH, get, save_both


def load_or_empty(path, columns):
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame(columns=columns)


def pull_periods_for_game(game_id) -> list[dict]:
    data = get(f"{BASE}/gamecenter/{game_id}/landing")
    if not data:
        return []

    home_abbrev = data.get("homeTeam", {}).get("abbrev")
    away_abbrev = data.get("awayTeam", {}).get("abbrev")
    reg_periods = data.get("regPeriods", 3)
    final_period_num = data.get("periodDescriptor", {}).get("number", reg_periods)
    went_to_shootout = bool(data.get("shootoutInUse")) and (
        data.get("periodDescriptor", {}).get("periodType") == "SO"
    )

    scoring = (data.get("summary") or {}).get("scoring", [])

    # Tally actual goals per period from the goal-by-goal list.
    period_info = {}  # period_num -> {"home": int, "away": int, "period_type": str}
    for period_entry in scoring:
        pdesc = period_entry.get("periodDescriptor", {})
        period_num = pdesc.get("number")
        period_type = pdesc.get("periodType")
        home_count = sum(1 for g in period_entry.get("goals", []) if g.get("isHome"))
        away_count = sum(1 for g in period_entry.get("goals", []) if not g.get("isHome"))
        period_info[period_num] = {
            "home": home_count,
            "away": away_count,
            "period_type": period_type,
        }

    # Build one row per team per period, covering every period that occurred
    # (including scoreless ones, which won't have an entry in `scoring`).
    rows = []
    for period_num in range(1, final_period_num + 1):
        info = period_info.get(period_num)
        if info:
            period_type = info["period_type"]
            home_goals = info["home"]
            away_goals = info["away"]
        else:
            period_type = "REG" if period_num <= reg_periods else "OT"
            home_goals = 0
            away_goals = 0
        for team, is_home, goals in (
            (home_abbrev, True, home_goals),
            (away_abbrev, False, away_goals),
        ):
            rows.append({
                "game_id": game_id,
                "period": period_num,
                "period_type": period_type,
                "team": team,
                "is_home": is_home,
                "goals": goals,
                "went_to_shootout": went_to_shootout,
            })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="max number of games to pull this run")
    args = parser.parse_args()

    if not GAMES_PATH.exists():
        print("No data/games.parquet found — run nhl_backfill.py first.")
        return

    games_df = pd.read_parquet(GAMES_PATH)
    existing_periods = load_or_empty(PERIODS_PATH, columns=[
        "game_id", "period", "period_type", "team", "is_home", "goals", "went_to_shootout",
    ])

    already_have = set(existing_periods["game_id"].unique()) if not existing_periods.empty else set()
    todo = [gid for gid in games_df["game_id"].unique() if gid not in already_have]
    if args.limit:
        todo = todo[: args.limit]

    print(f"Pulling period data for {len(todo)} games ({len(already_have)} already done)...")

    new_rows = []
    for i, gid in enumerate(todo, 1):
        new_rows.extend(pull_periods_for_game(gid))
        if i % 100 == 0:
            print(f"  {i}/{len(todo)} games processed")
        time.sleep(0.15)  # be polite to the API

    if new_rows:
        periods_df = pd.concat([existing_periods, pd.DataFrame(new_rows)], ignore_index=True)
        periods_df = periods_df.drop_duplicates(subset=["game_id", "period", "team"], keep="last")
        save_both(periods_df, PERIODS_PATH)
    else:
        print("No new period data pulled.")

    print("Done.")


if __name__ == "__main__":
    main()
