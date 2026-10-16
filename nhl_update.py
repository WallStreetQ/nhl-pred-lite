"""
nhl_update.py
--------------
Incremental updater — run this daily (e.g. via Task Scheduler / cron) during the season.
Pulls games from a given date (default: yesterday) and merges any newly-completed
games into data/games.parquet. Also refreshes the current season's standings snapshot
in data/team_season_stats.parquet.

Safe to re-run: rows are deduped by game_id / (season, team_abbrev), so nothing
gets duplicated.

Usage:
  python nhl_update.py                     # updates using yesterday's games
  python nhl_update.py --date 2026-01-15   # updates using a specific date
  python nhl_update.py --days-back 3       # sweep the last N days (catches makeups/postponements)
"""

import argparse
from datetime import date, timedelta

import pandas as pd

from nhl_common import BASE, GAMES_PATH, TEAM_STATS_PATH, get, save_both, season_id_for_date


def load_or_empty(path, columns):
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame(columns=columns)


def pull_games_for_date(d: date) -> pd.DataFrame:
    date_str = d.isoformat()
    data = get(f"{BASE}/score/{date_str}")
    if not data or "games" not in data:
        print(f"  [warn] no data for {date_str}")
        return pd.DataFrame()

    rows = []
    for g in data["games"]:
        if g.get("gameState") not in ("OFF", "FINAL"):
            continue  # only record completed games
        home = g.get("homeTeam", {})
        away = g.get("awayTeam", {})
        rows.append({
            "game_id": g.get("id"),
            "season": season_id_for_date(d),
            "game_date": date_str,
            "game_type": g.get("gameType"),
            "game_state": g.get("gameState"),
            "home_team": home.get("abbrev"),
            "away_team": away.get("abbrev"),
            "home_score": home.get("score"),
            "away_score": away.get("score"),
            "venue": (g.get("venue") or {}).get("default"),
        })
    print(f"  {date_str}: {len(rows)} completed games")
    return pd.DataFrame(rows)


def pull_current_standings() -> pd.DataFrame:
    data = get(f"{BASE}/standings/now")
    if not data or "standings" not in data:
        return pd.DataFrame()

    rows = []
    for team in data["standings"]:
        season = str(team.get("seasonId") or season_id_for_date(date.today()))
        rows.append({
            "season": season,
            "team_abbrev": team.get("teamAbbrev", {}).get("default"),
            "team_name": team.get("teamName", {}).get("default"),
            "games_played": team.get("gamesPlayed"),
            "wins": team.get("wins"),
            "losses": team.get("losses"),
            "ot_losses": team.get("otLosses"),
            "points": team.get("points"),
            "goals_for": team.get("goalFor"),
            "goals_against": team.get("goalAgainst"),
            "goal_diff": team.get("goalDifferential"),
        })
    print(f"  refreshed current-season standings for {len(rows)} teams")
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=None, help="YYYY-MM-DD (default: yesterday)")
    parser.add_argument("--days-back", type=int, default=1, help="sweep this many days ending at --date")
    args = parser.parse_args()

    target = date.fromisoformat(args.date) if args.date else date.today() - timedelta(days=1)

    existing_games = load_or_empty(GAMES_PATH, columns=[
        "game_id", "season", "game_date", "game_type", "game_state",
        "home_team", "away_team", "home_score", "away_score", "venue",
    ])

    new_frames = [existing_games]
    for offset in range(args.days_back):
        d = target - timedelta(days=offset)
        new_frames.append(pull_games_for_date(d))

    games_df = pd.concat(new_frames, ignore_index=True)
    games_df = games_df.drop_duplicates(subset=["game_id"], keep="last")
    save_both(games_df, GAMES_PATH)

    existing_stats = load_or_empty(TEAM_STATS_PATH, columns=[
        "season", "team_abbrev", "team_name", "games_played", "wins", "losses",
        "ot_losses", "points", "goals_for", "goals_against", "goal_diff",
    ])
    new_stats = pull_current_standings()
    if not new_stats.empty:
        stats_df = pd.concat([existing_stats, new_stats], ignore_index=True)
        stats_df = stats_df.drop_duplicates(subset=["season", "team_abbrev"], keep="last")
        save_both(stats_df, TEAM_STATS_PATH)

    print("Done.")


if __name__ == "__main__":
    main()
