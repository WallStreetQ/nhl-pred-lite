"""
nhl_backfill.py
----------------
Pulls the last 6 completed NHL seasons of:
  1) Team season stats (final standings snapshot per team per season)
  2) Every completed game (final score, date, teams)

Data source: the official NHL web API (api-web.nhle.com) — free, no key needed.
Reference: https://github.com/Zmalski/NHL-API-Reference

Output (in ./data/):
  team_season_stats.parquet / .csv
  games.parquet / .csv

Usage:
  python nhl_backfill.py                # last 6 completed seasons
  python nhl_backfill.py --seasons 8    # customize how many seasons back
"""

import argparse
import time

import pandas as pd

from nhl_common import BASE, GAMES_PATH, TEAM_CODES, TEAM_STATS_PATH, get, save_both, season_ids


def last_regular_season_date(games_df: pd.DataFrame) -> str | None:
    """Finds the latest date with a completed regular-season game (game_type == 2)."""
    if games_df.empty:
        return None
    reg = games_df[games_df["game_type"] == 2]
    if reg.empty:
        return None
    return str(reg["game_date"].max())


def pull_team_season_stats(season: str, snapshot_date: str | None) -> pd.DataFrame:
    """Uses the actual last day of the regular season (derived from games data) to
    get each team's final regular-season record. A fixed calendar date doesn't work
    here since playoffs start at different points each year and the standings-by-date
    endpoint returns empty once you're past the regular season window."""
    if not snapshot_date:
        print(f"  [warn] no regular-season game dates found for {season}, skipping team stats")
        return pd.DataFrame()

    data = get(f"{BASE}/standings/{snapshot_date}")
    if not data or "standings" not in data:
        print(f"  [warn] no standings data for {season} (tried {snapshot_date})")
        return pd.DataFrame()

    rows = []
    for team in data["standings"]:
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
    print(f"  team_season_stats: {len(rows)} teams for {season}")
    return pd.DataFrame(rows)


def pull_games_for_season(season: str) -> pd.DataFrame:
    """Pulls every team's season schedule and dedupes by game_id."""
    seen_games = {}
    for code in TEAM_CODES:
        data = get(f"{BASE}/club-schedule-season/{code}/{season}")
        if not data or "games" not in data:
            continue
        for g in data["games"]:
            gid = g.get("id")
            if gid is None or gid in seen_games:
                continue
            if g.get("gameState") not in ("OFF", "FINAL"):
                continue  # skip games that haven't been played yet
            home = g.get("homeTeam", {})
            away = g.get("awayTeam", {})
            seen_games[gid] = {
                "game_id": gid,
                "season": season,
                "game_date": g.get("gameDate"),
                "game_type": g.get("gameType"),
                "game_state": g.get("gameState"),
                "home_team": home.get("abbrev"),
                "away_team": away.get("abbrev"),
                "home_score": home.get("score"),
                "away_score": away.get("score"),
                "venue": (g.get("venue") or {}).get("default"),
            }
        time.sleep(0.15)  # be polite to the API

    print(f"  games: {len(seen_games)} unique games for {season}")
    return pd.DataFrame(list(seen_games.values()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", type=int, default=6, help="how many seasons back to pull")
    args = parser.parse_args()

    seasons = season_ids(args.seasons)
    print(f"Pulling seasons: {seasons}")

    all_team_stats = []
    all_games = []
    for season in seasons:
        print(f"Season {season}:")
        games_df = pull_games_for_season(season)
        all_games.append(games_df)
        snapshot_date = last_regular_season_date(games_df)
        all_team_stats.append(pull_team_season_stats(season, snapshot_date))

    team_stats_df = pd.concat(all_team_stats, ignore_index=True) if all_team_stats else pd.DataFrame()
    games_df = pd.concat(all_games, ignore_index=True) if all_games else pd.DataFrame()

    if not team_stats_df.empty:
        team_stats_df = team_stats_df.drop_duplicates(subset=["season", "team_abbrev"], keep="last")
        save_both(team_stats_df, TEAM_STATS_PATH)

    if not games_df.empty:
        games_df = games_df.drop_duplicates(subset=["game_id"], keep="last")
        save_both(games_df, GAMES_PATH)

    print("Done.")


if __name__ == "__main__":
    main()

