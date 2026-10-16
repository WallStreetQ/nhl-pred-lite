"""
nhl_goalies.py
---------------
Backfills per-goalie, per-game data for every game already in data/games.parquet.

Data source: the `boxscore` endpoint. Each team's `playerByGameStats.<side>.goalies`
list has every goalie who dressed, with:
  starter (bool), decision (W/L/OT), toi ("MM:SS"), shotsAgainst, saves, goalsAgainst,
  and strength splits as "saves/shots" strings (evenStrengthShotsAgainst, etc.).

NOTE: these starter flags exist only AFTER the game. The NHL API publishes no confirmed
pre-game starter, so for live predictions you need an outside probable-starter source
(see goalie_features.expected_goalie_effect). This file is for building history.

Output: data/game_goalies.parquet / .csv  — one row per goalie per game
(including 0:00 backups, which is useful: it tells you who dressed).

Run AFTER nhl_backfill.py (needs game IDs). Resume-friendly; checkpoints every 250 games.

Usage:
  python nhl_goalies.py                # pull every game missing goalie data
  python nhl_goalies.py --limit 500    # cap this run
"""

import argparse
import time

import pandas as pd

from nhl_common import BASE, DATA_DIR, GAMES_PATH, get, save_both

GOALIES_PATH = DATA_DIR / "game_goalies.parquet"

COLUMNS = [
    "game_id", "game_date", "season", "game_type", "team", "opponent", "is_home",
    "player_id", "name", "starter", "decision", "toi_sec",
    "shots_against", "saves", "goals_against",
    "es_sa", "es_ga", "pp_sa", "pp_ga", "sh_sa", "sh_ga",
]


def _split(s):
    """'33/35' -> (saves=33, shots=35). Missing/garbled -> (0, 0)."""
    try:
        a, b = str(s).split("/")
        return int(a), int(b)
    except (ValueError, AttributeError):
        return 0, 0


def _toi_seconds(s) -> int:
    try:
        m, sec = str(s).split(":")
        return int(m) * 60 + int(sec)
    except (ValueError, AttributeError):
        return 0


def parse_goalies(box: dict) -> list[dict]:
    """Turn one boxscore payload into goalie rows (both teams)."""
    if not box:
        return []
    game_id = box.get("id")
    stats = box.get("playerByGameStats") or {}
    teams = {
        "homeTeam": (box.get("homeTeam", {}).get("abbrev"), box.get("awayTeam", {}).get("abbrev"), True),
        "awayTeam": (box.get("awayTeam", {}).get("abbrev"), box.get("homeTeam", {}).get("abbrev"), False),
    }

    rows = []
    for side, (team, opp, is_home) in teams.items():
        side_rows = []
        for gk in (stats.get(side) or {}).get("goalies", []):
            es_saves, es_sa = _split(gk.get("evenStrengthShotsAgainst"))
            pp_saves, pp_sa = _split(gk.get("powerPlayShotsAgainst"))
            sh_saves, sh_sa = _split(gk.get("shorthandedShotsAgainst"))
            shots = gk.get("shotsAgainst")
            saves = gk.get("saves")
            if shots is None:
                saves, shots = _split(gk.get("saveShotsAgainst"))
            side_rows.append({
                "game_id": game_id,
                "game_date": box.get("gameDate"),
                "season": box.get("season"),
                "game_type": box.get("gameType"),
                "team": team,
                "opponent": opp,
                "is_home": is_home,
                "player_id": gk.get("playerId"),
                "name": (gk.get("name") or {}).get("default"),
                "starter": bool(gk.get("starter")),
                "decision": gk.get("decision"),
                "toi_sec": _toi_seconds(gk.get("toi")),
                "shots_against": shots or 0,
                "saves": saves or 0,
                "goals_against": gk.get("goalsAgainst") or 0,
                "es_sa": es_sa, "es_ga": gk.get("evenStrengthGoalsAgainst") or 0,
                "pp_sa": pp_sa, "pp_ga": gk.get("powerPlayGoalsAgainst") or 0,
                "sh_sa": sh_sa, "sh_ga": gk.get("shorthandedGoalsAgainst") or 0,
            })
        # Safety net: if the API omitted the starter flag, the goalie with the most ice time started.
        if side_rows and not any(r["starter"] for r in side_rows):
            max(side_rows, key=lambda r: r["toi_sec"])["starter"] = True
        rows.extend(side_rows)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="max games to pull this run")
    args = parser.parse_args()

    if not GAMES_PATH.exists():
        print("No data/games.parquet found — run nhl_backfill.py first.")
        return

    games_df = pd.read_parquet(GAMES_PATH)
    games_df = games_df[games_df["game_type"].isin([2, 3])]  # skip preseason
    existing = pd.read_parquet(GOALIES_PATH) if GOALIES_PATH.exists() else pd.DataFrame(columns=COLUMNS)
    have = set(existing["game_id"].unique())
    todo = [g for g in games_df["game_id"].unique() if g not in have]
    if args.limit:
        todo = todo[: args.limit]
    print(f"Pulling goalie data for {len(todo)} games ({len(have)} already done)...")

    buffer, frames = [], [existing] if not existing.empty else []
    for i, gid in enumerate(todo, 1):
        buffer.extend(parse_goalies(get(f"{BASE}/gamecenter/{gid}/boxscore")))
        if i % 250 == 0 or i == len(todo):
            if buffer:
                frames.append(pd.DataFrame(buffer, columns=COLUMNS))
                buffer = []
                out = pd.concat(frames, ignore_index=True).drop_duplicates(
                    subset=["game_id", "player_id"], keep="last")
                save_both(out, GOALIES_PATH)
                frames = [out]
            print(f"  {i}/{len(todo)} games processed")
        time.sleep(0.15)  # be polite to the API
    print("Done.")


if __name__ == "__main__":
    main()
