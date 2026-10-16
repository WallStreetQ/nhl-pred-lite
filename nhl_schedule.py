"""
nhl_schedule.py
----------------
Pulls upcoming (not yet started) regular-season and playoff games for the next few days into
data/upcoming.json. refresh.py feeds this into the model artifact so the app can show the slate,
flag back-to-backs automatically, and log pre-game predictions.

Endpoint: /v1/schedule/{date} returns the week starting at that date:
  {"gameWeek": [{"date": "YYYY-MM-DD", "games": [{"id", "gameType", "gameState", "startTimeUTC",
                                                 "homeTeam": {"abbrev"}, "awayTeam": {"abbrev"}}]}]}

Usage:
  python nhl_schedule.py              # today + next 3 days
  python nhl_schedule.py --days 7
"""

import argparse
import json
from datetime import date, timedelta

from nhl_common import BASE, DATA_DIR, get

UPCOMING_PATH = DATA_DIR / "upcoming.json"
NOT_STARTED = {"FUT", "PRE"}


def pull_upcoming(days: int = 4, start: date | None = None) -> list[dict]:
    start = start or date.today()
    end = start + timedelta(days=days - 1)
    data = get(f"{BASE}/schedule/{start.isoformat()}")
    if not data or "gameWeek" not in data:
        print("  [warn] no schedule data returned")
        return []
    out = []
    for day in data["gameWeek"]:
        d = day.get("date")
        if not d or not (start.isoformat() <= d <= end.isoformat()):
            continue
        for g in day.get("games", []):
            if g.get("gameType") not in (2, 3) or g.get("gameState") not in NOT_STARTED:
                continue
            out.append({
                "game_id": g.get("id"),
                "date": d,
                "start_utc": g.get("startTimeUTC"),
                "game_type": g.get("gameType"),
                "home": (g.get("homeTeam") or {}).get("abbrev"),
                "away": (g.get("awayTeam") or {}).get("abbrev"),
            })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=4, help="how many days ahead, including today (max 7)")
    args = ap.parse_args()
    games = pull_upcoming(min(args.days, 7))
    with open(UPCOMING_PATH, "w") as f:
        json.dump(games, f, indent=1)
    print(f"  saved {len(games)} upcoming games -> {UPCOMING_PATH}")


if __name__ == "__main__":
    main()
