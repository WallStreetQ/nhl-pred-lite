"""
nhl_common.py
--------------
Shared constants and helpers for the NHL data pipeline scripts.
"""

import time
from datetime import date
from pathlib import Path

import requests

BASE = "https://api-web.nhle.com/v1"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; nhl-data-script/1.0)"}

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

GAMES_PATH = DATA_DIR / "games.parquet"
TEAM_STATS_PATH = DATA_DIR / "team_season_stats.parquet"
PERIODS_PATH = DATA_DIR / "game_periods.parquet"

TEAM_CODES = [
    "ANA", "ARI", "BOS", "BUF", "CGY", "CAR", "CHI", "COL", "CBJ", "DAL",
    "DET", "EDM", "FLA", "LAK", "MIN", "MTL", "NSH", "NJD", "NYI", "NYR",
    "OTT", "PHI", "PIT", "SJS", "SEA", "STL", "TBL", "TOR", "UTA", "VAN",
    "VGK", "WSH", "WPG",
]
# ARI (Coyotes) and UTA (Utah) both appear since the franchise moved in 2024;
# whichever didn't exist in a given season simply returns no games for it.


def get(url: str, retries: int = 3, pause: float = 1.5):
    """GET a JSON endpoint with basic retry/backoff. Returns None on 404 or failure."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=20)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 404:
                return None
        except requests.RequestException:
            pass
        time.sleep(pause * (attempt + 1))
    return None


def season_ids(n_seasons: int):
    """Return the last n completed season IDs, e.g. '20192020', '20202021', ...

    A season starting in year Y is treated as complete once July of Y+1 has passed
    (playoffs always wrap by June).
    """
    today = date.today()
    last_completed_start_year = today.year - 1 if today.month >= 7 else today.year - 2
    return [
        f"{y}{y+1}" for y in range(
            last_completed_start_year - n_seasons + 1, last_completed_start_year + 1
        )
    ]


def season_id_for_date(d: date) -> str:
    """NHL seasons span Oct(Y)-June(Y+1). A date in Jan 2026 belongs to season 20252026."""
    start_year = d.year if d.month >= 7 else d.year - 1
    return f"{start_year}{start_year + 1}"


def save_both(df, path: Path):
    """Save a dataframe to its Parquet path, plus a sibling CSV for quick inspection."""
    df.to_parquet(path, index=False)
    df.to_csv(path.with_suffix(".csv"), index=False)
    print(f"  saved {len(df)} rows -> {path} (+ .csv)")
