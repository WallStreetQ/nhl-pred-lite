> **Streamlit app + season-refresh workflow: see [DEPLOY.md](DEPLOY.md).**

# NHL Historical + Live Data Pipeline

Pulls team season stats and full game results — including period-by-period scoring —
from the NHL's official free API (`api-web.nhle.com`) into local Parquet/CSV files.
No database or API key required. Built to run locally in VS Code; the output files
drop straight into Google Colab too if you want to move analysis there later.

## Setup

```bash
pip install pandas pyarrow requests
```

(`pyarrow` is what lets pandas read/write `.parquet` files.)

Files needed in the same folder:
- `nhl_common.py` — shared constants + request helper (imported by the other three)
- `nhl_backfill.py`
- `nhl_update.py`
- `nhl_periods.py`

## 1. Backfill 6 seasons of history

```bash
python nhl_backfill.py
```

Creates a `data/` folder with:
- `team_season_stats.parquet` / `.csv` — one row per team per season (record, points, goals for/against)
- `games.parquet` / `.csv` — every completed game (date, teams, final score, venue)

Takes a few minutes (32 teams × 6 seasons of schedule calls, rate-limited to be
polite to the API).

## 2. Backfill period-by-period scoring (optional, run once)

```bash
python nhl_periods.py
```

Reads the game IDs already in `data/games.parquet` and pulls the boxscore for each
one, extracting goals scored per period per team into `data/game_periods.parquet`.
This is one API call *per game* (thousands of calls across 6 seasons), so:
- It's resume-friendly — it skips games it's already pulled, so you can stop and
  restart anytime.
- Use `--limit 500` to do it in batches if you want to check progress along the way.

## 3. Keep it updated as the season progresses

Run daily once the season starts:

```bash
python nhl_update.py
```

Checks yesterday's games, merges in any that finished, and refreshes the current
season's standings. Safe to run repeatedly — rows are deduped by `game_id`, so
nothing gets duplicated. Re-run `nhl_periods.py` periodically afterward to pick up
period-level data for newly-added games.

**To automate `nhl_update.py`** on Windows, use Task Scheduler; on Mac/Linux, cron:

```
0 9 * * * cd /path/to/project && python nhl_update.py --days-back 2 >> update.log 2>&1
```

`--days-back 2` catches games whose final status posted a bit late, or postponed/
rescheduled games.

## Loading the data for analysis

```python
import pandas as pd

games = pd.read_parquet("data/games.parquet")
team_stats = pd.read_parquet("data/team_season_stats.parquet")
periods = pd.read_parquet("data/game_periods.parquet")
```

## Notes

- No API key required — this is NHL's own public web API, same one their site uses.
- I couldn't execute live test calls against `api-web.nhle.com` from my sandboxed
  environment (its network allowlist doesn't include that domain, and the API also
  blocks generic bot traffic from cloud fetchers). The endpoint paths and field
  names come from the community-maintained reference
  (github.com/Zmalski/NHL-API-Reference) and known response shapes. Run
  `nhl_backfill.py` locally first and skim the CSV output to confirm field names
  line up — the `linescore.byPeriod` structure in `nhl_periods.py` is the most
  likely spot to need a small tweak if the NHL has changed their response shape.
- Team codes list includes both `ARI` and `UTA` since the Coyotes/Utah franchise
  moved in 2024 — whichever didn't exist in a given season just returns no games.

-----usage

In cell 9 ( example prediction )

# use team abbrevaitons
home_team = 'FLA'
away_team = 'VGK'

prediction = predict_game(model, period_shares, home_team, away_team)
for k, v in prediction.items():
  print (f"{k}: {v}")

  #run cell

 #note to find the offical team code use
 print(sorted(games["home_team"].unique())) 