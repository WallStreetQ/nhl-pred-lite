# Running locally and publishing to Streamlit via GitHub

## How it fits together

```
YOUR MACHINE                                             GITHUB  →  STREAMLIT CLOUD
data/ (games, periods, goalies; stays local)
   │  python refresh.py  (fetch → retrain)
   ▼
model/model_artifact.json  (a few KB)  ── git push ──►  repo  ──►  app.py redeploys automatically
```

The app never touches raw data or the NHL API. It only reads `model/model_artifact.json`, so the repo stays tiny
and the deployed app can't break because an API changed. Updating the live app = producing a new model file and pushing it.

## Files

| File | Purpose | Runs where |
|---|---|---|
| `app.py` | The Streamlit app | local + Streamlit Cloud |
| `nhl_model.py`, `goalie_features.py` | Model code (training + prediction) | both |
| `model/model_artifact.json` | The trained model the app loads | generated locally, committed |
| `refresh.py` | One-command update + optional GitHub push | local |
| `nhl_update.py`, `nhl_periods.py`, `nhl_goalies.py`, `nhl_backfill.py`, `nhl_common.py` | Data pulls from the NHL API | local |
| `nhl_prediction_model.ipynb` | Exploration, backtest, tuning | local |
| `requirements.txt` | What Streamlit Cloud installs | Streamlit Cloud |
| `requirements-local.txt` | Full local install | local |

## First-time setup (local)

```bash
pip install -r requirements-local.txt
streamlit run app.py          # preview the app at http://localhost:8501
```

`data/` must contain `games.parquet`, `game_periods.parquet` (and `game_goalies.parquet` once `nhl_goalies.py` has finished).
Run everything from the project folder.

## Replacing your current Streamlit app

1. Put these files into the GitHub repo your Streamlit app deploys from (overwrite the old app's files).
2. If the old app's main file has a different name (e.g. `streamlit_app.py`), either delete it or open the app in Streamlit Cloud →
   **Manage app → Settings** and set **Main file path** to `app.py`.
3. Replace the old `requirements.txt` with the new one (statsmodels is no longer needed). You can delete the old `model_artifacts.pkl`.
4. Commit and push:

```bash
git add .
git commit -m "New model + app with goalie support"
git push
```

Streamlit Cloud redeploys on every push. If the app doesn't pick up new requirements, use **Manage app → Reboot app**.

Starting from an empty folder instead? `git init`, `git remote add origin <your-repo-url>`, commit, `git push -u origin main`,
then in Streamlit Cloud choose **New app** → your repo → main file `app.py`.

## Updating as the season progresses

```bash
python refresh.py --push
```

That one command: pulls new games, period data and goalie data → retrains → rewrites `model/model_artifact.json` →
commits and pushes **only if the model actually changed** → Streamlit redeploys in about a minute.

| Command | When |
|---|---|
| `python refresh.py --push` | Normal update (daily or every few days in-season) |
| `python refresh.py` | Update locally, check the app with `streamlit run app.py`, publish later |
| `python refresh.py --skip-fetch --push` | Retrain from data you already have (e.g. right after a goalie backfill) |
| `python refresh.py --days-back 5` | Sweep more days to catch postponed/late-posting games |

If any fetch step fails, nothing is published.

**Automating (optional).** Mac/Linux cron, daily at 9:00:
```
0 9 * * * cd /path/to/project && python refresh.py --push >> refresh.log 2>&1
```
Windows: Task Scheduler running the same command. Pushing unattended needs Git credentials set up (SSH key or credential manager).

## Goalies

- Until goalie data is complete (95% of games) the model runs without goalies and the app says "goalie data OFF".
- After `python nhl_goalies.py` finishes: `python refresh.py --skip-fetch --push`.
- In the app, pick each team's goalie once he's confirmed. Left on "Unknown", it uses each team's recent starter mix.
  The NHL API doesn't publish confirmed starters in advance, so this step is manual.

## Troubleshooting

| Symptom | Fix |
|---|---|
| App says the model file wasn't found | `model/model_artifact.json` isn't in the repo. Run `python refresh.py`, then `git add model/model_artifact.json` |
| `push failed` | Someone/something updated the repo. Run `git pull --rebase` then `git push` |
| `This folder isn't a git repository` | Follow "Starting from an empty folder" above, or clone your existing repo and copy the files in |
| "goalies: OFF - only X% of games" | `nhl_goalies.py` isn't finished. Re-run it (it resumes), then refresh |
| Changed teams don't show up (new season) | Run `refresh.py`; active teams are those with games in the last 365 days |
