"""
nhl_odds.py
------------
Pulls current NHL odds from The Odds API (https://the-odds-api.com) for:
    hardrockbet_fl   Hard Rock Bet, Florida feed  - the prices you can actually bet (PRIMARY)
    hardrockbet      Hard Rock Bet, main feed     - used for any game the FL feed doesn't carry
    pinnacle         sharp reference book         - the yardstick for "is the model's edge real?" (REFERENCE)
The app and edge log see a single "Hard Rock Bet" book per game, tagged with the feed it came from.
Markets: moneyline (h2h), puck line (spreads) and game total (totals), main lines only.

Each pull is appended to data/odds_snapshots.parquet (+ .csv), one row per book / market / side, with the
NHL game_id matched from data/upcoming.json. refresh.py calls pull_and_save() and puts the latest prices for
upcoming games into the model artifact, so the Streamlit app needs no API key.

API key (never commit it): set the environment variable ODDS_API_KEY, or put the key alone in a file
called odds_api_key.txt in the project folder (add that file to .gitignore).
    Windows (once, then open a new terminal):   setx ODDS_API_KEY "your-key-here"

Cost: one pull = 3 markets x 1 region-equivalent = about 3 credits (both books are requested by name, and the
API counts up to 10 named books as one region). The free plan has 500 credits/month; 2 pulls a day ~ 180.
Each pull prints the credits it used and how many remain, so you can confirm the cost.

Usage:
  python nhl_odds.py           # pull now, save a snapshot, print the matched games
  python nhl_odds.py --show    # print the latest saved snapshot (no API call)
  python nhl_odds.py --list-books   # which books the API returns for NHL right now (2 credits)
"""

import argparse
import json
import os
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
SNAP_PATH = DATA / "odds_snapshots.parquet"
UPCOMING = DATA / "upcoming.json"
KEY_FILE = ROOT / "odds_api_key.txt"
API_URL = "https://api.the-odds-api.com/v4/sports/icehockey_nhl/odds"
PRIMARY = "hardrock"                              # logical book the app and edge log use
PRIMARY_FEEDS = ["hardrockbet_fl", "hardrockbet"]  # Florida feed first; main Hard Rock feed fills any gaps
REFERENCE = "pinnacle"
BOOKS = PRIMARY_FEEDS + [REFERENCE]
MARKETS = ["h2h", "spreads", "totals"]
BOOK_NAMES = {"hardrock": "Hard Rock Bet", "hardrockbet_fl": "Hard Rock Bet (FL feed)",
              "hardrockbet": "Hard Rock Bet (main feed)", "pinnacle": "Pinnacle"}
MATCH_HOURS = 18            # an odds event matches a scheduled game if start times are within this many hours

_TEAMS = {
    "ANA": ["Anaheim Ducks"], "ARI": ["Arizona Coyotes"], "BOS": ["Boston Bruins"], "BUF": ["Buffalo Sabres"],
    "CGY": ["Calgary Flames"], "CAR": ["Carolina Hurricanes"], "CHI": ["Chicago Blackhawks"],
    "COL": ["Colorado Avalanche"], "CBJ": ["Columbus Blue Jackets"], "DAL": ["Dallas Stars"],
    "DET": ["Detroit Red Wings"], "EDM": ["Edmonton Oilers"], "FLA": ["Florida Panthers"],
    "LAK": ["Los Angeles Kings"], "MIN": ["Minnesota Wild"], "MTL": ["Montreal Canadiens"],
    "NSH": ["Nashville Predators"], "NJD": ["New Jersey Devils"], "NYI": ["New York Islanders"],
    "NYR": ["New York Rangers"], "OTT": ["Ottawa Senators"], "PHI": ["Philadelphia Flyers"],
    "PIT": ["Pittsburgh Penguins"], "SJS": ["San Jose Sharks"], "SEA": ["Seattle Kraken"],
    "STL": ["St Louis Blues"], "TBL": ["Tampa Bay Lightning"], "TOR": ["Toronto Maple Leafs"],
    "UTA": ["Utah Hockey Club", "Utah Mammoth", "Utah HC"], "VAN": ["Vancouver Canucks"],
    "VGK": ["Vegas Golden Knights"], "WSH": ["Washington Capitals"], "WPG": ["Winnipeg Jets"],
}


def _norm(name):
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return " ".join(s.replace(".", "").lower().split())


TEAM_ABBREV = {_norm(n): a for a, names in _TEAMS.items() for n in names}


def team_abbrev(name):
    return TEAM_ABBREV.get(_norm(name))


# ------------------------------------------------------------------------------------------
# Fetch + parse
# ------------------------------------------------------------------------------------------

def api_key():
    k = os.environ.get("ODDS_API_KEY", "").strip()
    if not k and KEY_FILE.exists():
        k = KEY_FILE.read_text().strip()
    return k or None


def fetch(key):
    """One API call -> (events list, usage dict)."""
    import requests
    params = {"apiKey": key, "bookmakers": ",".join(BOOKS), "markets": ",".join(MARKETS),
              "oddsFormat": "american", "dateFormat": "iso"}
    r = requests.get(API_URL, params=params, timeout=20)
    usage = {k: r.headers.get(f"x-requests-{k}") for k in ("used", "remaining", "last")}
    if r.status_code == 401:
        raise RuntimeError("The Odds API rejected the key (401). Check ODDS_API_KEY / odds_api_key.txt.")
    if r.status_code == 429:
        raise RuntimeError("Out of API credits or rate-limited (429). Credits reset monthly.")
    if r.status_code != 200:
        raise RuntimeError(f"The Odds API returned {r.status_code}: {r.text[:200]}")
    return r.json(), usage


def parse(events, pulled_at):
    """API events -> long rows: one per book / market / side."""
    rows, unknown = [], set()
    for ev in events or []:
        home, away = team_abbrev(ev.get("home_team")), team_abbrev(ev.get("away_team"))
        if not home or not away:
            unknown |= {n for n, a in ((ev.get("home_team"), home), (ev.get("away_team"), away)) if not a}
            continue
        for bk in ev.get("bookmakers", []):
            for mk in bk.get("markets", []):
                for oc in mk.get("outcomes", []):
                    name = oc.get("name", "")
                    if mk["key"] == "totals":
                        side = name.lower()
                    elif name == ev.get("home_team"):
                        side = "home"
                    elif name == ev.get("away_team"):
                        side = "away"
                    else:
                        continue
                    if side not in ("home", "away", "over", "under"):
                        continue
                    rows.append({"pulled_at": pulled_at, "event_id": ev.get("id"),
                                 "commence_time": ev.get("commence_time"), "home": home, "away": away,
                                 "book": bk.get("key"), "market": mk["key"], "side": side,
                                 "point": oc.get("point"), "price": oc.get("price"),
                                 "book_last_update": mk.get("last_update") or bk.get("last_update")})
    if unknown:
        print(f"  [warn] unknown team names in the odds feed (add them to _TEAMS): {sorted(unknown)}")
    return pd.DataFrame(rows, columns=["pulled_at", "event_id", "commence_time", "home", "away", "book", "market",
                                       "side", "point", "price", "book_last_update"])


def load_upcoming():
    return json.loads(UPCOMING.read_text()) if UPCOMING.exists() else []


def match_games(df, upcoming):
    """Adds game_id by matching (home, away) and start time to the NHL schedule in data/upcoming.json."""
    df = df.copy()
    df["game_id"] = pd.NA
    if df.empty or not upcoming:
        return df
    sched = pd.DataFrame(upcoming)
    if "start_utc" in sched and sched["start_utc"].notna().any():
        sched["start"] = pd.to_datetime(sched["start_utc"], utc=True, errors="coerce")
    else:
        sched["start"] = pd.NaT
    noon_et = pd.to_datetime(sched["date"]).dt.tz_localize("America/New_York") + pd.Timedelta(hours=19)
    sched["start"] = sched["start"].fillna(noon_et.dt.tz_convert("UTC"))
    ct = pd.to_datetime(df["commence_time"], utc=True, errors="coerce")
    for (home, away), idx in df.groupby(["home", "away"]).groups.items():
        cand = sched[(sched["home"] == home) & (sched["away"] == away)]
        if cand.empty:
            continue
        for i in idx:
            gap = (cand["start"] - ct[i]).abs()
            if gap.min() <= pd.Timedelta(hours=MATCH_HOURS):
                df.at[i, "game_id"] = int(cand.loc[gap.idxmin(), "game_id"])
    return df


def load_snapshots():
    for p, fn in ((SNAP_PATH, pd.read_parquet), (SNAP_PATH.with_suffix(".csv"), pd.read_csv)):
        if p.exists():
            try:
                return fn(p)
            except ImportError:
                continue
    return pd.DataFrame()


def save_snapshot(df):
    old = load_snapshots()
    out = pd.concat([old, df], ignore_index=True) if not old.empty else df
    DATA.mkdir(exist_ok=True)
    try:
        out.to_parquet(SNAP_PATH, index=False)
    except ImportError:
        pass
    out.to_csv(SNAP_PATH.with_suffix(".csv"), index=False)
    return out


def pull_and_save(key=None, log=print):
    """Pull, match to the schedule, append to the snapshot file. Returns the new rows (or None if no key)."""
    key = key or api_key()
    if not key:
        log("  odds: no API key (set ODDS_API_KEY or create odds_api_key.txt) - skipped")
        return None
    pulled_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    events, usage = fetch(key)
    df = match_games(parse(events, pulled_at), load_upcoming())
    save_snapshot(df)
    books = df.groupby("book")["event_id"].nunique().to_dict() if not df.empty else {}
    log(f"  odds: {len(events)} games in feed | " + ", ".join(f"{BOOK_NAMES.get(b, b)} {books.get(b, 0)}" for b in BOOKS)
        + f" | matched to schedule: {df.dropna(subset=['game_id'])['event_id'].nunique()}"
        + f" | credits used this call {usage.get('last')}, remaining {usage.get('remaining')}")
    if not any(books.get(b) for b in PRIMARY_FEEDS):
        log("  [warn] no Hard Rock prices from either feed. Run `python nhl_odds.py --list-books` to see which "
            "books the API is returning for NHL right now.")
    elif not books.get(PRIMARY_FEEDS[0]):
        log("  note: the Hard Rock FL feed has no NHL prices; using the main Hard Rock feed instead.")
    if not books.get(REFERENCE):
        log(f"  note: no {BOOK_NAMES[REFERENCE]} prices this pull (it may be temporarily missing from the feed)")
    return df


def list_books(key=None):
    """Which bookmaker keys the API returns for NHL right now (costs 2 credits: h2h x regions us, us2)."""
    import requests
    key = key or api_key()
    if not key:
        sys.exit("No API key - set ODDS_API_KEY or create odds_api_key.txt")
    r = requests.get(API_URL, params={"apiKey": key, "regions": "us,us2", "markets": "h2h"}, timeout=20)
    if r.status_code != 200:
        sys.exit(f"The Odds API returned {r.status_code}: {r.text[:200]}")
    counts = {}
    for ev in r.json():
        for bk in ev.get("bookmakers", []):
            counts[bk["key"]] = counts.get(bk["key"], 0) + 1
    print(f"{len(r.json())} NHL games in the feed. Books with moneyline prices (games priced):")
    for k, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {k:<22} {n}" + ("   <- used" if k in BOOKS else ""))
    print(f"credits used: {r.headers.get('x-requests-last')}, remaining: {r.headers.get('x-requests-remaining')}")


# ------------------------------------------------------------------------------------------
# For refresh.py: latest prices per upcoming game, in a compact JSON-safe block for the artifact
# ------------------------------------------------------------------------------------------

def latest_block(game_ids, snaps=None):
    """{"pulled_at", "primary", "reference", "books": {key: name}, "games": {game_id: {book: {market: {...}}}}}
    using each game's most recent snapshot per book."""
    snaps = load_snapshots() if snaps is None else snaps
    if snaps is None or snaps.empty or "game_id" not in snaps:
        return None
    s = snaps.dropna(subset=["game_id", "price"]).copy()
    s["game_id"] = s["game_id"].astype("int64")
    s = s[s["game_id"].isin({int(g) for g in game_ids})]
    if s.empty:
        return None
    last = s.groupby(["game_id", "book"])["pulled_at"].transform("max")
    s = s[s["pulled_at"] == last]
    # one Hard Rock entry per game: the FL feed when it has the game, otherwise the main feed
    s["feed_rank"] = s["book"].map({b: i for i, b in enumerate(PRIMARY_FEEDS)})
    hr = s[s["feed_rank"].notna()]
    best = hr.groupby("game_id")["feed_rank"].transform("min")
    hr = hr[hr["feed_rank"] == best].assign(source=lambda d: d["book"], book=PRIMARY)
    s = pd.concat([hr, s[s["feed_rank"].isna()].assign(source=lambda d: d["book"])], ignore_index=True)
    games = {}
    for (gid, book), g in s.groupby(["game_id", "book"]):
        b = {"pulled_at": g["pulled_at"].iloc[0], "commence_time": g["commence_time"].iloc[0],
             "source": g["source"].iloc[0]}
        for mkt, m in g.groupby("market"):
            sides = {r.side: int(r.price) for r in m.itertuples()}
            if mkt == "spreads":
                hp = m.loc[m["side"] == "home", "point"]
                if hp.empty or pd.isna(hp.iloc[0]):
                    continue
                sides["point"] = float(hp.iloc[0])          # home team's line, e.g. -1.5
            elif mkt == "totals":
                pt = m["point"].dropna()
                if pt.empty:
                    continue
                sides["point"] = float(pt.iloc[0])
            b[mkt] = sides
        games.setdefault(str(int(gid)), {})[book] = b
    return {"pulled_at": s["pulled_at"].max(), "primary": PRIMARY, "reference": REFERENCE,
            "books": {k: BOOK_NAMES.get(k, k) for k in [PRIMARY] + BOOKS}, "games": games}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--show", action="store_true", help="print the latest saved snapshot, no API call")
    ap.add_argument("--list-books", action="store_true", help="list the books the API returns for NHL (2 credits)")
    args = ap.parse_args()
    if args.list_books:
        list_books()
        return
    if args.show:
        s = load_snapshots()
        if s.empty:
            print("No snapshots yet.")
            return
        s = s[s["pulled_at"] == s["pulled_at"].max()]
    else:
        try:
            s = pull_and_save()
        except RuntimeError as e:
            sys.exit(f"[stopped] {e}")
        if s is None:
            sys.exit(1)
    if s.empty:
        print("No NHL games in the feed right now.")
        return
    pd.set_option("display.width", 200)
    short = {"hardrockbet_fl": "HRFL", "hardrockbet": "HR", "pinnacle": "PIN"}
    piv = s.assign(col=s["book"].map(lambda b: short.get(b, b)) + "_" + s["market"] + "_" + s["side"])
    print(piv.pivot_table(index=["game_id", "away", "home"], columns="col", values="price", aggfunc="first",
                          dropna=False).to_string())


if __name__ == "__main__":
    main()