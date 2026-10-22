"""
streamlit_app.py - NHL game predictor

Reads only model/model_artifact.json, model/prediction_log.csv and model/edge_log.csv (written by refresh.py).
No raw data, no API calls.  Run locally:  streamlit run streamlit_app.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from nhl_model import Predictor, load_artifact

ROOT = Path(__file__).parent
ART_PATH = ROOT / "model" / "model_artifact.json"
LOG_PATH = ROOT / "model" / "prediction_log.csv"
EDGE_PATH = ROOT / "model" / "edge_log.csv"
MODEL_CODE = ROOT / "nhl_model.py"
USUAL = "Unknown - use team's usual mix"
CUSTOM = "Custom matchup"
EARLY_GP = 10
STALE_HOURS = 6
NHL_MARKETS = ["Moneyline", "Puck line", "Total"]
EDGE_BUCKETS = [(-np.inf, 0.04, "Under 4 pts"), (0.04, 0.07, "4–7 pts"), (0.07, np.inf, "7+ pts")]

st.set_page_config(page_title="NHL Game Predictor", page_icon="🏒", layout="wide")

st.markdown("""
<style>
.block-container {max-width: 1150px; padding-top: 1.4rem; padding-bottom: 3rem;}
@media (max-width: 640px) { .block-container {padding-left: .75rem; padding-right: .75rem;} }
.card {border: 1px solid rgba(128,128,128,.28); border-radius: 14px; padding: 13px 15px; margin-bottom: 12px;
       background: rgba(128,128,128,.06);}
.head {display:flex; justify-content:space-between; align-items:center; gap:8px; flex-wrap:wrap;}
.matchup {font-weight:700; font-size:1.05rem;}
.muted {opacity:.68; font-size:.82rem;}
.chip {display:inline-block; padding:2px 9px; border-radius:999px; font-size:.74rem; font-weight:600;
       border:1px solid rgba(128,128,128,.35); white-space:nowrap;}
.chip.mk {background:rgba(249,115,22,.14); color:#f97316; border-color:rgba(249,115,22,.4);}
.chip.W {background:#16a34a; color:#fff; border-color:#16a34a;}
.chip.L {background:#dc2626; color:#fff; border-color:#dc2626;}
.chip.P {background:#6b7280; color:#fff; border-color:#6b7280;}
.chip.b2b {background:rgba(234,179,8,.15); color:#ca8a04; border-color:rgba(234,179,8,.45);}
.chip.warn {background:rgba(234,179,8,.15); color:#ca8a04; border-color:rgba(234,179,8,.45);}
.chip.live {background:rgba(239,68,68,.12); color:#ef4444; border-color:rgba(239,68,68,.4);}
.score {font-size:1.3rem; font-weight:700; margin:6px 0 2px;}
.sel {font-size:1.22rem; font-weight:800; margin:6px 0 2px;}
.grid {display:grid; grid-template-columns:repeat(auto-fit, minmax(118px, 1fr)); gap:8px; margin-top:10px;}
.tile {border-radius:10px; padding:7px 10px; background:rgba(128,128,128,.08);}
.tile .l {font-size:.7rem; opacity:.65; text-transform:uppercase; letter-spacing:.03em;}
.tile .v {font-size:1.02rem; font-weight:700;}
.tile .s {font-size:.74rem; opacity:.8;}
.lean {color:#f97316; font-weight:600;}
.stars {color:#f59e0b; letter-spacing:1px;}
.gl {margin-top:6px; font-size:.82rem; line-height:1.45;}
</style>""", unsafe_allow_html=True)


# ------------------------------------------------------------------ loading
@st.cache_resource(show_spinner=False)
def get_predictor(mtime: float, code_mtime: float = 0.0) -> Predictor:
    # Both file times are in the cache key, so a new model file OR new model code rebuilds the predictor
    # (otherwise a code-only deploy keeps serving an object built from the old class).
    return Predictor(load_artifact(ART_PATH))


@st.cache_data(show_spinner=False)
def get_log(mtime: float) -> pd.DataFrame:
    return pd.read_csv(LOG_PATH)


@st.cache_data(show_spinner=False)
def get_edges(mtime: float) -> pd.DataFrame:
    return pd.read_csv(EDGE_PATH)


# ------------------------------------------------------------------ formatting / odds helpers
def pct(p):
    return "-" if p is None or pd.isna(p) else f"{100 * p:.1f}%"


def american(p):
    if p is None or pd.isna(p) or p <= 0.0005 or p >= 0.9995:
        return "-"
    us = -100 * p / (1 - p) if p >= 0.5 else 100 * (1 - p) / p
    return f"{us:+.0f}"


def fair(p):
    return "-" if p <= 0.0005 or p >= 0.9995 else f"{1 / p:.2f} ({american(p)})"


def to_decimal(us):
    if us is None or pd.isna(us) or abs(us) < 100:
        return np.nan
    return 1 + (us / 100 if us > 0 else 100 / -us)


def price_txt(x):
    return "—" if x is None or pd.isna(x) else f"{int(x):+d}"


def et_time(iso, fmt="%a %I:%M %p"):
    t = pd.to_datetime(iso, utc=True, errors="coerce")
    return "" if pd.isna(t) else t.tz_convert("America/New_York").strftime(fmt).replace(" 0", " ")


def day_label(d):
    return pd.Timestamp(d).strftime("%a %b ") + str(pd.Timestamp(d).day)


def is_true(s):
    return s.astype(str).str.lower().isin(["true", "1"])


def num(s):
    return pd.to_numeric(s, errors="coerce")


def tile(label, value, sub=""):
    return f'<div class="tile"><div class="l">{label}</div><div class="v">{value}</div><div class="s">{sub}</div></div>'


def lean(proj, line, over="Over", under="Under"):
    if line is None or pd.isna(line):
        return ""
    d = proj - line
    return f'<span class="lean">{over if d > 0 else under} {abs(d):.2f}</span>'


def stars(edge):
    """Same tiers the Track record tab grades: under 4 pts, 4-7, 7+."""
    return 3 if edge >= 0.07 else (2 if edge >= 0.04 else 1)


def quarter_kelly(p, us):
    dec = to_decimal(us)
    if pd.isna(dec) or pd.isna(p):
        return 0.0
    b = dec - 1
    return max((b * p - (1 - p)) / b, 0.0) / 4


# ------------------------------------------------------------------ market table
def build_markets(r, home, away):
    """Rows of (group, market, p_win, p_push). Rows in a group are the sides of one market, used to
    remove the bookmaker's margin when you enter prices for every side."""
    rows = [("ml", f"{home} moneyline (incl OT/SO)", r["p_home_win"], 0.0),
            ("ml", f"{away} moneyline (incl OT/SO)", 1 - r["p_home_win"], 0.0),
            ("3w", f"{home} in regulation (3-way)", r["p_home_reg"], 0.0),
            ("3w", "Tie after regulation (3-way)", r["p_tie_reg"], 0.0),
            ("3w", f"{away} in regulation (3-way)", r["p_away_reg"], 0.0)]
    s = r["spread"]
    rows += [("pl", f"{home} {s['line']:+.1f}", s["cover"], s["push"]),
             ("pl", f"{away} {-s['line']:+.1f}", s["fail"], s["push"])]
    for line, alt in r["alt_spreads"].items():
        if abs(line - s["line"]) < 1e-9:
            continue
        g = f"alt{line}"
        rows += [(g, f"{home} {line:+.1f} (alt)", alt["cover"], alt["push"]),
                 (g, f"{away} {-line:+.1f} (alt)", alt["fail"], alt["push"])]
    for g, name, key in (("tot", "Game total", "total"), ("htt", f"{home} team total", "home_total"),
                         ("att", f"{away} team total", "away_total")):
        t = r[key]
        rows += [(g, f"{name} over {t['line']:g}", t["over"], t["push"]),
                 (g, f"{name} under {t['line']:g}", t["under"], t["push"])]
    rows += [("btts", "Both teams score: yes", r["btts"], 0.0), ("btts", "Both teams score: no", 1 - r["btts"], 0.0)]
    return pd.DataFrame(rows, columns=["group", "Market", "p", "push"])


def prefill_odds(mk, book, total_line, home_spread):
    """Book prices for the moneyline, puck line and game total rows (only where the line matches)."""
    pre = [None] * len(mk)
    if not book:
        return pre
    groups = mk["group"].tolist()

    def put(group, prices):
        idx = [i for i, g in enumerate(groups) if g == group]
        for i, p in zip(idx, prices):
            pre[i] = p
    if "h2h" in book:
        put("ml", [book["h2h"].get("home"), book["h2h"].get("away")])
    sp = book.get("spreads")
    if sp and abs(sp.get("point", 99) - home_spread) < 1e-9:
        put("pl", [sp.get("home"), sp.get("away")])
    tt = book.get("totals")
    if tt and abs(tt.get("point", 99) - total_line) < 1e-9:
        put("tot", [tt.get("over"), tt.get("under")])
    return pre


def price_markets(mk, odds):
    """Adds book no-vig probability, edge, EV and quarter-Kelly for rows that have a price."""
    mk = mk.copy()
    mk["Your odds"] = pd.to_numeric(pd.Series(odds, index=mk.index), errors="coerce")
    mk["dec"] = mk["Your odds"].map(to_decimal).astype(float)
    mk["implied"] = 1 / mk["dec"]
    # de-vig when every side of a market is priced; otherwise use the raw implied probability
    full = mk.groupby("group")["dec"].transform(lambda s: s.notna().all())
    over = mk.groupby("group")["implied"].transform("sum").replace(0, np.nan)
    mk["book"] = np.where(full, mk["implied"] / over, mk["implied"])
    p_res = 1 - mk["push"]                                    # probability the bet is settled
    mk["p_cond"] = mk["p"] / p_res.where(p_res > 0, 1)        # model win prob given no push
    mk["edge"] = mk["p_cond"] - mk["book"]
    lose = (1 - mk["p"] - mk["push"]).clip(lower=0)
    mk["ev"] = mk["p"] * (mk["dec"] - 1) - lose
    b = mk["dec"] - 1
    mk["kelly"] = ((b * mk["p_cond"] - (1 - mk["p_cond"])) / b).clip(lower=0) / 4
    return mk


# ------------------------------------------------------------------ load
if not ART_PATH.exists():
    st.error("model/model_artifact.json not found. Run `python refresh.py` locally, then commit and push the model folder.")
    st.stop()
ART_MT = ART_PATH.stat().st_mtime
CODE_MT = MODEL_CODE.stat().st_mtime if MODEL_CODE.exists() else 0.0
try:
    pred = get_predictor(ART_MT, CODE_MT)
except ValueError as e:
    st.error(str(e))
    st.stop()
meta = pred.meta
gp = meta.get("season_gp", {})
odds = pred.odds
PRIMARY = odds.get("primary", "hardrockbet_fl")
REFERENCE = odds.get("reference", "pinnacle")
BOOK = (odds.get("books") or {}).get(PRIMARY, "Hard Rock Bet")
REF_BOOK = (odds.get("books") or {}).get(REFERENCE, "Pinnacle")


@st.cache_data(show_spinner=False)
def slate_cards(art_mt: float, code_mt: float, primary: str) -> list[dict]:
    """Every slate game priced with each team's usual goalie mix, at the book's total and puck line when posted."""
    p_ = get_predictor(art_mt, code_mt)
    out = []
    for g in p_.slate:
        hb, ab = bool(g.get("home_b2b")), bool(g.get("away_b2b"))
        book = p_.game_odds(g["game_id"]).get(primary, {}) or {}
        kw = {}
        if (book.get("totals") or {}).get("point") is not None:
            kw["total_line"] = float(book["totals"]["point"])
        if (book.get("spreads") or {}).get("point") is not None:
            kw["home_spread"] = float(book["spreads"]["point"])
        try:
            r = p_.predict_game(g["home"], g["away"], p_.goalie_spec(g["home"], None, b2b=hb),
                                p_.goalie_spec(g["away"], None, b2b=ab), home_b2b=hb, away_b2b=ab, **kw)
        except ValueError:
            continue
        out.append({"g": g, "r": r, "book": book})
    return out


@st.cache_data(show_spinner=False)
def board_rows(art_mt: float, code_mt: float, primary: str) -> list[dict]:
    """Every priced side on the slate vs the book, with each team's usual goalie mix."""
    p_ = get_predictor(art_mt, code_mt)
    rows = []
    for g in p_.slate:
        go = p_.game_odds(g["game_id"])
        if not go.get(primary):
            continue
        hb, ab = bool(g.get("home_b2b")), bool(g.get("away_b2b"))
        try:
            er = p_.edge_board(g["home"], g["away"], go, p_.goalie_spec(g["home"], None, b2b=hb),
                               p_.goalie_spec(g["away"], None, b2b=ab), home_b2b=hb, away_b2b=ab)
        except ValueError:
            continue
        start = go[primary].get("commence_time") or g.get("start_utc")
        for e in er:
            rows.append({"date": g["date"], "start": start, "home": g["home"], "away": g["away"],
                         "game": f"{g['away']} @ {g['home']}", "market": e["market"], "label": e["label"],
                         "price": e["price"], "model": e["model_cond"], "fair": e["fair"], "edge": e["edge"],
                         "ev": e["ev"], "ref_fair": e["ref_fair"], "hold": e["hold"],
                         "src": go[primary].get("source", primary)})
    return rows


st.title("🏒 NHL Game Predictor")
st.caption(f"Model trained through **{meta['trained_through']}** · {meta['n_games']:,} games · "
           f"goalie data **{'ON' if pred.goalies_on else 'OFF'}**"
           + (f" · {BOOK} odds **{et_time(odds['pulled_at'])} ET**" if odds.get("pulled_at") else ""))

tab_board, tab_slate, tab_game, tab_track, tab_model = st.tabs(
    ["🎯 Edges board", "📅 Today's Slate", "🧮 Game Markets", "📈 Track Record", "📊 Model"])

# ================================================================== Edges board
with tab_board:
    if not odds.get("games"):
        st.info("No sportsbook odds in this model file yet. Set an Odds API key (see nhl_odds.py) and run "
                "`python refresh.py --skip-fetch --push`.")
    else:
        age_h = (pd.Timestamp.now(tz="UTC") - pd.to_datetime(odds["pulled_at"], utc=True)).total_seconds() / 3600
        if age_h > STALE_HOURS:
            st.warning(f"These prices are {age_h:.0f} hours old. Lines move; check {BOOK} before acting.")
        rows = board_rows(ART_MT, CODE_MT, PRIMARY)
        if not rows:
            st.info(f"No {BOOK} prices for the upcoming games yet. Books often post NHL lines the day before.")
        else:
            bd = pd.DataFrame(rows)
            now = pd.Timestamp.now(tz="UTC")
            bd["_start"] = [pd.to_datetime(s, utc=True) if s else pd.Timestamp(d, tz="America/New_York").tz_convert("UTC")
                            for s, d in zip(bd["start"], bd["date"])]
            bd["started"] = bd["_start"] <= now
            days = sorted(bd["date"].unique())
            upcoming = [d for d in days if not bd.loc[bd["date"] == d, "started"].all()]
            default_day = upcoming[0] if upcoming else days[-1]
            day = st.segmented_control("Day", days + ["All"], default=default_day, label_visibility="collapsed",
                                       format_func=lambda d: d if d == "All" else day_label(d), key="eb_day")
            c1, c2 = st.columns([3, 2])
            mkts = c1.pills("Markets", NHL_MARKETS, selection_mode="multi", default=NHL_MARKETS,
                            label_visibility="collapsed", key="eb_mk")
            thr = c2.select_slider("Minimum edge", [0, 2, 3, 4, 5, 7], 3, key="eb_thr",
                                   format_func=lambda x: f"{x}+ pts" if x else "Every side")
            day = day or default_day
            if day != "All":
                bd = bd[bd["date"] == day]
            show = bd[bd["market"].isin(mkts or NHL_MARKETS)]
            if thr:
                show = show[show["edge"] >= thr / 100]
            show = show.sort_values(["_start", "edge"], ascending=[True, False])
            st.caption(f"{len(show)} of {len(bd)} priced sides · {bd['game'].nunique()} games · average {BOOK} margin "
                       f"{100 * bd['hold'].mean():.1f}% · edge = model chance (pushes excluded) minus {BOOK}'s no-vig "
                       "chance · uses each team's usual goalie mix")
            if show.empty:
                st.info("Nothing clears the threshold right now. That's normal on many nights - the market is "
                        "usually close to the model.")
            else:
                cols = st.columns(2)
                for i, (_, e) in enumerate(show.iterrows()):
                    when = ('<span class="chip warn">started</span>' if e["started"]
                            else f'<span class="chip">{et_time(e["_start"])}</span>')
                    k = quarter_kelly(e["model"], e["price"])
                    has_ref = not pd.isna(e["ref_fair"])
                    ref_t = tile(REF_BOOK, pct(e["ref_fair"]) if has_ref else "—", "sharp fair" if has_ref else "not posted")
                    warn = ""
                    if has_ref and e["edge"] > 0 and e["ref_fair"] <= e["fair"]:
                        warn = (f'<div style="margin-top:6px"><span class="chip warn">⚠ {REF_BOOK} sides with '
                                f'{BOOK}, not the model</span></div>')
                    if e["src"] == "hardrockbet":
                        warn += ('<div style="margin-top:6px"><span class="chip">main Hard Rock feed - confirm '
                                 'the FL price</span></div>')
                    model_t = tile("Model / book", pct(e["model"]), f"book {pct(e['fair'])}")
                    edge_t = tile("Edge", f"{100 * e['edge']:+.1f} pts", f"EV {100 * e['ev']:+.1f}%")
                    kelly_t = tile("¼ Kelly", f"{100 * k:.1f}%" if k > 0 else "no bet", "of bankroll")
                    star_txt = "★" * stars(e["edge"]) if e["edge"] > 0 else ""
                    cols[i % 2].markdown(f"""
                    <div class="card">
                      <div class="head"><span class="chip mk">{e['market']}</span>{when}</div>
                      <div class="sel">{e['label']} <span class="muted">({price_txt(e['price'])})</span></div>
                      <div class="muted">{e['game']} · <span class="stars">{star_txt}</span></div>
                      <div class="grid">{model_t}{edge_t}{kelly_t}{ref_t}</div>{warn}
                    </div>""", unsafe_allow_html=True)
            st.caption(f"{REF_BOOK} is the sharp market's view at the same line: if it sides with {BOOK} rather than "
                       "the model, be sceptical of the edge. Set confirmed starters on the Game Markets tab to re-price a "
                       "game. The Track Record tab shows whether flagged edges have held up.")

# ================================================================== Slate
with tab_slate:
    cards = slate_cards(ART_MT, CODE_MT, PRIMARY)
    if not cards:
        st.info("No upcoming games in this model file. `refresh.py` pulls the next few days of the schedule.")
    else:
        dates = sorted({c["g"]["date"] for c in cards})
        day = st.segmented_control("Day", dates, default=dates[0], label_visibility="collapsed",
                                   format_func=day_label, key="sl_day")
        rows = [c for c in cards if c["g"]["date"] == (day or dates[0])]
        st.caption(f"{len(rows)} games · projections are final scores including OT/SO, with each team's usual goalie "
                   f"mix · lines: {BOOK}")
        cols = st.columns(2)
        for i, c in enumerate(rows):
            g, r, bk = c["g"], c["r"], c["book"]
            h, a = g["home"], g["away"]
            fav, pf = (h, r["p_home_win"]) if r["p_home_win"] >= 0.5 else (a, 1 - r["p_home_win"])
            b2b = "".join(f' <span class="chip b2b">{t} B2B</span>'
                          for t, on in ((a, g.get("away_b2b")), (h, g.get("home_b2b"))) if on)
            early = [t for t in (h, a) if gp.get(t, 99) < EARLY_GP]
            ml, sp, tt = bk.get("h2h") or {}, bk.get("spreads") or {}, bk.get("totals") or {}
            start = (bk.get("commence_time") if bk else None) or g.get("start_utc")

            # likeliest starter per team, from the usual-mix weights
            gl = []
            for t in (a, h):
                opts = [x for x in (r.get("goalie_info") or []) if x["team"] == t]
                if opts:
                    top = max(opts, key=lambda x: x["prob"])
                    gl.append(f"{t}: {top['name']} ({top['prob']:.0%})")
            goalie_html = f'<div class="gl">🥅 {" &nbsp;|&nbsp; ".join(gl)}</div>' if gl else ""

            s, tot = r["spread"], r["total"]
            win_t = tile("Win prob", f"{fav} {pf:.0%}", f"Fair {american(pf)}"
                         + (f" · ML {price_txt(ml.get('away'))}/{price_txt(ml.get('home'))}" if ml else ""))
            pl_t = tile("Puck line", f"{h} {s['line']:+g}", f"covers {pct(s['cover'])}"
                        + (f" · {price_txt(sp.get('home'))}" if sp else ""))
            tot_t = tile("Total", f"{r['exp_total']:.2f}",
                         (f"Line {tot['line']:g} {lean(r['exp_total'], tot['line'])} · O {pct(tot['over'])}" if tt
                          else f"no line · O {tot['line']:g} {pct(tot['over'])}"))
            a_t = tile(f"{a} goals", f"{r['exp_away']:.2f}")
            h_t = tile(f"{h} goals", f"{r['exp_home']:.2f}")
            per = r.get("periods") or []
            p1_t = tile("1st period", f"{per[0]['exp_total']:.2f}", f"O 1.5 {pct(per[0]['over_1_5'])}") if per else ""
            cols[i % 2].markdown(f"""
            <div class="card">
              <div class="head"><span class="matchup">{a} @ {h}{b2b}</span><span class="chip">{et_time(start)}</span></div>
              <div class="score">{a} {r['exp_away']:.2f} – {r['exp_home']:.2f} {h}</div>
              <div class="muted">OT/SO chance {pct(r['p_tie_reg'])}{(' · early season: ' + ', '.join(early)) if early else ''}</div>
              {goalie_html}
              <div class="grid">{win_t}{pl_t}{tot_t}{a_t}{h_t}{p1_t}</div>
            </div>""", unsafe_allow_html=True)
        st.caption("Use Game Markets to set confirmed starters and price any game at your own book's numbers.")


# ================================================================== Game markets
def render_game_tab():
    teams = pred.active_teams
    labels = {CUSTOM: None}
    for g in pred.slate:
        tag = " · ".join(f"{t} on B2B" for t, on in ((g["home"], g.get("home_b2b")), (g["away"], g.get("away_b2b"))) if on)
        labels[f"{g['date'][5:]}  {g['away']} @ {g['home']}" + (f"  ({tag})" if tag else "")] = g
    pick = st.selectbox("Game", list(labels), index=1 if len(labels) > 1 else 0, key="gm_pick",
                        help="Upcoming games come from the schedule pulled by refresh.py. Pick 'Custom matchup' for any pairing.")
    sg = labels[pick]
    k = str(sg["game_id"]) if sg else "custom"     # widget keys per game, so defaults reset when you switch

    c1, c2, c3, c4 = st.columns(4)
    def_away = sg["away"] if sg else ("VGK" if "VGK" in teams else teams[0])
    def_home = sg["home"] if sg else ("FLA" if "FLA" in teams else teams[1])
    away = c1.selectbox("Away team", teams, index=teams.index(def_away) if def_away in teams else 0, key=f"away_{k}")
    home = c2.selectbox("Home team", teams, index=teams.index(def_home) if def_home in teams else 1, key=f"home_{k}")
    away_b2b = c3.checkbox(f"{away} played yesterday", value=bool(sg and sg.get("away_b2b")), key=f"ab2b_{k}_{away}")
    home_b2b = c4.checkbox(f"{home} played yesterday", value=bool(sg and sg.get("home_b2b")), key=f"hb2b_{k}_{home}")
    if home == away:
        st.warning("Pick two different teams.")
        return

    # book prices for this game, only when the teams still match the scheduled game
    book = {}
    if sg and home == sg["home"] and away == sg["away"]:
        book = pred.game_odds(sg["game_id"]).get(PRIMARY, {}) or {}

    early = [t for t in (home, away) if gp.get(t, 99) < EARLY_GP]
    if early:
        st.info(f"Early season: {', '.join(f'{t} ({gp.get(t, 0)} GP)' for t in early)}. Ratings still lean on last "
                "season, pulled toward league average. Roster changes since then aren't reflected yet.")

    # ---- goalies ----
    home_gid = away_gid = None
    home_conf = away_conf = 1.0
    if pred.goalies_on:
        with st.expander("🥅 Goalies", expanded=True):
            gc1, gc2 = st.columns(2)
            for col, team, side in ((gc1, away, "away"), (gc2, home, "home")):
                gl = {USUAL: None}
                for o in pred.goalie_options(team):
                    gl[f"{o['name']}  ·  {o['starts']} starts  ·  {o['rating_gpg']:+.2f} g/g"] = o["player_id"]
                with col:
                    gid = gl[st.selectbox(f"{team} goalie", list(gl), key=f"g_{side}_{team}_{k}")]
                    conf = 1.0
                    if gid is not None:
                        conf = st.slider("Chance he starts", 50, 100, 100, 5, key=f"c_{side}_{team}_{k}",
                                         help="Below 100% the remainder goes to the team's other usual goalies.") / 100
                if side == "away":
                    away_gid, away_conf = gid, conf
                else:
                    home_gid, home_conf = gid, conf
            st.caption("Pick a starter once he's confirmed (typically morning of the game). 'Unknown' uses each team's "
                       "recent starter mix, shifted toward the backup on the second night of a back-to-back.")

    # ---- lines ----
    with st.expander("Lines", expanded=True):
        l1, l2, l3, l4 = st.columns(4)
        def_total = float((book.get("totals") or {}).get("point", 6.5))
        def_spread = float((book.get("spreads") or {}).get("point", -1.5))
        total_line = l1.number_input("Game total", 3.0, 10.0, def_total, 0.5, key=f"tl_{k}_{def_total}")
        home_spread = l2.number_input(f"{home} spread", -3.5, 3.5, def_spread, 0.5, help="-1.5 = home lays 1.5 goals",
                                      key=f"hs_{k}_{home}_{def_spread}")
        home_tt = l3.number_input(f"{home} team total", 0.5, 6.5, 2.5, 0.5, key=f"htt_{home}")
        away_tt = l4.number_input(f"{away} team total", 0.5, 6.5, 2.5, 0.5, key=f"att_{away}")
        if book:
            st.caption(f"Game total and puck line default to {BOOK}'s current lines.")

    hs = pred.goalie_spec(home, home_gid, home_conf, b2b=home_b2b)
    as_ = pred.goalie_spec(away, away_gid, away_conf, b2b=away_b2b)
    r = pred.predict_game(home, away, hs, as_, total_line=total_line, home_spread=home_spread,
                          home_team_line=home_tt, away_team_line=away_tt, home_b2b=home_b2b, away_b2b=away_b2b)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"{away} @ {home}", f"{r['exp_away']:.2f} – {r['exp_home']:.2f}", help="Expected final score incl. OT/SO winner")
    m2.metric("Expected total", f"{r['exp_total']:.2f}")
    m3.metric(f"{home} win probability", pct(r["p_home_win"]), help="Includes overtime / shootout")
    m4.metric("Goes to OT/SO", pct(r["p_tie_reg"]))

    # ---- markets + value check ----
    mk = build_markets(r, home, away)
    pre = prefill_odds(mk, book, total_line, home_spread)
    if any(p is not None for p in pre):
        st.caption(f"**Your odds** is prefilled with {BOOK} prices as of {et_time(odds.get('pulled_at'))} ET "
                   "(moneyline, puck line and game total at the book's line). Overwrite with the price you see. "
                   "Price both sides of a market and the book's margin is removed before comparing.")
    else:
        st.caption("Type your sportsbook's American odds (e.g. -115, +140) in **Your odds**. Price both sides of a market "
                   "and the book's margin is removed before comparing. On whole-number lines the push chance is shown "
                   "separately and fair odds exclude it, matching how books grade a push as a refund.")
    edit_key = f"odds_{home}_{away}_{total_line}_{home_spread}_{home_tt}_{away_tt}_{odds.get('pulled_at', '')}"
    shown = pd.DataFrame({"Market": mk["Market"], "Model": mk["p"].map(pct),
                          "Push": mk["push"].map(lambda x: pct(x) if x > 1e-9 else ""),
                          "Fair odds": (mk["p"] / (1 - mk["push"])).map(lambda p: fair(p) if p > 0 else "-"),
                          "Your odds": pd.array(pre, dtype="Int64")})
    edited = st.data_editor(shown, hide_index=True, width="stretch", key=edit_key,
                            disabled=["Market", "Model", "Push", "Fair odds"],
                            column_config={"Your odds": st.column_config.NumberColumn(
                                "Your odds", help="American odds, e.g. -110 or +125", step=1, format="%+d")})
    priced = price_markets(mk, edited["Your odds"].values)
    priced = priced[priced["dec"].notna()]
    if len(priced):
        st.markdown("**Value check**")
        st.dataframe(pd.DataFrame({
            "Market": priced["Market"], "Book (no-vig)": priced["book"].map(pct),
            "Model": priced["p_cond"].map(pct), "Edge": priced["edge"].map(lambda x: f"{100 * x:+.1f} pts"),
            "EV per $100": priced["ev"].map(lambda x: f"{100 * x:+.2f}"),
            "¼ Kelly": priced["kelly"].map(lambda x: f"{100 * x:.1f}%" if x > 0 else "no bet")}),
            hide_index=True, width="stretch")
        st.caption("Small edges (under ~2-3 points) are within this model's error. Stakes assume the model is right; "
                   "quarter Kelly is already a cautious fraction.")

    mp = pd.DataFrame(r["margin_pmf"])
    mp = mp[mp["margin"] != 0]
    st.markdown(f"**Final margin** ({home} minus {away})")
    st.bar_chart(mp.set_index("margin")["p"], x_label=f"{home} margin", y_label="Probability", height=230)

    # ---- periods ----
    with st.expander("⏱ Period breakdown"):
        st.dataframe(pd.DataFrame([{"Period": p["period"], f"{home} exp. goals": round(p["exp_home"], 2),
                                    f"{away} exp. goals": round(p["exp_away"], 2), "Exp. total": round(p["exp_total"], 2),
                                    "Over 0.5": pct(p["over_0_5"]), "Over 1.5": pct(p["over_1_5"]),
                                    "Both score": pct(p["btts"])} for p in r["periods"]]),
                     hide_index=True, width="stretch")
        K = meta.get("period_share_K")
        st.caption("Expected regulation goals split by period. " + (
            "Team-by-team differences in period scoring weren't distinguishable from noise, so every team uses the "
            "league split." if K and K > 1e5 else "Each team's period split is shrunk toward the league split."))

    # ---- goalie impact ----
    if pred.goalies_on:
        with st.expander("🥅 Goalie impact"):
            if r["goalie_info"]:
                st.dataframe(pd.DataFrame([{"Team": g["team"], "Goalie": g["name"], "Weight": pct(g["prob"]),
                                            "Rating (g/g vs avg)": f"{g['rating_gpg']:+.2f}", "Basis": g["basis"]}
                                           for g in r["goalie_info"]]), hide_index=True, width="stretch")
            if home_gid is None and away_gid is None:
                st.caption("Select a goalie above to see how much the pick moves the prediction.")
            else:
                base = pred.predict_game(home, away, None, None, total_line=total_line, home_spread=home_spread,
                                         home_b2b=home_b2b, away_b2b=away_b2b)
                st.dataframe(pd.DataFrame([
                    {"": f"{home} win probability", "Usual goalies": pct(base["p_home_win"]), "Selected": pct(r["p_home_win"]),
                     "Change": f"{100 * (r['p_home_win'] - base['p_home_win']):+.1f} pts"},
                    {"": "Expected total goals", "Usual goalies": f"{base['exp_total']:.2f}", "Selected": f"{r['exp_total']:.2f}",
                     "Change": f"{r['exp_total'] - base['exp_total']:+.2f}"},
                    {"": f"Total over {total_line:g}", "Usual goalies": pct(base["total"]["over"]),
                     "Selected": pct(r["total"]["over"]),
                     "Change": f"{100 * (r['total']['over'] - base['total']['over']):+.1f} pts"}]),
                    hide_index=True, width="stretch")
            beta = meta.get("goalie_beta")
            gxb = meta.get("goalie_xg_blend_used") or 0
            basis = ("save % only" if not gxb else
                     f"{100 * (1 - gxb):.0f}% save % / {100 * gxb:.0f}% goals saved above expected")
            if beta is not None:
                st.caption(f"Fitted goalie coefficient: {beta:+.2f} (near -1 = ratings trusted at face value; near 0 = "
                           f"goalies add little). Ratings use {basis}, shrunk toward league average.")


with tab_game:
    render_game_tab()


# ================================================================== Track record
def summary(df):
    """Same shape as the NBA summary, plus closing line value."""
    w, l, p = (df["result"] == "win").sum(), (df["result"] == "loss").sum(), (df["result"] == "push").sum()
    u = df["profit"].sum()
    staked = w + l
    clv = df["clv"].dropna() if "clv" in df else pd.Series(dtype=float)
    return {"Picks": len(df), "Record": f"{w}-{l}-{p}", "Win %": w / staked if staked else np.nan,
            "Units": u, "ROI": u / staked if staked else np.nan,
            "Beat close": (clv > 0).mean() if len(clv) else np.nan, "Avg CLV": clv.mean() if len(clv) else np.nan}


REC_CFG = {"Win %": st.column_config.NumberColumn(format="percent"),
           "Units": st.column_config.NumberColumn(format="%+.2f"),
           "ROI": st.column_config.NumberColumn(format="percent"),
           "Beat close": st.column_config.NumberColumn(format="percent"),
           "Avg CLV": st.column_config.NumberColumn(format="percent", help="Closing no-vig prob. minus your break-even prob.")}

with tab_track:
    # ---------------------------------------------------------------- flagged edges
    st.markdown(f"#### Flagged edges at {BOOK}")
    el = get_edges(EDGE_PATH.stat().st_mtime) if EDGE_PATH.exists() else pd.DataFrame()
    fl = el[is_true(el["flag"])] if "flag" in el else el.iloc[0:0]
    done = fl[fl["result"].notna()].copy() if "result" in fl else fl.iloc[0:0]
    if done.empty:
        st.info(f"{len(fl)} edges flagged so far; none settled yet. Results fill in on the first refresh after each game."
                if len(fl) else "No flagged edges yet. They're logged by refresh.py when the slate has posted odds.")
    else:
        done["date"] = pd.to_datetime(done["date"])
        for c in ("profit", "clv", "open_edge", "open_price", "close_price"):
            if c in done:
                done[c] = num(done[c])
        markets = [m for m in NHL_MARKETS if (done["market"] == m).any()] + \
            sorted(set(done["market"]) - set(NHL_MARKETS))

        f1, f2 = st.columns([2, 3])
        window = f1.segmented_control("Period", ["7D", "30D", "Season", "All"], default="All",
                                      label_visibility="collapsed", key="rec_win")
        mks = f2.pills("Market filter", markets, selection_mode="multi", default=markets,
                       label_visibility="collapsed", key="rec_mk")
        last = done["date"].max()
        if window in ("7D", "30D"):
            done = done[done["date"] > last - pd.Timedelta(days=int(window[:-1]))]
        elif window == "Season":                       # NHL season opens in October
            done = done[done["date"] >= pd.Timestamp(last.year if last.month >= 9 else last.year - 1, 9, 1)]
        done = done[done["market"].isin(mks or markets)]

        if done.empty:
            st.info("No settled flags in this window.")
        else:
            s = summary(done)
            t1, t2, t3, t4, t5 = st.columns(5)
            t1.metric("Record", s["Record"])
            t2.metric("Win %", pct(s["Win %"]), help="Break-even depends on the price; ROI is the better read")
            t3.metric("Units (1u flat)", f"{s['Units']:+.2f}")
            t4.metric("ROI", pct(s["ROI"]), help="Units per unit staked (pushes refunded)")
            t5.metric("Beat the close", pct(s["Beat close"]),
                      delta=f"{100 * s['Avg CLV']:+.2f} pts avg" if not pd.isna(s["Avg CLV"]) else None,
                      help=f"Closing line value: closing no-vig probability ({REF_BOOK} when available, else {BOOK}) "
                           "minus the break-even probability of the price taken. Positive and consistent is the "
                           "earliest sign of a real edge - it needs far fewer games than win/loss.")

            st.area_chart(done.groupby("date")["profit"].sum().cumsum().rename("Cumulative units"), height=210)

            st.dataframe(pd.DataFrame([{"Market": m, **summary(done[done["market"] == m])} for m in markets
                                       if (done["market"] == m).any()]),
                         hide_index=True, width="stretch", column_config=REC_CFG)
            if "open_edge" in done:   # NHL stand-in for the NBA star tiers: how big the edge was when flagged
                tiers = []
                for lo, hi, name in EDGE_BUCKETS:
                    sub = done[(done["open_edge"] >= lo) & (done["open_edge"] < hi)]
                    if len(sub):
                        tiers.append({"Edge size": name, **summary(sub)})
                if tiers:
                    st.dataframe(pd.DataFrame(tiers), hide_index=True, width="stretch", column_config=REC_CFG)

            allset = el[el["result"].notna()] if "result" in el else el.iloc[0:0]
            if len(allset):
                ap = num(allset["profit"])
                st.caption(f"For comparison, every logged side (flagged or not): {len(allset):,} settled, "
                           f"{100 * ap.sum() / max(int((allset['result'] != 'push').sum()), 1):+.1f}% ROI - roughly "
                           f"what betting without the model costs at {BOOK}'s margin.")

            with st.expander("Pick history"):
                hist = done.sort_values("date", ascending=False)
                st.dataframe(pd.DataFrame({
                    "Date": hist["date"].dt.strftime("%b %-d"), "Game": hist["away"] + " @ " + hist["home"],
                    "Pick": hist["open_label"], "Price": hist["open_price"].map(price_txt),
                    "Edge": hist["open_edge"].map(lambda x: f"{100 * x:+.1f}"),
                    "Close": hist["close_price"].map(price_txt),
                    "CLV": hist["clv"].map(lambda x: "—" if pd.isna(x) else f"{100 * x:+.1f}"),
                    "Final": hist["final"], "W/L": hist["result"].map({"win": "W", "loss": "L", "push": "P"}),
                    "Units": hist["profit"].map(lambda x: f"{x:+.2f}")}), hide_index=True, width="stretch")
            if len(done) < 200:
                st.caption(f"Only {len(done)} settled flags: win/loss and units are still mostly luck at this sample "
                           "size. Closing line value is the more reliable early read.")

    # ---------------------------------------------------------------- every game projected
    st.markdown("#### Every game the model projected")
    lg = get_log(LOG_PATH.stat().st_mtime) if LOG_PATH.exists() else pd.DataFrame()
    settled = lg[lg["home_score"].notna()].copy() if "home_score" in lg else pd.DataFrame()
    if settled.empty:
        st.info(f"{len(lg)} games logged so far; none settled yet." if not lg.empty
                else "No prediction log yet. It starts the first time refresh.py runs with games on the schedule.")
    else:
        y = (settled["home_score"] > settled["away_score"]).astype(int)
        p = settled["p_home_win"]
        reg = np.select([settled["home_reg"] > settled["away_reg"], settled["home_reg"] == settled["away_reg"]], [0, 1], 2)
        P3 = settled[["p_home_reg", "p_tie_reg", "p_away_reg"]].values
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Games settled", f"{len(settled):,}")
        a2.metric("Model's winner right", pct(((p >= 0.5) == (y == 1)).mean()), help="Including OT/SO")
        a3.metric("Moneyline Brier", f"{((p - y) ** 2).mean():.4f}",
                  delta=f"{((p - y) ** 2).mean() - ((y.mean() - y) ** 2).mean():+.4f} vs constant", delta_color="inverse",
                  help="Lower is better. 0.25 = always saying 50%.")
        a4.metric("3-way log-loss", f"{-np.log(P3[np.arange(len(settled)), reg]).mean():.3f}",
                  help="Regulation home / tie / away. Lower is better; ~1.04-1.05 is what the backtest achieved.")

        # Total vs market - only if refresh.py logs these columns
        if {"exp_total", "total_line"} <= set(settled.columns) and settled["total_line"].notna().any():
            tl = settled["total_line"].notna()
            act_tot = settled["home_score"] + settled["away_score"]
            res = act_tot[tl] - settled.loc[tl, "total_line"]
            won = np.where(settled.loc[tl, "exp_total"] > settled.loc[tl, "total_line"], res > 0, res < 0)
            st.dataframe(pd.DataFrame([{
                "Check": "Total vs market", "Games": int(tl.sum()),
                "Model MAE": (settled.loc[tl, "exp_total"] - act_tot[tl]).abs().mean(),
                "Market MAE": res.abs().mean(),
                "Model side record": f"{int(won[res != 0].sum())}-{int((~won)[res != 0].sum())}-{int((res == 0).sum())}"}]),
                hide_index=True, width="stretch", column_config={
                    "Model MAE": st.column_config.NumberColumn(format="%.2f"),
                    "Market MAE": st.column_config.NumberColumn(format="%.2f")})

        if len(settled) >= 40:
            cal = settled.assign(y=y, bucket=pd.cut(p, [0, .40, .45, .50, .55, .60, 1])).groupby("bucket", observed=True).agg(
                Predicted=("p_home_win", "mean"), Actual=("y", "mean"), Games=("y", "size")).reset_index()
            cal["bucket"] = cal["bucket"].astype(str)
            st.markdown("**Calibration (home win probability)**")
            st.dataframe(cal.round(3), hide_index=True, width="stretch")

        with st.expander("Recent games"):
            recent = settled.tail(25).iloc[::-1]
            pick_home = recent["p_home_win"] >= 0.5
            home_won = recent["home_score"] > recent["away_score"]
            st.dataframe(pd.DataFrame({
                "Date": pd.to_datetime(recent["date"]).dt.strftime("%b %-d"), "Game": recent["away"] + " @ " + recent["home"],
                "Model pick": np.where(pick_home, recent["home"], recent["away"]),
                "Pick win %": np.maximum(recent["p_home_win"], 1 - recent["p_home_win"]).map(pct),
                "Result": np.where(pick_home == home_won, "✓", "✗"),
                "Score": recent["away_score"].astype(int).astype(str) + "-" + recent["home_score"].astype(int).astype(str),
                "OT/SO": np.where(recent["home_reg"] == recent["away_reg"], "yes", "")}), hide_index=True, width="stretch")
        st.caption("Logged by refresh.py before each game with each team's usual goalie mix, so this is honest "
                   "out-of-sample performance.")

    # ---------------------------------------------------------------- backtest
    bt = meta.get("backtest")
    if bt:
        st.markdown(f"#### Walk-forward backtest ({bt['n_games']:,} games, seasons {', '.join(bt['seasons'])})")
        st.dataframe(pd.DataFrame([
            {"Check": "Moneyline Brier", "Model": f"{bt['win_brier']:.4f}", "Reference": f"{bt['const_brier']:.4f} (constant home rate)"},
            {"Check": "3-way regulation log-loss", "Model": f"{bt['reg3_logloss']:.4f}", "Reference": ""},
            {"Check": "OT/SO frequency", "Model": pct(bt["pred_tie"]), "Reference": f"{pct(bt['actual_tie'])} actual"},
            {"Check": "Over 5.5 rate", "Model": pct(bt["pred_o55"]), "Reference": f"{pct(bt['actual_o55'])} actual"},
            {"Check": "Over 6.5 rate", "Model": pct(bt["pred_o65"]), "Reference": f"{pct(bt['actual_o65'])} actual"},
            {"Check": "Home -1.5 covers", "Model": pct(bt["pred_cover"]), "Reference": f"{pct(bt['actual_cover'])} actual"},
        ]), hide_index=True, width="stretch")
        if bt.get("calibration"):
            st.dataframe(pd.DataFrame(bt["calibration"]).rename(columns={"bucket": "Home win prob.", "predicted": "Predicted",
                         "actual": "Actual", "n": "Games"}).round(3), hide_index=True, width="stretch")

# ================================================================== Model
with tab_model:
    st.markdown("**Team ratings** (log scale; higher defense = allows more)")
    st.dataframe(pred.team_ratings().round(3), width="stretch", height=420)
    leaders = pred.goalie_leaders()
    if leaders is not None:
        st.markdown("**Goalie ratings** (min 20 starts, active in the last ~13 months)")
        show = leaders.copy()
        show["sv_pct"] = show["sv_pct"].map(lambda x: f"{x:.3f}")
        show["rating_gpg"] = show["rating_gpg"].round(2)
        if "gsax" in show:
            show["gsax"] = show["gsax"].round(1)
        st.dataframe(show, width="stretch", height=420)
    else:
        st.caption("Goalie ratings appear once goalie data is included in the model.")
    b2b = meta.get("b2b_coef")
    st.caption(
        f"Recency half-life {meta['half_life_days']:g} days · ridge α {meta['alpha_l2']:g} "
        f"(effective {meta.get('alpha_eff', meta['alpha_l2']):.3f}) · past-season weight {meta.get('carryover', 1):g} · "
        f"shots blend {meta.get('shot_blend', 0):.0%} · xG blend {meta.get('xg_blend_used', 0):.0%}"
        + (f" · teams on a back-to-back score {100 * (np.exp(b2b) - 1):+.1f}%" if b2b is not None else "")
        + f" · fitted {meta['fitted_at']}. The model predicts regulation goals; ties are settled by one OT/SO goal and the "
        "score grid is calibrated to real overtime and empty-net frequencies. For information and entertainment only - "
        "not betting advice.")