"""
streamlit_app.py - NHL game predictor

Reads only model/model_artifact.json and model/prediction_log.csv (written by refresh.py).
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
USUAL = "Unknown - use team's usual mix"
CUSTOM = "Custom matchup"
EARLY_GP = 10

st.set_page_config(page_title="NHL Game Predictor", page_icon="🏒", layout="wide")


@st.cache_resource(show_spinner=False)
def get_predictor(mtime: float) -> Predictor:      # mtime in the key => reload if the file changes
    return Predictor(load_artifact(ART_PATH))


@st.cache_data(show_spinner=False)
def get_log(mtime: float) -> pd.DataFrame:
    return pd.read_csv(LOG_PATH)


# ------------------------------------------------------------------ odds helpers
def pct(p):
    return "-" if p is None or pd.isna(p) else f"{100 * p:.1f}%"


def american(p):
    if p <= 0.0005 or p >= 0.9995:
        return "-"
    us = -100 * p / (1 - p) if p >= 0.5 else 100 * (1 - p) / p
    return f"{us:+.0f}"


def fair(p):
    return "-" if p <= 0.0005 or p >= 0.9995 else f"{1 / p:.2f} ({american(p)})"


def to_decimal(us):
    if us is None or pd.isna(us) or abs(us) < 100:
        return np.nan
    return 1 + (us / 100 if us > 0 else 100 / -us)


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
try:
    pred = get_predictor(ART_PATH.stat().st_mtime)
except ValueError as e:
    st.error(str(e))
    st.stop()
meta = pred.meta
gp = meta.get("season_gp", {})

st.title("🏒 NHL Game Predictor")
st.caption(f"Model trained through **{meta['trained_through']}** · {meta['n_games']:,} games · "
           f"goalie data **{'ON' if pred.goalies_on else 'OFF'}**")

# ------------------------------------------------------------------ game picker
teams = pred.active_teams
slate = pred.slate
labels = {CUSTOM: None}
for g in slate:
    tag = " · ".join(t for t, on in ((f"{g['home']} on B2B", g.get("home_b2b")), (f"{g['away']} on B2B", g.get("away_b2b"))) if on)
    labels[f"{g['date']}  {g['away']} @ {g['home']}" + (f"  ({tag})" if tag else "")] = g
pick = st.selectbox("Game", list(labels), index=1 if len(labels) > 1 else 0,
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
    st.stop()

early = [t for t in (home, away) if gp.get(t, 99) < EARLY_GP]
if early:
    st.info(f"Early season: {', '.join(f'{t} ({gp.get(t, 0)} GP)' for t in early)}. Ratings still lean on last season, "
            "pulled toward league average. Roster changes since then aren't reflected yet.")

with st.expander("Lines", expanded=True):
    l1, l2, l3, l4 = st.columns(4)
    total_line = l1.number_input("Game total", 3.0, 10.0, 6.5, 0.5)
    home_spread = l2.number_input(f"{home} spread", -3.5, 3.5, -1.5, 0.5, help="-1.5 = home lays 1.5 goals")
    home_tt = l3.number_input(f"{home} team total", 0.5, 6.5, 2.5, 0.5, key=f"htt_{home}")
    away_tt = l4.number_input(f"{away} team total", 0.5, 6.5, 2.5, 0.5, key=f"att_{away}")

# ------------------------------------------------------------------ goalies
home_gid = away_gid = None
home_conf = away_conf = 1.0
if pred.goalies_on:
    st.subheader("Goalies")
    st.caption("Pick a starter once he's confirmed (typically morning of the game). 'Unknown' uses each team's "
               "recent starter mix, shifted toward the backup on the second night of a back-to-back.")
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

hs = pred.goalie_spec(home, home_gid, home_conf, b2b=home_b2b)
as_ = pred.goalie_spec(away, away_gid, away_conf, b2b=away_b2b)
r = pred.predict_game(home, away, hs, as_, total_line=total_line, home_spread=home_spread,
                      home_team_line=home_tt, away_team_line=away_tt, home_b2b=home_b2b, away_b2b=away_b2b)

# ------------------------------------------------------------------ headline
st.divider()
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric(f"{home} expected goals", f"{r['exp_home']:.2f}", help="Final score, including the OT/SO winner")
m2.metric(f"{away} expected goals", f"{r['exp_away']:.2f}")
m3.metric("Expected total", f"{r['exp_total']:.2f}")
m4.metric(f"{home} win probability", pct(r["p_home_win"]), help="Includes overtime / shootout")
m5.metric("Goes to OT/SO", pct(r["p_tie_reg"]))

tabs = st.tabs(["Game markets", "Today's slate", "Period markets", "Goalie impact", "Track record", "Model info"])
tab_game, tab_slate, tab_period, tab_goalie, tab_track, tab_model = tabs

with tab_game:
    mk = build_markets(r, home, away)
    st.caption("Type your sportsbook's American odds (e.g. -115, +140) in **Your odds**. Price both sides of a market "
               "and the book's margin is removed before comparing. On whole-number lines the push chance is shown separately "
               "and fair odds exclude it, matching how books grade a push as a refund.")
    edit_key = f"odds_{home}_{away}_{total_line}_{home_spread}_{home_tt}_{away_tt}"
    shown = pd.DataFrame({"Market": mk["Market"], "Model": mk["p"].map(pct),
                          "Push": mk["push"].map(lambda x: pct(x) if x > 1e-9 else ""),
                          "Fair odds": (mk["p"] / (1 - mk["push"])).map(lambda p: fair(p) if p > 0 else "-"),
                          "Your odds": [None] * len(mk)})
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
            "¼ Kelly stake": priced["kelly"].map(lambda x: f"{100 * x:.1f}% of bankroll" if x > 0 else "no bet")}),
            hide_index=True, width="stretch")
        st.caption("Small edges (under ~2-3 points) are within this model's error. Stakes assume the model is right; "
                   "quarter Kelly is already a cautious fraction.")
    mp = pd.DataFrame(r["margin_pmf"])
    mp["label"] = [f"{home} by {m}" if m > 0 else (f"{away} by {-m}" if m < 0 else "") for m in mp["margin"]]
    mp = mp[mp["margin"] != 0]
    st.markdown("**Final margin**")
    st.bar_chart(mp.set_index("margin")["p"], x_label=f"Final margin ({home} minus {away})", y_label="Probability")

with tab_slate:
    if not slate:
        st.info("No upcoming games in this model file. `refresh.py` pulls the next few days of the schedule.")
    else:
        rows = []
        for g in slate:
            try:
                gr = pred.predict_game(g["home"], g["away"],
                                       pred.goalie_spec(g["home"], None, b2b=g.get("home_b2b", False)),
                                       pred.goalie_spec(g["away"], None, b2b=g.get("away_b2b", False)),
                                       home_b2b=g.get("home_b2b", False), away_b2b=g.get("away_b2b", False))
            except ValueError:
                continue
            fav = g["home"] if gr["p_home_win"] >= 0.5 else g["away"]
            pf = max(gr["p_home_win"], 1 - gr["p_home_win"])
            rows.append({"Date": g["date"], "Game": f"{g['away']} @ {g['home']}",
                         "Back-to-back": ", ".join(t for t, on in ((g["away"], g.get("away_b2b")), (g["home"], g.get("home_b2b"))) if on),
                         "Model pick": fav, "Pick win %": pct(pf), "Fair ML": american(pf),
                         "Exp. total": f"{gr['exp_total']:.2f}", "Over 6.5": pct(gr["total"]["over"]),
                         "OT/SO": pct(gr["p_tie_reg"])})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("Uses each team's usual goalie mix. Pick a game at the top to set confirmed starters and compare prices.")

with tab_period:
    df = pd.DataFrame([{"Period": p["period"], f"{home} exp. goals": round(p["exp_home"], 2),
                        f"{away} exp. goals": round(p["exp_away"], 2), "Exp. total": round(p["exp_total"], 2),
                        "Over 0.5 goals": pct(p["over_0_5"]), "Over 1.5 goals": pct(p["over_1_5"]),
                        "Both teams score": pct(p["btts"])} for p in r["periods"]])
    st.dataframe(df, hide_index=True, width="stretch")
    K = meta.get("period_share_K")
    note = ("Team-by-team differences in period scoring weren't distinguishable from noise, so every team uses the "
            "league split." if K and K > 1e5 else "Each team's period split is shrunk toward the league split.")
    st.caption(f"Expected regulation goals split by period. {note}")

with tab_goalie:
    if not pred.goalies_on:
        st.info("Goalie data isn't in this model yet. Finish `nhl_goalies.py`, then run `python refresh.py --skip-fetch --push`.")
    else:
        if r["goalie_info"]:
            gi = pd.DataFrame([{"Team": g["team"], "Goalie": g["name"], "Weight": pct(g["prob"]),
                                "Rating (goals/game vs avg)": f"{g['rating_gpg']:+.2f}", "Basis": g["basis"]}
                               for g in r["goalie_info"]])
            st.markdown("**Goalies used in this prediction**")
            st.dataframe(gi, hide_index=True, width="stretch")
        base = pred.predict_game(home, away, None, None, total_line=total_line, home_spread=home_spread,
                                 home_b2b=home_b2b, away_b2b=away_b2b)
        if hs is None and as_ is None:
            st.info("Select at least one goalie above to see how much it moves the prediction.")
        else:
            imp = pd.DataFrame([
                {"": f"{home} win probability", "Usual goalies": pct(base["p_home_win"]), "Selected": pct(r["p_home_win"]),
                 "Change": f"{100 * (r['p_home_win'] - base['p_home_win']):+.1f} pts"},
                {"": "Expected total goals", "Usual goalies": f"{base['exp_total']:.2f}", "Selected": f"{r['exp_total']:.2f}",
                 "Change": f"{r['exp_total'] - base['exp_total']:+.2f}"},
                {"": f"Total over {total_line:g}", "Usual goalies": pct(base["total"]["over"]), "Selected": pct(r["total"]["over"]),
                 "Change": f"{100 * (r['total']['over'] - base['total']['over']):+.1f} pts"},
            ])
            st.markdown("**Impact of the goalie selection**" + (" (includes the back-to-back adjustment)" if (home_b2b or away_b2b) and home_gid is None and away_gid is None else ""))
            st.dataframe(imp, hide_index=True, width="stretch")
        beta = meta.get("goalie_beta")
        if beta is not None:
            st.caption(f"Fitted goalie coefficient: {beta:+.2f} (near -1 = ratings trusted at face value; near 0 = goalies add little). "
                       "Ratings use save % only, shrunk toward league average.")

with tab_track:
    bt = meta.get("backtest")
    if LOG_PATH.exists():
        lg = get_log(LOG_PATH.stat().st_mtime)
        done = lg[lg["home_score"].notna()].copy() if "home_score" in lg else lg.iloc[0:0]
        if done.empty:
            st.info(f"{len(lg)} games logged so far; none settled yet. Results fill in on the next refresh after each game.")
        else:
            y = (done["home_score"] > done["away_score"]).astype(int)
            p = done["p_home_win"]
            fav_right = ((p >= 0.5) == (y == 1)).mean()
            reg = np.select([done["home_reg"] > done["away_reg"], done["home_reg"] == done["away_reg"]], [0, 1], 2)
            P3 = done[["p_home_reg", "p_tie_reg", "p_away_reg"]].values
            t1, t2, t3, t4 = st.columns(4)
            t1.metric("Games settled", f"{len(done):,}")
            t2.metric("Model's moneyline pick won", pct(fav_right),
                      help="Share of settled games where the team the model gave a 50%+ win chance actually won "
                           "(including OT/SO). This is the model's pick, not the betting market's favourite.")
            t3.metric("Moneyline Brier", f"{((p - y) ** 2).mean():.4f}",
                      delta=f"{((p - y) ** 2).mean() - ((y.mean() - y) ** 2).mean():+.4f} vs constant home-win rate",
                      delta_color="inverse",
                      help="Lower is better. 0.25 = always saying 50%. The comparison is against always predicting "
                           "the home-win rate seen in these games; negative means the model did better.")
            t4.metric("3-way log-loss", f"{-np.log(P3[np.arange(len(done)), reg]).mean():.3f}",
                      help="Lower is better. ~1.04-1.05 is what the backtest achieved.")
            if len(done) >= 40:
                done["bucket"] = pd.cut(p, [0, .40, .45, .50, .55, .60, 1])
                cal = done.assign(y=y).groupby("bucket", observed=True).agg(
                    Predicted=("p_home_win", "mean"), Actual=("y", "mean"), Games=("y", "size")).reset_index()
                cal["bucket"] = cal["bucket"].astype(str)
                st.markdown("**Calibration (home win probability)**")
                st.dataframe(cal.round(3), hide_index=True, width="stretch")
            recent = done.tail(15).iloc[::-1]
            pick_home = recent["p_home_win"] >= 0.5
            home_won = recent["home_score"] > recent["away_score"]
            st.markdown("**Latest results**")
            st.dataframe(pd.DataFrame({
                "Date": recent["date"], "Game": recent["away"] + " @ " + recent["home"],
                "Model pick": np.where(pick_home, recent["home"], recent["away"]),
                "Pick win %": np.maximum(recent["p_home_win"], 1 - recent["p_home_win"]).map(pct),
                "Pick won": np.where(pick_home == home_won, "✓", "✗"),
                "Score": recent["away_score"].astype(int).astype(str) + "-" + recent["home_score"].astype(int).astype(str),
                "OT/SO": np.where(recent["home_reg"] == recent["away_reg"], "yes", "")}), hide_index=True, width="stretch")
            st.caption("Predictions are logged by refresh.py before each game with each team's usual goalie mix, so this "
                       "is honest out-of-sample performance.")
    else:
        st.info("No prediction log yet. It starts filling the first time refresh.py runs with upcoming games on the schedule.")
    if bt:
        st.markdown(f"**Walk-forward backtest** ({bt['n_games']:,} games, seasons {', '.join(bt['seasons'])})")
        st.dataframe(pd.DataFrame([
            {"Check": "Moneyline Brier", "Model": f"{bt['win_brier']:.4f}", "Reference": f"{bt['const_brier']:.4f} (constant home rate)"},
            {"Check": "3-way regulation log-loss", "Model": f"{bt['reg3_logloss']:.4f}", "Reference": ""},
            {"Check": "OT/SO frequency", "Model": pct(bt["pred_tie"]), "Reference": f"{pct(bt['actual_tie'])} actual"},
            {"Check": "Over 5.5 rate", "Model": pct(bt["pred_o55"]), "Reference": f"{pct(bt['actual_o55'])} actual"},
            {"Check": "Over 6.5 rate", "Model": pct(bt["pred_o65"]), "Reference": f"{pct(bt['actual_o65'])} actual"},
            {"Check": "Home -1.5 covers", "Model": pct(bt["pred_cover"]), "Reference": f"{pct(bt['actual_cover'])} actual"},
        ]), hide_index=True, width="stretch")
        if bt.get("calibration"):
            cal = pd.DataFrame(bt["calibration"]).rename(columns={"bucket": "Home win prob.", "predicted": "Predicted",
                                                                  "actual": "Actual", "n": "Games"})
            st.dataframe(cal.round(3), hide_index=True, width="stretch")

with tab_model:
    a, b = st.columns(2)
    with a:
        st.markdown("**Team ratings** (log scale; higher defense = allows more)")
        st.dataframe(pred.team_ratings().round(3), width="stretch", height=420)
    with b:
        leaders = pred.goalie_leaders()
        if leaders is not None:
            st.markdown("**Goalie ratings** (min 20 starts, active in the last ~13 months)")
            show = leaders.copy()
            show["sv_pct"] = show["sv_pct"].map(lambda x: f"{x:.3f}")
            show["rating_gpg"] = show["rating_gpg"].round(2)
            st.dataframe(show, width="stretch", height=420)
        else:
            st.markdown("**Goalie ratings** - not available until goalie data is included.")
    b2b = meta.get("b2b_coef")
    st.caption(
        f"Recency half-life {meta['half_life_days']:g} days · ridge α {meta['alpha_l2']:g} "
        f"(effective {meta.get('alpha_eff', meta['alpha_l2']):.3f}) · past-season weight {meta.get('carryover', 1):g} · "
        f"shots blend {meta.get('shot_blend', 0):.0%}"
        + (f" · teams on a back-to-back score {100 * (np.exp(b2b) - 1):+.1f}%" if b2b is not None else "")
        + f" · fitted {meta['fitted_at']}. The model predicts regulation goals; ties are settled by one OT/SO goal and the "
        "score grid is calibrated to real overtime and empty-net frequencies. For information and entertainment only - not betting advice.")
