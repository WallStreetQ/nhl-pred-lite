"""
app.py — NHL game predictor (Streamlit)

Reads only model/model_artifact.json (written by refresh.py). No raw data, no API calls.
Run locally:  streamlit run app.py
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from nhl_model import Predictor, load_artifact

ART_PATH = Path(__file__).parent / "model" / "model_artifact.json"
USUAL = "Unknown - use team's usual mix"

st.set_page_config(page_title="NHL Game Predictor", page_icon="🏒", layout="wide")


@st.cache_resource(show_spinner=False)
def get_predictor(mtime: float) -> Predictor:      # mtime in the key => reload if the file changes
    return Predictor(load_artifact(ART_PATH))


def pct(p):
    return f"{100 * p:.1f}%"


def fair(p):
    """Fair (no-vig) odds: decimal and American."""
    if p <= 0.0005 or p >= 0.9995:
        return "-"
    dec = 1 / p
    us = -100 * p / (1 - p) if p >= 0.5 else 100 * (1 - p) / p
    return f"{dec:.2f} ({us:+.0f})"


def market_rows(r, home, away):
    rows = [
        (f"{home} win (incl OT/SO)", r["p_home_win"]),
        (f"{away} win (incl OT/SO)", 1 - r["p_home_win"]),
        (f"{home} win in regulation", r["p_home_reg"]),
        (f"{away} win in regulation", r["p_away_reg"]),
        ("Tied after regulation", r["p_tie_reg"]),
    ]
    s = r["spread"]
    rows += [(f"{home} {s['line']:+.1f} covers", s["cover"]), (f"{away} {-s['line']:+.1f} covers", s["fail"])]
    for name, key in (("Game total", "total"), (f"{home} team total", "home_total"), (f"{away} team total", "away_total")):
        t = r[key]
        rows += [(f"{name} over {t['line']:g}", t["over"]), (f"{name} under {t['line']:g}", t["under"])]
    rows.append(("Both teams score (full game)", r["btts"]))
    return pd.DataFrame([{"Market": m, "Probability": pct(p), "Fair odds": fair(p)} for m, p in rows])


# ------------------------------------------------------------------ load
if not ART_PATH.exists():
    st.error("model/model_artifact.json not found. Run `python refresh.py` locally, then commit and push the model file.")
    st.stop()
pred = get_predictor(ART_PATH.stat().st_mtime)
meta = pred.meta

st.title("🏒 NHL Game Predictor")
st.caption(f"Model trained through **{meta['trained_through']}** · {meta['n_games']:,} games · "
           f"goalie data **{'ON' if pred.goalies_on else 'OFF'}**")

# ------------------------------------------------------------------ inputs
teams = pred.active_teams
c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
away = c1.selectbox("Away team", teams, index=teams.index("VGK") if "VGK" in teams else 0, key="away")
home = c2.selectbox("Home team", teams, index=teams.index("FLA") if "FLA" in teams else 1, key="home")
total_line = c3.number_input("Game total line", 3.0, 10.0, 6.5, 0.5)
home_spread = c4.number_input(f"Spread for home team (e.g. -1.5)", -3.5, 3.5, -1.5, 0.5)

if home == away:
    st.warning("Pick two different teams.")
    st.stop()

home_gid = away_gid = None
home_conf = away_conf = 1.0
if pred.goalies_on:
    st.subheader("Goalies")
    st.caption("Pick a starter once he's confirmed (typically morning of the game). Leave as 'Unknown' to use each team's recent starter mix.")
    gc1, gc2 = st.columns(2)
    for col, team, side in ((gc1, away, "away"), (gc2, home, "home")):
        opts = pred.goalie_options(team)
        labels = {USUAL: None}
        for o in opts:
            labels[f"{o['name']}  ·  {o['starts']} starts  ·  {o['rating_gpg']:+.2f} g/g"] = o["player_id"]
        with col:
            choice = st.selectbox(f"{team} goalie", list(labels), key=f"g_{side}_{team}")
            gid = labels[choice]
            conf = 1.0
            if gid is not None:
                conf = st.slider("Chance he starts", 50, 100, 100, 5, key=f"c_{side}_{team}",
                                 help="Below 100% the remainder goes to the team's other usual goalies.") / 100
        if side == "away":
            away_gid, away_conf = gid, conf
        else:
            home_gid, home_conf = gid, conf

hs = pred.goalie_spec(home, home_gid, home_conf)
as_ = pred.goalie_spec(away, away_gid, away_conf)
r = pred.predict_game(home, away, hs, as_, total_line=total_line, home_spread=home_spread)

# ------------------------------------------------------------------ headline
st.divider()
m1, m2, m3, m4 = st.columns(4)
m1.metric(f"{home} expected goals", f"{r['lam_home']:.2f}")
m2.metric(f"{away} expected goals", f"{r['lam_away']:.2f}")
m3.metric("Expected total", f"{r['lam_total']:.2f}")
m4.metric(f"{home} win probability", pct(r["p_home_win"]), help="Includes overtime / shootout")

tab_game, tab_period, tab_goalie, tab_model = st.tabs(["Game markets", "Period markets", "Goalie impact", "Model info"])

with tab_game:
    st.dataframe(market_rows(r, home, away), hide_index=True, width="stretch")
    st.caption("Fair odds are the no-vig price implied by the model's probability (decimal, with American in brackets). "
               "Spreads and moneyline settle on the final score including OT/SO.")

with tab_period:
    df = pd.DataFrame([{"Period": p["period"], f"{home} exp. goals": round(p["exp_home"], 2),
                        f"{away} exp. goals": round(p["exp_away"], 2), "Exp. total": round(p["exp_total"], 2),
                        "Both teams score": pct(p["btts"]), "Over 1.5 goals": pct(p["over_1_5"])} for p in r["periods"]])
    st.dataframe(df, hide_index=True, width="stretch")
    st.caption("Period figures split each team's expected goals by its historical share of goals in that period.")

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
        base = pred.predict_game(home, away, None, None, total_line=total_line, home_spread=home_spread)
        if home_gid is None and away_gid is None:
            st.info("Select at least one goalie above to see how much it moves the prediction.")
        else:
            imp = pd.DataFrame([
                {"": f"{home} win probability", "Usual goalies": pct(base["p_home_win"]), "Selected": pct(r["p_home_win"]),
                 "Change": f"{100 * (r['p_home_win'] - base['p_home_win']):+.1f} pts"},
                {"": "Expected total goals", "Usual goalies": f"{base['lam_total']:.2f}", "Selected": f"{r['lam_total']:.2f}",
                 "Change": f"{r['lam_total'] - base['lam_total']:+.2f}"},
                {"": f"P(total over {total_line:g})", "Usual goalies": pct(base["total"]["over"]), "Selected": pct(r["total"]["over"]),
                 "Change": f"{100 * (r['total']['over'] - base['total']['over']):+.1f} pts"},
            ])
            st.markdown("**Impact of your goalie selection**")
            st.dataframe(imp, hide_index=True, width="stretch")
        beta = meta.get("goalie_beta")
        if beta is not None:
            st.caption(f"Fitted goalie coefficient: {beta:+.2f} (near -1 = ratings trusted at face value; near 0 = goalies add little). "
                       "Ratings use save % only, shrunk toward league average.")

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
    st.caption(f"Recency half-life {meta['half_life_days']:g} days · ridge α {meta['alpha_l2']:g} · "
               f"fitted {meta['fitted_at']}. The Poisson model under-predicts overtime frequency, so tie-related numbers are approximate. "
               "For information and entertainment only - not betting advice.")
