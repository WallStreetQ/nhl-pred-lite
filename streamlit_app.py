"""
streamlit_app.py
------------------
NHL game prediction tool — deployed via Streamlit Community Cloud.

Reads ONLY model_coefficients.json (plain data, no pickle, no statsmodels) so this
app has a minimal, fast-to-cold-start dependency footprint. The heavy lifting (data
pipeline, model fitting, hyperparameter sweep) all happens offline in
nhl_prediction_model.ipynb — this file only reproduces the prediction math from the
coefficients that notebook exports.

To update the live app after refitting the model: overwrite model_coefficients.json
in this repo and push. Streamlit Community Cloud redeploys automatically.
"""

import json
from pathlib import Path

import numpy as np
import streamlit as st
from scipy.stats import poisson, skellam

st.set_page_config(page_title="NHL Predictions", page_icon="🏒", layout="centered")

COEFFICIENTS_PATH = Path(__file__).parent / "model_coefficients.json"


# ---------------------------------------------------------------------------
# Password gate
# ---------------------------------------------------------------------------
def check_password():
    """Simple shared-password gate. Password is set via Streamlit secrets, never
    committed to the repo (see secrets.toml.example)."""

    def password_entered():
        if st.session_state.get("password_input") == st.secrets.get("app_password"):
            st.session_state["password_correct"] = True
            del st.session_state["password_input"]
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct"):
        return True

    st.text_input(
        "Password", type="password", key="password_input", on_change=password_entered
    )
    if "password_correct" in st.session_state and not st.session_state["password_correct"]:
        st.error("Incorrect password.")
    return False


# ---------------------------------------------------------------------------
# Model loading (cached so it only reads the file once per deploy, not per click)
# ---------------------------------------------------------------------------
@st.cache_data
def load_coefficients():
    if not COEFFICIENTS_PATH.exists():
        return None
    with open(COEFFICIENTS_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Prediction math (mirrors nhl_prediction_model.ipynb sections 9 and 15 exactly —
# validated there to reproduce the full statsmodels model's output before shipping)
# ---------------------------------------------------------------------------
def predict_lambdas(coefs, home_team, away_team, home_rest, away_rest):
    intercept = coefs["intercept"]
    home_adv = coefs["home_advantage"]
    rest_coef = coefs["rest_days_coef"]
    opp_rest_coef = coefs["opp_rest_days_coef"]

    h = coefs["teams"].get(home_team, {"attack": 0.0, "defense": 0.0})
    a = coefs["teams"].get(away_team, {"attack": 0.0, "defense": 0.0})

    lam_home = np.exp(intercept + h["attack"] + a["defense"] + home_adv * 1
                       + rest_coef * home_rest + opp_rest_coef * away_rest)
    lam_away = np.exp(intercept + a["attack"] + h["defense"] + home_adv * 0
                       + rest_coef * away_rest + opp_rest_coef * home_rest)
    return float(lam_home), float(lam_away)


def btts_prob(lam_a, lam_b):
    p0_a = poisson.pmf(0, lam_a)
    p0_b = poisson.pmf(0, lam_b)
    return 1 - p0_a - p0_b + (p0_a * p0_b)


def predict_game(coefs, home_team, away_team, total_line, spread_line, home_rest, away_rest):
    lam_h, lam_a = predict_lambdas(coefs, home_team, away_team, home_rest, away_rest)
    total_lambda = lam_h + lam_a

    result = {
        "lam_h": lam_h, "lam_a": lam_a, "total_lambda": total_lambda,
        "p_total_over": 1 - poisson.cdf(int(total_line), total_lambda),
        "p_home_team_total_over": 1 - poisson.cdf(int(total_line / 2), lam_h),
        "p_away_team_total_over": 1 - poisson.cdf(int(total_line / 2), lam_a),
        "p_home_covers": 1 - skellam.cdf(int(spread_line - 0.5), lam_h, lam_a),
        "p_home_win": 1 - skellam.cdf(0, lam_h, lam_a),
        "p_away_win": skellam.cdf(-1, lam_h, lam_a),
        "p_tie": skellam.pmf(0, lam_h, lam_a),
        "p_btts": btts_prob(lam_h, lam_a),
        "periods": [],
    }

    period_shares = coefs["period_shares"]
    for period in ["1", "2", "3"]:
        share_h = period_shares.get(home_team, {}).get(period, 1 / 3)
        share_a = period_shares.get(away_team, {}).get(period, 1 / 3)
        p_lam_h = lam_h * share_h
        p_lam_a = lam_a * share_a
        result["periods"].append({
            "period": period, "lam_h": p_lam_h, "lam_a": p_lam_a,
            "total": p_lam_h + p_lam_a,
            "p_btts": btts_prob(p_lam_h, p_lam_a),
            "p_over_1p5": 1 - poisson.cdf(1, p_lam_h + p_lam_a),
        })

    return result


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
if not check_password():
    st.stop()

coefs = load_coefficients()

if coefs is None:
    st.error(
        "No model_coefficients.json found in this app. Run the notebook's export "
        "cell (section 14) and add the file to this repo."
    )
    st.stop()

st.title("🏒 NHL Game Predictions")
st.caption(
    f"Model trained through {coefs['trained_through_date']} · "
    f"last refit {coefs['fitted_at'][:10]}"
)

team_list = sorted(coefs["teams"].keys())

col1, col2 = st.columns(2)
with col1:
    home_team = st.selectbox("Home team", team_list, index=0)
with col2:
    away_options = [t for t in team_list if t != home_team]
    away_team = st.selectbox("Away team", away_options, index=0)

with st.expander("Advanced options (rest days, betting lines)"):
    default_rest = coefs["default_rest_days"]
    rest_cap = coefs.get("rest_days_cap", 6)
    c1, c2 = st.columns(2)
    with c1:
        home_rest = st.slider(f"{home_team} rest days", 0, rest_cap, int(default_rest))
    with c2:
        away_rest = st.slider(f"{away_team} rest days", 0, rest_cap, int(default_rest))
    total_line = st.number_input("Total goals line", value=6.5, step=0.5)
    spread_line = st.number_input(f"{home_team} spread line", value=0.5, step=0.5)

if st.button("Get prediction", type="primary"):
    r = predict_game(coefs, home_team, away_team, total_line, spread_line, home_rest, away_rest)

    st.subheader(f"{away_team} @ {home_team}")

    m1, m2, m3 = st.columns(3)
    m1.metric(f"{home_team} expected goals", f"{r['lam_h']:.2f}")
    m2.metric(f"{away_team} expected goals", f"{r['lam_a']:.2f}")
    m3.metric("Expected total", f"{r['total_lambda']:.2f}")

    st.markdown("#### Moneyline")
    c1, c2, c3 = st.columns(3)
    c1.metric(f"{home_team} wins", f"{r['p_home_win']:.0%}")
    c2.metric(f"{away_team} wins", f"{r['p_away_win']:.0%}")
    c3.metric("Tie after regulation", f"{r['p_tie']:.0%}")

    st.markdown(f"#### Spread ({home_team} {spread_line:+.1f})")
    st.metric(f"{home_team} covers", f"{r['p_home_covers']:.0%}")

    st.markdown(f"#### Totals (line {total_line})")
    c1, c2, c3 = st.columns(3)
    c1.metric("Over", f"{r['p_total_over']:.0%}")
    c2.metric(f"{home_team} team total over {total_line/2:.1f}", f"{r['p_home_team_total_over']:.0%}")
    c3.metric(f"{away_team} team total over {total_line/2:.1f}", f"{r['p_away_team_total_over']:.0%}")

    st.markdown("#### Both teams to score")
    st.metric("BTTS — full game", f"{r['p_btts']:.0%}")

    st.markdown("#### Period-by-period breakdown")
    for p in r["periods"]:
        st.write(
            f"**Period {p['period']}** — expected {p['lam_h']:.2f}–{p['lam_a']:.2f} "
            f"(total {p['total']:.2f}) · BTTS {p['p_btts']:.0%} · Over 1.5 {p['p_over_1p5']:.0%}"
        )

    st.caption(
        "Model estimates only, not betting advice. Predictions reflect historical "
        "scoring patterns and don't account for injuries, lineup changes, or "
        "goaltender starts."
    )
