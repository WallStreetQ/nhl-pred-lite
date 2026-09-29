"""
goalie_features.py
-------------------
Everything needed to add goalie information to the Poisson team-strength model.

Design in one paragraph
  A goalie's save% is a noisy statistic (a starter faces ~1,700 shots a season and the
  true spread between starters is small), so raw sv% must be shrunk hard toward the league
  mean. We build a leak-free, empirical-Bayes goalie rating (only games BEFORE each game are
  used), express it in "goals per game vs. an average goalie", and feed it into the Poisson
  regression as one extra column:  log(lambda) = intercept + attack_team + defense_opp
                                     + home + beta * opp_goalie_effect.
  beta is LEARNED: beta ~ 1 means the rating is trustworthy at face value, beta < 1 means
  it is still too noisy, beta ~ 0 means the goalie adds nothing beyond the team ratings.

Also fixes two issues in the current notebook fit (see review): the ridge penalty is applied
ONLY to team/opponent terms (not intercept / home / goalie), and shrinkage is toward the
league average rather than toward the alphabetical reference team.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import skellam

# ----------------------------------------------------------------------------------------
# 1. Penalized Poisson regression with a per-column penalty mask
# ----------------------------------------------------------------------------------------

def fit_poisson_ridge(X, y, w, penalized, alpha):
    """Weighted Poisson (log link) with L2 penalty only where `penalized` is True.
    Objective = -sum(w*(y*eta - exp(eta)))/sum(w) + 0.5*alpha*sum_{penalized} beta^2
    (same scale convention as sklearn's PoissonRegressor)."""
    w = np.asarray(w, float)
    sw = w.sum()
    pen = np.asarray(penalized, float)
    beta0 = np.zeros(X.shape[1])
    beta0[0] = np.log(np.average(y, weights=w))  # column 0 must be the intercept

    def f(b):
        eta = np.clip(X @ b, -20, 20)
        mu = np.exp(eta)
        val = -(w * (y * eta - mu)).sum() / sw + 0.5 * alpha * (pen * b * b).sum()
        grad = -(X.T @ (w * (y - mu))) / sw + alpha * pen * b
        return val, grad

    res = minimize(f, beta0, jac=True, method="L-BFGS-B", options={"maxiter": 500})
    return res.x


class PoissonTeamModel:
    """goals ~ intercept + team(attack) + opponent(defense) + is_home [+ extra columns]."""

    def __init__(self, teams, alpha=0.1, extra_cols=()):
        self.teams = list(teams)
        self.ix = {t: i for i, t in enumerate(self.teams)}
        self.alpha = alpha
        self.extra_cols = list(extra_cols)

    def _X(self, df):
        n, T = len(df), len(self.teams)
        X = np.zeros((n, 1 + 2 * T + 1 + len(self.extra_cols)))
        X[:, 0] = 1.0
        r = np.arange(n)
        X[r, 1 + df["team"].map(self.ix).values] = 1.0
        X[r, 1 + T + df["opponent"].map(self.ix).values] = 1.0
        X[:, 1 + 2 * T] = df["is_home"].values
        for j, c in enumerate(self.extra_cols):
            X[:, 2 + 2 * T + j] = df[c].values
        return X

    def fit(self, df, weight_col="fit_weight"):
        T = len(self.teams)
        X = self._X(df)
        pen = np.zeros(X.shape[1])
        pen[1: 1 + 2 * T] = 1.0  # team + opponent dummies only
        self.beta = fit_poisson_ridge(X, df["goals"].values, df[weight_col].values, pen, self.alpha)
        return self

    def predict(self, df):
        return np.exp(np.clip(self._X(df) @ self.beta, -20, 20))

    def coef(self, name):
        """Coefficient for is_home or one of the extra columns."""
        T = len(self.teams)
        if name == "is_home":
            return self.beta[1 + 2 * T]
        return self.beta[2 + 2 * T + self.extra_cols.index(name)]


def win_prob(lam_h, lam_a, home_ot_win=0.518):
    """P(home wins) INCLUDING overtime/shootout: regulation win + share of the tie mass.
    (The notebook's 1 - skellam.cdf(0) is regulation-only and omits ~22% of games.)
    0.518 = home win rate in games that went past regulation in games.csv."""
    return (1 - skellam.cdf(0, lam_h, lam_a)) + skellam.pmf(0, lam_h, lam_a) * home_ot_win


# ----------------------------------------------------------------------------------------
# 2. Leak-free goalie ratings
# ----------------------------------------------------------------------------------------

def estimate_prior_strength(gk, min_shots=500, k_bounds=(300.0, 6000.0)):
    """Estimate the shrinkage strength k (in shots) as  k = p(1-p) / tau^2,  where tau^2 is the
    variance of TRUE goalie talent in sv%.

    Uses the year-over-year COVARIANCE of the same goalie's season sv% (sampling noise is
    independent across seasons, so it cancels — far more stable than 'observed variance minus
    binomial noise', which can swing wildly with only a few seasons of data).
    Falls back to variance-minus-noise if fewer than 15 repeat goalie-seasons exist.
    Returns (k, tau, n_pairs). k is clipped to k_bounds; still validate with a CV sweep of k."""
    s = gk.groupby(["player_id", "season"]).agg(sa=("shots_against", "sum"), sv=("saves", "sum")).reset_index()
    s = s[s["sa"] >= min_shots].copy()
    s["p"] = s["sv"] / s["sa"]
    p = s["sv"].sum() / s["sa"].sum()

    nxt = s.assign(season=s["season"] + 10001)  # 20212022 -> 20222023
    pairs = s.merge(nxt, on=["player_id", "season"], suffixes=("", "_prev"))
    if len(pairs) >= 15:
        w = pairs["sa"] * pairs["sa_prev"] / (pairs["sa"] + pairs["sa_prev"])
        tau2 = float(np.average((pairs["p"] - p) * (pairs["p_prev"] - p), weights=w))
    else:
        obs_var = np.average((s["p"] - p) ** 2, weights=s["sa"])
        tau2 = float(obs_var - (p * (1 - p) / s["sa"]).mean())
    tau2 = max(tau2, 1e-8)
    k = float(np.clip(p * (1 - p) / tau2, *k_bounds))
    return k, float(np.sqrt(tau2)), len(pairs)


def build_goalie_ratings(gk, k_shots=1400.0, prior_offset=0.0, shots_per_game=None):
    """Adds columns to the goalie-game table, each computed ONLY from games strictly earlier:
        prior_shots        shots faced before this game
        rating_sv          shrunk sv% minus league sv% (positive = better than average)
        rating_gpg         same, in goals/game vs average goalie at league-average shot volume
    prior_offset (sv% pts, usually <= 0) sets the prior mean for goalies with little history
    relative to league average — e.g. -0.003 if unproven goalies tend to be below average.
    """
    gk = gk.sort_values(["game_date", "game_id"]).reset_index(drop=True).copy()
    if shots_per_game is None:
        shots_per_game = gk.groupby(["game_id", "team"])["shots_against"].sum().mean()

    # league sv% using only earlier DATES
    daily = gk.groupby("game_date")[["saves", "shots_against"]].sum().sort_index()
    cum = daily.cumsum().shift(1)
    lg_sv = (cum["saves"] / cum["shots_against"]).rename("lg_sv").reindex(daily.index)
    lg_sv = lg_sv.fillna(0.905)  # first day only
    gk = gk.merge(lg_sv.reset_index(), on="game_date", how="left")

    # goalie's own history, excluding the current game
    by = gk.groupby("player_id")
    gk["prior_saves"] = by["saves"].cumsum() - gk["saves"]
    gk["prior_shots"] = by["shots_against"].cumsum() - gk["shots_against"]

    prior_mean = gk["lg_sv"] + prior_offset
    shrunk = (gk["prior_saves"] + k_shots * prior_mean) / (gk["prior_shots"] + k_shots)
    gk["rating_sv"] = shrunk - gk["lg_sv"]
    gk["rating_gpg"] = gk["rating_sv"] * shots_per_game
    return gk


def starter_table(gk_rated):
    """One row per (game_id, team): the STARTER's pre-game rating (what is knowable once
    the starter is confirmed — the honest 'oracle' for backtesting)."""
    s = gk_rated[gk_rated["starter"]].sort_values("toi_sec", ascending=False)
    s = s.drop_duplicates(["game_id", "team"])
    return s[["game_id", "team", "player_id", "rating_gpg", "prior_shots"]]


def attach_opp_goalie_effect(long_df, starters, mean_goals=3.1, col="opp_goalie_effect"):
    """Adds `opp_goalie_effect` (log-lambda units; positive = opposing goalie is stingier,
    so the coefficient is expected to be NEGATIVE ~ -1 when ratings are well calibrated).
    Missing goalie data -> 0 (league average)."""
    m = starters.rename(columns={"team": "opponent", "rating_gpg": "opp_rating_gpg"})
    out = long_df.merge(m[["game_id", "opponent", "opp_rating_gpg"]], on=["game_id", "opponent"], how="left")
    out[col] = (out["opp_rating_gpg"].fillna(0.0) / mean_goals)
    return out.drop(columns=["opp_rating_gpg"])


# ----------------------------------------------------------------------------------------
# 3. Live use: the starter isn't known until ~morning skate
# ----------------------------------------------------------------------------------------

def rest_flags(games):
    """Days of rest and back-to-back flag per (game_id, team), from games.parquet alone.
    Backup goalies start disproportionately on the 2nd night of a back-to-back."""
    g = games[games["game_type"].isin([2, 3])]
    long = pd.concat([
        g[["game_id", "game_date", "home_team"]].rename(columns={"home_team": "team"}),
        g[["game_id", "game_date", "away_team"]].rename(columns={"away_team": "team"}),
    ]).sort_values(["team", "game_date"])
    long["game_date"] = pd.to_datetime(long["game_date"])
    long["rest_days"] = long.groupby("team")["game_date"].diff().dt.days - 1
    long["back_to_back"] = (long["rest_days"] == 0).astype(int)
    return long[["game_id", "team", "rest_days", "back_to_back"]]


def p_backup_by_rest(gk, rest):
    """Empirical P(backup starts) by back-to-back status. `backup` = a game's starter who is
    NOT the team's most-used goalie over the trailing 30 team games (simple, transparent)."""
    s = starter_table(build_goalie_ratings(gk))[["game_id", "team", "player_id"]]
    s = s.merge(gk[["game_id", "game_date"]].drop_duplicates(), on="game_id").sort_values(["team", "game_date"])
    def _is_backup(g):
        top = g["player_id"].rolling(30, min_periods=10).apply(lambda x: pd.Series(x).mode().iloc[0], raw=False)
        return (g["player_id"] != top.shift(1)).where(top.shift(1).notna())
    s["is_backup"] = s.groupby("team", group_keys=False).apply(_is_backup)
    s = s.merge(rest, on=["game_id", "team"])
    return s.groupby("back_to_back")["is_backup"].mean()


def expected_goalie_effect(candidates, mean_goals=3.1):
    """Expected log-lambda effect when the starter is uncertain.
    candidates: list of (rating_gpg, probability). Probabilities should sum to 1.
    Example: [(+0.30, 0.75), (-0.10, 0.25)]  -> starter 75% / backup 25%.
    (Approximation: averages in log-lambda space; fine for effects this small.)"""
    tot = sum(p for _, p in candidates)
    return sum(r * p for r, p in candidates) / tot / mean_goals
