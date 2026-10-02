"""
nhl_model.py
-------------
Shared model code for the local refresh pipeline AND the Streamlit app.

  train_artifact(data_dir)  -> dict     reads data/*.parquet|csv, fits the model, returns a
                                        JSON-safe "artifact" (coefficients, goalie ratings, ...)
  walk_forward(data_dir)    -> dict     leak-free backtest (used by refresh.py --backtest)
  save_artifact / load_artifact         tiny JSON file; this is the ONLY thing the app needs
  Predictor(artifact)                   turns the artifact into predictions for any matchup

Model (v3)
  * Target = REGULATION goals (periods 1-3). The final score is regulation + exactly one goal when
    regulation ends tied, so every full-game market is built from a joint regulation-score grid.
    This removes shootout noise from the ratings and prices OT/SO correctly.
  * Margin calibration: independent Poisson under-predicts regulation ties (~16% vs ~22%) and
    over-predicts 1-goal margins relative to 2-goal ones (empty-net goals). The joint grid is
    re-weighted by |regulation margin| with factors fitted to the data.
  * Season carryover: games from earlier seasons get extra down-weighting AND the ridge penalty is
    scaled up when little current-season data exists, so ratings regress toward average at each
    new season instead of carrying last year's roster forward unchanged.
  * Rest: back-to-back flags for the team and its opponent.
  * Shots: team ratings can be blended with a shots-based model (shots stabilise faster than goals).
  * ARI history is merged into UTA (same roster moved).

Depends on goalie_features.py (fitter + goalie ratings).
"""

import json
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson, skellam

from goalie_features import (PoissonTeamModel, estimate_prior_strength, build_goalie_ratings,
                             starter_table, attach_opp_goalie_effect, expected_goalie_effect)

ARTIFACT_VERSION = 3
DEFAULT_HALF_LIFE = 200      # days
DEFAULT_ALPHA = 0.03         # ridge strength on team ratings (per unit of current-season weight)
DEFAULT_CARRYOVER = 0.7      # weight multiplier per season boundary crossed (1 = off); walk-forward tuned
DEFAULT_SHOT_BLEND = 0.25    # share of team ratings from the shots model (0 = off); walk-forward tuned
TEAM_ALIASES = {"ARI": "UTA"}
MAX_GOALS = 12
MARGIN_BUCKETS = 5           # |margin| 0,1,2,3,4+
REST_COLS = ["b2b", "opp_b2b"]


# ======================================================================================
# Market math: joint regulation grid -> every market
# ======================================================================================

class GameDist:
    """Joint distribution of a game. P[h, a] = regulation score; F[h, a] = final score
    (a regulation tie is settled by one goal: home with prob home_ot_win)."""

    def __init__(self, lam_h, lam_a, home_ot_win, grid_w=None):
        k, D, T = _grid_index()
        P = np.outer(poisson.pmf(k, lam_h), poisson.pmf(k, lam_a))
        if grid_w:
            P = P * np.asarray(grid_w["margin"])[D] * np.asarray(grid_w["total"])[T]
        P /= P.sum()
        n = MAX_GOALS + 2
        F = np.zeros((n, n))
        F[:-1, :-1] = P
        diag = np.diag(P).copy()
        F[k, k] = 0.0
        F[k + 1, k] += diag * home_ot_win
        F[k, k + 1] += diag * (1 - home_ot_win)
        self.P, self.F, self.ot = P, F, home_ot_win
        h, a = np.indices(P.shape)
        self.p_home_reg = float(P[h > a].sum())
        self.p_away_reg = float(P[h < a].sum())
        self.p_tie = float(np.trace(P))
        self.p_home_win = self.p_home_reg + self.p_tie * home_ot_win
        hf, af = np.indices(F.shape)
        self._margin, self._total, self._hf, self._af = hf - af, hf + af, hf, af
        self.exp_home = float((F * hf).sum())
        self.exp_away = float((F * af).sum())

    def margin_pmf(self):
        """Final home margin -> probability."""
        d = np.arange(-(MAX_GOALS + 1), MAX_GOALS + 2)
        return d, np.array([self.F[self._margin == x].sum() for x in d])

    def total_pmf(self):
        t = np.arange(0, 2 * MAX_GOALS + 3)
        return t, np.bincount(self._total.ravel(), weights=self.F.ravel(), minlength=len(t))[:len(t)]

    @staticmethod
    def _ou(values, probs, line):
        over = float(probs[values > line + 1e-9].sum())
        push = float(probs[np.abs(values - line) < 1e-9].sum())
        return {"line": float(line), "over": over, "under": float(max(0.0, 1 - over - push)), "push": push}

    def total(self, line):
        return self._ou(*self.total_pmf(), line)

    def team_total(self, side, line):
        m = self.F.sum(axis=1) if side == "home" else self.F.sum(axis=0)
        return self._ou(np.arange(len(m)), m, line)

    def spread(self, home_spread):
        """Standard convention: -1.5 = home lays 1.5. Settled on the final score incl OT/SO."""
        d, p = self.margin_pmf()
        adj = d + home_spread
        cover, push = float(p[adj > 1e-9].sum()), float(p[np.abs(adj) < 1e-9].sum())
        return {"line": float(home_spread), "cover": cover, "push": push, "fail": float(max(0.0, 1 - cover - push))}

    def btts(self):
        return float(self.F[(self._hf > 0) & (self._af > 0)].sum())


def btts_prob(lam_a, lam_b):
    p0a, p0b = poisson.pmf(0, lam_a), poisson.pmf(0, lam_b)
    return float(1 - p0a - p0b + p0a * p0b)


TOTAL_BUCKETS = 12           # regulation total 0..10, 11+


def _grid_index():
    k = np.arange(MAX_GOALS + 1)
    D = np.minimum(np.abs(k[:, None] - k[None, :]), MARGIN_BUCKETS - 1)
    T = np.minimum(k[:, None] + k[None, :], TOTAL_BUCKETS - 1)
    return k, D, T


def fit_grid_weights(lam_h, lam_a, home_reg, away_reg, game_w=None, iters=40):
    """Multiplicative weights on |regulation margin| and on regulation total, fitted jointly by
    iterative proportional fitting so the model's average margin AND total frequencies match the data.
    (Margin alone fixes ties/empty-net margins but inflates totals; this keeps totals honest.)
    game_w: recency weights, so the weights learn distribution SHAPE where the ratings are calibrated,
    not scoring-level drift from older seasons."""
    k, D, T = _grid_index()
    base = poisson.pmf(k[None, :, None], np.asarray(lam_h)[:, None, None]) * \
        poisson.pmf(k[None, None, :], np.asarray(lam_a)[:, None, None])        # games x 13 x 13
    gw = np.ones(len(base)) if game_w is None else np.asarray(game_w, float)
    gw = gw / gw.sum()
    m_act = np.bincount(np.minimum(np.abs(home_reg - away_reg), MARGIN_BUCKETS - 1), weights=gw, minlength=MARGIN_BUCKETS)
    t_act = np.bincount(np.minimum(home_reg + away_reg, TOTAL_BUCKETS - 1), weights=gw, minlength=TOTAL_BUCKETS)
    m_act, t_act = m_act / m_act.sum(), t_act / t_act.sum()
    wm, wt = np.ones(MARGIN_BUCKETS), np.ones(TOTAL_BUCKETS)
    oD = [(D == j) for j in range(MARGIN_BUCKETS)]
    oT = [(T == j) for j in range(TOTAL_BUCKETS)]
    for _ in range(iters):
        for which in ("m", "t"):
            G = base * wm[D] * wt[T]
            G = G / G.sum(axis=(1, 2), keepdims=True)
            if which == "m":
                pred = np.array([G[:, o].sum(axis=1) @ gw for o in oD])
                wm *= np.where(pred > 0, m_act / np.maximum(pred, 1e-12), 1.0)
                wm /= wm[1]
            else:
                pred = np.array([G[:, o].sum(axis=1) @ gw for o in oT])
                wt *= np.where((pred > 1e-6) & (t_act > 0), t_act / np.maximum(pred, 1e-12), 1.0)
                wt /= wt[6]
    return {"margin": wm.tolist(), "total": wt.tolist()}


# ======================================================================================
# Data preparation
# ======================================================================================

def read_table(data_dir, name, required=True):
    data_dir = Path(data_dir)
    for ext, reader in ((".parquet", pd.read_parquet), (".csv", pd.read_csv)):
        p = data_dir / f"{name}{ext}"
        if not p.exists():
            continue
        try:
            return reader(p)
        except ImportError:  # pyarrow missing -> use the CSV twin
            continue
    if required:
        raise FileNotFoundError(f"{name}.parquet/.csv not found in {Path(data_dir).resolve()}")
    return None


def season_start(season):
    return int(season) // 10000


def season_for_date(d):
    d = pd.Timestamp(d)
    return d.year if d.month >= 7 else d.year - 1


def _alias(s):
    return s.replace(TEAM_ALIASES)


def _clean(obj):
    """Make nested data JSON-safe (numpy -> python, NaN -> None)."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    return obj


def load_data(data_dir="data", playoff_weight=0.5, use_goalies="auto", min_goalie_coverage=0.95, log=print):
    """Everything the fit and the backtest need, in one dict."""
    games = read_table(data_dir, "games")
    games["game_date"] = pd.to_datetime(games["game_date"])
    games["season"] = games["season"].astype(int)
    for c in ("home_team", "away_team"):
        games[c] = _alias(games[c])
    periods = read_table(data_dir, "game_periods", required=False)
    gk_raw = read_table(data_dir, "game_goalies", required=False)

    fin = games[games["game_state"].isin(["OFF", "FINAL"])].dropna(subset=["home_score", "away_score"]).copy()
    fin = fin[fin["game_type"].isin([2, 3])].sort_values(["game_date", "game_id"]).reset_index(drop=True)
    fin["type_weight"] = np.where(fin["game_type"] == 3, playoff_weight, 1.0)

    # ---- regulation goals (periods 1-3) ----
    fin["home_reg"], fin["away_reg"] = np.nan, np.nan
    if periods is not None and not periods.empty:
        pr = periods[periods["period"] <= 3].copy()
        pr["is_home"] = pr["is_home"].astype(str).str.lower().isin(["true", "1"])
        reg = pr.groupby(["game_id", "is_home"])["goals"].sum().unstack()
        if True in reg.columns:
            fin["home_reg"] = fin["game_id"].map(reg[True])
        if False in reg.columns:
            fin["away_reg"] = fin["game_id"].map(reg[False])
    hs, as_ = fin["home_score"], fin["away_score"]
    # usable = regulation score is consistent with the final (equal, or tied + one deciding goal)
    tied_ok = (fin["home_reg"] == fin["away_reg"]) & ((hs - fin["home_reg"]) + (as_ - fin["away_reg"]) == 1)
    same_ok = (fin["home_reg"] == hs) & (fin["away_reg"] == as_)
    bad = ~(tied_ok | same_ok)
    fin.loc[bad, "home_reg"] = hs[bad]
    fin.loc[bad, "away_reg"] = as_[bad]
    if bad.any():
        log(f"  {int(bad.sum())} games lack consistent period data - using final score as regulation score")
    fin["home_reg"] = fin["home_reg"].astype(int)
    fin["away_reg"] = fin["away_reg"].astype(int)

    ot_games = fin[(fin["game_type"] == 2) & (fin["home_reg"] == fin["away_reg"])]
    home_ot_win = float((ot_games["home_score"] > ot_games["away_score"]).mean()) if len(ot_games) > 100 else 0.518

    # ---- back-to-backs ----
    sched = pd.concat([fin[["game_id", "game_date", "home_team"]].rename(columns={"home_team": "team"}),
                       fin[["game_id", "game_date", "away_team"]].rename(columns={"away_team": "team"})])
    sched = sched.sort_values(["team", "game_date", "game_id"])
    sched["b2b"] = (sched.groupby("team")["game_date"].diff().dt.days == 1).astype(int)
    b2b = sched.set_index(["game_id", "team"])["b2b"]
    fin["home_b2b"] = b2b.reindex(list(zip(fin["game_id"], fin["home_team"]))).fillna(0).astype(int).values
    fin["away_b2b"] = b2b.reindex(list(zip(fin["game_id"], fin["away_team"]))).fillna(0).astype(int).values

    # ---- goalies ----
    goalies_on, gk_all, coverage = False, None, 0.0
    if gk_raw is not None:
        gk_raw = gk_raw[gk_raw["game_type"].isin([2, 3])].copy()
        gk_raw["game_date"] = pd.to_datetime(gk_raw["game_date"])
        gk_raw["season"] = gk_raw["season"].astype(int)
        gk_raw["team"] = _alias(gk_raw["team"])
        gk_raw["opponent"] = _alias(gk_raw["opponent"])
        gk_raw["starter"] = gk_raw["starter"].astype(str).str.lower().isin(["true", "1"])
        gk_raw = gk_raw.drop_duplicates(["game_id", "player_id"], keep="last")
        if use_goalies is not False:
            n_st = gk_raw[gk_raw["starter"]].groupby("game_id")["team"].nunique()
            good = set(n_st[n_st == 2].index)
            coverage = float(fin["game_id"].isin(good).mean())
            gk_all = gk_raw[gk_raw["game_id"].isin(good)].copy()
            goalies_on = (use_goalies is True) or coverage >= min_goalie_coverage
            if not goalies_on:
                log(f"goalies: OFF - only {coverage:.1%} of games have goalie rows (need {min_goalie_coverage:.0%})")
    else:
        log("goalies: OFF - no game_goalies file")

    # ---- long format (one row per team-game) ----
    cols = ["game_id", "season", "game_date", "team", "opponent", "goals", "type_weight", "is_home", "b2b", "opp_b2b"]
    h = fin.assign(team=fin["home_team"], opponent=fin["away_team"], goals=fin["home_reg"], is_home=1,
                   b2b=fin["home_b2b"], opp_b2b=fin["away_b2b"])[cols]
    a = fin.assign(team=fin["away_team"], opponent=fin["home_team"], goals=fin["away_reg"], is_home=0,
                   b2b=fin["away_b2b"], opp_b2b=fin["home_b2b"])[cols]
    long_base = pd.concat([h, a], ignore_index=True)

    # shots for = the opposing goalies' shots against (NaN where goalie data is missing)
    if gk_raw is not None:
        sf = gk_raw.groupby(["game_id", "opponent"])["shots_against"].sum()
        long_base["shots"] = sf.reindex(list(zip(long_base["game_id"], long_base["team"]))).values
    else:
        long_base["shots"] = np.nan

    teams = sorted(set(long_base["team"]) | set(long_base["opponent"]))
    mean_goals = float(long_base.loc[long_base["type_weight"] == 1.0, "goals"].mean())

    k_shots = spg = None
    if goalies_on:
        spg = float(gk_all.groupby(["game_id", "team"])["shots_against"].sum().mean())
        k_shots, tau, _ = estimate_prior_strength(gk_all)
        rated = build_goalie_ratings(gk_all, k_shots=k_shots, shots_per_game=spg)
        long_base = attach_opp_goalie_effect(long_base, starter_table(rated), mean_goals=mean_goals)
        log(f"goalies: ON ({coverage:.1%} coverage) | k={k_shots:,.0f} shots, talent SD {tau:.4f} sv%")

    return {"games": games, "fin": fin, "long": long_base, "periods": periods, "gk": gk_all,
            "teams": teams, "mean_goals": mean_goals, "home_ot_win": home_ot_win,
            "goalies_on": goalies_on, "coverage": coverage, "k_shots": k_shots, "spg": spg}


# ======================================================================================
# Fitting
# ======================================================================================

def _weights(df, as_of, half_life, carryover):
    days_ago = (as_of - df["game_date"]).dt.days.clip(lower=0)
    base = 0.5 ** (days_ago / half_life) * df["type_weight"]
    seasons_back = (season_for_date(as_of) - df["season"].map(season_start)).clip(lower=0)
    return base, base * carryover ** seasons_back


def fit_ratings(train, teams, as_of, half_life=DEFAULT_HALF_LIFE, alpha=DEFAULT_ALPHA,
                carryover=DEFAULT_CARRYOVER, shot_blend=DEFAULT_SHOT_BLEND, goalies=False):
    """Fit goals (and optionally shots) models on `train` (rows strictly before as_of).
    Returns (beta, extra_cols, alpha_eff, diagnostics)."""
    base, w = _weights(train, as_of, half_life, carryover)
    # The fit is normalised by sum(w). Scaling alpha by sum(base)/sum(w) keeps the penalty fixed relative
    # to the undiscounted data, so discounting old seasons genuinely shrinks ratings toward average.
    alpha_eff = alpha * float(base.sum()) / max(float(w.sum()), 1e-9)
    df = train.assign(fit_weight=w.values)
    extra = REST_COLS + (["opp_goalie_effect"] if goalies else [])
    gm = PoissonTeamModel(teams, alpha=alpha_eff, extra_cols=extra).fit(df)
    beta = gm.beta.copy()
    T = len(teams)
    diag = {"home_coef": float(gm.coef("is_home")), "b2b_coef": float(gm.coef("b2b")),
            "opp_b2b_coef": float(gm.coef("opp_b2b")),
            "goalie_beta": float(gm.coef("opp_goalie_effect")) if goalies else None}
    if shot_blend > 0:
        s = df.dropna(subset=["shots"])
        if len(s) > 200:
            sm = PoissonTeamModel(teams, alpha=alpha_eff, extra_cols=REST_COLS).fit(s.assign(goals=s["shots"]))
            beta[1:1 + 2 * T] = (1 - shot_blend) * gm.beta[1:1 + 2 * T] + shot_blend * sm.beta[1:1 + 2 * T]
    return beta, extra, alpha_eff, diag


def _recency(dates, as_of, half_life):
    return 0.5 ** ((as_of - pd.to_datetime(dates)).dt.days.clip(lower=0) / half_life).values


def _predict_rows(beta, teams, extra, rows):
    m = PoissonTeamModel(teams, extra_cols=extra)
    m.beta = np.asarray(beta)
    return m.predict(rows)


def _home_away_lams(beta, teams, extra, long_rows, game_ids):
    ev = long_rows[long_rows["game_id"].isin(game_ids)]
    evh = ev[ev["is_home"] == 1].set_index("game_id").loc[game_ids]
    eva = ev[ev["is_home"] == 0].set_index("game_id").loc[game_ids]
    return _predict_rows(beta, teams, extra, evh), _predict_rows(beta, teams, extra, eva)


def period_shares_shrunk(periods, teams, game_ids=None):
    """Each team's share of regulation goals by period, shrunk toward the league split. The prior
    strength K (in goals) comes from how much teams genuinely differ (method of moments)."""
    even = {"1": 1 / 3, "2": 1 / 3, "3": 1 / 3}
    if periods is None or periods.empty:
        return {t: dict(even) for t in teams}, None
    reg = periods[periods["period"].isin([1, 2, 3])].copy()
    if game_ids is not None:
        reg = reg[reg["game_id"].isin(set(game_ids))]
    reg["team"] = _alias(reg["team"])
    reg = reg[reg["team"].isin(teams)]
    tot = reg.groupby(["team", "period"])["goals"].sum().unstack(fill_value=0)
    league = tot.sum() / tot.values.sum()
    n = tot.sum(axis=1)
    Ks = []
    for p in tot.columns:
        share, pl = tot[p] / n, league[p]
        tau2 = float(np.average((share - pl) ** 2, weights=n) - np.average(pl * (1 - pl) / n, weights=n))
        Ks.append(pl * (1 - pl) / tau2 if tau2 > 0 else 1e6)
    K = float(np.clip(np.median(Ks), 200, 1e6))
    out = {}
    for t in teams:
        sh = (tot.loc[t] + K * league) / (n[t] + K) if t in tot.index else league
        out[t] = {str(int(p)): float(v) for p, v in sh.items()}
    return out, K


def b2b_same_goalie_rate(gk):
    """League rate at which the previous night's starter also starts the 2nd game of a back-to-back."""
    s = gk[gk["starter"]].sort_values(["team", "game_date", "game_id"]).drop_duplicates(["game_id", "team"])
    prev_date = s.groupby("team")["game_date"].shift()
    prev_pid = s.groupby("team")["player_id"].shift()
    b = (s["game_date"] - prev_date).dt.days == 1
    return float((s.loc[b, "player_id"] == prev_pid[b]).mean()) if b.sum() > 50 else 0.2


# ======================================================================================
# Training -> artifact
# ======================================================================================

def train_artifact(data_dir="data", half_life_days=DEFAULT_HALF_LIFE, alpha=DEFAULT_ALPHA,
                   carryover=DEFAULT_CARRYOVER, shot_blend=DEFAULT_SHOT_BLEND,
                   playoff_weight=0.5, use_goalies="auto", min_goalie_coverage=0.95,
                   pool_starts=25, as_of=None, upcoming=None, log=print):
    D = load_data(data_dir, playoff_weight, use_goalies, min_goalie_coverage, log)
    fin, long_base, teams = D["fin"], D["long"], D["teams"]
    last_date = fin["game_date"].max()
    # "as of" = later of the last game and today: in the off-season the new season already counts as current
    as_of = pd.Timestamp(as_of) if as_of is not None else max(last_date, pd.Timestamp(date.today()))
    recent = fin[fin["game_date"] >= last_date - pd.Timedelta(days=365)]
    active_teams = sorted(set(recent["home_team"]) | set(recent["away_team"]))
    log(f"games: {len(fin):,} through {last_date.date()} | {len(active_teams)} active teams | as of {as_of.date()}")

    beta, extra, alpha_eff, diag = fit_ratings(long_base, teams, as_of, half_life_days, alpha,
                                               carryover, shot_blend, D["goalies_on"])
    log(f"fit: home {diag['home_coef']:+.3f} | b2b {diag['b2b_coef']:+.3f} | opp b2b {diag['opp_b2b_coef']:+.3f}"
        + (f" | goalie beta {diag['goalie_beta']:+.2f}" if D["goalies_on"] else "")
        + f" | effective alpha {alpha_eff:.3f}")

    gids = fin["game_id"].values
    lh, la = _home_away_lams(beta, teams, extra, long_base, gids)
    grid_w = fit_grid_weights(lh, la, fin["home_reg"].values, fin["away_reg"].values,
                              _recency(fin["game_date"], as_of, half_life_days) * fin["type_weight"].values)
    log("margin weights for |reg margin| 0,1,2,3,4+: " + ", ".join(f"{x:.2f}" for x in grid_w["margin"]))

    shares, share_K = period_shares_shrunk(D["periods"], teams, fin["game_id"])

    cur = season_for_date(as_of)
    cs = fin[(fin["season"].map(season_start) == cur) & (fin["game_type"] == 2)]
    gp_all = pd.concat([cs["home_team"], cs["away_team"]]).value_counts()
    gp = {t: int(gp_all.get(t, 0)) for t in active_teams}

    goalie_block = None
    if D["goalies_on"]:
        g = D["gk"].sort_values(["game_date", "game_id"])
        k_shots, spg = D["k_shots"], D["spg"]
        lg = g["saves"].sum() / g["shots_against"].sum()
        t = g.groupby("player_id").agg(name=("name", "last"), team=("team", "last"), last_game=("game_date", "max"),
                                       starts=("starter", "sum"), saves=("saves", "sum"), shots=("shots_against", "sum"))
        t["sv_pct"] = t["saves"] / t["shots"].replace(0, np.nan)
        t["rating_gpg"] = ((t["saves"] + k_shots * lg) / (t["shots"] + k_shots) - lg) * spg
        table = t.reset_index()[["player_id", "name", "team", "last_game", "starts", "sv_pct", "rating_gpg"]]
        table["last_game"] = table["last_game"].dt.strftime("%Y-%m-%d")
        st = g[g["starter"]].drop_duplicates(["game_id", "team"])
        pools, last_starter = {}, {}
        for tm in teams:
            r = st[st["team"] == tm].tail(pool_starts)
            pools[tm] = r["player_id"].value_counts(normalize=True).to_dict() if len(r) else {}
            if len(r):
                last_starter[tm] = {"player_id": int(r["player_id"].iloc[-1]),
                                    "date": r["game_date"].iloc[-1].strftime("%Y-%m-%d")}
        goalie_block = {"k_shots": k_shots, "shots_per_game": spg, "table": table.to_dict("records"),
                        "pools": pools, "last_starter": last_starter, "b2b_same_goalie": b2b_same_goalie_rate(g)}

    last_played = pd.concat([fin[["game_date", "home_team"]].rename(columns={"home_team": "t"}),
                             fin[["game_date", "away_team"]].rename(columns={"away_team": "t"})]
                            ).groupby("t")["game_date"].max().dt.strftime("%Y-%m-%d").to_dict()
    slate = build_slate(upcoming or [], last_played, last_date)

    artifact = {
        "version": ARTIFACT_VERSION,
        "meta": {"fitted_at": datetime.now().isoformat(timespec="seconds"), "trained_through": str(last_date.date()),
                 "as_of": str(as_of.date()), "n_games": int(len(fin)), "half_life_days": half_life_days,
                 "alpha_l2": alpha, "alpha_eff": alpha_eff, "carryover": carryover, "shot_blend": shot_blend,
                 "goalies_on": D["goalies_on"], "goalie_coverage": D["coverage"], "goalie_beta": diag["goalie_beta"],
                 "home_coef": diag["home_coef"], "b2b_coef": diag["b2b_coef"], "opp_b2b_coef": diag["opp_b2b_coef"],
                 "period_share_K": share_K, "active_teams": active_teams, "season_gp": gp,
                 "team_aliases": TEAM_ALIASES},
        "teams": teams, "beta": beta.tolist(), "extra_cols": extra,
        "home_ot_win": D["home_ot_win"], "mean_goals": D["mean_goals"], "grid_weights": grid_w,
        "period_shares": shares, "goalies": goalie_block, "last_played": last_played, "slate": slate,
    }
    return _clean(artifact)


def build_slate(upcoming, last_played, last_date):
    """upcoming: list of {game_id, date, start_utc, home, away}. Adds back-to-back flags using each
    team's last completed game and any earlier game in the upcoming list itself."""
    out, played = [], dict(last_played)
    for g in sorted(upcoming, key=lambda g: (g["date"], g.get("start_utc") or "")):
        if pd.Timestamp(g["date"]) <= last_date:
            continue
        home, away = TEAM_ALIASES.get(g["home"], g["home"]), TEAM_ALIASES.get(g["away"], g["away"])
        flags = {}
        for side, tm in (("home", home), ("away", away)):
            prev = played.get(tm)
            flags[f"{side}_b2b"] = bool(prev and (pd.Timestamp(g["date"]) - pd.Timestamp(prev)).days == 1)
        out.append({**g, "home": home, "away": away, **flags})
        played[home] = played[away] = g["date"]
    return out


# ======================================================================================
# Walk-forward backtest
# ======================================================================================

def walk_forward(data_dir="data", n_seasons=3, half_life_days=DEFAULT_HALF_LIFE, alpha=DEFAULT_ALPHA,
                 carryover=DEFAULT_CARRYOVER, shot_blend=DEFAULT_SHOT_BLEND, goalies="auto",
                 margin_cal=True, rest=True, refit_days=30, D=None, log=print):
    """Refit every `refit_days` using only earlier games and predict the next block. Scores regular-season
    games in the last n_seasons. Goalie ratings are leak-free (each uses only earlier games)."""
    D = D or load_data(data_dir, use_goalies=goalies, log=log)
    fin, teams = D["fin"], D["teams"]
    long_base = D["long"] if rest else D["long"].assign(b2b=0, opp_b2b=0)
    use_g = D["goalies_on"] and goalies is not False
    seasons = sorted(fin["season"].unique())
    complete = [s for s in seasons if (fin["season"] == s).sum() > 500]
    tests = [s for s in complete[-n_seasons:] if s != seasons[0]]
    rows = []
    for s in tests:
        sg = fin[(fin["season"] == s) & (fin["game_type"] == 2)]
        edges = pd.date_range(sg["game_date"].min(), sg["game_date"].max() + pd.Timedelta(days=refit_days + 1),
                              freq=f"{refit_days}D")
        for a, b in zip(edges[:-1], edges[1:]):
            chunk = sg[(sg["game_date"] >= a) & (sg["game_date"] < b)]
            if chunk.empty:
                continue
            train = long_base[long_base["game_date"] < a]
            beta, extra, _, _ = fit_ratings(train, teams, a, half_life_days, alpha, carryover, shot_blend, use_g)
            mw = None
            if margin_cal:
                tg = fin[fin["game_date"] < a].tail(4000)
                tlh, tla = _home_away_lams(beta, teams, extra, train, tg["game_id"].values)
                mw = fit_grid_weights(tlh, tla, tg["home_reg"].values, tg["away_reg"].values,
                                      _recency(tg["game_date"], a, half_life_days) * tg["type_weight"].values)
            lh, la = _home_away_lams(beta, teams, extra, long_base, chunk["game_id"].values)
            for i, g in enumerate(chunk.itertuples()):
                gd = GameDist(lh[i], la[i], D["home_ot_win"], mw)
                tot = g.home_score + g.away_score
                rows.append({"season": s, "date": g.game_date, "lam_tot": lh[i] + la[i],
                             "reg_tot": g.home_reg + g.away_reg, "exp_final": gd.exp_home + gd.exp_away,
                             "act_final": tot, "y_win": int(g.home_score > g.away_score),
                             "p_win": gd.p_home_win, "p_h": gd.p_home_reg, "p_t": gd.p_tie, "p_a": gd.p_away_reg,
                             "y_reg": 0 if g.home_reg > g.away_reg else (1 if g.home_reg == g.away_reg else 2),
                             "p_o55": gd.total(5.5)["over"], "p_o65": gd.total(6.5)["over"],
                             "y_o55": int(tot > 5.5), "y_o65": int(tot > 6.5),
                             "p_cover": gd.spread(-1.5)["cover"], "y_cover": int(g.home_score - g.away_score >= 2),
                             "p_acover": gd.spread(-1.5)["fail"], "y_acover": int(g.home_score - g.away_score >= -1)})
    r = pd.DataFrame(rows)
    first = r.groupby("season")["date"].transform("min")
    r["early"] = (r["date"] - first).dt.days < 45
    return summarize_backtest(r, tests)


def summarize_backtest(r, tests):
    def brier(p, y): return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))
    def ll(p, y):
        p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6); y = np.asarray(y)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    preg = r[["p_h", "p_t", "p_a"]].values
    ll3 = float(-np.mean(np.log(np.clip(preg[np.arange(len(r)), r["y_reg"].values], 1e-6, 1))))
    return {"seasons": [str(s) for s in tests], "n_games": int(len(r)),
            "win_brier": brier(r["p_win"], r["y_win"]), "win_logloss": ll(r["p_win"], r["y_win"]),
            "const_brier": brier(np.full(len(r), r["y_win"].mean()), r["y_win"]),
            "reg3_logloss": ll3, "pred_tie": float(r["p_t"].mean()), "actual_tie": float((r["y_reg"] == 1).mean()),
            "o55_brier": brier(r["p_o55"], r["y_o55"]), "o65_brier": brier(r["p_o65"], r["y_o65"]),
            "pred_o55": float(r["p_o55"].mean()), "actual_o55": float(r["y_o55"].mean()),
            "pred_o65": float(r["p_o65"].mean()), "actual_o65": float(r["y_o65"].mean()),
            "pl_brier": brier(r["p_cover"], r["y_cover"]),
            "early_win_brier": brier(r.loc[r["early"], "p_win"], r.loc[r["early"], "y_win"]),
            "early_n": int(r["early"].sum()),
            "pred_cover": float(r["p_cover"].mean()), "actual_cover": float(r["y_cover"].mean()),
            "raw": r}


# ======================================================================================
# Save / load
# ======================================================================================

def save_artifact(artifact, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(artifact, f, separators=(",", ":"))
    return path


def load_artifact(path):
    with open(path) as f:
        return json.load(f)


# ======================================================================================
# Prediction
# ======================================================================================

class Predictor:
    """Everything the app needs, from the artifact alone."""

    def __init__(self, art):
        if art.get("version", 0) < 3:
            raise ValueError("Model file is from an older version. Run `python refresh.py --skip-fetch` to rebuild it.")
        self.art, self.meta = art, art["meta"]
        self.teams = art["teams"]
        self.active_teams = self.meta.get("active_teams") or self.teams
        self.aliases = self.meta.get("team_aliases", {})
        self.home_ot_win, self.mean_goals = art["home_ot_win"], art["mean_goals"]
        self.grid_w = art.get("grid_weights")
        self.extra = art["extra_cols"]
        self.beta = np.array(art["beta"])
        self.period_shares = {t: {int(k): v for k, v in s.items()} for t, s in art["period_shares"].items()}
        self.slate = art.get("slate", [])
        gb = art.get("goalies")
        self.goalies_on = bool(gb)
        if self.goalies_on:
            self.gtable = pd.DataFrame(gb["table"]).set_index("player_id")
            self.pools = {t: {int(k): float(v) for k, v in p.items()} for t, p in gb["pools"].items()}
            self.last_starter = gb.get("last_starter", {})
            self.b2b_same = gb.get("b2b_same_goalie", 0.2)

    # ---------- goalies ----------
    def goalie_options(self, team, max_age_days=400):
        if not self.goalies_on:
            return []
        cutoff = pd.Timestamp(self.meta["trained_through"]) - pd.Timedelta(days=max_age_days)
        t = self.gtable
        ids = set(self.pools.get(team, {})) | set(t[(t["team"] == team) & (pd.to_datetime(t["last_game"]) >= cutoff)].index)
        rows = [{"player_id": int(i), "name": t.loc[i, "name"], "starts": int(t.loc[i, "starts"]),
                 "rating_gpg": float(t.loc[i, "rating_gpg"]), "usual_share": self.pools.get(team, {}).get(int(i), 0.0)}
                for i in ids if i in t.index]
        return sorted(rows, key=lambda r: (-r["usual_share"], -r["starts"]))

    def goalie_spec(self, team, player_id=None, confidence=1.0, b2b=False):
        """None player -> usual mix (shifted toward the backup on the 2nd night of a back-to-back).
        Otherwise {player_id: weight}; if confidence < 1 the rest goes to the team's other usual goalies."""
        if player_id is None:
            return self.b2b_pool(team) if (b2b and self.goalies_on) else None
        if confidence >= 0.999:
            return {int(player_id): 1.0}
        others = {k: v for k, v in self.pools.get(team, {}).items() if k != int(player_id)}
        tot = sum(others.values())
        if tot <= 0:
            return {int(player_id): 1.0}
        return {int(player_id): float(confidence), **{k: (1 - confidence) * v / tot for k, v in others.items()}}

    def b2b_pool(self, team):
        """Last night's starter goes again at the league back-to-back rate; the rest is spread
        over the team's other usual goalies."""
        ls, pool = self.last_starter.get(team), self.pools.get(team, {})
        if not ls or not pool:
            return None
        pid = int(ls["player_id"])
        others = {k: v for k, v in pool.items() if k != pid}
        tot = sum(others.values())
        if tot <= 0:
            return None
        return {pid: self.b2b_same, **{k: (1 - self.b2b_same) * v / tot for k, v in others.items()}}

    def _effect(self, team, spec):
        if spec is None:
            probs, label = dict(self.pools.get(team, {})), "usual mix"
        else:
            probs, label = {int(k): float(v) for k, v in spec.items()}, "selected"
        if not probs:
            return 0.0, []
        tot = sum(probs.values())
        cands, info = [], []
        for pid, p in probs.items():
            r = float(self.gtable.loc[pid, "rating_gpg"]) if pid in self.gtable.index else 0.0
            nm = self.gtable.loc[pid, "name"] if pid in self.gtable.index else str(pid)
            cands.append((r, p / tot))
            info.append({"team": team, "name": nm, "prob": p / tot, "rating_gpg": r, "basis": label})
        return expected_goalie_effect(cands, self.mean_goals), info

    # ---------- core ----------
    def resolve(self, team):
        team = self.aliases.get(team, team)
        if team not in self.teams:
            raise ValueError(f"Unknown team '{team}'.")
        return team

    def lambdas(self, home, away, home_goalie=None, away_goalie=None, home_b2b=False, away_b2b=False):
        """Expected REGULATION goals for each side."""
        home, away = self.resolve(home), self.resolve(away)
        rows = pd.DataFrame({"team": [home, away], "opponent": [away, home], "is_home": [1, 0],
                             "b2b": [int(home_b2b), int(away_b2b)], "opp_b2b": [int(away_b2b), int(home_b2b)]})
        info = []
        if self.goalies_on:
            eff_h, i_away = self._effect(away, away_goalie)   # away goalie suppresses HOME scoring
            eff_a, i_home = self._effect(home, home_goalie)
            rows["opp_goalie_effect"] = [eff_h, eff_a]
            info = i_home + i_away
        lam = _predict_rows(self.beta, self.teams, self.extra, rows)
        return float(lam[0]), float(lam[1]), info

    def predict_game(self, home, away, home_goalie=None, away_goalie=None, total_line=6.5, home_spread=-1.5,
                     home_team_line=2.5, away_team_line=2.5, home_b2b=False, away_b2b=False):
        lh, la, info = self.lambdas(home, away, home_goalie, away_goalie, home_b2b, away_b2b)
        gd = GameDist(lh, la, self.home_ot_win, self.grid_w)
        sh_h = self.period_shares.get(self.resolve(home), {})
        sh_a = self.period_shares.get(self.resolve(away), {})
        periods = []
        for p in (1, 2, 3):
            ph, pa = lh * sh_h.get(p, 1 / 3), la * sh_a.get(p, 1 / 3)
            periods.append({"period": p, "exp_home": ph, "exp_away": pa, "exp_total": ph + pa,
                            "btts": btts_prob(ph, pa), "over_0_5": float(1 - poisson.pmf(0, ph + pa)),
                            "over_1_5": float(1 - poisson.cdf(1, ph + pa))})
        d, mp = gd.margin_pmf()
        keep = np.abs(d) <= 5
        return {
            "home": home, "away": away, "lam_home": lh, "lam_away": la,
            "exp_home": gd.exp_home, "exp_away": gd.exp_away, "exp_total": gd.exp_home + gd.exp_away,
            "p_home_win": gd.p_home_win, "p_home_reg": gd.p_home_reg, "p_away_reg": gd.p_away_reg,
            "p_tie_reg": gd.p_tie,
            "total": gd.total(total_line),
            "home_total": gd.team_total("home", home_team_line), "away_total": gd.team_total("away", away_team_line),
            "spread": gd.spread(home_spread),
            "alt_spreads": {s: gd.spread(s) for s in (-2.5, -1.5, 1.5, 2.5)},
            "btts": gd.btts(), "periods": periods, "goalie_info": info,
            "margin_pmf": {"margin": d[keep].tolist(), "p": mp[keep].tolist()},
        }

    # ---------- display helpers ----------
    def team_ratings(self):
        T = len(self.teams)
        b = self.beta
        df = pd.DataFrame({"attack": b[1:1 + T], "defense": b[1 + T:1 + 2 * T]}, index=self.teams)
        df["net_rating"] = df["attack"] - df["defense"]
        gp = self.meta.get("season_gp", {})
        df["gp_this_season"] = [gp.get(t, 0) for t in df.index]
        return df.loc[[t for t in self.teams if t in self.active_teams]].sort_values("net_rating", ascending=False)

    def goalie_leaders(self, min_starts=20):
        if not self.goalies_on:
            return None
        cutoff = pd.Timestamp(self.meta["trained_through"]) - pd.Timedelta(days=400)
        t = self.gtable
        t = t[(t["starts"] >= min_starts) & (pd.to_datetime(t["last_game"]) >= cutoff)]
        return t.sort_values("rating_gpg", ascending=False)[["name", "team", "starts", "sv_pct", "rating_gpg"]]
