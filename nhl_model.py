"""
nhl_model.py
-------------
Shared model code for the local refresh pipeline AND the Streamlit app.

  train_artifact(data_dir)  -> dict     reads data/*.parquet|csv, fits the model, returns a
                                        JSON-safe "artifact" (coefficients, goalie ratings, ...)
  save_artifact / load_artifact         tiny JSON file; this is the ONLY thing the app needs
  Predictor(artifact)                   turns the artifact into predictions for any matchup

Depends on goalie_features.py (fitter + goalie ratings). No raw data is needed to predict.
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skellam, poisson

from goalie_features import (PoissonTeamModel, estimate_prior_strength, build_goalie_ratings,
                             starter_table, attach_opp_goalie_effect, expected_goalie_effect)

ARTIFACT_VERSION = 2
DEFAULT_HALF_LIFE = 200   # days; chosen by walk-forward sweep on 2023-26 (flat optimum ~150-250)
DEFAULT_ALPHA = 0.03      # ridge strength on team ratings; same sweep


# ======================================================================================
# Market math (shared by app + anything else)
# ======================================================================================

def ou_probs(lam, line):
    """Over / under / push for a Poisson count vs a line (half or whole number)."""
    over = 1 - poisson.cdf(int(np.floor(line)), lam)
    push = float(poisson.pmf(int(line), lam)) if float(line).is_integer() else 0.0
    return {"line": float(line), "over": float(over), "under": float(1 - over - push), "push": push}


def btts_prob(lam_a, lam_b):
    p0a, p0b = poisson.pmf(0, lam_a), poisson.pmf(0, lam_b)
    return float(1 - p0a - p0b + p0a * p0b)


def final_margin_pmf(lam_h, lam_a, home_ot_win, max_goals=15):
    """Distribution of the FINAL home margin (home - away), NHL-style: a game tied after
    regulation is decided in OT/SO by exactly one goal (home wins with prob home_ot_win)."""
    d = np.arange(-max_goals, max_goals + 1)
    pmf = skellam.pmf(d, lam_h, lam_a)
    tie = pmf[d == 0].copy()
    pmf[d == 0] = 0.0
    pmf[d == 1] += tie * home_ot_win
    pmf[d == -1] += tie * (1 - home_ot_win)
    return d, pmf


def spread_probs(lam_h, lam_a, home_spread, home_ot_win):
    """Home team covers `home_spread` in standard betting convention (-1.5 = home lays 1.5,
    +1.5 = home gets 1.5), settled on the final score including OT/SO."""
    d, pmf = final_margin_pmf(lam_h, lam_a, home_ot_win)
    adj = d + home_spread
    cover = float(pmf[adj > 1e-9].sum())
    push = float(pmf[np.abs(adj) < 1e-9].sum())
    return {"line": float(home_spread), "cover": cover, "push": push, "fail": float(1 - cover - push)}


def home_win_prob(lam_h, lam_a, home_ot_win):
    d, pmf = final_margin_pmf(lam_h, lam_a, home_ot_win)
    return float(pmf[d > 0].sum())


# ======================================================================================
# Training
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
        raise FileNotFoundError(f"{name}.parquet/.csv not found in {data_dir.resolve()}")
    return None


def _to_long(f):
    cols = ["game_id", "season", "game_date", "team", "opponent", "goals", "opp_goals", "type_weight", "is_home"]
    h = f.rename(columns={"home_team": "team", "away_team": "opponent",
                          "home_score": "goals", "away_score": "opp_goals"}).assign(is_home=1)[cols]
    a = f.rename(columns={"away_team": "team", "home_team": "opponent",
                          "away_score": "goals", "home_score": "opp_goals"}).assign(is_home=0)[cols]
    return pd.concat([h, a], ignore_index=True)


def _add_recency_weight(df, half_life_days, as_of=None):
    as_of = as_of if as_of is not None else df["game_date"].max()
    days_ago = (as_of - df["game_date"]).dt.days.clip(lower=0)
    return df.assign(fit_weight=0.5 ** (days_ago / half_life_days) * df["type_weight"])


def _clean(obj):
    """Make nested data JSON-safe (numpy -> python, NaN -> None)."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    return obj


def train_artifact(data_dir="data", half_life_days=DEFAULT_HALF_LIFE, alpha=DEFAULT_ALPHA,
                   playoff_weight=0.5, use_goalies="auto", min_goalie_coverage=0.95,
                   pool_starts=25, log=print):
    """Fit everything and return a JSON-safe artifact dict."""
    games = read_table(data_dir, "games")
    games["game_date"] = pd.to_datetime(games["game_date"])
    games["season"] = games["season"].astype(int)
    periods = read_table(data_dir, "game_periods", required=False)
    gk_raw = read_table(data_dir, "game_goalies", required=False)

    finished = games[games["game_state"].isin(["OFF", "FINAL"])].dropna(subset=["home_score", "away_score"]).copy()
    finished = finished[finished["game_type"].isin([2, 3])]
    finished["type_weight"] = np.where(finished["game_type"] == 3, playoff_weight, 1.0)
    long_base = _to_long(finished)
    teams = sorted(set(long_base["team"]) | set(long_base["opponent"]))
    mean_goals = float(long_base.loc[long_base["type_weight"] == 1.0, "goals"].mean())

    home_ot_win = 0.518
    if periods is not None and not periods.empty:
        went_ot = finished["game_id"].map(periods.groupby("game_id")["period"].max()) > 3
        ot = finished[went_ot & (finished["game_type"] == 2)]
        if len(ot) > 100:
            home_ot_win = float((ot["home_score"] > ot["away_score"]).mean())

    last_date = finished["game_date"].max()
    recent = finished[finished["game_date"] >= last_date - pd.Timedelta(days=365)]
    active_teams = sorted(set(recent["home_team"]) | set(recent["away_team"]))
    log(f"games: {len(finished):,} through {last_date.date()} | {len(teams)} teams ({len(active_teams)} active)")

    # ---- goalie mode decision ----
    goalies_on, gk_all, coverage = False, None, 0.0
    if gk_raw is not None and use_goalies is not False:
        gk_raw = gk_raw[gk_raw["game_type"].isin([2, 3])].copy()
        gk_raw["game_date"] = pd.to_datetime(gk_raw["game_date"])
        gk_raw["season"] = gk_raw["season"].astype(int)
        gk_raw = gk_raw.drop_duplicates(["game_id", "player_id"], keep="last")
        n_st = gk_raw[gk_raw["starter"]].groupby("game_id")["team"].nunique()
        good = set(n_st[n_st == 2].index)
        coverage = float(finished["game_id"].isin(good).mean())
        gk_all = gk_raw[gk_raw["game_id"].isin(good)].copy()
        goalies_on = (use_goalies is True) or coverage >= min_goalie_coverage
        if not goalies_on:
            log(f"goalies: OFF - only {coverage:.1%} of games have goalie rows (need {min_goalie_coverage:.0%})")
    elif gk_raw is None:
        log("goalies: OFF - no game_goalies file")
    else:
        log("goalies: OFF (disabled)")

    long_df = _add_recency_weight(long_base, half_life_days)
    extra, k_shots, spg, goalie_block = [], None, None, None
    if goalies_on:
        spg = float(gk_all.groupby(["game_id", "team"])["shots_against"].sum().mean())
        k_shots, tau, n_pairs = estimate_prior_strength(gk_all)
        rated = build_goalie_ratings(gk_all, k_shots=k_shots, shots_per_game=spg)
        long_df = attach_opp_goalie_effect(long_df, starter_table(rated), mean_goals=mean_goals)
        extra = ["opp_goalie_effect"]
        log(f"goalies: ON ({coverage:.1%} coverage) | k={k_shots:,.0f} shots, talent SD {tau:.4f} sv% ({n_pairs} repeat seasons)")

    model = PoissonTeamModel(teams, alpha=alpha, extra_cols=extra).fit(long_df)
    goalie_beta = float(model.coef("opp_goalie_effect")) if goalies_on else None
    log(f"fit: home-ice coef {model.coef('is_home'):.3f}" + (f" | goalie beta {goalie_beta:+.2f}" if goalies_on else ""))

    # ---- period shares ----
    even = {"1": 1 / 3, "2": 1 / 3, "3": 1 / 3}
    shares = {t: dict(even) for t in teams}
    if periods is not None and not periods.empty:
        reg = periods[periods["period"].isin([1, 2, 3])]
        tot = reg.groupby(["team", "period"])["goals"].sum().unstack(fill_value=0)
        for t in teams:
            if t in tot.index and tot.loc[t].sum() > 0:
                shares[t] = {str(int(p)): float(v) for p, v in (tot.loc[t] / tot.loc[t].sum()).items()}

    # ---- goalie table + each team's recent starter mix ----
    if goalies_on:
        g = gk_all.sort_values(["game_date", "game_id"])
        lg = g["saves"].sum() / g["shots_against"].sum()
        t = g.groupby("player_id").agg(name=("name", "last"), team=("team", "last"), last_game=("game_date", "max"),
                                       starts=("starter", "sum"), saves=("saves", "sum"), shots=("shots_against", "sum"))
        t["sv_pct"] = t["saves"] / t["shots"].replace(0, np.nan)
        t["rating_gpg"] = ((t["saves"] + k_shots * lg) / (t["shots"] + k_shots) - lg) * spg
        table = t.reset_index()[["player_id", "name", "team", "last_game", "starts", "sv_pct", "rating_gpg"]]
        table["last_game"] = table["last_game"].dt.strftime("%Y-%m-%d")
        pools = {}
        for tm in teams:
            r = g[(g["team"] == tm) & g["starter"]].tail(pool_starts)
            pools[tm] = r["player_id"].value_counts(normalize=True).to_dict() if len(r) else {}
        goalie_block = {"k_shots": k_shots, "shots_per_game": spg, "table": table.to_dict("records"), "pools": pools}

    artifact = {
        "version": ARTIFACT_VERSION,
        "meta": {"fitted_at": datetime.now().isoformat(timespec="seconds"), "trained_through": str(last_date.date()),
                 "n_games": int(len(finished)), "half_life_days": half_life_days, "alpha_l2": alpha,
                 "goalies_on": goalies_on, "goalie_coverage": coverage, "goalie_beta": goalie_beta,
                 "active_teams": active_teams},
        "teams": teams, "beta": model.beta.tolist(), "extra_cols": extra,
        "home_ot_win": home_ot_win, "mean_goals": mean_goals,
        "period_shares": shares, "goalies": goalie_block,
    }
    return _clean(artifact)


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
        self.art, self.meta = art, art["meta"]
        self.teams = art["teams"]
        self.active_teams = self.meta.get("active_teams") or self.teams
        self.home_ot_win, self.mean_goals = art["home_ot_win"], art["mean_goals"]
        self.period_shares = {t: {int(k): v for k, v in s.items()} for t, s in art["period_shares"].items()}
        self.model = PoissonTeamModel(self.teams, alpha=self.meta["alpha_l2"], extra_cols=art["extra_cols"])
        self.model.beta = np.array(art["beta"])
        gb = art.get("goalies")
        self.goalies_on = bool(gb)
        if self.goalies_on:
            self.gtable = pd.DataFrame(gb["table"]).set_index("player_id")
            self.pools = {t: {int(k): float(v) for k, v in p.items()} for t, p in gb["pools"].items()}

    # ---------- goalies ----------
    def goalie_options(self, team, max_age_days=400):
        """Goalies who could plausibly start for `team`, most likely first."""
        if not self.goalies_on:
            return []
        cutoff = pd.Timestamp(self.meta["trained_through"]) - pd.Timedelta(days=max_age_days)
        t = self.gtable
        ids = set(self.pools.get(team, {})) | set(t[(t["team"] == team) & (pd.to_datetime(t["last_game"]) >= cutoff)].index)
        rows = [{"player_id": int(i), "name": t.loc[i, "name"], "starts": int(t.loc[i, "starts"]),
                 "rating_gpg": float(t.loc[i, "rating_gpg"]), "usual_share": self.pools.get(team, {}).get(int(i), 0.0)}
                for i in ids if i in t.index]
        return sorted(rows, key=lambda r: (-r["usual_share"], -r["starts"]))

    def goalie_spec(self, team, player_id=None, confidence=1.0):
        """None = unknown (usual mix). Otherwise a {player_id: weight} dict; if confidence < 1 the
        remainder goes to the team's other usual goalies."""
        if player_id is None:
            return None
        if confidence >= 0.999:
            return {int(player_id): 1.0}
        others = {k: v for k, v in self.pools.get(team, {}).items() if k != int(player_id)}
        tot = sum(others.values())
        spec = {int(player_id): float(confidence)}
        if tot > 0:
            spec.update({k: (1 - confidence) * v / tot for k, v in others.items()})
        else:
            spec = {int(player_id): 1.0}
        return spec

    def _lookup(self, team, query):
        if isinstance(query, (int, np.integer)):
            return int(query)
        q, t = str(query).lower().strip(), self.gtable
        for scope in (t[t["team"] == team], t):
            hit = scope[scope["name"].str.lower().str.contains(q, regex=False, na=False)]
            if len(hit):
                return int(hit.sort_values("last_game").index[-1])
        raise ValueError(f"No goalie matching '{query}' for {team}.")

    def _effect(self, team, spec):
        """(log-lambda effect, [info dicts]) for the goalie(s) that `team` will ice."""
        if spec is None:
            probs = dict(self.pools.get(team, {}))
            label = "usual mix"
        elif isinstance(spec, dict):
            probs, label = {self._lookup(team, k): float(v) for k, v in spec.items()}, "selected"
        else:
            probs, label = {self._lookup(team, spec): 1.0}, "selected"
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
    def _check(self, team):
        if team not in self.teams:
            raise ValueError(f"Unknown team '{team}'.")

    def lambdas(self, home, away, home_goalie=None, away_goalie=None):
        self._check(home); self._check(away)
        eff_h = eff_a = 0.0
        info = []
        if self.goalies_on:
            eff_h, i_away = self._effect(away, away_goalie)   # away goalie suppresses HOME scoring
            eff_a, i_home = self._effect(home, home_goalie)
            info = i_home + i_away
        rows = pd.DataFrame({"team": [home, away], "opponent": [away, home], "is_home": [1, 0]})
        if self.goalies_on:
            rows["opp_goalie_effect"] = [eff_h, eff_a]
        lam = self.model.predict(rows)
        return float(lam[0]), float(lam[1]), info

    def predict_game(self, home, away, home_goalie=None, away_goalie=None,
                     total_line=6.5, home_spread=-1.5, home_team_line=None, away_team_line=None):
        lh, la, info = self.lambdas(home, away, home_goalie, away_goalie)
        ot = self.home_ot_win
        htl = home_team_line if home_team_line is not None else total_line / 2
        atl = away_team_line if away_team_line is not None else total_line / 2
        periods = []
        for p in (1, 2, 3):
            ph = lh * self.period_shares.get(home, {}).get(p, 1 / 3)
            pa = la * self.period_shares.get(away, {}).get(p, 1 / 3)
            periods.append({"period": p, "exp_home": ph, "exp_away": pa, "exp_total": ph + pa,
                            "btts": btts_prob(ph, pa), "over_1_5": float(1 - poisson.cdf(1, ph + pa))})
        return {
            "home": home, "away": away, "lam_home": lh, "lam_away": la, "lam_total": lh + la,
            "p_home_win": home_win_prob(lh, la, ot),
            "p_home_reg": float(1 - skellam.cdf(0, lh, la)),
            "p_away_reg": float(skellam.cdf(-1, lh, la)),
            "p_tie_reg": float(skellam.pmf(0, lh, la)),
            "total": ou_probs(lh + la, total_line),
            "home_total": ou_probs(lh, htl), "away_total": ou_probs(la, atl),
            "spread": spread_probs(lh, la, home_spread, ot),
            "btts": btts_prob(lh, la), "periods": periods, "goalie_info": info,
        }

    # ---------- display helpers ----------
    def team_ratings(self):
        T = len(self.teams)
        b = self.model.beta
        df = pd.DataFrame({"attack": b[1:1 + T], "defense": b[1 + T:1 + 2 * T]}, index=self.teams)
        df["net_rating"] = df["attack"] - df["defense"]
        return df.loc[[t for t in self.teams if t in self.active_teams]].sort_values("net_rating", ascending=False)

    def goalie_leaders(self, min_starts=20):
        if not self.goalies_on:
            return None
        cutoff = pd.Timestamp(self.meta["trained_through"]) - pd.Timedelta(days=400)
        t = self.gtable
        t = t[(t["starts"] >= min_starts) & (pd.to_datetime(t["last_game"]) >= cutoff)]
        return t.sort_values("rating_gpg", ascending=False)[["name", "team", "starts", "sv_pct", "rating_gpg"]]
