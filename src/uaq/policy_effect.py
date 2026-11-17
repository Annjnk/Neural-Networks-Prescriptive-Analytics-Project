"""Weather-adjusted estimate of the 15 May 2019 raw-coal ban (H6, notebook 07).

Method (weather normalisation used as a counterfactual, in the style of Grange & Carslaw): a model of
log1p(daily PM2.5) on weather and calendar inputs is trained on PRE-ban days only (all months) and predicts the
"no-ban" level of each post-ban day. Effect % = observed mean / counterfactual mean - 1, reported on smog-season
(Nov-Feb) days. No date or trend variable is an input: it would absorb the ban.

Caveat: this is weather-adjusted, not causal. Population, fuel briquettes, COVID and warmer winters also changed.

Seal (project rule 1: test seal): no date after ban_effect.end (the 2024/25 test winter) is ever used; every public
function checks it. PM2.5 is never imputed: days without a valid daily mean are left out.

Back-transformation: log1p predictions are converted to means with Duan's smearing factor, computed from
out-of-bag residuals for the forest and in-sample residuals for OLS, separately for smog-season and other fit
days (log1p residuals are wider in winter), so observed and counterfactual means compare.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor

from . import io, weather
from .config import ROOT
from .daily import to_daily
from .periods import in_months, season_label, season_year

FORBIDDEN = ("date", "year", "trend", "time", "season", "post")      # never a counterfactual input


# ---------------------------------------------------------------------- data
def _end(cfg: dict) -> pd.Timestamp:
    return pd.Timestamp(cfg["ban_effect"]["end"])


def _ban(cfg: dict) -> pd.Timestamp:
    return pd.Timestamp(cfg["structural_break"])


def _check_seal(df: pd.DataFrame, cfg: dict) -> None:
    if (df["date"] > _end(cfg)).any():
        raise ValueError(f"dates after {cfg['ban_effect']['end']} found: the test winter is sealed")


def input_columns(cfg: dict) -> list[str]:
    """Weather + calendar inputs; raises if any looks like a date or trend variable."""
    be = cfg["ban_effect"]
    cols = list(be["weather"]) + list(be["calendar"])
    bad = [c for c in cols if any(k in c.lower() for k in FORBIDDEN)]
    if bad:
        raise ValueError(f"date/trend-like inputs are not allowed (they would absorb the ban): {bad}")
    return cols


def build_daily(cfg: dict, root: Path = ROOT, pm_hourly: pd.DataFrame | None = None,
                weather_hourly: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per day from pre_ban_sensitivity.start to ban_effect.end: date, season_year, smog, covid, post,
    pm25 (valid daily mean, NaN otherwise), weather and calendar columns."""
    be = cfg["ban_effect"]
    lo, hi = pd.Timestamp(cfg["forecast"]["periods"]["pre_ban_sensitivity"]["start"]), _end(cfg)
    h = io.load_hourly(cfg, root) if pm_hourly is None else pm_hourly
    h = h[h["ts_local"].dt.tz_localize(None).dt.normalize() <= hi]           # sealed hours never enter
    d = to_daily(h, cfg["quality"]["min_hours_per_day"])
    pm = d[d["valid"]].groupby("date")["pm25_daily"].mean()
    wh = weather.load_hourly(cfg, root) if weather_hourly is None else weather_hourly
    wd = weather.to_daily(wh[wh["ts_local"].dt.tz_localize(None).dt.normalize() <= hi]).set_index("date")

    days = pd.date_range(lo, hi, freq="D")
    df = pd.DataFrame({"date": days})
    df["season_year"] = season_year(df["date"], cfg.get("season_start_month", 7)).values
    df["smog"] = in_months(df["date"], cfg["smog_months"]).values
    cv = cfg["forecast"]["periods"]["covid_flag"]
    df["covid"] = ((df["date"] >= cv["start"]) & (df["date"] <= cv["end"])).values
    df["post"] = (df["date"] >= _ban(cfg)).values
    df["pm25"] = pm.reindex(days).values
    for c in be["weather"]:
        df[c] = wd[c].reindex(days).values
    doy, dow = days.day_of_year.to_numpy(), days.day_of_week.to_numpy()
    df["doy_sin"], df["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    df["dow_sin"], df["dow_cos"] = np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)
    _check_seal(df, cfg)
    return df


def coverage_by_season(df: pd.DataFrame) -> pd.DataFrame:
    """Smog-season days, valid days, share valid and observed mean per season; pre/post-ban and COVID flags."""
    s = df[df["smog"]]
    g = s.groupby("season_year")
    out = pd.DataFrame({"smog_days": g.size(), "valid_days": g["pm25"].count(), "observed_mean": g["pm25"].mean(),
                        "period": g["post"].all().map({True: "post-ban", False: "pre-ban"}), "covid": g["covid"].any()})
    out["share_valid"] = out["valid_days"] / out["smog_days"]
    out.index = [season_label(int(k)) for k in out.index]
    return out.rename_axis("season")[["period", "smog_days", "valid_days", "share_valid", "observed_mean", "covid"]]


# ---------------------------------------------------------------------- counterfactual
def _design(df: pd.DataFrame, cols: list[str], model: str) -> np.ndarray:
    X = df[cols].to_numpy(float)
    if model == "linear":                                      # OLS: intercept + inputs + squared temperatures
        sq = [df[c].to_numpy(float) ** 2 for c in ("temp_mean", "temp_min") if c in cols]
        X = np.column_stack([np.ones(len(df)), X, *sq])
    return X


def _fit_predict(df: pd.DataFrame, cfg: dict, fit_rows: np.ndarray, predict_rows: np.ndarray,
                 model: str, rf_params: dict | None = None) -> np.ndarray:
    """Fit log1p(pm25) on rows `fit_rows` (duplicates allowed), predict the mean level on `predict_rows`."""
    cols = input_columns(cfg)
    X = _design(df, cols, model)
    ok = ~np.isnan(X).any(axis=1)
    fit_rows = fit_rows[ok[fit_rows] & df["pm25"].notna().to_numpy()[fit_rows]]
    y = np.log1p(df["pm25"].to_numpy(float)[fit_rows])
    if model == "rf":
        m = RandomForestRegressor(**(rf_params or cfg["ban_effect"]["rf"]), oob_score=True, n_jobs=-1)
        m.fit(X[fit_rows], y)
        resid = y - m.oob_prediction_
        f = lambda A: m.predict(A)                                              # noqa: E731
    elif model == "linear":
        beta, *_ = np.linalg.lstsq(X[fit_rows], y, rcond=None)
        resid = y - X[fit_rows] @ beta
        f = lambda A: A @ beta                                                  # noqa: E731
    else:
        raise ValueError(f"unknown model {model!r}")
    # Duan's smearing factor, separately for smog-season and other days: log1p compresses low summer values,
    # so residual spread is larger in winter and one year-round factor would bias smog-day means low.
    smog = df["smog"].to_numpy(bool)
    smear = {k: float(np.mean(np.exp(resid[smog[fit_rows] == k]))) if (smog[fit_rows] == k).any()
             else float(np.mean(np.exp(resid))) for k in (True, False)}
    out = np.full(len(predict_rows), np.nan)
    good = predict_rows[ok[predict_rows]]
    factor = np.where(smog[good], smear[True], smear[False])
    out[ok[predict_rows]] = np.clip(np.exp(f(X[good])) * factor - 1, 0, None)
    return out


def counterfactual(df: pd.DataFrame, cfg: dict, fit_mask, predict_mask, model: str = "rf",
                   rf_params: dict | None = None) -> pd.Series:
    """No-ban daily PM2.5 for `predict_mask` days, from a model fitted on `fit_mask` days with a valid mean."""
    _check_seal(df, cfg)
    rows = np.flatnonzero(np.asarray(predict_mask))
    pred = _fit_predict(df, cfg, np.flatnonzero(np.asarray(fit_mask)), rows, model, rf_params)
    return pd.Series(pred, index=df.index[rows], name="counterfactual")


def effect_by_season(df: pd.DataFrame, pred: pd.Series) -> pd.DataFrame:
    """Per season (and pooled): observed and counterfactual smog-season means over days with both, effect % =
    observed / counterfactual - 1, number of days, COVID flag."""
    s = df.loc[pred.index].assign(cf=pred)
    s = s[s["smog"] & s["pm25"].notna() & s["cf"].notna()]
    g = s.groupby("season_year")
    out = pd.DataFrame({"observed": g["pm25"].mean(), "counterfactual": g["cf"].mean(),
                        "n_days": g.size(), "covid": g["covid"].any()})
    out.index = [season_label(int(k)) for k in out.index]
    out.loc["pooled"] = [s["pm25"].mean(), s["cf"].mean(), len(s), bool(s["covid"].any())]
    out["effect"] = out["observed"] / out["counterfactual"] - 1
    out["n_days"] = out["n_days"].astype(int)
    out["covid"] = out["covid"].astype(bool)
    return out.rename_axis("season")


def _masks(df: pd.DataFrame, cfg: dict):
    post = df["post"].to_numpy(bool)
    return ~post, post                                      # fit: every pre-ban day; predict: every post-ban day


def ban_effect(df: pd.DataFrame, cfg: dict, model: str = "rf", rf_params: dict | None = None):
    """Main estimate: counterfactual fitted on all pre-ban days, effect per post-ban season and pooled."""
    fit, post = _masks(df, cfg)
    pred = counterfactual(df, cfg, fit, post, model, rf_params)
    return effect_by_season(df, pred), pred


def block_bootstrap(df: pd.DataFrame, cfg: dict, model: str = "rf", n: int | None = None,
                    rf_params: dict | None = None):
    """95% interval for the effect: resample block_days-day blocks of the pre-ban fit days with replacement,
    refit, re-predict. Returns (interval table with lo/hi per season and pooled, replicate effects)."""
    _check_seal(df, cfg)
    b = cfg["ban_effect"]["bootstrap"]
    n = n or b["n"]
    fit, post = _masks(df, cfg)
    fit_rows = np.flatnonzero(fit & df["pm25"].notna().to_numpy())
    block = ((df["date"] - df["date"].min()).dt.days // b["block_days"]).to_numpy()[fit_rows]
    blocks = {k: fit_rows[block == k] for k in np.unique(block)}
    keys = np.array(list(blocks))
    rng = np.random.default_rng(b["seed"])
    post_rows = np.flatnonzero(post)
    reps = []
    for _ in range(n):
        rows = np.concatenate([blocks[k] for k in rng.choice(keys, size=len(keys), replace=True)])
        pred = pd.Series(_fit_predict(df, cfg, rows, post_rows, model, rf_params), index=df.index[post_rows])
        reps.append(effect_by_season(df, pred)["effect"])
    reps = pd.DataFrame(reps).reset_index(drop=True)
    interval = pd.DataFrame({"lo": reps.quantile(0.025), "hi": reps.quantile(0.975)})
    return interval, reps


def placebo(df: pd.DataFrame, cfg: dict, model: str = "rf", rf_params: dict | None = None) -> pd.Series:
    """Fit on pre-ban seasons before placebo_holdout_season, predict that (pre-ban) season: effect should be ~0."""
    hold = cfg["ban_effect"]["placebo_holdout_season"]
    pre = ~df["post"].to_numpy(bool)
    sy = df["season_year"].to_numpy()
    pred = counterfactual(df, cfg, pre & (sy < hold), pre & (sy == hold), model, rf_params)
    return effect_by_season(df, pred).loc[season_label(hold)]


def did_check(df: pd.DataFrame, cfg: dict) -> dict:
    """Cross-check: OLS of log1p(pm25) on a post-ban dummy + weather + calendar over every valid day to
    ban_effect.end, HAC (Newey-West) standard errors with maxlags = bootstrap block_days. Missing days are
    skipped, so HAC lags count observed days. Effect % = exp(b) - 1 with a 95% interval."""
    _check_seal(df, cfg)
    cols = input_columns(cfg)
    d = df[df["pm25"].notna()].dropna(subset=cols)
    X = sm.add_constant(d[["post"] + cols].astype(float))
    lags = cfg["ban_effect"]["bootstrap"]["block_days"]
    res = sm.OLS(np.log1p(d["pm25"].to_numpy(float)), X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    b, se = float(res.params["post"]), float(res.bse["post"])
    return {"coef": b, "se": se, "effect": float(np.expm1(b)), "lo": float(np.expm1(b - 1.96 * se)),
            "hi": float(np.expm1(b + 1.96 * se)), "pvalue": float(res.pvalues["post"]), "n_days": int(res.nobs),
            "hac_maxlags": lags, "r2": float(res.rsquared)}
