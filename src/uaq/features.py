"""Forecast features: one row per issue day D (forecast issued at the end of local day D for D+1).

No-leakage rule: every feature uses day D or earlier. The only exception is the labelled
`*_tplus1` weather columns (day D+1), which exist only when features.include_future_weather
is true, for the upper-bound experiment, and are excluded by feature_columns() unless
future=True. The target y is the mean PM2.5 of day D+1 and is never imputed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .periods import assign_split, in_months, season_year

BOOKKEEPING = {"date", "target_date", "y", "split", "smog", "covid", "season_year"}


def _hourly_of_day(hourly: pd.DataFrame, idx: pd.DatetimeIndex, fc: dict) -> pd.DataFrame:
    h = hourly[["ts_local", "pm25"]].copy()
    h["date"] = h["ts_local"].dt.tz_localize(None).dt.normalize()
    h["hour"] = h["ts_local"].dt.hour
    h = h.dropna(subset=["pm25"])
    lo, hi = fc["evening_hours"]
    ev = h[(h["hour"] >= lo) & (h["hour"] <= hi)].groupby("date")["pm25"].agg(["mean", "count"])
    last = h.sort_values("ts_local").groupby("date").last()
    out = pd.DataFrame(index=idx)
    out["pm_evening"] = ev["mean"].where(ev["count"] >= fc["min_evening_hours"])
    out["pm_last_hour"] = last["pm25"].where(last["hour"] >= 21)
    out["pm_max_hourly"] = h.groupby("date")["pm25"].max()
    return out


def build(daily: pd.DataFrame, hourly: pd.DataFrame, cfg: dict, weather_daily: pd.DataFrame | None = None) -> pd.DataFrame:
    """Feature table for ONE station (filter `daily` and `hourly` to it first)."""
    fc = cfg["features"]
    s = daily.set_index("date")["pm25_daily"].sort_index()
    idx = pd.date_range(s.index.min(), s.index.max(), freq="D", name="date")
    s = s.reindex(idx)

    f = pd.DataFrame(index=idx)
    f["pm_lag0"] = s
    for k in fc["pm25_lags"]:
        f[f"pm_lag{k}"] = s.shift(k)
    for k in fc["rolling"]:
        f[f"pm_mean{k}d"] = s.rolling(k, min_periods=k // 2 + 1).mean()
    f["pm_std7d"] = s.rolling(7, min_periods=4).std()
    f = f.join(_hourly_of_day(hourly, idx, fc))

    f["y"] = s.shift(-1)
    target = idx + pd.Timedelta(days=1)
    f["target_date"] = target
    doy, dow = target.day_of_year.to_numpy(), target.day_of_week.to_numpy()
    f["doy_sin"], f["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    f["dow_sin"], f["dow_cos"] = np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)
    f["weekend"] = (dow >= 5).astype(int)

    if weather_daily is not None:
        wd = weather_daily.set_index("date").drop(columns="n_hours", errors="ignore").sort_index()
        wd = wd.reindex(pd.date_range(wd.index.min(), wd.index.max(), freq="D"))
        w = wd.add_prefix("w_")
        for c in ("temp_mean", "pressure_mean", "blh_night"):
            if c in wd:
                w[f"w_{c}_change"] = wd[c].diff()
        if fc.get("include_future_weather", False):
            w = w.join(wd.shift(-1).add_prefix("w_").add_suffix("_tplus1"))
        f = f.join(w.reindex(idx))

    p = cfg["forecast"]["periods"]
    tgt = pd.Series(target)
    f["split"] = assign_split(tgt, p["split"]).values
    f["smog"] = in_months(tgt, cfg["smog_months"]).values
    f["covid"] = ((tgt >= pd.Timestamp(p["covid_flag"]["start"])) & (tgt <= pd.Timestamp(p["covid_flag"]["end"]))).values
    f["season_year"] = season_year(tgt).values
    return f.reset_index()


def feature_columns(frame: pd.DataFrame, weather: bool = True, future: bool = False) -> list[str]:
    cols = [c for c in frame.columns if c not in BOOKKEEPING]
    if not weather:
        cols = [c for c in cols if not c.startswith("w_")]
    if not future:
        cols = [c for c in cols if not c.endswith("_tplus1")]
    return cols


def complete_rows(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Rows with every feature in `cols` and the target present (PM2.5 is never imputed)."""
    return frame.dropna(subset=list(cols) + ["y"])


def make_sequences(frame: pd.DataFrame, cols: list[str], history_days: int):
    """(X[n, T, F], y[n], issue_dates[n]); a window ending on D is kept only if all T days have
    every column and y(D) is present, so windows never span a gap."""
    f = frame.set_index("date").sort_index()
    f = f.reindex(pd.date_range(f.index.min(), f.index.max(), freq="D"))
    M = f[list(cols)].to_numpy(float)
    y = f["y"].to_numpy(float)
    ok = (~np.isnan(M).any(axis=1)).astype(int)
    T = history_days
    full = np.convolve(ok, np.ones(T, dtype=int), mode="valid") == T     # window ending at i = T-1+j
    ends = np.flatnonzero(full) + T - 1
    ends = ends[~np.isnan(y[ends])]
    X = np.stack([M[e - T + 1:e + 1] for e in ends]) if len(ends) else np.empty((0, T, len(cols)))
    return X, y[ends], f.index[ends]
