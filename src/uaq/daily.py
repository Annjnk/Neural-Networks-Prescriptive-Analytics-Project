"""Hourly -> daily aggregation with an explicit completeness rule."""
from __future__ import annotations

import numpy as np
import pandas as pd


def to_daily(hourly: pd.DataFrame, min_hours: int = 18) -> pd.DataFrame:
    """Daily mean PM2.5 per station and local calendar day.

    A day is valid only if at least `min_hours` hourly values were observed;
    otherwise pm25_daily is NaN (missing is never treated as clean air).
    The result has one row for every station x day between the station's first
    and last observation, so gaps are visible rather than silently absent.
    """
    h = hourly.copy()
    h["date"] = h["ts_local"].dt.tz_localize(None).dt.normalize()
    g = h.groupby(["station_id", "date"])["pm25"]
    daily = pd.DataFrame({"n_hours": g.count(), "mean_all": g.mean(), "max_hourly": g.max()}).reset_index()

    # Expand to a complete station x day grid.
    frames = []
    for sid, sub in daily.groupby("station_id"):
        full = pd.DataFrame({"date": pd.date_range(sub["date"].min(), sub["date"].max(), freq="D")})
        full["station_id"] = sid
        frames.append(full.merge(sub, on=["station_id", "date"], how="left"))
    daily = pd.concat(frames, ignore_index=True)

    daily["n_hours"] = daily["n_hours"].fillna(0).astype(int)
    daily["valid"] = daily["n_hours"] >= min_hours
    daily["pm25_daily"] = np.where(daily["valid"], daily["mean_all"], np.nan)
    return daily.drop(columns="mean_all")[["station_id", "date", "n_hours", "valid", "pm25_daily", "max_hourly"]]


def wide(daily: pd.DataFrame) -> pd.DataFrame:
    """date x station matrix of valid daily means."""
    return daily.pivot(index="date", columns="station_id", values="pm25_daily").sort_index()
