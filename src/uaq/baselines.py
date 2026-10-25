"""Reference forecasts every neural network must beat.

All forecasts are for day D+1 and use only information available at the end of day D.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .periods import season_year


def make_targets(daily: pd.DataFrame) -> pd.DataFrame:
    """One row per station x issue day D, with y = valid mean of day D+1."""
    d = daily.sort_values(["station_id", "date"]).copy()
    d["y"] = d.groupby("station_id")["pm25_daily"].shift(-1)
    d["target_date"] = d["date"] + pd.Timedelta(days=1)
    return d


def persistence(frame: pd.DataFrame) -> pd.Series:
    """Tomorrow = today."""
    return frame["pm25_daily"].rename("persistence")


def rolling_mean(frame: pd.DataFrame, days: int = 3) -> pd.Series:
    """Tomorrow = mean of the last `days` valid days (min 1)."""
    return (frame.groupby("station_id")["pm25_daily"]
                 .transform(lambda s: s.rolling(days, min_periods=1).mean())
                 .rename(f"mean{days}d"))


def climatology(frame: pd.DataFrame, train_mask: pd.Series, window: int = 7) -> pd.Series:
    """Station-specific day-of-year mean for the TARGET date, fitted on training rows only.

    Day-of-year is smoothed with a +/- `window` day circular window.
    """
    tr = frame[train_mask & frame["y"].notna()]
    doy_tr = tr["target_date"].dt.dayofyear.clip(upper=365)
    table = {}
    for sid, g in tr.assign(doy=doy_tr).groupby("station_id"):
        prof = g.groupby("doy")["y"].mean().reindex(range(1, 366))
        vals = np.r_[prof.values[-window:], prof.values, prof.values[:window]]
        sm = pd.Series(vals).rolling(2 * window + 1, center=True, min_periods=1).mean().values[window:-window]
        table[sid] = pd.Series(sm, index=range(1, 366))
    doy = frame["target_date"].dt.dayofyear.clip(upper=365)
    out = [table[s].get(d, np.nan) if s in table else np.nan for s, d in zip(frame["station_id"], doy)]
    return pd.Series(out, index=frame.index, name="climatology")


def add_all(frame: pd.DataFrame, train_mask: pd.Series) -> pd.DataFrame:
    f = frame.copy()
    f["persistence"] = persistence(f)
    f["mean3d"] = rolling_mean(f, 3)
    f["climatology"] = climatology(f, train_mask)
    f["season_year"] = season_year(f["target_date"]).values
    return f
