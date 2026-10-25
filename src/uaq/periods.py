"""Season years, chronological splits and evaluation windows.

All functions take a Series/Index of local *dates* (day resolution) so the same
logic is used by the audit, the models and the GA.
"""
from __future__ import annotations

import pandas as pd


def season_year(dates: pd.Series | pd.DatetimeIndex, start_month: int = 7) -> pd.Series:
    """Season year = calendar year in which the season started.

    With start_month=7, 2019-11-15 and 2020-02-01 both belong to season 2019.
    """
    d = pd.to_datetime(pd.Series(dates))
    return (d.dt.year - (d.dt.month < start_month).astype(int)).rename("season_year")


def season_label(year: int) -> str:
    return f"{year}/{str(year + 1)[-2:]}"


def assign_split(dates: pd.Series | pd.DatetimeIndex, split_cfg: dict) -> pd.Series:
    """Label each date train/val/test, or 'outside' if it falls in none."""
    d = pd.to_datetime(pd.Series(dates))
    out = pd.Series("outside", index=d.index, name="split")
    for name in ("train", "val", "test"):
        lo, hi = pd.Timestamp(split_cfg[name]["start"]), pd.Timestamp(split_cfg[name]["end"])
        out[(d >= lo) & (d <= hi)] = name
    return out


def check_split_order(split_cfg: dict) -> None:
    """Raise if splits overlap or are out of chronological order."""
    t, v, s = (split_cfg[k] for k in ("train", "val", "test"))
    bounds = [pd.Timestamp(x) for x in (t["start"], t["end"], v["start"], v["end"], s["start"], s["end"])]
    if not all(a < b for a, b in zip(bounds, bounds[1:])):
        raise ValueError(f"Splits must be strictly chronological and non-overlapping: {bounds}")


def in_months(dates: pd.Series | pd.DatetimeIndex, months: list[int]) -> pd.Series:
    d = pd.to_datetime(pd.Series(dates))
    return d.dt.month.isin(months)


def rolling_folds(dates: pd.Series | pd.DatetimeIndex, folds_cfg: list[dict], core_start: str):
    """Yield (fold_no, train_mask, val_mask) for rolling-origin model selection."""
    d = pd.to_datetime(pd.Series(dates))
    start = pd.Timestamp(core_start)
    for i, f in enumerate(folds_cfg, 1):
        train = (d >= start) & (d <= pd.Timestamp(f["train_end"]))
        val = (d >= pd.Timestamp(f["val_start"])) & (d <= pd.Timestamp(f["val_end"]))
        yield i, train, val
