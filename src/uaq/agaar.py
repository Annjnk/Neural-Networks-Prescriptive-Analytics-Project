"""Load the agaar.gov.mn daily archive (copied from the data-mn repo) for the network part.

Expected files (copied unchanged from data-mn `tools/sources/agaar/raw/`):
    data/raw/agaar/agaar-stations.csv
    data/raw/agaar/daily/agaar-daily-YYYY-MM.csv
Daily values are the site's own daily means in µg/m³. The hourly files are NOT used:
the site deletes hourly concentrations once a month is validated, so from Jan 2026
they are only about half complete, while the daily files stay about 94% complete.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ROOT


def load_stations(cfg: dict, root: Path = ROOT) -> pd.DataFrame:
    inp = cfg["network"]["input"]
    st = pd.read_csv(root / inp["stations_file"], dtype={"station_code": str})
    st = st[st["aimag_en"].astype(str).str.contains(inp["region"], case=False, na=False)]
    return st[["station_code", "name_mn", "lat", "lon"]].reset_index(drop=True)


def clean(df: pd.DataFrame, rules: dict) -> pd.DataFrame:
    """Drop implausible PM2.5 station-days; first matching rule wins, recorded in `dropped_reason`."""
    d = df.sort_values(["station_code", "date"]).copy()
    d["dropped_reason"] = ""
    raw = d["pm25_raw"]

    neg = raw < 0
    above = (raw > rules["above_pm10_ratio"] * d["pm10"]) & (raw > rules["above_pm10_min"]) & d["pm10"].notna()
    med = d.groupby("date")["pm25_raw"].transform("median")
    spike = (raw > rules["spike_ratio"] * med) & (raw > rules["spike_min"])

    # Flat: the same value on N+ consecutive calendar days at one station (stuck sensor).
    same = (d.groupby("station_code")["pm25_raw"].diff() == 0) & (d.groupby("station_code")["date"].diff() == pd.Timedelta(days=1))
    run_id = (~same).cumsum()
    run_len = d.groupby(run_id)["pm25_raw"].transform("size")
    flat = (run_len >= rules["flat_days"]) & raw.notna()

    for name, mask in (("negative", neg), ("above_pm10", above), ("spike", spike), ("flat", flat)):
        d.loc[mask & (d["dropped_reason"] == ""), "dropped_reason"] = name
    d["pm25"] = np.where(d["dropped_reason"] == "", raw, np.nan)
    return d


def load_daily(cfg: dict, root: Path = ROOT) -> pd.DataFrame:
    """Long table: station_code, date, validated, pm25_raw, pm10, pm25 (cleaned), dropped_reason."""
    inp = cfg["network"]["input"]
    files = sorted(glob.glob(str(root / inp["daily_glob"])))
    if not files:
        raise FileNotFoundError(f"No agaar daily files match {root / inp['daily_glob']}. "
                                "Copy data-mn tools/sources/agaar/raw/ into data/raw/agaar/.")
    d = pd.concat([pd.read_csv(f, dtype={"station_code": str}) for f in files], ignore_index=True)
    st = load_stations(cfg, root)
    d = d[d["station_code"].isin(st["station_code"])].copy()
    d["date"] = pd.to_datetime(d["date"])
    d["pm25_raw"] = pd.to_numeric(d["pm25"], errors="coerce")
    d["pm10"] = pd.to_numeric(d["pm10"], errors="coerce")
    d = d.drop(columns="pm25").drop_duplicates(["station_code", "date"], keep="last")
    out = clean(d, cfg["network"]["cleaning"])
    out.attrs["files"] = len(files)
    return out[["station_code", "date", "validated", "pm25_raw", "pm10", "pm25", "dropped_reason"]]


def to_wide(daily: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """date x station matrix of cleaned daily PM2.5 over the network study period (gaps = NaN)."""
    p = cfg["network"]["period"]
    idx = pd.date_range(p["start"], p["end"], freq="D", name="date")
    w = daily.pivot(index="date", columns="station_code", values="pm25")
    return w.reindex(idx)


def coverage(wide: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Share of days with a valid value, station x split."""
    sp = cfg["network"]["split"]
    cols = {}
    for name in ("train", "val", "test"):
        rows = (wide.index >= sp[name]["start"]) & (wide.index <= sp[name]["end"])
        cols[name] = wide[rows].notna().mean()
    return pd.DataFrame(cols).round(3)


def eligible(cov: pd.DataFrame, min_frac: float) -> list[str]:
    return cov.index[(cov >= min_frac).all(axis=1)].tolist()
