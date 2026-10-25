"""Load raw OpenAQ PM2.5 exports into one tidy hourly table (forecasting part).

Output columns: station_id, station_name, ts_local (tz-aware, Asia/Ulaanbaatar),
pm25 (float, NaN where not observed), lat, lon, flag_implausible.
Raw files are never modified.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ROOT


def load_hourly(cfg: dict, root: Path = ROOT) -> pd.DataFrame:
    inp = cfg["forecast"]["input"]
    col = inp["columns"]
    files = sorted(glob.glob(str(root / inp["raw_glob"]), recursive=True))
    if not files:
        raise FileNotFoundError(f"No raw files match {root / inp['raw_glob']}")

    raw = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)

    if inp.get("location_ids"):
        raw = raw[raw[col["station_id"]].isin(inp["location_ids"])]
        if raw.empty:
            raise ValueError(f"No rows for location_ids {inp['location_ids']} in {len(files)} files")
    if col.get("parameter") in raw.columns and inp.get("parameter_filter"):
        raw = raw[raw[col["parameter"]].astype(str).str.lower() == inp["parameter_filter"]]

    # OpenAQ's docs call the column 'unit'; some exports say 'units'. Accept either.
    ucol = next((c for c in (col.get("units"), "unit", "units") if c in raw.columns), None)
    units_seen = sorted(raw[ucol].dropna().astype(str).unique()) if ucol else []

    ts = pd.to_datetime(raw[col["datetime"]], utc=True, errors="coerce", format="mixed")
    # Strings with an explicit offset are converted correctly by utc=True. Naive strings
    # are interpreted as UTC by pandas, which is only right if the source says so.
    if not inp.get("datetime_is_utc", False):
        naive = ~raw[col["datetime"]].astype(str).str.contains(r"(?:Z|[+-]\d\d:?\d\d)$", regex=True)
        if naive.any():
            local = pd.to_datetime(raw.loc[naive, col["datetime"]], errors="coerce", format="mixed")
            # Mongolia used DST in 2015-2016; ambiguous/non-existent local hours become NaT and are counted.
            ts.loc[naive] = (local.dt.tz_localize(cfg["timezone"], ambiguous="NaT", nonexistent="NaT")
                                  .dt.tz_convert("UTC"))

    df = pd.DataFrame({
        "station_id": raw[col["station_id"]].astype(str).values,
        "station_name": (raw[col["station_name"]] if col.get("station_name") in raw.columns
                         else raw[col["station_id"]]).astype(str).values,
        # Floor in UTC (always unambiguous), then convert to local time.
        "ts_local": ts.dt.floor("h").dt.tz_convert(cfg["timezone"]).values,
        "pm25": pd.to_numeric(raw[col["value"]], errors="coerce").values,
        "lat": pd.to_numeric(raw[col["lat"]], errors="coerce").values if col.get("lat") in raw.columns else np.nan,
        "lon": pd.to_numeric(raw[col["lon"]], errors="coerce").values if col.get("lon") in raw.columns else np.nan,
    })
    n_bad_ts = int(df["ts_local"].isna().sum())
    df = df.dropna(subset=["ts_local"])

    # Negative concentrations are sensor errors -> missing. Very high values are FLAGGED only:
    # Ulaanbaatar winter peaks are genuinely extreme.
    df.loc[df["pm25"] < 0, "pm25"] = np.nan
    df["flag_implausible"] = df["pm25"] > cfg["quality"]["plausible_max_hourly"]

    n_dupes = int(df.duplicated(["station_id", "ts_local"]).sum())
    df = (df.groupby(["station_id", "ts_local"], as_index=False)
            .agg(station_name=("station_name", "first"), pm25=("pm25", "mean"),
                 lat=("lat", "median"), lon=("lon", "median"),
                 flag_implausible=("flag_implausible", "max")))

    df.attrs.update(files=len(files), units_seen=units_seen, dropped_bad_timestamps=n_bad_ts,
                    duplicate_station_hours=n_dupes)
    return df.sort_values(["station_id", "ts_local"]).reset_index(drop=True)
