"""Generate a FAKE OpenAQ-style hourly PM2.5 file shaped like station 8160 (US Embassy),
so notebook 01 and the forecast audit can be dry-run before the real files are in place.
Never use these numbers in the report, and delete the file before the real run.

    python scripts/make_synthetic_data.py            # writes data/raw/pm25/_synthetic.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def make(n_stations: int = 1, start: str = "2016-11-09", end: str = "2025-03-31", seed: int = 0,
         first_id: int = 8160) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, end + " 23:00", freq="h", tz="Asia/Ulaanbaatar")
    doy, hour = idx.dayofyear.values, idx.hour.values
    winter = np.clip(np.cos(2 * np.pi * (doy - 15) / 365.25), 0, None) ** 1.5      # peaks mid-January
    diurnal = 1 + 0.5 * np.exp(-((hour - 21) % 24) ** 2 / 8) + 0.3 * np.exp(-((hour - 9) % 24) ** 2 / 4)
    post_ban = np.where(idx < pd.Timestamp("2019-05-15", tz="Asia/Ulaanbaatar"), 1.0, 0.6)

    # Shared weather driver (AR(1) on the daily scale) -> realistic multi-day episodes.
    days = pd.Series(idx.normalize()).factorize()[0]
    ar = np.zeros(days.max() + 1)
    for t in range(1, len(ar)):
        ar[t] = 0.75 * ar[t - 1] + rng.normal(0, 0.35)
    city = np.exp(ar[days])

    rows = []
    for s in range(n_stations):
        level = 1.0 if s == 0 else rng.uniform(0.6, 1.6)
        pm = ((12 + 180 * level * winter * post_ban * city) * diurnal * np.exp(rng.normal(0, 0.25, len(idx)))).astype(float)
        pm[rng.random(len(idx)) < 0.05] = np.nan                       # random missing hours
        o = rng.integers(0, len(idx) - 24 * 40)
        pm[o:o + 24 * rng.integers(5, 40)] = np.nan                    # one multi-week outage
        ok = ~np.isnan(pm)
        rows.append(pd.DataFrame({
            "location_id": first_id + s, "sensors_id": 90000 + s, "location": f"SYNTHETIC-{s:02d}",
            "datetime": idx[ok].strftime("%Y-%m-%dT%H:%M:%S%z").str.replace(r"(\d\d)(\d\d)$", r"\1:\2", regex=True),
            "lat": 47.928, "lon": 106.929, "parameter": "pm25", "units": "µg/m³", "value": pm[ok].round(1),
        }))
    return pd.concat(rows, ignore_index=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/pm25/_synthetic.csv")
    ap.add_argument("--stations", type=int, default=1)
    a = ap.parse_args()
    df = make(a.stations)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"wrote {len(df):,} rows to {a.out}  (FAKE data, delete before the real run)")
