"""Hourly weather for Ulaanbaatar from the Open-Meteo historical API (ERA5 reanalysis).

    python -m uaq.weather            # downloads once to weather.out_file
    python -m uaq.weather --force    # re-download

Why reanalysis: NAMEM's station observations are sold per value, so the study uses ERA5 via
Open-Meteo (free, no key). Limitation: ERA5's grid is ~25 km, which does not resolve the
Ulaanbaatar valley or its winter inversions; treat these as city-scale weather, not
station-level meteorology, and say so in the write-up.

Time: Mongolia used daylight saving time in 2015-2016, so local timestamps from an API cannot
be trusted. Data are requested in UTC (timezone=GMT) and converted to Asia/Ulaanbaatar here.

Attribution: Open-Meteo (Zippenfenig, 2023); ERA5, Copernicus Climate Change Service.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ROOT, load_config

ATTRIBUTION = "Open-Meteo (Zippenfenig, 2023); ERA5, Copernicus Climate Change Service"


def _request(url: str, params: dict, retries: int = 3) -> dict:
    """GET `url` with `params`; HTTP 400 raises ValueError with the body, other errors retry."""
    full = f"{url}?{urllib.parse.urlencode(params)}"
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(full, timeout=120) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 400:
                raise ValueError(e.read().decode("utf-8", errors="replace")) from e
            if attempt == retries - 1:
                raise
        except urllib.error.URLError:
            if attempt == retries - 1:
                raise
        time.sleep(5 * (attempt + 1))          # 5 s, then 10 s
    raise RuntimeError("unreachable")


def parse_response(js: dict) -> pd.DataFrame:
    h = dict(js["hourly"])
    df = pd.DataFrame({"time_utc": pd.to_datetime(h.pop("time"), utc=True)})
    for name, values in h.items():
        df[name] = values
    return df


def download(cfg: dict, root: Path = ROOT, force: bool = False, fetch=_request) -> Path:
    """Download weather.start..weather.end one calendar year per request; save CSV + meta JSON."""
    w = cfg["weather"]
    out = root / w["out_file"]
    if out.exists() and not force:
        print(f"{out} exists; not downloading (use force=True to refresh)")
        return out

    variables = list(w["hourly"])
    start, end = pd.Timestamp(w["start"]), pd.Timestamp(w["end"])
    frames, js = [], {}
    for year in range(start.year, end.year + 1):
        lo, hi = max(start, pd.Timestamp(year, 1, 1)), min(end, pd.Timestamp(year, 12, 31))
        while True:
            params = {"latitude": w["latitude"], "longitude": w["longitude"],
                      "start_date": lo.strftime("%Y-%m-%d"), "end_date": hi.strftime("%Y-%m-%d"),
                      "hourly": ",".join(variables), "timezone": "GMT", "wind_speed_unit": "ms"}
            if w.get("models"):
                params["models"] = w["models"]
            try:
                js = fetch(w["url"], params)
                break
            except ValueError as e:
                bad = [v for v in w.get("optional", []) if v in variables and v in str(e)]
                if not bad:
                    raise
                for v in bad:
                    variables.remove(v)
                    print(f"{v} not offered by the API; dropped for the rest of this download")
                frames = [f.drop(columns=bad, errors="ignore") for f in frames]
        frames.append(parse_response(js))
        print(f"{year}: {len(frames[-1]):,} hours")

    df = (pd.concat(frames, ignore_index=True)
            .drop_duplicates("time_utc").sort_values("time_utc").reset_index(drop=True))
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    meta = {"latitude": js.get("latitude"), "longitude": js.get("longitude"), "elevation": js.get("elevation"),
            "hourly_units": js.get("hourly_units"), "variables": variables,
            "requested_latitude": w["latitude"], "requested_longitude": w["longitude"],
            "downloaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "models": w.get("models"), "source": w["url"], "attribution": ATTRIBUTION}
    out.with_name(out.stem + ".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out


def load_hourly(cfg: dict, root: Path = ROOT) -> pd.DataFrame:
    df = pd.read_csv(root / cfg["weather"]["out_file"])
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
    df["ts_local"] = df["time_utc"].dt.tz_convert(cfg["timezone"])
    return df


def to_daily(wh: pd.DataFrame) -> pd.DataFrame:
    """Daily weather by LOCAL calendar day, for the variables present in `wh`."""
    h = wh.copy()
    h["date"] = h["ts_local"].dt.tz_localize(None).dt.normalize()
    h["hour"] = h["ts_local"].dt.hour
    g = h.groupby("date")
    out = pd.DataFrame({"n_hours": g.size()})
    if "temperature_2m" in h:
        t = g["temperature_2m"]
        out["temp_mean"], out["temp_min"] = t.mean(), t.min()
        out["temp_range"] = t.max() - t.min()
    if "relative_humidity_2m" in h:
        out["rh_mean"] = g["relative_humidity_2m"].mean()
    if "surface_pressure" in h:
        out["pressure_mean"] = g["surface_pressure"].mean()
    if "wind_speed_10m" in h:
        out["wind_mean"] = g["wind_speed_10m"].mean()
        out["calm_hours"] = g["wind_speed_10m"].agg(lambda s: int((s < 1.0).sum()))
        if "wind_direction_10m" in h:
            rad = np.deg2rad(h["wind_direction_10m"])
            h["u"] = -np.sin(rad) * h["wind_speed_10m"]
            h["v"] = -np.cos(rad) * h["wind_speed_10m"]
            out["wind_u"], out["wind_v"] = h.groupby("date")["u"].mean(), h.groupby("date")["v"].mean()
    if "boundary_layer_height" in h:
        out["blh_mean"] = g["boundary_layer_height"].mean()
        out["blh_night"] = h[h["hour"] <= 6].groupby("date")["boundary_layer_height"].mean()
    return out.reset_index()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    print(download(load_config(a.config), force=a.force))
