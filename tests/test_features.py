"""Tests for the weather download and the forecast features (no network).  Run with:  pytest -q"""
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uaq import features, weather  # noqa: E402
from uaq.config import load_config  # noqa: E402
from uaq.daily import to_daily  # noqa: E402

CFG = load_config()
TZ = "Asia/Ulaanbaatar"


def _pm_hourly(start, end, seed=0):
    rng = np.random.default_rng(seed)
    ts = pd.date_range(start, end + " 23:00", freq="h", tz=TZ)
    return pd.DataFrame({"station_id": "8160", "ts_local": ts, "pm25": rng.uniform(10, 300, len(ts))})


def _weather_hourly(start_utc, hours, seed=0):
    rng = np.random.default_rng(seed)
    t = pd.date_range(start_utc, periods=hours, freq="h", tz="UTC")
    wh = pd.DataFrame({"time_utc": t, "temperature_2m": rng.normal(-20, 5, hours),
                       "relative_humidity_2m": rng.uniform(40, 90, hours),
                       "surface_pressure": rng.normal(870, 5, hours),
                       "wind_speed_10m": rng.uniform(0, 5, hours),
                       "wind_direction_10m": rng.uniform(0, 360, hours),
                       "boundary_layer_height": rng.uniform(20, 800, hours)})
    wh["ts_local"] = wh["time_utc"].dt.tz_convert(TZ)
    return wh


def _features(pm, wh, cfg=CFG):
    return features.build(to_daily(pm), pm, cfg, weather.to_daily(wh))


def test_parse_response_and_download_with_fake_fetch(tmp_path):
    cfg = copy.deepcopy(CFG)
    cfg["weather"].update(out_file="w.csv", start="2020-12-30", end="2021-01-02")
    calls, ok_years = [], []

    def fake(url, params):
        calls.append(params)
        if "boundary_layer_height" in params["hourly"]:
            raise ValueError('{"reason": "Cannot initialize boundary_layer_height"}')
        t = pd.date_range(params["start_date"], params["end_date"] + " 23:00", freq="h")
        vs = params["hourly"].split(",")
        ok_years.append(params["start_date"][:4])
        return {"latitude": 47.9, "longitude": 106.9, "elevation": 1300.0,
                "hourly_units": {v: "x" for v in vs},
                "hourly": {"time": t.strftime("%Y-%m-%dT%H:%M").tolist(), **{v: [1.0] * len(t) for v in vs}}}

    path = weather.download(cfg, root=tmp_path, fetch=fake)
    df = pd.read_csv(path)
    assert len(df) == 96 and "boundary_layer_height" not in df.columns
    assert ok_years == ["2020", "2021"]                       # one successful request per year
    assert all(c["timezone"] == "GMT" and c["wind_speed_unit"] == "ms" for c in calls)
    assert (tmp_path / "w.meta.json").exists()
    parsed = weather.parse_response(fake("u", calls[-1]))
    assert str(parsed["time_utc"].dt.tz) == "UTC" and "temperature_2m" in parsed


def test_daily_weather_uses_local_calendar_days():
    wd = weather.to_daily(_weather_hourly("2020-10-31 00:00", 72))
    assert wd["n_hours"].iloc[0] == 16 and (wd["n_hours"].iloc[1:-1] == 24).all()
    assert {"temp_mean", "wind_u", "blh_night", "calm_hours"} <= set(wd.columns)


def test_no_feature_uses_the_target_day_or_later():
    pm = _pm_hourly("2020-11-01", "2020-12-10")
    wh = _weather_hourly("2020-10-31 16:00", 41 * 24 + 8)
    base = _features(pm, wh)

    cut = pd.Timestamp("2020-11-20", tz=TZ)
    pm2, wh2 = pm.copy(), wh.copy()
    pm2.loc[pm2["ts_local"] >= cut, "pm25"] = 9999.0
    wh2.loc[wh2["ts_local"] >= cut, ["temperature_2m", "boundary_layer_height"]] = 9999.0
    poisoned = _features(pm2, wh2)

    cols = features.feature_columns(base)
    before = base["date"] < "2020-11-20"
    pd.testing.assert_frame_equal(base.loc[before, cols], poisoned.loc[before, cols])
    hit = poisoned.loc[before & (poisoned["y"] == 9999.0), "date"]
    assert hit.tolist() == [pd.Timestamp("2020-11-19")]


def test_future_weather_columns_only_when_enabled():
    pm = _pm_hourly("2020-11-01", "2020-11-30")
    wh = _weather_hourly("2020-10-31 16:00", 32 * 24)
    assert not any(c.endswith("_tplus1") for c in _features(pm, wh).columns)

    cfg = copy.deepcopy(CFG)
    cfg["features"]["include_future_weather"] = True
    f = _features(pm, wh, cfg)
    fut = [c for c in f.columns if c.endswith("_tplus1")]
    assert fut and not set(fut) & set(features.feature_columns(f))
    assert set(fut) <= set(features.feature_columns(f, future=True))


def test_sequences_never_span_a_gap():
    pm = _pm_hourly("2020-11-01", "2020-12-10")
    gap = pm["ts_local"].dt.tz_localize(None).dt.normalize() == pd.Timestamp("2020-11-20")
    pm.loc[gap, "pm25"] = np.nan
    f = _features(pm, _weather_hourly("2020-10-31 16:00", 41 * 24 + 8))
    X, y, dates = features.make_sequences(f, ["pm_lag0", "w_temp_mean"], 7)
    g = pd.Timestamp("2020-11-20")
    assert len(dates) and not any(d - pd.Timedelta(days=6) <= g <= d for d in dates)
    assert not np.isnan(X).any() and not np.isnan(y).any()
    assert X.shape == (len(dates), 7, 2)


def test_target_is_next_day_and_split_uses_target_date():
    pm = _pm_hourly("2023-06-25", "2023-07-05")
    f = features.build(to_daily(pm), pm, CFG)
    i = f.index[f["date"] == "2023-06-30"][0]
    row = f.loc[i]
    assert row["target_date"] == pd.Timestamp("2023-07-01") and row["split"] == "val"
    assert row["y"] == f.loc[i + 1, "pm_lag0"]


def test_feature_columns_respects_drop_list():
    pm = _pm_hourly("2020-11-01", "2020-11-30")
    f = _features(pm, _weather_hourly("2020-10-31 16:00", 32 * 24))
    drop = ["w_blh_mean", "w_blh_night", "w_blh_night_change"]
    cols = features.feature_columns(f, drop=drop)
    assert not set(drop) & set(cols) and set(drop) <= set(features.feature_columns(f))
