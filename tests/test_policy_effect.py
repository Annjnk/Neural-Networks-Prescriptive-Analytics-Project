"""Tests for the weather-adjusted coal-ban estimate (synthetic data, no files).  Run with:  pytest -q"""
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uaq import policy_effect as pe  # noqa: E402
from uaq.config import load_config  # noqa: E402
from uaq.periods import in_months, season_year  # noqa: E402

CFG = load_config()
END = pd.Timestamp(CFG["ban_effect"]["end"])
BAN = pd.Timestamp(CFG["structural_break"])
START = pd.Timestamp(CFG["forecast"]["periods"]["pre_ban_sensitivity"]["start"])
FAST = copy.deepcopy(CFG)
FAST["ban_effect"]["rf"] = {"n_estimators": 100, "min_samples_leaf": 5, "random_state": 0}


def _synth(step=0.30, seed=0, missing=0.2, end=END):
    """Daily frame shaped like build_daily(): PM2.5 driven by weather, multiplied by (1 - step) after the ban."""
    rng = np.random.default_rng(seed)
    days = pd.date_range(START, end, freq="D")
    n = len(days)
    doy, dow = days.day_of_year.to_numpy(), days.day_of_week.to_numpy()
    winter = np.cos(2 * np.pi * (doy - 15) / 365.25)
    df = pd.DataFrame({"date": days})
    df["season_year"] = season_year(df["date"]).values
    df["smog"] = in_months(df["date"], CFG["smog_months"]).values
    df["covid"] = ((df["date"] >= "2020-03-01") & (df["date"] <= "2021-06-30")).values
    df["post"] = (df["date"] >= BAN).values
    df["temp_mean"] = -15 * winter + rng.normal(0, 4, n)
    df["temp_min"] = df["temp_mean"] - 6 + rng.normal(0, 1.5, n)
    df["temp_range"] = 12 + rng.normal(0, 2, n)
    df["rh_mean"] = 60 + rng.normal(0, 10, n)
    df["pressure_mean"] = 860 + rng.normal(0, 5, n)
    df["wind_mean"] = rng.gamma(2, 1.2, n)
    df["calm_hours"] = rng.integers(0, 12, n).astype(float)
    df["wind_u"], df["wind_v"] = rng.normal(0, 1.5, n), rng.normal(0, 1.5, n)
    df["doy_sin"], df["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    df["dow_sin"], df["dow_cos"] = np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)
    log_pm = 3.6 - 0.07 * df["temp_mean"] + 0.06 * df["calm_hours"] - 0.15 * df["wind_mean"] + rng.normal(0, 0.3, n)
    df["pm25"] = np.exp(log_pm) * np.where(df["post"], 1 - step, 1.0)
    df.loc[rng.random(n) < missing, "pm25"] = np.nan                          # missing days stay missing
    return df


def test_inputs_have_no_date_or_trend_column():
    cols = pe.input_columns(CFG)
    assert cols == CFG["ban_effect"]["weather"] + CFG["ban_effect"]["calendar"]
    assert not {"date", "season_year", "post", "trend"} & set(cols)
    bad = copy.deepcopy(CFG)
    bad["ban_effect"]["calendar"] = bad["ban_effect"]["calendar"] + ["trend"]
    with pytest.raises(ValueError):
        pe.input_columns(bad)


def test_no_date_after_the_end_is_ever_used():
    df = _synth(end=END + pd.Timedelta(days=60))                             # reaches into the test winter
    fit, post = ~df["post"].to_numpy(), df["post"].to_numpy()
    for call in (lambda: pe.counterfactual(df, FAST, fit, post, "linear"), lambda: pe.placebo(df, FAST, "linear"),
                 lambda: pe.block_bootstrap(df, FAST, "linear", n=2), lambda: pe.did_check(df, FAST)):
        with pytest.raises(ValueError):
            call()


def test_build_daily_stops_at_the_end_and_never_imputes():
    hours = pd.date_range(START, END + pd.Timedelta(days=90), freq="h", tz="Asia/Ulaanbaatar")
    pm = pd.DataFrame({"station_id": "8160", "ts_local": hours, "pm25": 50.0})
    pm.loc[pm["ts_local"].dt.date == pd.Timestamp("2018-01-10").date(), "pm25"] = np.nan     # an outage day
    utc = pd.date_range(START - pd.Timedelta(days=1), END + pd.Timedelta(days=90), freq="h", tz="UTC")
    wh = pd.DataFrame({"time_utc": utc, "temperature_2m": -10.0, "relative_humidity_2m": 60.0,
                       "surface_pressure": 860.0, "wind_speed_10m": 2.0, "wind_direction_10m": 90.0})
    wh["ts_local"] = wh["time_utc"].dt.tz_convert("Asia/Ulaanbaatar")
    df = pe.build_daily(CFG, pm_hourly=pm, weather_hourly=wh)
    assert df["date"].min() == START and df["date"].max() == END
    assert np.isnan(df.loc[df["date"] == "2018-01-10", "pm25"]).all()
    assert set(pe.input_columns(CFG)) <= set(df.columns)


@pytest.mark.parametrize("model", ["rf", "linear"])
def test_counterfactual_recovers_a_known_30_percent_step(model):
    eff, _ = pe.ban_effect(_synth(step=0.30), FAST, model)
    assert abs(eff.loc["pooled", "effect"] - (-0.30)) < 0.05, eff


def test_placebo_without_a_step_is_near_zero():
    r = pe.placebo(_synth(step=0.0, seed=1), FAST, "rf")
    assert abs(r["effect"]) < 0.05 and r["n_days"] > 0


def test_bootstrap_refits_on_pre_ban_blocks_only_and_brackets_the_estimate(monkeypatch):
    # The interval is model uncertainty only (pre-ban fit blocks resampled, post-ban days fixed), so it is
    # checked against the full-data estimate, not the synthetic truth (it covers that only ~75% of the time).
    df = _synth(step=0.30, seed=2)
    seen, real = [], pe._fit_predict

    def spy(df_, cfg, fit_rows, predict_rows, model, rf_params=None):
        seen.append((fit_rows, predict_rows))
        return real(df_, cfg, fit_rows, predict_rows, model, rf_params)

    monkeypatch.setattr(pe, "_fit_predict", spy)
    interval, reps = pe.block_bootstrap(df, FAST, "linear", n=40)
    est = pe.ban_effect(df, FAST, "linear")[0].loc["pooled", "effect"]
    post_rows = np.flatnonzero(df["post"].to_numpy())
    assert len(reps) == 40 and len(seen) == 41
    for fit_rows, predict_rows in seen[:40]:
        assert (df["date"].to_numpy()[fit_rows] < BAN).all()                  # refits use pre-ban days only
        assert len(set(fit_rows)) < len(fit_rows)                             # blocks drawn with replacement
        np.testing.assert_array_equal(predict_rows, post_rows)                # post-ban days stay fixed
    lo, hi = interval.loc["pooled", ["lo", "hi"]]
    assert lo < est < hi


def test_did_check_recovers_the_step():
    r = pe.did_check(_synth(step=0.30, seed=3), FAST)
    assert r["lo"] < -0.30 < r["hi"] and abs(r["effect"] + 0.30) < 0.05


def test_coverage_by_season_counts_valid_smog_days():
    df = _synth(step=0.0, missing=0.0)
    df.loc[(df["season_year"] == 2017) & df["smog"], "pm25"] = np.nan        # a whole missing winter
    cov = pe.coverage_by_season(df)
    assert cov.loc["2017/18", "valid_days"] == 0 and cov.loc["2018/19", "share_valid"] == 1.0
    assert cov.loc["2019/20", "period"] == "post-ban" and cov.loc["2018/19", "period"] == "pre-ban"
    assert bool(cov.loc["2020/21", "covid"]) and not bool(cov.loc["2021/22", "covid"])
