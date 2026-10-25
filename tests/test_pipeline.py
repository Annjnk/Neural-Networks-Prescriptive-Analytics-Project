"""Run with:  pytest -q   (from the repo root)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uaq import baselines, metrics, thresholds  # noqa: E402
from uaq.config import load_config  # noqa: E402
from uaq.daily import to_daily  # noqa: E402
from uaq.periods import assign_split, check_split_order, season_year  # noqa: E402

CFG = load_config()
FP = CFG["forecast"]["periods"]


def test_season_year_crosses_new_year():
    s = season_year(pd.Series(pd.to_datetime(["2019-07-01", "2019-12-31", "2020-02-15", "2020-06-30", "2020-07-01"])))
    assert s.tolist() == [2019, 2019, 2019, 2019, 2020]


def test_both_split_sets_are_chronological_and_contiguous():
    for sp in (FP["split"], CFG["network"]["split"]):
        check_split_order(sp)
        assert pd.Timestamp(sp["val"]["start"]) - pd.Timestamp(sp["train"]["end"]) == pd.Timedelta(days=1)
        assert pd.Timestamp(sp["test"]["start"]) - pd.Timestamp(sp["val"]["end"]) == pd.Timedelta(days=1)


def test_forecast_core_starts_after_coal_ban():
    assert pd.Timestamp(FP["core"]["start"]) > pd.Timestamp(CFG["structural_break"])
    assert pd.Timestamp(FP["pre_ban_sensitivity"]["end"]) < pd.Timestamp(FP["core"]["start"])


def test_every_forecast_split_contains_a_full_smog_season():
    for name in ("train", "val", "test"):
        dates = pd.Series(pd.date_range(FP["split"][name]["start"], FP["split"][name]["end"]))
        assert (dates.dt.month.isin(CFG["smog_months"])).sum() >= 119, name   # Nov-Feb = 119-120 days


def test_assign_split_labels_boundaries():
    d = pd.Series(pd.to_datetime(["2023-06-30", "2023-07-01", "2024-06-30", "2024-07-01", "2015-01-01"]))
    assert assign_split(d, FP["split"]).tolist() == ["train", "val", "val", "test", "outside"]


def _hourly(values_by_day: dict[str, list]):
    rows = []
    for day, vals in values_by_day.items():
        for h, v in enumerate(vals):
            rows.append({"station_id": "A", "ts_local": pd.Timestamp(day, tz="Asia/Ulaanbaatar") + pd.Timedelta(hours=h), "pm25": v})
    return pd.DataFrame(rows)


def test_incomplete_day_is_missing_not_low():
    d = to_daily(_hourly({"2020-01-01": [100.0] * 24, "2020-01-02": [5.0] * 10}), min_hours=18)
    assert d.loc[d["date"] == "2020-01-01", "pm25_daily"].item() == 100.0
    assert np.isnan(d.loc[d["date"] == "2020-01-02", "pm25_daily"].item())


def test_gap_days_appear_as_rows():
    d = to_daily(_hourly({"2020-01-01": [50.0] * 24, "2020-01-04": [50.0] * 24}))
    assert len(d) == 4 and d["valid"].sum() == 2


def test_persistence_target_is_next_day():
    f = baselines.make_targets(to_daily(_hourly({"2020-01-01": [10.0] * 24, "2020-01-02": [20.0] * 24, "2020-01-03": [30.0] * 24})))
    row = f[f["date"] == "2020-01-01"].iloc[0]
    assert row["pm25_daily"] == 10.0 and row["y"] == 20.0 and row["target_date"] == pd.Timestamp("2020-01-02")


def test_climatology_uses_training_rows_only():
    f = baselines.make_targets(to_daily(_hourly({f"2020-01-{d:02d}": [10.0] * 24 for d in range(1, 11)})))
    train = f["date"] < "2020-01-05"
    f.loc[~train, "y"] = 9999.0                      # poison non-training targets
    assert baselines.climatology(f, train).max() < 100


def test_event_metrics():
    m = metrics.events([10, 60, 80, 40], [60, 70, 20, 30], threshold=50)
    assert (m["tp"], m["false_alarms"], m["missed"]) == (1, 1, 1) and m["precision"] == 0.5 and m["recall"] == 0.5


def test_metrics_ignore_missing_pairs():
    r = metrics.regression([1.0, np.nan, 3.0], [1.0, 2.0, np.nan])
    assert r["n"] == 1 and r["mae"] == 0.0


def test_threshold_rule_picks_lowest_acceptable_official_line():
    vals = pd.Series([30.0] * 50 + [45.0] * 30 + [60.0] * 20)      # >37.5: 50%, >50: 20%
    name, thr, _ = thresholds.choose(thresholds.base_rates(vals, CFG), CFG)
    assert name == "mns_4585_2025" and thr == 37.5


def test_threshold_rule_falls_back_to_percentile():
    vals = pd.Series(np.linspace(80, 300, 100))                    # every official line exceeded always
    name, thr, _ = thresholds.choose(thresholds.base_rates(vals, CFG), CFG)
    assert name.startswith("train_p") and 80 < thr < 300


def test_forecast_eligibility_counts_partial_winters():
    from uaq.audit_forecast import eligibility, smog_days_per_season
    denom = smog_days_per_season(CFG)
    cov = pd.DataFrame([[0.0] * len(denom)], index=["8160"], columns=denom.index)
    cov[[2019, 2020]] = 1.0
    cov[[2021, 2022]] = 0.5
    cov[[2023, 2024]] = 0.9
    e = eligibility(cov, CFG).loc["8160"]
    assert e["good_train_winters"] == 2 and e["train_valid_smog_days"] >= 240 and bool(e["eligible"])
    cov[[2023]] = 0.6                                   # a weak validation winter still fails
    assert not bool(eligibility(cov, CFG).loc["8160", "eligible"])
