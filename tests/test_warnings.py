"""Tests for the fatigue-aware warning policy (synthetic, no files).  Run with:  pytest -q"""
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uaq import warnings as wp  # noqa: E402
from uaq.config import load_config  # noqa: E402
from uaq.periods import season_year  # noqa: E402

CFG = load_config()
W = CFG["warnings"]
VH = 220.0


def _val_days():
    """The two validation smog seasons as one daily index."""
    folds = CFG["forecast"]["periods"]["rolling_origin_folds"]
    d = pd.DatetimeIndex([])
    for f in folds:
        d = d.append(pd.date_range(f["val_start"], f["val_end"], freq="D"))
    return d[wp.validation_mask(d, CFG)]


def _synth(seed=0):
    rng = np.random.default_rng(seed)
    idx = _val_days()
    y = pd.Series(np.exp(rng.normal(4.6, 0.6, len(idx))), index=idx)
    f = (y * np.exp(rng.normal(0, 0.3, len(idx)))).rename("mlp (weather)")
    return y, f


def test_fatigue_rule_caps_consecutive_warnings():
    idx = pd.date_range("2023-12-01", periods=30, freq="D")
    lev = wp.apply_policy(pd.Series(500.0, index=idx), 100, 200, max_consecutive=2)
    assert lev.tolist()[:6] == [2, 2, 1, 2, 2, 1]
    assert wp._longest_run(lev) == 2
    gap = pd.Series(500.0, index=idx.delete([2]))                        # a calendar gap restarts the run
    assert wp.apply_policy(gap, 100, 200, 2).tolist()[:4] == [2, 2, 2, 2]
    assert wp._longest_run(wp.apply_policy(gap, 100, 200, 2)) == 2
    missing = pd.Series([500.0, np.nan, 500.0], index=idx[:3])
    assert wp.apply_policy(missing, 100, 200, 2).tolist() == [2, 0, 2]   # missing forecast -> no alert


def test_t1_must_be_below_t2():
    with pytest.raises(ValueError):
        wp.apply_policy(pd.Series([1.0], index=pd.date_range("2023-12-01", periods=1)), 200, 200, 2)


def test_search_respects_the_budget_in_each_winter_and_t1_below_t2():
    y, f = _synth()
    res = wp.search({"mlp (weather)": f}, y, CFG, VH)
    assert len(res) and (res["t1"] < res["t2"]).all()
    assert (res["max_l2_per_winter"] <= W["max_warning_days_per_winter"]).all()
    best = res.iloc[0]
    lev = wp.apply_policy(f, best["t1"], best["t2"], W["max_consecutive_warnings"])
    per_winter = (lev == 2).groupby(season_year(pd.Series(lev.index)).values).sum()
    assert len(per_winter) == 2 and (per_winter <= W["max_warning_days_per_winter"]).all()
    tight = copy.deepcopy(CFG)
    tight["warnings"]["max_warning_days_per_winter"] = 3
    assert (wp.search({"mlp (weather)": f}, y, tight, VH)["max_l2_per_winter"] <= 3).all()


def test_search_uses_only_validation_days():
    y, f = _synth()
    base = wp.search({"mlp (weather)": f}, y, CFG, VH)
    extra = pd.date_range("2024-11-01", "2025-02-28", freq="D")              # test winter, poisoned
    y2 = pd.concat([y, pd.Series(999.0, index=extra)])
    f2 = pd.concat([f, pd.Series(999.0, index=extra)]).rename(f.name)
    again = wp.search({"mlp (weather)": f2}, y2, CFG, VH)
    pd.testing.assert_frame_equal(base, again)


def test_score_counts_match_a_hand_made_example():
    idx = pd.date_range("2023-12-01", periods=6, freq="D")
    y = pd.Series([100, 180, 250, 300, 50, 190.0], index=idx)               # high > 170.8, very high > 220
    level = pd.Series([1, 1, 2, 1, 2, 0], index=idx)
    s = wp.score_policy(y, level, CFG, very_high=VH)
    assert (s["high_days"], s["very_high_days"]) == (4, 2)
    assert (s["high_caught"], s["very_high_caught_l2"], s["missed"]) == (3, 1, 1)
    assert (s["alerts"], s["alerts_l1"], s["alerts_l2"]) == (5, 3, 2)
    assert (s["false_alarms_l1"], s["false_alarms_l2"]) == (1, 1)
    assert s["weighted_recall"] == (1 + 2 + 1 + 0) / (1 + 2 + 2 + 1)        # 180 l1, 250 l2, 300 l1, 190 none
    assert s["caught_per_alert"] == 3 / 5 and s["longest_l2_run"] == 1


def test_input_series_max_with_today_needs_the_model():
    idx = pd.date_range("2023-12-01", periods=3, freq="D")
    preds = pd.DataFrame({"pred_mlp (weather)": [100.0, np.nan, 50.0]}, index=idx)
    feats = pd.DataFrame({"target_date": idx, "pm_lag0": [150.0, 300.0, np.nan]})
    s = wp.input_series("max(mlp, today)", preds, feats)
    assert s.iloc[0] == 150.0 and np.isnan(s.iloc[1]) and s.iloc[2] == 50.0


def test_single_threshold_rule_has_no_fatigue():
    idx = pd.date_range("2023-12-01", periods=5, freq="D")
    f = pd.Series(300.0, index=idx)
    lev, s = wp.single_threshold_rule(f, f, CFG, VH)
    assert lev.tolist() == [2] * 5 and s["longest_l2_run"] == 5


def test_evaluate_on_test_scores_only_test_smog_days_with_the_frozen_policy():
    idx = pd.date_range("2024-07-02", "2025-03-31", freq="D")
    rng = np.random.default_rng(0)
    y = np.exp(rng.normal(4.6, 0.6, len(idx)))
    feats = pd.DataFrame({"date": idx - pd.Timedelta(days=1), "target_date": idx, "y": y, "pm_lag0": y,
                          "split": "test"})
    preds = pd.DataFrame({"pred_mlp (weather)": y}, index=idx)
    days, s = wp.evaluate_on_test({"input": "mlp (weather)", "t1": 150, "t2": 250}, preds, feats, CFG, VH)
    assert days.index.month.isin(CFG["smog_months"]).all() and len(days) == s["days"]
    assert days["level"].isin([0, 1, 2]).all() and s["longest_l2_run"] <= W["max_consecutive_warnings"]
