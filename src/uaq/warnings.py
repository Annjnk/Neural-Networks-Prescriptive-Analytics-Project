"""Fatigue-aware day-ahead warning policy (H5, notebook 08).

A policy turns a next-day PM2.5 forecast into an alert level per day: 2 (warning) if the forecast >= t2,
1 (advisory) if t1 <= forecast < t2, else 0. Fatigue rule: after `max_consecutive` level-2 days in a row (calendar
days), the next day is downgraded to level 1. A missing forecast gives level 0.

The policy (input forecast, t1, t2) is chosen on the two validation winters only, under per-winter budgets for
level-2 days and for all alert days, by severity-weighted recall: a high day (observed > high) earns 1 if alerted at any level; a very-high day
(observed > the training 90th percentile) earns 2 at level 2, 1 at level 1. Ties go to fewer alerts.
The test winter is scored only by evaluate_on_test(), once, on 23 November (CLAUDE.md rule 1).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .periods import assign_split, in_months, season_label, season_year

_MAX = re.compile(r"max\((.+), today\)")


# ---------------------------------------------------------------------- inputs and days
def validation_mask(dates, cfg: dict) -> np.ndarray:
    """Target dates inside a rolling fold's validation window and in the smog season."""
    d = pd.DatetimeIndex(dates)
    m = np.zeros(len(d), bool)
    for f in cfg["forecast"]["periods"]["rolling_origin_folds"]:
        m |= (d >= pd.Timestamp(f["val_start"])) & (d <= pd.Timestamp(f["val_end"]))
    return m & in_months(pd.Series(d), cfg["smog_months"]).to_numpy()


def validation_days(features: pd.DataFrame, cfg: dict) -> pd.Series:
    """Observed y on validation smog-season target days, indexed by target_date."""
    f = features.set_index("target_date")
    y = f["y"][validation_mask(f.index, cfg)]
    return y.dropna().sort_index()


def very_high_value(features: pd.DataFrame, cfg: dict) -> float:
    """The very_high_quantile of training smog-season daily means (day D, as uaq.models.event_threshold)."""
    d = features["date"]
    train = (assign_split(d, cfg["forecast"]["periods"]["split"]) == "train").to_numpy()
    smog = in_months(d, cfg["smog_months"]).to_numpy()
    v = features.loc[train & smog, "pm_lag0"].dropna()
    return float(np.percentile(v, 100 * cfg["warnings"]["very_high_quantile"]))


def _model_name(x: str) -> str:
    return x if "(" in x else f"{x} (weather)"


def input_series(name: str, preds: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
    """Forecast per target day for an input name. `preds` has pred_<model> columns indexed by target_date.
    "max(x, today)" = max of model x and today's value (pm_lag0); NaN wherever model x has no forecast."""
    m = _MAX.fullmatch(name)
    model = _model_name(m.group(1)) if m else name
    s = preds[f"pred_{model}"].astype(float)
    if m:
        today = features.set_index("target_date")["pm_lag0"].reindex(s.index).astype(float)
        s = s.where(s.isna(), np.fmax(s, today))
    return s.rename(name).sort_index()


# ---------------------------------------------------------------------- policy
def _levels(vals: np.ndarray, day: np.ndarray, t1: float, t2: float, max_consecutive: int) -> np.ndarray:
    lev = np.where(np.isnan(vals), 0, np.where(vals >= t2, 2, np.where(vals >= t1, 1, 0)))
    run, prev = 0, None
    for i in range(len(lev)):
        if prev is None or day[i] - prev != 1:
            run = 0                                            # runs count calendar days
        if lev[i] == 2:
            if run >= max_consecutive:
                lev[i], run = 1, 0                             # fatigue: downgrade to an advisory
            else:
                run += 1
        else:
            run = 0
        prev = day[i]
    return lev


def _day_numbers(index) -> np.ndarray:
    return (pd.DatetimeIndex(index) - pd.Timestamp("1970-01-01")).days.to_numpy()


def apply_policy(f: pd.Series, t1: float, t2: float, max_consecutive: int) -> pd.Series:
    """Alert level per day (0 none, 1 advisory, 2 warning) for forecasts `f` indexed by target date."""
    if not t1 < t2:
        raise ValueError(f"need t1 < t2, got {t1}, {t2}")
    f = f.sort_index()
    return pd.Series(_levels(f.to_numpy(float), _day_numbers(f.index), t1, t2, max_consecutive),
                     index=f.index, name="level")


def _longest_run(level: pd.Series, k: int = 2) -> int:
    best = run = 0
    prev = None
    for d, v in zip(_day_numbers(level.index), level.to_numpy()):
        if v == k:
            run = run + 1 if (prev is not None and d - prev == 1) else 1     # a calendar gap starts a new run
        else:
            run = 0
        best, prev = max(best, run), d
    return best


def score_policy(y: pd.Series, level: pd.Series, cfg: dict, very_high: float | None = None) -> dict:
    """Scorecard of alert levels against observed y (aligned on target date; days without a level count as 0)."""
    w = cfg["warnings"]
    vh = very_high if very_high is not None else w["very_high"]
    lev = level.reindex(y.index).fillna(0).astype(int).to_numpy()
    yv = y.to_numpy(float)
    high, very = yv > w["high"], yv > vh
    credit = np.where(very, np.select([lev == 2, lev == 1], [2, 1], 0), np.where(high & (lev >= 1), 1, 0))
    weight = np.where(very, 2, np.where(high, 1, 0))
    alerts = int((lev >= 1).sum())
    caught = int((high & (lev >= 1)).sum())
    winters = pd.Series(lev == 2, index=y.index).groupby(season_year(pd.Series(y.index)).values).sum()
    out = {"days": len(yv), "high_days": int(high.sum()), "very_high_days": int(very.sum()),
           "high_caught": caught, "very_high_caught_l2": int((very & (lev == 2)).sum()),
           "weighted_recall": float(credit.sum() / weight.sum()) if weight.sum() else np.nan,
           "alerts": alerts, "alerts_l1": int((lev == 1).sum()), "alerts_l2": int((lev == 2).sum()),
           "false_alarms_l1": int(((lev == 1) & ~high).sum()), "false_alarms_l2": int(((lev == 2) & ~high).sum()),
           "missed": int((high & (lev == 0)).sum()),
           "longest_l2_run": _longest_run(pd.Series(lev, index=y.index)),
           "max_l2_per_winter": int(winters.max()) if len(winters) else 0,
           "caught_per_alert": caught / alerts if alerts else np.nan}
    out.update({f"l2_{season_label(int(k))}": int(v) for k, v in winters.items()})
    return out


# ---------------------------------------------------------------------- choice and baselines
def search(inputs: dict, y: pd.Series, cfg: dict, very_high: float) -> pd.DataFrame:
    """Every input x (t1 < t2) on the grid, validation days only. Keeps combinations that, in EACH validation
    winter, use at most max_alert_days_per_winter alert days (any level) and at most max_warning_days_per_winter
    level-2 days; sorted by weighted recall (desc), then fewer alerts, then config input order, t1, t2.
    The first row is the chosen policy."""
    w = cfg["warnings"]
    g = w["threshold_grid"]
    grid = np.arange(g["start"], g["stop"] + g["step"], g["step"])
    y = y[validation_mask(y.index, cfg)].sort_index()
    day = _day_numbers(y.index)
    yv = y.to_numpy(float)
    high, very = yv > w["high"], yv > very_high
    weight = np.where(very, 2, np.where(high, 1, 0))
    winter = season_year(pd.Series(y.index)).to_numpy()
    winters = np.unique(winter)
    rows = []
    for order, (name, f) in enumerate(inputs.items()):
        vals = f.reindex(y.index).to_numpy(float)
        for t1 in grid:
            for t2 in grid[grid > t1]:
                lev = _levels(vals, day, t1, t2, w["max_consecutive_warnings"])
                l2 = [(lev[winter == k] == 2).sum() for k in winters]
                alerts = [(lev[winter == k] >= 1).sum() for k in winters]
                if max(l2) > w["max_warning_days_per_winter"] or max(alerts) > w["max_alert_days_per_winter"]:
                    continue
                credit = np.where(very, np.select([lev == 2, lev == 1], [2, 1], 0), np.where(high & (lev >= 1), 1, 0))
                rows.append({"input": name, "order": order, "t1": float(t1), "t2": float(t2),
                             "weighted_recall": credit.sum() / weight.sum(), "alerts": int((lev >= 1).sum()),
                             "alerts_l2": int((lev == 2).sum()), "max_l2_per_winter": int(max(l2)),
                             "max_alerts_per_winter": int(max(alerts))})
    res = pd.DataFrame(rows).sort_values(["weighted_recall", "alerts", "order", "t1", "t2"],
                                         ascending=[False, True, True, True, True])
    return res.drop(columns="order").reset_index(drop=True)


def single_threshold_rule(f: pd.Series, y: pd.Series, cfg: dict, very_high: float, threshold: float | None = None):
    """Baseline: one alert level (scored as a warning) whenever the forecast >= threshold (default: high); no
    fatigue rule or budget. Returns (levels, scorecard)."""
    thr = cfg["warnings"]["high"] if threshold is None else threshold
    f = f.sort_index()
    lev = pd.Series(np.where(f.to_numpy(float) >= thr, 2, 0), index=f.index, name="level")
    return lev, score_policy(y, lev, cfg, very_high)


def matched_single_threshold(f: pd.Series, y: pd.Series, cfg: dict, very_high: float, alerts: int):
    """Single-threshold rule with the threshold set so it uses `alerts` alert days on the days of `y`: the threshold
    is the alerts-th largest forecast (ties can add days). For comparing caught-per-alert at an equal alert count.
    Returns (threshold, levels, scorecard)."""
    vals = np.sort(f.reindex(y.index).dropna().to_numpy(float))[::-1]
    thr = float(vals[min(max(alerts, 1), len(vals)) - 1])
    lev, card = single_threshold_rule(f.reindex(y.index), y, cfg, very_high, threshold=thr)
    return thr, lev, card


def evaluate_on_test(policy: dict, test_preds: pd.DataFrame, features: pd.DataFrame, cfg: dict, very_high: float):
    """Apply the FROZEN policy (input, t1, t2) to test-split forecasts (pred_<model> columns indexed by target_date,
    from fit_final_and_predict / fit_final_and_predict_lstm) and score it on observed test smog-season days.
    Runs ONCE, on 23 November. Returns (per-day frame, scorecard)."""
    f = features.set_index("target_date")
    test_days = f.index[(f["split"] == "test").to_numpy() & in_months(pd.Series(f.index), cfg["smog_months"]).to_numpy()]
    fc = input_series(policy["input"], test_preds, features)
    fc = fc[fc.index.isin(test_days)]
    lev = apply_policy(fc, policy["t1"], policy["t2"], cfg["warnings"]["max_consecutive_warnings"])
    y = f.loc[test_days, "y"].dropna()
    days = pd.DataFrame({"y": y, "forecast": fc.reindex(y.index), "level": lev.reindex(y.index).fillna(0).astype(int)})
    return days, score_policy(y, lev, cfg, very_high)
