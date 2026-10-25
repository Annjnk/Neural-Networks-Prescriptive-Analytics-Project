"""Forecast and event-detection metrics. NaN pairs are dropped, never filled."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _pair(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    m = ~(np.isnan(y) | np.isnan(yhat))
    return y[m], yhat[m]


def regression(y, yhat) -> dict:
    y, yhat = _pair(y, yhat)
    if len(y) == 0:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "r2": np.nan}
    err = yhat - y
    ss_tot = ((y - y.mean()) ** 2).sum()
    return {
        "n": int(len(y)),
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "r2": float(1 - (err ** 2).sum() / ss_tot) if ss_tot > 0 else np.nan,
    }


def events(y, yhat, threshold: float) -> dict:
    """Exceedance detection: an event is a daily mean above `threshold`."""
    y, yhat = _pair(y, yhat)
    obs, pred = y > threshold, yhat > threshold
    tp = int((obs & pred).sum()); fp = int((~obs & pred).sum()); fn = int((obs & ~pred).sum())
    precision = tp / (tp + fp) if tp + fp else np.nan
    recall = tp / (tp + fn) if tp + fn else np.nan
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else np.nan
    return {"threshold": threshold, "events": int(obs.sum()), "base_rate": float(obs.mean()) if len(y) else np.nan,
            "tp": tp, "false_alarms": fp, "missed": fn,
            "precision": precision, "recall": recall, "f1": f1}


def skill_score(model_err: float, baseline_err: float) -> float:
    """1 - model/baseline. Positive = better than the baseline."""
    return float(1 - model_err / baseline_err) if baseline_err else np.nan


def evaluate(frame: pd.DataFrame, pred_col: str, threshold: float, by: str | None = None) -> pd.DataFrame:
    """Regression + event metrics, optionally grouped (e.g. by='split')."""
    groups = frame.groupby(by) if by else [("all", frame)]
    rows = []
    for key, g in groups:
        r = {"group": key, "model": pred_col}
        r.update(regression(g["y"], g[pred_col]))
        r.update({k: v for k, v in events(g["y"], g[pred_col], threshold).items() if k != "threshold"})
        rows.append(r)
    return pd.DataFrame(rows)
