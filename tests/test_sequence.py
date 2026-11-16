"""Tests for the LSTM sequence module (synthetic frame, no files).  Run with:  pytest -q"""
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uaq import models, sequence  # noqa: E402
from uaq.config import load_config  # noqa: E402
from uaq.periods import assign_split, in_months  # noqa: E402

CFG = load_config()
P = CFG["forecast"]["periods"]
CORE, TEST = pd.Timestamp(P["core"]["start"]), pd.Timestamp(P["split"]["test"]["start"])
VAL_END = pd.Timestamp(P["split"]["val"]["end"])
T = 7
FAST = dict(hidden=8, dropout=0.0, lr=1e-2, weight_decay=1e-4, max_epochs=8, patience=3, batch_size=64,
            early_stop_tail=0.15)


def _frame(seed=0):
    """Issue days from 2018 (pre-core) to the end of the test split; inputs carry the next-day signal."""
    rng = np.random.default_rng(seed)
    date = pd.date_range("2018-01-01", "2025-03-30", freq="D")
    target = date + pd.Timedelta(days=1)
    doy, dow = target.day_of_year.to_numpy(), target.day_of_week.to_numpy()
    winter = np.cos(2 * np.pi * (doy - 15) / 365.25).clip(0)
    level = np.exp(3 + 2 * winter + rng.normal(0, 0.3, len(date)))
    f = pd.DataFrame({"date": date, "target_date": target, "pm_lag0": level, "pm_evening": level * 1.1,
                      "pm_last_hour": level * 1.2, "pm_max_hourly": level * 1.5, "pm_mean3d": level,
                      "doy_sin": np.sin(2 * np.pi * doy / 365.25), "doy_cos": np.cos(2 * np.pi * doy / 365.25),
                      "dow_sin": np.sin(2 * np.pi * dow / 7), "dow_cos": np.cos(2 * np.pi * dow / 7),
                      "weekend": (dow >= 5).astype(int), "w_temp_min": -25 * winter + rng.normal(0, 3, len(date)),
                      "w_temp_mean_change": rng.normal(0, 2, len(date)), "w_blh_mean": rng.uniform(50, 500, len(date))})
    f["y"] = np.r_[level[1:], np.nan]
    f["split"] = assign_split(f["target_date"], P["split"]).values
    f["smog"] = in_months(f["target_date"], CFG["smog_months"]).values
    return f


COLS = sequence.sequence_columns(_frame(), CFG, weather=True)


def _lstm(seed=0):
    return sequence.LSTMModel(COLS, seed=seed, **FAST)


def test_sequence_columns_are_daily_inputs_only():
    assert "w_temp_min" in COLS and "pm_lag0" in COLS
    assert not {"w_temp_mean_change", "pm_mean3d"} & set(COLS)
    assert not set(CFG["features"]["drop_columns"]) & set(COLS)          # w_blh_mean is dropped by config
    assert not any(c.startswith("w_") for c in sequence.sequence_columns(_frame(), CFG, weather=False))


def test_folds_respect_core_start_smog_scoring_and_test_seal():
    folds = sequence.sequence_folds(_frame(), COLS, T, CFG)
    assert len(folds) == len(P["rolling_origin_folds"])
    one = pd.Timedelta(days=1)
    for _, Xf, yf, df, Xs, ys, ds in folds:
        assert Xf.shape[1:] == (T, len(COLS)) and len(Xf) == len(yf) == len(df)
        assert df.max() < ds.min() and (df + one).min() >= CORE
        assert ((df + one) < TEST).all() and ((ds + one) < TEST).all()
        assert in_months(pd.Series(ds + one), CFG["smog_months"]).all()


def test_scaler_is_fitted_on_fit_windows_only():
    f = _frame()
    _, Xf, yf, _, Xs, _, ds = sequence.sequence_folds(f, COLS, T, CFG)[0]
    m = _lstm().fit(Xf, yf)
    A = m._prep(Xf)
    np.testing.assert_allclose(m.scaler_.mean_, A.reshape(-1, A.shape[2]).mean(axis=0))
    poisoned = f.copy()
    poisoned.loc[poisoned["date"] >= ds.min(), COLS[:4]] = 1e6     # score-period inputs only
    _, Xf2, yf2, _, _, _, _ = sequence.sequence_folds(poisoned, COLS, T, CFG)[0]
    np.testing.assert_array_equal(m.scaler_.mean_, _lstm().fit(Xf2, yf2).scaler_.mean_)


def test_same_seed_is_reproducible_and_seeds_differ():
    _, Xf, yf, _, Xs, _, _ = sequence.sequence_folds(_frame(), COLS, T, CFG)[0]
    a, b = _lstm(0).fit(Xf, yf).predict(Xs), _lstm(0).fit(Xf, yf).predict(Xs)
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, _lstm(1).fit(Xf, yf).predict(Xs))


def test_poisoning_score_window_targets_does_not_change_predictions():
    f = _frame()
    base = sequence.predictions_lstm(f, COLS, T, lambda: _lstm(0), CFG)
    for fold, *_, ds in sequence.sequence_folds(f, COLS, T, CFG):
        poisoned = f.copy()
        poisoned.loc[poisoned["date"].isin(ds), "y"] = 5000.0
        again = sequence.predictions_lstm(poisoned, COLS, T, lambda: _lstm(0), CFG)
        mine = again["fold"] == fold
        np.testing.assert_array_equal(base.loc[mine, "pred"].values, again.loc[mine, "pred"].values)
        assert (again.loc[mine, "y"] == 5000.0).all()


def test_predictions_are_finite_and_non_negative():
    ens = lambda: models.SeedEnsemble(_lstm, [0, 1])                      # noqa: E731
    p = sequence.predictions_lstm(_frame(), COLS, T, ens, CFG)["pred"].values
    assert len(p) and np.isfinite(p).all() and (p >= 0).all()


def test_history_days_fallback_triggers_only_below_the_minimum():
    f = _frame()
    long_T = CFG["forecast"]["history_days"]
    T_used, n1 = sequence.history_days_rule(f, CFG)
    cfg = copy.deepcopy(CFG)
    cfg["forecast"]["models"]["lstm_min_fit_sequences"] = n1               # exactly enough -> keep default
    assert sequence.history_days_rule(f, cfg) == (long_T, n1)
    cfg["forecast"]["models"]["lstm_min_fit_sequences"] = n1 + 1           # one short -> fall back
    assert sequence.history_days_rule(f, cfg) == (cfg["forecast"]["models"]["lstm_fallback_history_days"], n1)


def test_shared_availability_gives_identical_windows():
    f = _frame()
    f.loc[f.index[::37], "w_temp_min"] = np.nan                          # weather gaps
    nw = sequence.sequence_columns(f, CFG, weather=False)
    a = sequence.sequence_folds(f, COLS, T, CFG)
    b = sequence.sequence_folds(f, nw, T, CFG, availability_cols=COLS)
    for x, z in zip(a, b):
        assert x[3].equals(z[3]) and x[6].equals(z[6]) and z[1].shape[2] == len(nw)


def test_compare_common_days_uses_the_intersection():
    f = _frame()
    days = f.loc[f["smog"] & (f["split"] == "val"), "target_date"].iloc[:20]
    a = pd.DataFrame({"target_date": days.values, "fold": 2, "pred": 50.0})
    b = a.iloc[5:].assign(pred=60.0)
    per_fold, summary = sequence.compare_common_days(f, {"a": a, "b": b}, CFG, threshold=100.0)
    assert (per_fold["n"] == 15).all() and set(per_fold["model"]) == {"persistence", "a", "b"}


def test_fit_final_uses_core_to_validation_end_and_never_reads_test_targets():
    seen = {}

    class Spy(sequence.LSTMModel):
        def fit(self, X, y):
            seen["n"] = len(y)
            return super().fit(X, y)

    f = _frame()
    out = sequence.fit_final_and_predict_lstm(f, COLS, T, lambda: Spy(COLS, **FAST), CFG)
    assert (out["target_date"] >= TEST).all() and "y" not in out.columns and len(out)
    _, _, d = sequence._windows(f, COLS, T)
    tgt = d + pd.Timedelta(days=1)
    assert seen["n"] == int(((tgt >= CORE) & (tgt <= VAL_END)).sum())
    poisoned = f.copy()
    poisoned.loc[poisoned["split"] == "test", "y"] = np.nan                  # test targets unknown or poisoned
    again = sequence.fit_final_and_predict_lstm(poisoned, COLS, T, lambda: Spy(COLS, **FAST), CFG)
    np.testing.assert_array_equal(out["pred"].values, again["pred"].values)
