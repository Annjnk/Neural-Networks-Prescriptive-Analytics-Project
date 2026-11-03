"""Tests for the forecasting models and fold protocol (synthetic frame, no files).  Run with:  pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uaq import models  # noqa: E402
from uaq.config import load_config  # noqa: E402
from uaq.periods import assign_split, in_months  # noqa: E402

CFG = load_config()
P = CFG["forecast"]["periods"]
CORE, TEST = pd.Timestamp(P["core"]["start"]), pd.Timestamp(P["split"]["test"]["start"])
VAL_END = pd.Timestamp(P["split"]["val"]["end"])
COLS = ["pm_lag0", "pm_mean3d", "w_temp_min", "doy_sin", "doy_cos"]
FAST_MLP = dict(hidden=[16], dropout=0.0, weight_decay=1e-4, lr=1e-2, max_epochs=15, patience=5,
                batch_size=64, early_stop_tail=0.15)


def _frame(seed=0):
    """Daily issue days from 2018 (pre-core) to the end of the test split, with a learnable signal."""
    rng = np.random.default_rng(seed)
    date = pd.date_range("2018-01-01", "2025-03-30", freq="D")
    target = date + pd.Timedelta(days=1)
    doy = target.day_of_year.to_numpy()
    winter = np.cos(2 * np.pi * (doy - 15) / 365.25).clip(0)
    temp = -25 * winter + 10 + rng.normal(0, 3, len(date))
    level = np.exp(3 + 2 * winter + rng.normal(0, 0.3, len(date)))
    f = pd.DataFrame({"date": date, "target_date": target, "pm_lag0": level,
                      "pm_mean3d": pd.Series(level).rolling(3, min_periods=1).mean().values,
                      "w_temp_min": temp, "doy_sin": np.sin(2 * np.pi * doy / 365.25),
                      "doy_cos": np.cos(2 * np.pi * doy / 365.25)})
    f["y"] = np.r_[level[1:], np.nan]
    f["split"] = assign_split(f["target_date"], P["split"]).values
    f["smog"] = in_months(f["target_date"], CFG["smog_months"]).values
    return f


def _mlp(seed=0):
    return models.MLPModel(seed=seed, **FAST_MLP)


def test_fold_rows_respect_core_start_smog_scoring_and_test_seal():
    folds = models.fold_rows(_frame(), COLS, CFG)
    assert len(folds) == len(P["rolling_origin_folds"])
    for _, fit, score in folds:
        assert fit["target_date"].max() < score["target_date"].min()
        assert fit["target_date"].min() >= CORE
        assert score["smog"].all() and (~fit["smog"]).any()
        assert (fit["target_date"] < TEST).all() and (score["target_date"] < TEST).all()


def test_ridge_scaler_is_fitted_on_fit_rows_only():
    f = _frame()
    base = next(models.run_folds(f, COLS, lambda: models.RidgeModel(1.0), CFG))
    poisoned = f.copy()
    score_idx = base["score_df"].index
    poisoned.loc[score_idx, COLS] = 0.0
    again = next(models.run_folds(poisoned, COLS, lambda: models.RidgeModel(1.0), CFG))
    np.testing.assert_array_equal(base["model"].scaler_.mean_, again["model"].scaler_.mean_)
    np.testing.assert_allclose(base["model"].scaler_.mean_, base["fit_df"][COLS].mean().values)


def test_mlp_same_seed_is_reproducible_and_seeds_differ():
    _, fit, score = models.fold_rows(_frame(), COLS, CFG)[0]
    X, y, Xs = fit.set_index("target_date")[COLS], fit["y"].values, score.set_index("target_date")[COLS]
    a, b = _mlp(0).fit(X, y).predict(Xs), _mlp(0).fit(X, y).predict(Xs)
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, _mlp(1).fit(X, y).predict(Xs))


def test_poisoning_score_targets_does_not_change_mlp_predictions():
    # Poison each fold's OWN score rows (a later fold may legitimately fit on an earlier fold's window).
    f = _frame()
    base = models.predictions(f, COLS, lambda: _mlp(0), CFG)
    for fold, _, score in models.fold_rows(f, COLS, CFG):
        poisoned = f.copy()
        poisoned.loc[score.index, "y"] = 5000.0
        again = models.predictions(poisoned, COLS, lambda: _mlp(0), CFG)
        mine = again["fold"] == fold
        np.testing.assert_array_equal(base.loc[mine, "pred"].values, again.loc[mine, "pred"].values)
        assert (again.loc[mine, "y"] == 5000.0).all()


def test_all_models_predict_non_negative_finite_values():
    f = _frame()
    factories = {"persistence": models.Persistence, "climatology": models.Climatology,
                 "ridge": lambda: models.RidgeModel(1.0),
                 "hgb": lambda: models.HGBModel(max_depth=3, learning_rate=0.1, max_iter=50),
                 "mlp": lambda: _mlp(0),
                 "ensemble": lambda: models.SeedEnsemble(_mlp, [0, 1])}
    for name, factory in factories.items():
        p = models.predictions(f, COLS, factory, CFG)["pred"].values
        assert np.isfinite(p).all() and (p >= 0).all(), name


def test_seed_ensemble_is_the_mean_of_its_seeds():
    _, fit, score = models.fold_rows(_frame(), COLS, CFG)[0]
    X, Xs = fit.set_index("target_date")[COLS], score.set_index("target_date")[COLS]
    ens = models.SeedEnsemble(_mlp, [0, 1, 2]).fit(X, fit["y"].values)
    per_seed = ens.predict_per_seed(Xs)
    assert per_seed.shape == (3, len(Xs))
    np.testing.assert_allclose(ens.predict(Xs), per_seed.mean(axis=0))


def test_evaluate_extreme_rows_and_skill_use_the_same_rows_as_persistence():
    y = np.arange(1, 21, dtype=float) * 10                        # 10..200
    pers = y + 5                                                 # persistence MAE 5 everywhere
    pred = y - np.r_[np.zeros(18), [40.0, 60.0]]                 # misses only the two worst days
    df = pd.DataFrame({"y": y, "pm_lag0": pers})
    r = models.evaluate(df, pred, threshold=150.0, extreme_quantile=0.90)
    assert r["n_extreme"] == 2                                   # y > 90th percentile = 190, 200
    assert r["mae_extreme"] == 50.0 and r["bias_extreme"] == -50.0
    assert r["skill_mae_extreme"] == 1 - 50.0 / 5.0
    assert np.isclose(r["skill_mae"], 1 - (100.0 / 20) / 5.0)


def test_select_breaks_ties_towards_the_simpler_candidate():
    f = _frame()
    cands = {"simple": models.Persistence, "complex": models.Persistence}
    _, summary = models.select(f, COLS, cands, CFG, threshold=100.0)
    assert summary.index[summary["chosen"]].tolist() == ["simple"]


def test_missingness_rule_uses_training_rows_with_y_and_lag0():
    f = _frame()
    tr = f.index[(f["target_date"] >= CORE) & (f["target_date"] <= pd.Timestamp(P["split"]["train"]["end"]))]
    f["sparse"], f["dense"] = 1.0, 1.0
    f.loc[tr[::4], "sparse"] = np.nan                            # 25% missing -> dropped
    f.loc[tr[::50], "dense"] = np.nan                            # 2% missing -> kept
    f.loc[f.index[f["target_date"] < CORE], "dense"] = np.nan    # pre-core rows do not count
    share, dropped = models.missingness_rule(f, CFG)
    assert "sparse" in dropped and "dense" not in dropped and "pm_lag0" not in dropped


def test_fit_final_fits_only_core_to_validation_end_and_predicts_test():
    seen = {}

    class Spy(models.RidgeModel):
        def fit(self, X, y):
            seen["index"] = X.index
            return super().fit(X, y)

    out = models.fit_final_and_predict(_frame(), COLS, lambda: Spy(1.0), CFG)
    assert seen["index"].min() >= CORE and seen["index"].max() <= VAL_END
    assert (out["target_date"] >= TEST).all() and "y" not in out.columns
