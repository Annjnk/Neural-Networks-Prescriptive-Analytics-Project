"""Tests for the agaar cleaning rules and the network-design GA.  Run with:  pytest -q"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uaq import agaar, network  # noqa: E402
from uaq.config import load_config  # noqa: E402

CFG = load_config()


def _synthetic_network(n_days=120, seed=0):
    """6 stations: A,B,C share one signal, D,E share another, F is noise around both."""
    rng = np.random.default_rng(seed)
    s1, s2 = np.exp(rng.normal(3.5, 0.6, n_days)), np.exp(rng.normal(3.5, 0.6, n_days))
    noise = lambda: np.exp(rng.normal(0, 0.05, n_days))  # noqa: E731
    cols = {"A": s1 * noise(), "B": s1 * 1.2 * noise(), "C": s1 * 0.8 * noise(),
            "D": s2 * noise(), "E": s2 * 1.1 * noise(), "F": np.sqrt(s1 * s2) * noise()}
    w = pd.DataFrame(cols, index=pd.date_range("2025-10-01", periods=n_days))
    fit = np.arange(n_days) < 80
    return network.Problem(w, fit, ~fit, threshold=50.0)


def test_best_pair_takes_one_station_from_each_group():
    p = _synthetic_network()
    best = network.exhaustive(p, 2)
    groups = {p.stations[i] in ("A", "B", "C") for i in best}
    assert groups == {True, False}, p.names(best)


def test_fitness_falls_as_more_stations_are_kept():
    p = _synthetic_network()
    f = [p.fitness(network.exhaustive(p, k)) for k in (1, 2, 3, 4)]
    assert all(a >= b for a, b in zip(f, f[1:]))


def test_ga_always_returns_k_distinct_stations_and_finds_optimum():
    p = _synthetic_network()
    for k in (2, 3, 4):
        S, f, hist = network.genetic_algorithm(p, k, np.random.default_rng(1), population=20, generations=30)
        assert len(set(S)) == k == len(S) and all(0 <= i < p.n for i in S)
        assert math.isclose(f, p.fitness(network.exhaustive(p, k)))
        assert all(a >= b for a, b in zip(hist, hist[1:]))     # elitism: best never gets worse


def test_fitness_is_scored_only_on_the_eval_rows():
    p = _synthetic_network()
    base = p.fitness((0, 3))
    w = p.wide.copy()
    w.iloc[:80, 5] = w.iloc[:80, 5] * 1.0                      # unchanged fit rows
    w.iloc[80:, 1] = 9999.0                                    # poison eval rows of a dropped station
    poisoned = network.Problem(w, p.fit_mask, p.eval_mask, threshold=50.0)
    assert poisoned.fitness((0, 3)) > base * 10


def test_spatial_spread_picks_far_apart_stations():
    p = _synthetic_network()
    lat = np.array([47.90, 47.901, 47.902, 47.95, 47.951, 47.92])
    lon = np.array([106.80, 106.801, 106.802, 107.00, 107.001, 106.90])
    S = network.spatial_spread(p, 3, lat, lon)
    assert {0, 1, 2} & set(S) and {3, 4} & set(S)


def test_cleaning_rules():
    dates = pd.date_range("2025-11-01", periods=6)
    d = pd.DataFrame({"station_code": ["X"] * 6 + ["Y"] * 6 + ["Z"] * 6,
                      "date": list(dates) * 3,
                      "pm25_raw": [40, 40, 40, 40, 40, 41] + [-5, 60, 55, 900, 50, 52] + [45, 70, 50, 52, 48, 51],
                      "pm10": [np.nan] * 6 + [np.nan, 40, np.nan, np.nan, np.nan, np.nan] + [np.nan] * 6})
    out = agaar.clean(d, CFG["network"]["cleaning"]).set_index(["station_code", "date"])["dropped_reason"]
    assert (out.loc["X"].iloc[:5] == "flat").all() and out.loc["X"].iloc[5] == ""
    assert out.loc[("Y", dates[0])] == "negative"
    assert out.loc[("Y", dates[1])] == "above_pm10"
    assert out.loc[("Y", dates[3])] == "spike"
    assert (out.loc["Z"] == "").all()
