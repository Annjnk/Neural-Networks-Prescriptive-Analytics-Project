"""Monitoring-network design: which k stations should a reduced network keep?

Fitness of a kept subset S (lower is better), on a fit period and a scoring period:

    F(S) = mean_j wRMSE_j(S) / sigma  +  lambda * MissRate(S)

  wRMSE_j   for every DROPPED station j: a ridge model predicts j's daily mean from the
            kept stations on the same day (fitted on the fit period, scored on the scoring
            period, in µg/m³); days above the event threshold weigh `event_weight` times.
  sigma     standard deviation of all observed daily values in the fit period (puts wRMSE
            on a unit-free scale so lambda is comparable across datasets).
  MissRate  share of exceedance days AT DROPPED STATIONS (observed daily mean above the
            threshold) that the reduced network predicts below the threshold, i.e. local
            high-pollution days that would go unreported if those stations were removed.

Selection methods compared at each k: random, highest-mean, spatial spread, greedy forward,
exhaustive (when C(N, k) is small enough) and the genetic algorithm.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd


@dataclass
class Problem:
    wide: pd.DataFrame            # date x station, NaN = missing
    fit_mask: np.ndarray          # rows used to fit the ridge models
    eval_mask: np.ndarray         # rows used to score
    threshold: float
    ridge_alpha: float = 1.0
    log_transform: bool = True
    event_weight: float = 3.0
    lambda_miss: float = 0.5
    _cache: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        X = self.wide.to_numpy(float)
        self.stations = list(self.wide.columns)
        self.n = X.shape[1]
        self.X = X
        self.obs = ~np.isnan(X)
        Z = np.log1p(np.clip(X, 0, None)) if self.log_transform else X.copy()
        fit = self.fit_mask
        mu = np.nanmean(Z[fit], axis=0)
        sd = np.nanstd(Z[fit], axis=0)
        sd[sd == 0] = 1.0
        self.Z = Z
        # Standardised predictors; a missing kept-station value becomes 0 (= its fit-period mean).
        self.P = np.nan_to_num((Z - mu) / sd, nan=0.0)
        self.sigma = float(np.nanstd(X[fit]))

    # ------------------------------------------------------------------ fitness
    def details(self, subset) -> dict:
        S = tuple(sorted(subset))
        if S in self._cache:
            return self._cache[S]
        A = np.column_stack([np.ones(len(self.X)), self.P[:, S]])
        pen = self.ridge_alpha * np.eye(A.shape[1]); pen[0, 0] = 0.0
        errs, events, missed, false_alarms = [], 0, 0, 0
        for j in range(self.n):
            if j in S:
                continue
            fr = self.fit_mask & self.obs[:, j]
            Af = A[fr]
            beta = np.linalg.solve(Af.T @ Af + pen, Af.T @ self.Z[fr, j])
            pred = A @ beta
            pred = np.expm1(pred) if self.log_transform else pred
            er = self.eval_mask & self.obs[:, j]
            y, p = self.X[er, j], pred[er]
            w = np.where(y > self.threshold, self.event_weight, 1.0)
            errs.append(math.sqrt(float((w * (p - y) ** 2).sum() / w.sum())) if len(y) else np.nan)
            ev = y > self.threshold
            events += int(ev.sum())
            missed += int((ev & (p <= self.threshold)).sum())
            false_alarms += int((~ev & (p > self.threshold)).sum())
        wrmse = float(np.nanmean(errs)) if errs else 0.0
        miss_rate = missed / events if events else 0.0
        out = {"F": wrmse / self.sigma + self.lambda_miss * miss_rate, "wrmse": wrmse,
               "miss_rate": miss_rate, "events": events, "missed": missed, "false_alarms": false_alarms}
        self._cache[S] = out
        return out

    def fitness(self, subset) -> float:
        return self.details(subset)["F"]

    def names(self, subset) -> list[str]:
        return [self.stations[i] for i in sorted(subset)]


# ---------------------------------------------------------------------- selectors
def random_search(prob: Problem, k: int, draws: int, rng: np.random.Generator):
    best, scores = None, []
    for _ in range(draws):
        S = tuple(sorted(rng.choice(prob.n, k, replace=False)))
        f = prob.fitness(S)
        scores.append(f)
        if best is None or f < best[1]:
            best = (S, f)
    return best[0], np.array(scores)


def highest_mean(prob: Problem, k: int):
    means = np.nanmean(prob.X[prob.fit_mask], axis=0)
    return tuple(sorted(np.argsort(-means)[:k]))


def spatial_spread(prob: Problem, k: int, lat: np.ndarray, lon: np.ndarray):
    """Greedy k-centre on coordinates: start nearest the centroid, then add the farthest station."""
    x = (lon - lon.mean()) * 111.32 * math.cos(math.radians(lat.mean()))
    y = (lat - lat.mean()) * 110.57
    D = np.hypot(x[:, None] - x[None], y[:, None] - y[None])
    S = [int(np.argmin(np.hypot(x, y)))]
    while len(S) < k:
        S.append(int(np.argmax(D[:, S].min(axis=1))))
    return tuple(sorted(S))


def greedy_forward(prob: Problem, k_max: int) -> dict[int, tuple]:
    """Nested subsets for k = 1..k_max, each adding the station that lowers F most."""
    S, out = (), {}
    for k in range(1, k_max + 1):
        cand = [tuple(sorted(S + (j,))) for j in range(prob.n) if j not in S]
        S = min(cand, key=prob.fitness)
        out[k] = S
    return out


def exhaustive(prob: Problem, k: int):
    return min(combinations(range(prob.n), k), key=prob.fitness)


def genetic_algorithm(prob: Problem, k: int, rng: np.random.Generator, population: int = 60,
                      generations: int = 100, tournament: int = 3, elite: int = 2,
                      mutation_rate: float = 0.3):
    """Set-encoded GA: every individual is k distinct station indices, so all are feasible.

    Crossover keeps the parents' common stations and fills the rest from the stations only
    one parent has. Mutation swaps one kept station for one dropped station.
    Returns (best subset, best F, best-F-per-generation history).
    """
    n = prob.n

    def rand_ind():
        return tuple(sorted(rng.choice(n, k, replace=False)))

    def pick(pop, fit):
        idx = rng.choice(len(pop), tournament, replace=False)
        return pop[min(idx, key=lambda i: fit[i])]

    def crossover(a, b):
        common = set(a) & set(b)
        rest = list((set(a) | set(b)) - common)
        rng.shuffle(rest)
        return tuple(sorted(common | set(rest[: k - len(common)])))

    def mutate(s):
        if rng.random() >= mutation_rate or k == n:
            return s
        s = set(s)
        out_ = rng.choice(sorted(s)); in_ = rng.choice(sorted(set(range(n)) - s))
        return tuple(sorted((s - {out_}) | {in_}))

    pop = [rand_ind() for _ in range(population)]
    history = []
    for _ in range(generations):
        fit = [prob.fitness(s) for s in pop]
        order = np.argsort(fit)
        history.append(fit[order[0]])
        nxt = [pop[i] for i in order[:elite]]
        while len(nxt) < population:
            nxt.append(mutate(crossover(pick(pop, fit), pick(pop, fit))))
        pop = nxt
    fit = [prob.fitness(s) for s in pop]
    i = int(np.argmin(fit))
    return pop[i], fit[i], history


# ---------------------------------------------------------------------- comparison
def compare(prob: Problem, k_values, cfg: dict, lat=None, lon=None, run_exhaustive: bool = True) -> pd.DataFrame:
    """Run every method at every k on `prob`. Returns one row per (k, method)."""
    net = cfg["network"]
    ga_cfg = {key: v for key, v in net["ga"].items() if key != "seeds"}
    greedy = greedy_forward(prob, max(k_values))
    rows = []

    def add(k, method, S, secs, **extra):
        d = prob.details(S)
        rows.append({"k": k, "method": method, "F": d["F"], "wrmse": d["wrmse"], "miss_rate": d["miss_rate"],
                     "stations": prob.names(S), "seconds": round(secs, 2), **extra})

    for k in k_values:
        rng = np.random.default_rng(cfg["seed"] + k)
        t = time.time(); S, scores = random_search(prob, k, net["random_draws"], rng)
        add(k, "random (best of draws)", S, time.time() - t, random_median_F=float(np.median(scores)))
        add(k, "highest mean", highest_mean(prob, k), 0.0)
        if lat is not None:
            add(k, "spatial spread", spatial_spread(prob, k, lat, lon), 0.0)
        add(k, "greedy forward", greedy[k], 0.0)
        if run_exhaustive and math.comb(prob.n, k) <= net["exhaustive_max_subsets"]:
            t = time.time(); add(k, "exhaustive (optimum)", exhaustive(prob, k), time.time() - t)
        ga_runs = []
        for seed in net["ga"]["seeds"]:
            t = time.time()
            S, f, _ = genetic_algorithm(prob, k, np.random.default_rng(seed), **ga_cfg)
            ga_runs.append((f, S, time.time() - t))
        f_best, S_best, _ = min(ga_runs, key=lambda r: r[0])
        add(k, "genetic algorithm", S_best, float(np.mean([r[2] for r in ga_runs])),
            ga_seed_F_mean=float(np.mean([r[0] for r in ga_runs])), ga_seed_F_worst=float(max(r[0] for r in ga_runs)))
    return pd.DataFrame(rows)
