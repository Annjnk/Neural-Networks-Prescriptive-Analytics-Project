"""Next-day PM2.5 forecasting models and the rolling-fold selection protocol (notebook 05).

Protocol (CLAUDE.md, hard rules 2-6):
  - fit rows: core.start <= target_date <= fold train_end, every month (forecast.models.fit_on);
  - score rows: the fold's validation window, smog-season targets only;
  - select on the mean MAE of the rolling folds; the test split is never fitted, scored or returned,
    except by fit_final_and_predict(), which runs once, on 23 November;
  - every scaler, target transform and climatology is fitted on the fit rows only;
  - skill is always measured against persistence on the same rows.

Models take X as a DataFrame indexed by target_date (climatology uses the index) and y in µg/m³,
and predict µg/m³, clipped at 0.
"""
from __future__ import annotations

import copy
import itertools
import math

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from . import metrics, thresholds
from .features import complete_rows, feature_columns
from .periods import assign_split, in_months


# ---------------------------------------------------------------------- rules and rows
def missingness_rule(frame: pd.DataFrame, cfg: dict) -> tuple[pd.Series, list[str]]:
    """Share missing per feature on training rows (core.start .. train end, all months) where y and
    pm_lag0 are both present, and the features above features.max_missing_share. pm_lag0 is never dropped."""
    p = cfg["forecast"]["periods"]
    td = frame["target_date"]
    base = frame[(td >= pd.Timestamp(p["core"]["start"])) & (td <= pd.Timestamp(p["split"]["train"]["end"]))
                 & frame["y"].notna() & frame["pm_lag0"].notna()]
    share = base[feature_columns(frame)].isna().mean().sort_values(ascending=False)
    dropped = [c for c, v in share.items() if v > cfg["features"]["max_missing_share"] and c != "pm_lag0"]
    return share, dropped


def event_threshold(frame: pd.DataFrame, cfg: dict) -> tuple[str, float, str]:
    """Event threshold from training smog-season daily means of day D, exactly as notebook 01."""
    d = frame["date"]
    train = (assign_split(d, cfg["forecast"]["periods"]["split"]) == "train").values
    smog = in_months(d, cfg["smog_months"]).values
    return thresholds.choose(thresholds.base_rates(frame.loc[train & smog, "pm_lag0"], cfg), cfg)


def fold_rows(frame: pd.DataFrame, cols: list[str], cfg: dict) -> list[tuple[int, pd.DataFrame, pd.DataFrame]]:
    """(fold_no, fit_df, score_df) per rolling fold, both complete on `cols` + y."""
    p, m = cfg["forecast"]["periods"], cfg["forecast"]["models"]
    core, test_start = pd.Timestamp(p["core"]["start"]), pd.Timestamp(p["split"]["test"]["start"])
    cc = complete_rows(frame, cols)
    cc = cc[cc["target_date"] < test_start]                    # the test split never enters a fold
    td = cc["target_date"]
    out = []
    for i, f in enumerate(p["rolling_origin_folds"], 1):
        fit = cc[(td >= core) & (td <= pd.Timestamp(f["train_end"]))]
        if m.get("fit_on", "all_months") == "smog_only":
            fit = fit[fit["smog"]]
        score = cc[(td >= pd.Timestamp(f["val_start"])) & (td <= pd.Timestamp(f["val_end"])) & cc["smog"]]
        assert len(fit) and len(score), f"fold {i} has no fit or score rows"
        assert fit["target_date"].max() < score["target_date"].min(), f"fold {i}: fit rows overlap score rows"
        out.append((i, fit.sort_values("target_date"), score.sort_values("target_date")))
    return out


def _X(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df.set_index("target_date")[list(cols)]


# ---------------------------------------------------------------------- models
class Persistence:
    """Tomorrow = today (pm_lag0)."""
    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.clip(X["pm_lag0"].to_numpy(float), 0, None)


class Climatology:
    """Mean y by target day of year (fit rows only), smoothed with a circular ±window-day mean."""
    def __init__(self, window: int = 7):
        self.window = window

    def fit(self, X, y):
        doy = np.minimum(X.index.day_of_year, 365)
        prof = pd.Series(np.asarray(y, float)).groupby(doy).mean().reindex(range(1, 366))
        w = self.window
        vals = np.r_[prof.values[-w:], prof.values, prof.values[:w]]
        sm = pd.Series(vals).rolling(2 * w + 1, center=True, min_periods=1).mean().values[w:-w]
        self.table_ = pd.Series(sm, index=range(1, 366)).fillna(float(np.mean(y)))
        return self

    def predict(self, X):
        return np.clip(self.table_.reindex(np.minimum(X.index.day_of_year, 365)).to_numpy(float), 0, None)


class RidgeModel:
    """StandardScaler + Ridge on log1p(y)."""
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, X, y):
        self.scaler_ = StandardScaler().fit(X.to_numpy(float))
        self.model_ = Ridge(alpha=self.alpha).fit(self.scaler_.transform(X.to_numpy(float)), np.log1p(y))
        return self

    def predict(self, X):
        z = self.model_.predict(self.scaler_.transform(X.to_numpy(float)))
        return np.clip(np.expm1(z), 0, None)


class HGBModel:
    """HistGradientBoostingRegressor on log1p(y)."""
    def __init__(self, **params):
        self.params = params

    def fit(self, X, y):
        self.model_ = HistGradientBoostingRegressor(random_state=0, **self.params).fit(X.to_numpy(float), np.log1p(y))
        return self

    def predict(self, X):
        return np.clip(np.expm1(self.model_.predict(X.to_numpy(float))), 0, None)


class MLPModel:
    """Linear-ReLU-Dropout stack + Linear(1) on standardised log1p(y), CPU.

    Early stopping uses the chronologically last `early_stop_tail` of the FIT rows; the best
    weights are restored. history_ holds per-epoch train and early-stop MSE (standardised scale).
    """
    def __init__(self, hidden=(32,), dropout=0.0, weight_decay=1e-4, lr=1e-3, max_epochs=300,
                 patience=20, batch_size=64, early_stop_tail=0.15, seed=0):
        self.hidden, self.dropout, self.weight_decay, self.lr = list(hidden), dropout, weight_decay, lr
        self.max_epochs, self.patience, self.batch_size = max_epochs, patience, batch_size
        self.early_stop_tail, self.seed = early_stop_tail, seed

    def _net(self, n_in):
        layers, d = [], n_in
        for h in self.hidden:
            layers += [torch.nn.Linear(d, h), torch.nn.ReLU(), torch.nn.Dropout(self.dropout)]
            d = h
        return torch.nn.Sequential(*layers, torch.nn.Linear(d, 1))

    def fit(self, X, y):
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        order = np.argsort(X.index.values, kind="stable")              # chronological
        Xa, ya = X.to_numpy(float)[order], np.log1p(np.asarray(y, float))[order]
        self.scaler_ = StandardScaler().fit(Xa)
        self.y_mean_, self.y_std_ = float(ya.mean()), float(ya.std()) or 1.0
        Xt = torch.tensor(self.scaler_.transform(Xa), dtype=torch.float32)
        yt = torch.tensor((ya - self.y_mean_) / self.y_std_, dtype=torch.float32).unsqueeze(1)
        n_es = max(1, int(math.ceil(len(Xa) * self.early_stop_tail)))
        Xtr, ytr, Xes, yes = Xt[:-n_es], yt[:-n_es], Xt[-n_es:], yt[-n_es:]

        self.net_ = self._net(Xt.shape[1])
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn, gen = torch.nn.MSELoss(), torch.Generator().manual_seed(self.seed)
        best, best_state, wait, self.history_ = np.inf, None, 0, []
        for epoch in range(self.max_epochs):
            self.net_.train()
            for idx in torch.randperm(len(Xtr), generator=gen).split(self.batch_size):
                opt.zero_grad()
                loss_fn(self.net_(Xtr[idx]), ytr[idx]).backward()
                opt.step()
            self.net_.eval()
            with torch.no_grad():
                tr_loss, es_loss = loss_fn(self.net_(Xtr), ytr).item(), loss_fn(self.net_(Xes), yes).item()
            self.history_.append({"epoch": epoch + 1, "train": tr_loss, "early_stop": es_loss})
            if es_loss < best - 1e-9:
                best, best_state, wait, self.best_epoch_ = es_loss, copy.deepcopy(self.net_.state_dict()), 0, epoch + 1
            else:
                wait += 1
                if wait >= self.patience:
                    break
        self.net_.load_state_dict(best_state)
        self.net_.eval()
        return self

    def predict(self, X):
        with torch.no_grad():
            z = self.net_(torch.tensor(self.scaler_.transform(X.to_numpy(float)), dtype=torch.float32))
        return np.clip(np.expm1(z.numpy().ravel() * self.y_std_ + self.y_mean_), 0, None)


class SeedEnsemble:
    """One model per seed; predicts the mean of the seeds' predictions."""
    def __init__(self, model_factory, seeds):
        self.model_factory, self.seeds = model_factory, list(seeds)

    def fit(self, X, y):
        self.models_ = [self.model_factory(s).fit(X, y) for s in self.seeds]
        return self

    def predict_per_seed(self, X) -> np.ndarray:
        return np.vstack([m.predict(X) for m in self.models_])

    def predict(self, X):
        return self.predict_per_seed(X).mean(axis=0)


# ---------------------------------------------------------------------- candidates
def ridge_candidates(cfg: dict) -> dict:
    """Simplest first (largest alpha), so ties go to the simpler model."""
    return {f"ridge alpha={a:g}": (lambda a=a: RidgeModel(a))
            for a in sorted(cfg["forecast"]["models"]["ridge_alphas"], reverse=True)}


def hgb_candidates(cfg: dict) -> dict:
    """Simplest first: shallower trees, then the lower learning rate."""
    g = cfg["forecast"]["models"]["hgb_grid"]
    out = {}
    for d, lr, it in itertools.product(sorted(g["max_depth"]), sorted(g["learning_rate"]), sorted(g["max_iter"])):
        out[f"hgb depth={d} lr={lr:g} iter={it}"] = (lambda d=d, lr=lr, it=it:
                                                      HGBModel(max_depth=d, learning_rate=lr, max_iter=it))
    return out


def mlp_candidates(cfg: dict) -> dict:
    """5-seed ensembles; simplest first: fewer weights, then more dropout."""
    m = cfg["forecast"]["models"]
    g = m["mlp_grid"]
    fixed = {k: g[k] for k in ("max_epochs", "patience", "batch_size", "early_stop_tail")}
    out = {}
    for h, dr, wd, lr in itertools.product(sorted(g["hidden"], key=lambda h: (len(h), sum(h))),
                                           sorted(g["dropout"], reverse=True),
                                           g["weight_decay"], g["lr"]):
        params = dict(hidden=list(h), dropout=dr, weight_decay=wd, lr=lr, **fixed)
        name = f"mlp hidden={list(h)} dropout={dr:g} wd={wd:g} lr={lr:g}"
        out[name] = (lambda params=params: SeedEnsemble(lambda s: MLPModel(seed=s, **params), m["seeds"]))
    return out


# ---------------------------------------------------------------------- scoring
def evaluate(score_df: pd.DataFrame, pred, threshold: float, extreme_quantile: float) -> dict:
    """Regression + event metrics, skill vs persistence on the same rows, and worst-day metrics."""
    y, p = score_df["y"].to_numpy(float), np.asarray(pred, float)
    pers = score_df["pm_lag0"].to_numpy(float)
    out = metrics.regression(y, p)
    out.update({k: v for k, v in metrics.events(y, p, threshold).items() if k != "threshold"})
    out["skill_mae"] = metrics.skill_score(out["mae"], metrics.regression(y, pers)["mae"])
    ext = y > np.quantile(y, extreme_quantile)
    out["n_extreme"] = int(ext.sum())
    out["mae_extreme"] = float(np.abs(p[ext] - y[ext]).mean()) if ext.any() else np.nan
    out["bias_extreme"] = float((p[ext] - y[ext]).mean()) if ext.any() else np.nan
    out["skill_mae_extreme"] = metrics.skill_score(out["mae_extreme"], float(np.abs(pers[ext] - y[ext]).mean())) \
        if ext.any() else np.nan
    return out


def run_folds(frame: pd.DataFrame, cols: list[str], factory, cfg: dict):
    """Fit a fresh model per fold on the fit rows and predict the score rows."""
    for fold, fit_df, score_df in fold_rows(frame, cols, cfg):
        model = factory().fit(_X(fit_df, cols), fit_df["y"].to_numpy(float))
        yield {"fold": fold, "fit_df": fit_df, "score_df": score_df, "model": model,
               "pred": model.predict(_X(score_df, cols))}


def select(frame: pd.DataFrame, cols: list[str], candidates: dict, cfg: dict, threshold: float):
    """(per-fold table, summary). The chosen candidate has the lowest mean MAE over folds;
    ties (equal to 0.01 µg/m³) go to the earlier, i.e. simpler, candidate."""
    q = cfg["forecast"]["models"]["extreme_quantile"]
    rows = []
    for order, (name, factory) in enumerate(candidates.items()):
        for r in run_folds(frame, cols, factory, cfg):
            rows.append({"candidate": name, "order": order, "fold": r["fold"], "n_fit": len(r["fit_df"]),
                         **evaluate(r["score_df"], r["pred"], threshold, q)})
    per_fold = pd.DataFrame(rows)
    agg = {c: "mean" for c in ("mae", "rmse", "skill_mae", "precision", "recall", "f1",
                               "mae_extreme", "bias_extreme", "skill_mae_extreme")}
    agg.update({"missed": "sum", "events": "sum", "order": "first"})
    summary = per_fold.groupby("candidate", sort=False).agg(agg).rename(columns={"mae": "mean_mae"})
    summary = summary.assign(key=summary["mean_mae"].round(2)).sort_values(["key", "order"])
    summary["chosen"] = False
    summary.iloc[0, summary.columns.get_loc("chosen")] = True
    return per_fold.drop(columns="order"), summary.drop(columns=["key", "order"])


def predictions(frame: pd.DataFrame, cols: list[str], factory, cfg: dict) -> pd.DataFrame:
    """target_date, fold, y, pred for every fold's score rows."""
    return pd.concat([pd.DataFrame({"target_date": r["score_df"]["target_date"].values, "fold": r["fold"],
                                    "y": r["score_df"]["y"].values, "pred": r["pred"]})
                      for r in run_folds(frame, cols, factory, cfg)], ignore_index=True)


def fit_final_and_predict(frame: pd.DataFrame, cols: list[str], factory, cfg: dict) -> pd.DataFrame:
    """Fit on core.start <= target_date <= validation end, predict the test split.
    Runs ONCE, on 23 November (CLAUDE.md hard rule 1); returns target_date, smog, pred only."""
    p = cfg["forecast"]["periods"]
    cc = complete_rows(frame, cols)
    fit = cc[(cc["target_date"] >= pd.Timestamp(p["core"]["start"]))
             & (cc["target_date"] <= pd.Timestamp(p["split"]["val"]["end"]))]
    if cfg["forecast"]["models"].get("fit_on", "all_months") == "smog_only":
        fit = fit[fit["smog"]]
    test = frame[(frame["split"] == "test")].dropna(subset=list(cols))
    model = factory().fit(_X(fit, cols), fit["y"].to_numpy(float))
    return pd.DataFrame({"target_date": test["target_date"].values, "smog": test["smog"].values,
                         "pred": model.predict(_X(test, cols))})
