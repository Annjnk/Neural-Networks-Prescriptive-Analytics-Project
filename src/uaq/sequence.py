"""LSTM forecasts on windows of consecutive days (notebook 06).

Same protocol as uaq.models (CLAUDE.md, hard rules 2-6): fit windows have targets from core.start to the
fold's train end, all months; score windows have smog-season targets in the fold's validation window; test
windows are never returned, except by fit_final_and_predict_lstm(), which runs once, on the single test run (notebook 09), after the tag pre-test-v2.
A window ends on issue day D (its target is D+1) and comes from uaq.features.make_sequences, so it never
spans a gap. Windows are kept in chronological order, which the early-stopping split relies on.
"""
from __future__ import annotations

import copy
import itertools
import math

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from .features import make_sequences
from .models import SeedEnsemble, _inputs, evaluate


# ---------------------------------------------------------------------- columns, windows, folds
def sequence_columns(frame: pd.DataFrame, cfg: dict, weather: bool = True) -> list[str]:
    """forecast.models.sequence_columns, plus (with weather) every w_* column except *_change, *_tplus1
    and anything in features.drop_columns."""
    drop = set(cfg["features"]["drop_columns"])
    cols = [c for c in cfg["forecast"]["models"]["sequence_columns"] if c not in drop]
    if weather:
        cols += [c for c in frame.columns if c.startswith("w_") and not c.endswith(("_change", "_tplus1"))
                 and c not in drop]
    return cols


def _windows(frame: pd.DataFrame, cols: list[str], history_days: int, availability_cols=None):
    """Windows complete on `cols` (and on `availability_cols`, so two feature sets can share windows)."""
    extra = [c for c in (availability_cols or []) if c not in cols]
    X, y, d = make_sequences(frame, list(cols) + extra, history_days)
    return X[:, :, :len(cols)], y, pd.DatetimeIndex(d)


def _fold_masks(issue: pd.DatetimeIndex, smog: np.ndarray, cfg: dict):
    p, m = cfg["forecast"]["periods"], cfg["forecast"]["models"]
    target = issue + pd.Timedelta(days=1)
    open_ = target < pd.Timestamp(p["split"]["test"]["start"])          # the test split never enters a fold
    for i, f in enumerate(p["rolling_origin_folds"], 1):
        fit = open_ & (target >= pd.Timestamp(p["core"]["start"])) & (target <= pd.Timestamp(f["train_end"]))
        if m.get("fit_on", "all_months") == "smog_only":
            fit = fit & smog
        score = open_ & (target >= pd.Timestamp(f["val_start"])) & (target <= pd.Timestamp(f["val_end"])) & smog
        yield i, np.asarray(fit), np.asarray(score)


def _smog(frame: pd.DataFrame, issue: pd.DatetimeIndex) -> np.ndarray:
    return frame.set_index("date")["smog"].reindex(issue).fillna(False).to_numpy(bool)


def sequence_folds(frame: pd.DataFrame, cols: list[str], history_days: int, cfg: dict, availability_cols=None):
    """(fold_no, X_fit, y_fit, d_fit, X_score, y_score, d_score) per rolling fold; d = issue dates (window ends)."""
    X, y, d = _windows(frame, cols, history_days, availability_cols)
    out = []
    for i, fit, score in _fold_masks(d, _smog(frame, d), cfg):
        assert fit.any() and score.any(), f"fold {i} has no fit or score windows"
        assert d[fit].max() < d[score].min(), f"fold {i}: fit windows overlap score windows"
        out.append((i, X[fit], y[fit], d[fit], X[score], y[score], d[score]))
    return out


def history_days_rule(frame: pd.DataFrame, cfg: dict) -> tuple[int, int]:
    """(history_days to use, fold-1 fit windows at forecast.history_days), counted on the with-weather
    sequence columns. Falls back to lstm_fallback_history_days if fold 1 has fewer than lstm_min_fit_sequences."""
    m, T = cfg["forecast"]["models"], cfg["forecast"]["history_days"]
    _, _, d = _windows(frame, sequence_columns(frame, cfg, weather=True), T)
    _, fit, _ = next(_fold_masks(d, _smog(frame, d), cfg))
    n1 = int(fit.sum())
    return (T if n1 >= m["lstm_min_fit_sequences"] else m["lstm_fallback_history_days"]), n1


# ---------------------------------------------------------------------- model
class _LSTMNet(torch.nn.Module):
    def __init__(self, n_in: int, hidden: int, dropout: float):
        super().__init__()
        self.lstm = torch.nn.LSTM(n_in, hidden, batch_first=True)
        self.drop = torch.nn.Dropout(dropout)
        self.out = torch.nn.Linear(hidden, 1)

    def forward(self, x):
        h, _ = self.lstm(x)
        return self.out(self.drop(h[:, -1, :]))


class LSTMModel:
    """1-layer LSTM -> dropout -> Linear(1) on standardised log1p(y), CPU.

    Inputs: log1p on PM2.5 columns (uaq.models._inputs), then a per-feature StandardScaler fitted on the fit
    windows. X must be in chronological order: early stopping uses the last `early_stop_tail` of the FIT
    windows. Best weights are restored; history_ holds per-epoch train and early-stop MSE.
    """
    def __init__(self, columns, hidden=32, dropout=0.0, lr=1e-3, weight_decay=1e-4, max_epochs=300,
                 patience=20, batch_size=32, early_stop_tail=0.15, seed=0, log_prefix="pm_"):
        self.columns, self.hidden, self.dropout, self.lr = list(columns), hidden, dropout, lr
        self.weight_decay, self.max_epochs, self.patience = weight_decay, max_epochs, patience
        self.batch_size, self.early_stop_tail, self.seed, self.log_prefix = batch_size, early_stop_tail, seed, log_prefix

    def _prep(self, X: np.ndarray) -> np.ndarray:
        n, T, F = X.shape
        return _inputs(pd.DataFrame(X.reshape(n * T, F), columns=self.columns), self.log_prefix).reshape(n, T, F)

    def _scale(self, A: np.ndarray) -> torch.Tensor:
        n, T, F = A.shape
        return torch.tensor(self.scaler_.transform(A.reshape(n * T, F)).reshape(n, T, F), dtype=torch.float32)

    def fit(self, X, y):
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        A = self._prep(np.asarray(X, float))
        self.scaler_ = StandardScaler().fit(A.reshape(-1, A.shape[2]))
        ly = np.log1p(np.asarray(y, float))
        self.y_mean_, self.y_std_ = float(ly.mean()), float(ly.std()) or 1.0
        Xt = self._scale(A)
        yt = torch.tensor((ly - self.y_mean_) / self.y_std_, dtype=torch.float32).unsqueeze(1)
        n_es = max(1, int(math.ceil(len(A) * self.early_stop_tail)))
        Xtr, ytr, Xes, yes = Xt[:-n_es], yt[:-n_es], Xt[-n_es:], yt[-n_es:]

        self.net_ = _LSTMNet(A.shape[2], self.hidden, self.dropout)
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
            z = self.net_(self._scale(self._prep(np.asarray(X, float))))
        return np.clip(np.expm1(z.numpy().ravel() * self.y_std_ + self.y_mean_), 0, None)


def lstm_candidates(cfg: dict, cols: list[str]) -> dict:
    """5-seed LSTM ensembles over lstm_grid; smaller first (fewer units, then more dropout), so ties go to it."""
    m = cfg["forecast"]["models"]
    g = m["lstm_grid"]
    fixed = {k: g[k] for k in ("max_epochs", "patience", "batch_size", "early_stop_tail")}
    out = {}
    for h, dr, wd, lr in itertools.product(sorted(g["hidden"]), sorted(g["dropout"], reverse=True),
                                           g["weight_decay"], g["lr"]):
        params = dict(hidden=h, dropout=dr, weight_decay=wd, lr=lr, log_prefix=m["log_feature_prefix"], **fixed)
        name = f"lstm hidden={h} dropout={dr:g} wd={wd:g} lr={lr:g}"
        out[name] = (lambda params=params: SeedEnsemble(lambda s: LSTMModel(cols, seed=s, **params), m["seeds"]))
    return out


# ---------------------------------------------------------------------- folds, selection, predictions
def _score_df(frame: pd.DataFrame, issue: pd.DatetimeIndex) -> pd.DataFrame:
    """y and pm_lag0 (persistence) for the given issue days, as uaq.models.evaluate expects."""
    return frame.set_index("date").loc[issue, ["y", "pm_lag0"]].reset_index(drop=True)


def run_lstm_folds(frame, cols, history_days, factory, cfg, availability_cols=None):
    """Fit a fresh model per fold on the fit windows and predict the score windows."""
    for fold, Xf, yf, df, Xs, ys, ds in sequence_folds(frame, cols, history_days, cfg, availability_cols):
        model = factory().fit(Xf, yf)
        yield {"fold": fold, "model": model, "n_fit": len(yf), "X_score": Xs, "y_score": ys,
               "issue": ds, "pred": model.predict(Xs)}


def select_lstm(frame, cols, history_days, candidates: dict, cfg, threshold, availability_cols=None):
    """(per-fold table, summary). Lowest mean MAE over folds wins; ties (equal to 0.01 µg/m³) go to the
    earlier, i.e. smaller, candidate."""
    q = cfg["forecast"]["models"]["extreme_quantile"]
    rows = []
    for order, (name, factory) in enumerate(candidates.items()):
        for r in run_lstm_folds(frame, cols, history_days, factory, cfg, availability_cols):
            rows.append({"candidate": name, "order": order, "fold": r["fold"], "n_fit": r["n_fit"],
                         **evaluate(_score_df(frame, r["issue"]), r["pred"], threshold, q)})
    per_fold = pd.DataFrame(rows)
    agg = {c: "mean" for c in ("mae", "rmse", "skill_mae", "precision", "recall", "f1",
                               "mae_extreme", "bias_extreme", "skill_mae_extreme")}
    agg.update({"missed": "sum", "events": "sum", "order": "first"})
    summary = per_fold.groupby("candidate", sort=False).agg(agg).rename(columns={"mae": "mean_mae"})
    summary = summary.assign(key=summary["mean_mae"].round(2)).sort_values(["key", "order"])
    summary["chosen"] = False
    summary.iloc[0, summary.columns.get_loc("chosen")] = True
    return per_fold.drop(columns="order"), summary.drop(columns=["key", "order"])


def predictions_lstm(frame, cols, history_days, factory, cfg, availability_cols=None) -> pd.DataFrame:
    """target_date, fold, y, pred for every fold's score windows."""
    return pd.concat([pd.DataFrame({"target_date": r["issue"] + pd.Timedelta(days=1), "fold": r["fold"],
                                    "y": r["y_score"], "pred": r["pred"]})
                      for r in run_lstm_folds(frame, cols, history_days, factory, cfg, availability_cols)],
                     ignore_index=True)


def compare_common_days(frame: pd.DataFrame, preds: dict, cfg: dict, threshold: float):
    """Persistence and each model in `preds` (name -> target_date, fold, pred) on exactly the target days
    every model has, per fold. Returns (per-fold table, mean over folds; missed and events summed)."""
    q = cfg["forecast"]["models"]["extreme_quantile"]
    by_target = frame.set_index("target_date")
    rows = []
    for fold in sorted(set.intersection(*[set(p["fold"]) for p in preds.values()])):
        days = sorted(set.intersection(*[set(p.loc[p["fold"] == fold, "target_date"]) for p in preds.values()]))
        sdf = by_target.loc[days, ["y", "pm_lag0"]].reset_index(drop=True)
        series = {"persistence": sdf["pm_lag0"].to_numpy(float)}
        series.update({n: p[p["fold"] == fold].set_index("target_date").loc[days, "pred"].to_numpy(float)
                       for n, p in preds.items()})
        for name, pred in series.items():
            rows.append({"model": name, "fold": fold, **evaluate(sdf, pred, threshold, q)})
    per_fold = pd.DataFrame(rows)
    agg = {c: "mean" for c in ("mae", "rmse", "skill_mae", "precision", "recall", "f1",
                               "mae_extreme", "bias_extreme", "skill_mae_extreme")}
    agg.update({"n": "sum", "missed": "sum", "events": "sum"})
    return per_fold, per_fold.groupby("model", sort=False).agg(agg)


def fit_final_and_predict_lstm(frame, cols, history_days, factory, cfg) -> pd.DataFrame:
    """Fit on windows with core.start <= target_date <= validation end and predict the test split.
    Runs ONCE, on the single test run (notebook 09), after the tag pre-test-v2 (CLAUDE.md hard rule 1). Test-split targets are never read: they are replaced by
    a placeholder only so make_sequences keeps those windows. Returns target_date, smog, pred."""
    p = cfg["forecast"]["periods"]
    f = frame.copy()
    test = (f["split"] == "test").to_numpy()
    f.loc[test, "y"] = 0.0
    X, y, d = _windows(f, cols, history_days)
    target = d + pd.Timedelta(days=1)
    fit = (target >= pd.Timestamp(p["core"]["start"])) & (target <= pd.Timestamp(p["split"]["val"]["end"]))
    te = target >= pd.Timestamp(p["split"]["test"]["start"])
    model = factory().fit(X[np.asarray(fit)], y[np.asarray(fit)])
    return pd.DataFrame({"target_date": target[te], "smog": _smog(f, d[te]), "pred": model.predict(X[np.asarray(te)])})
