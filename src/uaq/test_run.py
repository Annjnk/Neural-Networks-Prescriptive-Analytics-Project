"""The single test evaluation (notebook 09). Runs ONCE, after the tag pre-test-v2.

This is the only module that reads test-split PM2.5 (and the network test months). It only CALLS the frozen
functions of uaq.models, uaq.sequence, uaq.warnings, uaq.network and uaq.audit_network with the choices recorded
before the test (configs/study_config.yaml, outputs/results/forecast_selected.json, warning_policy.json,
network_validation.csv). Nothing chosen earlier is changed. Evaluation constants below were fixed before the run.

Preconditions (check_preconditions): the working tree is clean; no frozen file changed since the tag pre-test-v2
(configs/ and earlier outputs/results/ files byte for byte; any changed .py file under src/uaq/ other than
test_run.py must have the same code, compared as ASTs without docstrings); and outputs/results/test_run_log.json does
not exist (no re-runs; the only allowed re-run fixes a bug in EVALUATION code and must be logged with its reason).
"""
from __future__ import annotations

import ast
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from . import audit_network, models, network, sequence
from . import warnings as wp
from .config import ROOT
from .features import complete_rows, feature_columns

TAG = "pre-test-v2"
RESULTS = Path("outputs") / "results"
LOG = RESULTS / "test_run_log.json"
FROZEN_FILES = ("configs/study_config.yaml", "outputs/results/forecast_selected.json",
                "outputs/results/warning_policy.json")
INPUT_FILES = ("data/processed/features_8160.csv",)       # git-ignored inputs: hashed in the log for provenance
NEW_OUTPUTS = (":(exclude,glob)outputs/results/test_*", ":(exclude)outputs/results/network_test.csv")
OUTPUT_FILES = ("test_forecast.csv", "test_warning.csv", "test_warning.json", "network_test.csv",
                "test_summary.json", "test_run_log.json")
BOOT_N, BOOT_BLOCK_DAYS, BOOT_SEED = 1000, 7, 0          # paired block bootstrap of MAE differences
FORECAST_MODELS = ("persistence", "climatology", "ridge (weather)", "ridge (no weather)", "hgb (weather)",
                   "hgb (no weather)", "mlp (weather)", "mlp (no weather)", "lstm (weather)", "lstm (no weather)")


# ---------------------------------------------------------------------- preconditions and log
def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True).stdout.strip()


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _code_without_docstrings(source: str) -> str:
    """AST dump with module, class and function docstrings removed (comments are not in the AST either)."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                node.body = node.body[1:] or [ast.Pass()]
    return ast.dump(tree)


def check_preconditions(root: Path = ROOT, tag: str = TAG, rerun_reason: str | None = None) -> dict:
    """Raise RuntimeError unless the tree is clean, no frozen file changed since `tag`, and no test-run log exists
    (or `rerun_reason` is given for a logged EVALUATION-code fix). Returns both commit hashes and file hashes."""
    root = Path(root)
    if _git(root, "status", "--porcelain"):
        raise RuntimeError("the working tree is not clean: commit or remove changes before the test run")
    try:
        tagged = _git(root, "rev-parse", f"{tag}^{{commit}}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"tag {tag!r} not found") from e
    head = _git(root, "rev-parse", "HEAD")
    frozen = _git(root, "diff", "--name-only", tag, "HEAD", "--", "configs/", "outputs/results/", *NEW_OUTPUTS)
    if frozen:
        raise RuntimeError(f"frozen files changed since {tag}: {frozen.split()}")
    doc_only = []
    for f in _git(root, "diff", "--name-only", tag, "HEAD", "--", "src/uaq/", ":(exclude)src/uaq/test_run.py").split():
        try:
            old, new = _git(root, "show", f"{tag}:{f}"), _git(root, "show", f"HEAD:{f}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"{f} was added or removed since {tag}") from e
        if not f.endswith(".py") or _code_without_docstrings(old) != _code_without_docstrings(new):
            raise RuntimeError(f"code changed since {tag}: {f}")
        doc_only.append(f)
    if (root / LOG).exists() and not rerun_reason:
        raise RuntimeError(f"{LOG} exists: the test run has already happened, no re-runs "
                           "(only an EVALUATION-code bug fix may re-run, with rerun_reason logged)")
    return {"tag": tag, "tag_commit": tagged, "head_commit": head, "docstring_only_changes": doc_only,
            "hashes": {f: sha256(root / f) for f in FROZEN_FILES},
            "input_hashes": {f: sha256(root / f) for f in INPUT_FILES if (root / f).exists()}}


def write_log(root: Path, info: dict, rerun_reason: str | None = None) -> dict:
    """Create the test-run log (or append a logged re-run to it) BEFORE any test data is read."""
    path = Path(root) / LOG
    entry = {**info, "run_time": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")}
    if path.exists():
        if not rerun_reason:
            raise RuntimeError(f"{LOG} exists and no rerun_reason was given")
        log = json.loads(path.read_text(encoding="utf-8"))
        log.setdefault("reruns", []).append({**entry, "reason": rerun_reason})
    else:
        log = entry
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(log, indent=2), encoding="utf-8")
    return log


def save_outputs(root: Path, outputs: dict) -> list[Path]:
    """Write test results; only names in OUTPUT_FILES (never configs/ or an earlier results file), never overwriting."""
    written = []
    for name, obj in outputs.items():
        if name not in OUTPUT_FILES or name == "test_run_log.json":
            raise ValueError(f"{name!r} is not a test-run output file")
        path = Path(root) / RESULTS / name
        if path.exists():
            raise FileExistsError(f"{path} exists: test outputs are written once")
        if name.endswith(".csv"):
            obj.to_csv(path, index=False)
        else:
            path.write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")
        written.append(path)
    return written


# ---------------------------------------------------------------------- forecasts
def test_days(feat: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Test-split smog-season target days with an observed y: target_date index, y, pm_lag0."""
    f = feat[(feat["split"] == "test") & feat["smog"].astype(bool) & feat["y"].notna()]
    return f.set_index("target_date")[["y", "pm_lag0"]].sort_index()


def _factories(feat: pd.DataFrame, cfg: dict, selected: dict) -> dict:
    drop = cfg["features"]["drop_columns"]
    cw, cnw = feature_columns(feat, weather=True, drop=drop), feature_columns(feat, weather=False, drop=drop)
    ch = selected["chosen"]
    cands = {"ridge": models.ridge_candidates(cfg), "hgb": models.hgb_candidates(cfg), "mlp": models.mlp_candidates(cfg)}
    out = {"persistence": ("rows", cw, models.Persistence), "climatology": ("rows", cw, models.Climatology)}
    for fam in ("ridge", "hgb", "mlp"):
        for fs, cols in (("weather", cw), ("no weather", cnw)):
            name = f"{fam} ({fs})"
            out[name] = ("rows", cols, cands[fam][ch[name]["config"]])
    for fs, w in (("weather", True), ("no weather", False)):
        cols = sequence.sequence_columns(feat, cfg, weather=w)
        name = f"lstm ({fs})"
        out[name] = ("seq", cols, sequence.lstm_candidates(cfg, cols)[selected["lstm"]["chosen"][name]["config"]])
    out["_cw"] = cw
    return out


def forecasts(feat: pd.DataFrame, cfg: dict, selected: dict) -> pd.DataFrame:
    """Refit every frozen model on core.start..validation end (as notebooks 05/06) and predict the test days.
    Row models use the notebook 05 ablation rows (complete for the with-weather features) for both feature sets."""
    days = test_days(feat, cfg)
    facs = _factories(feat, cfg, selected)
    ablation = complete_rows(feat, facs.pop("_cw"))
    table = days.copy()
    for name in FORECAST_MODELS:
        kind, cols, fac = facs[name]
        out = (models.fit_final_and_predict(ablation, cols, fac, cfg) if kind == "rows" else
               sequence.fit_final_and_predict_lstm(feat, cols, selected["lstm"]["history_days"], fac, cfg))
        table[f"pred_{name}"] = out.set_index("target_date")["pred"].reindex(table.index)
    return table


def score_forecasts(table: pd.DataFrame, cfg: dict, threshold: float) -> pd.DataFrame:
    """uaq.models.evaluate per model on the days it forecasts (skill vs persistence on those same days)."""
    q = cfg["forecast"]["models"]["extreme_quantile"]
    rows = []
    for c in [c for c in table.columns if c.startswith("pred_")]:
        d = table[table[c].notna() & table["pm_lag0"].notna()]
        rows.append({"model": c[5:], **models.evaluate(d[["y", "pm_lag0"]], d[c], threshold, q)})
    return pd.DataFrame(rows).set_index("model")


def common_days(table: pd.DataFrame, feat: pd.DataFrame, cfg: dict, threshold: float):
    """Persistence vs MLP vs LSTM on days with an LSTM window (uaq.sequence.compare_common_days, as notebook 06)."""
    preds = {n: pd.DataFrame({"target_date": table.index, "fold": "test", "pred": table[f"pred_{n}"].values}).dropna()
             for n in ("mlp (weather)", "lstm (weather)")}
    return sequence.compare_common_days(feat, preds, cfg, threshold)


def bootstrap_mae_difference(table: pd.DataFrame, model: str, n: int = BOOT_N, block_days: int = BOOT_BLOCK_DAYS,
                             seed: int = BOOT_SEED) -> dict:
    """Paired block bootstrap of MAE(model) - MAE(persistence) on the days the model forecasts."""
    d = table[table[f"pred_{model}"].notna() & table["pm_lag0"].notna()]
    e_m = np.abs(d[f"pred_{model}"].to_numpy(float) - d["y"].to_numpy(float))
    e_p = np.abs(d["pm_lag0"].to_numpy(float) - d["y"].to_numpy(float))
    block = ((d.index - d.index.min()).days // block_days).to_numpy()
    groups = [np.flatnonzero(block == b) for b in np.unique(block)]
    rng = np.random.default_rng(seed)
    reps = np.empty(n)
    for i in range(n):
        idx = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
        reps[i] = e_m[idx].mean() - e_p[idx].mean()
    return {"model": model, "n_days": len(d), "mae_difference": float(e_m.mean() - e_p.mean()),
            "lo": float(np.quantile(reps, 0.025)), "hi": float(np.quantile(reps, 0.975)),
            "resamples": n, "block_days": block_days, "seed": seed}


def weather_ablation(table: pd.DataFrame) -> pd.DataFrame:
    """MAE with vs without weather per family, on the days both variants forecast."""
    rows = []
    for fam in ("ridge", "hgb", "mlp", "lstm"):
        w, nw = table[f"pred_{fam} (weather)"], table[f"pred_{fam} (no weather)"]
        both = w.notna() & nw.notna()
        y = table.loc[both, "y"]
        rows.append({"family": fam, "n_days": int(both.sum()), "mae_weather": float((w[both] - y).abs().mean()),
                     "mae_no_weather": float((nw[both] - y).abs().mean())})
    out = pd.DataFrame(rows).set_index("family")
    out["difference"] = out["mae_weather"] - out["mae_no_weather"]
    return out


def verdict_h1(scores: pd.DataFrame, boot: dict) -> dict:
    skill = float(scores.loc["mlp (weather)", "skill_mae"])
    ok = skill > 0 and boot["hi"] < 0
    return {"verdict": "SUPPORTED" if ok else "NOT SUPPORTED", "mlp_skill_mae": skill,
            "mae_difference": boot["mae_difference"], "interval": [boot["lo"], boot["hi"]]}


def verdict_h2(ablation: pd.DataFrame) -> dict:
    d = float(ablation.loc["mlp", "difference"])
    return {"verdict": "SUPPORTED" if d < 0 else "NOT SUPPORTED", "mlp_mae_difference_weather_minus_none": d,
            "all_families": ablation["difference"].to_dict()}


# ---------------------------------------------------------------------- warning policy
def warning_results(table: pd.DataFrame, feat: pd.DataFrame, cfg: dict, policy: dict) -> dict:
    """Frozen policy and the three single-threshold rules on the test winter (uaq.warnings)."""
    pol = {k: policy[k] for k in ("input", "t1", "t2")}
    vh = policy["very_high"]
    preds = table[[c for c in table.columns if c.startswith("pred_")]]
    days, card = wp.evaluate_on_test(pol, preds, feat, cfg, vh)
    rules = {"policy": (days["level"], card)}
    for name, thr in (("single threshold, matched budget", policy["matched_single_threshold"]),
                      ("single threshold", cfg["warnings"]["high"]),
                      ("official line", cfg["thresholds"]["candidates"]["mns_4585_2025"])):
        rules[name] = wp.single_threshold_rule(days["forecast"], days["y"], cfg, vh, threshold=thr)
    cards = {}
    for name, (lev, c) in rules.items():
        any_level = (lev.reindex(days.index).fillna(0) >= 1).astype(int)
        cards[name] = {**c, "longest_alert_run": wp._longest_run(any_level, k=1)}
        days[f"level_{name}"] = lev.reindex(days.index).fillna(0).astype(int)
    h5 = cards["policy"]["caught_per_alert"] > cards["single threshold, matched budget"]["caught_per_alert"]
    # Pre-declared secondary outcome: descriptive, no verdict.
    fatigue = {name: {"longest_warning_run": cards[name]["longest_l2_run"], "warnings": cards[name]["alerts_l2"],
                      **{f"warnings {k[3:]}": v for k, v in cards[name].items() if k.startswith("l2_")}}
               for name in ("policy", "single threshold, matched budget")}
    return {"days": days.drop(columns="level"), "cards": cards, "fatigue": fatigue,
            "h5": {"verdict": "SUPPORTED" if h5 else "NOT SUPPORTED",
                   "policy_caught_per_alert": cards["policy"]["caught_per_alert"],
                   "matched_caught_per_alert": cards["single threshold, matched budget"]["caught_per_alert"]}}


# ---------------------------------------------------------------------- network design
def network_results(cfg: dict, root: Path = ROOT, eval_split: str = "test") -> dict:
    """Score the subsets each method chose on validation (network_validation.csv) on the test months, exactly as
    notebook 03's sealed cell; no selection is re-run. 'random (median)' replays notebook 03's random draws
    (same seeds) and takes the median F on the test months."""
    r = audit_network.prepare(cfg, root)
    m, fit = r["masks"], cfg["network"]["fitness"]
    val = pd.read_csv(Path(root) / RESULTS / "network_validation.csv")
    val["stations"] = val["stations"].map(ast.literal_eval)
    fit_mask = m["train"] | m["val"] if eval_split == "test" else m["train"]
    prob = network.Problem(r["wide"], fit_mask, m[eval_split], r["threshold"], **fit)
    idx = {c: i for i, c in enumerate(prob.stations)}
    rows = []
    for _, row in val.iterrows():
        S = tuple(idx[c] for c in row["stations"])
        rows.append({"k": int(row["k"]), "method": row["method"], "stations": row["stations"],
                     "F_validation": float(row["F"]), **{f"{a}_{eval_split}": b for a, b in prob.details(S).items()}})
    val_prob = network.Problem(r["wide"], m["train"], m["val"], r["threshold"], **fit)
    for k in sorted(val["k"].unique()):
        _, scores = network.random_search(prob, int(k), cfg["network"]["random_draws"], np.random.default_rng(cfg["seed"] + int(k)))
        best_val, _ = network.random_search(val_prob, int(k), cfg["network"]["random_draws"], np.random.default_rng(cfg["seed"] + int(k)))
        stored = val[(val["k"] == k) & val["method"].str.startswith("random")]["stations"].iloc[0]
        assert sorted(val_prob.names(best_val)) == sorted(stored), f"random draws do not replay notebook 03 at k={k}"
        rows.append({"k": int(k), "method": "random (median)", "stations": None, "F_validation": float(
            val[(val["k"] == k) & val["method"].str.startswith("random")]["random_median_F"].iloc[0]),
            f"F_{eval_split}": float(np.median(scores))})
    return {"table": pd.DataFrame(rows), "problem": prob, "threshold": r["threshold"]}


def verdict_h3(net: pd.DataFrame, col: str = "F_test") -> dict:
    """H3: for most k, the GA subset has lower F than random (median), highest-mean and spatial-spread, and F <=
    greedy. Returns per-k wins and the verdict."""
    t = net.pivot(index="k", columns="method", values=col)
    ga = t["genetic algorithm"]
    win = ((ga < t["random (median)"]) & (ga < t["highest mean"]) & (ga < t["spatial spread"])
           & (ga <= t["greedy forward"] + 1e-12))
    return {"verdict": "SUPPORTED" if win.sum() > len(win) / 2 else "NOT SUPPORTED",
            "k_won": [int(k) for k in win.index[win]], "k_total": int(len(win)),
            "per_k": {int(k): bool(v) for k, v in win.items()}}


def verdict_h4(net: pd.DataFrame, col: str = "F_test") -> dict:
    """H4 (validation): GA within 2% of the exhaustive optimum wherever exhaustive search ran; plus test F of both."""
    t = net.pivot(index="k", columns="method", values="F_validation")
    ex = t["exhaustive (optimum)"].dropna()
    within = (t.loc[ex.index, "genetic algorithm"] <= 1.02 * ex)
    tt = net.pivot(index="k", columns="method", values=col)
    return {"verdict": "SUPPORTED" if within.all() else "NOT SUPPORTED", "k_within_2pct": int(within.sum()),
            "k_total": int(len(ex)), "test_F_ga": {int(k): float(v) for k, v in tt["genetic algorithm"].items()},
            "test_F_validation_optimum": {int(k): float(v) for k, v in tt["exhaustive (optimum)"].items()}}
