"""Audit for the FORECASTING part (OpenAQ hourly, station 8160).

    python -m uaq.audit_forecast

Notebook 01 calls the same functions step by step. Writes to outputs/audit_forecast/:
    audit_report.md, coverage_station_season.csv, station_eligibility.csv,
    threshold_base_rates.csv, baseline_scores.csv, coverage_heatmap.png, smog_season_means.png
The test split is never scored here.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import baselines, metrics, thresholds
from .config import ROOT, load_config
from .daily import to_daily
from .io import load_hourly
from .periods import assign_split, check_split_order, in_months, season_label, season_year


def smog_days_per_season(cfg: dict) -> pd.Series:
    """Number of smog-season days in each season year of the study window."""
    p = cfg["forecast"]["periods"]
    lo, hi = pd.Timestamp(p["pre_ban_sensitivity"]["start"]), pd.Timestamp(p["core"]["end"])
    days = pd.DataFrame({"date": pd.date_range(lo, hi, freq="D")})
    days = days[in_months(days["date"], cfg["smog_months"]).values]
    days["season_year"] = season_year(days["date"]).values
    return days.groupby("season_year").size()


def coverage_table(daily: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Share of smog-season days with a valid daily mean, station x season year.
    The denominator is every smog-season day in that season (inside the study window),
    so a station that went offline scores low."""
    denom = smog_days_per_season(cfg)

    d = daily[in_months(daily["date"], cfg["smog_months"]).values & daily["valid"]].copy()
    d["season_year"] = season_year(d["date"]).values
    num = d.groupby(["station_id", "season_year"]).size().unstack(fill_value=0)
    return (num.reindex(columns=denom.index, fill_value=0) / denom).round(3)


def eligibility(cov: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    sp, thr = cfg["forecast"]["periods"]["split"], cfg["quality"]["min_valid_frac"]
    sy = lambda d: int(season_year(pd.Series([pd.Timestamp(d)]))[0])  # noqa: E731
    cols = lambda a, b: [c for c in cov.columns if a <= c <= b]  # noqa: E731
    tr, va, te = cols(sy(sp["train"]["start"]), sy(sp["train"]["end"])), sy(sp["val"]["start"]), \
        cols(sy(sp["test"]["start"]), sy(sp["test"]["end"]))
    out = pd.DataFrame(index=cov.index)
    out["good_train_winters"] = (cov[tr] >= thr).sum(axis=1)
    denom = smog_days_per_season(cfg)
    out["train_valid_smog_days"] = (cov[tr] * denom[tr]).sum(axis=1).round().astype(int)
    out["val_cov"] = cov[va] if va in cov.columns else 0.0
    out["test_cov_min"] = cov[te].min(axis=1) if te else 0.0
    out["eligible"] = ((out["train_valid_smog_days"] >= cfg["forecast"]["min_train_smog_days"])
                       & (out["val_cov"] >= thr) & (out["test_cov_min"] >= thr))
    return out


def threshold_table(daily: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    sp = cfg["forecast"]["periods"]["split"]
    tr = daily[(assign_split(daily["date"], sp) == "train").values
               & in_months(daily["date"], cfg["smog_months"]).values]
    return thresholds.base_rates(tr["pm25_daily"], cfg)


def baseline_scores(daily: pd.DataFrame, stations: list[str], thr: float, cfg: dict) -> pd.DataFrame:
    """Persistence / 3-day mean / climatology on train and val smog-season days."""
    p = cfg["forecast"]["periods"]
    core = daily[daily["station_id"].isin(stations)
                 & (daily["date"] >= pd.Timestamp(p["core"]["start"]))
                 & (daily["date"] <= pd.Timestamp(p["core"]["end"]))]
    if core.empty:
        return pd.DataFrame()
    f = baselines.make_targets(core)
    f["split"] = assign_split(f["target_date"], p["split"]).values
    f = baselines.add_all(f, f["split"] == "train")
    f = f[f["split"].isin(["train", "val"]) & in_months(f["target_date"], cfg["smog_months"]).values]
    return pd.concat([metrics.evaluate(f, m, thr, by="split") for m in ("persistence", "mean3d", "climatology")],
                     ignore_index=True)


def plot_coverage(cov: pd.DataFrame, thr: float, ax=None):
    ax = ax or plt.subplots(figsize=(1.0 + 0.7 * cov.shape[1], 0.9 + 0.35 * cov.shape[0]))[1]
    im = ax.imshow(cov.values, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(range(cov.shape[1]), [season_label(c) for c in cov.columns], rotation=45, ha="right")
    ax.set_yticks(range(cov.shape[0]), cov.index)
    for i in range(cov.shape[0]):
        for j in range(cov.shape[1]):
            v = cov.values[i, j]
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=7, color="white" if v < thr else "black")
    ax.set_title("Valid smog-season days (Nov–Feb) per station")
    ax.figure.colorbar(im, ax=ax, fraction=0.03, aspect=max(2, 10 * cov.shape[0]))
    ax.figure.tight_layout()
    return ax


def plot_season_means(daily: pd.DataFrame, cfg: dict, ax=None):
    d = daily[daily["valid"] & in_months(daily["date"], cfg["smog_months"]).values].copy()
    d["season_year"] = season_year(d["date"]).values
    city = d.groupby(["season_year", "date"])["pm25_daily"].median().groupby("season_year").mean()
    ax = ax or plt.subplots(figsize=(7, 3.5))[1]
    ax.bar([season_label(s) for s in city.index], city.values, color="#4C72B0")
    brk = int(season_year(pd.Series([pd.Timestamp(cfg["structural_break"])]))[0])
    if brk + 1 in city.index:
        ax.axvline(list(city.index).index(brk + 1) - 0.5, color="#C44E52", ls="--", label="raw-coal ban (May 2019)")
        ax.legend()
    ax.tick_params(axis="x", labelrotation=45)
    ax.set_ylabel("Mean daily PM2.5 (µg/m³)")
    ax.set_title("Smog-season PM2.5 by season")
    ax.figure.tight_layout()
    return ax


def run(config_path: str | None = None, root: Path = ROOT) -> Path:
    cfg = load_config(config_path)
    check_split_order(cfg["forecast"]["periods"]["split"])
    out = root / "outputs" / "audit_forecast"
    out.mkdir(parents=True, exist_ok=True)

    hourly = load_hourly(cfg, root)
    daily = to_daily(hourly, cfg["quality"]["min_hours_per_day"])
    daily.to_csv(out / "daily_all.csv", index=False)

    cov = coverage_table(daily, cfg); cov.to_csv(out / "coverage_station_season.csv")
    elig = eligibility(cov, cfg); elig.to_csv(out / "station_eligibility.csv")
    rates = threshold_table(daily, cfg); rates.to_csv(out / "threshold_base_rates.csv", index=False)
    chosen, thr, why = thresholds.choose(rates, cfg)
    stations = elig.index[elig["eligible"]].tolist()
    scores = baseline_scores(daily, stations, thr, cfg)
    scores.to_csv(out / "baseline_scores.csv", index=False)
    plot_coverage(cov, cfg["quality"]["min_valid_frac"]).figure.savefig(out / "coverage_heatmap.png", dpi=150)
    plot_season_means(daily, cfg).figure.savefig(out / "smog_season_means.png", dpi=150)
    plt.close("all")

    sp = cfg["forecast"]["periods"]["split"]
    lines = [
        "# Forecasting data audit", "",
        f"- Raw files read: {hourly.attrs.get('files')}",
        f"- Units seen: {hourly.attrs.get('units_seen')} (expected {cfg['forecast']['input']['expected_units']})",
        f"- Hourly rows: {len(hourly):,} across {hourly['station_id'].nunique()} station(s)",
        f"- Date range: {hourly['ts_local'].min()} → {hourly['ts_local'].max()}",
        f"- Duplicate station-hours merged: {hourly.attrs.get('duplicate_station_hours')}",
        f"- Unparseable timestamps dropped: {hourly.attrs.get('dropped_bad_timestamps')}",
        f"- Hourly values flagged implausible (>{cfg['quality']['plausible_max_hourly']}): "
        f"{int(hourly['flag_implausible'].sum())}",
        "", "## Decisions", "",
        f"- **Usable station(s):** {', '.join(stations) or 'NONE: check the coverage heatmap'}",
        f"- **Event threshold:** {chosen} = {thr} µg/m³. {why}",
        f"- **Splits:** train {sp['train']['start']}→{sp['train']['end']}, val {sp['val']['start']}→"
        f"{sp['val']['end']}, test {sp['test']['start']}→{sp['test']['end']} (sealed).",
        "", "## Station eligibility", "", elig.to_markdown(),
        "", "## Threshold base rates (training smog-season days)", "", rates.to_markdown(index=False, floatfmt=".3f"),
        "", "## Baselines, smog season (validation is the bar the neural nets must clear)", "",
        scores.to_markdown(index=False, floatfmt=".3f") if len(scores) else "_no usable station_",
    ]
    report = out / "audit_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


if __name__ == "__main__":
    matplotlib.use("Agg")      # command line only; notebooks keep their inline backend
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    print(run(ap.parse_args().config))
