"""Audit for the NETWORK part (agaar.gov.mn daily, Ulaanbaatar, winter 2025/26).

    python -m uaq.audit_network

Notebook 02 calls the same functions step by step. Writes to outputs/audit_network/:
    audit_report.md, station_coverage.csv, wide_daily_eligible.csv, threshold_base_rates.csv,
    coverage_heatmap.png, station_map.png, correlation.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import agaar, thresholds
from .config import ROOT, load_config
from .periods import check_split_order


def split_masks(wide: pd.DataFrame, cfg: dict) -> dict[str, np.ndarray]:
    sp = cfg["network"]["split"]
    return {s: np.asarray((wide.index >= sp[s]["start"]) & (wide.index <= sp[s]["end"])) for s in ("train", "val", "test")}


def prepare(cfg: dict, root: Path = ROOT) -> dict:
    """Everything the GA needs: eligible-station matrix, coordinates, masks, threshold."""
    check_split_order(cfg["network"]["split"])
    daily = agaar.load_daily(cfg, root)
    stations = agaar.load_stations(cfg, root)
    wide_all = agaar.to_wide(daily, cfg)
    cov = agaar.coverage(wide_all, cfg).reindex(stations["station_code"]).fillna(0.0)
    elig = agaar.eligible(cov, cfg["quality"]["min_valid_frac"])
    wide = wide_all[elig]
    masks = split_masks(wide, cfg)
    rates = thresholds.base_rates(wide[masks["train"]].stack(), cfg)
    name, thr, why = thresholds.choose(rates, cfg)
    st = stations.set_index("station_code")
    return {"daily": daily, "stations": stations, "coverage": cov, "eligible": elig, "wide": wide,
            "masks": masks, "rates": rates, "threshold_name": name, "threshold": thr, "threshold_why": why,
            "lat": st.loc[elig, "lat"].to_numpy(float), "lon": st.loc[elig, "lon"].to_numpy(float)}


def plot_coverage(cov: pd.DataFrame, names: pd.Series, thr: float, ax=None):
    ax = ax or plt.subplots(figsize=(6, 0.8 + 0.3 * len(cov)))[1]
    ax.imshow(cov.values, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(range(cov.shape[1]), cov.columns)
    ax.set_yticks(range(len(cov)), [f"{c} {names.get(c, '')}" for c in cov.index], fontsize=8)
    for i in range(cov.shape[0]):
        for j in range(cov.shape[1]):
            v = cov.values[i, j]
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=7, color="white" if v < thr else "black")
    ax.set_title("Valid days per split")
    ax.figure.tight_layout()
    return ax


def plot_map(stations: pd.DataFrame, eligible: list[str], ax=None):
    ax = ax or plt.subplots(figsize=(6, 5))[1]
    ok = stations["station_code"].isin(eligible)
    ax.scatter(stations.loc[~ok, "lon"], stations.loc[~ok, "lat"], c="lightgrey", label="not eligible")
    ax.scatter(stations.loc[ok, "lon"], stations.loc[ok, "lat"], c="#4C72B0", label="eligible")
    for _, r in stations.iterrows():
        ax.annotate(r["station_code"], (r["lon"], r["lat"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude"); ax.legend(fontsize=8)
    ax.set_title("Ulaanbaatar monitoring stations")
    ax.figure.tight_layout()
    return ax


def plot_correlation(wide: pd.DataFrame, mask: np.ndarray, ax=None):
    c = np.log1p(wide[mask]).corr()
    ax = ax or plt.subplots(figsize=(6, 5))[1]
    im = ax.imshow(c.values, vmin=0, vmax=1, cmap="magma")
    ax.set_xticks(range(len(c)), c.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(c)), c.index, fontsize=7)
    ax.set_title("Station correlation, training period (log PM2.5)")
    ax.figure.colorbar(im, ax=ax, fraction=0.04)
    ax.figure.tight_layout()
    return ax


def run(config_path: str | None = None, root: Path = ROOT) -> Path:
    cfg = load_config(config_path)
    out = root / "outputs" / "audit_network"
    out.mkdir(parents=True, exist_ok=True)
    r = prepare(cfg, root)
    names = r["stations"].set_index("station_code")["name_mn"]
    r["coverage"].to_csv(out / "station_coverage.csv")
    r["wide"].to_csv(out / "wide_daily_eligible.csv")
    r["rates"].to_csv(out / "threshold_base_rates.csv", index=False)
    plot_coverage(r["coverage"], names, cfg["quality"]["min_valid_frac"]).figure.savefig(out / "coverage_heatmap.png", dpi=150)
    plot_map(r["stations"], r["eligible"]).figure.savefig(out / "station_map.png", dpi=150)
    plot_correlation(r["wide"], r["masks"]["train"]).figure.savefig(out / "correlation.png", dpi=150)
    plt.close("all")

    sp, n = cfg["network"]["split"], len(r["eligible"])
    dropped = r["daily"]["dropped_reason"].replace("", "kept").value_counts().to_dict()
    lines = [
        "# Network data audit (agaar.gov.mn, Ulaanbaatar)", "",
        f"- Daily files read: {r['daily'].attrs.get('files')}",
        f"- Ulaanbaatar stations in catalogue: {len(r['stations'])}",
        f"- Station-days by cleaning outcome: {dropped}",
        "", "## Decisions", "",
        f"- **Eligible stations:** {n} (≥{cfg['quality']['min_valid_frac']:.0%} valid days in train, val AND test) → "
        f"{', '.join(r['eligible'])}",
        f"- **Exhaustive search feasible for every k:** {'yes' if max(__import__('math').comb(n, k) for k in range(1, n)) <= cfg['network']['exhaustive_max_subsets'] else 'no'}",
        f"- **Event threshold:** {r['threshold_name']} = {r['threshold']} µg/m³. {r['threshold_why']}",
        f"- **Splits:** train {sp['train']['start']}→{sp['train']['end']}, val {sp['val']['start']}→"
        f"{sp['val']['end']}, test {sp['test']['start']}→{sp['test']['end']} (sealed).",
        "", "## Coverage by split", "", r["coverage"].assign(name=names).to_markdown(floatfmt=".2f"),
        "", "## Threshold base rates (training station-days)", "", r["rates"].to_markdown(index=False, floatfmt=".3f"),
    ]
    report = out / "audit_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


if __name__ == "__main__":
    matplotlib.use("Agg")      # command line only; notebooks keep their inline backend
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    print(run(ap.parse_args().config))
