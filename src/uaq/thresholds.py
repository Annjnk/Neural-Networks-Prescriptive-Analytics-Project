"""Event-threshold choice, shared by both parts. Uses TRAINING data only."""
from __future__ import annotations

import numpy as np
import pandas as pd


def base_rates(train_values: pd.Series, cfg: dict) -> pd.DataFrame:
    """Exceedance rate of each candidate threshold on training daily means."""
    v = pd.Series(train_values, dtype=float).dropna()
    t = cfg["thresholds"]
    cands = dict(t["candidates"])
    q = t["fallback_percentile"]
    cands[f"train_p{q}"] = float(np.percentile(v, q)) if len(v) else np.nan
    return pd.DataFrame([{"threshold": k, "ug_m3": round(val, 1), "base_rate": float((v > val).mean())}
                         for k, val in cands.items()])


def choose(rates: pd.DataFrame, cfg: dict) -> tuple[str, float, str]:
    """Lowest official threshold exceeded on <= max_base_rate of training days,
    otherwise the study-defined training percentile."""
    t = cfg["thresholds"]
    r = rates.set_index("threshold")
    tried = []
    for name in sorted(t["candidates"], key=lambda k: t["candidates"][k]):
        rate = r.loc[name, "base_rate"]
        tried.append(f"{name} {rate:.0%}")
        if rate <= t["max_base_rate"]:
            return name, float(r.loc[name, "ug_m3"]), (
                f"Lowest official threshold exceeded on ≤ {t['max_base_rate']:.0%} of training days "
                f"(rates: {', '.join(tried)}).")
    pname = next(k for k in r.index if k.startswith("train_p"))
    return pname, float(r.loc[pname, "ug_m3"]), (
        f"Every official threshold is exceeded too often in training ({', '.join(tried)}), so a "
        f"study-defined training percentile is used. Report the official rates as context.")
