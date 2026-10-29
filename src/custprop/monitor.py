"""Checks for a model that scores new months.

- drift: how far the inputs and the scores have moved from the month the model was tested
  on, by the population stability index (PSI)
- expected_vs_actual: once a month's label window has passed, how many customers the
  calibrated scores expected to buy against how many did

PSI cuts the reference values into ten equal-count bins and compares the share of each
month in every bin: sum of (current - reference) * ln(current / reference). A common rule
of thumb reads below 0.1 as little change, 0.1 to 0.25 as worth a look and above 0.25 as
a real shift. Features with many equal values (counts of zero) get fewer bins.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def psi(reference, current, bins: int = 10, floor: float = 1e-4) -> float:
    ref, cur = np.asarray(reference, dtype=float), np.asarray(current, dtype=float)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)[1:-1]))

    def share(v):
        return np.bincount(np.searchsorted(edges, v, side="right"), minlength=len(edges) + 1) / len(v)

    r, c = np.clip(share(ref), floor, None), np.clip(share(cur), floor, None)  # empty bins would give log(0)
    return float(np.sum((c - r) * np.log(c / r)))


def level(value: float) -> str:
    return "stable" if value < 0.1 else "check" if value < 0.25 else "shift"


def drift(reference: pd.DataFrame, current: pd.DataFrame, columns) -> pd.DataFrame:
    """PSI per column, largest first, with the medians so a shift can be read in its own units."""
    rows = []
    for col in columns:
        v = psi(reference[col], current[col])
        rows.append({"column": col, "psi": v, "level": level(v),
                     "reference_median": float(np.median(reference[col])), "current_median": float(np.median(current[col]))})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).round(4).reset_index(drop=True)


def expected_vs_actual(p, y) -> dict:
    p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=float)
    return {"customers": len(p), "expected_buyers": float(p.sum()), "actual_buyers": float(y.sum()),
            "ratio": float(y.sum() / p.sum()) if p.sum() > 0 else np.nan}
