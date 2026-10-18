"""Purchase propensity models, evaluated on later months than they were trained on.

Snapshots are split by cutoff: the earliest months train, the next month(s) validate
(model choice and calibration), the latest month(s) test. A random split would let the
model see the same customer's later behaviour while training, which flatters it.

Candidates, from simplest up:
- recent_orders: no model, rank customers by how many orders they placed in the last 90 days
- logistic: logistic regression on log-scaled features
- gbm: gradient-boosted trees on all features
- gbm_segment: the same, with the customer's segment as an extra feature
- gbm_per_segment: one model per segment

A propensity score is used both to rank customers (who to contact first) and as a
probability (expected uptake), so both are measured: ranking by ROC AUC, average
precision and lift in the top tenth; probabilities by Brier score and expected calibration
error, before and after isotonic calibration on the validation month.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .snapshots import FEATURE_COLUMNS


def split_by_cutoff(df: pd.DataFrame, test_cutoffs: int = 1, valid_cutoffs: int = 1):
    cutoffs = sorted(df["cutoff"].unique())
    if len(cutoffs) < test_cutoffs + valid_cutoffs + 1:
        raise ValueError(f"need more than {test_cutoffs + valid_cutoffs} cutoffs, got {len(cutoffs)}")
    test = cutoffs[-test_cutoffs:]
    valid = cutoffs[-(test_cutoffs + valid_cutoffs):-test_cutoffs]
    return (df[~df["cutoff"].isin(test + valid)], df[df["cutoff"].isin(valid)], df[df["cutoff"].isin(test)])


# ---------------------------------------------------------------- metrics

def lift_at(y: np.ndarray, p: np.ndarray, top: float = 0.1) -> float:
    """Purchase rate among the top-scored share of customers, relative to the overall rate."""
    n = max(1, int(round(len(p) * top)))
    order = np.argsort(-p, kind="stable")[:n]
    base = y.mean()
    return float(y[order].mean() / base) if base > 0 else np.nan


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Average gap between predicted and observed rate over equal-count bins."""
    order = np.argsort(p, kind="stable")
    gaps = [abs(p[b].mean() - y[b].mean()) * len(b) for b in np.array_split(order, bins) if len(b)]
    return float(sum(gaps) / len(p))


def metrics(y, p) -> dict:
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    return {
        "roc_auc": roc_auc_score(y, p),
        "avg_precision": average_precision_score(y, p),
        "lift_top10": lift_at(y, p, 0.1),
        "brier": brier_score_loss(y, np.clip(p, 0, 1)),
        "ece": expected_calibration_error(y, p),
        "base_rate": y.mean(),
    }


def calibration_table(y, p, bins: int = 10) -> pd.DataFrame:
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    order = np.argsort(p, kind="stable")
    return pd.DataFrame(
        [{"predicted": p[b].mean(), "observed": y[b].mean(), "n": len(b)} for b in np.array_split(order, bins)]
    )


# ---------------------------------------------------------------- models

def _log1p_positive(x):
    return np.log1p(np.clip(x, 0, None))


def _logistic() -> Pipeline:
    return Pipeline([
        ("log", FunctionTransformer(_log1p_positive)),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, C=1.0)),
    ])


def _gbm(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                          l2_regularization=1.0, early_stopping=True,
                                          validation_fraction=0.15, random_state=seed)


class PerSegment:
    """One model per segment; segments too small or with a single class fall back to the global model."""

    def __init__(self, base, min_rows: int = 200):
        self.base, self.min_rows = base, min_rows

    def fit(self, x: pd.DataFrame, y: np.ndarray, segments: np.ndarray):
        self.fallback = clone(self.base).fit(x, y)
        self.models = {}
        for s in np.unique(segments):
            m = segments == s
            if m.sum() >= self.min_rows and len(np.unique(y[m])) == 2:
                self.models[s] = clone(self.base).fit(x[m], y[m])
        return self

    def predict_proba(self, x: pd.DataFrame, segments: np.ndarray) -> np.ndarray:
        p = self.fallback.predict_proba(x)[:, 1]
        for s, model in self.models.items():
            m = segments == s
            if m.any():
                p[m] = model.predict_proba(x[m])[:, 1]
        return p


@dataclass
class Result:
    name: str
    target: str
    test: dict
    test_calibrated: dict
    valid: dict
    model: object = None
    calibrator: IsotonicRegression | None = None
    test_scores: np.ndarray | None = field(default=None, repr=False)
    run_id: str | None = None  # set when the result is logged to MLflow


def _features(df: pd.DataFrame) -> pd.DataFrame:
    return df[FEATURE_COLUMNS].astype(float)


def compare_models(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame, target: str,
                   seed: int = 7, use_segments: bool = True) -> list[Result]:
    y_col = f"y_{target}"
    ytr, yva, yte = (d[y_col].to_numpy() for d in (train, valid, test))
    results: list[Result] = []

    def add(name, model, p_valid, p_test):
        iso = IsotonicRegression(out_of_bounds="clip").fit(p_valid, yva)
        results.append(Result(name, target, metrics(yte, p_test), metrics(yte, iso.predict(p_test)),
                              metrics(yva, p_valid), model, iso, p_test))

    # no model: more recent orders, more likely to order again
    def score(d):
        return d["orders_90d"].to_numpy() + d["orders_365d"].to_numpy() / 100  # longer history breaks ties

    add("recent_orders", None, score(valid), score(test))

    for name, est in (("logistic", _logistic()), ("gbm", _gbm(seed))):
        m = est.fit(_features(train), ytr)
        add(name, m, m.predict_proba(_features(valid))[:, 1], m.predict_proba(_features(test))[:, 1])

    if use_segments and "segment" in train:
        # one-hot over the segments of all three splits, so the columns line up
        segs = sorted(set(train["segment"]) | set(valid["segment"]) | set(test["segment"]))

        def seg_x(d):
            return _features(d).assign(**{f"segment_{s}": (d["segment"] == s).astype(float) for s in segs})
        m = _gbm(seed).fit(seg_x(train), ytr)
        add("gbm_segment", m, m.predict_proba(seg_x(valid))[:, 1], m.predict_proba(seg_x(test))[:, 1])

        per = PerSegment(_gbm(seed)).fit(_features(train), ytr, train["segment"].to_numpy())
        add("gbm_per_segment", per, per.predict_proba(_features(valid), valid["segment"].to_numpy()),
            per.predict_proba(_features(test), test["segment"].to_numpy()))
    return results


def results_table(results: list[Result]) -> pd.DataFrame:
    rows = []
    for r in results:
        row = {"target": r.target, "model": r.name}
        row.update({f"test_{k}": v for k, v in r.test.items()})
        row.update({"test_brier_calibrated": r.test_calibrated["brier"], "test_ece_calibrated": r.test_calibrated["ece"],
                    "valid_roc_auc": r.valid["roc_auc"]})
        rows.append(row)
    return pd.DataFrame(rows).round(4)
