"""Purchase propensity models, evaluated on later months than they were trained on.

Snapshots are split by cutoff: the earliest months train, the next month(s) validate
(model choice and calibration), the latest month(s) test. A random split would let the
model see the same customer's later behaviour while training, which flatters it.

Candidates, from simplest up:
- recent_orders: no model, rank customers by how many orders they placed in the last 90 days
- logistic, gbm: logistic regression and gradient-boosted trees on the first version's
  features (BASE_COLUMNS), kept so the effect of the new features is visible
- logistic_ext, gbm_ext: the same models on the extended features (EXTENDED_COLUMNS)
- logistic_segment, gbm_segment: base features plus the customer's segment, to see whether
  segments help a linear model that can't find groups by itself, and the trees that can
- gbm_ext_segment: extended trees with the customer's segment as an extra feature
- gbm_ext_per_segment: one extended model per segment

More views of the same data:
- metrics_by_segment: how well a model ranks customers within each segment
- metrics_on_subset: every model on a subset of the test month (customers with one order)
- rolling_evaluation: the comparison repeated with each of the last few months as the
  test month, so one unusual month doesn't decide the result
- learning_curve: the trees trained on only the most recent 1, 2, 4, 8 or all months
- group_importance: how much ranking quality drops when one group of features is shuffled

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

from .snapshots import BASE_COLUMNS, EXTENDED_COLUMNS


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
    features: list[str] = field(default_factory=list)  # columns the model takes, for registering it


def _features(df: pd.DataFrame, columns=BASE_COLUMNS) -> pd.DataFrame:
    return df[list(columns)].astype(float)


def _recent_orders(d: pd.DataFrame) -> np.ndarray:
    """The no-model baseline: more recent orders, more likely to order again."""
    return d["orders_90d"].to_numpy() + d["orders_365d"].to_numpy() / 100  # longer history breaks ties


def compare_models(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame, target: str,
                   seed: int = 7, use_segments: bool = True) -> list[Result]:
    y_col = f"y_{target}"
    ytr, yva, yte = (d[y_col].to_numpy() for d in (train, valid, test))
    results: list[Result] = []

    def add(name, model, p_valid, p_test, columns=()):
        iso = IsotonicRegression(out_of_bounds="clip").fit(p_valid, yva)
        results.append(Result(name, target, metrics(yte, p_test), metrics(yte, iso.predict(p_test)),
                              metrics(yva, p_valid), model, iso, p_test, features=list(columns)))

    add("recent_orders", None, _recent_orders(valid), _recent_orders(test))

    for suffix, cols in (("", BASE_COLUMNS), ("_ext", EXTENDED_COLUMNS)):
        for name, est in (("logistic", _logistic()), ("gbm", _gbm(seed))):
            m = est.fit(_features(train, cols), ytr)
            add(name + suffix, m, m.predict_proba(_features(valid, cols))[:, 1],
                m.predict_proba(_features(test, cols))[:, 1], cols)

    if use_segments and "segment" in train:
        # one-hot over the segments of all three splits, so the columns line up
        segs = sorted(set(train["segment"]) | set(valid["segment"]) | set(test["segment"]))

        def seg_x(d, cols):
            return _features(d, cols).assign(**{f"segment_{s}": (d["segment"] == s).astype(float) for s in segs})

        for name, est, cols in (("logistic_segment", _logistic(), BASE_COLUMNS),
                                ("gbm_segment", _gbm(seed), BASE_COLUMNS),
                                ("gbm_ext_segment", _gbm(seed), EXTENDED_COLUMNS)):
            m = est.fit(seg_x(train, cols), ytr)
            add(name, m, m.predict_proba(seg_x(valid, cols))[:, 1], m.predict_proba(seg_x(test, cols))[:, 1])

        x = {k: _features(d, EXTENDED_COLUMNS) for k, d in (("train", train), ("valid", valid), ("test", test))}
        per = PerSegment(_gbm(seed)).fit(x["train"], ytr, train["segment"].to_numpy())
        add("gbm_ext_per_segment", per, per.predict_proba(x["valid"], valid["segment"].to_numpy()),
            per.predict_proba(x["test"], test["segment"].to_numpy()))
    return results


def metrics_by_segment(test: pd.DataFrame, scores: np.ndarray, target: str) -> pd.DataFrame:
    """Ranking quality within each segment. A model can do well overall by ranking the
    segments against each other while ranking poorly inside one of them."""
    y = test[f"y_{target}"].to_numpy()
    rows = []
    for s in sorted(test["segment"].unique()):
        m = (test["segment"] == s).to_numpy()
        row = {"segment": s, "customers": int(m.sum()), "base_rate": float(y[m].mean())}
        if len(np.unique(y[m])) == 2:  # AUC needs both outcomes
            row.update({k: metrics(y[m], scores[m])[k] for k in ("roc_auc", "avg_precision", "lift_top10")})
        rows.append(row)
    return pd.DataFrame(rows).round(4)


def rolling_evaluation(df: pd.DataFrame, target: str, months: int = 4, seed: int = 7) -> pd.DataFrame:
    """Repeat the core comparison with each of the last `months` cutoffs as the test month.

    For each test month, the month before it is held back as validation and all earlier
    months train, as in the main comparison. No calibration or model choice happens here.
    It only checks how much the ranking quality moves from month to month.
    """
    cutoffs = sorted(df["cutoff"].unique())
    if len(cutoffs) < months + 2:
        raise ValueError(f"need at least {months + 2} cutoffs for {months} test months")
    rows = []
    for test_cutoff in cutoffs[-months:]:
        upto = df[df["cutoff"] <= test_cutoff]
        train, _, test = split_by_cutoff(upto, 1, 1)
        y_tr, y_te = train[f"y_{target}"].to_numpy(), test[f"y_{target}"].to_numpy()
        scores = {"recent_orders": _recent_orders(test)}
        for name, cols in (("gbm", BASE_COLUMNS), ("gbm_ext", EXTENDED_COLUMNS)):
            m = _gbm(seed).fit(_features(train, cols), y_tr)
            scores[name] = m.predict_proba(_features(test, cols))[:, 1]
        for name, p in scores.items():
            m = metrics(y_te, p)
            rows.append({"target": target, "test_cutoff": test_cutoff, "model": name,
                         **{k: m[k] for k in ("roc_auc", "avg_precision", "lift_top10", "base_rate")}})
    return pd.DataFrame(rows).round(4)


def results_table(results: list[Result]) -> pd.DataFrame:
    rows = []
    for r in results:
        row = {"target": r.target, "model": r.name}
        row.update({f"test_{k}": v for k, v in r.test.items()})
        row.update({"test_brier_calibrated": r.test_calibrated["brier"], "test_ece_calibrated": r.test_calibrated["ece"],
                    "valid_roc_auc": r.valid["roc_auc"]})
        rows.append(row)
    return pd.DataFrame(rows).round(4)


def metrics_on_subset(results: list[Result], test: pd.DataFrame, mask: np.ndarray, target: str) -> pd.DataFrame:
    """Every model's test-month ranking quality on a subset of customers."""
    y = test[f"y_{target}"].to_numpy()[mask]
    rows = []
    for r in results:
        m = metrics(y, r.test_scores[mask])
        rows.append({"target": target, "model": r.name, "customers": int(mask.sum()),
                     **{k: m[k] for k in ("roc_auc", "avg_precision", "lift_top10", "base_rate")}})
    return pd.DataFrame(rows).round(4)


def learning_curve(train: pd.DataFrame, test: pd.DataFrame, target: str, months=(1, 2, 4, 8, None),
                   seed: int = 7) -> pd.DataFrame:
    """Trees trained on only the most recent `n` training months (None: all of them)."""
    cutoffs = sorted(train["cutoff"].unique())
    y_te = test[f"y_{target}"].to_numpy()
    rows = []
    for n in months:
        keep = cutoffs if n is None else cutoffs[-n:]
        part = train[train["cutoff"].isin(keep)]
        for name, cols in (("gbm", BASE_COLUMNS), ("gbm_ext", EXTENDED_COLUMNS)):
            m = _gbm(seed).fit(_features(part, cols), part[f"y_{target}"].to_numpy())
            met = metrics(y_te, m.predict_proba(_features(test, cols))[:, 1])
            rows.append({"target": target, "model": name, "train_months": len(keep), "train_rows": len(part),
                         **{k: met[k] for k in ("roc_auc", "avg_precision")}})
    return pd.DataFrame(rows).round(4)


FEATURE_GROUPS = {
    "recency": ["recency_days"],
    "frequency": ["orders_30d", "orders_90d", "orders_365d"],
    "spend": ["spend_30d", "spend_90d", "spend_365d", "avg_order_value", "items"],
    "breadth": ["distinct_products", "tenure_days"],
    "returns": ["return_line_share"],
    "category_shares": [c for c in BASE_COLUMNS if c.startswith("share_")],
    "context": ["is_uk", "cutoff_month"],
    "trend": ["orders_prev_90d", "spend_prev_90d"],
    "seasonality": ["orders_same_window_last_year", "spend_same_window_last_year"],
    "regularity": ["gap_mean_days", "gap_cv"],
    "category_recency": [c for c in EXTENDED_COLUMNS if c.startswith("days_since_")],
}


def group_importance(model, test: pd.DataFrame, columns, target: str, groups=FEATURE_GROUPS,
                     repeats: int = 5, seed: int = 7) -> pd.DataFrame:
    """Drop in test ROC AUC when a group of features is shuffled together across customers.

    Shuffling a whole group at once keeps related features (orders over 30, 90 and 365 days)
    from standing in for each other, which happens when they are shuffled one at a time.
    """
    columns = list(columns)
    x = _features(test, columns)
    y = test[f"y_{target}"].to_numpy()
    base = roc_auc_score(y, model.predict_proba(x)[:, 1])
    rng = np.random.default_rng(seed)
    rows = []
    for group, cols in groups.items():
        cols = [c for c in cols if c in columns]
        if not cols:
            continue
        drops = []
        for _ in range(repeats):
            shuffled = x.copy()
            shuffled[cols] = x[cols].to_numpy()[rng.permutation(len(x))]
            drops.append(base - roc_auc_score(y, model.predict_proba(shuffled)[:, 1]))
        rows.append({"target": target, "group": group, "features": len(cols),
                     "auc_drop": float(np.mean(drops)), "auc_drop_sd": float(np.std(drops))})
    return pd.DataFrame(rows).sort_values("auc_drop", ascending=False).round(4).reset_index(drop=True)
