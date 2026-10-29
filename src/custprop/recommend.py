"""Next-best category: which product category to suggest to each customer.

One propensity model per category gives every customer a score for each category, and the
suggestion is the category with the highest score. That only works if the scores can be
compared across categories: a raw score of 0.3 for a rare category is not the same thing
as 0.3 for a common one. So each model is calibrated on the validation month and the
choice is made on calibrated probabilities. The raw scores are compared too, to see
whether calibration changes the choice.

Evaluated on test-month customers who bought from at least one of the categories in the
label window: a hit means a suggested category is one they bought.

Baselines, which need no model:
- popular: the categories most customers bought in the training months, the same for everyone
- most_bought: the categories with the largest share of the customer's past spend
- most_recent: the categories the customer bought from most recently
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from .propensity import _features, _gbm
from .snapshots import EXTENDED_COLUMNS


def category_scores(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame, categories,
                    seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Raw and calibrated test-month scores, one column per category."""
    raw, calibrated = {}, {}
    for c in categories:
        y = f"y_{c}"
        model = _gbm(seed).fit(_features(train, EXTENDED_COLUMNS), train[y].to_numpy())
        p_valid = model.predict_proba(_features(valid, EXTENDED_COLUMNS))[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip").fit(p_valid, valid[y].to_numpy())
        raw[c] = model.predict_proba(_features(test, EXTENDED_COLUMNS))[:, 1]
        calibrated[c] = iso.predict(raw[c])
    return pd.DataFrame(raw, index=test.index), pd.DataFrame(calibrated, index=test.index)


def top_k(scores: pd.DataFrame, k: int) -> np.ndarray:
    """The k highest-scored categories per row, best first. Ties go to the earlier column."""
    order = np.argsort(-scores.to_numpy(), axis=1, kind="stable")[:, :k]
    return np.asarray(scores.columns)[order]


def hits(bought: pd.DataFrame, picks: np.ndarray) -> np.ndarray:
    """1 where at least one of the picked categories was bought."""
    cols = {c: i for i, c in enumerate(bought.columns)}
    b = bought.to_numpy().astype(bool)
    idx = np.vectorize(cols.get)(picks)
    return b[np.arange(len(b))[:, None], idx].any(axis=1).astype(int)


def baseline_scores(train: pd.DataFrame, test: pd.DataFrame, categories) -> dict[str, pd.DataFrame]:
    """Scores that rank categories the way each baseline does. Popularity breaks ties, so a
    customer who never bought a category still gets the popular ones after their own."""
    popularity = pd.Series({c: train[f"y_{c}"].mean() for c in categories})
    tie = popularity.rank() / 1000  # small enough never to beat a real difference
    ones = pd.DataFrame(1.0, index=test.index, columns=list(categories))
    return {
        "popular": ones * tie,
        "most_bought": test[[f"share_{c}" for c in categories]].set_axis(list(categories), axis=1) + tie,
        "most_recent": -test[[f"days_since_{c}" for c in categories]].set_axis(list(categories), axis=1) + tie,
    }


def evaluate(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame, categories, ks=(1, 3),
             seed: int = 7, n_boot: int = 1000) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Hit rates per method and k, the model against the best baseline with a 95% interval,
    and how often each category is suggested against how often it is bought."""
    categories = list(categories)
    bought = test[[f"y_{c}" for c in categories]].set_axis(categories, axis=1)
    keep = bought.sum(axis=1) > 0
    raw, calibrated = category_scores(train, valid, test, categories, seed)
    methods = {**baseline_scores(train, test, categories), "model_raw": raw, "model_calibrated": calibrated}

    rows, hit_vectors = [], {}
    for k in ks:
        for name, scores in methods.items():
            h = hits(bought[keep], top_k(scores[keep], k))
            hit_vectors[(name, k)] = h
            rows.append({"method": name, "k": k, "customers": int(keep.sum()), "hit_rate": h.mean(),
                         "distinct_suggestions": len(np.unique(top_k(scores[keep], k)))})
    table = pd.DataFrame(rows).round(4)

    rng = np.random.default_rng(seed)
    diff_rows = []
    for k in ks:
        base = table[(table.k == k) & table.method.isin(["popular", "most_bought", "most_recent"])]
        best = base.sort_values("hit_rate").iloc[-1]["method"]
        a, b = hit_vectors[(best, k)], hit_vectors[("model_calibrated", k)]
        draws = [b[i].mean() - a[i].mean() for i in (rng.integers(0, len(a), len(a)) for _ in range(n_boot))]
        low, high = np.percentile(draws, [2.5, 97.5])
        diff_rows.append({"k": k, "model": "model_calibrated", "against": best, "diff": b.mean() - a.mean(),
                          "ci_low": low, "ci_high": high})
    differences = pd.DataFrame(diff_rows).round(4)

    first = pd.Series(top_k(calibrated[keep], 1)[:, 0]).value_counts(normalize=True)
    mix = pd.DataFrame({"category": categories,
                        "suggested_first": [first.get(c, 0.0) for c in categories],
                        "bought": [bought.loc[keep, c].mean() for c in categories]}).round(4)
    return table, differences, mix
