"""Customer segmentation on snapshot features: k-means and a Gaussian mixture, compared.

Segments are fitted on training cutoffs only, then applied to every snapshot, so the
test month's customers don't shape the segments the model later uses.

How many segments, and which method, is a judgement call. The functions here report
what the choice rests on: silhouette for k-means, BIC for the mixture, agreement between
two random seeds (stability), and how each segment differs on the original features.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

SEGMENT_FEATURES = ["recency_days", "orders_365d", "spend_365d", "distinct_products", "avg_order_value",
                    "return_line_share"]


def _matrix(df: pd.DataFrame) -> np.ndarray:
    x = df[SEGMENT_FEATURES].astype(float).clip(lower=0).to_numpy()
    return x


def _pipeline(method: str, k: int, seed: int) -> Pipeline:
    model = (KMeans(n_clusters=k, n_init=10, random_state=seed) if method == "kmeans"
             else GaussianMixture(n_components=k, covariance_type="full", random_state=seed))
    # heavy-tailed counts and amounts: log first, then put features on one scale
    return Pipeline([("log", FunctionTransformer(np.log1p)), ("scale", StandardScaler()), ("model", model)])


def compare(df: pd.DataFrame, ks=range(2, 9), seed: int = 7, sample: int = 5000) -> pd.DataFrame:
    """One row per (method, k) with the numbers the choice rests on."""
    x = _matrix(df)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x), size=min(sample, len(x)), replace=False)
    rows = []
    for method in ("kmeans", "gmm"):
        for k in ks:
            a, b = _pipeline(method, k, seed).fit(x), _pipeline(method, k, seed + 1).fit(x)
            la, lb = a.predict(x), b.predict(x)
            z = a[:-1].transform(x)
            row = {
                "method": method, "k": k,
                "silhouette": silhouette_score(z[idx], la[idx]) if len(set(la[idx])) > 1 else np.nan,
                "stability_ari": adjusted_rand_score(la, lb),
                "smallest_share": np.bincount(la, minlength=k).min() / len(la),
            }
            if method == "gmm":
                row["bic"] = a[-1].bic(z)
            rows.append(row)
    return pd.DataFrame(rows)


@dataclass
class Segmenter:
    method: str
    k: int
    pipeline: Pipeline
    order: np.ndarray  # cluster id from the model -> segment number

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.order[self.pipeline.predict(_matrix(df))]


def fit(df: pd.DataFrame, method: str, k: int, seed: int = 7) -> Segmenter:
    """Fit, then number segments by median recency, most recent first.

    The model's own cluster ids depend on row order, which Spark doesn't fix, so the same
    segments could come back with different numbers from one run to the next.
    """
    pipeline = _pipeline(method, k, seed).fit(_matrix(df))
    raw = pipeline.predict(_matrix(df))
    recency = pd.Series(df["recency_days"].to_numpy()).groupby(raw).median().reindex(range(k))
    ranked = recency.fillna(np.inf).sort_values(kind="stable").index.to_numpy()
    order = np.empty(k, dtype=int)
    order[ranked] = np.arange(k)
    return Segmenter(method, k, pipeline, order)


def profile(df: pd.DataFrame, segments: np.ndarray, extra: list[str] | None = None) -> pd.DataFrame:
    """Size and median feature values per segment, to name and sanity-check them."""
    cols = SEGMENT_FEATURES + (extra or [])
    out = df[cols].assign(segment=segments).groupby("segment").median()
    out.insert(0, "share", pd.Series(segments).value_counts(normalize=True).sort_index().to_numpy())
    return out.round(2)
