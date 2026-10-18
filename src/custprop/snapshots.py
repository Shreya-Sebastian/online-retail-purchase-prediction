"""Customer snapshots: features as of a cutoff date, labels from the window after it.

One row per (cutoff, customer). Features only use transactions strictly before the
cutoff and labels only use the `label_days` after it, so nothing about the future leaks
into the features. Building many monthly cutoffs gives the model more examples and lets
the evaluation train on earlier months and test on later ones, the way it would be used.

A customer is in a snapshot if they bought something in the `history_days` before the
cutoff: the population someone would actually score.
"""

from __future__ import annotations

from datetime import date

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from .categories import CATEGORIES

WINDOWS = (30, 90, 365)


def cutoffs_frame(spark: SparkSession, cutoffs: list[date]) -> DataFrame:
    return spark.createDataFrame([(c,) for c in cutoffs], "cutoff date")


def features(txns: DataFrame, cutoffs: DataFrame, history_days: int = 365) -> DataFrame:
    if history_days not in WINDOWS:
        raise ValueError(f"history_days must be one of {WINDOWS}, got {history_days}")
    days_before = F.datediff(F.col("cutoff"), F.col("invoice_date"))
    hist = txns.crossJoin(cutoffs).where((days_before > 0) & (days_before <= history_days))
    hist = hist.withColumn("days_before", days_before)
    buys = F.col("is_return") == F.lit(False)

    aggs = [
        F.min(F.when(buys, F.col("days_before"))).alias("recency_days"),
        F.max("days_before").alias("tenure_days"),  # within the history window
        F.countDistinct(F.when(buys, F.col("stock_code"))).alias("distinct_products"),
        F.sum(F.when(buys, F.col("quantity")).otherwise(0)).alias("items"),
        F.avg(F.col("is_return").cast("double")).alias("return_line_share"),
        F.max((F.col("country") == "United Kingdom").cast("int")).alias("is_uk"),
    ]
    for w in WINDOWS:
        in_w = buys & (F.col("days_before") <= w)
        aggs.append(F.countDistinct(F.when(in_w, F.col("invoice"))).alias(f"orders_{w}d"))
        aggs.append(F.sum(F.when(in_w, F.col("amount")).otherwise(0.0)).alias(f"spend_{w}d"))
    spend = F.sum(F.when(buys, F.col("amount")).otherwise(0.0))
    for cat in CATEGORIES:
        aggs.append(F.sum(F.when(buys & (F.col("category") == cat), F.col("amount")).otherwise(0.0)).alias(f"_spend_{cat}"))
    aggs.append(spend.alias("_spend_total"))

    feats = hist.groupBy("cutoff", "customer_id").agg(*aggs)
    total = F.col("_spend_total")
    for cat in CATEGORIES:
        feats = feats.withColumn(
            f"share_{cat}", F.when(total > 0, F.col(f"_spend_{cat}") / total).otherwise(0.0)
        )
    orders, spent = F.col(f"orders_{history_days}d"), F.col(f"spend_{history_days}d")
    return (
        feats.where(F.col("recency_days").isNotNull())  # only returns in the window: not an active buyer
        .withColumn("avg_order_value", F.when(orders > 0, spent / orders).otherwise(0.0))
        .withColumn("cutoff_month", F.month("cutoff"))
        .drop(*[f"_spend_{c}" for c in CATEGORIES], "_spend_total")
    )


def labels(txns: DataFrame, cutoffs: DataFrame, label_days: int = 90, targets=("any",)) -> DataFrame:
    """1 if the customer buys (any product, or one in the target category) within the window."""
    days_after = F.datediff(F.col("invoice_date"), F.col("cutoff"))
    future = (
        txns.where(~F.col("is_return"))
        .crossJoin(cutoffs)
        .where((days_after >= 0) & (days_after < label_days))
    )
    aggs = []
    for t in targets:
        hit = F.lit(True) if t == "any" else (F.col("category") == t)
        aggs.append(F.max(hit.cast("int")).alias(f"y_{t}"))
    return future.groupBy("cutoff", "customer_id").agg(*aggs)


def snapshots(txns: DataFrame, cutoffs: DataFrame, history_days: int = 365, label_days: int = 90,
              targets=("any",)) -> DataFrame:
    feats = features(txns, cutoffs, history_days)
    labs = labels(txns, cutoffs, label_days, targets)
    out = feats.join(labs, ["cutoff", "customer_id"], "left")
    return out.fillna(0, subset=[f"y_{t}" for t in targets])


FEATURE_COLUMNS = (
    ["recency_days", "tenure_days", "distinct_products", "items", "return_line_share", "is_uk",
     "avg_order_value", "cutoff_month"]
    + [f"{k}_{w}d" for w in WINDOWS for k in ("orders", "spend")]
    + [f"share_{c}" for c in CATEGORIES]
)
