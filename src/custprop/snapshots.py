"""Customer snapshots: features as of a cutoff date, labels from the window after it.

One row per (cutoff, customer). Features only use transactions strictly before the
cutoff and labels only use the `label_days` after it, so nothing about the future leaks
into the features. Building many monthly cutoffs gives the model more examples and lets
the evaluation train on earlier months and test on later ones, the way it would be used.

A customer is in a snapshot if they bought something in the `history_days` before the
cutoff: the population someone would actually score.

Two feature sets:
- BASE_COLUMNS: recency, frequency, spend, returns and category shares (the first version)
- EXTENDED_COLUMNS: the base set plus information the segments don't contain
  - seasonality: orders and spend in the same window one year earlier (did they buy in
    last year's September to November, for a September cutoff?)
    - this window lies inside the year of history, so it still uses only the past
  - regularity: mean and spread of the gaps between order days
  - trend: orders and spend in the 90 days before the last 90
  - category recency: days since the last purchase in each category
"""

from __future__ import annotations

from datetime import date

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from .categories import CATEGORIES

WINDOWS = (30, 90, 365)


def cutoffs_frame(spark: SparkSession, cutoffs: list[date]) -> DataFrame:
    return spark.createDataFrame([(c,) for c in cutoffs], "cutoff date")


def features(txns: DataFrame, cutoffs: DataFrame, history_days: int = 365, label_days: int = 90) -> DataFrame:
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
    # trend: the 90 days before the last 90
    prev = buys & (F.col("days_before") > 90) & (F.col("days_before") <= 180)
    aggs.append(F.countDistinct(F.when(prev, F.col("invoice"))).alias("orders_prev_90d"))
    aggs.append(F.sum(F.when(prev, F.col("amount")).otherwise(0.0)).alias("spend_prev_90d"))
    # seasonality: last year's version of the label window, [cutoff - 365, cutoff - 365 + label_days)
    season = buys & (F.col("days_before") > history_days - label_days) & (F.col("days_before") <= history_days)
    aggs.append(F.countDistinct(F.when(season, F.col("invoice"))).alias("orders_same_window_last_year"))
    aggs.append(F.sum(F.when(season, F.col("amount")).otherwise(0.0)).alias("spend_same_window_last_year"))
    # category recency; customers who never bought the category get one day more than the history
    for cat in CATEGORIES:
        aggs.append(F.min(F.when(buys & (F.col("category") == cat), F.col("days_before"))).alias(f"days_since_{cat}"))
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
    feats = (
        feats.where(F.col("recency_days").isNotNull())  # only returns in the window: not an active buyer
        .withColumn("avg_order_value", F.when(orders > 0, spent / orders).otherwise(0.0))
        .withColumn("cutoff_month", F.month("cutoff"))
        .drop(*[f"_spend_{c}" for c in CATEGORIES], "_spend_total")
        .fillna(history_days + 1, subset=[f"days_since_{c}" for c in CATEGORIES])
    )
    regularity = _regularity(hist)
    return (
        feats.join(regularity, ["cutoff", "customer_id"], "left")
        .fillna({"gap_mean_days": float(history_days), "gap_cv": 0.0})
    )


def _regularity(hist: DataFrame) -> DataFrame:
    """Mean gap between order days and its coefficient of variation (spread / mean).

    A customer with one order day has no gaps. They get a mean gap of the whole history
    and no spread, which the model can tell apart through the order counts.
    """
    days = (hist.where(~F.col("is_return"))
            .select("cutoff", "customer_id", "days_before").distinct())
    w = Window.partitionBy("cutoff", "customer_id").orderBy(F.desc("days_before"))
    gaps = (
        days.withColumn("gap", F.col("days_before") - F.lead("days_before").over(w))
        .where(F.col("gap").isNotNull())
    )
    return gaps.groupBy("cutoff", "customer_id").agg(
        F.avg("gap").alias("gap_mean_days"),
        (F.stddev_pop("gap") / F.avg("gap")).alias("gap_cv"),
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
    feats = features(txns, cutoffs, history_days, label_days)
    labs = labels(txns, cutoffs, label_days, targets)
    out = feats.join(labs, ["cutoff", "customer_id"], "left")
    return out.fillna(0, subset=[f"y_{t}" for t in targets])


BASE_COLUMNS = (
    ["recency_days", "tenure_days", "distinct_products", "items", "return_line_share", "is_uk",
     "avg_order_value", "cutoff_month"]
    + [f"{k}_{w}d" for w in WINDOWS for k in ("orders", "spend")]
    + [f"share_{c}" for c in CATEGORIES]
)
NEW_COLUMNS = (
    ["orders_prev_90d", "spend_prev_90d", "orders_same_window_last_year", "spend_same_window_last_year",
     "gap_mean_days", "gap_cv"]
    + [f"days_since_{c}" for c in CATEGORIES]
)
EXTENDED_COLUMNS = BASE_COLUMNS + NEW_COLUMNS
