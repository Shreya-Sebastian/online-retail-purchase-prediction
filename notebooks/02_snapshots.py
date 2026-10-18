# Databricks notebook source
# MAGIC %md
# MAGIC # 02 Customer snapshots
# MAGIC
# MAGIC One row per customer per monthly cutoff: features from the year before the cutoff, labels from the
# MAGIC 90 days after it. Features never see anything on or after the cutoff date. Writes `snapshots`.

# COMMAND ----------

import os
import sys

sys.path.append(os.path.abspath("../src"))

from pyspark.sql import functions as F

from custprop.config import Config, monthly_cutoffs
from custprop.snapshots import FEATURE_COLUMNS, cutoffs_frame, snapshots

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "custprop")
cfg = Config(catalog=dbutils.widgets.get("catalog"), schema=dbutils.widgets.get("schema"))

# COMMAND ----------

txns = spark.table(cfg.table("transactions"))
last_day = txns.agg(F.max("invoice_date")).first()[0]
cutoffs = monthly_cutoffs(cfg.first_cutoff, cfg.last_cutoff)
# every label window has to end before the data does, or late cutoffs get too few positives
assert all((last_day - c).days >= cfg.label_days for c in cutoffs), (last_day, cutoffs[-1])

snaps = snapshots(txns, cutoffs_frame(spark, cutoffs), cfg.history_days, cfg.label_days, cfg.targets)
snaps.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(cfg.table("snapshots"))

# COMMAND ----------

snaps = spark.table(cfg.table("snapshots"))
display(
    snaps.groupBy("cutoff")
    .agg(F.count("*").alias("customers"), *[F.round(F.avg(f"y_{t}"), 3).alias(f"rate_{t}") for t in cfg.targets])
    .orderBy("cutoff")
)

# COMMAND ----------

# MAGIC %md
# MAGIC Feature summary for the latest cutoff.

# COMMAND ----------

display(snaps.where(F.col("cutoff") == max(cutoffs)).select(*FEATURE_COLUMNS).summary("mean", "50%", "max"))
