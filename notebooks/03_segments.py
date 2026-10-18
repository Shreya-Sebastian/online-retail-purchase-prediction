# Databricks notebook source
# MAGIC %md
# MAGIC # 03 Customer segments
# MAGIC
# MAGIC Compares k-means and a Gaussian mixture over a range of k, on training cutoffs only. The choice of
# MAGIC method and k is made here by hand from the table below and recorded in the widgets, so it is visible
# MAGIC and repeatable. Writes `snapshots_segmented` and logs the segmenter to MLflow.

# COMMAND ----------

import os
import sys

sys.path.append(os.path.abspath("../src"))

import mlflow

from custprop import segment
from custprop.config import Config
from custprop.propensity import split_by_cutoff

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "custprop")
dbutils.widgets.dropdown("method", "kmeans", ["kmeans", "gmm"])
dbutils.widgets.text("k", "4")
cfg = Config(catalog=dbutils.widgets.get("catalog"), schema=dbutils.widgets.get("schema"))

snaps = spark.table(cfg.table("snapshots")).toPandas()
train, valid, test = split_by_cutoff(snaps, cfg.test_cutoffs, cfg.valid_cutoffs)
print(len(train), len(valid), len(test), "snapshot rows in train / valid / test")

# COMMAND ----------

table = segment.compare(train, ks=range(2, 9), seed=cfg.seed)
display(table)

# COMMAND ----------

# MAGIC %md
# MAGIC Pick a method and k that balance separation (silhouette, BIC) with stability across seeds and no tiny
# MAGIC segments, then set the widgets above. The profile shows whether the segments are interpretable.

# COMMAND ----------

method, k = dbutils.widgets.get("method"), int(dbutils.widgets.get("k"))
seg = segment.fit(train, method, k, cfg.seed)
display(segment.profile(train, seg.predict(train), extra=["share_kitchen", "share_bags", "share_christmas"]).reset_index())

# COMMAND ----------

mlflow.set_experiment(f"/Users/{spark.sql('select current_user()').first()[0]}/custprop")
with mlflow.start_run(run_name=f"segments-{method}-{k}"):
    mlflow.log_params({"method": method, "k": k, "features": ",".join(segment.SEGMENT_FEATURES)})
    row = table[(table.method == method) & (table.k == k)].iloc[0]
    mlflow.log_metrics({m: float(row[m]) for m in ("silhouette", "stability_ari", "smallest_share")})
    mlflow.log_table(table, "segment_comparison.json")
    mlflow.sklearn.log_model(seg.pipeline, "segmenter", input_example=train[segment.SEGMENT_FEATURES].head(5))

snaps["segment"] = seg.predict(snaps)
spark.createDataFrame(snaps).write.mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable(cfg.table("snapshots_segmented"))
