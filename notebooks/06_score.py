# Databricks notebook source
# MAGIC %md
# MAGIC # 06 Batch scoring and monitoring
# MAGIC
# MAGIC Scores every active customer for the months after the test month with the `champion` version of each
# MAGIC registered model, and checks the result:
# MAGIC
# MAGIC - drift: how far each feature and the scores have moved from the test month, by PSI
# MAGIC - expected against actual buyers on the test month, the last month whose label window is complete
# MAGIC
# MAGIC Writes `scores` (one row per cutoff, customer and target) and `score_drift`. The scored months' label
# MAGIC windows run past the end of the data, as they would for the current month, so they get an expected
# MAGIC number of buyers but no actual one.

# COMMAND ----------

import os
import sys

sys.path.append(os.path.abspath("../src"))

from datetime import datetime, timezone

import mlflow
import pandas as pd
from mlflow import MlflowClient
from pyspark.sql import functions as F

from custprop import monitor
from custprop.config import Config
from custprop.snapshots import cutoffs_frame, features

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "custprop")
cfg = Config(catalog=dbutils.widgets.get("catalog"), schema=dbutils.widgets.get("schema"))
mlflow.set_registry_uri("databricks-uc")

txns = spark.table(cfg.table("transactions"))
last_day = txns.agg(F.max("invoice_date")).first()[0]
current = (features(txns, cutoffs_frame(spark, list(cfg.score_cutoffs)), cfg.history_days, cfg.label_days)
           .orderBy("cutoff", "customer_id").toPandas())
snaps = spark.table(cfg.table("snapshots")).orderBy("cutoff", "customer_id").toPandas()
reference = snaps[snaps.cutoff == snaps.cutoff.max()]  # the test month
print(len(reference), "customers in the test month;", current.groupby("cutoff").size().to_dict())

# COMMAND ----------

client = MlflowClient()
scored_at = datetime.now(timezone.utc)
scores, drift, checks = [], [], []
for target in cfg.targets:
    name = cfg.table(f"propensity_{target}")
    version = client.get_model_version_by_alias(name, "champion").version
    model = mlflow.sklearn.load_model(f"models:/{name}@champion")
    p_ref = model.predict_proba(reference)[:, 1]
    checks.append({"target": target, "model_version": version, "cutoff": reference.cutoff.max(),
                   **monitor.expected_vs_actual(p_ref, reference[f"y_{target}"])})
    for cutoff, month in current.groupby("cutoff"):
        p = model.predict_proba(month)[:, 1]
        scores.append(month[["cutoff", "customer_id"]].assign(target=target, probability=p,
                                                               model_version=version, scored_at=scored_at))
        complete = (last_day - cutoff).days >= cfg.label_days
        checks.append({"target": target, "model_version": version, "cutoff": cutoff, "customers": len(p),
                       "expected_buyers": float(p.sum()), "label_window_complete": complete})
        cols = monitor.drift(reference.assign(score=p_ref), month.assign(score=p), model.features + ["score"])
        drift.append(cols.assign(target=target, cutoff=cutoff, model_version=version))

scores = pd.concat(scores, ignore_index=True)
drift = pd.concat(drift, ignore_index=True)
checks = pd.DataFrame(checks)

# COMMAND ----------

# the table holds the latest run: every run scores the same months again
spark.createDataFrame(scores).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(cfg.table("scores"))
spark.createDataFrame(drift).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(cfg.table("score_drift"))

# COMMAND ----------

# MAGIC %md
# MAGIC Expected buyers per target and month. The first row per target is the test month, where the actual
# MAGIC number is known.

# COMMAND ----------

display(checks)

# COMMAND ----------

# MAGIC %md
# MAGIC Drift against the test month: every column above 0.1, and the score itself.

# COMMAND ----------

display(drift[(drift.psi >= 0.1) | (drift.column == "score")].sort_values(["target", "cutoff", "psi"], ascending=[True, True, False]))

# COMMAND ----------

mlflow.set_experiment(f"/Users/{spark.sql('select current_user()').first()[0]}/custprop")
with mlflow.start_run(run_name="batch-scoring"):
    mlflow.log_params({"score_cutoffs": ",".join(str(c) for c in cfg.score_cutoffs),
                       "reference_cutoff": str(reference.cutoff.max())})
    mlflow.log_metric("rows_scored", len(scores))
    mlflow.log_table(checks.astype({"cutoff": str}), "expected_buyers.json")
    mlflow.log_table(drift.astype({"cutoff": str}), "drift.json")
