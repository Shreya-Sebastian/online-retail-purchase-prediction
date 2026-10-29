# Databricks notebook source
# MAGIC %md
# MAGIC # 05 Next-best category
# MAGIC
# MAGIC One calibrated propensity model per category, and the category with the highest probability as the
# MAGIC suggestion for each customer. Compared on the test month with three suggestions that need no model:
# MAGIC the most popular categories, the customer's own most-bought categories and their most recent ones.
# MAGIC A hit means the customer bought from a suggested category in the 90 days after the cutoff. Only
# MAGIC customers who bought from at least one of the categories count.

# COMMAND ----------

import os
import sys

sys.path.append(os.path.abspath("../src"))

import mlflow

from custprop import recommend
from custprop.config import Config
from custprop.propensity import split_by_cutoff

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "custprop")
cfg = Config(catalog=dbutils.widgets.get("catalog"), schema=dbutils.widgets.get("schema"))

snaps = spark.table(cfg.table("snapshots_segmented")).orderBy("cutoff", "customer_id").toPandas()  # fixed row order, so reruns match
train, valid, test = split_by_cutoff(snaps, cfg.test_cutoffs, cfg.valid_cutoffs)
table, differences, mix = recommend.evaluate(train, valid, test, cfg.recommend_categories, seed=cfg.seed)

# COMMAND ----------

display(table)

# COMMAND ----------

# MAGIC %md
# MAGIC The calibrated model against the best of the three baselines, with a 95% interval from resampling the
# MAGIC test customers.

# COMMAND ----------

display(differences)

# COMMAND ----------

# MAGIC %md
# MAGIC How often each category is the first suggestion, against how often it is bought.

# COMMAND ----------

display(mix)

# COMMAND ----------

mlflow.set_experiment(f"/Users/{spark.sql('select current_user()').first()[0]}/custprop")
with mlflow.start_run(run_name="next-best-category"):
    mlflow.log_params({"categories": ",".join(cfg.recommend_categories), "test_cutoff": str(test.cutoff.max())})
    for _, row in table.iterrows():
        mlflow.log_metric(f"hit_at_{row.k}_{row.method}", row.hit_rate)
    mlflow.log_table(table, "hit_rates.json")
    mlflow.log_table(differences, "model_vs_best_baseline.json")
    mlflow.log_table(mix, "suggested_vs_bought.json")
