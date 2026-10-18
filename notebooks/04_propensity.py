# Databricks notebook source
# MAGIC %md
# MAGIC # 04 Propensity models
# MAGIC
# MAGIC For each target (any purchase, and purchases in a category) compares a no-model baseline, logistic
# MAGIC regression and gradient-boosted trees on the first version's features, the same two models on the
# MAGIC extended features, and two segment variants. Trained on earlier cutoffs, calibrated and chosen on the
# MAGIC validation month, scored on the test month. Every model is an MLflow run. The best plain model by
# MAGIC validation average precision is registered in Unity Catalog.
# MAGIC
# MAGIC Then two checks on the same data: ranking quality within each segment, and a rolling evaluation with
# MAGIC each of the last four months as the test month.

# COMMAND ----------

import os
import sys

sys.path.append(os.path.abspath("../src"))

import matplotlib.pyplot as plt
import mlflow
import pandas as pd

from custprop.config import Config
from custprop.propensity import (calibration_table, compare_models, metrics_by_segment, results_table,
                                 rolling_evaluation, split_by_cutoff)

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "custprop")
dbutils.widgets.text("register", "true")
cfg = Config(catalog=dbutils.widgets.get("catalog"), schema=dbutils.widgets.get("schema"))

snaps = spark.table(cfg.table("snapshots_segmented")).orderBy("cutoff", "customer_id").toPandas()  # fixed row order, so reruns match
train, valid, test = split_by_cutoff(snaps, cfg.test_cutoffs, cfg.valid_cutoffs)
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(f"/Users/{spark.sql('select current_user()').first()[0]}/custprop")

# COMMAND ----------

def calibration_plot(y, p_raw, p_cal, title):
    fig, ax = plt.subplots(figsize=(4, 4))
    for p, label in ((p_raw, "raw"), (p_cal, "isotonic")):
        t = calibration_table(y, p)
        ax.plot(t.predicted, t.observed, marker="o", label=label)
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1)
    ax.set(xlabel="predicted", ylabel="observed", title=title)
    ax.legend()
    return fig


# only models that take the plain feature table are registered; the segment variants would also
# need the segmenter at scoring time, so they are compared but not registered
REGISTERABLE = ("logistic", "gbm", "logistic_ext", "gbm_ext")

tables, by_segment = [], []
for target in cfg.targets:
    results = compare_models(train, valid, test, target, cfg.seed)
    tables.append(results_table(results))
    for r in results:
        with mlflow.start_run(run_name=f"{target}-{r.name}") as run:
            mlflow.log_params({"target": target, "model": r.name, "label_days": cfg.label_days,
                               "features": len(r.features), "train_cutoffs": train.cutoff.nunique(),
                               "test_cutoff": str(test.cutoff.max())})
            mlflow.log_metrics({f"test_{k}": v for k, v in r.test.items()})
            mlflow.log_metrics({f"test_cal_{k}": v for k, v in r.test_calibrated.items()})
            mlflow.log_metrics({f"valid_{k}": v for k, v in r.valid.items()})
            y = test[f"y_{target}"].to_numpy()
            mlflow.log_figure(calibration_plot(y, r.test_scores, r.calibrator.predict(r.test_scores),
                                               f"{target} / {r.name}"), "calibration.png")
            plt.close("all")
            r.run_id = run.info.run_id

    best = max((r for r in results if r.name in REGISTERABLE), key=lambda r: r.valid["avg_precision"])
    print(f"{target}: best on the validation month is {best.name}")
    by_segment.append(metrics_by_segment(test, best.test_scores, target).assign(target=target, model=best.name))
    if dbutils.widgets.get("register") == "true":
        with mlflow.start_run(run_id=best.run_id):
            mlflow.sklearn.log_model(best.model, "model", input_example=test[best.features].astype(float).head(5),
                                     registered_model_name=cfg.table(f"propensity_{target}"))

comparison = pd.concat(tables, ignore_index=True)
display(comparison)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Within each segment
# MAGIC
# MAGIC How well the chosen model ranks customers inside each segment, on the test month.

# COMMAND ----------

segment_table = pd.concat(by_segment, ignore_index=True)
display(segment_table)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Rolling evaluation
# MAGIC
# MAGIC The baseline and the trees on both feature sets, with each of the last four months as the test month.

# COMMAND ----------

rolling = pd.concat([rolling_evaluation(snaps, t, months=4, seed=cfg.seed) for t in cfg.targets], ignore_index=True)
display(rolling)
display(rolling.groupby(["target", "model"])[["roc_auc", "avg_precision", "lift_top10"]]
        .agg(["mean", "min", "max"]).round(3).reset_index())

# COMMAND ----------

with mlflow.start_run(run_name="evaluation-tables"):
    mlflow.log_table(comparison, "comparison.json")
    mlflow.log_table(segment_table, "by_segment.json")
    mlflow.log_table(rolling.assign(test_cutoff=rolling.test_cutoff.astype(str)), "rolling.json")
