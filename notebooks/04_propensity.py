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
# MAGIC Then more checks on the same data: ranking quality within each segment, every model on customers
# MAGIC with a single order, a rolling evaluation over the last four months, a learning curve, and which
# MAGIC feature groups the extended trees rely on. The expected outcome of the last three is written in the
# MAGIC README before the run.

# COMMAND ----------

import os
import sys

sys.path.append(os.path.abspath("../src"))

import matplotlib.pyplot as plt
import mlflow
import pandas as pd

from custprop.config import Config
from custprop.propensity import (calibration_table, compare_models, group_importance, learning_curve,
                                 metrics_by_segment, metrics_on_subset, results_table, rolling_evaluation,
                                 split_by_cutoff)

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

# customers with a single order in the past year: the order-count baseline can barely rank them
one_order = (test["orders_365d"] == 1).to_numpy()

tables, by_segment, single, importance = [], [], [], []
for target in cfg.targets:
    results = compare_models(train, valid, test, target, cfg.seed)
    tables.append(results_table(results))
    single.append(metrics_on_subset(results, test, one_order, target))
    ext = next(r for r in results if r.name == "gbm_ext")
    importance.append(group_importance(ext.model, test, ext.features, target, seed=cfg.seed))
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
# MAGIC ## Segments in a linear model
# MAGIC
# MAGIC Logistic regression and the trees on the base features, with and without the segment.

# COMMAND ----------

wanted = ["logistic", "logistic_segment", "gbm", "gbm_segment"]
display(comparison[comparison.model.isin(wanted)][["target", "model", "test_roc_auc", "test_avg_precision"]]
        .pivot(index="target", columns="model").round(4))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Customers with one order
# MAGIC
# MAGIC Every model on the test-month customers with a single order in the past year.

# COMMAND ----------

single_table = pd.concat(single, ignore_index=True)
display(single_table)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Which features the extended trees use
# MAGIC
# MAGIC Drop in test ROC AUC when one group of features is shuffled, averaged over five shuffles.

# COMMAND ----------

importance_table = pd.concat(importance, ignore_index=True)
display(importance_table)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Learning curve
# MAGIC
# MAGIC The trees trained on only the most recent 1, 2, 4, 8 or all training months, scored on the test month.

# COMMAND ----------

curve = pd.concat([learning_curve(train, test, t, seed=cfg.seed) for t in cfg.targets], ignore_index=True)
display(curve)

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
    mlflow.log_table(single_table, "one_order_customers.json")
    mlflow.log_table(importance_table, "group_importance.json")
    mlflow.log_table(curve, "learning_curve.json")
