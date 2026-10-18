# Databricks notebook source
# MAGIC %md
# MAGIC # 04 Propensity models
# MAGIC
# MAGIC For each target (any purchase, and purchases in a category) compares a no-model baseline, logistic
# MAGIC regression, gradient-boosted trees, trees with the segment as a feature and one model per segment.
# MAGIC Trained on earlier cutoffs, calibrated and chosen on the validation month, scored on the test month.
# MAGIC Every model is an MLflow run. The better of logistic regression and the trees, by validation average
# MAGIC precision, is registered in Unity Catalog.

# COMMAND ----------

import os
import sys

sys.path.append(os.path.abspath("../src"))

import matplotlib.pyplot as plt
import mlflow
import pandas as pd

from custprop.config import Config
from custprop.propensity import calibration_table, compare_models, results_table, split_by_cutoff
from custprop.snapshots import FEATURE_COLUMNS

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "custprop")
dbutils.widgets.text("register", "true")
cfg = Config(catalog=dbutils.widgets.get("catalog"), schema=dbutils.widgets.get("schema"))

snaps = spark.table(cfg.table("snapshots_segmented")).toPandas()
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


tables = []
for target in cfg.targets:
    results = compare_models(train, valid, test, target, cfg.seed)
    tables.append(results_table(results))
    for r in results:
        with mlflow.start_run(run_name=f"{target}-{r.name}") as run:
            mlflow.log_params({"target": target, "model": r.name, "label_days": cfg.label_days,
                               "train_cutoffs": train.cutoff.nunique(), "test_cutoff": str(test.cutoff.max())})
            mlflow.log_metrics({f"test_{k}": v for k, v in r.test.items()})
            mlflow.log_metrics({f"test_cal_{k}": v for k, v in r.test_calibrated.items()})
            mlflow.log_metrics({f"valid_{k}": v for k, v in r.valid.items()})
            y = test[f"y_{target}"].to_numpy()
            mlflow.log_figure(calibration_plot(y, r.test_scores, r.calibrator.predict(r.test_scores),
                                               f"{target} / {r.name}"), "calibration.png")
            plt.close("all")
            r.run_id = run.info.run_id

    # only the models that take the plain feature table are registered; the segment variants would also
    # need the segmenter at scoring time, so they are compared but not registered
    best = max((r for r in results if r.name in ("logistic", "gbm")), key=lambda r: r.valid["avg_precision"])
    if dbutils.widgets.get("register") == "true":
        with mlflow.start_run(run_id=best.run_id):
            mlflow.sklearn.log_model(best.model, "model", input_example=test[FEATURE_COLUMNS].astype(float).head(5),
                                     registered_model_name=cfg.table(f"propensity_{target}"))

display(pd.concat(tables, ignore_index=True))
