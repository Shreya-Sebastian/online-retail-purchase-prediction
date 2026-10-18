# Databricks notebook source
# MAGIC %md
# MAGIC # 01 Ingest and clean
# MAGIC
# MAGIC Loads Online Retail II (UCI) into a Unity Catalog volume, cleans it and writes two Delta tables:
# MAGIC
# MAGIC - `raw_transactions`: the export as it is, both yearly sheets stacked
# MAGIC - `transactions`: one row per purchased or returned product line, with a coarse category
# MAGIC
# MAGIC If the download is blocked (outbound internet can be restricted), download the zip from the UCI page in a
# MAGIC browser and upload `online_retail_II.xlsx` to the volume (Catalog > workspace > custprop > raw).

# COMMAND ----------

# MAGIC %pip install openpyxl
# MAGIC %restart_python

# COMMAND ----------

import os
import sys

sys.path.append(os.path.abspath("../src"))  # the package sits next to the notebooks in the Git folder

from custprop.clean import clean, clean_report
from custprop.config import DATA_URL, Config

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "custprop")
cfg = Config(catalog=dbutils.widgets.get("catalog"), schema=dbutils.widgets.get("schema"))

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {cfg.catalog}.{cfg.schema}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {cfg.catalog}.{cfg.schema}.raw")

# COMMAND ----------

import io
import urllib.request
import zipfile

xlsx = f"{cfg.volume}/online_retail_II.xlsx"
if not os.path.exists(xlsx):
    with urllib.request.urlopen(DATA_URL, timeout=120) as resp:
        archive = zipfile.ZipFile(io.BytesIO(resp.read()))
    name = next(n for n in archive.namelist() if n.endswith(".xlsx"))
    with open(xlsx, "wb") as fh:
        fh.write(archive.read(name))
print(xlsx, os.path.getsize(xlsx) // 1_000_000, "MB")

# COMMAND ----------

import pandas as pd

sheets = pd.read_excel(xlsx, sheet_name=None, dtype={"Invoice": str, "StockCode": str})
raw_pdf = pd.concat(sheets.values(), ignore_index=True)
spark.createDataFrame(raw_pdf).write.mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable(cfg.table("raw_transactions"))
print(f"{len(raw_pdf):,} raw rows from sheets {list(sheets)}")

# COMMAND ----------

raw = spark.table(cfg.table("raw_transactions"))
clean(raw).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(cfg.table("transactions"))
txns = spark.table(cfg.table("transactions"))
display(pd.DataFrame([clean_report(raw, txns)]))

# COMMAND ----------

# MAGIC %md
# MAGIC How much of the spend the keyword categories cover. "other" is everything no rule matched.

# COMMAND ----------

from pyspark.sql import Window
from pyspark.sql import functions as F

display(
    txns.where(~F.col("is_return"))
    .groupBy("category")
    .agg(F.round(F.sum("amount")).alias("spend"), F.countDistinct("stock_code").alias("products"))
    .withColumn("spend_share", F.round(F.col("spend") / F.sum("spend").over(Window.partitionBy()), 3))
    .orderBy(F.desc("spend"))
)
