import os
import sys

import pytest

# PySpark starts its Python workers with this interpreter; on Windows it otherwise picks up "python" from PATH.
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)


@pytest.fixture(scope="session")
def spark():
    from pyspark.sql import SparkSession

    s = (SparkSession.builder.master("local[2]").appName("custprop-tests")
         .config("spark.sql.shuffle.partitions", "4")
         .config("spark.sql.session.timeZone", "UTC")
         .config("spark.ui.enabled", "false")
         # convert pandas through Arrow with no fallback, as Databricks does
         .config("spark.sql.execution.arrow.pyspark.enabled", "true")
         .config("spark.sql.execution.arrow.pyspark.fallback.enabled", "false")
         .getOrCreate())
    yield s
    s.stop()
