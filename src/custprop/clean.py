"""Raw Online Retail II rows -> one clean row per purchased (or returned) line.

The raw export mixes several things that are not customer purchases:
- cancellations: invoice numbers starting with "C", negative quantities;
- rows without a customer id (guest checkouts), which can't be followed over time;
- postage, bank charges, manual adjustments and samples under special stock codes;
- zero or negative prices (damaged stock, write-offs).

Cancellations are kept, flagged as returns, because the return rate is a useful feature.
Everything else that isn't a product sale is dropped, and `clean_report` says how much.
"""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from .categories import category_expr

# Stock codes that aren't products.
NON_PRODUCT_CODES = ("POST", "DOT", "M", "D", "C2", "S", "B", "AMAZONFEE", "BANK CHARGES", "PADS", "CRUK", "ADJUST", "ADJUST2", "TEST001", "TEST002", "GIFT")

RAW_COLUMNS = {
    "Invoice": "invoice",
    "StockCode": "stock_code",
    "Description": "description",
    "Quantity": "quantity",
    "InvoiceDate": "invoice_ts",
    "Price": "price",
    "Customer ID": "customer_id",
    "Country": "country",
}


TEXT_COLUMNS = ("Invoice", "StockCode", "Description", "Country")


def prepare_raw_pandas(pdf):
    """Make the Excel export safe to hand to Spark and to store as a Delta table.

    - Excel cells in a text column are not always text: some descriptions and stock codes
      are numbers, and empty cells come back as NaN. A pandas column mixing those can't be
      converted to a Spark column, so every text column becomes one string type with real
      missing values.
    - Delta doesn't allow spaces in column names ("Customer ID"), so the columns get the
      snake_case names used everywhere else.
    """
    import pandas as pd

    out = pdf.copy()
    for col in TEXT_COLUMNS:
        if col in out:
            out[col] = out[col].map(lambda v: None if pd.isna(v) else str(v)).astype("string")
    return out.rename(columns=RAW_COLUMNS)


def standardise(raw: DataFrame) -> DataFrame:
    """Accept the export with its original column names or already renamed."""
    df = raw
    if "Invoice" in raw.columns:
        df = raw.select([F.col(f"`{src}`").alias(dst) for src, dst in RAW_COLUMNS.items()])
    # Read through pandas, a missing customer id can be NaN, which Spark keeps as a number
    # and would cast to customer 0. Make it a real null first.
    if dict(df.dtypes)["customer_id"] in ("double", "float"):
        df = df.withColumn("customer_id", F.when(~F.isnan("customer_id"), F.col("customer_id")))
    return df


def clean(raw: DataFrame) -> DataFrame:
    df = standardise(raw)
    code = F.upper(F.trim(F.col("stock_code").cast("string")))
    df = (
        df.withColumn("invoice", F.trim(F.col("invoice").cast("string")))
        .withColumn("stock_code", code)
        .withColumn("customer_id", F.col("customer_id").cast("long").cast("string"))
        .withColumn("invoice_ts", F.col("invoice_ts").cast("timestamp"))
        .withColumn("is_return", F.col("invoice").startswith("C") | (F.col("quantity") < 0))
    )
    df = df.where(
        F.col("customer_id").isNotNull()
        & (F.col("price") > 0)
        & (F.col("quantity") != 0)
        & ~F.col("stock_code").isin(*NON_PRODUCT_CODES)
        & ~F.col("stock_code").startswith("GIFT_")
    )
    return (
        df.withColumn("invoice_date", F.to_date("invoice_ts"))
        .withColumn("amount", F.round(F.col("quantity") * F.col("price"), 2))
        .withColumn("category", category_expr(F.col("description")))
        .dropDuplicates()  # the two yearly sheets overlap in December 2010
        .select("invoice", "invoice_ts", "invoice_date", "customer_id", "country", "stock_code",
                "description", "category", "quantity", "price", "amount", "is_return")
    )


def clean_report(raw: DataFrame, cleaned: DataFrame) -> dict:
    """Row counts before and after, so the notebook can show what was dropped and why."""
    df = standardise(raw)
    code = F.upper(F.trim(F.col("stock_code").cast("string")))
    row = df.agg(
        F.count("*").alias("raw_rows"),
        F.sum(F.col("customer_id").isNull().cast("int")).alias("no_customer"),
        F.sum((F.col("price") <= 0).cast("int")).alias("non_positive_price"),
        F.sum((code.isin(*NON_PRODUCT_CODES) | code.startswith("GIFT_")).cast("int")).alias("non_product"),
    ).first().asDict()
    c = cleaned.agg(
        F.count("*").alias("clean_rows"),
        F.sum(F.col("is_return").cast("int")).alias("return_rows"),
        F.countDistinct("customer_id").alias("customers"),
        F.avg((F.col("category") != "other").cast("double")).alias("category_coverage"),
    ).first().asDict()
    return {**row, **c}
