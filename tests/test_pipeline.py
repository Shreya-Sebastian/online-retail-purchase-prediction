from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from custprop import propensity, segment
from custprop.categories import category_of
from custprop.clean import clean, clean_report, prepare_raw_pandas
from custprop.config import monthly_cutoffs
from custprop.snapshots import FEATURE_COLUMNS, cutoffs_frame, features, labels, snapshots
from custprop.synthetic import raw_transactions


@pytest.mark.parametrize("desc, cat", [
    ("WHITE HANGING HEART T-LIGHT HOLDER", "lighting"),
    ("LUNCH BAG RED RETROSPOT", "bags"),
    ("PAPER CHAIN KIT 50'S CHRISTMAS", "christmas"),  # christmas wins over stationery
    ("REGENCY CAKESTAND 3 TIER", "kitchen"),
    ("POPCORN HOLDER", "other"),
    (None, "other"),
])
def test_categories(desc, cat):
    assert category_of(desc) == cat


def test_monthly_cutoffs_cross_year():
    assert monthly_cutoffs(date(2010, 11, 1), date(2011, 2, 1)) == [
        date(2010, 11, 1), date(2010, 12, 1), date(2011, 1, 1), date(2011, 2, 1)]


@pytest.fixture(scope="module")
def raw_pdf():
    return raw_transactions(customers=150)


@pytest.fixture(scope="module")
def txns(spark, raw_pdf):
    return clean(spark.createDataFrame(raw_pdf))


def test_clean_drops_noise_and_flags_returns(spark, raw_pdf, txns):
    rows = txns.toPandas()
    assert "POST" not in set(rows["stock_code"])
    assert rows["customer_id"].notna().all() and (rows["price"] > 0).all()
    assert rows.loc[rows["invoice"].str.startswith("C"), "is_return"].all()
    assert rows["customer_id"].iloc[0].isdigit()  # 12346.0 -> "12346"
    report = clean_report(spark.createDataFrame(raw_pdf), txns)
    assert report["no_customer"] == 1 and report["non_positive_price"] == 1 and report["non_product"] == 1


def _tiny(spark):
    rows = [  # customer A buys twice before the cutoff and once after; B only before
        ("1", datetime(2011, 1, 10), "A", "LUNCH BAG RED", 2, 1.65, False),
        ("2", datetime(2011, 2, 20), "A", "CAKE TIN", 1, 4.95, False),
        ("C2", datetime(2011, 2, 25), "A", "CAKE TIN", -1, 4.95, True),
        ("3", datetime(2011, 3, 15), "A", "JUMBO BAG", 1, 1.95, False),
        ("4", datetime(2010, 12, 1), "B", "CAKE TIN", 3, 4.95, False),
        ("5", datetime(2011, 3, 1), "B", "CAKE TIN", 1, 4.95, False),  # on the cutoff day: label, not feature
    ]
    pdf = pd.DataFrame(rows, columns=["invoice", "invoice_ts", "customer_id", "description", "quantity", "price", "is_return"])
    pdf["invoice_date"] = pdf["invoice_ts"].dt.date
    pdf["stock_code"] = pdf["description"]
    pdf["country"] = "United Kingdom"
    pdf["amount"] = pdf["quantity"] * pdf["price"]
    pdf["category"] = pdf["description"].map(category_of)
    return spark.createDataFrame(pdf)


def test_features_use_only_history(spark):
    t = _tiny(spark)
    f = features(t, cutoffs_frame(spark, [date(2011, 3, 1)])).toPandas().set_index("customer_id")
    a, b = f.loc["A"], f.loc["B"]
    assert a["recency_days"] == 9 and a["orders_365d"] == 2  # the March purchase is in the future
    assert a["orders_30d"] == 1 and round(a["spend_365d"], 2) == 8.25
    assert a["return_line_share"] == pytest.approx(1 / 3)
    assert round(a["share_bags"], 3) == round(3.30 / 8.25, 3)
    assert b["recency_days"] == 90 and b["orders_365d"] == 1  # the purchase on the cutoff day is not history


def test_labels_use_only_the_window_after(spark):
    t = _tiny(spark)
    y = labels(t, cutoffs_frame(spark, [date(2011, 3, 1)]), 90, ("any", "bags", "kitchen")).toPandas().set_index("customer_id")
    assert y.loc["A", "y_any"] == 1 and y.loc["A", "y_bags"] == 1 and y.loc["A", "y_kitchen"] == 0
    assert y.loc["B", "y_any"] == 1 and y.loc["B", "y_kitchen"] == 1  # bought on the cutoff day itself
    y_short = labels(t, cutoffs_frame(spark, [date(2011, 1, 1)]), 30).toPandas().set_index("customer_id")
    assert list(y_short.index) == ["A"]


def test_end_to_end(spark, txns):
    cut = monthly_cutoffs(date(2010, 9, 1), date(2011, 6, 1))
    snaps = snapshots(txns, cutoffs_frame(spark, cut), 365, 90, ("any", "kitchen")).toPandas()
    assert not snaps[FEATURE_COLUMNS].isna().any().any()
    assert snaps.groupby(["cutoff", "customer_id"]).size().max() == 1

    train, valid, test = propensity.split_by_cutoff(snaps)
    assert train["cutoff"].max() < valid["cutoff"].min() < test["cutoff"].min()

    table = segment.compare(train, ks=range(2, 4))
    assert set(table["method"]) == {"kmeans", "gmm"} and table["silhouette"].notna().all()
    seg = segment.fit(train, "kmeans", 3)
    for d in (train, valid, test):
        d["segment"] = seg.predict(d)
    assert segment.profile(train, train["segment"].to_numpy())["share"].sum() == pytest.approx(1)

    results = propensity.compare_models(train, valid, test, "any")
    names = [r.name for r in results]
    assert names == ["recent_orders", "logistic", "gbm", "gbm_segment", "gbm_per_segment"]
    out = propensity.results_table(results)
    assert out["test_roc_auc"].between(0, 1).all()
    assert out.loc[out["model"] == "gbm", "test_roc_auc"].item() > 0.6  # there is signal to find


def test_metrics_on_known_values():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    m = propensity.metrics(y, p)
    assert m["roc_auc"] == 1.0 and m["base_rate"] == 0.5
    assert propensity.lift_at(y, p, 0.5) == 2.0
    assert propensity.expected_calibration_error(np.array([0, 1]), np.array([0.0, 1.0]), bins=2) == 0.0


def test_split_needs_enough_cutoffs():
    df = pd.DataFrame({"cutoff": [date(2011, 1, 1), date(2011, 2, 1)]})
    with pytest.raises(ValueError):
        propensity.split_by_cutoff(df)


def test_mixed_type_text_columns_reach_spark(spark, raw_pdf):
    # as in the real export: a numeric description, an empty one, a numeric stock code
    pdf = raw_pdf.head(5).copy()
    pdf["Description"] = pd.Series(["WHITE MUG", 21258, np.nan, 3.5, "BAG"], dtype=object)
    pdf["StockCode"] = pd.Series(["85123A", 22423, "POST", 22423, "20725"], dtype=object)
    with pytest.raises(Exception):
        spark.createDataFrame(pdf)  # fails without prepare_raw_pandas, as on Databricks
    rows = spark.createDataFrame(prepare_raw_pandas(pdf)).toPandas()
    assert rows["description"].tolist()[:3] == ["WHITE MUG", "21258", None]
    assert rows["stock_code"].tolist()[1] == "22423"


def test_prepared_export_has_delta_safe_names_and_cleans_the_same(spark, raw_pdf, txns):
    prepared = spark.createDataFrame(prepare_raw_pandas(raw_pdf))
    assert all(" " not in c for c in prepared.columns)  # Delta rejects "Customer ID"
    assert clean(prepared).count() == txns.count()
    report = clean_report(prepared, txns)
    assert report["no_customer"] == 1 and report["non_product"] == 1
