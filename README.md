# online-retail-purchase-prediction

Customer segmentation and purchase prediction on Databricks, using two years of
transactions from a UK online gift shop (Online Retail II, UCI).

The questions:

- Which groups of customers are there, going by how recently, how often and how much they buy?
- Who is likely to buy again in the next 90 days, and who is likely to buy from a given
  product category (kitchen, bags)?
- Does knowing a customer's segment improve those predictions?

## Pipeline

Four notebooks, run in order as one Databricks job (`databricks.yml`):

| Notebook | Writes | What it does |
|---|---|---|
| `01_ingest` | `raw_transactions`, `transactions` | Downloads the data to a Unity Catalog volume, stacks both yearly sheets, cleans them into Delta tables |
| `02_snapshots` | `snapshots` | One row per customer per monthly cutoff: features from the year before, labels from the 90 days after |
| `03_segments` | `snapshots_segmented` | Compares k-means and a Gaussian mixture, fits the chosen one on training months, logs it to MLflow |
| `04_propensity` | MLflow runs, registered models | Trains and compares the propensity models per target and registers the better of logistic regression and boosted trees in Unity Catalog |

The logic lives in `src/custprop` so it can be tested outside Databricks. The
notebooks only load data, call it, and log the results.

### Cleaning

The raw export mixes purchases with other things. Cancellations (invoice numbers
starting with C) are kept and flagged as returns, because the return rate is a
feature. Rows without a customer id, non-product codes (postage, bank charges,
manual adjustments) and zero prices are dropped. The notebook reports how many rows
each rule removes.

The data has no product hierarchy, so categories come from keyword rules on the item
description. The notebook shows how much of the spend they cover.

### Snapshots and leakage

Features for a cutoff only use transactions before that date, and labels only use the
90 days from the cutoff on. A purchase on the cutoff day itself counts as the future.
The tests check this boundary. The training, validation and test sets are split by
cutoff month, not at random. Otherwise the model would train on a customer's later
behaviour and be tested on their earlier behaviour, which flatters it.

### Models and evaluation

For each target the same comparison is run:

- **recent_orders**: no model, rank by number of orders in the last 90 days
- **logistic**: logistic regression on log-scaled features
- **gbm**: gradient-boosted trees
- **gbm_segment**: gradient-boosted trees with the segment as a feature
- **gbm_per_segment**: one model per segment

A propensity score is used to rank customers (who to contact first) and as a
probability (how many will respond), so both are measured on the test month:

- ranking: ROC AUC, average precision, lift in the top 10%
- probabilities: Brier score and expected calibration error, before and after
  isotonic calibration fitted on the validation month

## Results

From the pipeline on Databricks Free Edition (serverless), on the full dataset. The
notebooks were first run one by one, then end to end as one Databricks job (four tasks,
7 minutes 41 seconds).

### Data

| | |
|---|---|
| Raw rows (both yearly sheets) | 1,067,371 |
| No customer id | 243,007 |
| Zero or negative price | 6,207 |
| Non-product codes | 5,931 |
| Clean rows (returns included) | 794,165 (17,586 returns) |
| Customers | 5,875 |

The dropped counts overlap, and exact duplicates are removed as well. About 23% of rows
have no customer id. They can't be followed over time, so everything below is about
customers who ordered while logged in. The keyword categories cover 72% of spend.

19 monthly snapshots (March 2010 to September 2011) have 1,700 to 4,300 active customers
each. The share who buy again within 90 days peaks in September (0.58 in 2010) and is
lowest after Christmas (0.33). The model trains on cutoffs up to July 2011, is calibrated
and chosen on August 2011, and is tested on September 2011.

### Segments

k-means with k from 2 to 8 and a Gaussian mixture were compared on the training months.
k-means with k = 3 had the highest silhouette of all 14 options (0.32, against 0.23 for
k = 4). It gave the same segments for every seed (adjusted Rand index 1.00), and no
segment was under 5%. The Gaussian mixture separated less well at every k and was unstable
at k = 3. Its BIC kept improving as k grew, which says little with this many rows.

| Segment | Share | Last order | Orders / year | Spend / year | Returns |
|---|---|---|---|---|---|
| Active regulars | 40% | 31 days ago | 5 | €1,722 | 1% of lines |
| Occasional or lapsing | 55% | 114 days ago | 1 | €316 | 0% |
| High-return | 5% | 105 days ago | 2 | €379 | 20% of lines |

(medians per segment)

### Propensity models (test month: September 2011)

| Target | Base rate | Model | ROC AUC | Avg. precision | Lift, top 10% | Calibration error (raw → isotonic) |
|---|---|---|---|---|---|---|
| any purchase | 0.50 | recent orders (no model) | 0.747 | 0.749 | 1.84 | (not a probability) → 0.073 |
| | | logistic regression | 0.764 | 0.782 | 1.91 | 0.070 → 0.065 |
| | | boosted trees | **0.772** | **0.792** | **1.93** | 0.057 → 0.042 |
| kitchen | 0.43 | recent orders | 0.727 | 0.671 | 1.94 | → 0.065 |
| | | logistic regression | 0.767 | 0.732 | 2.10 | 0.054 → 0.060 |
| | | boosted trees | **0.778** | **0.750** | **2.17** | 0.055 → 0.047 |
| bags | 0.28 | recent orders | 0.715 | 0.507 | 2.28 | → 0.047 |
| | | logistic regression | 0.806 | 0.660 | 2.86 | 0.034 → 0.041 |
| | | boosted trees | **0.827** | **0.702** | **3.03** | 0.041 → 0.024 |

What the numbers say:

- **Whether a customer buys again is mostly explained by recent activity.** Counting
  recent orders already reaches an AUC of 0.75, and boosted trees add 0.025.
- **The models help most for category purchases.** For bags, the trees reach
  0.83 against 0.72 for the baseline, and average precision rises from 0.51 to 0.70. What
  a customer bought before (their share of spend per category) matters, and order counts
  don't capture it. Kitchen items make up so much of the range that predicting them is
  close to predicting any purchase.
- **Segments did not improve the predictions.** Adding the segment as a feature changed
  nothing (any purchase: 0.771 against 0.772). One model per segment looked best on the
  validation month (0.819) but was slightly worse on the test month (0.768). The segments
  are built from the same features the trees already use, so they add no information. They
  are more useful for describing customers than for predicting their purchases.
- **The trees' probabilities were already close to calibrated.** Isotonic calibration on
  the validation month lowered the trees' calibration error for all three targets. For
  logistic regression it helped only for any purchase and made kitchen and bags slightly
  worse. The baseline needs it, because order counts are not probabilities.

For each target, notebook 04 registers the better of logistic regression and boosted
trees in Unity Catalog as `propensity_<target>`, choosing by average precision on the
validation month.

### Limitations

- One test month. September is in the pre-Christmas season, and validation AUC (August)
  was higher than test AUC by 0.03 to 0.04. A rolling evaluation over several months would
  give a better estimate.
- Snapshots before December 2010 have less than a year of history, because the data
  starts in December 2009. Those months have fewer and more recently active customers,
  and their year-long features are cut short.
- Categories come from keywords in item descriptions. 28% of spend is "other".
- Customers without an id (23% of rows) are left out entirely.

## Running it

### On Databricks (Free Edition)

1. Create a Free Edition account at databricks.com and open the workspace.
2. Workspace > Create > Git folder, and paste this repository's URL.
3. Run the notebooks in order, or deploy the job with the Databricks CLI:

   ```bash
   databricks bundle deploy
   databricks bundle run custprop_pipeline
   ```

   Set your workspace URL in `databricks.yml` first.

Everything is written to the `workspace.custprop` schema. The catalog and schema are
notebook widgets and job parameters.

If the download in `01_ingest` is blocked, download the zip from the UCI page and
upload `online_retail_II.xlsx` to the `raw` volume by hand.

### Tests, locally

The tests run the Spark code on a small synthetic dataset (`custprop.synthetic`) with
local Spark, so they need Java 17.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scriptsctivate
pip install -e ".[dev]"
pytest
```

## Code layout

```
src/custprop/
  config.py      table names, windows, cutoffs, targets
  clean.py       raw export -> clean transactions
  categories.py  keyword categories from item descriptions
  snapshots.py   features and labels per customer per cutoff
  segment.py     k-means vs Gaussian mixture: comparison, fit, profiles
  propensity.py  time-based split, models, metrics, calibration
  synthetic.py   small synthetic shop for tests
notebooks/       the four Databricks notebooks
databricks.yml   the job, as a Databricks Asset Bundle
```
