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

Not run on the real data yet.

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
