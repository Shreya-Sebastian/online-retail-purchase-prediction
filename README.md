# online-retail-purchase-prediction

Customer segmentation and purchase prediction on Databricks, using two years of real
transactions from a UK online gift shop (Online Retail II, UCI): about a million rows and
5,875 customers after cleaning.

The aim is to rank customers by how likely they are to buy again in the next 90 days, or to
buy from a given product category (kitchen, bags), and to estimate how many will.

## Key findings

- **The models beat a simple order-count rule in every month tested.** Over four test months
  (June to September 2011), boosted trees reached a ROC AUC of 0.84 for bag purchases,
  against 0.75 for ranking customers by their recent orders. Average precision rose from
  0.50 to 0.69.
- **The largest gain is for category purchases, and it comes from purchase history in that
  category.** Shuffling a customer's category history costs the bags model more than any
  other feature group, while the order-count rule has no access to it.
- **The model adds most where customers have least history.** Among customers with a
  single order in the past year (about a third of the test month), the order-count rule
  can hardly rank anyone (ROC AUC 0.55 to 0.56). There the trees' lead over it is 1.5 to
  1.9 times as large as over all customers, reaching 0.16 ROC AUC for bags.
- **The probabilities can be used as they are.** After calibration on the validation
  month, the trees' expected calibration error was below 0.05 for every target, so the
  scores also estimate how many customers will buy.
- **One month of history is enough to train on.** Trees trained on the most recent month
  alone (about 4,300 customers) scored the same as trees trained on all 17 months, which
  keeps retraining cheap.
- **Three interpretable customer segments**, chosen from 14 candidate models: active
  regulars (40%), occasional or lapsing buyers (55%) and a small high-return group (5%).
- **It runs end to end as a Databricks job:** PySpark cleaning into Delta tables in Unity
  Catalog, MLflow tracking for every model, the best model per target registered in Unity
  Catalog, and tests for the data logic.

Detailed tables are in [docs/results.md](docs/results.md).

## How it was evaluated

These rules were set at the start and kept for every experiment:

- **No leakage.** Features for a cutoff use only the year before it, and labels only the 90
  days from the cutoff on. Tests check the boundary, including a purchase on the cutoff day
  itself.
- **Split by time, not at random.** Models train on earlier months, are chosen and
  calibrated on the next month, and are tested on the month after that, the way they would
  be used.
- **A baseline first.** Every model is compared with ranking customers by their recent
  orders, so any gain is measured against something a business could do without a model.
- **Both uses of a score are measured.** Ranking (ROC AUC, average precision, lift in the
  top 10%) for deciding who to contact, and calibration (Brier score, calibration error)
  for estimating how many will respond.
- **Choices are made on the validation month.** The test month is only used to report.

## How the work developed

The project went through three rounds, each set up by what the previous one showed.

1. **Baseline comparison.** Logistic regression and boosted trees against the order-count
   rule, for any purchase, kitchen and bags. The trees were best for every target, with a
   modest gain for any purchase and a large one for bags. Segments were compared as an
   extra feature and as one model per segment.
2. **Testing what the model might be missing.** Four groups of features were added that the
   segments don't contain: seasonality, regularity of orders, trend and category recency.
   The comparison was repeated over four test months instead of one, and per segment.
3. **Predictions written down before running.** Four further checks were set up where a
   difference was expected, with the expected outcome committed before the run: a learning
   curve, feature group importance, segments in a linear model, and customers with a single
   order. Three of the seven predicted outcomes held, and the rest changed the picture, as
   described below.

## What did not help, and what that shows

- **The extra features did not improve the trees.** Seasonality, regularity, trend and
  category recency changed their ROC AUC by at most 0.003 in any of twelve month and target
  combinations. Logistic regression gained a little (bags 0.806 to 0.814). The trees
  already got the same information from recency, frequency, spend and category shares.
- **A feature the model relies on is not necessarily one that adds value.** The extended
  trees rely on regularity more than on anything else for any purchase, yet adding it did
  not raise their score. The feature groups overlap, so when one is shuffled the model can
  partly fall back on another.
- **Segments describe customers but do not improve predictions.** Adding the segment
  changed no model's ROC AUC by more than 0.007, including logistic regression. The segments
  are built from the same features the models use. Within each segment the model is weakest
  for the occasional or lapsing buyers (ROC AUC 0.64 for any purchase), who have the least
  history to go on.
- **More history did not help.** This was predicted the other way, and is listed among the
  findings above.

## Limitations

- All test months fall within one year (June to September 2011, summer and early autumn),
  so the results can't show how the models hold up from one year to the next.
- Snapshots before December 2010 have less than a year of history, because the data starts
  in December 2009. Those months have fewer, more recently active customers.
- Categories come from keyword rules on item descriptions, and 28% of spend falls under
  "other".
- Customers without an id (23% of rows) are left out entirely.
- The explanations for why segments and extra features did not help are my reading of the
  results, not separately tested.

## Next steps

1. **Better categories.** Group products by their descriptions with text embeddings
   instead of keywords, to cover the 28% of spend now in "other".
2. **Product-level features for one-order customers,** where the model adds most and has
   least to go on: what was in their one basket.
3. **Batch scoring.** A scheduled job that scores current customers into a Delta table,
   with a check on how the score distribution moves from month to month.
4. **A second dataset,** to see whether the findings hold for another shop.

## Pipeline

Four notebooks, run in order. They were run one by one and then as one Databricks job set
up in the UI. `databricks.yml` defines the same job as an Asset Bundle, but that file has
not been deployed or tested.

| Notebook | Writes | What it does |
|---|---|---|
| `01_ingest` | `raw_transactions`, `transactions` | Downloads the data to a Unity Catalog volume, stacks both yearly sheets, cleans them into Delta tables |
| `02_snapshots` | `snapshots` | One row per customer per monthly cutoff: features from the year before, labels from the 90 days after |
| `03_segments` | `snapshots_segmented` | Compares k-means and a Gaussian mixture, fits the chosen one on training months, logs it to MLflow |
| `04_propensity` | MLflow runs, registered models | Trains and compares the models per target, runs the further checks and registers the best model per target in Unity Catalog |

The logic lives in `src/custprop` so it can be tested outside Databricks. The notebooks
only load data, call it, and log the results.

### Cleaning

The raw export mixes purchases with other things. Cancellations (invoice numbers starting
with C) are kept and flagged as returns, because the return rate is a feature. Rows without
a customer id, non-product codes (postage, bank charges, manual adjustments) and zero or
negative prices are dropped. The notebook reports how many rows each rule matches. The
counts overlap.

### Models

For each target the same comparison is run:

- **recent_orders**: no model, rank by number of orders in the last 90 days
- **logistic**, **gbm**: logistic regression on log-scaled features, and gradient-boosted
  trees, both on the base features (recency, frequency, spend, returns, category shares)
- **logistic_ext**, **gbm_ext**: the same two models on the extended features (seasonality,
  regularity, trend, category recency)
- **logistic_segment**, **gbm_segment**, **gbm_ext_segment**: with the segment as a feature
- **gbm_ext_per_segment**: one extended model per segment

The better of logistic regression and the trees, by average precision on the validation
month, is registered in Unity Catalog as `propensity_<target>`.

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

If the download in `01_ingest` is blocked, download the zip from the UCI page and upload
`online_retail_II.xlsx` to the `raw` volume by hand.

### Tests, locally

The tests run the Spark code on a small synthetic dataset (`custprop.synthetic`) with local
Spark, so they need Java 17.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

## Code layout

```
src/custprop/
  config.py      table names, windows, cutoffs, targets, chosen segmentation
  clean.py       raw export -> clean transactions
  categories.py  keyword categories from item descriptions
  snapshots.py   features and labels per customer per cutoff
  segment.py     k-means vs Gaussian mixture: comparison, fit, profiles
  propensity.py  time-based split, models, metrics, calibration, further checks
  synthetic.py   small synthetic shop for tests
notebooks/       the four Databricks notebooks
docs/results.md  detailed results
databricks.yml   the job, as a Databricks Asset Bundle
```
