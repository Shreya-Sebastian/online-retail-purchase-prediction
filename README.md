# online-retail-purchase-prediction

Customer segmentation and purchase prediction on Databricks, using two years of
transactions from a UK online gift shop (Online Retail II, UCI).

The questions:

- Which groups of customers are there, going by how recently, how often and how much they buy?
- Who is likely to buy again in the next 90 days, and who is likely to buy from a given
  product category (kitchen, bags)?
- Does knowing a customer's segment improve those predictions?

## Pipeline

Four notebooks, run in order. They were run one by one and then as one Databricks job set up in
the UI. `databricks.yml` defines the same job as an Asset Bundle, but that file has not been
deployed or tested.

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
manual adjustments) and zero or negative prices are dropped. The notebook reports how many
rows each rule matches. The counts overlap.

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
- **logistic**, **gbm**: logistic regression on log-scaled features, and gradient-boosted
  trees, both on the base features (recency, frequency, spend, returns, category shares)
- **logistic_ext**, **gbm_ext**: the same two models on the extended features (below)
- **logistic_segment**, **gbm_segment**: base features plus the segment
- **gbm_ext_segment**: extended trees with the segment as a feature
- **gbm_ext_per_segment**: one extended model per segment

The extended features add information the segments don't contain:

- seasonality: orders and spend in the same 90-day window one year earlier
- regularity: the mean gap between order days, and how much the gaps vary
- trend: orders and spend in the 90 days before the last 90
- category recency: days since the last purchase in each category

All of them come from the year before the cutoff, so they don't leak future data.

More checks use the same data:

- **within each segment**: how well the chosen model ranks customers inside each segment
- **one-order customers**: every model on the test-month customers with a single order in
  the past year
- **rolling evaluation**: the baseline and both tree models, with each of the last four
  months as the test month, to see how much the scores move from month to month
- **learning curve**: the trees trained on only the most recent 1, 2, 4, 8 or all
  training months
- **feature group importance**: the drop in test ROC AUC when one group of features is
  shuffled across customers, for the extended trees

A propensity score is used to rank customers (who to contact first) and as a
probability (how many will respond), so both are measured on the test month:

- ranking: ROC AUC, average precision, lift in the top 10%
- probabilities: Brier score and expected calibration error, before and after
  isotonic calibration fitted on the validation month

## Results

### Third version: predictions written before the run

In the second version the new features made no difference to the trees. These four checks are set up where a difference is expected, with the expected outcome
written here and committed before running them. Each will be reported as supported or not.

1. **Learning curve.** For every target, the trees trained on all training months score
   at least 0.01 ROC AUC higher than trees trained on the most recent month only. The gain
   from 8 months to all months is smaller than the gain from 1 month to 2.
2. **Feature group importance** (extended trees, test month). For any purchase, the
   largest drop in ROC AUC comes from shuffling recency or frequency. For bags, category
   shares or category recency is among the two largest drops. If so, this supports the
   explanation for the bags result in the first version, which so far is only a likely
   reason.
3. **Segments in a linear model.** Adding the segment raises logistic regression's test ROC
   AUC for every target, and changes the base trees' ROC AUC by less than 0.005. A linear
   model can't find these groups itself, and the trees can.
4. **One-order customers.** On test-month customers with one order in the past year, the
   gap between the base trees and the order-count baseline is larger than on all customers,
   for every target. The extended features help these customers more than they help all
   customers, for at least two of the three targets.

### Second version: extended features

The plan, fixed before the run:

- The new features were chosen from what the first version's results suggested was
  missing, not by trying features against the test month.
- The model to register is chosen on the validation month (August 2011), as before.
- The test month (September 2011) is looked at once, and the result is reported here
  whether the new features help or not.
- The first version's results below stay as they are, for comparison.

**The new features did not make the trees better.** Seasonality, regularity, trend and
category recency added nothing the trees could not already get from recency, frequency,
spend and category shares. Logistic regression gained a little, and the trees'
probabilities came out somewhat better calibrated.

Test month, September 2011:

| Target | Model | Base features: ROC AUC / avg. precision / calibration error | Extended features |
|---|---|---|---|
| any purchase | logistic regression | 0.764 / 0.782 / 0.070 | 0.767 / 0.785 / 0.065 |
| | boosted trees | 0.770 / 0.791 / 0.057 | 0.767 / 0.790 / 0.048 |
| kitchen | logistic regression | 0.767 / 0.732 / 0.054 | 0.769 / 0.737 / 0.050 |
| | boosted trees | 0.778 / 0.751 / 0.044 | 0.778 / 0.751 / 0.034 |
| bags | logistic regression | 0.806 / 0.660 / 0.034 | 0.814 / 0.676 / 0.033 |
| | boosted trees | 0.826 / 0.699 / 0.037 | 0.825 / 0.702 / 0.022 |

(calibration error is before isotonic calibration)

The segment variants, now on the extended features, again scored the same as the plain
trees, within 0.007 ROC AUC. On the validation month the extended trees were chosen for any
purchase and kitchen, and the base trees for bags.

**Rolling evaluation.** The same comparison with each of June, July, August and September
2011 as the test month. ROC AUC, mean and range over the four months:

| Target | Recent orders (no model) | Boosted trees, base | Boosted trees, extended |
|---|---|---|---|
| any purchase | 0.771 (0.747 to 0.783) | 0.794 (0.770 to 0.807) | 0.794 (0.767 to 0.810) |
| kitchen | 0.761 (0.727 to 0.784) | 0.796 (0.778 to 0.810) | 0.797 (0.778 to 0.812) |
| bags | 0.753 (0.715 to 0.776) | 0.842 (0.826 to 0.852) | 0.842 (0.825 to 0.856) |

- The trees beat the baseline in every month. The margin was 0.02 for any purchase, 0.03
  to 0.05 for kitchen, and 0.08 to 0.11 for bags.
- The difference between base and extended features was never more than 0.003 in either
  direction.
- September was the hardest of the four months for every model. The single-month test in
  the first version was a little pessimistic, not flattering.

**Within each segment** (September 2011, the model chosen for each target):

| Segment | Customers | Bought again (any) | ROC AUC: any / kitchen / bags |
|---|---|---|---|
| Active regulars | 1,683 | 0.75 | 0.75 / 0.75 / 0.81 |
| Occasional or lapsing | 2,462 | 0.34 | 0.64 / 0.65 / 0.73 |
| High-return | 179 | 0.32 | 0.67 / 0.73 / 0.80 |

Ranking customers within a segment is harder than ranking all of them together. Part of
the overall ROC AUC comes from telling the groups apart. The model is weakest inside the
largest group, the occasional or lapsing buyers, probably because most of them have only
one or two orders to go on. The high-return group is small (179 customers), so its numbers
are rough.

Two fixes came out of this run. Spark returns rows in no fixed order, so the trees came out
slightly different on a rerun (ROC AUC 0.770 against 0.772 for any purchase), and k-means
could give the same segments different numbers. The notebooks now sort rows before
modelling, and segments are numbered by recency, most recent first.

### First version: base features

From the pipeline on Databricks Free Edition (serverless), on the full dataset. The
notebooks were first run one by one, then end to end as one Databricks job (four tasks,
7 minutes 41 seconds).

#### Data

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
each. Apart from the earliest months (see Limitations), the share who buy again within 90
days is highest for the September cutoffs (0.58 in 2010, 0.50 in 2011) and lowest after
Christmas (0.33). The model trains on cutoffs up to July 2011, is calibrated
and chosen on August 2011, and is tested on September 2011.

#### Segments

k-means with k from 2 to 8 and a Gaussian mixture were compared on the training months.
k-means with k = 3 had the highest silhouette of all 14 options (0.32, against 0.23 for
k = 4). Two random seeds gave the same segments (adjusted Rand index 1.00), and no
segment was under 5%. The Gaussian mixture separated less well at every k and was unstable
at k = 3. Its BIC kept improving as k grew, which says little with this many rows.

| Segment | Share | Last order | Orders / year | Spend / year | Returns |
|---|---|---|---|---|---|
| Active regulars | 40% | 31 days ago | 5 | €1,722 | 1% of lines |
| Occasional or lapsing | 55% | 114 days ago | 1 | €316 | 0% |
| High-return | 5% | 105 days ago | 2 | €379 | 20% of lines |

(shares and medians are over training snapshots, so a customer counts once per month they appear in)

#### Propensity models (test month: September 2011)

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
  0.83 against 0.72 for the baseline, and average precision rises from 0.51 to 0.70. A
  likely reason is that the models see what a customer bought before (their share of spend
  per category), while the baseline only counts orders. I did not test this directly.
  Kitchen purchases are almost as common as any purchase (0.43 against 0.50 in the test
  month), so that target behaves much like the first one.
- **Segments did not improve the predictions.** Adding the segment as a feature changed
  nothing (any purchase: 0.771 against 0.772). One model per segment looked best on the
  validation month (0.819) but was slightly worse on the test month (0.768). The segments
  are built from features the trees already use, which is probably why they add nothing
  here. They
  are more useful for describing customers than for predicting their purchases.
- **The trees' probabilities were already close to calibrated.** Isotonic calibration on
  the validation month lowered the trees' calibration error for all three targets. For
  logistic regression it helped only for any purchase and made kitchen and bags slightly
  worse. The baseline needs it, because order counts are not probabilities.

For each target, notebook 04 registers the better of logistic regression and boosted
trees in Unity Catalog as `propensity_<target>`, choosing by average precision on the
validation month.

### Limitations

- The first version used one test month. September is in the pre-Christmas season, and
  validation AUC (August) was higher than test AUC by 0.015 to 0.05, depending on the
  model. The rolling evaluation in the second version covers four months, all from one
  year, so it still can't show year-to-year changes.
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
source .venv/bin/activate        # Windows: .venv\Scripts\activate
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
