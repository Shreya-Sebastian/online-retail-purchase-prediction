# Databricks notebook source
# MAGIC %md
# MAGIC # 07 Figures
# MAGIC
# MAGIC Draws the figures in `docs/figures/` and saves them to the volume (`<volume>/figures`):
# MAGIC
# MAGIC - `segments_map.png`: training snapshots on two principal components, coloured by segment. The components are
# MAGIC   only for drawing; the segments were found on all six features.
# MAGIC - `segment_profiles.png`: median recency, orders, spend and returns per segment
# MAGIC - `rolling_auc.png`: ROC AUC in each of ten test months, the order-count rule against the boosted trees
# MAGIC - `lead_certainty.png`: how sure the trees' lead over the rule is (bootstrap intervals and five seeds)
# MAGIC
# MAGIC Reads `snapshots_segmented` and the round-4 tables in `docs/round4/`.

# COMMAND ----------

import os
import sys

sys.path.append(os.path.abspath("../src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from custprop.config import Config
from custprop.propensity import split_by_cutoff
from custprop.segment import SEGMENT_FEATURES

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "custprop")
cfg = Config(catalog=dbutils.widgets.get("catalog"), schema=dbutils.widgets.get("schema"))
OUT = f"{cfg.volume}/figures"
os.makedirs(OUT, exist_ok=True)
ROUND4 = os.path.abspath("../docs/round4")

# One look for every figure: recessive axes, thin marks, text in ink colours, never the series colour.
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA, RULE = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans",
})


def save(fig, name):
    fig.savefig(f"{OUT}/{name}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Segments

# COMMAND ----------

snaps = spark.table(cfg.table("snapshots_segmented")).orderBy("cutoff", "customer_id").toPandas()
train, _, _ = split_by_cutoff(snaps, cfg.test_cutoffs, cfg.valid_cutoffs)

# Name the segments from their profile, the same way the README does.
med = train.groupby("segment")[SEGMENT_FEATURES].median()
high_return = med["return_line_share"].idxmax()
active = med.drop(high_return)["orders_365d"].idxmax()
names = {active: "Active regulars", high_return: "High-return"}
names.update({s: "Occasional or lapsing" for s in med.index if s not in names})
ORDER = ["Active regulars", "Occasional or lapsing", "High-return"]
COLOUR = dict(zip(ORDER, [BLUE, ORANGE, AQUA]))
train["segment_name"] = train["segment"].map(names)
share = train["segment_name"].value_counts(normalize=True)

x = np.log1p(train[SEGMENT_FEATURES].astype(float).clip(lower=0).to_numpy())
xy = PCA(n_components=2, random_state=cfg.seed).fit_transform(StandardScaler().fit_transform(x))
sample = np.random.default_rng(cfg.seed).choice(len(train), size=min(4000, len(train)), replace=False)

fig, ax = plt.subplots(figsize=(6.4, 4.6))
for name in ORDER:  # largest first would hide the small group, so draw in fixed order with the small one last
    idx = sample[train["segment_name"].to_numpy()[sample] == name]
    ax.scatter(xy[idx, 0], xy[idx, 1], s=9, alpha=0.45, color=COLOUR[name], linewidths=0,
               label=f"{name} ({share[name]:.0%})")
ax.set_title("Customer segments (k-means, k = 3)", loc="left")
ax.set_xlabel("Component 1 (for drawing only)")
ax.set_ylabel("Component 2")
ax.legend(frameon=False, markerscale=2, loc="best")
save(fig, "segments_map.png")

# COMMAND ----------

panels = [("recency_days", "Days since last order", "{:.0f}"), ("orders_365d", "Orders in the past year", "{:.0f}"),
          ("spend_365d", "Spend in the past year (£)", "{:,.0f}"), ("return_line_share", "Share of lines returned", "{:.0%}")]
prof = train.groupby("segment_name")[[c for c, _, _ in panels]].median().reindex(ORDER)
fig, axes = plt.subplots(1, 4, figsize=(11, 2.6), sharey=True)
for ax, (col, title, fmt) in zip(axes, panels):
    vals = prof[col].to_numpy()
    ax.barh(ORDER, vals, color=[COLOUR[n] for n in ORDER], height=0.6)
    for i, v in enumerate(vals):
        ax.text(v, i, " " + fmt.format(v), va="center", fontsize=8, color=INK)
    ax.set_title(title, loc="left", fontsize=9)
    ax.set_xlim(0, vals.max() * 1.35 if vals.max() > 0 else 1)
    ax.grid(axis="y", visible=False)
    ax.set_xticks([])
axes[0].invert_yaxis()
fig.suptitle("What sets the segments apart (medians over training months)", x=0.01, ha="left",
             fontsize=11, fontweight="bold")
fig.tight_layout()
save(fig, "segment_profiles.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results over ten test months

# COMMAND ----------

rolling = pd.read_csv(f"{ROUND4}/rolling_10_months.csv", parse_dates=["test_cutoff"])
TARGETS = [("any", "Any purchase"), ("kitchen", "Kitchen"), ("bags", "Bags")]
MODELS = [("recent_orders", "Rule: recent orders", RULE), ("gbm_ext", "Boosted trees", BLUE)]
fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
for ax, (t, title) in zip(axes, TARGETS):
    means = []
    for m, label, colour in MODELS:
        d = rolling[(rolling.target == t) & (rolling.model == m)].sort_values("test_cutoff")
        ax.plot(d.test_cutoff, d.roc_auc, color=colour, linewidth=2, marker="o", markersize=4, label=label)
        means.append(d.roc_auc.mean())
    ax.set_title(title, loc="left")
    ax.text(0.02, 0.97, f"ten-month mean: rule {means[0]:.3f}, trees {means[1]:.3f}", transform=ax.transAxes,
            fontsize=8, color=INK2, va="top")
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b\n%Y"))
    ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator(bymonth=[12, 3, 6, 9]))
axes[0].set_ylim(top=0.885)  # room for the mean labels above the lines
axes[0].set_ylabel("ROC AUC on the test month")
axes[0].legend(frameon=False, loc="lower left")
fig.suptitle("The trees beat the rule in all ten test months, for every target",
             x=0.01, ha="left", fontsize=11, fontweight="bold")
fig.tight_layout()
save(fig, "rolling_auc.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ## How sure the lead is

# COMMAND ----------

lead = pd.read_csv(f"{ROUND4}/tuned_lead.csv").set_index("target")
seeds = pd.read_csv(f"{ROUND4}/seeds.csv")
fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.0), sharey=True)
ys = np.arange(len(TARGETS))[::-1]
for y, (t, title) in zip(ys, TARGETS):
    r = lead.loc[t]
    a.plot([r.ci_low, r.ci_high], [y, y], color=BLUE, linewidth=2)
    a.plot(r["diff"], y, "o", color=BLUE, markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5)
    a.annotate(f"+{r['diff']:.3f} ({r.ci_low:.3f} to {r.ci_high:.3f})", (r.ci_high, y), xytext=(6, 0),
               textcoords="offset points", va="center", fontsize=8, color=INK2)
    d = seeds[(seeds.target == t) & (seeds.model == "gbm_ext")]
    b.plot(d.lead_over_rule, np.full(len(d), y), "o", color=BLUE, alpha=0.7, markersize=7,
           markeredgecolor=SURFACE, markeredgewidth=1.5)
    b.annotate(f"{d.lead_over_rule.min():.3f} to {d.lead_over_rule.max():.3f}", (d.lead_over_rule.max(), y),
               xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=INK2)
for ax in (a, b):
    ax.axvline(0, color=INK2, linewidth=1)
    ax.set_xlim(-0.01, 0.16)
    ax.grid(axis="y", visible=False)
a.set_yticks(ys, [title for _, title in TARGETS])
a.set_title("Lead with a 95% bootstrap interval", loc="left")
a.set_xlabel("ROC AUC, trees minus rule (test month)")
b.set_title("Lead with five different seeds", loc="left")
b.set_xlabel("ROC AUC, trees minus rule (test month)")
fig.suptitle("The lead over the rule is above zero however it is checked", x=0.01, ha="left",
             fontsize=11, fontweight="bold")
fig.tight_layout()
save(fig, "lead_certainty.png")
