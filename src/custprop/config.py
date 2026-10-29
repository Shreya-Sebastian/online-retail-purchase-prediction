"""Settings shared by the notebooks, the job and the tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

# Online Retail II (UCI): two years of transactions from a UK online gift shop.
DATA_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"


@dataclass(frozen=True)
class Config:
    catalog: str = "workspace"  # the Free Edition's default catalog
    schema: str = "custprop"
    history_days: int = 365  # features look this far back from a cutoff
    label_days: int = 90  # a "purchase" label means buying within this window after the cutoff
    first_cutoff: date = date(2010, 3, 1)  # leaves a few months of history before the first snapshot
    last_cutoff: date = date(2011, 9, 1)  # last cutoff whose label window fits before the data ends
    # months scored by the batch job: after the test month, so their label windows aren't complete yet,
    # as for a model scoring the current month
    score_cutoffs: tuple[date, ...] = (date(2011, 10, 1), date(2011, 11, 1), date(2011, 12, 1))
    test_cutoffs: int = 1  # latest cutoffs held out for testing
    valid_cutoffs: int = 1  # the cutoffs before those, for calibration and model choice
    targets: tuple[str, ...] = ("any", "kitchen", "bags")
    # categories the next-best-category suggestion chooses between; "other" is not something to suggest
    recommend_categories: tuple[str, ...] = ("christmas", "bags", "kitchen", "lighting", "stationery", "decor", "toys")
    # Chosen from the comparison in 03_segments on the real data: k-means with k=3 had the highest
    # silhouette (0.32) and near-perfect stability across seeds (ARI 1.00), with no segment under 5%.
    segment_method: str = "kmeans"
    segment_k: int = 3
    seed: int = 7
    extra: dict = field(default_factory=dict)

    @property
    def label_targets(self) -> tuple[str, ...]:
        """Every target a snapshot needs a label for: the propensity targets and each suggestible category."""
        return tuple(dict.fromkeys(self.targets + self.recommend_categories))

    def table(self, name: str) -> str:
        return f"{self.catalog}.{self.schema}.{name}"

    @property
    def volume(self) -> str:
        return f"/Volumes/{self.catalog}/{self.schema}/raw"


def monthly_cutoffs(first: date, last: date) -> list[date]:
    out, d = [], first
    while d <= last:
        out.append(d)
        d = date(d.year + d.month // 12, d.month % 12 + 1, 1)
    return out
