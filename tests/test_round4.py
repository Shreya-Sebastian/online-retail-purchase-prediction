import numpy as np
import pandas as pd
import pytest

from custprop.propensity import (GBM_DEFAULTS, _gbm, rolling_evaluation, seed_sensitivity, split_by_cutoff,
                                 tune_models, tuning_settings)
from custprop.snapshots import EXTENDED_COLUMNS


@pytest.fixture(scope="module")
def snaps():
    """A small random snapshot table with the real columns: 12 monthly cutoffs, where buying
    again depends on recent orders, so the models have something to find."""
    rng = np.random.default_rng(0)
    rows = []
    for k, cutoff in enumerate(pd.date_range("2010-03-01", periods=12, freq="MS")):
        n = 300
        df = pd.DataFrame({c: rng.gamma(2.0, 2.0, n) for c in EXTENDED_COLUMNS})
        df["orders_90d"] = rng.poisson(2, n)
        df["orders_365d"] = df["orders_90d"] + rng.poisson(4, n)
        logit = -1 + 0.6 * df["orders_90d"] - 0.1 * df[EXTENDED_COLUMNS[0]]
        for t in ("any", "kitchen", "bags"):
            df[f"y_{t}"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
        df["cutoff"] = cutoff.date()
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def test_defaults_unchanged():
    # rounds 1 to 3 must reproduce: the default trees are the same as before round 4
    m = _gbm(7)
    assert (m.max_iter, m.learning_rate, m.max_leaf_nodes, m.l2_regularization, m.min_samples_leaf) == \
        (300, 0.05, 31, 1.0, 20)
    assert GBM_DEFAULTS["min_samples_leaf"] == 20  # scikit-learn's own default


def test_tuning_settings_are_fixed_and_shared():
    assert tuning_settings(5) == tuning_settings(5)
    assert len(tuning_settings(20)) == 20


def test_seed_sensitivity_shape(snaps):
    train, _, test = split_by_cutoff(snaps, 1, 1)
    out = seed_sensitivity(train, test, "any", seeds=(7, 11))
    assert set(out["model"]) == {"gbm", "gbm_ext"} and set(out["seed"]) == {7, 11}
    assert out["roc_auc"].between(0, 1).all()


def test_tune_models_chooses_on_validation_only(snaps):
    train, valid, test = split_by_cutoff(snaps, 1, 1)
    cand, summary, lead = tune_models(train, valid, test, "any", n_settings=3, n_boot=50)
    assert set(summary["model"]) == {"logistic", "logistic_ext", "gbm", "gbm_ext"}
    for _, row in summary.iterrows():
        pool = cand[(cand.model == row.model)]
        # the kept setting has the best validation average precision, whatever its test score
        assert row.tuned_valid_avg_precision == pool.valid_avg_precision.max()
    assert lead.loc[0, "against"] == "recent_orders"


def test_rolling_ten_months(snaps):
    out = rolling_evaluation(snaps, "any", months=10)
    assert out["test_cutoff"].nunique() == 10
