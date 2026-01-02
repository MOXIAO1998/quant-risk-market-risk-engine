import numpy as np
import pandas as pd
import pytest

from risk_engine.backtest import (
    compute_var_hits,
    kupiec_pof_test,
    christoffersen_independence_test,
    christoffersen_conditional_coverage_test,
)


def test_compute_var_hits_loss_positive():
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    r = pd.Series([0.0, -0.01, -0.03, 0.02, -0.02], index=idx)
    var = pd.Series([np.nan, 0.02, 0.02, 0.02, 0.02], index=idx)  # first nan dropped

    df = compute_var_hits(r, var, loss_positive=True)
    assert df.shape[0] == 4
    # violation if r < -0.02
    expected = [0, 1, 0, 0]  # dates 2..5: -0.01 no, -0.03 yes, 0.02 no, -0.02 no (strict <)
    assert df["hit"].tolist() == expected


def test_compute_var_hits_return_quantile_mode():
    idx = pd.date_range("2020-01-01", periods=4, freq="D")
    r = pd.Series([-0.01, -0.03, 0.01, -0.02], index=idx)
    q = pd.Series([-0.02, -0.02, -0.02, -0.02], index=idx)  # return-quantile threshold
    df = compute_var_hits(r, q, loss_positive=False)
    # violation if r < q (strict)
    assert df["hit"].tolist() == [0, 1, 0, 0]


def test_kupiec_lr_zero_when_violation_rate_matches_expected():
    # If x/n == p, LR should be ~0.
    # alpha=0.95 => p=0.05. Choose n=100, x=5.
    hits = pd.Series([1] * 5 + [0] * 95)
    lr, _ = kupiec_pof_test(hits, alpha=0.95)
    assert np.isclose(lr, 0.0, atol=1e-12)


def test_kupiec_lr_positive_when_mismatch():
    hits = pd.Series([1] * 10 + [0] * 90)  # x=10, n=100, p=0.05 mismatch
    lr, _ = kupiec_pof_test(hits, alpha=0.95)
    assert lr > 0.0


def test_christoffersen_independence_lr_zero_when_markov_equals_iid():
    # Construct hits with N00=N01=N10=N11=2 (pi0=pi1=pi=0.5) -> LR_ind = 0
    # Sequence length 9 gives 8 transitions:
    # 0->0,0->1,1->1,1->0 repeated twice
    hits = pd.Series([0, 0, 1, 1, 0, 0, 1, 1, 0], dtype=int)
    lr, _ = christoffersen_independence_test(hits)
    assert np.isclose(lr, 0.0, atol=1e-12)


def test_christoffersen_independence_degenerate_returns_nan():
    # Never visits state 1 -> transitions from 1 undefined
    hits = pd.Series([0] * 50, dtype=int)
    lr, p = christoffersen_independence_test(hits)
    assert np.isnan(lr)
    assert p is None


def test_conditional_coverage_is_sum_of_components_when_defined():
    hits = pd.Series([0, 0, 1, 1, 0, 0, 1, 1, 0], dtype=int)
    lr_pof, _ = kupiec_pof_test(hits, alpha=0.5)  # p=0.5 matches, lr_pof=0
    lr_ind, _ = christoffersen_independence_test(hits)  # lr_ind=0
    lr_cc, _ = christoffersen_conditional_coverage_test(hits, alpha=0.5)
    assert np.isclose(lr_cc, lr_pof + lr_ind, atol=1e-12)
