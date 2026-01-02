import numpy as np
import pandas as pd
import pytest

from risk_engine.portfolio import (
    make_equal_weight_portfolio,
    make_portfolio_from_dict,
    portfolio_returns,
    portfolio_pnl,
    Portfolio,
)


def test_equal_weight_sums_to_one():
    p = make_equal_weight_portfolio(["A", "B", "C", "D"])
    assert np.isclose(p.weights.sum(), 1.0)


def test_portfolio_returns_weighted_sum():
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    rets = pd.DataFrame({"A": [0.01, 0.02, -0.01], "B": [0.00, 0.01, 0.03]}, index=idx)
    p = make_portfolio_from_dict({"A": 0.6, "B": 0.4}, normalize=False)
    pr = portfolio_returns(rets, p)
    expected = 0.6 * rets["A"] + 0.4 * rets["B"]
    assert np.allclose(pr.values, expected.values)


def test_portfolio_pnl_equity_curve():
    idx = pd.date_range("2020-01-01", periods=2, freq="D")
    r = pd.Series([0.10, -0.05], index=idx, name="portfolio_return")
    out = portfolio_pnl(r, notional=100.0, initial_value=100.0)
    assert np.isclose(out.loc[idx[0], "pnl"], 10.0)
    assert np.isclose(out.loc[idx[1], "pnl"], -5.0)
    assert np.isclose(out.loc[idx[0], "equity"], 110.0)
    assert np.isclose(out.loc[idx[1], "equity"], 104.5)


def test_portfolio_returns_missing_ticker_raises():
    idx = pd.date_range("2020-01-01", periods=2, freq="D")
    rets = pd.DataFrame({"A": [0.0, 0.0]}, index=idx)
    p = make_portfolio_from_dict({"A": 0.5, "B": 0.5})
    with pytest.raises(ValueError):
        portfolio_returns(rets, p)
def test_portfolio_returns_zero_weights_gives_zero():
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    rets = pd.DataFrame({"A": [0.01, -0.02, 0.03]}, index=idx)
    p = Portfolio(weights=pd.Series({"A": 0.0}))
    pr = portfolio_returns(rets, p)
    assert np.allclose(pr.values, 0.0)


def test_portfolio_returns_rejects_nan_returns():
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    rets = pd.DataFrame({"A": [0.01, np.nan, 0.03]}, index=idx)
    p = make_portfolio_from_dict({"A": 1.0})
    with pytest.raises(ValueError):
        portfolio_returns(rets, p)


def test_make_portfolio_from_dict_normalizes():
    p = make_portfolio_from_dict({"A": 2.0, "B": 1.0}, normalize=True)
    assert np.isclose(p.weights.sum(), 1.0)
    assert np.isclose(p.weights["A"], 2.0 / 3.0)
    assert np.isclose(p.weights["B"], 1.0 / 3.0)


def test_make_portfolio_from_dict_rejects_negative():
    with pytest.raises(ValueError):
        make_portfolio_from_dict({"A": 1.0, "B": -0.1})


def test_portfolio_pnl_requires_positive_notional_and_initial_value():
    idx = pd.date_range("2020-01-01", periods=2, freq="D")
    r = pd.Series([0.01, 0.02], index=idx)
    with pytest.raises(ValueError):
        portfolio_pnl(r, notional=0.0, initial_value=100.0)
    with pytest.raises(ValueError):
        portfolio_pnl(r, notional=100.0, initial_value=0.0)


def test_portfolio_returns_column_order_irrelevant():
    idx = pd.date_range("2020-01-01", periods=2, freq="D")
    rets1 = pd.DataFrame({"A": [0.01, 0.02], "B": [0.03, 0.04]}, index=idx)
    rets2 = rets1[["B", "A"]]  # swapped
    p = make_portfolio_from_dict({"A": 0.6, "B": 0.4}, normalize=False)
    pr1 = portfolio_returns(rets1, p)
    pr2 = portfolio_returns(rets2, p)
    assert np.allclose(pr1.values, pr2.values)