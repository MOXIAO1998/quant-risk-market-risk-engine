import numpy as np
import pandas as pd
import pytest

from risk_engine.data import align_prices, compute_returns,_ensure_datetime_index


def test_compute_returns_simple_basic():
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    prices = pd.DataFrame({"A": [100.0, 110.0, 121.0]}, index=idx)
    rets = compute_returns(prices, method="simple")
    assert rets.shape == (2, 1)
    assert np.isclose(rets.iloc[0, 0], 0.10)
    assert np.isclose(rets.iloc[1, 0], 0.10)


def test_compute_returns_log_basic():
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    prices = pd.DataFrame({"A": [100.0, 110.0, 121.0]}, index=idx)
    rets = compute_returns(prices, method="log")
    assert np.isclose(rets.iloc[0, 0], np.log(110 / 100))
    assert np.isclose(rets.iloc[1, 0], np.log(121 / 110))


def test_align_prices_inner_drops_nans():
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    prices = pd.DataFrame({"A": [1.0, np.nan, 3.0], "B": [1.0, 2.0, 3.0]}, index=idx)
    aligned = align_prices(prices, how="inner")
    assert len(aligned) == 2
    assert aligned.isna().sum().sum() == 0


def test_compute_returns_raises_on_nan_prices():
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    prices = pd.DataFrame({"A": [1.0, np.nan, 3.0]}, index=idx)
    with pytest.raises(ValueError):
        compute_returns(prices, method="simple")


def test_align_prices_outer_keeps_nans():
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    prices = pd.DataFrame({"A": [1.0, np.nan, 3.0], "B": [1.0, 2.0, 3.0]}, index=idx)
    aligned = align_prices(prices, how="outer")
    assert len(aligned) == 3
    assert aligned.isna().sum().sum() == 1


def test_align_prices_invalid_how_raises():
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    prices = pd.DataFrame({"A": [1.0, 2.0, 3.0]}, index=idx)
    with pytest.raises(ValueError):
        align_prices(prices, how="banana")


def test_compute_returns_invalid_method_raises():
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    prices = pd.DataFrame({"A": [1.0, 2.0, 3.0]}, index=idx)
    with pytest.raises(ValueError):
        compute_returns(prices, method="banana")


def test_compute_returns_constant_prices_zero_returns():
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    prices = pd.DataFrame({"A": [100.0] * 5}, index=idx)
    rets = compute_returns(prices, method="simple")
    assert np.allclose(rets["A"].values, 0.0)


def test_compute_returns_scale_invariance_simple():
    # If prices are multiplied by a constant, simple returns are unchanged.
    idx = pd.date_range("2020-01-01", periods=4, freq="D")
    prices = pd.DataFrame({"A": [100.0, 105.0, 103.0, 110.0]}, index=idx)
    rets1 = compute_returns(prices, method="simple")
    rets2 = compute_returns(prices * 7.5, method="simple")
    assert np.allclose(rets1.values, rets2.values)


def test_compute_returns_log_vs_simple_small_moves():
    # For small returns, log return approx equals simple return.
    idx = pd.date_range("2020-01-01", periods=4, freq="D")
    prices = pd.DataFrame({"A": [100.0, 100.5, 100.2, 100.8]}, index=idx)
    r_simple = compute_returns(prices, method="simple")["A"].values
    r_log = compute_returns(prices, method="log")["A"].values
    assert np.allclose(r_simple, r_log, atol=1e-4)


def test_ensure_datetime_index_raises_on_non_datetime_index():
    prices = pd.DataFrame({"A": [1.0, 2.0, 3.0]}, index=[1, 2, 3])
    with pytest.raises(TypeError):
        _ensure_datetime_index(prices)


def test_compute_returns_drops_first_row_only():
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    prices = pd.DataFrame({"A": [100.0, 110.0, 121.0]}, index=idx)
    rets = compute_returns(prices, method="simple")
    # returns should start from second date
    assert rets.index[0] == idx[1]
    assert len(rets) == len(prices) - 1