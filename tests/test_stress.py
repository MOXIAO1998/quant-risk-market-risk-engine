import numpy as np
import pandas as pd
import pytest

from risk_engine.stress import (
    equity_curve,
    max_drawdown,
    apply_parametric_shock,
    historical_window_stress,
    summarize_stress,
)


def _make_returns():
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    r = pd.Series([0.01, -0.02, 0.03, -0.01, 0.00], index=idx, name="r")
    return r


def test_equity_curve_basic():
    r = _make_returns()
    eq = equity_curve(r, initial_value=1.0)
    assert np.isclose(eq.iloc[0], 1.0 * (1.0 + 0.01))
    assert np.isclose(eq.iloc[-1], (1.01) * (0.98) * (1.03) * (0.99) * (1.0))


def test_max_drawdown_basic():
    idx = pd.date_range("2020-01-01", periods=4, freq="D")
    eq = pd.Series([1.0, 1.2, 1.0, 1.1], index=idx)
    # peak=1.2, trough=1.0 => drawdown = 1 - 1.0/1.2 = 0.166666...
    mdd = max_drawdown(eq)
    assert np.isclose(mdd, 1.0 - (1.0 / 1.2))


def test_apply_parametric_shock_additive_all_dates():
    r = _make_returns()
    shocked = apply_parametric_shock(r, shock=-0.10, mode="additive")
    assert np.allclose(shocked.values, (r.values - 0.10))


def test_apply_parametric_shock_additive_single_date():
    r = _make_returns()
    d = r.index[2]
    shocked = apply_parametric_shock(r, shock=-0.10, mode="additive", shock_date=d)
    expected = r.copy()
    expected.loc[d] = expected.loc[d] - 0.10
    assert np.allclose(shocked.values, expected.values)


def test_apply_parametric_shock_multiplicative():
    r = _make_returns()
    shocked = apply_parametric_shock(r, shock=-0.10, mode="multiplicative")
    expected = (1.0 + r) * (1.0 - 0.10) - 1.0
    assert np.allclose(shocked.values, expected.values)


def test_apply_parametric_shock_invalid_mode_raises():
    r = _make_returns()
    with pytest.raises(ValueError):
        apply_parametric_shock(r, shock=-0.1, mode="banana")


def test_apply_parametric_shock_date_not_found_raises():
    r = _make_returns()
    with pytest.raises(ValueError):
        apply_parametric_shock(r, shock=-0.1, mode="additive", shock_date=pd.Timestamp("2099-01-01"))


def test_historical_window_stress_basic():
    r = _make_returns()
    res = historical_window_stress(r, start="2020-01-02", end="2020-01-04", name="win")
    # window returns are [-0.02, 0.03, -0.01]
    eq = (1.0 - 0.02) * (1.0 + 0.03) * (1.0 - 0.01)
    assert np.isclose(res.cumulative_return, eq - 1.0)
    assert res.start == pd.Timestamp("2020-01-02")
    assert res.end == pd.Timestamp("2020-01-04")


def test_historical_window_stress_empty_raises():
    r = _make_returns()
    with pytest.raises(ValueError):
        historical_window_stress(r, start="2019-01-01", end="2019-01-10")


def test_summarize_stress_table():
    r = _make_returns()
    s1 = apply_parametric_shock(r, shock=-0.05, mode="additive")
    s2 = apply_parametric_shock(r, shock=-0.10, mode="multiplicative")

    df = summarize_stress(r, {"add_5": s1, "mult_10": s2})
    assert "cumulative_return" in df.columns
    assert "max_drawdown" in df.columns
    assert set(df.index.tolist()) == {"add_5", "mult_10"}
