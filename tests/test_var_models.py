import numpy as np
import pandas as pd
import pytest

from risk_engine.var_models import historical_var, gaussian_var, ewma_var


def _make_returns(n=400, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    r = pd.Series(rng.normal(0.0002, 0.01, size=n), index=idx, name="r")
    return r


def test_var_sign_convention_flag():
    r = _make_returns()
    v_pos = historical_var(r, alpha=0.95, window=100, loss_positive=True)
    v_retq = historical_var(r, alpha=0.95, window=100, loss_positive=False)
    # same dates, same magnitude, opposite sign where defined
    mask = v_pos.notna()
    assert np.allclose(v_pos[mask].values, (-v_retq[mask]).values)


def test_var_monotonicity_hs():
    r = _make_returns()
    v95 = historical_var(r, alpha=0.95, window=100, loss_positive=True)
    v99 = historical_var(r, alpha=0.99, window=100, loss_positive=True)
    mask = v95.notna() & v99.notna()
    assert np.all(v99[mask].values >= v95[mask].values)


def test_var_scale_invariance_all_models():
    r = _make_returns()
    k = 2.5
    for fn in (historical_var, gaussian_var, ewma_var):
        v1 = fn(r, alpha=0.99, window=120, loss_positive=True)
        v2 = fn(k * r, alpha=0.99, window=120, loss_positive=True)
        mask = v1.notna() & v2.notna()
        assert np.allclose(v2[mask].values, (k * v1[mask]).values, rtol=1e-10, atol=1e-12)


def test_no_lookahead_last_point_not_affected_by_current_return():
    # Change the last return massively; VaR at the last date should not change
    # because it uses returns up to t-1.
    r = _make_returns(n=300, seed=1)
    r2 = r.copy()
    r2.iloc[-1] = -0.99  # huge loss today

    v1 = historical_var(r, alpha=0.99, window=100, loss_positive=True)
    v2 = historical_var(r2, alpha=0.99, window=100, loss_positive=True)

    # Compare VaR at the last timestamp
    assert np.isclose(v1.iloc[-1], v2.iloc[-1])


def test_window_too_small_raises():
    r = _make_returns()
    with pytest.raises(ValueError):
        historical_var(r, alpha=0.99, window=1)


def test_alpha_out_of_range_raises():
    r = _make_returns()
    with pytest.raises(ValueError):
        gaussian_var(r, alpha=1.0, window=100)
    with pytest.raises(ValueError):
        gaussian_var(r, alpha=0.0, window=100)


def test_var_defined_only_after_window():
    r = _make_returns(n=260)
    w = 100
    v = gaussian_var(r, alpha=0.99, window=w, loss_positive=True)
    # because we shift(1), first valid is at index w (0-based) + 1? rolling needs w shifted obs
    # specifically: r_shift has first non-nan at index 1, so first full window ends at index w
    assert v.iloc[:w].isna().all()
    assert v.iloc[w:].notna().any()


def test_models_return_series_shape_and_index():
    r = _make_returns()
    for fn in (historical_var, gaussian_var, ewma_var):
        v = fn(r, alpha=0.99, window=120, loss_positive=True)
        assert isinstance(v, pd.Series)
        assert v.index.equals(r.index)
        assert len(v) == len(r)
