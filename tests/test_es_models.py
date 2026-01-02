import numpy as np
import pandas as pd
import pytest

from risk_engine.var_models import historical_var, gaussian_var, ewma_var
from risk_engine.es_models import historical_es, gaussian_es, ewma_es


def _make_returns(n=500, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    r = pd.Series(rng.normal(0.0002, 0.01, size=n), index=idx, name="r")
    return r


def test_es_sign_convention_flag():
    r = _make_returns()
    es_pos = historical_es(r, alpha=0.95, window=120, loss_positive=True)
    es_ret = historical_es(r, alpha=0.95, window=120, loss_positive=False)
    mask = es_pos.notna()
    assert np.allclose(es_pos[mask].values, (-es_ret[mask]).values)


def test_es_vs_var_historical_es_more_conservative():
    r = _make_returns()
    w = 150
    alpha = 0.99
    var = historical_var(r, alpha=alpha, window=w, loss_positive=True)
    es = historical_es(r, alpha=alpha, window=w, loss_positive=True)
    mask = var.notna() & es.notna()
    # ES (expected tail loss) should be >= VaR (tail threshold)
    assert np.all(es[mask].values >= var[mask].values)


def test_es_vs_var_gaussian_es_more_conservative():
    r = _make_returns()
    w = 150
    alpha = 0.99
    var = gaussian_var(r, alpha=alpha, window=w, loss_positive=True)
    es = gaussian_es(r, alpha=alpha, window=w, loss_positive=True)
    mask = var.notna() & es.notna()
    assert np.all(es[mask].values >= var[mask].values)


def test_es_vs_var_ewma_es_more_conservative():
    r = _make_returns()
    w = 150
    alpha = 0.99
    var = ewma_var(r, alpha=alpha, window=w, loss_positive=True)
    es = ewma_es(r, alpha=alpha, window=w, loss_positive=True)
    mask = var.notna() & es.notna()
    assert np.all(es[mask].values >= var[mask].values)


def test_es_monotonicity_historical():
    r = _make_returns()
    w = 120
    es95 = historical_es(r, alpha=0.95, window=w, loss_positive=True)
    es99 = historical_es(r, alpha=0.99, window=w, loss_positive=True)
    mask = es95.notna() & es99.notna()
    assert np.all(es99[mask].values >= es95[mask].values)


def test_es_scale_invariance_all_models():
    r = _make_returns()
    k = 1.7
    for fn in (historical_es, gaussian_es, ewma_es):
        e1 = fn(r, alpha=0.99, window=160, loss_positive=True)
        e2 = fn(k * r, alpha=0.99, window=160, loss_positive=True)
        mask = e1.notna() & e2.notna()
        assert np.allclose(e2[mask].values, (k * e1[mask]).values, rtol=1e-10, atol=1e-12)


def test_no_lookahead_last_point_not_affected_by_current_return():
    r = _make_returns(n=350, seed=2)
    r2 = r.copy()
    r2.iloc[-1] = -0.99  # huge loss today

    es1 = gaussian_es(r, alpha=0.99, window=120, loss_positive=True)
    es2 = gaussian_es(r2, alpha=0.99, window=120, loss_positive=True)

    assert np.isclose(es1.iloc[-1], es2.iloc[-1])


def test_invalid_alpha_and_window_raises():
    r = _make_returns()
    with pytest.raises(ValueError):
        historical_es(r, alpha=1.0, window=100)
    with pytest.raises(ValueError):
        gaussian_es(r, alpha=0.0, window=100)
    with pytest.raises(ValueError):
        ewma_es(r, alpha=0.99, window=1)
