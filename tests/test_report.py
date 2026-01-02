import numpy as np
import pandas as pd
import pytest

from risk_engine.report import generate_risk_report


def _make_returns(n=400, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series(rng.normal(0.0002, 0.01, size=n), index=idx, name="r")


def test_generate_risk_report_smoke():
    r = _make_returns()
    rep = generate_risk_report(
        r,
        alpha=0.99,
        window=120,
        var_model="HS",
        es_model="HS",
        basel_window=50,
        loss_positive=True,
        historical_stress_window=None,
    )

    # basic structural checks
    assert rep.alpha == 0.99
    assert rep.window == 120
    assert rep.var_model == "HS"
    assert rep.es_model == "HS"

    assert isinstance(rep.var, pd.Series)
    assert isinstance(rep.es, pd.Series)
    assert rep.var.index.equals(r.index)
    assert rep.es.index.equals(r.index)

    # backtest fields exist
    assert rep.backtest.n > 0
    assert 0 <= rep.backtest.viol_rate <= 1

    # basel light exists
    assert isinstance(rep.basel_light, pd.Series)
    assert rep.basel_light.notna().any()

    # stress table exists
    assert "cumulative_return" in rep.stress_table.columns
    assert "max_drawdown" in rep.stress_table.columns
    assert rep.stress_table.shape[0] >= 1


def test_report_loss_positive_semantics():
    r = _make_returns()
    rep = generate_risk_report(r, alpha=0.95, window=100, var_model="GAUSSIAN", es_model="GAUSSIAN", basel_window=50)

    # where defined, VaR/ES should be non-negative under loss_positive=True
    v = rep.var.dropna()
    e = rep.es.dropna()
    assert (v.values >= 0).all()
    assert (e.values >= 0).all()


def test_report_invalid_alpha_raises():
    r = _make_returns()
    with pytest.raises(ValueError):
        generate_risk_report(r, alpha=1.0, window=100)
