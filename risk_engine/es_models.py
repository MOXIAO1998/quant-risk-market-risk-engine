from __future__ import annotations

from statistics import NormalDist
from typing import Optional
from .var_models import _validate_inputs, _as_loss, historical_var, rolling_mean_std,gaussian_var, ewma_var

import numpy as np
import pandas as pd

def historical_es(
    returns: pd.Series,
    alpha: float = 0.95,
    window: int = 252,
    *,
    loss_positive: bool = True,
) -> pd.Series:
    """
    Historical ES (Expected Shortfall / CVaR).

    Steps (no look-ahead):
      1) q_t = Q_{1-alpha}(r_{t-window:t-1})  (return-space quantile, typically negative)
         We reuse historical_var(..., loss_positive=False) to compute q_t.
      2) ES_return(t) = mean( r_{t-window:t-1} | r <= q_t )

    Output:
      - loss_positive=True  => ES_loss = -ES_return (positive)
      - loss_positive=False => ES_return (typically negative)
    """
    _validate_inputs(returns, alpha, window)

    # return-space quantile threshold, already no-lookahead due to shift(1) inside historical_var
    q = historical_var(returns, alpha=alpha, window=window, loss_positive=False)

    # the same shifted returns used for no-lookahead window content
    r_shift = returns.shift(1)

    es_vals = np.full(len(returns), np.nan, dtype=float)

    q_np = q.to_numpy()
    r_np = r_shift.to_numpy()

    for t in range(len(returns)):
        qt = q_np[t]
        if np.isnan(qt):
            continue

        start = t - window + 1
        if start < 0:
            continue

        w = r_np[start : t + 1]  # length=window, corresponds to original r[t-window : t-1]
        # w may contain nan only at the very beginning; but when q is defined, this window is valid.
        tail = w[w <= qt]
        if tail.size == 0:
            # extremely unlikely; fallback to quantile itself
            es_vals[t] = float(qt)
        else:
            es_vals[t] = float(np.mean(tail))

    es_return = pd.Series(es_vals, index=returns.index, name=f"HS_ES_{alpha:.4f}")

    out = (-es_return) if loss_positive else es_return
    out.name = f"HS_ES_{alpha:.4f}"
    return out



def gaussian_es(
    returns: pd.Series,
    alpha: float = 0.95,
    window: int = 252,
    *,
    loss_positive: bool = True,
    ddof: int = 1,
) -> pd.Series:
    """
    Gaussian ES in return space (no look-ahead, rolling estimates):

    Let p = 1 - alpha (left-tail probability), z = Phi^{-1}(p).
    For Normal(mu, sigma^2), ES in the left tail is:
      ES_return = mu - sigma * (phi(z) / p)

    Output sign controlled by loss_positive.
    """
    _validate_inputs(returns, alpha, window)
    if ddof not in (0, 1):
        raise ValueError("ddof must be 0 or 1")

    # r_shift = returns.shift(1)
    # mu = r_shift.rolling(window=window, min_periods=window).mean()
    # sigma = r_shift.rolling(window=window, min_periods=window).std(ddof=ddof)
    mu, sigma = rolling_mean_std(returns, window=window,ddof=ddof)

    p = 1.0 - alpha
    nd = NormalDist()
    z = nd.inv_cdf(p)
    phi = nd.pdf(z)

    es_return = mu - sigma * (phi / p)
    out = _as_loss(es_return, loss_positive=loss_positive)
    out.name = f"Gaussian_ES_{alpha:.4f}"
    return out


def ewma_es(
    returns: pd.Series,
    alpha: float = 0.95,
    window: int = 252,
    *,
    loss_positive: bool = True,
    lam: float = 0.94,
    use_mean: bool = False,
) -> pd.Series:
    """
    EWMA ES using RiskMetrics-style volatility with a Normal assumption.

    sigma_t^2 = lam*sigma_{t-1}^2 + (1-lam)*r_{t-1}^2 (no look-ahead)
    ES_return(t) = mu_t - sigma_t * (phi(z)/p)   if use_mean else 0 - sigma_t*(phi(z)/p)
    where p = 1-alpha and z = Phi^{-1}(p).

    This is consistent with EWMA-VaR (Normal) used in many practical settings.
    """
    _validate_inputs(returns, alpha, window)
    if not (0.0 < lam < 1.0):
        raise ValueError("lam must be in (0, 1)")

    r = returns.astype(float).to_numpy()
    n = len(r)

    if n <= window:
        out = pd.Series(np.full(n, np.nan), index=returns.index, name=f"EWMA_ES_{alpha:.4f}")
        return out

    # shifted returns r_{t-1} for sigma_t
    r_shift = np.roll(r, 1)
    r_shift[0] = np.nan

    sigma2 = np.full(n, np.nan, dtype=float)

    seed_var = np.var(r[:window], ddof=1)
    sigma2[window] = seed_var

    for t in range(window + 1, n):
        prev = sigma2[t - 1]
        rt_1 = r_shift[t]
        sigma2[t] = lam * prev + (1.0 - lam) * (rt_1 ** 2)

    sigma = np.sqrt(sigma2)

    p = 1.0 - alpha
    nd = NormalDist()
    z = nd.inv_cdf(p)
    phi = nd.pdf(z)

    if use_mean:
        mu = pd.Series(returns.shift(1)).rolling(window=window, min_periods=window).mean().to_numpy()
        es = mu - sigma * (phi / p)
    else:
        es = -sigma * (phi / p)  # mu=0

    es = pd.Series(es, index=returns.index)
    out = _as_loss(es, loss_positive=loss_positive)
    out.name = f"EWMA_ES_{alpha:.4f}"
    return out
