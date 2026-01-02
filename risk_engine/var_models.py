from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist
from typing import Literal, Optional

import numpy as np
import pandas as pd

def _validate_inputs(returns: pd.Series, alpha: float, window: int) -> None:
    if not isinstance(returns, pd.Series):
        raise TypeError("returns must be a pandas Series")
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise TypeError("returns index must be a DatetimeIndex")
    if returns.isna().any():
        raise ValueError("returns contains NaNs; align/fill before VaR modeling")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")
    if window <= 1:
        raise ValueError("window must be >= 2")


def _as_loss(var_return_quantile: pd.Series, *, loss_positive: bool = True) -> pd.Series:
    """
    Convert a return-quantile (typically negative) into:
      - loss_positive=True: positive loss threshold => VaR = -q
      - loss_positive=False: return quantile itself
    """
    return -var_return_quantile if loss_positive else var_return_quantile


def historical_var(
    returns: pd.Series,
    alpha: float = 0.95,
    window: int = 252,  # year
    *,
    loss_positive: bool = True) -> pd.Series:
    """
    Historical Simulation VaR.

    Computes q_{1-alpha} of past returns (rolling window, no look-ahead):
      q_t = Q_{1-alpha}(r_{t-window : t-1})
      VaR_t = -q_t  if loss_positive else q_t
    """
    _validate_inputs(returns, alpha, window)

    r_shift = returns.shift(1)  # exclude current day to avoid look-ahead, shift down by 1 index and leave NaN
    q = r_shift.rolling(window=window, min_periods=window).quantile(1.0 - alpha)
    out = _as_loss(q, loss_positive=loss_positive)
    out.name = f"HS_VaR_{alpha:.4f}"
    return out


def rolling_mean_std(returns: pd.Series, window: int, *, ddof: int = 1) -> tuple[pd.Series, pd.Series]:
    r_shift = returns.shift(1)
    mu = r_shift.rolling(window=window, min_periods=window).mean()
    sigma = r_shift.rolling(window=window, min_periods=window).std(ddof=ddof)
    return mu, sigma


def gaussian_var(
    returns: pd.Series,
    alpha: float = 0.95,
    window: int = 252,
    *,
    loss_positive: bool = True,
    ddof: int = 1) -> pd.Series:
    """
    Parametric (Gaussian) VaR:
      q_t = mu_t + sigma_t * z_{1-alpha}
    estimated from past returns in a rolling window, no look-ahead.
    """
    _validate_inputs(returns, alpha, window)
    if ddof not in (0, 1):
        raise ValueError("ddof must be 0 or 1")

    # r_shift = returns.shift(1)
    # mu = r_shift.rolling(window=window, min_periods=window).mean()
    # sigma = r_shift.rolling(window=window, min_periods=window).std(ddof=ddof)
    mu, sigma = rolling_mean_std(returns, window=window, ddof=ddof)

    z = NormalDist().inv_cdf(1.0 - alpha)  # negative for alpha>0.5
    q = mu + sigma * z

    out = _as_loss(q, loss_positive=loss_positive)
    out.name = f"Gaussian_VaR_{alpha:.4f}"
    return out

def ewma_var(
    returns: pd.Series,
    alpha: float = 0.95,
    window: int = 252,
    *,
    loss_positive: bool = True,
    lam: float = 0.95, # lam typically will be in (0.9, 1)
    use_mean: bool = False,
) -> pd.Series:
    """
    EWMA (RiskMetrics-style) VaR.

    Volatility recursion (no look-ahead):
      sigma_t^2 = lam*sigma_{t-1}^2 + (1-lam)*r_{t-1}^2

    For prediction at day t, we use sigma_t computed using r_{t-1}.
    We require an initial variance seed; we use the sample variance of the
    first 'window' shifted returns.

    If use_mean=True, include rolling mean in q_t (rare in RiskMetrics):
      q_t = mu_t + sigma_t * z_{1-alpha}
    else:
      q_t = 0 + sigma_t * z_{1-alpha}
    """
    _validate_inputs(returns, alpha, window)
    if not (0.0 < lam < 1.0):
        raise ValueError("lam must be in (0, 1)")

    r = returns.astype(float).to_numpy()
    n = len(r)

    # shifted returns for no look-ahead: r_{t-1} feeds sigma_t
    r_shift = np.roll(r, 1)
    r_shift[0] = np.nan

    sigma2 = np.full(n, np.nan, dtype=float)

    # seed at index = window (i.e., when we first have window shifted returns available)
    # We need r_shift[1:window+1] corresponds to original returns[0:window] seed_end = window
    # inclusive index for original? We'll just use first 'window' real returns.
    if n <= window:
        # not enough data to produce any VaR
        out = pd.Series(np.full(n, np.nan), index=returns.index, name=f"EWMA_VaR_{alpha:.4f}")
        return out

    seed_var = np.var(r[:window], ddof=1)
    sigma2[window] = seed_var

    # propagate from window+1 onward
    for t in range(window + 1, n):
        prev = sigma2[t - 1]
        rt_1 = r_shift[t]  # equals r[t-1]
        sigma2[t] = lam * prev + (1.0 - lam) * (rt_1 ** 2)

    sigma = np.sqrt(sigma2)
    z = NormalDist().inv_cdf(1.0 - alpha)

    if use_mean:
        mu = pd.Series(returns.shift(1)).rolling(window=window, min_periods=window).mean().to_numpy()
        q = mu + sigma * z
    else:
        q = sigma * z

    q = pd.Series(q, index=returns.index)
    out = _as_loss(q, loss_positive=loss_positive)
    out.name = f"EWMA_VaR_{alpha:.4f}"
    return out