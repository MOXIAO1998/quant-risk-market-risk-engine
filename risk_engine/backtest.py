from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import Optional, Tuple


import numpy as np
import pandas as pd


def _safe_log(x: float, eps: float = 1e-15) -> float:
    return log(max(x, eps))


def _chi2_sf(stat: float, df: int) -> Optional[float]:
    """
    Optional chi-square survival function (p-value).
    Uses scipy if available; otherwise returns None (no extra deps).
    """
    try:
        from scipy.stats import chi2  # type: ignore
        return float(chi2.sf(stat, df=df))
    except Exception:
        return None


@dataclass(frozen=True)
class BacktestResult:
    n: int                 # number of evaluated days
    n_viol: int            # number of violations
    viol_rate: float       # n_viol / n
    expected_rate: float   # 1 - alpha
    kupiec_lr: float
    kupiec_pvalue: Optional[float]
    christoffersen_lr_ind: float
    christoffersen_pvalue_ind: Optional[float]
    lr_cc: float
    pvalue_cc: Optional[float]


def compute_var_hits(
    returns: pd.Series,
    var: pd.Series,
    *,
    loss_positive: bool = True,
) -> pd.DataFrame:
    """
    Build backtest dataset for VaR.

    returns: daily returns r_t
    var: VaR series aligned by date (prediction for the same date)
         If loss_positive=True, var_t is positive loss threshold, and violation is:
            r_t < -var_t
         If loss_positive=False, var_t is return-quantile threshold (usually negative), violation is:
            r_t < var_t
    """
    if not isinstance(returns, pd.Series) or not isinstance(var, pd.Series):
        raise TypeError("returns and var must be pandas Series")
    if not isinstance(returns.index, pd.DatetimeIndex) or not isinstance(var.index, pd.DatetimeIndex):
        raise TypeError("returns and var must have DatetimeIndex")
    if returns.isna().any():
        raise ValueError("returns contains NaNs")
    # var may have NaNs at the beginning due to rolling windows; that's OK.

    df = pd.DataFrame({"return": returns, "var": var}).dropna(subset=["var"])
    if loss_positive:
        hit = (df["return"] < -df["var"]).astype(int)
    else:
        hit = (df["return"] < df["var"]).astype(int)

    df["hit"] = hit
    return df


def kupiec_pof_test(hits: pd.Series, alpha: float) -> Tuple[float, Optional[float]]:
    """
    Kupiec Proportion of Failures (POF) LR test.

    hits: 0/1 violation indicator
    alpha: VaR confidence level (e.g., 0.99). Expected violation rate p = 1 - alpha.

    Returns: (LR_pof, p_value_or_None) with df=1.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")
    if not isinstance(hits, pd.Series):
        raise TypeError("hits must be a pandas Series")

    h = hits.dropna().astype(int).to_numpy()
    if h.size == 0:
        raise ValueError("hits is empty after dropping NaNs")
    if not np.isin(h, [0, 1]).all():
        raise ValueError("hits must be 0/1")

    n = int(h.size)
    x = int(h.sum())
    p = 1.0 - alpha

    # MLE under alternative
    phat = x / n

    # Likelihoods (use safe logs for x=0 or x=n)
    lnL_null = (n - x) * _safe_log(1.0 - p) + x * _safe_log(p)
    lnL_alt = (n - x) * _safe_log(1.0 - phat) + x * _safe_log(phat)

    lr = -2.0 * (lnL_null - lnL_alt)
    pval = _chi2_sf(lr, df=1)
    return float(lr), pval


def christoffersen_independence_test(hits: pd.Series) -> Tuple[float, Optional[float]]:
    """
    Christoffersen (1998) independence LR test for violations.

    hits: 0/1 series. Uses transition counts:
      N00, N01, N10, N11

    Returns: (LR_ind, p_value_or_None) with df=1.

    If transitions are degenerate (never visits a state), returns (nan, None).
    """
    if not isinstance(hits, pd.Series):
        raise TypeError("hits must be a pandas Series")

    h = hits.dropna().astype(int).to_numpy()
    if h.size < 2:
        raise ValueError("hits must have at least 2 observations")
    if not np.isin(h, [0, 1]).all():
        raise ValueError("hits must be 0/1")

    h0 = h[:-1]
    h1 = h[1:]

    n00 = int(np.sum((h0 == 0) & (h1 == 0)))
    n01 = int(np.sum((h0 == 0) & (h1 == 1)))
    n10 = int(np.sum((h0 == 1) & (h1 == 0)))
    n11 = int(np.sum((h0 == 1) & (h1 == 1)))

    # If we never observe transitions from 0 or from 1, independence test is ill-defined
    if (n00 + n01) == 0 or (n10 + n11) == 0:
        return float("nan"), None

    pi0 = n01 / (n00 + n01)
    pi1 = n11 / (n10 + n11)
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    lnL_ind = (
        (n00 + n10) * _safe_log(1.0 - pi) + (n01 + n11) * _safe_log(pi)
    )
    lnL_markov = (
        n00 * _safe_log(1.0 - pi0) + n01 * _safe_log(pi0) +
        n10 * _safe_log(1.0 - pi1) + n11 * _safe_log(pi1)
    )

    lr = -2.0 * (lnL_ind - lnL_markov)
    pval = _chi2_sf(lr, df=1)
    return float(lr), pval


def christoffersen_conditional_coverage_test(hits: pd.Series, alpha: float) -> Tuple[float, Optional[float]]:
    """
    Christoffersen conditional coverage test:
      LR_cc = LR_pof + LR_ind, df=2
    """
    lr_pof, _ = kupiec_pof_test(hits, alpha=alpha)
    lr_ind, _ = christoffersen_independence_test(hits)

    if np.isnan(lr_ind):
        return float("nan"), None

    lr_cc = lr_pof + lr_ind
    pval = _chi2_sf(lr_cc, df=2)
    return float(lr_cc), pval


def backtest_var(
    returns: pd.Series,
    var: pd.Series,
    alpha: float,
    *,
    loss_positive: bool = True,
) -> BacktestResult:
    """
    Convenience wrapper: compute hits + Kupiec + Christoffersen tests.

    Returns a BacktestResult with p-values when scipy is available.
    """
    df = compute_var_hits(returns, var, loss_positive=loss_positive)
    hits = df["hit"]

    n = int(hits.shape[0])
    x = int(hits.sum())
    viol_rate = x / n if n > 0 else float("nan")
    expected = 1.0 - alpha

    lr_pof, p_pof = kupiec_pof_test(hits, alpha=alpha)
    lr_ind, p_ind = christoffersen_independence_test(hits)
    lr_cc, p_cc = christoffersen_conditional_coverage_test(hits, alpha=alpha)

    return BacktestResult(
        n=n,
        n_viol=x,
        viol_rate=float(viol_rate),
        expected_rate=float(expected),
        kupiec_lr=float(lr_pof),
        kupiec_pvalue=p_pof,
        christoffersen_lr_ind=float(lr_ind),
        christoffersen_pvalue_ind=p_ind,
        lr_cc=float(lr_cc),
        pvalue_cc=p_cc,
    )
