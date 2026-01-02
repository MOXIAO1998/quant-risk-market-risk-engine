from __future__ import annotations  # if python version < 3.11

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Portfolio:
    weights: pd.Series  # index=tickers


def make_equal_weight_portfolio(tickers: list[str]) -> Portfolio:
    if len(tickers) == 0:
        raise ValueError("tickers must be non-empty")
    w = pd.Series(1.0 / len(tickers), index=tickers, dtype=float)
    return Portfolio(weights=w)


def make_portfolio_from_dict(weights: Dict[str, float], normalize: bool = True) -> Portfolio:
    if not weights:
        raise ValueError("weights must be non-empty")
    w = pd.Series(weights, dtype=float)

    if (w < 0).any():
        raise ValueError("baseline project assumes long-only weights (non-negative).")

    if normalize:
        s = float(w.sum())
        if s <= 0:
            raise ValueError("sum(weights) must be positive")
        w = w / s

    return Portfolio(weights=w)


def portfolio_returns(returns: pd.DataFrame, portfolio: Portfolio) -> pd.Series:
    """
    Daily portfolio return: r_p,t = sum_i w_i * r_i,t
    Static weights in M1. No look-ahead (uses same-day returns only).
    """
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise TypeError("returns index must be DatetimeIndex")

    w = portfolio.weights
    missing = set(w.index) - set(returns.columns)
    if missing:
        raise ValueError(f"returns missing tickers: {sorted(missing)}")

    X = returns[w.index].copy()
    if X.isna().any().any():
        raise ValueError("returns contain NaNs; align/fill first.")

    pr = X.to_numpy() @ w.to_numpy()
    return pd.Series(pr, index=X.index, name="portfolio_return")


def portfolio_pnl(
    portfolio_return: pd.Series,
    notional: float = 1_000_000.0,
    initial_value: float = 1_000_000.0,
) -> pd.DataFrame:
    """
    Convert return series to PnL and equity curve.
    pnl_t = notional * r_t
    equity_t = initial_value * Π_{s<=t} (1 + r_s)
    """
    if notional <= 0 or initial_value <= 0:
        raise ValueError("notional and initial_value must be positive")

    r = portfolio_return.astype(float)
    pnl = notional * r
    equity = initial_value * (1.0 + r).cumprod()

    return pd.DataFrame({"return": r, "pnl": pnl, "equity": equity})
