from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StressResult:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp
    cumulative_return: float
    max_drawdown: float


def _validate_returns(returns: pd.Series) -> None:
    if not isinstance(returns, pd.Series):
        raise TypeError("returns must be a pandas Series")
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise TypeError("returns must have a DatetimeIndex")
    if returns.isna().any():
        raise ValueError("returns contains NaNs")
    returns.sort_index(inplace=False)


def equity_curve(returns: pd.Series, initial_value: float = 1.0) -> pd.Series:
    """
    Equity curve from returns: E_t = E_0 * Π(1 + r_t)
    """
    _validate_returns(returns)
    if initial_value <= 0:
        raise ValueError("initial_value must be positive")
    eq = initial_value * (1.0 + returns.astype(float)).cumprod()
    eq.name = "equity"
    return eq


def max_drawdown(equity: pd.Series) -> float:
    """
    Max drawdown in [0, 1): max peak-to-trough decline as fraction.
    """
    if not isinstance(equity, pd.Series):
        raise TypeError("equity must be a pandas Series")
    if equity.isna().any():
        raise ValueError("equity contains NaNs")
    if (equity <= 0).any():
        raise ValueError("equity must be strictly positive")

    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    return float(dd.min() * -1.0)  # positive drawdown magnitude


def apply_parametric_shock(
    returns: pd.Series,
    *,
    shock: float,
    mode: str = "additive",
    shock_date: Optional[pd.Timestamp] = None,
) -> pd.Series:
    """
    Apply a simple parametric stress to returns.

    mode:
      - "additive": r_t' = r_t + shock  (e.g., shock=-0.10 means -10% shock)
      - "multiplicative": r_t' = (1+r_t)*(1+shock) - 1  (scales gross returns)
    shock_date:
      - None => apply to ALL dates (useful for "scenario shift" style)
      - specific date => apply only that day (classic one-day shock)
    """
    _validate_returns(returns)
    if mode not in {"additive", "multiplicative"}:
        raise ValueError("mode must be 'additive' or 'multiplicative'")

    out = returns.astype(float).copy()

    if shock_date is not None:
        if shock_date not in out.index:
            raise ValueError("shock_date not found in returns index")
        idx = [shock_date]
    else:
        idx = out.index

    if mode == "additive":
        out.loc[idx] = out.loc[idx] + shock
    else:
        out.loc[idx] = (1.0 + out.loc[idx]) * (1.0 + shock) - 1.0

    out.name = "stressed_return"
    return out


def historical_window_stress(
    returns: pd.Series,
    *,
    start: str,
    end: str,
    name: str = "historical_window",
) -> StressResult:
    """
    Stress result using realized returns in a historical window [start, end].
    """
    _validate_returns(returns)
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    if s > e:
        raise ValueError("start must be <= end")

    window = returns.loc[(returns.index >= s) & (returns.index <= e)]
    if window.empty:
        raise ValueError("no observations in the specified window")

    eq = equity_curve(window, initial_value=1.0)
    cum_ret = float(eq.iloc[-1] - 1.0)
    mdd = max_drawdown(eq)

    return StressResult(
        name=name,
        start=window.index[0],
        end=window.index[-1],
        cumulative_return=cum_ret,
        max_drawdown=mdd,
    )


def summarize_stress(
    returns: pd.Series,
    scenarios: Dict[str, pd.Series],
) -> pd.DataFrame:
    """
    Summarize multiple stressed return series into a table.
    Each scenario series must align on the same DatetimeIndex as returns.
    Output columns: cumulative_return, max_drawdown
    """
    _validate_returns(returns)

    rows = []
    for name, r_stress in scenarios.items():
        if not isinstance(r_stress, pd.Series):
            raise TypeError(f"scenario '{name}' must be a pandas Series")
        if not r_stress.index.equals(returns.index):
            raise ValueError(f"scenario '{name}' index must equal base returns index")
        if r_stress.isna().any():
            raise ValueError(f"scenario '{name}' contains NaNs")

        eq = equity_curve(r_stress, initial_value=1.0)
        rows.append(
            {
                "scenario": name,
                "cumulative_return": float(eq.iloc[-1] - 1.0),
                "max_drawdown": float(max_drawdown(eq)),
            }
        )

    df = pd.DataFrame(rows).set_index("scenario").sort_index()
    return df
