from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TrafficLightThresholds:
    green_max: int = 4        # 0..4
    yellow_max: int = 9       # 5..9, >=10 is red

    def color(self, n_viol: int) -> str:
        if n_viol <= self.green_max:
            return "GREEN"
        if n_viol <= self.yellow_max:
            return "YELLOW"
        return "RED"


def rolling_violations(hits: pd.Series, window: int = 250) -> pd.Series:
    """
    Rolling sum of violations over a fixed window.
    hits must be 0/1, DatetimeIndex recommended.
    """
    if not isinstance(hits, pd.Series):
        raise TypeError("hits must be a pandas Series")
    if hits.isna().any():
        raise ValueError("hits contains NaNs")
    if not np.isin(hits.astype(int).to_numpy(), [0, 1]).all():
        raise ValueError("hits must contain only 0/1 values")
    if window <= 1:
        raise ValueError("window must be >= 2")

    v = hits.astype(int).rolling(window=window, min_periods=window).sum()
    v.name = "n_viol"
    return v


def traffic_light(
    hits: pd.Series,
    window: int = 250,
    *,
    thresholds: Optional[TrafficLightThresholds] = None,
) -> pd.Series:
    """
    Map rolling violation counts to Basel traffic-light colors.

    Returns a Series of strings: GREEN/YELLOW/RED, with NaN for dates
    where the rolling window is not yet available.
    """
    if thresholds is None:
        thresholds = TrafficLightThresholds()

    v = rolling_violations(hits, window=window)
    out = v.copy().astype("object")

    for i in range(len(out)):
        if pd.isna(out.iat[i]):
            out.iat[i] = np.nan
        else:
            out.iat[i] = thresholds.color(int(out.iat[i]))

    out.name = "traffic_light"
    return out
