from __future__ import annotations # if python version is less than 3.11

from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

import yfinance as yf

@dataclass(frozen=True)
class MarketData:
    prices: pd.DataFrame    # index=DatetimeIndex, columns=tickers
    returns: pd.DataFrame   # aligned daily returns, starts from 2nd price date i.e. (p_t - p_t-1)/(p_t-1)


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a pandas.DatetimeIndex")
    # make timezone-naive for consistency
    if df.index.tz is not None:
        df = df.copy()
        df.index = df.index.tz_convert(None)
    return df.sort_index()

def align_prices(prices: pd.DataFrame, how:str ="inner") -> pd.DataFrame:
    """
       Align prices across tickers by date.
       - inner: keep only dates where all tickers have prices (recommended), keep intersection
       - outer: keep union (may contain NaNs)
    """
    prices = _ensure_datetime_index(prices)
    if how not in ["inner", "outer"]:
        raise ValueError("‘how’ must be either 'inner' or 'outer'")
    if how == "inner":
        return prices.dropna(axis = 0, how = "any")
    else:
        return prices

def compute_returns(prices: pd.DataFrame, method: str = "simple") -> pd.DataFrame:
    """
        Compute returns from prices.
        - simple: r_t = P_t / P_{t-1} - 1
        - log:    r_t = log(P_t) - log(P_{t-1})
    """
    prices = _ensure_datetime_index(prices)

    if prices.isna().any().any():
        raise ValueError("Prices contain NaNs. Align/fill before computing returns.")

    if method not in ["simple", "log"]:
        raise ValueError("'method' must be 'simple' or 'log'")

    if method == "simple":
        rets = prices.pct_change()      # (row_t - row_t-1)/ (row_t-1_
    else:
        rets = np.log(prices).diff()   # log(row_t) - log(row_t-1)
    return rets.dropna(how="all")

def download_prices_yfinance(
        tickers:Iterable[str],
        start:str,
        end:Optional[str] = None,   # to the newest price data
        price_field_preference: Optional[List[str]] = None) -> pd.DataFrame:

    tickers = list(tickers)
    if not tickers:
        raise ValueError("tickers must be non-empty")

    if price_field_preference is None:
        price_field_preference = ["Adj Close", "Close"]

    df = yf.download(
        tickers = tickers,
        start = start,
        end = end,
        auto_adjust= False,
        progress = False,
        group_by = "column"
    )

    if df is None or len(df) == 0:
        raise RuntimeError("No data returned from yfinance.")

    # MultiIndex if multiple tickers
    if isinstance(df.columns, pd.MultiIndex):
        px = None
        for field in price_field_preference:
            if field in df.columns.get_level_values(0):
                px = df[field].copy()
                break
        if px is None:
            raise RuntimeError(f"None of fields {price_field_preference} found in yfinance output.")
    else:
        # single ticker
        px = None
        for field in price_field_preference:
            if field in df.columns:
                px = df[field].to_frame(tickers[0]).copy()
                break
        if px is None:
            raise RuntimeError(f"None of fields {price_field_preference} found in yfinance output.")

    px = px.rename_axis("date")
    px = _ensure_datetime_index(px)

    if (px <= 0).any().any():
        raise ValueError("Downloaded prices contain non-positive values.")

    return px


def load_market_data_default(
    start: str = "2016-01-01",
    end: Optional[str] = None,
    tickers: Optional[List[str]] = None,
    return_method: str = "simple",
) -> MarketData:
    """
    Default dataset used throughout the project: multi-asset ETF proxies.
    """
    if tickers is None:
        tickers = ["SPY", "QQQ", "TLT", "GLD", "HYG"]

    prices_raw = download_prices_yfinance(tickers=tickers, start=start, end=end)
    prices = align_prices(prices_raw, how="inner")
    returns = compute_returns(prices, method=return_method)

    # sanity: returns dates must be subset of prices dates
    if not returns.index.isin(prices.index).all():
        raise AssertionError("Returns index is not aligned to prices index.")

    return MarketData(prices=prices, returns=returns)