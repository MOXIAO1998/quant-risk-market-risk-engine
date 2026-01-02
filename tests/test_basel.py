import numpy as np
import pandas as pd
import pytest

from risk_engine.basel import rolling_violations, traffic_light, TrafficLightThresholds


def test_rolling_violations_basic():
    idx = pd.date_range("2020-01-01", periods=6, freq="D")
    hits = pd.Series([0, 1, 0, 1, 1, 0], index=idx)
    v = rolling_violations(hits, window=3)

    # first two are NaN, then rolling sums:
    # [0,1,0]->1, [1,0,1]->2, [0,1,1]->2, [1,1,0]->2
    assert v.isna().iloc[0]
    assert v.isna().iloc[1]
    assert v.iloc[2] == 1
    assert v.iloc[3] == 2
    assert v.iloc[4] == 2
    assert v.iloc[5] == 2


def test_traffic_light_mapping_default_thresholds():
    idx = pd.date_range("2020-01-01", periods=12, freq="D")

    # Case 1: GREEN then YELLOW in consecutive windows
    # Make first 10-day window have 4 ones, next 10-day window have 5 ones.
    # Window at t=10 (idx[9]): hits[0:10]
    # Window at t=11 (idx[10]): hits[1:11]
    hits = pd.Series(
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0],  # first 10 => 4, next window => 4-1+1 = 4? wait:
        index=idx,
    )
    # The above still keeps 4; so we instead drop a 0 and add a 1 in the shifted window:
    hits = pd.Series(
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # first 10 => 5 (YELLOW) not desired
        index=idx,
    )

    # Let's do it explicitly with two separate sequences to avoid confusion:

    # GREEN sequence: exactly 4 ones in first 10
    hits_green = pd.Series([1,1,1,1,0,0,0,0,0,0, 0,0], index=idx)
    tl_green = traffic_light(hits_green, window=10)
    assert tl_green.iloc[:9].isna().all()
    assert tl_green.iloc[9] == "GREEN"  # 4
    assert tl_green.iloc[10] == "GREEN" # still 3 or 4, definitely not yellow

    # YELLOW sequence: exactly 5 ones in first 10
    hits_yellow = pd.Series([1,1,1,1,1,0,0,0,0,0, 0,0], index=idx)
    tl_yellow = traffic_light(hits_yellow, window=10)
    assert tl_yellow.iloc[:9].isna().all()
    assert tl_yellow.iloc[9] == "YELLOW"  # 5

    # RED sequence: 10 ones in first 10
    hits_red = pd.Series([1]*10 + [0]*2, index=idx)
    tl_red = traffic_light(hits_red, window=10)
    assert tl_red.iloc[:9].isna().all()
    assert tl_red.iloc[9] == "RED"  # 10


def test_custom_thresholds():
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    hits = pd.Series([1, 0, 1, 0, 1], index=idx)
    th = TrafficLightThresholds(green_max=1, yellow_max=2)  # 0..1 green, 2 yellow, >=3 red
    tl = traffic_light(hits, window=3, thresholds=th)
    # rolling sums: [1,0,1]=2 -> YELLOW, [0,1,0]=1 -> GREEN, [1,0,1]=2 -> YELLOW
    assert tl.iloc[0:2].isna().all()
    assert tl.iloc[2] == "YELLOW"
    assert tl.iloc[3] == "GREEN"
    assert tl.iloc[4] == "YELLOW"


def test_invalid_hits_raises():
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    with pytest.raises(ValueError):
        rolling_violations(pd.Series([0, 1, np.nan], index=idx), window=2)
    with pytest.raises(ValueError):
        rolling_violations(pd.Series([0, 2, 1], index=idx), window=2)
    with pytest.raises(ValueError):
        traffic_light(pd.Series([0, 2, 1], index=idx), window=2)
