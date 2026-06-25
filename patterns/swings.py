"""Swing high / low (pivot) detection for chart pattern analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SwingPoint:
    idx: int
    price: float
    kind: str  # "high" or "low"


def find_swings(
    df: pd.DataFrame,
    high_col: str = "rth_high",
    low_col: str = "rth_low",
    window: int = 5,
) -> tuple[list[SwingPoint], list[SwingPoint]]:
    """
    Identify swing highs and lows using a centered rolling window.
    A swing high at i requires high[i] == max(high[i-window : i+window+1]).
    """
    highs = df[high_col].values
    lows = df[low_col].values
    n = len(df)

    swing_highs: list[SwingPoint] = []
    swing_lows: list[SwingPoint] = []

    for i in range(window, n - window):
        hi_slice = highs[i - window : i + window + 1]
        lo_slice = lows[i - window : i + window + 1]
        if highs[i] == np.max(hi_slice):
            swing_highs.append(SwingPoint(i, float(highs[i]), "high"))
        if lows[i] == np.min(lo_slice):
            swing_lows.append(SwingPoint(i, float(lows[i]), "low"))

    return swing_highs, swing_lows


def ohlc_frame(daily: pd.DataFrame) -> pd.DataFrame:
    """Standard OHLCV view from daily CSV RTH columns."""
    return daily.assign(
        open=daily["rth_open"],
        high=daily["rth_high"],
        low=daily["rth_low"],
        close=daily["rth_close"],
        volume=daily["rth_volume"],
    )
