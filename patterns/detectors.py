"""Chart pattern detection: continuations and reversals."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from patterns.swings import SwingPoint, find_swings, ohlc_frame

PATTERN_CATEGORIES = {
    "double_bottom": "bullish_reversal",
    "inverse_head_shoulders": "bullish_reversal",
    "falling_wedge": "bullish_reversal",
    "hammer": "bullish_reversal",
    "double_top": "bearish_reversal",
    "head_shoulders": "bearish_reversal",
    "rising_wedge": "bearish_reversal",
    "shooting_star": "bearish_reversal",
    "bull_flag": "bullish_continuation",
    "ascending_triangle": "bullish_continuation",
    "bear_flag": "bearish_continuation",
    "descending_triangle": "bearish_continuation",
}


@dataclass
class PatternMatch:
    pattern: str
    category: str
    end_idx: int
    end_date: object
    confidence: float
    neckline: Optional[float] = None
    metadata: dict = field(default_factory=dict)


def _near(a: float, b: float, tol_pct: float) -> bool:
    mid = (abs(a) + abs(b)) / 2 or 1.0
    return abs(a - b) / mid <= tol_pct


def _slope(xs: np.ndarray, ys: np.ndarray) -> float:
    if len(xs) < 2:
        return 0.0
    coef = np.polyfit(xs, ys, 1)
    return float(coef[0])


# ---------------------------------------------------------------------------
# Reversals — multi-swing
# ---------------------------------------------------------------------------

def detect_double_top(
    df: pd.DataFrame,
    swing_highs: list[SwingPoint],
    tol_pct: float = 0.015,
    min_sep: int = 10,
    max_sep: int = 60,
) -> list[PatternMatch]:
    matches = []
    for i in range(len(swing_highs) - 1):
        a, b = swing_highs[i], swing_highs[i + 1]
        sep = b.idx - a.idx
        if sep < min_sep or sep > max_sep:
            continue
        if not _near(a.price, b.price, tol_pct):
            continue
        between = df.iloc[a.idx : b.idx + 1]
        neckline = float(between["low"].min())
        confirm_idx = b.idx
        for j in range(b.idx + 1, min(b.idx + 10, len(df))):
            if df.iloc[j]["close"] < neckline:
                confirm_idx = j
                break
        else:
            continue
        conf = 1.0 - abs(a.price - b.price) / a.price / tol_pct
        matches.append(PatternMatch(
            "double_top", PATTERN_CATEGORIES["double_top"],
            confirm_idx, df.iloc[confirm_idx]["date"],
            max(0.3, min(1.0, conf)),
            neckline,
            {"peak_a": a.price, "peak_b": b.price},
        ))
    return matches


def detect_double_bottom(
    df: pd.DataFrame,
    swing_lows: list[SwingPoint],
    tol_pct: float = 0.015,
    min_sep: int = 10,
    max_sep: int = 60,
) -> list[PatternMatch]:
    matches = []
    for i in range(len(swing_lows) - 1):
        a, b = swing_lows[i], swing_lows[i + 1]
        sep = b.idx - a.idx
        if sep < min_sep or sep > max_sep:
            continue
        if not _near(a.price, b.price, tol_pct):
            continue
        between = df.iloc[a.idx : b.idx + 1]
        neckline = float(between["high"].max())
        confirm_idx = b.idx
        for j in range(b.idx + 1, min(b.idx + 10, len(df))):
            if df.iloc[j]["close"] > neckline:
                confirm_idx = j
                break
        else:
            continue
        conf = 1.0 - abs(a.price - b.price) / a.price / tol_pct
        matches.append(PatternMatch(
            "double_bottom", PATTERN_CATEGORIES["double_bottom"],
            confirm_idx, df.iloc[confirm_idx]["date"],
            max(0.3, min(1.0, conf)),
            neckline,
            {"trough_a": a.price, "trough_b": b.price},
        ))
    return matches


def detect_head_shoulders(
    df: pd.DataFrame,
    swing_highs: list[SwingPoint],
    shoulder_tol: float = 0.03,
    head_min_lift: float = 0.01,
) -> list[PatternMatch]:
    matches = []
    for i in range(len(swing_highs) - 2):
        ls, head, rs = swing_highs[i], swing_highs[i + 1], swing_highs[i + 2]
        if head.price <= ls.price * (1 + head_min_lift):
            continue
        if head.price <= rs.price * (1 + head_min_lift):
            continue
        if not _near(ls.price, rs.price, shoulder_tol):
            continue
        seg = df.iloc[ls.idx : rs.idx + 1]
        neckline = float(seg["low"].min())
        confirm_idx = rs.idx
        for j in range(rs.idx + 1, min(rs.idx + 10, len(df))):
            if df.iloc[j]["close"] < neckline:
                confirm_idx = j
                break
        else:
            continue
        matches.append(PatternMatch(
            "head_shoulders", PATTERN_CATEGORIES["head_shoulders"],
            confirm_idx, df.iloc[confirm_idx]["date"],
            0.7,
            neckline,
            {"left": ls.price, "head": head.price, "right": rs.price},
        ))
    return matches


def detect_inverse_head_shoulders(
    df: pd.DataFrame,
    swing_lows: list[SwingPoint],
    shoulder_tol: float = 0.03,
    head_min_drop: float = 0.01,
) -> list[PatternMatch]:
    matches = []
    for i in range(len(swing_lows) - 2):
        ls, head, rs = swing_lows[i], swing_lows[i + 1], swing_lows[i + 2]
        if head.price >= ls.price * (1 - head_min_drop):
            continue
        if head.price >= rs.price * (1 - head_min_drop):
            continue
        if not _near(ls.price, rs.price, shoulder_tol):
            continue
        seg = df.iloc[ls.idx : rs.idx + 1]
        neckline = float(seg["high"].max())
        confirm_idx = rs.idx
        for j in range(rs.idx + 1, min(rs.idx + 10, len(df))):
            if df.iloc[j]["close"] > neckline:
                confirm_idx = j
                break
        else:
            continue
        matches.append(PatternMatch(
            "inverse_head_shoulders", PATTERN_CATEGORIES["inverse_head_shoulders"],
            confirm_idx, df.iloc[confirm_idx]["date"],
            0.7,
            neckline,
            {"left": ls.price, "head": head.price, "right": rs.price},
        ))
    return matches


# ---------------------------------------------------------------------------
# Reversals — wedges
# ---------------------------------------------------------------------------

def detect_falling_wedge(
    df: pd.DataFrame,
    swing_highs: list[SwingPoint],
    swing_lows: list[SwingPoint],
    lookback: int = 40,
    min_swings: int = 3,
) -> list[PatternMatch]:
    """Both trendlines fall; lower line falls faster → convergence → bullish breakout."""
    matches = []
    n = len(df)
    for end in range(lookback, n):
        start = end - lookback
        highs = [s for s in swing_highs if start <= s.idx <= end]
        lows = [s for s in swing_lows if start <= s.idx <= end]
        if len(highs) < min_swings or len(lows) < min_swings:
            continue
        hx = np.array([h.idx - start for h in highs])
        hy = np.array([h.price for h in highs])
        lx = np.array([l.idx - start for l in lows])
        ly = np.array([l.price for l in lows])
        hs, ls = _slope(hx, hy), _slope(lx, ly)
        if hs >= 0 or ls >= 0:
            continue
        if ls >= hs:  # lower falls faster → converging
            continue
        resist = float(hy[-1])
        if df.iloc[end]["close"] <= resist:
            continue
        matches.append(PatternMatch(
            "falling_wedge", PATTERN_CATEGORIES["falling_wedge"],
            end, df.iloc[end]["date"], 0.6, resist,
            {"high_slope": hs, "low_slope": ls},
        ))
    return _dedupe_by_end(matches, min_gap=15)


def detect_rising_wedge(
    df: pd.DataFrame,
    swing_highs: list[SwingPoint],
    swing_lows: list[SwingPoint],
    lookback: int = 40,
    min_swings: int = 3,
) -> list[PatternMatch]:
    """Both trendlines rise; upper rises slower → bearish breakdown."""
    matches = []
    n = len(df)
    for end in range(lookback, n):
        start = end - lookback
        highs = [s for s in swing_highs if start <= s.idx <= end]
        lows = [s for s in swing_lows if start <= s.idx <= end]
        if len(highs) < min_swings or len(lows) < min_swings:
            continue
        hx = np.array([h.idx - start for h in highs])
        hy = np.array([h.price for h in highs])
        lx = np.array([l.idx - start for l in lows])
        ly = np.array([l.price for l in lows])
        hs, ls = _slope(hx, hy), _slope(lx, ly)
        if hs <= 0 or ls <= 0:
            continue
        if hs >= ls:  # upper rises slower → converging
            continue
        support = float(ly[-1])
        if df.iloc[end]["close"] >= support:
            continue
        matches.append(PatternMatch(
            "rising_wedge", PATTERN_CATEGORIES["rising_wedge"],
            end, df.iloc[end]["date"], 0.6, support,
            {"high_slope": hs, "low_slope": ls},
        ))
    return _dedupe_by_end(matches, min_gap=15)


# ---------------------------------------------------------------------------
# Reversals — candlesticks
# ---------------------------------------------------------------------------

def detect_hammer(df: pd.DataFrame, lookback_trend: int = 10) -> list[PatternMatch]:
    matches = []
    for i in range(lookback_trend, len(df)):
        row = df.iloc[i]
        o, h, l, c = row["open"], row["high"], row["low"], row["close"]
        body = abs(c - o)
        rng = h - l
        if rng <= 0 or body <= 0:
            continue
        lower_shadow = min(o, c) - l
        upper_shadow = h - max(o, c)
        if lower_shadow < 2 * body or upper_shadow > body:
            continue
        prior = df.iloc[i - lookback_trend : i]["close"]
        if prior.iloc[-1] >= prior.iloc[0]:  # need prior downtrend
            continue
        matches.append(PatternMatch(
            "hammer", PATTERN_CATEGORIES["hammer"],
            i, row["date"], 0.55, None,
            {"body_ratio": body / rng},
        ))
    return matches


def detect_shooting_star(df: pd.DataFrame, lookback_trend: int = 10) -> list[PatternMatch]:
    matches = []
    for i in range(lookback_trend, len(df)):
        row = df.iloc[i]
        o, h, l, c = row["open"], row["high"], row["low"], row["close"]
        body = abs(c - o)
        rng = h - l
        if rng <= 0 or body <= 0:
            continue
        upper_shadow = h - max(o, c)
        lower_shadow = min(o, c) - l
        if upper_shadow < 2 * body or lower_shadow > body:
            continue
        prior = df.iloc[i - lookback_trend : i]["close"]
        if prior.iloc[-1] <= prior.iloc[0]:  # need prior uptrend
            continue
        matches.append(PatternMatch(
            "shooting_star", PATTERN_CATEGORIES["shooting_star"],
            i, row["date"], 0.55, None,
            {"body_ratio": body / rng},
        ))
    return matches


# ---------------------------------------------------------------------------
# Continuations — flags & triangles
# ---------------------------------------------------------------------------

def detect_bull_flag(
    df: pd.DataFrame,
    pole_bars: int = 10,
    pole_min_ret: float = 0.03,
    flag_bars: int = 15,
    flag_max_ret: float = 0.02,
) -> list[PatternMatch]:
    matches = []
    for end in range(pole_bars + flag_bars, len(df)):
        pole_start = end - pole_bars - flag_bars
        pole_end = end - flag_bars
        pole_ret = (df.iloc[pole_end]["close"] - df.iloc[pole_start]["close"]) / df.iloc[pole_start]["close"]
        if pole_ret < pole_min_ret:
            continue
        flag = df.iloc[pole_end : end + 1]
        flag_ret = (flag.iloc[-1]["close"] - flag.iloc[0]["close"]) / flag.iloc[0]["close"]
        if flag_ret > 0 or abs(flag_ret) > flag_max_ret:
            continue
        x = np.arange(len(flag))
        hs = _slope(x, flag["high"].values)
        ls = _slope(x, flag["low"].values)
        if hs > 0 or ls > 0:  # drift down channel
            continue
        if df.iloc[end]["close"] <= flag["high"].max():
            continue
        matches.append(PatternMatch(
            "bull_flag", PATTERN_CATEGORIES["bull_flag"],
            end, df.iloc[end]["date"], 0.65,
            float(flag["high"].max()),
            {"pole_ret": pole_ret, "flag_ret": flag_ret},
        ))
    return _dedupe_by_end(matches, min_gap=20)


def detect_bear_flag(
    df: pd.DataFrame,
    pole_bars: int = 10,
    pole_min_ret: float = 0.03,
    flag_bars: int = 15,
    flag_max_ret: float = 0.02,
) -> list[PatternMatch]:
    matches = []
    for end in range(pole_bars + flag_bars, len(df)):
        pole_start = end - pole_bars - flag_bars
        pole_end = end - flag_bars
        pole_ret = (df.iloc[pole_end]["close"] - df.iloc[pole_start]["close"]) / df.iloc[pole_start]["close"]
        if pole_ret > -pole_min_ret:
            continue
        flag = df.iloc[pole_end : end + 1]
        flag_ret = (flag.iloc[-1]["close"] - flag.iloc[0]["close"]) / flag.iloc[0]["close"]
        if flag_ret < 0 or abs(flag_ret) > flag_max_ret:
            continue
        x = np.arange(len(flag))
        hs = _slope(x, flag["high"].values)
        ls = _slope(x, flag["low"].values)
        if hs < 0 or ls < 0:
            continue
        if df.iloc[end]["close"] >= flag["low"].min():
            continue
        matches.append(PatternMatch(
            "bear_flag", PATTERN_CATEGORIES["bear_flag"],
            end, df.iloc[end]["date"], 0.65,
            float(flag["low"].min()),
            {"pole_ret": pole_ret, "flag_ret": flag_ret},
        ))
    return _dedupe_by_end(matches, min_gap=20)


def detect_ascending_triangle(
    df: pd.DataFrame,
    swing_highs: list[SwingPoint],
    swing_lows: list[SwingPoint],
    lookback: int = 50,
    flat_tol: float = 0.012,
) -> list[PatternMatch]:
    matches = []
    n = len(df)
    for end in range(lookback, n):
        start = end - lookback
        highs = [s for s in swing_highs if start <= s.idx <= end]
        lows = [s for s in swing_lows if start <= s.idx <= end]
        if len(highs) < 2 or len(lows) < 2:
            continue
        flat_top = np.mean([h.price for h in highs])
        if max(abs(h.price - flat_top) / flat_top for h in highs) > flat_tol:
            continue
        lx = np.array([l.idx for l in lows])
        ly = np.array([l.price for l in lows])
        if _slope(lx, ly) <= 0:
            continue
        resist = float(flat_top)
        if df.iloc[end]["close"] <= resist:
            continue
        matches.append(PatternMatch(
            "ascending_triangle", PATTERN_CATEGORIES["ascending_triangle"],
            end, df.iloc[end]["date"], 0.6, resist,
            {"resistance": resist, "low_slope": _slope(lx, ly)},
        ))
    return _dedupe_by_end(matches, min_gap=20)


def detect_descending_triangle(
    df: pd.DataFrame,
    swing_highs: list[SwingPoint],
    swing_lows: list[SwingPoint],
    lookback: int = 50,
    flat_tol: float = 0.012,
) -> list[PatternMatch]:
    matches = []
    n = len(df)
    for end in range(lookback, n):
        start = end - lookback
        highs = [s for s in swing_highs if start <= s.idx <= end]
        lows = [s for s in swing_lows if start <= s.idx <= end]
        if len(highs) < 2 or len(lows) < 2:
            continue
        flat_bot = np.mean([l.price for l in lows])
        if max(abs(l.price - flat_bot) / flat_bot for l in lows) > flat_tol:
            continue
        hx = np.array([h.idx for h in highs])
        hy = np.array([h.price for h in highs])
        if _slope(hx, hy) >= 0:
            continue
        support = float(flat_bot)
        if df.iloc[end]["close"] >= support:
            continue
        matches.append(PatternMatch(
            "descending_triangle", PATTERN_CATEGORIES["descending_triangle"],
            end, df.iloc[end]["date"], 0.6, support,
            {"support": support, "high_slope": _slope(hx, hy)},
        ))
    return _dedupe_by_end(matches, min_gap=20)


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------

def _dedupe_by_end(matches: list[PatternMatch], min_gap: int) -> list[PatternMatch]:
    if not matches:
        return []
    matches = sorted(matches, key=lambda m: m.end_idx)
    kept = [matches[0]]
    for m in matches[1:]:
        if m.end_idx - kept[-1].end_idx >= min_gap:
            kept.append(m)
    return kept


def scan_all_patterns(daily: pd.DataFrame, swing_window: int = 5) -> list[PatternMatch]:
    df = ohlc_frame(daily)
    swing_highs, swing_lows = find_swings(df, window=swing_window)

    all_matches: list[PatternMatch] = []
    all_matches.extend(detect_double_top(df, swing_highs))
    all_matches.extend(detect_double_bottom(df, swing_lows))
    all_matches.extend(detect_head_shoulders(df, swing_highs))
    all_matches.extend(detect_inverse_head_shoulders(df, swing_lows))
    all_matches.extend(detect_falling_wedge(df, swing_highs, swing_lows))
    all_matches.extend(detect_rising_wedge(df, swing_highs, swing_lows))
    all_matches.extend(detect_hammer(df))
    all_matches.extend(detect_shooting_star(df))
    all_matches.extend(detect_bull_flag(df))
    all_matches.extend(detect_bear_flag(df))
    all_matches.extend(detect_ascending_triangle(df, swing_highs, swing_lows))
    all_matches.extend(detect_descending_triangle(df, swing_highs, swing_lows))

    return sorted(all_matches, key=lambda m: m.end_idx)
