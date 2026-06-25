"""Forward-return evaluation for detected chart patterns."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from patterns.detectors import PATTERN_CATEGORIES, PatternMatch, scan_all_patterns
from patterns.swings import ohlc_frame

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"

FORWARD_HORIZONS = (5, 10, 20)
BULLISH_CATEGORIES = {"bullish_reversal", "bullish_continuation"}
BEARISH_CATEGORIES = {"bearish_reversal", "bearish_continuation"}


@dataclass
class PatternStats:
    pattern: str
    category: str
    symbol: str
    count: int
    horizon: int
    mean_forward_return: float
    median_forward_return: float
    hit_rate: float
    baseline_mean: float
    baseline_hit_rate: float
    excess_return: float
    t_stat: Optional[float]
    p_value: Optional[float]


def forward_return(closes: np.ndarray, idx: int, horizon: int) -> Optional[float]:
    if idx + horizon >= len(closes):
        return None
    return float((closes[idx + horizon] - closes[idx]) / closes[idx])


def evaluate_matches(
    daily: pd.DataFrame,
    matches: list[PatternMatch],
    symbol: str,
    horizons: tuple[int, ...] = FORWARD_HORIZONS,
) -> list[PatternStats]:
    df = ohlc_frame(daily)
    closes = df["close"].values
    results: list[PatternStats] = []

    by_pattern: dict[str, list[PatternMatch]] = {}
    for m in matches:
        by_pattern.setdefault(m.pattern, []).append(m)

    for pattern, pmatches in by_pattern.items():
        category = PATTERN_CATEGORIES[pattern]
        bullish = category in BULLISH_CATEGORIES

        for horizon in horizons:
            fwd: list[float] = []
            for m in pmatches:
                r = forward_return(closes, m.end_idx, horizon)
                if r is not None:
                    fwd.append(r)

            if len(fwd) < 3:
                continue

            fwd_arr = np.array(fwd)
            if bullish:
                hits = fwd_arr > 0
            else:
                hits = fwd_arr < 0

            # Baseline: all valid confirmation bars with enough forward data
            baseline_fwd = []
            for i in range(len(closes) - horizon):
                r = forward_return(closes, i, horizon)
                if r is not None:
                    baseline_fwd.append(r)
            base = np.array(baseline_fwd)
            base_hits = base > 0 if bullish else base < 0

            t_stat, p_val = None, None
            if len(fwd) >= 5:
                t_stat, p_val = stats.ttest_1samp(fwd_arr, 0.0 if bullish else 0.0)
                if not bullish:
                    # test if mean < 0
                    t_stat, p_val = stats.ttest_1samp(fwd_arr, 0.0)

            results.append(PatternStats(
                pattern=pattern,
                category=category,
                symbol=symbol,
                count=len(fwd),
                horizon=horizon,
                mean_forward_return=float(np.mean(fwd_arr)),
                median_forward_return=float(np.median(fwd_arr)),
                hit_rate=float(np.mean(hits)),
                baseline_mean=float(np.mean(base)),
                baseline_hit_rate=float(np.mean(base_hits)),
                excess_return=float(np.mean(fwd_arr) - np.mean(base)),
                t_stat=float(t_stat) if t_stat is not None else None,
                p_value=float(p_val) if p_val is not None else None,
            ))

    return results


def evaluate_symbol(daily: pd.DataFrame, symbol: str) -> dict:
    matches = scan_all_patterns(daily)
    stats_list = evaluate_matches(daily, matches, symbol)

    by_category: dict[str, int] = {}
    for m in matches:
        by_category[m.category] = by_category.get(m.category, 0) + 1

    return {
        "symbol": symbol,
        "total_detections": len(matches),
        "by_category": by_category,
        "by_pattern": _count_by_pattern(matches),
        "forward_stats": [asdict(s) for s in stats_list],
        "recent_matches": [
            {
                "pattern": m.pattern,
                "category": m.category,
                "date": str(m.end_date),
                "confidence": m.confidence,
                "neckline": m.neckline,
            }
            for m in matches[-15:]
        ],
    }


def _count_by_pattern(matches: list[PatternMatch]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for m in matches:
        counts[m.pattern] = counts.get(m.pattern, 0) + 1
    return counts


def save_pattern_results(payload: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    path = RESULTS_DIR / "pattern_analysis.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return path


def summarize_category_performance(all_stats: list[PatternStats]) -> pd.DataFrame:
    rows = []
    for s in all_stats:
        rows.append({
            "category": s.category,
            "symbol": s.symbol,
            "horizon": s.horizon,
            "patterns": s.pattern,
            "count": s.count,
            "mean_fwd_ret": s.mean_forward_return,
            "hit_rate": s.hit_rate,
            "baseline_hit": s.baseline_hit_rate,
            "excess": s.excess_return,
            "p_value": s.p_value,
        })
    return pd.DataFrame(rows)
