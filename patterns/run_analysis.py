"""
Scan SPY/QQQ daily RTH bars for chart patterns and evaluate forward returns.

Usage:
    python patterns/run_analysis.py
    python patterns/run_analysis.py --symbol SPY
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modeling.common import DATA_DIR, SYMBOLS
from patterns.evaluate import (
    PatternStats,
    evaluate_symbol,
    save_pattern_results,
    summarize_category_performance,
)


def load_daily(symbol: str) -> pd.DataFrame:
    path = DATA_DIR / f"{symbol.lower()}_daily.csv"
    return pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)


def print_report(results: dict) -> None:
    print("\n" + "=" * 64)
    print("  CHART PATTERN ANALYSIS")
    print("=" * 64)

    for sym_result in results["symbols"]:
        symbol = sym_result["symbol"]
        print(f"\n--- {symbol} ---")
        print(f"  Total detections: {sym_result['total_detections']}")
        print(f"  By category: {sym_result['by_category']}")
        print(f"  By pattern:  {sym_result['by_pattern']}")

        stats = sym_result["forward_stats"]
        if not stats:
            print("  (insufficient detections for forward-return stats)")
            continue

        df = pd.DataFrame(stats)
        # Show 10-day horizon summary
        h10 = df[df["horizon"] == 10].sort_values("count", ascending=False)
        if h10.empty:
            h10 = df.sort_values("count", ascending=False)

        print(f"\n  Forward returns (10d horizon where available):")
        print(f"  {'Pattern':<26} {'N':>4} {'Mean':>8} {'Hit%':>6} {'Base%':>6} {'Excess':>8} {'p':>7}")
        print("  " + "-" * 62)
        for _, row in h10.head(12).iterrows():
            p = row.get("p_value")
            ps = f"{p:.3f}" if p is not None and pd.notna(p) else "  n/a"
            print(
                f"  {row['pattern']:<26} {int(row['count']):>4} "
                f"{row['mean_forward_return']*100:>+7.2f}% "
                f"{row['hit_rate']*100:>5.1f}% "
                f"{row['baseline_hit_rate']*100:>5.1f}% "
                f"{row['excess_return']*100:>+7.2f}% "
                f"{ps:>7}"
            )

    # Cross-symbol category rollup at 10d
    all_stats = []
    for sym_result in results["symbols"]:
        for s in sym_result["forward_stats"]:
            all_stats.append(PatternStats(**s))

    if all_stats:
        rollup = summarize_category_performance(all_stats)
        h10 = rollup[rollup["horizon"] == 10]
        if not h10.empty:
            print("\n--- Category rollup (10d, all symbols) ---")
            cat = h10.groupby("category").agg(
                count=("count", "sum"),
                mean_fwd=("mean_fwd_ret", "mean"),
                hit_rate=("hit_rate", "mean"),
                baseline_hit=("baseline_hit", "mean"),
            )
            for cat_name, row in cat.iterrows():
                edge = row["hit_rate"] - row["baseline_hit"]
                print(
                    f"  {cat_name:<25} n={int(row['count']):>4}  "
                    f"mean={row['mean_fwd']*100:+.2f}%  "
                    f"hit={row['hit_rate']*100:.1f}% (base {row['baseline_hit']*100:.1f}%, edge {edge*100:+.1f}pp)"
                )


def main():
    parser = argparse.ArgumentParser(description="Chart pattern detection and testing")
    parser.add_argument("--symbol", choices=SYMBOLS, action="append", dest="symbols")
    args = parser.parse_args()
    symbols = args.symbols or SYMBOLS

    payload = {"symbols": []}
    for symbol in symbols:
        print(f"Scanning {symbol}...")
        daily = load_daily(symbol)
        sym_result = evaluate_symbol(daily, symbol)
        payload["symbols"].append(sym_result)

    path = save_pattern_results(payload)
    print_report(payload)
    print(f"\nSaved → {path}")


if __name__ == "__main__":
    main()
