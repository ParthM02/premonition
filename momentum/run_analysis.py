"""
Momentum trading backtests on SPY/QQQ daily RTH data.

Usage:
    python momentum/run_analysis.py
    python momentum/run_analysis.py --cost-bps 15
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modeling.common import DATA_DIR, RESULTS_DIR, SYMBOLS, load_symbol_data
from momentum.backtest import (
    DEFAULT_COST_BPS,
    backtest_cross_sectional,
    run_single_asset_backtests,
    _perf_stats,
)
from momentum.walkforward import (
    run_vol_momentum_comparison,
    walk_forward_tsmom,
)


def load_daily(symbol: str) -> pd.DataFrame:
    path = DATA_DIR / f"{symbol.lower()}_daily.csv"
    return pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)


def print_report(
    results: list[dict],
    xs_result: dict | None,
    wf_results: list[dict] | None = None,
    vol_cmp: list[dict] | None = None,
) -> None:
    print("\n" + "=" * 72)
    print("  MOMENTUM TRADING BACKTESTS  (net of transaction costs)")
    print("=" * 72)

    df = pd.DataFrame(results)
    if df.empty:
        print("No results.")
        return

    for symbol in df["symbol"].unique():
        sub = df[df["symbol"] == symbol].sort_values("sharpe", ascending=False)
        print(f"\n--- {symbol} (full sample) ---")
        print(f"  {'Strategy':<26} {'AnnRet':>7} {'Sharpe':>7} {'MaxDD':>7} {'Hit%':>6} {'vs B&H':>8} {'Holdout':>8}")
        print("  " + "-" * 68)
        for _, row in sub.iterrows():
            print(
                f"  {row['strategy']:<26} "
                f"{row['ann_return']*100:>+6.1f}% "
                f"{row['sharpe']:>7.2f} "
                f"{row['max_drawdown']*100:>6.1f}% "
                f"{row['hit_rate']*100:>5.1f}% "
                f"{row['excess_vs_bh']*100:>+7.1f}% "
                f"{row['holdout_return']*100:>+7.1f}%"
            )
        bh = sub.iloc[0]["buy_hold_return"]
        print(f"\n  Buy & hold total return: {bh*100:+.1f}%")

    if xs_result:
        print(f"\n--- Cross-sectional SPY vs QQQ ({xs_result['lookback']}d lookback) ---")
        print(f"  Ann return:  {xs_result['ann_return']*100:+.2f}%")
        print(f"  Sharpe:      {xs_result['sharpe']:.2f}")
        print(f"  Max DD:      {xs_result['max_drawdown']*100:.1f}%")
        print(f"  vs blended B&H: {xs_result['excess_vs_bh']*100:+.1f}%")
        print(f"  Holdout:     {xs_result['holdout_return']*100:+.1f}%  (excess {xs_result['holdout_excess']*100:+.1f}%)")

    if vol_cmp:
        print("\n--- Momentum + vol scaling (20d, full sample) ---")
        vdf = pd.DataFrame(vol_cmp)
        for symbol in vdf["symbol"].unique():
            sub = vdf[vdf["symbol"] == symbol]
            print(f"  {symbol}:")
            for _, row in sub.iterrows():
                print(
                    f"    {row['strategy']:<24} Sharpe {row['sharpe']:.2f}  "
                    f"Ann {row['ann_return']*100:+.1f}%  MaxDD {row['max_drawdown']*100:.1f}%"
                )

    if wf_results:
        print("\n--- Walk-forward tuned tsmom (OOS from day 252) ---")
        print(f"  {'Strategy':<24} {'Sym':>4} {'OOS Sh':>7} {'OOS Ret':>8} {'Holdout':>8} {'Static LB':>10} {'Static Sh':>10}")
        print("  " + "-" * 72)
        for row in wf_results:
            print(
                f"  {row['strategy']:<24} {row['symbol']:>4} "
                f"{row['oos_sharpe']:>7.2f} "
                f"{row['oos_total_return']*100:>+7.1f}% "
                f"{row['holdout_return']*100:>+7.1f}% "
                f"{row['best_static_lookback']:>10}d "
                f"{row['static_sharpe']:>10.2f}"
            )


def main():
    parser = argparse.ArgumentParser(description="Momentum strategy backtests")
    parser.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS,
                        help="Round-trip cost in bps per position change (default 10)")
    parser.add_argument("--lookback", type=int, default=20, help="XS momentum lookback")
    args = parser.parse_args()

    all_results: list[dict] = []
    wf_results: list[dict] = []
    vol_cmp: list[dict] = []
    dailies = {s: load_daily(s) for s in SYMBOLS}
    vol_data = {s: load_symbol_data(s)["realized_vol"] for s in SYMBOLS}

    for symbol in SYMBOLS:
        print(f"Backtesting {symbol}...")
        bt = run_single_asset_backtests(dailies[symbol], symbol, args.cost_bps)
        all_results.extend([r.to_dict() for r in bt])

        print(f"  Walk-forward tune {symbol}...")
        wf = walk_forward_tsmom(dailies[symbol], symbol, cost_bps=args.cost_bps)
        wf_vol = walk_forward_tsmom(
            dailies[symbol], symbol, cost_bps=args.cost_bps,
            vol_scaled=True, realized_vol=vol_data[symbol],
        )
        wf_results.extend([wf.to_dict(), wf_vol.to_dict()])

        vol_cmp.extend(run_vol_momentum_comparison(
            dailies[symbol], symbol, vol_data[symbol], args.cost_bps
        ))

    # Cross-sectional
    print("Backtesting cross-sectional SPY vs QQQ...")
    xs_net, xs_pos = backtest_cross_sectional(
        dailies["SPY"], dailies["QQQ"], lookback=args.lookback, cost_bps=args.cost_bps
    )
    merged_bh = (
        dailies["SPY"]["rth_return"].values + dailies["QQQ"]["rth_return"].values
    ) / 2
    bh = pd.Series(merged_bh, index=xs_net.index)
    xs_stats = _perf_stats(xs_net, xs_pos, bh, cost_bps=args.cost_bps)
    xs_result = {"lookback": args.lookback, **xs_stats}

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cost_bps": args.cost_bps,
        "single_asset": all_results,
        "cross_sectional": xs_result,
        "walk_forward": wf_results,
        "vol_momentum_comparison": vol_cmp,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "momentum_analysis.json"
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    print_report(all_results, xs_result, wf_results, vol_cmp)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
