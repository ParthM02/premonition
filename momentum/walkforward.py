"""Walk-forward parameter tuning for time-series momentum."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from momentum.backtest import (
    DEFAULT_COST_BPS,
    _perf_stats,
    backtest_tsmom,
    backtest_tsmom_vol_scaled,
)

INITIAL_TRAIN = 252
ROLL_STEP = 21
DEFAULT_LOOKBACKS = (10, 15, 20, 30, 40, 60)


@dataclass
class WalkForwardResult:
    symbol: str
    strategy: str
    oos_sharpe: float
    oos_ann_return: float
    oos_max_dd: float
    oos_total_return: float
    oos_excess_vs_bh: float
    holdout_sharpe: float
    holdout_return: float
    holdout_excess: float
    n_oos_days: int
    param_history: list[dict]
    best_static_lookback: int
    static_sharpe: float

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "strategy": self.strategy,
            "oos_sharpe": self.oos_sharpe,
            "oos_ann_return": self.oos_ann_return,
            "oos_max_dd": self.oos_max_dd,
            "oos_total_return": self.oos_total_return,
            "oos_excess_vs_bh": self.oos_excess_vs_bh,
            "holdout_sharpe": self.holdout_sharpe,
            "holdout_return": self.holdout_return,
            "holdout_excess": self.holdout_excess,
            "n_oos_days": self.n_oos_days,
            "param_history": self.param_history[-10:],
            "best_static_lookback": self.best_static_lookback,
            "static_sharpe": self.static_sharpe,
        }


def _oos_segment_returns(
    daily: pd.DataFrame,
    lookback: int,
    start: int,
    end: int,
    cost_bps: float,
    vol_scaled: bool = False,
    realized_vol: pd.Series | None = None,
) -> tuple[list[float], list[float]]:
    """Generate OOS daily returns for days [start, end) using fixed lookback."""
    ret = daily["rth_return"].values
    close = daily["rth_close"].values
    vol = realized_vol.values if realized_vol is not None else None

    returns: list[float] = []
    positions: list[float] = []
    prev_pos = 0.0

    for i in range(start, end):
        if i < lookback:
            returns.append(0.0)
            positions.append(0.0)
            continue

        mom = close[i - 1] / close[i - 1 - lookback] - 1
        pos = 1.0 if mom > 0 else 0.0

        if vol_scaled and vol is not None and i >= 21:
            v = vol[i - 1]
            v_med = np.nanmedian(vol[max(0, i - 252) : i])
            if v > 0 and v_med > 0:
                scale = float(np.clip(v_med / v, 0.5, 1.5))
                pos *= scale

        gross = prev_pos * ret[i]
        cost = abs(pos - prev_pos) * cost_bps / 10_000
        returns.append(gross - cost)
        positions.append(pos)
        prev_pos = pos

    return returns, positions


def walk_forward_tsmom(
    daily: pd.DataFrame,
    symbol: str,
    lookbacks: tuple[int, ...] = DEFAULT_LOOKBACKS,
    train_min: int = INITIAL_TRAIN,
    step: int = ROLL_STEP,
    cost_bps: float = DEFAULT_COST_BPS,
    vol_scaled: bool = False,
    realized_vol: pd.Series | None = None,
) -> WalkForwardResult:
    """
    Expanding-window walk-forward: every `step` days, pick best lookback on
    train history by in-sample Sharpe, then trade OOS for the next `step` days.
    """
    n = len(daily)
    bh = daily["rth_return"]

    oos_returns: list[float] = []
    oos_positions: list[float] = []
    param_history: list[dict] = []

    # Also find best static lookback on full pre-holdout sample for comparison
    static_best_lb, static_best_sharpe = 20, -np.inf
    for lb in lookbacks:
        net, pos = backtest_tsmom(daily, lb, hold=1, cost_bps=cost_bps)
        stats = _perf_stats(net, pos, bh, cost_bps=cost_bps)
        if stats and stats["sharpe"] > static_best_sharpe:
            static_best_sharpe = stats["sharpe"]
            static_best_lb = lb

    t = train_min
    while t < n:
        train_slice = daily.iloc[:t]
        best_lb, best_sharpe = 20, -np.inf

        for lb in lookbacks:
            if t <= lb + 30:
                continue
            net, pos = backtest_tsmom(train_slice, lb, hold=1, cost_bps=cost_bps)
            stats = _perf_stats(net, pos, train_slice["rth_return"], cost_bps=cost_bps)
            if stats and stats["sharpe"] > best_sharpe:
                best_sharpe = stats["sharpe"]
                best_lb = lb

        test_end = min(t + step, n)
        seg_ret, seg_pos = _oos_segment_returns(
            daily, best_lb, t, test_end, cost_bps, vol_scaled, realized_vol
        )
        oos_returns.extend(seg_ret)
        oos_positions.extend(seg_pos)
        param_history.append({
            "oos_start": str(daily.iloc[t]["date"]),
            "lookback": best_lb,
            "train_sharpe": best_sharpe,
        })
        t = test_end

    oos_series = pd.Series(oos_returns)
    pos_series = pd.Series(oos_positions)
    bh_oos = bh.iloc[train_min : train_min + len(oos_series)].reset_index(drop=True)
    oos_series.index = bh_oos.index
    pos_series.index = bh_oos.index

    stats = _perf_stats(oos_series, pos_series, bh_oos, cost_bps=cost_bps)
    strat_name = "wf_tsmom_vol_scaled" if vol_scaled else "wf_tsmom"

    return WalkForwardResult(
        symbol=symbol,
        strategy=strat_name,
        oos_sharpe=stats["sharpe"],
        oos_ann_return=stats["ann_return"],
        oos_max_dd=stats["max_drawdown"],
        oos_total_return=stats["total_return"],
        oos_excess_vs_bh=stats["excess_vs_bh"],
        holdout_sharpe=stats["holdout_sharpe"],
        holdout_return=stats["holdout_return"],
        holdout_excess=stats["holdout_excess"],
        n_oos_days=len(oos_series),
        param_history=param_history,
        best_static_lookback=static_best_lb,
        static_sharpe=static_best_sharpe,
    )


def run_vol_momentum_comparison(
    daily: pd.DataFrame,
    symbol: str,
    realized_vol: pd.Series,
    cost_bps: float = DEFAULT_COST_BPS,
) -> list[dict]:
    """Compare plain 20d tsmom vs vol-scaled variant (full sample)."""
    bh = daily["rth_return"]
    rows = []

    for name, fn in [
        ("tsmom_20d", lambda: backtest_tsmom(daily, 20, hold=1, cost_bps=cost_bps)),
        ("tsmom_20d_vol_scaled", lambda: backtest_tsmom_vol_scaled(
            daily, lookback=20, realized_vol=realized_vol, cost_bps=cost_bps
        )),
    ]:
        net, pos = fn()
        stats = _perf_stats(net, pos, bh, cost_bps=cost_bps)
        rows.append({"strategy": name, "symbol": symbol, **stats})

    return rows
