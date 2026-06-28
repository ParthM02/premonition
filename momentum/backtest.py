"""
Momentum strategy signals and vectorized backtests on daily RTH data.

Strategies:
  - Time-series momentum (5/20/60d lookback, 1d hold)
  - 12-1 month momentum (skip recent 21 days)
  - 52-week high proximity
  - Cross-sectional SPY vs QQQ relative strength
  - Dual momentum (absolute + relative filter)
  - Premarket → RTH session continuation (same-day)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS_YEAR = 252
DEFAULT_COST_BPS = 10  # round-trip per position change


@dataclass
class BacktestResult:
    strategy: str
    symbol: str
    horizon: str
    n_trades: int
    ann_return: float
    ann_vol: float
    sharpe: float
    max_drawdown: float
    hit_rate: float
    avg_trade_return: float
    total_return: float
    buy_hold_return: float
    excess_vs_bh: float
    holdout_return: float
    holdout_sharpe: float
    holdout_excess: float

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "symbol": self.symbol,
            "horizon": self.horizon,
            "n_trades": self.n_trades,
            "ann_return": self.ann_return,
            "ann_vol": self.ann_vol,
            "sharpe": self.sharpe,
            "max_drawdown": self.max_drawdown,
            "hit_rate": self.hit_rate,
            "avg_trade_return": self.avg_trade_return,
            "total_return": self.total_return,
            "buy_hold_return": self.buy_hold_return,
            "excess_vs_bh": self.excess_vs_bh,
            "holdout_return": self.holdout_return,
            "holdout_sharpe": self.holdout_sharpe,
            "holdout_excess": self.holdout_excess,
        }


def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1
    return float(dd.min())


def _perf_stats(
    strat_returns: pd.Series,
    positions: pd.Series,
    bh_returns: pd.Series,
    holdout_days: int = 126,
    cost_bps: float = DEFAULT_COST_BPS,
) -> dict:
    """Compute performance from daily strategy returns (already net of costs)."""
    strat_returns = strat_returns.dropna()
    if strat_returns.empty:
        return {}

    equity = (1 + strat_returns).cumprod()
    ann_ret = float((equity.iloc[-1]) ** (TRADING_DAYS_YEAR / len(strat_returns)) - 1)
    ann_vol = float(strat_returns.std() * np.sqrt(TRADING_DAYS_YEAR))
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

    bh_eq = (1 + bh_returns.loc[strat_returns.index]).cumprod()
    bh_total = float(bh_eq.iloc[-1] - 1)

    # Trades: position changes
    pos = positions.loc[strat_returns.index].fillna(0)
    trades = int((pos.diff().abs() > 0).sum())

    # Hit rate on active days only
    active = pos != 0
    if active.any():
        hit = float((strat_returns[active] > 0).mean())
        avg_trade = float(strat_returns[active].mean())
    else:
        hit, avg_trade = 0.0, 0.0

    holdout = strat_returns.iloc[-holdout_days:] if len(strat_returns) > holdout_days else strat_returns
    bh_holdout = bh_returns.loc[holdout.index]
    h_eq = (1 + holdout).cumprod()
    h_ret = float(h_eq.iloc[-1] - 1) if len(h_eq) else 0.0
    h_vol = float(holdout.std() * np.sqrt(TRADING_DAYS_YEAR)) if len(holdout) > 1 else 0.0
    h_sharpe = h_ret / h_vol if h_vol > 0 else 0.0
    bh_h_eq = (1 + bh_holdout).cumprod()
    bh_h_ret = float(bh_h_eq.iloc[-1] - 1) if len(bh_h_eq) else 0.0

    return {
        "n_trades": trades,
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": _max_drawdown(equity),
        "hit_rate": hit,
        "avg_trade_return": avg_trade,
        "total_return": float(equity.iloc[-1] - 1),
        "buy_hold_return": bh_total,
        "excess_vs_bh": float(equity.iloc[-1] - 1) - bh_total,
        "holdout_return": h_ret,
        "holdout_sharpe": h_sharpe,
        "holdout_excess": h_ret - bh_h_ret,
    }


def _apply_costs(position: pd.Series, gross_ret: pd.Series, cost_bps: float) -> pd.Series:
    turnover = position.diff().abs().fillna(position.abs())
    cost = turnover * (cost_bps / 10_000)
    return gross_ret - cost


def backtest_tsmom(
    daily: pd.DataFrame,
    lookback: int,
    hold: int = 1,
    long_only: bool = True,
    cost_bps: float = DEFAULT_COST_BPS,
) -> tuple[pd.Series, pd.Series]:
    """
    Time-series momentum: go long if past `lookback` return > 0 (else flat/short).
    Position held for `hold` days (non-overlapping rebalance every `hold` days).
    """
    ret = daily["rth_return"]
    mom = daily["rth_close"].pct_change(lookback)

    signal = np.sign(mom)
    if long_only:
        signal = signal.clip(lower=0)

    # Rebalance every `hold` days: forward-fill signal
    position = pd.Series(0.0, index=daily.index)
    for i in range(lookback, len(daily), hold):
        position.iloc[i : i + hold] = signal.iloc[i]

    gross = position.shift(1).fillna(0) * ret
    net = _apply_costs(position.shift(1).fillna(0), gross, cost_bps)
    return net, position


def backtest_12_1_mom(
    daily: pd.DataFrame,
    lookback: int = 252,
    skip: int = 21,
    cost_bps: float = DEFAULT_COST_BPS,
) -> tuple[pd.Series, pd.Series]:
    """Classic 12-1 momentum: return from t-252 to t-21."""
    close = daily["rth_close"]
    mom = close.shift(skip) / close.shift(lookback) - 1
    position = (mom > 0).astype(float)
    gross = position.shift(1).fillna(0) * daily["rth_return"]
    net = _apply_costs(position.shift(1).fillna(0), gross, cost_bps)
    return net, position


def backtest_52w_high(
    daily: pd.DataFrame,
    window: int = 252,
    threshold: float = 0.95,
    cost_bps: float = DEFAULT_COST_BPS,
) -> tuple[pd.Series, pd.Series]:
    """Long when close is within `threshold` of rolling 52-week high."""
    close = daily["rth_close"]
    rolling_max = close.rolling(window).max()
    proximity = close / rolling_max
    position = (proximity >= threshold).astype(float)
    gross = position.shift(1).fillna(0) * daily["rth_return"]
    net = _apply_costs(position.shift(1).fillna(0), gross, cost_bps)
    return net, position


def backtest_premarket_continuation(
    daily: pd.DataFrame,
    cost_bps: float = DEFAULT_COST_BPS,
) -> tuple[pd.Series, pd.Series]:
    """Same-day: long RTH if premarket return > 0, else flat."""
    position = (daily["premarket_return"] > 0).astype(float)
    gross = position * daily["rth_return"]  # same-day, no shift
    net = gross - position.diff().abs().fillna(0) * (cost_bps / 10_000)
    return net, position


def backtest_first30_continuation(
    daily: pd.DataFrame,
    cost_bps: float = DEFAULT_COST_BPS,
) -> tuple[pd.Series, pd.Series]:
    """Same-day: bet rest-of-RTH continues first-30-min direction (enter at 10:00)."""
    position = (daily["first30_rth_return"] > 0).astype(float)
    gross = position * daily["rest_rth_return"]
    net = gross - position.diff().abs().fillna(0) * (cost_bps / 10_000)
    return net, position


def backtest_cross_sectional(
    spy: pd.DataFrame,
    qqq: pd.DataFrame,
    lookback: int = 20,
    cost_bps: float = DEFAULT_COST_BPS,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Relative momentum: each day hold the asset with stronger past `lookback` return.
    Returns (spy_net, qqq_net, winner_label) where winner_label is 1=SPY, -1=QQQ, 0=cash.
    """
    merged = pd.merge(
        spy[["date", "rth_close", "rth_return"]].rename(
            columns={"rth_close": "spy_close", "rth_return": "spy_ret"}
        ),
        qqq[["date", "rth_close", "rth_return"]].rename(
            columns={"rth_close": "qqq_close", "rth_return": "qqq_ret"}
        ),
        on="date",
    )
    spy_mom = merged["spy_close"].pct_change(lookback)
    qqq_mom = merged["qqq_close"].pct_change(lookback)

    spy_pos = (spy_mom > qqq_mom).astype(float)
    qqq_pos = (qqq_mom > spy_mom).astype(float)

    spy_gross = spy_pos.shift(1).fillna(0) * merged["spy_ret"]
    qqq_gross = qqq_pos.shift(1).fillna(0) * merged["qqq_ret"]

    spy_net = _apply_costs(spy_pos.shift(1).fillna(0), spy_gross, cost_bps)
    qqq_net = _apply_costs(qqq_pos.shift(1).fillna(0), qqq_gross, cost_bps)

    # Portfolio: hold one at a time
    port_ret = spy_net + qqq_net
    winner = spy_pos.shift(1).fillna(0) - qqq_pos.shift(1).fillna(0)
    port_ret.index = merged.index
    winner.index = merged.index
    return port_ret, winner


def backtest_dual_momentum(
    daily: pd.DataFrame,
    lookback: int = 252,
    skip: int = 21,
    cost_bps: float = DEFAULT_COST_BPS,
) -> tuple[pd.Series, pd.Series]:
    """
    Absolute + time-series filter: long only when 12-1 return > 0, else cash (T-bill proxy = 0).
    """
    close = daily["rth_close"]
    mom = close.shift(skip) / close.shift(lookback) - 1
    position = (mom > 0).astype(float)
    gross = position.shift(1).fillna(0) * daily["rth_return"]
    net = _apply_costs(position.shift(1).fillna(0), gross, cost_bps)
    return net, position


def backtest_tsmom_vol_scaled(
    daily: pd.DataFrame,
    lookback: int = 20,
    realized_vol: pd.Series | None = None,
    vol_window: int = 20,
    cost_bps: float = DEFAULT_COST_BPS,
) -> tuple[pd.Series, pd.Series]:
    """
    Time-series momentum with vol scaling: size up when trailing vol is below
    its expanding median (clip 0.5x–1.5x). Uses realized intraday vol if provided,
    else rolling close-to-close std.
    """
    ret = daily["rth_return"]
    close = daily["rth_close"]
    mom = close.pct_change(lookback)

    if realized_vol is not None:
        vol = realized_vol.reindex(daily.index).fillna(
            close.pct_change().rolling(vol_window).std()
        )
    else:
        vol = close.pct_change().rolling(vol_window).std()

    vol_median = vol.expanding(min_periods=vol_window).median()
    scale = (vol_median / vol).clip(0.5, 1.5).fillna(1.0)

    signal = (mom > 0).astype(float)
    position = (signal * scale).shift(1).fillna(0)
    gross = position * ret
    net = _apply_costs(position, gross, cost_bps)
    return net, position


def run_single_asset_backtests(
    daily: pd.DataFrame,
    symbol: str,
    cost_bps: float = DEFAULT_COST_BPS,
) -> list[BacktestResult]:
    bh = daily["rth_return"]
    results: list[BacktestResult] = []

    configs = [
        ("tsmom_5d", lambda: backtest_tsmom(daily, 5, hold=1)),
        ("tsmom_20d", lambda: backtest_tsmom(daily, 20, hold=1)),
        ("tsmom_60d", lambda: backtest_tsmom(daily, 60, hold=1)),
        ("tsmom_20d_hold5d", lambda: backtest_tsmom(daily, 20, hold=5)),
        ("mom_12_1", lambda: backtest_12_1_mom(daily)),
        ("52w_high", lambda: backtest_52w_high(daily)),
        ("dual_momentum", lambda: backtest_dual_momentum(daily)),
        ("premarket_continuation", lambda: backtest_premarket_continuation(daily)),
        ("first30_continuation", lambda: backtest_first30_continuation(daily)),
    ]

    for name, fn in configs:
        net, pos = fn()
        stats = _perf_stats(net, pos, bh, cost_bps=cost_bps)
        if not stats:
            continue
        results.append(BacktestResult(
            strategy=name, symbol=symbol, horizon="daily",
            **stats,
        ))

    return results
