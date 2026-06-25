"""Ornstein-Uhlenbeck / AR(1) mean reversion on returns and SPY-QQQ spread."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modeling.common import (
    INITIAL_TRAIN,
    aggregate_walk_forward_predictions,
    load_merged_symbols,
    load_symbol_data,
    save_results,
)

warnings.filterwarnings("ignore")


def fit_ar1(series: np.ndarray) -> dict:
    """Fit AR(1): x_t = mu + phi * (x_{t-1} - mu) + eps."""
    x = series.astype(float)
    mu = float(np.mean(x))
    demeaned = x - mu
    if len(demeaned) < 2:
        return {"mu": mu, "phi": 0.0, "half_life": np.inf}
    phi = float(np.corrcoef(demeaned[:-1], demeaned[1:])[0, 1])
    phi = np.clip(phi, -0.99, 0.99)
    if abs(phi) < 1e-8:
        half_life = np.inf
    else:
        half_life = float(-np.log(2) / np.log(abs(phi))) if abs(phi) < 1 else np.inf
    return {"mu": mu, "phi": phi, "half_life": half_life}


def adf_test(series: np.ndarray) -> dict:
    try:
        stat, pval, _, _, crit, _ = adfuller(series, autolag="AIC")
        return {
            "adf_stat": float(stat),
            "p_value": float(pval),
            "stationary_5pct": bool(pval < 0.05),
            "crit_5pct": float(crit["5%"]),
        }
    except Exception as e:
        return {"error": str(e)}


def forecast_ar1(params: dict, x_last: float) -> float:
    mu, phi = params["mu"], params["phi"]
    return mu + phi * (x_last - mu)


def run_mean_reversion_series(
    series: np.ndarray,
    series_name: str,
    symbol: str,
) -> dict:
    n = len(series)
    y_true_all, y_pred_all, train_means = [], [], []
    params_history = []

    for i in range(INITIAL_TRAIN, n):
        train = series[:i]
        params = fit_ar1(train)
        if len(params_history) == 0 or i % 21 == 0:
            params_history.append({"index": i, **params})

        pred = forecast_ar1(params, float(series[i - 1]))
        y_true_all.append(float(series[i]))
        y_pred_all.append(pred)
        train_means.append(params["mu"])

    metrics = aggregate_walk_forward_predictions(
        y_true_all, y_pred_all, train_means, compute_direction=True
    )
    holdout = aggregate_walk_forward_predictions(
        y_true_all[-126:], y_pred_all[-126:], train_means[-126:],
        compute_direction=True,
    )

    full_params = fit_ar1(series)
    adf = adf_test(series)

    return {
        "symbol": symbol,
        "series": series_name,
        "target": "next_day_return",
        "ou_params_full_sample": full_params,
        "adf_test": adf,
        "walk_forward": metrics.to_dict(),
        "holdout": holdout.to_dict(),
        "params_history": params_history[:10],
    }


def run_spread_mean_reversion() -> dict:
    merged = load_merged_symbols()
    spread = (merged["spy_rth_return"] - merged["qqq_rth_return"]).values
    return run_mean_reversion_series(spread, "spy_minus_qqq_rth_spread", "SPY_QQQ")


def main():
    parser = argparse.ArgumentParser(description="Run mean reversion (OU/AR1) models")
    args = parser.parse_args()

    results = {"runs": []}
    for symbol in ("SPY", "QQQ"):
        for col, name in (("rth_return", "rth_return"), ("premarket_return", "premarket_return")):
            df = load_symbol_data(symbol)
            series = df[col].values
            print(f"Mean reversion: {symbol} / {name}")
            run = run_mean_reversion_series(series, name, symbol)
            results["runs"].append(run)
            ou = run["ou_params_full_sample"]
            wf = run["walk_forward"]
            print(f"  phi={ou['phi']:.4f}  half_life={ou['half_life']:.1f}d  RMSE={wf['rmse']:.6f}")

    print("Mean reversion: SPY-QQQ spread")
    spread_run = run_spread_mean_reversion()
    results["runs"].append(spread_run)
    ou = spread_run["ou_params_full_sample"]
    print(f"  phi={ou['phi']:.4f}  half_life={ou['half_life']:.1f}d")

    path = save_results("mean_reversion", results)
    print(f"\nSaved → {path}")


if __name__ == "__main__":
    main()
