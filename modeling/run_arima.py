"""ARIMA models for next-day return forecasting."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
from statsmodels.tsa.arima.model import ARIMA

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modeling.common import (
    SYMBOLS,
    INITIAL_TRAIN,
    aggregate_walk_forward_predictions,
    get_return_series,
    save_results,
)

warnings.filterwarnings("ignore")

DEFAULT_ORDER = (1, 0, 1)


def run_arima(symbol: str, refit_every: int = 63) -> dict:
    returns = get_return_series(symbol)
    y = returns.values.astype(float)
    dates = returns.index
    n = len(y)

    y_true_all, y_pred_all, train_means = [], [], []
    last_fit_idx = -1
    fitted_model = None
    order = DEFAULT_ORDER

    for i in range(INITIAL_TRAIN, n):
        train_idx = np.arange(0, i)
        train_series = y[train_idx]
        train_mean = float(np.mean(train_series))

        if fitted_model is None or (i - 1 - last_fit_idx) >= refit_every:
            try:
                fitted_model = ARIMA(train_series, order=order).fit()
                last_fit_idx = i - 1
            except Exception:
                fitted_model = None

        if fitted_model is not None:
            try:
                fc = fitted_model.forecast(steps=1)
                pred = float(fc.iloc[0] if hasattr(fc, "iloc") else fc[0])
            except Exception:
                pred = train_mean
        else:
            pred = train_mean

        y_true_all.append(float(y[i]))
        y_pred_all.append(pred)
        train_means.append(train_mean)

    metrics = aggregate_walk_forward_predictions(
        y_true_all, y_pred_all, train_means, compute_direction=True
    )
    holdout_metrics = aggregate_walk_forward_predictions(
        y_true_all[-126:], y_pred_all[-126:], train_means[-126:],
        compute_direction=True,
    )

    return {
        "symbol": symbol,
        "target": "next_day_return",
        "arima_order": list(order),
        "walk_forward": metrics.to_dict(),
        "holdout": holdout_metrics.to_dict(),
        "refit_every": refit_every,
    }


def main():
    parser = argparse.ArgumentParser(description="Run ARIMA models")
    parser.add_argument("--refit-every", type=int, default=63)
    args = parser.parse_args()

    results = {"runs": []}
    for symbol in SYMBOLS:
        print(f"ARIMA: {symbol}")
        run = run_arima(symbol, args.refit_every)
        results["runs"].append(run)
        wf = run["walk_forward"]
        print(f"  OOS RMSE={wf['rmse']:.6f}  R²={wf['oos_r2']:.4f}")

    path = save_results("arima", results)
    print(f"\nSaved → {path}")


if __name__ == "__main__":
    main()
