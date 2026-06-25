"""GARCH(1,1) volatility forecasting vs rolling baseline."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from arch import arch_model

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modeling.common import (
    SYMBOLS,
    INITIAL_TRAIN,
    aggregate_walk_forward_predictions,
    get_return_series,
    get_vol_series,
    save_results,
)

warnings.filterwarnings("ignore")


def run_garch(symbol: str, refit_every: int = 21) -> dict:
    returns = get_return_series(symbol)
    realized_vol = get_vol_series(symbol)
    ret = returns.values.astype(float) * 100  # scale for numerical stability
    vol_true = realized_vol.values.astype(float)
    dates = returns.index
    n = len(ret)

    y_true_all, y_pred_garch, y_pred_roll, train_means = [], [], [], []
    last_fit_idx = -1
    garch_res = None

    for i in range(INITIAL_TRAIN, n):
        train_idx = np.arange(0, i)
        if garch_res is None or (i - 1 - last_fit_idx) >= refit_every:
            try:
                am = arch_model(ret[train_idx], mean="Zero", vol="Garch", p=1, q=1)
                garch_res = am.fit(disp="off", show_warning=False)
                last_fit_idx = i - 1
            except Exception:
                garch_res = None

        train_mean_vol = float(np.mean(vol_true[train_idx]))
        roll_vol = float(np.std(ret[train_idx[-20:]])) / 100 if len(train_idx) >= 20 else train_mean_vol

        if garch_res is not None:
            try:
                fc = garch_res.forecast(horizon=1)
                var_fc = float(fc.variance.iloc[-1, 0])
                garch_vol = np.sqrt(var_fc) / 100
            except Exception:
                garch_vol = roll_vol
        else:
            garch_vol = roll_vol

        y_true_all.append(float(vol_true[i]))
        y_pred_garch.append(garch_vol)
        y_pred_roll.append(roll_vol)
        train_means.append(train_mean_vol)

    garch_metrics = aggregate_walk_forward_predictions(
        y_true_all, y_pred_garch, train_means, compute_qlike=True
    )
    roll_metrics = aggregate_walk_forward_predictions(
        y_true_all, y_pred_roll, train_means, compute_qlike=True
    )

    holdout_slice = slice(-126, None)
    holdout_garch = aggregate_walk_forward_predictions(
        y_true_all[holdout_slice], y_pred_garch[holdout_slice], train_means[holdout_slice],
        compute_qlike=True,
    )
    holdout_roll = aggregate_walk_forward_predictions(
        y_true_all[holdout_slice], y_pred_roll[holdout_slice], train_means[holdout_slice],
        compute_qlike=True,
    )

    return {
        "symbol": symbol,
        "target": "next_day_vol",
        "garch": {
            "walk_forward": garch_metrics.to_dict(),
            "holdout": holdout_garch.to_dict(),
        },
        "rolling_20d": {
            "walk_forward": roll_metrics.to_dict(),
            "holdout": holdout_roll.to_dict(),
        },
        "conditional_vol_series": {
            str(dates[INITIAL_TRAIN + i]): float(y_pred_garch[i])
            for i in range(min(50, len(y_pred_garch)))
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Run GARCH volatility models")
    parser.add_argument("--refit-every", type=int, default=21)
    args = parser.parse_args()

    results = {"runs": []}
    for symbol in SYMBOLS:
        print(f"GARCH: {symbol}")
        run = run_garch(symbol, args.refit_every)
        results["runs"].append(run)
        g = run["garch"]["walk_forward"]
        r = run["rolling_20d"]["walk_forward"]
        print(f"  GARCH RMSE={g['rmse']:.6f}  QLIKE={g.get('qlike', 0):.4f}")
        print(f"  Rolling RMSE={r['rmse']:.6f}  QLIKE={r.get('qlike', 0):.4f}")

    path = save_results("garch", results)
    print(f"\nSaved → {path}")


if __name__ == "__main__":
    main()
