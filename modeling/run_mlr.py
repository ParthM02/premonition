"""Multiple linear regression with Ridge OOS predictions and OLS inference."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modeling.common import (
    SYMBOLS,
    TARGETS,
    aggregate_walk_forward_predictions,
    build_feature_frame,
    get_walk_forward_splits,
    save_results,
)


def run_mlr_for_target(symbol: str, target_name: str, alpha: float = 1.0) -> dict:
    frame = build_feature_frame(symbol, target_name)
    feature_cols = frame.attrs["feature_cols"]
    X = frame[feature_cols].values
    y = frame["target"].values
    n = len(frame)

    splits = get_walk_forward_splits(n)
    y_true_all, y_pred_all, train_means = [], [], []
    ols_summary = None

    for split in splits:
        X_train, y_train = X[split.train_idx], y[split.train_idx]
        X_pred = X[split.predict_idx]
        y_actual = y[split.predict_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_pred_s = scaler.transform(X_pred)

        model = Ridge(alpha=alpha)
        model.fit(X_train_s, y_train)
        preds = model.predict(X_pred_s)

        train_mean = float(np.mean(y_train))
        for yt, yp in zip(y_actual, preds):
            y_true_all.append(float(yt))
            y_pred_all.append(float(yp))
            train_means.append(train_mean)

        if ols_summary is None and len(split.train_idx) >= 252:
            X_const = sm.add_constant(X_train)
            ols = sm.OLS(y_train, X_const).fit(cov_type="HC3")
            coefs = {feature_cols[i]: float(ols.params[i + 1]) for i in range(len(feature_cols))}
            pvals = {feature_cols[i]: float(ols.pvalues[i + 1]) for i in range(len(feature_cols))}
            ols_summary = {
                "r_squared": float(ols.rsquared),
                "adj_r_squared": float(ols.rsquared_adj),
                "coefficients": coefs,
                "p_values": pvals,
            }

    compute_dir = target_name in ("same_day_rth", "next_day_return")
    compute_qlike = target_name == "next_day_vol"
    metrics = aggregate_walk_forward_predictions(
        y_true_all, y_pred_all, train_means, compute_dir, compute_qlike
    )

    holdout_mask_start = n - 126
    holdout_true = [t for i, t in enumerate(y_true_all) if i >= len(y_true_all) - 126]
    holdout_pred = [p for i, p in enumerate(y_pred_all) if i >= len(y_pred_all) - 126]
    holdout_metrics = aggregate_walk_forward_predictions(
        holdout_true, holdout_pred, train_means[-126:] if len(train_means) >= 126 else train_means,
        compute_dir, compute_qlike,
    )

    return {
        "symbol": symbol,
        "target": target_name,
        "target_description": TARGETS[target_name],
        "walk_forward": metrics.to_dict(),
        "holdout": holdout_metrics.to_dict(),
        "ols_in_sample": ols_summary,
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
    }


def main():
    parser = argparse.ArgumentParser(description="Run MLR models")
    parser.add_argument("--alpha", type=float, default=1.0, help="Ridge regularization")
    args = parser.parse_args()

    results = {"runs": []}
    for symbol in SYMBOLS:
        for target_name in TARGETS:
            print(f"MLR: {symbol} / {target_name}")
            run = run_mlr_for_target(symbol, target_name, args.alpha)
            results["runs"].append(run)
            wf = run["walk_forward"]
            print(f"  OOS RMSE={wf['rmse']:.6f}  R²={wf['oos_r2']:.4f}  n={wf['n_samples']}")

    path = save_results("mlr", results)
    print(f"\nSaved → {path}")


if __name__ == "__main__":
    main()
