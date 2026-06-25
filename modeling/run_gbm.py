"""Gradient boosting (XGBoost) for all prediction targets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modeling.common import (
    SYMBOLS,
    TARGETS,
    aggregate_walk_forward_predictions,
    build_feature_frame,
    get_walk_forward_splits,
    save_results,
)


def run_gbm_for_target(
    symbol: str,
    target_name: str,
    max_depth: int = 3,
    n_estimators: int = 200,
) -> dict:
    frame = build_feature_frame(symbol, target_name)
    feature_cols = frame.attrs["feature_cols"]
    X = frame[feature_cols].values
    y = frame["target"].values
    n = len(frame)

    splits = get_walk_forward_splits(n)
    y_true_all, y_pred_all, train_means = [], [], []
    importance_accum = np.zeros(len(feature_cols))

    for split in splits:
        X_train, y_train = X[split.train_idx], y[split.train_idx]
        X_pred = X[split.predict_idx]
        y_actual = y[split.predict_idx]

        val_size = max(int(len(split.train_idx) * 0.15), 20)
        X_fit, y_fit = X_train[:-val_size], y_train[:-val_size]
        X_val, y_val = X_train[-val_size:], y_train[-val_size:]

        scaler = StandardScaler()
        X_fit_s = scaler.fit_transform(X_fit)
        X_val_s = scaler.transform(X_val)
        X_pred_s = scaler.transform(X_pred)

        model = XGBRegressor(
            max_depth=max_depth,
            n_estimators=n_estimators,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
            early_stopping_rounds=20,
        )
        model.fit(X_fit_s, y_fit, eval_set=[(X_val_s, y_val)], verbose=False)
        preds = model.predict(X_pred_s)
        importance_accum += model.feature_importances_

        train_mean = float(np.mean(y_train))
        for yt, yp in zip(y_actual, preds):
            y_true_all.append(float(yt))
            y_pred_all.append(float(yp))
            train_means.append(train_mean)

    compute_dir = target_name in ("same_day_rth", "next_day_return")
    compute_qlike = target_name == "next_day_vol"
    metrics = aggregate_walk_forward_predictions(
        y_true_all, y_pred_all, train_means, compute_dir, compute_qlike
    )
    holdout_metrics = aggregate_walk_forward_predictions(
        y_true_all[-126:], y_pred_all[-126:], train_means[-126:],
        compute_dir, compute_qlike,
    )

    imp = importance_accum / max(len(splits), 1)
    top_idx = np.argsort(imp)[::-1][:10]
    top_features = [
        {"feature": feature_cols[i], "importance": float(imp[i])}
        for i in top_idx
    ]

    return {
        "symbol": symbol,
        "target": target_name,
        "target_description": TARGETS[target_name],
        "walk_forward": metrics.to_dict(),
        "holdout": holdout_metrics.to_dict(),
        "top_features": top_features,
        "n_features": len(feature_cols),
    }


def main():
    parser = argparse.ArgumentParser(description="Run XGBoost models")
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--n-estimators", type=int, default=200)
    args = parser.parse_args()

    results = {"runs": []}
    for symbol in SYMBOLS:
        for target_name in TARGETS:
            print(f"GBM: {symbol} / {target_name}")
            run = run_gbm_for_target(symbol, target_name, args.max_depth, args.n_estimators)
            results["runs"].append(run)
            wf = run["walk_forward"]
            print(f"  OOS RMSE={wf['rmse']:.6f}  R²={wf['oos_r2']:.4f}")
            print(f"  Top feature: {run['top_features'][0]['feature']}")

    path = save_results("gbm", results)
    print(f"\nSaved → {path}")


if __name__ == "__main__":
    main()
