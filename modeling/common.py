"""
Shared data loading, feature engineering, walk-forward evaluation, and metrics
for SPY/QQQ modeling scripts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

SYMBOLS = ["SPY", "QQQ"]

INITIAL_TRAIN = 252
ROLL_STEP = 21
HOLDOUT_DAYS = 126

TARGETS = {
    "same_day_rth": "Same-day RTH return from premarket signals",
    "next_day_return": "Next-day RTH return",
    "next_day_vol": "Next-day realized volatility",
}


@dataclass
class WalkForwardSplit:
    train_idx: np.ndarray
    predict_idx: np.ndarray
    is_holdout: bool = False


@dataclass
class EvalMetrics:
    rmse: float
    mae: float
    oos_r2: float
    n_samples: int
    accuracy: Optional[float] = None
    balanced_accuracy: Optional[float] = None
    qlike: Optional[float] = None
    naive_rmse: Optional[float] = None

    def to_dict(self) -> dict:
        d = {
            "rmse": self.rmse,
            "mae": self.mae,
            "oos_r2": self.oos_r2,
            "n_samples": self.n_samples,
        }
        if self.accuracy is not None:
            d["accuracy"] = self.accuracy
        if self.balanced_accuracy is not None:
            d["balanced_accuracy"] = self.balanced_accuracy
        if self.qlike is not None:
            d["qlike"] = self.qlike
        if self.naive_rmse is not None:
            d["naive_rmse"] = self.naive_rmse
        return d


def load_daily(symbol: str) -> pd.DataFrame:
    path = DATA_DIR / f"{symbol.lower()}_daily.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def load_rth_bars(symbol: str) -> pd.DataFrame:
    path = DATA_DIR / f"{symbol.lower()}_rth_bars.csv"
    df = pd.read_csv(path)
    df["dt"] = pd.to_datetime(df["dt"], utc=True)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("dt").reset_index(drop=True)


def compute_realized_vol(rth_bars: pd.DataFrame) -> pd.Series:
    """Std of M30 log-returns within each RTH session."""
    bars = rth_bars.copy()
    bars["log_ret"] = np.log(bars["close"] / bars["close"].shift(1))
    vol = bars.groupby("date")["log_ret"].std()
    vol.index = pd.to_datetime(vol.index)
    return vol.rename("realized_vol")


def load_symbol_data(symbol: str) -> pd.DataFrame:
    daily = load_daily(symbol)
    rth_bars = load_rth_bars(symbol)
    realized = compute_realized_vol(rth_bars)
    daily = daily.merge(
        realized.reset_index().rename(columns={"date": "date"}),
        on="date",
        how="left",
    )
    daily["realized_vol"] = daily["realized_vol"].fillna(daily["rth_return"].abs())
    return daily


def load_merged_symbols() -> pd.DataFrame:
    """SPY and QQQ daily data merged on date with prefixed columns."""
    spy = load_symbol_data("SPY")
    qqq = load_symbol_data("QQQ")
    spy = spy.add_prefix("spy_").rename(columns={"spy_date": "date"})
    qqq = qqq.add_prefix("qqq_").rename(columns={"qqq_date": "date"})
    return pd.merge(spy, qqq, on="date", how="inner").sort_values("date").reset_index(drop=True)


def _add_lags_and_rolls(df: pd.DataFrame, col: str, prefix: str) -> pd.DataFrame:
    for lag in (1, 2, 5, 20):
        df[f"{prefix}_lag_{lag}"] = df[col].shift(lag)
    for window in (5, 20):
        df[f"{prefix}_roll_mean_{window}"] = df[col].shift(1).rolling(window).mean()
        df[f"{prefix}_roll_std_{window}"] = df[col].shift(1).rolling(window).std()
    return df


def _add_dow(df: pd.DataFrame) -> pd.DataFrame:
    dow = pd.to_datetime(df["date"]).dt.dayofweek
    for i in range(5):
        df[f"dow_{i}"] = (dow == i).astype(int)
    return df


def build_feature_frame(symbol: str, target_name: str) -> pd.DataFrame:
    """
    Build a feature matrix and target column for one symbol and target.
    All features are causally available at prediction time (no lookahead).
    """
    sym = symbol.lower()
    df = load_symbol_data(symbol)
    merged = load_merged_symbols() if symbol == "QQQ" else None

    df = _add_lags_and_rolls(df, "rth_return", "rth_ret")
    df = _add_lags_and_rolls(df, "premarket_return", "pre_ret")
    df["realized_vol_lag_1"] = df["realized_vol"].shift(1)
    df["realized_vol_roll_mean_5"] = df["realized_vol"].shift(1).rolling(5).mean()
    df["realized_vol_roll_mean_20"] = df["realized_vol"].shift(1).rolling(20).mean()
    df["pre_vol_ratio"] = df["pre_volume"] / df["pre_volume"].shift(1).rolling(20).mean()
    df["rth_vol_ratio"] = df["rth_volume"] / df["rth_volume"].shift(1).rolling(20).mean()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = _add_dow(df)

    if merged is not None:
        for col in ("premarket_return", "rth_return", "last30_pre_return", "gap_pct"):
            df[f"spy_{col}"] = merged[f"spy_{col}"]
            df[f"spy_{col}_lag_1"] = merged[f"spy_{col}"].shift(1)

    if target_name == "same_day_rth":
        df["target"] = df["rth_return"]
        feature_cols = [
            "premarket_return", "gap_pct", "last30_pre_return",
            "rth_ret_lag_1", "rth_ret_lag_2", "rth_ret_lag_5",
            "rth_ret_roll_mean_5", "rth_ret_roll_std_5", "rth_ret_roll_std_20",
            "realized_vol_lag_1", "realized_vol_roll_mean_5",
            "pre_vol_ratio", "dow_0", "dow_1", "dow_2", "dow_3", "dow_4",
        ]
        if symbol == "QQQ":
            feature_cols += [
                "spy_premarket_return", "spy_gap_pct", "spy_last30_pre_return",
                "spy_premarket_return_lag_1", "spy_rth_return_lag_1",
            ]
    elif target_name == "next_day_return":
        df["target"] = df["rth_return"].shift(-1)
        feature_cols = [
            "rth_return", "premarket_return", "gap_pct", "last30_pre_return",
            "rth_ret_lag_1", "rth_ret_lag_2", "rth_ret_lag_5", "rth_ret_lag_20",
            "rth_ret_roll_mean_5", "rth_ret_roll_mean_20",
            "rth_ret_roll_std_5", "rth_ret_roll_std_20",
            "pre_ret_lag_1", "realized_vol_lag_1",
            "realized_vol_roll_mean_5", "realized_vol_roll_mean_20",
            "dow_0", "dow_1", "dow_2", "dow_3", "dow_4",
        ]
        if symbol == "QQQ":
            feature_cols += ["spy_rth_return", "spy_premarket_return", "spy_rth_return_lag_1"]
    elif target_name == "next_day_vol":
        df["target"] = df["realized_vol"].shift(-1)
        feature_cols = [
            "realized_vol", "rth_return", "premarket_return",
            "realized_vol_lag_1", "realized_vol_roll_mean_5", "realized_vol_roll_mean_20",
            "rth_ret_roll_std_5", "rth_ret_roll_std_20",
            "rth_ret_lag_1", "pre_ret_lag_1",
            "dow_0", "dow_1", "dow_2", "dow_3", "dow_4",
        ]
        if symbol == "QQQ":
            feature_cols += ["spy_realized_vol", "spy_rth_return"]
            df["spy_realized_vol"] = merged["spy_realized_vol"]
    else:
        raise ValueError(f"Unknown target: {target_name}")

    feature_cols = [c for c in feature_cols if c in df.columns]
    out = df[["date", "target"] + feature_cols].dropna().reset_index(drop=True)
    out.attrs["feature_cols"] = feature_cols
    out.attrs["target_name"] = target_name
    out.attrs["symbol"] = symbol
    return out


def get_walk_forward_splits(
    n: int,
    initial_train: int = INITIAL_TRAIN,
    step: int = ROLL_STEP,
    holdout: int = HOLDOUT_DAYS,
) -> list[WalkForwardSplit]:
    """Expanding-window splits; last `holdout` rows marked as holdout."""
    holdout_start = max(n - holdout, initial_train + 1)
    splits: list[WalkForwardSplit] = []

    t = initial_train
    while t < n:
        train_idx = np.arange(0, t)
        if t >= holdout_start:
            # Holdout region: predict one day at a time with expanding train
            if t < n:
                splits.append(WalkForwardSplit(train_idx, np.array([t]), is_holdout=True))
            t += 1
            continue
        end = min(t + step, holdout_start)
        predict_idx = np.arange(t, end)
        if len(predict_idx) > 0:
            splits.append(WalkForwardSplit(train_idx, predict_idx, is_holdout=False))
        t = end

    return splits


def iter_simple_splits(n: int, initial_train: int = INITIAL_TRAIN) -> Iterator[tuple[np.ndarray, int]]:
    """Yield (train_idx, predict_idx) for each day from initial_train to n-1."""
    for i in range(initial_train, n):
        yield np.arange(0, i), i


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train_mean: Optional[float] = None,
    compute_direction: bool = False,
    compute_qlike: bool = False,
) -> EvalMetrics:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    n = len(y_true)
    if n == 0:
        return EvalMetrics(rmse=np.nan, mae=np.nan, oos_r2=np.nan, n_samples=0)

    residuals = y_true - y_pred
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mae = float(np.mean(np.abs(residuals)))
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    oos_r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    naive_rmse = None
    if y_train_mean is not None:
        naive_rmse = float(np.sqrt(np.mean((y_true - y_train_mean) ** 2)))

    accuracy = balanced_accuracy = None
    if compute_direction:
        pred_sign = np.sign(y_pred)
        true_sign = np.sign(y_true)
        nonzero = true_sign != 0
        if nonzero.any():
            accuracy = float(np.mean(pred_sign[nonzero] == true_sign[nonzero]))
            pos = true_sign > 0
            neg = true_sign < 0
            tpr = float(np.mean(pred_sign[pos] == 1)) if pos.any() else 0.0
            tnr = float(np.mean(pred_sign[neg] == -1)) if neg.any() else 0.0
            balanced_accuracy = (tpr + tnr) / 2

    qlike = None
    if compute_qlike:
        vol_pred = np.maximum(y_pred, 1e-8)
        vol_true = np.maximum(y_true, 1e-8)
        qlike = float(np.mean(np.log(vol_pred ** 2) + (vol_true ** 2) / (vol_pred ** 2)))

    return EvalMetrics(
        rmse=rmse,
        mae=mae,
        oos_r2=oos_r2,
        n_samples=n,
        accuracy=accuracy,
        balanced_accuracy=balanced_accuracy,
        qlike=qlike,
        naive_rmse=naive_rmse,
    )


def aggregate_walk_forward_predictions(
    y_true_list: list[float],
    y_pred_list: list[float],
    y_train_means: list[float],
    compute_direction: bool = False,
    compute_qlike: bool = False,
) -> EvalMetrics:
    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)
    train_mean = float(np.mean(y_train_means)) if y_train_means else None
    return compute_regression_metrics(
        y_true, y_pred, train_mean, compute_direction, compute_qlike
    )


def save_results(model_name: str, payload: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        **payload,
        "model": model_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    path = RESULTS_DIR / f"{model_name}.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return path


def load_all_results() -> dict[str, dict]:
    if not RESULTS_DIR.exists():
        return {}
    results = {}
    for path in sorted(RESULTS_DIR.glob("*.json")):
        if path.name == "combined_summary.json":
            continue
        with open(path) as f:
            results[path.stem] = json.load(f)
    return results


def get_return_series(symbol: str) -> pd.Series:
    df = load_symbol_data(symbol)
    return df.set_index("date")["rth_return"]


def get_vol_series(symbol: str) -> pd.Series:
    df = load_symbol_data(symbol)
    return df.set_index("date")["realized_vol"]
