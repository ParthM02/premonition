"""
Fetch premarket (PRE) and regular trading hours (RTH) M30 bars for SPY, QQQ,
and TQQQ from the Webull API. Paginates backwards using the new start_time /
end_time parameters (added in webull-openapi-python-sdk >= 2.0.9) to build a
multi-year history rather than the ~4-month window the single 1200-bar limit
allows.

Usage:
    python fetch_bars.py [--start YYYY-MM-DD]  (default: 2022-01-01)

Output (per symbol, e.g. spy_*):
    data/spy_daily.csv       — one row per trading day, aggregates + features
    data/spy_pre_bars.csv    — raw M30 PRE intraday bars
    data/spy_rth_bars.csv    — raw M30 RTH intraday bars

Key columns in daily CSV:
    date,
    pre_open/close/high/low/volume,
    rth_open/close/high/low/volume,
    premarket_return, rth_return, gap_pct,
    last30_pre_return,      # 9:00–9:30 ET bar vs prev RTH close
    first30_rth_return,     # 9:30–10:00 ET bar open-to-close
    rest_rth_return,        # 10:00 ET → 4pm
    resolution
"""

import os
import json
import time
import logging
import warnings
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

load_dotenv(override=True)

from webull.core.client import ApiClient
from webull.data.data_client import DataClient
from webull.data.common.category import Category
from webull.data.common.timespan import Timespan

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SYMBOLS = ["SPY", "QQQ", "TQQQ"]
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

BAR_COUNT = "1200"       # API maximum per request
MAX_BATCHES = 30         # hard safety limit (~30 × 90 days = ~7 years max)
INTER_BATCH_SLEEP = 0.25 # seconds between paginated requests (rate-limit courtesy)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

def build_client() -> DataClient:
    key = os.getenv("WEBULL_APP_KEY")
    secret = os.getenv("WEBULL_APP_SECRET")
    if not key or not secret:
        raise RuntimeError("WEBULL_APP_KEY and WEBULL_APP_SECRET must be set in .env")
    api = ApiClient(key.strip(), secret.strip(), "us")
    api.add_endpoint("us", "api.webull.com")
    return DataClient(api)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _parse_response(resp) -> list[dict]:
    """
    Extract a flat list of bar dicts from any Webull API response shape:
      - bare list  [...]
      - {"data": [...]}  or  {"data": {"list": [...]}}
      - single-symbol paginated  [{tickerId, symbol, time, ...}, ...]
    """
    if hasattr(resp, "json"):
        body = resp.json()
    elif isinstance(resp, (dict, list)):
        body = resp
    else:
        raise ValueError(f"Unexpected response type: {type(resp)}")

    if isinstance(body, list):
        return body
    if isinstance(body, dict):
        data = body.get("data", body)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return data.get("list", data.get("bars", []))
    raise ValueError(f"Cannot parse bars: {json.dumps(body, default=str)[:400]}")


def _bars_to_df(bars: list[dict], timespan: Timespan) -> pd.DataFrame:
    """Parse a raw bar list into a tidy DataFrame with a UTC-aware 'dt' column."""
    df = pd.DataFrame(bars)
    df.columns = [c.lower() for c in df.columns]

    ts_col = next((c for c in ("timestamp", "time", "date") if c in df.columns), None)
    if ts_col is None:
        raise KeyError(f"No timestamp column. Columns: {df.columns.tolist()}")

    ts = df[ts_col]
    if pd.api.types.is_numeric_dtype(ts):
        df["dt"] = pd.to_datetime(ts, unit="ms", utc=True)
    else:
        df["dt"] = pd.to_datetime(ts, utc=True)

    if timespan != Timespan.D:
        df["dt"] = df["dt"].dt.tz_convert("America/New_York")
    df["date"] = df["dt"].dt.date

    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.sort_values("dt").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Paginated fetch
# ---------------------------------------------------------------------------

def fetch_all_bars(
    dc: DataClient,
    symbol: str,
    session: str,
    start_dt: datetime,
) -> pd.DataFrame:
    """
    Fetch M30 bars for one symbol×session, paginating backwards until either:
      - the API returns an empty response, or
      - the oldest bar in the batch is before start_dt, or
      - MAX_BATCHES batches have been fetched.

    Returns a single deduplicated DataFrame sorted by dt (ascending).
    """
    all_dfs: list[pd.DataFrame] = []
    end_time_ms: Optional[int] = None
    start_ts = start_dt.timestamp() * 1000  # ms, for comparison

    for batch_num in range(1, MAX_BATCHES + 1):
        resp = dc.market_data.get_history_bar(
            symbol,
            Category.US_ETF,
            Timespan.M30,
            count=BAR_COUNT,
            trading_sessions=[session],
            end_time=end_time_ms,
        )
        bars = _parse_response(resp)
        if not bars:
            print(f"    batch {batch_num}: empty response — done")
            break

        df_batch = _bars_to_df(bars, Timespan.M30)
        all_dfs.append(df_batch)

        oldest_dt = df_batch["dt"].min()
        newest_dt = df_batch["dt"].max()
        oldest_ms = int(oldest_dt.timestamp() * 1000)

        print(
            f"    batch {batch_num}: {len(df_batch)} bars  "
            f"[{oldest_dt.strftime('%Y-%m-%d')} → {newest_dt.strftime('%Y-%m-%d')}]"
        )

        if oldest_ms <= start_ts:
            print(f"    reached target start date — done")
            break

        # Advance cursor to just before the oldest bar in this batch
        end_time_ms = oldest_ms - 1

        if batch_num < MAX_BATCHES:
            time.sleep(INTER_BATCH_SLEEP)

    if not all_dfs:
        raise RuntimeError(f"No bars returned for {symbol} {session}")

    combined = (
        pd.concat(all_dfs, ignore_index=True)
        .drop_duplicates(subset=["dt"])
        .sort_values("dt")
        .reset_index(drop=True)
    )

    # Trim to start_dt
    combined = combined[combined["dt"] >= pd.Timestamp(start_dt)].reset_index(drop=True)

    return combined


# ---------------------------------------------------------------------------
# Daily aggregation
# ---------------------------------------------------------------------------

def _aggregate_intraday(bars: pd.DataFrame, prefix: str) -> pd.DataFrame:
    agg = (
        bars.groupby("date")
        .agg(open=("open", "first"), close=("close", "last"),
             high=("high", "max"), low=("low", "min"), volume=("volume", "sum"))
        .reset_index()
    )
    return agg.rename(columns={c: f"{prefix}_{c}"
                                for c in ("open", "close", "high", "low", "volume")})


def _extract_window_features(
    pre_bars: pd.DataFrame, rth_bars: pd.DataFrame
) -> pd.DataFrame:
    """
    Extract per-day features for the two 30-minute edge windows:

    last30_pre   : 9:00–9:30 ET  (final half-hour of premarket)
    first30_rth  : 9:30–10:00 ET (opening half-hour of RTH)
    rest_rth     : 10:00 ET → close
    """
    last_pre = pre_bars[
        (pre_bars["dt"].dt.hour == 9) & (pre_bars["dt"].dt.minute == 0)
    ][["date", "open", "close"]].rename(
        columns={"open": "last30_pre_open", "close": "last30_pre_close"}
    )

    first_rth = rth_bars[
        (rth_bars["dt"].dt.hour == 9) & (rth_bars["dt"].dt.minute == 30)
    ][["date", "open", "close"]].rename(
        columns={"open": "first30_rth_open", "close": "first30_rth_close"}
    )

    rest_rth_open = (
        rth_bars[rth_bars["dt"].dt.hour == 10][["date", "open"]]
        .groupby("date")["open"].first()
        .reset_index().rename(columns={"open": "rest_rth_open"})
    )
    rest_rth_close = (
        rth_bars.groupby("date")["close"].last()
        .reset_index().rename(columns={"close": "rest_rth_close"})
    )

    return (
        last_pre
        .merge(first_rth, on="date", how="inner")
        .merge(rest_rth_open, on="date", how="left")
        .merge(rest_rth_close, on="date", how="left")
    )


def build_daily_features(
    pre_daily: pd.DataFrame,
    rth_daily: pd.DataFrame,
    pre_bars: Optional[pd.DataFrame] = None,
    rth_bars: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    df = pd.merge(pre_daily, rth_daily, on="date", how="inner")
    df = df.sort_values("date").reset_index(drop=True)

    df["prev_rth_close"] = df["rth_close"].shift(1)
    df["premarket_return"] = (df["pre_close"] - df["prev_rth_close"]) / df["prev_rth_close"]
    df["rth_return"] = (df["rth_close"] - df["rth_open"]) / df["rth_open"]
    df["gap_pct"] = (df["rth_open"] - df["pre_close"]) / df["pre_close"]

    if pre_bars is not None and rth_bars is not None:
        win = _extract_window_features(pre_bars, rth_bars)
        df = df.merge(win, on="date", how="left")
        df["last30_pre_return"] = (
            (df["last30_pre_close"] - df["prev_rth_close"]) / df["prev_rth_close"]
        )
        df["first30_rth_return"] = (
            (df["first30_rth_close"] - df["first30_rth_open"]) / df["first30_rth_open"]
        )
        df["rest_rth_return"] = (
            (df["rest_rth_close"] - df["rest_rth_open"]) / df["rest_rth_open"]
        )
    else:
        for col in ("last30_pre_return", "first30_rth_return", "rest_rth_return"):
            df[col] = float("nan")

    df["resolution"] = "M30"
    return df.dropna(subset=["premarket_return", "rth_return"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Per-symbol orchestration
# ---------------------------------------------------------------------------

def fetch_symbol(dc: DataClient, symbol: str, start_dt: datetime) -> pd.DataFrame:
    print(f"\n{'='*56}")
    print(f"  {symbol}  (target start: {start_dt.date()})")
    print(f"{'='*56}")

    print(f"  Fetching PRE M30 bars...")
    pre_bars = fetch_all_bars(dc, symbol, "PRE", start_dt)
    pre_daily = _aggregate_intraday(pre_bars, "pre")
    print(f"  PRE total: {len(pre_bars)} bars → {len(pre_daily)} days "
          f"({pre_bars['date'].min()} → {pre_bars['date'].max()})")

    print(f"  Fetching RTH M30 bars...")
    rth_bars = fetch_all_bars(dc, symbol, "RTH", start_dt)
    rth_daily = _aggregate_intraday(rth_bars, "rth")
    print(f"  RTH total: {len(rth_bars)} bars → {len(rth_daily)} days "
          f"({rth_bars['date'].min()} → {rth_bars['date'].max()})")

    sym_l = symbol.lower()
    pre_bars.to_csv(DATA_DIR / f"{sym_l}_pre_bars.csv", index=False)
    rth_bars.to_csv(DATA_DIR / f"{sym_l}_rth_bars.csv", index=False)

    daily = build_daily_features(pre_daily, rth_daily, pre_bars, rth_bars)

    out_path = DATA_DIR / f"{sym_l}_daily.csv"
    daily.to_csv(out_path, index=False)
    print(f"  Saved {len(daily)} trading days → {out_path}")
    return daily


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fetch Webull historical bars")
    parser.add_argument(
        "--start", default="2022-01-01",
        help="Earliest date to fetch back to (YYYY-MM-DD), default 2022-01-01"
    )
    args = parser.parse_args()
    start_dt = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    print(f"Connecting to Webull API...")
    dc = build_client()
    print(f"Connected. Fetching bars back to {start_dt.date()}.\n")

    results = {}
    for symbol in SYMBOLS:
        try:
            results[symbol] = fetch_symbol(dc, symbol, start_dt)
        except Exception as e:
            print(f"\n[ERROR] {symbol}: {e}")
            raise

    print("\n" + "=" * 56)
    print("  SUMMARY")
    print("=" * 56)
    for symbol, df in results.items():
        pm = df["premarket_return"]
        rth = df["rth_return"]
        corr = pm.corr(rth)
        dir_match = ((pm > 0) == (rth > 0)).mean()
        print(f"\n{symbol} — {len(df)} days  ({df['date'].min()} → {df['date'].max()})")
        print(f"  Premarket return  mean={pm.mean()*100:+.3f}%  std={pm.std()*100:.3f}%")
        print(f"  RTH return        mean={rth.mean()*100:+.3f}%  std={rth.std()*100:.3f}%")
        print(f"  Pearson r (PM→RTH): {corr:+.4f}")
        print(f"  Directional match:  {dir_match*100:.1f}%")


if __name__ == "__main__":
    main()
