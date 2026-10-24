"""Data loading for the ASM project: index prices and India VIX, with local CSV caching.

Tickers used:
    ^NSEI      NIFTY 50 (primary index)
    ^GSPC      S&P 500 (secondary index, for robustness checks)
    ^INDIAVIX  India VIX (external regime validation, NIFTY options implied vol)
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

TICKERS = {
    "nifty50": "^NSEI",
    "sp500": "^GSPC",
    "india_vix": "^INDIAVIX",
}


def _cache_path(name: str) -> Path:
    return RAW_DIR / f"{name}.csv"


def download_prices(
    name: str,
    ticker: str,
    start: str = "2008-01-01",
    end: str | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Download OHLCV data for a ticker and cache it as CSV. Returns a DataFrame
    indexed by Date with columns Open, High, Low, Close, Volume.
    """
    cache_file = _cache_path(name)
    if cache_file.exists() and not force:
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        return df

    raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if raw.empty:
        raise ValueError(f"No data returned for ticker {ticker!r}")

    # yfinance returns MultiIndex columns (Price, Ticker) even for a single ticker.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index.name = "Date"
    df.to_csv(cache_file)
    return df


def load_all(start: str = "2008-01-01", end: str | None = None, force: bool = False) -> dict[str, pd.DataFrame]:
    """Download (or load from cache) all series used in this project."""
    return {
        name: download_prices(name, ticker, start=start, end=end, force=force)
        for name, ticker in TICKERS.items()
    }


def load_close(name: str, start: str = "2008-01-01", end: str | None = None, force: bool = False) -> pd.Series:
    """Convenience: just the Close price series for one named ticker."""
    ticker = TICKERS[name]
    df = download_prices(name, ticker, start=start, end=end, force=force)
    s = df["Close"].dropna()
    s.name = name
    return s


if __name__ == "__main__":
    data = load_all()
    for name, df in data.items():
        print(f"{name}: {df.shape[0]} rows, {df.index.min().date()} to {df.index.max().date()}")
