"""fetch_market_data.py
Download daily market data for a US multi-asset analysis from 2006-01-01 to today.

Data sources used:
- Equities/ETFs: Yahoo Finance via `yfinance` (adjusted close prices).
- US Treasury yields (10y and 2y): FRED via `pandas_datareader` (DGS10, DGS2).

Outputs saved to `data/raw/` and `data/` as CSV files.

All comments and variable names are in English.
"""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import List

import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RAW_DIR = DATA_DIR / "raw"
START_DATE = "2006-01-01"
END_DATE = date.today().isoformat()


def ensure_dirs() -> None:
    """Create data directories if they do not exist."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)


def fetch_equities(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """Fetch adjusted close prices for a list of tickers using yfinance.

    Returns a DataFrame with dates as index and tickers as columns.
    """
    logging.info("Downloading equities/ETFs from Yahoo Finance: %s", tickers)
    # yfinance.download returns a multi-column DataFrame when multiple tickers
    df = yf.download(tickers, start=start, end=end, progress=False, threads=True, auto_adjust=True)
    if isinstance(df, pd.DataFrame) and "Adj Close" in df.columns:
        prices = df["Adj Close"].copy()
    else:
        # If a single ticker was passed, yf returns a Series
        prices = df["Adj Close"] if "Adj Close" in df else df

    # Ensure columns are simple ticker strings
    if isinstance(prices, pd.DataFrame):
        prices.columns = [str(c) for c in prices.columns]
    prices.index = pd.to_datetime(prices.index)
    return prices.sort_index()


def fetch_us_yields(start: str, end: str) -> pd.DataFrame:
    """Fetch US 10-year and 2-year constant maturity yields from FRED.

    The FRED series are `DGS10` and `DGS2` (daily; many days are reported).
    """
    fred_series = ["DGS10", "DGS2"]
    logging.info("Downloading yields from FRED: %s", fred_series)
    yields = pdr.DataReader(fred_series, "fred", start, end)
    yields = yields.rename(columns={"DGS10": "US10Y", "DGS2": "US2Y"})
    yields.index = pd.to_datetime(yields.index)
    return yields.sort_index()


def combine_and_fill(equities: pd.DataFrame, yields: pd.DataFrame) -> pd.DataFrame:
    """Combine equities and yields into a single DataFrame and forward-fill missing data.

    Forward-filling is a pragmatic choice for aligning daily series with different reporting
    calendars; students can choose alternative imputation methods later.
    """
    logging.info("Combining equity prices and yields into single DataFrame")
    combined = pd.concat([equities, yields], axis=1)
    combined = combined.sort_index()
    # Forward-fill then backfill any leading NaNs
    combined = combined.ffill().bfill()
    return combined


def save_outputs(equities: pd.DataFrame, yields: pd.DataFrame, combined: pd.DataFrame) -> None:
    """Save raw and combined data to CSV files in the `data` folder."""
    equities_path = RAW_DIR / "equities_adj_close.csv"
    yields_path = RAW_DIR / "us_yields.csv"
    combined_path = DATA_DIR / "market_data_combined.csv"

    logging.info("Saving equities to %s", equities_path)
    equities.to_csv(equities_path)
    logging.info("Saving yields to %s", yields_path)
    yields.to_csv(yields_path)
    logging.info("Saving combined data to %s", combined_path)
    combined.to_csv(combined_path)


def main() -> None:
    """Main entry point for data fetching.

    The multi-asset portfolio (equal-weight, no rebalancing) includes the following
    equity/ETF tickers and fixed-income items:

    AAPL, AMZN, BAC, BRK-B, CVX, ENPH, GLD, GME, GOOGL, JNJ, JPM, MSFT, NVDA, PG, XOM,
    HYG (corporate high-yield ETF), plus US 10y and 2y yields from FRED.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ensure_dirs()

    tickers = [
        "AAPL",
        "AMZN",
        "BAC",
        "BRK-B",
        "CVX",
        "ENPH",
        "GLD",
        "GME",
        "GOOGL",
        "JNJ",
        "JPM",
        "MSFT",
        "NVDA",
        "PG",
        "XOM",
        "HYG",
    ]

    equities = fetch_equities(tickers, START_DATE, END_DATE)
    yields = fetch_us_yields(START_DATE, END_DATE)
    combined = combine_and_fill(equities, yields)
    save_outputs(equities, yields, combined)


if __name__ == "__main__":
    main()
