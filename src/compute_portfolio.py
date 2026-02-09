"""compute_portfolio.py
Construct an equal-weight, buy-and-hold (no rebalancing) portfolio from the
columns in `data/market_data_combined.csv`. Saves portfolio value, returns
and summary metrics to the `data/` folder.

Assumptions:
- Columns representing price series (equities, ETFs) are used directly.
- Columns named like `US10Y` or `US2Y` are treated as annual yield levels (percent)
  and converted to a simple price proxy by treating the daily return as yield/100/252.

All comments are in English and the market considered is US.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
COMBINED_PATH = DATA_DIR / "market_data_combined.csv"
OUTPUT_DIR = DATA_DIR


@dataclass
class PortfolioMetrics:
    cumulative_return: float
    cagr: float
    annualized_vol: float
    max_drawdown: float


def load_combined(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def is_yield_column(name: str) -> bool:
    name_upper = name.upper()
    return name_upper.startswith("US") and name_upper.endswith("Y")


def prepare_price_like_series(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all columns to price-like series starting at 1.

    - Price columns (assumed for equities/ETFs): normalize by dividing by first value.
    - Yield columns (`US10Y`, `US2Y`): convert to daily returns via yield/100/252,
      then build a cumulative product series starting at 1.
    """
    price_df = pd.DataFrame(index=df.index)
    for col in df.columns:
        series = df[col].astype(float).copy()
        if is_yield_column(col):
            # Treat value as annual yield in percent. Approximate daily return.
            daily_ret = series / 100.0 / 252.0
            price_like = (1.0 + daily_ret).cumprod()
            price_like = price_like / price_like.iloc[0]
        else:
            # Price-like series: normalize to 1 at first observation
            price_like = series / series.iloc[0]
        price_df[col] = price_like
    return price_df


def build_equal_weight_buy_and_hold(price_df: pd.DataFrame) -> pd.Series:
    """Construct buy-and-hold portfolio value series with equal initial weights.

    No rebalancing is performed: we allocate initial capital proportionally
    and hold constant shares thereafter.
    """
    n = price_df.shape[1]
    w0 = 1.0 / n
    # Portfolio value = sum( w0 * price_t / price_0 )
    pv = price_df.divide(price_df.iloc[0]).mul(w0).sum(axis=1)
    # Normalize to 1 at start (represents starting capital)
    pv = pv / pv.iloc[0]
    return pv


def compute_returns(portfolio_values: pd.Series) -> pd.Series:
    return portfolio_values.pct_change().dropna()


def compute_metrics(portfolio_values: pd.Series, returns: pd.Series) -> PortfolioMetrics:
    total_days = (portfolio_values.index[-1] - portfolio_values.index[0]).days
    trading_days = returns.shape[0]
    cumulative_return = portfolio_values.iloc[-1] / portfolio_values.iloc[0] - 1.0
    # Annualize using trading days ~ 252
    cagr = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) ** (252.0 / trading_days) - 1.0
    annualized_vol = returns.std(ddof=1) * sqrt(252.0)
    # Max drawdown
    roll_max = portfolio_values.cummax()
    drawdown = (portfolio_values / roll_max) - 1.0
    max_dd = drawdown.min()
    return PortfolioMetrics(cumulative_return=cumulative_return, cagr=cagr, annualized_vol=annualized_vol, max_drawdown=max_dd)


def save_outputs(portfolio_values: pd.Series, returns: pd.Series, metrics: PortfolioMetrics) -> None:
    pv_path = OUTPUT_DIR / "portfolio_value_buy_and_hold.csv"
    ret_path = OUTPUT_DIR / "portfolio_returns_buy_and_hold.csv"
    metrics_path = OUTPUT_DIR / "portfolio_metrics_buy_and_hold.txt"

    portfolio_values.to_csv(pv_path, header=["portfolio_value"] )
    returns.to_csv(ret_path, header=["portfolio_return"] )
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"Cumulative Return: {metrics.cumulative_return:.6f}\n")
        f.write(f"CAGR (ann.): {metrics.cagr:.6f}\n")
        f.write(f"Annualized Volatility: {metrics.annualized_vol:.6f}\n")
        f.write(f"Max Drawdown: {metrics.max_drawdown:.6f}\n")


def main() -> None:
    df = load_combined(COMBINED_PATH)
    price_df = prepare_price_like_series(df)
    portfolio_values = build_equal_weight_buy_and_hold(price_df)
    returns = compute_returns(portfolio_values)
    metrics = compute_metrics(portfolio_values, returns)
    save_outputs(portfolio_values, returns, metrics)


if __name__ == "__main__":
    main()
