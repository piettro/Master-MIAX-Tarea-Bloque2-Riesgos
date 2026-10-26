"""Proveedor de precios Yahoo Finance. Único módulo (con fred.py) que toca la red."""
from __future__ import annotations

from typing import Sequence

import pandas as pd
import yfinance as yf

from stresslab import config


class YahooPriceProvider:
    """Precios ajustados por splits y dividendos (auto_adjust=True)."""

    def get_prices(self, tickers: Sequence[str], start: str = config.START_DATE,
                   end: str | None = None) -> pd.DataFrame:
        """Descarga Close ajustado; lanza error si el resultado está vacío."""
        raw = yf.download(list(tickers), start=start, end=end, auto_adjust=True,
                          progress=False, group_by="column")
        if raw is None or raw.empty:
            raise RuntimeError("Yahoo Finance devolvió datos vacíos")
        close = raw["Close"]
        if isinstance(close, pd.Series):
            close = close.to_frame(list(tickers)[0])
        close.index = pd.to_datetime(close.index).tz_localize(None)
        return close[list(tickers)]
