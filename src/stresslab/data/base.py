"""Protocols de datos: único contrato del que depende la lógica de negocio."""
from __future__ import annotations

from typing import Protocol, Sequence

import pandas as pd


class PriceProvider(Protocol):
    """Precios ajustados diarios; columnas = tickers, índice = fechas."""

    def get_prices(self, tickers: Sequence[str], start: str, end: str | None = None) -> pd.DataFrame:
        ...


class RateProvider(Protocol):
    """Rendimientos (en %) diarios de series de tipos; columnas = series."""

    def get_rates(self, series: Sequence[str], start: str, end: str | None = None) -> pd.DataFrame:
        ...
