"""Carga con degradación elegante: fuente viva -> caché CSV, avisando del fallo."""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

from stresslab import config
from stresslab.data import csv_cache
from stresslab.data.base import PriceProvider, RateProvider

PRICE_TICKERS: tuple[str, ...] = (*config.EQUITIES, config.CREDIT_ETF, config.MARKET_TICKER, config.VIX_TICKER)
RATE_SERIES: tuple[str, ...] = (*config.FRED_SERIES.values(), config.CREDIT_OAS)


@dataclass
class MarketData:
    """Datos crudos (precios y tipos) más los avisos de degradación."""

    prices: pd.DataFrame
    rates: pd.DataFrame
    warnings: list[str] = field(default_factory=list)


def clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Calendario de bolsa limpio: quita festivos con solo VIX y la barra del día en curso (incompleta)."""
    out = prices.dropna(subset=[config.MARKET_TICKER])
    return out[out.index < pd.Timestamp.today().normalize()]


def _with_fallback(live: Callable[[], pd.DataFrame], cache: Callable[[], pd.DataFrame],
                   save: Callable[[pd.DataFrame], object], label: str, avisos: list[str]) -> pd.DataFrame:
    """Intenta la fuente viva; si falla, usa la caché y avisa."""
    try:
        df = live()
        save(df)
        return df
    except Exception as exc:  # noqa: BLE001 - cualquier fallo de red degrada a caché
        msg = f"{label}: fuente viva falló ({type(exc).__name__}: {exc}); se usa la caché data/raw"
        warnings.warn(msg, stacklevel=2)
        avisos.append(msg)
        return cache()


def load_market_data(offline: bool | None = None, raw_dir: Path = config.RAW_DIR,
                     prices: PriceProvider | None = None, rates: RateProvider | None = None) -> MarketData:
    """Carga precios y tipos. Offline => solo caché; online => viva con respaldo en caché."""
    offline = config.is_offline() if offline is None else offline
    cache = csv_cache.CsvProvider(raw_dir)
    avisos: list[str] = []
    if offline:
        return MarketData(clean_prices(cache.get_prices(PRICE_TICKERS)), cache.get_rates(RATE_SERIES), avisos)
    if prices is None or rates is None:  # Importación diferida: solo si hay red.
        from stresslab.data.fred import FredRateProvider
        from stresslab.data.yahoo import YahooPriceProvider
        prices, rates = prices or YahooPriceProvider(), rates or FredRateProvider()
    px = _with_fallback(lambda: prices.get_prices(PRICE_TICKERS, config.START_DATE),
                        lambda: cache.get_prices(PRICE_TICKERS), lambda d: csv_cache.update_csv(
                            csv_cache.PRICES_FILE, d, raw_dir), "precios", avisos)
    rt = _with_fallback(lambda: rates.get_rates(RATE_SERIES, config.START_DATE),
                        lambda: cache.get_rates(RATE_SERIES), lambda d: csv_cache.update_csv(
                            csv_cache.RATES_FILE, d, raw_dir), "tipos", avisos)
    return MarketData(clean_prices(px), rt, avisos)
