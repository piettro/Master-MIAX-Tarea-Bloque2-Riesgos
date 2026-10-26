"""Factores de estado del HMM (D-04): variaciones, no niveles, comparables a retornos."""
from __future__ import annotations

import numpy as np
import pandas as pd

from stresslab import config

FACTORS: tuple[str, ...] = ("ret_sp500", "dlog_vix", "d_curva_10y2y", "d_oas_hy")


def build_features(prices: pd.DataFrame, rates: pd.DataFrame) -> pd.DataFrame:
    """S&P 500 (retorno), Δlog VIX, Δ(10a−2a) y Δ OAS HY, sobre el calendario de bolsa."""
    px = prices[[config.MARKET_TICKER, config.VIX_TICKER]].dropna()
    tipos = rates.reindex(px.index.union(rates.index)).ffill(limit=5).reindex(px.index)
    curva = tipos[config.FRED_SERIES["UST10Y"]] - tipos[config.FRED_SERIES["UST2Y"]]
    out = pd.DataFrame({
        FACTORS[0]: px[config.MARKET_TICKER].pct_change(),
        FACTORS[1]: np.log(px[config.VIX_TICKER]).diff(),
        FACTORS[2]: curva.diff(),
        FACTORS[3]: tipos[config.CREDIT_OAS].diff(),
    })
    return out.dropna()


def standardize(x: pd.DataFrame) -> pd.DataFrame:
    """Z-score por columna (media y desviación de toda la muestra)."""
    return (x - x.mean()) / x.std()
