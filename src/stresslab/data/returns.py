"""Retornos de equities y de bonos (aproximación por duración y convexidad)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from stresslab import config


def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Retornos simples diarios; no rellena huecos (los NaN se conservan)."""
    return prices.pct_change(fill_method=None).iloc[1:]


def par_bond_duration_convexity(y: np.ndarray, maturity: float,
                                freq: int = config.BOND_COUPON_FREQ) -> tuple[np.ndarray, np.ndarray]:
    """Duración modificada y convexidad de un bono a la par con cupón = y (y en decimal)."""
    y = np.asarray(y, dtype=float)
    n = int(round(maturity * freq))
    k = np.arange(1, n + 1)[:, None]  # Periodos de pago.
    c = y[None, :] / freq  # Cupón por periodo = rendimiento (bono par).
    v = (1.0 + c) ** (-k)
    cash = np.where(k == n, 1.0 + c, c)  # Cupones + principal al final.
    t = k / freq
    price = (cash * v).sum(axis=0)  # = 1 por construcción.
    d_mod = (t * cash * v).sum(axis=0) / price / (1.0 + y / freq)
    conv = (cash * v * t * (t + 1.0 / freq)).sum(axis=0) / price / (1.0 + y / freq) ** 2
    return d_mod, conv


def bond_returns(yields_pct: pd.Series, maturity: float) -> pd.Series:
    """r_t ≈ y_{t-1}/252 − D·Δy + ½·C·Δy² con D y C del bono par al rendimiento previo."""
    y = (yields_pct / 100.0).astype(float)
    prev = y.shift(1)
    dy = y - prev
    ok = prev.notna() & dy.notna()
    d_mod, conv = par_bond_duration_convexity(prev[ok].to_numpy(), maturity)
    out = pd.Series(np.nan, index=y.index)
    out[ok] = prev[ok] / config.TRADING_DAYS - d_mod * dy[ok] + 0.5 * conv * dy[ok] ** 2
    return out.iloc[1:].dropna()


def credit_returns(oas_pct: pd.Series, riskfree_pct: pd.Series) -> pd.Series:
    """Retorno del crédito high yield: r_t ≈ y_{t-1}/252 − D·Δy_t con y = OAS + tipo libre de riesgo (D-39)."""
    y = ((oas_pct + riskfree_pct) / 100.0).astype(float)
    return (y.shift(1) / config.TRADING_DAYS - config.CREDIT_SPREAD_DURATION * y.diff()).iloc[1:].dropna()


def asset_returns(prices: pd.DataFrame, rates: pd.DataFrame) -> pd.DataFrame:
    """Retornos de la cartera: equities (precios), HYG (OAS de FRED, D-02) y bonos 10a/2a (duración)."""
    eq = simple_returns(prices[list(config.EQUITIES)])
    tipos = rates.reindex(prices.index.union(rates.index)).ffill(limit=5).reindex(prices.index)
    libre = tipos[[config.FRED_SERIES["UST10Y"], config.FRED_SERIES["UST2Y"]]].mean(axis=1)  # Proxy de tipo a 5 años.
    credito = credit_returns(tipos[config.CREDIT_OAS], libre).rename(config.CREDIT_ASSET).reindex(eq.index)
    bonos = bond_return_table(tipos).reindex(eq.index)
    return pd.concat([eq, credito, bonos], axis=1)


def bond_return_table(rates: pd.DataFrame) -> pd.DataFrame:
    """Retornos de los bonos de config.BOND_MATURITIES desde DGS10/DGS2."""
    cols = {name: bond_returns(rates[config.FRED_SERIES[name]].dropna(), m)
            for name, m in config.BOND_MATURITIES.items()}
    return pd.DataFrame(cols)
