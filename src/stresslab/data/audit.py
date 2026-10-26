"""Auditoría de calidad: historias cortas, huecos, outliers y saltos (R-DAT-04)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stresslab import config
from stresslab.data.returns import simple_returns


def _robust_z(r: pd.Series) -> pd.Series:
    """Retorno en sigmas robustas (mediana y MAD escalado a normal)."""
    mad = (r - r.median()).abs().median() * 1.4826
    return (r - r.median()) / mad if mad > 0 else r * 0.0


def audit_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Tabla de auditoría por activo; marca pero NUNCA elimina datos."""
    rets = simple_returns(prices)
    calendario = prices.index
    inicio_ref = pd.Timestamp(config.START_DATE)
    filas = {}
    for col in prices.columns:
        s = prices[col].dropna()
        r = rets[col].dropna()
        z = _robust_z(r)
        gaps = calendario[(calendario >= s.index[0]) & (calendario <= s.index[-1])].difference(s.index)
        filas[col] = {
            "inicio": s.index[0].date(), "fin": s.index[-1].date(), "n_obs": len(s),
            "historia_corta": bool((s.index[0] - inicio_ref).days > config.SHORT_HISTORY_DAYS),
            "huecos_internos": len(gaps),
            "retorno_max_abs": float(r.abs().max()), "fecha_max": r.abs().idxmax().date(),
            "n_outliers": int((z.abs() > config.OUTLIER_MAD_Z).sum()),
            "n_saltos": int((r.abs() > config.JUMP_THRESHOLD).sum()),
            "n_precios_no_positivos": int((s <= 0).sum()),
        }
    return pd.DataFrame(filas).T


@dataclass(frozen=True)
class ProxyModel:
    """r_i = alpha + beta·r_mercado + eps; eps con desviación por estado (o única)."""

    alpha: float
    beta: float
    resid_std: tuple[float, ...]  # Un valor por estado (uno solo si no hay estados).


def fit_market_proxy(target: pd.Series, market: pd.Series, states: pd.Series | None = None,
                     n_states: int = config.N_STATES) -> ProxyModel:
    """Regresión MCO del activo sobre el mercado en el solape; residuo por estado si se dan."""
    df = pd.concat([target, market], axis=1, keys=["y", "x"]).dropna()
    beta, alpha = np.polyfit(df["x"], df["y"], 1)
    resid = df["y"] - (alpha + beta * df["x"])
    if states is None:
        return ProxyModel(float(alpha), float(beta), (float(resid.std()),))
    st = states.reindex(resid.index)
    stds = tuple(float(resid[st == k].std()) if (st == k).sum() > 2 else float(resid.std())
                 for k in range(n_states))
    return ProxyModel(float(alpha), float(beta), stds)


def fill_with_proxy(target: pd.Series, market: pd.Series, model: ProxyModel,
                    rng: np.random.Generator, states: pd.Series | None = None) -> pd.Series:
    """Rellena los NaN de target con alpha + beta·mercado + ruido idiosincrático (documentado)."""
    falta = target.index[target.isna()].intersection(market.dropna().index)
    if states is None:
        sd = np.full(len(falta), model.resid_std[0])
    else:
        sd = np.array([model.resid_std[int(k)] for k in states.reindex(falta).fillna(0)])
    ruido = rng.standard_normal(len(falta)) * sd
    out = target.copy()
    out.loc[falta] = model.alpha + model.beta * market.loc[falta] + ruido
    return out
