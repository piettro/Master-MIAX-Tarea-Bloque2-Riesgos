"""Métricas de riesgo. Convención: VaR y ES son pérdidas positivas; ES = media de pérdidas ≥ VaR."""
from __future__ import annotations

import numpy as np
from scipy import stats

from stresslab import config


def var(returns: np.ndarray, level: float = config.VAR_LEVEL) -> float:
    """VaR histórico/simulado: cuantil `level` de la pérdida (−retorno)."""
    return float(np.quantile(-np.asarray(returns), level))


def es(returns: np.ndarray, level: float = config.VAR_LEVEL) -> float:
    """Expected Shortfall: media de las pérdidas iguales o superiores al VaR."""
    losses = -np.asarray(returns)
    return float(losses[losses >= np.quantile(losses, level)].mean())


def normal_var_es(mu: float, sigma: float, level: float = config.VAR_LEVEL) -> tuple[float, float]:
    """VaR y ES paramétricos bajo normalidad (referencia): VaR=−μ+zσ, ES=−μ+σ·φ(z)/(1−α)."""
    z = stats.norm.ppf(level)
    return float(-mu + z * sigma), float(-mu + sigma * stats.norm.pdf(z) / (1.0 - level))


def max_drawdown(wealth: np.ndarray) -> np.ndarray:
    """Máxima caída desde máximos (valor positivo) de cada trayectoria; wealth tiene forma (n, T+1)."""
    pico = np.maximum.accumulate(wealth, axis=1)
    return (1.0 - wealth / pico).max(axis=1)


def annualize_vol(daily_std: float) -> float:
    """Volatilidad diaria a anual por raíz del tiempo."""
    return float(daily_std * np.sqrt(config.TRADING_DAYS))
