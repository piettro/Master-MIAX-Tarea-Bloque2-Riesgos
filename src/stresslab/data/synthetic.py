"""Proveedor sintético: cadena de Markov + dependencia con parámetros conocidos.

Sirve para tests (cero red) y modo offline. Al conocer la verdad, los tests
pueden comprobar que HMM, cópulas y simulador la recuperan.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from stresslab import config


@dataclass(frozen=True)
class SyntheticParams:
    """Parámetros verdaderos. Estado 0 = calma, 1 = crisis."""

    transition: np.ndarray = field(  # Persistencia alta: crisis dura ~25 días, calma ~100.
        default_factory=lambda: np.array([[0.99, 0.01], [0.04, 0.96]])
    )
    daily_mean: tuple[float, float] = (0.0005, -0.0015)  # Deriva positiva en calma, negativa en crisis.
    daily_vol: tuple[float, float] = (0.01, 0.03)  # La volatilidad se triplica en crisis.
    rho: tuple[float, float] = (0.3, 0.7)  # Equicorrelación: sube en crisis.
    t_df: tuple[float | None, float | None] = (None, 4.0)  # Gauss en calma, t en crisis (colas).


@dataclass(frozen=True)
class SyntheticSample:
    """Retornos simples diarios y estados verdaderos."""

    returns: pd.DataFrame
    states: np.ndarray


def _equicorr_chol(n: int, rho: float) -> np.ndarray:
    """Cholesky de una matriz de equicorrelación rho."""
    corr = np.full((n, n), rho)
    np.fill_diagonal(corr, 1.0)
    return np.linalg.cholesky(corr)


def simulate_states(p: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Cadena de Markov de n pasos; arranca en la distribución estacionaria."""
    eigval, eigvec = np.linalg.eig(p.T)
    pi = np.real(eigvec[:, np.argmax(np.real(eigval))])
    pi = pi / pi.sum()
    u = rng.random(n)
    states = np.empty(n, dtype=int)
    states[0] = int(u[0] > pi[0])
    cum = np.cumsum(p, axis=1)
    for t in range(1, n):
        states[t] = int(u[t] > cum[states[t - 1], 0])
    return states


def _state_shocks(n: int, k: int, state: int, prm: SyntheticParams, rng: np.random.Generator) -> np.ndarray:
    """Shocks de varianza unitaria con la dependencia del estado (Gauss o t)."""
    z = rng.standard_normal((n, k)) @ _equicorr_chol(k, prm.rho[state]).T
    df = prm.t_df[state]
    if df is None:
        return z
    w = rng.chisquare(df, size=(n, 1)) / df
    return z / np.sqrt(w) * np.sqrt((df - 2.0) / df)  # Reescala a varianza 1.


def generate(n_days: int, n_assets: int, params: SyntheticParams | None = None,
             seed: int | None = None, start: str = config.START_DATE) -> SyntheticSample:
    """Genera retornos de n_assets activos con regímenes ocultos conocidos."""
    prm = params or SyntheticParams()
    rng = config.rng(seed)
    states = simulate_states(prm.transition, n_days, rng)
    rets = np.empty((n_days, n_assets))
    for s in (0, 1):
        idx = states == s
        shocks = _state_shocks(int(idx.sum()), n_assets, s, prm, rng)
        rets[idx] = prm.daily_mean[s] + prm.daily_vol[s] * shocks
    dates = pd.bdate_range(start, periods=n_days)
    return SyntheticSample(pd.DataFrame(rets, index=dates), states)


class SyntheticProvider:
    """Implementa PriceProvider y RateProvider sin tocar la red."""

    def __init__(self, n_days: int = 1500, params: SyntheticParams | None = None,
                 seed: int | None = None) -> None:
        self.n_days, self.params, self.seed = n_days, params, seed
        self.states: np.ndarray | None = None  # Estados verdaderos de la última llamada.

    def get_prices(self, tickers: Sequence[str], start: str = config.START_DATE,
                   end: str | None = None) -> pd.DataFrame:
        """Precios sintéticos (base 100) con regímenes conocidos."""
        sample = generate(self.n_days, len(tickers), self.params, self.seed, start)
        self.states = sample.states
        prices = 100.0 * (1.0 + sample.returns).cumprod()
        prices.columns = list(tickers)
        return prices if end is None else prices.loc[:end]

    def get_rates(self, series: Sequence[str], start: str = config.START_DATE,
                  end: str | None = None) -> pd.DataFrame:
        """Rendimientos sintéticos (%) con más ruido en crisis."""
        rng = config.rng(self.seed)
        prm = self.params or SyntheticParams()
        states = simulate_states(prm.transition, self.n_days, rng)
        vol = np.where(states == 1, 0.08, 0.03)  # Puntos porcentuales diarios; más ruido en crisis.
        steps = rng.standard_normal((self.n_days, len(series))) * vol[:, None]
        base = np.linspace(3.0, 1.5, len(series))  # Curva inicial plausible (10a > 2a).
        levels = np.clip(base + np.cumsum(steps, axis=0), 0.05, None)
        idx = pd.bdate_range(start, periods=self.n_days)
        rates = pd.DataFrame(levels, index=idx, columns=list(series))
        return rates if end is None else rates.loc[:end]
