"""Marginales condicionales por activo y estado: momentos y ajuste Normal vs Student-t."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from stresslab import config

STATE_NAMES = {0: "calma", 1: "crisis"}
MIN_OBS = 60  # Mínimo de observaciones por (activo, estado) para ajustar; por debajo, la t es inestable.
DF_BOUNDS = (3.0, 200.0)  # Suelo 3 (D-26): varianza finita sin colas absurdas; >200 equivale a la normal.


@dataclass(frozen=True)
class Marginal:
    """Distribución ajustada de los retornos diarios de un activo en un estado."""

    family: str  # "normal" o "t"
    params: tuple[float, ...]  # normal: (mu, sigma); t: (df, mu, sigma)
    loglik: float
    n: int

    @property
    def n_params(self) -> int:
        return len(self.params)

    @property
    def aic(self) -> float:
        return 2 * self.n_params - 2 * self.loglik

    @property
    def bic(self) -> float:
        return self.n_params * np.log(self.n) - 2 * self.loglik

    @property
    def dist(self):
        return stats.norm(*self.params) if self.family == "normal" else stats.t(*self.params)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        return self.dist.cdf(x)

    def ppf(self, u: np.ndarray) -> np.ndarray:
        return self.dist.ppf(u)


def conditional_moments(returns: pd.DataFrame, states: pd.Series) -> pd.DataFrame:
    """Media, volatilidad, asimetría y curtosis (exceso) diarias por activo y estado."""
    filas = {}
    for col in returns.columns:
        for k, nombre in STATE_NAMES.items():
            x = returns[col][states.reindex(returns.index) == k].dropna()
            filas[(col, nombre)] = {
                "n": len(x), "media": x.mean(), "vol": x.std(),
                "vol_anual": x.std() * np.sqrt(config.TRADING_DAYS),
                "skew": stats.skew(x), "kurt_exceso": stats.kurtosis(x),
            }
    return pd.DataFrame(filas).T.rename_axis(["activo", "estado"])


def fit_normal(x: np.ndarray) -> Marginal:
    mu, sigma = stats.norm.fit(x)
    return Marginal("normal", (mu, sigma), float(stats.norm.logpdf(x, mu, sigma).sum()), len(x))


def fit_t(x: np.ndarray) -> Marginal:
    """MLE de la t con g.l. acotados a DF_BOUNDS y localización/escala re-estimadas."""
    df, _, _ = stats.t.fit(x)
    df = float(np.clip(df, *DF_BOUNDS))
    _, mu, sigma = stats.t.fit(x, f0=df)
    return Marginal("t", (df, mu, sigma), float(stats.t.logpdf(x, df, mu, sigma).sum()), len(x))


def select_marginal(x: np.ndarray, criterio: str = "bic") -> tuple[Marginal, Marginal, Marginal]:
    """Ajusta Normal y t y devuelve (elegida, normal, t) según AIC o BIC."""
    n, t = fit_normal(x), fit_t(x)
    elegida = min((n, t), key=lambda m: getattr(m, criterio))
    return elegida, n, t


def fit_all(returns: pd.DataFrame, states: pd.Series, criterio: str = "bic"
            ) -> tuple[pd.DataFrame, dict[tuple[str, int], Marginal]]:
    """Tabla de ajuste y diccionario (activo, estado) -> Marginal elegida."""
    filas, elegidas = [], {}
    for col in returns.columns:
        for k, nombre in STATE_NAMES.items():
            x = returns[col][states.reindex(returns.index) == k].dropna().to_numpy()
            if len(x) < MIN_OBS:
                continue
            m, n, t = select_marginal(x, criterio)
            elegidas[(col, k)] = m
            filas.append({"activo": col, "estado": nombre, "n": len(x), "aic_normal": n.aic, "aic_t": t.aic,
                          "bic_normal": n.bic, "bic_t": t.bic, "df_t": t.params[0], "elegida": m.family})
    return pd.DataFrame(filas).set_index(["activo", "estado"]), elegidas


def refuge_answers(returns: pd.DataFrame, states: pd.Series, market: pd.Series) -> dict[str, float]:
    """Respuestas numéricas de R-MAR-02: vol de HYG y comportamiento de GLD como refugio."""
    st = states.reindex(returns.index)
    mom = conditional_moments(returns, states)
    c = {k: market.reindex(returns.index)[st == k] for k in (0, 1)}
    gld = returns["GLD"]
    return {
        "hyg_vol_calma": float(mom.loc[("HYG", "calma"), "vol"]),
        "hyg_vol_crisis": float(mom.loc[("HYG", "crisis"), "vol"]),
        "hyg_ratio_vol": float(mom.loc[("HYG", "crisis"), "vol"] / mom.loc[("HYG", "calma"), "vol"]),
        "gld_media_calma": float(mom.loc[("GLD", "calma"), "media"]),
        "gld_media_crisis": float(mom.loc[("GLD", "crisis"), "media"]),
        "gld_corr_sp500_calma": float(gld[st == 0].corr(c[0])),
        "gld_corr_sp500_crisis": float(gld[st == 1].corr(c[1])),
        "sp500_media_crisis": float(c[1].mean()),
        "gld_dias_positivos_crisis": float((gld[st == 1].dropna() > 0).mean()),
        "gld_ratio_vol": float(mom.loc[("GLD", "crisis"), "vol"] / mom.loc[("GLD", "calma"), "vol"]),
    }


def qq_points(x: np.ndarray, m: Marginal) -> tuple[np.ndarray, np.ndarray]:
    """Cuantiles teóricos de la marginal ajustada frente a los empíricos (para el QQ)."""
    p = (np.arange(1, len(x) + 1) - 0.5) / len(x)
    return m.ppf(p), np.sort(x)
