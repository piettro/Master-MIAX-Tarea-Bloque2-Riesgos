"""Cópulas Gauss y t por estado: ajuste sobre pseudo-observaciones, muestreo y dependencia en colas."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar

T_DF_BOUNDS = (2.5, 100.0)  # g.l. de la t-cópula: >100 equivale a Gauss; <2,5 es inestable con pocos datos.
TAIL_Q = 0.05  # Cuantil para la dependencia de cola empírica (5 % peor de cada activo).


@dataclass(frozen=True)
class CopulaFit:
    """Cópula elíptica ajustada: family = 'gauss' o 't'."""

    family: str
    corr: np.ndarray
    df: float | None
    loglik: float
    n: int

    @property
    def dim(self) -> int:
        return self.corr.shape[0]

    @property
    def n_params(self) -> int:
        return self.dim * (self.dim - 1) // 2 + (1 if self.family == "t" else 0)

    @property
    def aic(self) -> float:
        return 2 * self.n_params - 2 * self.loglik

    @property
    def bic(self) -> float:
        return self.n_params * np.log(self.n) - 2 * self.loglik

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """n vectores U ∈ (0,1)^d de la cópula; la t usa mezcla de escala chi-cuadrado."""
        z = rng.standard_normal((n, self.dim)) @ np.linalg.cholesky(self.corr).T
        if self.family == "gauss":
            return stats.norm.cdf(z)
        w = rng.chisquare(self.df, size=(n, 1)) / self.df
        return stats.t.cdf(z / np.sqrt(w), self.df)


def pseudo_obs(x: np.ndarray) -> np.ndarray:
    """Pseudo-observaciones: rango/(n+1) por columna (marginales uniformes sin supuestos)."""
    return (stats.rankdata(x, axis=0) / (len(x) + 1)).astype(float)


def nearest_corr(r: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Proyecta a matriz de correlación definida positiva (recorta autovalores)."""
    w, v = np.linalg.eigh((r + r.T) / 2)
    m = (v * np.clip(w, eps, None)) @ v.T
    d = np.sqrt(np.diag(m))
    return m / np.outer(d, d)


def mean_offdiag(c: np.ndarray) -> float:
    """Media de los elementos fuera de la diagonal."""
    d = c.shape[0]
    return float((c.sum() - np.trace(c)) / (d * (d - 1)))


def fit_gauss(u: np.ndarray) -> CopulaFit:
    """Gauss: correlación de los scores normales; log-verosimilitud de la densidad de cópula."""
    z = stats.norm.ppf(u)
    r = nearest_corr(np.corrcoef(z, rowvar=False))
    inv = np.linalg.inv(r)
    ll = -0.5 * len(u) * np.linalg.slogdet(r)[1] - 0.5 * np.einsum("ij,jk,ik->", z, inv - np.eye(len(r)), z)
    return CopulaFit("gauss", r, None, float(ll), len(u))


def _tau_corr(u: np.ndarray) -> np.ndarray:
    """Correlación elíptica por inversión de la tau de Kendall: ρ = sin(π·τ/2)."""
    d = u.shape[1]
    tau = np.eye(d)
    for i in range(d):
        for j in range(i + 1, d):
            tau[i, j] = tau[j, i] = stats.kendalltau(u[:, i], u[:, j])[0]
    return nearest_corr(np.sin(np.pi * tau / 2))


def _t_loglik(u: np.ndarray, r: np.ndarray, df: float) -> float:
    x = stats.t.ppf(u, df)
    joint = stats.multivariate_t.logpdf(x, loc=np.zeros(len(r)), shape=r, df=df)
    return float((joint - stats.t.logpdf(x, df).sum(axis=1)).sum())


def fit_t(u: np.ndarray) -> CopulaFit:
    """t: R por tau de Kendall (robusta) y g.l. por MLE unidimensional con R fija."""
    r = _tau_corr(u)
    res = minimize_scalar(lambda v: -_t_loglik(u, r, v), bounds=T_DF_BOUNDS, method="bounded")
    return CopulaFit("t", r, float(res.x), float(-res.fun), len(u))


def lower_tail_dependence(u: np.ndarray, q: float = TAIL_Q) -> float:
    """λ_L empírico medio por pares: P(U_j<q | U_i<q), promediado sobre los pares."""
    b = (u < q).astype(float)
    joint = b.T @ b / len(u)
    d = u.shape[1]
    return float((joint.sum() - np.trace(joint)) / (d * (d - 1)) / q)


def implied_tail_dependence(fit: CopulaFit, rng: np.random.Generator, n: int = 200_000, q: float = TAIL_Q) -> float:
    """Mismo estimador aplicado a una muestra de la cópula ajustada (comparable con el empírico)."""
    return lower_tail_dependence(fit.sample(n, rng), q)


def bootstrap_mean_corr(x: np.ndarray, rng: np.random.Generator, n_boot: int = 1000) -> dict[str, float]:
    """Correlación media fuera de diagonal con IC 95 % por bootstrap de días (iid)."""
    n = len(x)
    medias = np.array([mean_offdiag(np.corrcoef(x[rng.integers(0, n, n)], rowvar=False)) for _ in range(n_boot)])
    return {"media": mean_offdiag(np.corrcoef(x, rowvar=False)),
            "ic_inf": float(np.percentile(medias, 2.5)), "ic_sup": float(np.percentile(medias, 97.5))}


def bootstrap_corr_diff(x0: np.ndarray, x1: np.ndarray, rng: np.random.Generator, n_boot: int = 1000) -> dict[str, float]:
    """Diferencia (crisis − calma) de la correlación media con IC 95 %; remuestreo independiente por estado."""
    def media(x: np.ndarray) -> float:
        return mean_offdiag(np.corrcoef(x[rng.integers(0, len(x), len(x))], rowvar=False))

    d = np.array([media(x1) - media(x0) for _ in range(n_boot)])
    punto = mean_offdiag(np.corrcoef(x1, rowvar=False)) - mean_offdiag(np.corrcoef(x0, rowvar=False))
    return {"diferencia": punto, "ic_inf": float(np.percentile(d, 2.5)), "ic_sup": float(np.percentile(d, 97.5)),
            "p_menor_igual_cero": float((d <= 0).mean())}


def portfolio_vol(vols: np.ndarray, corr: np.ndarray) -> float:
    """Volatilidad de una cartera equiponderada (rebalanceo diario, solo diagnóstico) dadas vols y correlación."""
    w = np.full(len(vols), 1.0 / len(vols))
    cov = corr * np.outer(vols, vols)
    return float(np.sqrt(w @ cov @ w))


def risk_decomposition(x0: np.ndarray, x1: np.ndarray) -> dict[str, float]:
    """Cuánto del salto de riesgo en crisis viene de la volatilidad y cuánto de la correlación."""
    v0, v1 = x0.std(axis=0, ddof=1), x1.std(axis=0, ddof=1)
    c0, c1 = np.corrcoef(x0, rowvar=False), np.corrcoef(x1, rowvar=False)
    return {"vol_calma": portfolio_vol(v0, c0), "vol_crisis": portfolio_vol(v1, c1),
            "vol_solo_vols_crisis": portfolio_vol(v1, c0), "vol_solo_corr_crisis": portfolio_vol(v0, c1)}
