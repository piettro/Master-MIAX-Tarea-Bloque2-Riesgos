"""HMM gaussiano de 2 estados: ajuste, orden calma/crisis, filtrado, duración mínima y episodios."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from stresslab import config


@dataclass
class RegimeFit:
    """Modelo ajustado; order[k] = índice interno de hmmlearn del estado k (0 calma, 1 crisis)."""

    model: GaussianHMM
    order: np.ndarray
    loglik: float

    @property
    def transmat(self) -> np.ndarray:
        return self.model.transmat_[np.ix_(self.order, self.order)]

    @property
    def startprob(self) -> np.ndarray:
        return self.model.startprob_[self.order]

    @property
    def means(self) -> np.ndarray:
        return self.model.means_[self.order]

    @property
    def covars(self) -> np.ndarray:
        return self.model.covars_[self.order]


def fit_hmm(x: np.ndarray, n_states: int = config.N_STATES, n_init: int = config.HMM_N_INIT,
            seed: int | None = None) -> RegimeFit:
    """EM con varias inicializaciones; ordena estados por varianza total (0 = calma)."""
    rng = config.rng(seed)
    best: GaussianHMM | None = None
    best_ll = -np.inf
    for s in rng.integers(0, 2**31 - 1, size=n_init):
        m = GaussianHMM(n_components=n_states, covariance_type=config.HMM_COVARIANCE,
                        n_iter=config.HMM_N_ITER, random_state=int(s))
        try:
            m.fit(x)
            ll = m.score(x)
        except ValueError:  # Inicialización degenerada: se descarta.
            continue
        if ll > best_ll:
            best, best_ll = m, ll
    if best is None:
        raise RuntimeError("EM no convergió en ninguna inicialización")
    varianza = [np.trace(c) for c in best.covars_]
    return RegimeFit(best, np.argsort(varianza), float(best_ll))


def smoothed_proba(fit: RegimeFit, x: np.ndarray) -> np.ndarray:
    """P(estado_t | toda la serie) con columnas ordenadas (usa información futura: look-ahead)."""
    return fit.model.predict_proba(x)[:, fit.order]


def filtered_proba(fit: RegimeFit, x: np.ndarray) -> np.ndarray:
    """P(estado_t | datos hasta t): algoritmo forward, sin look-ahead."""
    mu, cov, pi, a = fit.means, fit.covars, fit.startprob, fit.transmat
    log_b = np.column_stack([multivariate_normal.logpdf(x, mu[k], cov[k]) for k in range(len(pi))])
    out = np.empty_like(log_b)
    log_alpha = np.log(pi) + log_b[0]
    out[0] = np.exp(log_alpha - logsumexp(log_alpha))
    for t in range(1, len(x)):
        log_alpha = logsumexp(np.log(out[t - 1])[:, None] + np.log(a), axis=0) + log_b[t]
        out[t] = np.exp(log_alpha - logsumexp(log_alpha))
    return out


def _runs(s: np.ndarray) -> list[list[int]]:
    """Tramos consecutivos [inicio, fin_exclusivo, estado]."""
    cortes = np.flatnonzero(np.diff(s)) + 1
    ini = np.r_[0, cortes]
    fin = np.r_[cortes, len(s)]
    return [[int(i), int(f), int(s[i])] for i, f in zip(ini, fin)]


def enforce_min_duration(states: np.ndarray, min_days: int) -> np.ndarray:
    """Absorbe el tramo más corto (< min_days) en el estado vecino hasta que ninguno lo sea."""
    s = states.astype(int).copy()
    while True:
        runs = _runs(s)
        if len(runs) == 1 or min(f - i for i, f, _ in runs) >= min_days:
            return s
        i, f, est = min(runs, key=lambda r: (r[1] - r[0], r[0]))
        s[i:f] = 1 - est


def hard_states(p_crisis: np.ndarray, threshold: float = config.CRISIS_THRESHOLD,
                min_days: int = config.MIN_EPISODE_DAYS) -> np.ndarray:
    """Decisión dura: umbral sobre P(crisis) y filtro de duración mínima."""
    return enforce_min_duration((p_crisis > threshold).astype(int), min_days)


def transition_matrix(states: np.ndarray) -> np.ndarray:
    """Matriz de transición empírica de una secuencia 0/1."""
    n = np.zeros((2, 2))
    np.add.at(n, (states[:-1], states[1:]), 1)
    return n / n.sum(axis=1, keepdims=True)


def regime_stats(states: np.ndarray) -> dict[str, object]:
    """% de días, duración media observada, nº de cambios, matriz y duración esperada 1/(1−pii)."""
    p = transition_matrix(states)
    runs = _runs(states)
    dur = {k: [f - i for i, f, e in runs if e == k] for k in (0, 1)}
    return {
        "pct_crisis": float(states.mean()),
        "n_cambios": len(runs) - 1,
        "duracion_media": {k: float(np.mean(v)) if v else float("nan") for k, v in dur.items()},
        "duracion_esperada": {k: float(1.0 / (1.0 - p[k, k])) if p[k, k] < 1 else float("inf") for k in (0, 1)},
        "transmat": p.tolist(),
    }


def _narrativa(ini: pd.Timestamp, fin: pd.Timestamp) -> str:
    """Etiqueta el episodio con el suceso conocido que más días solapa."""
    mejor, mejor_dias = "Sin etiqueta histórica: revisar manualmente", 0
    for a, b, texto in config.KNOWN_EVENTS:
        dias = (min(fin, pd.Timestamp(b)) - max(ini, pd.Timestamp(a))).days
        if dias > mejor_dias:
            mejor, mejor_dias = texto, dias
    return mejor


def episodes_table(states: pd.Series, sp500: pd.Series, vix: pd.Series) -> pd.DataFrame:
    """Un episodio de crisis por fila: fechas, duración, caída del S&P, pico del VIX y narrativa."""
    filas = []
    for i, f, e in _runs(states.to_numpy()):
        if e != 1:
            continue
        idx = states.index[i:f]
        px = sp500.reindex(idx)
        previo = sp500.loc[:idx[0]].iloc[-2] if sp500.index.get_loc(idx[0]) > 0 else px.iloc[0]
        filas.append({
            "inicio": idx[0].date(), "fin": idx[-1].date(), "dias": len(idx),
            "caida_sp500": float(px.min() / previo - 1.0), "pico_vix": float(vix.reindex(idx).max()),
            "narrativa": _narrativa(idx[0], idx[-1]),
        })
    return pd.DataFrame(filas)


def duration_sensitivity(p_crisis: np.ndarray) -> pd.DataFrame:
    """Sensibilidad de la decisión dura a la duración mínima."""
    filas = {}
    for m in config.MIN_EPISODE_SENSITIVITY:
        st = hard_states(p_crisis, min_days=m)
        e = regime_stats(st)
        filas[m] = {"pct_crisis": e["pct_crisis"], "n_cambios": e["n_cambios"],
                    "n_episodios_crisis": int(len([r for r in _runs(st) if r[2] == 1]))}
    return pd.DataFrame(filas).T.rename_axis("min_dias")
