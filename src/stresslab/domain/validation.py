"""Validación real vs simulado: regímenes (B), riesgo (C), cartera (A) y comportamiento en estrés."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from stresslab.domain import copulas, risk
from stresslab.domain.simulator import stationary

PERCENTILES = (0.1, 1, 5, 25, 50, 75, 95, 99, 99.9)


def regime_path_stats(states: np.ndarray) -> dict[str, float]:
    """% de días en crisis, cambios y duración media (truncada al horizonte) sobre trayectorias (n, T)."""
    out = {"pct_dias_crisis": float(states.mean()),
           "cambios_por_ventana": float((np.diff(states, axis=1) != 0).sum(axis=1).mean())}
    for k, nombre in ((0, "calma"), (1, "crisis")):
        entradas = (states[:, 0] == k).sum() + ((states[:, 1:] == k) & (states[:, :-1] != k)).sum()
        out[f"duracion_media_{nombre}"] = float((states == k).sum() / entradas) if entradas else float("nan")
    return out


def regime_table(sim_states: np.ndarray, real_state_windows: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    """R-VAL-02: real (ventanas móviles) frente a simulado y valores teóricos de la cadena."""
    real, sim = regime_path_stats(real_state_windows), regime_path_stats(sim_states)
    teorico = {"pct_dias_crisis": float(stationary(p)[1]),
               "duracion_media_calma": float(1 / (1 - p[0, 0])), "duracion_media_crisis": float(1 / (1 - p[1, 1]))}
    return pd.DataFrame({"real_ventanas": real, "simulado": sim, "teorico_cadena": teorico})


def risk_table(real_daily: np.ndarray, real_w: np.ndarray, sim_daily: np.ndarray, sim_w: np.ndarray) -> pd.DataFrame:
    """R-VAL-03: vol, VaR/ES 99 % diario y a 6 meses y drawdown, real frente a simulado."""
    def col(daily: np.ndarray, w: np.ndarray) -> dict[str, float]:
        d, term, mdd = daily.ravel(), w[:, -1] - 1.0, risk.max_drawdown(w)
        return {"vol_diaria": float(d.std()), "vol_anualizada": risk.annualize_vol(float(d.std())),
                "var99_diario": risk.var(d), "es99_diario": risk.es(d),
                "var99_normal_diario": risk.normal_var_es(d.mean(), d.std())[0],
                "es99_normal_diario": risk.normal_var_es(d.mean(), d.std())[1],
                "var99_6m": risk.var(term), "es99_6m": risk.es(term), "retorno_6m_mediano": float(np.median(term)),
                "mdd_medio_6m": float(mdd.mean()), "mdd_p95_6m": float(np.quantile(mdd, 0.95))}
    return pd.DataFrame({"real": col(real_daily, real_w), "simulado": col(sim_daily, sim_w)})


def window_coverage(real_term: np.ndarray, sim_term: np.ndarray) -> dict[str, float]:
    """Cobertura de la banda p5-p95 y posición media de los retornos reales a 6 meses en la distribución simulada."""
    lo, hi = np.quantile(sim_term, [0.05, 0.95])
    ranks = np.searchsorted(np.sort(sim_term), real_term) / len(sim_term)
    return {"p5_sim": float(lo), "p95_sim": float(hi), "cobertura_p5_p95_real": float(((real_term >= lo) & (real_term <= hi)).mean()),
            "percentil_medio_real": float(ranks.mean()), "ks_estadistico": float(stats.ks_2samp(real_term, sim_term)[0])}


def daily_percentiles(real_daily: np.ndarray, sim_daily: np.ndarray) -> pd.DataFrame:
    """Percentiles del retorno diario de la cartera, real frente a simulado (R-VAL-01)."""
    return pd.DataFrame({"real": np.percentile(real_daily.ravel(), PERCENTILES),
                         "simulado": np.percentile(sim_daily.ravel(), PERCENTILES)},
                        index=pd.Index(PERCENTILES, name="percentil"))


def stress_check(sim_ret: np.ndarray, sim_states: np.ndarray, real_ret: np.ndarray, real_states: np.ndarray,
                 rng: np.random.Generator, n_sub: int = 100_000) -> pd.DataFrame:
    """R-VAL-04: vol, correlación y dependencia de cola por estado, real frente a simulado."""
    flat, fs = sim_ret.reshape(-1, sim_ret.shape[-1]), sim_states.reshape(-1)
    filas = {}
    for k, nombre in ((0, "calma"), (1, "crisis")):
        sel = np.flatnonzero(fs == k)
        sel = rng.choice(sel, size=min(n_sub, len(sel)), replace=False)
        for etiqueta, x in (("real", real_ret[real_states == k]), ("simulado", flat[sel])):
            u = copulas.pseudo_obs(x)
            filas[(nombre, etiqueta)] = {
                "n": len(x), "vol_mediana_activos": float(np.median(x.std(axis=0))),
                "corr_media": copulas.mean_offdiag(np.corrcoef(x, rowvar=False)),
                "lambda_L_5pct": copulas.lower_tail_dependence(u),
                "curtosis_exceso_mediana": float(np.median(stats.kurtosis(x, axis=0)))}
    return pd.DataFrame(filas).T.rename_axis(["estado", "fuente"])
