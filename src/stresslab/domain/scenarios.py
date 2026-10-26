"""Escenarios de estrés: recalibración del estado de crisis (E1, E2) y shock paramétrico (E3)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from stresslab import config
from stresslab.domain import copulas, marginals, portfolio, risk
from stresslab.domain.copulas import CopulaFit
from stresslab.domain.marginals import Marginal


def window_returns(returns: pd.DataFrame, states: pd.Series, window: tuple[str, str],
                   min_obs: int = config.SCEN_MIN_OBS) -> tuple[pd.DataFrame, str]:
    """Retornos de los días de crisis de la ventana; si son pocos, todos los de la ventana."""
    r = returns.loc[window[0]:window[1]]
    crisis = r[states.reindex(r.index) == 1]
    return (crisis, "días de crisis") if len(crisis) >= min_obs else (r, "todos los días de la ventana")


def calibrate_crisis(window_ret: pd.DataFrame, names: list[str]) -> tuple[dict[tuple[str, int], Marginal], CopulaFit]:
    """Marginales (Normal/t por BIC) y cópula t del estado de crisis con los datos de la ventana."""
    x = window_ret[names].dropna()
    marg = {(n, 1): marginals.select_marginal(x[n].to_numpy())[0] for n in names}
    return marg, copulas.fit_t(copulas.pseudo_obs(x.to_numpy()))


def with_crisis(base_marg: dict, base_cops: dict[int, CopulaFit], crisis_marg: dict,
                crisis_cop: CopulaFit) -> tuple[dict, dict[int, CopulaFit]]:
    """Modelo base con el estado de crisis sustituido por el recalibrado (la calma no cambia)."""
    return {**base_marg, **crisis_marg}, {0: base_cops[0], 1: crisis_cop}


def stress_marginal(m: Marginal, vol_mult: float) -> Marginal:
    """Volatilidad exactamente × vol_mult y un grado de libertad menos (suelo DF_BOUNDS)."""
    if m.family != "t":
        return Marginal("normal", (m.params[0], m.params[1] * vol_mult), float("nan"), m.n)
    df, mu, scale = m.params
    df2 = max(marginals.DF_BOUNDS[0], df - 1.0)
    factor = np.sqrt(((df2 - 2.0) / df2) / ((df - 2.0) / df))  # Compensa el cambio de varianza al bajar los g.l.
    return Marginal("t", (df2, mu, scale * vol_mult * factor), float("nan"), m.n)


def stress_copula(c: CopulaFit, df: float, lam: float) -> CopulaFit:
    """Cópula t con R' = (1−λ)R + λJ (J = unos) y g.l. reducidos; R' sigue siendo correlación válida."""
    r = (1.0 - lam) * c.corr + lam * np.ones_like(c.corr)
    return CopulaFit("t", r, min(df, c.df or df), float("nan"), c.n)


def e3_model(base_marg: dict, base_cops: dict[int, CopulaFit], names: list[str]) -> tuple[dict, dict[int, CopulaFit]]:
    """E3 pánico de liquidez crediticia: solo se estresa el estado de crisis."""
    marg = {**base_marg, **{(n, 1): stress_marginal(base_marg[(n, 1)], config.E3_VOL_MULT) for n in names}}
    return marg, {0: base_cops[0], 1: stress_copula(base_cops[1], config.E3_COPULA_DF, config.E3_LAMBDA)}


def subset_copula(c: CopulaFit, idx: list[int]) -> CopulaFit:
    """Submatriz de una cópula elíptica: sigue siendo la misma familia con los mismos g.l."""
    return CopulaFit(c.family, c.corr[np.ix_(idx, idx)], c.df, c.loglik, c.n)


def scenario_metrics(w: np.ndarray) -> dict[str, float]:
    """VaR/ES 99 % (diario, 1 mes, 6 meses), volatilidad, drawdown y probabilidad de pérdida > 20 %."""
    d, mdd = portfolio.daily_returns(w).ravel(), risk.max_drawdown(w)
    m1, m6 = w[:, 21] - 1.0, w[:, -1] - 1.0
    return {"vol_anualizada": risk.annualize_vol(float(d.std())),
            "var99_diario": risk.var(d), "es99_diario": risk.es(d),
            "var99_1m": risk.var(m1), "es99_1m": risk.es(m1),
            "var99_6m": risk.var(m6), "es99_6m": risk.es(m6),
            "mdd_medio_6m": float(mdd.mean()), "mdd_p95_6m": float(np.quantile(mdd, 0.95)),
            "retorno_mediano_6m": float(np.median(m6)), "prob_perdida_20pct_6m": float((m6 < -0.20).mean())}


def validate_against_window(w: np.ndarray, real_window: np.ndarray) -> dict[str, float]:
    """Misma validación de R-ESC-01/02: dónde cae la ventana real dentro de la distribución simulada."""
    h = min(len(real_window), w.shape[1] - 1)
    rw = portfolio.wealth_paths(real_window[None, :h, :])
    term = float(rw[0, -1] - 1.0)
    sim_term = w[:, h] - 1.0
    mdd_real = float(risk.max_drawdown(rw)[0])
    return {"dias_ventana_real": h, "retorno_real": term, "percentil_retorno_real": float((sim_term < term).mean()),
            "mdd_real": mdd_real, "percentil_mdd_real": float((risk.max_drawdown(w[:, : h + 1]) < mdd_real).mean())}
