"""Pipeline: ejecuta las fases con traza. Fase 2 = regímenes."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stresslab import config
from stresslab.data import audit
from stresslab.data.returns import asset_returns
from stresslab.domain import copulas, features, marginals, portfolio, regimes, scenarios, simulator, validation
from stresslab.orchestration.trace import Trace


@dataclass
class RegimeResult:
    """Salidas de la Fase 2."""

    X: pd.DataFrame
    p_smooth: pd.Series
    p_filter: pd.Series
    states: pd.Series
    fit: regimes.RegimeFit
    episodes: pd.DataFrame
    sensitivity: pd.DataFrame
    summary: dict[str, object]


def run_regimes(prices: pd.DataFrame, rates: pd.DataFrame, trace: Trace) -> RegimeResult:
    """Factores -> HMM -> probabilidades -> estados duros -> estadísticas y episodios."""
    with trace.step("factores", "Δ S&P, Δlog VIX, Δ(10a-2a), Δ OAS; z-score") as st:
        x = features.build_features(prices, rates)
        z = features.standardize(x)
        st.n_obs = len(z)
    with trace.step("hmm", f"GaussianHMM {config.N_STATES} estados, EM x{config.HMM_N_INIT} inicios") as st:
        fit = regimes.fit_hmm(z.to_numpy())
        st.n_obs, st.nota = len(z), f"loglik={fit.loglik:.1f}"
    with trace.step("probabilidades", "suavizada (smoothing) y filtrada (forward)") as st:
        p_s = pd.Series(regimes.smoothed_proba(fit, z.to_numpy())[:, 1], index=z.index)
        p_f = pd.Series(regimes.filtered_proba(fit, z.to_numpy())[:, 1], index=z.index)
        st.n_obs = len(z)
    with trace.step("estados_duros", f"umbral {config.CRISIS_THRESHOLD} + duración mínima {config.MIN_EPISODE_DAYS} d") as st:
        states = pd.Series(regimes.hard_states(p_s.to_numpy()), index=z.index)
        stats = regimes.regime_stats(states.to_numpy())
        st.n_obs, st.nota = len(states), f"{stats['pct_crisis']:.1%} días en crisis"
    with trace.step("episodios", "tabla de episodios de crisis y sensibilidad") as st:
        sp, vix = prices[config.MARKET_TICKER].dropna(), prices[config.VIX_TICKER].dropna()
        epi = regimes.episodes_table(states, sp, vix)
        sens = regimes.duration_sensitivity(p_s.to_numpy())
        st.n_obs, st.nota = len(epi), "episodios de crisis"
    resumen = {
        **stats, "loglik": fit.loglik,
        "transmat_hmm": fit.transmat.tolist(),
        "medias_z": fit.means.tolist(),
        "acuerdo_filtrado_suavizado": float(((p_f > 0.5) == (p_s > 0.5)).mean()),
        "acuerdo_hmm_solo_sp500": None,
    }
    res = RegimeResult(x, p_s, p_f, states, fit, epi, sens, resumen)
    with trace.step("hmm_solo_sp500", "contraste D-04: HMM solo sobre el S&P 500") as st:
        res.summary["acuerdo_hmm_solo_sp500"] = sp500_only_agreement(prices, res)
        st.n_obs = len(states)
    return res


@dataclass
class MarginalResult:
    """Salidas de la Fase 3."""

    returns: pd.DataFrame
    moments: pd.DataFrame
    fit_table: pd.DataFrame
    marginals: dict
    refuge: dict[str, float]


def run_marginals(prices: pd.DataFrame, rates: pd.DataFrame, states: pd.Series, trace: Trace) -> MarginalResult:
    """Retornos de cartera -> momentos condicionales -> ajuste Normal/t -> respuestas HYG y GLD."""
    with trace.step("retornos_activos", "equities+HYG (precio) y bonos (duración)") as st:
        rets = asset_returns(prices, rates).reindex(states.index)
        st.n_obs, st.nota = int(rets.notna().sum().sum()), f"{rets.shape[1]} activos"
    with trace.step("momentos", "media, vol, skew, curtosis por activo y estado") as st:
        mom = marginals.conditional_moments(rets, states)
        st.n_obs = len(rets)
    with trace.step("ajuste_marginales", "Normal vs t por MLE, selección BIC") as st:
        tabla, elegidas = marginals.fit_all(rets, states)
        st.n_obs, st.nota = len(tabla), f"{sum(m.family == 't' for m in elegidas.values())} t de {len(elegidas)}"
    with trace.step("refugio", "vol HYG y comportamiento de GLD") as st:
        mercado = prices[config.MARKET_TICKER].pct_change().reindex(states.index)
        refugio = marginals.refuge_answers(rets, states, mercado)
        st.n_obs = len(states)
    return MarginalResult(rets, mom, tabla, elegidas, refugio)


@dataclass
class CopulaResult:
    """Salidas de la Fase 4."""

    names: list[str]
    corr: dict[int, np.ndarray]
    fits: dict[int, dict[str, copulas.CopulaFit]]
    table: pd.DataFrame
    summary: dict[str, object]


def _estado_profile(x: dict[int, np.ndarray], corr: dict[int, np.ndarray], u: dict[int, np.ndarray],
                    prices: pd.DataFrame, rates: pd.DataFrame, st: pd.Series) -> dict[str, float]:
    """Características que distinguen crisis de calma (R-COP-04), todas medibles."""
    d = x[0].shape[1]
    iu = np.triu_indices(d, 1)
    niv = st.index
    vix = prices[config.VIX_TICKER].reindex(niv)
    oas = rates[config.CREDIT_OAS].reindex(niv, method="ffill")
    return {
        "mediana_ratio_vol": float(np.median(x[1].std(axis=0) / x[0].std(axis=0))),
        "corr_media_calma": copulas.mean_offdiag(corr[0]), "corr_media_crisis": copulas.mean_offdiag(corr[1]),
        "spearman_medio_calma": copulas.mean_offdiag(np.corrcoef(u[0], rowvar=False)),
        "spearman_medio_crisis": copulas.mean_offdiag(np.corrcoef(u[1], rowvar=False)),
        "pct_pares_corr_sube": float((corr[1][iu] > corr[0][iu]).mean()),
        "lambda_L_calma": copulas.lower_tail_dependence(u[0]), "lambda_L_crisis": copulas.lower_tail_dependence(u[1]),
        "vix_medio_calma": float(vix[st == 0].mean()), "vix_medio_crisis": float(vix[st == 1].mean()),
        "oas_medio_calma": float(oas[st == 0].mean()), "oas_medio_crisis": float(oas[st == 1].mean()),
    }


def run_copulas(returns: pd.DataFrame, states: pd.Series, prices: pd.DataFrame, rates: pd.DataFrame,
                trace: Trace) -> CopulaResult:
    """Ventana común -> Pearson -> bootstrap -> Gauss y t por estado -> cola -> perfil de estados."""
    rng = config.rng()
    with trace.step("ventana_comun", "días con los 18 activos (D-05)") as st:
        r = returns.dropna()
        s = states.reindex(r.index)
        x = {k: r[s == k].to_numpy() for k in (0, 1)}
        st.n_obs, st.nota = len(r), f"calma={len(x[0])}, crisis={len(x[1])}, desde {r.index[0].date()}"
    corr = {k: np.corrcoef(x[k], rowvar=False) for k in (0, 1)}
    with trace.step("bootstrap_correlaciones", "1000 remuestreos de días, IC 95 %") as st:
        boot = {k: copulas.bootstrap_mean_corr(x[k], rng) for k in (0, 1)}
        boot["dif"] = copulas.bootstrap_corr_diff(x[0], x[1], rng)
        st.n_obs = len(r)
    fits, u, filas = {}, {}, []
    for k, nombre in marginals.STATE_NAMES.items():
        with trace.step(f"copulas_{nombre}", "Gauss (scores) y t (tau de Kendall + MLE de df)") as st:
            u[k] = copulas.pseudo_obs(x[k])
            fits[k] = {"gauss": copulas.fit_gauss(u[k]), "t": copulas.fit_t(u[k])}
            emp = copulas.lower_tail_dependence(u[k])
            for fam, f in fits[k].items():
                filas.append({"estado": nombre, "familia": fam, "loglik": f.loglik, "aic": f.aic, "bic": f.bic,
                              "df": f.df, "lambda_L_ajustada": copulas.implied_tail_dependence(f, rng),
                              "lambda_L_empirica": emp})
            st.n_obs, st.nota = len(u[k]), f"df_t={fits[k]['t'].df:.1f}"
    tabla = pd.DataFrame(filas).set_index(["estado", "familia"])
    perfil = _estado_profile(x, corr, u, prices, rates, s)
    resumen = {"n_ventana": len(r), "n_calma": len(x[0]), "n_crisis": len(x[1]), "bootstrap": boot,
               "descomposicion_riesgo": copulas.risk_decomposition(x[0], x[1]), "perfil_estados": perfil,
               "elegida_bic": {nm: min(fits[k], key=lambda f: getattr(fits[k][f], "bic"))
                               for k, nm in marginals.STATE_NAMES.items()}}
    return CopulaResult(list(r.columns), corr, fits, tabla, resumen)


@dataclass
class SimulationResult:
    """Salidas de la Fase 5 (escenario base)."""

    sim: simulator.SimResult
    wealth: np.ndarray
    real_wealth: np.ndarray
    transmat: np.ndarray
    fan: np.ndarray  # Cuantiles 5, 25, 50, 75, 95 de la riqueza simulada: (T+1, 5).
    sim_daily: np.ndarray
    real_daily: np.ndarray
    tables: dict[str, pd.DataFrame]
    summary: dict[str, object]


def base_copulas(cop: CopulaResult) -> dict[int, copulas.CopulaFit]:
    """Cópula elegida por BIC en cada estado."""
    elegidas = cop.summary["elegida_bic"]
    return {k: cop.fits[k][elegidas[nombre]] for k, nombre in marginals.STATE_NAMES.items()}


def run_simulation(mar: MarginalResult, cop: CopulaResult, states: pd.Series, trace: Trace,
                   n_sims: int = config.N_SIMS, horizon: int = config.HORIZON_DAYS) -> SimulationResult:
    """Monte Carlo base (S_0 estacionaria) y validación A/B/C frente a ventanas móviles reales."""
    rng = config.rng()
    real = mar.returns[cop.names].dropna()
    real_ret, real_st = real.to_numpy(), states.reindex(real.index).to_numpy()
    cops = base_copulas(cop)
    p = regimes.transition_matrix(states.to_numpy())
    with trace.step("montecarlo", f"{n_sims} trayectorias x {horizon} d: Markov -> cópula -> marginales") as st:
        hist = mar.returns[cop.names].to_numpy()  # Historia completa por activo (con NaN) para los topes.
        sim = simulator.simulate(p, mar.marginals, cops, cop.names, n_sims, horizon, rng, "estacionaria", hist)
        st.n_obs, st.nota = n_sims * horizon, f"{sim.n_clipped} retornos topados de {sim.returns.size}"
    with trace.step("validacion", "ventanas móviles reales de 126 d vs simulado (A/B/C y estrés)") as st:
        w = portfolio.wealth_paths(sim.returns)
        rw = portfolio.wealth_paths(portfolio.rolling_windows(real_ret, horizon))
        sd, rd = portfolio.daily_returns(w), portfolio.daily_returns(rw)
        sin = [i for i, n in enumerate(cop.names) if n != "GME"]  # Sensibilidad: GME (enero 2021) domina las colas reales.
        w_x = portfolio.wealth_paths(sim.returns[:, :, sin])
        rw_x = portfolio.wealth_paths(portfolio.rolling_windows(real_ret[:, sin], horizon))
        tablas = {
            "regimenes": validation.regime_table(sim.states, portfolio.rolling_windows(states.to_numpy(), horizon), p),
            "riesgo": validation.risk_table(rd, rw, sd, w),
            "riesgo_sin_GME": validation.risk_table(portfolio.daily_returns(rw_x), rw_x, portfolio.daily_returns(w_x), w_x),
            "percentiles_diarios": validation.daily_percentiles(rd, sd),
            "estres": validation.stress_check(sim.returns, sim.states, real_ret, real_st, rng),
        }
        cobertura = validation.window_coverage(rw[:, -1] - 1.0, w[:, -1] - 1.0)
        st.n_obs = len(rw)
    fan = np.quantile(w, [0.05, 0.25, 0.5, 0.75, 0.95], axis=0).T
    resumen = {"transmat_usada": p.tolist(), "n_sims": n_sims, "horizon": horizon, "n_clipped": sim.n_clipped,
               "n_ventanas_reales": len(rw), "cobertura": cobertura,
               "tablas": {k: v.round(6).reset_index().astype(str).to_dict("records") for k, v in tablas.items()}}
    return SimulationResult(sim, w, rw, p, fan, sd, rd, tablas, resumen)


@dataclass
class ScenarioResult:
    """Salidas de la Fase 6."""

    outcomes: dict[str, dict]
    comparativa: pd.DataFrame
    calibracion: pd.DataFrame
    perfil_crisis: pd.DataFrame
    sensibilidad: pd.DataFrame
    validacion_ventana: pd.DataFrame
    summary: dict[str, object]


def _corr_bonos_acciones(corr: np.ndarray, names: list[str]) -> float:
    """Correlación media entre los bonos (10a, 2a) y las acciones del universo (fallo de diversificación si > 0)."""
    bonos = [i for i, n in enumerate(names) if n in config.BOND_MATURITIES]
    acciones = [i for i, n in enumerate(names) if n in config.EQUITIES]
    return float(corr[np.ix_(bonos, acciones)].mean())


def _simular_escenario(etiqueta: str, marg: dict, cops: dict, names: list[str], hist: np.ndarray, n: int,
                       p: np.ndarray, rng: np.random.Generator, real_window: np.ndarray | None = None) -> dict:
    """Simula arrancando en crisis y devuelve riqueza, métricas, perfil del estado de crisis y validación."""
    sim = simulator.simulate(p, marg, cops, names, n, config.HORIZON_DAYS, rng, 1, hist)
    w = portfolio.wealth_paths(sim.returns)
    flat, fs = sim.returns.reshape(-1, len(names)), sim.states.reshape(-1)
    sel = rng.choice(np.flatnonzero(fs == 1), size=100_000, replace=False)
    x = flat[sel]
    perfil = {"vol_mediana_activos": float(np.median(x.std(axis=0))),
              "corr_media": copulas.mean_offdiag(np.corrcoef(x, rowvar=False)),
              "lambda_L_5pct": copulas.lower_tail_dependence(copulas.pseudo_obs(x))}
    val = scenarios.validate_against_window(w, real_window) if real_window is not None else None
    return {"etiqueta": etiqueta, "w": w, "metricas": scenarios.scenario_metrics(w), "perfil": perfil,
            "validacion": val, "n_clipped": sim.n_clipped, "fan": np.quantile(w, [0.05, 0.25, 0.5, 0.75, 0.95], axis=0).T}


def run_scenarios(prices: pd.DataFrame, mar: MarginalResult, cop: CopulaResult, states: pd.Series,
                  base: SimulationResult, trace: Trace) -> ScenarioResult:
    """E1 (2022), E2 (2008, con proxy de ENPH) y E3 (pánico de liquidez), más sensibilidades."""
    rng = config.rng()
    names, p = cop.names, base.transmat
    base_cops = base_copulas(cop)
    with trace.step("proxy_enph", "MCO sobre S&P 500 + ruido idiosincrático por estado (D-21)") as st:
        mercado = prices[config.MARKET_TICKER].pct_change().reindex(states.index)
        modelo = audit.fit_market_proxy(mar.returns["ENPH"], mercado, states)
        lleno = mar.returns.copy()
        lleno["ENPH"] = audit.fill_with_proxy(mar.returns["ENPH"], mercado, modelo, config.rng(), states)
        ruidoso = audit.ProxyModel(modelo.alpha, modelo.beta, tuple(s * config.PROXY_NOISE_MULT for s in modelo.resid_std))
        lleno_x = mar.returns.copy()
        lleno_x["ENPH"] = audit.fill_with_proxy(mar.returns["ENPH"], mercado, ruidoso, config.rng(), states)
        st.n_obs, st.nota = int(mar.returns["ENPH"].isna().sum()), f"beta={modelo.beta:.2f}, rellena días previos a 2012"
    hist = lleno[names].to_numpy()
    out, calib = {}, []

    def calibrar(etq: str, ventana: tuple[str, str], datos: pd.DataFrame, nombres: list[str]):
        w, modo = scenarios.window_returns(datos[nombres], states, ventana)
        cm, cc = scenarios.calibrate_crisis(w, nombres)
        calib.append({"escenario": etq, "ventana": f"{ventana[0]} a {ventana[1]}", "base": modo, "n_obs": len(w.dropna()),
                      "df_copula": cc.df, "corr_media_copula": copulas.mean_offdiag(cc.corr),
                      "corr_bonos_acciones": _corr_bonos_acciones(cc.corr, nombres)})
        return cm, cc, w

    for etq, vent, datos in (("E1", config.E1_WINDOW, lleno), ("E2", config.E2_WINDOW, lleno)):
        with trace.step(f"escenario_{etq}", f"recalibra crisis con {vent[0]}..{vent[1]}; arranca en crisis") as st:
            cm, cc, w = calibrar(etq, vent, datos, names)
            marg, cops = scenarios.with_crisis(mar.marginals, base_cops, cm, cc)
            real_w = datos[names].loc[vent[0]:].iloc[: config.HORIZON_DAYS].to_numpy()
            out[etq] = _simular_escenario(etq, marg, cops, names, hist, config.N_SIMS, p, rng, real_w)
            out[etq]["real_w"] = portfolio.wealth_paths(real_w[None])
            out[etq]["perfil_real"] = {"vol_mediana_activos": float(np.median(w[names].dropna().std())),
                                       "corr_media": copulas.mean_offdiag(np.corrcoef(w[names].dropna().to_numpy(), rowvar=False))}
            st.n_obs, st.nota = config.N_SIMS * config.HORIZON_DAYS, f"{out[etq]['n_clipped']} topados"
    with trace.step("escenario_E3", f"pánico de liquidez: vol x{config.E3_VOL_MULT}, df copula {config.E3_COPULA_DF}, λ={config.E3_LAMBDA}") as st:
        marg, cops = scenarios.e3_model(mar.marginals, base_cops, names)
        out["E3"] = _simular_escenario("E3", marg, cops, names, hist, config.N_SIMS, p, rng)
        st.n_obs, st.nota = config.N_SIMS * config.HORIZON_DAYS, f"{out['E3']['n_clipped']} topados"
    sens = []
    with trace.step("sensibilidades", f"ventana alternativa, ruido x{config.PROXY_NOISE_MULT} de la proxy, sin ENPH ({config.SENS_SIMS} sims)") as st:
        def sens_run(etq: str, base_etq: str, ventana: tuple[str, str], datos: pd.DataFrame, nombres: list[str]):
            cm, cc, _ = calibrar(etq, ventana, datos, nombres)
            idx = [names.index(n) for n in nombres]
            bm = {k: v for k, v in mar.marginals.items() if k[0] in nombres}
            marg, cops = scenarios.with_crisis(bm, {0: scenarios.subset_copula(base_cops[0], idx), 1: cc}, cm, cc)
            r = _simular_escenario(etq, marg, cops, nombres, datos[nombres].to_numpy(), config.SENS_SIMS, p, rng)
            sens.append({"escenario": base_etq, "variante": etq, **{k: r["metricas"][k] for k in
                         ("vol_anualizada", "var99_6m", "es99_6m", "mdd_medio_6m", "prob_perdida_20pct_6m")}})
        sens_run("E1 ventana a oct-2022", "E1", config.E1_WINDOW_ALT, lleno, names)
        sens_run("E2 ventana post-Lehman", "E2", config.E2_WINDOW_ALT, lleno, names)
        sens_run("E2 ruido proxy x2", "E2", config.E2_WINDOW, lleno_x, names)
        sens_run("E2 sin ENPH", "E2", config.E2_WINDOW, lleno, [n for n in names if n != "ENPH"])
        st.n_obs = len(sens)
    base_m = scenarios.scenario_metrics(base.wealth)
    comp = pd.DataFrame({"Base": base_m, **{k: v["metricas"] for k, v in out.items()}})
    perfil = pd.DataFrame({k: v["perfil"] for k, v in out.items()}).T
    val = pd.DataFrame({k: v["validacion"] for k, v in out.items() if v["validacion"]}).T
    resumen = {"corr_bonos_acciones_base": {"calma": _corr_bonos_acciones(cop.corr[0], names),
                                            "crisis": _corr_bonos_acciones(cop.corr[1], names)},
               "modelo_proxy": {"alpha": modelo.alpha, "beta": modelo.beta, "resid_std": list(modelo.resid_std)},
               "comparativa": comp.round(6).to_dict(), "calibracion": calib, "perfil_crisis": perfil.round(6).to_dict("index"),
               "validacion_ventana_real": val.round(6).to_dict("index"), "sensibilidad": sens,
               "perfil_real_ventana": {k: v.get("perfil_real") for k, v in out.items() if "perfil_real" in v},
               "n_clipped": {k: v["n_clipped"] for k, v in out.items()}}
    return ScenarioResult(out, comp, pd.DataFrame(calib), perfil, pd.DataFrame(sens), val, resumen)


def build_results(reg: RegimeResult, mar: MarginalResult, cop: CopulaResult, sim: SimulationResult,
                  esc: ScenarioResult) -> dict[str, object]:
    """Diccionario de resultados (única fuente de cifras del informe y del notebook)."""
    return {
        "regimenes": reg.summary, "sensibilidad_duracion": reg.sensitivity.to_dict("index"),
        "episodios": reg.episodes.astype(str).to_dict("records"), "refugio": mar.refuge,
        "simulacion_base": sim.summary, "escenarios": esc.summary,
        "copulas": {**cop.summary, "ajuste": {f"{a}|{b}": v for (a, b), v in cop.table.to_dict("index").items()}},
        "marginales_elegidas": {f"{a}|{k}": {"familia": m.family, "params": m.params}
                                for (a, k), m in mar.marginals.items()},
        "parametros": {"seed": config.SEED, "n_sims": config.N_SIMS, "horizonte_dias": config.HORIZON_DAYS,
                       "nivel_var": config.VAR_LEVEL, "min_episodio_dias": config.MIN_EPISODE_DAYS,
                       "inicio_datos": config.START_DATE, "n_activos": len(cop.names),
                       "primer_dia_modelo": str(reg.states.index[0].date()), "ultimo_dia_modelo": str(reg.states.index[-1].date()),
                       "n_dias_modelo": len(reg.states), "umbral_crisis": config.CRISIS_THRESHOLD,
                       "e3_vol_mult": config.E3_VOL_MULT, "cola_q_pct": 100 * copulas.TAIL_Q, "banda_pct": [5, 95], "ic_pct": 95},
    }


def sp500_only_agreement(prices: pd.DataFrame, res: RegimeResult) -> float:
    """Acuerdo entre el HMM multifactor y un HMM solo sobre el S&P 500 (D-04: contraste)."""
    r = res.X[[features.FACTORS[0]]]
    z = ((r - r.mean()) / r.std()).to_numpy()
    f = regimes.fit_hmm(z)
    est = regimes.hard_states(regimes.smoothed_proba(f, z)[:, 1])
    return float((est == res.states.to_numpy()).mean())
