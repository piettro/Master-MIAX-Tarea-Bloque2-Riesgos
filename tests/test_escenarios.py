"""Tests de la Fase 6: calibración y estrés de escenarios (solo datos sintéticos)."""
import numpy as np
import pandas as pd
import pytest

from stresslab import config
from stresslab.data.synthetic import generate
from stresslab.domain import copulas, marginals, portfolio, scenarios, simulator

NAMES = ["A", "B", "C", "D"]
P = np.array([[0.99, 0.01], [0.04, 0.96]])


@pytest.fixture(scope="module")
def modelo():
    s = generate(30_000, 4, seed=51)
    r = s.returns.copy()
    r.columns = NAMES
    st = pd.Series(s.states, index=r.index)
    _, marg = marginals.fit_all(r, st)
    cops = {k: copulas.fit_t(copulas.pseudo_obs(r[st == k].to_numpy()[:5000])) for k in (0, 1)}
    return r, st, marg, cops


def test_stress_marginal_multiplica_la_vol_exactamente() -> None:
    m = marginals.Marginal("t", (4.5, 0.0, 0.005), 0.0, 1000)
    s = scenarios.stress_marginal(m, 1.5)
    assert s.params[0] == 3.5 and s.dist.std() == pytest.approx(1.5 * m.dist.std(), rel=1e-9)


def test_stress_marginal_respeta_el_suelo_de_df() -> None:
    m = marginals.Marginal("t", (3.0, 0.0, 0.01), 0.0, 1000)
    assert scenarios.stress_marginal(m, 1.5).params[0] == marginals.DF_BOUNDS[0]


def test_stress_copula_mezcla_correlaciones_y_es_valida(modelo) -> None:
    c = modelo[3][1]
    e = scenarios.stress_copula(c, config.E3_COPULA_DF, 0.3)
    esperado = 0.7 * copulas.mean_offdiag(c.corr) + 0.3
    assert copulas.mean_offdiag(e.corr) == pytest.approx(esperado, abs=1e-9)
    assert np.all(np.linalg.eigvalsh(e.corr) > 0) and e.df <= c.df


def test_ventana_usa_dias_de_crisis_o_cae_a_todos() -> None:
    idx = pd.bdate_range("2022-01-03", periods=200)
    r = pd.DataFrame(np.zeros((200, 2)), index=idx, columns=["A", "B"])
    st = pd.Series(0, index=idx)
    st.iloc[:80] = 1
    a, modo_a = scenarios.window_returns(r, st, ("2022-01-01", "2022-12-31"), 60)
    b, modo_b = scenarios.window_returns(r, st, ("2022-01-01", "2022-12-31"), 100)
    assert len(a) == 80 and modo_a == "días de crisis" and len(b) == 200 and "todos" in modo_b


def test_calibracion_recupera_crisis_sintetica(modelo) -> None:
    r, st, _, _ = modelo
    cm, cc = scenarios.calibrate_crisis(r[st == 1].iloc[:600], NAMES)
    assert cm[("A", 1)].params[-1] > 0.015  # La escala de crisis (vol 3 %) es mayor que la de calma.
    assert copulas.mean_offdiag(cc.corr) == pytest.approx(0.7, abs=0.08)


def test_subset_copula_mantiene_familia_y_df(modelo) -> None:
    c = modelo[3][1]
    s = scenarios.subset_copula(c, [0, 2, 3])
    assert s.corr.shape == (3, 3) and s.df == c.df and np.allclose(np.diag(s.corr), 1)


def test_e3_aumenta_el_riesgo_y_arrancar_en_crisis_tambien(modelo) -> None:
    r, _, marg, cops = modelo
    hist = r.to_numpy()
    base = simulator.simulate(P, marg, cops, NAMES, 2500, 126, config.rng(1), "estacionaria", hist)
    crisis = simulator.simulate(P, marg, cops, NAMES, 2500, 126, config.rng(1), 1, hist)
    m3, c3 = scenarios.e3_model(marg, cops, NAMES)
    e3 = simulator.simulate(P, m3, c3, NAMES, 2500, 126, config.rng(1), 1, hist)
    met = [scenarios.scenario_metrics(portfolio.wealth_paths(s.returns)) for s in (base, crisis, e3)]
    assert met[0]["es99_6m"] < met[1]["es99_6m"] < met[2]["es99_6m"]
    assert met[2]["vol_anualizada"] > met[1]["vol_anualizada"]


def test_metricas_y_validacion_contra_ventana() -> None:
    w = np.tile(np.linspace(1.0, 0.8, 127), (50, 1))
    m = scenarios.scenario_metrics(w)
    assert m["var99_6m"] == pytest.approx(0.2) and m["prob_perdida_20pct_6m"] == 0.0
    real = np.full((126, 2), -0.002)
    v = scenarios.validate_against_window(w, real)
    assert v["dias_ventana_real"] == 126 and 0.0 <= v["percentil_retorno_real"] <= 1.0
