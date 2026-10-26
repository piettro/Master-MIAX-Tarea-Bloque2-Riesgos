"""Tests de la Fase 5: simulador, cartera, riesgo y validación (solo datos sintéticos)."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from stresslab import config
from stresslab.data.synthetic import SyntheticParams, generate
from stresslab.domain import copulas, marginals, portfolio, regimes, risk, simulator, validation

NAMES = ["A", "B", "C", "D"]
P = np.array([[0.99, 0.01], [0.04, 0.96]])


@pytest.fixture(scope="module")
def modelo():
    s = generate(30_000, 4, seed=41)
    r = s.returns.copy()
    r.columns = NAMES
    st = pd.Series(s.states, index=r.index)
    _, marg = marginals.fit_all(r, st)
    cops = {k: copulas.fit_t(copulas.pseudo_obs(r[st == k].to_numpy()[:6000])) for k in (0, 1)}
    return r, st, marg, cops


def _sim(modelo, seed=1, n=2000, h=126, **kw):
    r, _, marg, cops = modelo
    return simulator.simulate(P, marg, cops, NAMES, n, h, config.rng(seed), hist=r.to_numpy(), **kw)


def test_riesgo_normal_cerrado() -> None:
    x = np.random.default_rng(0).normal(0, 1, 2_000_000)
    assert risk.var(x) == pytest.approx(2.326, abs=0.02)
    assert risk.es(x) == pytest.approx(2.665, abs=0.03)
    v, e = risk.normal_var_es(0.0, 1.0)
    assert v == pytest.approx(2.3263, abs=1e-3) and e == pytest.approx(2.6652, abs=1e-3) and e >= v


def test_riqueza_buy_and_hold_formula_cerrada() -> None:
    r = np.zeros((1, 2, 2))
    r[0, :, 0], r[0, :, 1] = [0.10, 0.10], [-0.10, 0.0]  # Dos activos, dos días.
    w = portfolio.wealth_paths(r)
    esperado = [1.0, (1.1 + 0.9) / 2, (1.1 * 1.1 + 0.9 * 1.0) / 2]
    assert np.allclose(w[0], esperado)


def test_max_drawdown_y_retornos_diarios() -> None:
    w = np.array([[1.0, 1.2, 0.6, 0.9]])
    assert risk.max_drawdown(w)[0] == pytest.approx(0.5)
    assert np.allclose(portfolio.daily_returns(w)[0], [0.2, -0.5, 0.5])


def test_ventanas_moviles_forma() -> None:
    x = np.arange(20.0).reshape(10, 2)
    assert portfolio.rolling_windows(x, 4).shape == (7, 4, 2)
    assert portfolio.rolling_windows(np.arange(10.0), 4).shape == (7, 4)


def test_mismo_seed_identico_y_seed_distinto_no(modelo) -> None:
    a, b, c = _sim(modelo, 7, 300, 30), _sim(modelo, 7, 300, 30), _sim(modelo, 8, 300, 30)
    assert np.array_equal(a.returns, b.returns) and np.array_equal(a.states, b.states)
    assert not np.array_equal(a.returns, c.returns)


def test_transiciones_empiricas_cercanas_a_la_matriz(modelo) -> None:
    s = _sim(modelo, 3, 4000, 126).states
    emp = np.zeros((2, 2))
    np.add.at(emp, (s[:, :-1].ravel(), s[:, 1:].ravel()), 1)
    emp /= emp.sum(axis=1, keepdims=True)
    assert np.allclose(emp, P, atol=0.01)


def test_estado_inicial_fijo_en_crisis(modelo) -> None:
    s = _sim(modelo, 4, 3000, 5, initial=1).states
    assert s[:, 0].mean() == pytest.approx(0.96, abs=0.02)  # Primer día: P(seguir en crisis) = p11.


def test_simulador_recupera_vol_y_correlacion_por_estado(modelo) -> None:
    s = _sim(modelo, 5, 3000, 126)
    flat, st = s.returns.reshape(-1, 4), s.states.reshape(-1)
    for k, vol in enumerate(SyntheticParams().daily_vol):
        assert flat[st == k].std(axis=0).mean() == pytest.approx(vol, rel=0.12)
    assert copulas.mean_offdiag(np.corrcoef(flat[st == 1], rowvar=False)) == pytest.approx(0.7, abs=0.06)


def test_tope_de_retornos_se_aplica(modelo) -> None:
    r = modelo[0]
    lo, hi = simulator.return_bounds(r.to_numpy())
    s = _sim(modelo, 6, 500, 126)
    assert (s.returns >= lo - 1e-12).all() and (s.returns <= hi + 1e-12).all()


def test_tablas_de_validacion_sobre_sinteticos(modelo) -> None:
    r, st, _, _ = modelo
    s = _sim(modelo, 9, 1500, 126)
    w = portfolio.wealth_paths(s.returns)
    rw = portfolio.wealth_paths(portfolio.rolling_windows(r.to_numpy()[:5000], 126))
    sd, rd = portfolio.daily_returns(w), portfolio.daily_returns(rw)
    t = validation.risk_table(rd, rw, sd, w)
    assert t.loc["es99_diario"].min() >= t.loc["var99_diario"].min()
    assert t.loc["vol_diaria", "simulado"] == pytest.approx(t.loc["vol_diaria", "real"], rel=0.25)
    rg = validation.regime_table(s.states, portfolio.rolling_windows(st.to_numpy()[:5000], 126), P)
    assert rg.loc["pct_dias_crisis", "simulado"] == pytest.approx(rg.loc["pct_dias_crisis", "teorico_cadena"], abs=0.03)
    cov = validation.window_coverage(rw[:, -1] - 1, w[:, -1] - 1)
    assert 0.5 < cov["cobertura_p5_p95_real"] <= 1.0
