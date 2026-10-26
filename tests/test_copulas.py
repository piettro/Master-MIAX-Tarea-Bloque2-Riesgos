"""Tests de la Fase 4 con datos sintéticos (calma Gauss ρ=0,3; crisis t df=4 ρ=0,7)."""
import numpy as np
import pytest

from stresslab import config
from stresslab.data.synthetic import generate
from stresslab.domain import copulas

D = 6


@pytest.fixture(scope="module")
def datos():
    s = generate(40_000, D, seed=31)
    x = s.returns.to_numpy()
    return x[s.states == 0][:6000], x[s.states == 1][:4000]


def test_t_copula_recupera_df_y_correlacion(datos) -> None:
    _, x1 = datos
    f = copulas.fit_t(copulas.pseudo_obs(x1))
    assert 2.5 < f.df < 7
    assert copulas.mean_offdiag(f.corr) == pytest.approx(0.7, abs=0.05)


def test_gauss_recupera_correlacion_en_calma(datos) -> None:
    x0, _ = datos
    f = copulas.fit_gauss(copulas.pseudo_obs(x0))
    assert copulas.mean_offdiag(f.corr) == pytest.approx(0.3, abs=0.04)


def test_seleccion_gauss_en_calma_y_t_en_crisis(datos) -> None:
    for x, ganadora in ((datos[0], "gauss"), (datos[1], "t")):
        u = copulas.pseudo_obs(x)
        g, t = copulas.fit_gauss(u), copulas.fit_t(u)
        assert min((g, t), key=lambda f: f.aic).family == ganadora


def test_muestreo_reproducible_y_uniforme(datos) -> None:
    f = copulas.fit_t(copulas.pseudo_obs(datos[1]))
    a, b = f.sample(2000, config.rng(1)), f.sample(2000, config.rng(1))
    assert np.array_equal(a, b)
    assert abs(a.mean() - 0.5) < 0.02 and a.min() > 0 and a.max() < 1


def test_t_tiene_mas_dependencia_de_cola_que_gauss(datos) -> None:
    u = copulas.pseudo_obs(datos[1])
    lam_emp = copulas.lower_tail_dependence(u)
    lam_t = copulas.implied_tail_dependence(copulas.fit_t(u), config.rng(2))
    lam_g = copulas.implied_tail_dependence(copulas.fit_gauss(u), config.rng(2))
    assert lam_t > lam_g and abs(lam_t - lam_emp) < abs(lam_g - lam_emp)


def test_bootstrap_y_correlaciones_no_llegan_a_uno(datos) -> None:
    x0, x1 = datos
    b0 = copulas.bootstrap_mean_corr(x0, config.rng(3), 200)
    b1 = copulas.bootstrap_mean_corr(x1, config.rng(3), 200)
    dif = copulas.bootstrap_corr_diff(x0, x1, config.rng(3), 200)
    assert b0["ic_inf"] < b0["media"] < b0["ic_sup"] and b1["ic_sup"] < 0.9
    assert dif["ic_inf"] > 0 and dif["p_menor_igual_cero"] == 0.0


def test_descomposicion_de_riesgo_coherente(datos) -> None:
    d = copulas.risk_decomposition(*datos)
    assert d["vol_crisis"] > d["vol_solo_vols_crisis"] > d["vol_calma"]
    assert d["vol_crisis"] > d["vol_solo_corr_crisis"] > d["vol_calma"]


def test_nearest_corr_devuelve_matriz_valida() -> None:
    malo = np.array([[1, 0.9, -0.9], [0.9, 1, 0.9], [-0.9, 0.9, 1]])
    c = copulas.nearest_corr(malo)
    assert np.all(np.linalg.eigvalsh(c) > 0) and np.allclose(np.diag(c), 1)
