"""Tests de la Fase 3 con datos sintéticos (calma Gauss, crisis t con 4 g.l.)."""
import numpy as np
import pandas as pd
import pytest

from stresslab import config
from stresslab.data.returns import asset_returns
from stresslab.data.synthetic import SyntheticProvider, generate
from stresslab.domain import marginals


@pytest.fixture(scope="module")
def muestra():
    s = generate(40_000, 3, seed=21)
    r = s.returns.copy()
    r.columns = ["HYG", "GLD", "X"]
    return r, pd.Series(s.states, index=r.index)


def test_momentos_condicionales_recuperan_vol_verdadera(muestra) -> None:
    r, st = muestra
    mom = marginals.conditional_moments(r, st)
    assert mom.loc[("HYG", "calma"), "vol"] == pytest.approx(0.01, rel=0.05)
    assert mom.loc[("HYG", "crisis"), "vol"] == pytest.approx(0.03, rel=0.08)
    assert mom.loc[("HYG", "crisis"), "kurt_exceso"] > 3 > mom.loc[("HYG", "calma"), "kurt_exceso"]


def test_seleccion_normal_en_calma_y_t_en_crisis(muestra) -> None:
    r, st = muestra
    tabla, elegidas = marginals.fit_all(r, st)
    assert tabla.loc[("HYG", "calma"), "elegida"] == "normal"
    assert tabla.loc[("HYG", "crisis"), "elegida"] == "t"
    assert 2.5 < tabla.loc[("HYG", "crisis"), "df_t"] < 6.5  # Verdad: 4.


def test_ppf_cdf_ida_y_vuelta(muestra) -> None:
    r, st = muestra
    m, _, _ = marginals.select_marginal(r["X"][st == 1].to_numpy())
    u = np.array([0.01, 0.5, 0.99])
    assert np.allclose(m.cdf(m.ppf(u)), u)


def test_df_se_acota_cuando_los_datos_son_normales() -> None:
    x = np.random.default_rng(1).normal(0, 0.01, 5000)
    assert marginals.fit_t(x).params[0] <= marginals.DF_BOUNDS[1]


def test_suelo_de_df_en_colas_extremas() -> None:
    x = np.random.default_rng(2).standard_t(2.2, 8000) * 0.01
    assert marginals.fit_t(x).params[0] >= marginals.DF_BOUNDS[0]


def test_respuestas_refugio_y_retornos_de_activos(muestra) -> None:
    r, st = muestra
    mercado = r["X"]
    resp = marginals.refuge_answers(r, st, mercado)
    assert resp["hyg_ratio_vol"] == pytest.approx(3.0, rel=0.1)
    assert resp["gld_corr_sp500_crisis"] > resp["gld_corr_sp500_calma"]  # Correlación sube (verdad 0,7 vs 0,3).


def test_asset_returns_incluye_bonos_y_18_columnas() -> None:
    prov = SyntheticProvider(n_days=300, seed=5)
    cols = [*config.EQUITIES, config.CREDIT_ETF]
    px = prov.get_prices(cols)
    rt = prov.get_rates(["DGS10", "DGS2", config.CREDIT_OAS])
    out = asset_returns(px, rt)
    assert out.shape[1] == 18 and {"UST10Y", "UST2Y"} <= set(out.columns)
