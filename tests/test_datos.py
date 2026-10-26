"""Tests de la Fase 1: caché, degradación, retornos de bonos, auditoría y proxy (solo sintéticos)."""
import json
import warnings

import numpy as np
import pandas as pd
import pytest

from stresslab import config
from stresslab.data import audit, csv_cache, returns
from stresslab.data.loader import PRICE_TICKERS, RATE_SERIES, load_market_data
from stresslab.data.synthetic import SyntheticProvider
from stresslab.orchestration.trace import Trace, write_results


class ProveedorRoto:
    """Simula caída de la fuente viva."""

    def get_prices(self, *a, **k):
        raise ConnectionError("sin red")

    get_rates = get_prices


def test_duracion_bono_par_formula_cerrada() -> None:
    d, c = returns.par_bond_duration_convexity(np.array([0.04]), 10.0)
    cerrada = (1 / 0.04) * (1 - 1 / 1.02**20)
    assert d[0] == pytest.approx(cerrada, rel=1e-9) and c[0] > 0


def test_bono_rendimiento_constante_devenga_carry() -> None:
    y = pd.Series(4.0, index=pd.bdate_range("2020-01-01", periods=10))
    assert returns.bond_returns(y, 10.0).iloc[0] == pytest.approx(0.04 / 252)


def test_bono_sube_rendimiento_pierde_precio() -> None:
    y = pd.Series([4.0, 4.5], index=pd.bdate_range("2020-01-01", periods=2))
    r = returns.bond_returns(y, 10.0).iloc[0]
    assert r == pytest.approx(-8.175 * 0.005 + 0.5 * 80 * 0.005**2, abs=0.004)  # D·Δy domina


def test_retorno_de_credito_desde_oas() -> None:
    idx = pd.bdate_range("2020-01-01", periods=3)
    oas, rf = pd.Series([4.0, 4.0, 5.0], index=idx), pd.Series([2.0, 2.0, 2.0], index=idx)
    r = returns.credit_returns(oas, rf)
    assert r.iloc[0] == pytest.approx(0.06 / 252)  # Spread constante: solo carry (OAS + libre de riesgo).
    assert r.iloc[1] == pytest.approx(0.06 / 252 - config.CREDIT_SPREAD_DURATION * 0.01)  # +100 pb de rendimiento.


def test_auditoria_detecta_historia_corta_hueco_y_split() -> None:
    p = SyntheticProvider(n_days=600, seed=2).get_prices(["A", "B"])
    p.iloc[:300, 1] = np.nan  # Historia corta en B.
    p.iloc[400:403, 0] = np.nan  # Hueco interno en A.
    p.iloc[500:, 0] = p.iloc[500:, 0] * 0.4  # Split sin ajustar: salto > 40 %.
    t = audit.audit_prices(p)
    assert t.loc["B", "historia_corta"] and not t.loc["A", "historia_corta"]
    assert t.loc["A", "huecos_internos"] == 3 and t.loc["A", "n_saltos"] >= 1


def test_proxy_recupera_beta_y_rellena_sin_alterar_observado() -> None:
    rng = config.rng(5)
    m = pd.Series(rng.normal(0, 0.01, 800), index=pd.bdate_range("2010-01-01", periods=800))
    y = 0.0001 + 1.5 * m + rng.normal(0, 0.005, 800)
    y.iloc[:200] = np.nan
    modelo = audit.fit_market_proxy(y, m)
    assert modelo.beta == pytest.approx(1.5, abs=0.1)
    lleno = audit.fill_with_proxy(y, m, modelo, config.rng(1))
    assert lleno.notna().all() and lleno.iloc[200:].equals(y.iloc[200:])
    otra = audit.fill_with_proxy(y, m, modelo, config.rng(1))
    assert lleno.equals(otra)  # Misma semilla, mismo relleno.


def test_cache_ida_y_vuelta_y_modo_offline(tmp_path) -> None:
    prov = SyntheticProvider(n_days=100, seed=1)
    csv_cache.update_csv(csv_cache.PRICES_FILE, prov.get_prices(list(PRICE_TICKERS)), tmp_path)
    csv_cache.update_csv(csv_cache.RATES_FILE, prov.get_rates(list(RATE_SERIES)), tmp_path)
    datos = load_market_data(offline=True, raw_dir=tmp_path)
    assert list(datos.prices.columns) == list(PRICE_TICKERS) and not datos.warnings


def test_degradacion_usa_cache_y_avisa(tmp_path) -> None:
    prov = SyntheticProvider(n_days=100, seed=1)
    csv_cache.update_csv(csv_cache.PRICES_FILE, prov.get_prices(list(PRICE_TICKERS)), tmp_path)
    csv_cache.update_csv(csv_cache.RATES_FILE, prov.get_rates(list(RATE_SERIES)), tmp_path)
    roto = ProveedorRoto()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        datos = load_market_data(offline=False, raw_dir=tmp_path, prices=roto, rates=roto)
    assert len(datos.warnings) == 2 and len(w) == 2 and len(datos.prices) == 100


def test_limpieza_quita_festivos_solo_vix_y_barra_de_hoy() -> None:
    from stresslab.data.loader import clean_prices

    idx = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06", pd.Timestamp.today().normalize()])
    p = pd.DataFrame({config.MARKET_TICKER: [1.0, np.nan, 2.0, 3.0], config.VIX_TICKER: [10.0, 11.0, 12.0, 13.0]}, index=idx)
    assert list(clean_prices(p).index) == [idx[0], idx[2]]


def test_fuente_viva_actualiza_cache(tmp_path) -> None:
    prov = SyntheticProvider(n_days=80, seed=4)
    load_market_data(offline=False, raw_dir=tmp_path, prices=prov, rates=prov)
    assert (tmp_path / csv_cache.PRICES_FILE).exists()


def test_traza_y_results_json(tmp_path) -> None:
    tr = Trace()
    with tr.step("datos", "csv", n_obs=10) as s:
        s.nota = "ok"
    ruta = write_results({"x": 1}, tr, tmp_path / "results.json")
    cargado = json.loads(ruta.read_text(encoding="utf-8"))
    assert cargado["trace"][0]["paso"] == "datos" and cargado["seed"] == config.SEED
    assert list(tr.to_frame().columns) == ["paso", "metodo", "n_obs", "seed", "segundos", "nota"]
