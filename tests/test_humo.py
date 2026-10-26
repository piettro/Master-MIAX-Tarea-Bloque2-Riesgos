"""Tests de humo de la Fase 0: config, Protocols y proveedor sintético."""
import numpy as np

from stresslab import config
from stresslab.data.base import PriceProvider, RateProvider
from stresslab.data.synthetic import SyntheticParams, SyntheticProvider, generate


def test_config_coherente() -> None:
    assert len(config.EQUITIES) == 15
    assert config.N_SIMS >= 10_000 and config.HORIZON_DAYS == 126


def test_proveedor_cumple_protocols() -> None:
    prov: PriceProvider = SyntheticProvider(n_days=50, seed=1)
    rat: RateProvider = SyntheticProvider(n_days=50, seed=1)
    assert prov.get_prices(["A", "B"]).shape == (50, 2)
    assert list(rat.get_rates(["DGS10", "DGS2"]).columns) == ["DGS10", "DGS2"]


def test_mismo_seed_mismos_resultados() -> None:
    a = generate(300, 4, seed=7).returns
    b = generate(300, 4, seed=7).returns
    c = generate(300, 4, seed=8).returns
    assert a.equals(b) and not a.equals(c)


def test_sintetico_recupera_parametros_verdaderos() -> None:
    s = generate(20_000, 5, seed=3)
    for st, vol in enumerate(SyntheticParams().daily_vol):
        assert abs(s.returns.values[s.states == st].std() - vol) / vol < 0.05
    corr = np.corrcoef(s.returns.values[s.states == 1].T)
    assert abs(corr[np.triu_indices(5, 1)].mean() - 0.7) < 0.05


def test_modo_offline_por_variable(monkeypatch) -> None:
    monkeypatch.setenv("OFFLINE", "true")
    assert config.is_offline()
