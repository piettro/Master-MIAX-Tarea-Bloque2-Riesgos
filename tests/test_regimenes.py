"""Tests de la Fase 2 con datos sintéticos de verdad conocida."""
import numpy as np
import pandas as pd
import pytest

from stresslab.data.synthetic import generate
from stresslab.domain import regimes
from stresslab.domain.features import standardize


def _muestra(n: int = 3000):
    s = generate(n, 4, seed=11)
    return standardize(s.returns), s.states


def test_hmm_recupera_estados_conocidos() -> None:
    z, verdad = _muestra()
    fit = regimes.fit_hmm(z.to_numpy(), n_init=3, seed=1)
    pred = (regimes.smoothed_proba(fit, z.to_numpy())[:, 1] > 0.5).astype(int)
    assert (pred == verdad).mean() > 0.9


def test_estado_0_es_calma_y_1_crisis() -> None:
    z, _ = _muestra()
    fit = regimes.fit_hmm(z.to_numpy(), n_init=3, seed=2)
    assert np.trace(fit.covars[0]) < np.trace(fit.covars[1])
    assert fit.transmat[0, 0] > 0.9 and np.allclose(fit.transmat.sum(axis=1), 1)


def test_filtrado_igual_a_suavizado_sobre_prefijo() -> None:
    z, _ = _muestra(800)
    fit = regimes.fit_hmm(z.to_numpy(), n_init=2, seed=3)
    filt = regimes.filtered_proba(fit, z.to_numpy())
    t = 400
    prefijo = regimes.smoothed_proba(fit, z.to_numpy()[: t + 1])[-1]
    assert np.allclose(filt[t], prefijo, atol=1e-6)


def test_duracion_minima_elimina_episodios_cortos() -> None:
    rng = np.random.default_rng(0)
    crudo = (rng.random(2000) < 0.3).astype(int)  # Ruido: muchísimos tramos de 1-3 días.
    for m in (10, 15, 20):
        runs = regimes._runs(regimes.enforce_min_duration(crudo, m))
        assert min(f - i for i, f, _ in runs) >= m


def test_estadisticas_de_regimen() -> None:
    s = np.array([0] * 90 + [1] * 10 + [0] * 90 + [1] * 10)
    e = regimes.regime_stats(s)
    assert e["pct_crisis"] == 0.1 and e["n_cambios"] == 3
    assert e["duracion_media"][1] == 10.0


def test_tabla_de_episodios() -> None:
    idx = pd.bdate_range("2020-01-01", periods=120)
    st = pd.Series(0, index=idx)
    st.iloc[40:70] = 1
    sp = pd.Series(100.0, index=idx)
    sp.iloc[40:70] = 80.0
    vix = pd.Series(15.0, index=idx)
    vix.iloc[50] = 60.0
    t = regimes.episodes_table(st, sp, vix)
    assert len(t) == 1 and t.loc[0, "dias"] == 30
    assert t.loc[0, "caida_sp500"] == pytest.approx(-0.2) and t.loc[0, "pico_vix"] == 60.0
