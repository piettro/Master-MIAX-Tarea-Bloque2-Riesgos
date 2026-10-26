"""Monte Carlo: cadena de Markov de regímenes + cópula del estado + marginales del estado."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from stresslab import config
from stresslab.domain.copulas import CopulaFit
from stresslab.domain.marginals import Marginal

CHUNK = 250_000  # Filas por bloque al muestrear cópulas: limita la memoria (≈ 36 MB por bloque).
U_EPS = 1e-10  # Margen para que ppf nunca reciba exactamente 0 o 1.


@dataclass(frozen=True)
class SimResult:
    """states: (n, T) en {0,1}; returns: (n, T, N) retornos simples diarios; n_clipped: topes aplicados."""

    states: np.ndarray
    returns: np.ndarray
    n_clipped: int


def stationary(p: np.ndarray) -> np.ndarray:
    """Distribución estacionaria de la cadena (autovector izquierdo de autovalor 1)."""
    w, v = np.linalg.eig(p.T)
    pi = np.real(v[:, np.argmax(np.real(w))])
    return pi / pi.sum()


def simulate_states(p: np.ndarray, n: int, horizon: int, s0: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Estados S_1..S_T de n trayectorias a partir de S_0 (vectorizado sobre trayectorias)."""
    u = rng.random((n, horizon))
    s = np.empty((n, horizon), dtype=np.int8)
    prev = s0.astype(int)
    for t in range(horizon):
        s[:, t] = (u[:, t] > p[prev, 0]).astype(np.int8)  # Dos estados: P(ir a 0) = p[prev, 0].
        prev = s[:, t].astype(int)
    return s


def initial_states(p: np.ndarray, n: int, initial: str | int, rng: np.random.Generator) -> np.ndarray:
    """S_0: 'estacionaria' (base, D-08) o un estado fijo (0 calma, 1 crisis)."""
    if initial == "estacionaria":
        return (rng.random(n) > stationary(p)[0]).astype(int)
    return np.full(n, int(initial))


def draw_returns(flat_states: np.ndarray, marginals: dict[tuple[str, int], Marginal],
                 copulas: dict[int, CopulaFit], names: list[str], rng: np.random.Generator,
                 bounds: tuple[np.ndarray, np.ndarray] | None = None) -> tuple[np.ndarray, int]:
    """R_t,i = F⁻¹_{i,S_t}(U_t,i) con U_t de la cópula del estado S_t; devuelve (retornos, nº topados)."""
    out = np.empty((len(flat_states), len(names)))
    for k in (0, 1):
        idx = np.flatnonzero(flat_states == k)
        for i in range(0, len(idx), CHUNK):
            sel = idx[i:i + CHUNK]
            u = np.clip(copulas[k].sample(len(sel), rng), U_EPS, 1.0 - U_EPS)
            for j, name in enumerate(names):
                out[sel, j] = marginals[(name, k)].ppf(u[:, j])
    clipped = 0
    if bounds is not None:
        lo, hi = bounds
        clipped = int(((out < lo) | (out > hi)).sum())
        out = np.clip(out, lo, hi)
    return out, clipped


def return_bounds(hist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Topes por activo: ±SIM_CLIP_MULT × extremo histórico, y nunca por debajo de −99 %."""
    lo = np.maximum(config.SIM_CLIP_MULT * np.nanmin(hist, axis=0), -0.99)
    hi = config.SIM_CLIP_MULT * np.nanmax(hist, axis=0)
    return lo, hi


def simulate(p: np.ndarray, marginals: dict[tuple[str, int], Marginal], copulas: dict[int, CopulaFit],
             names: list[str], n_sims: int, horizon: int, rng: np.random.Generator,
             initial: str | int = "estacionaria", hist: np.ndarray | None = None) -> SimResult:
    """Simula n_sims trayectorias de `horizon` días: cadena de Markov -> cópula -> marginales."""
    s0 = initial_states(p, n_sims, initial, rng)
    states = simulate_states(p, n_sims, horizon, s0, rng)
    bounds = return_bounds(hist) if hist is not None else None
    flat, clipped = draw_returns(states.reshape(-1), marginals, copulas, names, rng, bounds)
    return SimResult(states, flat.reshape(n_sims, horizon, len(names)), clipped)
