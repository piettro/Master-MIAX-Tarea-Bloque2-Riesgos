"""Cartera equiponderada buy & hold (sin rebalanceo): riqueza, retornos diarios y ventanas reales."""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def wealth_paths(returns: np.ndarray, chunk: int = 1000) -> np.ndarray:
    """W_t = (1/N)·Σ_i Π_{s≤t}(1+r_{i,s}), con W_0 = 1. Entrada (n, T, N); salida (n, T+1)."""
    n, t, _ = returns.shape
    out = np.ones((n, t + 1))
    for i in range(0, n, chunk):
        out[i:i + chunk, 1:] = np.cumprod(1.0 + returns[i:i + chunk], axis=1).mean(axis=2)
    return out


def daily_returns(wealth: np.ndarray) -> np.ndarray:
    """Retorno diario de la cartera: W_t / W_{t-1} − 1; salida (n, T)."""
    return wealth[:, 1:] / wealth[:, :-1] - 1.0


def rolling_windows(x: np.ndarray, horizon: int) -> np.ndarray:
    """Ventanas móviles de `horizon` días: (T, N) -> (n_ventanas, horizon, N); (T,) -> (n_ventanas, horizon)."""
    v = sliding_window_view(x, horizon, axis=0)
    return np.moveaxis(v, -1, 1) if x.ndim == 2 else v
