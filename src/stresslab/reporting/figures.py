"""Figuras (títulos en español)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure


def plot_qq(returns: pd.DataFrame, states: pd.Series, marg: dict, assets: list[str]) -> Figure:
    """QQ de la marginal elegida frente a los datos: fila = estado, columna = activo."""
    from stresslab.domain.marginals import STATE_NAMES, qq_points

    fig, axes = plt.subplots(2, len(assets), figsize=(3.4 * len(assets), 6.4), squeeze=False)
    for j, a in enumerate(assets):
        for k in (0, 1):
            ax = axes[k, j]
            m = marg.get((a, k))
            if m is None:
                ax.axis("off")
                continue
            x = returns[a][states.reindex(returns.index) == k].dropna().to_numpy()
            teo, emp = qq_points(x, m)
            ax.scatter(teo, emp, s=4)
            lim = [min(teo.min(), emp.min()), max(teo.max(), emp.max())]
            ax.plot(lim, lim, color="red", lw=0.8)
            ax.set_title(f"{a} · {STATE_NAMES[k]} · {m.family}")
            ax.set_xlabel("Cuantil teórico")
            ax.set_ylabel("Cuantil empírico")
    fig.suptitle("QQ de las marginales ajustadas (rojo = ajuste perfecto)")
    fig.tight_layout()
    return fig


def plot_correlaciones(c0, c1, names: list[str], boot: dict) -> Figure:
    """Mapas de calor de Pearson en calma y crisis y correlación media con su IC 95 %."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), gridspec_kw={"width_ratios": [1, 1, 0.55]})
    for ax, c, t in ((axes[0], c0, "Calma"), (axes[1], c1, "Crisis")):
        im = ax.imshow(c, vmin=-0.2, vmax=1, cmap="viridis")
        ax.set_xticks(range(len(names)), names, rotation=90, fontsize=7)
        ax.set_yticks(range(len(names)), names, fontsize=7)
        ax.set_title(f"Correlación de Pearson · {t}")
    fig.colorbar(im, ax=axes[:2], shrink=0.8)
    ax = axes[2]
    for i, k in enumerate((0, 1)):
        b = boot[k]
        ax.errorbar(i, b["media"], yerr=[[b["media"] - b["ic_inf"]], [b["ic_sup"] - b["media"]]], fmt="o", capsize=6)
    ax.axhline(1.0, color="red", ls="--", lw=0.8)
    ax.set_xticks([0, 1], ["Calma", "Crisis"])
    ax.set_ylim(0, 1.05)
    ax.set_title("Correlación media (IC 95 %); línea roja = 1")
    return fig


def plot_abanico(fan, real_wealth, titulo: str = "Riqueza de la cartera a 6 meses: abanico simulado y ventana real") -> Figure:
    """Abanico p5-p25-p50-p75-p95 de la riqueza simulada con trayectorias reales de 126 días encima."""
    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(fan.shape[0])
    ax.fill_between(x, fan[:, 0], fan[:, 4], color="tab:blue", alpha=0.2, label="p5-p95 simulado")
    ax.fill_between(x, fan[:, 1], fan[:, 3], color="tab:blue", alpha=0.35, label="p25-p75 simulado")
    ax.plot(x, fan[:, 2], color="tab:blue", label="mediana simulada")
    if real_wealth is not None:
        ax.plot(x, real_wealth[-1], color="black", lw=1.8, label="real: últimos 126 días" if len(real_wealth) > 1 else "real: ventana histórica")
        if len(real_wealth) > 1:
            ax.plot(x, real_wealth[0], color="gray", lw=1, label="real: primera ventana")
    ax.set_title(titulo)
    ax.set_xlabel("Días hábiles")
    ax.set_ylabel("Riqueza (inicio = 1)")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_hist_retornos(real_daily, sim_daily) -> Figure:
    """Histograma (escala log) de retornos diarios de la cartera: real frente a simulado."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    lim = float(max(abs(real_daily).max(), 0.1))
    bins = [-lim + 2 * lim * i / 120 for i in range(121)]
    ax.hist(sim_daily.ravel(), bins=bins, density=True, alpha=0.5, label="simulado")
    ax.hist(real_daily.ravel(), bins=bins, density=True, alpha=0.5, label="real")
    ax.set_yscale("log")
    ax.set_title("Retorno diario de la cartera: real vs simulado")
    ax.set_xlabel("Retorno diario")
    ax.set_ylabel("Densidad (escala log)")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_escenarios(terminales: dict, comparativa: pd.DataFrame) -> Figure:
    """Distribución del retorno a 6 meses por escenario y barras de VaR/ES 99 % a 6 meses."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.8))
    for nombre, t in terminales.items():
        a.hist(t, bins=100, histtype="step", density=True, label=nombre, lw=1.5)
    a.set_title("Retorno de la cartera a 6 meses por escenario")
    a.set_xlabel("Retorno a 126 días")
    a.set_ylabel("Densidad")
    a.legend()
    x = range(len(comparativa.columns))
    b.bar([i - 0.2 for i in x], comparativa.loc["var99_6m"], width=0.4, label="VaR 99 %")
    b.bar([i + 0.2 for i in x], comparativa.loc["es99_6m"], width=0.4, label="ES 99 %")
    b.set_xticks(list(x), list(comparativa.columns))
    b.set_title("Pérdida a 6 meses (positiva) por escenario")
    b.set_ylabel("Pérdida sobre el capital")
    b.legend()
    fig.tight_layout()
    return fig


def plot_regimes(sp500: pd.Series, states: pd.Series, titulo: str = "S&P 500 por régimen (azul = crisis)") -> Figure:
    """S&P 500 en escala log con las crisis sombreadas en azul y la calma en blanco."""
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(sp500.index, sp500.values, color="black", lw=0.8)
    ax.set_yscale("log")
    s = states.reindex(sp500.index).fillna(0).to_numpy()
    ini = None
    for t, v in enumerate(s):
        if v == 1 and ini is None:
            ini = t
        if (v == 0 or t == len(s) - 1) and ini is not None:
            ax.axvspan(sp500.index[ini], sp500.index[t], color="tab:blue", alpha=0.35, lw=0)
            ini = None
    ax.set_title(titulo)
    ax.set_ylabel("Nivel del S&P 500 (escala log)")
    ax.set_xlabel("Fecha")
    fig.tight_layout()
    return fig
