"""Ejecuta todo el pipeline (fases 1-6) sobre data/raw, sin red; guarda figuras, tablas y results.json."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import pandas as pd

from stresslab import config
from stresslab.data.loader import load_market_data
from stresslab.orchestration import pipeline
from stresslab.orchestration.trace import Trace, write_results
from stresslab.reporting.figures import (plot_abanico, plot_correlaciones, plot_escenarios, plot_hist_retornos, plot_qq,
                                          plot_regimes)


def guardar_figuras(datos, reg, mar, cop, sim, esc) -> None:
    """Figuras de cada fase en outputs/ (las usa el informe PDF)."""
    out = config.OUTPUT_DIR
    terminales = {"Base": sim.wealth[:, -1] - 1.0, **{k: v["w"][:, -1] - 1.0 for k, v in esc.outcomes.items()}}
    plot_escenarios(terminales, esc.comparativa).savefig(out / "fig_escenarios.png", dpi=120)
    for k in ("E1", "E2"):
        o = esc.outcomes[k]
        plot_abanico(o["fan"], o["real_w"], f"Escenario {k}: abanico simulado (arranca en crisis) y ventana real").savefig(
            out / f"fig_abanico_{k}.png", dpi=120)
    plot_abanico(esc.outcomes["E3"]["fan"], None, "Escenario E3 (pánico de liquidez): abanico simulado").savefig(
        out / "fig_abanico_E3.png", dpi=120)
    plot_abanico(sim.fan, sim.real_wealth).savefig(out / "fig_abanico.png", dpi=120)
    plot_hist_retornos(sim.real_daily, sim.sim_daily).savefig(out / "fig_hist_retornos.png", dpi=120)
    sp = datos.prices[config.MARKET_TICKER].reindex(reg.states.index)
    plot_regimes(sp, reg.states).savefig(out / "fig_regimenes.png", dpi=130)
    plot_qq(mar.returns, reg.states, mar.marginals, ["HYG", "GLD", "GME", "JPM"]).savefig(out / "fig_qq.png", dpi=110)
    plot_correlaciones(cop.corr[0], cop.corr[1], cop.names, cop.summary["bootstrap"]).savefig(
        out / "fig_correlaciones.png", dpi=110)


def guardar_tablas(reg, mar, cop, sim, esc) -> None:
    """Tablas CSV de cada fase en outputs/."""
    out = config.OUTPUT_DIR
    for nombre, tabla in sim.tables.items():
        tabla.to_csv(out / f"validacion_{nombre}.csv")
    esc.comparativa.to_csv(out / "escenarios_comparativa.csv")
    esc.calibracion.to_csv(out / "escenarios_calibracion.csv", index=False)
    esc.sensibilidad.to_csv(out / "escenarios_sensibilidad.csv", index=False)
    cop.table.to_csv(out / "ajuste_copulas.csv")
    reg.episodes.to_csv(out / "episodios_crisis.csv", index=False)
    mar.moments.to_csv(out / "momentos_condicionales.csv")
    mar.fit_table.to_csv(out / "ajuste_marginales.csv")


def main() -> None:
    """Pipeline completo con traza."""
    pd.set_option("display.width", 220)
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    datos = load_market_data(offline=True)
    trace = Trace()
    reg = pipeline.run_regimes(datos.prices, datos.rates, trace)
    mar = pipeline.run_marginals(datos.prices, datos.rates, reg.states, trace)
    cop = pipeline.run_copulas(mar.returns, reg.states, datos.prices, datos.rates, trace)
    sim = pipeline.run_simulation(mar, cop, reg.states, trace)
    esc = pipeline.run_scenarios(datos.prices, mar, cop, reg.states, sim, trace)
    guardar_figuras(datos, reg, mar, cop, sim, esc)
    guardar_tablas(reg, mar, cop, sim, esc)
    write_results(pipeline.build_results(reg, mar, cop, sim, esc), trace)
    print(trace.to_frame().to_string())
    print(esc.comparativa.round(4).to_string())


if __name__ == "__main__":
    main()
