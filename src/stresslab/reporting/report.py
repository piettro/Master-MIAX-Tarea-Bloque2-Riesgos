"""Informe ejecutivo PDF: todas las cifras salen de outputs/results.json (R-INF-06)."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

TEMPLATES = Path(__file__).parent / "templates"
NOMBRES = {"Base": "Crisis base", "E1": "E1 · Estanflación 2022", "E2": "E2 · Crisis de crédito 2008",
           "E3": "E3 · Pánico de liquidez"}
FILAS = (("vol_anualizada", "Volatilidad anual"), ("var99_diario", "VaR diario"), ("es99_diario", "ES diario"),
         ("var99_1m", "VaR mensual"), ("es99_1m", "ES mensual"), ("var99_6m", "VaR a seis meses"),
         ("es99_6m", "ES a seis meses"), ("mdd_medio_6m", "Caída máxima media (6 meses)"),
         ("prob_perdida_20pct_6m", "Prob. de perder más del 20 % (6 meses)"))


def num(x: float, d: int = 1) -> str:
    """Número con coma decimal."""
    return f"{x:.{d}f}".replace(".", ",")


def signed(x: float, d: int = 2) -> str:
    """Número con signo explícito y coma decimal."""
    return f"{x:+.{d}f}".replace(".", ",")


def pct(x: float, d: int = 1) -> str:
    """Porcentaje con coma decimal."""
    return f"{num(100 * x, d)} %"


def load_results(path: Path) -> dict[str, Any]:
    """Lee results.json."""
    return json.loads(path.read_text(encoding="utf-8"))


def _tabla_escenarios(comp: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    """Filas (etiqueta, valores por escenario) con formato porcentual."""
    return [{"etiqueta": etq, "valores": [pct(comp[c][k]) for c in NOMBRES]} for k, etq in FILAS]


def _episodios(r: dict[str, Any], n: int = 5) -> list[dict[str, str]]:
    """Los n episodios de crisis con mayor caída del S&P 500."""
    orden = sorted(r["episodios"], key=lambda e: float(e["caida_sp500"]))[:n]
    return [{"periodo": f"{e['inicio']} a {e['fin']}", "dias": e["dias"], "caida": pct(-float(e["caida_sp500"]), 0),
             "vix": num(float(e["pico_vix"]), 0), "narrativa": e["narrativa"]} for e in orden]


def build_context(r: dict[str, Any]) -> dict[str, Any]:
    """Valores formateados que consume la plantilla; no hay cifras escritas a mano en ella."""
    reg, cop, esc = r["regimenes"], r["copulas"], r["escenarios"]
    comp, par, perf = esc["comparativa"], r["parametros"], r["copulas"]["perfil_estados"]
    riesgo = {f["index"]: {"real": float(f["real"]), "sim": float(f["simulado"])}
              for f in r["simulacion_base"]["tablas"]["riesgo_sin_GME"]}
    val = esc["validacion_ventana_real"]
    sens = esc["sensibilidad"]
    rango = max(abs(s["es99_6m"] - comp[s["escenario"]]["es99_6m"]) for s in sens)
    calib = {c["escenario"]: c for c in esc["calibracion"]}
    cb = esc["corr_bonos_acciones_base"]
    ref = r["refugio"]
    e = {k: {m: comp[k][m] for m, _ in FILAS} for k in comp}
    return {
        "fecha": date.today().strftime("%d/%m/%Y"), "seed": par["seed"], "nivel": int(round(100 * par["nivel_var"])),
        "n_activos": par["n_activos"], "inicio": par["primer_dia_modelo"], "fin": par["ultimo_dia_modelo"],
        "n_sims": f"{par['n_sims']:,}".replace(",", "."), "horizonte": par["horizonte_dias"],
        "umbral": num(par["umbral_crisis"], 1), "min_dias": par["min_episodio_dias"],
        "e3_vol": num(par["e3_vol_mult"], 1), "cola_q": num(par["cola_q_pct"], 0),
        "ic_pct": par["ic_pct"], "banda_inf": par["banda_pct"][0], "banda_sup": par["banda_pct"][1],
        "nombres": NOMBRES,
        "pct_crisis": pct(reg["pct_crisis"]), "dur_crisis": num(reg["duracion_esperada"]["1"], 0),
        "dur_calma": num(reg["duracion_esperada"]["0"], 0), "n_cambios": reg["n_cambios"],
        "ratio_vol": num(perf["mediana_ratio_vol"]), "corr_calma": num(perf["corr_media_calma"], 2),
        "corr_crisis": num(perf["corr_media_crisis"], 2), "pct_pares": pct(perf["pct_pares_corr_sube"], 0),
        "corr_ic_inf": num(cop["bootstrap"]["1"]["ic_inf"], 2), "corr_ic_sup": num(cop["bootstrap"]["1"]["ic_sup"], 2),
        "lambda_calma": num(perf["lambda_L_calma"], 2), "lambda_crisis": num(perf["lambda_L_crisis"], 2),
        "lambda_e3": num(esc["perfil_crisis"]["E3"]["lambda_L_5pct"], 2),
        "vix_calma": num(perf["vix_medio_calma"], 0), "vix_crisis": num(perf["vix_medio_crisis"], 0),
        "oas_calma": num(perf["oas_medio_calma"], 1), "oas_crisis": num(perf["oas_medio_crisis"], 1),
        "dist_cop_calma": num(cop["ajuste"]["calma|t"]["df"], 0), "dist_cop_crisis": num(cop["ajuste"]["crisis|t"]["df"], 0),
        "hyg_ratio": num(ref["hyg_ratio_vol"]), "gld_corr": num(ref["gld_corr_sp500_crisis"], 2),
        "gld_pos": pct(ref["gld_dias_positivos_crisis"], 0), "gld_ratio": num(ref["gld_ratio_vol"]),
        "tabla": _tabla_escenarios(comp), "columnas": list(NOMBRES.values()), "episodios": _episodios(r),
        "e": {k: {m: pct(v) for m, v in e[k].items()} for k in e},
        "gap_es_var_e2": num(100 * (e["E2"]["es99_6m"] - e["E2"]["var99_6m"]), 1),
        "corr_ba": {"e1": signed(calib["E1"]["corr_bonos_acciones"]), "base": signed(cb["crisis"]),
                    "e2": signed(calib["E2"]["corr_bonos_acciones"])},
        "n_e1": calib["E1"]["n_obs"], "n_e2": calib["E2"]["n_obs"],
        "perc_e1": pct(val["E1"]["percentil_retorno_real"], 0), "perc_e2": pct(val["E2"]["percentil_retorno_real"], 0),
        "perdida_real_e2": pct(-val["E2"]["retorno_real"], 0),
        "dur_sim": num(reg["duracion_esperada"]["1"], 0),
        "cobertura": pct(r["simulacion_base"]["cobertura"]["cobertura_p5_p95_real"], 0),
        "vol_real_sin": pct(riesgo["vol_anualizada"]["real"]), "vol_sim_sin": pct(riesgo["vol_anualizada"]["sim"]),
        "var6_real": pct(riesgo["var99_6m"]["real"], 0), "var6_sim": pct(riesgo["var99_6m"]["sim"], 0),
        "beta_proxy": num(esc["modelo_proxy"]["beta"], 1), "rango_sens": num(100 * rango, 1),
        "n_crisis_cop": cop["n_crisis"], "n_cal": cop["n_ventana"],
    }


def render_html(r: dict[str, Any]) -> str:
    """HTML del informe a partir de results.json (StrictUndefined: falla si falta una cifra)."""
    env = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=True, undefined=StrictUndefined)
    return env.get_template("informe.html.j2").render(**build_context(r))


def build_pdf(results_path: Path, fig_dir: Path, out_pdf: Path) -> int:
    """Genera el PDF con WeasyPrint y devuelve el número de páginas."""
    from weasyprint import HTML

    html = render_html(load_results(results_path))
    doc = HTML(string=html, base_url=str(fig_dir) + "/").render()
    doc.write_pdf(str(out_pdf))
    return len(doc.pages)
