"""Celdas narrativas del notebook (markdown y código de uso). El código del motor lo incrusta build_notebook.py."""
from __future__ import annotations

PORTADA = r'''# Práctica B2-2 · Escenarios de estrés y cambios de régimen de mercado
**Módulo de Gestión de Riesgos · Máster MIAX**

Este notebook construye, de principio a fin, un motor de *stress testing* con cambios de régimen:

1. **Datos** diarios de EE. UU. desde 2006 (15 acciones/ETF, HYG y los bonos del Tesoro a 10 y 2 años) con auditoría de calidad.
2. **Régimen de mercado** (calma/crisis) con un *Hidden Markov Model* gaussiano de 2 estados.
3. **Marginales condicionales** por activo y estado (Normal frente a Student-t).
4. **Cópulas** por estado (Gauss frente a t) y contraste de «las correlaciones tienden a 1».
5. **Simulador Monte Carlo** (10.000 trayectorias × 126 días) y su validación frente a los datos reales.
6. **Tres escenarios de estrés** (estanflación 2022, crisis de crédito 2008 y pánico de liquidez crediticia) con VaR y ES al 99 %.

**Cómo leerlo.** El código del motor está incrustado al principio (una celda por módulo, con comentarios) para que este
único fichero sea autocontenido; después, cada sección ejecuta una fase y explica **qué se ve y por qué importa**.
El notebook no necesita red: incluye una instantánea de los datos descargados (ver sección de datos).

**Reglas de diseño.** Interpretabilidad ante todo: no hay Deep Learning ni cajas negras (solo HMM, cópulas, Monte Carlo y estadística descriptiva);
una única semilla propaga todo el azar (mismo *seed*, mismos resultados); cada paso deja una traza al final.'''

CONFIG_MD = r'''## 0. Motor de cálculo (código de `src/stresslab` incrustado)
Cada celda de código de esta sección es un módulo del paquete, en orden de dependencia. Los módulos se envuelven en una función
para conservar sus espacios de nombres (`config.SEED`, `regimes.fit_hmm`, …) sin imports locales. **Se pueden saltar en una primera lectura**:
las secciones 1-7 usan estas funciones y comentan los resultados.

* **Capa de datos**: contratos (`Protocol`), caché CSV, proveedores de Yahoo Finance y FRED (solo se usan si se pide actualizar), auditoría y retornos.
* **Capa de dominio**: factores, regímenes, marginales, cópulas, simulador, cartera, riesgo, escenarios y validación.
* **Orquestación y figuras**: traza por paso y pipeline.'''

SETUP = r'''from __future__ import annotations

import base64, gzip, json, os, sys, time, types, warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator, Protocol, Sequence
from contextlib import contextmanager

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
plt.rcParams["figure.dpi"] = 100

pct = lambda x, d=1: f"{100 * x:.{d}f} %".replace(".", ",")  # Porcentaje con coma decimal.
'''

DATOS_MD = r'''## 1. Datos: fuentes, filtrado y auditoría de calidad
**Universo.** AAPL, AMZN, BAC, BRK-B, CVX, ENPH, GLD, GME, GOOGL, JNJ, JPM, MSFT, NVDA, PG, XOM, el crédito high yield **HYG (BAMLH0A0HYM2)** y los bonos del Tesoro a **10 y 2 años**
(18 activos). Precios diarios ajustados por *splits* y dividendos de Yahoo Finance para las acciones; rendimientos diarios de FRED (DGS10, DGS2) y el diferencial de crédito
OAS de ICE BofA US High Yield (BAMLH0A0HYM2). Según la aclaración del profesor (decisión D-02), **«HYG (BAMLH0A0HYM2)» es la serie OAS de FRED**, así que el activo de crédito se construye
desde ella (ver más abajo) y la misma serie sirve además de **factor de estado** del HMM. El ETF HYG de Yahoo se descarga solo como contraste de esa construcción y no entra en la cartera.

**Reproducibilidad.** Por defecto el notebook lee la caché `data/raw/` y, si no existe, la reconstruye desde la **instantánea incrustada** en la celda siguiente (así se ejecuta
sin red y da siempre los mismos números). Para refrescar los datos desde internet hay que lanzar el notebook con la variable de entorno `ACTUALIZAR_DATOS=true`;
si una fuente falla, el flujo continúa con la caché y avisa.

**Bonos.** FRED da *rendimientos*, no precios. Se aproxima el retorno diario de un bono a la par con cupón semestral igual al rendimiento previo:
`r_t ≈ y_{t-1}/252 − D_mod · Δy_t + ½ · C · Δy_t²`, con duración modificada `D_mod` y convexidad `C` calculadas analíticamente. Capta el *carry*, el riesgo de tipos y la convexidad; ignora el *roll-down* y los diferenciales.

**Crédito high yield.** El OAS es un diferencial, no un precio. Se construye el retorno diario con su rendimiento `y = OAS + tipo libre de riesgo` (media de DGS10 y DGS2 como proxy a 5 años):
`r_t ≈ y_{t-1}/252 − D · Δy_t` con **D = 4 años** (D-39), que coincide con la duración típica del índice y con la estimada por MCO contra el ETF HYG (≈ 3,9). Incluye carry y sensibilidad a spread **y** a tipos
(en 2022 dio −11,5 % frente a −10,8 % del ETF); ignora convexidad e impagos. Contraste con el ETF: la correlación diaria es moderada (≈ 0,6) porque el ETF cotiza con descuentos/primas intradía respecto al índice y el OAS es un cierre diario.'''

DATOS_CODE = r'''# Instantánea de los datos (CSV comprimidos): permite ejecutar el notebook sin red.
SNAPSHOT = __SNAPSHOT__


def restaurar_cache_si_falta() -> str:
    """Escribe la instantánea en data/raw si la caché no existe y dice de dónde vienen los datos."""
    raw = config.RAW_DIR
    if (raw / "prices.csv").exists() and (raw / "rates.csv").exists():
        return f"caché local en {raw}"
    raw.mkdir(parents=True, exist_ok=True)
    for nombre, blob in SNAPSHOT.items():
        (raw / nombre).write_bytes(gzip.decompress(base64.b64decode(blob)))
    return f"instantánea incrustada escrita en {raw}"


ACTUALIZAR = os.environ.get("ACTUALIZAR_DATOS", "false").strip().lower() == "true"
origen = restaurar_cache_si_falta()
datos = load_market_data(offline=not ACTUALIZAR)
print("Origen de los datos:", origen, "| actualizar desde la red:", ACTUALIZAR)
for aviso in datos.warnings:
    print("AVISO:", aviso)
print(f"Precios: {datos.prices.shape[0]} días × {datos.prices.shape[1]} series, "
      f"{datos.prices.index[0].date()} → {datos.prices.index[-1].date()}")
print(f"Tipos:   {datos.rates.shape[0]} días × {datos.rates.shape[1]} series, "
      f"{datos.rates.index[0].date()} → {datos.rates.index[-1].date()}")
trace = Trace()  # Traza del pipeline: se muestra al final.
'''

AUDIT_CODE = r'''auditoria = audit.audit_prices(datos.prices)
display(auditoria.drop(columns=["n_precios_no_positivos"]))'''

AUDIT_MD = r'''**Qué mira la auditoría** (y por qué): *historias cortas* (ENPH cotiza desde 2012, así que para 2008 faltan datos y hará falta una proxy; el ETF HYG desde 2007, pero solo es contraste),
*huecos* internos del calendario, *saltos* diarios superiores al 40 % (posibles *splits* mal ajustados) y *outliers* a más de 8 sigmas robustas (MAD).
La auditoría **marca pero no elimina**: GME en enero de 2021 es un fenómeno real (el *short squeeze*), y borrarlo escondería precisamente el tipo de riesgo que estamos midiendo.
No hay huecos internos ni precios no positivos; el único activo con saltos por encima del 40 % aparte de GME y ENPH (2020) es el índice VIX, algo normal en un índice de volatilidad.'''

BONOS_CODE = r'''bonos = returns.bond_return_table(datos.rates)
display(bonos.describe().T[["count", "mean", "std", "min", "max"]])
anio = bonos.loc["2022"].sum()
display(Markdown(f"Retorno acumulado de los bonos en 2022 según el modelo de duración: **10 años {pct(anio['UST10Y'])}** y **2 años {pct(anio['UST2Y'])}** "
                 "(coherente con lo que sufrieron los ETF de Tesoro ese año: la renta fija también cayó)."))'''

REG_MD = r'''## 2. Regímenes de mercado: HMM gaussiano de 2 estados
**Idea.** El mercado alterna entre un régimen de *calma* (volatilidad baja, correlaciones moderadas) y otro de *crisis*. No observamos el régimen: lo **inferimos** de los datos
con un *Hidden Markov Model* (estado oculto que solo depende del estado anterior; parámetros por máxima verosimilitud con el algoritmo EM).

**Variables de estado** (decisión D-04): retorno del S&P 500, Δlog del VIX, Δ de la pendiente 10a−2a y Δ del diferencial de crédito OAS. Son **variaciones, no niveles**
(comparables a retornos y estacionarias) y cubren las tres dimensiones de una crisis: caída de la renta variable, miedo implícito y estrés de crédito, más la señal macro de la curva.
Se estandarizan antes de ajustar.

**Etiquetas.** hmmlearn numera los estados arbitrariamente; se reordenan para que el **estado 0 sea el de menor varianza (calma)** y el 1 la crisis.
**Decisión dura.** Se usa la probabilidad *suavizada* de crisis con umbral 0,5 y una **duración mínima de 15 días** aplicada a ambos estados (D-06): un régimen económico dura semanas o meses; una «crisis de 3 días» es ruido de volatilidad.

**Smoothing frente a filtering.** La probabilidad suavizada usa **toda la serie** (incluido el futuro de cada día): es lo correcto para *describir* la historia, pero tiene sesgo de *look-ahead*
y no sería una señal operable en tiempo real. La probabilidad *filtrada* (solo información hasta *t*) se calcula también para medir cuánto cambia la foto.'''

REG_CODE = r'''reg = run_regimes(datos.prices, datos.rates, trace)
s = reg.summary
resumen = pd.DataFrame({
    "% de días": [1 - s["pct_crisis"], s["pct_crisis"]],
    "duración media observada (días)": [s["duracion_media"][0], s["duracion_media"][1]],
    "duración esperada 1/(1−pᵢᵢ) (días)": [s["duracion_esperada"][0], s["duracion_esperada"][1]],
}, index=["calma", "crisis"])
display(resumen)
display(pd.DataFrame(s["transmat"], index=["desde calma", "desde crisis"], columns=["a calma", "a crisis"]))
display(Markdown(
    f"**Lectura.** El {pct(s['pct_crisis'])} de los días está en crisis, con {s['n_cambios']} cambios de régimen en {len(reg.states)} días. "
    f"La calma es muy persistente (p₀₀ = {s['transmat'][0][0]:.3f}, dura ≈ {s['duracion_esperada'][0]:.0f} días) y la crisis también, aunque menos "
    f"(p₁₁ = {s['transmat'][1][1]:.3f}, ≈ {s['duracion_esperada'][1]:.0f} días): las crisis son bastante más cortas que los periodos de calma. "
    f"La probabilidad filtrada y la suavizada coinciden en el {pct(s['acuerdo_filtrado_suavizado'])} de los días, así que el sesgo de *look-ahead* afecta sobre todo a *cuándo* empieza y acaba cada episodio. "
    f"Un HMM solo sobre el S&P 500 coincide con el multifactor en el {pct(s['acuerdo_hmm_solo_sp500'])} de los días: los factores de volatilidad y crédito cambian una de cada diez decisiones. "
    "(La matriz de transición de los estados duros es la que usará el simulador, por coherencia con los episodios que reportamos; la del HMM, p₁₁ menor, generaría crisis demasiado cortas.)"))'''

REG_FIG_CODE = r'''sp = datos.prices[config.MARKET_TICKER].reindex(reg.states.index)
figures.plot_regimes(sp, reg.states)
plt.show()'''

REG_FIG_MD = r'''**El gráfico.** S&P 500 en escala logarítmica; **azul = crisis, blanco = calma**. Las bandas azules coinciden con los episodios que reconocemos: la crisis financiera de 2008-2009 (la banda más larga), la deuda europea de 2010-2011,
el COVID de 2020, la estanflación de 2022 y los aranceles de 2025. Se ve que las crisis son **raras pero concentradas**, y que la banda de 2022 aparece partida en varios episodios (el mercado «respiró» entre subidas de tipos).'''

EPI_CODE = r'''epi = reg.episodes.copy()
epi["caida_sp500"] = epi["caida_sp500"].map(pct)
epi["pico_vix"] = epi["pico_vix"].round(1)
display(epi)
peor = reg.episodes.loc[reg.episodes["caida_sp500"].idxmin()]
largo = reg.episodes.loc[reg.episodes["dias"].idxmax()]
display(Markdown(f"**Por qué cada episodio es crisis.** La tabla combina tres evidencias independientes del HMM: duración, caída máxima del S&P 500 y pico del VIX, y una narrativa económica "
                 f"(etiqueta histórica por solapamiento de fechas; lo que no solapa queda «sin etiqueta»). El más profundo es el que empieza el {peor['inicio']} (caída del {pct(-peor['caida_sp500'])}, VIX {peor['pico_vix']:.0f}); "
                 f"el más largo dura {largo['dias']} días (desde {largo['inicio']})."))
display(reg.sensitivity.rename(columns={"pct_crisis": "% días en crisis"}))
display(Markdown("**Sensibilidad a la duración mínima (10/15/20 días).** Cambia el número de episodios (los cortos se absorben o se funden) pero casi no cambia el % de días en crisis: la conclusión sobre *cuánto tiempo* pasa el mercado en crisis es robusta."))'''

MAR_MD = r'''## 3. Distribuciones marginales condicionales
Para cada activo y estado se calculan **media, volatilidad, asimetría y curtosis**, y se ajustan dos candidatas: **Normal** y **Student-t** (colas gruesas), elegidas por **BIC** (que penaliza el parámetro extra de la t;
también se guarda el AIC). Se limitan los grados de libertad a ≥ 3 (D-26): por debajo de 3 el ajuste está dominado por muchos retornos cercanos a cero y exagera las colas, lo que en 10.000×126 días produciría caídas diarias absurdas.'''

MAR_CODE = r'''mar = run_marginals(datos.prices, datos.rates, reg.states, trace)
vistos = ["AAPL", "BAC", "GLD", "GME", "HYG", "JPM", "UST10Y", "UST2Y", "XOM"]
display(mar.moments.loc[vistos].rename(columns={"kurt_exceso": "curtosis (exceso)", "vol_anual": "vol. anual"}))
display(Markdown("**Cómo leer la tabla.** En crisis la volatilidad sube en casi todos los activos (los bancos BAC y JPM se multiplican por 3-4; GLD y los bonos por 1,5-2); las excepciones son ENPH y GME, que ya eran extremadamente volátiles en calma. La curtosis (exceso sobre una Normal) es alta **en ambos regímenes**: los retornos diarios tienen colas gruesas siempre, no solo en las crisis."))
n_t = sum(m.family == "t" for m in mar.marginals.values())
suelo = [i for i in mar.fit_table.index[mar.fit_table["df_t"] <= marginals.DF_BOUNDS[0] + 1e-6]]
normales = [i for i in mar.fit_table.index[mar.fit_table["elegida"] == "normal"]]
display(Markdown(f"**Normal frente a t.** La t gana en **{n_t} de {len(mar.marginals)}** ajustes (18 activos × 2 estados), también en calma. "
                 f"En {len(suelo)} casos los grados de libertad quedan en el suelo de 3 ({', '.join(f'{a} en {e}' for a, e in suelo)}): "
                 "colas muy pesadas que el simulador mantiene, pero acotadas."
                 + (f" La Normal gana en: {', '.join(f'{a} en {e}' for a, e in normales)}." if normales else "")))'''

REFUGIO_CODE = r'''r = mar.refuge
display(Markdown(
    f"### Respuestas de la práctica\n"
    f"* **¿Cuánto sube la volatilidad de HYG de calma a crisis?** De {pct(r['hyg_vol_calma'], 2)} a {pct(r['hyg_vol_crisis'], 2)} diaria: **×{r['hyg_ratio_vol']:.1f}**. "
    f"El crédito de alto rendimiento se comporta como un activo de riesgo (primo de liquidez + riesgo de impago), no como un bono refugio.\n"
    f"* **¿Sigue siendo GLD un refugio?** En crisis el oro rinde de media {pct(r['gld_media_crisis'], 3)} al día frente a {pct(r['sp500_media_crisis'], 3)} del S&P 500, "
    f"su correlación con el S&P 500 es {r['gld_corr_sp500_crisis']:.2f} (en calma {r['gld_corr_sp500_calma']:.2f}) y sube en el {pct(r['gld_dias_positivos_crisis'])} de los días de crisis. "
    f"**Sigue siendo un diversificador** (no cae con las acciones), pero su volatilidad también sube ×{r['gld_ratio_vol']:.1f}: refugio relativo, no inmunidad."))'''

QQ_CODE = r'''figures.plot_qq(mar.returns, reg.states, mar.marginals, ["HYG", "GLD", "GME", "JPM"])
plt.show()'''

QQ_MD = r'''**El gráfico (QQ).** Cada punto compara un cuantil de los datos (eje vertical) con el de la distribución ajustada (eje horizontal); la recta roja es el ajuste perfecto. Los puntos pegados a la recta indican buen ajuste.
Las **colas** (extremos) son lo que importa para el riesgo: en GLD y JPM la t sigue bien los datos incluso en los extremos; en **HYG en calma** la t con 3 g.l. sobreestima las colas (los puntos se aplanan en los extremos) y en **GME** los datos tienen una cola derecha mucho más extrema que cualquier ajuste (enero de 2021). Es un ajuste conservador en las pérdidas.'''

COP_MD = r'''## 4. Dependencia: correlaciones y cópulas por estado
**Cópula (teorema de Sklar).** Separa las marginales de la **estructura de dependencia**: cada activo se transforma a una uniforme (pseudo-observación por rangos) y la cópula modela cómo se mueven juntas.
La Gauss no tiene dependencia en las colas; la **t sí** (simétrica): captura que las caídas extremas tienden a coincidir. Se estiman en la **ventana común** (días con los 18 activos, desde que cotiza ENPH, abril de 2012),
por lo que la crisis de 2008 queda fuera de la estimación de dependencia (el escenario E2 la recalibra con una proxy). Las correlaciones de la t se estiman con la tau de Kendall (robusta a outliers) y los grados de libertad por máxima verosimilitud.'''

COP_CODE = r'''cop = run_copulas(mar.returns, reg.states, datos.prices, datos.rates, trace)
c = cop.summary
display(Markdown(f"Ventana común: **{c['n_ventana']}** días ({c['n_calma']} en calma y {c['n_crisis']} en crisis)."))
figures.plot_correlaciones(cop.corr[0], cop.corr[1], cop.names, c["bootstrap"])
plt.show()
b = c["bootstrap"]
display(Markdown(
    f"**Contraste «en crisis las correlaciones tienden a 1».** La correlación media entre pares pasa de **{b[0]['media']:.2f}** (IC 95 % {b[0]['ic_inf']:.2f}–{b[0]['ic_sup']:.2f}) en calma a **{b[1]['media']:.2f}** "
    f"(IC 95 % {b[1]['ic_inf']:.2f}–{b[1]['ic_sup']:.2f}) en crisis; la diferencia es {b['dif']['diferencia']:.2f} (IC 95 % {b['dif']['ic_inf']:.2f}–{b['dif']['ic_sup']:.2f}; "
    f"{pct(c['perfil_estados']['pct_pares_corr_sube'], 0)} de los pares suben). **Suben mucho y de forma significativa, pero no tienden a 1** en esta cartera diversificada: "
    "los bonos del Tesoro y el oro diluyen la media (son los cuadrados oscuros del mapa). Entre algunos pares de acciones (bancos entre sí, grandes tecnológicas entre sí) sí superan 0,8 en crisis."))'''

COP_FIG_MD = r'''**El gráfico.** Izquierda y centro: mapas de calor de la correlación de Pearson (amarillo = alta, morado = negativa). El de crisis es claramente más claro: todo el bloque de acciones se «enciende» a la vez; los bonos (últimas dos filas) siguen en tonos oscuros: negativos frente a las acciones. Derecha: correlación media con su intervalo del 95 % (bootstrap de días); la línea roja es el valor 1 que habría si «todo cayera junto».'''

COP_FIT_CODE = r'''display(cop.table[["loglik", "aic", "bic", "df", "lambda_L_ajustada", "lambda_L_empirica"]])
display(Markdown(
    f"**Elección de la cópula (D-07).** La t gana por AIC y BIC en ambos estados (g.l. {cop.fits[0]['t'].df:.1f} en calma y {cop.fits[1]['t'].df:.1f} en crisis; en calma es casi gaussiana). "
    "La *dependencia de cola inferior* λ_L (probabilidad de que dos activos estén a la vez en su peor 5 % de días) es más fiel con la t que con la Gauss en ambos estados: "
    "la Gauss subestima el riesgo conjunto. Clayton (un solo parámetro) queda descartada por rígida en 18 dimensiones."))'''

PREG_MD = r'''### Pregunta intermedia: ¿qué distingue realmente un estado del otro?'''

PREG_CODE = r'''p = c["perfil_estados"]
d = c["descomposicion_riesgo"]
evidencia = pd.DataFrame({
    "calma": [p["vix_medio_calma"], p["oas_medio_calma"], p["corr_media_calma"], p["lambda_L_calma"], d["vol_calma"]],
    "crisis": [p["vix_medio_crisis"], p["oas_medio_crisis"], p["corr_media_crisis"], p["lambda_L_crisis"], d["vol_crisis"]],
}, index=["VIX medio", "OAS high yield medio (pp)", "Correlación media", "Dependencia de cola λ_L (5 %)", "Vol. diaria de cartera equiponderada"])
display(evidencia)
sube_vol = (d["vol_solo_vols_crisis"] - d["vol_calma"]) / (d["vol_crisis"] - d["vol_calma"])
sube_corr = (d["vol_solo_corr_crisis"] - d["vol_calma"]) / (d["vol_crisis"] - d["vol_calma"])
display(Markdown(
    f"**Respuesta.** Los dos estados se distinguen en cuatro cosas medibles: **(1) volatilidad** (mediana por activo ×{p['mediana_ratio_vol']:.1f}); **(2) correlación** ({p['corr_media_calma']:.2f} → {p['corr_media_crisis']:.2f}); "
    f"**(3) dependencia de cola** ({p['lambda_L_calma']:.2f} → {p['lambda_L_crisis']:.2f}); y **(4) variables de mercado** (VIX {p['vix_medio_calma']:.0f} → {p['vix_medio_crisis']:.0f}; OAS {p['oas_medio_calma']:.1f} → {p['oas_medio_crisis']:.1f} pp). "
    f"¿Qué pesa más? Si en una cartera equiponderada se cambian solo las volatilidades a las de crisis, el riesgo sube {pct(sube_vol, 0)} del salto total; si se cambian solo las correlaciones, {pct(sube_corr, 0)}; el resto es **interacción** (volatilidades y correlaciones altas se refuerzan). "
    "**Interpretación económica:** una crisis es un régimen en el que los inversores dejan de diferenciar entre activos (la liquidez y el miedo dominan sobre los fundamentales), por eso sube la dependencia conjunta, y en el que la incertidumbre se paga (más VIX y más diferencial de crédito)."))'''

SIM_MD = r'''## 5. Simulador Monte Carlo y validación
**Mecánica (por trayectoria y día).** (1) El estado inicial se toma de la **distribución estacionaria** de la cadena (D-08; sin información de coyuntura, el mercado puede estar en cualquier régimen con su frecuencia de largo plazo); (2) el estado de cada día se sortea con la matriz de transición
`S_t ~ P[S_{t-1}, ·]`; (3) se genera un vector de uniformes `U_t` de la **cópula del estado activo**; (4) cada activo se obtiene con la inversa de su marginal del estado: `R_{t,i} = F⁻¹_{i,S_t}(U_{t,i})`. La cartera es **equiponderada sin rebalanceo** (*buy & hold*):
`W_t = (1/N)·Σ_i Π_{s≤t}(1+r_{i,s})`. Los retornos diarios simulados se topan en ±2× el extremo histórico de cada activo (afecta a una fracción mínima; D-31).

**Validación (D-09).** Se compara con **ventanas móviles reales de 126 días** (solapadas, con la riqueza reiniciada a 1 en cada una) y, para regímenes, con toda la serie 2006-2026.'''

SIM_CODE = r'''sim = run_simulation(mar, cop, reg.states, trace)
t = sim.tables
display(Markdown("### A. Cartera: abanico simulado y retornos diarios"))
figures.plot_abanico(sim.fan, sim.real_wealth)
plt.show()
figures.plot_hist_retornos(sim.real_daily, sim.sim_daily)
plt.show()
display(t["percentiles_diarios"])'''

SIM_FIG_MD = r'''**Los gráficos.** *Abanico*: la franja azul clara es el rango p5-p95 de la riqueza simulada de la cartera, la oscura p25-p75 y la línea azul la mediana; las líneas negra y gris son dos ventanas **reales** de 126 días (las últimas y las primeras). La realidad cae dentro del abanico: el modelo no «se sale del mapa».
*Histograma* (escala logarítmica, para ver las colas): el simulado reproduce el cuerpo de la distribución diaria real; en las colas extremas el real tiene días peores y mejores por GME (enero de 2021), que el tope de ±2× del simulador no replica por diseño.'''

SIM_B_CODE = r'''display(Markdown("### B. Regímenes: real frente a simulado"))
display(t["regimenes"])
display(Markdown("(*Real* = ventanas móviles de 126 días de los estados duros 2006-2026; *teórico* = valor de la cadena. Las duraciones medias de las ventanas están truncadas por el horizonte de 126 días, por eso son menores que las esperadas.)"))'''

SIM_C_CODE = r'''display(Markdown("### C. Riesgo: volatilidad, VaR 99 %, ES 99 % y drawdown (pérdidas en positivo)"))
display(t["riesgo"])
display(Markdown("**Sensibilidad quitando GME** (el *short squeeze* de enero de 2021 domina las colas de la cartera real *buy & hold*, porque su peso se disparaba antes de desplomarse):"))
display(t["riesgo_sin_GME"])
cob = sim.summary["cobertura"]
rg, rk = t["riesgo"], t["riesgo_sin_GME"]
display(Markdown(
    f"**Lectura.** *Regímenes*: el simulador reproduce el % de días en crisis, los cambios y las duraciones. *Riesgo*: con GME, la volatilidad real ({pct(rg.loc['vol_anualizada', 'real'])}) supera a la simulada ({pct(rg.loc['vol_anualizada', 'simulado'])}) por el *squeeze*; "
    f"sin GME es la simulada la que es algo mayor ({pct(rk.loc['vol_anualizada', 'simulado'])} frente a {pct(rk.loc['vol_anualizada', 'real'])}): **el modelo es conservador**. El VaR 99 % diario sin GME es {pct(rk.loc['var99_diario', 'simulado'], 2)} simulado frente a {pct(rk.loc['var99_diario', 'real'], 2)} real. "
    f"A 6 meses el VaR 99 % simulado ({pct(rg.loc['var99_6m', 'simulado'])}) está muy por encima del histórico ({pct(rg.loc['var99_6m', 'real'])}): el histórico a 6 meses 2012-2026 se apoya en unas {len(sim.real_wealth) / 126:.0f} ventanas independientes y **no contiene una crisis sostenida tipo 2008**; no es un buen árbitro del percentil 99. "
    f"El {pct(cob['cobertura_p5_p95_real'], 0)} de las ventanas reales cae dentro de la banda p5-p95 simulada (el 90 % sería lo ideal). "
    f"Fíjate también en el **VaR/ES paramétrico normal** (sin GME, con la volatilidad real): el VaR normal diario es {pct(rk.loc['var99_normal_diario', 'real'], 2)} frente a {pct(rk.loc['var99_diario', 'real'], 2)} histórico y el ES normal {pct(rk.loc['es99_normal_diario', 'real'], 2)} frente a {pct(rk.loc['es99_diario', 'real'], 2)}: "
    "la hipótesis de normalidad subestima la pérdida extrema de las colas gruesas, sobre todo en el ES. Es el motivo de usar ES y no solo VaR (el VaR no es subaditivo y ignora la severidad más allá del umbral). "
    "(Con GME la volatilidad real se infla y el VaR normal pasa a *sobrestimar*: por eso la comparación limpia es la de la tabla sin GME.)"))'''

SIM_D_CODE = r'''display(Markdown("### D. Comportamiento en estrés (R-VAL-04): ¿suben la volatilidad y los co-movimientos extremos?"))
display(t["estres"])
display(Markdown("En ambos regímenes la volatilidad mediana por activo, la correlación media y la dependencia de cola simuladas están cerca de las reales; la curtosis simulada es mayor (las colas de la t con 3-4 g.l. son más pesadas que las observadas, otra señal de conservadurismo). "
                 "En crisis suben a la vez la volatilidad, la correlación y la dependencia de cola, como en los datos."))'''

ESC_MD = r'''## 6. Escenarios de estrés
En los tres escenarios **solo se sustituye el estado de crisis** (la calma y la cadena de Markov no cambian) y las simulaciones **arrancan en crisis**. Se reportan VaR 99 % y ES 99 % **diario, a 1 mes (21 días) y a 6 meses (126 días)**, siempre como pérdida positiva sobre el capital; ES = media de las pérdidas iguales o superiores al VaR.

* **E1 · Estanflación 2022** (enero-diciembre de 2022): acciones y bonos cayeron a la vez porque la inflación obligó a subir tipos; se recalibran marginales y cópula con los 187 días de crisis de 2022.
* **E2 · Crisis de crédito 2008** (septiembre de 2008 a marzo de 2009): ENPH no cotizaba, así que se usa una **proxy** (ENPH = α + β·S&P 500 + ruido idiosincrático por estado; β ≈ 1,6 estimada en 2012-2026) y se hace análisis de sensibilidad. Es un supuesto fuerte: la «proxy» es una cartera sintética.
* **E3 · Pánico de liquidez crediticia** (escenario propio, D-11): en un pánico de liquidez los inversores venden lo que *pueden* vender, no lo que *quieren*; los diferenciales de crédito se abren y **la diversificación falla**. Se modela con volatilidad ×1,5 (exacta), un grado de libertad menos en las marginales,
  cópula t con 3 g.l. y correlaciones `R' = 0,7·R + 0,3·J` (J = matriz de unos). Es el mecanismo que no captura ningún dato histórico concreto: 2022 fue un problema de tipos e inflación y 2008 de crédito hipotecario; aquí la dependencia conjunta de cola es el protagonista.'''

ESC_CODE = r'''esc = run_scenarios(datos.prices, mar, cop, reg.states, sim, trace)
etiquetas = {"vol_anualizada": "Volatilidad anual", "var99_diario": "VaR 99 % diario", "es99_diario": "ES 99 % diario",
             "var99_1m": "VaR 99 % 1 mes", "es99_1m": "ES 99 % 1 mes", "var99_6m": "VaR 99 % 6 meses", "es99_6m": "ES 99 % 6 meses",
             "mdd_medio_6m": "Max. drawdown medio 6m", "mdd_p95_6m": "Max. drawdown p95 6m",
             "retorno_mediano_6m": "Retorno mediano 6m", "prob_perdida_20pct_6m": "Prob. pérdida > 20 % a 6m"}
display(esc.comparativa.rename(index=etiquetas))
terminales = {"Base": sim.wealth[:, -1] - 1.0, **{k: v["w"][:, -1] - 1.0 for k, v in esc.outcomes.items()}}
figures.plot_escenarios(terminales, esc.comparativa)
plt.show()'''

ESC_FIG_MD = r'''**El gráfico.** Izquierda: distribución del retorno de la cartera a 6 meses en cada escenario; cuanto más desplazada a la izquierda y más ancha, peor. E1 se parece a la base (la estanflación fue una sangría lenta, no un desplome);
E2 y E3 desplazan y ensanchan la distribución. Derecha: VaR y ES al 99 % a 6 meses; el ES siempre supera al VaR porque promedia todo lo que pasa más allá del umbral.'''

ESC_VAL_CODE = r'''display(Markdown("### Calibración y validación contra la ventana real"))
display(esc.calibracion)
display(esc.validacion_ventana.rename(columns={"retorno_real": "retorno real 6m", "percentil_retorno_real": "percentil en el escenario"}))
cb = esc.summary["corr_bonos_acciones_base"]
cm = esc.comparativa
display(Markdown(
    f"**Qué enseña cada escenario.** (1) **E1**: el ES 99 % a 6 meses ({pct(cm.loc['es99_6m', 'E1'])}) es *menor* que el de la crisis base ({pct(cm.loc['es99_6m', 'Base'])}) porque 2022 fue una sangría lenta (cópula casi gaussiana) y la crisis base incluye el COVID; "
    f"lo que cambia es el **signo de la cobertura**: la correlación media bono-acción es {esc.calibracion.loc[0, 'corr_bonos_acciones']:+.2f} en 2022, {cb['crisis']:+.2f} en la crisis base y {esc.calibracion.loc[1, 'corr_bonos_acciones']:+.2f} en 2008. "
    f"Un VaR de cola no capta que, en 2022, la renta fija dejó de proteger. (2) **E2** es el más duro de los dos escenarios históricos (volatilidad {pct(cm.loc['vol_anualizada', 'E2'])} anual) y arrastra una probabilidad de perder más del 20 % a 6 meses del {pct(cm.loc['prob_perdida_20pct_6m', 'E2'], 0)}. "
    f"(3) **E3** tiene un VaR/ES similar a E2 ({pct(cm.loc['es99_6m', 'E3'])} frente a {pct(cm.loc['es99_6m', 'E2'])}) pero por otra vía: la dependencia conjunta de cola sube (λ_L {esc.perfil_crisis.loc['E3', 'lambda_L_5pct']:.2f} frente a {esc.perfil_crisis.loc['E1', 'lambda_L_5pct']:.2f} en E1). "
    f"**Validación**: las ventanas reales de E1 y E2 caen en los percentiles {pct(esc.validacion_ventana.loc['E1', 'percentil_retorno_real'], 0)} y {pct(esc.validacion_ventana.loc['E2', 'percentil_retorno_real'], 0)} de su escenario: dentro del rango, pero en la parte mala, "
    "lo que sugiere que la persistencia de la crisis simulada (≈ 50 días de media) es menor que la de 2008 (230 días)."))'''

ESC_SENS_CODE = r'''display(esc.sensibilidad)
display(Markdown("**Sensibilidad (3.000 trayectorias por variante).** Cambiar la ventana de calibración, duplicar el ruido de la proxy de ENPH o quitar ENPH mueve el ES 99 % a 6 meses menos de 3 puntos porcentuales: las conclusiones no dependen de la proxy."))
fig_e = [("E1", esc.outcomes["E1"]), ("E2", esc.outcomes["E2"])]
for nombre, o in fig_e:
    figures.plot_abanico(o["fan"], o["real_w"], f"Escenario {nombre}: abanico simulado (arranca en crisis) y ventana real")
    plt.show()
figures.plot_abanico(esc.outcomes["E3"]["fan"], None, "Escenario E3 (pánico de liquidez): abanico simulado")
plt.show()
display(Markdown("**Los abanicos.** La línea negra de E1 y E2 es la evolución real de la cartera en los primeros 126 días del periodo; queda dentro del abanico simulado pero en su mitad inferior. El abanico de E3 no tiene equivalente real: es un escenario hipotético."))'''

FIN_MD = r'''## 7. Conclusiones, limitaciones y traza del pipeline
**Conclusiones para el Comité de Riesgos.**
1. El riesgo de la cartera **no es estable**: un régimen de crisis (alrededor de un sexto de los días) multiplica por ≈ 2 la volatilidad y más que duplica la correlación media; modelar con parámetros únicos subestima las pérdidas extremas.
2. **VaR y ES al 99 %** crecen de forma muy distinta según el tipo de crisis: la estanflación casi no aumenta la pérdida de cola, pero **elimina la cobertura de la renta fija**; el crédito y el pánico de liquidez la duplican.
3. La diversificación funciona **menos de lo que se cree** cuando más hace falta (la correlación sube), y el oro y los bonos son diversificadores, no seguros.

**Limitaciones y riesgo de modelo.** (i) Solo dos regímenes y la persistencia de la crisis simulada es menor que la de 2008; (ii) la dependencia se estima en 2012-2026 (370 días de crisis para 153 correlaciones): ruidosa; (iii) la proxy de ENPH es sintética;
(iv) los retornos diarios dentro de un estado son independientes (sin agrupamiento de volatilidad dentro del régimen); (v) las ventanas reales solapadas no son independientes, por lo que su VaR al 99 % es poco informativo; (vi) las colas derechas simuladas (GME, ENPH compuestos) son poco realistas y afectan a la media, no al VaR/ES;
(vii) los bonos se aproximan por duración y convexidad, sin diferencial ni *roll-down*. Recomendaciones de gestión: monitorizar el régimen con la probabilidad *filtrada* (sin *look-ahead*), fijar límites con **ES** y no solo con VaR, y no confiar en la cobertura de renta fija ante shocks de inflación.'''

FIN_CODE = r'''resultados = build_results(reg, mar, cop, sim, esc)
ruta = write_results(resultados, trace)
print("Resultados guardados en", ruta)
display(trace.to_frame())'''

TEORIA_MD = r'''### Anexo: teoría mínima
* **HMM.** El estado oculto sigue una cadena de Markov (la transición solo depende del estado actual); se estima maximizando la verosimilitud de los datos observados (Baum-Welch/EM) y devuelve la probabilidad de cada estado por día y la matriz de transición. *Filtering* = información hasta *t*; *smoothing* = toda la serie.
* **Cópula.** Por el teorema de Sklar, cualquier distribución conjunta = marginales + cópula. La t tiene dependencia de cola simétrica; la Gauss no.
* **VaR frente a ES.** El VaR (cuantil de la pérdida) no es subaditivo y no dice nada de la severidad más allá del umbral; el ES (media de las pérdidas ≥ VaR) es una medida coherente.
* **Por qué fallaron los modelos en 2022.** Supusieron normalidad y correlaciones estáticas bono-acción negativas; acciones y bonos cayeron a la vez.'''
