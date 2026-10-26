"""Configuración central. Cada parámetro lleva su justificación."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# Semilla única: todo el azar sale de aquí vía numpy.random.Generator (reproducibilidad, R-TRV-03).
SEED: int = 20260215

# Inicio exigido por el enunciado (R-DAT-01); incluye la crisis de 2008 para el escenario E2.
START_DATE: str = "2006-01-01"

# Equities del universo (R-DAT-02), ya ajustados por splits y dividendos.
EQUITIES: tuple[str, ...] = (
    "AAPL", "AMZN", "BAC", "BRK-B", "CVX", "ENPH", "GLD", "GME",
    "GOOGL", "JNJ", "JPM", "MSFT", "NVDA", "PG", "XOM",
)

# Crédito: el activo «HYG (BAMLH0A0HYM2)» es la serie OAS de FRED (respuesta del profesor, D-02); se convierte
# en retorno con carry + duración de spread (ver CREDIT_SPREAD_DURATION). El ETF HYG de Yahoo se conserva
# solo como contraste de la proxy (no entra en la cartera). Bonos 10a/2a vía FRED (D-03).
CREDIT_ASSET: str = "HYG"
CREDIT_ETF: str = "HYG"

# Duración efectiva del crédito high yield respecto a su rendimiento (OAS + tipo libre de riesgo): 4 años.
# Coincide con el valor típico del índice ICE BofA US High Yield y con la estimada por MCO contra el ETF HYG
# (≈ 3,9 en 2007-2026; 3,8 antes de 2020 y 4,1 desde 2020). Ignora convexidad y pérdidas por impago (D-39).
CREDIT_SPREAD_DURATION: float = 4.0
FRED_SERIES: dict[str, str] = {"UST10Y": "DGS10", "UST2Y": "DGS2"}

# Spread de crédito OAS de ICE BofA US High Yield (FRED): factor de estado, no activo (D-02).
CREDIT_OAS: str = "BAMLH0A0HYM2"

# Duración/convexidad de los bonos par sintéticos: vencimiento en años (10a y 2a, enunciado) (D-03).
BOND_MATURITIES: dict[str, float] = {"UST10Y": 10.0, "UST2Y": 2.0}

# Cupones semestrales: convención de los Treasuries; fija duración y convexidad del bono par.
BOND_COUPON_FREQ: int = 2

# Umbral de outlier: retorno a más de 8 sigmas robustas (MAD). Alto a propósito, para no eliminar
# colas reales (GME 2021 se marca pero no se borra; spec §3).
OUTLIER_MAD_Z: float = 8.0

# Un salto diario de más del 40 % sin explicación sugiere split mal ajustado; se revisa a mano.
JUMP_THRESHOLD: float = 0.40

# Una historia que arranca más de 30 días tras START_DATE se considera corta (ENPH 2012, HYG 2007).
SHORT_HISTORY_DAYS: int = 30

# Proxy de mercado y volatilidad para los factores de estado (D-04).
MARKET_TICKER: str = "^GSPC"
VIX_TICKER: str = "^VIX"

# Horizonte de simulación: 6 meses ≈ 126 días hábiles (enunciado F4).
HORIZON_DAYS: int = 126

# Trayectorias Monte Carlo: mínimo exigido 10.000 (R-SIM-01).
N_SIMS: int = 10_000

# Días hábiles por año para anualizar volatilidad y rendimientos de bonos.
TRADING_DAYS: int = 252

# Nivel de confianza de VaR/ES (enunciado F5; estándar de riesgo de mercado).
VAR_LEVEL: float = 0.99

# Estados del HMM: calma/crisis. Más estados queda fuera de alcance (spec §1).
N_STATES: int = 2

# HMM: covarianza completa porque los factores (acción, VIX, crédito) están correlacionados en crisis.
HMM_COVARIANCE: str = "full"

# Iteraciones máximas de EM: holgadas; EM converge antes con tolerancia por defecto.
HMM_N_ITER: int = 300

# EM es sensible a la inicialización (óptimos locales): se prueban varias semillas y gana la mayor verosimilitud.
HMM_N_INIT: int = 10

# Umbral de decisión dura sobre P(crisis) suavizada (D-06): 0,5 = estado más probable.
CRISIS_THRESHOLD: float = 0.5

# Duración mínima de un episodio en días (D-06): evita «crisis de 3 días»; un mes de mercado ≈ 21 días,
# por eso 15 queda en el rango 10-20 de la spec. Se reporta sensibilidad.
MIN_EPISODE_DAYS: int = 15
MIN_EPISODE_SENSITIVITY: tuple[int, ...] = (10, 15, 20)

# Episodios históricos conocidos para etiquetar crisis detectadas (inicio, fin, narrativa económica).
KNOWN_EVENTS: tuple[tuple[str, str, str], ...] = (
    ("2007-07-01", "2009-06-30", "Crisis financiera global: hipotecas subprime, quiebra de Lehman, congelación del crédito"),
    ("2010-04-15", "2010-07-15", "Crisis de deuda soberana europea y «flash crash»"),
    ("2011-07-15", "2011-11-30", "Rebaja de rating de EE. UU. y contagio de la crisis europea"),
    ("2015-08-15", "2016-03-15", "Devaluación del yuan, caída del petróleo y miedo a desaceleración global"),
    ("2018-01-25", "2018-04-15", "Repunte de volatilidad (febrero 2018) y subida de tipos"),
    ("2018-10-01", "2019-01-15", "Venta masiva del 4T 2018: Fed restrictiva y guerra comercial"),
    ("2020-02-15", "2020-06-30", "Pandemia de COVID-19: parón económico y crisis de liquidez"),
    ("2020-09-01", "2020-10-31", "Venta de megacaps tecnológicas tras el rally de verano de 2020"),
    ("2023-03-01", "2023-04-15", "Quiebra de Silicon Valley Bank y estrés en bancos regionales"),
    ("2022-01-01", "2022-12-31", "Estanflación: inflación alta y subidas agresivas de tipos; caen acciones y bonos a la vez"),
    ("2025-03-01", "2025-06-15", "Aranceles de abril de 2025 y shock de incertidumbre comercial"),
)

# Tope de los retornos diarios simulados: ±2× el extremo histórico de cada activo (y nunca < -99 %).
# La t con 3 g.l. tiene colas tan pesadas que unas pocas de 22 millones de extracciones serían
# irreales (+300 %); el tope se aplica a una fracción ínfima de datos, que se reporta.
SIM_CLIP_MULT: float = 2.0

# Ventanas de calibración de los escenarios (D-10): E1 = año 2022 completo; E2 = septiembre 2008
# (quiebra de Lehman) a marzo 2009 (suelo del S&P 500). Las alternativas miden la sensibilidad a la ventana.
E1_WINDOW: tuple[str, str] = ("2022-01-01", "2022-12-31")
E1_WINDOW_ALT: tuple[str, str] = ("2022-01-01", "2022-10-31")
E2_WINDOW: tuple[str, str] = ("2008-09-01", "2009-03-31")
E2_WINDOW_ALT: tuple[str, str] = ("2008-09-15", "2009-03-31")

# Mínimo de días de crisis en la ventana para recalibrar; por debajo se usan todos los días de la ventana.
SCEN_MIN_OBS: int = 60

# E3 «pánico de liquidez crediticia» (D-11): volatilidad ×1,5; t de 3 g.l. (la más gruesa admitida)
# y correlaciones R' = (1-λ)R + λJ con λ = 0,3 (recetas de la spec §5.5).
E3_VOL_MULT: float = 1.5
E3_COPULA_DF: float = 3.0
E3_LAMBDA: float = 0.3

# Trayectorias de las sensibilidades: menos que las 10.000 oficiales para acotar el tiempo de cálculo.
SENS_SIMS: int = 3000

# Ruido idiosincrático de la proxy de ENPH multiplicado en la sensibilidad de E2.
PROXY_NOISE_MULT: float = 2.0

# Rutas del proyecto.
ROOT: Path = Path(__file__).resolve().parents[2]
RAW_DIR: Path = ROOT / "data" / "raw"
OUTPUT_DIR: Path = ROOT / "outputs"


def rng(seed: int | None = None) -> np.random.Generator:
    """Generador reproducible; sin argumento usa la semilla única SEED."""
    return np.random.default_rng(SEED if seed is None else seed)


def is_offline() -> bool:
    """Modo offline: OFFLINE=true o caché en data/raw/ (spec §4, regla 2)."""
    if os.environ.get("OFFLINE", "").strip().lower() in {"1", "true", "yes"}:
        return True
    return RAW_DIR.is_dir() and any(RAW_DIR.glob("*.csv"))
