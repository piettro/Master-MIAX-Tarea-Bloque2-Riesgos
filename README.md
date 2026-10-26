# Práctica B2-2 · Escenarios de estrés y cambios de régimen de mercado

Motor de *stress testing* para el Módulo de Gestión de Riesgos (MIAX): HMM de 2 estados (calma/crisis),
marginales y cópulas por estado, Monte Carlo (10.000 × 126 días), VaR/ES 99 % y tres escenarios de estrés.

**Entregables**

| Entregable | Fichero | Cómo se genera |
|---|---|---|
| Notebook autocontenido (70 %) | `notebooks/Practica_B2-2_Escenarios_Estres.ipynb` | `python scripts/build_notebook.py --ejecutar` |
| Resumen ejecutivo PDF ≤ 3 págs. (30 %) | `outputs/Informe_Ejecutivo_B2-2.pdf` | `python scripts/build_report.py` |

Fuente de verdad: `docs/01_MATRIZ_REQUISITOS.md`, `docs/02_SPEC_PRODUCTO.md`, `docs/03_DECISIONES_Y_PREGUNTAS.md`
(decisiones y su justificación económica). Estado de cumplimiento: `docs/04_QA_MATRIZ.md`.

## Instalación

```bash
py -3.12 -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m pip install -e .
```

Copiar `.env.example` a `.env` y rellenar `FRED_API_KEY` (opcional: sin clave se usa el CSV público de FRED; nunca se versiona).

## Uso

```bash
.venv/Scripts/python scripts/run_pipeline.py     # fases 1-6: figuras, tablas y outputs/results.json (sin red)
.venv/Scripts/python scripts/build_report.py     # PDF desde results.json (falla si pasa de 3 páginas)
.venv/Scripts/python scripts/build_notebook.py --ejecutar   # notebook autocontenido con las salidas
.venv/Scripts/python scripts/fetch_rates.py      # (manual, con red) refresca DGS10/DGS2 en data/raw
.venv/Scripts/python scripts/fetch_data.py       # (manual, con red) refresca precios y tipos
```

## Tests

```bash
.venv/Scripts/python -m pytest              # rápidos (≈ 1 min): solo datos sintéticos, red bloqueada
.venv/Scripts/python -m pytest -m slow      # ejecuta el notebook con nbconvert en una carpeta vacía (≈ 5 min)
```

Los tests de informe y notebook necesitan haber ejecutado antes `run_pipeline.py`, `build_report.py` y `build_notebook.py`.

## Estructura

- `src/stresslab/config.py`: semilla única y parámetros, cada uno con su justificación.
- `src/stresslab/data/`: Protocols (`base.py`), caché CSV, proveedor sintético, auditoría, retornos y, únicos con acceso a red, `yahoo.py` y `fred.py`.
- `src/stresslab/domain/`: factores, regímenes, marginales, cópulas, simulador, cartera, riesgo, escenarios y validación.
- `src/stresslab/orchestration/`: `pipeline.py` y `trace.py` (traza por paso y `results.json`).
- `src/stresslab/reporting/`: figuras, plantilla del informe (`templates/informe.html.j2`) y generador de PDF.
- `scripts/`: `build_notebook.py` incrusta el código de `src/` en celdas sin imports locales; `notebook_narrativa.py` guarda el texto del notebook.
- `data/raw/` (caché CSV versionada), `data/bronze/` (CSV originales previos, sin tocar), `outputs/`, `docs/`, `tests/`.

## Reglas clave

- **Modo offline:** con `data/raw/` con CSV o `OFFLINE=true` nada toca la red. El notebook incluye una instantánea de los datos y la restaura si falta la caché; para refrescar desde la red se lanza con `ACTUALIZAR_DATOS=true`.
- **Reproducibilidad:** una sola semilla (`config.SEED`) propagada con `numpy.random.Generator`; el notebook y el pipeline dan los mismos resultados.
- **Prohibido Deep Learning** (torch, tensorflow, keras, sklearn.neural_network): lo vigila `tests/test_guardrails.py`, que además impide importar yfinance/fredapi fuera de `yahoo.py`/`fred.py`.
- **Cero cifras a mano en el informe:** todo sale de `outputs/results.json`; un test comprueba que la plantilla no contiene números.
- **Sin secretos en el repo:** claves por variable de entorno.
