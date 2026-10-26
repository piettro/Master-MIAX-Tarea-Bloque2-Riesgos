"""Genera el notebook AUTOCONTENIDO: incrusta el código de src/ (sin imports locales) y añade la narrativa.

Uso:  python scripts/build_notebook.py            # solo construye
      python scripts/build_notebook.py --ejecutar # construye y ejecuta (guarda salidas en el .ipynb)
"""
from __future__ import annotations

import ast
import base64
import gzip
import re
import sys
import textwrap
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

sys.path.insert(0, str(Path(__file__).resolve().parent))
import notebook_narrativa as nar  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "stresslab"
SALIDA = ROOT / "notebooks" / "Practica_B2-2_Escenarios_Estres.ipynb"

# (espacio de nombres, ruta relativa, módulo opcional). Orden = dependencias.
MODULOS: list[tuple[str, str, bool]] = [
    ("config", "config.py", False), ("base", "data/base.py", False), ("csv_cache", "data/csv_cache.py", False),
    ("fred", "data/fred.py", True), ("yahoo", "data/yahoo.py", True), ("loader", "data/loader.py", False),
    ("returns", "data/returns.py", False), ("audit", "data/audit.py", False),
    ("features", "domain/features.py", False), ("regimes", "domain/regimes.py", False),
    ("marginals", "domain/marginals.py", False), ("copulas", "domain/copulas.py", False),
    ("simulator", "domain/simulator.py", False), ("portfolio", "domain/portfolio.py", False),
    ("risk", "domain/risk.py", False), ("scenarios", "domain/scenarios.py", False),
    ("validation", "domain/validation.py", False), ("trace", "orchestration/trace.py", False),
    ("pipeline", "orchestration/pipeline.py", False), ("figures", "reporting/figures.py", False),
]
NOMBRES = {m[0] for m in MODULOS}
IMPORT_LOCAL = re.compile(r"^(\s*)from stresslab(?:\.([\w.]+))? import (.+?)\s*$")
EXPORTAR_ALGUNOS = "return types.SimpleNamespace(**{k: v for k, v in locals().items() if not k.startswith('__')})"
ROOT_ORIGINAL = "ROOT: Path = Path(__file__).resolve().parents[2]"
ROOT_NOTEBOOK = ('ROOT: Path = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()  '
                 "# En el notebook no hay __file__: la raíz es la carpeta que contiene notebooks/ (o el directorio actual).")


def _reescribir_import(linea: str) -> str | None:
    """Sustituye un import local por la asignación desde el espacio de nombres del módulo."""
    m = IMPORT_LOCAL.match(linea)
    if not m:
        return None
    sangria, ruta, nombres = m.groups()
    if "(" in nombres:
        raise ValueError(f"Import multilínea no soportado: {linea}")
    modulo = (ruta or "").split(".")[-1]
    destino = [n.strip() for n in nombres.split(",")]
    if not ruta or all(n in NOMBRES for n in destino):  # from stresslab[.pkg] import modulo(s): ya son globales.
        return f"{sangria}pass  # import local sustituido por el espacio de nombres del notebook"
    lineas = []
    for n in destino:
        origen, _, alias = n.partition(" as ")
        lineas.append(f"{sangria}{(alias or origen).strip()} = {modulo}.{origen.strip()}")
    return "\n".join(lineas)


def _nombres_de_modulo(ruta: str) -> set[str]:
    """Nombres definidos a nivel de módulo (funciones, clases y asignaciones)."""
    nombres: set[str] = set()
    for n in ast.parse((SRC / ruta).read_text(encoding="utf-8")).body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            nombres.add(n.name)
        elif isinstance(n, (ast.Assign, ast.AnnAssign)):
            for t in (n.targets if isinstance(n, ast.Assign) else [n.target]):
                if isinstance(t, ast.Name):
                    nombres.add(t.id)
    return nombres


TOPLEVEL = {m[0]: _nombres_de_modulo(m[1]) for m in MODULOS}


def _comprobar_sombras(nombre: str, codigo: str) -> None:
    """Falla si un local oculta un módulo y la función accede a un nombre de ese módulo (p. ej. `copulas.fit_t`)."""
    for fn in ast.walk(ast.parse(codigo)):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            locales = {a.arg for a in fn.args.args + fn.args.kwonlyargs}
            locales |= {t.id for n in ast.walk(fn) if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
            for n in ast.walk(fn):
                if (isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id in locales
                        and n.attr in TOPLEVEL.get(n.value.id, set())):
                    raise ValueError(f"{nombre}.{fn.name}: «{n.value.id}» oculta el módulo y se accede a {n.attr}")


def envolver_modulo(nombre: str, ruta: str, opcional: bool) -> str:
    """Convierte un módulo de src/ en una celda de código sin imports locales."""
    codigo = (SRC / ruta).read_text(encoding="utf-8")
    codigo = codigo.replace("from __future__ import annotations\n", "")
    if nombre == "config":
        assert ROOT_ORIGINAL in codigo
        codigo = codigo.replace(ROOT_ORIGINAL, ROOT_NOTEBOOK)
    _comprobar_sombras(nombre, codigo)
    lineas = [_reescribir_import(l) or l for l in codigo.splitlines()]
    cuerpo = textwrap.indent("\n".join(lineas), "    ")
    cabecera = f"# === módulo: {ruta} ===\n"
    fn = f"_modulo_{nombre}"
    envuelto = f"{cabecera}def {fn}():\n{cuerpo}\n    {EXPORTAR_ALGUNOS}\n\n"
    if opcional:
        return (envuelto + f"try:\n    {nombre} = {fn}()\nexcept ImportError as _e:  # Librería de red ausente: el notebook sigue con la caché.\n"
                f"    {nombre} = None\n    warnings.warn(f\"{ruta}: no se pudo importar ({{_e}}); solo se usará la caché\")\ndel {fn}\n")
    return envuelto + f"{nombre} = {fn}()\ndel {fn}\n"


def instantanea() -> str:
    """Diccionario Python con los CSV de data/raw comprimidos (gzip) y en base64."""
    partes = {}
    for nombre in ("prices.csv", "rates.csv"):
        blob = base64.b64encode(gzip.compress((ROOT / "data" / "raw" / nombre).read_bytes(), 9)).decode()
        partes[nombre] = "(\n" + "\n".join(f'    "{blob[i:i + 100]}"' for i in range(0, len(blob), 100)) + "\n)"
    return "{\n" + ",\n".join(f'    "{k}": {v}' for k, v in partes.items()) + ",\n}"


def construir() -> nbformat.NotebookNode:
    """Ensambla todas las celdas en orden."""
    md, code = new_markdown_cell, new_code_cell
    c = [md(nar.PORTADA), code(nar.SETUP), md(nar.CONFIG_MD)]
    c += [code(envolver_modulo(*m)) for m in MODULOS]
    # Atajos a las funciones del pipeline usadas en las secciones (espacios de nombres ya definidos).
    c.append(code("run_regimes, run_marginals, run_copulas = pipeline.run_regimes, pipeline.run_marginals, pipeline.run_copulas\n"
                  "run_simulation, run_scenarios, build_results = pipeline.run_simulation, pipeline.run_scenarios, pipeline.build_results\n"
                  "load_market_data, Trace, write_results = loader.load_market_data, trace.Trace, trace.write_results"))
    c += [md(nar.DATOS_MD), code(nar.DATOS_CODE.replace("__SNAPSHOT__", instantanea())), code(nar.AUDIT_CODE), md(nar.AUDIT_MD), code(nar.BONOS_CODE)]
    c += [md(nar.REG_MD), code(nar.REG_CODE), code(nar.REG_FIG_CODE), md(nar.REG_FIG_MD), code(nar.EPI_CODE)]
    c += [md(nar.MAR_MD), code(nar.MAR_CODE), code(nar.REFUGIO_CODE), code(nar.QQ_CODE), md(nar.QQ_MD)]
    c += [md(nar.COP_MD), code(nar.COP_CODE), md(nar.COP_FIG_MD), code(nar.COP_FIT_CODE), md(nar.PREG_MD), code(nar.PREG_CODE)]
    c += [md(nar.SIM_MD), code(nar.SIM_CODE), md(nar.SIM_FIG_MD), code(nar.SIM_B_CODE), code(nar.SIM_C_CODE), code(nar.SIM_D_CODE)]
    c += [md(nar.ESC_MD), code(nar.ESC_CODE), md(nar.ESC_FIG_MD), code(nar.ESC_VAL_CODE), code(nar.ESC_SENS_CODE)]
    c += [md(nar.FIN_MD), code(nar.FIN_CODE), md(nar.TEORIA_MD)]
    nb = new_notebook(cells=c)
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python"}
    return nb


def ejecutar(nb: nbformat.NotebookNode) -> nbformat.NotebookNode:
    """Ejecuta el notebook desde notebooks/ (equivale a `nbconvert --execute`)."""
    from nbconvert.preprocessors import ExecutePreprocessor

    ExecutePreprocessor(timeout=1800, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(SALIDA.parent)}})
    return nb


def main() -> None:
    nb = construir()
    if "--ejecutar" in sys.argv:
        nb = ejecutar(nb)
    SALIDA.parent.mkdir(exist_ok=True)
    nbformat.write(nb, SALIDA)
    print(f"Notebook escrito en {SALIDA} ({SALIDA.stat().st_size / 1e6:.1f} MB, {len(nb.cells)} celdas)")


if __name__ == "__main__":
    main()
