"""Guardrails deterministas: sin Deep Learning y sin red fuera de yahoo.py/fred.py."""
import json
import re
from pathlib import Path
from typing import Iterator

ROOT = Path(__file__).resolve().parents[1]
EXCLUIR = {".venv", "__pycache__", ".git", ".pytest_cache", "bronze"}
PROHIBIDOS = r"(torch|tensorflow|keras|sklearn\.neural_network)"
IMPORT_PROHIBIDO = re.compile(rf"^\s*(import|from)\s+{PROHIBIDOS}\b", re.M)
REQ_PROHIBIDO = re.compile(r"^\s*(torch|tensorflow|keras|scikit-learn)\b", re.M | re.I)
IMPORT_RED = re.compile(r"^\s*(import|from)\s+(yfinance|fredapi|pandas_datareader)\b", re.M)


def _ficheros(sufijos: tuple[str, ...]) -> Iterator[Path]:
    """Ficheros del repo con esos sufijos, sin entornos ni cachés."""
    for p in ROOT.rglob("*"):
        if p.is_file() and p.suffix in sufijos and not (set(p.relative_to(ROOT).parts) & EXCLUIR):
            yield p


def _texto(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")


def test_sin_deep_learning() -> None:
    este = Path(__file__).resolve()
    malos = [str(p) for p in _ficheros((".py", ".ipynb")) if p != este and IMPORT_PROHIBIDO.search(_texto(p))]
    req = ROOT / "requirements.txt"
    if req.exists() and REQ_PROHIBIDO.search(_texto(req)):
        malos.append(str(req))
    assert not malos, f"Deep Learning prohibido: {malos}"


def _celdas_codigo(p: Path) -> list[str]:
    """Fuente de las celdas de código de un notebook."""
    return ["".join(c["source"]) for c in json.loads(_texto(p))["cells"] if c["cell_type"] == "code"]


def test_red_solo_en_yahoo_y_fred() -> None:
    este = Path(__file__).resolve()
    malos = [str(p) for p in _ficheros((".py",))
             if p != este and p.name not in {"yahoo.py", "fred.py"} and IMPORT_RED.search(_texto(p))]
    # En el notebook autocontenido solo pueden aparecer en las celdas generadas desde yahoo.py y fred.py.
    for nb in _ficheros((".ipynb",)):
        for celda in _celdas_codigo(nb):
            if IMPORT_RED.search(celda) and not re.match(r"# === módulo: data/(yahoo|fred)\.py", celda):
                malos.append(f"{nb}: {celda[:60]!r}")
    assert not malos, f"Imports de red fuera de yahoo.py/fred.py: {malos}"


def test_notebook_sin_imports_locales_y_sin_deep_learning() -> None:
    nb = ROOT / "notebooks" / "Practica_B2-2_Escenarios_Estres.ipynb"
    if not nb.exists():
        return
    codigo = "\n".join(_celdas_codigo(nb))
    assert not re.search(r"^\s*(import|from)\s+stresslab\b", codigo, re.M), "El notebook importa stresslab"
    assert not IMPORT_PROHIBIDO.search(codigo), "Deep Learning en el notebook"


def test_dominio_no_importa_proveedores_vivos() -> None:
    patron = re.compile(r"^\s*(import|from)\s+stresslab\.data\.(yahoo|fred)\b", re.M)
    for carpeta in ("domain", "orchestration"):
        for p in (ROOT / "src" / "stresslab" / carpeta).rglob("*.py"):
            assert not patron.search(_texto(p)), f"{p} importa un proveedor vivo"


def test_guardrail_detecta_import_prohibido() -> None:
    assert IMPORT_PROHIBIDO.search("import torch\n")
    assert IMPORT_PROHIBIDO.search("from sklearn.neural_network import MLPRegressor")
    assert not IMPORT_PROHIBIDO.search("import numpy as np")
