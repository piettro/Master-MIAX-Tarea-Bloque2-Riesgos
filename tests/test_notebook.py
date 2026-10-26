"""Notebook autocontenido: estructura (rápido) y ejecución real con nbconvert en una carpeta vacía (lento)."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import nbformat
import pytest

from stresslab import config

NB = config.ROOT / "notebooks" / "Practica_B2-2_Escenarios_Estres.ipynb"
SECCIONES = ("## 1. Datos", "## 2. Regímenes", "## 3. Distribuciones marginales", "## 4. Dependencia",
             "## 5. Simulador", "## 6. Escenarios", "## 7. Conclusiones")
necesita_nb = pytest.mark.skipif(not NB.exists(), reason="ejecuta scripts/build_notebook.py primero")


@necesita_nb
def test_estructura_del_notebook() -> None:
    nb = nbformat.read(NB, as_version=4)
    md = "\n".join(c.source for c in nb.cells if c.cell_type == "markdown")
    assert all(s in md for s in SECCIONES)
    codigo = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    assert "# === módulo: domain/regimes.py" in codigo and "def run_simulation" in codigo
    assert "SEED" in codigo and "SNAPSHOT" in codigo


@necesita_nb
@pytest.mark.slow
def test_nbconvert_execute_en_carpeta_vacia(tmp_path: Path) -> None:
    """R-NB-01: el .ipynb solo, sin repo ni red, se ejecuta de principio a fin y reproduce los resultados."""
    (tmp_path / "notebooks").mkdir()
    copia = tmp_path / "notebooks" / NB.name
    shutil.copy(NB, copia)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.run([sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute",
                           "--ExecutePreprocessor.timeout=1800", "--output", "ejecutado.ipynb", str(copia)],
                          cwd=tmp_path / "notebooks", env=env, capture_output=True, text=True, timeout=2400)
    assert proc.returncode == 0, proc.stderr[-2000:]
    nuevo = json.loads((tmp_path / "outputs" / "results.json").read_text(encoding="utf-8"))
    ref = json.loads((config.OUTPUT_DIR / "results.json").read_text(encoding="utf-8"))
    for esc in ("Base", "E1", "E2", "E3"):
        assert nuevo["escenarios"]["comparativa"][esc] == pytest.approx(ref["escenarios"]["comparativa"][esc], rel=1e-9)
    assert nuevo["regimenes"]["pct_crisis"] == pytest.approx(ref["regimenes"]["pct_crisis"])
