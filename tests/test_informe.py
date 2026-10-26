"""Tests del informe: sin cifras a mano en la plantilla, ≤ 3 páginas y cifras ligadas a results.json."""
import copy
import json
import re

import pytest

from stresslab import config
from stresslab.reporting import report

RESULTADOS = config.OUTPUT_DIR / "results.json"
necesita_resultados = pytest.mark.skipif(not RESULTADOS.exists(), reason="ejecuta scripts/run_pipeline.py primero")


def test_plantilla_sin_cifras_escritas_a_mano() -> None:
    """R-INF-06: solo se admiten años de episodios históricos y el nombre «S&P 500»."""
    t = (report.TEMPLATES / "informe.html.j2").read_text(encoding="utf-8")
    t = re.sub(r"<style>.*?</style>", "", t, flags=re.S)
    t = re.sub(r"\{\{.*?\}\}|\{%.*?%\}", "", t)
    t = re.sub(r"<[^>]+>", "", t)  # Solo cuenta el texto visible, no los atributos ni las etiquetas.
    t = re.sub(r"\b(19|20)\d{2}\b", "", t).replace("S&amp;P 500", "")
    assert not re.findall(r"\d+", t), f"Cifras en la plantilla: {re.findall(r'.{15}\d+.{10}', t)[:5]}"


@necesita_resultados
def test_cifras_dependen_de_results_json() -> None:
    r = report.load_results(RESULTADOS)
    html0 = report.render_html(r)
    r2 = copy.deepcopy(r)
    r2["escenarios"]["comparativa"]["E2"]["es99_6m"] = 0.123456
    html1 = report.render_html(r2)
    assert "12,3 %" in html1 and "12,3 %" not in html0


@necesita_resultados
def test_informe_pdf_maximo_tres_paginas(tmp_path) -> None:
    from pypdf import PdfReader

    destino = tmp_path / "informe.pdf"
    paginas = report.build_pdf(RESULTADOS, config.OUTPUT_DIR, destino)
    assert paginas == len(PdfReader(destino).pages) and 1 <= paginas <= 3


@necesita_resultados
def test_results_json_tiene_las_claves_del_informe() -> None:
    r = json.loads(RESULTADOS.read_text(encoding="utf-8"))
    for clave in ("regimenes", "copulas", "escenarios", "refugio", "parametros", "simulacion_base", "episodios"):
        assert clave in r
    assert {"Base", "E1", "E2", "E3"} == set(r["escenarios"]["comparativa"])
