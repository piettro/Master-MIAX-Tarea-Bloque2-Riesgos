"""La matriz de QA cubre todos los requisitos de la matriz oficial."""
import re

from stresslab import config


def _ids(texto: str) -> set[str]:
    return set(re.findall(r"R-[A-Z]{3}-\d{2}", texto))


def test_qa_cubre_todos_los_requisitos() -> None:
    docs = config.ROOT / "docs"
    matriz = _ids((docs / "01_MATRIZ_REQUISITOS.md").read_text(encoding="utf-8"))
    qa = _ids((docs / "04_QA_MATRIZ.md").read_text(encoding="utf-8"))
    assert matriz and not (matriz - qa), f"Faltan en el QA: {sorted(matriz - qa)}"
