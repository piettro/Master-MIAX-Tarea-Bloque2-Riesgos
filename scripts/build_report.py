"""Genera outputs/Informe_Ejecutivo_B2-2.pdf desde outputs/results.json y las figuras de outputs/."""
from __future__ import annotations

from stresslab import config
from stresslab.reporting.report import build_pdf

MAX_PAGINAS = 3


def main() -> None:
    """Construye el PDF y verifica el límite de páginas."""
    destino = config.OUTPUT_DIR / "Informe_Ejecutivo_B2-2.pdf"
    paginas = build_pdf(config.OUTPUT_DIR / "results.json", config.OUTPUT_DIR, destino)
    print(f"{destino} · {paginas} páginas")
    if paginas > MAX_PAGINAS:
        raise SystemExit(f"El informe tiene {paginas} páginas (máximo {MAX_PAGINAS})")


if __name__ == "__main__":
    main()
