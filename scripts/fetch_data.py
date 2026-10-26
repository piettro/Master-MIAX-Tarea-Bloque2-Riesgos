"""Descarga precios (Yahoo) y tipos (FRED) a data/raw. Es el ÚNICO paso con red; ejecutar a mano."""
from __future__ import annotations

from stresslab import config
from stresslab.data.loader import load_market_data


def main() -> None:
    """Fuerza modo online; si una fuente falla, avisa y conserva la caché."""
    datos = load_market_data(offline=False)
    for aviso in datos.warnings:
        print("AVISO:", aviso)
    print("prices:", datos.prices.shape, "rates:", datos.rates.shape, "desde", config.START_DATE)


if __name__ == "__main__":
    main()
