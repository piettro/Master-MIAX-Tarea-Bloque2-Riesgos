"""Audita la caché data/raw y guarda outputs/auditoria_datos.csv (sin red)."""
from __future__ import annotations

from stresslab import config
from stresslab.data import audit
from stresslab.data.loader import load_market_data


def main() -> None:
    """Escribe y muestra la tabla de auditoría de precios."""
    tabla = audit.audit_prices(load_market_data(offline=True).prices)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(config.OUTPUT_DIR / "auditoria_datos.csv", index_label="activo")
    print(tabla.to_string())


if __name__ == "__main__":
    main()
