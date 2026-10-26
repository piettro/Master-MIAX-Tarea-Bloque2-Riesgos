"""Migra data/bronze/ a la caché data/raw/ (sin red). Descarta GS10/GS2 (mensuales)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from stresslab import config
from stresslab.data import csv_cache

BRONZE = config.ROOT / "data" / "bronze"


def _leer(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, parse_dates=True)


def main() -> None:
    """Une los CSV de precios y la serie OAS en prices.csv y rates.csv."""
    precios = [_leer(p) for p in sorted(BRONZE.glob("equities_adj_close_*.csv"))]
    prices = pd.concat(precios, axis=1).sort_index()
    rates = _leer(BRONZE / "us_yields_BAMLH0A0HYM2.csv").sort_index()
    csv_cache.update_csv(csv_cache.PRICES_FILE, prices)
    csv_cache.update_csv(csv_cache.RATES_FILE, rates)
    print(f"prices.csv: {prices.shape}, {prices.index[0].date()} -> {prices.index[-1].date()}")
    print(f"rates.csv: {rates.shape} (faltan DGS10/DGS2 diarios: scripts/fetch_data.py)")


if __name__ == "__main__":
    main()
