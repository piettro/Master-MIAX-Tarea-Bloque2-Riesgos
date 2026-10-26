"""Caché CSV en data/raw: lectura/escritura y proveedor offline."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from stresslab import config

PRICES_FILE = "prices.csv"  # Equities, HYG, ^GSPC y ^VIX (precios ajustados).
RATES_FILE = "rates.csv"  # DGS10, DGS2 y OAS (en %).


def read_csv(name: str, raw_dir: Path = config.RAW_DIR) -> pd.DataFrame:
    """Lee un CSV de la caché con índice de fechas."""
    path = raw_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Falta {path}; ejecuta scripts/fetch_data.py o scripts/migrate_bronze.py")
    return pd.read_csv(path, index_col=0, parse_dates=True).sort_index()


def update_csv(name: str, df: pd.DataFrame, raw_dir: Path = config.RAW_DIR) -> Path:
    """Fusiona df con la caché existente (lo nuevo prevalece) y la guarda."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / name
    merged = df.combine_first(read_csv(name, raw_dir)) if path.exists() else df
    merged.sort_index().to_csv(path, index_label="Date")
    return path


class CsvProvider:
    """PriceProvider y RateProvider sobre data/raw (cero red)."""

    def __init__(self, raw_dir: Path = config.RAW_DIR) -> None:
        self.raw_dir = raw_dir

    def _select(self, name: str, cols: Sequence[str], start: str, end: str | None) -> pd.DataFrame:
        df = read_csv(name, self.raw_dir)
        faltan = [c for c in cols if c not in df.columns]
        if faltan:
            raise KeyError(f"{name} no contiene {faltan}")
        return df.loc[start:end, list(cols)]

    def get_prices(self, tickers: Sequence[str], start: str = config.START_DATE,
                   end: str | None = None) -> pd.DataFrame:
        """Precios desde la caché."""
        return self._select(PRICES_FILE, tickers, start, end)

    def get_rates(self, series: Sequence[str], start: str = config.START_DATE,
                  end: str | None = None) -> pd.DataFrame:
        """Rendimientos (%) desde la caché."""
        return self._select(RATES_FILE, series, start, end)
