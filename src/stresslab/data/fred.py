"""Proveedor FRED. Único módulo (con yahoo.py) que toca la red junto a Yahoo."""
from __future__ import annotations

import os
from typing import Sequence

import pandas as pd
from fredapi import Fred

from stresslab import config

# Endpoint público de FRED que no requiere clave (se usa si falta FRED_API_KEY).
_PUBLIC_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={id}"


class FredRateProvider:
    """Series diarias de FRED en %. Clave solo por variable de entorno FRED_API_KEY."""

    def __init__(self) -> None:
        key = os.environ.get("FRED_API_KEY", "").strip()
        self._client = Fred(api_key=key) if key else None

    def _one(self, series_id: str, start: str, end: str | None) -> pd.Series:
        if self._client is not None:
            s = self._client.get_series(series_id, observation_start=start, observation_end=end)
        else:
            df = pd.read_csv(_PUBLIC_CSV.format(id=series_id), index_col=0, parse_dates=True,
                             na_values=["."])
            s = df.iloc[:, 0].loc[start:end]
        s.index = pd.to_datetime(s.index)
        return pd.to_numeric(s, errors="coerce").rename(series_id)

    def get_rates(self, series: Sequence[str], start: str = config.START_DATE,
                  end: str | None = None) -> pd.DataFrame:
        """Descarga cada serie y las alinea por fecha."""
        return pd.concat([self._one(s, start, end) for s in series], axis=1)
