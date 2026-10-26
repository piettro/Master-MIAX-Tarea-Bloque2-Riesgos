"""Descarga solo DGS10 y DGS2 de FRED y los fusiona en data/raw/rates.csv (paso con red, manual)."""
from __future__ import annotations

from stresslab import config
from stresslab.data import csv_cache
from stresslab.data.fred import FredRateProvider


def main() -> None:
    """Baja los rendimientos diarios 10a/2a y actualiza la caché."""
    series = list(config.FRED_SERIES.values())
    df = FredRateProvider().get_rates(series, config.START_DATE)
    csv_cache.update_csv(csv_cache.RATES_FILE, df)
    print(df.notna().sum().to_string(), "\nDesde", df.index[0].date(), "hasta", df.index[-1].date())


if __name__ == "__main__":
    main()
