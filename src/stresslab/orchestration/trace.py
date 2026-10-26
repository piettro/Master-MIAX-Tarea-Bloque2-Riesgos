"""Traza del pipeline: (paso, método, n_obs, seed, segundos, nota) y results.json."""
from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from stresslab import config


@dataclass
class TraceStep:
    """Un paso registrado del pipeline."""

    paso: str
    metodo: str
    n_obs: int
    seed: int
    segundos: float
    nota: str = ""


@dataclass
class Trace:
    """Colección de pasos; se muestra en el notebook y se guarda en results.json."""

    steps: list[TraceStep] = field(default_factory=list)

    @contextmanager
    def step(self, paso: str, metodo: str, n_obs: int = 0, seed: int = config.SEED) -> Iterator[TraceStep]:
        """Mide un paso; el cuerpo puede completar n_obs y nota sobre el objeto devuelto."""
        registro = TraceStep(paso, metodo, n_obs, seed, 0.0)
        t0 = time.perf_counter()
        try:
            yield registro
        finally:
            registro.segundos = round(time.perf_counter() - t0, 4)
            self.steps.append(registro)

    def to_frame(self) -> pd.DataFrame:
        """Traza como tabla."""
        return pd.DataFrame([asdict(s) for s in self.steps])


def write_results(results: dict[str, Any], trace: Trace, path: Path | None = None) -> Path:
    """Escribe outputs/results.json (única fuente de cifras del informe)."""
    path = path or config.OUTPUT_DIR / "results.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"seed": config.SEED, "trace": [asdict(s) for s in trace.steps], **results}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return path
