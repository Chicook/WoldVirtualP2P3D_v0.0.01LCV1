"""Paquete red neuronal (50 neuronas en 5 familias)."""
from __future__ import annotations

from typing import Any, Dict, Final

__all__ = ["FAMILIAS"]

FAMILIAS: Final[Dict[str, str]] = {
    "ENRN": "Entrada recurrente no lineal (10)",
    "RF_SL": "Aprendizaje supervisado residual (10)",
    "RF_EN": "Retroalimentacion y refuerzo (10)",
    "RNP": "Plasticidad sinaptica (10)",
    "SLRN": "Optimizadores supervisados (10)",
}
