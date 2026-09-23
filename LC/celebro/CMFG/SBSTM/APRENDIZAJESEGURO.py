"""Aprendizaje seguro sobre los pesos propios de LucIA.

La fuente puede ser Ollama, LM Studio u OpenRouter. Este módulo nunca intenta
editar los pesos del proveedor externo: solo protege y actualiza PSNRCV.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[4]
PSNRL_DIR = ROOT_DIR / "LC" / "celebro" / "PSNRL"
SNAPSHOT = PSNRL_DIR / "lucia_pesos_previos.npz"
SNAPSHOT_META = PSNRL_DIR / "lucia_pesos_previos.json"
REPLAY = PSNRL_DIR / "experiencias_lucia.jsonl"
WEIGHT_ATTRS = (
    "pesos", "sesgo", "q_online", "q_objetivo", "pesos_actor",
    "pesos_critic", "q_table", "pesos_actor_global", "pesos_q_principal",
)

SPECIALTY_KEYWORDS = {
    "EN": {"pregunta", "entrada", "texto", "percepcion", "sensacion", "contexto"},
    "RFSL": {"explica", "razon", "codigo", "estructura", "analisis", "aprendizaje"},
    "RFEN": {"decide", "accion", "objetivo", "plan", "prioridad", "recompensa"},
    "RNP": {"memoria", "peso", "neural", "optimizacion", "sinapsis", "patron"},
    "SLRN": {"responde", "sintesis", "generaliza", "regla", "solucion", "lenguaje"},
}


_APRENDIZAJE: Optional[AprendizajeSeguroLucIA] = None


def get_aprendizaje_seguro() -> AprendizajeSeguroLucIA:
    global _APRENDIZAJE
    if _APRENDIZAJE is None:
        _APRENDIZAJE = AprendizajeSeguroLucIA()
    return _APRENDIZAJE
from APRENDIZAJESEGURO_AprendizajeSeguroLucIA import AprendizajeSeguroLucIA  # CLASSPACK
