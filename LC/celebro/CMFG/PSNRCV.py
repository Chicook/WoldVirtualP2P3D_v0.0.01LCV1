"""
PSNRCV.py - Conversor de Respuestas y Consultas a Pesos Neuronales (Arquitectura 2026)
=====================================================================================
Orquesta las 50 neuronas de los 5 subsistemas de WoldVirtualP2P3D_v0.0.01LCV1_DEVPYMD:
1. ENRN (10 neuronas): Percepcion y entrada sensorial (Basica a Sigmoid).
2. RF_SL (10 neuronas): Memoria y optimizacion supervisada (GBM, SVM, RF, NN, DT, NB, KNN, etc.).
3. RF_EN (10 neuronas): Aprendizaje por refuerzo (QLearning, PG, AC, DQN, A3C, PPO, SAC, TD3, etc.).
4. RNP (10 neuronas): Optimizacion de pesos neuronales (AdamW, RAdam, Lookahead, SAM, SWATS, etc.).
5. SLRN (10 neuronas): Optimizadores supervisados 2026 (Backpropagation, SGD, RMSprop, Adam, etc.).
6. PSNRL: Persistencia activa en tiempo real de pesos durante la sesion.
"""
from __future__ import annotations
import os, sys, json, time, math, logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np

CURRENT_FILE = Path(__file__).resolve()
CMFG_DIR = CURRENT_FILE.parent
CELEBRO_DIR = CMFG_DIR.parent
ROOT_DIR = CELEBRO_DIR.parent.parent
PSNRL_DIR = CELEBRO_DIR / "PSNRL"
PSNRL_DIR.mkdir(parents=True, exist_ok=True)

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# 1. ENRN (10 neuronas)
import LC.celebro.red_neuronal.ENRN as enrn_pkg

# 2. RF_SL (10 neuronas)
import LC.celebro.red_neuronal.RF_SL.RFSL1 as rfsl_pkg

# 3. RF_EN (10 neuronas)
import LC.celebro.red_neuronal.RF_EN as rfen_pkg

# 4. RNP (10 neuronas)
import LC.celebro.red_neuronal.RNP as rnp_pkg

# 5. SLRN (10 neuronas)
import LC.celebro.red_neuronal.SLRN as slrn_pkg
from LC.celebro.red_neuronal.SLRN import SupervisedLearningNeuralConfig

logger = logging.getLogger("WoldVirtualP2P3D.PSNRCV")

from LC.celebro.CMFG.neural_math import (
    NeuralMathPrecision2026,
    SemanticEncoderSHA,
    SemanticEncoder4D,
)


# ===========================================================================
# CLASE PRINCIPAL: CONVERSOR RESPUESTA PESOS (PSNRCV) - 50 NEURONAS
# ===========================================================================
_conversor_global: Optional[ConversorRespuestaPesos] = None


def get_conversor_pesos(learning_rate: float = 0.01) -> ConversorRespuestaPesos:
    """Retorna la instancia global del conversor neuronal de pesos."""
    global _conversor_global
    if _conversor_global is None:
        _conversor_global = ConversorRespuestaPesos(learning_rate=learning_rate)
    return _conversor_global


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Iniciando prueba unitaria de PSNRCV 2026 con 50 neuronas...")
    conversor = get_conversor_pesos()
    res1 = conversor.procesar_consulta_a_pesos("Hola LucIA, ¿cómo funciona tu red neuronal completa?")
    print("Consulta procesada. Delta aplicada:", res1["norma_delta_aplicada"])
    res2 = conversor.asimilar_respuestas_y_calcular_sintesis(
        "Hola LucIA", "Las 50 neuronas en ENRN, RF_SL, RF_EN, RNP y SLRN están sincronizadas.", "qwen2.5:7b"
    )
    print("Respuesta asimilada. Deriva acumulada:", res2["deriva_acumulada"])
    archivos = list(PSNRL_DIR.glob("*.*"))
    print(f"Archivos verificados en PSNRL ({len(archivos)}): {[f.name for f in archivos]}")
from LC.celebro.CMFG.PSNRCV_ConversorRespuestaPesos import ConversorRespuestaPesos  # CLASSPACK
