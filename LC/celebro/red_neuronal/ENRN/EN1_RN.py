"""
NeuronaRefuerzoDQN - version expandida (400-450 lineas).
Modelo free OpenRouter. Subsistema HRCNTR.
"""
from __future__ import annotations

import importlib
import sys as _sys
_mod_orig = importlib.import_module('celebro\red_neuronal\RF_EN\RFEN1_RN_4.py')
NeuronaRefuerzoDQN = getattr(_mod_orig, 'NeuronaRefuerzoDQN', None)


"""RFEN1_RN_4.py - Neurona de Refuerzo con DQN (refactor ampliado y documentado).

Este módulo implementa una neurona de aprendizaje por refuerzo basada en
Deep Q‑Network (DQN) totalmente en NumPy.  La clase :class:`NeuronaRefuerzoDQN`
extiende la base mínima :class:`NeuronaRefuerzoBase` y añade:
    * inicialización de pesos con varios esquemas (He, Xavier, Lecun, ortogonal, espectral);
    * memoria de replay y muestreo por lotes;
    * red de política y red objetivo con actualización periódica;
    * política ε‑greedy con decaimiento;
    * rutinas de entrenamiento, validación de entradas y utilidades de guardado;
    * documentación extensa, ejemplos de uso y registro de eventos.

El código es totalmente ejecutable y se puede usar como bloque de construcción
para experimentos de refuerzo simples (por ejemplo, entornos tipo bandit o
CartPole simulado).  Todas las funciones incluyen comprobaciones de tipo y
rango, y se registran mediante el módulo estándar :mod:`logging`.
"""

# --------------------------------------------------------------------------- #
# Imports y configuración global
# --------------------------------------------------------------------------- #
import math
import time
import logging
from typing import Tuple, Optional, Dict, Any, List, Deque
from collections import deque
import numpy as np

# Configuración global de precisión y semilla aleatoria (reutilizada por los
# inicializadores y por la propia neurona).
LUCIA_RL_CONFIG = {
    'precision': 'float32',
    'random_seed': 42,
    'default_learning_rate': 0.001,
}

# --------------------------------------------------------------------------- #
# Clase base (minimalista) – evita importar un módulo .base inexistente
# --------------------------------------------------------------------------- #
class NeuronaRefuerzoBase:
    """Clase base mínima para todas las neuronas de refuerzo del proyecto.

    Sólo define la estructura común: tamaños de entrada/salida, nombre,
    pesos, sesgo e históricos de activaciones y gradientes.  Los métodos
    :meth:`inicializar_pesos` y :meth:`forward` deben ser implementados por
    las subclases concretas.
    """

    def __init__(self, input_size: int, output_size: int,
                 nombre: str = "NeuronaRefuerzo"):
        self.input_size: int = int(input_size)
        self.output_size: int = int(output_size)
        self.nombre: str = str(nombre)
        self.pesos: Optional[np.ndarray] = None
        self.sesgo: Optional[np.ndarray] = None
        self.historial_activaciones: List[np.ndarray] = []
        self.historial_gradientes: List[Any] = []

    def inicializar_pesos(self) -> None:
        """Inicializa los pesos y el sesgo.  Debe ser sobrescrito."""
        raise NotImplementedError

    def forward(self, e: np.ndarray) -> np.ndarray:
        """Propagación hacia adelante.  Debe ser sobrescrito."""
        raise NotImplementedError

    def resetear_historial(self) -> None:
        """Limpia los históricos de activaciones y gradientes."""
        self.historial_activaciones = []
        self.historial_gradientes = []


# --------------------------------------------------------------------------- #
# Inicializadores de pesos (He, Xavier, Lecun, ortogonal, espectral)
# --------------------------------------------------------------------------- #
class InicializadoresRL:
    """Colección de inicializadores usados frecuentemente en RL."""

    @staticmethod
    def _rng(seed: int = 42) -> np.random.Generator:
        """Generador de números aleatorios NumPy con semilla fija."""
        return np.random.default_rng(seed)

    @classmethod
    def he(cls, shape, fan_in=1, seed=42):
        """Inicialización He (recomendada para ReLU)."""
        return cls._rng(seed).normal(
            0.0,
            math.sqrt(2.0 / max(1, fan_in)),
            shape,
        ).astype(LUCIA_RL_CONFIG['precision'])

    @classmethod
    def xavier(cls, shape, fan_in=1, fan_out=1, seed=42):
        """Inicialización Xavier/Glorot (útil para tanh y sigmoid)."""
        lim = math.sqrt(6.0 / max(1, fan_in + fan_out))
        return cls._rng(seed).uniform(-lim, lim, shape).astype(
            LUCIA_RL_CONFIG['precision']
        )

    @classmethod
    def lecun(cls, shape, fan_in=1, seed=42):