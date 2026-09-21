"""
lucIA.CORE.initializers - Algoritmos de inicialización de pesos neuronales
"""

import numpy as np
from typing import Tuple, Union, Optional


def he_initialization(shape: Union[Tuple[int, ...], int] = (1, 1), fan_in: Optional[int] = None) -> np.ndarray:
    """
    Inicialización He (Kaiming), óptima para activaciones ReLU y derivadas.
    """
    if isinstance(shape, int):
        shape = (shape,)
    if fan_in is None:
        fan_in = shape[0] if len(shape) > 0 else 1
    std = np.sqrt(2.0 / max(1, fan_in))
    return np.random.randn(*shape).astype(np.float32) * std


def xavier_initialization(shape: Union[Tuple[int, ...], int] = (1, 1),
                          fan_in: Optional[int] = None,
                          fan_out: Optional[int] = None) -> np.ndarray:
    """
    Inicialización Xavier (Glorot), óptima para Sigmoid y Tanh.
    """
    if isinstance(shape, int):
        shape = (shape,)
    if fan_in is None:
        fan_in = shape[0] if len(shape) > 0 else 1
    if fan_out is None:
        fan_out = shape[1] if len(shape) > 1 else shape[0]
    std = np.sqrt(2.0 / max(1, (fan_in + fan_out)))
    return np.random.randn(*shape).astype(np.float32) * std


def lecun_initialization(shape: Union[Tuple[int, ...], int] = (1, 1), fan_in: Optional[int] = None) -> np.ndarray:
    """
    Inicialización LeCun, óptima para funciones SELU y autoencoders.
    """
    if isinstance(shape, int):
        shape = (shape,)
    if fan_in is None:
        fan_in = shape[0] if len(shape) > 0 else 1
    std = np.sqrt(1.0 / max(1, fan_in))
    return np.random.randn(*shape).astype(np.float32) * std
