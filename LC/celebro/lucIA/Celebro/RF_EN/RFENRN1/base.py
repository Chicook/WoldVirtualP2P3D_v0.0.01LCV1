"""
base.py - Clases Base y Funciones de Utilidad para Neuronas de Refuerzo
========================================================================

Este módulo contiene las clases base y funciones de utilidad que son compartidas
por todas las neuronas de refuerzo del sistema LucIA.

Autor: LucIA Development Team
Versión: 1.0.0
"""

import numpy as np
import math
from typing import Tuple, Optional, Dict, Any, List
import logging
from collections import deque

# Configuración global para LucIA Reinforcement Learning
LUCIA_RL_CONFIG = {
    'precision': 'float32',
    'random_seed': 42,
    'default_learning_rate': 0.001,
    'default_gamma': 0.995,
    'default_epsilon': 0.3,
    'default_epsilon_decay': 0.98,
    'default_epsilon_min': 0.01,
    'default_tau': 0.005,
    'default_batch_size': 32,
    'default_memory_size': 10000
}

# Clase base abstracta para todas las neuronas de refuerzo


class NeuronaRefuerzoBase:
    """
    Clase base abstracta para todas las neuronas de refuerzo de LucIA.
    Define la interfaz común y funcionalidades básicas.
    """

    def __init__(self, input_size: int, output_size: int, nombre: str = "NeuronaRefuerzoBase"):
        self.input_size = input_size
        self.output_size = output_size
        self.nombre = nombre
        self.pesos = None
        self.sesgo = None
        self.historial_entrenamiento = []
        self.historial_recompensas = []
        self.historial_politicas = []
        self.historial_valores = []
        self.estadisticas_entrenamiento = {
            'episodios': 0,
            'pasos_totales': 0,
            'recompensa_promedio': 0.0,
            'recompensa_maxima': float('-inf'),
            'recompensa_minima': float('inf'),
            'convergencia': 0.0,
            'estabilidad': 0.0
        }

    def inicializar_pesos(self) -> None:
        """Inicializa los pesos de la neurona. Debe ser implementado por las subclases."""
        raise NotImplementedError("Subclases deben implementar inicializar_pesos()")

    def forward(self, estado: np.ndarray) -> np.ndarray:
        """
        Propagación hacia adelante.

        Args:
            estado: Estado actual del entorno

        Returns:
            Acción o valor estimado
        """
        if self.pesos is None:
            self.inicializar_pesos()

        suma_ponderada = np.dot(estado, self.pesos) + self.sesgo
        return suma_ponderada

    def backward(self, gradiente_salida: np.ndarray, estado: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Propagación hacia atrás.

        Args:
            gradiente_salida: Gradiente de la salida
            estado: Estado original

        Returns:
            Tupla con (gradiente_pesos, gradiente_sesgo)
        """
        if self.pesos is None:
            raise ValueError("Pesos no inicializados")

        gradiente_pesos = np.dot(estado.T, gradiente_salida)
        gradiente_sesgo = np.sum(gradiente_salida, axis=0, keepdims=True)

        return gradiente_pesos, gradiente_sesgo

    def actualizar_pesos(self, gradiente_pesos: np.ndarray, gradiente_sesgo: np.ndarray,
                         learning_rate: float = None) -> None:
        """
        Actualiza los pesos usando descenso de gradiente.

        Args:
            gradiente_pesos: Gradiente de los pesos
            gradiente_sesgo: Gradiente del sesgo
            learning_rate: Tasa de aprendizaje
        """
        if learning_rate is None:
            learning_rate = LUCIA_RL_CONFIG['default_learning_rate']

        self.pesos -= learning_rate * gradiente_pesos
        self.sesgo -= learning_rate * gradiente_sesgo

    def obtener_estadisticas(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas de la neurona.

        Returns:
            Diccionario con estadísticas
        """
        if self.pesos is None:
            return {'estado': 'no_inicializada'}

        return {
            'nombre': self.nombre,
            'forma_pesos': self.pesos.shape,
            'media_pesos': np.mean(self.pesos),
            'std_pesos': np.std(self.pesos),
            'min_peso': np.min(self.pesos),
            'max_peso': np.max(self.pesos),
            'media_sesgo': np.mean(self.sesgo),
            'std_sesgo': np.std(self.sesgo),
            'episodios': self.estadisticas_entrenamiento['episodios'],
            'pasos_totales': self.estadisticas_entrenamiento['pasos_totales'],
            'recompensa_promedio': self.estadisticas_entrenamiento['recompensa_promedio']
        }

    def resetear_historial(self) -> None:
        """Resetea el historial de entrenamiento."""
        self.historial_entrenamiento.clear()
        self.historial_recompensas.clear()
        self.historial_politicas.clear()
        self.historial_valores.clear()

    def __str__(self) -> str:
        return f"{self.nombre}(entrada={self.input_size}, salida={self.output_size})"

    def __repr__(self) -> str:
        return self.__str__()


# Funciones de utilidad para inicialización de pesos
def inicializar_pesos_he(shape: Tuple[int, ...], fan_in: Optional[int] = None) -> np.ndarray:
    """Inicialización de He para redes con activación ReLU."""
    if fan_in is None:
        fan_in = shape[0]
    stddev = math.sqrt(2.0 / fan_in)
    return np.random.normal(0, stddev, shape).astype(LUCIA_RL_CONFIG['precision'])


def inicializar_pesos_xavier(shape: Tuple[int, ...], fan_in: Optional[int] = None,
                             fan_out: Optional[int] = None) -> np.ndarray:
    """Inicialización de Xavier/Glorot."""
    if fan_in is None:
        fan_in = shape[0]
    if fan_out is None:
        fan_out = shape[1] if len(shape) > 1 else shape[0]

    limit = math.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, shape).astype(LUCIA_RL_CONFIG['precision'])


def inicializar_pesos_lecun(shape: Tuple[int, ...], fan_in: Optional[int] = None) -> np.ndarray:
    """Inicialización de LeCun para redes con activación tanh."""
    if fan_in is None:
        fan_in = shape[0]
    stddev = math.sqrt(1.0 / fan_in)
    return np.random.normal(0, stddev, shape).astype(LUCIA_RL_CONFIG['precision'])


def inicializar_pesos_ortogonal(shape: Tuple[int, ...]) -> np.ndarray:
    """Inicialización ortogonal para redes de refuerzo."""
    if len(shape) == 2:
        a = np.random.randn(shape[0], shape[1])
        u, _, v = np.linalg.svd(a, full_matrices=False)
        return u.astype(LUCIA_RL_CONFIG['precision'])
    else:
        flat_shape = (shape[0], np.prod(shape[1:]))
        a = np.random.randn(*flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        return u.reshape(shape).astype(LUCIA_RL_CONFIG['precision'])


def inicializar_pesos_espectral(shape: Tuple[int, ...], fan_in: Optional[int] = None) -> np.ndarray:
    """Inicialización espectral para redes de refuerzo."""
    if fan_in is None:
        fan_in = shape[0]

    stddev = math.sqrt(1.0 / fan_in)
    weights = np.random.normal(0, stddev, shape)

    if len(shape) == 2:
        u, s, v = np.linalg.svd(weights, full_matrices=False)
        s = np.clip(s, 0, 1.0)
        weights = u @ np.diag(s) @ v

    return weights.astype(LUCIA_RL_CONFIG['precision'])


def calcular_recompensa_descontada(recompensas: List[float], gamma: float = 0.99) -> List[float]:
    """Calcula la recompensa descontada."""
    recompensas_descontadas = []
    recompensa_acumulada = 0

    for recompensa in reversed(recompensas):
        recompensa_acumulada = recompensa + gamma * recompensa_acumulada
        recompensas_descontadas.insert(0, recompensa_acumulada)

    return recompensas_descontadas


def normalizar_recompensas(recompensas: List[float], epsilon: float = 1e-8) -> List[float]:
    """Normaliza las recompensas."""
    recompensas_array = np.array(recompensas)
    media = np.mean(recompensas_array)
    std = np.std(recompensas_array)

    if std < epsilon:
        return recompensas

    return ((recompensas_array - media) / (std + epsilon)).tolist()


def calcular_gae(recompensas: List[float], valores: List[float],
                 gamma: float = 0.99, lambda_gae: float = 0.95) -> List[float]:
    """Calcula el Advantage Estimator Generalizado (GAE)."""
    advantages = []
    advantage = 0

    for t in reversed(range(len(recompensas))):
        if t == len(recompensas) - 1:
            next_value = 0
        else:
            next_value = valores[t + 1]

        delta = recompensas[t] + gamma * next_value - valores[t]
        advantage = delta + gamma * lambda_gae * advantage
        advantages.insert(0, advantage)

    return advantages


def aplicar_clip_gradientes(gradientes: List[np.ndarray], max_norm: float = 1.0) -> List[np.ndarray]:
    """Aplica recorte de gradientes."""
    total_norm = 0
    for grad in gradientes:
        total_norm += np.sum(grad ** 2)
    total_norm = math.sqrt(total_norm)

    if total_norm > max_norm:
        clip_coef = max_norm / (total_norm + 1e-8)
        gradientes = [grad * clip_coef for grad in gradientes]

    return gradientes


def crear_buffer_experiencia(tamaño: int = 10000) -> deque:
    """Crea un buffer de experiencia."""
    return deque(maxlen=tamaño)


def muestrear_buffer(buffer: deque, tamaño_muestra: int) -> List[Dict[str, Any]]:
    """Muestrea un batch del buffer de experiencia."""
    if len(buffer) < tamaño_muestra:
        return list(buffer)

    import random
    return random.sample(list(buffer), tamaño_muestra)
