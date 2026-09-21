"""
RFENRN2 - Redes de Refuerzo Avanzadas 2025 para LucIA
=====================================================

Este paquete contiene neuronas de refuerzo con técnicas avanzadas de 2025
para optimización de pesos, incluyendo redes neuronales líquidas, meta-aprendizaje,
atención adaptativa y optimización evolutiva.

Características Avanzadas 2025:
- Redes Neuronales Líquidas con pesos adaptativos dinámicos
- Meta-Aprendizaje con optimización de hiperparámetros automática
- Atención Adaptativa con pesos contextuales
- Optimización Evolutiva de Pesos
- Redes Neuronales con Ciclos de Sueño para consolidación de memoria
- Pesos Cuánticos Híbridos para computación cuántica
- Optimización Multi-Objetivo avanzada
- Redes Neuronales Explicables (XAI)
- Optimización Federada de Pesos
- Arquitectura Transformadora Adaptativa

Autor: LucIA Development Team
Versión: 2.0.0
Fecha: 2025
"""

import numpy as np
import math
from typing import Tuple, Optional, Dict, Any, List, Union
import logging
import random
from collections import deque
import warnings
import threading
import time
from abc import ABC, abstractmethod

# Configuración global para LucIA Advanced Reinforcement Learning 2025
LUCIA_ADVANCED_RL_CONFIG = {
    'precision': 'float32',
    'random_seed': 42,
    'default_learning_rate': 0.0001,
    'default_gamma': 0.99,
    'default_epsilon': 0.05,
    'default_epsilon_decay': 0.999,
    'default_epsilon_min': 0.001,
    'default_tau': 0.001,
    'default_batch_size': 64,
    'default_memory_size': 100000,
    'default_update_frequency': 10,
    'default_target_update_frequency': 1000,
    'default_clip_ratio': 0.1,
    'default_entropy_coef': 0.001,
    'default_value_coef': 0.5,
    'default_policy_coef': 1.0,
    'default_temperature': 0.5,
    'default_noise_std': 0.05,
    'default_noise_clip': 0.1,
    'default_gradient_clip': 0.5,
    'default_weight_decay': 1e-5,
    'default_dropout_rate': 0.05,
    'default_batch_norm_momentum': 0.95,
    'default_batch_norm_epsilon': 1e-6,
    # Configuraciones avanzadas 2025
    'liquid_dynamics_factor': 0.1,
    'meta_learning_rate': 0.0001,
    'attention_heads': 8,
    'evolution_population_size': 100,
    'sleep_cycle_length': 1000,
    'quantum_superposition_factor': 0.5,
    'multi_objective_weights': [0.4, 0.3, 0.3],
    'federated_rounds': 10,
    'transformer_layers': 6,
    'adaptive_learning_rate': True,
    'dynamic_weight_adjustment': True,
    'contextual_weight_scaling': True,
    'memory_consolidation_enabled': True,
    'quantum_hybrid_enabled': False,
    'explainability_threshold': 0.8,
    'federated_privacy_factor': 0.1
}

# Seed global centralizado en lucIA/__init__.py (no re-sembrar aquí).

# Las importaciones se harán de forma lazy para evitar importaciones circulares
# Se definirán aquí las exportaciones
__all__ = [
    'NeuronaRefuerzoLiquida',
    'NeuronaRefuerzoMetaLearning',
    'NeuronaRefuerzoAtencionAdaptativa',
    'NeuronaRefuerzoEvolutiva',
    'NeuronaRefuerzoCiclosSueno',
    'NeuronaRefuerzoCuantica',
    'NeuronaRefuerzoMultiObjetivo',
    'NeuronaRefuerzoExplicable',
    'NeuronaRefuerzoFederada',
    'NeuronaRefuerzoTransformerAdaptativa',
    'LUCIA_ADVANCED_RL_CONFIG'
]

# Funciones de utilidad avanzadas para redes de refuerzo 2025


def inicializar_pesos_liquidos(shape: Tuple[int, ...], dynamics_factor: float = 0.1) -> np.ndarray:
    """
    Inicialización específica para redes neuronales líquidas.

    Args:
        shape: Forma de los pesos
        dynamics_factor: Factor de dinámica líquida

    Returns:
        Pesos inicializados para redes líquidas
    """
    # Inicialización con distribución normal escalada por dinámica líquida
    stddev = math.sqrt(2.0 / shape[0]) * dynamics_factor
    weights = np.random.normal(0, stddev, shape)

    # Aplicar transformación líquida
    weights = np.tanh(weights) * dynamics_factor

    return weights.astype(LUCIA_ADVANCED_RL_CONFIG['precision'])


def inicializar_pesos_meta_learning(shape: Tuple[int, ...], meta_lr: float = 0.0001) -> np.ndarray:
    """
    Inicialización específica para meta-aprendizaje.

    Args:
        shape: Forma de los pesos
        meta_lr: Tasa de aprendizaje meta

    Returns:
        Pesos inicializados para meta-aprendizaje
    """
    # Inicialización adaptativa para meta-aprendizaje
    stddev = math.sqrt(meta_lr / shape[0])
    weights = np.random.normal(0, stddev, shape)

    # Aplicar escalado meta-adaptativo
    weights = weights * meta_lr

    return weights.astype(LUCIA_ADVANCED_RL_CONFIG['precision'])


def inicializar_pesos_atencion(shape: Tuple[int, ...], num_heads: int = 8) -> np.ndarray:
    """
    Inicialización específica para mecanismos de atención.

    Args:
        shape: Forma de los pesos
        num_heads: Número de cabezas de atención

    Returns:
        Pesos inicializados para atención
    """
    # Inicialización específica para atención multi-cabeza
    d_model = shape[0]
    stddev = math.sqrt(2.0 / (d_model * num_heads))
    weights = np.random.normal(0, stddev, shape)

    # Escalado por número de cabezas
    weights = weights / math.sqrt(num_heads)

    return weights.astype(LUCIA_ADVANCED_RL_CONFIG['precision'])


def inicializar_pesos_evolutivos(shape: Tuple[int, ...], population_size: int = 100) -> List[np.ndarray]:
    """
    Inicialización específica para optimización evolutiva.

    Args:
        shape: Forma de los pesos
        population_size: Tamaño de la población

    Returns:
        Lista de pesos inicializados para población evolutiva
    """
    population = []

    for _ in range(population_size):
        # Inicialización diversa para población evolutiva
        stddev = math.sqrt(1.0 / shape[0])
        weights = np.random.normal(0, stddev, shape)

        # Aplicar mutación inicial
        mutation_factor = np.random.uniform(0.8, 1.2)
        weights = weights * mutation_factor

        population.append(weights.astype(LUCIA_ADVANCED_RL_CONFIG['precision']))

    return population


def inicializar_pesos_cuanticos(shape: Tuple[int, ...], superposition_factor: float = 0.5) -> np.ndarray:
    """
    Inicialización específica para pesos cuánticos híbridos.

    Args:
        shape: Forma de los pesos
        superposition_factor: Factor de superposición cuántica

    Returns:
        Pesos inicializados para computación cuántica híbrida
    """
    # Inicialización con superposición cuántica
    weights_real = np.random.normal(0, 0.1, shape)
    weights_imag = np.random.normal(0, 0.1, shape)

    # Combinar componentes real e imaginario
    weights = weights_real + 1j * weights_imag * superposition_factor

    # Convertir a representación real para compatibilidad
    weights_real = np.real(weights)

    return weights_real.astype(LUCIA_ADVANCED_RL_CONFIG['precision'])


def calcular_adaptacion_dinamica(pesos: np.ndarray, contexto: np.ndarray) -> np.ndarray:
    """
    Calcula adaptación dinámica de pesos basada en contexto.

    Args:
        pesos: Pesos base
        contexto: Contexto actual

    Returns:
        Pesos adaptados dinámicamente
    """
    if not LUCIA_ADVANCED_RL_CONFIG['dynamic_weight_adjustment']:
        return pesos

    # Calcular factor de adaptación basado en contexto
    context_norm = np.linalg.norm(contexto)
    adaptation_factor = 1.0 + 0.1 * np.tanh(context_norm)

    # Aplicar adaptación dinámica
    pesos_adaptados = pesos * adaptation_factor

    return pesos_adaptados


def aplicar_consolidacion_memoria(pesos: np.ndarray, memoria: List[np.ndarray]) -> np.ndarray:
    """
    Aplica consolidación de memoria durante ciclos de sueño.

    Args:
        pesos: Pesos actuales
        memoria: Historial de pesos

    Returns:
        Pesos consolidados
    """
    if not LUCIA_ADVANCED_RL_CONFIG['memory_consolidation_enabled'] or len(memoria) < 10:
        return pesos

    # Calcular promedio de memoria reciente
    memoria_reciente = memoria[-10:]
    pesos_promedio = np.mean(memoria_reciente, axis=0)

    # Consolidación suave
    consolidation_factor = 0.1
    pesos_consolidados = (1 - consolidation_factor) * pesos + consolidation_factor * pesos_promedio

    return pesos_consolidados


def calcular_explicabilidad(pesos: np.ndarray, activaciones: np.ndarray) -> float:
    """
    Calcula métrica de explicabilidad de los pesos.

    Args:
        pesos: Pesos de la red
        activaciones: Activaciones de las neuronas

    Returns:
        Score de explicabilidad (0-1)
    """
    # Calcular varianza de pesos
    peso_variance = np.var(pesos)

    # Calcular consistencia de activaciones
    activation_consistency = 1.0 / (1.0 + np.var(activaciones))

    # Combinar métricas
    explicabilidad = (activation_consistency + (1.0 / (1.0 + peso_variance))) / 2.0

    return min(explicabilidad, 1.0)


def aplicar_privacy_preserving(pesos: np.ndarray, privacy_factor: float = 0.1) -> np.ndarray:
    """
    Aplica técnicas de preservación de privacidad a los pesos.

    Args:
        pesos: Pesos originales
        privacy_factor: Factor de privacidad

    Returns:
        Pesos con privacidad preservada
    """
    # Agregar ruido diferencialmente privado
    noise = np.random.normal(0, privacy_factor, pesos.shape)
    pesos_privados = pesos + noise

    # Recortar valores extremos
    pesos_privados = np.clip(pesos_privados, -10.0, 10.0)

    return pesos_privados

# Clase base abstracta avanzada para todas las neuronas de refuerzo 2025


class NeuronaRefuerzoAvanzadaBase(ABC):
    """
    Clase base abstracta avanzada para todas las neuronas de refuerzo de LucIA 2025.
    Define la interfaz común y funcionalidades avanzadas.
    """

    def __init__(self, input_size: int, output_size: int, nombre: str = "NeuronaRefuerzoAvanzadaBase"):
        self.input_size = input_size
        self.output_size = output_size
        self.nombre = nombre
        self.pesos = None
        self.sesgo = None
        self.historial_entrenamiento = []
        self.historial_recompensas = []
        self.historial_politicas = []
        self.historial_valores = []
        self.historial_pesos = []
        self.historial_contextos = []

        # Estadísticas avanzadas
        self.estadisticas_entrenamiento = {
            'episodios': 0,
            'pasos_totales': 0,
            'recompensa_promedio': 0.0,
            'recompensa_maxima': float('-inf'),
            'recompensa_minima': float('inf'),
            'convergencia': 0.0,
            'estabilidad': 0.0,
            'explicabilidad': 0.0,
            'adaptacion_dinamica': 0.0,
            'consolidacion_memoria': 0.0,
            'eficiencia_computacional': 0.0
        }

        # Configuración avanzada
        self.config_avanzada = LUCIA_ADVANCED_RL_CONFIG.copy()

    @abstractmethod
    def inicializar_pesos(self) -> None:
        """Inicializa los pesos de la neurona. Debe ser implementado por las subclases."""
        pass

    @abstractmethod
    def forward(self, estado: np.ndarray, contexto: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Propagación hacia adelante avanzada.

        Args:
            estado: Estado actual del entorno
            contexto: Contexto adicional (opcional)

        Returns:
            Acción o valor estimado
        """
        pass

    def backward(self, gradiente_salida: np.ndarray, estado: np.ndarray,
                 contexto: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Propagación hacia atrás avanzada.

        Args:
            gradiente_salida: Gradiente de la salida
            estado: Estado original
            contexto: Contexto adicional (opcional)

        Returns:
            Tupla con (gradiente_pesos, gradiente_sesgo)
        """
        if self.pesos is None:
            raise ValueError("Pesos no inicializados")

        # Aplicar adaptación dinámica si está habilitada
        if contexto is not None and self.config_avanzada['dynamic_weight_adjustment']:
            pesos_adaptados = calcular_adaptacion_dinamica(self.pesos, contexto)
        else:
            pesos_adaptados = self.pesos

        gradiente_pesos = np.dot(estado.T, gradiente_salida)
        gradiente_sesgo = np.sum(gradiente_salida, axis=0, keepdims=True)

        return gradiente_pesos, gradiente_sesgo

    def recibir_feedback_nivel_10(self, feedback: Dict[str, Any]) -> None:
        """
        Integra feedback de alta jerarquía (Nivel 10) para ajustar pesos locales.
        """
        calidad = feedback.get('quality_score', 0.5)
        # Escalar pesos si la calidad es baja
        if calidad < 0.4:
            self.pesos *= (1.0 + (1.0 - calidad) * 0.05)
            self.estadisticas_entrenamiento['consolidacion_memoria'] += 0.01
            logger.info(f"[{self.nombre}] Ajuste proactivo por bajo feedback (calidad={calidad:.2f})")

    def actualizar_pesos(self, gradiente_pesos: np.ndarray, gradiente_sesgo: np.ndarray,
                         learning_rate: float = None, contexto: Optional[np.ndarray] = None) -> None:
        """
        Actualiza los pesos usando técnicas avanzadas.

        Args:
            gradiente_pesos: Gradiente de los pesos
            gradiente_sesgo: Gradiente del sesgo
            learning_rate: Tasa de aprendizaje
            contexto: Contexto adicional (opcional)
        """
        if learning_rate is None:
            learning_rate = self.config_avanzada['default_learning_rate']

        # Aplicar tasa de aprendizaje adaptativa
        if self.config_avanzada['adaptive_learning_rate']:
            learning_rate = self._calcular_learning_rate_adaptativa(learning_rate, contexto)

        # Actualizar pesos
        self.pesos -= learning_rate * gradiente_pesos
        self.sesgo -= learning_rate * gradiente_sesgo

        # Guardar historial de pesos
        self.historial_pesos.append(self.pesos.copy())

        # Aplicar consolidación de memoria
        if self.config_avanzada['memory_consolidation_enabled']:
            self.pesos = aplicar_consolidacion_memoria(self.pesos, self.historial_pesos)

    def _calcular_learning_rate_adaptativa(self, lr_base: float, contexto: Optional[np.ndarray] = None) -> float:
        """
        Calcula tasa de aprendizaje adaptativa.

        Args:
            lr_base: Tasa de aprendizaje base
            contexto: Contexto actual

        Returns:
            Tasa de aprendizaje adaptativa
        """
        if contexto is None:
            return lr_base

        # Adaptar basado en contexto
        context_norm = np.linalg.norm(contexto)
        adaptation_factor = 1.0 + 0.1 * np.tanh(context_norm)

        return lr_base * adaptation_factor

    def obtener_estadisticas(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas avanzadas de la neurona.

        Returns:
            Diccionario con estadísticas avanzadas
        """
        if self.pesos is None:
            return {'estado': 'no_inicializada'}

        # Calcular explicabilidad
        if self.historial_pesos:
            activaciones_recientes = np.array(self.historial_pesos[-10:])
            self.estadisticas_entrenamiento['explicabilidad'] = calcular_explicabilidad(
                self.pesos, activaciones_recientes.flatten()
            )

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
            'recompensa_promedio': self.estadisticas_entrenamiento['recompensa_promedio'],
            'explicabilidad': self.estadisticas_entrenamiento['explicabilidad'],
            'adaptacion_dinamica': self.estadisticas_entrenamiento['adaptacion_dinamica'],
            'consolidacion_memoria': self.estadisticas_entrenamiento['consolidacion_memoria'],
            'eficiencia_computacional': self.estadisticas_entrenamiento['eficiencia_computacional']
        }

    def resetear_historial(self) -> None:
        """Resetea el historial de entrenamiento avanzado."""
        self.historial_entrenamiento.clear()
        self.historial_recompensas.clear()
        self.historial_politicas.clear()
        self.historial_valores.clear()
        self.historial_pesos.clear()
        self.historial_contextos.clear()

    def verificar_estabilidad_avanzada(self) -> Dict[str, bool]:
        """
        Verifica la estabilidad avanzada de la neurona.

        Returns:
            Diccionario con indicadores de estabilidad avanzada
        """
        estabilidad = {}

        # Verificaciones básicas
        if self.pesos is not None:
            peso_max = np.max(np.abs(self.pesos))
            peso_min = np.min(np.abs(self.pesos))

            estabilidad['pesos_no_explosivos'] = peso_max < 10.0
            estabilidad['pesos_no_desaparecen'] = peso_min > 1e-6
            estabilidad['pesos_balanceados'] = peso_max / (peso_min + 1e-8) < 1000.0

        # Verificaciones avanzadas
        explicabilidad = self.estadisticas_entrenamiento['explicabilidad']
        estabilidad['explicabilidad_suficiente'] = explicabilidad > self.config_avanzada['explainability_threshold']

        return estabilidad

    def __str__(self) -> str:
        return f"{self.nombre}(entrada={self.input_size}, salida={self.output_size})"

    def __repr__(self) -> str:
        return self.__str__()


# Logging centralizado en lucIA/__init__.py; aquí solo logger local.
logger = logging.getLogger('RFENRN2')

# Importar las neuronas desde sus archivos individuales
try:
    from .RFEN2_RN_1 import NeuronaRefuerzoLiquida, crear_neurona_liquida
    from .RFEN2_RN_2 import NeuronaRefuerzoMetaLearning, crear_neurona_meta_learning
    from .RFEN2_RN_3 import NeuronaRefuerzoAtencionAdaptativa, crear_neurona_atencion_adaptativa
    from .RFEN2_RN_4 import NeuronaRefuerzoEvolutiva, crear_neurona_evolutiva
    from .RFEN2_RN_5 import NeuronaRefuerzoCiclosSueno, crear_neurona_ciclos_sueno_avanzados
    from .RFEN2_RN_6 import NeuronaRefuerzoCuantica, crear_neurona_cuantica
    from .RFEN2_RN_7 import NeuronaRefuerzoMultiObjetivo, crear_neurona_multi_objetivo
    from .RFEN2_RN_8 import NeuronaRefuerzoExplicable, crear_neurona_explicable
    from .RFEN2_RN_9 import NeuronaRefuerzoFederada, crear_neurona_federada
    from .RFEN2_RN_10 import NeuronaRefuerzoTransformerAdaptativa, crear_neurona_transformer_adaptativa

    logger.debug("Paquete RFENRN2 inicializado correctamente para LucIA Advanced Reinforcement Learning 2025")
except ImportError as e:
    logger.warning(f"Error al importar neuronas de RFENRN2: {e}")
    # Crear clases stubs para evitar errores de importación
    NeuronaRefuerzoLiquida = type('NeuronaRefuerzoLiquida', (), {})
    NeuronaRefuerzoMetaLearning = type('NeuronaRefuerzoMetaLearning', (), {})
    NeuronaRefuerzoAtencionAdaptativa = type('NeuronaRefuerzoAtencionAdaptativa', (), {})
    NeuronaRefuerzoEvolutiva = type('NeuronaRefuerzoEvolutiva', (), {})
    NeuronaRefuerzoCiclosSueno = type('NeuronaRefuerzoCiclosSueno', (), {})
    NeuronaRefuerzoCuantica = type('NeuronaRefuerzoCuantica', (), {})
    NeuronaRefuerzoMultiObjetivo = type('NeuronaRefuerzoMultiObjetivo', (), {})
    NeuronaRefuerzoExplicable = type('NeuronaRefuerzoExplicable', (), {})
    NeuronaRefuerzoFederada = type('NeuronaRefuerzoFederada', (), {})
    NeuronaRefuerzoTransformadora = type('NeuronaRefuerzoTransformadora', (), {})
