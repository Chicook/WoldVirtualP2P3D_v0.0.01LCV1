"""
RF_RFENRN1/__init__.py - Inicialización del Paquete de Neuronas de Refuerzo Avanzadas RFENRN1
============================================================================================

Este archivo inicializa el paquete RF_RFENRN1, que contiene implementaciones de neuronas
de refuerzo con técnicas avanzadas de optimización de pesos y algoritmos de refuerzo
especializados para 2025.

Define configuraciones globales, una clase base abstracta para las neuronas de refuerzo
y funciones de utilidad compartidas para optimización avanzada de pesos.

Autor: LucIA Development Team
Versión: 2.0.0
Fecha: 2025
"""

from .RF_RFEN1_RN_1_10 import NeuronaRefuerzoIMPALAAvanzada, crear_neurona_impala_avanzada
from .RF_RFEN1_RN_1_9 import NeuronaRefuerzoRainbowDQNAvanzada, crear_neurona_rainbow_dqn_avanzada
from .RF_RFEN1_RN_1_8 import NeuronaRefuerzoTD3Avanzada, crear_neurona_td3_avanzada
from .RF_RFEN1_RN_1_7 import NeuronaRefuerzoSACAvanzada, crear_neurona_sac_avanzada
from .RF_RFEN1_RN_1_6 import NeuronaRefuerzoPPOAvanzada, crear_neurona_ppo_avanzada
from .RF_RFEN1_RN_1_5 import NeuronaRefuerzoA3CAvanzada, crear_neurona_a3c_avanzada
from .RF_RFEN1_RN_1_4 import NeuronaRefuerzoDQNAvanzada, crear_neurona_dqn_avanzada
from .RF_RFEN1_RN_1_3 import NeuronaRefuerzoActorCriticAvanzada, crear_neurona_actor_critic_avanzada
from .RF_RFEN1_RN_1_2 import NeuronaRefuerzoPolicyGradientAvanzada, crear_neurona_policy_gradient_avanzada
from .RF_RFEN1_RN_1_1 import NeuronaRefuerzoQLearningAvanzada, crear_neurona_q_learning_avanzada
import numpy as np
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional
import time
from collections import deque

# Configuración global para las neuronas de refuerzo avanzadas RFENRN1
LUCIA_RL_CONFIG_RFENRN1 = {
    'precision': np.float32,
    'default_learning_rate': 0.001,
    'default_gamma': 0.99,
    'default_epsilon': 0.1,
    'default_buffer_size': 100000,
    'default_batch_size': 64,
    'default_target_update_frequency': 200,
    'default_entropy_coef': 0.001,
    'default_value_coef': 0.5,
    'default_clip_ratio': 0.2,
    'default_lambda_gae': 0.95,
    'default_momentum': 0.9,
    'default_beta1': 0.9,
    'default_beta2': 0.999,
    'default_epsilon_adam': 1e-8,
    'default_weight_decay': 1e-4,
    'default_dropout_rate': 0.1,
    'default_batch_norm_momentum': 0.9,
    'default_batch_norm_eps': 1e-5,
    'seed': 42
}

# Configuración del logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('RF_RFENRN1')


class NeuronaRefuerzoBaseRFENRN1(ABC):
    """
    Clase base abstracta para todas las neuronas de refuerzo en RF_RFENRN1.
    Define la interfaz común y funcionalidades básicas con técnicas avanzadas de 2025.
    """

    def __init__(self, input_size: int, output_size: int, nombre: str = "NeuronaBaseRFENRN1"):
        self.input_size = input_size
        self.output_size = output_size
        self.nombre = nombre
        self.pasos_totales = 0
        self.historial_recompensas = deque(maxlen=10000)
        self.historial_perdidas = deque(maxlen=10000)
        self.historial_pesos = deque(maxlen=1000)

        # Métricas avanzadas de optimización
        self.metricas_optimizacion = {
            'gradient_norm': 0.0,
            'weight_norm': 0.0,
            'learning_rate_actual': LUCIA_RL_CONFIG_RFENRN1['default_learning_rate'],
            'momentum_actual': LUCIA_RL_CONFIG_RFENRN1['default_momentum'],
            'convergence_rate': 0.0,
            'stability_score': 0.0,
            'efficiency_score': 0.0
        }

        # Historial de métricas avanzadas
        self.historial_gradientes = deque(maxlen=1000)
        self.historial_learning_rates = deque(maxlen=1000)
        self.historial_momentum = deque(maxlen=1000)

        # Evitar invocar __str__ del hijo antes de que inicialice sus propios atributos
        logger.info(
            "NeuronaRefuerzoBaseRFENRN1 creada: nombre=%s entrada=%s salida=%s",
            self.nombre,
            self.input_size,
            self.output_size,
        )

    @abstractmethod
    def inicializar_pesos(self) -> None:
        """Inicializa los pesos de la neurona con técnicas avanzadas."""
        pass

    @abstractmethod
    def forward(self, estado: np.ndarray, **kwargs) -> Any:
        """Realiza la propagación hacia adelante."""
        pass

    @abstractmethod
    def entrenar_paso(self, **kwargs) -> Optional[float]:
        """Realiza un paso de entrenamiento con optimización avanzada."""
        pass

    @abstractmethod
    def obtener_estadisticas(self) -> Dict[str, Any]:
        """Obtiene estadísticas de rendimiento de la neurona."""
        pass

    @abstractmethod
    def verificar_estabilidad(self) -> Dict[str, bool]:
        """Verifica la estabilidad de la neurona."""
        pass

    def resetear_historial(self) -> None:
        """Resetea el historial de la neurona."""
        self.historial_recompensas.clear()
        self.historial_perdidas.clear()
        self.historial_pesos.clear()
        self.historial_gradientes.clear()
        self.historial_learning_rates.clear()
        self.historial_momentum.clear()
        self.pasos_totales = 0
        logger.debug(f"Historial de {self.nombre} reseteado.")

    def __str__(self) -> str:
        return f"{self.nombre}(entrada={self.input_size}, salida={self.output_size})"

    def __repr__(self) -> str:
        return self.__str__()

# --- Funciones de Utilidad Avanzadas para Inicialización de Pesos ---


def inicializar_pesos_he_avanzado(shape: Tuple[int, ...], gain: float = np.sqrt(2.0)) -> np.ndarray:
    """
    Inicialización He avanzada con factor de ganancia configurable.
    Optimizada para activaciones ReLU y variantes.
    """
    fan_in = shape[0]
    std_dev = gain * np.sqrt(2.0 / fan_in)
    return np.random.normal(0, std_dev, shape).astype(LUCIA_RL_CONFIG_RFENRN1['precision'])


def inicializar_pesos_xavier_avanzado(shape: Tuple[int, ...], gain: float = 1.0) -> np.ndarray:
    """
    Inicialización Xavier/Glorot avanzada con factor de ganancia.
    Optimizada para activaciones sigmoide/tanh.
    """
    fan_in = shape[0]
    fan_out = shape[1] if len(shape) > 1 else shape[0]
    limit = gain * np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(low=-limit, high=limit, size=shape).astype(LUCIA_RL_CONFIG_RFENRN1['precision'])


def inicializar_pesos_ortogonal_avanzado(shape: Tuple[int, ...], gain: float = 1.0) -> np.ndarray:
    """
    Inicialización ortogonal avanzada con factor de ganancia.
    Mantiene la ortogonalidad para estabilidad.
    """
    if len(shape) < 2:
        raise ValueError("Shape must have at least 2 dimensions for orthogonal initialization.")

    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    return (gain * q.reshape(shape)).astype(LUCIA_RL_CONFIG_RFENRN1['precision'])


def inicializar_pesos_le_cun_avanzado(shape: Tuple[int, ...], gain: float = 1.0) -> np.ndarray:
    """
    Inicialización LeCun avanzada con factor de ganancia.
    Optimizada para activaciones SELU.
    """
    fan_in = shape[0]
    std_dev = gain * np.sqrt(1.0 / fan_in)
    return np.random.normal(0, std_dev, shape).astype(LUCIA_RL_CONFIG_RFENRN1['precision'])


def inicializar_pesos_kaiming_avanzado(shape: Tuple[int, ...], mode: str = 'fan_in',
                                       nonlinearity: str = 'relu') -> np.ndarray:
    """
    Inicialización Kaiming avanzada con modo y no linealidad configurables.
    """
    fan_in = shape[0]
    fan_out = shape[1] if len(shape) > 1 else shape[0]

    if mode == 'fan_in':
        fan = fan_in
    elif mode == 'fan_out':
        fan = fan_out
    else:
        fan = (fan_in + fan_out) / 2

    if nonlinearity == 'relu':
        gain = np.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        gain = np.sqrt(2.0 / (1 + 0.01**2))  # Para leaky_relu con alpha=0.01
    else:
        gain = 1.0

    std_dev = gain * np.sqrt(1.0 / fan)
    return np.random.normal(0, std_dev, shape).astype(LUCIA_RL_CONFIG_RFENRN1['precision'])

# --- Funciones de Utilidad Avanzadas para RL ---


def calcular_gae_avanzado(recompensas: List[float], valores_estado: List[float],
                          gamma: float, lambda_gae: float, siguiente_estado_valor: float = 0.0,
                          terminado: bool = False) -> np.ndarray:
    """
    Calcula las ventajas generalizadas (GAE) con optimizaciones avanzadas.
    """
    advantages = []
    last_advantage = 0

    for t in reversed(range(len(recompensas))):
        if t == len(recompensas) - 1:
            next_value = siguiente_estado_valor * (1 - int(terminado))
        else:
            next_value = valores_estado[t + 1]

        delta = recompensas[t] + gamma * next_value - valores_estado[t]
        advantage = delta + gamma * lambda_gae * last_advantage
        advantages.insert(0, advantage)
        last_advantage = advantage

    return np.array(advantages, dtype=LUCIA_RL_CONFIG_RFENRN1['precision'])


def aplicar_clip_gradientes_avanzado(gradientes: List[np.ndarray], max_norm: float = 0.5,
                                     method: str = 'global') -> List[np.ndarray]:
    """
    Aplica clipping avanzado a los gradientes con diferentes métodos.

    Args:
        gradientes: Lista de gradientes
        max_norm: Norma máxima permitida
        method: Método de clipping ('global', 'per_layer', 'adaptive')
    """
    if method == 'global':
        # Clipping global tradicional
        total_norm = np.sqrt(sum(np.sum(g**2) for g in gradientes))
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            return [g * clip_coef for g in gradientes]
        return gradientes

    elif method == 'per_layer':
        # Clipping por capa
        clipped_gradientes = []
        for g in gradientes:
            layer_norm = np.sqrt(np.sum(g**2))
            clip_coef = max_norm / (layer_norm + 1e-6)
            if clip_coef < 1:
                clipped_gradientes.append(g * clip_coef)
            else:
                clipped_gradientes.append(g)
        return clipped_gradientes

    elif method == 'adaptive':
        # Clipping adaptativo basado en historial
        if len(gradientes) == 0:
            return gradientes

        # Calcular norma promedio histórica (simplificado)
        current_norm = np.sqrt(sum(np.sum(g**2) for g in gradientes))
        adaptive_max_norm = max_norm * (1.0 + 0.1 * np.tanh(current_norm - max_norm))

        total_norm = np.sqrt(sum(np.sum(g**2) for g in gradientes))
        clip_coef = adaptive_max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            return [g * clip_coef for g in gradientes]
        return gradientes

    else:
        return gradientes


def calcular_momentum_avanzado(gradientes: List[np.ndarray], momentum_historial: List[np.ndarray],
                               momentum: float = 0.9, nesterov: bool = False) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Calcula momentum avanzado con opción Nesterov.

    Args:
        gradientes: Gradientes actuales
        momentum_historial: Historial de momentum
        momentum: Factor de momentum
        nesterov: Si usar momentum de Nesterov

    Returns:
        Tupla con gradientes con momentum y nuevo historial
    """
    if len(momentum_historial) == 0:
        # Inicializar historial
        momentum_historial = [np.zeros_like(g) for g in gradientes]

    gradientes_momentum = []
    nuevo_historial = []

    for i, (grad, hist) in enumerate(zip(gradientes, momentum_historial)):
        # Actualizar momentum
        nuevo_momentum = momentum * hist + grad

        if nesterov:
            # Momentum de Nesterov: mirar hacia adelante
            gradiente_momentum = grad + momentum * nuevo_momentum
        else:
            # Momentum estándar
            gradiente_momentum = nuevo_momentum

        gradientes_momentum.append(gradiente_momentum)
        nuevo_historial.append(nuevo_momentum)

    return gradientes_momentum, nuevo_historial


def aplicar_adam_avanzado(gradientes: List[np.ndarray], m_historial: List[np.ndarray],
                          v_historial: List[np.ndarray], learning_rate: float = 0.001,
                          beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8,
                          weight_decay: float = 0.0, amsgrad: bool = False) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Aplica optimización Adam avanzada con opciones adicionales.

    Args:
        gradientes: Gradientes actuales
        m_historial: Historial de primeros momentos
        v_historial: Historial de segundos momentos
        learning_rate: Tasa de aprendizaje
        beta1: Factor de decaimiento para primer momento
        beta2: Factor de decaimiento para segundo momento
        epsilon: Término de estabilidad
        weight_decay: Decaimiento de pesos
        amsgrad: Si usar AMSGrad

    Returns:
        Tupla con gradientes optimizados y nuevos historiales
    """
    if len(m_historial) == 0:
        m_historial = [np.zeros_like(g) for g in gradientes]
    if len(v_historial) == 0:
        v_historial = [np.zeros_like(g) for g in gradientes]

    gradientes_adam = []
    nuevo_m_historial = []
    nuevo_v_historial = []

    for i, (grad, m, v) in enumerate(zip(gradientes, m_historial, v_historial)):
        # Aplicar weight decay
        if weight_decay > 0:
            grad = grad + weight_decay * grad

        # Actualizar momentos
        m_new = beta1 * m + (1 - beta1) * grad
        v_new = beta2 * v + (1 - beta2) * (grad ** 2)

        # AMSGrad: mantener el máximo de v
        if amsgrad:
            v_new = np.maximum(v_new, v)

        # Calcular bias correction
        m_corrected = m_new / (1 - beta1)
        v_corrected = v_new / (1 - beta2)

        # Calcular gradiente optimizado
        gradiente_adam = learning_rate * m_corrected / (np.sqrt(v_corrected) + epsilon)

        gradientes_adam.append(gradiente_adam)
        nuevo_m_historial.append(m_new)
        nuevo_v_historial.append(v_new)

    return gradientes_adam, nuevo_m_historial, nuevo_v_historial


def aplicar_rmsprop_avanzado(gradientes: List[np.ndarray], v_historial: List[np.ndarray],
                             learning_rate: float = 0.001, alpha: float = 0.99,
                             epsilon: float = 1e-8, centered: bool = False) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Aplica optimización RMSprop avanzada con opción centrada.

    Args:
        gradientes: Gradientes actuales
        v_historial: Historial de segundos momentos
        learning_rate: Tasa de aprendizaje
        alpha: Factor de decaimiento
        epsilon: Término de estabilidad
        centered: Si usar RMSprop centrado

    Returns:
        Tupla con gradientes optimizados y nuevo historial
    """
    if len(v_historial) == 0:
        v_historial = [np.zeros_like(g) for g in gradientes]

    gradientes_rmsprop = []
    nuevo_v_historial = []

    for i, (grad, v) in enumerate(zip(gradientes, v_historial)):
        # Actualizar segundo momento
        v_new = alpha * v + (1 - alpha) * (grad ** 2)

        if centered:
            # RMSprop centrado: restar la media
            v_new = v_new - np.mean(grad) ** 2

        # Calcular gradiente optimizado
        gradiente_rmsprop = learning_rate * grad / (np.sqrt(v_new) + epsilon)

        gradientes_rmsprop.append(gradiente_rmsprop)
        nuevo_v_historial.append(v_new)

    return gradientes_rmsprop, nuevo_v_historial


def calcular_learning_rate_adaptativa(learning_rate_base: float, paso: int,
                                      schedule_type: str = 'cosine',
                                      warmup_steps: int = 1000,
                                      decay_steps: int = 10000) -> float:
    """
    Calcula una tasa de aprendizaje adaptativa según diferentes esquemas.

    Args:
        learning_rate_base: Tasa de aprendizaje base
        paso: Paso actual de entrenamiento
        schedule_type: Tipo de programación ('cosine', 'exponential', 'polynomial', 'step')
        warmup_steps: Pasos de calentamiento
        decay_steps: Pasos de decaimiento

    Returns:
        Tasa de aprendizaje adaptativa
    """
    if paso < warmup_steps:
        # Fase de calentamiento
        return learning_rate_base * paso / warmup_steps

    if schedule_type == 'cosine':
        # Decaimiento coseno
        progress = (paso - warmup_steps) / decay_steps
        progress = min(progress, 1.0)
        return learning_rate_base * 0.5 * (1 + np.cos(np.pi * progress))

    elif schedule_type == 'exponential':
        # Decaimiento exponencial
        decay_rate = 0.96
        decay_steps_actual = decay_steps // 10
        return learning_rate_base * (decay_rate ** (paso // decay_steps_actual))

    elif schedule_type == 'polynomial':
        # Decaimiento polinomial
        progress = (paso - warmup_steps) / decay_steps
        progress = min(progress, 1.0)
        power = 2.0
        return learning_rate_base * ((1 - progress) ** power)

    elif schedule_type == 'step':
        # Decaimiento por pasos
        step_size = decay_steps // 3
        decay_factor = 0.1
        return learning_rate_base * (decay_factor ** (paso // step_size))

    else:
        return learning_rate_base


def aplicar_batch_normalization_avanzada(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
                                         running_mean: np.ndarray, running_var: np.ndarray,
                                         momentum: float = 0.9, epsilon: float = 1e-5,
                                         training: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Aplica normalización por lotes avanzada.

    Args:
        x: Entrada a normalizar
        gamma: Parámetros de escala
        beta: Parámetros de sesgo
        running_mean: Media móvil
        running_var: Varianza móvil
        momentum: Factor de momentum para estadísticas móviles
        epsilon: Término de estabilidad
        training: Si está en modo entrenamiento

    Returns:
        Tupla con salida normalizada, nueva media móvil y nueva varianza móvil
    """
    if training:
        # Calcular estadísticas del lote
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)

        # Actualizar estadísticas móviles
        new_running_mean = momentum * running_mean + (1 - momentum) * batch_mean
        new_running_var = momentum * running_var + (1 - momentum) * batch_var

        # Normalizar usando estadísticas del lote
        x_norm = (x - batch_mean) / np.sqrt(batch_var + epsilon)
    else:
        # Usar estadísticas móviles
        x_norm = (x - running_mean) / np.sqrt(running_var + epsilon)
        new_running_mean = running_mean
        new_running_var = running_var

    # Aplicar escala y sesgo
    output = gamma * x_norm + beta

    return output, new_running_mean, new_running_var


def aplicar_dropout_avanzado(x: np.ndarray, dropout_rate: float = 0.1,
                             training: bool = True, noise_shape: Optional[Tuple] = None) -> np.ndarray:
    """
    Aplica dropout avanzado con forma de ruido configurable.

    Args:
        x: Entrada
        dropout_rate: Tasa de dropout
        training: Si está en modo entrenamiento
        noise_shape: Forma del ruido (opcional)

    Returns:
        Salida con dropout aplicado
    """
    if not training or dropout_rate == 0:
        return x

    if noise_shape is None:
        noise_shape = x.shape

    # Generar máscara de dropout
    dropout_mask = np.random.random(noise_shape) > dropout_rate

    # Aplicar dropout
    output = x * dropout_mask / (1 - dropout_rate)

    return output


# Importar todas las neuronas de refuerzo especializadas

logger.info("Paquete RF_RFENRN1 inicializado correctamente con configuraciones avanzadas de optimización de pesos.")
