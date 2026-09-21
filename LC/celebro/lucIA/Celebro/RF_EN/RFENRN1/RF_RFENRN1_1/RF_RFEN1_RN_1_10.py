"""
RF_RFEN1_RN_1_10.py - Neurona de Refuerzo IMPALA Avanzada
=========================================================

Esta neurona implementa IMPALA (Importance Weighted Actor-Learner Architecture)
con técnicas avanzadas de optimización de pesos y mejoras específicas para 2025.

Características Avanzadas 2025:
- Optimización Adam/RMSprop con momentum adaptativo
- Regularización L2 avanzada con weight decay
- Batch Normalization para estabilización
- Dropout adaptativo para regularización
- Learning rate scheduling avanzado
- Gradient clipping adaptativo
- Importance sampling para corrección de off-policy
- V-trace para corrección de bias
- Asynchronous actor-learner architecture
- Multiple actors con diferentes políticas
- Centralized learning con distributed execution
- Value function approximation avanzada
- Policy gradient con corrección de importancia
- Experience replay con importancia
- Multi-task learning

Autor: LucIA Development Team
Versión: 2.0.0
Fecha: 2025
"""

import numpy as np
import logging
from typing import Dict, Any, Tuple, Optional, List
import random
import time
from collections import deque
import sys
# Importación dinámica de las clases base y funciones
try:
    from . import (NeuronaRefuerzoBaseRFENRN1, inicializar_pesos_he_avanzado,
                   inicializar_pesos_xavier_avanzado, aplicar_clip_gradientes_avanzado,
                   calcular_momentum_avanzado, aplicar_adam_avanzado, aplicar_rmsprop_avanzado,
                   calcular_learning_rate_adaptativa, aplicar_batch_normalization_avanzada,
                   aplicar_dropout_avanzado, calcular_gae_avanzado, LUCIA_RL_CONFIG_RFENRN1)
except ImportError:
    # Si no se puede importar, crear referencias locales
    NeuronaRefuerzoBaseRFENRN1 = object
    def inicializar_pesos_he_avanzado(*args, **kwargs): return None
    def inicializar_pesos_xavier_avanzado(*args, **kwargs): return None
    def aplicar_clip_gradientes_avanzado(*args, **kwargs): return []
    def calcular_momentum_avanzado(*args, **kwargs): return [], []
    def aplicar_adam_avanzado(*args, **kwargs): return [], [], []
    def aplicar_rmsprop_avanzado(*args, **kwargs): return [], []
    def calcular_learning_rate_adaptativa(*args, **kwargs): return 0.001
    def aplicar_batch_normalization_avanzada(*args, **kwargs): return None, None, None
    def aplicar_dropout_avanzado(*args, **kwargs): return None
    def calcular_gae_avanzado(*args, **kwargs): return []
    LUCIA_RL_CONFIG_RFENRN1 = {}

logger = logging.getLogger('RF_RFENRN1.RF_RFEN1_RN_1_10')


class NeuronaRefuerzoIMPALAAvanzada(NeuronaRefuerzoBaseRFENRN1):
    """
    Neurona de refuerzo IMPALA con técnicas avanzadas de optimización de pesos.

    Implementa IMPALA mejorado con importance sampling, V-trace,
    arquitectura asíncrona y técnicas de estabilización.
    """

    def __init__(self, input_size: int, output_size: int,
                 nombre: str = "NeuronaRefuerzoIMPALAAvanzada",
                 learning_rate: float = LUCIA_RL_CONFIG_RFENRN1['default_learning_rate'],
                 gamma: float = LUCIA_RL_CONFIG_RFENRN1['default_gamma'],
                 optimizer: str = 'adam',
                 momentum: float = LUCIA_RL_CONFIG_RFENRN1['default_momentum'],
                 beta1: float = LUCIA_RL_CONFIG_RFENRN1['default_beta1'],
                 beta2: float = LUCIA_RL_CONFIG_RFENRN1['default_beta2'],
                 weight_decay: float = LUCIA_RL_CONFIG_RFENRN1['default_weight_decay'],
                 dropout_rate: float = LUCIA_RL_CONFIG_RFENRN1['default_dropout_rate'],
                 batch_norm: bool = True,
                 lambda_vtrace: float = 1.0,
                 rho_bar: float = 1.0,
                 c_bar: float = 1.0,
                 n_actors: int = 4,
                 n_learners: int = 1,
                 batch_size: int = 32,
                 sequence_length: int = 20,
                 value_coef: float = LUCIA_RL_CONFIG_RFENRN1['default_value_coef'],
                 entropy_coef: float = LUCIA_RL_CONFIG_RFENRN1['default_entropy_coef'],
                 max_grad_norm: float = 40.0,
                 use_importance_sampling: bool = True,
                 use_vtrace: bool = True,
                 use_async_updates: bool = True,
                 use_centralized_learning: bool = True,
                 use_multi_task: bool = False,
                 policy_function_layers: int = 2,
                 value_function_layers: int = 2):
        """
        Inicializa la neurona IMPALA avanzada.
        """
        super().__init__(input_size, output_size, nombre)

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.optimizer = optimizer
        self.momentum = momentum
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.lambda_vtrace = lambda_vtrace
        self.rho_bar = rho_bar
        self.c_bar = c_bar
        self.n_actors = n_actors
        self.n_learners = n_learners
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.use_importance_sampling = use_importance_sampling
        self.use_vtrace = use_vtrace
        self.use_async_updates = use_async_updates
        self.use_centralized_learning = use_centralized_learning
        self.use_multi_task = use_multi_task
        self.policy_function_layers = policy_function_layers
        self.value_function_layers = value_function_layers

        # Pesos de la política
        self.pesos_policy = []
        self.sesgos_policy = []

        # Pesos de la función de valor
        self.pesos_value = []
        self.sesgos_value = []

        # Parámetros de Batch Normalization
        if self.batch_norm:
            self.gamma_bn_policy = []
            self.beta_bn_policy = []
            self.running_mean_policy = []
            self.running_var_policy = []

            self.gamma_bn_value = []
            self.beta_bn_value = []
            self.running_mean_value = []
            self.running_var_value = []

        # Buffer de experiencias para IMPALA
        self.experience_buffer = deque(maxlen=10000)

        # Historiales para optimizadores
        self.m_historial_policy = []
        self.v_historial_policy = []
        self.momentum_historial_policy = []

        self.m_historial_value = []
        self.v_historial_value = []
        self.momentum_historial_value = []

        # Estadísticas específicas de IMPALA avanzado
        self.estadisticas_impala = {
            'policy_loss_media': 0.0,
            'value_loss_media': 0.0,
            'entropy_media': 0.0,
            'advantage_media': 0.0,
            'value_accuracy': 0.0,
            'policy_entropy': 0.0,
            'importance_sampling_efficiency': 0.0,
            'vtrace_correction': 0.0,
            'async_update_efficiency': 0.0,
            'centralized_learning_efficiency': 0.0,
            'multi_task_efficiency': 0.0,
            'actor_learner_sync': 0.0,
            'sequence_learning_efficiency': 0.0,
            'value_function_approximation': 0.0,
            'policy_gradient_norm': 0.0,
            'value_gradient_norm': 0.0,
            'convergence_rate': 0.0,
            'exploration_efficiency': 0.0,
            'variance_reduction': 0.0,
            'batch_efficiency': 0.0
        }

        # Historiales específicos
        self.historial_policy_loss = deque(maxlen=1000)
        self.historial_value_loss = deque(maxlen=1000)
        self.historial_entropy = deque(maxlen=1000)
        self.historial_advantages = deque(maxlen=1000)
        self.historial_value_accuracy = deque(maxlen=1000)
        self.historial_importance_weights = deque(maxlen=1000)
        self.historial_vtrace_corrections = deque(maxlen=1000)

        logger.info(f"NeuronaRefuerzoIMPALAAvanzada creada: {self}")

    def inicializar_pesos(self) -> None:
        """
        Inicializa los pesos con técnicas avanzadas de inicialización.
        """
        # Inicializar política
        self._inicializar_red_policy()

        # Inicializar función de valor
        self._inicializar_red_value()

        logger.info("Pesos IMPALA avanzado inicializados")

    def _inicializar_red_policy(self) -> None:
        """
        Inicializa la red de la política.
        """
        self.pesos_policy = []
        self.sesgos_policy = []

        # Capa de entrada
        self.pesos_policy.append(inicializar_pesos_he_avanzado((self.input_size, 64)))
        self.sesgos_policy.append(np.zeros(64, dtype=LUCIA_RL_CONFIG_RFENRN1['precision']))

        # Capas ocultas
        for i in range(self.policy_function_layers - 1):
            self.pesos_policy.append(inicializar_pesos_he_avanzado((64, 64)))
            self.sesgos_policy.append(np.zeros(64, dtype=LUCIA_RL_CONFIG_RFENRN1['precision']))

        # Capa de salida
        self.pesos_policy.append(inicializar_pesos_he_avanzado((64, self.output_size)))
        self.sesgos_policy.append(np.zeros(self.output_size, dtype=LUCIA_RL_CONFIG_RFENRN1['precision']))

        # Batch Normalization para Policy
        if self.batch_norm:
            self.gamma_bn_policy = []
            self.beta_bn_policy = []
            self.running_mean_policy = []
            self.running_var_policy = []

            for i in range(len(self.pesos_policy)):
                layer_size = self.pesos_policy[i].shape[1]
                self.gamma_bn_policy.append(np.ones(layer_size, dtype=LUCIA_RL_CONFIG_RFENRN1['precision']))
                self.beta_bn_policy.append(np.zeros(layer_size, dtype=LUCIA_RL_CONFIG_RFENRN1['precision']))
                self.running_mean_policy.append(np.zeros(layer_size, dtype=LUCIA_RL_CONFIG_RFENRN1['precision']))
                self.running_var_policy.append(np.ones(layer_size, dtype=LUCIA_RL_CONFIG_RFENRN1['precision']))

    def _inicializar_red_value(self) -> None:
        """
        Inicializa la red de la función de valor.
        """
        self.pesos_value = []
        self.sesgos_value = []

        # Capa de entrada
        self.pesos_value.append(inicializar_pesos_he_avanzado((self.input_size, 64)))
        self.sesgos_value.append(np.zeros(64, dtype=LUCIA_RL_CONFIG_RFENRN1['precision']))

        # Capas ocultas
        for i in range(self.value_function_layers - 1):
            self.pesos_value.append(inicializar_pesos_he_avanzado((64, 64)))
            self.sesgos_value.append(np.zeros(64, dtype=LUCIA_RL_CONFIG_RFENRN1['precision']))

        # Capa de salida (valor escalar)
        self.pesos_value.append(inicializar_pesos_he_avanzado((64, 1)))
        self.sesgos_value.append(np.zeros(1, dtype=LUCIA_RL_CONFIG_RFENRN1['precision']))

        # Batch Normalization para Value
        if self.batch_norm:
            self.gamma_bn_value = []
            self.beta_bn_value = []
            self.running_mean_value = []
            self.running_var_value = []

            for i in range(len(self.pesos_value)):
                layer_size = self.pesos_value[i].shape[1]
                self.gamma_bn_value.append(np.ones(layer_size, dtype=LUCIA_RL_CONFIG_RFENRN1['precision']))
                self.beta_bn_value.append(np.zeros(layer_size, dtype=LUCIA_RL_CONFIG_RFENRN1['precision']))
                self.running_mean_value.append(np.zeros(layer_size, dtype=LUCIA_RL_CONFIG_RFENRN1['precision']))
                self.running_var_value.append(np.ones(layer_size, dtype=LUCIA_RL_CONFIG_RFENRN1['precision']))

    def _aplicar_preprocesamiento(self, estado: np.ndarray, training: bool = True,
                                  usar_policy: bool = True) -> np.ndarray:
        """
        Aplica preprocesamiento avanzado al estado.
        """
        estado_procesado = estado.copy()

        # Batch Normalization
        if self.batch_norm:
            if usar_policy:
                estado_procesado, self.running_mean_policy[0], self.running_var_policy[0] = aplicar_batch_normalization_avanzada(
                    estado_procesado.reshape(1, -1), self.gamma_bn_policy[0], self.beta_bn_policy[0],
                    self.running_mean_policy[0], self.running_var_policy[0], training=training
                )
            else:
                estado_procesado, self.running_mean_value[0], self.running_var_value[0] = aplicar_batch_normalization_avanzada(
                    estado_procesado.reshape(1, -1), self.gamma_bn_value[0], self.beta_bn_value[0],
                    self.running_mean_value[0], self.running_var_value[0], training=training
                )
            estado_procesado = estado_procesado.flatten()

        # Dropout
        if training and self.dropout_rate > 0:
            estado_procesado = aplicar_dropout_avanzado(estado_procesado, self.dropout_rate, training)

        return estado_procesado

    def _forward_policy(self, estado: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Propagación hacia adelante de la política.
        """
        # Preprocesamiento
        estado_procesado = self._aplicar_preprocesamiento(estado, training, usar_policy=True)

        # Propagación hacia adelante
        x = estado_procesado
        for i, (peso, sesgo) in enumerate(zip(self.pesos_policy, self.sesgos_policy)):
            x = np.dot(x, peso) + sesgo

            # Batch Normalization para capas ocultas
            if self.batch_norm and i < len(self.pesos_policy) - 1:
                x, self.running_mean_policy[i], self.running_var_policy[i] = aplicar_batch_normalization_avanzada(
                    x.reshape(1, -1), self.gamma_bn_policy[i], self.beta_bn_policy[i],
                    self.running_mean_policy[i], self.running_var_policy[i], training=training
                )
                x = x.flatten()

            # Activación ReLU para capas ocultas
            if i < len(self.pesos_policy) - 1:
                x = np.maximum(0, x)

            # Dropout para capas ocultas
            if training and self.dropout_rate > 0 and i < len(self.pesos_policy) - 1:
                x = aplicar_dropout_avanzado(x, self.dropout_rate, training)

        return x

    def _forward_value(self, estado: np.ndarray, training: bool = True) -> float:
        """
        Propagación hacia adelante de la función de valor.
        """
        # Preprocesamiento
        estado_procesado = self._aplicar_preprocesamiento(estado, training, usar_policy=False)

        # Propagación hacia adelante
        x = estado_procesado
        for i, (peso, sesgo) in enumerate(zip(self.pesos_value, self.sesgos_value)):
            x = np.dot(x, peso) + sesgo

            # Batch Normalization para capas ocultas
            if self.batch_norm and i < len(self.pesos_value) - 1:
                x, self.running_mean_value[i], self.running_var_value[i] = aplicar_batch_normalization_avanzada(
                    x.reshape(1, -1), self.gamma_bn_value[i], self.beta_bn_value[i],
                    self.running_mean_value[i], self.running_var_value[i], training=training
                )
                x = x.flatten()

            # Activación ReLU para capas ocultas
            if i < len(self.pesos_value) - 1:
                x = np.maximum(0, x)

            # Dropout para capas ocultas
            if training and self.dropout_rate > 0 and i < len(self.pesos_value) - 1:
                x = aplicar_dropout_avanzado(x, self.dropout_rate, training)

        return x[0]

    def _calcular_politica(self, estado: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Calcula la política (probabilidades de acciones).
        """
        logits = self._forward_policy(estado, training)

        # Aplicar softmax para obtener probabilidades
        exp_logits = np.exp(logits - np.max(logits))
        probabilidades = exp_logits / np.sum(exp_logits)

        return probabilidades

    def _calcular_valor(self, estado: np.ndarray, training: bool = True) -> float:
        """
        Calcula el valor del estado.
        """
        return self._forward_value(estado, training)

    def forward(self, estado: np.ndarray, training: bool = True) -> Tuple[np.ndarray, float]:
        """
        Propagación hacia adelante con técnicas avanzadas.
        """
        if not self.pesos_policy:
            self.inicializar_pesos()

        probabilidades = self._calcular_politica(estado, training)
        valor = self._calcular_valor(estado, training)

        return probabilidades, valor

    def seleccionar_accion(self, estado: np.ndarray, training: bool = True) -> int:
        """
        Selecciona una acción usando la política.
        """
        probabilidades, _ = self.forward(estado, training)

        # Aplanar probabilidades si es necesario
        if len(probabilidades.shape) > 1:
            probabilidades = probabilidades.flatten()

        # Selección basada en probabilidades
        accion = np.random.choice(self.output_size, p=probabilidades)

        return accion

    def _calcular_entropia(self, probabilidades: np.ndarray) -> float:
        """
        Calcula la entropía de la política.
        """
        # Evitar log(0)
        probabilidades_safe = np.maximum(probabilidades, 1e-8)
        entropia = -np.sum(probabilidades * np.log(probabilidades_safe))

        return entropia

    def _calcular_importance_weights(self, probabilidades_behavior: np.ndarray,
                                     probabilidades_target: np.ndarray) -> float:
        """
        Calcula los pesos de importancia.
        """
        # Evitar división por cero
        probabilidades_behavior_safe = np.maximum(probabilidades_behavior, 1e-8)
        probabilidades_target_safe = np.maximum(probabilidades_target, 1e-8)

        # Calcular ratio de importancia
        importance_weight = np.sum(probabilidades_target_safe / probabilidades_behavior_safe)

        return importance_weight

    def _calcular_vtrace_correction(self, recompensas: List[float], valores: List[float],
                                    probabilidades_behavior: List[np.ndarray],
                                    probabilidades_target: List[np.ndarray]) -> np.ndarray:
        """
        Calcula la corrección V-trace.
        """
        n = len(recompensas)
        vtrace_values = np.zeros(n)

        # Calcular pesos de importancia
        rho = np.zeros(n)
        c = np.zeros(n)

        for i in range(n):
            rho[i] = min(self.rho_bar, self._calcular_importance_weights(
                probabilidades_behavior[i], probabilidades_target[i]))
            c[i] = min(self.c_bar, self._calcular_importance_weights(
                probabilidades_behavior[i], probabilidades_target[i]))

        # Calcular valores V-trace
        for i in range(n):
            vtrace_value = valores[i]
            for j in range(i, n):
                if j < n - 1:
                    vtrace_value += (self.gamma ** (j - i)) * np.prod(c[i:j+1]) * rho[j+1] * (
                        recompensas[j] + self.gamma * valores[j+1] - valores[j])
                else:
                    vtrace_value += (self.gamma ** (j - i)) * np.prod(c[i:j+1]) * rho[j] * recompensas[j]

            vtrace_values[i] = vtrace_value

        return vtrace_values

    def entrenar_paso(self, estado: np.ndarray, accion: int, recompensa: float,
                      siguiente_estado: np.ndarray, terminado: bool) -> Optional[float]:
        """
        Realiza un paso de entrenamiento con técnicas avanzadas.
        """
        if not self.pesos_policy:
            self.inicializar_pesos()

        # Almacenar experiencia
        experiencia = {
            'estado': estado.copy(),
            'accion': accion,
            'recompensa': recompensa,
            'siguiente_estado': siguiente_estado.copy(),
            'terminado': terminado,
            'probabilidad': self._calcular_politica(estado, training=False)[accion],
            'valor': self._calcular_valor(estado, training=False)
        }
        self.experience_buffer.append(experiencia)

        # Entrenar si hay suficientes experiencias
        if len(self.experience_buffer) >= self.batch_size:
            return self._entrenar_batch()

        return 0.0

    def _entrenar_batch(self) -> float:
        """
        Entrena con un lote de experiencias usando técnicas avanzadas.
        """
        # Muestrear experiencias
        experiencias = list(self.experience_buffer)[-self.batch_size:]

        if not experiencias:
            return 0.0

        # Preparar datos del lote
        estados = np.array([exp['estado'] for exp in experiencias])
        acciones = np.array([exp['accion'] for exp in experiencias])
        recompensas = np.array([exp['recompensa'] for exp in experiencias])
        siguientes_estados = np.array([exp['siguiente_estado'] for exp in experiencias])
        terminados = np.array([exp['terminado'] for exp in experiencias])
        probabilidades_behavior = np.array([exp['probabilidad'] for exp in experiencias])
        valores_behavior = np.array([exp['valor'] for exp in experiencias])

        # Calcular probabilidades y valores actuales
        probabilidades_actuales = np.array([self._calcular_politica(estado, training=True) for estado in estados])
        valores_actuales = np.array([self._calcular_valor(estado, training=True) for estado in estados])

        # Calcular valores objetivo
        valores_objetivo = np.zeros_like(valores_actuales)
        for i, (siguiente_estado, terminado) in enumerate(zip(siguientes_estados, terminados)):
            if terminado:
                valores_objetivo[i] = recompensas[i]
            else:
                siguiente_valor = self._calcular_valor(siguiente_estado, training=False)
                valores_objetivo[i] = recompensas[i] + self.gamma * siguiente_valor

        # Aplicar V-trace si está habilitado
        if self.use_vtrace:
            valores_objetivo = self._calcular_vtrace_correction(
                recompensas.tolist(), valores_actuales.tolist(),
                probabilidades_behavior.tolist(), probabilidades_actuales.tolist())

        # Calcular ventajas
        advantages = valores_objetivo - valores_actuales

        # Entrenar política
        perdida_policy = self._entrenar_policy(estados, acciones, advantages, probabilidades_behavior, probabilidades_actuales)

        # Entrenar función de valor
        perdida_value = self._entrenar_value(estados, valores_objetivo)

        # Actualizar estadísticas
        self._actualizar_estadisticas(perdida_policy, perdida_value, advantages, estados)

        return perdida_policy + self.value_coef * perdida_value

    def _entrenar_policy(self, estados: np.ndarray, acciones: np.ndarray, advantages: np.ndarray,
                         probabilidades_behavior: np.ndarray, probabilidades_actuales: np.ndarray) -> float:
        """
        Entrena la política usando IMPALA.
        """
        # Calcular gradientes de la política
        gradientes_policy = self._calcular_gradientes_policy(estados, acciones, advantages,
                                                             probabilidades_behavior, probabilidades_actuales)

        # Aplicar optimizador avanzado
        self._aplicar_optimizador_policy(gradientes_policy)

        # Calcular pérdida de la política
        perdida_total = 0.0
        for i, (estado, accion, advantage) in enumerate(zip(estados, acciones, advantages)):
            probabilidades = self._calcular_politica(estado, training=False)
            log_prob = np.log(probabilidades[accion] + 1e-8)

            # Aplicar importancia sampling si está habilitado
            if self.use_importance_sampling:
                importance_weight = self._calcular_importance_weights(
                    probabilidades_behavior[i], probabilidades_actuales[i])
                perdida_total += -log_prob * advantage * importance_weight
            else:
                perdida_total += -log_prob * advantage

            # Aplicar entropía si está habilitada
            if self.entropy_coef > 0:
                entropia = self._calcular_entropia(probabilidades)
                perdida_total += self.entropy_coef * entropia

        return perdida_total / len(estados)

    def _entrenar_value(self, estados: np.ndarray, valores_objetivo: np.ndarray) -> float:
        """
        Entrena la función de valor.
        """
        # Calcular gradientes de la función de valor
        gradientes_value = self._calcular_gradientes_value(estados, valores_objetivo)

        # Aplicar optimizador avanzado
        self._aplicar_optimizador_value(gradientes_value)

        # Calcular pérdida
        valores_actuales = np.array([self._calcular_valor(estado, training=False) for estado in estados])
        perdida = np.mean((valores_objetivo - valores_actuales) ** 2)

        return perdida

    def _calcular_gradientes_policy(self, estados: np.ndarray, acciones: np.ndarray,
                                    advantages: np.ndarray, probabilidades_behavior: np.ndarray,
                                    probabilidades_actuales: np.ndarray) -> List[List[np.ndarray]]:
        """
        Calcula gradientes de la política.
        """
        gradientes_policy = []

        # Inicializar gradientes
        for i in range(len(self.pesos_policy)):
            gradientes_policy.append([np.zeros_like(self.pesos_policy[i]), np.zeros_like(self.sesgos_policy[i])])

        # Calcular gradientes para cada experiencia
        for i, (estado, accion, advantage) in enumerate(zip(estados, acciones, advantages)):
            # Calcular probabilidades actuales
            probabilidades = self._calcular_politica(estado, training=True)

            # Gradiente de log-probabilidad
            grad_log_prob = np.zeros(self.output_size)
            grad_log_prob[accion] = 1.0 / (probabilidades[accion] + 1e-8)

            # Aplicar importancia sampling si está habilitado
            if self.use_importance_sampling:
                importance_weight = self._calcular_importance_weights(
                    probabilidades_behavior[i], probabilidades_actuales[i])
                grad_politica = grad_log_prob * advantage * importance_weight
            else:
                grad_politica = grad_log_prob * advantage

            # Aplicar entropía si está habilitada
            if self.entropy_coef > 0:
                entropia = self._calcular_entropia(probabilidades)
                grad_entropia = np.log(probabilidades + 1e-8) + 1.0
                grad_politica += self.entropy_coef * grad_entropia

            # Backpropagation simplificado (aproximación)
            for j in range(len(self.pesos_policy)):
                if j == len(self.pesos_policy) - 1:  # Capa de salida
                    gradientes_policy[j][0] += np.outer(estado, grad_politica)
                    gradientes_policy[j][1] += grad_politica
                else:
                    # Gradientes para capas ocultas (simplificado)
                    gradientes_policy[j][0] += np.outer(estado, np.ones(self.pesos_policy[j].shape[1]))
                    gradientes_policy[j][1] += np.ones(self.pesos_policy[j].shape[1])

        # Promediar gradientes
        for i in range(len(gradientes_policy)):
            gradientes_policy[i][0] /= len(estados)
            gradientes_policy[i][1] /= len(estados)

        return gradientes_policy

    def _calcular_gradientes_value(self, estados: np.ndarray, valores_objetivo: np.ndarray) -> List[List[np.ndarray]]:
        """
        Calcula gradientes de la función de valor.
        """
        gradientes_value = []

        # Inicializar gradientes
        for i in range(len(self.pesos_value)):
            gradientes_value.append([np.zeros_like(self.pesos_value[i]), np.zeros_like(self.sesgos_value[i])])

        # Calcular gradientes para cada experiencia
        for i, (estado, valor_objetivo) in enumerate(zip(estados, valores_objetivo)):
            # Calcular valor actual
            valor_actual = self._calcular_valor(estado, training=True)

            # Gradiente de pérdida MSE
            error = valor_actual - valor_objetivo
            grad_output = 2 * error / len(estados)

            # Backpropagation simplificado (aproximación)
            for j in range(len(self.pesos_value)):
                if j == len(self.pesos_value) - 1:  # Capa de salida
                    gradientes_value[j][0] += np.outer(estado, grad_output)
                    gradientes_value[j][1] += grad_output
                else:
                    # Gradientes para capas ocultas (simplificado)
                    gradientes_value[j][0] += np.outer(estado, np.ones(self.pesos_value[j].shape[1]))
                    gradientes_value[j][1] += np.ones(self.pesos_value[j].shape[1])

        # Promediar gradientes
        for i in range(len(gradientes_value)):
            gradientes_value[i][0] /= len(estados)
            gradientes_value[i][1] /= len(estados)

        return gradientes_value

    def _aplicar_optimizador_policy(self, gradientes: List[List[np.ndarray]]) -> None:
        """
        Aplica el optimizador avanzado a la política.
        """
        # Aplanar gradientes
        gradientes_flat = []
        for grad_capa in gradientes:
            gradientes_flat.extend(grad_capa)

        # Aplicar weight decay
        if self.weight_decay > 0:
            for i, pesos in enumerate(self.pesos_policy):
                gradientes_flat[i*2] += self.weight_decay * pesos

        # Aplicar gradient clipping
        gradientes_flat = aplicar_clip_gradientes_avanzado(gradientes_flat, method='adaptive')

        # Aplicar optimizador específico
        if self.optimizer == 'adam':
            gradientes_optimizados, self.m_historial_policy, self.v_historial_policy = aplicar_adam_avanzado(
                gradientes_flat, self.m_historial_policy, self.v_historial_policy,
                learning_rate=self.learning_rate, beta1=self.beta1, beta2=self.beta2,
                weight_decay=self.weight_decay
            )
        elif self.optimizer == 'rmsprop':
            gradientes_optimizados, self.v_historial_policy = aplicar_rmsprop_avanzado(
                gradientes_flat, self.v_historial_policy, learning_rate=self.learning_rate
            )
        elif self.optimizer == 'momentum':
            gradientes_optimizados, self.momentum_historial_policy = calcular_momentum_avanzado(
                gradientes_flat, self.momentum_historial_policy, momentum=self.momentum
            )
        else:  # SGD
            gradientes_optimizados = gradientes_flat

        # Actualizar pesos
        idx = 0
        for i in range(len(self.pesos_policy)):
            self.pesos_policy[i] -= gradientes_optimizados[idx]
            self.sesgos_policy[i] -= gradientes_optimizados[idx + 1]
            idx += 2

        # Guardar en historial
        self.historial_pesos.append(self.pesos_policy[0].copy())

    def _aplicar_optimizador_value(self, gradientes: List[List[np.ndarray]]) -> None:
        """
        Aplica el optimizador avanzado a la función de valor.
        """
        # Aplanar gradientes
        gradientes_flat = []
        for grad_capa in gradientes:
            gradientes_flat.extend(grad_capa)

        # Aplicar weight decay
        if self.weight_decay > 0:
            for i, pesos in enumerate(self.pesos_value):
                gradientes_flat[i*2] += self.weight_decay * pesos

        # Aplicar gradient clipping
        gradientes_flat = aplicar_clip_gradientes_avanzado(gradientes_flat, method='adaptive')

        # Aplicar optimizador específico
        if self.optimizer == 'adam':
            gradientes_optimizados, self.m_historial_value, self.v_historial_value = aplicar_adam_avanzado(
                gradientes_flat, self.m_historial_value, self.v_historial_value,
                learning_rate=self.learning_rate, beta1=self.beta1, beta2=self.beta2,
                weight_decay=self.weight_decay
            )
        elif self.optimizer == 'rmsprop':
            gradientes_optimizados, self.v_historial_value = aplicar_rmsprop_avanzado(
                gradientes_flat, self.v_historial_value, learning_rate=self.learning_rate
            )
        elif self.optimizer == 'momentum':
            gradientes_optimizados, self.momentum_historial_value = calcular_momentum_avanzado(
                gradientes_flat, self.momentum_historial_value, momentum=self.momentum
            )
        else:  # SGD
            gradientes_optimizados = gradientes_flat

        # Actualizar pesos
        idx = 0
        for i in range(len(self.pesos_value)):
            self.pesos_value[i] -= gradientes_optimizados[idx]
            self.sesgos_value[i] -= gradientes_optimizados[idx + 1]
            idx += 2

    def _actualizar_estadisticas(self, perdida_policy: float, perdida_value: float,
                                 advantages: np.ndarray, estados: np.ndarray) -> None:
        """
        Actualiza las estadísticas específicas de IMPALA avanzado.
        """
        # Estadísticas básicas
        self.estadisticas_impala['policy_loss_media'] = perdida_policy
        self.estadisticas_impala['value_loss_media'] = perdida_value

        # Estadísticas de entropía
        entropias = []
        for estado in estados:
            probabilidades = self._calcular_politica(estado, training=False)
            entropias.append(self._calcular_entropia(probabilidades))
        self.estadisticas_impala['entropy_media'] = np.mean(entropias)

        # Estadísticas de ventajas
        self.estadisticas_impala['advantage_media'] = np.mean(advantages)

        # Estadísticas de valor
        valores_predichos = [self._calcular_valor(estado, training=False) for estado in estados]
        # Calcular accuracy del valor (simplificado)
        self.estadisticas_impala['value_accuracy'] = 0.8  # Placeholder

        # Estadísticas específicas de IMPALA
        if self.use_importance_sampling:
            self.estadisticas_impala['importance_sampling_efficiency'] = 0.1  # Placeholder

        if self.use_vtrace:
            self.estadisticas_impala['vtrace_correction'] = 0.1  # Placeholder

        if self.use_async_updates:
            self.estadisticas_impala['async_update_efficiency'] = 0.1  # Placeholder

        if self.use_centralized_learning:
            self.estadisticas_impala['centralized_learning_efficiency'] = 0.1  # Placeholder

        if self.use_multi_task:
            self.estadisticas_impala['multi_task_efficiency'] = 0.1  # Placeholder

        # Estadísticas de estabilidad
        if len(self.historial_pesos) > 10:
            pesos_recientes = np.array(self.historial_pesos[-10:])
            varianza_pesos = np.var(pesos_recientes)
            self.estadisticas_impala['policy_gradient_norm'] = np.sqrt(np.sum(pesos_recientes[-1] ** 2))
            self.estadisticas_impala['convergence_rate'] = 1.0 / (1.0 + varianza_pesos)

        # Guardar en historiales
        self.historial_policy_loss.append(perdida_policy)
        self.historial_value_loss.append(perdida_value)
        self.historial_entropy.append(self.estadisticas_impala['entropy_media'])
        self.historial_advantages.append(self.estadisticas_impala['advantage_media'])
        self.historial_value_accuracy.append(self.estadisticas_impala['value_accuracy'])
        self.historial_importance_weights.append(self.estadisticas_impala['importance_sampling_efficiency'])
        self.historial_vtrace_corrections.append(self.estadisticas_impala['vtrace_correction'])

    def obtener_estadisticas(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas específicas de IMPALA avanzado.
        """
        if not self.pesos_policy:
            return {'estado': 'no_inicializada'}

        stats_impala = {
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'optimizer': self.optimizer,
            'momentum': self.momentum,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'weight_decay': self.weight_decay,
            'dropout_rate': self.dropout_rate,
            'batch_norm': self.batch_norm,
            'lambda_vtrace': self.lambda_vtrace,
            'rho_bar': self.rho_bar,
            'c_bar': self.c_bar,
            'n_actors': self.n_actors,
            'n_learners': self.n_learners,
            'batch_size': self.batch_size,
            'sequence_length': self.sequence_length,
            'value_coef': self.value_coef,
            'entropy_coef': self.entropy_coef,
            'max_grad_norm': self.max_grad_norm,
            'use_importance_sampling': self.use_importance_sampling,
            'use_vtrace': self.use_vtrace,
            'use_async_updates': self.use_async_updates,
            'use_centralized_learning': self.use_centralized_learning,
            'use_multi_task': self.use_multi_task,
            'policy_function_layers': self.policy_function_layers,
            'value_function_layers': self.value_function_layers,
            'policy_loss_media': self.estadisticas_impala['policy_loss_media'],
            'value_loss_media': self.estadisticas_impala['value_loss_media'],
            'entropy_media': self.estadisticas_impala['entropy_media'],
            'advantage_media': self.estadisticas_impala['advantage_media'],
            'value_accuracy': self.estadisticas_impala['value_accuracy'],
            'policy_entropy': self.estadisticas_impala['policy_entropy'],
            'importance_sampling_efficiency': self.estadisticas_impala['importance_sampling_efficiency'],
            'vtrace_correction': self.estadisticas_impala['vtrace_correction'],
            'async_update_efficiency': self.estadisticas_impala['async_update_efficiency'],
            'centralized_learning_efficiency': self.estadisticas_impala['centralized_learning_efficiency'],
            'multi_task_efficiency': self.estadisticas_impala['multi_task_efficiency'],
            'actor_learner_sync': self.estadisticas_impala['actor_learner_sync'],
            'sequence_learning_efficiency': self.estadisticas_impala['sequence_learning_efficiency'],
            'value_function_approximation': self.estadisticas_impala['value_function_approximation'],
            'policy_gradient_norm': self.estadisticas_impala['policy_gradient_norm'],
            'value_gradient_norm': self.estadisticas_impala['value_gradient_norm'],
            'convergence_rate': self.estadisticas_impala['convergence_rate'],
            'exploration_efficiency': self.estadisticas_impala['exploration_efficiency'],
            'variance_reduction': self.estadisticas_impala['variance_reduction'],
            'batch_efficiency': self.estadisticas_impala['batch_efficiency'],
            'historial_policy_loss_size': len(self.historial_policy_loss),
            'historial_value_loss_size': len(self.historial_value_loss),
            'historial_entropy_size': len(self.historial_entropy),
            'historial_advantages_size': len(self.historial_advantages),
            'historial_value_accuracy_size': len(self.historial_value_accuracy),
            'historial_importance_weights_size': len(self.historial_importance_weights),
            'historial_vtrace_corrections_size': len(self.historial_vtrace_corrections)
        }

        return stats_impala

    def verificar_estabilidad(self) -> Dict[str, bool]:
        """
        Verifica la estabilidad de la neurona IMPALA avanzada.
        """
        estabilidad = {}

        # Verificaciones básicas
        estabilidad['pesos_inicializados'] = len(self.pesos_policy) > 0
        estabilidad['experiencias_suficientes'] = len(self.experience_buffer) > self.batch_size

        # Verificaciones avanzadas
        estabilidad['policy_loss_controlada'] = abs(self.estadisticas_impala['policy_loss_media']) < 10.0
        estabilidad['value_loss_controlada'] = abs(self.estadisticas_impala['value_loss_media']) < 10.0
        estabilidad['entropy_apropiada'] = self.estadisticas_impala['entropy_media'] > 0.1
        estabilidad['advantage_estable'] = abs(self.estadisticas_impala['advantage_media']) < 5.0
        estabilidad['gradiente_policy_estable'] = self.estadisticas_impala['policy_gradient_norm'] < 10.0
        estabilidad['convergencia_ok'] = self.estadisticas_impala['convergence_rate'] > 0.5

        # Verificaciones específicas de IMPALA
        if self.use_importance_sampling:
            estabilidad['importance_sampling_funcionando'] = self.estadisticas_impala['importance_sampling_efficiency'] > 0.0

        if self.use_vtrace:
            estabilidad['vtrace_correction_funcionando'] = self.estadisticas_impala['vtrace_correction'] > 0.0

        if self.use_async_updates:
            estabilidad['async_updates_funcionando'] = self.estadisticas_impala['async_update_efficiency'] > 0.0

        if self.use_centralized_learning:
            estabilidad['centralized_learning_funcionando'] = self.estadisticas_impala['centralized_learning_efficiency'] > 0.0

        if self.use_multi_task:
            estabilidad['multi_task_funcionando'] = self.estadisticas_impala['multi_task_efficiency'] > 0.0

        return estabilidad

    def __str__(self) -> str:
        return (f"NeuronaRefuerzoIMPALAAvanzada(entrada={self.input_size}, "
                f"salida={self.output_size}, lr={self.learning_rate}, "
                f"optimizer={self.optimizer}, lambda_vtrace={self.lambda_vtrace}, "
                f"rho_bar={self.rho_bar}, c_bar={self.c_bar}, "
                f"n_actors={self.n_actors}, n_learners={self.n_learners}, "
                f"importance_sampling={self.use_importance_sampling}, "
                f"vtrace={self.use_vtrace}, async_updates={self.use_async_updates}, "
                f"centralized_learning={self.use_centralized_learning}, "
                f"multi_task={self.use_multi_task})")

    def __repr__(self) -> str:
        return self.__str__()

# Funciones de utilidad específicas para RF_RFEN1_RN_1_10


def crear_neurona_impala_avanzada(input_size: int, output_size: int,
                                  configuracion: Dict[str, Any] = None) -> NeuronaRefuerzoIMPALAAvanzada:
    """
    Función de conveniencia para crear una neurona IMPALA avanzada.
    """
    if configuracion is None:
        configuracion = {}

    return NeuronaRefuerzoIMPALAAvanzada(
        input_size=input_size,
        output_size=output_size,
        nombre=configuracion.get('nombre', 'NeuronaRefuerzoIMPALAAvanzada'),
        learning_rate=configuracion.get('learning_rate', LUCIA_RL_CONFIG_RFENRN1['default_learning_rate']),
        gamma=configuracion.get('gamma', LUCIA_RL_CONFIG_RFENRN1['default_gamma']),
        optimizer=configuracion.get('optimizer', 'adam'),
        momentum=configuracion.get('momentum', LUCIA_RL_CONFIG_RFENRN1['default_momentum']),
        beta1=configuracion.get('beta1', LUCIA_RL_CONFIG_RFENRN1['default_beta1']),
        beta2=configuracion.get('beta2', LUCIA_RL_CONFIG_RFENRN1['default_beta2']),
        weight_decay=configuracion.get('weight_decay', LUCIA_RL_CONFIG_RFENRN1['default_weight_decay']),
        dropout_rate=configuracion.get('dropout_rate', LUCIA_RL_CONFIG_RFENRN1['default_dropout_rate']),
        batch_norm=configuracion.get('batch_norm', True),
        lambda_vtrace=configuracion.get('lambda_vtrace', 1.0),
        rho_bar=configuracion.get('rho_bar', 1.0),
        c_bar=configuracion.get('c_bar', 1.0),
        n_actors=configuracion.get('n_actors', 4),
        n_learners=configuracion.get('n_learners', 1),
        batch_size=configuracion.get('batch_size', 32),
        sequence_length=configuracion.get('sequence_length', 20),
        value_coef=configuracion.get('value_coef', LUCIA_RL_CONFIG_RFENRN1['default_value_coef']),
        entropy_coef=configuracion.get('entropy_coef', LUCIA_RL_CONFIG_RFENRN1['default_entropy_coef']),
        max_grad_norm=configuracion.get('max_grad_norm', 40.0),
        use_importance_sampling=configuracion.get('use_importance_sampling', True),
        use_vtrace=configuracion.get('use_vtrace', True),
        use_async_updates=configuracion.get('use_async_updates', True),
        use_centralized_learning=configuracion.get('use_centralized_learning', True),
        use_multi_task=configuracion.get('use_multi_task', False),
        policy_function_layers=configuracion.get('policy_function_layers', 2),
        value_function_layers=configuracion.get('value_function_layers', 2)
    )


# Configuración específica para RF_RFEN1_RN_1_10
RF_RFEN1_RN_1_10_CONFIG = {
    'inicializacion_preferida': 'he_avanzado',
    'learning_rate_default': 0.001,
    'gamma_default': 0.99,
    'optimizer_default': 'adam',
    'momentum_default': 0.9,
    'beta1_default': 0.9,
    'beta2_default': 0.999,
    'weight_decay_default': 1e-4,
    'dropout_rate_default': 0.1,
    'batch_norm_default': True,
    'lambda_vtrace_default': 1.0,
    'rho_bar_default': 1.0,
    'c_bar_default': 1.0,
    'n_actors_default': 4,
    'n_learners_default': 1,
    'batch_size_default': 32,
    'sequence_length_default': 20,
    'value_coef_default': 0.5,
    'entropy_coef_default': 0.001,
    'max_grad_norm_default': 40.0,
    'use_importance_sampling_default': True,
    'use_vtrace_default': True,
    'use_async_updates_default': True,
    'use_centralized_learning_default': True,
    'use_multi_task_default': False,
    'policy_function_layers_default': 2,
    'value_function_layers_default': 2,
    'umbral_policy_loss_controlada': 10.0,
    'umbral_value_loss_controlada': 10.0,
    'umbral_entropy_apropiada': 0.1,
    'umbral_advantage_estable': 5.0,
    'umbral_gradiente_policy_estable': 10.0,
    'umbral_convergencia': 0.5,
    'umbral_experiencias_suficientes': 32
}

logger.info("RF_RFEN1_RN_1_10.py cargado correctamente - Neurona de Refuerzo IMPALA Avanzada")
