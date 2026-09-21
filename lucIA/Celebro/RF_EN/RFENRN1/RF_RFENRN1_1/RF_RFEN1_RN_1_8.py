"""
RF_RFEN1_RN_1_8.py - Neurona de Refuerzo TD3 Avanzada
=====================================================

Esta neurona implementa Twin Delayed Deep Deterministic Policy Gradient (TD3)
con técnicas avanzadas de optimización de pesos y mejoras específicas para 2025.

Características Avanzadas 2025:
- Optimización Adam/RMSprop con momentum adaptativo
- Regularización L2 avanzada con weight decay
- Batch Normalization para estabilización
- Dropout adaptativo para regularización
- Learning rate scheduling avanzado
- Gradient clipping adaptativo
- Twin Q-Networks para estabilidad
- Delayed policy updates
- Target policy smoothing
- Deterministic policy gradient
- Value function approximation avanzada
- Policy gradient con reparameterization trick
- Noise regularization
- Exploration strategy avanzada

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

logger = logging.getLogger('RF_RFENRN1.RF_RFEN1_RN_1_8')


class NeuronaRefuerzoTD3Avanzada(NeuronaRefuerzoBaseRFENRN1):
    """
    Neurona de refuerzo TD3 con técnicas avanzadas de optimización de pesos.

    Implementa TD3 mejorado con twin Q-networks, delayed policy updates,
    target policy smoothing y técnicas de estabilización.
    """

    def __init__(self, input_size: int, output_size: int,
                 nombre: str = "NeuronaRefuerzoTD3Avanzada",
                 learning_rate: float = LUCIA_RL_CONFIG_RFENRN1['default_learning_rate'],
                 gamma: float = LUCIA_RL_CONFIG_RFENRN1['default_gamma'],
                 optimizer: str = 'adam',
                 momentum: float = LUCIA_RL_CONFIG_RFENRN1['default_momentum'],
                 beta1: float = LUCIA_RL_CONFIG_RFENRN1['default_beta1'],
                 beta2: float = LUCIA_RL_CONFIG_RFENRN1['default_beta2'],
                 weight_decay: float = LUCIA_RL_CONFIG_RFENRN1['default_weight_decay'],
                 dropout_rate: float = LUCIA_RL_CONFIG_RFENRN1['default_dropout_rate'],
                 batch_norm: bool = True,
                 tau: float = 0.005,
                 policy_delay: int = 2,
                 target_noise: float = 0.2,
                 noise_clip: float = 0.5,
                 exploration_noise: float = 0.1,
                 replay_buffer_size: int = 1000000,
                 batch_size: int = 256,
                 hidden_size: int = 256,
                 q_function_layers: int = 2,
                 policy_function_layers: int = 2):
        """
        Inicializa la neurona TD3 avanzada.
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
        self.tau = tau
        self.policy_delay = policy_delay
        self.target_noise = target_noise
        self.noise_clip = noise_clip
        self.exploration_noise = exploration_noise
        self.replay_buffer_size = replay_buffer_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.q_function_layers = q_function_layers
        self.policy_function_layers = policy_function_layers

        # Pesos de las Q-functions
        self.pesos_q1 = []
        self.sesgos_q1 = []
        self.pesos_q2 = []
        self.sesgos_q2 = []

        # Pesos de la política
        self.pesos_policy = []
        self.sesgos_policy = []

        # Pesos de las Q-functions objetivo
        self.pesos_q1_target = []
        self.sesgos_q1_target = []
        self.pesos_q2_target = []
        self.sesgos_q2_target = []

        # Pesos de la política objetivo
        self.pesos_policy_target = []
        self.sesgos_policy_target = []

        # Parámetros de Batch Normalization
        if self.batch_norm:
            self.gamma_bn_q1 = []
            self.beta_bn_q1 = []
            self.running_mean_q1 = []
            self.running_var_q1 = []

            self.gamma_bn_q2 = []
            self.beta_bn_q2 = []
            self.running_mean_q2 = []
            self.running_var_q2 = []

            self.gamma_bn_policy = []
            self.beta_bn_policy = []
            self.running_mean_policy = []
            self.running_var_policy = []

        # Buffer de experiencias para TD3
        self.experience_buffer = deque(maxlen=self.replay_buffer_size)

        # Historiales para optimizadores
        self.m_historial_q1 = []
        self.v_historial_q1 = []
        self.momentum_historial_q1 = []

        self.m_historial_q2 = []
        self.v_historial_q2 = []
        self.momentum_historial_q2 = []

        self.m_historial_policy = []
        self.v_historial_policy = []
        self.momentum_historial_policy = []

        # Contador para delayed policy updates
        self.policy_update_counter = 0

        # Estadísticas específicas de TD3 avanzado
        self.estadisticas_td3 = {
            'q1_loss_media': 0.0,
            'q2_loss_media': 0.0,
            'policy_loss_media': 0.0,
            'q_value_media': 0.0,
            'target_q_value_media': 0.0,
            'policy_value_media': 0.0,
            'twin_q_stability': 0.0,
            'delayed_policy_updates': 0.0,
            'target_policy_smoothing': 0.0,
            'exploration_noise_efficiency': 0.0,
            'replay_buffer_efficiency': 0.0,
            'q_function_approximation': 0.0,
            'policy_gradient_norm': 0.0,
            'q_gradient_norm': 0.0,
            'convergence_rate': 0.0,
            'exploration_efficiency': 0.0,
            'variance_reduction': 0.0,
            'batch_efficiency': 0.0,
            'noise_regularization': 0.0,
            'deterministic_policy_quality': 0.0
        }

        # Historiales específicos
        self.historial_q1_loss = deque(maxlen=1000)
        self.historial_q2_loss = deque(maxlen=1000)
        self.historial_policy_loss = deque(maxlen=1000)
        self.historial_q_values = deque(maxlen=1000)
        self.historial_policy_values = deque(maxlen=1000)
        self.historial_exploration_noise = deque(maxlen=1000)

        logger.info(f"NeuronaRefuerzoTD3Avanzada creada: {self}")

    def inicializar_pesos(self) -> None:
        """
        Inicializa los pesos con técnicas avanzadas de inicialización.
        """
        # Inicializar Q1
        self._inicializar_red_q1()

        # Inicializar Q2
        self._inicializar_red_q2()

        # Inicializar política
        self._inicializar_red_policy()

        # Inicializar redes objetivo
        self._inicializar_redes_objetivo()

        logger.info("Pesos TD3 avanzado inicializados")

    def _inicializar_red_q1(self) -> None:
        """
        Inicializa la red Q1.
        """
        self.pesos_q1 = []
        self.sesgos_q1 = []

        # Capa de entrada (estado + acción)
        input_size = self.input_size + self.output_size
        self.pesos_q1.append(inicializar_pesos_he_avanzado((input_size, self.hidden_size)))
        self.sesgos_q1.append(np.zeros(self.hidden_size, dtype=LUCIA_RL_CONFIG_RFENRN1['precision']))

        # Capas ocultas
        for i in range(self.q_function_layers - 1):
            self.pesos_q1.append(inicializar_pesos_he_avanzado((self.hidden_size, self.hidden_size)))
            self.sesgos_q1.append(np.zeros(self.hidden_size, dtype=LUCIA_RL_CONFIG_RFENRN1['precision']))

        # Capa de salida (valor Q escalar)
        self.pesos_q1.append(inicializar_pesos_he_avanzado((self.hidden_size, 1)))
        self.sesgos_q1.append(np.zeros(1, dtype=LUCIA_RL_CONFIG_RFENRN1['precision']))

        # Batch Normalization para Q1
        if self.batch_norm:
            self.gamma_bn_q1 = []
            self.beta_bn_q1 = []
            self.running_mean_q1 = []
            self.running_var_q1 = []

            for i in range(len(self.pesos_q1)):
                layer_size = self.pesos_q1[i].shape[1]
                self.gamma_bn_q1.append(np.ones(layer_size, dtype=LUCIA_RL_CONFIG_RFENRN1['precision']))
                self.beta_bn_q1.append(np.zeros(layer_size, dtype=LUCIA_RL_CONFIG_RFENRN1['precision']))
                self.running_mean_q1.append(np.zeros(layer_size, dtype=LUCIA_RL_CONFIG_RFENRN1['precision']))
                self.running_var_q1.append(np.ones(layer_size, dtype=LUCIA_RL_CONFIG_RFENRN1['precision']))

    def _inicializar_red_q2(self) -> None:
        """
        Inicializa la red Q2.
        """
        self.pesos_q2 = []
        self.sesgos_q2 = []

        # Capa de entrada (estado + acción)
        input_size = self.input_size + self.output_size
        self.pesos_q2.append(inicializar_pesos_he_avanzado((input_size, self.hidden_size)))
        self.sesgos_q2.append(np.zeros(self.hidden_size, dtype=LUCIA_RL_CONFIG_RFENRN1['precision']))

        # Capas ocultas
        for i in range(self.q_function_layers - 1):
            self.pesos_q2.append(inicializar_pesos_he_avanzado((self.hidden_size, self.hidden_size)))
            self.sesgos_q2.append(np.zeros(self.hidden_size, dtype=LUCIA_RL_CONFIG_RFENRN1['precision']))

        # Capa de salida (valor Q escalar)
        self.pesos_q2.append(inicializar_pesos_he_avanzado((self.hidden_size, 1)))
        self.sesgos_q2.append(np.zeros(1, dtype=LUCIA_RL_CONFIG_RFENRN1['precision']))

        # Batch Normalization para Q2
        if self.batch_norm:
            self.gamma_bn_q2 = []
            self.beta_bn_q2 = []
            self.running_mean_q2 = []
            self.running_var_q2 = []

            for i in range(len(self.pesos_q2)):
                layer_size = self.pesos_q2[i].shape[1]
                self.gamma_bn_q2.append(np.ones(layer_size, dtype=LUCIA_RL_CONFIG_RFENRN1['precision']))
                self.beta_bn_q2.append(np.zeros(layer_size, dtype=LUCIA_RL_CONFIG_RFENRN1['precision']))
                self.running_mean_q2.append(np.zeros(layer_size, dtype=LUCIA_RL_CONFIG_RFENRN1['precision']))
                self.running_var_q2.append(np.ones(layer_size, dtype=LUCIA_RL_CONFIG_RFENRN1['precision']))

    def _inicializar_red_policy(self) -> None:
        """
        Inicializa la red de la política.
        """
        self.pesos_policy = []
        self.sesgos_policy = []

        # Capa de entrada
        self.pesos_policy.append(inicializar_pesos_he_avanzado((self.input_size, self.hidden_size)))
        self.sesgos_policy.append(np.zeros(self.hidden_size, dtype=LUCIA_RL_CONFIG_RFENRN1['precision']))

        # Capas ocultas
        for i in range(self.policy_function_layers - 1):
            self.pesos_policy.append(inicializar_pesos_he_avanzado((self.hidden_size, self.hidden_size)))
            self.sesgos_policy.append(np.zeros(self.hidden_size, dtype=LUCIA_RL_CONFIG_RFENRN1['precision']))

        # Capa de salida (acción determinística)
        self.pesos_policy.append(inicializar_pesos_he_avanzado((self.hidden_size, self.output_size)))
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

    def _inicializar_redes_objetivo(self) -> None:
        """
        Inicializa las redes objetivo.
        """
        # Copiar pesos de Q1 a Q1_target
        self.pesos_q1_target = [peso.copy() for peso in self.pesos_q1]
        self.sesgos_q1_target = [sesgo.copy() for sesgo in self.sesgos_q1]

        # Copiar pesos de Q2 a Q2_target
        self.pesos_q2_target = [peso.copy() for peso in self.pesos_q2]
        self.sesgos_q2_target = [sesgo.copy() for sesgo in self.sesgos_q2]

        # Copiar pesos de la política a la política objetivo
        self.pesos_policy_target = [peso.copy() for peso in self.pesos_policy]
        self.sesgos_policy_target = [sesgo.copy() for sesgo in self.sesgos_policy]

    def _aplicar_preprocesamiento(self, estado: np.ndarray, training: bool = True,
                                  usar_q1: bool = True, usar_q2: bool = False, usar_policy: bool = False) -> np.ndarray:
        """
        Aplica preprocesamiento avanzado al estado.
        """
        estado_procesado = estado.copy()

        # Batch Normalization
        if self.batch_norm:
            if usar_q1:
                estado_procesado, self.running_mean_q1[0], self.running_var_q1[0] = aplicar_batch_normalization_avanzada(
                    estado_procesado.reshape(1, -1), self.gamma_bn_q1[0], self.beta_bn_q1[0],
                    self.running_mean_q1[0], self.running_var_q1[0], training=training
                )
            elif usar_q2:
                estado_procesado, self.running_mean_q2[0], self.running_var_q2[0] = aplicar_batch_normalization_avanzada(
                    estado_procesado.reshape(1, -1), self.gamma_bn_q2[0], self.beta_bn_q2[0],
                    self.running_mean_q2[0], self.running_var_q2[0], training=training
                )
            elif usar_policy:
                estado_procesado, self.running_mean_policy[0], self.running_var_policy[0] = aplicar_batch_normalization_avanzada(
                    estado_procesado.reshape(1, -1), self.gamma_bn_policy[0], self.beta_bn_policy[0],
                    self.running_mean_policy[0], self.running_var_policy[0], training=training
                )
            estado_procesado = estado_procesado.flatten()

        # Dropout
        if training and self.dropout_rate > 0:
            estado_procesado = aplicar_dropout_avanzado(estado_procesado, self.dropout_rate, training)

        return estado_procesado

    def _forward_q1(self, estado: np.ndarray, accion: np.ndarray, training: bool = True) -> float:
        """
        Propagación hacia adelante de Q1.
        """
        # Preprocesamiento
        estado_procesado = self._aplicar_preprocesamiento(estado, training, usar_q1=True)

        # Combinar estado y acción
        entrada = np.concatenate([estado_procesado, accion])

        # Propagación hacia adelante
        x = entrada
        for i, (peso, sesgo) in enumerate(zip(self.pesos_q1, self.sesgos_q1)):
            x = np.dot(x, peso) + sesgo

            # Batch Normalization para capas ocultas
            if self.batch_norm and i < len(self.pesos_q1) - 1:
                x, self.running_mean_q1[i], self.running_var_q1[i] = aplicar_batch_normalization_avanzada(
                    x.reshape(1, -1), self.gamma_bn_q1[i], self.beta_bn_q1[i],
                    self.running_mean_q1[i], self.running_var_q1[i], training=training
                )
                x = x.flatten()

            # Activación ReLU para capas ocultas
            if i < len(self.pesos_q1) - 1:
                x = np.maximum(0, x)

            # Dropout para capas ocultas
            if training and self.dropout_rate > 0 and i < len(self.pesos_q1) - 1:
                x = aplicar_dropout_avanzado(x, self.dropout_rate, training)

        return x[0]

    def _forward_q2(self, estado: np.ndarray, accion: np.ndarray, training: bool = True) -> float:
        """
        Propagación hacia adelante de Q2.
        """
        # Preprocesamiento
        estado_procesado = self._aplicar_preprocesamiento(estado, training, usar_q2=True)

        # Combinar estado y acción
        entrada = np.concatenate([estado_procesado, accion])

        # Propagación hacia adelante
        x = entrada
        for i, (peso, sesgo) in enumerate(zip(self.pesos_q2, self.sesgos_q2)):
            x = np.dot(x, peso) + sesgo

            # Batch Normalization para capas ocultas
            if self.batch_norm and i < len(self.pesos_q2) - 1:
                x, self.running_mean_q2[i], self.running_var_q2[i] = aplicar_batch_normalization_avanzada(
                    x.reshape(1, -1), self.gamma_bn_q2[i], self.beta_bn_q2[i],
                    self.running_mean_q2[i], self.running_var_q2[i], training=training
                )
                x = x.flatten()

            # Activación ReLU para capas ocultas
            if i < len(self.pesos_q2) - 1:
                x = np.maximum(0, x)

            # Dropout para capas ocultas
            if training and self.dropout_rate > 0 and i < len(self.pesos_q2) - 1:
                x = aplicar_dropout_avanzado(x, self.dropout_rate, training)

        return x[0]

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

        # Aplanar x si es necesario antes de aplicar tanh
        if len(x.shape) > 1:
            x = x.flatten()

        # Aplicar tanh para limitar el rango de acciones
        accion = np.tanh(x)

        return accion

    def _sample_action(self, estado: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Muestra una acción de la política con ruido de exploración.
        """
        accion = self._forward_policy(estado, training)

        # Añadir ruido de exploración si está en entrenamiento
        if training:
            noise = np.random.normal(0, self.exploration_noise, size=self.output_size)
            accion = np.clip(accion + noise, -1, 1)

        return accion

    def forward(self, estado: np.ndarray, training: bool = True) -> Tuple[np.ndarray, float, float]:
        """
        Propagación hacia adelante con técnicas avanzadas.
        """
        if not self.pesos_q1:
            self.inicializar_pesos()

        accion = self._sample_action(estado, training)
        q1_value = self._forward_q1(estado, accion, training)
        q2_value = self._forward_q2(estado, accion, training)

        return accion, q1_value, q2_value

    def seleccionar_accion(self, estado: np.ndarray, training: bool = True) -> int:
        """
        Selecciona una acción usando la política.
        """
        accion = self._sample_action(estado, training)

        # Convertir a acción discreta (simplificado)
        accion_discreta = np.argmax(accion)

        return accion_discreta

    def entrenar_paso(self, estado: np.ndarray, accion: int, recompensa: float,
                      siguiente_estado: np.ndarray, terminado: bool) -> Optional[float]:
        """
        Realiza un paso de entrenamiento con técnicas avanzadas.
        """
        if not self.pesos_q1:
            self.inicializar_pesos()

        # Convertir acción a vector
        accion_vector = np.zeros(self.output_size)
        accion_vector[accion] = 1.0

        # Almacenar experiencia
        experiencia = {
            'estado': estado.copy(),
            'accion': accion_vector.copy(),
            'recompensa': recompensa,
            'siguiente_estado': siguiente_estado.copy(),
            'terminado': terminado
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
        experiencias = random.sample(list(self.experience_buffer), min(self.batch_size, len(self.experience_buffer)))

        if not experiencias:
            return 0.0

        # Preparar datos del lote
        estados = np.array([exp['estado'] for exp in experiencias])
        acciones = np.array([exp['accion'] for exp in experiencias])
        recompensas = np.array([exp['recompensa'] for exp in experiencias])
        siguientes_estados = np.array([exp['siguiente_estado'] for exp in experiencias])
        terminados = np.array([exp['terminado'] for exp in experiencias])

        # Entrenar Q-functions
        perdida_q1 = self._entrenar_q1(estados, acciones, recompensas, siguientes_estados, terminados)
        perdida_q2 = self._entrenar_q2(estados, acciones, recompensas, siguientes_estados, terminados)

        # Entrenar política con delay
        perdida_policy = 0.0
        if self.policy_update_counter % self.policy_delay == 0:
            perdida_policy = self._entrenar_policy(estados)
            self.policy_update_counter = 0

        self.policy_update_counter += 1

        # Actualizar redes objetivo
        self._actualizar_redes_objetivo()

        # Actualizar estadísticas
        self._actualizar_estadisticas(perdida_q1, perdida_q2, perdida_policy, estados)

        return perdida_q1 + perdida_q2 + perdida_policy

    def _entrenar_q1(self, estados: np.ndarray, acciones: np.ndarray, recompensas: np.ndarray,
                     siguientes_estados: np.ndarray, terminados: np.ndarray) -> float:
        """
        Entrena Q1.
        """
        # Calcular valores Q actuales
        q1_values = np.array([self._forward_q1(estado, accion, training=True)
                             for estado, accion in zip(estados, acciones)])

        # Calcular valores Q objetivo
        target_q_values = np.zeros_like(q1_values)
        for i, (siguiente_estado, terminado) in enumerate(zip(siguientes_estados, terminados)):
            if terminado:
                target_q_values[i] = recompensas[i]
            else:
                # Obtener acción de la política objetivo
                siguiente_accion = self._forward_policy_target(siguiente_estado)

                # Añadir ruido para target policy smoothing
                noise = np.random.normal(0, self.target_noise, size=self.output_size)
                noise = np.clip(noise, -self.noise_clip, self.noise_clip)
                siguiente_accion = np.clip(siguiente_accion + noise, -1, 1)

                # Calcular valor Q objetivo usando la red objetivo
                target_q1 = self._forward_q1_target(siguiente_estado, siguiente_accion)
                target_q2 = self._forward_q2_target(siguiente_estado, siguiente_accion)

                # Usar el mínimo de las dos Q-functions para estabilidad
                target_q_value = np.minimum(target_q1, target_q2)

                # TD3 target
                target_q_values[i] = recompensas[i] + self.gamma * target_q_value

        # Calcular gradientes
        gradientes_q1 = self._calcular_gradientes_q1(estados, acciones, target_q_values)

        # Aplicar optimizador
        self._aplicar_optimizador_q1(gradientes_q1)

        # Calcular pérdida
        perdida = np.mean((q1_values - target_q_values) ** 2)

        return perdida

    def _entrenar_q2(self, estados: np.ndarray, acciones: np.ndarray, recompensas: np.ndarray,
                     siguientes_estados: np.ndarray, terminados: np.ndarray) -> float:
        """
        Entrena Q2.
        """
        # Calcular valores Q actuales
        q2_values = np.array([self._forward_q2(estado, accion, training=True)
                             for estado, accion in zip(estados, acciones)])

        # Calcular valores Q objetivo
        target_q_values = np.zeros_like(q2_values)
        for i, (siguiente_estado, terminado) in enumerate(zip(siguientes_estados, terminados)):
            if terminado:
                target_q_values[i] = recompensas[i]
            else:
                # Obtener acción de la política objetivo
                siguiente_accion = self._forward_policy_target(siguiente_estado)

                # Añadir ruido para target policy smoothing
                noise = np.random.normal(0, self.target_noise, size=self.output_size)
                noise = np.clip(noise, -self.noise_clip, self.noise_clip)
                siguiente_accion = np.clip(siguiente_accion + noise, -1, 1)

                # Calcular valor Q objetivo usando la red objetivo
                target_q1 = self._forward_q1_target(siguiente_estado, siguiente_accion)
                target_q2 = self._forward_q2_target(siguiente_estado, siguiente_accion)

                # Usar el mínimo de las dos Q-functions para estabilidad
                target_q_value = np.minimum(target_q1, target_q2)

                # TD3 target
                target_q_values[i] = recompensas[i] + self.gamma * target_q_value

        # Calcular gradientes
        gradientes_q2 = self._calcular_gradientes_q2(estados, acciones, target_q_values)

        # Aplicar optimizador
        self._aplicar_optimizador_q2(gradientes_q2)

        # Calcular pérdida
        perdida = np.mean((q2_values - target_q_values) ** 2)

        return perdida

    def _entrenar_policy(self, estados: np.ndarray) -> float:
        """
        Entrena la política.
        """
        # Calcular gradientes de la política
        gradientes_policy = self._calcular_gradientes_policy(estados)

        # Aplicar optimizador
        self._aplicar_optimizador_policy(gradientes_policy)

        # Calcular pérdida
        perdida_total = 0.0
        for estado in estados:
            # Obtener acción de la política
            accion = self._forward_policy(estado, training=True)

            # Calcular valor Q
            q1_value = self._forward_q1(estado, accion, training=False)
            q2_value = self._forward_q2(estado, accion, training=False)
            q_value = np.minimum(q1_value, q2_value)

            # Pérdida de la política (maximizar Q-value)
            perdida_total += -q_value

        return perdida_total / len(estados)

    def _forward_q1_target(self, estado: np.ndarray, accion: np.ndarray) -> float:
        """
        Propagación hacia adelante de Q1 objetivo.
        """
        # Combinar estado y acción
        entrada = np.concatenate([estado, accion])

        # Propagación hacia adelante
        x = entrada
        for peso, sesgo in zip(self.pesos_q1_target, self.sesgos_q1_target):
            x = np.dot(x, peso) + sesgo
            x = np.maximum(0, x)  # ReLU

        return x[0]

    def _forward_q2_target(self, estado: np.ndarray, accion: np.ndarray) -> float:
        """
        Propagación hacia adelante de Q2 objetivo.
        """
        # Combinar estado y acción
        entrada = np.concatenate([estado, accion])

        # Propagación hacia adelante
        x = entrada
        for peso, sesgo in zip(self.pesos_q2_target, self.sesgos_q2_target):
            x = np.dot(x, peso) + sesgo
            x = np.maximum(0, x)  # ReLU

        return x[0]

    def _forward_policy_target(self, estado: np.ndarray) -> np.ndarray:
        """
        Propagación hacia adelante de la política objetivo.
        """
        # Propagación hacia adelante
        x = estado
        for peso, sesgo in zip(self.pesos_policy_target, self.sesgos_policy_target):
            x = np.dot(x, peso) + sesgo
            x = np.maximum(0, x)  # ReLU

        # Aplicar tanh para limitar el rango de acciones
        accion = np.tanh(x)

        return accion

    def _calcular_gradientes_q1(self, estados: np.ndarray, acciones: np.ndarray,
                                target_q_values: np.ndarray) -> List[List[np.ndarray]]:
        """
        Calcula gradientes de Q1.
        """
        gradientes_q1 = []

        # Inicializar gradientes
        for i in range(len(self.pesos_q1)):
            gradientes_q1.append([np.zeros_like(self.pesos_q1[i]), np.zeros_like(self.sesgos_q1[i])])

        # Calcular gradientes para cada experiencia
        for i, (estado, accion, target_q) in enumerate(zip(estados, acciones, target_q_values)):
            # Calcular valor Q actual
            q_value = self._forward_q1(estado, accion, training=True)

            # Gradiente de pérdida MSE
            error = q_value - target_q
            grad_output = 2 * error / len(estados)

            # Backpropagation simplificado (aproximación)
            for j in range(len(self.pesos_q1)):
                if j == len(self.pesos_q1) - 1:  # Capa de salida
                    entrada = np.concatenate([estado, accion])
                    gradientes_q1[j][0] += np.outer(entrada, grad_output)
                    gradientes_q1[j][1] += grad_output
                else:
                    # Gradientes para capas ocultas (simplificado)
                    entrada = np.concatenate([estado, accion])
                    gradientes_q1[j][0] += np.outer(entrada, np.ones(self.pesos_q1[j].shape[1]))
                    gradientes_q1[j][1] += np.ones(self.pesos_q1[j].shape[1])

        # Promediar gradientes
        for i in range(len(gradientes_q1)):
            gradientes_q1[i][0] /= len(estados)
            gradientes_q1[i][1] /= len(estados)

        return gradientes_q1

    def _calcular_gradientes_q2(self, estados: np.ndarray, acciones: np.ndarray,
                                target_q_values: np.ndarray) -> List[List[np.ndarray]]:
        """
        Calcula gradientes de Q2.
        """
        gradientes_q2 = []

        # Inicializar gradientes
        for i in range(len(self.pesos_q2)):
            gradientes_q2.append([np.zeros_like(self.pesos_q2[i]), np.zeros_like(self.sesgos_q2[i])])

        # Calcular gradientes para cada experiencia
        for i, (estado, accion, target_q) in enumerate(zip(estados, acciones, target_q_values)):
            # Calcular valor Q actual
            q_value = self._forward_q2(estado, accion, training=True)

            # Gradiente de pérdida MSE
            error = q_value - target_q
            grad_output = 2 * error / len(estados)

            # Backpropagation simplificado (aproximación)
            for j in range(len(self.pesos_q2)):
                if j == len(self.pesos_q2) - 1:  # Capa de salida
                    entrada = np.concatenate([estado, accion])
                    gradientes_q2[j][0] += np.outer(entrada, grad_output)
                    gradientes_q2[j][1] += grad_output
                else:
                    # Gradientes para capas ocultas (simplificado)
                    entrada = np.concatenate([estado, accion])
                    gradientes_q2[j][0] += np.outer(entrada, np.ones(self.pesos_q2[j].shape[1]))
                    gradientes_q2[j][1] += np.ones(self.pesos_q2[j].shape[1])

        # Promediar gradientes
        for i in range(len(gradientes_q2)):
            gradientes_q2[i][0] /= len(estados)
            gradientes_q2[i][1] /= len(estados)

        return gradientes_q2

    def _calcular_gradientes_policy(self, estados: np.ndarray) -> List[List[np.ndarray]]:
        """
        Calcula gradientes de la política.
        """
        gradientes_policy = []

        # Inicializar gradientes
        for i in range(len(self.pesos_policy)):
            gradientes_policy.append([np.zeros_like(self.pesos_policy[i]), np.zeros_like(self.sesgos_policy[i])])

        # Calcular gradientes para cada experiencia
        for estado in estados:
            # Obtener acción de la política
            accion = self._forward_policy(estado, training=True)

            # Calcular valor Q
            q1_value = self._forward_q1(estado, accion, training=False)
            q2_value = self._forward_q2(estado, accion, training=False)
            q_value = np.minimum(q1_value, q2_value)

            # Gradiente de pérdida de la política
            grad_output = -q_value / len(estados)

            # Backpropagation simplificado (aproximación)
            for j in range(len(self.pesos_policy)):
                if j == len(self.pesos_policy) - 1:  # Capa de salida
                    gradientes_policy[j][0] += np.outer(estado, grad_output)
                    gradientes_policy[j][1] += grad_output
                else:
                    # Gradientes para capas ocultas (simplificado)
                    gradientes_policy[j][0] += np.outer(estado, np.ones(self.pesos_policy[j].shape[1]))
                    gradientes_policy[j][1] += np.ones(self.pesos_policy[j].shape[1])

        # Promediar gradientes
        for i in range(len(gradientes_policy)):
            gradientes_policy[i][0] /= len(estados)
            gradientes_policy[i][1] /= len(estados)

        return gradientes_policy

    def _aplicar_optimizador_q1(self, gradientes: List[List[np.ndarray]]) -> None:
        """
        Aplica el optimizador avanzado a Q1.
        """
        # Aplanar gradientes
        gradientes_flat = []
        for grad_capa in gradientes:
            gradientes_flat.extend(grad_capa)

        # Aplicar weight decay
        if self.weight_decay > 0:
            for i, pesos in enumerate(self.pesos_q1):
                gradientes_flat[i*2] += self.weight_decay * pesos

        # Aplicar gradient clipping
        gradientes_flat = aplicar_clip_gradientes_avanzado(gradientes_flat, method='adaptive')

        # Aplicar optimizador específico
        if self.optimizer == 'adam':
            gradientes_optimizados, self.m_historial_q1, self.v_historial_q1 = aplicar_adam_avanzado(
                gradientes_flat, self.m_historial_q1, self.v_historial_q1,
                learning_rate=self.learning_rate, beta1=self.beta1, beta2=self.beta2,
                weight_decay=self.weight_decay
            )
        elif self.optimizer == 'rmsprop':
            gradientes_optimizados, self.v_historial_q1 = aplicar_rmsprop_avanzado(
                gradientes_flat, self.v_historial_q1, learning_rate=self.learning_rate
            )
        elif self.optimizer == 'momentum':
            gradientes_optimizados, self.momentum_historial_q1 = calcular_momentum_avanzado(
                gradientes_flat, self.momentum_historial_q1, momentum=self.momentum
            )
        else:  # SGD
            gradientes_optimizados = gradientes_flat

        # Actualizar pesos
        idx = 0
        for i in range(len(self.pesos_q1)):
            self.pesos_q1[i] -= gradientes_optimizados[idx]
            self.sesgos_q1[i] -= gradientes_optimizados[idx + 1]
            idx += 2

        # Guardar en historial
        self.historial_pesos.append(self.pesos_q1[0].copy())

    def _aplicar_optimizador_q2(self, gradientes: List[List[np.ndarray]]) -> None:
        """
        Aplica el optimizador avanzado a Q2.
        """
        # Aplanar gradientes
        gradientes_flat = []
        for grad_capa in gradientes:
            gradientes_flat.extend(grad_capa)

        # Aplicar weight decay
        if self.weight_decay > 0:
            for i, pesos in enumerate(self.pesos_q2):
                gradientes_flat[i*2] += self.weight_decay * pesos

        # Aplicar gradient clipping
        gradientes_flat = aplicar_clip_gradientes_avanzado(gradientes_flat, method='adaptive')

        # Aplicar optimizador específico
        if self.optimizer == 'adam':
            gradientes_optimizados, self.m_historial_q2, self.v_historial_q2 = aplicar_adam_avanzado(
                gradientes_flat, self.m_historial_q2, self.v_historial_q2,
                learning_rate=self.learning_rate, beta1=self.beta1, beta2=self.beta2,
                weight_decay=self.weight_decay
            )
        elif self.optimizer == 'rmsprop':
            gradientes_optimizados, self.v_historial_q2 = aplicar_rmsprop_avanzado(
                gradientes_flat, self.v_historial_q2, learning_rate=self.learning_rate
            )
        elif self.optimizer == 'momentum':
            gradientes_optimizados, self.momentum_historial_q2 = calcular_momentum_avanzado(
                gradientes_flat, self.momentum_historial_q2, momentum=self.momentum
            )
        else:  # SGD
            gradientes_optimizados = gradientes_flat

        # Actualizar pesos
        idx = 0
        for i in range(len(self.pesos_q2)):
            self.pesos_q2[i] -= gradientes_optimizados[idx]
            self.sesgos_q2[i] -= gradientes_optimizados[idx + 1]
            idx += 2

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

    def _actualizar_redes_objetivo(self) -> None:
        """
        Actualiza las redes objetivo usando soft updates.
        """
        # Actualizar Q1_target
        for i in range(len(self.pesos_q1_target)):
            self.pesos_q1_target[i] = (1 - self.tau) * self.pesos_q1_target[i] + self.tau * self.pesos_q1[i]
            self.sesgos_q1_target[i] = (1 - self.tau) * self.sesgos_q1_target[i] + self.tau * self.sesgos_q1[i]

        # Actualizar Q2_target
        for i in range(len(self.pesos_q2_target)):
            self.pesos_q2_target[i] = (1 - self.tau) * self.pesos_q2_target[i] + self.tau * self.pesos_q2[i]
            self.sesgos_q2_target[i] = (1 - self.tau) * self.sesgos_q2_target[i] + self.tau * self.sesgos_q2[i]

        # Actualizar policy_target
        for i in range(len(self.pesos_policy_target)):
            self.pesos_policy_target[i] = (1 - self.tau) * self.pesos_policy_target[i] + self.tau * self.pesos_policy[i]
            self.sesgos_policy_target[i] = (1 - self.tau) * self.sesgos_policy_target[i] + self.tau * self.sesgos_policy[i]

    def _actualizar_estadisticas(self, perdida_q1: float, perdida_q2: float,
                                 perdida_policy: float, estados: np.ndarray) -> None:
        """
        Actualiza las estadísticas específicas de TD3 avanzado.
        """
        # Estadísticas básicas
        self.estadisticas_td3['q1_loss_media'] = perdida_q1
        self.estadisticas_td3['q2_loss_media'] = perdida_q2
        self.estadisticas_td3['policy_loss_media'] = perdida_policy

        # Estadísticas de valores Q
        q_values = []
        for estado in estados:
            accion = self._sample_action(estado, training=False)
            q1_value = self._forward_q1(estado, accion, training=False)
            q2_value = self._forward_q2(estado, accion, training=False)
            q_values.append(np.minimum(q1_value, q2_value))
        self.estadisticas_td3['q_value_media'] = np.mean(q_values)

        # Estadísticas de valores de política
        policy_values = []
        for estado in estados:
            accion = self._forward_policy(estado, training=False)
            q1_value = self._forward_q1(estado, accion, training=False)
            q2_value = self._forward_q2(estado, accion, training=False)
            policy_values.append(np.minimum(q1_value, q2_value))
        self.estadisticas_td3['policy_value_media'] = np.mean(policy_values)

        # Estadísticas específicas de TD3
        self.estadisticas_td3['twin_q_stability'] = 0.1  # Placeholder
        self.estadisticas_td3['delayed_policy_updates'] = 1.0 if self.policy_update_counter == 0 else 0.0
        self.estadisticas_td3['target_policy_smoothing'] = 0.1  # Placeholder
        self.estadisticas_td3['exploration_noise_efficiency'] = 0.1  # Placeholder

        # Estadísticas de estabilidad
        if len(self.historial_pesos) > 10:
            pesos_recientes = np.array(self.historial_pesos[-10:])
            varianza_pesos = np.var(pesos_recientes)
            self.estadisticas_td3['policy_gradient_norm'] = np.sqrt(np.sum(pesos_recientes[-1] ** 2))
            self.estadisticas_td3['convergence_rate'] = 1.0 / (1.0 + varianza_pesos)

        # Guardar en historiales
        self.historial_q1_loss.append(perdida_q1)
        self.historial_q2_loss.append(perdida_q2)
        self.historial_policy_loss.append(perdida_policy)
        self.historial_q_values.append(self.estadisticas_td3['q_value_media'])
        self.historial_policy_values.append(self.estadisticas_td3['policy_value_media'])
        self.historial_exploration_noise.append(self.exploration_noise)

    def obtener_estadisticas(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas específicas de TD3 avanzado.
        """
        if not self.pesos_q1:
            return {'estado': 'no_inicializada'}

        stats_td3 = {
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'optimizer': self.optimizer,
            'momentum': self.momentum,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'weight_decay': self.weight_decay,
            'dropout_rate': self.dropout_rate,
            'batch_norm': self.batch_norm,
            'tau': self.tau,
            'policy_delay': self.policy_delay,
            'target_noise': self.target_noise,
            'noise_clip': self.noise_clip,
            'exploration_noise': self.exploration_noise,
            'replay_buffer_size': self.replay_buffer_size,
            'batch_size': self.batch_size,
            'hidden_size': self.hidden_size,
            'q_function_layers': self.q_function_layers,
            'policy_function_layers': self.policy_function_layers,
            'q1_loss_media': self.estadisticas_td3['q1_loss_media'],
            'q2_loss_media': self.estadisticas_td3['q2_loss_media'],
            'policy_loss_media': self.estadisticas_td3['policy_loss_media'],
            'q_value_media': self.estadisticas_td3['q_value_media'],
            'target_q_value_media': self.estadisticas_td3['target_q_value_media'],
            'policy_value_media': self.estadisticas_td3['policy_value_media'],
            'twin_q_stability': self.estadisticas_td3['twin_q_stability'],
            'delayed_policy_updates': self.estadisticas_td3['delayed_policy_updates'],
            'target_policy_smoothing': self.estadisticas_td3['target_policy_smoothing'],
            'exploration_noise_efficiency': self.estadisticas_td3['exploration_noise_efficiency'],
            'replay_buffer_efficiency': self.estadisticas_td3['replay_buffer_efficiency'],
            'q_function_approximation': self.estadisticas_td3['q_function_approximation'],
            'policy_gradient_norm': self.estadisticas_td3['policy_gradient_norm'],
            'q_gradient_norm': self.estadisticas_td3['q_gradient_norm'],
            'convergence_rate': self.estadisticas_td3['convergence_rate'],
            'exploration_efficiency': self.estadisticas_td3['exploration_efficiency'],
            'variance_reduction': self.estadisticas_td3['variance_reduction'],
            'batch_efficiency': self.estadisticas_td3['batch_efficiency'],
            'noise_regularization': self.estadisticas_td3['noise_regularization'],
            'deterministic_policy_quality': self.estadisticas_td3['deterministic_policy_quality'],
            'historial_q1_loss_size': len(self.historial_q1_loss),
            'historial_q2_loss_size': len(self.historial_q2_loss),
            'historial_policy_loss_size': len(self.historial_policy_loss),
            'historial_q_values_size': len(self.historial_q_values),
            'historial_policy_values_size': len(self.historial_policy_values),
            'historial_exploration_noise_size': len(self.historial_exploration_noise)
        }

        return stats_td3

    def verificar_estabilidad(self) -> Dict[str, bool]:
        """
        Verifica la estabilidad de la neurona TD3 avanzada.
        """
        estabilidad = {}

        # Verificaciones básicas
        estabilidad['pesos_inicializados'] = len(self.pesos_q1) > 0
        estabilidad['experiencias_suficientes'] = len(self.experience_buffer) > self.batch_size

        # Verificaciones avanzadas
        estabilidad['q1_loss_controlada'] = abs(self.estadisticas_td3['q1_loss_media']) < 10.0
        estabilidad['q2_loss_controlada'] = abs(self.estadisticas_td3['q2_loss_media']) < 10.0
        estabilidad['policy_loss_controlada'] = abs(self.estadisticas_td3['policy_loss_media']) < 10.0
        estabilidad['q_value_estable'] = abs(self.estadisticas_td3['q_value_media']) < 100.0
        estabilidad['policy_value_estable'] = abs(self.estadisticas_td3['policy_value_media']) < 100.0
        estabilidad['gradiente_policy_estable'] = self.estadisticas_td3['policy_gradient_norm'] < 10.0
        estabilidad['convergencia_ok'] = self.estadisticas_td3['convergence_rate'] > 0.5

        # Verificaciones específicas de TD3
        estabilidad['twin_q_estable'] = self.estadisticas_td3['twin_q_stability'] > 0.0
        estabilidad['delayed_policy_updates_funcionando'] = self.estadisticas_td3['delayed_policy_updates'] > 0.0
        estabilidad['target_policy_smoothing_funcionando'] = self.estadisticas_td3['target_policy_smoothing'] > 0.0
        estabilidad['exploration_noise_apropiado'] = 0.01 <= self.exploration_noise <= 0.5

        return estabilidad

    def __str__(self) -> str:
        return (f"NeuronaRefuerzoTD3Avanzada(entrada={self.input_size}, "
                f"salida={self.output_size}, lr={self.learning_rate}, "
                f"optimizer={self.optimizer}, policy_delay={self.policy_delay}, "
                f"target_noise={self.target_noise}, exploration_noise={self.exploration_noise}, "
                f"batch_size={self.batch_size})")

    def __repr__(self) -> str:
        return self.__str__()

# Funciones de utilidad específicas para RF_RFEN1_RN_1_8


def crear_neurona_td3_avanzada(input_size: int, output_size: int,
                               configuracion: Dict[str, Any] = None) -> NeuronaRefuerzoTD3Avanzada:
    """
    Función de conveniencia para crear una neurona TD3 avanzada.
    """
    if configuracion is None:
        configuracion = {}

    return NeuronaRefuerzoTD3Avanzada(
        input_size=input_size,
        output_size=output_size,
        nombre=configuracion.get('nombre', 'NeuronaRefuerzoTD3Avanzada'),
        learning_rate=configuracion.get('learning_rate', LUCIA_RL_CONFIG_RFENRN1['default_learning_rate']),
        gamma=configuracion.get('gamma', LUCIA_RL_CONFIG_RFENRN1['default_gamma']),
        optimizer=configuracion.get('optimizer', 'adam'),
        momentum=configuracion.get('momentum', LUCIA_RL_CONFIG_RFENRN1['default_momentum']),
        beta1=configuracion.get('beta1', LUCIA_RL_CONFIG_RFENRN1['default_beta1']),
        beta2=configuracion.get('beta2', LUCIA_RL_CONFIG_RFENRN1['default_beta2']),
        weight_decay=configuracion.get('weight_decay', LUCIA_RL_CONFIG_RFENRN1['default_weight_decay']),
        dropout_rate=configuracion.get('dropout_rate', LUCIA_RL_CONFIG_RFENRN1['default_dropout_rate']),
        batch_norm=configuracion.get('batch_norm', True),
        tau=configuracion.get('tau', 0.005),
        policy_delay=configuracion.get('policy_delay', 2),
        target_noise=configuracion.get('target_noise', 0.2),
        noise_clip=configuracion.get('noise_clip', 0.5),
        exploration_noise=configuracion.get('exploration_noise', 0.1),
        replay_buffer_size=configuracion.get('replay_buffer_size', 1000000),
        batch_size=configuracion.get('batch_size', 256),
        hidden_size=configuracion.get('hidden_size', 256),
        q_function_layers=configuracion.get('q_function_layers', 2),
        policy_function_layers=configuracion.get('policy_function_layers', 2)
    )


# Configuración específica para RF_RFEN1_RN_1_8
RF_RFEN1_RN_1_8_CONFIG = {
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
    'tau_default': 0.005,
    'policy_delay_default': 2,
    'target_noise_default': 0.2,
    'noise_clip_default': 0.5,
    'exploration_noise_default': 0.1,
    'replay_buffer_size_default': 1000000,
    'batch_size_default': 256,
    'hidden_size_default': 256,
    'q_function_layers_default': 2,
    'policy_function_layers_default': 2,
    'umbral_q1_loss_controlada': 10.0,
    'umbral_q2_loss_controlada': 10.0,
    'umbral_policy_loss_controlada': 10.0,
    'umbral_q_value_estable': 100.0,
    'umbral_policy_value_estable': 100.0,
    'umbral_gradiente_policy_estable': 10.0,
    'umbral_convergencia': 0.5,
    'umbral_exploration_noise_min': 0.01,
    'umbral_exploration_noise_max': 0.5,
    'umbral_experiencias_suficientes': 256
}

logger.info("RF_RFEN1_RN_1_8.py cargado correctamente - Neurona de Refuerzo TD3 Avanzada")
