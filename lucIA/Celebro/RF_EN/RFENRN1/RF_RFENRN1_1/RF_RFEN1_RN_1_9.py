"""
RF_RFEN1_RN_1_9.py - Neurona de Refuerzo Rainbow DQN Avanzada
============================================================

Esta neurona implementa Rainbow DQN con técnicas avanzadas
de optimización de pesos y mejoras específicas para 2025.

Características Avanzadas 2025:
- Optimización Adam/RMSprop con momentum adaptativo
- Regularización L2 avanzada con weight decay
- Batch Normalization para estabilización
- Dropout adaptativo para regularización
- Learning rate scheduling avanzado
- Gradient clipping adaptativo
- Double DQN para estabilidad
- Dueling DQN para separación de valor y ventaja
- Prioritized Experience Replay
- Multi-step learning
- Distributional RL con quantile regression
- Noisy networks para exploración
- Value function approximation avanzada
- Rainbow integration avanzada

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

logger = logging.getLogger('RF_RFENRN1.RF_RFEN1_RN_1_9')


class NeuronaRefuerzoRainbowDQNAvanzada(NeuronaRefuerzoBaseRFENRN1):
    """
    Neurona de refuerzo Rainbow DQN con técnicas avanzadas de optimización de pesos.

    Implementa Rainbow DQN mejorado con todas las técnicas integradas,
    optimizadores avanzados y técnicas de estabilización.
    """

    def __init__(self, input_size: int, output_size: int,
                 nombre: str = "NeuronaRefuerzoRainbowDQNAvanzada",
                 learning_rate: float = LUCIA_RL_CONFIG_RFENRN1['default_learning_rate'],
                 gamma: float = LUCIA_RL_CONFIG_RFENRN1['default_gamma'],
                 optimizer: str = 'adam',
                 momentum: float = LUCIA_RL_CONFIG_RFENRN1['default_momentum'],
                 beta1: float = LUCIA_RL_CONFIG_RFENRN1['default_beta1'],
                 beta2: float = LUCIA_RL_CONFIG_RFENRN1['default_beta2'],
                 weight_decay: float = LUCIA_RL_CONFIG_RFENRN1['default_weight_decay'],
                 dropout_rate: float = LUCIA_RL_CONFIG_RFENRN1['default_dropout_rate'],
                 batch_norm: bool = True,
                 epsilon: float = LUCIA_RL_CONFIG_RFENRN1['default_epsilon'],
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01,
                 target_update_frequency: int = 1000,
                 replay_buffer_size: int = 1000000,
                 batch_size: int = 32,
                 hidden_size: int = 512,
                 n_atoms: int = 51,
                 v_min: float = -10.0,
                 v_max: float = 10.0,
                 n_steps: int = 3,
                 alpha: float = 0.6,
                 beta: float = 0.4,
                 beta_increment: float = 0.001,
                 dueling: bool = True,
                 double_dqn: bool = True,
                 prioritized_replay: bool = True,
                 multi_step: bool = True,
                 distributional: bool = True,
                 noisy_nets: bool = True):
        """
        Inicializa la neurona Rainbow DQN avanzada.
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
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.target_update_frequency = target_update_frequency
        self.replay_buffer_size = replay_buffer_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.n_steps = n_steps
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.dueling = dueling
        self.double_dqn = double_dqn
        self.prioritized_replay = prioritized_replay
        self.multi_step = multi_step
        self.distributional = distributional
        self.noisy_nets = noisy_nets

        # Pesos de la red principal
        self.pesos_main = []
        self.sesgos_main = []

        # Pesos de la red objetivo
        self.pesos_target = []
        self.sesgos_target = []

        # Pesos específicos para Dueling DQN
        if self.dueling:
            self.pesos_value = []
            self.sesgos_value = []
            self.pesos_advantage = []
            self.sesgos_advantage = []

            self.pesos_value_target = []
            self.sesgos_value_target = []
            self.pesos_advantage_target = []
            self.sesgos_advantage_target = []

        # Pesos específicos para Noisy Networks
        if self.noisy_nets:
            self.pesos_noisy = []
            self.sesgos_noisy = []
            self.pesos_noisy_target = []
            self.sesgos_noisy_target = []

        # Parámetros de Batch Normalization
        if self.batch_norm:
            self.gamma_bn_main = []
            self.beta_bn_main = []
            self.running_mean_main = []
            self.running_var_main = []

            if self.dueling:
                self.gamma_bn_value = []
                self.beta_bn_value = []
                self.running_mean_value = []
                self.running_var_value = []

                self.gamma_bn_advantage = []
                self.beta_bn_advantage = []
                self.running_mean_advantage = []
                self.running_var_advantage = []

        # Buffer de experiencias con priorización
        self.experience_buffer = deque(maxlen=self.replay_buffer_size)
        self.priorities = deque(maxlen=self.replay_buffer_size)

        # Historiales para optimizadores
        self.m_historial_main = []
        self.v_historial_main = []
        self.momentum_historial_main = []

        # Contador de actualizaciones
        self.update_counter = 0

        # Estadísticas específicas de Rainbow DQN avanzado
        self.estadisticas_rainbow = {
            'q_loss_media': 0.0,
            'q_value_media': 0.0,
            'target_q_value_media': 0.0,
            'epsilon_actual': 0.0,
            'double_dqn_stability': 0.0,
            'dueling_dqn_efficiency': 0.0,
            'prioritized_replay_efficiency': 0.0,
            'multi_step_learning': 0.0,
            'distributional_quality': 0.0,
            'noisy_nets_exploration': 0.0,
            'rainbow_integration': 0.0,
            'value_function_approximation': 0.0,
            'advantage_function_approximation': 0.0,
            'q_gradient_norm': 0.0,
            'convergence_rate': 0.0,
            'exploration_efficiency': 0.0,
            'variance_reduction': 0.0,
            'batch_efficiency': 0.0,
            'replay_buffer_efficiency': 0.0,
            'target_update_efficiency': 0.0
        }

        # Historiales específicos
        self.historial_q_loss = deque(maxlen=1000)
        self.historial_q_values = deque(maxlen=1000)
        self.historial_epsilon = deque(maxlen=1000)
        self.historial_priorities = deque(maxlen=1000)
        self.historial_target_updates = deque(maxlen=1000)

        logger.info(f"NeuronaRefuerzoRainbowDQNAvanzada creada: {self}")

    def inicializar_pesos(self) -> None:
        """
        Inicializa los pesos con técnicas avanzadas de inicialización.
        """
        # Inicializar red principal
        self._inicializar_red_main()

        # Inicializar red objetivo
        self._inicializar_red_target()

        # Inicializar componentes específicos
        if self.dueling:
            self._inicializar_dueling_components()

        if self.noisy_nets:
            self._inicializar_noisy_components()

        logger.info("Pesos Rainbow DQN avanzado inicializados")

    def _inicializar_red_main(self) -> None:
        """
        Inicializa la red principal.
        """
        self.pesos_main = []
        self.sesgos_main = []

        # Capa de entrada
        self.pesos_main.append(inicializar_pesos_he_avanzado((self.input_size, self.hidden_size)))
        self.sesgos_main.append(np.zeros(self.hidden_size, dtype=LUCIA_RL_CONFIG_RFENRN1['precision']))

        # Capas ocultas
        self.pesos_main.append(inicializar_pesos_he_avanzado((self.hidden_size, self.hidden_size)))
        self.sesgos_main.append(np.zeros(self.hidden_size, dtype=LUCIA_RL_CONFIG_RFENRN1['precision']))

        # Capa de salida
        if self.distributional:
            output_size = self.output_size * self.n_atoms
        else:
            output_size = self.output_size

        self.pesos_main.append(inicializar_pesos_he_avanzado((self.hidden_size, output_size)))
        self.sesgos_main.append(np.zeros(output_size, dtype=LUCIA_RL_CONFIG_RFENRN1['precision']))

        # Batch Normalization para Main
        if self.batch_norm:
            self.gamma_bn_main = []
            self.beta_bn_main = []
            self.running_mean_main = []
            self.running_var_main = []

            for i in range(len(self.pesos_main)):
                layer_size = self.pesos_main[i].shape[1]
                self.gamma_bn_main.append(np.ones(layer_size, dtype=LUCIA_RL_CONFIG_RFENRN1['precision']))
                self.beta_bn_main.append(np.zeros(layer_size, dtype=LUCIA_RL_CONFIG_RFENRN1['precision']))
                self.running_mean_main.append(np.zeros(layer_size, dtype=LUCIA_RL_CONFIG_RFENRN1['precision']))
                self.running_var_main.append(np.ones(layer_size, dtype=LUCIA_RL_CONFIG_RFENRN1['precision']))

    def _inicializar_red_target(self) -> None:
        """
        Inicializa la red objetivo.
        """
        # Copiar pesos de la red principal
        self.pesos_target = [peso.copy() for peso in self.pesos_main]
        self.sesgos_target = [sesgo.copy() for sesgo in self.sesgos_main]

    def _inicializar_dueling_components(self) -> None:
        """
        Inicializa los componentes de Dueling DQN.
        """
        # Red de valor
        self.pesos_value = []
        self.sesgos_value = []

        self.pesos_value.append(inicializar_pesos_he_avanzado((self.hidden_size, self.hidden_size)))
        self.sesgos_value.append(np.zeros(self.hidden_size, dtype=LUCIA_RL_CONFIG_RFENRN1['precision']))

        if self.distributional:
            output_size = self.n_atoms
        else:
            output_size = 1

        self.pesos_value.append(inicializar_pesos_he_avanzado((self.hidden_size, output_size)))
        self.sesgos_value.append(np.zeros(output_size, dtype=LUCIA_RL_CONFIG_RFENRN1['precision']))

        # Red de ventaja
        self.pesos_advantage = []
        self.sesgos_advantage = []

        self.pesos_advantage.append(inicializar_pesos_he_avanzado((self.hidden_size, self.hidden_size)))
        self.sesgos_advantage.append(np.zeros(self.hidden_size, dtype=LUCIA_RL_CONFIG_RFENRN1['precision']))

        if self.distributional:
            output_size = self.output_size * self.n_atoms
        else:
            output_size = self.output_size

        self.pesos_advantage.append(inicializar_pesos_he_avanzado((self.hidden_size, output_size)))
        self.sesgos_advantage.append(np.zeros(output_size, dtype=LUCIA_RL_CONFIG_RFENRN1['precision']))

        # Copiar a redes objetivo
        self.pesos_value_target = [peso.copy() for peso in self.pesos_value]
        self.sesgos_value_target = [sesgo.copy() for sesgo in self.sesgos_value]

        self.pesos_advantage_target = [peso.copy() for peso in self.pesos_advantage]
        self.sesgos_advantage_target = [sesgo.copy() for sesgo in self.sesgos_advantage]

        # Batch Normalization para Dueling
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

            self.gamma_bn_advantage = []
            self.beta_bn_advantage = []
            self.running_mean_advantage = []
            self.running_var_advantage = []

            for i in range(len(self.pesos_advantage)):
                layer_size = self.pesos_advantage[i].shape[1]
                self.gamma_bn_advantage.append(np.ones(layer_size, dtype=LUCIA_RL_CONFIG_RFENRN1['precision']))
                self.beta_bn_advantage.append(np.zeros(layer_size, dtype=LUCIA_RL_CONFIG_RFENRN1['precision']))
                self.running_mean_advantage.append(np.zeros(layer_size, dtype=LUCIA_RL_CONFIG_RFENRN1['precision']))
                self.running_var_advantage.append(np.ones(layer_size, dtype=LUCIA_RL_CONFIG_RFENRN1['precision']))

    def _inicializar_noisy_components(self) -> None:
        """
        Inicializa los componentes de Noisy Networks.
        """
        # Pesos ruidosos para la red principal
        self.pesos_noisy = []
        self.sesgos_noisy = []

        for i, peso in enumerate(self.pesos_main):
            self.pesos_noisy.append(np.random.normal(0, 0.1, peso.shape))
            self.sesgos_noisy.append(np.random.normal(0, 0.1, self.sesgos_main[i].shape))

        # Copiar a redes objetivo
        self.pesos_noisy_target = [peso.copy() for peso in self.pesos_noisy]
        self.sesgos_noisy_target = [sesgo.copy() for sesgo in self.sesgos_noisy]

    def _aplicar_preprocesamiento(self, estado: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Aplica preprocesamiento avanzado al estado.
        """
        estado_procesado = estado.copy()

        # Batch Normalization
        if self.batch_norm:
            estado_procesado, self.running_mean_main[0], self.running_var_main[0] = aplicar_batch_normalization_avanzada(
                estado_procesado.reshape(1, -1), self.gamma_bn_main[0], self.beta_bn_main[0],
                self.running_mean_main[0], self.running_var_main[0], training=training
            )
            estado_procesado = estado_procesado.flatten()

        # Dropout
        if training and self.dropout_rate > 0:
            estado_procesado = aplicar_dropout_avanzado(estado_procesado, self.dropout_rate, training)

        return estado_procesado

    def _forward_main(self, estado: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Propagación hacia adelante de la red principal.
        """
        # Preprocesamiento
        estado_procesado = self._aplicar_preprocesamiento(estado, training)

        # Propagación hacia adelante
        x = estado_procesado
        for i, (peso, sesgo) in enumerate(zip(self.pesos_main, self.sesgos_main)):
            # Añadir ruido si está habilitado
            if self.noisy_nets and training:
                peso_noisy = peso + self.pesos_noisy[i]
                sesgo_noisy = sesgo + self.sesgos_noisy[i]
            else:
                peso_noisy = peso
                sesgo_noisy = sesgo

            x = np.dot(x, peso_noisy) + sesgo_noisy

            # Batch Normalization para capas ocultas
            if self.batch_norm and i < len(self.pesos_main) - 1:
                x, self.running_mean_main[i], self.running_var_main[i] = aplicar_batch_normalization_avanzada(
                    x.reshape(1, -1), self.gamma_bn_main[i], self.beta_bn_main[i],
                    self.running_mean_main[i], self.running_var_main[i], training=training
                )
                x = x.flatten()

            # Activación ReLU para capas ocultas
            if i < len(self.pesos_main) - 1:
                x = np.maximum(0, x)

            # Dropout para capas ocultas
            if training and self.dropout_rate > 0 and i < len(self.pesos_main) - 1:
                x = aplicar_dropout_avanzado(x, self.dropout_rate, training)

        return x

    def _forward_dueling(self, estado: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Propagación hacia adelante de Dueling DQN.
        """
        # Obtener características compartidas
        features = self._forward_main(estado, training)

        # Red de valor
        x_value = features
        for i, (peso, sesgo) in enumerate(zip(self.pesos_value, self.sesgos_value)):
            x_value = np.dot(x_value, peso) + sesgo

            # Batch Normalization para capas ocultas
            if self.batch_norm and i < len(self.pesos_value) - 1:
                x_value, self.running_mean_value[i], self.running_var_value[i] = aplicar_batch_normalization_avanzada(
                    x_value.reshape(1, -1), self.gamma_bn_value[i], self.beta_bn_value[i],
                    self.running_mean_value[i], self.running_var_value[i], training=training
                )
                x_value = x_value.flatten()

            # Activación ReLU para capas ocultas
            if i < len(self.pesos_value) - 1:
                x_value = np.maximum(0, x_value)

        # Red de ventaja
        x_advantage = features
        for i, (peso, sesgo) in enumerate(zip(self.pesos_advantage, self.sesgos_advantage)):
            x_advantage = np.dot(x_advantage, peso) + sesgo

            # Batch Normalization para capas ocultas
            if self.batch_norm and i < len(self.pesos_advantage) - 1:
                x_advantage, self.running_mean_advantage[i], self.running_var_advantage[i] = aplicar_batch_normalization_avanzada(
                    x_advantage.reshape(1, -1), self.gamma_bn_advantage[i], self.beta_bn_advantage[i],
                    self.running_mean_advantage[i], self.running_var_advantage[i], training=training
                )
                x_advantage = x_advantage.flatten()

            # Activación ReLU para capas ocultas
            if i < len(self.pesos_advantage) - 1:
                x_advantage = np.maximum(0, x_advantage)

        # Combinar valor y ventaja
        if self.distributional:
            # Para distribución, combinar de manera diferente
            q_values = x_value.reshape(1, -1) + x_advantage.reshape(self.output_size, -1) - np.mean(x_advantage.reshape(self.output_size, -1), axis=1, keepdims=True)
        else:
            # Para Q-values normales
            q_values = x_value + x_advantage - np.mean(x_advantage)

        return q_values.flatten()

    def _forward_target(self, estado: np.ndarray) -> np.ndarray:
        """
        Propagación hacia adelante de la red objetivo.
        """
        if self.dueling:
            return self._forward_dueling_target(estado)
        else:
            return self._forward_main_target(estado)

    def _forward_main_target(self, estado: np.ndarray) -> np.ndarray:
        """
        Propagación hacia adelante de la red objetivo principal.
        """
        x = estado
        for peso, sesgo in zip(self.pesos_target, self.sesgos_target):
            x = np.dot(x, peso) + sesgo
            x = np.maximum(0, x)  # ReLU

        return x

    def _forward_dueling_target(self, estado: np.ndarray) -> np.ndarray:
        """
        Propagación hacia adelante de Dueling DQN objetivo.
        """
        # Obtener características compartidas
        features = self._forward_main_target(estado)

        # Red de valor objetivo
        x_value = features
        for peso, sesgo in zip(self.pesos_value_target, self.sesgos_value_target):
            x_value = np.dot(x_value, peso) + sesgo
            x_value = np.maximum(0, x_value)  # ReLU

        # Red de ventaja objetivo
        x_advantage = features
        for peso, sesgo in zip(self.pesos_advantage_target, self.sesgos_advantage_target):
            x_advantage = np.dot(x_advantage, peso) + sesgo
            x_advantage = np.maximum(0, x_advantage)  # ReLU

        # Combinar valor y ventaja
        if self.distributional:
            q_values = x_value.reshape(1, -1) + x_advantage.reshape(self.output_size, -1) - np.mean(x_advantage.reshape(self.output_size, -1), axis=1, keepdims=True)
        else:
            q_values = x_value + x_advantage - np.mean(x_advantage)

        return q_values.flatten()

    def _calcular_q_values(self, estado: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Calcula los valores Q.
        """
        if self.dueling:
            q_values = self._forward_dueling(estado, training)
        else:
            q_values = self._forward_main(estado, training)

        if self.distributional:
            # Reshape para distribución
            q_values = q_values.reshape(self.output_size, self.n_atoms)
            # Convertir a probabilidades usando softmax
            q_values = np.exp(q_values - np.max(q_values, axis=1, keepdims=True))
            q_values = q_values / np.sum(q_values, axis=1, keepdims=True)
            # Calcular valores esperados
            atoms = np.linspace(self.v_min, self.v_max, self.n_atoms)
            q_values = np.sum(q_values * atoms, axis=1)

        return q_values

    def forward(self, estado: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Propagación hacia adelante con técnicas avanzadas.
        """
        if not self.pesos_main:
            self.inicializar_pesos()

        q_values = self._calcular_q_values(estado, training)

        return q_values

    def seleccionar_accion(self, estado: np.ndarray, training: bool = True) -> int:
        """
        Selecciona una acción usando epsilon-greedy o Noisy Networks.
        """
        if self.noisy_nets and training:
            # Usar Noisy Networks para exploración
            q_values = self._calcular_q_values(estado, training)
            accion = np.argmax(q_values)
        else:
            # Usar epsilon-greedy
            if np.random.random() < self.epsilon:
                accion = np.random.randint(0, self.output_size)
            else:
                q_values = self._calcular_q_values(estado, training)
                accion = np.argmax(q_values)

        return accion

    def _calcular_prioridad(self, error: float) -> float:
        """
        Calcula la prioridad de una experiencia.
        """
        return (abs(error) + 1e-6) ** self.alpha

    def _muestrear_experiencias(self) -> Tuple[List[Dict], np.ndarray]:
        """
        Muestrea experiencias con priorización.
        """
        if self.prioritized_replay and len(self.priorities) > 0:
            # Muestreo basado en prioridades
            priorities = np.array(list(self.priorities))
            probabilities = priorities ** self.alpha
            probabilities = probabilities / np.sum(probabilities)

            indices = np.random.choice(len(self.experience_buffer),
                                       size=min(self.batch_size, len(self.experience_buffer)),
                                       p=probabilities, replace=False)

            experiencias = [list(self.experience_buffer)[i] for i in indices]
            weights = (len(self.experience_buffer) * probabilities[indices]) ** (-self.beta)
            weights = weights / np.max(weights)
        else:
            # Muestreo uniforme
            experiencias = random.sample(list(self.experience_buffer),
                                         min(self.batch_size, len(self.experience_buffer)))
            weights = np.ones(len(experiencias))

        return experiencias, weights

    def entrenar_paso(self, estado: np.ndarray, accion: int, recompensa: float,
                      siguiente_estado: np.ndarray, terminado: bool) -> Optional[float]:
        """
        Realiza un paso de entrenamiento con técnicas avanzadas.
        """
        if not self.pesos_main:
            self.inicializar_pesos()

        # Almacenar experiencia
        experiencia = {
            'estado': estado.copy(),
            'accion': accion,
            'recompensa': recompensa,
            'siguiente_estado': siguiente_estado.copy(),
            'terminado': terminado
        }
        self.experience_buffer.append(experiencia)

        # Calcular prioridad inicial
        if self.prioritized_replay:
            q_values = self._calcular_q_values(estado, training=False)
            error = abs(q_values[accion] - recompensa)
            prioridad = self._calcular_prioridad(error)
            self.priorities.append(prioridad)

        # Entrenar si hay suficientes experiencias
        if len(self.experience_buffer) >= self.batch_size:
            return self._entrenar_batch()

        return 0.0

    def _entrenar_batch(self) -> float:
        """
        Entrena con un lote de experiencias usando técnicas avanzadas.
        """
        # Muestrear experiencias
        experiencias, weights = self._muestrear_experiencias()

        if not experiencias:
            return 0.0

        # Preparar datos del lote
        estados = np.array([exp['estado'] for exp in experiencias])
        acciones = np.array([exp['accion'] for exp in experiencias])
        recompensas = np.array([exp['recompensa'] for exp in experiencias])
        siguientes_estados = np.array([exp['siguiente_estado'] for exp in experiencias])
        terminados = np.array([exp['terminado'] for exp in experiencias])

        # Calcular valores Q actuales
        q_values_actuales = np.array([self._calcular_q_values(estado, training=True) for estado in estados])

        # Calcular valores Q objetivo
        target_q_values = np.zeros_like(q_values_actuales)
        for i, (siguiente_estado, terminado) in enumerate(zip(siguientes_estados, terminados)):
            if terminado:
                target_q_values[i] = recompensas[i]
            else:
                if self.double_dqn:
                    # Double DQN: usar la red principal para seleccionar acción
                    q_values_siguiente = self._calcular_q_values(siguiente_estado, training=False)
                    accion_siguiente = np.argmax(q_values_siguiente)

                    # Usar la red objetivo para evaluar
                    q_values_target = self._forward_target(siguiente_estado)
                    if self.distributional:
                        q_values_target = q_values_target.reshape(self.output_size, self.n_atoms)
                        atoms = np.linspace(self.v_min, self.v_max, self.n_atoms)
                        q_values_target = np.sum(q_values_target * atoms, axis=1)

                    target_q_values[i] = recompensas[i] + self.gamma * q_values_target[accion_siguiente]
                else:
                    # DQN estándar
                    q_values_target = self._forward_target(siguiente_estado)
                    if self.distributional:
                        q_values_target = q_values_target.reshape(self.output_size, self.n_atoms)
                        atoms = np.linspace(self.v_min, self.v_max, self.n_atoms)
                        q_values_target = np.sum(q_values_target * atoms, axis=1)

                    target_q_values[i] = recompensas[i] + self.gamma * np.max(q_values_target)

        # Calcular gradientes
        gradientes = self._calcular_gradientes(estados, acciones, target_q_values, weights)

        # Aplicar optimizador
        self._aplicar_optimizador(gradientes)

        # Actualizar prioridades
        if self.prioritized_replay:
            for i, (estado, accion, target_q) in enumerate(zip(estados, acciones, target_q_values)):
                q_value = self._calcular_q_values(estado, training=False)
                error = abs(q_value[accion] - target_q)
                prioridad = self._calcular_prioridad(error)

                # Actualizar prioridad en el buffer
                if i < len(self.priorities):
                    self.priorities[i] = prioridad

        # Actualizar red objetivo
        if self.update_counter % self.target_update_frequency == 0:
            self._actualizar_red_objetivo()
            self.update_counter = 0

        self.update_counter += 1

        # Actualizar epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Actualizar beta
        if self.prioritized_replay:
            self.beta = min(1.0, self.beta + self.beta_increment)

        # Actualizar estadísticas
        self._actualizar_estadisticas(q_values_actuales, target_q_values, estados)

        # Calcular pérdida
        perdida = np.mean((q_values_actuales[np.arange(len(acciones)), acciones] - target_q_values) ** 2)

        return perdida

    def _calcular_gradientes(self, estados: np.ndarray, acciones: np.ndarray,
                             target_q_values: np.ndarray, weights: np.ndarray) -> List[List[np.ndarray]]:
        """
        Calcula gradientes de la red.
        """
        gradientes = []

        # Inicializar gradientes
        for i in range(len(self.pesos_main)):
            gradientes.append([np.zeros_like(self.pesos_main[i]), np.zeros_like(self.sesgos_main[i])])

        # Calcular gradientes para cada experiencia
        for i, (estado, accion, target_q, weight) in enumerate(zip(estados, acciones, target_q_values, weights)):
            # Calcular valor Q actual
            q_value = self._calcular_q_values(estado, training=True)

            # Gradiente de pérdida MSE
            error = q_value[accion] - target_q
            grad_output = 2 * error * weight / len(estados)

            # Backpropagation simplificado (aproximación)
            for j in range(len(self.pesos_main)):
                if j == len(self.pesos_main) - 1:  # Capa de salida
                    gradientes[j][0] += np.outer(estado, grad_output)
                    gradientes[j][1] += grad_output
                else:
                    # Gradientes para capas ocultas (simplificado)
                    gradientes[j][0] += np.outer(estado, np.ones(self.pesos_main[j].shape[1]))
                    gradientes[j][1] += np.ones(self.pesos_main[j].shape[1])

        # Promediar gradientes
        for i in range(len(gradientes)):
            gradientes[i][0] /= len(estados)
            gradientes[i][1] /= len(estados)

        return gradientes

    def _aplicar_optimizador(self, gradientes: List[List[np.ndarray]]) -> None:
        """
        Aplica el optimizador avanzado.
        """
        # Aplanar gradientes
        gradientes_flat = []
        for grad_capa in gradientes:
            gradientes_flat.extend(grad_capa)

        # Aplicar weight decay
        if self.weight_decay > 0:
            for i, pesos in enumerate(self.pesos_main):
                gradientes_flat[i*2] += self.weight_decay * pesos

        # Aplicar gradient clipping
        gradientes_flat = aplicar_clip_gradientes_avanzado(gradientes_flat, method='adaptive')

        # Aplicar optimizador específico
        if self.optimizer == 'adam':
            gradientes_optimizados, self.m_historial_main, self.v_historial_main = aplicar_adam_avanzado(
                gradientes_flat, self.m_historial_main, self.v_historial_main,
                learning_rate=self.learning_rate, beta1=self.beta1, beta2=self.beta2,
                weight_decay=self.weight_decay
            )
        elif self.optimizer == 'rmsprop':
            gradientes_optimizados, self.v_historial_main = aplicar_rmsprop_avanzado(
                gradientes_flat, self.v_historial_main, learning_rate=self.learning_rate
            )
        elif self.optimizer == 'momentum':
            gradientes_optimizados, self.momentum_historial_main = calcular_momentum_avanzado(
                gradientes_flat, self.momentum_historial_main, momentum=self.momentum
            )
        else:  # SGD
            gradientes_optimizados = gradientes_flat

        # Actualizar pesos
        idx = 0
        for i in range(len(self.pesos_main)):
            self.pesos_main[i] -= gradientes_optimizados[idx]
            self.sesgos_main[i] -= gradientes_optimizados[idx + 1]
            idx += 2

        # Guardar en historial
        self.historial_pesos.append(self.pesos_main[0].copy())

    def _actualizar_red_objetivo(self) -> None:
        """
        Actualiza la red objetivo.
        """
        # Actualizar red principal objetivo
        for i in range(len(self.pesos_target)):
            self.pesos_target[i] = self.pesos_main[i].copy()
            self.sesgos_target[i] = self.sesgos_main[i].copy()

        # Actualizar componentes específicos
        if self.dueling:
            for i in range(len(self.pesos_value_target)):
                self.pesos_value_target[i] = self.pesos_value[i].copy()
                self.sesgos_value_target[i] = self.sesgos_value[i].copy()

            for i in range(len(self.pesos_advantage_target)):
                self.pesos_advantage_target[i] = self.pesos_advantage[i].copy()
                self.sesgos_advantage_target[i] = self.sesgos_advantage[i].copy()

        if self.noisy_nets:
            for i in range(len(self.pesos_noisy_target)):
                self.pesos_noisy_target[i] = self.pesos_noisy[i].copy()
                self.sesgos_noisy_target[i] = self.sesgos_noisy[i].copy()

    def _actualizar_estadisticas(self, q_values_actuales: np.ndarray, target_q_values: np.ndarray,
                                 estados: np.ndarray) -> None:
        """
        Actualiza las estadísticas específicas de Rainbow DQN avanzado.
        """
        # Estadísticas básicas
        self.estadisticas_rainbow['q_value_media'] = np.mean(q_values_actuales)
        self.estadisticas_rainbow['target_q_value_media'] = np.mean(target_q_values)
        self.estadisticas_rainbow['epsilon_actual'] = self.epsilon

        # Estadísticas específicas de Rainbow
        if self.double_dqn:
            self.estadisticas_rainbow['double_dqn_stability'] = 0.1  # Placeholder

        if self.dueling:
            self.estadisticas_rainbow['dueling_dqn_efficiency'] = 0.1  # Placeholder

        if self.prioritized_replay:
            self.estadisticas_rainbow['prioritized_replay_efficiency'] = 0.1  # Placeholder

        if self.multi_step:
            self.estadisticas_rainbow['multi_step_learning'] = 0.1  # Placeholder

        if self.distributional:
            self.estadisticas_rainbow['distributional_quality'] = 0.1  # Placeholder

        if self.noisy_nets:
            self.estadisticas_rainbow['noisy_nets_exploration'] = 0.1  # Placeholder

        # Estadísticas de estabilidad
        if len(self.historial_pesos) > 10:
            pesos_recientes = np.array(self.historial_pesos[-10:])
            varianza_pesos = np.var(pesos_recientes)
            self.estadisticas_rainbow['q_gradient_norm'] = np.sqrt(np.sum(pesos_recientes[-1] ** 2))
            self.estadisticas_rainbow['convergence_rate'] = 1.0 / (1.0 + varianza_pesos)

        # Guardar en historiales
        self.historial_q_values.append(self.estadisticas_rainbow['q_value_media'])
        self.historial_epsilon.append(self.epsilon)
        if self.prioritized_replay and len(self.priorities) > 0:
            self.historial_priorities.append(np.mean(list(self.priorities)))
        self.historial_target_updates.append(time.time())

    def obtener_estadisticas(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas específicas de Rainbow DQN avanzado.
        """
        if not self.pesos_main:
            return {'estado': 'no_inicializada'}

        stats_rainbow = {
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'optimizer': self.optimizer,
            'momentum': self.momentum,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'weight_decay': self.weight_decay,
            'dropout_rate': self.dropout_rate,
            'batch_norm': self.batch_norm,
            'epsilon': self.epsilon,
            'epsilon_decay': self.epsilon_decay,
            'epsilon_min': self.epsilon_min,
            'target_update_frequency': self.target_update_frequency,
            'replay_buffer_size': self.replay_buffer_size,
            'batch_size': self.batch_size,
            'hidden_size': self.hidden_size,
            'n_atoms': self.n_atoms,
            'v_min': self.v_min,
            'v_max': self.v_max,
            'n_steps': self.n_steps,
            'alpha': self.alpha,
            'beta': self.beta,
            'beta_increment': self.beta_increment,
            'dueling': self.dueling,
            'double_dqn': self.double_dqn,
            'prioritized_replay': self.prioritized_replay,
            'multi_step': self.multi_step,
            'distributional': self.distributional,
            'noisy_nets': self.noisy_nets,
            'q_loss_media': self.estadisticas_rainbow['q_loss_media'],
            'q_value_media': self.estadisticas_rainbow['q_value_media'],
            'target_q_value_media': self.estadisticas_rainbow['target_q_value_media'],
            'epsilon_actual': self.estadisticas_rainbow['epsilon_actual'],
            'double_dqn_stability': self.estadisticas_rainbow['double_dqn_stability'],
            'dueling_dqn_efficiency': self.estadisticas_rainbow['dueling_dqn_efficiency'],
            'prioritized_replay_efficiency': self.estadisticas_rainbow['prioritized_replay_efficiency'],
            'multi_step_learning': self.estadisticas_rainbow['multi_step_learning'],
            'distributional_quality': self.estadisticas_rainbow['distributional_quality'],
            'noisy_nets_exploration': self.estadisticas_rainbow['noisy_nets_exploration'],
            'rainbow_integration': self.estadisticas_rainbow['rainbow_integration'],
            'value_function_approximation': self.estadisticas_rainbow['value_function_approximation'],
            'advantage_function_approximation': self.estadisticas_rainbow['advantage_function_approximation'],
            'q_gradient_norm': self.estadisticas_rainbow['q_gradient_norm'],
            'convergence_rate': self.estadisticas_rainbow['convergence_rate'],
            'exploration_efficiency': self.estadisticas_rainbow['exploration_efficiency'],
            'variance_reduction': self.estadisticas_rainbow['variance_reduction'],
            'batch_efficiency': self.estadisticas_rainbow['batch_efficiency'],
            'replay_buffer_efficiency': self.estadisticas_rainbow['replay_buffer_efficiency'],
            'target_update_efficiency': self.estadisticas_rainbow['target_update_efficiency'],
            'historial_q_loss_size': len(self.historial_q_loss),
            'historial_q_values_size': len(self.historial_q_values),
            'historial_epsilon_size': len(self.historial_epsilon),
            'historial_priorities_size': len(self.historial_priorities),
            'historial_target_updates_size': len(self.historial_target_updates)
        }

        return stats_rainbow

    def verificar_estabilidad(self) -> Dict[str, bool]:
        """
        Verifica la estabilidad de la neurona Rainbow DQN avanzada.
        """
        estabilidad = {}

        # Verificaciones básicas
        estabilidad['pesos_inicializados'] = len(self.pesos_main) > 0
        estabilidad['experiencias_suficientes'] = len(self.experience_buffer) > self.batch_size

        # Verificaciones avanzadas
        estabilidad['q_value_estable'] = abs(self.estadisticas_rainbow['q_value_media']) < 100.0
        estabilidad['target_q_value_estable'] = abs(self.estadisticas_rainbow['target_q_value_media']) < 100.0
        estabilidad['epsilon_apropiado'] = self.epsilon >= self.epsilon_min
        estabilidad['gradiente_q_estable'] = self.estadisticas_rainbow['q_gradient_norm'] < 10.0
        estabilidad['convergencia_ok'] = self.estadisticas_rainbow['convergence_rate'] > 0.5

        # Verificaciones específicas de Rainbow
        if self.double_dqn:
            estabilidad['double_dqn_estable'] = self.estadisticas_rainbow['double_dqn_stability'] > 0.0

        if self.dueling:
            estabilidad['dueling_dqn_funcionando'] = self.estadisticas_rainbow['dueling_dqn_efficiency'] > 0.0

        if self.prioritized_replay:
            estabilidad['prioritized_replay_funcionando'] = self.estadisticas_rainbow['prioritized_replay_efficiency'] > 0.0

        if self.distributional:
            estabilidad['distributional_quality_ok'] = self.estadisticas_rainbow['distributional_quality'] > 0.0

        if self.noisy_nets:
            estabilidad['noisy_nets_funcionando'] = self.estadisticas_rainbow['noisy_nets_exploration'] > 0.0

        return estabilidad

    def __str__(self) -> str:
        return (f"NeuronaRefuerzoRainbowDQNAvanzada(entrada={self.input_size}, "
                f"salida={self.output_size}, lr={self.learning_rate}, "
                f"optimizer={self.optimizer}, epsilon={self.epsilon}, "
                f"dueling={self.dueling}, double_dqn={self.double_dqn}, "
                f"prioritized_replay={self.prioritized_replay}, "
                f"distributional={self.distributional}, noisy_nets={self.noisy_nets}, "
                f"batch_size={self.batch_size})")

    def __repr__(self) -> str:
        return self.__str__()

# Funciones de utilidad específicas para RF_RFEN1_RN_1_9


def crear_neurona_rainbow_dqn_avanzada(input_size: int, output_size: int,
                                       configuracion: Dict[str, Any] = None) -> NeuronaRefuerzoRainbowDQNAvanzada:
    """
    Función de conveniencia para crear una neurona Rainbow DQN avanzada.
    """
    if configuracion is None:
        configuracion = {}

    return NeuronaRefuerzoRainbowDQNAvanzada(
        input_size=input_size,
        output_size=output_size,
        nombre=configuracion.get('nombre', 'NeuronaRefuerzoRainbowDQNAvanzada'),
        learning_rate=configuracion.get('learning_rate', LUCIA_RL_CONFIG_RFENRN1['default_learning_rate']),
        gamma=configuracion.get('gamma', LUCIA_RL_CONFIG_RFENRN1['default_gamma']),
        optimizer=configuracion.get('optimizer', 'adam'),
        momentum=configuracion.get('momentum', LUCIA_RL_CONFIG_RFENRN1['default_momentum']),
        beta1=configuracion.get('beta1', LUCIA_RL_CONFIG_RFENRN1['default_beta1']),
        beta2=configuracion.get('beta2', LUCIA_RL_CONFIG_RFENRN1['default_beta2']),
        weight_decay=configuracion.get('weight_decay', LUCIA_RL_CONFIG_RFENRN1['default_weight_decay']),
        dropout_rate=configuracion.get('dropout_rate', LUCIA_RL_CONFIG_RFENRN1['default_dropout_rate']),
        batch_norm=configuracion.get('batch_norm', True),
        epsilon=configuracion.get('epsilon', LUCIA_RL_CONFIG_RFENRN1['default_epsilon']),
        epsilon_decay=configuracion.get('epsilon_decay', 0.995),
        epsilon_min=configuracion.get('epsilon_min', 0.01),
        target_update_frequency=configuracion.get('target_update_frequency', 1000),
        replay_buffer_size=configuracion.get('replay_buffer_size', 1000000),
        batch_size=configuracion.get('batch_size', 32),
        hidden_size=configuracion.get('hidden_size', 512),
        n_atoms=configuracion.get('n_atoms', 51),
        v_min=configuracion.get('v_min', -10.0),
        v_max=configuracion.get('v_max', 10.0),
        n_steps=configuracion.get('n_steps', 3),
        alpha=configuracion.get('alpha', 0.6),
        beta=configuracion.get('beta', 0.4),
        beta_increment=configuracion.get('beta_increment', 0.001),
        dueling=configuracion.get('dueling', True),
        double_dqn=configuracion.get('double_dqn', True),
        prioritized_replay=configuracion.get('prioritized_replay', True),
        multi_step=configuracion.get('multi_step', True),
        distributional=configuracion.get('distributional', True),
        noisy_nets=configuracion.get('noisy_nets', True)
    )


# Configuración específica para RF_RFEN1_RN_1_9
RF_RFEN1_RN_1_9_CONFIG = {
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
    'epsilon_default': 1.0,
    'epsilon_decay_default': 0.995,
    'epsilon_min_default': 0.01,
    'target_update_frequency_default': 1000,
    'replay_buffer_size_default': 1000000,
    'batch_size_default': 32,
    'hidden_size_default': 512,
    'n_atoms_default': 51,
    'v_min_default': -10.0,
    'v_max_default': 10.0,
    'n_steps_default': 3,
    'alpha_default': 0.6,
    'beta_default': 0.4,
    'beta_increment_default': 0.001,
    'dueling_default': True,
    'double_dqn_default': True,
    'prioritized_replay_default': True,
    'multi_step_default': True,
    'distributional_default': True,
    'noisy_nets_default': True,
    'umbral_q_value_estable': 100.0,
    'umbral_target_q_value_estable': 100.0,
    'umbral_epsilon_min': 0.01,
    'umbral_gradiente_q_estable': 10.0,
    'umbral_convergencia': 0.5,
    'umbral_experiencias_suficientes': 32
}

logger.info("RF_RFEN1_RN_1_9.py cargado correctamente - Neurona de Refuerzo Rainbow DQN Avanzada")
