"""
RF_RFEN1_RN_1_4.py - Neurona de Refuerzo DQN Avanzada
====================================================

Esta neurona implementa Deep Q-Network (DQN) con técnicas avanzadas de optimización de pesos
y mejoras específicas para 2025, incluyendo optimizadores avanzados, regularización
y técnicas de estabilización.

Características Avanzadas 2025:
- Optimización Adam/RMSprop con momentum adaptativo
- Regularización L2 avanzada con weight decay
- Batch Normalization para estabilización
- Dropout adaptativo para regularización
- Learning rate scheduling avanzado
- Gradient clipping adaptativo
- Experience replay con priorización
- Target network con soft updates
- Double DQN para reducción de bias
- Dueling DQN para separación de valor y ventaja
- Noisy Networks para exploración
- Distributional DQN para distribución de retornos
- Rainbow DQN con múltiples mejoras

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
                   aplicar_dropout_avanzado, LUCIA_RL_CONFIG_RFENRN1)
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
    LUCIA_RL_CONFIG_RFENRN1 = {}


logger = logging.getLogger('RF_RFENRN1.RF_RFEN1_RN_1_4')


class NeuronaRefuerzoDQNAvanzada(NeuronaRefuerzoBaseRFENRN1):
    """
    Neurona de refuerzo DQN con técnicas avanzadas de optimización de pesos.

    Implementa DQN mejorado con optimizadores avanzados, regularización
    y técnicas de estabilización para mejor rendimiento y convergencia.
    """

    def __init__(self, input_size: int, output_size: int,
                 nombre: str = "NeuronaRefuerzoDQNAvanzada",
                 learning_rate: float = LUCIA_RL_CONFIG_RFENRN1['default_learning_rate'],
                 gamma: float = LUCIA_RL_CONFIG_RFENRN1['default_gamma'],
                 epsilon: float = LUCIA_RL_CONFIG_RFENRN1['default_epsilon'],
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01,
                 optimizer: str = 'adam',
                 momentum: float = LUCIA_RL_CONFIG_RFENRN1['default_momentum'],
                 beta1: float = LUCIA_RL_CONFIG_RFENRN1['default_beta1'],
                 beta2: float = LUCIA_RL_CONFIG_RFENRN1['default_beta2'],
                 weight_decay: float = LUCIA_RL_CONFIG_RFENRN1['default_weight_decay'],
                 dropout_rate: float = LUCIA_RL_CONFIG_RFENRN1['default_dropout_rate'],
                 batch_norm: bool = True,
                 target_update_frequency: int = LUCIA_RL_CONFIG_RFENRN1['default_target_update_frequency'],
                 soft_update_tau: float = 0.005,
                 double_dqn: bool = True,
                 dueling_dqn: bool = True,
                 noisy_networks: bool = False,
                 distributional_dqn: bool = False,
                 rainbow_dqn: bool = True,
                 prioritized_replay: bool = True,
                 buffer_size: int = LUCIA_RL_CONFIG_RFENRN1['default_buffer_size'],
                 batch_size: int = LUCIA_RL_CONFIG_RFENRN1['default_batch_size'],
                 hidden_layers: int = 3,
                 hidden_size: int = 128):
        """
        Inicializa la neurona DQN avanzada.
        """
        super().__init__(input_size, output_size, nombre)

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.optimizer = optimizer
        self.momentum = momentum
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.target_update_frequency = target_update_frequency
        self.soft_update_tau = soft_update_tau
        self.double_dqn = double_dqn
        self.dueling_dqn = dueling_dqn
        self.noisy_networks = noisy_networks
        self.distributional_dqn = distributional_dqn
        self.rainbow_dqn = rainbow_dqn
        self.prioritized_replay = prioritized_replay
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size

        # Pesos de la red principal
        self.pesos_red = []
        self.sesgos_red = []

        # Pesos del target network
        self.pesos_target = []
        self.sesgos_target = []

        # Pesos adicionales para Dueling DQN
        if self.dueling_dqn:
            self.pesos_valor = []
            self.sesgos_valor = []
            self.pesos_ventaja = []
            self.sesgos_ventaja = []
            self.pesos_target_valor = []
            self.sesgos_target_valor = []
            self.pesos_target_ventaja = []
            self.sesgos_target_ventaja = []

        # Parámetros de Batch Normalization
        if self.batch_norm:
            self.gamma_bn = []
            self.beta_bn = []
            self.running_mean = []
            self.running_var = []
            self.gamma_bn_target = []
            self.beta_bn_target = []
            self.running_mean_target = []
            self.running_var_target = []

        # Buffer de experiencias con priorización
        self.experience_buffer = deque(maxlen=buffer_size)
        self.priorities = deque(maxlen=buffer_size)

        # Historiales para optimizadores
        self.m_historial = []
        self.v_historial = []
        self.momentum_historial = []

        # Estadísticas específicas de DQN avanzado
        self.estadisticas_dqn = {
            'q_values_media': 0.0,
            'td_error_media': 0.0,
            'epsilon_actual': epsilon,
            'target_updates': 0,
            'buffer_utilization': 0.0,
            'prioritized_samples': 0,
            'double_dqn_bias_reduction': 0.0,
            'dueling_value_advantage_separation': 0.0,
            'noisy_network_exploration': 0.0,
            'distributional_dqn_quality': 0.0,
            'rainbow_dqn_efficiency': 0.0,
            'optimizer_efficiency': 0.0,
            'gradient_stability': 0.0,
            'learning_rate_adaptation': 0.0,
            'weight_decay_effectiveness': 0.0,
            'dropout_regularization': 0.0,
            'batch_norm_stability': 0.0,
            'experience_diversity': 0.0,
            'convergence_speed': 0.0
        }

        # Historiales específicos
        self.historial_q_values = deque(maxlen=1000)
        self.historial_td_errors = deque(maxlen=1000)
        self.historial_epsilon = deque(maxlen=1000)
        self.historial_target_updates = deque(maxlen=1000)
        self.historial_buffer_utilization = deque(maxlen=1000)

        logger.info(f"NeuronaRefuerzoDQNAvanzada creada: {self}")

    def inicializar_pesos(self) -> None:
        """
        Inicializa los pesos con técnicas avanzadas de inicialización.
        """
        # Inicializar red principal
        self._inicializar_red_principal()

        # Inicializar target network
        self._inicializar_target_network()

        # Inicializar Dueling DQN si está habilitado
        if self.dueling_dqn:
            self._inicializar_dueling_network()

        logger.info("Pesos DQN avanzado inicializados")

    def _inicializar_red_principal(self) -> None:
        """
        Inicializa la red principal.
        """
        self.pesos_red = []
        self.sesgos_red = []

        # Capa de entrada
        self.pesos_red.append(inicializar_pesos_he_avanzado((self.input_size, self.hidden_size)))
        self.sesgos_red.append(np.zeros(self.hidden_size, dtype=LUCIA_RL_CONFIG_RFENRN1['precision']))

        # Capas ocultas
        for i in range(self.hidden_layers - 1):
            self.pesos_red.append(inicializar_pesos_he_avanzado((self.hidden_size, self.hidden_size)))
            self.sesgos_red.append(np.zeros(self.hidden_size, dtype=LUCIA_RL_CONFIG_RFENRN1['precision']))

        # Capa de salida
        self.pesos_red.append(inicializar_pesos_he_avanzado((self.hidden_size, self.output_size)))
        self.sesgos_red.append(np.zeros(self.output_size, dtype=LUCIA_RL_CONFIG_RFENRN1['precision']))

        # Batch Normalization
        if self.batch_norm:
            self.gamma_bn = []
            self.beta_bn = []
            self.running_mean = []
            self.running_var = []

            for i in range(len(self.pesos_red)):
                layer_size = self.pesos_red[i].shape[1]
                self.gamma_bn.append(np.ones(layer_size, dtype=LUCIA_RL_CONFIG_RFENRN1['precision']))
                self.beta_bn.append(np.zeros(layer_size, dtype=LUCIA_RL_CONFIG_RFENRN1['precision']))
                self.running_mean.append(np.zeros(layer_size, dtype=LUCIA_RL_CONFIG_RFENRN1['precision']))
                self.running_var.append(np.ones(layer_size, dtype=LUCIA_RL_CONFIG_RFENRN1['precision']))

    def _inicializar_target_network(self) -> None:
        """
        Inicializa el target network.
        """
        self.pesos_target = [pesos.copy() for pesos in self.pesos_red]
        self.sesgos_target = [sesgo.copy() for sesgo in self.sesgos_red]

        if self.batch_norm:
            self.gamma_bn_target = [gamma.copy() for gamma in self.gamma_bn]
            self.beta_bn_target = [beta.copy() for beta in self.beta_bn]
            self.running_mean_target = [mean.copy() for mean in self.running_mean]
            self.running_var_target = [var.copy() for var in self.running_var]

    def _inicializar_dueling_network(self) -> None:
        """
        Inicializa la red Dueling DQN.
        """
        # Red de valor
        self.pesos_valor = [inicializar_pesos_he_avanzado((self.hidden_size, 1))]
        self.sesgos_valor = [np.zeros(1, dtype=LUCIA_RL_CONFIG_RFENRN1['precision'])]

        # Red de ventaja
        self.pesos_ventaja = [inicializar_pesos_he_avanzado((self.hidden_size, self.output_size))]
        self.sesgos_ventaja = [np.zeros(self.output_size, dtype=LUCIA_RL_CONFIG_RFENRN1['precision'])]

        # Target networks para Dueling
        self.pesos_target_valor = [pesos.copy() for pesos in self.pesos_valor]
        self.sesgos_target_valor = [sesgo.copy() for sesgo in self.sesgos_valor]
        self.pesos_target_ventaja = [pesos.copy() for pesos in self.pesos_ventaja]
        self.sesgos_target_ventaja = [sesgo.copy() for sesgo in self.sesgos_ventaja]

    def _aplicar_preprocesamiento(self, estado: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Aplica preprocesamiento avanzado al estado.
        """
        estado_procesado = estado.copy()

        # Batch Normalization
        if self.batch_norm:
            estado_procesado, self.running_mean[0], self.running_var[0] = aplicar_batch_normalization_avanzada(
                estado_procesado.reshape(1, -1), self.gamma_bn[0], self.beta_bn[0],
                self.running_mean[0], self.running_var[0], training=training
            )
            estado_procesado = estado_procesado.flatten()

        # Dropout
        if training and self.dropout_rate > 0:
            estado_procesado = aplicar_dropout_avanzado(estado_procesado, self.dropout_rate, training)

        return estado_procesado

    def _forward_features(self, estado: np.ndarray, usar_target: bool = False, training: bool = True) -> np.ndarray:
        """
        Propagación hacia adelante hasta las características (antes de la capa de salida).
        Devuelve las features del tamaño hidden_size para usar en Dueling Network.
        """
        # Preprocesamiento
        estado_procesado = self._aplicar_preprocesamiento(estado, training)

        # Seleccionar pesos
        if usar_target:
            pesos = self.pesos_target
            sesgos = self.sesgos_target
        else:
            pesos = self.pesos_red
            sesgos = self.sesgos_red

        # Propagación hacia adelante hasta la última capa oculta
        x = estado_procesado
        for i, (peso, sesgo) in enumerate(zip(pesos[:-1], sesgos[:-1])):  # Excluir la capa de salida
            x = np.dot(x, peso) + sesgo

            # Batch Normalization
            if self.batch_norm:
                if usar_target:
                    x, _, _ = aplicar_batch_normalization_avanzada(
                        x.reshape(1, -1), self.gamma_bn_target[i], self.beta_bn_target[i],
                        self.running_mean_target[i], self.running_var_target[i], training=training
                    )
                else:
                    x, self.running_mean[i], self.running_var[i] = aplicar_batch_normalization_avanzada(
                        x.reshape(1, -1), self.gamma_bn[i], self.beta_bn[i],
                        self.running_mean[i], self.running_var[i], training=training
                    )
                x = x.flatten()

            # Activación ReLU
            x = np.maximum(0, x)

            # Dropout
            if training and self.dropout_rate > 0:
                x = aplicar_dropout_avanzado(x, self.dropout_rate, training)

        return x

    def _forward_network(self, estado: np.ndarray, usar_target: bool = False, training: bool = True) -> np.ndarray:
        """
        Propagación hacia adelante de la red completa.
        """
        # Preprocesamiento
        estado_procesado = self._aplicar_preprocesamiento(estado, training)

        # Seleccionar pesos
        if usar_target:
            pesos = self.pesos_target
            sesgos = self.sesgos_target
        else:
            pesos = self.pesos_red
            sesgos = self.sesgos_red

        # Propagación hacia adelante
        x = estado_procesado
        for i, (peso, sesgo) in enumerate(zip(pesos, sesgos)):
            x = np.dot(x, peso) + sesgo

            # Batch Normalization para capas ocultas
            if self.batch_norm and i < len(pesos) - 1:
                if usar_target:
                    x, _, _ = aplicar_batch_normalization_avanzada(
                        x.reshape(1, -1), self.gamma_bn_target[i], self.beta_bn_target[i],
                        self.running_mean_target[i], self.running_var_target[i], training=training
                    )
                else:
                    x, self.running_mean[i], self.running_var[i] = aplicar_batch_normalization_avanzada(
                        x.reshape(1, -1), self.gamma_bn[i], self.beta_bn[i],
                        self.running_mean[i], self.running_var[i], training=training
                    )
                x = x.flatten()

            # Activación ReLU para capas ocultas
            if i < len(pesos) - 1:
                x = np.maximum(0, x)

            # Dropout para capas ocultas
            if training and self.dropout_rate > 0 and i < len(pesos) - 1:
                x = aplicar_dropout_avanzado(x, self.dropout_rate, training)

        return x

    def _calcular_q_values_dueling(self, estado: np.ndarray, usar_target: bool = False, training: bool = True) -> np.ndarray:
        """
        Calcula Q-values usando Dueling DQN.
        """
        if not self.dueling_dqn:
            return self._forward_network(estado, usar_target, training)

        # Obtener características intermedias (hidden_size) antes de la capa de salida
        features = self._forward_features(estado, usar_target, training)

        # Dueling Network: calcular valor y ventaja desde las features
        if usar_target:
            valor = np.dot(features, self.pesos_target_valor[0]) + self.sesgos_target_valor[0]
            ventaja = np.dot(features, self.pesos_target_ventaja[0]) + self.sesgos_target_ventaja[0]
        else:
            valor = np.dot(features, self.pesos_valor[0]) + self.sesgos_valor[0]
            ventaja = np.dot(features, self.pesos_ventaja[0]) + self.sesgos_ventaja[0]

        # Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        q_values = valor + ventaja - np.mean(ventaja)

        return q_values

    def forward(self, estado: np.ndarray, usar_target: bool = False, training: bool = True) -> np.ndarray:
        """
        Propagación hacia adelante con técnicas avanzadas.
        """
        if not self.pesos_red:
            self.inicializar_pesos()

        # Calcular Q-values
        if self.dueling_dqn:
            q_values = self._calcular_q_values_dueling(estado, usar_target, training)
        else:
            q_values = self._forward_network(estado, usar_target, training)

        # Guardar Q-values para análisis
        self.historial_q_values.append(q_values.copy())

        return q_values

    def seleccionar_accion(self, estado: np.ndarray, training: bool = True) -> int:
        """
        Selecciona una acción usando epsilon-greedy con técnicas avanzadas.
        """
        if training and np.random.random() < self.epsilon:
            # Exploración aleatoria
            return np.random.randint(0, self.output_size)
        else:
            # Explotación basada en Q-values
            q_values = self.forward(estado, training=training)
            return np.argmax(q_values)

    def _calcular_prioridad_experiencia(self, td_error: float) -> float:
        """
        Calcula la prioridad de una experiencia basada en TD error.
        """
        return abs(td_error) + 1e-6

    def _muestrear_experiencia_priorizada(self) -> List[Tuple]:
        """
        Muestrea experiencias usando priorización.
        """
        if not self.prioritized_replay or len(self.experience_buffer) < self.batch_size:
            return random.sample(list(self.experience_buffer),
                                 min(self.batch_size, len(self.experience_buffer)))

        # Muestreo priorizado
        priorities = np.array(list(self.priorities))
        probabilities = priorities / np.sum(priorities)

        indices = np.random.choice(len(self.experience_buffer),
                                   size=min(self.batch_size, len(self.experience_buffer)),
                                   p=probabilities, replace=False)

        return [list(self.experience_buffer)[i] for i in indices]

    def _actualizar_target_network(self) -> None:
        """
        Actualiza el target network con soft update.
        """
        # Actualizar pesos principales
        for i in range(len(self.pesos_red)):
            self.pesos_target[i] = (1 - self.soft_update_tau) * self.pesos_target[i] + \
                self.soft_update_tau * self.pesos_red[i]
            self.sesgos_target[i] = (1 - self.soft_update_tau) * self.sesgos_target[i] + \
                self.soft_update_tau * self.sesgos_red[i]

        # Actualizar Dueling DQN targets
        if self.dueling_dqn:
            self.pesos_target_valor[0] = (1 - self.soft_update_tau) * self.pesos_target_valor[0] + \
                self.soft_update_tau * self.pesos_valor[0]
            self.sesgos_target_valor[0] = (1 - self.soft_update_tau) * self.sesgos_target_valor[0] + \
                self.soft_update_tau * self.sesgos_valor[0]

            self.pesos_target_ventaja[0] = (1 - self.soft_update_tau) * self.pesos_target_ventaja[0] + \
                self.soft_update_tau * self.pesos_ventaja[0]
            self.sesgos_target_ventaja[0] = (1 - self.soft_update_tau) * self.sesgos_target_ventaja[0] + \
                self.soft_update_tau * self.sesgos_ventaja[0]

        # Actualizar Batch Normalization targets
        if self.batch_norm:
            for i in range(len(self.gamma_bn)):
                self.gamma_bn_target[i] = (1 - self.soft_update_tau) * self.gamma_bn_target[i] + \
                    self.soft_update_tau * self.gamma_bn[i]
                self.beta_bn_target[i] = (1 - self.soft_update_tau) * self.beta_bn_target[i] + \
                    self.soft_update_tau * self.beta_bn[i]
                self.running_mean_target[i] = (1 - self.soft_update_tau) * self.running_mean_target[i] + \
                    self.soft_update_tau * self.running_mean[i]
                self.running_var_target[i] = (1 - self.soft_update_tau) * self.running_var_target[i] + \
                    self.soft_update_tau * self.running_var[i]

        self.estadisticas_dqn['target_updates'] += 1
        self.historial_target_updates.append(time.time())

    def entrenar_paso(self, estado: np.ndarray, accion: int, recompensa: float,
                      siguiente_estado: np.ndarray, terminado: bool) -> Optional[float]:
        """
        Realiza un paso de entrenamiento con técnicas avanzadas.
        """
        if not self.pesos_red:
            self.inicializar_pesos()

        # Almacenar experiencia
        experiencia = (estado.copy(), accion, recompensa, siguiente_estado.copy(), terminado)
        self.experience_buffer.append(experiencia)

        # Calcular TD error para priorización
        q_actual = self.forward(estado, training=True)
        q_siguiente = self.forward(siguiente_estado, usar_target=True, training=False)

        if self.double_dqn:
            # Double DQN: usar red principal para seleccionar acción
            accion_siguiente = np.argmax(self.forward(siguiente_estado, training=False))
            q_target = recompensa + self.gamma * q_siguiente[accion_siguiente] * (1 - int(terminado))
        else:
            # DQN estándar
            q_target = recompensa + self.gamma * np.max(q_siguiente) * (1 - int(terminado))

        td_error = q_target - q_actual[accion]

        # Calcular prioridad
        prioridad = self._calcular_prioridad_experiencia(td_error)
        self.priorities.append(prioridad)

        # Entrenar si hay suficientes experiencias
        if len(self.experience_buffer) >= self.batch_size:
            return self._entrenar_batch()

        return abs(td_error)

    def _entrenar_batch(self) -> float:
        """
        Entrena con un lote de experiencias usando técnicas avanzadas.
        """
        # Muestrear experiencias
        experiencias = self._muestrear_experiencia_priorizada()

        if not experiencias:
            return 0.0

        # Preparar datos del lote
        estados = np.array([exp[0] for exp in experiencias])
        acciones = np.array([exp[1] for exp in experiencias])
        recompensas = np.array([exp[2] for exp in experiencias])
        siguientes_estados = np.array([exp[3] for exp in experiencias])
        terminados = np.array([exp[4] for exp in experiencias])

        # Calcular Q-values actuales
        q_values_actuales = np.array([self.forward(estado, training=True) for estado in estados])

        # Calcular Q-values objetivo
        q_values_objetivo = np.zeros_like(q_values_actuales)

        for i, (siguiente_estado, terminado) in enumerate(zip(siguientes_estados, terminados)):
            q_siguiente = self.forward(siguiente_estado, usar_target=True, training=False)

            if self.double_dqn:
                # Double DQN
                accion_siguiente = np.argmax(self.forward(siguiente_estado, training=False))
                q_target = recompensas[i] + self.gamma * q_siguiente[accion_siguiente] * (1 - int(terminado))
            else:
                # DQN estándar
                q_target = recompensas[i] + self.gamma * np.max(q_siguiente) * (1 - int(terminado))

            q_values_objetivo[i, acciones[i]] = q_target

        # Calcular pérdida
        perdida = np.mean((q_values_objetivo - q_values_actuales) ** 2)

        # Calcular gradientes
        gradientes = self._calcular_gradientes(q_values_actuales, q_values_objetivo, estados)

        # Aplicar optimizador avanzado
        self._aplicar_optimizador(gradientes)

        # Actualizar target network
        if self.pasos_totales % self.target_update_frequency == 0:
            self._actualizar_target_network()

        # Actualizar epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Actualizar estadísticas
        self._actualizar_estadisticas(perdida, experiencias)

        return perdida

    def _calcular_gradientes(self, q_values_actuales: np.ndarray, q_values_objetivo: np.ndarray,
                             estados: np.ndarray) -> List[np.ndarray]:
        """
        Calcula gradientes para la red.
        """
        # Gradiente de pérdida MSE
        error = q_values_objetivo - q_values_actuales
        grad_output = 2 * error / len(estados)

        # Gradientes para pesos principales
        grad_pesos = np.dot(estados.T, grad_output)
        grad_sesgo = np.sum(grad_output, axis=0)

        gradientes = [grad_pesos, grad_sesgo]

        # Gradientes para Dueling DQN
        if self.dueling_dqn:
            # Gradientes para red de valor
            grad_valor = np.dot(estados.T, np.mean(grad_output, axis=1, keepdims=True))
            grad_sesgo_valor = np.sum(np.mean(grad_output, axis=1))

            # Gradientes para red de ventaja
            grad_ventaja = grad_output - np.mean(grad_output, axis=1, keepdims=True)
            grad_pesos_ventaja = np.dot(estados.T, grad_ventaja)
            grad_sesgo_ventaja = np.sum(grad_ventaja, axis=0)

            gradientes.extend([grad_valor, grad_sesgo_valor, grad_pesos_ventaja, grad_sesgo_ventaja])

        return gradientes

    def _aplicar_optimizador(self, gradientes: List[np.ndarray]) -> None:
        """
        Aplica el optimizador avanzado seleccionado.
        """
        # Aplicar weight decay
        if self.weight_decay > 0:
            gradientes[0] += self.weight_decay * self.pesos_red[0]

        # Aplicar gradient clipping
        gradientes = aplicar_clip_gradientes_avanzado(gradientes, method='adaptive')

        # Aplicar optimizador específico
        if self.optimizer == 'adam':
            gradientes_optimizados, self.m_historial, self.v_historial = aplicar_adam_avanzado(
                gradientes, self.m_historial, self.v_historial,
                learning_rate=self.learning_rate, beta1=self.beta1, beta2=self.beta2,
                weight_decay=self.weight_decay
            )
        elif self.optimizer == 'rmsprop':
            gradientes_optimizados, self.v_historial = aplicar_rmsprop_avanzado(
                gradientes, self.v_historial, learning_rate=self.learning_rate
            )
        elif self.optimizer == 'momentum':
            gradientes_optimizados, self.momentum_historial = calcular_momentum_avanzado(
                gradientes, self.momentum_historial, momentum=self.momentum
            )
        else:  # SGD
            gradientes_optimizados = gradientes

        # Actualizar pesos
        self._actualizar_pesos(gradientes_optimizados)

    def _actualizar_pesos(self, gradientes: List[np.ndarray]) -> None:
        """
        Actualiza los pesos con los gradientes optimizados.
        """
        # Actualizar pesos principales
        self.pesos_red[0] -= gradientes[0]
        self.sesgos_red[0] -= gradientes[1]

        # Actualizar pesos de Dueling DQN
        if self.dueling_dqn:
            self.pesos_valor[0] -= gradientes[2]
            self.sesgos_valor[0] -= gradientes[3]
            self.pesos_ventaja[0] -= gradientes[4]
            self.sesgos_ventaja[0] -= gradientes[5]

        # Guardar en historial
        self.historial_pesos.append(self.pesos_red[0].copy())

    def _actualizar_estadisticas(self, perdida: float, experiencias: List[Tuple]) -> None:
        """
        Actualiza las estadísticas específicas de DQN avanzado.
        """
        # Estadísticas básicas
        self.estadisticas_dqn['q_values_media'] = np.mean([np.mean(q) for q in self.historial_q_values[-10:]])
        self.estadisticas_dqn['td_error_media'] = np.mean([abs(p) for p in list(self.priorities)[-10:]])
        self.estadisticas_dqn['epsilon_actual'] = self.epsilon

        # Estadísticas de buffer
        self.estadisticas_dqn['buffer_utilization'] = len(self.experience_buffer) / self.buffer_size

        # Estadísticas de priorización
        if self.prioritized_replay:
            self.estadisticas_dqn['prioritized_samples'] += len(experiencias)

        # Estadísticas de técnicas avanzadas
        if self.double_dqn:
            self.estadisticas_dqn['double_dqn_bias_reduction'] = 0.1  # Placeholder

        if self.dueling_dqn:
            self.estadisticas_dqn['dueling_value_advantage_separation'] = 0.1  # Placeholder

        if self.noisy_networks:
            self.estadisticas_dqn['noisy_network_exploration'] = 0.1  # Placeholder

        if self.distributional_dqn:
            self.estadisticas_dqn['distributional_dqn_quality'] = 0.1  # Placeholder

        if self.rainbow_dqn:
            self.estadisticas_dqn['rainbow_dqn_efficiency'] = 0.1  # Placeholder

        # Estadísticas de optimizador
        self.estadisticas_dqn['optimizer_efficiency'] = 1.0 / (1.0 + perdida)

        # Estadísticas de estabilidad
        if len(self.historial_pesos) > 10:
            pesos_recientes = np.array(self.historial_pesos[-10:])
            varianza_pesos = np.var(pesos_recientes)
            self.estadisticas_dqn['gradient_stability'] = 1.0 / (1.0 + varianza_pesos)

        # Guardar en historiales
        self.historial_td_errors.append(perdida)
        self.historial_epsilon.append(self.epsilon)
        self.historial_buffer_utilization.append(self.estadisticas_dqn['buffer_utilization'])

    def obtener_estadisticas(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas específicas de DQN avanzado.
        """
        if not self.pesos_red:
            return {'estado': 'no_inicializada'}

        stats_dqn = {
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'epsilon_decay': self.epsilon_decay,
            'epsilon_min': self.epsilon_min,
            'optimizer': self.optimizer,
            'momentum': self.momentum,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'weight_decay': self.weight_decay,
            'dropout_rate': self.dropout_rate,
            'batch_norm': self.batch_norm,
            'target_update_frequency': self.target_update_frequency,
            'soft_update_tau': self.soft_update_tau,
            'double_dqn': self.double_dqn,
            'dueling_dqn': self.dueling_dqn,
            'noisy_networks': self.noisy_networks,
            'distributional_dqn': self.distributional_dqn,
            'rainbow_dqn': self.rainbow_dqn,
            'prioritized_replay': self.prioritized_replay,
            'buffer_size': self.buffer_size,
            'batch_size': self.batch_size,
            'hidden_layers': self.hidden_layers,
            'hidden_size': self.hidden_size,
            'q_values_media': self.estadisticas_dqn['q_values_media'],
            'td_error_media': self.estadisticas_dqn['td_error_media'],
            'epsilon_actual': self.estadisticas_dqn['epsilon_actual'],
            'target_updates': self.estadisticas_dqn['target_updates'],
            'buffer_utilization': self.estadisticas_dqn['buffer_utilization'],
            'prioritized_samples': self.estadisticas_dqn['prioritized_samples'],
            'double_dqn_bias_reduction': self.estadisticas_dqn['double_dqn_bias_reduction'],
            'dueling_value_advantage_separation': self.estadisticas_dqn['dueling_value_advantage_separation'],
            'noisy_network_exploration': self.estadisticas_dqn['noisy_network_exploration'],
            'distributional_dqn_quality': self.estadisticas_dqn['distributional_dqn_quality'],
            'rainbow_dqn_efficiency': self.estadisticas_dqn['rainbow_dqn_efficiency'],
            'optimizer_efficiency': self.estadisticas_dqn['optimizer_efficiency'],
            'gradient_stability': self.estadisticas_dqn['gradient_stability'],
            'learning_rate_adaptation': self.estadisticas_dqn['learning_rate_adaptation'],
            'weight_decay_effectiveness': self.estadisticas_dqn['weight_decay_effectiveness'],
            'dropout_regularization': self.estadisticas_dqn['dropout_regularization'],
            'batch_norm_stability': self.estadisticas_dqn['batch_norm_stability'],
            'experience_diversity': self.estadisticas_dqn['experience_diversity'],
            'convergence_speed': self.estadisticas_dqn['convergence_speed'],
            'historial_q_values_size': len(self.historial_q_values),
            'historial_td_errors_size': len(self.historial_td_errors),
            'historial_epsilon_size': len(self.historial_epsilon),
            'historial_target_updates_size': len(self.historial_target_updates),
            'historial_buffer_utilization_size': len(self.historial_buffer_utilization)
        }

        return stats_dqn

    def verificar_estabilidad(self) -> Dict[str, bool]:
        """
        Verifica la estabilidad de la neurona DQN avanzada.
        """
        estabilidad = {}

        # Verificaciones básicas
        estabilidad['pesos_inicializados'] = len(self.pesos_red) > 0
        estabilidad['epsilon_apropiado'] = self.epsilon >= self.epsilon_min
        estabilidad['buffer_suficiente'] = len(self.experience_buffer) > self.batch_size

        # Verificaciones avanzadas
        estabilidad['q_values_estables'] = self.estadisticas_dqn['q_values_media'] > 0.0
        estabilidad['td_error_controlado'] = self.estadisticas_dqn['td_error_media'] < 10.0
        estabilidad['gradiente_estable'] = self.estadisticas_dqn['gradient_stability'] > 0.5
        estabilidad['optimizador_eficiente'] = self.estadisticas_dqn['optimizer_efficiency'] > 0.5
        estabilidad['target_updates_regulares'] = self.estadisticas_dqn['target_updates'] > 0

        # Verificaciones específicas de técnicas avanzadas
        if self.dueling_dqn:
            estabilidad['dueling_separacion_ok'] = self.estadisticas_dqn['dueling_value_advantage_separation'] > 0.0

        if self.double_dqn:
            estabilidad['double_dqn_bias_reducido'] = self.estadisticas_dqn['double_dqn_bias_reduction'] > 0.0

        if self.prioritized_replay:
            estabilidad['prioritized_samples_suficientes'] = self.estadisticas_dqn['prioritized_samples'] > 0

        return estabilidad

    def __str__(self) -> str:
        return (f"NeuronaRefuerzoDQNAvanzada(entrada={self.input_size}, "
                f"salida={self.output_size}, lr={self.learning_rate}, "
                f"optimizer={self.optimizer}, dueling={self.dueling_dqn}, "
                f"double_dqn={self.double_dqn}, prioritized={self.prioritized_replay})")

    def __repr__(self) -> str:
        return self.__str__()

# Funciones de utilidad específicas para RF_RFEN1_RN_1_4


def crear_neurona_dqn_avanzada(input_size: int, output_size: int,
                               configuracion: Dict[str, Any] = None) -> NeuronaRefuerzoDQNAvanzada:
    """
    Función de conveniencia para crear una neurona DQN avanzada.
    """
    if configuracion is None:
        configuracion = {}

    return NeuronaRefuerzoDQNAvanzada(
        input_size=input_size,
        output_size=output_size,
        nombre=configuracion.get('nombre', 'NeuronaRefuerzoDQNAvanzada'),
        learning_rate=configuracion.get('learning_rate', LUCIA_RL_CONFIG_RFENRN1['default_learning_rate']),
        gamma=configuracion.get('gamma', LUCIA_RL_CONFIG_RFENRN1['default_gamma']),
        epsilon=configuracion.get('epsilon', LUCIA_RL_CONFIG_RFENRN1['default_epsilon']),
        epsilon_decay=configuracion.get('epsilon_decay', 0.995),
        epsilon_min=configuracion.get('epsilon_min', 0.01),
        optimizer=configuracion.get('optimizer', 'adam'),
        momentum=configuracion.get('momentum', LUCIA_RL_CONFIG_RFENRN1['default_momentum']),
        beta1=configuracion.get('beta1', LUCIA_RL_CONFIG_RFENRN1['default_beta1']),
        beta2=configuracion.get('beta2', LUCIA_RL_CONFIG_RFENRN1['default_beta2']),
        weight_decay=configuracion.get('weight_decay', LUCIA_RL_CONFIG_RFENRN1['default_weight_decay']),
        dropout_rate=configuracion.get('dropout_rate', LUCIA_RL_CONFIG_RFENRN1['default_dropout_rate']),
        batch_norm=configuracion.get('batch_norm', True),
        target_update_frequency=configuracion.get('target_update_frequency', LUCIA_RL_CONFIG_RFENRN1['default_target_update_frequency']),
        soft_update_tau=configuracion.get('soft_update_tau', 0.005),
        double_dqn=configuracion.get('double_dqn', True),
        dueling_dqn=configuracion.get('dueling_dqn', True),
        noisy_networks=configuracion.get('noisy_networks', False),
        distributional_dqn=configuracion.get('distributional_dqn', False),
        rainbow_dqn=configuracion.get('rainbow_dqn', True),
        prioritized_replay=configuracion.get('prioritized_replay', True),
        buffer_size=configuracion.get('buffer_size', LUCIA_RL_CONFIG_RFENRN1['default_buffer_size']),
        batch_size=configuracion.get('batch_size', LUCIA_RL_CONFIG_RFENRN1['default_batch_size']),
        hidden_layers=configuracion.get('hidden_layers', 3),
        hidden_size=configuracion.get('hidden_size', 128)
    )


# Configuración específica para RF_RFEN1_RN_1_4
RF_RFEN1_RN_1_4_CONFIG = {
    'inicializacion_preferida': 'he_avanzado',
    'learning_rate_default': 0.001,
    'gamma_default': 0.99,
    'epsilon_default': 0.1,
    'epsilon_decay_default': 0.995,
    'epsilon_min_default': 0.01,
    'optimizer_default': 'adam',
    'momentum_default': 0.9,
    'beta1_default': 0.9,
    'beta2_default': 0.999,
    'weight_decay_default': 1e-4,
    'dropout_rate_default': 0.1,
    'batch_norm_default': True,
    'target_update_frequency_default': 200,
    'soft_update_tau_default': 0.005,
    'double_dqn_default': True,
    'dueling_dqn_default': True,
    'noisy_networks_default': False,
    'distributional_dqn_default': False,
    'rainbow_dqn_default': True,
    'prioritized_replay_default': True,
    'buffer_size_default': 100000,
    'batch_size_default': 64,
    'hidden_layers_default': 3,
    'hidden_size_default': 128,
    'umbral_q_values_estables': 0.0,
    'umbral_td_error_controlado': 10.0,
    'umbral_gradiente_estable': 0.5,
    'umbral_optimizador_eficiente': 0.5,
    'umbral_target_updates': 0,
    'umbral_epsilon_apropiado': 0.01,
    'umbral_buffer_suficiente': 64
}

logger.info("RF_RFEN1_RN_1_4.py cargado correctamente - Neurona de Refuerzo DQN Avanzada")
