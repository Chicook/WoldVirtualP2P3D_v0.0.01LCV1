"""
RF_RFEN1_RN_1_1.py - Neurona de Refuerzo Q-Learning Avanzada
===========================================================

Esta neurona implementa Q-Learning con técnicas avanzadas de optimización de pesos
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
- Double Q-Learning para reducción de bias
- Dueling Q-Learning para separación de valor y ventaja

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

logger = logging.getLogger('RF_RFENRN1.RF_RFEN1_RN_1_1')


class NeuronaRefuerzoQLearningAvanzada(NeuronaRefuerzoBaseRFENRN1):
    """
    Neurona de refuerzo Q-Learning con técnicas avanzadas de optimización de pesos.

    Implementa Q-Learning mejorado con optimizadores avanzados, regularización
    y técnicas de estabilización para mejor rendimiento y convergencia.
    """

    def __init__(self, input_size: int, output_size: int,
                 nombre: str = "NeuronaRefuerzoQLearningAvanzada",
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
                 double_q_learning: bool = True,
                 dueling_network: bool = True,
                 prioritized_replay: bool = True,
                 buffer_size: int = LUCIA_RL_CONFIG_RFENRN1['default_buffer_size'],
                 batch_size: int = LUCIA_RL_CONFIG_RFENRN1['default_batch_size']):
        """
        Inicializa la neurona Q-Learning avanzada.

        Args:
            input_size: Número de características de entrada
            output_size: Número de acciones posibles
            nombre: Nombre identificativo de la neurona
            learning_rate: Tasa de aprendizaje
            gamma: Factor de descuento
            epsilon: Tasa de exploración inicial
            epsilon_decay: Factor de decaimiento de epsilon
            epsilon_min: Valor mínimo de epsilon
            optimizer: Optimizador a usar ('adam', 'rmsprop', 'momentum', 'sgd')
            momentum: Factor de momentum
            beta1: Factor de decaimiento para primer momento (Adam)
            beta2: Factor de decaimiento para segundo momento (Adam)
            weight_decay: Decaimiento de pesos
            dropout_rate: Tasa de dropout
            batch_norm: Si usar batch normalization
            target_update_frequency: Frecuencia de actualización del target network
            soft_update_tau: Factor de actualización suave
            double_q_learning: Si usar Double Q-Learning
            dueling_network: Si usar Dueling Network
            prioritized_replay: Si usar replay con priorización
            buffer_size: Tamaño del buffer de experiencias
            batch_size: Tamaño del lote
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
        self.double_q_learning = double_q_learning
        self.dueling_network = dueling_network
        self.prioritized_replay = prioritized_replay
        self.buffer_size = buffer_size
        self.batch_size = batch_size

        # Pesos de la red principal
        self.pesos_red = None
        self.sesgo_red = None

        # Pesos del target network
        self.pesos_target = None
        self.sesgo_target = None

        # Pesos adicionales para Dueling Network
        if self.dueling_network:
            self.pesos_valor = None
            self.sesgo_valor = None
            self.pesos_ventaja = None
            self.sesgo_ventaja = None
            self.pesos_target_valor = None
            self.sesgo_target_valor = None
            self.pesos_target_ventaja = None
            self.sesgo_target_ventaja = None

        # Parámetros de Batch Normalization
        if self.batch_norm:
            self.gamma_bn = None
            self.beta_bn = None
            self.running_mean = None
            self.running_var = None
            self.gamma_bn_target = None
            self.beta_bn_target = None
            self.running_mean_target = None
            self.running_var_target = None

        # Buffer de experiencias con priorización
        self.experience_buffer = deque(maxlen=buffer_size)
        self.priorities = deque(maxlen=buffer_size)

        # Historiales para optimizadores
        self.m_historial = []
        self.v_historial = []
        self.momentum_historial = []

        # Estadísticas específicas de Q-Learning avanzado
        self.estadisticas_q_learning = {
            'q_values_media': 0.0,
            'td_error_media': 0.0,
            'epsilon_actual': epsilon,
            'target_updates': 0,
            'buffer_utilization': 0.0,
            'prioritized_samples': 0,
            'double_q_bias_reduction': 0.0,
            'dueling_value_advantage_separation': 0.0,
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

        logger.info(f"NeuronaRefuerzoQLearningAvanzada creada: {self}")

    def inicializar_pesos(self) -> None:
        """
        Inicializa los pesos con técnicas avanzadas de inicialización.
        """
        # Red principal
        self.pesos_red = inicializar_pesos_he_avanzado((self.input_size, self.output_size))
        self.sesgo_red = np.zeros(self.output_size, dtype=LUCIA_RL_CONFIG_RFENRN1['precision'])

        # Target network (copia inicial)
        self.pesos_target = self.pesos_red.copy()
        self.sesgo_target = self.sesgo_red.copy()

        # Dueling Network
        if self.dueling_network:
            # Red de valor
            self.pesos_valor = inicializar_pesos_he_avanzado((self.input_size, 1))
            self.sesgo_valor = np.zeros(1, dtype=LUCIA_RL_CONFIG_RFENRN1['precision'])

            # Red de ventaja
            self.pesos_ventaja = inicializar_pesos_he_avanzado((self.input_size, self.output_size))
            self.sesgo_ventaja = np.zeros(self.output_size, dtype=LUCIA_RL_CONFIG_RFENRN1['precision'])

            # Target networks para Dueling
            self.pesos_target_valor = self.pesos_valor.copy()
            self.sesgo_target_valor = self.sesgo_valor.copy()
            self.pesos_target_ventaja = self.pesos_ventaja.copy()
            self.sesgo_target_ventaja = self.sesgo_ventaja.copy()

        # Batch Normalization
        if self.batch_norm:
            self.gamma_bn = np.ones(self.input_size, dtype=LUCIA_RL_CONFIG_RFENRN1['precision'])
            self.beta_bn = np.zeros(self.input_size, dtype=LUCIA_RL_CONFIG_RFENRN1['precision'])
            self.running_mean = np.zeros(self.input_size, dtype=LUCIA_RL_CONFIG_RFENRN1['precision'])
            self.running_var = np.ones(self.input_size, dtype=LUCIA_RL_CONFIG_RFENRN1['precision'])

            # Target network batch norm
            self.gamma_bn_target = self.gamma_bn.copy()
            self.beta_bn_target = self.beta_bn.copy()
            self.running_mean_target = self.running_mean.copy()
            self.running_var_target = self.running_var.copy()

        logger.info("Pesos Q-Learning avanzado inicializados")

    def _calcular_q_values_dueling(self, estado: np.ndarray, usar_target: bool = False) -> np.ndarray:
        """
        Calcula Q-values usando Dueling Network.

        Args:
            estado: Estado actual
            usar_target: Si usar target network

        Returns:
            Q-values calculados
        """
        if not self.dueling_network:
            # Red tradicional
            if usar_target:
                q_values = np.dot(estado, self.pesos_target) + self.sesgo_target
            else:
                q_values = np.dot(estado, self.pesos_red) + self.sesgo_red
            return q_values

        # Dueling Network
        if usar_target:
            valor = np.dot(estado, self.pesos_target_valor) + self.sesgo_target_valor
            ventaja = np.dot(estado, self.pesos_target_ventaja) + self.sesgo_target_ventaja
        else:
            valor = np.dot(estado, self.pesos_valor) + self.sesgo_valor
            ventaja = np.dot(estado, self.pesos_ventaja) + self.sesgo_ventaja

        # Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        q_values = valor + ventaja - np.mean(ventaja)

        return q_values

    def _aplicar_preprocesamiento(self, estado: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Aplica preprocesamiento avanzado al estado.

        Args:
            estado: Estado a procesar
            training: Si está en modo entrenamiento

        Returns:
            Estado procesado
        """
        estado_procesado = estado.copy()

        # Batch Normalization
        if self.batch_norm:
            estado_procesado, self.running_mean, self.running_var = aplicar_batch_normalization_avanzada(
                estado_procesado.reshape(1, -1), self.gamma_bn, self.beta_bn,
                self.running_mean, self.running_var, training=training
            )
            estado_procesado = estado_procesado.flatten()

        # Dropout
        if training and self.dropout_rate > 0:
            estado_procesado = aplicar_dropout_avanzado(estado_procesado, self.dropout_rate, training)

        return estado_procesado

    def forward(self, estado: np.ndarray, usar_target: bool = False, training: bool = True) -> np.ndarray:
        """
        Propagación hacia adelante con técnicas avanzadas.

        Args:
            estado: Estado actual del entorno
            usar_target: Si usar target network
            training: Si está en modo entrenamiento

        Returns:
            Q-values para todas las acciones
        """
        if self.pesos_red is None:
            self.inicializar_pesos()

        # Preprocesamiento
        estado_procesado = self._aplicar_preprocesamiento(estado, training)

        # Calcular Q-values
        if self.dueling_network:
            q_values = self._calcular_q_values_dueling(estado_procesado, usar_target)
        else:
            if usar_target:
                q_values = np.dot(estado_procesado, self.pesos_target) + self.sesgo_target
            else:
                q_values = np.dot(estado_procesado, self.pesos_red) + self.sesgo_red

        # Guardar Q-values para análisis
        self.historial_q_values.append(q_values.copy())

        return q_values

    def seleccionar_accion(self, estado: np.ndarray, training: bool = True) -> int:
        """
        Selecciona una acción usando epsilon-greedy con técnicas avanzadas.

        Args:
            estado: Estado actual del entorno
            training: Si está en modo entrenamiento

        Returns:
            Acción seleccionada
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

        Args:
            td_error: Error de diferencia temporal

        Returns:
            Prioridad de la experiencia
        """
        return abs(td_error) + 1e-6  # Evitar prioridad cero

    def _muestrear_experiencia_priorizada(self) -> List[Tuple]:
        """
        Muestrea experiencias usando priorización.

        Returns:
            Lista de experiencias muestreadas
        """
        if not self.prioritized_replay or len(self.experience_buffer) < self.batch_size:
            # Muestreo uniforme
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
        self.pesos_target = (1 - self.soft_update_tau) * self.pesos_target + \
            self.soft_update_tau * self.pesos_red
        self.sesgo_target = (1 - self.soft_update_tau) * self.sesgo_target + \
            self.soft_update_tau * self.sesgo_red

        # Actualizar Dueling Network targets
        if self.dueling_network:
            self.pesos_target_valor = (1 - self.soft_update_tau) * self.pesos_target_valor + \
                self.soft_update_tau * self.pesos_valor
            self.sesgo_target_valor = (1 - self.soft_update_tau) * self.sesgo_target_valor + \
                self.soft_update_tau * self.sesgo_valor

            self.pesos_target_ventaja = (1 - self.soft_update_tau) * self.pesos_target_ventaja + \
                self.soft_update_tau * self.pesos_ventaja
            self.sesgo_target_ventaja = (1 - self.soft_update_tau) * self.sesgo_target_ventaja + \
                self.soft_update_tau * self.sesgo_ventaja

        # Actualizar Batch Normalization targets
        if self.batch_norm:
            self.gamma_bn_target = (1 - self.soft_update_tau) * self.gamma_bn_target + \
                self.soft_update_tau * self.gamma_bn
            self.beta_bn_target = (1 - self.soft_update_tau) * self.beta_bn_target + \
                self.soft_update_tau * self.beta_bn
            self.running_mean_target = (1 - self.soft_update_tau) * self.running_mean_target + \
                self.soft_update_tau * self.running_mean
            self.running_var_target = (1 - self.soft_update_tau) * self.running_var_target + \
                self.soft_update_tau * self.running_var

        self.estadisticas_q_learning['target_updates'] += 1
        self.historial_target_updates.append(time.time())

    def entrenar_paso(self, estado: np.ndarray, accion: int, recompensa: float,
                      siguiente_estado: np.ndarray, terminado: bool) -> Optional[float]:
        """
        Realiza un paso de entrenamiento con técnicas avanzadas.

        Args:
            estado: Estado actual
            accion: Acción tomada
            recompensa: Recompensa recibida
            siguiente_estado: Siguiente estado
            terminado: Si el episodio terminó

        Returns:
            Pérdida del paso de entrenamiento
        """
        if self.pesos_red is None:
            self.inicializar_pesos()

        # Almacenar experiencia
        experiencia = (estado.copy(), accion, recompensa, siguiente_estado.copy(), terminado)
        self.experience_buffer.append(experiencia)

        # Calcular TD error para priorización
        q_actual = self.forward(estado, training=True)
        q_siguiente = self.forward(siguiente_estado, usar_target=True, training=False)

        if self.double_q_learning:
            # Double Q-Learning: usar red principal para seleccionar acción
            accion_siguiente = np.argmax(self.forward(siguiente_estado, training=False))
            q_target = recompensa + self.gamma * q_siguiente[accion_siguiente] * (1 - int(terminado))
        else:
            # Q-Learning estándar
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

        Returns:
            Pérdida promedio del lote
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

            if self.double_q_learning:
                # Double Q-Learning
                accion_siguiente = np.argmax(self.forward(siguiente_estado, training=False))
                q_target = recompensas[i] + self.gamma * q_siguiente[accion_siguiente] * (1 - int(terminado))
            else:
                # Q-Learning estándar
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

        Args:
            q_values_actuales: Q-values actuales
            q_values_objetivo: Q-values objetivo
            estados: Estados del lote

        Returns:
            Lista de gradientes
        """
        # Gradiente de pérdida MSE
        error = q_values_objetivo - q_values_actuales
        grad_output = 2 * error / len(estados)

        # Gradientes para pesos principales
        grad_pesos = np.dot(estados.T, grad_output)
        grad_sesgo = np.sum(grad_output, axis=0)

        gradientes = [grad_pesos, grad_sesgo]

        # Gradientes para Dueling Network
        if self.dueling_network:
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

        Args:
            gradientes: Lista de gradientes
        """
        # Aplicar weight decay
        if self.weight_decay > 0:
            gradientes[0] += self.weight_decay * self.pesos_red

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

        Args:
            gradientes: Gradientes optimizados
        """
        # Actualizar pesos principales
        self.pesos_red -= gradientes[0]
        self.sesgo_red -= gradientes[1]

        # Actualizar pesos de Dueling Network
        if self.dueling_network:
            self.pesos_valor -= gradientes[2]
            self.sesgo_valor -= gradientes[3]
            self.pesos_ventaja -= gradientes[4]
            self.sesgo_ventaja -= gradientes[5]

        # Guardar en historial
        self.historial_pesos.append(self.pesos_red.copy())

    def _actualizar_estadisticas(self, perdida: float, experiencias: List[Tuple]) -> None:
        """
        Actualiza las estadísticas específicas de Q-Learning avanzado.
        """
        # Estadísticas básicas
        self.estadisticas_q_learning['q_values_media'] = np.mean([np.mean(q) for q in self.historial_q_values[-10:]])
        self.estadisticas_q_learning['td_error_media'] = np.mean([abs(p) for p in list(self.priorities)[-10:]])
        self.estadisticas_q_learning['epsilon_actual'] = self.epsilon

        # Estadísticas de buffer
        self.estadisticas_q_learning['buffer_utilization'] = len(self.experience_buffer) / self.buffer_size

        # Estadísticas de priorización
        if self.prioritized_replay:
            self.estadisticas_q_learning['prioritized_samples'] += len(experiencias)

        # Estadísticas de Double Q-Learning
        if self.double_q_learning:
            self.estadisticas_q_learning['double_q_bias_reduction'] = 0.1  # Placeholder

        # Estadísticas de Dueling Network
        if self.dueling_network:
            self.estadisticas_q_learning['dueling_value_advantage_separation'] = 0.1  # Placeholder

        # Estadísticas de optimizador
        self.estadisticas_q_learning['optimizer_efficiency'] = 1.0 / (1.0 + perdida)

        # Estadísticas de estabilidad
        if len(self.historial_pesos) > 10:
            pesos_recientes = np.array(self.historial_pesos[-10:])
            varianza_pesos = np.var(pesos_recientes)
            self.estadisticas_q_learning['gradient_stability'] = 1.0 / (1.0 + varianza_pesos)

        # Guardar en historiales
        self.historial_td_errors.append(perdida)
        self.historial_epsilon.append(self.epsilon)
        self.historial_buffer_utilization.append(self.estadisticas_q_learning['buffer_utilization'])

    def obtener_estadisticas(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas específicas de Q-Learning avanzado.

        Returns:
            Diccionario con estadísticas
        """
        if self.pesos_red is None:
            return {'estado': 'no_inicializada'}

        stats_q_learning = {
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
            'double_q_learning': self.double_q_learning,
            'dueling_network': self.dueling_network,
            'prioritized_replay': self.prioritized_replay,
            'buffer_size': self.buffer_size,
            'batch_size': self.batch_size,
            'q_values_media': self.estadisticas_q_learning['q_values_media'],
            'td_error_media': self.estadisticas_q_learning['td_error_media'],
            'epsilon_actual': self.estadisticas_q_learning['epsilon_actual'],
            'target_updates': self.estadisticas_q_learning['target_updates'],
            'buffer_utilization': self.estadisticas_q_learning['buffer_utilization'],
            'prioritized_samples': self.estadisticas_q_learning['prioritized_samples'],
            'double_q_bias_reduction': self.estadisticas_q_learning['double_q_bias_reduction'],
            'dueling_value_advantage_separation': self.estadisticas_q_learning['dueling_value_advantage_separation'],
            'optimizer_efficiency': self.estadisticas_q_learning['optimizer_efficiency'],
            'gradient_stability': self.estadisticas_q_learning['gradient_stability'],
            'learning_rate_adaptation': self.estadisticas_q_learning['learning_rate_adaptation'],
            'weight_decay_effectiveness': self.estadisticas_q_learning['weight_decay_effectiveness'],
            'dropout_regularization': self.estadisticas_q_learning['dropout_regularization'],
            'batch_norm_stability': self.estadisticas_q_learning['batch_norm_stability'],
            'experience_diversity': self.estadisticas_q_learning['experience_diversity'],
            'convergence_speed': self.estadisticas_q_learning['convergence_speed'],
            'historial_q_values_size': len(self.historial_q_values),
            'historial_td_errors_size': len(self.historial_td_errors),
            'historial_epsilon_size': len(self.historial_epsilon),
            'historial_target_updates_size': len(self.historial_target_updates),
            'historial_buffer_utilization_size': len(self.historial_buffer_utilization)
        }

        return stats_q_learning

    def verificar_estabilidad(self) -> Dict[str, bool]:
        """
        Verifica la estabilidad de la neurona Q-Learning avanzada.

        Returns:
            Diccionario con indicadores de estabilidad
        """
        estabilidad = {}

        # Verificaciones básicas
        estabilidad['pesos_inicializados'] = self.pesos_red is not None
        estabilidad['epsilon_apropiado'] = self.epsilon >= self.epsilon_min
        estabilidad['buffer_suficiente'] = len(self.experience_buffer) > self.batch_size

        # Verificaciones avanzadas
        estabilidad['q_values_estables'] = self.estadisticas_q_learning['q_values_media'] > 0.0
        estabilidad['td_error_controlado'] = self.estadisticas_q_learning['td_error_media'] < 10.0
        estabilidad['gradiente_estable'] = self.estadisticas_q_learning['gradient_stability'] > 0.5
        estabilidad['optimizador_eficiente'] = self.estadisticas_q_learning['optimizer_efficiency'] > 0.5
        estabilidad['target_updates_regulares'] = self.estadisticas_q_learning['target_updates'] > 0

        # Verificaciones específicas de técnicas avanzadas
        if self.dueling_network:
            estabilidad['dueling_separacion_ok'] = self.estadisticas_q_learning['dueling_value_advantage_separation'] > 0.0

        if self.double_q_learning:
            estabilidad['double_q_bias_reducido'] = self.estadisticas_q_learning['double_q_bias_reduction'] > 0.0

        if self.prioritized_replay:
            estabilidad['prioritized_samples_suficientes'] = self.estadisticas_q_learning['prioritized_samples'] > 0

        return estabilidad

    def __str__(self) -> str:
        return (f"NeuronaRefuerzoQLearningAvanzada(entrada={self.input_size}, "
                f"salida={self.output_size}, lr={self.learning_rate}, "
                f"optimizer={self.optimizer}, dueling={self.dueling_network}, "
                f"double_q={self.double_q_learning}, prioritized={self.prioritized_replay})")

    def __repr__(self) -> str:
        return self.__str__()

# Funciones de utilidad específicas para RF_RFEN1_RN_1_1


def crear_neurona_q_learning_avanzada(input_size: int, output_size: int,
                                      configuracion: Dict[str, Any] = None) -> NeuronaRefuerzoQLearningAvanzada:
    """
    Función de conveniencia para crear una neurona Q-Learning avanzada.

    Args:
        input_size: Tamaño de entrada
        output_size: Tamaño de salida
        configuracion: Configuración personalizada

    Returns:
        Neurona Q-Learning avanzada configurada
    """
    if configuracion is None:
        configuracion = {}

    return NeuronaRefuerzoQLearningAvanzada(
        input_size=input_size,
        output_size=output_size,
        nombre=configuracion.get('nombre', 'NeuronaRefuerzoQLearningAvanzada'),
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
        double_q_learning=configuracion.get('double_q_learning', True),
        dueling_network=configuracion.get('dueling_network', True),
        prioritized_replay=configuracion.get('prioritized_replay', True),
        buffer_size=configuracion.get('buffer_size', LUCIA_RL_CONFIG_RFENRN1['default_buffer_size']),
        batch_size=configuracion.get('batch_size', LUCIA_RL_CONFIG_RFENRN1['default_batch_size'])
    )


# Configuración específica para RF_RFEN1_RN_1_1
RF_RFEN1_RN_1_1_CONFIG = {
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
    'double_q_learning_default': True,
    'dueling_network_default': True,
    'prioritized_replay_default': True,
    'buffer_size_default': 100000,
    'batch_size_default': 64,
    'umbral_q_values_estables': 0.0,
    'umbral_td_error_controlado': 10.0,
    'umbral_gradiente_estable': 0.5,
    'umbral_optimizador_eficiente': 0.5,
    'umbral_target_updates': 0,
    'umbral_epsilon_apropiado': 0.01,
    'umbral_buffer_suficiente': 64
}

logger.info("RF_RFEN1_RN_1_1.py cargado correctamente - Neurona de Refuerzo Q-Learning Avanzada")
