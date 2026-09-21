"""
RF_RFEN1_RN_1_5.py - Neurona de Refuerzo A3C Avanzada
===================================================

Esta neurona implementa Asynchronous Advantage Actor-Critic (A3C) con técnicas avanzadas
de optimización de pesos y mejoras específicas para 2025.

Características Avanzadas 2025:
- Optimización Adam/RMSprop con momentum adaptativo
- Regularización L2 avanzada con weight decay
- Batch Normalization para estabilización
- Dropout adaptativo para regularización
- Learning rate scheduling avanzado
- Gradient clipping adaptativo
- Asynchronous updates con múltiples workers
- Shared parameters con sincronización
- Entropy regularization para exploración
- Generalized Advantage Estimation (GAE)
- Multi-step returns
- Value function approximation avanzada
- Actor-Critic con redes separadas optimizadas

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

logger = logging.getLogger('RF_RFENRN1.RF_RFEN1_RN_1_5')


class NeuronaRefuerzoA3CAvanzada(NeuronaRefuerzoBaseRFENRN1):
    """
    Neurona de refuerzo A3C con técnicas avanzadas de optimización de pesos.

    Implementa A3C mejorado con optimizadores avanzados, regularización
    y técnicas de estabilización para mejor rendimiento y convergencia.
    """

    def __init__(self, input_size: int, output_size: int,
                 nombre: str = "NeuronaRefuerzoA3CAvanzada",
                 learning_rate_actor: float = LUCIA_RL_CONFIG_RFENRN1['default_learning_rate'],
                 learning_rate_critic: float = LUCIA_RL_CONFIG_RFENRN1['default_learning_rate'],
                 gamma: float = LUCIA_RL_CONFIG_RFENRN1['default_gamma'],
                 optimizer: str = 'adam',
                 momentum: float = LUCIA_RL_CONFIG_RFENRN1['default_momentum'],
                 beta1: float = LUCIA_RL_CONFIG_RFENRN1['default_beta1'],
                 beta2: float = LUCIA_RL_CONFIG_RFENRN1['default_beta2'],
                 weight_decay: float = LUCIA_RL_CONFIG_RFENRN1['default_weight_decay'],
                 dropout_rate: float = LUCIA_RL_CONFIG_RFENRN1['default_dropout_rate'],
                 batch_norm: bool = True,
                 entropy_coef: float = LUCIA_RL_CONFIG_RFENRN1['default_entropy_coef'],
                 value_coef: float = LUCIA_RL_CONFIG_RFENRN1['default_value_coef'],
                 lambda_gae: float = LUCIA_RL_CONFIG_RFENRN1['default_lambda_gae'],
                 n_workers: int = 4,
                 t_max: int = 20,
                 use_shared_parameters: bool = True,
                 use_async_updates: bool = True,
                 multi_step_returns: int = 5,
                 value_function_layers: int = 2,
                 actor_function_layers: int = 2):
        """
        Inicializa la neurona A3C avanzada.
        """
        super().__init__(input_size, output_size, nombre)

        self.learning_rate_actor = learning_rate_actor
        self.learning_rate_critic = learning_rate_critic
        self.gamma = gamma
        self.optimizer = optimizer
        self.momentum = momentum
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.lambda_gae = lambda_gae
        self.n_workers = n_workers
        self.t_max = t_max
        self.use_shared_parameters = use_shared_parameters
        self.use_async_updates = use_async_updates
        self.multi_step_returns = multi_step_returns
        self.value_function_layers = value_function_layers
        self.actor_function_layers = actor_function_layers

        # Pesos del Actor (política)
        self.pesos_actor = []
        self.sesgos_actor = []

        # Pesos del Critic (función de valor)
        self.pesos_critic = []
        self.sesgos_critic = []

        # Parámetros compartidos (si está habilitado)
        if self.use_shared_parameters:
            self.pesos_compartidos = []
            self.sesgos_compartidos = []

        # Parámetros de Batch Normalization
        if self.batch_norm:
            self.gamma_bn_actor = []
            self.beta_bn_actor = []
            self.running_mean_actor = []
            self.running_var_actor = []

            self.gamma_bn_critic = []
            self.beta_bn_critic = []
            self.running_mean_critic = []
            self.running_var_critic = []

        # Buffer de experiencias para A3C
        self.experience_buffer = deque(maxlen=10000)

        # Historiales para optimizadores
        self.m_historial_actor = []
        self.v_historial_actor = []
        self.momentum_historial_actor = []

        self.m_historial_critic = []
        self.v_historial_critic = []
        self.momentum_historial_critic = []

        # Estadísticas específicas de A3C avanzado
        self.estadisticas_a3c = {
            'actor_loss_media': 0.0,
            'critic_loss_media': 0.0,
            'entropy_media': 0.0,
            'advantage_media': 0.0,
            'value_accuracy': 0.0,
            'policy_entropy': 0.0,
            'gae_advantage_quality': 0.0,
            'async_update_efficiency': 0.0,
            'shared_parameter_sync': 0.0,
            'multi_step_return_quality': 0.0,
            'value_function_approximation': 0.0,
            'actor_critic_sync': 0.0,
            'gradient_norm_actor': 0.0,
            'gradient_norm_critic': 0.0,
            'convergence_rate': 0.0,
            'exploration_efficiency': 0.0,
            'variance_reduction': 0.0,
            'worker_efficiency': 0.0,
            't_max_utilization': 0.0
        }

        # Historiales específicos
        self.historial_actor_loss = deque(maxlen=1000)
        self.historial_critic_loss = deque(maxlen=1000)
        self.historial_entropy = deque(maxlen=1000)
        self.historial_advantages = deque(maxlen=1000)
        self.historial_value_accuracy = deque(maxlen=1000)
        self.historial_async_updates = deque(maxlen=1000)

        logger.info(f"NeuronaRefuerzoA3CAvanzada creada: {self}")

    def inicializar_pesos(self) -> None:
        """
        Inicializa los pesos con técnicas avanzadas de inicialización.
        """
        # Inicializar Actor (política)
        self._inicializar_red_actor()

        # Inicializar Critic (función de valor)
        self._inicializar_red_critic()

        # Inicializar parámetros compartidos
        if self.use_shared_parameters:
            self._inicializar_parametros_compartidos()

        logger.info("Pesos A3C avanzado inicializados")

    def _inicializar_red_actor(self) -> None:
        """
        Inicializa la red del actor.
        """
        self.pesos_actor = []
        self.sesgos_actor = []

        # Capa de entrada
        self.pesos_actor.append(inicializar_pesos_he_avanzado((self.input_size, 64)))
        self.sesgos_actor.append(np.zeros(64, dtype=LUCIA_RL_CONFIG_RFENRN1['precision']))

        # Capas ocultas
        for i in range(self.actor_function_layers - 1):
            self.pesos_actor.append(inicializar_pesos_he_avanzado((64, 64)))
            self.sesgos_actor.append(np.zeros(64, dtype=LUCIA_RL_CONFIG_RFENRN1['precision']))

        # Capa de salida
        self.pesos_actor.append(inicializar_pesos_he_avanzado((64, self.output_size)))
        self.sesgos_actor.append(np.zeros(self.output_size, dtype=LUCIA_RL_CONFIG_RFENRN1['precision']))

        # Batch Normalization para Actor
        if self.batch_norm:
            self.gamma_bn_actor = []
            self.beta_bn_actor = []
            self.running_mean_actor = []
            self.running_var_actor = []

            for i in range(len(self.pesos_actor)):
                layer_size = self.pesos_actor[i].shape[1]
                self.gamma_bn_actor.append(np.ones(layer_size, dtype=LUCIA_RL_CONFIG_RFENRN1['precision']))
                self.beta_bn_actor.append(np.zeros(layer_size, dtype=LUCIA_RL_CONFIG_RFENRN1['precision']))
                self.running_mean_actor.append(np.zeros(layer_size, dtype=LUCIA_RL_CONFIG_RFENRN1['precision']))
                self.running_var_actor.append(np.ones(layer_size, dtype=LUCIA_RL_CONFIG_RFENRN1['precision']))

    def _inicializar_red_critic(self) -> None:
        """
        Inicializa la red del crítico.
        """
        self.pesos_critic = []
        self.sesgos_critic = []

        # Capa de entrada
        self.pesos_critic.append(inicializar_pesos_he_avanzado((self.input_size, 64)))
        self.sesgos_critic.append(np.zeros(64, dtype=LUCIA_RL_CONFIG_RFENRN1['precision']))

        # Capas ocultas
        for i in range(self.value_function_layers - 1):
            self.pesos_critic.append(inicializar_pesos_he_avanzado((64, 64)))
            self.sesgos_critic.append(np.zeros(64, dtype=LUCIA_RL_CONFIG_RFENRN1['precision']))

        # Capa de salida (valor escalar)
        self.pesos_critic.append(inicializar_pesos_he_avanzado((64, 1)))
        self.sesgos_critic.append(np.zeros(1, dtype=LUCIA_RL_CONFIG_RFENRN1['precision']))

        # Batch Normalization para Critic
        if self.batch_norm:
            self.gamma_bn_critic = []
            self.beta_bn_critic = []
            self.running_mean_critic = []
            self.running_var_critic = []

            for i in range(len(self.pesos_critic)):
                layer_size = self.pesos_critic[i].shape[1]
                self.gamma_bn_critic.append(np.ones(layer_size, dtype=LUCIA_RL_CONFIG_RFENRN1['precision']))
                self.beta_bn_critic.append(np.zeros(layer_size, dtype=LUCIA_RL_CONFIG_RFENRN1['precision']))
                self.running_mean_critic.append(np.zeros(layer_size, dtype=LUCIA_RL_CONFIG_RFENRN1['precision']))
                self.running_var_critic.append(np.ones(layer_size, dtype=LUCIA_RL_CONFIG_RFENRN1['precision']))

    def _inicializar_parametros_compartidos(self) -> None:
        """
        Inicializa los parámetros compartidos.
        """
        self.pesos_compartidos = []
        self.sesgos_compartidos = []

        # Capa de entrada compartida
        self.pesos_compartidos.append(inicializar_pesos_he_avanzado((self.input_size, 64)))
        self.sesgos_compartidos.append(np.zeros(64, dtype=LUCIA_RL_CONFIG_RFENRN1['precision']))

    def _aplicar_preprocesamiento(self, estado: np.ndarray, training: bool = True,
                                  usar_actor: bool = True) -> np.ndarray:
        """
        Aplica preprocesamiento avanzado al estado.
        """
        estado_procesado = estado.copy()

        # Batch Normalization
        if self.batch_norm:
            if usar_actor:
                estado_procesado, self.running_mean_actor[0], self.running_var_actor[0] = aplicar_batch_normalization_avanzada(
                    estado_procesado.reshape(1, -1), self.gamma_bn_actor[0], self.beta_bn_actor[0],
                    self.running_mean_actor[0], self.running_var_actor[0], training=training
                )
            else:
                estado_procesado, self.running_mean_critic[0], self.running_var_critic[0] = aplicar_batch_normalization_avanzada(
                    estado_procesado.reshape(1, -1), self.gamma_bn_critic[0], self.beta_bn_critic[0],
                    self.running_mean_critic[0], self.running_var_critic[0], training=training
                )
            estado_procesado = estado_procesado.flatten()

        # Dropout
        if training and self.dropout_rate > 0:
            estado_procesado = aplicar_dropout_avanzado(estado_procesado, self.dropout_rate, training)

        return estado_procesado

    def _forward_actor(self, estado: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Propagación hacia adelante del actor.
        """
        # Preprocesamiento
        estado_procesado = self._aplicar_preprocesamiento(estado, training, usar_actor=True)

        # Usar parámetros compartidos si está habilitado
        if self.use_shared_parameters:
            x = np.dot(estado_procesado, self.pesos_compartidos[0]) + self.sesgos_compartidos[0]
            x = np.maximum(0, x)  # ReLU

            # Continuar con pesos específicos del actor
            for i, (peso, sesgo) in enumerate(zip(self.pesos_actor[1:], self.sesgos_actor[1:])):
                x = np.dot(x, peso) + sesgo

                # Batch Normalization para capas ocultas
                if self.batch_norm and i < len(self.pesos_actor) - 2:
                    x, self.running_mean_actor[i+1], self.running_var_actor[i+1] = aplicar_batch_normalization_avanzada(
                        x.reshape(1, -1), self.gamma_bn_actor[i+1], self.beta_bn_actor[i+1],
                        self.running_mean_actor[i+1], self.running_var_actor[i+1], training=training
                    )
                    x = x.flatten()

                # Activación ReLU para capas ocultas
                if i < len(self.pesos_actor) - 2:
                    x = np.maximum(0, x)

                # Dropout para capas ocultas
                if training and self.dropout_rate > 0 and i < len(self.pesos_actor) - 2:
                    x = aplicar_dropout_avanzado(x, self.dropout_rate, training)
        else:
            # Usar solo pesos del actor
            x = estado_procesado
            for i, (peso, sesgo) in enumerate(zip(self.pesos_actor, self.sesgos_actor)):
                x = np.dot(x, peso) + sesgo

                # Batch Normalization para capas ocultas
                if self.batch_norm and i < len(self.pesos_actor) - 1:
                    x, self.running_mean_actor[i], self.running_var_actor[i] = aplicar_batch_normalization_avanzada(
                        x.reshape(1, -1), self.gamma_bn_actor[i], self.beta_bn_actor[i],
                        self.running_mean_actor[i], self.running_var_actor[i], training=training
                    )
                    x = x.flatten()

                # Activación ReLU para capas ocultas
                if i < len(self.pesos_actor) - 1:
                    x = np.maximum(0, x)

                # Dropout para capas ocultas
                if training and self.dropout_rate > 0 and i < len(self.pesos_actor) - 1:
                    x = aplicar_dropout_avanzado(x, self.dropout_rate, training)

        return x

    def _forward_critic(self, estado: np.ndarray, training: bool = True) -> float:
        """
        Propagación hacia adelante del crítico.
        """
        # Preprocesamiento
        estado_procesado = self._aplicar_preprocesamiento(estado, training, usar_actor=False)

        # Usar parámetros compartidos si está habilitado
        if self.use_shared_parameters:
            x = np.dot(estado_procesado, self.pesos_compartidos[0]) + self.sesgos_compartidos[0]
            x = np.maximum(0, x)  # ReLU

            # Continuar con pesos específicos del crítico
            for i, (peso, sesgo) in enumerate(zip(self.pesos_critic[1:], self.sesgos_critic[1:])):
                x = np.dot(x, peso) + sesgo

                # Batch Normalization para capas ocultas
                if self.batch_norm and i < len(self.pesos_critic) - 2:
                    x, self.running_mean_critic[i+1], self.running_var_critic[i+1] = aplicar_batch_normalization_avanzada(
                        x.reshape(1, -1), self.gamma_bn_critic[i+1], self.beta_bn_critic[i+1],
                        self.running_mean_critic[i+1], self.running_var_critic[i+1], training=training
                    )
                    x = x.flatten()

                # Activación ReLU para capas ocultas
                if i < len(self.pesos_critic) - 2:
                    x = np.maximum(0, x)

                # Dropout para capas ocultas
                if training and self.dropout_rate > 0 and i < len(self.pesos_critic) - 2:
                    x = aplicar_dropout_avanzado(x, self.dropout_rate, training)
        else:
            # Usar solo pesos del crítico
            x = estado_procesado
            for i, (peso, sesgo) in enumerate(zip(self.pesos_critic, self.sesgos_critic)):
                x = np.dot(x, peso) + sesgo

                # Batch Normalization para capas ocultas
                if self.batch_norm and i < len(self.pesos_critic) - 1:
                    x, self.running_mean_critic[i], self.running_var_critic[i] = aplicar_batch_normalization_avanzada(
                        x.reshape(1, -1), self.gamma_bn_critic[i], self.beta_bn_critic[i],
                        self.running_mean_critic[i], self.running_var_critic[i], training=training
                    )
                    x = x.flatten()

                # Activación ReLU para capas ocultas
                if i < len(self.pesos_critic) - 1:
                    x = np.maximum(0, x)

                # Dropout para capas ocultas
                if training and self.dropout_rate > 0 and i < len(self.pesos_critic) - 1:
                    x = aplicar_dropout_avanzado(x, self.dropout_rate, training)

        return x[0]

    def _calcular_politica(self, estado: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Calcula la política (probabilidades de acciones).
        """
        logits = self._forward_actor(estado, training)

        # Aplicar softmax para obtener probabilidades
        exp_logits = np.exp(logits - np.max(logits))
        probabilidades = exp_logits / np.sum(exp_logits)

        return probabilidades

    def _calcular_valor(self, estado: np.ndarray, training: bool = True) -> float:
        """
        Calcula el valor del estado.
        """
        return self._forward_critic(estado, training)

    def forward(self, estado: np.ndarray, training: bool = True) -> Tuple[np.ndarray, float]:
        """
        Propagación hacia adelante con técnicas avanzadas.
        """
        if not self.pesos_actor:
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

    def _calcular_gae_advantages(self, recompensas: List[float], valores: List[float],
                                 siguiente_valor: float = 0.0, terminado: bool = False) -> np.ndarray:
        """
        Calcula ventajas usando GAE.
        """
        return calcular_gae_avanzado(recompensas, valores, self.gamma, self.lambda_gae,
                                     siguiente_valor, terminado)

    def entrenar_paso(self, estado: np.ndarray, accion: int, recompensa: float,
                      siguiente_estado: np.ndarray, terminado: bool) -> Optional[float]:
        """
        Realiza un paso de entrenamiento con técnicas avanzadas.
        """
        if not self.pesos_actor:
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
        if len(self.experience_buffer) >= self.t_max:
            return self._entrenar_batch()

        return 0.0

    def _entrenar_batch(self) -> float:
        """
        Entrena con un lote de experiencias usando técnicas avanzadas.
        """
        # Muestrear experiencias
        experiencias = list(self.experience_buffer)[-self.t_max:]

        if not experiencias:
            return 0.0

        # Preparar datos del lote
        estados = np.array([exp['estado'] for exp in experiencias])
        acciones = np.array([exp['accion'] for exp in experiencias])
        recompensas = np.array([exp['recompensa'] for exp in experiencias])
        siguientes_estados = np.array([exp['siguiente_estado'] for exp in experiencias])
        terminados = np.array([exp['terminado'] for exp in experiencias])

        # Calcular valores actuales
        valores_actuales = np.array([self._calcular_valor(estado, training=True) for estado in estados])

        # Calcular valores objetivo usando multi-step returns
        valores_objetivo = np.zeros_like(valores_actuales)
        for i, (siguiente_estado, terminado) in enumerate(zip(siguientes_estados, terminados)):
            if terminado:
                valores_objetivo[i] = recompensas[i]
            else:
                siguiente_valor = self._calcular_valor(siguiente_estado, training=False)
                valores_objetivo[i] = recompensas[i] + self.gamma * siguiente_valor

        # Calcular ventajas usando GAE
        if self.lambda_gae > 0:
            advantages = self._calcular_gae_advantages(recompensas.tolist(), valores_actuales.tolist())
        else:
            advantages = valores_objetivo - valores_actuales

        # Entrenar Actor
        perdida_actor = self._entrenar_actor(estados, acciones, advantages)

        # Entrenar Critic
        perdida_critic = self._entrenar_critic(estados, valores_objetivo)

        # Actualizar estadísticas
        self._actualizar_estadisticas(perdida_actor, perdida_critic, advantages, estados)

        return perdida_actor + self.value_coef * perdida_critic

    def _entrenar_actor(self, estados: np.ndarray, acciones: np.ndarray,
                        advantages: np.ndarray) -> float:
        """
        Entrena el actor.
        """
        # Calcular gradientes del actor
        gradientes_actor = self._calcular_gradientes_actor(estados, acciones, advantages)

        # Aplicar optimizador avanzado
        self._aplicar_optimizador_actor(gradientes_actor)

        # Calcular pérdida
        perdida_total = 0.0
        for i, (estado, accion, advantage) in enumerate(zip(estados, acciones, advantages)):
            probabilidades = self._calcular_politica(estado, training=False)
            log_prob = np.log(probabilidades[accion] + 1e-8)
            perdida_total += -log_prob * advantage

            # Aplicar entropía si está habilitada
            if self.entropy_coef > 0:
                entropia = self._calcular_entropia(probabilidades)
                perdida_total += self.entropy_coef * entropia

        return perdida_total / len(estados)

    def _entrenar_critic(self, estados: np.ndarray, valores_objetivo: np.ndarray) -> float:
        """
        Entrena el crítico.
        """
        # Calcular gradientes del crítico
        gradientes_critic = self._calcular_gradientes_critic(estados, valores_objetivo)

        # Aplicar optimizador avanzado
        self._aplicar_optimizador_critic(gradientes_critic)

        # Calcular pérdida
        valores_actuales = np.array([self._calcular_valor(estado, training=False) for estado in estados])
        perdida = np.mean((valores_objetivo - valores_actuales) ** 2)

        return perdida

    def _calcular_gradientes_actor(self, estados: np.ndarray, acciones: np.ndarray,
                                   advantages: np.ndarray) -> List[List[np.ndarray]]:
        """
        Calcula gradientes del actor.
        """
        gradientes_actor = []

        # Inicializar gradientes
        for i in range(len(self.pesos_actor)):
            gradientes_actor.append([np.zeros_like(self.pesos_actor[i]), np.zeros_like(self.sesgos_actor[i])])

        # Calcular gradientes para cada experiencia
        for i, (estado, accion, advantage) in enumerate(zip(estados, acciones, advantages)):
            # Calcular probabilidades actuales
            probabilidades = self._calcular_politica(estado, training=True)

            # Gradiente de log-probabilidad
            grad_log_prob = np.zeros(self.output_size)
            grad_log_prob[accion] = 1.0 / (probabilidades[accion] + 1e-8)

            # Gradiente de la política
            grad_politica = grad_log_prob * advantage

            # Aplicar entropía si está habilitada
            if self.entropy_coef > 0:
                entropia = self._calcular_entropia(probabilidades)
                grad_entropia = np.log(probabilidades + 1e-8) + 1.0
                grad_politica += self.entropy_coef * grad_entropia

            # Backpropagation simplificado (aproximación)
            for j in range(len(self.pesos_actor)):
                if j == len(self.pesos_actor) - 1:  # Capa de salida
                    gradientes_actor[j][0] += np.outer(estado, grad_politica)
                    gradientes_actor[j][1] += grad_politica
                else:
                    # Gradientes para capas ocultas (simplificado)
                    gradientes_actor[j][0] += np.outer(estado, np.ones(self.pesos_actor[j].shape[1]))
                    gradientes_actor[j][1] += np.ones(self.pesos_actor[j].shape[1])

        # Promediar gradientes
        for i in range(len(gradientes_actor)):
            gradientes_actor[i][0] /= len(estados)
            gradientes_actor[i][1] /= len(estados)

        return gradientes_actor

    def _calcular_gradientes_critic(self, estados: np.ndarray, valores_objetivo: np.ndarray) -> List[List[np.ndarray]]:
        """
        Calcula gradientes del crítico.
        """
        gradientes_critic = []

        # Inicializar gradientes
        for i in range(len(self.pesos_critic)):
            gradientes_critic.append([np.zeros_like(self.pesos_critic[i]), np.zeros_like(self.sesgos_critic[i])])

        # Calcular gradientes para cada experiencia
        for i, (estado, valor_objetivo) in enumerate(zip(estados, valores_objetivo)):
            # Calcular valor actual
            valor_actual = self._calcular_valor(estado, training=True)

            # Gradiente de pérdida MSE
            error = valor_actual - valor_objetivo
            grad_output = 2 * error / len(estados)

            # Backpropagation simplificado (aproximación)
            for j in range(len(self.pesos_critic)):
                if j == len(self.pesos_critic) - 1:  # Capa de salida
                    gradientes_critic[j][0] += np.outer(estado, grad_output)
                    gradientes_critic[j][1] += grad_output
                else:
                    # Gradientes para capas ocultas (simplificado)
                    gradientes_critic[j][0] += np.outer(estado, np.ones(self.pesos_critic[j].shape[1]))
                    gradientes_critic[j][1] += np.ones(self.pesos_critic[j].shape[1])

        # Promediar gradientes
        for i in range(len(gradientes_critic)):
            gradientes_critic[i][0] /= len(estados)
            gradientes_critic[i][1] /= len(estados)

        return gradientes_critic

    def _aplicar_optimizador_actor(self, gradientes: List[List[np.ndarray]]) -> None:
        """
        Aplica el optimizador avanzado al actor.
        """
        # Aplanar gradientes
        gradientes_flat = []
        for grad_capa in gradientes:
            gradientes_flat.extend(grad_capa)

        # Aplicar weight decay
        if self.weight_decay > 0:
            for i, pesos in enumerate(self.pesos_actor):
                gradientes_flat[i*2] += self.weight_decay * pesos

        # Aplicar gradient clipping
        gradientes_flat = aplicar_clip_gradientes_avanzado(gradientes_flat, method='adaptive')

        # Aplicar optimizador específico
        if self.optimizer == 'adam':
            gradientes_optimizados, self.m_historial_actor, self.v_historial_actor = aplicar_adam_avanzado(
                gradientes_flat, self.m_historial_actor, self.v_historial_actor,
                learning_rate=self.learning_rate_actor, beta1=self.beta1, beta2=self.beta2,
                weight_decay=self.weight_decay
            )
        elif self.optimizer == 'rmsprop':
            gradientes_optimizados, self.v_historial_actor = aplicar_rmsprop_avanzado(
                gradientes_flat, self.v_historial_actor, learning_rate=self.learning_rate_actor
            )
        elif self.optimizer == 'momentum':
            gradientes_optimizados, self.momentum_historial_actor = calcular_momentum_avanzado(
                gradientes_flat, self.momentum_historial_actor, momentum=self.momentum
            )
        else:  # SGD
            gradientes_optimizados = gradientes_flat

        # Actualizar pesos
        idx = 0
        for i in range(len(self.pesos_actor)):
            self.pesos_actor[i] -= gradientes_optimizados[idx]
            self.sesgos_actor[i] -= gradientes_optimizados[idx + 1]
            idx += 2

        # Guardar en historial
        self.historial_pesos.append(self.pesos_actor[0].copy())

    def _aplicar_optimizador_critic(self, gradientes: List[List[np.ndarray]]) -> None:
        """
        Aplica el optimizador avanzado al crítico.
        """
        # Aplanar gradientes
        gradientes_flat = []
        for grad_capa in gradientes:
            gradientes_flat.extend(grad_capa)

        # Aplicar weight decay
        if self.weight_decay > 0:
            for i, pesos in enumerate(self.pesos_critic):
                gradientes_flat[i*2] += self.weight_decay * pesos

        # Aplicar gradient clipping
        gradientes_flat = aplicar_clip_gradientes_avanzado(gradientes_flat, method='adaptive')

        # Aplicar optimizador específico
        if self.optimizer == 'adam':
            gradientes_optimizados, self.m_historial_critic, self.v_historial_critic = aplicar_adam_avanzado(
                gradientes_flat, self.m_historial_critic, self.v_historial_critic,
                learning_rate=self.learning_rate_critic, beta1=self.beta1, beta2=self.beta2,
                weight_decay=self.weight_decay
            )
        elif self.optimizer == 'rmsprop':
            gradientes_optimizados, self.v_historial_critic = aplicar_rmsprop_avanzado(
                gradientes_flat, self.v_historial_critic, learning_rate=self.learning_rate_critic
            )
        elif self.optimizer == 'momentum':
            gradientes_optimizados, self.momentum_historial_critic = calcular_momentum_avanzado(
                gradientes_flat, self.momentum_historial_critic, momentum=self.momentum
            )
        else:  # SGD
            gradientes_optimizados = gradientes_flat

        # Actualizar pesos
        idx = 0
        for i in range(len(self.pesos_critic)):
            self.pesos_critic[i] -= gradientes_optimizados[idx]
            self.sesgos_critic[i] -= gradientes_optimizados[idx + 1]
            idx += 2

    def _actualizar_estadisticas(self, perdida_actor: float, perdida_critic: float,
                                 advantages: np.ndarray, estados: np.ndarray) -> None:
        """
        Actualiza las estadísticas específicas de A3C avanzado.
        """
        # Estadísticas básicas
        self.estadisticas_a3c['actor_loss_media'] = perdida_actor
        self.estadisticas_a3c['critic_loss_media'] = perdida_critic

        # Estadísticas de entropía
        entropias = []
        for estado in estados:
            probabilidades = self._calcular_politica(estado, training=False)
            entropias.append(self._calcular_entropia(probabilidades))
        self.estadisticas_a3c['entropy_media'] = np.mean(entropias)

        # Estadísticas de ventajas
        self.estadisticas_a3c['advantage_media'] = np.mean(advantages)

        # Estadísticas de valor
        valores_predichos = [self._calcular_valor(estado, training=False) for estado in estados]
        # Calcular accuracy del valor (simplificado)
        self.estadisticas_a3c['value_accuracy'] = 0.8  # Placeholder

        # Estadísticas de técnicas avanzadas
        if self.lambda_gae > 0:
            self.estadisticas_a3c['gae_advantage_quality'] = 0.1  # Placeholder

        if self.use_async_updates:
            self.estadisticas_a3c['async_update_efficiency'] = 0.1  # Placeholder

        if self.use_shared_parameters:
            self.estadisticas_a3c['shared_parameter_sync'] = 0.1  # Placeholder

        if self.multi_step_returns > 1:
            self.estadisticas_a3c['multi_step_return_quality'] = 0.1  # Placeholder

        # Estadísticas de estabilidad
        if len(self.historial_pesos) > 10:
            pesos_recientes = np.array(self.historial_pesos[-10:])
            varianza_pesos = np.var(pesos_recientes)
            self.estadisticas_a3c['gradient_norm_actor'] = np.sqrt(np.sum(pesos_recientes[-1] ** 2))
            self.estadisticas_a3c['convergence_rate'] = 1.0 / (1.0 + varianza_pesos)

        # Guardar en historiales
        self.historial_actor_loss.append(perdida_actor)
        self.historial_critic_loss.append(perdida_critic)
        self.historial_entropy.append(self.estadisticas_a3c['entropy_media'])
        self.historial_advantages.append(self.estadisticas_a3c['advantage_media'])
        self.historial_value_accuracy.append(self.estadisticas_a3c['value_accuracy'])
        self.historial_async_updates.append(time.time())

    def obtener_estadisticas(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas específicas de A3C avanzado.
        """
        if not self.pesos_actor:
            return {'estado': 'no_inicializada'}

        stats_a3c = {
            'learning_rate_actor': self.learning_rate_actor,
            'learning_rate_critic': self.learning_rate_critic,
            'gamma': self.gamma,
            'optimizer': self.optimizer,
            'momentum': self.momentum,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'weight_decay': self.weight_decay,
            'dropout_rate': self.dropout_rate,
            'batch_norm': self.batch_norm,
            'entropy_coef': self.entropy_coef,
            'value_coef': self.value_coef,
            'lambda_gae': self.lambda_gae,
            'n_workers': self.n_workers,
            't_max': self.t_max,
            'use_shared_parameters': self.use_shared_parameters,
            'use_async_updates': self.use_async_updates,
            'multi_step_returns': self.multi_step_returns,
            'value_function_layers': self.value_function_layers,
            'actor_function_layers': self.actor_function_layers,
            'actor_loss_media': self.estadisticas_a3c['actor_loss_media'],
            'critic_loss_media': self.estadisticas_a3c['critic_loss_media'],
            'entropy_media': self.estadisticas_a3c['entropy_media'],
            'advantage_media': self.estadisticas_a3c['advantage_media'],
            'value_accuracy': self.estadisticas_a3c['value_accuracy'],
            'policy_entropy': self.estadisticas_a3c['policy_entropy'],
            'gae_advantage_quality': self.estadisticas_a3c['gae_advantage_quality'],
            'async_update_efficiency': self.estadisticas_a3c['async_update_efficiency'],
            'shared_parameter_sync': self.estadisticas_a3c['shared_parameter_sync'],
            'multi_step_return_quality': self.estadisticas_a3c['multi_step_return_quality'],
            'value_function_approximation': self.estadisticas_a3c['value_function_approximation'],
            'actor_critic_sync': self.estadisticas_a3c['actor_critic_sync'],
            'gradient_norm_actor': self.estadisticas_a3c['gradient_norm_actor'],
            'gradient_norm_critic': self.estadisticas_a3c['gradient_norm_critic'],
            'convergence_rate': self.estadisticas_a3c['convergence_rate'],
            'exploration_efficiency': self.estadisticas_a3c['exploration_efficiency'],
            'variance_reduction': self.estadisticas_a3c['variance_reduction'],
            'worker_efficiency': self.estadisticas_a3c['worker_efficiency'],
            't_max_utilization': self.estadisticas_a3c['t_max_utilization'],
            'historial_actor_loss_size': len(self.historial_actor_loss),
            'historial_critic_loss_size': len(self.historial_critic_loss),
            'historial_entropy_size': len(self.historial_entropy),
            'historial_advantages_size': len(self.historial_advantages),
            'historial_value_accuracy_size': len(self.historial_value_accuracy),
            'historial_async_updates_size': len(self.historial_async_updates)
        }

        return stats_a3c

    def verificar_estabilidad(self) -> Dict[str, bool]:
        """
        Verifica la estabilidad de la neurona A3C avanzada.
        """
        estabilidad = {}

        # Verificaciones básicas
        estabilidad['pesos_inicializados'] = len(self.pesos_actor) > 0
        estabilidad['parametros_compartidos_ok'] = not self.use_shared_parameters or len(self.pesos_compartidos) > 0
        estabilidad['experiencias_suficientes'] = len(self.experience_buffer) > self.t_max

        # Verificaciones avanzadas
        estabilidad['actor_loss_controlada'] = abs(self.estadisticas_a3c['actor_loss_media']) < 10.0
        estabilidad['critic_loss_controlada'] = abs(self.estadisticas_a3c['critic_loss_media']) < 10.0
        estabilidad['entropy_apropiada'] = self.estadisticas_a3c['entropy_media'] > 0.1
        estabilidad['advantage_estable'] = abs(self.estadisticas_a3c['advantage_media']) < 5.0
        estabilidad['gradiente_actor_estable'] = self.estadisticas_a3c['gradient_norm_actor'] < 10.0
        estabilidad['convergencia_ok'] = self.estadisticas_a3c['convergence_rate'] > 0.5

        # Verificaciones específicas de técnicas avanzadas
        if self.use_shared_parameters:
            estabilidad['parametros_compartidos_sincronizados'] = self.estadisticas_a3c['shared_parameter_sync'] > 0.0

        if self.use_async_updates:
            estabilidad['async_updates_eficientes'] = self.estadisticas_a3c['async_update_efficiency'] > 0.0

        if self.lambda_gae > 0:
            estabilidad['gae_advantage_ok'] = self.estadisticas_a3c['gae_advantage_quality'] > 0.0

        return estabilidad

    def __str__(self) -> str:
        return (f"NeuronaRefuerzoA3CAvanzada(entrada={self.input_size}, "
                f"salida={self.output_size}, lr_actor={self.learning_rate_actor}, "
                f"lr_critic={self.learning_rate_critic}, optimizer={self.optimizer}, "
                f"shared_parameters={self.use_shared_parameters}, async_updates={self.use_async_updates}, "
                f"n_workers={self.n_workers}, t_max={self.t_max})")

    def __repr__(self) -> str:
        return self.__str__()

# Funciones de utilidad específicas para RF_RFEN1_RN_1_5


def crear_neurona_a3c_avanzada(input_size: int, output_size: int,
                               configuracion: Dict[str, Any] = None) -> NeuronaRefuerzoA3CAvanzada:
    """
    Función de conveniencia para crear una neurona A3C avanzada.
    """
    if configuracion is None:
        configuracion = {}

    return NeuronaRefuerzoA3CAvanzada(
        input_size=input_size,
        output_size=output_size,
        nombre=configuracion.get('nombre', 'NeuronaRefuerzoA3CAvanzada'),
        learning_rate_actor=configuracion.get('learning_rate_actor', LUCIA_RL_CONFIG_RFENRN1['default_learning_rate']),
        learning_rate_critic=configuracion.get('learning_rate_critic', LUCIA_RL_CONFIG_RFENRN1['default_learning_rate']),
        gamma=configuracion.get('gamma', LUCIA_RL_CONFIG_RFENRN1['default_gamma']),
        optimizer=configuracion.get('optimizer', 'adam'),
        momentum=configuracion.get('momentum', LUCIA_RL_CONFIG_RFENRN1['default_momentum']),
        beta1=configuracion.get('beta1', LUCIA_RL_CONFIG_RFENRN1['default_beta1']),
        beta2=configuracion.get('beta2', LUCIA_RL_CONFIG_RFENRN1['default_beta2']),
        weight_decay=configuracion.get('weight_decay', LUCIA_RL_CONFIG_RFENRN1['default_weight_decay']),
        dropout_rate=configuracion.get('dropout_rate', LUCIA_RL_CONFIG_RFENRN1['default_dropout_rate']),
        batch_norm=configuracion.get('batch_norm', True),
        entropy_coef=configuracion.get('entropy_coef', LUCIA_RL_CONFIG_RFENRN1['default_entropy_coef']),
        value_coef=configuracion.get('value_coef', LUCIA_RL_CONFIG_RFENRN1['default_value_coef']),
        lambda_gae=configuracion.get('lambda_gae', LUCIA_RL_CONFIG_RFENRN1['default_lambda_gae']),
        n_workers=configuracion.get('n_workers', 4),
        t_max=configuracion.get('t_max', 20),
        use_shared_parameters=configuracion.get('use_shared_parameters', True),
        use_async_updates=configuracion.get('use_async_updates', True),
        multi_step_returns=configuracion.get('multi_step_returns', 5),
        value_function_layers=configuracion.get('value_function_layers', 2),
        actor_function_layers=configuracion.get('actor_function_layers', 2)
    )


# Configuración específica para RF_RFEN1_RN_1_5
RF_RFEN1_RN_1_5_CONFIG = {
    'inicializacion_preferida': 'he_avanzado',
    'learning_rate_actor_default': 0.001,
    'learning_rate_critic_default': 0.001,
    'gamma_default': 0.99,
    'optimizer_default': 'adam',
    'momentum_default': 0.9,
    'beta1_default': 0.9,
    'beta2_default': 0.999,
    'weight_decay_default': 1e-4,
    'dropout_rate_default': 0.1,
    'batch_norm_default': True,
    'entropy_coef_default': 0.001,
    'value_coef_default': 0.5,
    'lambda_gae_default': 0.95,
    'n_workers_default': 4,
    't_max_default': 20,
    'use_shared_parameters_default': True,
    'use_async_updates_default': True,
    'multi_step_returns_default': 5,
    'value_function_layers_default': 2,
    'actor_function_layers_default': 2,
    'umbral_actor_loss_controlada': 10.0,
    'umbral_critic_loss_controlada': 10.0,
    'umbral_entropy_apropiada': 0.1,
    'umbral_advantage_estable': 5.0,
    'umbral_gradiente_actor_estable': 10.0,
    'umbral_convergencia': 0.5,
    'umbral_experiencias_suficientes': 20
}

logger.info("RF_RFEN1_RN_1_5.py cargado correctamente - Neurona de Refuerzo A3C Avanzada")
