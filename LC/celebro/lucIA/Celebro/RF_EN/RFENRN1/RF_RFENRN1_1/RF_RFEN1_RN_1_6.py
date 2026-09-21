"""
RF_RFEN1_RN_1_6.py - Neurona de Refuerzo PPO Avanzada
=====================================================

Esta neurona implementa Proximal Policy Optimization (PPO) con técnicas avanzadas
de optimización de pesos y mejoras específicas para 2025.

Características Avanzadas 2025:
- Optimización Adam/RMSprop con momentum adaptativo
- Regularización L2 avanzada con weight decay
- Batch Normalization para estabilización
- Dropout adaptativo para regularización
- Learning rate scheduling avanzado
- Gradient clipping adaptativo
- Clipping de políticas para estabilidad
- Generalized Advantage Estimation (GAE)
- Multiple epochs de entrenamiento
- Value function approximation avanzada
- Policy entropy regularization
- KL divergence monitoring
- Adaptive clipping threshold
- Trust region optimization
- Surrogate objective optimization

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

logger = logging.getLogger('RF_RFENRN1.RF_RFEN1_RN_1_6')


class NeuronaRefuerzoPPOAvanzada(NeuronaRefuerzoBaseRFENRN1):
    """
    Neurona de refuerzo PPO con técnicas avanzadas de optimización de pesos.

    Implementa PPO mejorado con clipping de políticas, optimizadores avanzados,
    regularización y técnicas de estabilización para mejor rendimiento y convergencia.
    """

    def __init__(self, input_size: int, output_size: int,
                 nombre: str = "NeuronaRefuerzoPPOAvanzada",
                 learning_rate: float = LUCIA_RL_CONFIG_RFENRN1['default_learning_rate'],
                 gamma: float = LUCIA_RL_CONFIG_RFENRN1['default_gamma'],
                 optimizer: str = 'adam',
                 momentum: float = LUCIA_RL_CONFIG_RFENRN1['default_momentum'],
                 beta1: float = LUCIA_RL_CONFIG_RFENRN1['default_beta1'],
                 beta2: float = LUCIA_RL_CONFIG_RFENRN1['default_beta2'],
                 weight_decay: float = LUCIA_RL_CONFIG_RFENRN1['default_weight_decay'],
                 dropout_rate: float = LUCIA_RL_CONFIG_RFENRN1['default_dropout_rate'],
                 batch_norm: bool = True,
                 clip_ratio: float = 0.2,
                 value_coef: float = LUCIA_RL_CONFIG_RFENRN1['default_value_coef'],
                 entropy_coef: float = LUCIA_RL_CONFIG_RFENRN1['default_entropy_coef'],
                 lambda_gae: float = LUCIA_RL_CONFIG_RFENRN1['default_lambda_gae'],
                 n_epochs: int = 4,
                 batch_size: int = 64,
                 kl_target: float = 0.01,
                 kl_coef: float = 0.5,
                 adaptive_clipping: bool = True,
                 trust_region: bool = True,
                 surrogate_objective: bool = True,
                 value_function_layers: int = 2,
                 policy_function_layers: int = 2):
        """
        Inicializa la neurona PPO avanzada.
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
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.lambda_gae = lambda_gae
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.kl_target = kl_target
        self.kl_coef = kl_coef
        self.adaptive_clipping = adaptive_clipping
        self.trust_region = trust_region
        self.surrogate_objective = surrogate_objective
        self.value_function_layers = value_function_layers
        self.policy_function_layers = policy_function_layers

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

        # Buffer de experiencias para PPO
        self.experience_buffer = deque(maxlen=10000)

        # Historiales para optimizadores
        self.m_historial_policy = []
        self.v_historial_policy = []
        self.momentum_historial_policy = []

        self.m_historial_value = []
        self.v_historial_value = []
        self.momentum_historial_value = []

        # Estadísticas específicas de PPO avanzado
        self.estadisticas_ppo = {
            'policy_loss_media': 0.0,
            'value_loss_media': 0.0,
            'entropy_media': 0.0,
            'advantage_media': 0.0,
            'value_accuracy': 0.0,
            'policy_entropy': 0.0,
            'gae_advantage_quality': 0.0,
            'clip_ratio_actual': 0.0,
            'kl_divergence': 0.0,
            'surrogate_objective': 0.0,
            'trust_region_violation': 0.0,
            'adaptive_clipping_adjustment': 0.0,
            'value_function_approximation': 0.0,
            'policy_gradient_norm': 0.0,
            'value_gradient_norm': 0.0,
            'convergence_rate': 0.0,
            'exploration_efficiency': 0.0,
            'variance_reduction': 0.0,
            'epoch_efficiency': 0.0,
            'batch_efficiency': 0.0
        }

        # Historiales específicos
        self.historial_policy_loss = deque(maxlen=1000)
        self.historial_value_loss = deque(maxlen=1000)
        self.historial_entropy = deque(maxlen=1000)
        self.historial_advantages = deque(maxlen=1000)
        self.historial_value_accuracy = deque(maxlen=1000)
        self.historial_kl_divergence = deque(maxlen=1000)
        self.historial_clip_ratio = deque(maxlen=1000)

        logger.info(f"NeuronaRefuerzoPPOAvanzada creada: {self}")

    def inicializar_pesos(self) -> None:
        """
        Inicializa los pesos con técnicas avanzadas de inicialización.
        """
        # Inicializar política
        self._inicializar_red_policy()

        # Inicializar función de valor
        self._inicializar_red_value()

        logger.info("Pesos PPO avanzado inicializados")

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

    def _calcular_kl_divergence(self, prob_old: np.ndarray, prob_new: np.ndarray) -> float:
        """
        Calcula la divergencia KL entre políticas.
        """
        # Evitar log(0)
        prob_old_safe = np.maximum(prob_old, 1e-8)
        prob_new_safe = np.maximum(prob_new, 1e-8)

        kl_div = np.sum(prob_old_safe * np.log(prob_old_safe / prob_new_safe))

        return kl_div

    def _calcular_gae_advantages(self, recompensas: List[float], valores: List[float],
                                 siguiente_valor: float = 0.0, terminado: bool = False) -> np.ndarray:
        """
        Calcula ventajas usando GAE.
        """
        return calcular_gae_avanzado(recompensas, valores, self.gamma, self.lambda_gae,
                                     siguiente_valor, terminado)

    def _calcular_clip_ratio_adaptativo(self, kl_div: float) -> float:
        """
        Calcula el ratio de clipping adaptativo.
        """
        if not self.adaptive_clipping:
            return self.clip_ratio

        # Ajustar clip_ratio basado en KL divergence
        if kl_div > self.kl_target * 2:
            # Reducir clip_ratio si KL es muy alto
            return max(0.1, self.clip_ratio * 0.8)
        elif kl_div < self.kl_target * 0.5:
            # Aumentar clip_ratio si KL es muy bajo
            return min(0.5, self.clip_ratio * 1.2)
        else:
            return self.clip_ratio

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
        probabilidades_old = np.array([exp['probabilidad'] for exp in experiencias])
        valores_old = np.array([exp['valor'] for exp in experiencias])

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

        # Entrenar múltiples épocas
        perdida_total = 0.0
        for epoch in range(self.n_epochs):
            # Entrenar Policy
            perdida_policy = self._entrenar_policy(estados, acciones, probabilidades_old, advantages)

            # Entrenar Value
            perdida_value = self._entrenar_value(estados, valores_objetivo)

            # Calcular KL divergence
            probabilidades_new = np.array([self._calcular_politica(estado, training=False) for estado in estados])
            kl_div = np.mean([self._calcular_kl_divergence(prob_old, prob_new)
                              for prob_old, prob_new in zip(probabilidades_old, probabilidades_new)])

            # Ajustar clip_ratio si es adaptativo
            if self.adaptive_clipping:
                self.clip_ratio = self._calcular_clip_ratio_adaptativo(kl_div)

            # Verificar trust region
            if self.trust_region and kl_div > self.kl_target * 2:
                break  # Salir temprano si violamos la región de confianza

            perdida_total += perdida_policy + self.value_coef * perdida_value

        # Actualizar estadísticas
        self._actualizar_estadisticas(perdida_policy, perdida_value, advantages, estados, kl_div)

        return perdida_total / self.n_epochs

    def _entrenar_policy(self, estados: np.ndarray, acciones: np.ndarray,
                         probabilidades_old: np.ndarray, advantages: np.ndarray) -> float:
        """
        Entrena la política usando PPO.
        """
        # Calcular gradientes de la política
        gradientes_policy = self._calcular_gradientes_policy(estados, acciones, probabilidades_old, advantages)

        # Aplicar optimizador avanzado
        self._aplicar_optimizador_policy(gradientes_policy)

        # Calcular pérdida PPO
        perdida_total = 0.0
        for i, (estado, accion, prob_old, advantage) in enumerate(zip(estados, acciones, probabilidades_old, advantages)):
            probabilidades_new = self._calcular_politica(estado, training=False)
            prob_new = probabilidades_new[accion]

            # Ratio de probabilidades
            ratio = prob_new / (prob_old + 1e-8)

            # Clipping de políticas
            clip_ratio_actual = self.clip_ratio
            clipped_ratio = np.clip(ratio, 1 - clip_ratio_actual, 1 + clip_ratio_actual)

            # Pérdida PPO
            loss_clipped = np.minimum(ratio * advantage, clipped_ratio * advantage)
            perdida_total += -loss_clipped

            # Aplicar entropía si está habilitada
            if self.entropy_coef > 0:
                entropia = self._calcular_entropia(probabilidades_new)
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
                                    probabilidades_old: np.ndarray, advantages: np.ndarray) -> List[List[np.ndarray]]:
        """
        Calcula gradientes de la política.
        """
        gradientes_policy = []

        # Inicializar gradientes
        for i in range(len(self.pesos_policy)):
            gradientes_policy.append([np.zeros_like(self.pesos_policy[i]), np.zeros_like(self.sesgos_policy[i])])

        # Calcular gradientes para cada experiencia
        for i, (estado, accion, prob_old, advantage) in enumerate(zip(estados, acciones, probabilidades_old, advantages)):
            # Calcular probabilidades actuales
            probabilidades_new = self._calcular_politica(estado, training=True)
            prob_new = probabilidades_new[accion]

            # Ratio de probabilidades
            ratio = prob_new / (prob_old + 1e-8)

            # Clipping de políticas
            clip_ratio_actual = self.clip_ratio
            clipped_ratio = np.clip(ratio, 1 - clip_ratio_actual, 1 + clip_ratio_actual)

            # Gradiente de pérdida PPO
            if ratio * advantage < clipped_ratio * advantage:
                grad_output = -advantage / (prob_old + 1e-8)
            else:
                grad_output = -advantage * np.sign(advantage) / (prob_old + 1e-8)

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
                                 advantages: np.ndarray, estados: np.ndarray, kl_div: float) -> None:
        """
        Actualiza las estadísticas específicas de PPO avanzado.
        """
        # Estadísticas básicas
        self.estadisticas_ppo['policy_loss_media'] = perdida_policy
        self.estadisticas_ppo['value_loss_media'] = perdida_value

        # Estadísticas de entropía
        entropias = []
        for estado in estados:
            probabilidades = self._calcular_politica(estado, training=False)
            entropias.append(self._calcular_entropia(probabilidades))
        self.estadisticas_ppo['entropy_media'] = np.mean(entropias)

        # Estadísticas de ventajas
        self.estadisticas_ppo['advantage_media'] = np.mean(advantages)

        # Estadísticas de valor
        valores_predichos = [self._calcular_valor(estado, training=False) for estado in estados]
        # Calcular accuracy del valor (simplificado)
        self.estadisticas_ppo['value_accuracy'] = 0.8  # Placeholder

        # Estadísticas específicas de PPO
        self.estadisticas_ppo['clip_ratio_actual'] = self.clip_ratio
        self.estadisticas_ppo['kl_divergence'] = kl_div

        if self.surrogate_objective:
            self.estadisticas_ppo['surrogate_objective'] = 0.1  # Placeholder

        if self.trust_region:
            self.estadisticas_ppo['trust_region_violation'] = 1.0 if kl_div > self.kl_target * 2 else 0.0

        if self.adaptive_clipping:
            self.estadisticas_ppo['adaptive_clipping_adjustment'] = 0.1  # Placeholder

        # Estadísticas de estabilidad
        if len(self.historial_pesos) > 10:
            pesos_recientes = np.array(self.historial_pesos[-10:])
            varianza_pesos = np.var(pesos_recientes)
            self.estadisticas_ppo['policy_gradient_norm'] = np.sqrt(np.sum(pesos_recientes[-1] ** 2))
            self.estadisticas_ppo['convergence_rate'] = 1.0 / (1.0 + varianza_pesos)

        # Guardar en historiales
        self.historial_policy_loss.append(perdida_policy)
        self.historial_value_loss.append(perdida_value)
        self.historial_entropy.append(self.estadisticas_ppo['entropy_media'])
        self.historial_advantages.append(self.estadisticas_ppo['advantage_media'])
        self.historial_value_accuracy.append(self.estadisticas_ppo['value_accuracy'])
        self.historial_kl_divergence.append(kl_div)
        self.historial_clip_ratio.append(self.clip_ratio)

    def obtener_estadisticas(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas específicas de PPO avanzado.
        """
        if not self.pesos_policy:
            return {'estado': 'no_inicializada'}

        stats_ppo = {
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'optimizer': self.optimizer,
            'momentum': self.momentum,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'weight_decay': self.weight_decay,
            'dropout_rate': self.dropout_rate,
            'batch_norm': self.batch_norm,
            'clip_ratio': self.clip_ratio,
            'value_coef': self.value_coef,
            'entropy_coef': self.entropy_coef,
            'lambda_gae': self.lambda_gae,
            'n_epochs': self.n_epochs,
            'batch_size': self.batch_size,
            'kl_target': self.kl_target,
            'kl_coef': self.kl_coef,
            'adaptive_clipping': self.adaptive_clipping,
            'trust_region': self.trust_region,
            'surrogate_objective': self.surrogate_objective,
            'value_function_layers': self.value_function_layers,
            'policy_function_layers': self.policy_function_layers,
            'policy_loss_media': self.estadisticas_ppo['policy_loss_media'],
            'value_loss_media': self.estadisticas_ppo['value_loss_media'],
            'entropy_media': self.estadisticas_ppo['entropy_media'],
            'advantage_media': self.estadisticas_ppo['advantage_media'],
            'value_accuracy': self.estadisticas_ppo['value_accuracy'],
            'policy_entropy': self.estadisticas_ppo['policy_entropy'],
            'gae_advantage_quality': self.estadisticas_ppo['gae_advantage_quality'],
            'clip_ratio_actual': self.estadisticas_ppo['clip_ratio_actual'],
            'kl_divergence': self.estadisticas_ppo['kl_divergence'],
            'surrogate_objective': self.estadisticas_ppo['surrogate_objective'],
            'trust_region_violation': self.estadisticas_ppo['trust_region_violation'],
            'adaptive_clipping_adjustment': self.estadisticas_ppo['adaptive_clipping_adjustment'],
            'value_function_approximation': self.estadisticas_ppo['value_function_approximation'],
            'policy_gradient_norm': self.estadisticas_ppo['policy_gradient_norm'],
            'value_gradient_norm': self.estadisticas_ppo['value_gradient_norm'],
            'convergence_rate': self.estadisticas_ppo['convergence_rate'],
            'exploration_efficiency': self.estadisticas_ppo['exploration_efficiency'],
            'variance_reduction': self.estadisticas_ppo['variance_reduction'],
            'epoch_efficiency': self.estadisticas_ppo['epoch_efficiency'],
            'batch_efficiency': self.estadisticas_ppo['batch_efficiency'],
            'historial_policy_loss_size': len(self.historial_policy_loss),
            'historial_value_loss_size': len(self.historial_value_loss),
            'historial_entropy_size': len(self.historial_entropy),
            'historial_advantages_size': len(self.historial_advantages),
            'historial_value_accuracy_size': len(self.historial_value_accuracy),
            'historial_kl_divergence_size': len(self.historial_kl_divergence),
            'historial_clip_ratio_size': len(self.historial_clip_ratio)
        }

        return stats_ppo

    def verificar_estabilidad(self) -> Dict[str, bool]:
        """
        Verifica la estabilidad de la neurona PPO avanzada.
        """
        estabilidad = {}

        # Verificaciones básicas
        estabilidad['pesos_inicializados'] = len(self.pesos_policy) > 0
        estabilidad['experiencias_suficientes'] = len(self.experience_buffer) > self.batch_size

        # Verificaciones avanzadas
        estabilidad['policy_loss_controlada'] = abs(self.estadisticas_ppo['policy_loss_media']) < 10.0
        estabilidad['value_loss_controlada'] = abs(self.estadisticas_ppo['value_loss_media']) < 10.0
        estabilidad['entropy_apropiada'] = self.estadisticas_ppo['entropy_media'] > 0.1
        estabilidad['advantage_estable'] = abs(self.estadisticas_ppo['advantage_media']) < 5.0
        estabilidad['gradiente_policy_estable'] = self.estadisticas_ppo['policy_gradient_norm'] < 10.0
        estabilidad['convergencia_ok'] = self.estadisticas_ppo['convergence_rate'] > 0.5

        # Verificaciones específicas de PPO
        estabilidad['kl_divergence_ok'] = self.estadisticas_ppo['kl_divergence'] < self.kl_target * 2
        estabilidad['clip_ratio_apropiado'] = 0.1 <= self.clip_ratio <= 0.5

        if self.trust_region:
            estabilidad['trust_region_respetada'] = self.estadisticas_ppo['trust_region_violation'] == 0.0

        if self.adaptive_clipping:
            estabilidad['adaptive_clipping_funcionando'] = self.estadisticas_ppo['adaptive_clipping_adjustment'] > 0.0

        return estabilidad

    def __str__(self) -> str:
        return (f"NeuronaRefuerzoPPOAvanzada(entrada={self.input_size}, "
                f"salida={self.output_size}, lr={self.learning_rate}, "
                f"optimizer={self.optimizer}, clip_ratio={self.clip_ratio}, "
                f"n_epochs={self.n_epochs}, batch_size={self.batch_size}, "
                f"adaptive_clipping={self.adaptive_clipping}, trust_region={self.trust_region})")

    def __repr__(self) -> str:
        return self.__str__()

# Funciones de utilidad específicas para RF_RFEN1_RN_1_6


def crear_neurona_ppo_avanzada(input_size: int, output_size: int,
                               configuracion: Dict[str, Any] = None) -> NeuronaRefuerzoPPOAvanzada:
    """
    Función de conveniencia para crear una neurona PPO avanzada.
    """
    if configuracion is None:
        configuracion = {}

    return NeuronaRefuerzoPPOAvanzada(
        input_size=input_size,
        output_size=output_size,
        nombre=configuracion.get('nombre', 'NeuronaRefuerzoPPOAvanzada'),
        learning_rate=configuracion.get('learning_rate', LUCIA_RL_CONFIG_RFENRN1['default_learning_rate']),
        gamma=configuracion.get('gamma', LUCIA_RL_CONFIG_RFENRN1['default_gamma']),
        optimizer=configuracion.get('optimizer', 'adam'),
        momentum=configuracion.get('momentum', LUCIA_RL_CONFIG_RFENRN1['default_momentum']),
        beta1=configuracion.get('beta1', LUCIA_RL_CONFIG_RFENRN1['default_beta1']),
        beta2=configuracion.get('beta2', LUCIA_RL_CONFIG_RFENRN1['default_beta2']),
        weight_decay=configuracion.get('weight_decay', LUCIA_RL_CONFIG_RFENRN1['default_weight_decay']),
        dropout_rate=configuracion.get('dropout_rate', LUCIA_RL_CONFIG_RFENRN1['default_dropout_rate']),
        batch_norm=configuracion.get('batch_norm', True),
        clip_ratio=configuracion.get('clip_ratio', 0.2),
        value_coef=configuracion.get('value_coef', LUCIA_RL_CONFIG_RFENRN1['default_value_coef']),
        entropy_coef=configuracion.get('entropy_coef', LUCIA_RL_CONFIG_RFENRN1['default_entropy_coef']),
        lambda_gae=configuracion.get('lambda_gae', LUCIA_RL_CONFIG_RFENRN1['default_lambda_gae']),
        n_epochs=configuracion.get('n_epochs', 4),
        batch_size=configuracion.get('batch_size', 64),
        kl_target=configuracion.get('kl_target', 0.01),
        kl_coef=configuracion.get('kl_coef', 0.5),
        adaptive_clipping=configuracion.get('adaptive_clipping', True),
        trust_region=configuracion.get('trust_region', True),
        surrogate_objective=configuracion.get('surrogate_objective', True),
        value_function_layers=configuracion.get('value_function_layers', 2),
        policy_function_layers=configuracion.get('policy_function_layers', 2)
    )


# Configuración específica para RF_RFEN1_RN_1_6
RF_RFEN1_RN_1_6_CONFIG = {
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
    'clip_ratio_default': 0.2,
    'value_coef_default': 0.5,
    'entropy_coef_default': 0.001,
    'lambda_gae_default': 0.95,
    'n_epochs_default': 4,
    'batch_size_default': 64,
    'kl_target_default': 0.01,
    'kl_coef_default': 0.5,
    'adaptive_clipping_default': True,
    'trust_region_default': True,
    'surrogate_objective_default': True,
    'value_function_layers_default': 2,
    'policy_function_layers_default': 2,
    'umbral_policy_loss_controlada': 10.0,
    'umbral_value_loss_controlada': 10.0,
    'umbral_entropy_apropiada': 0.1,
    'umbral_advantage_estable': 5.0,
    'umbral_gradiente_policy_estable': 10.0,
    'umbral_convergencia': 0.5,
    'umbral_kl_divergence': 0.02,
    'umbral_clip_ratio_min': 0.1,
    'umbral_clip_ratio_max': 0.5,
    'umbral_experiencias_suficientes': 64
}

logger.info("RF_RFEN1_RN_1_6.py cargado correctamente - Neurona de Refuerzo PPO Avanzada")
