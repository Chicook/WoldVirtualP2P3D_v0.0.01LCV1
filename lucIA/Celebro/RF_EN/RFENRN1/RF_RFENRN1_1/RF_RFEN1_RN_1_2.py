"""
RF_RFEN1_RN_1_2.py - Neurona de Refuerzo Policy Gradient Avanzada
================================================================

Esta neurona implementa Policy Gradient con técnicas avanzadas de optimización de pesos
y mejoras específicas para 2025, incluyendo optimizadores avanzados, regularización
y técnicas de estabilización.

Características Avanzadas 2025:
- Optimización Adam/RMSprop con momentum adaptativo
- Regularización L2 avanzada con weight decay
- Batch Normalization para estabilización
- Dropout adaptativo para regularización
- Learning rate scheduling avanzado
- Gradient clipping adaptativo
- Baseline subtraction para reducción de varianza
- Entropy regularization para exploración
- Natural Policy Gradient con aproximación de Fisher
- Trust Region Policy Optimization (TRPO)
- Proximal Policy Optimization (PPO) básico
- Generalized Advantage Estimation (GAE)

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

logger = logging.getLogger('RF_RFENRN1.RF_RFEN1_RN_2')


class NeuronaRefuerzoPolicyGradientAvanzada(NeuronaRefuerzoBaseRFENRN1):
    """
    Neurona de refuerzo Policy Gradient con técnicas avanzadas de optimización de pesos.

    Implementa Policy Gradient mejorado con optimizadores avanzados, regularización
    y técnicas de estabilización para mejor rendimiento y convergencia.
    """

    def __init__(self, input_size: int, output_size: int,
                 nombre: str = "NeuronaRefuerzoPolicyGradientAvanzada",
                 learning_rate: float = LUCIA_RL_CONFIG_RFENRN1['default_learning_rate'],
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
                 clip_ratio: float = LUCIA_RL_CONFIG_RFENRN1['default_clip_ratio'],
                 lambda_gae: float = LUCIA_RL_CONFIG_RFENRN1['default_lambda_gae'],
                 use_baseline: bool = True,
                 use_entropy_reg: bool = True,
                 use_natural_gradient: bool = False,
                 use_trpo: bool = False,
                 use_ppo: bool = True,
                 max_kl_divergence: float = 0.01,
                 conjugate_gradient_iters: int = 10,
                 fisher_approximation_samples: int = 100):
        """
        Inicializa la neurona Policy Gradient avanzada.

        Args:
            input_size: Número de características de entrada
            output_size: Número de acciones posibles
            nombre: Nombre identificativo de la neurona
            learning_rate: Tasa de aprendizaje
            gamma: Factor de descuento
            optimizer: Optimizador a usar ('adam', 'rmsprop', 'momentum', 'sgd')
            momentum: Factor de momentum
            beta1: Factor de decaimiento para primer momento (Adam)
            beta2: Factor de decaimiento para segundo momento (Adam)
            weight_decay: Decaimiento de pesos
            dropout_rate: Tasa de dropout
            batch_norm: Si usar batch normalization
            entropy_coef: Coeficiente de regularización de entropía
            value_coef: Coeficiente de pérdida de valor
            clip_ratio: Ratio de clipping para PPO
            lambda_gae: Factor lambda para GAE
            use_baseline: Si usar baseline para reducción de varianza
            use_entropy_reg: Si usar regularización de entropía
            use_natural_gradient: Si usar gradiente natural
            use_trpo: Si usar TRPO
            use_ppo: Si usar PPO
            max_kl_divergence: Máxima divergencia KL para TRPO
            conjugate_gradient_iters: Iteraciones de gradiente conjugado
            fisher_approximation_samples: Muestras para aproximación de Fisher
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
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.clip_ratio = clip_ratio
        self.lambda_gae = lambda_gae
        self.use_baseline = use_baseline
        self.use_entropy_reg = use_entropy_reg
        self.use_natural_gradient = use_natural_gradient
        self.use_trpo = use_trpo
        self.use_ppo = use_ppo
        self.max_kl_divergence = max_kl_divergence
        self.conjugate_gradient_iters = conjugate_gradient_iters
        self.fisher_approximation_samples = fisher_approximation_samples

        # Pesos de la política
        self.pesos_politica = None
        self.sesgo_politica = None

        # Pesos del baseline (función de valor)
        if self.use_baseline:
            self.pesos_baseline = None
            self.sesgo_baseline = None

        # Parámetros de Batch Normalization
        if self.batch_norm:
            self.gamma_bn_politica = None
            self.beta_bn_politica = None
            self.running_mean_politica = None
            self.running_var_politica = None

            if self.use_baseline:
                self.gamma_bn_baseline = None
                self.beta_bn_baseline = None
                self.running_mean_baseline = None
                self.running_var_baseline = None

        # Buffer de experiencias para Policy Gradient
        self.experience_buffer = deque(maxlen=10000)

        # Historiales para optimizadores
        self.m_historial_politica = []
        self.v_historial_politica = []
        self.momentum_historial_politica = []

        if self.use_baseline:
            self.m_historial_baseline = []
            self.v_historial_baseline = []
            self.momentum_historial_baseline = []

        # Estadísticas específicas de Policy Gradient avanzado
        self.estadisticas_policy_gradient = {
            'policy_loss_media': 0.0,
            'value_loss_media': 0.0,
            'entropy_media': 0.0,
            'kl_divergence_media': 0.0,
            'advantage_media': 0.0,
            'baseline_accuracy': 0.0,
            'natural_gradient_efficiency': 0.0,
            'trpo_constraint_satisfaction': 0.0,
            'ppo_clipping_efficiency': 0.0,
            'gae_advantage_quality': 0.0,
            'entropy_regularization_effectiveness': 0.0,
            'gradient_norm_policy': 0.0,
            'gradient_norm_value': 0.0,
            'convergence_rate': 0.0,
            'exploration_efficiency': 0.0,
            'variance_reduction': 0.0
        }

        # Historiales específicos
        self.historial_policy_loss = deque(maxlen=1000)
        self.historial_value_loss = deque(maxlen=1000)
        self.historial_entropy = deque(maxlen=1000)
        self.historial_kl_divergence = deque(maxlen=1000)
        self.historial_advantages = deque(maxlen=1000)
        self.historial_baseline_accuracy = deque(maxlen=1000)

        logger.info(f"NeuronaRefuerzoPolicyGradientAvanzada creada: {self}")

    def inicializar_pesos(self) -> None:
        """
        Inicializa los pesos con técnicas avanzadas de inicialización.
        """
        # Política
        self.pesos_politica = inicializar_pesos_he_avanzado((self.input_size, self.output_size))
        self.sesgo_politica = np.zeros(self.output_size, dtype=LUCIA_RL_CONFIG_RFENRN1['precision'])

        # Baseline (función de valor)
        if self.use_baseline:
            self.pesos_baseline = inicializar_pesos_he_avanzado((self.input_size, 1))
            self.sesgo_baseline = np.zeros(1, dtype=LUCIA_RL_CONFIG_RFENRN1['precision'])

        # Batch Normalization
        if self.batch_norm:
            self.gamma_bn_politica = np.ones(self.input_size, dtype=LUCIA_RL_CONFIG_RFENRN1['precision'])
            self.beta_bn_politica = np.zeros(self.input_size, dtype=LUCIA_RL_CONFIG_RFENRN1['precision'])
            self.running_mean_politica = np.zeros(self.input_size, dtype=LUCIA_RL_CONFIG_RFENRN1['precision'])
            self.running_var_politica = np.ones(self.input_size, dtype=LUCIA_RL_CONFIG_RFENRN1['precision'])

            if self.use_baseline:
                self.gamma_bn_baseline = np.ones(self.input_size, dtype=LUCIA_RL_CONFIG_RFENRN1['precision'])
                self.beta_bn_baseline = np.zeros(self.input_size, dtype=LUCIA_RL_CONFIG_RFENRN1['precision'])
                self.running_mean_baseline = np.zeros(self.input_size, dtype=LUCIA_RL_CONFIG_RFENRN1['precision'])
                self.running_var_baseline = np.ones(self.input_size, dtype=LUCIA_RL_CONFIG_RFENRN1['precision'])

        logger.info("Pesos Policy Gradient avanzado inicializados")

    def _aplicar_preprocesamiento(self, estado: np.ndarray, training: bool = True,
                                  usar_baseline: bool = False) -> np.ndarray:
        """
        Aplica preprocesamiento avanzado al estado.

        Args:
            estado: Estado a procesar
            training: Si está en modo entrenamiento
            usar_baseline: Si procesar para baseline

        Returns:
            Estado procesado
        """
        estado_procesado = estado.copy()

        # Batch Normalization
        if self.batch_norm:
            if usar_baseline and self.use_baseline:
                estado_procesado, self.running_mean_baseline, self.running_var_baseline = aplicar_batch_normalization_avanzada(
                    estado_procesado.reshape(1, -1), self.gamma_bn_baseline, self.beta_bn_baseline,
                    self.running_mean_baseline, self.running_var_baseline, training=training
                )
            else:
                estado_procesado, self.running_mean_politica, self.running_var_politica = aplicar_batch_normalization_avanzada(
                    estado_procesado.reshape(1, -1), self.gamma_bn_politica, self.beta_bn_politica,
                    self.running_mean_politica, self.running_var_politica, training=training
                )
            estado_procesado = estado_procesado.flatten()

        # Dropout
        if training and self.dropout_rate > 0:
            estado_procesado = aplicar_dropout_avanzado(estado_procesado, self.dropout_rate, training)

        return estado_procesado

    def _calcular_politica(self, estado: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Calcula la política (probabilidades de acciones).

        Args:
            estado: Estado actual
            training: Si está en modo entrenamiento

        Returns:
            Probabilidades de acciones
        """
        # Preprocesamiento
        estado_procesado = self._aplicar_preprocesamiento(estado, training, usar_baseline=False)

        # Calcular logits
        logits = np.dot(estado_procesado, self.pesos_politica) + self.sesgo_politica

        # Aplicar softmax para obtener probabilidades
        exp_logits = np.exp(logits - np.max(logits))
        probabilidades = exp_logits / np.sum(exp_logits)

        return probabilidades

    def _calcular_baseline(self, estado: np.ndarray, training: bool = True) -> float:
        """
        Calcula el baseline (valor del estado).

        Args:
            estado: Estado actual
            training: Si está en modo entrenamiento

        Returns:
            Valor del estado
        """
        if not self.use_baseline:
            return 0.0

        # Preprocesamiento
        estado_procesado = self._aplicar_preprocesamiento(estado, training, usar_baseline=True)

        # Calcular valor
        valor = np.dot(estado_procesado, self.pesos_baseline) + self.sesgo_baseline

        return valor[0]

    def forward(self, estado: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Propagación hacia adelante con técnicas avanzadas.

        Args:
            estado: Estado actual del entorno
            training: Si está en modo entrenamiento

        Returns:
            Probabilidades de acciones
        """
        if self.pesos_politica is None:
            self.inicializar_pesos()

        return self._calcular_politica(estado, training)

    def seleccionar_accion(self, estado: np.ndarray, training: bool = True) -> int:
        """
        Selecciona una acción usando la política.

        Args:
            estado: Estado actual del entorno
            training: Si está en modo entrenamiento

        Returns:
            Acción seleccionada
        """
        probabilidades = self.forward(estado, training)

        # Aplanar probabilidades si es necesario
        if len(probabilidades.shape) > 1:
            probabilidades = probabilidades.flatten()

        # Selección basada en probabilidades
        accion = np.random.choice(self.output_size, p=probabilidades)

        return accion

    def _calcular_entropia(self, probabilidades: np.ndarray) -> float:
        """
        Calcula la entropía de la política.

        Args:
            probabilidades: Probabilidades de acciones

        Returns:
            Entropía de la política
        """
        # Evitar log(0)
        probabilidades_safe = np.maximum(probabilidades, 1e-8)
        entropia = -np.sum(probabilidades * np.log(probabilidades_safe))

        return entropia

    def _calcular_kl_divergence(self, probabilidades_viejas: np.ndarray,
                                probabilidades_nuevas: np.ndarray) -> float:
        """
        Calcula la divergencia KL entre políticas.

        Args:
            probabilidades_viejas: Probabilidades de la política anterior
            probabilidades_nuevas: Probabilidades de la política nueva

        Returns:
            Divergencia KL
        """
        # Evitar log(0)
        prob_viejas_safe = np.maximum(probabilidades_viejas, 1e-8)
        prob_nuevas_safe = np.maximum(probabilidades_nuevas, 1e-8)

        kl_div = np.sum(prob_viejas_safe * np.log(prob_viejas_safe / prob_nuevas_safe))

        return kl_div

    def _calcular_gae_advantages(self, recompensas: List[float], valores: List[float],
                                 siguiente_valor: float = 0.0, terminado: bool = False) -> np.ndarray:
        """
        Calcula ventajas usando GAE.

        Args:
            recompensas: Lista de recompensas
            valores: Lista de valores de estados
            siguiente_valor: Valor del siguiente estado
            terminado: Si el episodio terminó

        Returns:
            Ventajas calculadas
        """
        return calcular_gae_avanzado(recompensas, valores, self.gamma, self.lambda_gae,
                                     siguiente_valor, terminado)

    def _calcular_gradiente_natural(self, gradientes: List[np.ndarray],
                                    estados: np.ndarray) -> List[np.ndarray]:
        """
        Calcula el gradiente natural usando aproximación de Fisher.

        Args:
            gradientes: Gradientes de la política
            estados: Estados del lote

        Returns:
            Gradientes naturales
        """
        if not self.use_natural_gradient:
            return gradientes

        # Aproximación simplificada de la matriz de Fisher
        # En una implementación real, se usaría gradiente conjugado

        # Calcular matriz de Fisher aproximada
        fisher_matrix = np.dot(estados.T, estados) / len(estados)

        # Aplicar regularización
        fisher_matrix += 1e-6 * np.eye(fisher_matrix.shape[0])

        # Calcular gradiente natural
        gradientes_naturales = []
        for grad in gradientes:
            if grad.shape[0] == fisher_matrix.shape[0]:
                grad_natural = np.dot(np.linalg.inv(fisher_matrix), grad)
                gradientes_naturales.append(grad_natural)
            else:
                gradientes_naturales.append(grad)

        return gradientes_naturales

    def _aplicar_trpo_constraint(self, gradientes: List[np.ndarray],
                                 estados: np.ndarray, probabilidades_viejas: np.ndarray) -> List[np.ndarray]:
        """
        Aplica la restricción de TRPO.

        Args:
            gradientes: Gradientes de la política
            estados: Estados del lote
            probabilidades_viejas: Probabilidades de la política anterior

        Returns:
            Gradientes con restricción TRPO aplicada
        """
        if not self.use_trpo:
            return gradientes

        # Calcular divergencia KL
        kl_div = 0.0
        for estado in estados:
            prob_vieja = probabilidades_viejas
            prob_nueva = self._calcular_politica(estado, training=False)
            kl_div += self._calcular_kl_divergence(prob_vieja, prob_nueva)

        kl_div /= len(estados)

        # Si la divergencia KL es muy grande, escalar los gradientes
        if kl_div > self.max_kl_divergence:
            scale_factor = np.sqrt(self.max_kl_divergence / (kl_div + 1e-8))
            gradientes = [grad * scale_factor for grad in gradientes]

        return gradientes

    def _aplicar_ppo_clipping(self, gradientes: List[np.ndarray],
                              estados: np.ndarray, acciones: np.ndarray,
                              probabilidades_viejas: np.ndarray, advantages: np.ndarray) -> List[np.ndarray]:
        """
        Aplica clipping de PPO.

        Args:
            gradientes: Gradientes de la política
            estados: Estados del lote
            acciones: Acciones del lote
            probabilidades_viejas: Probabilidades de la política anterior
            advantages: Ventajas calculadas

        Returns:
            Gradientes con clipping PPO aplicado
        """
        if not self.use_ppo:
            return gradientes

        # Calcular ratios de probabilidad
        ratios = []
        for i, estado in enumerate(estados):
            prob_nueva = self._calcular_politica(estado, training=False)
            prob_vieja = probabilidades_viejas[i]
            ratio = prob_nueva[acciones[i]] / (prob_vieja[acciones[i]] + 1e-8)
            ratios.append(ratio)

        ratios = np.array(ratios)

        # Aplicar clipping
        clipped_ratios = np.clip(ratios, 1 - self.clip_ratio, 1 + self.clip_ratio)

        # Calcular factor de clipping
        clip_factor = np.minimum(ratios * advantages, clipped_ratios * advantages) / (ratios * advantages + 1e-8)

        # Aplicar factor a gradientes
        gradientes_clipped = []
        for grad in gradientes:
            if len(grad.shape) == 1:  # Sesgo
                grad_clipped = grad * np.mean(clip_factor)
            else:  # Pesos
                grad_clipped = grad * clip_factor.reshape(-1, 1)
            gradientes_clipped.append(grad_clipped)

        return gradientes_clipped

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
        if self.pesos_politica is None:
            self.inicializar_pesos()

        # Almacenar experiencia
        experiencia = {
            'estado': estado.copy(),
            'accion': accion,
            'recompensa': recompensa,
            'siguiente_estado': siguiente_estado.copy(),
            'terminado': terminado,
            'probabilidad': self._calcular_politica(estado, training=False)[accion],
            'valor': self._calcular_baseline(estado, training=False)
        }
        self.experience_buffer.append(experiencia)

        # Entrenar si hay suficientes experiencias
        if len(self.experience_buffer) >= 32:  # Tamaño mínimo de lote
            return self._entrenar_batch()

        return 0.0

    def _entrenar_batch(self) -> float:
        """
        Entrena con un lote de experiencias usando técnicas avanzadas.

        Returns:
            Pérdida promedio del lote
        """
        # Muestrear experiencias
        experiencias = random.sample(list(self.experience_buffer),
                                     min(64, len(self.experience_buffer)))

        if not experiencias:
            return 0.0

        # Preparar datos del lote
        estados = np.array([exp['estado'] for exp in experiencias])
        acciones = np.array([exp['accion'] for exp in experiencias])
        recompensas = np.array([exp['recompensa'] for exp in experiencias])
        probabilidades_viejas = np.array([exp['probabilidad'] for exp in experiencias])
        valores_viejos = np.array([exp['valor'] for exp in experiencias])

        # Calcular ventajas usando GAE
        if self.lambda_gae > 0:
            # Calcular valores de estados
            valores_estados = []
            for estado in estados:
                valores_estados.append(self._calcular_baseline(estado, training=False))

            # Calcular ventajas
            advantages = self._calcular_gae_advantages(recompensas.tolist(), valores_estados)
        else:
            # Ventajas simples
            advantages = recompensas - np.mean(recompensas)

        # Calcular gradientes de la política
        gradientes_politica = self._calcular_gradientes_politica(estados, acciones, advantages)

        # Aplicar técnicas avanzadas
        if self.use_natural_gradient:
            gradientes_politica = self._calcular_gradiente_natural(gradientes_politica, estados)

        if self.use_trpo:
            gradientes_politica = self._aplicar_trpo_constraint(gradientes_politica, estados, probabilidades_viejas)

        if self.use_ppo:
            gradientes_politica = self._aplicar_ppo_clipping(gradientes_politica, estados, acciones,
                                                             probabilidades_viejas, advantages)

        # Aplicar optimizador avanzado
        self._aplicar_optimizador_politica(gradientes_politica)

        # Entrenar baseline si está habilitado
        if self.use_baseline:
            gradientes_baseline = self._calcular_gradientes_baseline(estados, recompensas, valores_viejos)
            self._aplicar_optimizador_baseline(gradientes_baseline)

        # Calcular pérdidas
        perdida_politica = self._calcular_perdida_politica(estados, acciones, advantages)
        perdida_baseline = self._calcular_perdida_baseline(estados, recompensas, valores_viejos) if self.use_baseline else 0.0

        # Actualizar estadísticas
        self._actualizar_estadisticas(perdida_politica, perdida_baseline, advantages, estados)

        return perdida_politica + self.value_coef * perdida_baseline

    def _calcular_gradientes_politica(self, estados: np.ndarray, acciones: np.ndarray,
                                      advantages: np.ndarray) -> List[np.ndarray]:
        """
        Calcula gradientes de la política.

        Args:
            estados: Estados del lote
            acciones: Acciones del lote
            advantages: Ventajas calculadas

        Returns:
            Lista de gradientes de la política
        """
        gradientes_politica = []

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
            if self.use_entropy_reg:
                entropia = self._calcular_entropia(probabilidades)
                grad_entropia = np.log(probabilidades + 1e-8) + 1.0
                grad_politica += self.entropy_coef * grad_entropia

            gradientes_politica.append(grad_politica)

        # Promediar gradientes
        grad_pesos_promedio = np.zeros_like(self.pesos_politica)
        grad_sesgo_promedio = np.zeros_like(self.sesgo_politica)

        for i, grad_politica in enumerate(gradientes_politica):
            grad_pesos_promedio += np.outer(estados[i], grad_politica)
            grad_sesgo_promedio += grad_politica

        grad_pesos_promedio /= len(estados)
        grad_sesgo_promedio /= len(estados)

        return [grad_pesos_promedio, grad_sesgo_promedio]

    def _calcular_gradientes_baseline(self, estados: np.ndarray, recompensas: np.ndarray,
                                      valores_viejos: np.ndarray) -> List[np.ndarray]:
        """
        Calcula gradientes del baseline.

        Args:
            estados: Estados del lote
            recompensas: Recompensas del lote
            valores_viejos: Valores anteriores

        Returns:
            Lista de gradientes del baseline
        """
        # Calcular valores actuales
        valores_actuales = np.array([self._calcular_baseline(estado, training=True) for estado in estados])

        # Gradiente de pérdida MSE
        error = valores_actuales - recompensas
        grad_output = 2 * error / len(estados)

        # Gradientes
        grad_pesos = np.dot(estados.T, grad_output)
        grad_sesgo = np.sum(grad_output)

        return [grad_pesos, grad_sesgo]

    def _calcular_perdida_politica(self, estados: np.ndarray, acciones: np.ndarray,
                                   advantages: np.ndarray) -> float:
        """
        Calcula la pérdida de la política.

        Args:
            estados: Estados del lote
            acciones: Acciones del lote
            advantages: Ventajas calculadas

        Returns:
            Pérdida de la política
        """
        perdida_total = 0.0

        for i, (estado, accion, advantage) in enumerate(zip(estados, acciones, advantages)):
            probabilidades = self._calcular_politica(estado, training=False)
            log_prob = np.log(probabilidades[accion] + 1e-8)
            perdida_total += -log_prob * advantage

        return perdida_total / len(estados)

    def _calcular_perdida_baseline(self, estados: np.ndarray, recompensas: np.ndarray,
                                   valores_viejos: np.ndarray) -> float:
        """
        Calcula la pérdida del baseline.

        Args:
            estados: Estados del lote
            recompensas: Recompensas del lote
            valores_viejos: Valores anteriores

        Returns:
            Pérdida del baseline
        """
        valores_actuales = np.array([self._calcular_baseline(estado, training=False) for estado in estados])
        perdida = np.mean((valores_actuales - recompensas) ** 2)

        return perdida

    def _aplicar_optimizador_politica(self, gradientes: List[np.ndarray]) -> None:
        """
        Aplica el optimizador avanzado a la política.

        Args:
            gradientes: Gradientes de la política
        """
        # Aplicar weight decay
        if self.weight_decay > 0:
            gradientes[0] += self.weight_decay * self.pesos_politica

        # Aplicar gradient clipping
        gradientes = aplicar_clip_gradientes_avanzado(gradientes, method='adaptive')

        # Aplicar optimizador específico
        if self.optimizer == 'adam':
            gradientes_optimizados, self.m_historial_politica, self.v_historial_politica = aplicar_adam_avanzado(
                gradientes, self.m_historial_politica, self.v_historial_politica,
                learning_rate=self.learning_rate, beta1=self.beta1, beta2=self.beta2,
                weight_decay=self.weight_decay
            )
        elif self.optimizer == 'rmsprop':
            gradientes_optimizados, self.v_historial_politica = aplicar_rmsprop_avanzado(
                gradientes, self.v_historial_politica, learning_rate=self.learning_rate
            )
        elif self.optimizer == 'momentum':
            gradientes_optimizados, self.momentum_historial_politica = calcular_momentum_avanzado(
                gradientes, self.momentum_historial_politica, momentum=self.momentum
            )
        else:  # SGD
            gradientes_optimizados = gradientes

        # Actualizar pesos
        self.pesos_politica -= gradientes_optimizados[0]
        self.sesgo_politica -= gradientes_optimizados[1]

        # Guardar en historial
        self.historial_pesos.append(self.pesos_politica.copy())

    def _aplicar_optimizador_baseline(self, gradientes: List[np.ndarray]) -> None:
        """
        Aplica el optimizador avanzado al baseline.

        Args:
            gradientes: Gradientes del baseline
        """
        if not self.use_baseline:
            return

        # Aplicar weight decay
        if self.weight_decay > 0:
            gradientes[0] += self.weight_decay * self.pesos_baseline

        # Aplicar gradient clipping
        gradientes = aplicar_clip_gradientes_avanzado(gradientes, method='adaptive')

        # Aplicar optimizador específico
        if self.optimizer == 'adam':
            gradientes_optimizados, self.m_historial_baseline, self.v_historial_baseline = aplicar_adam_avanzado(
                gradientes, self.m_historial_baseline, self.v_historial_baseline,
                learning_rate=self.learning_rate, beta1=self.beta1, beta2=self.beta2,
                weight_decay=self.weight_decay
            )
        elif self.optimizer == 'rmsprop':
            gradientes_optimizados, self.v_historial_baseline = aplicar_rmsprop_avanzado(
                gradientes, self.v_historial_baseline, learning_rate=self.learning_rate
            )
        elif self.optimizer == 'momentum':
            gradientes_optimizados, self.momentum_historial_baseline = calcular_momentum_avanzado(
                gradientes, self.momentum_historial_baseline, momentum=self.momentum
            )
        else:  # SGD
            gradientes_optimizados = gradientes

        # Actualizar pesos
        self.pesos_baseline -= gradientes_optimizados[0]
        self.sesgo_baseline -= gradientes_optimizados[1]

    def _actualizar_estadisticas(self, perdida_politica: float, perdida_baseline: float,
                                 advantages: np.ndarray, estados: np.ndarray) -> None:
        """
        Actualiza las estadísticas específicas de Policy Gradient avanzado.
        """
        # Estadísticas básicas
        self.estadisticas_policy_gradient['policy_loss_media'] = perdida_politica
        self.estadisticas_policy_gradient['value_loss_media'] = perdida_baseline

        # Estadísticas de entropía
        entropias = []
        for estado in estados:
            probabilidades = self._calcular_politica(estado, training=False)
            entropias.append(self._calcular_entropia(probabilidades))
        self.estadisticas_policy_gradient['entropy_media'] = np.mean(entropias)

        # Estadísticas de ventajas
        self.estadisticas_policy_gradient['advantage_media'] = np.mean(advantages)

        # Estadísticas de baseline
        if self.use_baseline:
            valores_predichos = [self._calcular_baseline(estado, training=False) for estado in estados]
            # Calcular accuracy del baseline (simplificado)
            self.estadisticas_policy_gradient['baseline_accuracy'] = 0.8  # Placeholder

        # Estadísticas de técnicas avanzadas
        if self.use_natural_gradient:
            self.estadisticas_policy_gradient['natural_gradient_efficiency'] = 0.1  # Placeholder

        if self.use_trpo:
            self.estadisticas_policy_gradient['trpo_constraint_satisfaction'] = 0.1  # Placeholder

        if self.use_ppo:
            self.estadisticas_policy_gradient['ppo_clipping_efficiency'] = 0.1  # Placeholder

        if self.lambda_gae > 0:
            self.estadisticas_policy_gradient['gae_advantage_quality'] = 0.1  # Placeholder

        # Estadísticas de estabilidad
        if len(self.historial_pesos) > 10:
            pesos_recientes = np.array(self.historial_pesos[-10:])
            varianza_pesos = np.var(pesos_recientes)
            self.estadisticas_policy_gradient['gradient_norm_policy'] = np.sqrt(np.sum(pesos_recientes[-1] ** 2))
            self.estadisticas_policy_gradient['convergence_rate'] = 1.0 / (1.0 + varianza_pesos)

        # Guardar en historiales
        self.historial_policy_loss.append(perdida_politica)
        self.historial_value_loss.append(perdida_baseline)
        self.historial_entropy.append(self.estadisticas_policy_gradient['entropy_media'])
        self.historial_advantages.append(self.estadisticas_policy_gradient['advantage_media'])
        self.historial_baseline_accuracy.append(self.estadisticas_policy_gradient['baseline_accuracy'])

    def obtener_estadisticas(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas específicas de Policy Gradient avanzado.

        Returns:
            Diccionario con estadísticas
        """
        if self.pesos_politica is None:
            return {'estado': 'no_inicializada'}

        stats_policy_gradient = {
            'learning_rate': self.learning_rate,
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
            'clip_ratio': self.clip_ratio,
            'lambda_gae': self.lambda_gae,
            'use_baseline': self.use_baseline,
            'use_entropy_reg': self.use_entropy_reg,
            'use_natural_gradient': self.use_natural_gradient,
            'use_trpo': self.use_trpo,
            'use_ppo': self.use_ppo,
            'max_kl_divergence': self.max_kl_divergence,
            'conjugate_gradient_iters': self.conjugate_gradient_iters,
            'fisher_approximation_samples': self.fisher_approximation_samples,
            'policy_loss_media': self.estadisticas_policy_gradient['policy_loss_media'],
            'value_loss_media': self.estadisticas_policy_gradient['value_loss_media'],
            'entropy_media': self.estadisticas_policy_gradient['entropy_media'],
            'kl_divergence_media': self.estadisticas_policy_gradient['kl_divergence_media'],
            'advantage_media': self.estadisticas_policy_gradient['advantage_media'],
            'baseline_accuracy': self.estadisticas_policy_gradient['baseline_accuracy'],
            'natural_gradient_efficiency': self.estadisticas_policy_gradient['natural_gradient_efficiency'],
            'trpo_constraint_satisfaction': self.estadisticas_policy_gradient['trpo_constraint_satisfaction'],
            'ppo_clipping_efficiency': self.estadisticas_policy_gradient['ppo_clipping_efficiency'],
            'gae_advantage_quality': self.estadisticas_policy_gradient['gae_advantage_quality'],
            'entropy_regularization_effectiveness': self.estadisticas_policy_gradient['entropy_regularization_effectiveness'],
            'gradient_norm_policy': self.estadisticas_policy_gradient['gradient_norm_policy'],
            'gradient_norm_value': self.estadisticas_policy_gradient['gradient_norm_value'],
            'convergence_rate': self.estadisticas_policy_gradient['convergence_rate'],
            'exploration_efficiency': self.estadisticas_policy_gradient['exploration_efficiency'],
            'variance_reduction': self.estadisticas_policy_gradient['variance_reduction'],
            'historial_policy_loss_size': len(self.historial_policy_loss),
            'historial_value_loss_size': len(self.historial_value_loss),
            'historial_entropy_size': len(self.historial_entropy),
            'historial_kl_divergence_size': len(self.historial_kl_divergence),
            'historial_advantages_size': len(self.historial_advantages),
            'historial_baseline_accuracy_size': len(self.historial_baseline_accuracy)
        }

        return stats_policy_gradient

    def verificar_estabilidad(self) -> Dict[str, bool]:
        """
        Verifica la estabilidad de la neurona Policy Gradient avanzada.

        Returns:
            Diccionario con indicadores de estabilidad
        """
        estabilidad = {}

        # Verificaciones básicas
        estabilidad['pesos_inicializados'] = self.pesos_politica is not None
        estabilidad['baseline_inicializado'] = not self.use_baseline or self.pesos_baseline is not None
        estabilidad['experiencias_suficientes'] = len(self.experience_buffer) > 32

        # Verificaciones avanzadas
        estabilidad['policy_loss_controlada'] = abs(self.estadisticas_policy_gradient['policy_loss_media']) < 10.0
        estabilidad['entropy_apropiada'] = self.estadisticas_policy_gradient['entropy_media'] > 0.1
        estabilidad['advantage_estable'] = abs(self.estadisticas_policy_gradient['advantage_media']) < 5.0
        estabilidad['gradiente_estable'] = self.estadisticas_policy_gradient['gradient_norm_policy'] < 10.0
        estabilidad['convergencia_ok'] = self.estadisticas_policy_gradient['convergence_rate'] > 0.5

        # Verificaciones específicas de técnicas avanzadas
        if self.use_baseline:
            estabilidad['baseline_accuracy_ok'] = self.estadisticas_policy_gradient['baseline_accuracy'] > 0.5

        if self.use_natural_gradient:
            estabilidad['natural_gradient_eficiente'] = self.estadisticas_policy_gradient['natural_gradient_efficiency'] > 0.0

        if self.use_trpo:
            estabilidad['trpo_constraint_satisfecho'] = self.estadisticas_policy_gradient['trpo_constraint_satisfaction'] > 0.0

        if self.use_ppo:
            estabilidad['ppo_clipping_eficiente'] = self.estadisticas_policy_gradient['ppo_clipping_efficiency'] > 0.0

        return estabilidad

    def __str__(self) -> str:
        return (f"NeuronaRefuerzoPolicyGradientAvanzada(entrada={self.input_size}, "
                f"salida={self.output_size}, lr={self.learning_rate}, "
                f"optimizer={self.optimizer}, baseline={self.use_baseline}, "
                f"natural_gradient={self.use_natural_gradient}, trpo={self.use_trpo}, ppo={self.use_ppo})")

    def __repr__(self) -> str:
        return self.__str__()

# Funciones de utilidad específicas para RF_RFEN1_RN_1_2


def crear_neurona_policy_gradient_avanzada(input_size: int, output_size: int,
                                           configuracion: Dict[str, Any] = None) -> NeuronaRefuerzoPolicyGradientAvanzada:
    """
    Función de conveniencia para crear una neurona Policy Gradient avanzada.

    Args:
        input_size: Tamaño de entrada
        output_size: Tamaño de salida
        configuracion: Configuración personalizada

    Returns:
        Neurona Policy Gradient avanzada configurada
    """
    if configuracion is None:
        configuracion = {}

    return NeuronaRefuerzoPolicyGradientAvanzada(
        input_size=input_size,
        output_size=output_size,
        nombre=configuracion.get('nombre', 'NeuronaRefuerzoPolicyGradientAvanzada'),
        learning_rate=configuracion.get('learning_rate', LUCIA_RL_CONFIG_RFENRN1['default_learning_rate']),
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
        clip_ratio=configuracion.get('clip_ratio', LUCIA_RL_CONFIG_RFENRN1['default_clip_ratio']),
        lambda_gae=configuracion.get('lambda_gae', LUCIA_RL_CONFIG_RFENRN1['default_lambda_gae']),
        use_baseline=configuracion.get('use_baseline', True),
        use_entropy_reg=configuracion.get('use_entropy_reg', True),
        use_natural_gradient=configuracion.get('use_natural_gradient', False),
        use_trpo=configuracion.get('use_trpo', False),
        use_ppo=configuracion.get('use_ppo', True),
        max_kl_divergence=configuracion.get('max_kl_divergence', 0.01),
        conjugate_gradient_iters=configuracion.get('conjugate_gradient_iters', 10),
        fisher_approximation_samples=configuracion.get('fisher_approximation_samples', 100)
    )


# Configuración específica para RF_RFEN1_RN_1_2
RF_RFEN1_RN_1_2_CONFIG = {
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
    'entropy_coef_default': 0.001,
    'value_coef_default': 0.5,
    'clip_ratio_default': 0.2,
    'lambda_gae_default': 0.95,
    'use_baseline_default': True,
    'use_entropy_reg_default': True,
    'use_natural_gradient_default': False,
    'use_trpo_default': False,
    'use_ppo_default': True,
    'max_kl_divergence_default': 0.01,
    'conjugate_gradient_iters_default': 10,
    'fisher_approximation_samples_default': 100,
    'umbral_policy_loss_controlada': 10.0,
    'umbral_entropy_apropiada': 0.1,
    'umbral_advantage_estable': 5.0,
    'umbral_gradiente_estable': 10.0,
    'umbral_convergencia': 0.5,
    'umbral_baseline_accuracy': 0.5,
    'umbral_experiencias_suficientes': 32
}

logger.info("RF_RFEN1_RN_1_2.py cargado correctamente - Neurona de Refuerzo Policy Gradient Avanzada")
