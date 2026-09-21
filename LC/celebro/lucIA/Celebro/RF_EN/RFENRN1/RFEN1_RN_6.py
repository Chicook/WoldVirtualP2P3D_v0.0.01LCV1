"""
RFEN1_RN_6.py - Neurona de Refuerzo con Proximal Policy Optimization (PPO)
===========================================================================

Esta neurona implementa el algoritmo PPO con optimización de pesos adaptativos
y técnicas avanzadas de clipping y normalización para estabilidad.

Características:
- PPO con clipping adaptativo y normalización de ventajas
- Inicialización de pesos específica para políticas proximales
- Monitoreo de ratio de políticas y estabilidad
- Adaptación automática de hiperparámetros PPO

Autor: LucIA Development Team
Versión: 1.0.0
"""

import numpy as np
import math
from typing import Tuple, Optional, Dict, Any, List
import logging
from collections import deque

# Importar las clases base y funciones de utilidad desde base.py
from .base import (
    NeuronaRefuerzoBase,
    inicializar_pesos_he,
    inicializar_pesos_xavier,
    inicializar_pesos_lecun,
    inicializar_pesos_ortogonal,
    inicializar_pesos_espectral,
    LUCIA_RL_CONFIG
)

logger = logging.getLogger('RFENRN1.RFEN1_RN_6')


class NeuronaRefuerzoPPO(NeuronaRefuerzoBase):
    """
    Neurona de refuerzo con algoritmo Proximal Policy Optimization (PPO).

    Esta neurona implementa PPO con clipping adaptativo, normalización de ventajas
    y optimización de pesos específica para políticas proximales.
    """

    def __init__(self, input_size: int, output_size: int,
                 nombre: str = "NeuronaRefuerzoPPO",
                 learning_rate: float = 0.0003,
                 gamma: float = 0.99,
                 lambda_gae: float = 0.95,
                 clip_ratio: float = 0.2,
                 clip_ratio_adaptativo: bool = True,
                 entropy_coef: float = 0.01,
                 value_coef: float = 0.5,
                 max_grad_norm: float = 0.5,
                 usar_ortogonal: bool = True,
                 epochs_ppo: int = 4):
        """
        Inicializa la neurona de refuerzo PPO.

        Args:
            input_size: Número de características de entrada
            output_size: Número de acciones posibles
            nombre: Nombre identificativo de la neurona
            learning_rate: Tasa de aprendizaje
            gamma: Factor de descuento
            lambda_gae: Parámetro lambda para GAE
            clip_ratio: Ratio de clipping inicial
            clip_ratio_adaptativo: Si usar clipping adaptativo
            entropy_coef: Coeficiente de entropía para regularización
            value_coef: Coeficiente para la función de valor
            max_grad_norm: Norma máxima para recorte de gradientes
            usar_ortogonal: Si usar inicialización ortogonal
            epochs_ppo: Número de épocas PPO por actualización
        """
        super().__init__(input_size, output_size, nombre)

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.clip_ratio = clip_ratio
        self.clip_ratio_adaptativo = clip_ratio_adaptativo
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.usar_ortogonal = usar_ortogonal
        self.epochs_ppo = epochs_ppo

        # Pesos separados para Actor y Critic
        self.pesos_actor = None
        self.sesgo_actor = None
        self.pesos_critic = None
        self.sesgo_critic = None

        # Buffer para almacenar experiencias
        self.buffer_estados = deque()
        self.buffer_acciones = deque()
        self.buffer_recompensas = deque()
        self.buffer_valores = deque()
        self.buffer_log_probs = deque()
        self.buffer_advantages = deque()
        self.buffer_returns = deque()

        # Estadísticas específicas de PPO
        self.estadisticas_ppo = {
            'policy_loss_medio': 0.0,
            'value_loss_medio': 0.0,
            'entropy_loss_medio': 0.0,
            'total_loss_medio': 0.0,
            'clip_ratio_actual': 0.0,
            'kl_divergencia': 0.0,
            'explained_variance': 0.0,
            'policy_ratio_medio': 0.0,
            'advantage_norma': 0.0,
            'gradiente_norma': 0.0,
            'epochs_completados': 0
        }

        # Historial para análisis
        self.historial_policy_loss = []
        self.historial_value_loss = []
        self.historial_entropy_loss = []
        self.historial_clip_ratios = []
        self.historial_kl_divergences = []
        self.historial_policy_ratios = []

        logger.info(f"NeuronaRefuerzoPPO creada: {self}")

    def inicializar_pesos(self) -> None:
        """
        Inicializa los pesos del Actor y Critic con técnicas específicas para PPO.
        """
        if self.usar_ortogonal:
            # Inicialización ortogonal para estabilidad en PPO
            self.pesos_actor = inicializar_pesos_ortogonal((self.input_size, self.output_size))
            self.pesos_critic = inicializar_pesos_ortogonal((self.input_size, 1))
        else:
            # Inicialización Xavier como alternativa
            self.pesos_actor = inicializar_pesos_xavier(
                (self.input_size, self.output_size),
                fan_in=self.input_size,
                fan_out=self.output_size
            )
            self.pesos_critic = inicializar_pesos_xavier(
                (self.input_size, 1),
                fan_in=self.input_size,
                fan_out=1
            )

        # Inicializar sesgos con ceros
        self.sesgo_actor = np.zeros((1, self.output_size), dtype=LUCIA_RL_CONFIG['precision'])
        self.sesgo_critic = np.zeros((1, 1), dtype=LUCIA_RL_CONFIG['precision'])

        logger.info(f"Pesos PPO inicializados con {'ortogonal' if self.usar_ortogonal else 'Xavier'}")

    def forward_actor(self, estado: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Propagación hacia adelante del Actor.

        Args:
            estado: Estado actual del entorno

        Returns:
            Tupla con (probabilidades_accion, log_probabilidades)
        """
        if self.pesos_actor is None:
            self.inicializar_pesos()

        # Calcular logits del actor
        logits = np.dot(estado, self.pesos_actor) + self.sesgo_actor

        # Aplicar softmax para obtener probabilidades
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probabilidades = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # Calcular log probabilidades
        log_probabilidades = np.log(probabilidades + 1e-8)

        return probabilidades, log_probabilidades

    def forward_critic(self, estado: np.ndarray) -> np.ndarray:
        """
        Propagación hacia adelante del Critic.

        Args:
            estado: Estado actual del entorno

        Returns:
            Valor estimado del estado
        """
        if self.pesos_critic is None:
            self.inicializar_pesos()

        # Calcular valor del estado
        valor = np.dot(estado, self.pesos_critic) + self.sesgo_critic

        return valor

    def forward(self, estado: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Propagación hacia adelante completa (Actor + Critic).

        Args:
            estado: Estado actual del entorno

        Returns:
            Tupla con (probabilidades_accion, log_probabilidades, valor_estado)
        """
        probabilidades, log_probabilidades = self.forward_actor(estado)
        valor = self.forward_critic(estado)

        return probabilidades, log_probabilidades, valor

    def seleccionar_accion(self, estado: np.ndarray) -> Tuple[int, float, float, float]:
        """
        Selecciona una acción usando el Actor-Critic.

        Args:
            estado: Estado actual del entorno

        Returns:
            Tupla con (acción, probabilidad_accion, log_probabilidad, valor_estado)
        """
        probabilidades, log_probabilidades, valor = self.forward(estado)

        # Muestrear acción según las probabilidades
        accion = np.random.choice(self.output_size, p=probabilidades[0])
        probabilidad_accion = probabilidades[0, accion]
        log_probabilidad = log_probabilidades[0, accion]
        valor_estado = valor[0, 0]

        return accion, probabilidad_accion, log_probabilidad, valor_estado

    def agregar_experiencia(self, estado: np.ndarray, accion: int, recompensa: float,
                            valor: float, log_prob: float) -> None:
        """
        Agrega una experiencia al buffer.

        Args:
            estado: Estado actual
            accion: Acción tomada
            recompensa: Recompensa recibida
            valor: Valor estimado del estado
            log_prob: Log probabilidad de la acción
        """
        self.buffer_estados.append(estado.copy())
        self.buffer_acciones.append(accion)
        self.buffer_recompensas.append(recompensa)
        self.buffer_valores.append(valor)
        self.buffer_log_probs.append(log_prob)

    def calcular_gae_advantages(self, recompensas: List[float], valores: List[float]) -> List[float]:
        """
        Calcula las ventajas usando GAE (Generalized Advantage Estimation).

        Args:
            recompensas: Lista de recompensas
            valores: Lista de valores estimados

        Returns:
            Lista de ventajas calculadas
        """
        advantages = []
        advantage = 0

        for t in reversed(range(len(recompensas))):
            if t == len(recompensas) - 1:
                next_value = 0
            else:
                next_value = valores[t + 1]

            delta = recompensas[t] + self.gamma * next_value - valores[t]
            advantage = delta + self.gamma * self.lambda_gae * advantage
            advantages.insert(0, advantage)

        return advantages

    def calcular_returns(self, recompensas: List[float], valores: List[float]) -> List[float]:
        """
        Calcula los returns descontados.

        Args:
            recompensas: Lista de recompensas
            valores: Lista de valores estimados

        Returns:
            Lista de returns calculados
        """
        returns = []
        return_acumulado = 0

        for t in reversed(range(len(recompensas))):
            if t == len(recompensas) - 1:
                next_value = 0
            else:
                next_value = valores[t + 1]

            return_acumulado = recompensas[t] + self.gamma * return_acumulado
            returns.insert(0, return_acumulado)

        return returns

    def normalizar_advantages(self, advantages: List[float]) -> List[float]:
        """
        Normaliza las ventajas para estabilizar el entrenamiento.

        Args:
            advantages: Lista de ventajas

        Returns:
            Lista de ventajas normalizadas
        """
        advantages_array = np.array(advantages)
        mean_adv = np.mean(advantages_array)
        std_adv = np.std(advantages_array)

        if std_adv < 1e-8:
            return advantages

        normalized_advantages = (advantages_array - mean_adv) / (std_adv + 1e-8)

        # Guardar estadísticas
        self.estadisticas_ppo['advantage_norma'] = np.linalg.norm(normalized_advantages)

        return normalized_advantages.tolist()

    def calcular_policy_loss(self, estados: List[np.ndarray], acciones: List[int],
                             log_probs_old: List[float], advantages: List[float]) -> Tuple[float, float]:
        """
        Calcula la pérdida de política con clipping PPO.

        Args:
            estados: Lista de estados
            acciones: Lista de acciones
            log_probs_old: Log probabilidades antiguas
            advantages: Lista de ventajas

        Returns:
            Tupla con (policy_loss, kl_divergencia)
        """
        policy_loss_total = 0
        kl_divergencia_total = 0

        for estado, accion, log_prob_old, advantage in zip(estados, acciones, log_probs_old, advantages):
            # Calcular probabilidades actuales
            _, log_probabilidades = self.forward_actor(estado)
            log_prob_actual = log_probabilidades[0, accion]

            # Calcular ratio de políticas
            ratio = np.exp(log_prob_actual - log_prob_old)

            # Clipping PPO
            clipped_ratio = np.clip(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)

            # Pérdida de política con clipping
            policy_loss = -np.minimum(ratio * advantage, clipped_ratio * advantage)
            policy_loss_total += policy_loss

            # KL divergencia
            kl_divergencia = log_prob_old - log_prob_actual
            kl_divergencia_total += kl_divergencia

        # Normalizar
        n_muestras = len(estados)
        policy_loss_medio = policy_loss_total / n_muestras
        kl_divergencia_medio = kl_divergencia_total / n_muestras

        # Guardar estadísticas
        self.estadisticas_ppo['policy_ratio_medio'] = np.mean([np.exp(log_probabilidades[0, accion] - log_prob_old)
                                                               for estado, accion, log_prob_old in zip(estados, acciones, log_probs_old)])
        self.estadisticas_ppo['kl_divergencia'] = kl_divergencia_medio

        return policy_loss_medio, kl_divergencia_medio

    def calcular_value_loss(self, estados: List[np.ndarray], returns: List[float]) -> float:
        """
        Calcula la pérdida de valor.

        Args:
            estados: Lista de estados
            returns: Lista de returns objetivo

        Returns:
            Pérdida de valor promedio
        """
        value_loss_total = 0

        for estado, return_objetivo in zip(estados, returns):
            valor_actual = self.forward_critic(estado)[0, 0]
            value_loss = 0.5 * ((valor_actual - return_objetivo) ** 2)
            value_loss_total += value_loss

        value_loss_medio = value_loss_total / len(estados)

        return value_loss_medio

    def calcular_entropy_loss(self, estados: List[np.ndarray]) -> float:
        """
        Calcula la pérdida de entropía para regularización.

        Args:
            estados: Lista de estados

        Returns:
            Pérdida de entropía promedio
        """
        entropy_loss_total = 0

        for estado in estados:
            probabilidades, _ = self.forward_actor(estado)
            entropy = -np.sum(probabilidades * np.log(probabilidades + 1e-8), axis=1)
            entropy_loss_total += np.mean(entropy)

        entropy_loss_medio = entropy_loss_total / len(estados)

        return entropy_loss_medio

    def calcular_gradientes_ppo(self, estados: List[np.ndarray], acciones: List[int],
                                log_probs_old: List[float], advantages: List[float],
                                returns: List[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calcula los gradientes para PPO.

        Args:
            estados: Lista de estados
            acciones: Lista de acciones
            log_probs_old: Log probabilidades antiguas
            advantages: Lista de ventajas
            returns: Lista de returns

        Returns:
            Tupla con (grad_actor_pesos, grad_actor_sesgo, grad_critic_pesos, grad_critic_sesgo)
        """
        if self.pesos_actor is None or self.pesos_critic is None:
            raise ValueError("Pesos no inicializados")

        # Calcular pérdidas
        policy_loss, kl_divergencia = self.calcular_policy_loss(estados, acciones, log_probs_old, advantages)
        value_loss = self.calcular_value_loss(estados, returns)
        entropy_loss = self.calcular_entropy_loss(estados)

        # Pérdida total
        total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss

        # Guardar estadísticas
        self.estadisticas_ppo['policy_loss_medio'] = policy_loss
        self.estadisticas_ppo['value_loss_medio'] = value_loss
        self.estadisticas_ppo['entropy_loss_medio'] = entropy_loss
        self.estadisticas_ppo['total_loss_medio'] = total_loss

        # Calcular gradientes (simplificado para este ejemplo)
        grad_actor_pesos = np.zeros_like(self.pesos_actor)
        grad_actor_sesgo = np.zeros_like(self.sesgo_actor)
        grad_critic_pesos = np.zeros_like(self.pesos_critic)
        grad_critic_sesgo = np.zeros_like(self.sesgo_critic)

        # Gradientes del Actor
        for estado, accion, log_prob_old, advantage in zip(estados, acciones, log_probs_old, advantages):
            _, log_probabilidades = self.forward_actor(estado)
            log_prob_actual = log_probabilidades[0, accion]

            ratio = np.exp(log_prob_actual - log_prob_old)
            clipped_ratio = np.clip(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)

            grad_log_prob = np.zeros(self.output_size)
            grad_log_prob[accion] = 1.0 / (log_probabilidades[0, accion] + 1e-8)

            grad_actor_pesos += np.outer(estado[0], grad_log_prob) * advantage
            grad_actor_sesgo += grad_log_prob * advantage

        # Gradientes del Critic
        for estado, return_objetivo in zip(estados, returns):
            valor_actual = self.forward_critic(estado)[0, 0]
            error_valor = return_objetivo - valor_actual

            grad_critic_pesos += np.outer(estado[0], np.array([error_valor]))
            grad_critic_sesgo += np.array([[error_valor]])

        # Normalizar gradientes
        n_muestras = len(estados)
        grad_actor_pesos /= n_muestras
        grad_actor_sesgo /= n_muestras
        grad_critic_pesos /= n_muestras
        grad_critic_sesgo /= n_muestras

        # Recorte de gradientes
        grad_norm = np.linalg.norm(np.concatenate([grad_actor_pesos.flatten(), grad_critic_pesos.flatten()]))
        if grad_norm > self.max_grad_norm:
            clip_coef = self.max_grad_norm / (grad_norm + 1e-8)
            grad_actor_pesos *= clip_coef
            grad_actor_sesgo *= clip_coef
            grad_critic_pesos *= clip_coef
            grad_critic_sesgo *= clip_coef

        self.estadisticas_ppo['gradiente_norma'] = grad_norm

        return grad_actor_pesos, grad_actor_sesgo, grad_critic_pesos, grad_critic_sesgo

    def actualizar_pesos(self, grad_actor_pesos: np.ndarray, grad_actor_sesgo: np.ndarray,
                         grad_critic_pesos: np.ndarray, grad_critic_sesgo: np.ndarray) -> None:
        """
        Actualiza los pesos del Actor y Critic.

        Args:
            grad_actor_pesos: Gradiente de los pesos del actor
            grad_actor_sesgo: Gradiente del sesgo del actor
            grad_critic_pesos: Gradiente de los pesos del crítico
            grad_critic_sesgo: Gradiente del sesgo del crítico
        """
        if self.pesos_actor is None or self.pesos_critic is None:
            raise ValueError("Pesos no inicializados")

        # Actualizar pesos del Actor
        self.pesos_actor += self.learning_rate * grad_actor_pesos
        self.sesgo_actor += self.learning_rate * grad_actor_sesgo

        # Actualizar pesos del Critic
        self.pesos_critic += self.learning_rate * grad_critic_pesos
        self.sesgo_critic += self.learning_rate * grad_critic_sesgo

        # Adaptar clip ratio si está habilitado
        if self.clip_ratio_adaptativo:
            self._adaptar_clip_ratio()

        # Guardar historial
        self.historial_policy_loss.append(self.estadisticas_ppo['policy_loss_medio'])
        self.historial_value_loss.append(self.estadisticas_ppo['value_loss_medio'])
        self.historial_entropy_loss.append(self.estadisticas_ppo['entropy_loss_medio'])
        self.historial_clip_ratios.append(self.clip_ratio)
        self.historial_kl_divergences.append(self.estadisticas_ppo['kl_divergencia'])

    def _adaptar_clip_ratio(self) -> None:
        """
        Adapta el clip ratio basado en la KL divergencia.
        """
        kl_divergencia = self.estadisticas_ppo['kl_divergencia']

        if kl_divergencia > 0.02:  # KL demasiado alta
            self.clip_ratio *= 0.9
        elif kl_divergencia < 0.01:  # KL demasiado baja
            self.clip_ratio *= 1.1

        # Limitar clip ratio
        self.clip_ratio = np.clip(self.clip_ratio, 0.1, 0.3)
        self.estadisticas_ppo['clip_ratio_actual'] = self.clip_ratio

    def entrenar_ppo(self) -> Dict[str, float]:
        """
        Realiza el entrenamiento PPO completo.

        Returns:
            Diccionario con estadísticas de entrenamiento
        """
        if len(self.buffer_estados) < self.epochs_ppo:
            return {'error': 'buffer_insuficiente'}

        # Convertir buffers a listas
        estados = list(self.buffer_estados)
        acciones = list(self.buffer_acciones)
        recompensas = list(self.buffer_recompensas)
        valores = list(self.buffer_valores)
        log_probs_old = list(self.buffer_log_probs)

        # Calcular advantages y returns
        advantages = self.calcular_gae_advantages(recompensas, valores)
        returns = self.calcular_returns(recompensas, valores)

        # Normalizar advantages
        advantages = self.normalizar_advantages(advantages)

        # Entrenar por múltiples épocas
        for epoch in range(self.epochs_ppo):
            # Calcular gradientes
            grad_actor_pesos, grad_actor_sesgo, grad_critic_pesos, grad_critic_sesgo = \
                self.calcular_gradientes_ppo(estados, acciones, log_probs_old, advantages, returns)

            # Actualizar pesos
            self.actualizar_pesos(grad_actor_pesos, grad_actor_sesgo, grad_critic_pesos, grad_critic_sesgo)

            self.estadisticas_ppo['epochs_completados'] += 1

        # Limpiar buffer
        self.buffer_estados.clear()
        self.buffer_acciones.clear()
        self.buffer_recompensas.clear()
        self.buffer_valores.clear()
        self.buffer_log_probs.clear()

        return {
            'policy_loss': self.estadisticas_ppo['policy_loss_medio'],
            'value_loss': self.estadisticas_ppo['value_loss_medio'],
            'entropy_loss': self.estadisticas_ppo['entropy_loss_medio'],
            'total_loss': self.estadisticas_ppo['total_loss_medio'],
            'kl_divergencia': self.estadisticas_ppo['kl_divergencia'],
            'clip_ratio': self.clip_ratio,
            'epochs': self.epochs_ppo
        }

    def obtener_estadisticas_ppo(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas específicas de PPO.

        Returns:
            Diccionario con estadísticas de PPO
        """
        if self.pesos_actor is None:
            return {'estado': 'no_inicializada'}

        stats_ppo = {
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'lambda_gae': self.lambda_gae,
            'clip_ratio': self.clip_ratio,
            'clip_ratio_adaptativo': self.clip_ratio_adaptativo,
            'entropy_coef': self.entropy_coef,
            'value_coef': self.value_coef,
            'max_grad_norm': self.max_grad_norm,
            'epochs_ppo': self.epochs_ppo,
            'policy_loss_medio': self.estadisticas_ppo['policy_loss_medio'],
            'value_loss_medio': self.estadisticas_ppo['value_loss_medio'],
            'entropy_loss_medio': self.estadisticas_ppo['entropy_loss_medio'],
            'total_loss_medio': self.estadisticas_ppo['total_loss_medio'],
            'kl_divergencia': self.estadisticas_ppo['kl_divergencia'],
            'policy_ratio_medio': self.estadisticas_ppo['policy_ratio_medio'],
            'advantage_norma': self.estadisticas_ppo['advantage_norma'],
            'gradiente_norma': self.estadisticas_ppo['gradiente_norma'],
            'epochs_completados': self.estadisticas_ppo['epochs_completados']
        }

        return stats_ppo

    def verificar_estabilidad(self) -> Dict[str, bool]:
        """
        Verifica la estabilidad de la neurona PPO.

        Returns:
            Diccionario con indicadores de estabilidad
        """
        estabilidad = {}

        # Verificar KL divergencia
        kl_divergencia = self.estadisticas_ppo['kl_divergencia']
        estabilidad['kl_divergencia_ok'] = 0.01 <= kl_divergencia <= 0.02

        # Verificar clip ratio
        estabilidad['clip_ratio_ok'] = 0.1 <= self.clip_ratio <= 0.3

        # Verificar pérdidas
        policy_loss = self.estadisticas_ppo['policy_loss_medio']
        value_loss = self.estadisticas_ppo['value_loss_medio']
        estabilidad['policy_loss_estable'] = abs(policy_loss) < 10.0
        estabilidad['value_loss_estable'] = value_loss < 10.0

        # Verificar gradientes
        gradiente_norma = self.estadisticas_ppo['gradiente_norma']
        estabilidad['gradientes_no_explosivos'] = gradiente_norma < 10.0
        estabilidad['gradientes_no_desaparecen'] = gradiente_norma > 1e-8

        # Verificar pesos
        if self.pesos_actor is not None and self.pesos_critic is not None:
            peso_actor_max = np.max(np.abs(self.pesos_actor))
            peso_critic_max = np.max(np.abs(self.pesos_critic))

            estabilidad['pesos_actor_no_explosivos'] = peso_actor_max < 5.0
            estabilidad['pesos_critic_no_explosivos'] = peso_critic_max < 5.0
        else:
            estabilidad['pesos_actor_no_explosivos'] = True
            estabilidad['pesos_critic_no_explosivos'] = True

        return estabilidad

    def reinicializar_con_parametros(self, learning_rate: float = None,
                                     gamma: float = None,
                                     clip_ratio: float = None,
                                     entropy_coef: float = None) -> None:
        """
        Reinicializa la neurona con nuevos parámetros.

        Args:
            learning_rate: Nueva tasa de aprendizaje
            gamma: Nuevo factor de descuento
            clip_ratio: Nuevo clip ratio
            entropy_coef: Nuevo coeficiente de entropía
        """
        if learning_rate is not None:
            self.learning_rate = learning_rate
        if gamma is not None:
            self.gamma = gamma
        if clip_ratio is not None:
            self.clip_ratio = clip_ratio
        if entropy_coef is not None:
            self.entropy_coef = entropy_coef

        # Reinicializar pesos
        self.inicializar_pesos()

        # Resetear historial
        self.resetear_historial()

        # Limpiar buffers y estadísticas específicas
        self.buffer_estados.clear()
        self.buffer_acciones.clear()
        self.buffer_recompensas.clear()
        self.buffer_valores.clear()
        self.buffer_log_probs.clear()

        self.historial_policy_loss.clear()
        self.historial_value_loss.clear()
        self.historial_entropy_loss.clear()
        self.historial_clip_ratios.clear()
        self.historial_kl_divergences.clear()
        self.historial_policy_ratios.clear()

        # Resetear contadores
        self.estadisticas_ppo['epochs_completados'] = 0

        logger.info(f"Neurona PPO reinicializada: lr={self.learning_rate}, "
                    f"gamma={self.gamma}, clip_ratio={self.clip_ratio}")

    def __str__(self) -> str:
        return (f"NeuronaRefuerzoPPO(entrada={self.input_size}, "
                f"salida={self.output_size}, lr={self.learning_rate}, "
                f"gamma={self.gamma}, clip_ratio={self.clip_ratio:.3f})")

    def __repr__(self) -> str:
        return self.__str__()

# Funciones de utilidad específicas para RFEN1_RN_6


def crear_neurona_ppo(input_size: int, output_size: int,
                      configuracion: Dict[str, Any] = None) -> NeuronaRefuerzoPPO:
    """
    Función de conveniencia para crear una neurona PPO.

    Args:
        input_size: Número de características de entrada
        output_size: Número de acciones
        configuracion: Configuración adicional

    Returns:
        Instancia de NeuronaRefuerzoPPO
    """
    if configuracion is None:
        configuracion = {}

    return NeuronaRefuerzoPPO(
        input_size=input_size,
        output_size=output_size,
        nombre=configuracion.get('nombre', 'NeuronaRefuerzoPPO'),
        learning_rate=configuracion.get('learning_rate', 0.0003),
        gamma=configuracion.get('gamma', 0.99),
        lambda_gae=configuracion.get('lambda_gae', 0.95),
        clip_ratio=configuracion.get('clip_ratio', 0.2),
        clip_ratio_adaptativo=configuracion.get('clip_ratio_adaptativo', True),
        entropy_coef=configuracion.get('entropy_coef', 0.01),
        value_coef=configuracion.get('value_coef', 0.5),
        max_grad_norm=configuracion.get('max_grad_norm', 0.5),
        usar_ortogonal=configuracion.get('usar_ortogonal', True),
        epochs_ppo=configuracion.get('epochs_ppo', 4)
    )


# Configuración específica para RFEN1_RN_6
RFEN6_CONFIG = {
    'inicializacion_preferida': 'ortogonal',
    'learning_rate_default': 0.0003,
    'gamma_default': 0.99,
    'lambda_gae_default': 0.95,
    'clip_ratio_default': 0.2,
    'clip_ratio_adaptativo_default': True,
    'entropy_coef_default': 0.01,
    'value_coef_default': 0.5,
    'max_grad_norm_default': 0.5,
    'epochs_ppo_default': 4,
    'umbral_kl_divergencia_min': 0.01,
    'umbral_kl_divergencia_max': 0.02
}

logger.info("RFEN1_RN_6.py cargado correctamente - Neurona de Refuerzo PPO Adaptativo")
