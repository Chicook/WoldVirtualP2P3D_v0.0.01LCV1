"""
RFEN1_RN_3.py - Neurona de Refuerzo con Actor-Critic
====================================================

Esta neurona implementa el algoritmo Actor-Critic con optimización de pesos
avanzada usando técnicas de 2025 como normalización espectral y meta-aprendizaje.

Características:
- Actor-Critic con redes separadas optimizadas
- Normalización espectral para estabilidad
- Meta-aprendizaje adaptativo
- Monitoreo de convergencia dual

Autor: LucIA Development Team
Versión: 1.0.0
"""

import numpy as np
import math
from typing import Tuple, Optional, Dict, Any, List
import logging

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

logger = logging.getLogger('RFENRN1.RFEN1_RN_3')


class NeuronaRefuerzoActorCritic(NeuronaRefuerzoBase):
    """
    Neurona de refuerzo con algoritmo Actor-Critic avanzado.

    Esta neurona implementa Actor-Critic con técnicas avanzadas de optimización
    de pesos incluyendo normalización espectral y meta-aprendizaje.
    """

    def __init__(self, input_size: int, output_size: int,
                 nombre: str = "NeuronaRefuerzoActorCritic",
                 learning_rate_actor: float = 0.001,
                 learning_rate_critic: float = 0.002,
                 gamma: float = 0.99,
                 lambda_gae: float = 0.95,
                 usar_espectral: bool = True,
                 meta_learning: bool = True):
        """
        Inicializa la neurona de refuerzo Actor-Critic.

        Args:
            input_size: Número de características de entrada
            output_size: Número de acciones posibles
            nombre: Nombre identificativo de la neurona
            learning_rate_actor: Tasa de aprendizaje del actor
            learning_rate_critic: Tasa de aprendizaje del crítico
            gamma: Factor de descuento
            lambda_gae: Parámetro lambda para GAE
            usar_espectral: Si usar normalización espectral
            meta_learning: Si usar meta-aprendizaje adaptativo
        """
        super().__init__(input_size, output_size, nombre)

        self.learning_rate_actor = learning_rate_actor
        self.learning_rate_critic = learning_rate_critic
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.usar_espectral = usar_espectral
        self.meta_learning = meta_learning

        # Pesos separados para Actor y Critic
        self.pesos_actor = None
        self.sesgo_actor = None
        self.pesos_critic = None
        self.sesgo_critic = None

        # Parámetros de normalización espectral
        self.espectral_actor = None
        self.espectral_critic = None

        # Estadísticas específicas de Actor-Critic
        self.estadisticas_ac = {
            'convergencia_actor': 0.0,
            'convergencia_critic': 0.0,
            'estabilidad_actor': 0.0,
            'estabilidad_critic': 0.0,
            'advantage_media': 0.0,
            'advantage_std': 0.0,
            'valor_error_medio': 0.0,
            'policy_loss_medio': 0.0,
            'meta_adaptaciones': 0,
            'espectral_updates': 0
        }

        # Historial para análisis
        self.historial_actor_loss = []
        self.historial_critic_loss = []
        self.historial_advantages = []
        self.historial_valores = []
        self.historial_politicas = []

        logger.info(f"NeuronaRefuerzoActorCritic creada: {self}")

    def inicializar_pesos(self) -> None:
        """
        Inicializa los pesos del Actor y Critic con técnicas avanzadas.
        """
        if self.usar_espectral:
            # Inicialización espectral para estabilidad
            self.pesos_actor = inicializar_pesos_espectral((self.input_size, self.output_size))
            self.pesos_critic = inicializar_pesos_espectral((self.input_size, 1))

            # Inicializar parámetros de normalización espectral
            self.espectral_actor = np.random.randn(self.output_size)
            self.espectral_critic = np.random.randn(1)
        else:
            # Inicialización ortogonal como alternativa
            self.pesos_actor = inicializar_pesos_ortogonal((self.input_size, self.output_size))
            self.pesos_critic = inicializar_pesos_ortogonal((self.input_size, 1))

        # Inicializar sesgos con ceros
        self.sesgo_actor = np.zeros((1, self.output_size), dtype=LUCIA_RL_CONFIG['precision'])
        self.sesgo_critic = np.zeros((1, 1), dtype=LUCIA_RL_CONFIG['precision'])

        logger.info(f"Pesos Actor-Critic inicializados con {'espectral' if self.usar_espectral else 'ortogonal'}")

    def _aplicar_normalizacion_espectral(self, pesos: np.ndarray, espectral_params: np.ndarray) -> np.ndarray:
        """
        Aplica normalización espectral a los pesos.

        Args:
            pesos: Pesos a normalizar
            espectral_params: Parámetros espectrales

        Returns:
            Pesos normalizados espectralmente
        """
        if not self.usar_espectral:
            return pesos

        # Calcular valores singulares
        u, s, v = np.linalg.svd(pesos, full_matrices=False)

        # Normalizar valores singulares
        s_normalized = s / (np.max(s) + 1e-8)

        # Aplicar parámetros espectrales
        s_espectral = s_normalized * espectral_params[:len(s)]

        # Reconstruir pesos
        pesos_normalizados = u @ np.diag(s_espectral) @ v

        return pesos_normalizados

    def forward_actor(self, estado: np.ndarray) -> np.ndarray:
        """
        Propagación hacia adelante del Actor.

        Args:
            estado: Estado actual del entorno

        Returns:
            Probabilidades de acción (softmax)
        """
        if self.pesos_actor is None:
            self.inicializar_pesos()

        # Aplicar normalización espectral si está habilitada
        pesos_actor = self._aplicar_normalizacion_espectral(self.pesos_actor, self.espectral_actor)

        # Calcular logits del actor
        logits = np.dot(estado, pesos_actor) + self.sesgo_actor

        # Aplicar softmax para obtener probabilidades
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probabilidades = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # Guardar historial
        self.historial_politicas.append(probabilidades.copy())

        return probabilidades

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

        # Aplicar normalización espectral si está habilitada
        pesos_critic = self._aplicar_normalizacion_espectral(self.pesos_critic, self.espectral_critic)

        # Calcular valor del estado
        valor = np.dot(estado, pesos_critic) + self.sesgo_critic

        # Guardar historial
        self.historial_valores.append(valor.copy())

        return valor

    def forward(self, estado: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Propagación hacia adelante completa (Actor + Critic).

        Args:
            estado: Estado actual del entorno

        Returns:
            Tupla con (probabilidades_accion, valor_estado)
        """
        probabilidades = self.forward_actor(estado)
        valor = self.forward_critic(estado)

        return probabilidades, valor

    def seleccionar_accion(self, estado: np.ndarray) -> Tuple[int, float, float]:
        """
        Selecciona una acción usando el Actor-Critic.

        Args:
            estado: Estado actual del entorno

        Returns:
            Tupla con (acción, probabilidad_accion, valor_estado)
        """
        probabilidades, valor = self.forward(estado)

        # Muestrear acción según las probabilidades
        accion = np.random.choice(self.output_size, p=probabilidades[0])
        probabilidad_accion = probabilidades[0, accion]
        valor_estado = valor[0, 0]

        return accion, probabilidad_accion, valor_estado

    def calcular_gae_advantages(self, recompensas: List[float],
                                valores: List[float]) -> List[float]:
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

        # Guardar historial
        self.historial_advantages.extend(advantages)

        return advantages

    def calcular_gradientes_ac(self, estados: List[np.ndarray],
                               acciones: List[int],
                               advantages: List[float],
                               valores_objetivo: List[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calcula los gradientes del Actor y Critic.

        Args:
            estados: Lista de estados
            acciones: Lista de acciones tomadas
            advantages: Lista de ventajas calculadas
            valores_objetivo: Lista de valores objetivo

        Returns:
            Tupla con (grad_actor_pesos, grad_actor_sesgo, grad_critic_pesos, grad_critic_sesgo)
        """
        if self.pesos_actor is None or self.pesos_critic is None:
            raise ValueError("Pesos no inicializados")

        # Gradientes del Actor
        grad_actor_pesos = np.zeros_like(self.pesos_actor)
        grad_actor_sesgo = np.zeros_like(self.sesgo_actor)

        # Gradientes del Critic
        grad_critic_pesos = np.zeros_like(self.pesos_critic)
        grad_critic_sesgo = np.zeros_like(self.sesgo_critic)

        actor_loss_total = 0
        critic_loss_total = 0

        for estado, accion, advantage, valor_objetivo in zip(estados, acciones, advantages, valores_objetivo):
            # Calcular probabilidades y valor actual
            probabilidades = self.forward_actor(estado)
            valor_actual = self.forward_critic(estado)

            # Gradiente del Actor (Policy Gradient)
            grad_log_prob = np.zeros(self.output_size)
            grad_log_prob[accion] = 1.0 / (probabilidades[0, accion] + 1e-8)

            grad_actor_pesos += np.outer(estado[0], grad_log_prob) * advantage
            grad_actor_sesgo += grad_log_prob * advantage

            actor_loss_total += -np.log(probabilidades[0, accion] + 1e-8) * advantage

            # Gradiente del Critic (Value Function)
            error_valor = valor_objetivo - valor_actual[0, 0]
            grad_critic_pesos += np.outer(estado[0], np.array([error_valor]))
            grad_critic_sesgo += np.array([[error_valor]])

            critic_loss_total += 0.5 * (error_valor ** 2)

        # Normalizar por el número de muestras
        n_muestras = len(estados)
        grad_actor_pesos /= n_muestras
        grad_actor_sesgo /= n_muestras
        grad_critic_pesos /= n_muestras
        grad_critic_sesgo /= n_muestras

        # Guardar pérdidas
        self.historial_actor_loss.append(actor_loss_total / n_muestras)
        self.historial_critic_loss.append(critic_loss_total / n_muestras)

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
        self.pesos_actor += self.learning_rate_actor * grad_actor_pesos
        self.sesgo_actor += self.learning_rate_actor * grad_actor_sesgo

        # Actualizar pesos del Critic
        self.pesos_critic += self.learning_rate_critic * grad_critic_pesos
        self.sesgo_critic += self.learning_rate_critic * grad_critic_sesgo

        # Meta-aprendizaje adaptativo
        if self.meta_learning:
            self._adaptar_meta_learning()

        # Actualizar normalización espectral
        if self.usar_espectral:
            self._actualizar_espectral()

        # Calcular convergencia
        self._calcular_convergencia_ac()

    def _adaptar_meta_learning(self) -> None:
        """
        Adapta los parámetros usando meta-aprendizaje.
        """
        if len(self.historial_actor_loss) < 10 or len(self.historial_critic_loss) < 10:
            return

        # Calcular tendencias de pérdida
        actor_trend = np.mean(np.diff(self.historial_actor_loss[-10:]))
        critic_trend = np.mean(np.diff(self.historial_critic_loss[-10:]))

        # Adaptar tasas de aprendizaje
        if actor_trend > 0:  # Pérdida aumentando
            self.learning_rate_actor *= 0.95
        elif actor_trend < -0.01:  # Pérdida disminuyendo rápidamente
            self.learning_rate_actor *= 1.05

        if critic_trend > 0:  # Pérdida aumentando
            self.learning_rate_critic *= 0.95
        elif critic_trend < -0.01:  # Pérdida disminuyendo rápidamente
            self.learning_rate_critic *= 1.05

        # Limitar tasas de aprendizaje
        self.learning_rate_actor = np.clip(self.learning_rate_actor, 1e-6, 1e-2)
        self.learning_rate_critic = np.clip(self.learning_rate_critic, 1e-6, 1e-2)

        self.estadisticas_ac['meta_adaptaciones'] += 1

    def _actualizar_espectral(self) -> None:
        """
        Actualiza los parámetros de normalización espectral.
        """
        if not self.usar_espectral:
            return

        # Actualizar parámetros espectrales del Actor
        if len(self.historial_actor_loss) > 5:
            actor_loss_reciente = np.mean(self.historial_actor_loss[-5:])
            if actor_loss_reciente > np.mean(self.historial_actor_loss[-10:-5]):
                # Pérdida aumentando, reducir normalización
                self.espectral_actor *= 0.99
            else:
                # Pérdida disminuyendo, aumentar normalización
                self.espectral_actor *= 1.01

        # Actualizar parámetros espectrales del Critic
        if len(self.historial_critic_loss) > 5:
            critic_loss_reciente = np.mean(self.historial_critic_loss[-5:])
            if critic_loss_reciente > np.mean(self.historial_critic_loss[-10:-5]):
                # Pérdida aumentando, reducir normalización
                self.espectral_critic *= 0.99
            else:
                # Pérdida disminuyendo, aumentar normalización
                self.espectral_critic *= 1.01

        # Limitar parámetros espectrales
        self.espectral_actor = np.clip(self.espectral_actor, 0.1, 2.0)
        self.espectral_critic = np.clip(self.espectral_critic, 0.1, 2.0)

        self.estadisticas_ac['espectral_updates'] += 1

    def _calcular_convergencia_ac(self) -> None:
        """
        Calcula la convergencia del Actor y Critic.
        """
        if len(self.historial_actor_loss) < 10 or len(self.historial_critic_loss) < 10:
            return

        # Convergencia del Actor
        actor_losses = np.array(self.historial_actor_loss[-10:])
        actor_varianza = np.var(actor_losses)
        self.estadisticas_ac['convergencia_actor'] = 1.0 / (1.0 + actor_varianza)

        # Convergencia del Critic
        critic_losses = np.array(self.historial_critic_loss[-10:])
        critic_varianza = np.var(critic_losses)
        self.estadisticas_ac['convergencia_critic'] = 1.0 / (1.0 + critic_varianza)

    def calcular_estabilidad_ac(self) -> Tuple[float, float]:
        """
        Calcula la estabilidad del Actor y Critic.

        Returns:
            Tupla con (estabilidad_actor, estabilidad_critic)
        """
        estabilidad_actor = 0.0
        estabilidad_critic = 0.0

        if len(self.historial_politicas) >= 20:
            # Estabilidad del Actor basada en políticas
            politicas_array = np.array(self.historial_politicas[-20:])
            varianza_politicas = np.var(politicas_array, axis=0)
            estabilidad_actor = 1.0 / (1.0 + np.mean(varianza_politicas))

        if len(self.historial_valores) >= 20:
            # Estabilidad del Critic basada en valores
            valores_array = np.array(self.historial_valores[-20:])
            varianza_valores = np.var(valores_array, axis=0)
            estabilidad_critic = 1.0 / (1.0 + np.mean(varianza_valores))

        self.estadisticas_ac['estabilidad_actor'] = estabilidad_actor
        self.estadisticas_ac['estabilidad_critic'] = estabilidad_critic

        return estabilidad_actor, estabilidad_critic

    def obtener_estadisticas_ac(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas específicas de Actor-Critic.

        Returns:
            Diccionario con estadísticas de Actor-Critic
        """
        if self.pesos_actor is None or self.pesos_critic is None:
            return {'estado': 'no_inicializada'}

        estabilidad_actor, estabilidad_critic = self.calcular_estabilidad_ac()

        stats_ac = {
            'learning_rate_actor': self.learning_rate_actor,
            'learning_rate_critic': self.learning_rate_critic,
            'gamma': self.gamma,
            'lambda_gae': self.lambda_gae,
            'convergencia_actor': self.estadisticas_ac['convergencia_actor'],
            'convergencia_critic': self.estadisticas_ac['convergencia_critic'],
            'estabilidad_actor': estabilidad_actor,
            'estabilidad_critic': estabilidad_critic,
            'meta_adaptaciones': self.estadisticas_ac['meta_adaptaciones'],
            'espectral_updates': self.estadisticas_ac['espectral_updates'],
            'usar_espectral': self.usar_espectral,
            'meta_learning': self.meta_learning
        }

        # Estadísticas de ventajas
        if self.historial_advantages:
            advantages_array = np.array(self.historial_advantages[-100:])
            stats_ac.update({
                'advantage_media': np.mean(advantages_array),
                'advantage_std': np.std(advantages_array),
                'advantage_min': np.min(advantages_array),
                'advantage_max': np.max(advantages_array)
            })

        # Estadísticas de pérdidas
        if self.historial_actor_loss:
            stats_ac['actor_loss_medio'] = np.mean(self.historial_actor_loss[-10:])
        if self.historial_critic_loss:
            stats_ac['critic_loss_medio'] = np.mean(self.historial_critic_loss[-10:])

        return stats_ac

    def verificar_estabilidad(self) -> Dict[str, bool]:
        """
        Verifica la estabilidad de la neurona Actor-Critic.

        Returns:
            Diccionario con indicadores de estabilidad
        """
        estabilidad = {}

        # Verificar convergencia
        convergencia_actor = self.estadisticas_ac['convergencia_actor']
        convergencia_critic = self.estadisticas_ac['convergencia_critic']
        estabilidad['actor_convergencia_ok'] = convergencia_actor > 0.7
        estabilidad['critic_convergencia_ok'] = convergencia_critic > 0.7

        # Verificar estabilidad
        estabilidad_actor, estabilidad_critic = self.calcular_estabilidad_ac()
        estabilidad['actor_estable'] = estabilidad_actor > 0.7
        estabilidad['critic_estable'] = estabilidad_critic > 0.7

        # Verificar tasas de aprendizaje
        estabilidad['lr_actor_ok'] = 1e-6 <= self.learning_rate_actor <= 1e-2
        estabilidad['lr_critic_ok'] = 1e-6 <= self.learning_rate_critic <= 1e-2

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

    def reinicializar_con_parametros(self, learning_rate_actor: float = None,
                                     learning_rate_critic: float = None,
                                     gamma: float = None,
                                     lambda_gae: float = None) -> None:
        """
        Reinicializa la neurona con nuevos parámetros.

        Args:
            learning_rate_actor: Nueva tasa de aprendizaje del actor
            learning_rate_critic: Nueva tasa de aprendizaje del crítico
            gamma: Nuevo factor de descuento
            lambda_gae: Nuevo parámetro lambda para GAE
        """
        if learning_rate_actor is not None:
            self.learning_rate_actor = learning_rate_actor
        if learning_rate_critic is not None:
            self.learning_rate_critic = learning_rate_critic
        if gamma is not None:
            self.gamma = gamma
        if lambda_gae is not None:
            self.lambda_gae = lambda_gae

        # Reinicializar pesos
        self.inicializar_pesos()

        # Resetear historial
        self.resetear_historial()

        # Limpiar estadísticas específicas
        self.historial_actor_loss.clear()
        self.historial_critic_loss.clear()
        self.historial_advantages.clear()
        self.historial_valores.clear()
        self.historial_politicas.clear()

        # Resetear contadores
        self.estadisticas_ac['meta_adaptaciones'] = 0
        self.estadisticas_ac['espectral_updates'] = 0

        logger.info(f"Neurona Actor-Critic reinicializada: lr_actor={self.learning_rate_actor}, "
                    f"lr_critic={self.learning_rate_critic}")

    def __str__(self) -> str:
        return (f"NeuronaRefuerzoActorCritic(entrada={self.input_size}, "
                f"salida={self.output_size}, lr_actor={self.learning_rate_actor}, "
                f"lr_critic={self.learning_rate_critic}, espectral={self.usar_espectral})")

    def __repr__(self) -> str:
        return self.__str__()

# Funciones de utilidad específicas para RFEN1_RN_3


def crear_neurona_actor_critic(input_size: int, output_size: int,
                               configuracion: Dict[str, Any] = None) -> NeuronaRefuerzoActorCritic:
    """
    Función de conveniencia para crear una neurona Actor-Critic.

    Args:
        input_size: Número de características de entrada
        output_size: Número de acciones
        configuracion: Configuración adicional

    Returns:
        Instancia de NeuronaRefuerzoActorCritic
    """
    if configuracion is None:
        configuracion = {}

    return NeuronaRefuerzoActorCritic(
        input_size=input_size,
        output_size=output_size,
        nombre=configuracion.get('nombre', 'NeuronaRefuerzoActorCritic'),
        learning_rate_actor=configuracion.get('learning_rate_actor', 0.001),
        learning_rate_critic=configuracion.get('learning_rate_critic', 0.002),
        gamma=configuracion.get('gamma', 0.99),
        lambda_gae=configuracion.get('lambda_gae', 0.95),
        usar_espectral=configuracion.get('usar_espectral', True),
        meta_learning=configuracion.get('meta_learning', True)
    )


# Configuración específica para RFEN1_RN_3
RFEN3_CONFIG = {
    'inicializacion_preferida': 'espectral',
    'learning_rate_actor_default': 0.001,
    'learning_rate_critic_default': 0.002,
    'gamma_default': 0.99,
    'lambda_gae_default': 0.95,
    'meta_learning_default': True,
    'espectral_default': True,
    'umbral_convergencia': 0.7,
    'umbral_estabilidad': 0.7
}

logger.info("RFEN1_RN_3.py cargado correctamente - Neurona de Refuerzo Actor-Critic Avanzado")
