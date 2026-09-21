"""
RFEN1_RN_5.py - Neurona de Refuerzo con Asynchronous Advantage Actor-Critic (A3C)
==================================================================================

Esta neurona implementa el algoritmo A3C con optimización de pesos distribuidos
y sincronización asíncrona para entrenamiento paralelo eficiente.

Características:
- A3C con múltiples workers asíncronos
- Inicialización de pesos distribuida y sincronizada
- Monitoreo de convergencia multi-worker
- Adaptación automática de hiperparámetros distribuidos

Autor: LucIA Development Team
Versión: 1.0.0
"""

import numpy as np
import math
from typing import Tuple, Optional, Dict, Any, List
import logging
import threading
import time
from collections import deque
import random

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

logger = logging.getLogger('RFENRN1.RFEN1_RN_5')


class NeuronaRefuerzoA3C(NeuronaRefuerzoBase):
    """
    Neurona de refuerzo con algoritmo Asynchronous Advantage Actor-Critic (A3C).

    Esta neurona implementa A3C con múltiples workers asíncronos,
    optimización de pesos distribuidos y sincronización eficiente.
    """

    def __init__(self, input_size: int, output_size: int,
                 nombre: str = "NeuronaRefuerzoA3C",
                 learning_rate_actor: float = 0.001,
                 learning_rate_critic: float = 0.002,
                 gamma: float = 0.99,
                 lambda_gae: float = 0.95,
                 num_workers: int = 4,
                 update_frequency: int = 20,
                 usar_ortogonal: bool = True,
                 entropy_coef: float = 0.01):
        """
        Inicializa la neurona de refuerzo A3C.

        Args:
            input_size: Número de características de entrada
            output_size: Número de acciones posibles
            nombre: Nombre identificativo de la neurona
            learning_rate_actor: Tasa de aprendizaje del actor
            learning_rate_critic: Tasa de aprendizaje del crítico
            gamma: Factor de descuento
            lambda_gae: Parámetro lambda para GAE
            num_workers: Número de workers asíncronos
            update_frequency: Frecuencia de actualización global
            usar_ortogonal: Si usar inicialización ortogonal
            entropy_coef: Coeficiente de entropía para regularización
        """
        super().__init__(input_size, output_size, nombre)

        self.learning_rate_actor = learning_rate_actor
        self.learning_rate_critic = learning_rate_critic
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.num_workers = num_workers
        self.update_frequency = update_frequency
        self.usar_ortogonal = usar_ortogonal
        self.entropy_coef = entropy_coef

        # Pesos globales (maestro) y locales (workers)
        self.pesos_actor_global = None
        self.sesgo_actor_global = None
        self.pesos_critic_global = None
        self.sesgo_critic_global = None

        # Pesos locales para cada worker
        self.pesos_actor_locales = {}
        self.sesgo_actor_locales = {}
        self.pesos_critic_locales = {}
        self.sesgo_critic_locales = {}

        # Lock para sincronización
        self.pesos_lock = threading.Lock()

        # Estadísticas específicas de A3C
        self.estadisticas_a3c = {
            'convergencia_actor_global': 0.0,
            'convergencia_critic_global': 0.0,
            'estabilidad_actor_global': 0.0,
            'estabilidad_critic_global': 0.0,
            'sincronizaciones': 0,
            'updates_globales': 0,
            'advantage_media': 0.0,
            'advantage_std': 0.0,
            'entropia_media': 0.0,
            'workers_activos': 0
        }

        # Historial para análisis
        self.historial_actor_loss_global = []
        self.historial_critic_loss_global = []
        self.historial_advantages_global = []
        self.historial_entropias_global = []
        self.historial_sincronizaciones = []

        logger.info(f"NeuronaRefuerzoA3C creada: {self} con {num_workers} workers")

    def inicializar_pesos(self) -> None:
        """
        Inicializa los pesos globales y locales para todos los workers.
        """
        if self.usar_ortogonal:
            # Inicialización ortogonal para estabilidad distribuida
            self.pesos_actor_global = inicializar_pesos_ortogonal((self.input_size, self.output_size))
            self.pesos_critic_global = inicializar_pesos_ortogonal((self.input_size, 1))
        else:
            # Inicialización Xavier como alternativa
            self.pesos_actor_global = inicializar_pesos_xavier(
                (self.input_size, self.output_size),
                fan_in=self.input_size,
                fan_out=self.output_size
            )
            self.pesos_critic_global = inicializar_pesos_xavier(
                (self.input_size, 1),
                fan_in=self.input_size,
                fan_out=1
            )

        # Inicializar sesgos globales con ceros
        self.sesgo_actor_global = np.zeros((1, self.output_size), dtype=LUCIA_RL_CONFIG['precision'])
        self.sesgo_critic_global = np.zeros((1, 1), dtype=LUCIA_RL_CONFIG['precision'])

        # Inicializar pesos locales para cada worker
        for worker_id in range(self.num_workers):
            self.pesos_actor_locales[worker_id] = self.pesos_actor_global.copy()
            self.sesgo_actor_locales[worker_id] = self.sesgo_actor_global.copy()
            self.pesos_critic_locales[worker_id] = self.pesos_critic_global.copy()
            self.sesgo_critic_locales[worker_id] = self.sesgo_critic_global.copy()

        logger.info(f"Pesos A3C inicializados con {'ortogonal' if self.usar_ortogonal else 'Xavier'} "
                    f"para {self.num_workers} workers")

    def forward_actor(self, estado: np.ndarray, worker_id: int = 0) -> np.ndarray:
        """
        Propagación hacia adelante del Actor para un worker específico.

        Args:
            estado: Estado actual del entorno
            worker_id: ID del worker

        Returns:
            Probabilidades de acción (softmax)
        """
        if self.pesos_actor_global is None:
            self.inicializar_pesos()

        # Usar pesos locales del worker
        pesos_actor = self.pesos_actor_locales.get(worker_id, self.pesos_actor_global)
        sesgo_actor = self.sesgo_actor_locales.get(worker_id, self.sesgo_actor_global)

        # Calcular logits del actor
        logits = np.dot(estado, pesos_actor) + sesgo_actor

        # Aplicar softmax para obtener probabilidades
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probabilidades = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        return probabilidades

    def forward_critic(self, estado: np.ndarray, worker_id: int = 0) -> np.ndarray:
        """
        Propagación hacia adelante del Critic para un worker específico.

        Args:
            estado: Estado actual del entorno
            worker_id: ID del worker

        Returns:
            Valor estimado del estado
        """
        if self.pesos_critic_global is None:
            self.inicializar_pesos()

        # Usar pesos locales del worker
        pesos_critic = self.pesos_critic_locales.get(worker_id, self.pesos_critic_global)
        sesgo_critic = self.sesgo_critic_locales.get(worker_id, self.sesgo_critic_global)

        # Calcular valor del estado
        valor = np.dot(estado, pesos_critic) + sesgo_critic

        return valor

    def forward(self, estado: np.ndarray, worker_id: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Propagación hacia adelante completa (Actor + Critic) para un worker.

        Args:
            estado: Estado actual del entorno
            worker_id: ID del worker

        Returns:
            Tupla con (probabilidades_accion, valor_estado)
        """
        probabilidades = self.forward_actor(estado, worker_id)
        valor = self.forward_critic(estado, worker_id)

        return probabilidades, valor

    def seleccionar_accion(self, estado: np.ndarray, worker_id: int = 0) -> Tuple[int, float, float]:
        """
        Selecciona una acción usando el Actor-Critic de un worker.

        Args:
            estado: Estado actual del entorno
            worker_id: ID del worker

        Returns:
            Tupla con (acción, probabilidad_accion, valor_estado)
        """
        probabilidades, valor = self.forward(estado, worker_id)

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

        return advantages

    def calcular_gradientes_worker(self, worker_id: int, estados: List[np.ndarray],
                                   acciones: List[int], advantages: List[float],
                                   valores_objetivo: List[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calcula los gradientes para un worker específico.

        Args:
            worker_id: ID del worker
            estados: Lista de estados
            acciones: Lista de acciones tomadas
            advantages: Lista de ventajas calculadas
            valores_objetivo: Lista de valores objetivo

        Returns:
            Tupla con (grad_actor_pesos, grad_actor_sesgo, grad_critic_pesos, grad_critic_sesgo)
        """
        # Gradientes del Actor
        grad_actor_pesos = np.zeros_like(self.pesos_actor_global)
        grad_actor_sesgo = np.zeros_like(self.sesgo_actor_global)

        # Gradientes del Critic
        grad_critic_pesos = np.zeros_like(self.pesos_critic_global)
        grad_critic_sesgo = np.zeros_like(self.sesgo_critic_global)

        actor_loss_total = 0
        critic_loss_total = 0

        for estado, accion, advantage, valor_objetivo in zip(estados, acciones, advantages, valores_objetivo):
            # Calcular probabilidades y valor actual
            probabilidades = self.forward_actor(estado, worker_id)
            valor_actual = self.forward_critic(estado, worker_id)

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

        return grad_actor_pesos, grad_actor_sesgo, grad_critic_pesos, grad_critic_sesgo

    def actualizar_pesos_worker(self, worker_id: int, grad_actor_pesos: np.ndarray,
                                grad_actor_sesgo: np.ndarray, grad_critic_pesos: np.ndarray,
                                grad_critic_sesgo: np.ndarray) -> None:
        """
        Actualiza los pesos locales de un worker específico.

        Args:
            worker_id: ID del worker
            grad_actor_pesos: Gradiente de los pesos del actor
            grad_actor_sesgo: Gradiente del sesgo del actor
            grad_critic_pesos: Gradiente de los pesos del crítico
            grad_critic_sesgo: Gradiente del sesgo del crítico
        """
        # Actualizar pesos locales del worker
        self.pesos_actor_locales[worker_id] += self.learning_rate_actor * grad_actor_pesos
        self.sesgo_actor_locales[worker_id] += self.learning_rate_actor * grad_actor_sesgo
        self.pesos_critic_locales[worker_id] += self.learning_rate_critic * grad_critic_pesos
        self.sesgo_critic_locales[worker_id] += self.learning_rate_critic * grad_critic_sesgo

    def sincronizar_pesos_globales(self) -> None:
        """
        Sincroniza los pesos globales con los pesos locales de todos los workers.
        """
        with self.pesos_lock:
            # Promediar pesos de todos los workers
            pesos_actor_promedio = np.zeros_like(self.pesos_actor_global)
            sesgo_actor_promedio = np.zeros_like(self.sesgo_actor_global)
            pesos_critic_promedio = np.zeros_like(self.pesos_critic_global)
            sesgo_critic_promedio = np.zeros_like(self.sesgo_critic_global)

            for worker_id in range(self.num_workers):
                pesos_actor_promedio += self.pesos_actor_locales[worker_id]
                sesgo_actor_promedio += self.sesgo_actor_locales[worker_id]
                pesos_critic_promedio += self.pesos_critic_locales[worker_id]
                sesgo_critic_promedio += self.sesgo_critic_locales[worker_id]

            # Promediar
            pesos_actor_promedio /= self.num_workers
            sesgo_actor_promedio /= self.num_workers
            pesos_critic_promedio /= self.num_workers
            sesgo_critic_promedio /= self.num_workers

            # Actualizar pesos globales
            self.pesos_actor_global = pesos_actor_promedio
            self.sesgo_actor_global = sesgo_actor_promedio
            self.pesos_critic_global = pesos_critic_promedio
            self.sesgo_critic_global = sesgo_critic_promedio

            # Actualizar pesos locales con los globales
            for worker_id in range(self.num_workers):
                self.pesos_actor_locales[worker_id] = self.pesos_actor_global.copy()
                self.sesgo_actor_locales[worker_id] = self.sesgo_actor_global.copy()
                self.pesos_critic_locales[worker_id] = self.pesos_critic_global.copy()
                self.sesgo_critic_locales[worker_id] = self.sesgo_critic_global.copy()

            self.estadisticas_a3c['sincronizaciones'] += 1
            self.historial_sincronizaciones.append(time.time())

    def calcular_convergencia_global(self) -> None:
        """
        Calcula la convergencia global basada en la estabilidad de los pesos.
        """
        if len(self.historial_sincronizaciones) < 5:
            return

        # Calcular varianza de los pesos globales en el tiempo
        if hasattr(self, 'historial_pesos_actor_global'):
            pesos_array = np.array(self.historial_pesos_actor_global[-5:])
            varianza_pesos = np.var(pesos_array, axis=0)
            varianza_promedio = np.mean(varianza_pesos)

            # Convergencia inversamente proporcional a la varianza
            convergencia = 1.0 / (1.0 + varianza_promedio)
            self.estadisticas_a3c['convergencia_actor_global'] = convergencia
            self.estadisticas_a3c['convergencia_critic_global'] = convergencia

    def calcular_estabilidad_global(self) -> Tuple[float, float]:
        """
        Calcula la estabilidad global del Actor y Critic.

        Returns:
            Tupla con (estabilidad_actor, estabilidad_critic)
        """
        estabilidad_actor = 0.0
        estabilidad_critic = 0.0

        # Calcular estabilidad basada en la consistencia entre workers
        if len(self.pesos_actor_locales) > 1:
            pesos_actor_array = np.array(list(self.pesos_actor_locales.values()))
            varianza_workers = np.var(pesos_actor_array, axis=0)
            estabilidad_actor = 1.0 / (1.0 + np.mean(varianza_workers))

        if len(self.pesos_critic_locales) > 1:
            pesos_critic_array = np.array(list(self.pesos_critic_locales.values()))
            varianza_workers = np.var(pesos_critic_array, axis=0)
            estabilidad_critic = 1.0 / (1.0 + np.mean(varianza_workers))

        self.estadisticas_a3c['estabilidad_actor_global'] = estabilidad_actor
        self.estadisticas_a3c['estabilidad_critic_global'] = estabilidad_critic

        return estabilidad_actor, estabilidad_critic

    def obtener_estadisticas_a3c(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas específicas de A3C.

        Returns:
            Diccionario con estadísticas de A3C
        """
        if self.pesos_actor_global is None:
            return {'estado': 'no_inicializada'}

        estabilidad_actor, estabilidad_critic = self.calcular_estabilidad_global()

        stats_a3c = {
            'learning_rate_actor': self.learning_rate_actor,
            'learning_rate_critic': self.learning_rate_critic,
            'gamma': self.gamma,
            'lambda_gae': self.lambda_gae,
            'num_workers': self.num_workers,
            'update_frequency': self.update_frequency,
            'convergencia_actor_global': self.estadisticas_a3c['convergencia_actor_global'],
            'convergencia_critic_global': self.estadisticas_a3c['convergencia_critic_global'],
            'estabilidad_actor_global': estabilidad_actor,
            'estabilidad_critic_global': estabilidad_critic,
            'sincronizaciones': self.estadisticas_a3c['sincronizaciones'],
            'updates_globales': self.estadisticas_a3c['updates_globales'],
            'entropy_coef': self.entropy_coef
        }

        return stats_a3c

    def verificar_estabilidad(self) -> Dict[str, bool]:
        """
        Verifica la estabilidad de la neurona A3C.

        Returns:
            Diccionario con indicadores de estabilidad
        """
        estabilidad = {}

        # Verificar convergencia global
        convergencia_actor = self.estadisticas_a3c['convergencia_actor_global']
        convergencia_critic = self.estadisticas_a3c['convergencia_critic_global']
        estabilidad['actor_convergencia_ok'] = convergencia_actor > 0.7
        estabilidad['critic_convergencia_ok'] = convergencia_critic > 0.7

        # Verificar estabilidad global
        estabilidad_actor, estabilidad_critic = self.calcular_estabilidad_global()
        estabilidad['actor_estable'] = estabilidad_actor > 0.7
        estabilidad['critic_estable'] = estabilidad_critic > 0.7

        # Verificar sincronización
        estabilidad['sincronizacion_ok'] = self.estadisticas_a3c['sincronizaciones'] > 0

        # Verificar pesos globales
        if self.pesos_actor_global is not None and self.pesos_critic_global is not None:
            peso_actor_max = np.max(np.abs(self.pesos_actor_global))
            peso_critic_max = np.max(np.abs(self.pesos_critic_global))

            estabilidad['pesos_actor_no_explosivos'] = peso_actor_max < 5.0
            estabilidad['pesos_critic_no_explosivos'] = peso_critic_max < 5.0
        else:
            estabilidad['pesos_actor_no_explosivos'] = True
            estabilidad['pesos_critic_no_explosivos'] = True

        return estabilidad

    def reinicializar_con_parametros(self, learning_rate_actor: float = None,
                                     learning_rate_critic: float = None,
                                     gamma: float = None,
                                     lambda_gae: float = None,
                                     num_workers: int = None) -> None:
        """
        Reinicializa la neurona con nuevos parámetros.

        Args:
            learning_rate_actor: Nueva tasa de aprendizaje del actor
            learning_rate_critic: Nueva tasa de aprendizaje del crítico
            gamma: Nuevo factor de descuento
            lambda_gae: Nuevo parámetro lambda para GAE
            num_workers: Nuevo número de workers
        """
        if learning_rate_actor is not None:
            self.learning_rate_actor = learning_rate_actor
        if learning_rate_critic is not None:
            self.learning_rate_critic = learning_rate_critic
        if gamma is not None:
            self.gamma = gamma
        if lambda_gae is not None:
            self.lambda_gae = lambda_gae
        if num_workers is not None:
            self.num_workers = num_workers

        # Reinicializar pesos
        self.inicializar_pesos()

        # Resetear historial
        self.resetear_historial()

        # Limpiar estadísticas específicas
        self.historial_actor_loss_global.clear()
        self.historial_critic_loss_global.clear()
        self.historial_advantages_global.clear()
        self.historial_entropias_global.clear()
        self.historial_sincronizaciones.clear()

        # Resetear contadores
        self.estadisticas_a3c['sincronizaciones'] = 0
        self.estadisticas_a3c['updates_globales'] = 0

        logger.info(f"Neurona A3C reinicializada: lr_actor={self.learning_rate_actor}, "
                    f"lr_critic={self.learning_rate_critic}, workers={self.num_workers}")

    def __str__(self) -> str:
        return (f"NeuronaRefuerzoA3C(entrada={self.input_size}, "
                f"salida={self.output_size}, lr_actor={self.learning_rate_actor}, "
                f"lr_critic={self.learning_rate_critic}, workers={self.num_workers})")

    def __repr__(self) -> str:
        return self.__str__()

# Funciones de utilidad específicas para RFEN1_RN_5


def crear_neurona_a3c(input_size: int, output_size: int,
                      configuracion: Dict[str, Any] = None) -> NeuronaRefuerzoA3C:
    """
    Función de conveniencia para crear una neurona A3C.

    Args:
        input_size: Número de características de entrada
        output_size: Número de acciones
        configuracion: Configuración adicional

    Returns:
        Instancia de NeuronaRefuerzoA3C
    """
    if configuracion is None:
        configuracion = {}

    return NeuronaRefuerzoA3C(
        input_size=input_size,
        output_size=output_size,
        nombre=configuracion.get('nombre', 'NeuronaRefuerzoA3C'),
        learning_rate_actor=configuracion.get('learning_rate_actor', 0.001),
        learning_rate_critic=configuracion.get('learning_rate_critic', 0.002),
        gamma=configuracion.get('gamma', 0.99),
        lambda_gae=configuracion.get('lambda_gae', 0.95),
        num_workers=configuracion.get('num_workers', 4),
        update_frequency=configuracion.get('update_frequency', 20),
        usar_ortogonal=configuracion.get('usar_ortogonal', True),
        entropy_coef=configuracion.get('entropy_coef', 0.01)
    )


# Configuración específica para RFEN1_RN_5
RFEN5_CONFIG = {
    'inicializacion_preferida': 'ortogonal',
    'learning_rate_actor_default': 0.001,
    'learning_rate_critic_default': 0.002,
    'gamma_default': 0.99,
    'lambda_gae_default': 0.95,
    'num_workers_default': 4,
    'update_frequency_default': 20,
    'entropy_coef_default': 0.01,
    'umbral_convergencia': 0.7,
    'umbral_estabilidad': 0.7
}

logger.info("RFEN1_RN_5.py cargado correctamente - Neurona de Refuerzo A3C Distribuida")
