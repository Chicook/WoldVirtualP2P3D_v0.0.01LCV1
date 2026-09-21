"""
RFEN1_RN_10.py - Neurona de Refuerzo con IMPALA (Importance Weighted Actor-Learner Architecture)
=================================================================================================

Esta neurona implementa el algoritmo IMPALA con optimización de pesos distribuidos
y arquitectura actor-learner para escalabilidad masiva.

Características:
- IMPALA con arquitectura actor-learner distribuida
- Inicialización de pesos específica para escalabilidad
- Monitoreo de convergencia distribuida
- Adaptación automática de hiperparámetros IMPALA

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

logger = logging.getLogger('RFENRN1.RFEN1_RN_10')


class NeuronaRefuerzoIMPALA(NeuronaRefuerzoBase):
    """
    Neurona de refuerzo con algoritmo IMPALA (Importance Weighted Actor-Learner Architecture).

    Esta neurona implementa IMPALA con arquitectura actor-learner distribuida,
    optimización de pesos específica para escalabilidad masiva.
    """

    def __init__(self, input_size: int, output_size: int,
                 nombre: str = "NeuronaRefuerzoIMPALA",
                 learning_rate: float = 0.0003,
                 gamma: float = 0.99,
                 lambda_gae: float = 0.95,
                 num_actors: int = 8,
                 num_learners: int = 2,
                 batch_size: int = 32,
                 sequence_length: int = 20,
                 usar_ortogonal: bool = True,
                 entropy_coef: float = 0.01,
                 value_coef: float = 0.5,
                 max_grad_norm: float = 0.5,
                 update_frequency: int = 100):
        """
        Inicializa la neurona de refuerzo IMPALA.

        Args:
            input_size: Número de características de entrada
            output_size: Número de acciones posibles
            nombre: Nombre identificativo de la neurona
            learning_rate: Tasa de aprendizaje
            gamma: Factor de descuento
            lambda_gae: Parámetro lambda para GAE
            num_actors: Número de actores distribuidos
            num_learners: Número de learners distribuidos
            batch_size: Tamaño del batch para entrenamiento
            sequence_length: Longitud de secuencias para entrenamiento
            usar_ortogonal: Si usar inicialización ortogonal
            entropy_coef: Coeficiente de entropía para regularización
            value_coef: Coeficiente para la función de valor
            max_grad_norm: Norma máxima para recorte de gradientes
            update_frequency: Frecuencia de actualización global
        """
        super().__init__(input_size, output_size, nombre)

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.num_actors = num_actors
        self.num_learners = num_learners
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.usar_ortogonal = usar_ortogonal
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.update_frequency = update_frequency

        # Pesos globales (learner) y locales (actors)
        self.pesos_actor_global = None
        self.sesgo_actor_global = None
        self.pesos_critic_global = None
        self.sesgo_critic_global = None

        # Pesos locales para cada actor
        self.pesos_actor_locales = {}
        self.sesgo_actor_locales = {}
        self.pesos_critic_locales = {}
        self.sesgo_critic_locales = {}

        # Buffer distribuido para experiencias
        self.buffer_experiencias = {}
        self.buffer_prioridades = {}

        # Lock para sincronización
        self.pesos_lock = threading.Lock()

        # Estadísticas específicas de IMPALA
        self.estadisticas_impala = {
            'actor_loss_medio': 0.0,
            'critic_loss_medio': 0.0,
            'entropy_loss_medio': 0.0,
            'total_loss_medio': 0.0,
            'importance_weights_medio': 0.0,
            'importance_weights_std': 0.0,
            'convergencia_actor': 0.0,
            'convergencia_critic': 0.0,
            'estabilidad_actor': 0.0,
            'estabilidad_critic': 0.0,
            'sincronizaciones': 0,
            'updates_globales': 0,
            'muestras_totales': 0,
            'gradiente_norma_actor': 0.0,
            'gradiente_norma_critic': 0.0,
            'actors_activos': 0,
            'learners_activos': 0
        }

        # Historial para análisis
        self.historial_actor_loss = []
        self.historial_critic_loss = []
        self.historial_entropy_loss = []
        self.historial_importance_weights = []
        self.historial_sincronizaciones = []

        logger.info(f"NeuronaRefuerzoIMPALA creada: {self} con {num_actors} actores y {num_learners} learners")

    def inicializar_pesos(self) -> None:
        """
        Inicializa los pesos globales y locales para todos los actores.
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

        # Inicializar pesos locales para cada actor
        for actor_id in range(self.num_actors):
            self.pesos_actor_locales[actor_id] = self.pesos_actor_global.copy()
            self.sesgo_actor_locales[actor_id] = self.sesgo_actor_global.copy()
            self.pesos_critic_locales[actor_id] = self.pesos_critic_global.copy()
            self.sesgo_critic_locales[actor_id] = self.sesgo_critic_global.copy()

            # Inicializar buffers locales
            self.buffer_experiencias[actor_id] = deque(maxlen=1000)
            self.buffer_prioridades[actor_id] = deque(maxlen=1000)

        logger.info(f"Pesos IMPALA inicializados con {'ortogonal' if self.usar_ortogonal else 'Xavier'} "
                    f"para {self.num_actors} actores")

    def forward_actor(self, estado: np.ndarray, actor_id: int = 0) -> np.ndarray:
        """
        Propagación hacia adelante del Actor para un actor específico.

        Args:
            estado: Estado actual del entorno
            actor_id: ID del actor

        Returns:
            Probabilidades de acción (softmax)
        """
        if self.pesos_actor_global is None:
            self.inicializar_pesos()

        # Usar pesos locales del actor
        pesos_actor = self.pesos_actor_locales.get(actor_id, self.pesos_actor_global)
        sesgo_actor = self.sesgo_actor_locales.get(actor_id, self.sesgo_actor_global)

        # Calcular logits del actor
        logits = np.dot(estado, pesos_actor) + sesgo_actor

        # Aplicar softmax para obtener probabilidades
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probabilidades = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        return probabilidades

    def forward_critic(self, estado: np.ndarray, actor_id: int = 0) -> np.ndarray:
        """
        Propagación hacia adelante del Critic para un actor específico.

        Args:
            estado: Estado actual del entorno
            actor_id: ID del actor

        Returns:
            Valor estimado del estado
        """
        if self.pesos_critic_global is None:
            self.inicializar_pesos()

        # Usar pesos locales del actor
        pesos_critic = self.pesos_critic_locales.get(actor_id, self.pesos_critic_global)
        sesgo_critic = self.sesgo_critic_locales.get(actor_id, self.sesgo_critic_global)

        # Calcular valor del estado
        valor = np.dot(estado, pesos_critic) + sesgo_critic

        return valor

    def forward(self, estado: np.ndarray, actor_id: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Propagación hacia adelante completa (Actor + Critic) para un actor.

        Args:
            estado: Estado actual del entorno
            actor_id: ID del actor

        Returns:
            Tupla con (probabilidades_accion, valor_estado)
        """
        probabilidades = self.forward_actor(estado, actor_id)
        valor = self.forward_critic(estado, actor_id)

        return probabilidades, valor

    def seleccionar_accion(self, estado: np.ndarray, actor_id: int = 0) -> Tuple[int, float, float]:
        """
        Selecciona una acción usando el Actor-Critic de un actor.

        Args:
            estado: Estado actual del entorno
            actor_id: ID del actor

        Returns:
            Tupla con (acción, probabilidad_accion, valor_estado)
        """
        probabilidades, valor = self.forward(estado, actor_id)

        # Muestrear acción según las probabilidades
        accion = np.random.choice(self.output_size, p=probabilidades[0])
        probabilidad_accion = probabilidades[0, accion]
        valor_estado = valor[0, 0]

        return accion, probabilidad_accion, valor_estado

    def agregar_experiencia(self, actor_id: int, estado: np.ndarray, accion: int,
                            recompensa: float, siguiente_estado: np.ndarray,
                            terminado: bool, probabilidad_accion: float) -> None:
        """
        Agrega una experiencia al buffer de un actor específico.

        Args:
            actor_id: ID del actor
            estado: Estado actual
            accion: Acción tomada
            recompensa: Recompensa recibida
            siguiente_estado: Siguiente estado
            terminado: Si el episodio ha terminado
            probabilidad_accion: Probabilidad de la acción tomada
        """
        experiencia = {
            'estado': estado.copy(),
            'accion': accion,
            'recompensa': recompensa,
            'siguiente_estado': siguiente_estado.copy(),
            'terminado': terminado,
            'probabilidad_accion': probabilidad_accion,
            'timestamp': time.time()
        }

        self.buffer_experiencias[actor_id].append(experiencia)
        self.estadisticas_impala['muestras_totales'] = sum(len(buf) for buf in self.buffer_experiencias.values())

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

    def calcular_importance_weights(self, experiencias: List[Dict[str, Any]], actor_id: int) -> List[float]:
        """
        Calcula los pesos de importancia para las experiencias.

        Args:
            experiencias: Lista de experiencias
            actor_id: ID del actor

        Returns:
            Lista de pesos de importancia
        """
        importance_weights = []

        for experiencia in experiencias:
            estado = experiencia['estado']
            accion = experiencia['accion']
            probabilidad_antigua = experiencia['probabilidad_accion']

            # Calcular probabilidad actual con pesos globales
            probabilidades_actuales = self.forward_actor(estado, actor_id)
            probabilidad_actual = probabilidades_actuales[0, accion]

            # Calcular peso de importancia
            importance_weight = probabilidad_actual / (probabilidad_antigua + 1e-8)

            # Recortar peso de importancia para estabilidad
            importance_weight = np.clip(importance_weight, 0.1, 10.0)

            importance_weights.append(importance_weight)

        # Guardar estadísticas
        if importance_weights:
            self.estadisticas_impala['importance_weights_medio'] = np.mean(importance_weights)
            self.estadisticas_impala['importance_weights_std'] = np.std(importance_weights)
            self.historial_importance_weights.extend(importance_weights)

        return importance_weights

    def calcular_gradientes_impala(self, experiencias: List[Dict[str, Any]],
                                   actor_id: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calcula los gradientes para IMPALA usando pesos de importancia.

        Args:
            experiencias: Lista de experiencias
            actor_id: ID del actor

        Returns:
            Tupla con (grad_actor_pesos, grad_actor_sesgo, grad_critic_pesos, grad_critic_sesgo)
        """
        if self.pesos_actor_global is None:
            raise ValueError("Pesos no inicializados")

        # Calcular pesos de importancia
        importance_weights = self.calcular_importance_weights(experiencias, actor_id)

        # Extraer datos de las experiencias
        estados = [exp['estado'] for exp in experiencias]
        acciones = [exp['accion'] for exp in experiencias]
        recompensas = [exp['recompensa'] for exp in experiencias]
        siguientes_estados = [exp['siguiente_estado'] for exp in experiencias]
        terminados = [exp['terminado'] for exp in experiencias]

        # Calcular valores y ventajas
        valores = [self.forward_critic(estado, actor_id)[0, 0] for estado in estados]
        advantages = self.calcular_gae_advantages(recompensas, valores)

        # Gradientes del Actor
        grad_actor_pesos = np.zeros_like(self.pesos_actor_global)
        grad_actor_sesgo = np.zeros_like(self.sesgo_actor_global)

        # Gradientes del Critic
        grad_critic_pesos = np.zeros_like(self.pesos_critic_global)
        grad_critic_sesgo = np.zeros_like(self.sesgo_critic_global)

        actor_loss_total = 0
        critic_loss_total = 0
        entropy_loss_total = 0

        for estado, accion, advantage, importance_weight, valor in zip(estados, acciones, advantages, importance_weights, valores):
            # Calcular probabilidades actuales
            probabilidades = self.forward_actor(estado, actor_id)

            # Gradiente del Actor con peso de importancia
            grad_log_prob = np.zeros(self.output_size)
            grad_log_prob[accion] = 1.0 / (probabilidades[0, accion] + 1e-8)

            grad_actor_pesos += np.outer(estado[0], grad_log_prob) * advantage * importance_weight
            grad_actor_sesgo += grad_log_prob * advantage * importance_weight

            actor_loss_total += -np.log(probabilidades[0, accion] + 1e-8) * advantage * importance_weight

            # Gradiente del Critic
            valor_objetivo = advantage + valor
            error_valor = valor_objetivo - valor

            grad_critic_pesos += np.outer(estado[0], np.array([error_valor])) * importance_weight
            grad_critic_sesgo += np.array([[error_valor]]) * importance_weight

            critic_loss_total += 0.5 * (error_valor ** 2) * importance_weight

            # Pérdida de entropía
            entropy = -np.sum(probabilidades * np.log(probabilidades + 1e-8), axis=1)
            entropy_loss_total += np.mean(entropy) * importance_weight

        # Normalizar por el número de muestras
        n_muestras = len(experiencias)
        grad_actor_pesos /= n_muestras
        grad_actor_sesgo /= n_muestras
        grad_critic_pesos /= n_muestras
        grad_critic_sesgo /= n_muestras

        # Guardar estadísticas
        self.estadisticas_impala['actor_loss_medio'] = actor_loss_total / n_muestras
        self.estadisticas_impala['critic_loss_medio'] = critic_loss_total / n_muestras
        self.estadisticas_impala['entropy_loss_medio'] = entropy_loss_total / n_muestras
        self.estadisticas_impala['total_loss_medio'] = (self.estadisticas_impala['actor_loss_medio'] +
                                                        self.value_coef * self.estadisticas_impala['critic_loss_medio'] -
                                                        self.entropy_coef * self.estadisticas_impala['entropy_loss_medio'])

        return grad_actor_pesos, grad_actor_sesgo, grad_critic_pesos, grad_critic_sesgo

    def actualizar_pesos_learner(self, grad_actor_pesos: np.ndarray, grad_actor_sesgo: np.ndarray,
                                 grad_critic_pesos: np.ndarray, grad_critic_sesgo: np.ndarray) -> None:
        """
        Actualiza los pesos globales del learner.

        Args:
            grad_actor_pesos: Gradiente de los pesos del actor
            grad_actor_sesgo: Gradiente del sesgo del actor
            grad_critic_pesos: Gradiente de los pesos del crítico
            grad_critic_sesgo: Gradiente del sesgo del crítico
        """
        # Verificar que los pesos globales estén inicializados
        if self.pesos_actor_global is None or self.pesos_critic_global is None:
            self.inicializar_pesos()

        with self.pesos_lock:
            # Recorte de gradientes
            grad_norm_actor = np.linalg.norm(grad_actor_pesos)
            grad_norm_critic = np.linalg.norm(grad_critic_pesos)

            if grad_norm_actor > self.max_grad_norm:
                grad_actor_pesos *= self.max_grad_norm / (grad_norm_actor + 1e-8)
                grad_actor_sesgo *= self.max_grad_norm / (grad_norm_actor + 1e-8)

            if grad_norm_critic > self.max_grad_norm:
                grad_critic_pesos *= self.max_grad_norm / (grad_norm_critic + 1e-8)
                grad_critic_sesgo *= self.max_grad_norm / (grad_norm_critic + 1e-8)

            # Asegurar que los pesos no son None después de inicialización
            assert self.pesos_actor_global is not None
            assert self.sesgo_actor_global is not None
            assert self.pesos_critic_global is not None
            assert self.sesgo_critic_global is not None

            # Actualizar pesos globales
            self.pesos_actor_global += self.learning_rate * grad_actor_pesos
            self.sesgo_actor_global += self.learning_rate * grad_actor_sesgo
            self.pesos_critic_global += self.learning_rate * grad_critic_pesos
            self.sesgo_critic_global += self.learning_rate * grad_critic_sesgo

            # Guardar estadísticas
            self.estadisticas_impala['gradiente_norma_actor'] = grad_norm_actor
            self.estadisticas_impala['gradiente_norma_critic'] = grad_norm_critic
            self.estadisticas_impala['updates_globales'] += 1

    def sincronizar_actores(self) -> None:
        """
        Sincroniza todos los actores con los pesos globales.
        """
        # Verificar que los pesos globales estén inicializados
        if self.pesos_actor_global is None or self.pesos_critic_global is None:
            self.inicializar_pesos()

        # Asegurar que los pesos no son None después de inicialización
        assert self.pesos_actor_global is not None
        assert self.sesgo_actor_global is not None
        assert self.pesos_critic_global is not None
        assert self.sesgo_critic_global is not None

        with self.pesos_lock:
            for actor_id in range(self.num_actors):
                self.pesos_actor_locales[actor_id] = self.pesos_actor_global.copy()
                self.sesgo_actor_locales[actor_id] = self.sesgo_actor_global.copy()
                self.pesos_critic_locales[actor_id] = self.pesos_critic_global.copy()
                self.sesgo_critic_locales[actor_id] = self.sesgo_critic_global.copy()

            self.estadisticas_impala['sincronizaciones'] += 1
            self.historial_sincronizaciones.append(time.time())

    def entrenar_step(self, actor_id: int) -> Optional[Dict[str, float]]:
        """
        Realiza un paso de entrenamiento para un actor específico.

        Args:
            actor_id: ID del actor

        Returns:
            Diccionario con estadísticas de entrenamiento o None si no hay suficientes muestras
        """
        if len(self.buffer_experiencias[actor_id]) < self.batch_size:
            return None

        # Muestrear experiencias del buffer del actor
        experiencias = random.sample(list(self.buffer_experiencias[actor_id]), self.batch_size)

        # Calcular gradientes
        grad_actor_pesos, grad_actor_sesgo, grad_critic_pesos, grad_critic_sesgo = \
            self.calcular_gradientes_impala(experiencias, actor_id)

        # Actualizar pesos globales
        self.actualizar_pesos_learner(grad_actor_pesos, grad_actor_sesgo, grad_critic_pesos, grad_critic_sesgo)

        # Sincronizar actores periódicamente
        if self.estadisticas_impala['updates_globales'] % self.update_frequency == 0:
            self.sincronizar_actores()

        # Guardar historial
        self.historial_actor_loss.append(self.estadisticas_impala['actor_loss_medio'])
        self.historial_critic_loss.append(self.estadisticas_impala['critic_loss_medio'])
        self.historial_entropy_loss.append(self.estadisticas_impala['entropy_loss_medio'])

        return {
            'actor_loss': self.estadisticas_impala['actor_loss_medio'],
            'critic_loss': self.estadisticas_impala['critic_loss_medio'],
            'entropy_loss': self.estadisticas_impala['entropy_loss_medio'],
            'total_loss': self.estadisticas_impala['total_loss_medio'],
            'importance_weights_medio': self.estadisticas_impala['importance_weights_medio'],
            'sincronizaciones': self.estadisticas_impala['sincronizaciones']
        }

    def calcular_convergencia_global(self) -> None:
        """
        Calcula la convergencia global basada en la estabilidad de los pesos.
        """
        if len(self.historial_sincronizaciones) < 5:
            return

        # Calcular convergencia basada en la estabilidad de las pérdidas recientes
        if len(self.historial_actor_loss) >= 5:
            actor_loss_reciente = self.historial_actor_loss[-5:]
            varianza_actor_loss = np.var(actor_loss_reciente)

            # Convergencia inversamente proporcional a la varianza
            convergencia = 1.0 / (1.0 + varianza_actor_loss)
            self.estadisticas_impala['convergencia_actor'] = convergencia
            self.estadisticas_impala['convergencia_critic'] = convergencia

    def calcular_estabilidad_global(self) -> Tuple[float, float]:
        """
        Calcula la estabilidad global del Actor y Critic.

        Returns:
            Tupla con (estabilidad_actor, estabilidad_critic)
        """
        estabilidad_actor = 0.0
        estabilidad_critic = 0.0

        # Calcular estabilidad basada en la consistencia entre actores
        if len(self.pesos_actor_locales) > 1:
            pesos_actor_array = np.array(list(self.pesos_actor_locales.values()))
            varianza_actors = np.var(pesos_actor_array, axis=0)
            estabilidad_actor = 1.0 / (1.0 + np.mean(varianza_actors))

        if len(self.pesos_critic_locales) > 1:
            pesos_critic_array = np.array(list(self.pesos_critic_locales.values()))
            varianza_actors = np.var(pesos_critic_array, axis=0)
            estabilidad_critic = 1.0 / (1.0 + np.mean(varianza_actors))

        self.estadisticas_impala['estabilidad_actor'] = estabilidad_actor
        self.estadisticas_impala['estabilidad_critic'] = estabilidad_critic

        return estabilidad_actor, estabilidad_critic

    def obtener_estadisticas_impala(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas específicas de IMPALA.

        Returns:
            Diccionario con estadísticas de IMPALA
        """
        if self.pesos_actor_global is None:
            return {'estado': 'no_inicializada'}

        estabilidad_actor, estabilidad_critic = self.calcular_estabilidad_global()

        stats_impala = {
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'lambda_gae': self.lambda_gae,
            'num_actors': self.num_actors,
            'num_learners': self.num_learners,
            'batch_size': self.batch_size,
            'sequence_length': self.sequence_length,
            'entropy_coef': self.entropy_coef,
            'value_coef': self.value_coef,
            'max_grad_norm': self.max_grad_norm,
            'update_frequency': self.update_frequency,
            'actor_loss_medio': self.estadisticas_impala['actor_loss_medio'],
            'critic_loss_medio': self.estadisticas_impala['critic_loss_medio'],
            'entropy_loss_medio': self.estadisticas_impala['entropy_loss_medio'],
            'total_loss_medio': self.estadisticas_impala['total_loss_medio'],
            'importance_weights_medio': self.estadisticas_impala['importance_weights_medio'],
            'importance_weights_std': self.estadisticas_impala['importance_weights_std'],
            'convergencia_actor': self.estadisticas_impala['convergencia_actor'],
            'convergencia_critic': self.estadisticas_impala['convergencia_critic'],
            'estabilidad_actor': estabilidad_actor,
            'estabilidad_critic': estabilidad_critic,
            'sincronizaciones': self.estadisticas_impala['sincronizaciones'],
            'updates_globales': self.estadisticas_impala['updates_globales'],
            'muestras_totales': self.estadisticas_impala['muestras_totales']
        }

        return stats_impala

    def verificar_estabilidad(self) -> Dict[str, bool]:
        """
        Verifica la estabilidad de la neurona IMPALA.

        Returns:
            Diccionario con indicadores de estabilidad
        """
        estabilidad = {}

        # Verificar convergencia global
        convergencia_actor = self.estadisticas_impala['convergencia_actor']
        convergencia_critic = self.estadisticas_impala['convergencia_critic']
        estabilidad['actor_convergencia_ok'] = convergencia_actor > 0.7
        estabilidad['critic_convergencia_ok'] = convergencia_critic > 0.7

        # Verificar estabilidad global
        estabilidad_actor, estabilidad_critic = self.calcular_estabilidad_global()
        estabilidad['actor_estable'] = estabilidad_actor > 0.7
        estabilidad['critic_estable'] = estabilidad_critic > 0.7

        # Verificar sincronización
        estabilidad['sincronizacion_ok'] = self.estadisticas_impala['sincronizaciones'] > 0

        # Verificar pesos de importancia
        importance_weights_medio = self.estadisticas_impala['importance_weights_medio']
        estabilidad['importance_weights_estables'] = 0.5 <= importance_weights_medio <= 2.0

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

    def reinicializar_con_parametros(self, learning_rate: Optional[float] = None,
                                     gamma: Optional[float] = None,
                                     lambda_gae: Optional[float] = None,
                                     num_actors: Optional[int] = None) -> None:
        """
        Reinicializa la neurona con nuevos parámetros.

        Args:
            learning_rate: Nueva tasa de aprendizaje
            gamma: Nuevo factor de descuento
            lambda_gae: Nuevo parámetro lambda para GAE
            num_actors: Nuevo número de actores
        """
        if learning_rate is not None:
            self.learning_rate = learning_rate
        if gamma is not None:
            self.gamma = gamma
        if lambda_gae is not None:
            self.lambda_gae = lambda_gae
        if num_actors is not None:
            self.num_actors = num_actors

        # Reinicializar pesos
        self.inicializar_pesos()

        # Resetear historial
        self.resetear_historial()

        # Limpiar buffers y estadísticas específicas
        for actor_id in range(self.num_actors):
            self.buffer_experiencias[actor_id].clear()
            self.buffer_prioridades[actor_id].clear()

        self.historial_actor_loss.clear()
        self.historial_critic_loss.clear()
        self.historial_entropy_loss.clear()
        self.historial_importance_weights.clear()
        self.historial_sincronizaciones.clear()

        # Resetear contadores
        self.estadisticas_impala['sincronizaciones'] = 0
        self.estadisticas_impala['updates_globales'] = 0
        self.estadisticas_impala['muestras_totales'] = 0

        logger.info(f"Neurona IMPALA reinicializada: lr={self.learning_rate}, "
                    f"gamma={self.gamma}, actors={self.num_actors}")

    def __str__(self) -> str:
        return (f"NeuronaRefuerzoIMPALA(entrada={self.input_size}, "
                f"salida={self.output_size}, lr={self.learning_rate}, "
                f"gamma={self.gamma}, actors={self.num_actors}, "
                f"learners={self.num_learners})")

    def __repr__(self) -> str:
        return self.__str__()

# Funciones de utilidad específicas para RFEN1_RN_10


def crear_neurona_impala(input_size: int, output_size: int,
                         configuracion: Optional[Dict[str, Any]] = None) -> NeuronaRefuerzoIMPALA:
    """
    Función de conveniencia para crear una neurona IMPALA.

    Args:
        input_size: Número de características de entrada
        output_size: Número de acciones
        configuracion: Configuración adicional

    Returns:
        Instancia de NeuronaRefuerzoIMPALA
    """
    if configuracion is None:
        configuracion = {}

    return NeuronaRefuerzoIMPALA(
        input_size=input_size,
        output_size=output_size,
        nombre=configuracion.get('nombre', 'NeuronaRefuerzoIMPALA'),
        learning_rate=configuracion.get('learning_rate', 0.0003),
        gamma=configuracion.get('gamma', 0.99),
        lambda_gae=configuracion.get('lambda_gae', 0.95),
        num_actors=configuracion.get('num_actors', 8),
        num_learners=configuracion.get('num_learners', 2),
        batch_size=configuracion.get('batch_size', 32),
        sequence_length=configuracion.get('sequence_length', 20),
        usar_ortogonal=configuracion.get('usar_ortogonal', True),
        entropy_coef=configuracion.get('entropy_coef', 0.01),
        value_coef=configuracion.get('value_coef', 0.5),
        max_grad_norm=configuracion.get('max_grad_norm', 0.5),
        update_frequency=configuracion.get('update_frequency', 100)
    )


# Configuración específica para RFEN1_RN_10
RFEN10_CONFIG = {
    'inicializacion_preferida': 'ortogonal',
    'learning_rate_default': 0.0003,
    'gamma_default': 0.99,
    'lambda_gae_default': 0.95,
    'num_actors_default': 8,
    'num_learners_default': 2,
    'batch_size_default': 32,
    'sequence_length_default': 20,
    'entropy_coef_default': 0.01,
    'value_coef_default': 0.5,
    'max_grad_norm_default': 0.5,
    'update_frequency_default': 100,
    'umbral_convergencia': 0.7,
    'umbral_estabilidad': 0.7
}

logger.info("RFEN1_RN_10.py cargado correctamente - Neurona de Refuerzo IMPALA Distribuida")
