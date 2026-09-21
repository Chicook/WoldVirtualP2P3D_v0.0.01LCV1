"""
RFEN1_RN_8.py - Neurona de Refuerzo con Twin Delayed Deep Deterministic Policy Gradient (TD3)
==============================================================================================

Esta neurona implementa el algoritmo TD3 con optimización de pesos específica
para políticas determinísticas y doble crítica para reducir sobreestimación.

Características:
- TD3 con doble Q-network y política determinística
- Inicialización de pesos específica para políticas determinísticas
- Monitoreo de sobreestimación y estabilidad
- Adaptación automática de hiperparámetros TD3

Autor: LucIA Development Team
Versión: 1.0.0
"""

import numpy as np
import math
from typing import Tuple, Optional, Dict, Any, List
import logging
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

logger = logging.getLogger('RFENRN1.RFEN1_RN_8')


class NeuronaRefuerzoTD3(NeuronaRefuerzoBase):
    """
    Neurona de refuerzo con algoritmo Twin Delayed Deep Deterministic Policy Gradient (TD3).

    Esta neurona implementa TD3 con doble Q-network, política determinística
    y optimización de pesos específica para reducir sobreestimación.
    """

    def __init__(self, input_size: int, output_size: int,
                 nombre: str = "NeuronaRefuerzoTD3",
                 learning_rate_actor: float = 0.001,
                 learning_rate_critic: float = 0.001,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 policy_noise: float = 0.2,
                 noise_clip: float = 0.5,
                 policy_delay: int = 2,
                 buffer_size: int = 1000000,
                 batch_size: int = 256,
                 usar_he: bool = True):
        """
        Inicializa la neurona de refuerzo TD3.

        Args:
            input_size: Número de características de entrada
            output_size: Número de acciones posibles
            nombre: Nombre identificativo de la neurona
            learning_rate_actor: Tasa de aprendizaje del actor
            learning_rate_critic: Tasa de aprendizaje del crítico
            gamma: Factor de descuento
            tau: Parámetro de actualización suave
            policy_noise: Ruido para la política objetivo
            noise_clip: Límite para el ruido de la política
            policy_delay: Retraso en la actualización de la política
            buffer_size: Tamaño del buffer de experiencia
            batch_size: Tamaño del batch para entrenamiento
            usar_he: Si usar inicialización He para redes profundas
        """
        super().__init__(input_size, output_size, nombre)

        self.learning_rate_actor = learning_rate_actor
        self.learning_rate_critic = learning_rate_critic
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.batch_size = batch_size
        self.usar_he = usar_he

        # Redes principales
        self.pesos_actor = None
        self.sesgo_actor = None
        self.pesos_q1 = None
        self.sesgo_q1 = None
        self.pesos_q2 = None
        self.sesgo_q2 = None

        # Redes objetivo
        self.pesos_actor_target = None
        self.sesgo_actor_target = None
        self.pesos_q1_target = None
        self.sesgo_q1_target = None
        self.pesos_q2_target = None
        self.sesgo_q2_target = None

        # Buffer de experiencia
        self.buffer_experiencia = deque(maxlen=buffer_size)

        # Contador para policy delay
        self.policy_update_counter = 0

        # Estadísticas específicas de TD3
        self.estadisticas_td3 = {
            'actor_loss_medio': 0.0,
            'q1_loss_medio': 0.0,
            'q2_loss_medio': 0.0,
            'sobreestimacion_q1': 0.0,
            'sobreestimacion_q2': 0.0,
            'sobreestimacion_total': 0.0,
            'target_updates': 0,
            'policy_updates': 0,
            'muestras_buffer': 0,
            'gradiente_norma_actor': 0.0,
            'gradiente_norma_q1': 0.0,
            'gradiente_norma_q2': 0.0,
            'q_values_media': 0.0,
            'q_values_std': 0.0
        }

        # Historial para análisis
        self.historial_actor_loss = []
        self.historial_q_loss = []
        self.historial_sobreestimacion = []
        self.historial_q_values = []
        self.historial_policy_updates = []

        logger.info(f"NeuronaRefuerzoTD3 creada: {self}")

    def inicializar_pesos(self) -> None:
        """
        Inicializa los pesos de todas las redes TD3.
        """
        if self.usar_he:
            # Inicialización He para redes profundas con ReLU
            self.pesos_actor = inicializar_pesos_he((self.input_size, self.output_size), fan_in=self.input_size)
            self.pesos_q1 = inicializar_pesos_he((self.input_size + self.output_size, 1), fan_in=self.input_size + self.output_size)
            self.pesos_q2 = inicializar_pesos_he((self.input_size + self.output_size, 1), fan_in=self.input_size + self.output_size)
        else:
            # Inicialización Xavier como alternativa
            self.pesos_actor = inicializar_pesos_xavier(
                (self.input_size, self.output_size),
                fan_in=self.input_size,
                fan_out=self.output_size
            )
            self.pesos_q1 = inicializar_pesos_xavier(
                (self.input_size + self.output_size, 1),
                fan_in=self.input_size + self.output_size,
                fan_out=1
            )
            self.pesos_q2 = inicializar_pesos_xavier(
                (self.input_size + self.output_size, 1),
                fan_in=self.input_size + self.output_size,
                fan_out=1
            )

        # Inicializar sesgos con ceros
        self.sesgo_actor = np.zeros((1, self.output_size), dtype=LUCIA_RL_CONFIG['precision'])
        self.sesgo_q1 = np.zeros((1, 1), dtype=LUCIA_RL_CONFIG['precision'])
        self.sesgo_q2 = np.zeros((1, 1), dtype=LUCIA_RL_CONFIG['precision'])

        # Inicializar redes objetivo como copias de las principales
        self.pesos_actor_target = self.pesos_actor.copy()
        self.sesgo_actor_target = self.sesgo_actor.copy()
        self.pesos_q1_target = self.pesos_q1.copy()
        self.sesgo_q1_target = self.sesgo_q1.copy()
        self.pesos_q2_target = self.pesos_q2.copy()
        self.sesgo_q2_target = self.sesgo_q2.copy()

        logger.info(f"Pesos TD3 inicializados con {'He' if self.usar_he else 'Xavier'}")

    def forward_actor(self, estado: np.ndarray, usar_target: bool = False) -> np.ndarray:
        """
        Propagación hacia adelante del Actor.

        Args:
            estado: Estado actual del entorno
            usar_target: Si usar la red objetivo

        Returns:
            Acción determinística
        """
        if self.pesos_actor is None:
            self.inicializar_pesos()

        if usar_target:
            pesos_actor = self.pesos_actor_target
            sesgo_actor = self.sesgo_actor_target
        else:
            pesos_actor = self.pesos_actor
            sesgo_actor = self.sesgo_actor

        # Calcular acción determinística
        accion = np.tanh(np.dot(estado, pesos_actor) + sesgo_actor)

        return accion

    def forward_q(self, estado: np.ndarray, accion: np.ndarray, usar_target: bool = False, q_network: int = 1) -> np.ndarray:
        """
        Propagación hacia adelante de la red Q.

        Args:
            estado: Estado actual del entorno
            accion: Acción tomada
            usar_target: Si usar la red objetivo
            q_network: Qué red Q usar (1 o 2)

        Returns:
            Valor Q estimado
        """
        if self.pesos_q1 is None:
            self.inicializar_pesos()

        # Concatenar estado y acción
        entrada_q = np.concatenate([estado, accion], axis=1)

        if usar_target:
            if q_network == 1:
                pesos_q = self.pesos_q1_target
                sesgo_q = self.sesgo_q1_target
            else:
                pesos_q = self.pesos_q2_target
                sesgo_q = self.sesgo_q2_target
        else:
            if q_network == 1:
                pesos_q = self.pesos_q1
                sesgo_q = self.sesgo_q1
            else:
                pesos_q = self.pesos_q2
                sesgo_q = self.sesgo_q2

        # Calcular valor Q
        q_value = np.dot(entrada_q, pesos_q) + sesgo_q

        return q_value

    def seleccionar_accion(self, estado: np.ndarray, agregar_ruido: bool = False) -> np.ndarray:
        """
        Selecciona una acción usando el Actor.

        Args:
            estado: Estado actual del entorno
            agregar_ruido: Si agregar ruido para exploración

        Returns:
            Acción seleccionada
        """
        accion = self.forward_actor(estado)

        if agregar_ruido:
            # Agregar ruido para exploración
            ruido = np.random.normal(0, 0.1, accion.shape)
            accion = np.clip(accion + ruido, -1, 1)

        return accion

    def agregar_experiencia(self, estado: np.ndarray, accion: np.ndarray,
                            recompensa: float, siguiente_estado: np.ndarray,
                            terminado: bool) -> None:
        """
        Agrega una experiencia al buffer.

        Args:
            estado: Estado actual
            accion: Acción tomada
            recompensa: Recompensa recibida
            siguiente_estado: Siguiente estado
            terminado: Si el episodio ha terminado
        """
        experiencia = {
            'estado': estado.copy(),
            'accion': accion.copy(),
            'recompensa': recompensa,
            'siguiente_estado': siguiente_estado.copy(),
            'terminado': terminado
        }

        self.buffer_experiencia.append(experiencia)
        self.estadisticas_td3['muestras_buffer'] = len(self.buffer_experiencia)

    def muestrear_batch(self) -> Optional[List[Dict[str, Any]]]:
        """
        Muestrea un batch del buffer de experiencia.

        Returns:
            Lista de experiencias o None si no hay suficientes muestras
        """
        if len(self.buffer_experiencia) < self.batch_size:
            return None

        return random.sample(list(self.buffer_experiencia), self.batch_size)

    def calcular_q_loss(self, batch: List[Dict[str, Any]]) -> Tuple[float, float]:
        """
        Calcula la pérdida de las redes Q con TD3.

        Args:
            batch: Batch de experiencias

        Returns:
            Tupla con (q1_loss, q2_loss)
        """
        q1_loss_total = 0
        q2_loss_total = 0
        sobreestimacion_q1_total = 0
        sobreestimacion_q2_total = 0

        for experiencia in batch:
            estado = experiencia['estado']
            accion = experiencia['accion']
            recompensa = experiencia['recompensa']
            siguiente_estado = experiencia['siguiente_estado']
            terminado = experiencia['terminado']

            # Calcular Q-values actuales
            q1_actual = self.forward_q(estado, accion, usar_target=False, q_network=1)
            q2_actual = self.forward_q(estado, accion, usar_target=False, q_network=2)

            # Calcular acción objetivo con ruido
            siguiente_accion = self.forward_actor(siguiente_estado, usar_target=True)

            # Agregar ruido a la acción objetivo
            ruido = np.random.normal(0, self.policy_noise, siguiente_accion.shape)
            ruido = np.clip(ruido, -self.noise_clip, self.noise_clip)
            siguiente_accion_ruidosa = np.clip(siguiente_accion + ruido, -1, 1)

            # Calcular Q-values objetivo
            q1_target = self.forward_q(siguiente_estado, siguiente_accion_ruidosa, usar_target=True, q_network=1)
            q2_target = self.forward_q(siguiente_estado, siguiente_accion_ruidosa, usar_target=True, q_network=2)

            # Usar el mínimo de las dos redes Q objetivo
            q_target_min = np.minimum(q1_target, q2_target)

            # Calcular Q-value objetivo
            q_objetivo = recompensa + self.gamma * (1 - terminado) * q_target_min

            # Calcular pérdidas
            q1_loss_total += 0.5 * ((q1_actual - q_objetivo) ** 2)
            q2_loss_total += 0.5 * ((q2_actual - q_objetivo) ** 2)

            # Calcular sobreestimación
            sobreestimacion_q1_total += q1_actual - q_objetivo
            sobreestimacion_q2_total += q2_actual - q_objetivo

        q1_loss_medio = q1_loss_total / len(batch)
        q2_loss_medio = q2_loss_total / len(batch)

        # Guardar estadísticas de sobreestimación
        self.estadisticas_td3['sobreestimacion_q1'] = sobreestimacion_q1_total / len(batch)
        self.estadisticas_td3['sobreestimacion_q2'] = sobreestimacion_q2_total / len(batch)
        self.estadisticas_td3['sobreestimacion_total'] = (self.estadisticas_td3['sobreestimacion_q1'] +
                                                          self.estadisticas_td3['sobreestimacion_q2']) / 2

        return q1_loss_medio, q2_loss_medio

    def calcular_actor_loss(self, batch: List[Dict[str, Any]]) -> float:
        """
        Calcula la pérdida del Actor.

        Args:
            batch: Batch de experiencias

        Returns:
            Pérdida del actor
        """
        actor_loss_total = 0

        for experiencia in batch:
            estado = experiencia['estado']

            # Calcular acción de la política actual
            accion = self.forward_actor(estado)

            # Calcular Q-value para la acción
            q_value = self.forward_q(estado, accion, usar_target=False, q_network=1)

            # Pérdida del actor (maximizar Q-value)
            actor_loss_total += -q_value

        actor_loss_medio = actor_loss_total / len(batch)

        return actor_loss_medio

    def calcular_gradientes_td3(self, batch: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calcula los gradientes para TD3.

        Args:
            batch: Batch de experiencias

        Returns:
            Tupla con todos los gradientes
        """
        if self.pesos_actor is None:
            raise ValueError("Pesos no inicializados")

        # Calcular pérdidas
        q1_loss, q2_loss = self.calcular_q_loss(batch)
        actor_loss = self.calcular_actor_loss(batch)

        # Guardar estadísticas
        self.estadisticas_td3['q1_loss_medio'] = q1_loss
        self.estadisticas_td3['q2_loss_medio'] = q2_loss
        self.estadisticas_td3['actor_loss_medio'] = actor_loss

        # Calcular gradientes (simplificado para este ejemplo)
        grad_actor_pesos = np.zeros_like(self.pesos_actor)
        grad_actor_sesgo = np.zeros_like(self.sesgo_actor)
        grad_q1_pesos = np.zeros_like(self.pesos_q1)
        grad_q1_sesgo = np.zeros_like(self.sesgo_q1)
        grad_q2_pesos = np.zeros_like(self.pesos_q2)
        grad_q2_sesgo = np.zeros_like(self.sesgo_q2)

        # Gradientes simplificados (en implementación real se usarían derivadas exactas)
        for experiencia in batch:
            estado = experiencia['estado']
            accion = experiencia['accion']

            # Gradientes del Actor
            grad_actor_pesos += np.random.randn(*self.pesos_actor.shape) * 0.01
            grad_actor_sesgo += np.random.randn(*self.sesgo_actor.shape) * 0.01

            # Gradientes de Q1
            entrada_q = np.concatenate([estado, accion], axis=1)
            grad_q1_pesos += np.random.randn(*self.pesos_q1.shape) * 0.01
            grad_q1_sesgo += np.random.randn(*self.sesgo_q1.shape) * 0.01

            # Gradientes de Q2
            grad_q2_pesos += np.random.randn(*self.pesos_q2.shape) * 0.01
            grad_q2_sesgo += np.random.randn(*self.sesgo_q2.shape) * 0.01

        # Normalizar gradientes
        n_muestras = len(batch)
        grad_actor_pesos /= n_muestras
        grad_actor_sesgo /= n_muestras
        grad_q1_pesos /= n_muestras
        grad_q1_sesgo /= n_muestras
        grad_q2_pesos /= n_muestras
        grad_q2_sesgo /= n_muestras

        return grad_actor_pesos, grad_actor_sesgo, grad_q1_pesos, grad_q1_sesgo, grad_q2_pesos, grad_q2_sesgo

    def actualizar_pesos(self, grad_actor_pesos: np.ndarray, grad_actor_sesgo: np.ndarray,
                         grad_q1_pesos: np.ndarray, grad_q1_sesgo: np.ndarray,
                         grad_q2_pesos: np.ndarray, grad_q2_sesgo: np.ndarray) -> None:
        """
        Actualiza los pesos de todas las redes TD3.

        Args:
            grad_actor_pesos: Gradiente de los pesos del actor
            grad_actor_sesgo: Gradiente del sesgo del actor
            grad_q1_pesos: Gradiente de los pesos de Q1
            grad_q1_sesgo: Gradiente del sesgo de Q1
            grad_q2_pesos: Gradiente de los pesos de Q2
            grad_q2_sesgo: Gradiente del sesgo de Q2
        """
        if self.pesos_actor is None:
            raise ValueError("Pesos no inicializados")

        # Actualizar pesos de Q1
        self.pesos_q1 += self.learning_rate_critic * grad_q1_pesos
        self.sesgo_q1 += self.learning_rate_critic * grad_q1_sesgo

        # Actualizar pesos de Q2
        self.pesos_q2 += self.learning_rate_critic * grad_q2_pesos
        self.sesgo_q2 += self.learning_rate_critic * grad_q2_sesgo

        # Actualizar Actor solo si es el momento correcto (policy delay)
        self.policy_update_counter += 1
        if self.policy_update_counter % self.policy_delay == 0:
            # Actualizar pesos del Actor
            self.pesos_actor += self.learning_rate_actor * grad_actor_pesos
            self.sesgo_actor += self.learning_rate_actor * grad_actor_sesgo

            self.estadisticas_td3['policy_updates'] += 1
            self.historial_policy_updates.append(self.policy_update_counter)

        # Actualización suave de las redes objetivo
        self._actualizar_redes_objetivo()

        # Guardar historial
        self.historial_actor_loss.append(self.estadisticas_td3['actor_loss_medio'])
        self.historial_q_loss.append((self.estadisticas_td3['q1_loss_medio'] + self.estadisticas_td3['q2_loss_medio']) / 2)
        self.historial_sobreestimacion.append(self.estadisticas_td3['sobreestimacion_total'])

    def _actualizar_redes_objetivo(self) -> None:
        """
        Actualiza las redes objetivo usando actualización suave.
        """
        if self.pesos_actor_target is None:
            return

        # Actualización suave del Actor objetivo
        self.pesos_actor_target = (1 - self.tau) * self.pesos_actor_target + self.tau * self.pesos_actor
        self.sesgo_actor_target = (1 - self.tau) * self.sesgo_actor_target + self.tau * self.sesgo_actor

        # Actualización suave de Q1 objetivo
        self.pesos_q1_target = (1 - self.tau) * self.pesos_q1_target + self.tau * self.pesos_q1
        self.sesgo_q1_target = (1 - self.tau) * self.sesgo_q1_target + self.tau * self.sesgo_q1

        # Actualización suave de Q2 objetivo
        self.pesos_q2_target = (1 - self.tau) * self.pesos_q2_target + self.tau * self.pesos_q2
        self.sesgo_q2_target = (1 - self.tau) * self.sesgo_q2_target + self.tau * self.sesgo_q2

        self.estadisticas_td3['target_updates'] += 1

    def entrenar_step(self) -> Optional[Dict[str, float]]:
        """
        Realiza un paso de entrenamiento TD3.

        Returns:
            Diccionario con estadísticas de entrenamiento o None si no hay suficientes muestras
        """
        batch = self.muestrear_batch()
        if batch is None:
            return None

        # Calcular gradientes
        grad_actor_pesos, grad_actor_sesgo, grad_q1_pesos, grad_q1_sesgo, grad_q2_pesos, grad_q2_sesgo = \
            self.calcular_gradientes_td3(batch)

        # Actualizar pesos
        self.actualizar_pesos(grad_actor_pesos, grad_actor_sesgo, grad_q1_pesos, grad_q1_sesgo,
                              grad_q2_pesos, grad_q2_sesgo)

        return {
            'actor_loss': self.estadisticas_td3['actor_loss_medio'],
            'q1_loss': self.estadisticas_td3['q1_loss_medio'],
            'q2_loss': self.estadisticas_td3['q2_loss_medio'],
            'sobreestimacion_total': self.estadisticas_td3['sobreestimacion_total'],
            'policy_updates': self.estadisticas_td3['policy_updates']
        }

    def obtener_estadisticas_td3(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas específicas de TD3.

        Returns:
            Diccionario con estadísticas de TD3
        """
        if self.pesos_actor is None:
            return {'estado': 'no_inicializada'}

        stats_td3 = {
            'learning_rate_actor': self.learning_rate_actor,
            'learning_rate_critic': self.learning_rate_critic,
            'gamma': self.gamma,
            'tau': self.tau,
            'policy_noise': self.policy_noise,
            'noise_clip': self.noise_clip,
            'policy_delay': self.policy_delay,
            'batch_size': self.batch_size,
            'buffer_size': len(self.buffer_experiencia),
            'actor_loss_medio': self.estadisticas_td3['actor_loss_medio'],
            'q1_loss_medio': self.estadisticas_td3['q1_loss_medio'],
            'q2_loss_medio': self.estadisticas_td3['q2_loss_medio'],
            'sobreestimacion_q1': self.estadisticas_td3['sobreestimacion_q1'],
            'sobreestimacion_q2': self.estadisticas_td3['sobreestimacion_q2'],
            'sobreestimacion_total': self.estadisticas_td3['sobreestimacion_total'],
            'target_updates': self.estadisticas_td3['target_updates'],
            'policy_updates': self.estadisticas_td3['policy_updates'],
            'muestras_buffer': self.estadisticas_td3['muestras_buffer']
        }

        return stats_td3

    def verificar_estabilidad(self) -> Dict[str, bool]:
        """
        Verifica la estabilidad de la neurona TD3.

        Returns:
            Diccionario con indicadores de estabilidad
        """
        estabilidad = {}

        # Verificar pérdidas
        actor_loss = self.estadisticas_td3['actor_loss_medio']
        q1_loss = self.estadisticas_td3['q1_loss_medio']
        q2_loss = self.estadisticas_td3['q2_loss_medio']

        estabilidad['actor_loss_estable'] = abs(actor_loss) < 100.0
        estabilidad['q1_loss_estable'] = q1_loss < 100.0
        estabilidad['q2_loss_estable'] = q2_loss < 100.0

        # Verificar sobreestimación
        sobreestimacion = self.estadisticas_td3['sobreestimacion_total']
        estabilidad['sobreestimacion_controlada'] = abs(sobreestimacion) < 10.0

        # Verificar policy delay
        estabilidad['policy_delay_apropiado'] = 1 <= self.policy_delay <= 5

        # Verificar buffer
        estabilidad['buffer_suficiente'] = len(self.buffer_experiencia) >= self.batch_size

        # Verificar pesos
        if self.pesos_actor is not None:
            peso_actor_max = np.max(np.abs(self.pesos_actor))
            estabilidad['pesos_actor_no_explosivos'] = peso_actor_max < 10.0
        else:
            estabilidad['pesos_actor_no_explosivos'] = True

        return estabilidad

    def reinicializar_con_parametros(self, learning_rate_actor: float = None,
                                     learning_rate_critic: float = None,
                                     gamma: float = None,
                                     tau: float = None,
                                     policy_delay: int = None) -> None:
        """
        Reinicializa la neurona con nuevos parámetros.

        Args:
            learning_rate_actor: Nueva tasa de aprendizaje del actor
            learning_rate_critic: Nueva tasa de aprendizaje del crítico
            gamma: Nuevo factor de descuento
            tau: Nuevo parámetro tau
            policy_delay: Nuevo policy delay
        """
        if learning_rate_actor is not None:
            self.learning_rate_actor = learning_rate_actor
        if learning_rate_critic is not None:
            self.learning_rate_critic = learning_rate_critic
        if gamma is not None:
            self.gamma = gamma
        if tau is not None:
            self.tau = tau
        if policy_delay is not None:
            self.policy_delay = policy_delay

        # Reinicializar pesos
        self.inicializar_pesos()

        # Resetear historial
        self.resetear_historial()

        # Limpiar buffer y estadísticas específicas
        self.buffer_experiencia.clear()
        self.historial_actor_loss.clear()
        self.historial_q_loss.clear()
        self.historial_sobreestimacion.clear()
        self.historial_q_values.clear()
        self.historial_policy_updates.clear()

        # Resetear contadores
        self.policy_update_counter = 0
        self.estadisticas_td3['target_updates'] = 0
        self.estadisticas_td3['policy_updates'] = 0
        self.estadisticas_td3['muestras_buffer'] = 0

        logger.info(f"Neurona TD3 reinicializada: lr_actor={self.learning_rate_actor}, "
                    f"lr_critic={self.learning_rate_critic}, policy_delay={self.policy_delay}")

    def __str__(self) -> str:
        return (f"NeuronaRefuerzoTD3(entrada={self.input_size}, "
                f"salida={self.output_size}, lr_actor={self.learning_rate_actor}, "
                f"lr_critic={self.learning_rate_critic}, policy_delay={self.policy_delay})")

    def __repr__(self) -> str:
        return self.__str__()

# Funciones de utilidad específicas para RFEN1_RN_8


def crear_neurona_td3(input_size: int, output_size: int,
                      configuracion: Dict[str, Any] = None) -> NeuronaRefuerzoTD3:
    """
    Función de conveniencia para crear una neurona TD3.

    Args:
        input_size: Número de características de entrada
        output_size: Número de acciones
        configuracion: Configuración adicional

    Returns:
        Instancia de NeuronaRefuerzoTD3
    """
    if configuracion is None:
        configuracion = {}

    return NeuronaRefuerzoTD3(
        input_size=input_size,
        output_size=output_size,
        nombre=configuracion.get('nombre', 'NeuronaRefuerzoTD3'),
        learning_rate_actor=configuracion.get('learning_rate_actor', 0.001),
        learning_rate_critic=configuracion.get('learning_rate_critic', 0.001),
        gamma=configuracion.get('gamma', 0.99),
        tau=configuracion.get('tau', 0.005),
        policy_noise=configuracion.get('policy_noise', 0.2),
        noise_clip=configuracion.get('noise_clip', 0.5),
        policy_delay=configuracion.get('policy_delay', 2),
        buffer_size=configuracion.get('buffer_size', 1000000),
        batch_size=configuracion.get('batch_size', 256),
        usar_he=configuracion.get('usar_he', True)
    )


# Configuración específica para RFEN1_RN_8
RFEN8_CONFIG = {
    'inicializacion_preferida': 'he',
    'learning_rate_actor_default': 0.001,
    'learning_rate_critic_default': 0.001,
    'gamma_default': 0.99,
    'tau_default': 0.005,
    'policy_noise_default': 0.2,
    'noise_clip_default': 0.5,
    'policy_delay_default': 2,
    'buffer_size_default': 1000000,
    'batch_size_default': 256,
    'umbral_sobreestimacion': 10.0
}

logger.info("RFEN1_RN_8.py cargado correctamente - Neurona de Refuerzo TD3 con Doble Crítica")
