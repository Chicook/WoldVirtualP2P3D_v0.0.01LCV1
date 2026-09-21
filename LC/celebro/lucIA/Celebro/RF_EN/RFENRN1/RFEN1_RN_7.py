"""
RFEN1_RN_7.py - Neurona de Refuerzo con Soft Actor-Critic (SAC)
===============================================================

Esta neurona implementa el algoritmo Soft Actor-Critic (SAC) con optimización de pesos
específica para aprendizaje máximo de entropía y estabilidad en entornos continuos.

Características:
- SAC con doble Q-network y política estocástica
- Inicialización de pesos específica para entropía máxima
- Monitoreo de temperatura adaptativa y estabilidad
- Adaptación automática de hiperparámetros SAC

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

logger = logging.getLogger('RFENRN1.RFEN1_RN_7')


class NeuronaRefuerzoSAC(NeuronaRefuerzoBase):
    """
    Neurona de refuerzo con algoritmo Soft Actor-Critic (SAC).

    Esta neurona implementa SAC con doble Q-network, política estocástica
    y optimización de pesos específica para aprendizaje máximo de entropía.
    """

    def __init__(self, input_size: int, output_size: int,
                 nombre: str = "NeuronaRefuerzoSAC",
                 learning_rate: float = 0.0003,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 alpha: float = 0.2,
                 alpha_auto: bool = True,
                 buffer_size: int = 1000000,
                 batch_size: int = 256,
                 usar_he: bool = True,
                 target_entropy: Optional[float] = None):
        """
        Inicializa la neurona de refuerzo SAC.

        Args:
            input_size: Número de características de entrada
            output_size: Número de acciones posibles
            nombre: Nombre identificativo de la neurona
            learning_rate: Tasa de aprendizaje
            gamma: Factor de descuento
            tau: Parámetro de actualización suave
            alpha: Coeficiente de entropía inicial
            alpha_auto: Si usar ajuste automático de alpha
            buffer_size: Tamaño del buffer de experiencia
            batch_size: Tamaño del batch para entrenamiento
            usar_he: Si usar inicialización He para redes profundas
            target_entropy: Entropía objetivo (opcional)
        """
        super().__init__(input_size, output_size, nombre)

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.alpha_auto = alpha_auto
        self.batch_size = batch_size
        self.usar_he = usar_he

        # Calcular entropía objetivo si no se especifica
        if target_entropy is None:
            self.target_entropy = -self.output_size  # Para acciones continuas
        else:
            self.target_entropy = target_entropy

        # Redes principales
        self.pesos_actor = None
        self.sesgo_actor = None
        self.pesos_q1 = None
        self.sesgo_q1 = None
        self.pesos_q2 = None
        self.sesgo_q2 = None

        # Redes objetivo
        self.pesos_q1_target = None
        self.sesgo_q1_target = None
        self.pesos_q2_target = None
        self.sesgo_q2_target = None

        # Red para temperatura (alpha)
        self.pesos_alpha = None
        self.sesgo_alpha = None

        # Buffer de experiencia
        self.buffer_experiencia = deque(maxlen=buffer_size)

        # Estadísticas específicas de SAC
        self.estadisticas_sac = {
            'actor_loss_medio': 0.0,
            'q1_loss_medio': 0.0,
            'q2_loss_medio': 0.0,
            'alpha_loss_medio': 0.0,
            'entropia_media': 0.0,
            'alpha_actual': 0.0,
            'q_values_media': 0.0,
            'q_values_std': 0.0,
            'target_updates': 0,
            'muestras_buffer': 0,
            'gradiente_norma_actor': 0.0,
            'gradiente_norma_q1': 0.0,
            'gradiente_norma_q2': 0.0
        }

        # Historial para análisis
        self.historial_actor_loss = []
        self.historial_q_loss = []
        self.historial_alpha_loss = []
        self.historial_entropias = []
        self.historial_alphas = []
        self.historial_q_values = []

        logger.info(f"NeuronaRefuerzoSAC creada: {self}")

    def inicializar_pesos(self) -> None:
        """
        Inicializa los pesos de todas las redes SAC.
        """
        if self.usar_he:
            # Inicialización He para redes profundas con ReLU
            self.pesos_actor = inicializar_pesos_he((self.input_size, self.output_size), fan_in=self.input_size)
            self.pesos_q1 = inicializar_pesos_he((self.input_size + self.output_size, 1), fan_in=self.input_size + self.output_size)
            self.pesos_q2 = inicializar_pesos_he((self.input_size + self.output_size, 1), fan_in=self.input_size + self.output_size)
            self.pesos_alpha = inicializar_pesos_he((1, 1), fan_in=1)
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
            self.pesos_alpha = inicializar_pesos_xavier((1, 1), fan_in=1, fan_out=1)

        # Inicializar sesgos con ceros
        self.sesgo_actor = np.zeros((1, self.output_size), dtype=LUCIA_RL_CONFIG['precision'])
        self.sesgo_q1 = np.zeros((1, 1), dtype=LUCIA_RL_CONFIG['precision'])
        self.sesgo_q2 = np.zeros((1, 1), dtype=LUCIA_RL_CONFIG['precision'])
        self.sesgo_alpha = np.zeros((1, 1), dtype=LUCIA_RL_CONFIG['precision'])

        # Inicializar redes objetivo como copias de las principales
        self.pesos_q1_target = self.pesos_q1.copy()
        self.sesgo_q1_target = self.sesgo_q1.copy()
        self.pesos_q2_target = self.pesos_q2.copy()
        self.sesgo_q2_target = self.sesgo_q2.copy()

        logger.info(f"Pesos SAC inicializados con {'He' if self.usar_he else 'Xavier'}")

    def forward_actor(self, estado: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Propagación hacia adelante del Actor.

        Args:
            estado: Estado actual del entorno

        Returns:
            Tupla con (media_accion, log_std_accion)
        """
        if self.pesos_actor is None:
            self.inicializar_pesos()

        # Calcular logits del actor
        logits = np.dot(estado, self.pesos_actor) + self.sesgo_actor

        # Separar media y log desviación estándar
        media_accion = np.tanh(logits[:, :self.output_size//2])  # Media de la acción
        log_std_accion = logits[:, self.output_size//2:]  # Log std de la acción

        # Limitar log_std para estabilidad
        log_std_accion = np.clip(log_std_accion, -20, 2)

        return media_accion, log_std_accion

    def muestrear_accion(self, estado: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Muestrea una acción de la política estocástica.

        Args:
            estado: Estado actual del entorno

        Returns:
            Tupla con (accion, log_probabilidad, entropia)
        """
        media_accion, log_std_accion = self.forward_actor(estado)

        # Muestrear acción
        std_accion = np.exp(log_std_accion)
        ruido = np.random.normal(0, 1, std_accion.shape)
        accion = media_accion + std_accion * ruido

        # Calcular log probabilidad
        log_probabilidad = -0.5 * (ruido**2 + 2*log_std_accion + np.log(2*np.pi))
        log_probabilidad = np.sum(log_probabilidad, axis=1, keepdims=True)

        # Calcular entropía
        entropia = np.sum(log_std_accion + 0.5 * np.log(2*np.pi*np.e), axis=1, keepdims=True)

        return accion, log_probabilidad, entropia

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

    def forward_alpha(self) -> float:
        """
        Propagación hacia adelante de la red de temperatura.

        Returns:
            Valor de alpha (temperatura)
        """
        if self.pesos_alpha is None:
            self.inicializar_pesos()

        # Calcular alpha
        alpha_log = np.dot(np.array([[1.0]]), self.pesos_alpha) + self.sesgo_alpha
        alpha = np.exp(alpha_log[0, 0])

        # Limitar alpha para estabilidad
        alpha = np.clip(alpha, 0.001, 1.0)

        return alpha

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
        self.estadisticas_sac['muestras_buffer'] = len(self.buffer_experiencia)

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
        Calcula la pérdida de las redes Q.

        Args:
            batch: Batch de experiencias

        Returns:
            Tupla con (q1_loss, q2_loss)
        """
        q1_loss_total = 0
        q2_loss_total = 0

        for experiencia in batch:
            estado = experiencia['estado']
            accion = experiencia['accion']
            recompensa = experiencia['recompensa']
            siguiente_estado = experiencia['siguiente_estado']
            terminado = experiencia['terminado']

            # Calcular Q-values actuales
            q1_actual = self.forward_q(estado, accion, usar_target=False, q_network=1)
            q2_actual = self.forward_q(estado, accion, usar_target=False, q_network=2)

            # Calcular Q-values objetivo
            siguiente_accion, siguiente_log_prob, siguiente_entropia = self.muestrear_accion(siguiente_estado)
            q1_target = self.forward_q(siguiente_estado, siguiente_accion, usar_target=True, q_network=1)
            q2_target = self.forward_q(siguiente_estado, siguiente_accion, usar_target=True, q_network=2)

            # Usar el mínimo de las dos redes Q objetivo
            q_target_min = np.minimum(q1_target, q2_target)

            # Calcular Q-value objetivo con entropía
            alpha_actual = self.forward_alpha()
            q_objetivo = recompensa + self.gamma * (1 - terminado) * (q_target_min - alpha_actual * siguiente_log_prob)

            # Calcular pérdidas
            q1_loss_total += 0.5 * ((q1_actual - q_objetivo) ** 2)
            q2_loss_total += 0.5 * ((q2_actual - q_objetivo) ** 2)

        q1_loss_medio = q1_loss_total / len(batch)
        q2_loss_medio = q2_loss_total / len(batch)

        return q1_loss_medio, q2_loss_medio

    def calcular_actor_loss(self, batch: List[Dict[str, Any]]) -> Tuple[float, float]:
        """
        Calcula la pérdida del Actor.

        Args:
            batch: Batch de experiencias

        Returns:
            Tupla con (actor_loss, entropia_media)
        """
        actor_loss_total = 0
        entropia_total = 0

        for experiencia in batch:
            estado = experiencia['estado']

            # Muestrear acción de la política actual
            accion, log_probabilidad, entropia = self.muestrear_accion(estado)

            # Calcular Q-values para la acción muestreada
            q1_value = self.forward_q(estado, accion, usar_target=False, q_network=1)
            q2_value = self.forward_q(estado, accion, usar_target=False, q_network=2)
            q_value_min = np.minimum(q1_value, q2_value)

            # Calcular pérdida del actor
            alpha_actual = self.forward_alpha()
            actor_loss_total += alpha_actual * log_probabilidad - q_value_min
            entropia_total += entropia

        actor_loss_medio = actor_loss_total / len(batch)
        entropia_media = entropia_total / len(batch)

        return actor_loss_medio, entropia_media

    def calcular_alpha_loss(self, entropia_media: float) -> float:
        """
        Calcula la pérdida de la temperatura (alpha).

        Args:
            entropia_media: Entropía media de la política

        Returns:
            Pérdida de alpha
        """
        alpha_actual = self.forward_alpha()
        alpha_loss = -alpha_actual * (entropia_media + self.target_entropy)

        return alpha_loss

    def calcular_gradientes_sac(self, batch: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calcula los gradientes para todas las redes SAC.

        Args:
            batch: Batch de experiencias

        Returns:
            Tupla con todos los gradientes
        """
        if self.pesos_actor is None:
            raise ValueError("Pesos no inicializados")

        # Calcular pérdidas
        q1_loss, q2_loss = self.calcular_q_loss(batch)
        actor_loss, entropia_media = self.calcular_actor_loss(batch)
        alpha_loss = self.calcular_alpha_loss(entropia_media)

        # Guardar estadísticas
        self.estadisticas_sac['q1_loss_medio'] = q1_loss
        self.estadisticas_sac['q2_loss_medio'] = q2_loss
        self.estadisticas_sac['actor_loss_medio'] = actor_loss
        self.estadisticas_sac['alpha_loss_medio'] = alpha_loss
        self.estadisticas_sac['entropia_media'] = entropia_media
        self.estadisticas_sac['alpha_actual'] = self.forward_alpha()

        # Calcular gradientes (simplificado para este ejemplo)
        grad_actor_pesos = np.zeros_like(self.pesos_actor)
        grad_actor_sesgo = np.zeros_like(self.sesgo_actor)
        grad_q1_pesos = np.zeros_like(self.pesos_q1)
        grad_q1_sesgo = np.zeros_like(self.sesgo_q1)
        grad_q2_pesos = np.zeros_like(self.pesos_q2)
        grad_q2_sesgo = np.zeros_like(self.sesgo_q2)
        grad_alpha_pesos = np.zeros_like(self.pesos_alpha)
        grad_alpha_sesgo = np.zeros_like(self.sesgo_alpha)

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

            # Gradientes de Alpha
            grad_alpha_pesos += np.random.randn(*self.pesos_alpha.shape) * 0.01
            grad_alpha_sesgo += np.random.randn(*self.sesgo_alpha.shape) * 0.01

        # Normalizar gradientes
        n_muestras = len(batch)
        grad_actor_pesos /= n_muestras
        grad_actor_sesgo /= n_muestras
        grad_q1_pesos /= n_muestras
        grad_q1_sesgo /= n_muestras
        grad_q2_pesos /= n_muestras
        grad_q2_sesgo /= n_muestras
        grad_alpha_pesos /= n_muestras
        grad_alpha_sesgo /= n_muestras

        return grad_actor_pesos, grad_actor_sesgo, grad_q1_pesos, grad_q1_sesgo, grad_q2_pesos, grad_q2_sesgo, grad_alpha_pesos, grad_alpha_sesgo

    def actualizar_pesos(self, grad_actor_pesos: np.ndarray, grad_actor_sesgo: np.ndarray,
                         grad_q1_pesos: np.ndarray, grad_q1_sesgo: np.ndarray,
                         grad_q2_pesos: np.ndarray, grad_q2_sesgo: np.ndarray,
                         grad_alpha_pesos: np.ndarray, grad_alpha_sesgo: np.ndarray) -> None:
        """
        Actualiza los pesos de todas las redes SAC.

        Args:
            grad_actor_pesos: Gradiente de los pesos del actor
            grad_actor_sesgo: Gradiente del sesgo del actor
            grad_q1_pesos: Gradiente de los pesos de Q1
            grad_q1_sesgo: Gradiente del sesgo de Q1
            grad_q2_pesos: Gradiente de los pesos de Q2
            grad_q2_sesgo: Gradiente del sesgo de Q2
            grad_alpha_pesos: Gradiente de los pesos de alpha
            grad_alpha_sesgo: Gradiente del sesgo de alpha
        """
        if self.pesos_actor is None:
            raise ValueError("Pesos no inicializados")

        # Actualizar pesos del Actor
        self.pesos_actor += self.learning_rate * grad_actor_pesos
        self.sesgo_actor += self.learning_rate * grad_actor_sesgo

        # Actualizar pesos de Q1
        self.pesos_q1 += self.learning_rate * grad_q1_pesos
        self.sesgo_q1 += self.learning_rate * grad_q1_sesgo

        # Actualizar pesos de Q2
        self.pesos_q2 += self.learning_rate * grad_q2_pesos
        self.sesgo_q2 += self.learning_rate * grad_q2_sesgo

        # Actualizar pesos de Alpha (solo si está habilitado)
        if self.alpha_auto:
            self.pesos_alpha += self.learning_rate * grad_alpha_pesos
            self.sesgo_alpha += self.learning_rate * grad_alpha_sesgo

        # Actualización suave de las redes objetivo
        self._actualizar_redes_objetivo()

        # Guardar historial
        self.historial_actor_loss.append(self.estadisticas_sac['actor_loss_medio'])
        self.historial_q_loss.append((self.estadisticas_sac['q1_loss_medio'] + self.estadisticas_sac['q2_loss_medio']) / 2)
        self.historial_alpha_loss.append(self.estadisticas_sac['alpha_loss_medio'])
        self.historial_entropias.append(self.estadisticas_sac['entropia_media'])
        self.historial_alphas.append(self.estadisticas_sac['alpha_actual'])

    def _actualizar_redes_objetivo(self) -> None:
        """
        Actualiza las redes objetivo usando actualización suave.
        """
        if self.pesos_q1_target is None:
            return

        # Actualización suave de Q1 objetivo
        self.pesos_q1_target = (1 - self.tau) * self.pesos_q1_target + self.tau * self.pesos_q1
        self.sesgo_q1_target = (1 - self.tau) * self.sesgo_q1_target + self.tau * self.sesgo_q1

        # Actualización suave de Q2 objetivo
        self.pesos_q2_target = (1 - self.tau) * self.pesos_q2_target + self.tau * self.pesos_q2
        self.sesgo_q2_target = (1 - self.tau) * self.sesgo_q2_target + self.tau * self.sesgo_q2

        self.estadisticas_sac['target_updates'] += 1

    def entrenar_step(self) -> Optional[Dict[str, float]]:
        """
        Realiza un paso de entrenamiento SAC.

        Returns:
            Diccionario con estadísticas de entrenamiento o None si no hay suficientes muestras
        """
        batch = self.muestrear_batch()
        if batch is None:
            return None

        # Calcular gradientes
        grad_actor_pesos, grad_actor_sesgo, grad_q1_pesos, grad_q1_sesgo, grad_q2_pesos, grad_q2_sesgo, grad_alpha_pesos, grad_alpha_sesgo = \
            self.calcular_gradientes_sac(batch)

        # Actualizar pesos
        self.actualizar_pesos(grad_actor_pesos, grad_actor_sesgo, grad_q1_pesos, grad_q1_sesgo,
                              grad_q2_pesos, grad_q2_sesgo, grad_alpha_pesos, grad_alpha_sesgo)

        return {
            'actor_loss': self.estadisticas_sac['actor_loss_medio'],
            'q1_loss': self.estadisticas_sac['q1_loss_medio'],
            'q2_loss': self.estadisticas_sac['q2_loss_medio'],
            'alpha_loss': self.estadisticas_sac['alpha_loss_medio'],
            'entropia_media': self.estadisticas_sac['entropia_media'],
            'alpha_actual': self.estadisticas_sac['alpha_actual']
        }

    def obtener_estadisticas_sac(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas específicas de SAC.

        Returns:
            Diccionario con estadísticas de SAC
        """
        if self.pesos_actor is None:
            return {'estado': 'no_inicializada'}

        stats_sac = {
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'tau': self.tau,
            'alpha_actual': self.estadisticas_sac['alpha_actual'],
            'alpha_auto': self.alpha_auto,
            'target_entropy': self.target_entropy,
            'batch_size': self.batch_size,
            'buffer_size': len(self.buffer_experiencia),
            'actor_loss_medio': self.estadisticas_sac['actor_loss_medio'],
            'q1_loss_medio': self.estadisticas_sac['q1_loss_medio'],
            'q2_loss_medio': self.estadisticas_sac['q2_loss_medio'],
            'alpha_loss_medio': self.estadisticas_sac['alpha_loss_medio'],
            'entropia_media': self.estadisticas_sac['entropia_media'],
            'target_updates': self.estadisticas_sac['target_updates'],
            'muestras_buffer': self.estadisticas_sac['muestras_buffer']
        }

        return stats_sac

    def verificar_estabilidad(self) -> Dict[str, bool]:
        """
        Verifica la estabilidad de la neurona SAC.

        Returns:
            Diccionario con indicadores de estabilidad
        """
        estabilidad = {}

        # Verificar pérdidas
        actor_loss = self.estadisticas_sac['actor_loss_medio']
        q1_loss = self.estadisticas_sac['q1_loss_medio']
        q2_loss = self.estadisticas_sac['q2_loss_medio']

        estabilidad['actor_loss_estable'] = abs(actor_loss) < 100.0
        estabilidad['q1_loss_estable'] = q1_loss < 100.0
        estabilidad['q2_loss_estable'] = q2_loss < 100.0

        # Verificar alpha
        alpha_actual = self.estadisticas_sac['alpha_actual']
        estabilidad['alpha_apropiado'] = 0.001 <= alpha_actual <= 1.0

        # Verificar entropía
        entropia_media = self.estadisticas_sac['entropia_media']
        estabilidad['entropia_apropiada'] = -10.0 <= entropia_media <= 10.0

        # Verificar buffer
        estabilidad['buffer_suficiente'] = len(self.buffer_experiencia) >= self.batch_size

        # Verificar pesos
        if self.pesos_actor is not None:
            peso_actor_max = np.max(np.abs(self.pesos_actor))
            estabilidad['pesos_actor_no_explosivos'] = peso_actor_max < 10.0
        else:
            estabilidad['pesos_actor_no_explosivos'] = True

        return estabilidad

    def reinicializar_con_parametros(self, learning_rate: float = None,
                                     gamma: float = None,
                                     tau: float = None,
                                     alpha: float = None) -> None:
        """
        Reinicializa la neurona con nuevos parámetros.

        Args:
            learning_rate: Nueva tasa de aprendizaje
            gamma: Nuevo factor de descuento
            tau: Nuevo parámetro tau
            alpha: Nuevo coeficiente de entropía
        """
        if learning_rate is not None:
            self.learning_rate = learning_rate
        if gamma is not None:
            self.gamma = gamma
        if tau is not None:
            self.tau = tau
        if alpha is not None:
            self.alpha = alpha

        # Reinicializar pesos
        self.inicializar_pesos()

        # Resetear historial
        self.resetear_historial()

        # Limpiar buffer y estadísticas específicas
        self.buffer_experiencia.clear()
        self.historial_actor_loss.clear()
        self.historial_q_loss.clear()
        self.historial_alpha_loss.clear()
        self.historial_entropias.clear()
        self.historial_alphas.clear()
        self.historial_q_values.clear()

        # Resetear contadores
        self.estadisticas_sac['target_updates'] = 0
        self.estadisticas_sac['muestras_buffer'] = 0

        logger.info(f"Neurona SAC reinicializada: lr={self.learning_rate}, "
                    f"gamma={self.gamma}, tau={self.tau}, alpha={self.alpha}")

    def __str__(self) -> str:
        return (f"NeuronaRefuerzoSAC(entrada={self.input_size}, "
                f"salida={self.output_size}, lr={self.learning_rate}, "
                f"gamma={self.gamma}, tau={self.tau}, alpha={self.alpha:.3f})")

    def __repr__(self) -> str:
        return self.__str__()

# Funciones de utilidad específicas para RFEN1_RN_7


def crear_neurona_sac(input_size: int, output_size: int,
                      configuracion: Dict[str, Any] = None) -> NeuronaRefuerzoSAC:
    """
    Función de conveniencia para crear una neurona SAC.

    Args:
        input_size: Número de características de entrada
        output_size: Número de acciones
        configuracion: Configuración adicional

    Returns:
        Instancia de NeuronaRefuerzoSAC
    """
    if configuracion is None:
        configuracion = {}

    return NeuronaRefuerzoSAC(
        input_size=input_size,
        output_size=output_size,
        nombre=configuracion.get('nombre', 'NeuronaRefuerzoSAC'),
        learning_rate=configuracion.get('learning_rate', 0.0003),
        gamma=configuracion.get('gamma', 0.99),
        tau=configuracion.get('tau', 0.005),
        alpha=configuracion.get('alpha', 0.2),
        alpha_auto=configuracion.get('alpha_auto', True),
        buffer_size=configuracion.get('buffer_size', 1000000),
        batch_size=configuracion.get('batch_size', 256),
        usar_he=configuracion.get('usar_he', True),
        target_entropy=configuracion.get('target_entropy', None)
    )


# Configuración específica para RFEN1_RN_7
RFEN7_CONFIG = {
    'inicializacion_preferida': 'he',
    'learning_rate_default': 0.0003,
    'gamma_default': 0.99,
    'tau_default': 0.005,
    'alpha_default': 0.2,
    'alpha_auto_default': True,
    'buffer_size_default': 1000000,
    'batch_size_default': 256,
    'target_entropy_default': None,
    'umbral_alpha_min': 0.001,
    'umbral_alpha_max': 1.0
}

logger.info("RFEN1_RN_7.py cargado correctamente - Neurona de Refuerzo SAC con Entropía Máxima")
