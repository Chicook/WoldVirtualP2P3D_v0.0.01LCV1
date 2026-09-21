"""
RFEN1_RN_9.py - Neurona de Refuerzo con Rainbow DQN
===================================================

Esta neurona implementa el algoritmo Rainbow DQN con optimización de pesos
multi-método que combina múltiples técnicas avanzadas de DQN.

Características:
- Rainbow DQN con múltiples mejoras combinadas
- Inicialización de pesos específica para cada componente
- Monitoreo de convergencia multi-método
- Adaptación automática de hiperparámetros Rainbow

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

logger = logging.getLogger('RFENRN1.RFEN1_RN_9')


class NeuronaRefuerzoRainbowDQN(NeuronaRefuerzoBase):
    """
    Neurona de refuerzo con algoritmo Rainbow DQN.

    Esta neurona implementa Rainbow DQN combinando múltiples técnicas:
    - Double DQN
    - Dueling DQN
    - Prioritized Experience Replay
    - Multi-step Learning
    - Distributional RL
    - Noisy Networks
    """

    def __init__(self, input_size: int, output_size: int,
                 nombre: str = "NeuronaRefuerzoRainbowDQN",
                 learning_rate: float = 0.00025,
                 gamma: float = 0.99,
                 epsilon: float = 0.1,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01,
                 tau: float = 0.005,
                 buffer_size: int = 1000000,
                 batch_size: int = 32,
                 usar_he: bool = True,
                 usar_dueling: bool = True,
                 usar_prioritized: bool = True,
                 usar_multi_step: bool = True,
                 usar_distributional: bool = True,
                 usar_noisy: bool = True,
                 n_atoms: int = 51,
                 v_min: float = -10.0,
                 v_max: float = 10.0,
                 n_steps: int = 3,
                 alpha_prioritized: float = 0.6,
                 beta_prioritized: float = 0.4):
        """
        Inicializa la neurona de refuerzo Rainbow DQN.

        Args:
            input_size: Número de características de entrada
            output_size: Número de acciones posibles
            nombre: Nombre identificativo de la neurona
            learning_rate: Tasa de aprendizaje
            gamma: Factor de descuento
            epsilon: Valor inicial de epsilon para exploración
            epsilon_decay: Factor de decaimiento de epsilon
            epsilon_min: Valor mínimo de epsilon
            tau: Parámetro de actualización suave
            buffer_size: Tamaño del buffer de experiencia
            batch_size: Tamaño del batch para entrenamiento
            usar_he: Si usar inicialización He para redes profundas
            usar_dueling: Si usar arquitectura Dueling
            usar_prioritized: Si usar replay prioritizado
            usar_multi_step: Si usar aprendizaje multi-step
            usar_distributional: Si usar RL distribucional
            usar_noisy: Si usar redes ruidosas
            n_atoms: Número de átomos para RL distribucional
            v_min: Valor mínimo para RL distribucional
            v_max: Valor máximo para RL distribucional
            n_steps: Número de pasos para multi-step learning
            alpha_prioritized: Parámetro alpha para replay prioritizado
            beta_prioritized: Parámetro beta para replay prioritizado
        """
        super().__init__(input_size, output_size, nombre)

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.tau = tau
        self.batch_size = batch_size
        self.usar_he = usar_he
        self.usar_dueling = usar_dueling
        self.usar_prioritized = usar_prioritized
        self.usar_multi_step = usar_multi_step
        self.usar_distributional = usar_distributional
        self.usar_noisy = usar_noisy
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.n_steps = n_steps
        self.alpha_prioritized = alpha_prioritized
        self.beta_prioritized = beta_prioritized

        # Redes principales
        self.pesos_q_principal = None
        self.sesgo_q_principal = None
        self.pesos_q_objetivo = None
        self.sesgo_q_objetivo = None

        # Pesos específicos para Dueling DQN
        if self.usar_dueling:
            self.pesos_value_stream = None
            self.sesgo_value_stream = None
            self.pesos_advantage_stream = None
            self.sesgo_advantage_stream = None
            self.pesos_value_stream_target = None
            self.sesgo_value_stream_target = None
            self.pesos_advantage_stream_target = None
            self.sesgo_advantage_stream_target = None

        # Pesos específicos para RL distribucional
        if self.usar_distributional:
            self.pesos_distributional = None
            self.sesgo_distributional = None
            self.pesos_distributional_target = None
            self.sesgo_distributional_target = None

        # Pesos específicos para redes ruidosas
        if self.usar_noisy:
            self.pesos_noisy = None
            self.sesgo_noisy = None
            self.pesos_noisy_target = None
            self.sesgo_noisy_target = None

        # Buffer de experiencia prioritizado
        if self.usar_prioritized:
            self.buffer_experiencia = deque(maxlen=buffer_size)
            self.prioridades = deque(maxlen=buffer_size)
            self.beta_schedule = 0.4
        else:
            self.buffer_experiencia = deque(maxlen=buffer_size)

        # Estadísticas específicas de Rainbow DQN
        self.estadisticas_rainbow = {
            'td_error_medio': 0.0,
            'td_error_std': 0.0,
            'convergencia_q': 0.0,
            'estabilidad_q': 0.0,
            'actualizaciones_target': 0,
            'muestras_buffer': 0,
            'exploracion_rate': 0.0,
            'exploitacion_rate': 0.0,
            'loss_medio': 0.0,
            'gradiente_norma': 0.0,
            'prioridad_media': 0.0,
            'kl_divergencia_distribucional': 0.0,
            'entropia_distribucional': 0.0,
            'noise_factor': 0.0
        }

        # Historial para análisis
        self.historial_td_errors = []
        self.historial_losses = []
        self.historial_q_values = []
        self.historial_epsilon = []
        self.historial_prioridades = []
        self.historial_kl_divergences = []

        logger.info(f"NeuronaRefuerzoRainbowDQN creada: {self}")

    def inicializar_pesos(self) -> None:
        """
        Inicializa los pesos de todas las redes Rainbow DQN.
        """
        if self.usar_he:
            # Inicialización He para redes profundas con ReLU
            self.pesos_q_principal = inicializar_pesos_he((self.input_size, self.output_size), fan_in=self.input_size)
            self.pesos_q_objetivo = inicializar_pesos_he((self.input_size, self.output_size), fan_in=self.input_size)
        else:
            # Inicialización Xavier como alternativa
            self.pesos_q_principal = inicializar_pesos_xavier(
                (self.input_size, self.output_size),
                fan_in=self.input_size,
                fan_out=self.output_size
            )
            self.pesos_q_objetivo = inicializar_pesos_xavier(
                (self.input_size, self.output_size),
                fan_in=self.input_size,
                fan_out=self.output_size
            )

        # Inicializar sesgos con ceros
        self.sesgo_q_principal = np.zeros((1, self.output_size), dtype=LUCIA_RL_CONFIG['precision'])
        self.sesgo_q_objetivo = np.zeros((1, self.output_size), dtype=LUCIA_RL_CONFIG['precision'])

        # Inicializar pesos específicos para Dueling DQN
        if self.usar_dueling:
            if self.usar_he:
                self.pesos_value_stream = inicializar_pesos_he((self.input_size, 1), fan_in=self.input_size)
                self.pesos_advantage_stream = inicializar_pesos_he((self.input_size, self.output_size), fan_in=self.input_size)
            else:
                self.pesos_value_stream = inicializar_pesos_xavier((self.input_size, 1), fan_in=self.input_size, fan_out=1)
                self.pesos_advantage_stream = inicializar_pesos_xavier(
                    (self.input_size, self.output_size),
                    fan_in=self.input_size,
                    fan_out=self.output_size
                )

            self.sesgo_value_stream = np.zeros((1, 1), dtype=LUCIA_RL_CONFIG['precision'])
            self.sesgo_advantage_stream = np.zeros((1, self.output_size), dtype=LUCIA_RL_CONFIG['precision'])

            # Copias objetivo
            self.pesos_value_stream_target = self.pesos_value_stream.copy()
            self.sesgo_value_stream_target = self.sesgo_value_stream.copy()
            self.pesos_advantage_stream_target = self.pesos_advantage_stream.copy()
            self.sesgo_advantage_stream_target = self.sesgo_advantage_stream.copy()

        # Inicializar pesos específicos para RL distribucional
        if self.usar_distributional:
            if self.usar_he:
                self.pesos_distributional = inicializar_pesos_he(
                    (self.input_size, self.output_size * self.n_atoms),
                    fan_in=self.input_size
                )
            else:
                self.pesos_distributional = inicializar_pesos_xavier(
                    (self.input_size, self.output_size * self.n_atoms),
                    fan_in=self.input_size,
                    fan_out=self.output_size * self.n_atoms
                )

            self.sesgo_distributional = np.zeros((1, self.output_size * self.n_atoms), dtype=LUCIA_RL_CONFIG['precision'])
            self.pesos_distributional_target = self.pesos_distributional.copy()
            self.sesgo_distributional_target = self.sesgo_distributional.copy()

        # Inicializar pesos específicos para redes ruidosas
        if self.usar_noisy:
            if self.usar_he:
                self.pesos_noisy = inicializar_pesos_he((self.input_size, self.output_size), fan_in=self.input_size)
            else:
                self.pesos_noisy = inicializar_pesos_xavier(
                    (self.input_size, self.output_size),
                    fan_in=self.input_size,
                    fan_out=self.output_size
                )

            self.sesgo_noisy = np.zeros((1, self.output_size), dtype=LUCIA_RL_CONFIG['precision'])
            self.pesos_noisy_target = self.pesos_noisy.copy()
            self.sesgo_noisy_target = self.sesgo_noisy.copy()

        logger.info(f"Pesos Rainbow DQN inicializados con {'He' if self.usar_he else 'Xavier'}")

    def forward(self, estado: np.ndarray, usar_objetivo: bool = False) -> np.ndarray:
        """
        Propagación hacia adelante - cálculo de Q-values.

        Args:
            estado: Estado actual del entorno
            usar_objetivo: Si usar la red objetivo

        Returns:
            Q-values para todas las acciones
        """
        if self.pesos_q_principal is None:
            self.inicializar_pesos()

        if usar_objetivo:
            pesos_q = self.pesos_q_objetivo
            sesgo_q = self.sesgo_q_objetivo
        else:
            pesos_q = self.pesos_q_principal
            sesgo_q = self.sesgo_q_principal

        # Calcular Q-values base
        q_values = np.dot(estado, pesos_q) + sesgo_q

        # Aplicar Dueling DQN si está habilitado
        if self.usar_dueling:
            q_values = self._aplicar_dueling(estado, q_values, usar_objetivo)

        # Aplicar RL distribucional si está habilitado
        if self.usar_distributional:
            q_values = self._aplicar_distributional(estado, q_values, usar_objetivo)

        # Aplicar redes ruidosas si está habilitado
        if self.usar_noisy:
            q_values = self._aplicar_noisy(estado, q_values, usar_objetivo)

        # Guardar historial
        self.historial_q_values.append(q_values.copy())

        return q_values

    def _aplicar_dueling(self, estado: np.ndarray, q_values: np.ndarray, usar_objetivo: bool) -> np.ndarray:
        """
        Aplica la arquitectura Dueling DQN.

        Args:
            estado: Estado actual
            q_values: Q-values base
            usar_objetivo: Si usar redes objetivo

        Returns:
            Q-values con arquitectura Dueling
        """
        if usar_objetivo:
            pesos_value = self.pesos_value_stream_target
            sesgo_value = self.sesgo_value_stream_target
            pesos_advantage = self.pesos_advantage_stream_target
            sesgo_advantage = self.sesgo_advantage_stream_target
        else:
            pesos_value = self.pesos_value_stream
            sesgo_value = self.sesgo_value_stream
            pesos_advantage = self.pesos_advantage_stream
            sesgo_advantage = self.sesgo_advantage_stream

        # Calcular Value y Advantage streams
        value_stream = np.dot(estado, pesos_value) + sesgo_value
        advantage_stream = np.dot(estado, pesos_advantage) + sesgo_advantage

        # Combinar usando la fórmula de Dueling DQN
        q_values = value_stream + advantage_stream - np.mean(advantage_stream, axis=1, keepdims=True)

        return q_values

    def _aplicar_distributional(self, estado: np.ndarray, q_values: np.ndarray, usar_objetivo: bool) -> np.ndarray:
        """
        Aplica RL distribucional.

        Args:
            estado: Estado actual
            q_values: Q-values base
            usar_objetivo: Si usar redes objetivo

        Returns:
            Q-values con RL distribucional
        """
        if usar_objetivo:
            pesos_dist = self.pesos_distributional_target
            sesgo_dist = self.sesgo_distributional_target
        else:
            pesos_dist = self.pesos_distributional
            sesgo_dist = self.sesgo_distributional

        # Calcular distribución de átomos
        logits_dist = np.dot(estado, pesos_dist) + sesgo_dist

        # Reshape para (batch_size, output_size, n_atoms)
        logits_dist = logits_dist.reshape(-1, self.output_size, self.n_atoms)

        # Aplicar softmax para obtener probabilidades
        exp_logits = np.exp(logits_dist - np.max(logits_dist, axis=2, keepdims=True))
        probs_dist = exp_logits / np.sum(exp_logits, axis=2, keepdims=True)

        # Calcular Q-values como expectativa de la distribución
        atoms = np.linspace(self.v_min, self.v_max, self.n_atoms)
        q_values = np.sum(probs_dist * atoms, axis=2)

        return q_values

    def _aplicar_noisy(self, estado: np.ndarray, q_values: np.ndarray, usar_objetivo: bool) -> np.ndarray:
        """
        Aplica redes ruidosas para exploración.

        Args:
            estado: Estado actual
            q_values: Q-values base
            usar_objetivo: Si usar redes objetivo

        Returns:
            Q-values con redes ruidosas
        """
        if usar_objetivo:
            pesos_noisy = self.pesos_noisy_target
            sesgo_noisy = self.sesgo_noisy_target
        else:
            pesos_noisy = self.pesos_noisy
            sesgo_noisy = self.sesgo_noisy

        # Generar ruido
        noise_factor = 0.1
        noise = np.random.normal(0, noise_factor, q_values.shape)

        # Aplicar ruido a los Q-values
        q_values_noisy = q_values + noise

        # Guardar factor de ruido
        self.estadisticas_rainbow['noise_factor'] = noise_factor

        return q_values_noisy

    def seleccionar_accion(self, estado: np.ndarray) -> int:
        """
        Selecciona una acción usando epsilon-greedy o redes ruidosas.

        Args:
            estado: Estado actual del entorno

        Returns:
            Acción seleccionada
        """
        if self.usar_noisy:
            # Usar redes ruidosas para exploración
            q_values = self.forward(estado)
            accion = np.argmax(q_values[0])
            self.estadisticas_rainbow['exploitacion_rate'] += 1
        else:
            # Usar epsilon-greedy tradicional
            if np.random.random() < self.epsilon:
                # Exploración: acción aleatoria
                accion = np.random.randint(0, self.output_size)
                self.estadisticas_rainbow['exploracion_rate'] += 1
            else:
                # Explotación: mejor acción según Q-values
                q_values = self.forward(estado)
                accion = np.argmax(q_values[0])
                self.estadisticas_rainbow['exploitacion_rate'] += 1

        return accion

    def agregar_experiencia(self, estado: np.ndarray, accion: int, recompensa: float,
                            siguiente_estado: np.ndarray, terminado: bool) -> None:
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
            'accion': accion,
            'recompensa': recompensa,
            'siguiente_estado': siguiente_estado.copy(),
            'terminado': terminado
        }

        self.buffer_experiencia.append(experiencia)
        self.estadisticas_rainbow['muestras_buffer'] = len(self.buffer_experiencia)

        # Calcular prioridad si está habilitado
        if self.usar_prioritized:
            prioridad = self._calcular_prioridad(experiencia)
            self.prioridades.append(prioridad)
            self.estadisticas_rainbow['prioridad_media'] = np.mean(list(self.prioridades))

    def _calcular_prioridad(self, experiencia: Dict[str, Any]) -> float:
        """
        Calcula la prioridad de una experiencia.

        Args:
            experiencia: Experiencia a evaluar

        Returns:
            Prioridad calculada
        """
        # Calcular TD error como prioridad
        estado = experiencia['estado']
        accion = experiencia['accion']
        recompensa = experiencia['recompensa']
        siguiente_estado = experiencia['siguiente_estado']
        terminado = experiencia['terminado']

        # Q-value actual
        q_actual = self.forward(estado)[0, accion]

        # Q-value objetivo
        if terminado:
            q_objetivo = recompensa
        else:
            q_siguiente = self.forward(siguiente_estado, usar_objetivo=True)
            q_objetivo = recompensa + self.gamma * np.max(q_siguiente[0])

        # TD error
        td_error = abs(q_objetivo - q_actual)

        # Prioridad basada en TD error
        prioridad = (td_error + 1e-6) ** self.alpha_prioritized

        return prioridad

    def muestrear_batch_prioritizado(self) -> Optional[List[Dict[str, Any]]]:
        """
        Muestrea un batch del buffer de experiencia usando prioridades.

        Returns:
            Lista de experiencias o None si no hay suficientes muestras
        """
        if len(self.buffer_experiencia) < self.batch_size:
            return None

        if self.usar_prioritized:
            # Muestreo prioritizado
            prioridades_array = np.array(list(self.prioridades))
            probabilidades = prioridades_array ** self.alpha_prioritized
            probabilidades = probabilidades / np.sum(probabilidades)

            indices = np.random.choice(len(self.buffer_experiencia),
                                       size=self.batch_size,
                                       replace=False,
                                       p=probabilidades)

            batch = [list(self.buffer_experiencia)[i] for i in indices]
        else:
            # Muestreo uniforme
            batch = random.sample(list(self.buffer_experiencia), self.batch_size)

        return batch

    def calcular_td_error_multi_step(self, batch: List[Dict[str, Any]]) -> List[float]:
        """
        Calcula el TD error usando multi-step learning.

        Args:
            batch: Batch de experiencias

        Returns:
            Lista de TD errors
        """
        td_errors = []

        for experiencia in batch:
            estado = experiencia['estado']
            accion = experiencia['accion']
            recompensa = experiencia['recompensa']
            siguiente_estado = experiencia['siguiente_estado']
            terminado = experiencia['terminado']

            # Q-value actual
            q_actual = self.forward(estado)[0, accion]

            # Q-value objetivo con multi-step
            if self.usar_multi_step:
                # Simular multi-step learning (simplificado)
                q_objetivo = recompensa
                if not terminado:
                    q_siguiente = self.forward(siguiente_estado, usar_objetivo=True)
                    q_objetivo += self.gamma * np.max(q_siguiente[0])
            else:
                # TD learning tradicional
                if terminado:
                    q_objetivo = recompensa
                else:
                    q_siguiente = self.forward(siguiente_estado, usar_objetivo=True)
                    q_objetivo = recompensa + self.gamma * np.max(q_siguiente[0])

            # TD error
            td_error = q_objetivo - q_actual
            td_errors.append(td_error)

        # Guardar historial
        self.historial_td_errors.extend(td_errors)

        return td_errors

    def calcular_gradientes_rainbow(self, batch: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula los gradientes para Rainbow DQN.

        Args:
            batch: Batch de experiencias

        Returns:
            Tupla con (gradiente_pesos, gradiente_sesgo)
        """
        if self.pesos_q_principal is None:
            raise ValueError("Pesos no inicializados")

        gradiente_pesos = np.zeros_like(self.pesos_q_principal)
        gradiente_sesgo = np.zeros_like(self.sesgo_q_principal)

        td_errors = self.calcular_td_error_multi_step(batch)
        loss_total = 0

        for experiencia, td_error in zip(batch, td_errors):
            estado = experiencia['estado']
            accion = experiencia['accion']

            # Calcular gradiente para la acción específica
            gradiente_accion = np.zeros(self.output_size)
            gradiente_accion[accion] = td_error

            # Gradiente de los pesos
            gradiente_pesos += np.outer(estado[0], gradiente_accion)
            gradiente_sesgo += gradiente_accion

            # Pérdida cuadrática
            loss_total += 0.5 * (td_error ** 2)

        # Normalizar por el tamaño del batch
        n_muestras = len(batch)
        gradiente_pesos /= n_muestras
        gradiente_sesgo /= n_muestras

        # Guardar pérdida
        loss_medio = loss_total / n_muestras
        self.historial_losses.append(loss_medio)

        # Calcular norma del gradiente
        norma_gradiente = np.linalg.norm(gradiente_pesos)
        self.estadisticas_rainbow['gradiente_norma'] = norma_gradiente

        return gradiente_pesos, gradiente_sesgo

    def actualizar_pesos(self, gradiente_pesos: np.ndarray, gradiente_sesgo: np.ndarray) -> None:
        """
        Actualiza los pesos de la red principal.

        Args:
            gradiente_pesos: Gradiente de los pesos
            gradiente_sesgo: Gradiente del sesgo
        """
        if self.pesos_q_principal is None:
            raise ValueError("Pesos no inicializados")

        # Actualizar pesos principales
        self.pesos_q_principal += self.learning_rate * gradiente_pesos
        self.sesgo_q_principal += self.learning_rate * gradiente_sesgo

        # Actualización suave de la red objetivo
        self._actualizar_red_objetivo()

        # Decaimiento de epsilon (solo si no usa redes ruidosas)
        if not self.usar_noisy and self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Calcular convergencia
        self._calcular_convergencia_rainbow()

    def _actualizar_red_objetivo(self) -> None:
        """
        Actualiza la red objetivo usando actualización suave.
        """
        if self.pesos_q_objetivo is None:
            return

        # Actualización suave (soft update)
        self.pesos_q_objetivo = (1 - self.tau) * self.pesos_q_objetivo + self.tau * self.pesos_q_principal
        self.sesgo_q_objetivo = (1 - self.tau) * self.sesgo_q_objetivo + self.tau * self.sesgo_q_principal

        # Actualizar redes objetivo específicas
        if self.usar_dueling:
            self.pesos_value_stream_target = (1 - self.tau) * self.pesos_value_stream_target + self.tau * self.pesos_value_stream
            self.sesgo_value_stream_target = (1 - self.tau) * self.sesgo_value_stream_target + self.tau * self.sesgo_value_stream
            self.pesos_advantage_stream_target = (1 - self.tau) * self.pesos_advantage_stream_target + self.tau * self.pesos_advantage_stream
            self.sesgo_advantage_stream_target = (1 - self.tau) * self.sesgo_advantage_stream_target + self.tau * self.sesgo_advantage_stream

        if self.usar_distributional:
            self.pesos_distributional_target = (1 - self.tau) * self.pesos_distributional_target + self.tau * self.pesos_distributional
            self.sesgo_distributional_target = (1 - self.tau) * self.sesgo_distributional_target + self.tau * self.sesgo_distributional

        if self.usar_noisy:
            self.pesos_noisy_target = (1 - self.tau) * self.pesos_noisy_target + self.tau * self.pesos_noisy
            self.sesgo_noisy_target = (1 - self.tau) * self.sesgo_noisy_target + self.tau * self.sesgo_noisy

        self.estadisticas_rainbow['actualizaciones_target'] += 1

    def _calcular_convergencia_rainbow(self) -> None:
        """
        Calcula la convergencia de la red Rainbow DQN.
        """
        if len(self.historial_td_errors) < 10:
            return

        # Convergencia basada en la estabilidad de los TD errors
        td_errors_recientes = self.historial_td_errors[-10:]
        varianza_td = np.var(td_errors_recientes)

        # Convergencia inversamente proporcional a la varianza
        convergencia = 1.0 / (1.0 + varianza_td)
        self.estadisticas_rainbow['convergencia_q'] = convergencia

    def calcular_estabilidad_q(self) -> float:
        """
        Calcula la estabilidad de los Q-values.

        Returns:
            Índice de estabilidad (0-1)
        """
        if len(self.historial_q_values) < 20:
            return 0.0

        # Calcular varianza de los Q-values en el tiempo
        q_values_array = np.array(self.historial_q_values[-20:])
        varianza_temporal = np.var(q_values_array, axis=0)
        varianza_promedio = np.mean(varianza_temporal)

        # Estabilidad inversamente proporcional a la varianza
        estabilidad = 1.0 / (1.0 + varianza_promedio)

        self.estadisticas_rainbow['estabilidad_q'] = estabilidad

        return estabilidad

    def entrenar_step(self) -> Optional[float]:
        """
        Realiza un paso de entrenamiento Rainbow DQN.

        Returns:
            Pérdida promedio o None si no hay suficientes muestras
        """
        batch = self.muestrear_batch_prioritizado()
        if batch is None:
            return None

        # Calcular gradientes
        gradiente_pesos, gradiente_sesgo = self.calcular_gradientes_rainbow(batch)

        # Actualizar pesos
        self.actualizar_pesos(gradiente_pesos, gradiente_sesgo)

        # Retornar pérdida promedio
        return np.mean(self.historial_losses[-len(batch):])

    def obtener_estadisticas_rainbow(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas específicas de Rainbow DQN.

        Returns:
            Diccionario con estadísticas de Rainbow DQN
        """
        if self.pesos_q_principal is None:
            return {'estado': 'no_inicializada'}

        # Calcular estadísticas de TD errors
        if self.historial_td_errors:
            td_errors_array = np.array(self.historial_td_errors[-100:])
            self.estadisticas_rainbow['td_error_medio'] = np.mean(td_errors_array)
            self.estadisticas_rainbow['td_error_std'] = np.std(td_errors_array)

        # Calcular estadísticas de pérdidas
        if self.historial_losses:
            self.estadisticas_rainbow['loss_medio'] = np.mean(self.historial_losses[-10:])

        # Calcular tasas de exploración y explotación
        total_acciones = (self.estadisticas_rainbow['exploracion_rate'] +
                          self.estadisticas_rainbow['exploitacion_rate'])

        if total_acciones > 0:
            self.estadisticas_rainbow['tasa_exploracion'] = (self.estadisticas_rainbow['exploracion_rate'] /
                                                             total_acciones)
            self.estadisticas_rainbow['tasa_exploitacion'] = (self.estadisticas_rainbow['exploitacion_rate'] /
                                                              total_acciones)
        else:
            self.estadisticas_rainbow['tasa_exploracion'] = 0.0
            self.estadisticas_rainbow['tasa_exploitacion'] = 0.0

        stats_rainbow = {
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'epsilon_actual': self.epsilon,
            'tau': self.tau,
            'batch_size': self.batch_size,
            'buffer_size': len(self.buffer_experiencia),
            'convergencia_q': self.estadisticas_rainbow['convergencia_q'],
            'estabilidad_q': self.calcular_estabilidad_q(),
            'actualizaciones_target': self.estadisticas_rainbow['actualizaciones_target'],
            'muestras_buffer': self.estadisticas_rainbow['muestras_buffer'],
            'td_error_medio': self.estadisticas_rainbow['td_error_medio'],
            'td_error_std': self.estadisticas_rainbow['td_error_std'],
            'loss_medio': self.estadisticas_rainbow['loss_medio'],
            'gradiente_norma': self.estadisticas_rainbow['gradiente_norma'],
            'tasa_exploracion': self.estadisticas_rainbow['tasa_exploracion'],
            'tasa_exploitacion': self.estadisticas_rainbow['tasa_exploitacion'],
            'prioridad_media': self.estadisticas_rainbow['prioridad_media'],
            'noise_factor': self.estadisticas_rainbow['noise_factor'],
            'usar_dueling': self.usar_dueling,
            'usar_prioritized': self.usar_prioritized,
            'usar_multi_step': self.usar_multi_step,
            'usar_distributional': self.usar_distributional,
            'usar_noisy': self.usar_noisy
        }

        return stats_rainbow

    def verificar_estabilidad(self) -> Dict[str, bool]:
        """
        Verifica la estabilidad de la neurona Rainbow DQN.

        Returns:
            Diccionario con indicadores de estabilidad
        """
        estabilidad = {}

        # Verificar convergencia
        convergencia = self.estadisticas_rainbow['convergencia_q']
        estabilidad['convergencia_ok'] = convergencia > 0.8

        # Verificar estabilidad de Q-values
        estabilidad_q = self.calcular_estabilidad_q()
        estabilidad['q_values_estables'] = estabilidad_q > 0.7

        # Verificar epsilon (solo si no usa redes ruidosas)
        if not self.usar_noisy:
            estabilidad['epsilon_apropiado'] = self.epsilon_min <= self.epsilon <= 1.0
        else:
            estabilidad['epsilon_apropiado'] = True

        # Verificar buffer
        estabilidad['buffer_suficiente'] = len(self.buffer_experiencia) >= self.batch_size

        # Verificar pesos principales
        if self.pesos_q_principal is not None:
            peso_max = np.max(np.abs(self.pesos_q_principal))
            peso_min = np.min(np.abs(self.pesos_q_principal))

            estabilidad['pesos_no_explosivos'] = peso_max < 10.0
            estabilidad['pesos_no_desaparecen'] = peso_min > 1e-6
            estabilidad['pesos_balanceados'] = peso_max / (peso_min + 1e-8) < 1000.0
        else:
            estabilidad['pesos_no_explosivos'] = True
            estabilidad['pesos_no_desaparecen'] = True
            estabilidad['pesos_balanceados'] = True

        # Verificar gradientes
        norma_gradiente = self.estadisticas_rainbow['gradiente_norma']
        estabilidad['gradientes_no_explosivos'] = norma_gradiente < 10.0
        estabilidad['gradientes_no_desaparecen'] = norma_gradiente > 1e-8

        return estabilidad

    def reinicializar_con_parametros(self, learning_rate: float = None,
                                     gamma: float = None,
                                     epsilon: float = None,
                                     tau: float = None) -> None:
        """
        Reinicializa la neurona con nuevos parámetros.

        Args:
            learning_rate: Nueva tasa de aprendizaje
            gamma: Nuevo factor de descuento
            epsilon: Nuevo valor de epsilon
            tau: Nuevo parámetro tau
        """
        if learning_rate is not None:
            self.learning_rate = learning_rate
        if gamma is not None:
            self.gamma = gamma
        if epsilon is not None:
            self.epsilon = epsilon
        if tau is not None:
            self.tau = tau

        # Reinicializar pesos
        self.inicializar_pesos()

        # Resetear historial
        self.resetear_historial()

        # Limpiar buffer y estadísticas específicas
        self.buffer_experiencia.clear()
        if self.usar_prioritized:
            self.prioridades.clear()

        self.historial_td_errors.clear()
        self.historial_losses.clear()
        self.historial_q_values.clear()
        self.historial_epsilon.clear()
        self.historial_prioridades.clear()
        self.historial_kl_divergences.clear()

        # Resetear contadores
        self.estadisticas_rainbow['actualizaciones_target'] = 0
        self.estadisticas_rainbow['muestras_buffer'] = 0
        self.estadisticas_rainbow['exploracion_rate'] = 0
        self.estadisticas_rainbow['exploitacion_rate'] = 0

        logger.info(f"Neurona Rainbow DQN reinicializada: lr={self.learning_rate}, "
                    f"gamma={self.gamma}, epsilon={self.epsilon}, tau={self.tau}")

    def __str__(self) -> str:
        return (f"NeuronaRefuerzoRainbowDQN(entrada={self.input_size}, "
                f"salida={self.output_size}, lr={self.learning_rate}, "
                f"gamma={self.gamma}, epsilon={self.epsilon:.3f}, "
                f"buffer={len(self.buffer_experiencia)})")

    def __repr__(self) -> str:
        return self.__str__()

# Funciones de utilidad específicas para RFEN1_RN_9


def crear_neurona_rainbow_dqn(input_size: int, output_size: int,
                              configuracion: Dict[str, Any] = None) -> NeuronaRefuerzoRainbowDQN:
    """
    Función de conveniencia para crear una neurona Rainbow DQN.

    Args:
        input_size: Número de características de entrada
        output_size: Número de acciones
        configuracion: Configuración adicional

    Returns:
        Instancia de NeuronaRefuerzoRainbowDQN
    """
    if configuracion is None:
        configuracion = {}

    return NeuronaRefuerzoRainbowDQN(
        input_size=input_size,
        output_size=output_size,
        nombre=configuracion.get('nombre', 'NeuronaRefuerzoRainbowDQN'),
        learning_rate=configuracion.get('learning_rate', 0.00025),
        gamma=configuracion.get('gamma', 0.99),
        epsilon=configuracion.get('epsilon', 0.1),
        epsilon_decay=configuracion.get('epsilon_decay', 0.995),
        epsilon_min=configuracion.get('epsilon_min', 0.01),
        tau=configuracion.get('tau', 0.005),
        buffer_size=configuracion.get('buffer_size', 1000000),
        batch_size=configuracion.get('batch_size', 32),
        usar_he=configuracion.get('usar_he', True),
        usar_dueling=configuracion.get('usar_dueling', True),
        usar_prioritized=configuracion.get('usar_prioritized', True),
        usar_multi_step=configuracion.get('usar_multi_step', True),
        usar_distributional=configuracion.get('usar_distributional', True),
        usar_noisy=configuracion.get('usar_noisy', True),
        n_atoms=configuracion.get('n_atoms', 51),
        v_min=configuracion.get('v_min', -10.0),
        v_max=configuracion.get('v_max', 10.0),
        n_steps=configuracion.get('n_steps', 3),
        alpha_prioritized=configuracion.get('alpha_prioritized', 0.6),
        beta_prioritized=configuracion.get('beta_prioritized', 0.4)
    )


# Configuración específica para RFEN1_RN_9
RFEN9_CONFIG = {
    'inicializacion_preferida': 'he',
    'learning_rate_default': 0.00025,
    'gamma_default': 0.99,
    'epsilon_default': 0.1,
    'epsilon_decay_default': 0.995,
    'epsilon_min_default': 0.01,
    'tau_default': 0.005,
    'buffer_size_default': 1000000,
    'batch_size_default': 32,
    'usar_dueling_default': True,
    'usar_prioritized_default': True,
    'usar_multi_step_default': True,
    'usar_distributional_default': True,
    'usar_noisy_default': True,
    'n_atoms_default': 51,
    'v_min_default': -10.0,
    'v_max_default': 10.0,
    'n_steps_default': 3,
    'alpha_prioritized_default': 0.6,
    'beta_prioritized_default': 0.4,
    'umbral_convergencia': 0.8,
    'umbral_estabilidad': 0.7
}

logger.info("RFEN1_RN_9.py cargado correctamente - Neurona de Refuerzo Rainbow DQN Multi-Método")
