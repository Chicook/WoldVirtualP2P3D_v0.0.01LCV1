"""
RFEN1_RN_4.py - Neurona de Refuerzo con Deep Q-Network (DQN)
============================================================

Esta neurona implementa el algoritmo Deep Q-Network (DQN) con optimización de pesos
específica para redes neuronales profundas y experiencia replay.

Características:
- DQN con experiencia replay y red objetivo
- Inicialización de pesos específica para redes profundas
- Monitoreo de estabilidad de la red objetivo
- Adaptación automática de hiperparámetros

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

logger = logging.getLogger('RFENRN1.RFEN1_RN_4')


class NeuronaRefuerzoDQN(NeuronaRefuerzoBase):
    """
    Neurona de refuerzo con algoritmo Deep Q-Network (DQN).

    Esta neurona implementa DQN con experiencia replay, red objetivo
    y optimización de pesos específica para redes profundas.
    """

    def __init__(self, input_size: int, output_size: int,
                 nombre: str = "NeuronaRefuerzoDQN",
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,
                 epsilon: float = 0.1,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01,
                 tau: float = 0.005,
                 buffer_size: int = 10000,
                 batch_size: int = 32,
                 usar_he: bool = True):
        """
        Inicializa la neurona de refuerzo DQN.

        Args:
            input_size: Número de características de entrada
            output_size: Número de acciones posibles
            nombre: Nombre identificativo de la neurona
            learning_rate: Tasa de aprendizaje
            gamma: Factor de descuento
            epsilon: Valor inicial de epsilon para exploración
            epsilon_decay: Factor de decaimiento de epsilon
            epsilon_min: Valor mínimo de epsilon
            tau: Parámetro de actualización suave de la red objetivo
            buffer_size: Tamaño del buffer de experiencia
            batch_size: Tamaño del batch para entrenamiento
            usar_he: Si usar inicialización He para redes profundas
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

        # Red principal y red objetivo
        self.pesos_principal = None
        self.sesgo_principal = None
        self.pesos_objetivo = None
        self.sesgo_objetivo = None

        # Buffer de experiencia
        self.buffer_experiencia = deque(maxlen=buffer_size)

        # Estadísticas específicas de DQN
        self.estadisticas_dqn = {
            'td_error_medio': 0.0,
            'td_error_std': 0.0,
            'convergencia_q': 0.0,
            'estabilidad_q': 0.0,
            'actualizaciones_target': 0,
            'muestras_buffer': 0,
            'exploracion_rate': 0.0,
            'exploitacion_rate': 0.0,
            'loss_medio': 0.0,
            'gradiente_norma': 0.0
        }

        # Historial para análisis
        self.historial_td_errors = []
        self.historial_losses = []
        self.historial_q_values = []
        self.historial_epsilon = []

        logger.info(f"NeuronaRefuerzoDQN creada: {self}")

    def inicializar_pesos(self) -> None:
        """
        Inicializa los pesos de la red principal y objetivo.
        """
        if self.usar_he:
            # Inicialización He para redes profundas con ReLU
            self.pesos_principal = inicializar_pesos_he(
                (self.input_size, self.output_size),
                fan_in=self.input_size
            )
            self.pesos_objetivo = inicializar_pesos_he(
                (self.input_size, self.output_size),
                fan_in=self.input_size
            )
        else:
            # Inicialización Xavier como alternativa
            self.pesos_principal = inicializar_pesos_xavier(
                (self.input_size, self.output_size),
                fan_in=self.input_size,
                fan_out=self.output_size
            )
            self.pesos_objetivo = inicializar_pesos_xavier(
                (self.input_size, self.output_size),
                fan_in=self.input_size,
                fan_out=self.output_size
            )

        # Inicializar sesgos con ceros
        self.sesgo_principal = np.zeros((1, self.output_size), dtype=LUCIA_RL_CONFIG['precision'])
        self.sesgo_objetivo = np.zeros((1, self.output_size), dtype=LUCIA_RL_CONFIG['precision'])

        logger.info(f"Pesos DQN inicializados con {'He' if self.usar_he else 'Xavier'}")

    def forward(self, estado: np.ndarray, usar_objetivo: bool = False) -> np.ndarray:
        """
        Propagación hacia adelante - cálculo de Q-values.

        Args:
            estado: Estado actual del entorno
            usar_objetivo: Si usar la red objetivo

        Returns:
            Q-values para todas las acciones
        """
        if self.pesos_principal is None:
            self.inicializar_pesos()

        if usar_objetivo:
            pesos = self.pesos_objetivo
            sesgo = self.sesgo_objetivo
        else:
            pesos = self.pesos_principal
            sesgo = self.sesgo_principal

        # Calcular Q-values
        q_values = np.dot(estado, pesos) + sesgo

        # Guardar historial
        self.historial_q_values.append(q_values.copy())

        return q_values

    def seleccionar_accion(self, estado: np.ndarray) -> int:
        """
        Selecciona una acción usando epsilon-greedy.

        Args:
            estado: Estado actual del entorno

        Returns:
            Acción seleccionada
        """
        if np.random.random() < self.epsilon:
            # Exploración: acción aleatoria
            accion = np.random.randint(0, self.output_size)
            self.estadisticas_dqn['exploracion_rate'] += 1
        else:
            # Explotación: mejor acción según Q-values
            q_values = self.forward(estado)
            accion = np.argmax(q_values[0])
            self.estadisticas_dqn['exploitacion_rate'] += 1

        return accion

    def agregar_experiencia(self, estado: np.ndarray, accion: int,
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
            'accion': accion,
            'recompensa': recompensa,
            'siguiente_estado': siguiente_estado.copy(),
            'terminado': terminado
        }

        self.buffer_experiencia.append(experiencia)
        self.estadisticas_dqn['muestras_buffer'] = len(self.buffer_experiencia)

    def muestrear_batch(self) -> Optional[List[Dict[str, Any]]]:
        """
        Muestrea un batch del buffer de experiencia.

        Returns:
            Lista de experiencias o None si no hay suficientes muestras
        """
        if len(self.buffer_experiencia) < self.batch_size:
            return None

        return random.sample(list(self.buffer_experiencia), self.batch_size)

    def calcular_td_error(self, batch: List[Dict[str, Any]]) -> List[float]:
        """
        Calcula el TD error para un batch de experiencias.

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

            # Q-value objetivo
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

    def calcular_gradientes(self, batch: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula los gradientes para un batch de experiencias.

        Args:
            batch: Batch de experiencias

        Returns:
            Tupla con (gradiente_pesos, gradiente_sesgo)
        """
        if self.pesos_principal is None:
            raise ValueError("Pesos no inicializados")

        gradiente_pesos = np.zeros_like(self.pesos_principal)
        gradiente_sesgo = np.zeros_like(self.sesgo_principal)

        td_errors = self.calcular_td_error(batch)
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
        self.estadisticas_dqn['gradiente_norma'] = norma_gradiente

        return gradiente_pesos, gradiente_sesgo

    def actualizar_pesos(self, gradiente_pesos: np.ndarray, gradiente_sesgo: np.ndarray) -> None:
        """
        Actualiza los pesos de la red principal.

        Args:
            gradiente_pesos: Gradiente de los pesos
            gradiente_sesgo: Gradiente del sesgo
        """
        if self.pesos_principal is None:
            raise ValueError("Pesos no inicializados")

        # Actualizar pesos principales
        self.pesos_principal += self.learning_rate * gradiente_pesos
        self.sesgo_principal += self.learning_rate * gradiente_sesgo

        # Actualización suave de la red objetivo
        self._actualizar_red_objetivo()

        # Decaimiento de epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Calcular convergencia
        self._calcular_convergencia_dqn()

    def _actualizar_red_objetivo(self) -> None:
        """
        Actualiza la red objetivo usando actualización suave.
        """
        if self.pesos_objetivo is None:
            return

        # Actualización suave (soft update)
        self.pesos_objetivo = (1 - self.tau) * self.pesos_objetivo + self.tau * self.pesos_principal
        self.sesgo_objetivo = (1 - self.tau) * self.sesgo_objetivo + self.tau * self.sesgo_principal

        self.estadisticas_dqn['actualizaciones_target'] += 1

    def _calcular_convergencia_dqn(self) -> None:
        """
        Calcula la convergencia de la red DQN.
        """
        if len(self.historial_td_errors) < 10:
            return

        # Convergencia basada en la estabilidad de los TD errors
        td_errors_recientes = self.historial_td_errors[-10:]
        varianza_td = np.var(td_errors_recientes)

        # Convergencia inversamente proporcional a la varianza
        convergencia = 1.0 / (1.0 + varianza_td)
        self.estadisticas_dqn['convergencia_q'] = convergencia

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

        self.estadisticas_dqn['estabilidad_q'] = estabilidad

        return estabilidad

    def entrenar_step(self) -> Optional[float]:
        """
        Realiza un paso de entrenamiento.

        Returns:
            Pérdida promedio o None si no hay suficientes muestras
        """
        batch = self.muestrear_batch()
        if batch is None:
            return None

        # Calcular gradientes
        gradiente_pesos, gradiente_sesgo = self.calcular_gradientes(batch)

        # Actualizar pesos
        self.actualizar_pesos(gradiente_pesos, gradiente_sesgo)

        # Retornar pérdida promedio
        return np.mean(self.historial_losses[-len(batch):])

    def obtener_estadisticas_dqn(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas específicas de DQN.

        Returns:
            Diccionario con estadísticas de DQN
        """
        if self.pesos_principal is None:
            return {'estado': 'no_inicializada'}

        # Calcular estadísticas de TD errors
        if self.historial_td_errors:
            td_errors_array = np.array(self.historial_td_errors[-100:])
            self.estadisticas_dqn['td_error_medio'] = np.mean(td_errors_array)
            self.estadisticas_dqn['td_error_std'] = np.std(td_errors_array)

        # Calcular estadísticas de pérdidas
        if self.historial_losses:
            self.estadisticas_dqn['loss_medio'] = np.mean(self.historial_losses[-10:])

        # Calcular tasas de exploración y explotación
        total_acciones = (self.estadisticas_dqn['exploracion_rate'] +
                          self.estadisticas_dqn['exploitacion_rate'])

        if total_acciones > 0:
            self.estadisticas_dqn['tasa_exploracion'] = (self.estadisticas_dqn['exploracion_rate'] /
                                                         total_acciones)
            self.estadisticas_dqn['tasa_exploitacion'] = (self.estadisticas_dqn['exploitacion_rate'] /
                                                          total_acciones)
        else:
            self.estadisticas_dqn['tasa_exploracion'] = 0.0
            self.estadisticas_dqn['tasa_exploitacion'] = 0.0

        stats_dqn = {
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'epsilon_actual': self.epsilon,
            'tau': self.tau,
            'batch_size': self.batch_size,
            'buffer_size': len(self.buffer_experiencia),
            'convergencia_q': self.estadisticas_dqn['convergencia_q'],
            'estabilidad_q': self.calcular_estabilidad_q(),
            'actualizaciones_target': self.estadisticas_dqn['actualizaciones_target'],
            'muestras_buffer': self.estadisticas_dqn['muestras_buffer'],
            'td_error_medio': self.estadisticas_dqn['td_error_medio'],
            'td_error_std': self.estadisticas_dqn['td_error_std'],
            'loss_medio': self.estadisticas_dqn['loss_medio'],
            'gradiente_norma': self.estadisticas_dqn['gradiente_norma'],
            'tasa_exploracion': self.estadisticas_dqn['tasa_exploracion'],
            'tasa_exploitacion': self.estadisticas_dqn['tasa_exploitacion']
        }

        return stats_dqn

    def verificar_estabilidad(self) -> Dict[str, bool]:
        """
        Verifica la estabilidad de la neurona DQN.

        Returns:
            Diccionario con indicadores de estabilidad
        """
        estabilidad = {}

        # Verificar convergencia
        convergencia = self.estadisticas_dqn['convergencia_q']
        estabilidad['convergencia_ok'] = convergencia > 0.8

        # Verificar estabilidad de Q-values
        estabilidad_q = self.calcular_estabilidad_q()
        estabilidad['q_values_estables'] = estabilidad_q > 0.7

        # Verificar epsilon
        estabilidad['epsilon_apropiado'] = self.epsilon_min <= self.epsilon <= 1.0

        # Verificar buffer
        estabilidad['buffer_suficiente'] = len(self.buffer_experiencia) >= self.batch_size

        # Verificar pesos principales
        if self.pesos_principal is not None:
            peso_max = np.max(np.abs(self.pesos_principal))
            peso_min = np.min(np.abs(self.pesos_principal))

            estabilidad['pesos_no_explosivos'] = peso_max < 10.0
            estabilidad['pesos_no_desaparecen'] = peso_min > 1e-6
            estabilidad['pesos_balanceados'] = peso_max / (peso_min + 1e-8) < 1000.0
        else:
            estabilidad['pesos_no_explosivos'] = True
            estabilidad['pesos_no_desaparecen'] = True
            estabilidad['pesos_balanceados'] = True

        # Verificar gradientes
        norma_gradiente = self.estadisticas_dqn['gradiente_norma']
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
        self.historial_td_errors.clear()
        self.historial_losses.clear()
        self.historial_q_values.clear()
        self.historial_epsilon.clear()

        # Resetear contadores
        self.estadisticas_dqn['actualizaciones_target'] = 0
        self.estadisticas_dqn['muestras_buffer'] = 0
        self.estadisticas_dqn['exploracion_rate'] = 0
        self.estadisticas_dqn['exploitacion_rate'] = 0

        logger.info(f"Neurona DQN reinicializada: lr={self.learning_rate}, "
                    f"gamma={self.gamma}, epsilon={self.epsilon}, tau={self.tau}")

    def __str__(self) -> str:
        return (f"NeuronaRefuerzoDQN(entrada={self.input_size}, "
                f"salida={self.output_size}, lr={self.learning_rate}, "
                f"gamma={self.gamma}, epsilon={self.epsilon:.3f}, "
                f"buffer={len(self.buffer_experiencia)})")

    def __repr__(self) -> str:
        return self.__str__()

# Funciones de utilidad específicas para RFEN1_RN_4


def crear_neurona_dqn(input_size: int, output_size: int,
                      configuracion: Dict[str, Any] = None) -> NeuronaRefuerzoDQN:
    """
    Función de conveniencia para crear una neurona DQN.

    Args:
        input_size: Número de características de entrada
        output_size: Número de acciones
        configuracion: Configuración adicional

    Returns:
        Instancia de NeuronaRefuerzoDQN
    """
    if configuracion is None:
        configuracion = {}

    return NeuronaRefuerzoDQN(
        input_size=input_size,
        output_size=output_size,
        nombre=configuracion.get('nombre', 'NeuronaRefuerzoDQN'),
        learning_rate=configuracion.get('learning_rate', 0.001),
        gamma=configuracion.get('gamma', 0.99),
        epsilon=configuracion.get('epsilon', 0.1),
        epsilon_decay=configuracion.get('epsilon_decay', 0.995),
        epsilon_min=configuracion.get('epsilon_min', 0.01),
        tau=configuracion.get('tau', 0.005),
        buffer_size=configuracion.get('buffer_size', 10000),
        batch_size=configuracion.get('batch_size', 32),
        usar_he=configuracion.get('usar_he', True)
    )


# Configuración específica para RFEN1_RN_4
RFEN4_CONFIG = {
    'inicializacion_preferida': 'he',
    'learning_rate_default': 0.001,
    'gamma_default': 0.99,
    'epsilon_default': 0.1,
    'epsilon_decay_default': 0.995,
    'epsilon_min_default': 0.01,
    'tau_default': 0.005,
    'buffer_size_default': 10000,
    'batch_size_default': 32,
    'umbral_convergencia': 0.8,
    'umbral_estabilidad': 0.7
}

logger.info("RFEN1_RN_4.py cargado correctamente - Neurona de Refuerzo DQN")
