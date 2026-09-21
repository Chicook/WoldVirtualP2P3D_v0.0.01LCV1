"""
RFEN1_RN_1.py - Neurona de Refuerzo con Q-Learning
===================================================

Esta neurona implementa el algoritmo Q-Learning clásico con optimización de pesos
específica para aprendizaje por refuerzo. Incluye técnicas avanzadas de 2025.

Características:
- Q-Learning con tabla Q optimizada
- Inicialización de pesos específica para Q-Learning
- Monitoreo de convergencia y estabilidad
- Adaptación automática de hiperparámetros

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

logger = logging.getLogger('RFENRN1.RFEN1_RN_1')


class NeuronaRefuerzoQLearning(NeuronaRefuerzoBase):
    """
    Neurona de refuerzo con algoritmo Q-Learning.

    Esta neurona implementa Q-Learning con optimización de pesos específica
    para aprendizaje por refuerzo, incluyendo técnicas avanzadas de 2025.
    """

    def __init__(self, input_size: int, output_size: int,
                 nombre: str = "NeuronaRefuerzoQLearning",
                 learning_rate: float = 0.1,
                 gamma: float = 0.99,
                 epsilon: float = 0.1,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01,
                 usar_xavier: bool = True):
        """
        Inicializa la neurona de refuerzo Q-Learning.

        Args:
            input_size: Número de estados posibles
            output_size: Número de acciones posibles
            nombre: Nombre identificativo de la neurona
            learning_rate: Tasa de aprendizaje
            gamma: Factor de descuento
            epsilon: Valor inicial de epsilon para exploración
            epsilon_decay: Factor de decaimiento de epsilon
            epsilon_min: Valor mínimo de epsilon
            usar_xavier: Si usar inicialización Xavier
        """
        super().__init__(input_size, output_size, nombre)

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.usar_xavier = usar_xavier

        # Tabla Q para almacenar valores Q(s,a)
        self.q_table = None

        # Estadísticas específicas de Q-Learning
        self.estadisticas_qlearning = {
            'convergencia_q': 0.0,
            'estabilidad_q': 0.0,
            'exploracion_rate': 0.0,
            'exploitacion_rate': 0.0,
            'q_values_media': 0.0,
            'q_values_std': 0.0,
            'q_values_min': 0.0,
            'q_values_max': 0.0,
            'actualizaciones_q': 0,
            'convergencia_detectada': False
        }

        # Historial de valores Q para análisis
        self.historial_q_values = []
        self.historial_epsilon = []
        self.historial_convergencia = []

        logger.info(f"NeuronaRefuerzoQLearning creada: {self}")

    def inicializar_pesos(self) -> None:
        """
        Inicializa la tabla Q con valores optimizados.
        """
        if self.usar_xavier:
            # Inicialización Xavier para la tabla Q
            self.q_table = inicializar_pesos_xavier(
                (self.input_size, self.output_size),
                fan_in=self.input_size,
                fan_out=self.output_size
            )
        else:
            # Inicialización con ceros (método clásico)
            self.q_table = np.zeros((self.input_size, self.output_size),
                                    dtype=LUCIA_RL_CONFIG['precision'])

        # Inicializar sesgo con ceros
        self.sesgo = np.zeros((1, self.output_size), dtype=LUCIA_RL_CONFIG['precision'])

        logger.info(f"Tabla Q inicializada con {'Xavier' if self.usar_xavier else 'ceros'}: "
                    f"forma={self.q_table.shape}")

    def forward(self, estado: np.ndarray) -> np.ndarray:
        """
        Propagación hacia adelante - selección de acción.

        Args:
            estado: Estado actual del entorno

        Returns:
            Acción seleccionada
        """
        if self.q_table is None:
            self.inicializar_pesos()

        # Obtener valores Q para el estado actual
        q_values = self.q_table[estado]

        # Estrategia epsilon-greedy
        if np.random.random() < self.epsilon:
            # Exploración: acción aleatoria
            accion = np.random.randint(0, self.output_size)
            self.estadisticas_qlearning['exploracion_rate'] += 1
        else:
            # Explotación: mejor acción según Q-values
            accion = np.argmax(q_values)
            self.estadisticas_qlearning['exploitacion_rate'] += 1

        # Guardar historial
        self.historial_q_values.append(q_values.copy())
        self.historial_epsilon.append(self.epsilon)

        return np.array([accion])

    def actualizar_q_value(self, estado: int, accion: int, recompensa: float,
                           siguiente_estado: int, terminado: bool = False) -> None:
        """
        Actualiza el valor Q usando la ecuación de Bellman.

        Args:
            estado: Estado actual
            accion: Acción tomada
            recompensa: Recompensa recibida
            siguiente_estado: Siguiente estado
            terminado: Si el episodio ha terminado
        """
        if self.q_table is None:
            self.inicializar_pesos()

        # Valor Q actual
        q_actual = self.q_table[estado, accion]

        # Calcular valor Q objetivo
        if terminado:
            q_objetivo = recompensa
        else:
            q_max_siguiente = np.max(self.q_table[siguiente_estado])
            q_objetivo = recompensa + self.gamma * q_max_siguiente

        # Actualizar valor Q usando la ecuación de Bellman
        self.q_table[estado, accion] = q_actual + self.learning_rate * (q_objetivo - q_actual)

        # Actualizar estadísticas
        self.estadisticas_qlearning['actualizaciones_q'] += 1

        # Decaimiento de epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Calcular convergencia
        self._calcular_convergencia_q()

    def _calcular_convergencia_q(self) -> None:
        """
        Calcula la convergencia de la tabla Q.
        """
        if len(self.historial_q_values) < 10:
            return

        # Calcular cambio promedio en valores Q
        cambios_q = []
        for i in range(1, len(self.historial_q_values)):
            cambio = np.mean(np.abs(self.historial_q_values[i] - self.historial_q_values[i-1]))
            cambios_q.append(cambio)

        # Convergencia basada en la estabilidad de los cambios
        if cambios_q:
            cambio_promedio = np.mean(cambios_q[-10:])  # Últimos 10 cambios
            convergencia = 1.0 - min(cambio_promedio, 1.0)
            self.estadisticas_qlearning['convergencia_q'] = convergencia

            # Detectar convergencia
            if convergencia > 0.95 and len(cambios_q) > 100:
                self.estadisticas_qlearning['convergencia_detectada'] = True

            self.historial_convergencia.append(convergencia)

    def calcular_estabilidad_q(self) -> float:
        """
        Calcula la estabilidad de los valores Q.

        Returns:
            Índice de estabilidad (0-1)
        """
        if len(self.historial_q_values) < 20:
            return 0.0

        # Calcular varianza de los valores Q en el tiempo
        q_values_array = np.array(self.historial_q_values[-20:])
        varianza_temporal = np.var(q_values_array, axis=0)
        varianza_promedio = np.mean(varianza_temporal)

        # Estabilidad inversamente proporcional a la varianza
        estabilidad = 1.0 / (1.0 + varianza_promedio)

        self.estadisticas_qlearning['estabilidad_q'] = estabilidad

        return estabilidad

    def obtener_estadisticas_q(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas específicas de Q-Learning.

        Returns:
            Diccionario con estadísticas de Q-Learning
        """
        if self.q_table is None:
            return {'estado': 'no_inicializada'}

        # Calcular estadísticas de la tabla Q
        q_flat = self.q_table.flatten()

        stats_q = {
            'q_values_media': np.mean(q_flat),
            'q_values_std': np.std(q_flat),
            'q_values_min': np.min(q_flat),
            'q_values_max': np.max(q_flat),
            'epsilon_actual': self.epsilon,
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'convergencia_q': self.estadisticas_qlearning['convergencia_q'],
            'estabilidad_q': self.calcular_estabilidad_q(),
            'actualizaciones_q': self.estadisticas_qlearning['actualizaciones_q'],
            'convergencia_detectada': self.estadisticas_qlearning['convergencia_detectada']
        }

        # Calcular tasas de exploración y explotación
        total_acciones = (self.estadisticas_qlearning['exploracion_rate'] +
                          self.estadisticas_qlearning['exploitacion_rate'])

        if total_acciones > 0:
            stats_q['tasa_exploracion'] = (self.estadisticas_qlearning['exploracion_rate'] /
                                           total_acciones)
            stats_q['tasa_exploitacion'] = (self.estadisticas_qlearning['exploitacion_rate'] /
                                            total_acciones)
        else:
            stats_q['tasa_exploracion'] = 0.0
            stats_q['tasa_exploitacion'] = 0.0

        return stats_q

    def verificar_estabilidad(self) -> Dict[str, bool]:
        """
        Verifica la estabilidad de la neurona Q-Learning.

        Returns:
            Diccionario con indicadores de estabilidad
        """
        estabilidad = {}

        # Verificar convergencia
        convergencia = self.estadisticas_qlearning['convergencia_q']
        estabilidad['convergencia_ok'] = convergencia > 0.8

        # Verificar estabilidad de Q-values
        estabilidad_q = self.calcular_estabilidad_q()
        estabilidad['q_values_estables'] = estabilidad_q > 0.7

        # Verificar epsilon
        estabilidad['epsilon_apropiado'] = self.epsilon_min <= self.epsilon <= 1.0

        # Verificar tabla Q
        if self.q_table is not None:
            q_max = np.max(np.abs(self.q_table))
            q_min = np.min(np.abs(self.q_table))

            estabilidad['q_values_no_explosivos'] = q_max < 100.0
            estabilidad['q_values_no_desaparecen'] = q_min > -100.0
            estabilidad['q_values_balanceados'] = q_max / (q_min + 1e-8) < 1000.0
        else:
            estabilidad['q_values_no_explosivos'] = True
            estabilidad['q_values_no_desaparecen'] = True
            estabilidad['q_values_balanceados'] = True

        return estabilidad

    def reinicializar_con_parametros(self, learning_rate: float = None,
                                     gamma: float = None,
                                     epsilon: float = None,
                                     epsilon_decay: float = None,
                                     epsilon_min: float = None) -> None:
        """
        Reinicializa la neurona con nuevos parámetros.

        Args:
            learning_rate: Nueva tasa de aprendizaje
            gamma: Nuevo factor de descuento
            epsilon: Nuevo valor de epsilon
            epsilon_decay: Nuevo factor de decaimiento
            epsilon_min: Nuevo valor mínimo de epsilon
        """
        if learning_rate is not None:
            self.learning_rate = learning_rate
        if gamma is not None:
            self.gamma = gamma
        if epsilon is not None:
            self.epsilon = epsilon
        if epsilon_decay is not None:
            self.epsilon_decay = epsilon_decay
        if epsilon_min is not None:
            self.epsilon_min = epsilon_min

        # Reinicializar tabla Q
        self.inicializar_pesos()

        # Resetear historial
        self.resetear_historial()

        # Limpiar estadísticas específicas
        self.historial_q_values.clear()
        self.historial_epsilon.clear()
        self.historial_convergencia.clear()

        # Resetear contadores
        self.estadisticas_qlearning['exploracion_rate'] = 0
        self.estadisticas_qlearning['exploitacion_rate'] = 0
        self.estadisticas_qlearning['actualizaciones_q'] = 0
        self.estadisticas_qlearning['convergencia_detectada'] = False

        logger.info(f"Neurona Q-Learning reinicializada: lr={self.learning_rate}, "
                    f"gamma={self.gamma}, epsilon={self.epsilon}")

    def obtener_mejor_politica(self) -> np.ndarray:
        """
        Obtiene la mejor política basada en los valores Q actuales.

        Returns:
            Array con la mejor acción para cada estado
        """
        if self.q_table is None:
            raise ValueError("Tabla Q no inicializada")

        return np.argmax(self.q_table, axis=1)

    def obtener_valores_q(self, estado: int) -> np.ndarray:
        """
        Obtiene los valores Q para un estado específico.

        Args:
            estado: Estado del que obtener valores Q

        Returns:
            Array con valores Q para todas las acciones
        """
        if self.q_table is None:
            raise ValueError("Tabla Q no inicializada")

        return self.q_table[estado].copy()

    def __str__(self) -> str:
        return (f"NeuronaRefuerzoQLearning(entrada={self.input_size}, "
                f"salida={self.output_size}, lr={self.learning_rate}, "
                f"gamma={self.gamma}, epsilon={self.epsilon:.3f})")

    def __repr__(self) -> str:
        return self.__str__()

# Funciones de utilidad específicas para RFEN1_RN_1


def crear_neurona_qlearning(input_size: int, output_size: int,
                            configuracion: Dict[str, Any] = None) -> NeuronaRefuerzoQLearning:
    """
    Función de conveniencia para crear una neurona Q-Learning.

    Args:
        input_size: Número de estados
        output_size: Número de acciones
        configuracion: Configuración adicional

    Returns:
        Instancia de NeuronaRefuerzoQLearning
    """
    if configuracion is None:
        configuracion = {}

    return NeuronaRefuerzoQLearning(
        input_size=input_size,
        output_size=output_size,
        nombre=configuracion.get('nombre', 'NeuronaRefuerzoQLearning'),
        learning_rate=configuracion.get('learning_rate', 0.1),
        gamma=configuracion.get('gamma', 0.99),
        epsilon=configuracion.get('epsilon', 0.1),
        epsilon_decay=configuracion.get('epsilon_decay', 0.995),
        epsilon_min=configuracion.get('epsilon_min', 0.01),
        usar_xavier=configuracion.get('usar_xavier', True)
    )


def analizar_convergencia_qlearning(neurona: NeuronaRefuerzoQLearning) -> Dict[str, Any]:
    """
    Analiza la convergencia de una neurona Q-Learning.

    Args:
        neurona: Neurona Q-Learning a analizar

    Returns:
        Diccionario con análisis de convergencia
    """
    analisis = {
        'convergencia_q': neurona.estadisticas_qlearning['convergencia_q'],
        'estabilidad_q': neurona.calcular_estabilidad_q(),
        'epsilon_actual': neurona.epsilon,
        'actualizaciones_q': neurona.estadisticas_qlearning['actualizaciones_q'],
        'convergencia_detectada': neurona.estadisticas_qlearning['convergencia_detectada'],
        'estadisticas_q': neurona.obtener_estadisticas_q(),
        'estabilidad_verificada': neurona.verificar_estabilidad()
    }

    return analisis


def comparar_inicializaciones_qlearning(input_size: int, output_size: int,
                                        episodios: int = 1000) -> Dict[str, Any]:
    """
    Compara diferentes métodos de inicialización para Q-Learning.

    Args:
        input_size: Número de estados
        output_size: Número de acciones
        episodios: Número de episodios para la comparación

    Returns:
        Diccionario con estadísticas comparativas
    """
    # Crear neuronas con diferentes inicializaciones
    neurona_xavier = NeuronaRefuerzoQLearning(input_size, output_size, usar_xavier=True)
    neurona_ceros = NeuronaRefuerzoQLearning(input_size, output_size, usar_xavier=False)

    # Simular entrenamiento básico
    for episodio in range(episodios):
        estado = np.random.randint(0, input_size)
        accion = neurona_xavier.forward(np.array([estado]))[0]
        recompensa = np.random.random()
        siguiente_estado = np.random.randint(0, input_size)

        neurona_xavier.actualizar_q_value(estado, accion, recompensa, siguiente_estado)

        estado = np.random.randint(0, input_size)
        accion = neurona_ceros.forward(np.array([estado]))[0]
        recompensa = np.random.random()
        siguiente_estado = np.random.randint(0, input_size)

        neurona_ceros.actualizar_q_value(estado, accion, recompensa, siguiente_estado)

    # Calcular estadísticas comparativas
    comparacion = {
        'xavier': {
            'convergencia_q': neurona_xavier.estadisticas_qlearning['convergencia_q'],
            'estabilidad_q': neurona_xavier.calcular_estabilidad_q(),
            'q_values_media': np.mean(neurona_xavier.q_table),
            'q_values_std': np.std(neurona_xavier.q_table),
            'actualizaciones_q': neurona_xavier.estadisticas_qlearning['actualizaciones_q']
        },
        'ceros': {
            'convergencia_q': neurona_ceros.estadisticas_qlearning['convergencia_q'],
            'estabilidad_q': neurona_ceros.calcular_estabilidad_q(),
            'q_values_media': np.mean(neurona_ceros.q_table),
            'q_values_std': np.std(neurona_ceros.q_table),
            'actualizaciones_q': neurona_ceros.estadisticas_qlearning['actualizaciones_q']
        }
    }

    # Calcular ratios comparativos
    comparacion['ratios'] = {
        'convergencia_q': comparacion['xavier']['convergencia_q'] / comparacion['ceros']['convergencia_q'],
        'estabilidad_q': comparacion['xavier']['estabilidad_q'] / comparacion['ceros']['estabilidad_q'],
        'q_values_std': comparacion['xavier']['q_values_std'] / comparacion['ceros']['q_values_std']
    }

    return comparacion


# Configuración específica para RFEN1_RN_1
RFEN1_CONFIG = {
    'inicializacion_preferida': 'xavier',
    'learning_rate_default': 0.1,
    'gamma_default': 0.99,
    'epsilon_default': 0.1,
    'epsilon_decay_default': 0.995,
    'epsilon_min_default': 0.01,
    'umbral_convergencia': 0.95,
    'umbral_estabilidad': 0.7
}

logger.info("RFEN1_RN_1.py cargado correctamente - Neurona de Refuerzo Q-Learning")
