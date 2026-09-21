"""
RFEN1_RN_2.py - Neurona de Refuerzo con Policy Gradient
======================================================

Esta neurona implementa el algoritmo Policy Gradient con optimización de pesos
específica para aprendizaje por refuerzo basado en políticas.

Características:
- Policy Gradient con optimización de políticas
- Inicialización de pesos específica para políticas
- Monitoreo de gradientes de política
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

logger = logging.getLogger('RFENRN1.RFEN1_RN_2')


class NeuronaRefuerzoPolicyGradient(NeuronaRefuerzoBase):
    """
    Neurona de refuerzo con algoritmo Policy Gradient.

    Esta neurona implementa Policy Gradient con optimización de pesos específica
    para aprendizaje por refuerzo basado en políticas.
    """

    def __init__(self, input_size: int, output_size: int,
                 nombre: str = "NeuronaRefuerzoPolicyGradient",
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,
                 usar_ortogonal: bool = True,
                 entropy_coef: float = 0.01):
        """
        Inicializa la neurona de refuerzo Policy Gradient.

        Args:
            input_size: Número de características de entrada
            output_size: Número de acciones posibles
            nombre: Nombre identificativo de la neurona
            learning_rate: Tasa de aprendizaje
            gamma: Factor de descuento
            usar_ortogonal: Si usar inicialización ortogonal
            entropy_coef: Coeficiente de entropía para regularización
        """
        super().__init__(input_size, output_size, nombre)

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.usar_ortogonal = usar_ortogonal
        self.entropy_coef = entropy_coef

        # Estadísticas específicas de Policy Gradient
        self.estadisticas_pg = {
            'gradiente_politica_norma': 0.0,
            'entropia_politica': 0.0,
            'probabilidad_accion_media': 0.0,
            'probabilidad_accion_std': 0.0,
            'convergencia_politica': 0.0,
            'estabilidad_politica': 0.0,
            'actualizaciones_politica': 0,
            'exploracion_efectiva': 0.0
        }

        # Historial para análisis
        self.historial_politicas = []
        self.historial_gradientes = []
        self.historial_entropias = []
        self.historial_recompensas_descontadas = []

        logger.info(f"NeuronaRefuerzoPolicyGradient creada: {self}")

    def inicializar_pesos(self) -> None:
        """
        Inicializa los pesos de la política con inicialización ortogonal.
        """
        if self.usar_ortogonal:
            # Inicialización ortogonal para políticas
            self.pesos = inicializar_pesos_ortogonal((self.input_size, self.output_size))
        else:
            # Inicialización Xavier como alternativa
            limit = math.sqrt(6.0 / (self.input_size + self.output_size))
            self.pesos = np.random.uniform(-limit, limit, (self.input_size, self.output_size))
            self.pesos = self.pesos.astype(LUCIA_RL_CONFIG['precision'])

        # Inicializar sesgo con ceros
        self.sesgo = np.zeros((1, self.output_size), dtype=LUCIA_RL_CONFIG['precision'])

        logger.info(f"Pesos de política inicializados con {'ortogonal' if self.usar_ortogonal else 'Xavier'}")

    def forward(self, estado: np.ndarray) -> np.ndarray:
        """
        Propagación hacia adelante - cálculo de probabilidades de acción.

        Args:
            estado: Estado actual del entorno

        Returns:
            Probabilidades de acción (softmax)
        """
        if self.pesos is None:
            self.inicializar_pesos()

        # Calcular logits
        logits = np.dot(estado, self.pesos) + self.sesgo

        # Aplicar softmax para obtener probabilidades
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))  # Estabilidad numérica
        probabilidades = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # Guardar historial
        self.historial_politicas.append(probabilidades.copy())

        # Calcular entropía de la política
        entropia = -np.sum(probabilidades * np.log(probabilidades + 1e-8), axis=1)
        self.historial_entropias.append(np.mean(entropia))

        return probabilidades

    def seleccionar_accion(self, estado: np.ndarray) -> Tuple[int, float]:
        """
        Selecciona una acción basada en la política actual.

        Args:
            estado: Estado actual del entorno

        Returns:
            Tupla con (acción_seleccionada, probabilidad_accion)
        """
        probabilidades = self.forward(estado)

        # Muestrear acción según las probabilidades
        accion = np.random.choice(self.output_size, p=probabilidades[0])
        probabilidad_accion = probabilidades[0, accion]

        return accion, probabilidad_accion

    def calcular_gradiente_politica(self, estados: List[np.ndarray],
                                    acciones: List[int],
                                    recompensas_descontadas: List[float]) -> np.ndarray:
        """
        Calcula el gradiente de la política usando REINFORCE.

        Args:
            estados: Lista de estados
            acciones: Lista de acciones tomadas
            recompensas_descontadas: Lista de recompensas descontadas

        Returns:
            Gradiente de la política
        """
        if self.pesos is None:
            raise ValueError("Pesos no inicializados")

        gradiente_politica = np.zeros_like(self.pesos)
        gradiente_sesgo = np.zeros_like(self.sesgo)

        for estado, accion, recompensa_descontada in zip(estados, acciones, recompensas_descontadas):
            # Calcular probabilidades de acción
            probabilidades = self.forward(estado)

            # Calcular gradiente de log-probabilidad
            gradiente_log_prob = np.zeros(self.output_size)
            gradiente_log_prob[accion] = 1.0 / (probabilidad[accion] + 1e-8)

            # Calcular gradiente de la política
            gradiente_politica += np.outer(estado, gradiente_log_prob) * recompensa_descontada
            gradiente_sesgo += gradiente_log_prob * recompensa_descontada

        # Normalizar por el número de muestras
        n_muestras = len(estados)
        gradiente_politica /= n_muestras
        gradiente_sesgo /= n_muestras

        # Agregar término de entropía para exploración
        if self.entropy_coef > 0:
            gradiente_entropia = self._calcular_gradiente_entropia(estados)
            gradiente_politica += self.entropy_coef * gradiente_entropia

        # Guardar historial
        norma_gradiente = np.linalg.norm(gradiente_politica)
        self.historial_gradientes.append(norma_gradiente)
        self.estadisticas_pg['gradiente_politica_norma'] = norma_gradiente

        return gradiente_politica, gradiente_sesgo

    def _calcular_gradiente_entropia(self, estados: List[np.ndarray]) -> np.ndarray:
        """
        Calcula el gradiente de entropía para regularización.

        Args:
            estados: Lista de estados

        Returns:
            Gradiente de entropía
        """
        gradiente_entropia = np.zeros_like(self.pesos)

        for estado in estados:
            probabilidades = self.forward(estado)

            # Calcular gradiente de entropía
            for accion in range(self.output_size):
                gradiente_log_prob = np.zeros(self.output_size)
                gradiente_log_prob[accion] = 1.0 / (probabilidades[0, accion] + 1e-8)

                # Gradiente de entropía
                entropia_grad = -(1 + np.log(probabilidades[0, accion] + 1e-8))
                gradiente_entropia += np.outer(estado, gradiente_log_prob) * entropia_grad

        return gradiente_entropia / len(estados)

    def actualizar_politica(self, gradiente_politica: np.ndarray,
                            gradiente_sesgo: np.ndarray) -> None:
        """
        Actualiza la política usando el gradiente calculado.

        Args:
            gradiente_politica: Gradiente de los pesos de la política
            gradiente_sesgo: Gradiente del sesgo
        """
        if self.pesos is None:
            raise ValueError("Pesos no inicializados")

        # Actualizar pesos
        self.pesos += self.learning_rate * gradiente_politica
        self.sesgo += self.learning_rate * gradiente_sesgo

        # Actualizar estadísticas
        self.estadisticas_pg['actualizaciones_politica'] += 1

        # Calcular convergencia
        self._calcular_convergencia_politica()

    def _calcular_convergencia_politica(self) -> None:
        """
        Calcula la convergencia de la política.
        """
        if len(self.historial_gradientes) < 10:
            return

        # Convergencia basada en la estabilidad de los gradientes
        gradientes_recientes = self.historial_gradientes[-10:]
        varianza_gradientes = np.var(gradientes_recientes)

        # Convergencia inversamente proporcional a la varianza
        convergencia = 1.0 / (1.0 + varianza_gradientes)
        self.estadisticas_pg['convergencia_politica'] = convergencia

    def calcular_estabilidad_politica(self) -> float:
        """
        Calcula la estabilidad de la política.

        Returns:
            Índice de estabilidad (0-1)
        """
        if len(self.historial_politicas) < 20:
            return 0.0

        # Calcular varianza de las probabilidades en el tiempo
        politicas_array = np.array(self.historial_politicas[-20:])
        varianza_temporal = np.var(politicas_array, axis=0)
        varianza_promedio = np.mean(varianza_temporal)

        # Estabilidad inversamente proporcional a la varianza
        estabilidad = 1.0 / (1.0 + varianza_promedio)

        self.estadisticas_pg['estabilidad_politica'] = estabilidad

        return estabilidad

    def calcular_exploracion_efectiva(self) -> float:
        """
        Calcula la exploración efectiva basada en la entropía.

        Returns:
            Índice de exploración efectiva (0-1)
        """
        if not self.historial_entropias:
            return 0.0

        # Exploración efectiva basada en la entropía promedio
        entropia_promedio = np.mean(self.historial_entropias[-10:])
        entropia_maxima = math.log(self.output_size)  # Entropía máxima posible

        exploracion = entropia_promedio / entropia_maxima
        self.estadisticas_pg['exploracion_efectiva'] = exploracion

        return exploracion

    def obtener_estadisticas_pg(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas específicas de Policy Gradient.

        Returns:
            Diccionario con estadísticas de Policy Gradient
        """
        if self.pesos is None:
            return {'estado': 'no_inicializada'}

        # Calcular estadísticas de la política
        if self.historial_politicas:
            politicas_array = np.array(self.historial_politicas[-10:])
            self.estadisticas_pg['probabilidad_accion_media'] = np.mean(politicas_array)
            self.estadisticas_pg['probabilidad_accion_std'] = np.std(politicas_array)

        if self.historial_entropias:
            self.estadisticas_pg['entropia_politica'] = np.mean(self.historial_entropias[-10:])

        stats_pg = {
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'entropy_coef': self.entropy_coef,
            'convergencia_politica': self.estadisticas_pg['convergencia_politica'],
            'estabilidad_politica': self.calcular_estabilidad_politica(),
            'exploracion_efectiva': self.calcular_exploracion_efectiva(),
            'actualizaciones_politica': self.estadisticas_pg['actualizaciones_politica'],
            'gradiente_politica_norma': self.estadisticas_pg['gradiente_politica_norma'],
            'entropia_politica': self.estadisticas_pg['entropia_politica'],
            'probabilidad_accion_media': self.estadisticas_pg['probabilidad_accion_media'],
            'probabilidad_accion_std': self.estadisticas_pg['probabilidad_accion_std']
        }

        return stats_pg

    def verificar_estabilidad(self) -> Dict[str, bool]:
        """
        Verifica la estabilidad de la neurona Policy Gradient.

        Returns:
            Diccionario con indicadores de estabilidad
        """
        estabilidad = {}

        # Verificar convergencia
        convergencia = self.estadisticas_pg['convergencia_politica']
        estabilidad['convergencia_ok'] = convergencia > 0.7

        # Verificar estabilidad de la política
        estabilidad_politica = self.calcular_estabilidad_politica()
        estabilidad['politica_estable'] = estabilidad_politica > 0.7

        # Verificar exploración efectiva
        exploracion = self.calcular_exploracion_efectiva()
        estabilidad['exploracion_efectiva'] = 0.1 <= exploracion <= 0.9

        # Verificar gradientes
        norma_gradiente = self.estadisticas_pg['gradiente_politica_norma']
        estabilidad['gradientes_no_explosivos'] = norma_gradiente < 10.0
        estabilidad['gradientes_no_desaparecen'] = norma_gradiente > 1e-8

        # Verificar pesos
        if self.pesos is not None:
            peso_max = np.max(np.abs(self.pesos))
            peso_min = np.min(np.abs(self.pesos))

            estabilidad['pesos_no_explosivos'] = peso_max < 5.0
            estabilidad['pesos_no_desaparecen'] = peso_min > 1e-6
            estabilidad['pesos_balanceados'] = peso_max / peso_min < 1000.0
        else:
            estabilidad['pesos_no_explosivos'] = True
            estabilidad['pesos_no_desaparecen'] = True
            estabilidad['pesos_balanceados'] = True

        return estabilidad

    def reinicializar_con_parametros(self, learning_rate: float = None,
                                     gamma: float = None,
                                     entropy_coef: float = None) -> None:
        """
        Reinicializa la neurona con nuevos parámetros.

        Args:
            learning_rate: Nueva tasa de aprendizaje
            gamma: Nuevo factor de descuento
            entropy_coef: Nuevo coeficiente de entropía
        """
        if learning_rate is not None:
            self.learning_rate = learning_rate
        if gamma is not None:
            self.gamma = gamma
        if entropy_coef is not None:
            self.entropy_coef = entropy_coef

        # Reinicializar pesos
        self.inicializar_pesos()

        # Resetear historial
        self.resetear_historial()

        # Limpiar estadísticas específicas
        self.historial_politicas.clear()
        self.historial_gradientes.clear()
        self.historial_entropias.clear()
        self.historial_recompensas_descontadas.clear()

        # Resetear contadores
        self.estadisticas_pg['actualizaciones_politica'] = 0

        logger.info(f"Neurona Policy Gradient reinicializada: lr={self.learning_rate}, "
                    f"gamma={self.gamma}, entropy_coef={self.entropy_coef}")

    def __str__(self) -> str:
        return (f"NeuronaRefuerzoPolicyGradient(entrada={self.input_size}, "
                f"salida={self.output_size}, lr={self.learning_rate}, "
                f"gamma={self.gamma}, entropy_coef={self.entropy_coef})")

    def __repr__(self) -> str:
        return self.__str__()

# Funciones de utilidad específicas para RFEN1_RN_2


def crear_neurona_policy_gradient(input_size: int, output_size: int,
                                  configuracion: Dict[str, Any] = None) -> NeuronaRefuerzoPolicyGradient:
    """
    Función de conveniencia para crear una neurona Policy Gradient.

    Args:
        input_size: Número de características de entrada
        output_size: Número de acciones
        configuracion: Configuración adicional

    Returns:
        Instancia de NeuronaRefuerzoPolicyGradient
    """
    if configuracion is None:
        configuracion = {}

    return NeuronaRefuerzoPolicyGradient(
        input_size=input_size,
        output_size=output_size,
        nombre=configuracion.get('nombre', 'NeuronaRefuerzoPolicyGradient'),
        learning_rate=configuracion.get('learning_rate', 0.001),
        gamma=configuracion.get('gamma', 0.99),
        usar_ortogonal=configuracion.get('usar_ortogonal', True),
        entropy_coef=configuracion.get('entropy_coef', 0.01)
    )


# Configuración específica para RFEN1_RN_2
RFEN2_CONFIG = {
    'inicializacion_preferida': 'ortogonal',
    'learning_rate_default': 0.001,
    'gamma_default': 0.99,
    'entropy_coef_default': 0.01,
    'umbral_convergencia': 0.7,
    'umbral_estabilidad': 0.7,
    'umbral_exploracion_min': 0.1,
    'umbral_exploracion_max': 0.9
}

logger.info("RFEN1_RN_2.py cargado correctamente - Neurona de Refuerzo Policy Gradient")
